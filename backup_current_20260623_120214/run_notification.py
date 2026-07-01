#!/usr/bin/env python
"""Run stock-alert notifications — entry point for Task Scheduler.

Usage::

    python run_notification.py --session morning     # 寄り付き (9:05)
    python run_notification.py --session evening      # 引け     (15:35)
    python run_notification.py --session evening --dry-run   # Discordに送信せず検出のみ
    python run_notification.py --session test --test          # テスト通知を1件送信

The script:
  1. Reads ``.env`` for ``DISCORD_WEBHOOK_URL``
  2. Reads ``notification_config.json`` for enabled checkers & params
  3. Loads the target ticker list (favorites from ``artifacts/favorites.json``)
  4. Runs each enabled checker against each ticker
  5. Deduplicates against ``notification_state.json``
  6. Sends new alerts to Discord
  7. Writes updated state and a log file
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable (for data, indicators, etc.)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
import os

load_dotenv(ROOT / ".env")

from alert_checker import ALL_CHECKERS, Alert, get_weekly_macd_status, remove_fired_price_alerts, remove_fired_date_alerts
from notifier import DiscordNotifier

# Paths
CONFIG_PATH = ROOT / "notification_config.json"
STATE_PATH = ROOT / "notification_state.json"
FAVORITES_PATH = ROOT / "artifacts" / "favorites.json"
LOG_DIR = ROOT / "logs"

SESSION_LABELS = {
    "morning": "寄り付き (9:05)",
    "evening": "引け (15:35)",
    "test":    "テスト",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_favorites() -> dict[str, str]:
    """Load favorites as {ticker: name}. Handles both dict and legacy list."""
    raw = _load_json(FAVORITES_PATH)
    if isinstance(raw, list):
        return {t: "" for t in raw}
    if isinstance(raw, dict):
        return {str(k): str(v) if v else "" for k, v in raw.items()}
    return {}


def _setup_logging(session: str) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    log_file = LOG_DIR / f"notification_{today}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("run_notification")
    logger.info("=== session=%s  started ===", session)
    return logger


def _macd_distance_text(
    macd_value: float,
    signal_value: float,
    cross_type: str | None = None,
) -> str:
    """Describe the current distance to the next MACD cross."""
    if cross_type == "gc":
        return "⬆️ **GC発生**"
    if cross_type == "dc":
        return "⬇️ **DC発生**"
    gap = abs(macd_value - signal_value)
    next_cross = "GC" if macd_value <= signal_value else "DC"
    return f"{next_cross}まであと `{gap:.2f}`"


def _notification_mode(ticker_cfg: dict) -> str:
    """Return all / alerts_only / off with legacy boolean compatibility."""
    mode = ticker_cfg.get("notification_mode")
    if mode in {"all", "alerts_only", "off"}:
        return mode
    return "all" if ticker_cfg.get("notifications_enabled", True) else "off"


def _collect_ticker_statuses(
    tickers: dict[str, str],
    checks_config: dict,
    global_defaults: dict,
    tickers_cfg: dict,
    fired_tickers: set[str] | None = None,
) -> list[dict]:
    """Collect MACD and price alert status for all tickers for summary display."""
    from alert_checker import get_weekly_macd_status, _resample_to_weekly
    from data import load_ohlcv
    from indicators import macd as calc_macd

    statuses = []
    macd_params = checks_config.get("weekly_macd_cross", {}).get("macd_params", {})
    daily_macd_params = checks_config.get("daily_macd_cross", {}).get("macd_params", {})
    fired_tickers = fired_tickers or set()

    for ticker, name in tickers.items():
        ticker_cfg = tickers_cfg.get(ticker, {})
        
        notification_mode = _notification_mode(ticker_cfg)
        if notification_mode == "off":
            continue
        if notification_mode == "alerts_only" and ticker not in fired_tickers:
            continue

        checks = []
        df_daily = None
        current_price = None
        previous_change_pct = None
        try:
            df_daily = load_ohlcv(ticker, interval="1d")
            if df_daily is not None and not df_daily.empty:
                close = df_daily["Close"].dropna()
                if not close.empty:
                    current_price = float(close.iloc[-1])
                if len(close) >= 2 and float(close.iloc[-2]) != 0:
                    previous_change_pct = (
                        float(close.iloc[-1]) / float(close.iloc[-2]) - 1.0
                    ) * 100.0
        except Exception:
            pass

        # Weekly MACD status
        w_enabled = ticker_cfg.get("weekly_macd_cross", global_defaults.get("weekly_macd_cross", True))
        if w_enabled:
            try:
                ws = get_weekly_macd_status(
                    ticker, name,
                    fast=macd_params.get("fast", 12),
                    slow=macd_params.get("slow", 26),
                    signal=macd_params.get("signal", 9),
                )
                if ws:
                    status = "bullish" if ws.position == "bullish" else "bearish"
                    distance = _macd_distance_text(
                        ws.macd_val, ws.signal_val, ws.cross_type,
                    )
                    checks.append({
                        "type": "weekly_macd_cross",
                        "label": "週足MACD",
                        "status": status,
                        "detail": (
                            f"MACD `{ws.macd_val:+.2f}` / Signal `{ws.signal_val:+.2f}`"
                            f" ｜ {distance}"
                        ),
                    })
            except Exception:
                pass

        # Daily MACD status
        d_enabled = ticker_cfg.get("daily_macd_cross", global_defaults.get("daily_macd_cross", False))
        if d_enabled:
            try:
                if df_daily is not None and len(df_daily) >= 35:
                    fast = daily_macd_params.get("fast", 12)
                    slow = daily_macd_params.get("slow", 26)
                    signal = daily_macd_params.get("signal", 9)
                    macd_df = calc_macd(df_daily["Close"], fast=fast, slow=slow, signal=signal)
                    if len(macd_df["macd"]) >= 2:
                        curr_m = float(macd_df["macd"].iloc[-1])
                        curr_s = float(macd_df["signal"].iloc[-1])
                        prev_m = float(macd_df["macd"].iloc[-2])
                        prev_s = float(macd_df["signal"].iloc[-2])
                        d_status = "bullish" if curr_m > curr_s else "bearish"
                        cross_type = None
                        if prev_m <= prev_s and curr_m > curr_s:
                            cross_type = "gc"
                        elif prev_m >= prev_s and curr_m < curr_s:
                            cross_type = "dc"
                        distance = _macd_distance_text(curr_m, curr_s, cross_type)
                        checks.append({
                            "type": "daily_macd_cross",
                            "label": "日足MACD",
                            "status": d_status,
                            "detail": (
                                f"MACD `{curr_m:+.2f}` / Signal `{curr_s:+.2f}`"
                                f" ｜ {distance}"
                            ),
                        })
            except Exception:
                pass

        # Price alert status
        price_alerts = ticker_cfg.get("price_alerts", [])
        if price_alerts:
            try:
                if df_daily is not None and not df_daily.empty:
                    current = float(df_daily["Close"].iloc[-1])
                    for pa in price_alerts:
                        target = pa["price"]
                        direction = pa.get("direction", "above")
                        dir_label = "上抜け" if direction == "above" else "下抜け"
                        diff = current - target
                        diff_pct = (diff / target) * 100 if target else 0
                        checks.append({
                            "type": "price_alert",
                            "label": "価格アラート",
                            "status": "watching",
                            "detail": f"目標: `{target:,.0f}`円 ({dir_label}) 現在: `{current:,.0f}`円 (差: {diff:+,.0f}円 / {diff_pct:+.1f}%)",
                        })
            except Exception:
                pass

        # Date alert status
        date_alerts = ticker_cfg.get("date_alerts", [])
        if date_alerts:
            for da in date_alerts:
                date_str = da.get("date", "")
                label = da.get("label", date_str)
                checks.append({
                    "type": "date_alert",
                    "label": "日付アラート",
                    "status": "watching",
                    "detail": f"予定日: `{date_str}` ({label})",
                })

        # Trendlines status
        trendlines = ticker_cfg.get("trendlines", [])
        if trendlines:
            checks.append({
                "type": "trendline_alert",
                "label": "トレンドライン",
                "status": "watching",
                "detail": f"設定本数: `{len(trendlines)}`本",
            })

        statuses.append({
            "ticker": ticker,
            "name": name,
            "current_price": current_price,
            "previous_change_pct": previous_change_pct,
            "checks": checks,
        })

    return statuses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stock alert notification runner")
    parser.add_argument(
        "--session",
        choices=["morning", "evening", "test"],
        default="evening",
        help="Which session is running (morning=9:05, evening=15:35)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect alerts but do NOT send to Discord",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send a single test notification and exit",
    )
    args = parser.parse_args()

    logger = _setup_logging(args.session)
    session_label = SESSION_LABELS.get(args.session, args.session)

    # ── Webhook URL ──────────────────────────────────────────────────
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        logger.error("DISCORD_WEBHOOK_URL is not set in .env")
        sys.exit(1)

    notifier = DiscordNotifier(webhook_url)

    # ── Test mode ────────────────────────────────────────────────────
    if args.test:
        logger.info("Sending test notification …")
        ok = notifier.send_embed(
            title="🔔 テスト通知",
            description="stock_future 通知システムのテストです。",
            fields=[],
            footer=f"session={args.session}",
        )
        logger.info("Test send result: %s", "OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)

    # ── Load config ──────────────────────────────────────────────────
    config = _load_json(CONFIG_PATH)
    if not config.get("enabled", True):
        logger.info("Notifications are disabled in config - exiting.")
        return

    # ── Load target tickers ──────────────────────────────────────────
    target = config.get("target", "favorites")
    if target == "favorites":
        tickers = _load_favorites()
        if not tickers:
            logger.warning("No favorites found in %s - nothing to check.", FAVORITES_PATH)
            return
        logger.info("Target: favorites (%d tickers)", len(tickers))
    else:
        # Future: could support "all" or explicit list
        logger.error("Unsupported target type: %s", target)
        return

    # ── Load state ───────────────────────────────────────────────────
    state = _load_json(STATE_PATH)

    # ── Run checkers ─────────────────────────────────────────────────
    checks_config = config.get("checks", {})
    global_defaults = config.get("global_defaults", {})
    tickers_cfg = config.get("tickers", {})
    all_alerts: list[Alert] = []
    errors = 0

    for checker_type, checker_cls in ALL_CHECKERS.items():
        ck_config = checks_config.get(checker_type, {})
        if not ck_config.get("enabled", False):
            logger.debug("Checker %s is disabled — skipping", checker_type)
            continue

        checker = checker_cls()
        # Use the real notification state for deduplication in all sessions,
        # including test. This makes the test session produce the same results
        # as a regular morning/evening run. State is never *saved* for test
        # sessions (see below), so the dedup entries won't persist.
        ck_state = state.setdefault(checker_type, {})
        logger.info("Running checker: %s", checker_type)

        for ticker, name in tickers.items():
            # Check override or default
            ticker_cfg = tickers_cfg.get(ticker, {})
            
            if _notification_mode(ticker_cfg) == "off":
                continue

            default_val = global_defaults.get(checker_type, True if checker_type == "weekly_macd_cross" else False)
            is_enabled = ticker_cfg.get(checker_type, default_val)

            if not is_enabled:
                logger.debug("Checker %s is disabled for %s — skipping", checker_type, ticker)
                continue

            # Prepare check-specific config for this ticker
            ticker_check_config = ck_config.copy()
            if checker_type == "price_alert":
                price_alerts = ticker_cfg.get("price_alerts", [])
                if not price_alerts:
                    continue
                # Inject the ticker key into each alert for backward compatibility
                ticker_alerts = []
                for pa in price_alerts:
                    pa_copy = pa.copy()
                    pa_copy["ticker"] = ticker
                    ticker_alerts.append(pa_copy)
                ticker_check_config["alerts"] = ticker_alerts
            elif checker_type == "trendline_alert":
                trendlines = ticker_cfg.get("trendlines", [])
                if not trendlines:
                    continue
                ticker_check_config["trendlines"] = trendlines

            try:
                alerts = checker.check(ticker, name, ticker_check_config, ck_state)
                all_alerts.extend(alerts)
                for a in alerts:
                    logger.info("ALERT: %s", a.message)
            except Exception:
                logger.exception("Error checking %s (%s) with %s", ticker, name, checker_type)
                errors += 1

    # ── Send alerts ──────────────────────────────────────────────────
    sent_count = 0
    sent_alerts: list[Alert] = []  # Track successfully sent alerts
    for alert in all_alerts:
        if args.dry_run:
            logger.info("[DRY-RUN] Would send: %s", alert.message)
            sent_count += 1
            continue

        ok = False
        if alert.alert_type == "weekly_macd_cross":
            d = alert.details
            ok = notifier.send_macd_alert(
                ticker=alert.ticker,
                name=alert.name,
                cross_type=d["cross_type"],
                macd_val=d["macd"],
                signal_val=d["signal"],
                close_price=d["close"],
                cross_date=d["cross_date"],
                session=session_label,
                interval_label="週足",
            )
        elif alert.alert_type == "daily_macd_cross":
            d = alert.details
            ok = notifier.send_macd_alert(
                ticker=alert.ticker,
                name=alert.name,
                cross_type=d["cross_type"],
                macd_val=d["macd"],
                signal_val=d["signal"],
                close_price=d["close"],
                cross_date=d["cross_date"],
                session=session_label,
                interval_label="日足",
            )
        elif alert.alert_type == "price_alert":
            d = alert.details
            ok = notifier.send_price_alert(
                ticker=alert.ticker,
                name=alert.name,
                target_price=d["target_price"],
                current_price=d["current_price"],
                direction=d["direction"],
            )
        elif alert.alert_type == "date_alert":
            d = alert.details
            label = d.get("label", d.get("date", ""))
            da_date = d.get("date", "")
            ok = notifier.send_embed(
                title=f"📅 日付アラート — {alert.name}",
                description=(
                    f"**{alert.name}** ({alert.ticker})\n"
                    f"**{label}** [{da_date}]"
                ),
                color=0x2E6B47,
            )
        elif alert.alert_type == "trendline_alert":
            ok = notifier.send_embed(
                title=f"📈 トレンドライン突破 — {alert.name}",
                description=alert.message,
                color=0xD65A31,
            )
        else:
            logger.warning("Unknown alert type: %s - skipping", alert.alert_type)
            continue

        if ok:
            sent_count += 1
            sent_alerts.append(alert)
            logger.info("Sent: %s", alert.message)
        else:
            logger.error("Failed to send: %s", alert.message)
            errors += 1

    # ── Auto-remove fired price alerts from config ─────────────────────────
    if sent_alerts and not args.dry_run:
        remove_fired_price_alerts(sent_alerts, CONFIG_PATH)
        remove_fired_date_alerts(sent_alerts, CONFIG_PATH)

    # ── Save state ───────────────────────────────────────────────────
    if args.session != "test":
        _save_json(STATE_PATH, state)
        logger.info("State saved to %s", STATE_PATH)
    else:
        logger.info("Test session - state saving skipped.")

    # ── Summary ──────────────────────────────────────────────────────
    logger.info(
        "Done: checked=%d  alerts=%d  sent=%d  errors=%d",
        len(tickers), len(all_alerts), sent_count, errors,
    )

    if not args.dry_run:
        logger.info("Collecting ticker statuses for summary …")
        ticker_statuses = _collect_ticker_statuses(
            tickers, checks_config, global_defaults, tickers_cfg,
            fired_tickers={alert.ticker for alert in all_alerts},
        )
        notifier.send_detailed_summary(
            session=session_label,
            ticker_statuses=ticker_statuses,
            fired_alerts=all_alerts,
        )


    logger.info("=== session=%s  finished ===", args.session)


if __name__ == "__main__":
    main()
