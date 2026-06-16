"""Alert checkers — extensible framework for stock notification conditions.

Each checker implements the ``AlertChecker`` interface:

    class MyChecker(AlertChecker):
        def check(self, ticker, name, df_daily, state) -> list[Alert]:
            ...

Add a new checker → register it in ``ALL_CHECKERS`` → it will be picked up
automatically by ``run_notification.py``.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from data import load_ohlcv
from indicators import macd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert data class
# ---------------------------------------------------------------------------
@dataclass
class Alert:
    """Represents a single notification event."""
    alert_type: str          # e.g. "weekly_macd_cross"
    ticker: str              # e.g. "7203.T"
    name: str                # e.g. "トヨタ自動車"
    message: str             # Human-readable one-liner
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def state_key(self) -> str:
        """Unique key used for dedup in notification_state.json."""
        return f"{self.alert_type}:{self.ticker}"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class AlertChecker(ABC):
    """Abstract base for all alert-condition checkers."""

    checker_type: str = "base"

    @abstractmethod
    def check(
        self,
        ticker: str,
        name: str,
        config: dict[str, Any],
        state: dict[str, Any],
    ) -> list[Alert]:
        """Run the check for a single ticker.

        Parameters
        ----------
        ticker : str
            Yahoo-finance style ticker, e.g. ``"7203.T"``.
        name : str
            Display name for the ticker.
        config : dict
            Checker-specific config from ``notification_config.json``.
        state : dict
            Previous notification state for this checker type (mutable;
            callers persist after the run).

        Returns
        -------
        list[Alert]
            Zero or more alerts to send. Empty list = no alert.
        """


# ---------------------------------------------------------------------------
# Weekly MACD Golden/Dead Cross checker
# ---------------------------------------------------------------------------
def _resample_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly bars (Mon–Fri, ending Friday).

    Replicates the logic already used in ``data.py`` / ``chart_app.py``
    so that the weekly MACD matches what the user sees on the chart.
    """
    weekly = df_daily.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna(subset=["Close"])
    return weekly


class WeeklyMACDCrossChecker(AlertChecker):
    """Detects Golden Cross / Dead Cross on the weekly MACD.

    A GC occurs when the MACD line crosses *above* the signal line.
    A DC occurs when the MACD line crosses *below* the signal line.
    Only the most recent completed weekly bar is evaluated.
    """

    checker_type = "weekly_macd_cross"

    def check(
        self,
        ticker: str,
        name: str,
        config: dict[str, Any],
        state: dict[str, Any],
    ) -> list[Alert]:
        params = config.get("macd_params", {})
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)

        # Load daily data and resample to weekly
        df_daily = load_ohlcv(ticker, interval="1d")
        if df_daily is None or len(df_daily) < slow * 7:
            logger.warning(
                "%s (%s): insufficient data for weekly MACD", ticker, name,
            )
            return []

        df_weekly = _resample_to_weekly(df_daily)
        if len(df_weekly) < slow + signal:
            return []

        # Calculate MACD on weekly closes
        macd_df = macd(df_weekly["Close"], fast=fast, slow=slow, signal=signal)
        macd_line = macd_df["macd"]
        signal_line = macd_df["signal"]

        # We need at least 2 bars to detect a cross
        if len(macd_line) < 2:
            return []

        curr_macd = macd_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        curr_signal = signal_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]
        close_price = df_weekly["Close"].iloc[-1]
        bar_date = str(df_weekly.index[-1].date())

        cross_type: str | None = None
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            cross_type = "gc"
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            cross_type = "dc"

        if cross_type is None:
            return []

        # Dedup: skip if we already notified this exact cross
        prev_state = state.get(ticker, {})
        if (
            prev_state.get("last_cross") == cross_type
            and prev_state.get("cross_date") == bar_date
        ):
            logger.debug(
                "%s: already notified %s on %s — skipping",
                ticker, cross_type, bar_date,
            )
            return []

        # Record new state (caller persists)
        state[ticker] = {
            "last_cross": cross_type,
            "cross_date": bar_date,
        }

        cross_label = "ゴールデンクロス" if cross_type == "gc" else "デッドクロス"
        return [
            Alert(
                alert_type=self.checker_type,
                ticker=ticker,
                name=name,
                message=f"週足 MACD {cross_label}: {ticker} {name}",
                details={
                    "cross_type": cross_type,
                    "macd": float(curr_macd),
                    "signal": float(curr_signal),
                    "close": float(close_price),
                    "cross_date": bar_date,
                },
            )
        ]


# ---------------------------------------------------------------------------
# Daily MACD Golden/Dead Cross checker
# ---------------------------------------------------------------------------
class DailyMACDCrossChecker(AlertChecker):
    """Detects Golden Cross / Dead Cross on the daily MACD.

    A GC occurs when the MACD line crosses *above* the signal line.
    A DC occurs when the MACD line crosses *below* the signal line.
    Only the most recent daily bar is evaluated.
    """

    checker_type = "daily_macd_cross"

    def check(
        self,
        ticker: str,
        name: str,
        config: dict[str, Any],
        state: dict[str, Any],
    ) -> list[Alert]:
        params = config.get("macd_params", {})
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)

        # Load daily data
        df_daily = load_ohlcv(ticker, interval="1d")
        if df_daily is None or len(df_daily) < slow + signal:
            logger.warning(
                "%s (%s): insufficient data for daily MACD", ticker, name,
            )
            return []

        # Calculate MACD on daily closes
        macd_df = macd(df_daily["Close"], fast=fast, slow=slow, signal=signal)
        macd_line = macd_df["macd"]
        signal_line = macd_df["signal"]

        # We need at least 2 bars to detect a cross
        if len(macd_line) < 2:
            return []

        curr_macd = macd_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        curr_signal = signal_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]
        close_price = df_daily["Close"].iloc[-1]
        bar_date = str(df_daily.index[-1].date())

        cross_type: str | None = None
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            cross_type = "gc"
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            cross_type = "dc"

        if cross_type is None:
            return []

        # Dedup: skip if we already notified this exact cross
        prev_state = state.get(ticker, {})
        if (
            prev_state.get("last_cross") == cross_type
            and prev_state.get("cross_date") == bar_date
        ):
            logger.debug(
                "%s: already notified %s on %s — skipping",
                ticker, cross_type, bar_date,
            )
            return []

        # Record new state
        state[ticker] = {
            "last_cross": cross_type,
            "cross_date": bar_date,
        }

        cross_label = "ゴールデンクロス" if cross_type == "gc" else "デッドクロス"
        return [
            Alert(
                alert_type=self.checker_type,
                ticker=ticker,
                name=name,
                message=f"日足 MACD {cross_label}: {ticker} {name}",
                details={
                    "cross_type": cross_type,
                    "macd": float(curr_macd),
                    "signal": float(curr_signal),
                    "close": float(close_price),
                    "cross_date": bar_date,
                },
            )
        ]


# ---------------------------------------------------------------------------
# Price alert checker (skeleton for future use)
# ---------------------------------------------------------------------------
class PriceAlertChecker(AlertChecker):
    """Check whether a stock's price has *crossed* a user-specified level.

    Uses **transition-based detection**: an alert fires only when the
    price crosses the target between two consecutive daily bars.

    - ``above``: previous close < target AND current close >= target
    - ``below``: previous close > target AND current close <= target

    Fired alerts include ``config_ticker``, ``config_price``, and
    ``config_direction`` in their details so that
    ``remove_fired_price_alerts()`` can locate and delete the
    corresponding entry from ``notification_config.json``.
    """

    checker_type = "price_alert"

    def check(
        self,
        ticker: str,
        name: str,
        config: dict[str, Any],
        state: dict[str, Any],
    ) -> list[Alert]:
        alerts_cfg = config.get("alerts", [])
        if not alerts_cfg:
            return []

        df = load_ohlcv(ticker, interval="1d")
        if df is None or len(df) < 2:
            # Need at least 2 bars for transition detection
            return []

        # Transition-based detection: compare the previous bar's close
        # with the current bar's close to detect a level crossing.
        prev_price = float(df["Close"].iloc[-2])
        current_price = float(df["Close"].iloc[-1])
        results: list[Alert] = []

        for acfg in alerts_cfg:
            if acfg.get("ticker") != ticker:
                continue
            target = float(acfg["price"])
            direction = acfg.get("direction", "above")

            # Transition-based cross check:
            #   above → previous close was below target, current close is at/above
            #   below → previous close was above target, current close is at/below
            if direction == "above":
                triggered = prev_price < target and current_price >= target
            elif direction == "below":
                triggered = prev_price > target and current_price <= target
            else:
                triggered = False

            if not triggered:
                continue

            # Dedup: skip if we already notified this exact cross
            state_key = f"{ticker}_{direction}_{target}"
            if state.get(state_key, {}).get("notified"):
                continue

            state[state_key] = {"notified": True, "price": current_price}
            dir_label = "上抜け" if direction == "above" else "下抜け"
            results.append(
                Alert(
                    alert_type=self.checker_type,
                    ticker=ticker,
                    name=name,
                    message=f"価格アラート {dir_label}: {ticker} {name} → {current_price:,.0f}円",
                    details={
                        "target_price": target,
                        "current_price": current_price,
                        "prev_price": prev_price,
                        "direction": direction,
                        # Config identification keys for auto-removal
                        "config_ticker": ticker,
                        "config_price": target,
                        "config_direction": direction,
                    },
                )
            )

        return results


def remove_fired_price_alerts(
    fired_alerts: list[Alert],
    config_path: Path,
) -> None:
    """Remove fired price alerts from ``notification_config.json``.

    For each fired ``price_alert`` Alert, this function locates the
    matching entry in ``tickers[ticker].price_alerts`` (by price and
    direction) and removes it.  If the ticker's ``price_alerts`` list
    becomes empty, the ``price_alert`` enabled flag is also set to
    ``False``.

    Parameters
    ----------
    fired_alerts : list[Alert]
        Alerts that were successfully fired.  Only entries with
        ``alert_type == "price_alert"`` are processed; others are
        ignored.
    config_path : Path
        Path to ``notification_config.json``.
    """
    price_alerts = [
        a for a in fired_alerts if a.alert_type == "price_alert"
    ]
    if not price_alerts:
        return

    # Load config
    if not config_path.exists():
        logger.warning(
            "Config file not found at %s — cannot remove fired alerts",
            config_path,
        )
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tickers_cfg = config.get("tickers", {})

    modified = False
    for alert in price_alerts:
        d = alert.details
        ticker = d.get("config_ticker", alert.ticker)
        target_price = d.get("config_price")
        direction = d.get("config_direction")
        if target_price is None or direction is None:
            logger.warning(
                "Alert for %s missing config_price/config_direction — skipping removal",
                ticker,
            )
            continue

        ticker_entry = tickers_cfg.get(ticker)
        if ticker_entry is None:
            continue

        pa_list: list[dict] = ticker_entry.get("price_alerts", [])
        # Find and remove matching entry (first match only)
        for i, pa in enumerate(pa_list):
            if (
                float(pa.get("price", 0)) == float(target_price)
                and pa.get("direction", "above") == direction
            ):
                pa_list.pop(i)
                modified = True
                logger.info(
                    "Removed fired price alert: %s %s %.1f",
                    ticker, direction, target_price,
                )
                break

        # If no price alerts remain, disable the price_alert flag
        if not pa_list:
            ticker_entry["price_alert"] = False
            logger.info(
                "No price alerts left for %s — set price_alert=False",
                ticker,
            )

    if modified:
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Updated config saved to %s", config_path)


# ---------------------------------------------------------------------------
# Registry — add new checkers here
# ---------------------------------------------------------------------------
ALL_CHECKERS: dict[str, type[AlertChecker]] = {
    WeeklyMACDCrossChecker.checker_type: WeeklyMACDCrossChecker,
    DailyMACDCrossChecker.checker_type: DailyMACDCrossChecker,
    PriceAlertChecker.checker_type: PriceAlertChecker,
}


# ---------------------------------------------------------------------------
# Status report helper (used by --report flag)
# ---------------------------------------------------------------------------
@dataclass
class MACDStatus:
    """Current weekly MACD status for a single ticker."""
    ticker: str
    name: str
    macd_val: float
    signal_val: float
    hist_val: float
    close_price: float
    bar_date: str
    position: str       # "bullish" or "bearish"
    cross_type: str | None   # "gc" / "dc" / None (no cross this week)


def get_weekly_macd_status(
    ticker: str,
    name: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDStatus | None:
    """Return the current weekly MACD status for a ticker."""
    df_daily = load_ohlcv(ticker, interval="1d")
    if df_daily is None or len(df_daily) < slow * 7:
        return None

    df_weekly = _resample_to_weekly(df_daily)
    if len(df_weekly) < slow + signal:
        return None

    macd_df = macd(df_weekly["Close"], fast=fast, slow=slow, signal=signal)
    macd_line = macd_df["macd"]
    signal_line = macd_df["signal"]
    hist = macd_df["hist"]

    if len(macd_line) < 2:
        return None

    curr_macd = float(macd_line.iloc[-1])
    prev_macd = float(macd_line.iloc[-2])
    curr_signal = float(signal_line.iloc[-1])
    prev_signal = float(signal_line.iloc[-2])
    curr_hist = float(hist.iloc[-1])
    close_price = float(df_weekly["Close"].iloc[-1])
    bar_date = str(df_weekly.index[-1].date())

    # Determine cross on latest bar
    cross_type: str | None = None
    if prev_macd <= prev_signal and curr_macd > curr_signal:
        cross_type = "gc"
    elif prev_macd >= prev_signal and curr_macd < curr_signal:
        cross_type = "dc"

    position = "bullish" if curr_macd > curr_signal else "bearish"

    return MACDStatus(
        ticker=ticker,
        name=name,
        macd_val=curr_macd,
        signal_val=curr_signal,
        hist_val=curr_hist,
        close_price=close_price,
        bar_date=bar_date,
        position=position,
        cross_type=cross_type,
    )
