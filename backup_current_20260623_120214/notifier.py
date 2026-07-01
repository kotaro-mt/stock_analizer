"""Discord Webhook notification sender.

Sends richly-formatted Embed messages to a Discord channel via Webhook.
Handles rate-limiting (Discord allows ~30 req/min per webhook) and
transient HTTP errors with simple exponential back-off.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Discord Embed colour constants (decimal RGB)
COLOR_GREEN = 0x00C853   # ゴールデンクロス
COLOR_RED   = 0xFF1744   # デッドクロス
COLOR_BLUE  = 0x2979FF   # 情報通知
COLOR_AMBER = 0xFFAB00   # 警告


@dataclass
class EmbedField:
    """A single field inside a Discord Embed."""
    name: str
    value: str
    inline: bool = True


class DiscordNotifier:
    """Thin wrapper around the Discord Webhook API.

    Usage::

        notifier = DiscordNotifier(webhook_url)
        notifier.send_embed(
            title="MACD ゴールデンクロス",
            description="7203.T トヨタ自動車",
            color=COLOR_GREEN,
            fields=[EmbedField("終値", "2,845 円")],
        )
    """

    # Discord rate-limit: stay well under 30 req/min
    MIN_INTERVAL_SEC = 2.5
    MAX_RETRIES = 3

    def __init__(self, webhook_url: str) -> None:
        if not webhook_url:
            raise ValueError("DISCORD_WEBHOOK_URL is empty or not set")
        self.webhook_url = webhook_url
        self._last_sent: float = 0.0

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        """Sleep if we're sending too fast."""
        elapsed = time.time() - self._last_sent
        if elapsed < self.MIN_INTERVAL_SEC:
            time.sleep(self.MIN_INTERVAL_SEC - elapsed)

    def send_embed(
        self,
        title: str,
        description: str,
        color: int = COLOR_BLUE,
        fields: list[EmbedField] | None = None,
        footer: str | None = None,
    ) -> bool:
        """Send a single Embed message.  Returns True on success."""
        embed: dict[str, Any] = {
            "title": title,
            "description": description,
            "color": color,
        }
        if fields:
            embed["fields"] = [
                {"name": f.name, "value": f.value, "inline": f.inline}
                for f in fields
            ]
        if footer:
            embed["footer"] = {"text": footer}

        payload = {"embeds": [embed]}

        for attempt in range(1, self.MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=15,
                )
                self._last_sent = time.time()

                if resp.status_code == 204:
                    return True
                if resp.status_code == 429:
                    # Rate-limited — honour Retry-After header
                    retry_after = resp.json().get("retry_after", 5)
                    logger.warning(
                        "Discord rate-limited; retrying after %.1fs", retry_after,
                    )
                    time.sleep(float(retry_after))
                    continue
                logger.error(
                    "Discord webhook error %s: %s", resp.status_code, resp.text,
                )
            except requests.RequestException as exc:
                logger.error("Discord webhook request failed: %s", exc)
                if attempt < self.MAX_RETRIES:
                    time.sleep(2 ** attempt)

        return False

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------
    def send_macd_alert(
        self,
        ticker: str,
        name: str,
        cross_type: str,      # "gc" or "dc"
        macd_val: float,
        signal_val: float,
        close_price: float,
        cross_date: str,
        session: str = "",
        interval_label: str = "週足",
    ) -> bool:
        """Send a formatted MACD Golden/Dead Cross alert."""
        is_gc = cross_type == "gc"
        emoji = "🟢" if is_gc else "🔴"
        cross_label = "ゴールデンクロス" if is_gc else "デッドクロス"
        trend_emoji = "📈" if is_gc else "📉"
        color = COLOR_GREEN if is_gc else COLOR_RED

        title = f"{emoji} {interval_label} MACD {cross_label}"
        description = f"{trend_emoji} **{ticker}　{name}**"

        fields = [
            EmbedField("MACD", f"`{macd_val:+.2f}`"),
            EmbedField("Signal", f"`{signal_val:+.2f}`"),
            EmbedField("終値", f"`{close_price:,.0f}` 円"),
            EmbedField("クロス日", f"`{cross_date}`（{interval_label}）", inline=False),
        ]

        footer_parts = ["stock_future 自動通知"]
        if session:
            footer_parts.append(f"セッション: {session}")
        footer = " ｜ ".join(footer_parts)

        return self.send_embed(
            title=title,
            description=description,
            color=color,
            fields=fields,
            footer=footer,
        )

    def send_price_alert(
        self,
        ticker: str,
        name: str,
        target_price: float,
        current_price: float,
        direction: str,       # "above" or "below"
    ) -> bool:
        """Send a price-level alert (placeholder for future use)."""
        emoji = "⬆️" if direction == "above" else "⬇️"
        title = f"{emoji} 価格アラート"
        description = f"**{ticker}　{name}**"

        fields = [
            EmbedField("現在値", f"`{current_price:,.0f}` 円"),
            EmbedField("設定価格", f"`{target_price:,.0f}` 円"),
            EmbedField("方向", f"`{'上抜け' if direction == 'above' else '下抜け'}`"),
        ]

        return self.send_embed(
            title=title,
            description=description,
            color=COLOR_AMBER,
            fields=fields,
            footer="stock_future 自動通知",
        )

    def send_summary(
        self,
        session: str,
        total_checked: int,
        alerts_sent: int,
        errors: int = 0,
    ) -> bool:
        """Send a simple run-summary message after a check session completes.

        This is a lightweight fallback.  For richer per-ticker detail,
        use :meth:`send_detailed_summary` instead.
        """
        # For test sessions, always send the summary so the user gets confirmation.
        is_test = "テスト" in session or "test" in session.lower()
        if alerts_sent == 0 and errors == 0 and not is_test:
            # Don't spam — skip summary when nothing happened
            return True

        emoji = "✅" if errors == 0 else "⚠️"
        title = f"{emoji} 通知チェック完了"
        description = f"セッション: **{session}**"
        fields = [
            EmbedField("チェック銘柄数", f"`{total_checked}`"),
            EmbedField("アラート送信数", f"`{alerts_sent}`"),
        ]
        if errors:
            fields.append(EmbedField("エラー数", f"`{errors}`"))

        return self.send_embed(
            title=title,
            description=description,
            color=COLOR_BLUE if errors == 0 else COLOR_AMBER,
            fields=fields,
            footer="stock_future 自動通知",
        )

    # ------------------------------------------------------------------
    # Detailed summary (primary method for post-session reporting)
    # ------------------------------------------------------------------

    # Status-to-emoji mapping used by send_detailed_summary
    _STATUS_EMOJI: dict[str, str] = {
        "bullish":  "🟢",
        "positive": "🟢",
        "bearish":  "🔴",
        "negative": "🔴",
        "watching": "👀",
    }

    def _build_ticker_block(
        self,
        ts: dict,
        fired_tickers: set[str],
    ) -> str:
        """Build the description block for a single ticker.

        Parameters
        ----------
        ts:
            A ticker-status dict with keys ``ticker``, ``name``, and
            ``checks`` (list of check-result dicts).
        fired_tickers:
            Set of ticker symbols that had alerts fired this session.

        Returns
        -------
        str
            Formatted markdown block ready for Discord Embed description.
        """
        ticker = ts["ticker"]
        name = ts.get("name", "")
        checks: list[dict] = ts.get("checks", [])

        # Highlight tickers that fired alerts
        if ticker in fired_tickers:
            header = f"⚡ **{ticker}　{name}** ── アラート発火"
        else:
            header = f"**{ticker}　{name}**"

        change_pct = ts.get("previous_change_pct")
        if change_pct is None:
            price_line = "　前日比: `—`"
        else:
            arrow = "▲" if change_pct > 0 else "▼" if change_pct < 0 else "→"
            price_line = f"　{arrow} 前日比: `{change_pct:+.2f}%`"

        check_lines: list[str] = []
        for chk in checks:
            emoji = self._STATUS_EMOJI.get(chk.get("status", ""), "⚪")
            label = chk.get("label", chk.get("type", ""))
            detail = chk.get("detail", "")
            check_lines.append(f"　{emoji} {label}: {detail}")

        lines = [header, price_line, *check_lines]
        return "\n".join(lines)

    def send_detailed_summary(
        self,
        session: str,
        ticker_statuses: list[dict],
        fired_alerts: list,
    ) -> bool:
        """Send a comprehensive summary of all monitored tickers.

        If the content exceeds the Discord Embed description limit
        (~4096 chars), the message is automatically split into multiple
        embeds at ticker boundaries so no single ticker's info is broken
        across messages.

        Parameters
        ----------
        session:
            Human-readable session label, e.g. ``"寄り付き (9:05)"``.
        ticker_statuses:
            List of dicts, each containing ``ticker``, ``name``, and
            ``checks`` (list of ``{type, label, status, detail}`` dicts).
        fired_alerts:
            List of Alert objects that were fired during this session.
            Each object must expose a ``.ticker`` attribute.

        Returns
        -------
        bool
            ``True`` if all messages were sent successfully.
        """
        # Collect tickers that fired alerts for highlighting
        fired_tickers: set[str] = {
            getattr(a, "ticker", None) for a in fired_alerts
        }
        fired_tickers.discard(None)

        # Build per-ticker blocks
        blocks: list[str] = [
            self._build_ticker_block(ts, fired_tickers)
            for ts in ticker_statuses
        ]

        # Stats line shown at the top of the first message
        stats_line = (
            f"セッション: **{session}**\n"
            f"チェック銘柄数: `{len(ticker_statuses)}` ｜ "
            f"アラート発火数: `{len(fired_alerts)}`\n"
            f"{'━' * 30}"
        )

        # --- Split blocks into pages that fit within 4000 chars --------
        max_chars = 4000
        pages: list[list[str]] = []
        current_page: list[str] = []
        # The first page starts with the stats line
        current_len = len(stats_line) + 2  # +2 for trailing \n\n

        for block in blocks:
            # +2 accounts for the "\n\n" separator between blocks
            needed = len(block) + 2
            if current_page and (current_len + needed) > max_chars:
                pages.append(current_page)
                current_page = []
                current_len = 0
            current_page.append(block)
            current_len += needed

        if current_page:
            pages.append(current_page)

        if not pages:
            pages = [[]]

        # --- Send each page as a separate embed -----------------------
        total_pages = len(pages)
        all_ok = True

        for idx, page_blocks in enumerate(pages, start=1):
            if total_pages == 1:
                title = "📋 通知サマリー"
            else:
                title = f"📋 通知サマリー ({idx}/{total_pages})"

            parts: list[str] = []
            if idx == 1:
                parts.append(stats_line)
            parts.extend(page_blocks)
            description = "\n\n".join(parts)

            color = COLOR_AMBER if fired_alerts else COLOR_BLUE

            ok = self.send_embed(
                title=title,
                description=description,
                color=color,
                footer="stock_future 自動通知",
            )
            if not ok:
                all_ok = False

        return all_ok

    def send_macd_report(
        self,
        statuses: list,
        session: str = "",
    ) -> bool:
        """Send a consolidated weekly MACD status report for all tickers.

        ``statuses`` is a list of ``MACDStatus`` objects (from alert_checker).
        """
        if not statuses:
            return self.send_embed(
                title="📋 週足 MACD レポート",
                description="対象銘柄がありません。",
                color=COLOR_BLUE,
                footer="stock_future 自動通知",
            )

        lines: list[str] = []
        for s in statuses:
            # Position emoji
            if s.cross_type == "gc":
                icon = "🟢⬆️"
                label = "**GC発生**"
            elif s.cross_type == "dc":
                icon = "🔴⬇️"
                label = "**DC発生**"
            elif s.position == "bullish":
                icon = "🟢"
                label = "強気"
            else:
                icon = "🔴"
                label = "弱気"

            lines.append(
                f"{icon} **{s.ticker}** {s.name}\n"
                f"　　終値 `{s.close_price:,.0f}` 円 ｜ "
                f"MACD `{s.macd_val:+.2f}` / Signal `{s.signal_val:+.2f}` ｜ "
                f"{label}"
            )

        # Discord Embed description has a 4096 char limit; split if needed
        description = "\n\n".join(lines)
        if len(description) > 4000:
            description = description[:4000] + "\n…（以下省略）"

        footer_parts = ["stock_future 自動通知"]
        if session:
            footer_parts.append(session)

        return self.send_embed(
            title="📋 週足 MACD レポート",
            description=description,
            color=COLOR_BLUE,
            footer=" ｜ ".join(footer_parts),
        )
