"""Streamlit UI component for managing Discord notifications and per-ticker alerts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Paths
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "notification_config.json"
FAVORITES_PATH = ROOT / "artifacts" / "favorites.json"

# Load env for webhook check
load_dotenv(ROOT / ".env")


def load_config() -> dict:
    """Load, migrate, and return the notification config."""
    default_structure = {
        "enabled": True,
        "target": "favorites",
        "global_defaults": {
            "weekly_macd_cross": True,
            "daily_macd_cross": False,
            "price_alert": True
        },
        "tickers": {},
        "checks": {
            "weekly_macd_cross": {
                "enabled": True,
                "macd_params": {"fast": 12, "slow": 26, "signal": 9}
            },
            "daily_macd_cross": {
                "enabled": True,
                "macd_params": {"fast": 12, "slow": 26, "signal": 9}
            },
            "price_alert": {
                "enabled": True
            }
        }
    }

    if not CONFIG_PATH.exists():
        return default_structure

    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        config = {}

    # Migrate / normalize old settings
    if "enabled" not in config:
        config["enabled"] = True
    if "target" not in config:
        config["target"] = "favorites"
    if "global_defaults" not in config:
        config["global_defaults"] = default_structure["global_defaults"]
    if "tickers" not in config:
        config["tickers"] = {}
    if "checks" not in config:
        config["checks"] = default_structure["checks"]

    # Ensure all checkers are globally enabled under "checks"
    for ck in ["weekly_macd_cross", "daily_macd_cross", "price_alert"]:
        if ck not in config["checks"]:
            config["checks"][ck] = {"enabled": True}
        else:
            config["checks"][ck]["enabled"] = True

        if ck in ["weekly_macd_cross", "daily_macd_cross"]:
            if "macd_params" not in config["checks"][ck]:
                config["checks"][ck]["macd_params"] = {"fast": 12, "slow": 26, "signal": 9}

    # Persist migrated config immediately if it changed
    try:
        current_on_disk = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
        serialized = json.dumps(config, ensure_ascii=False, indent=2)
        if current_on_disk != serialized:
            save_config(config)
    except Exception:
        pass

    return config


def save_config(config: dict) -> None:
    """Save config dict to notification_config.json."""
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    try:
        from git_utils import git_push_changes
        git_push_changes("Update notification config via UI")
    except Exception:
        pass


def load_favorites() -> dict[str, str]:
    """Load favorites directly from favorites.json."""
    if not FAVORITES_PATH.exists():
        return {}
    try:
        raw = json.loads(FAVORITES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, list):
        return {t: "" for t in raw}
    if isinstance(raw, dict):
        return {str(k): str(v) if v is not None else "" for k, v in raw.items()}
    return {}


def show_notification_ui(name_lookup: dict[str, str]) -> None:
    """Render the notification settings screen.

    Layout:
      Section 01 — Status & Actions: master toggle, webhook status, test buttons
      Section 02 — Detection Defaults: global MACD cross checkboxes
      Section 03 — Per-Ticker Settings: compact grid with inline price alerts
    """
    st.markdown("## 🔔 自動通知設定")
    st.markdown(
        "<div style='font-family: var(--font-display); font-style: italic; "
        "color: var(--ink-muted); margin-bottom: 1.5rem;'>"
        "お気に入り銘柄の週足/日足MACDクロス、および設定価格への到達判定を検知し、"
        "Discordへ自動通知します。"
        "</div>",
        unsafe_allow_html=True,
    )

    # Load configuration
    config = load_config()

    # ------------------------------------------------------------------ #
    # Section 01 — Status & Actions (top card)
    # ------------------------------------------------------------------ #
    st.markdown(
        "<div class='kt-section-label'>"
        "<span class='kt-section-num'>01</span>ステータス & アクション"
        "</div>",
        unsafe_allow_html=True,
    )

    # Webhook status
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if webhook_url:
        masked = webhook_url[:30] + "..." if len(webhook_url) > 30 else webhook_url
        webhook_html = (
            f"<span style='color: var(--forest);'>✅ 接続中</span>"
            f"&nbsp;&nbsp;<code style='color: var(--ink-muted); font-size: 0.75rem;'>"
            f"{masked}</code>"
        )
    else:
        webhook_html = (
            "<span style='color: var(--shu);'>⚠️ 未設定</span>"
            "&nbsp;&nbsp;<span style='font-size: 0.75rem; color: var(--ink-muted);'>"
            "<code>.env</code> に <code>DISCORD_WEBHOOK_URL</code> を追加してください"
            "</span>"
        )

    # Top row: toggle + webhook status
    col_toggle, col_status = st.columns([1, 2])
    with col_toggle:
        new_enabled = st.toggle(
            "通知システム ON / OFF",
            value=config.get("enabled", True),
            key="master_toggle",
        )
        if new_enabled != config.get("enabled", True):
            config["enabled"] = new_enabled
            save_config(config)
            st.rerun()
        status_label = (
            "🟢 **有効** — 定期チェック時（9:05 / 15:35）に通知が実行されます"
            if new_enabled
            else "⚫ **無効** — すべての通知送信がスキップされます"
        )
        st.markdown(
            f"<div style='font-size: 0.8rem; margin-top: 4px;'>{status_label}</div>",
            unsafe_allow_html=True,
        )

    with col_status:
        st.markdown(
            f"<div style='font-size: 0.85rem; margin-top: 6px;'>"
            f"<strong style='font-family: var(--font-mono); font-size: 0.7rem; "
            f"letter-spacing: 0.08em;'>WEBHOOK</strong>&nbsp;&nbsp;{webhook_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Test buttons — placed directly in the status card for immediate access
    st.markdown(
        "<div style='margin-top: 12px; padding-top: 10px; "
        "border-top: 1px dashed var(--border);'></div>",
        unsafe_allow_html=True,
    )
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button(
            "📢 テスト通知送信",
            use_container_width=True,
            type="primary",
        ):
            with st.spinner("テスト通知を送信中..."):
                try:
                    cmd = [
                        sys.executable,
                        str(ROOT / "run_notification.py"),
                        "--session", "test",
                        "--test",
                    ]
                    res = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=30,
                    )
                    if res.returncode == 0:
                        st.success(
                            "テスト通知の送信完了。Discord チャンネルを確認してください。"
                        )
                    else:
                        st.error(
                            f"テスト通知の送信失敗:\n{res.stderr or res.stdout}"
                        )
                except Exception as e:
                    st.error(f"実行中にエラーが発生しました: {e}")
    with col_btn2:
        if st.button(
            "🔍 チェック実行テスト",
            use_container_width=True,
            type="primary",
        ):
            with st.spinner("スキャン & 送信実行中..."):
                try:
                    cmd = [
                        sys.executable,
                        str(ROOT / "run_notification.py"),
                        "--session", "test",
                    ]
                    res = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=60,
                    )
                    if res.returncode == 0:
                        st.success(
                            "通知チェックと Discord への送信が正常に完了しました。"
                        )
                        st.text_area("実行ログ:", res.stdout, height=250)
                    else:
                        st.error(
                            f"実行・送信に失敗しました:\n{res.stderr or res.stdout}"
                        )
                except Exception as e:
                    st.error(f"実行中にエラーが発生しました: {e}")

    # ------------------------------------------------------------------ #
    # Section 02 — Detection Defaults
    # ------------------------------------------------------------------ #
    st.markdown(
        "<div class='kt-section-label'>"
        "<span class='kt-section-num'>02</span>検知条件デフォルト"
        "</div>",
        unsafe_allow_html=True,
    )

    col_wd, col_dd = st.columns(2)
    with col_wd:
        w_default = config["global_defaults"].get("weekly_macd_cross", True)
        new_w_default = st.checkbox(
            "週足 MACD クロス (GC/DC)",
            value=w_default,
            key="g_weekly_macd",
        )
        if new_w_default != w_default:
            config["global_defaults"]["weekly_macd_cross"] = new_w_default
            save_config(config)
            st.rerun()
    with col_dd:
        d_default = config["global_defaults"].get("daily_macd_cross", False)
        new_d_default = st.checkbox(
            "日足 MACD クロス (GC/DC)",
            value=d_default,
            key="g_daily_macd",
        )
        if new_d_default != d_default:
            config["global_defaults"]["daily_macd_cross"] = new_d_default
            save_config(config)
            st.rerun()

    st.caption(
        "※ ここで設定したデフォルトは、個別設定のない銘柄に適用されます。"
    )

    # ------------------------------------------------------------------ #
    # Section 03 — Per-Ticker Settings (expander list with pagination)
    # ------------------------------------------------------------------ #
    st.markdown(
        "<div class='kt-section-label'>"
        "<span class='kt-section-num'>03</span>お気に入り銘柄別の個別設定"
        "</div>",
        unsafe_allow_html=True,
    )

    favs = load_favorites()

    if not favs:
        st.info(
            "お気に入り登録されている銘柄がありません。"
            "「チャート分析」画面でお気に入りを追加してください。"
        )
    else:
        # Search filter
        search = st.text_input(
            "🔍 銘柄を検索 (コードまたは名称)", "", key="ticker_search"
        ).strip()

        # Reset visible count if search term changes
        if "visible_tickers_count" not in st.session_state:
            st.session_state.visible_tickers_count = 5
        if "prev_ticker_search" not in st.session_state:
            st.session_state.prev_ticker_search = ""

        if search != st.session_state.prev_ticker_search:
            st.session_state.visible_tickers_count = 5
            st.session_state.prev_ticker_search = search

        filtered_favs: dict[str, str] = {}
        for t, name in favs.items():
            disp_name = name or name_lookup.get(t, "")
            if (
                search.lower() in t.lower()
                or search.lower() in disp_name.lower()
            ):
                filtered_favs[t] = disp_name

        if not filtered_favs:
            st.caption("一致する銘柄はありません。")
        else:
            # Paginated list of expanders
            visible_count = st.session_state.visible_tickers_count
            filtered_list = list(filtered_favs.items())
            visible_tickers = filtered_list[:visible_count]

            for t, name in visible_tickers:
                t_cfg = config.setdefault("tickers", {}).setdefault(t, {})
                price_alerts: list[dict] = t_cfg.setdefault("price_alerts", [])

                # Load current values
                w_val = t_cfg.get(
                    "weekly_macd_cross",
                    config["global_defaults"]["weekly_macd_cross"],
                )
                d_val = t_cfg.get(
                    "daily_macd_cross",
                    config["global_defaults"]["daily_macd_cross"],
                )

                # Format expander label with summary
                status_parts = []
                status_parts.append(f"週足: {'✅' if w_val else '❌'}")
                status_parts.append(f"日足: {'✅' if d_val else '❌'}")
                if price_alerts:
                    status_parts.append(f"価格: {len(price_alerts)}件")
                else:
                    status_parts.append("価格: なし")
                status_str = " ｜ ".join(status_parts)

                label = f"{t}  {name}" if name else t
                expander_label = f"📊 {label}　（{status_str}）"

                with st.expander(expander_label, expanded=False):
                    # MACD Settings
                    st.markdown(
                        "<div style='font-size: 0.85rem; font-weight: bold; margin-bottom: 0.5rem;'>"
                        "📈 MACD 自動検知条件"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    col_w, col_d = st.columns(2)
                    with col_w:
                        new_w = st.checkbox(
                            "週足 MACD クロスを検知する",
                            value=w_val,
                            key=f"w_chk_{t}",
                        )
                        if new_w != w_val:
                            t_cfg["weekly_macd_cross"] = new_w
                            save_config(config)
                            st.rerun()
                    with col_d:
                        new_d = st.checkbox(
                            "日足 MACD クロスを検知する",
                            value=d_val,
                            key=f"d_chk_{t}",
                        )
                        if new_d != d_val:
                            t_cfg["daily_macd_cross"] = new_d
                            save_config(config)
                            st.rerun()

                    st.markdown(
                        "<div style='border-bottom: 1px solid var(--border); margin: 0.5rem 0;'></div>",
                        unsafe_allow_html=True,
                    )

                    # Price Alerts
                    st.markdown(
                        "<div style='font-size: 0.85rem; font-weight: bold; margin-bottom: 0.5rem;'>"
                        "🔔 設定中の価格アラート"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    if price_alerts:
                        for i, pa in enumerate(price_alerts):
                            arrow = "📈 上抜け" if pa["direction"] == "above" else "📉 下抜け"
                            col_info, col_del = st.columns([8, 2])
                            col_info.markdown(
                                f"<div style='font-size: 0.85rem; padding-top: 4px;'>"
                                f"`{pa['price']:,.0f}`円 ({arrow})"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            if col_del.button(
                                "🗑️ 削除",
                                key=f"del_pa_{t}_{i}",
                                use_container_width=True,
                            ):
                                price_alerts.pop(i)
                                t_cfg["price_alert"] = len(price_alerts) > 0
                                save_config(config)
                                st.toast(f"{t} の価格条件を削除しました。")
                                st.rerun()
                    else:
                        st.caption("設定されている価格アラートはありません。")

                    st.markdown(
                        "<div style='border-bottom: 1px solid var(--border); margin: 0.5rem 0;'></div>",
                        unsafe_allow_html=True,
                    )

                    # Add new Price Alert
                    st.markdown(
                        "<div style='font-size: 0.85rem; font-weight: bold; margin-bottom: 0.5rem;'>"
                        "➕ 新規価格アラート追加"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    with st.form(key=f"add_alert_form_{t}", clear_on_submit=True):
                        col_f1, col_f2, col_f3 = st.columns([5, 4, 3])
                        new_price = col_f1.number_input(
                            "ターゲット価格 (円)",
                            min_value=0.0,
                            step=10.0,
                            value=0.0,
                            key=f"inp_price_{t}",
                        )
                        new_dir = col_f2.selectbox(
                            "方向",
                            ["above", "below"],
                            format_func=lambda x: (
                                "📈 上抜け" if x == "above" else "📉 下抜け"
                            ),
                            key=f"inp_dir_{t}",
                        )
                        submit = col_f3.form_submit_button("追加")

                        if submit:
                            if new_price <= 0:
                                st.error("0より大きい有効な価格を入力してください。")
                            else:
                                price_alerts.append(
                                    {
                                        "price": float(new_price),
                                        "direction": new_dir,
                                    }
                                )
                                t_cfg["price_alert"] = True
                                save_config(config)
                                st.toast(f"{t} に価格アラートを追加しました。")
                                st.rerun()

            # More/Less buttons
            if len(filtered_list) > 5:
                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                col_more, col_less = st.columns(2) if visible_count > 5 else (st.columns(1)[0], None)

                if col_more.button("🔽 もっと見る", use_container_width=True):
                    st.session_state.visible_tickers_count += 5
                    st.rerun()

                if col_less is not None:
                    if col_less.button("🔼 元に戻す (5件表示)", use_container_width=True):
                        st.session_state.visible_tickers_count = 5
                        st.rerun()
