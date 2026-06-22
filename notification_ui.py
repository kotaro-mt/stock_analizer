"""Streamlit UI component for managing Discord notifications and per-ticker alerts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from git_utils import git_push_changes

# Paths
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "notification_config.json"
FAVORITES_PATH = ROOT / "artifacts" / "favorites.json"

NOTIFICATION_CSS = """
<style>
.notif-hero {
    position: relative;
    overflow: hidden;
    padding: 30px 34px 28px;
    margin: 4px 0 22px;
    border: 1px solid rgba(148, 163, 184, .18);
    border-radius: 22px;
    background:
      radial-gradient(circle at 92% 12%, rgba(56, 189, 248, .15), transparent 34%),
      linear-gradient(135deg, rgba(15, 23, 42, .98), rgba(15, 23, 42, .88));
    box-shadow: 0 18px 48px rgba(2, 6, 23, .25);
}
.notif-hero::after {
    content: "◌";
    position: absolute;
    right: 28px;
    top: -34px;
    font-size: 10rem;
    line-height: 1;
    color: rgba(125, 211, 252, .06);
}
.notif-eyebrow {
    color: #7dd3fc;
    font: 700 .68rem/1.2 var(--font-mono, monospace);
    letter-spacing: .18em;
    text-transform: uppercase;
}
.notif-title {
    margin: 7px 0 8px;
    color: #f8fafc;
    font: 700 clamp(1.8rem, 4vw, 3rem)/1.05 var(--font-display, sans-serif);
    letter-spacing: -.035em;
}
.notif-lead {
    max-width: 650px;
    margin: 0;
    color: #94a3b8;
    font-size: .88rem;
    line-height: 1.75;
}
.notif-chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.notif-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 11px;
    border: 1px solid rgba(148, 163, 184, .2);
    border-radius: 999px;
    background: rgba(255,255,255,.045);
    color: #cbd5e1;
    font: 600 .72rem/1 var(--font-mono, monospace);
}
.notif-dot { width: 7px; height: 7px; border-radius: 50%; background: #64748b; }
.notif-dot.on { background: #34d399; box-shadow: 0 0 0 4px rgba(52,211,153,.12); }
.notif-dot.warn { background: #fb7185; box-shadow: 0 0 0 4px rgba(251,113,133,.1); }
.notif-section-head { margin: 30px 0 12px; }
.notif-section-kicker {
    color: #64748b;
    font: 700 .65rem/1 var(--font-mono, monospace);
    letter-spacing: .16em;
}
.notif-section-title {
    margin-top: 5px;
    color: var(--ink, #e2e8f0);
    font: 700 1.08rem/1.3 var(--font-display, sans-serif);
}
.notif-section-copy { margin-top: 3px; color: var(--ink-muted, #94a3b8); font-size: .78rem; }
.notif-stat {
    min-height: 92px;
    padding: 17px 18px;
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 15px;
    background: rgba(15,23,42,.38);
}
.notif-stat-label { color: #64748b; font: 700 .64rem/1 var(--font-mono, monospace); letter-spacing: .12em; }
.notif-stat-value { margin-top: 9px; color: var(--ink, #e2e8f0); font-size: 1.05rem; font-weight: 700; }
.notif-stat-note { margin-top: 4px; color: #64748b; font-size: .68rem; }
div[data-testid="stExpander"] {
    margin-bottom: 10px;
    border: 1px solid rgba(148,163,184,.16) !important;
    border-radius: 14px !important;
    background: rgba(15,23,42,.24) !important;
    overflow: hidden;
}
div[data-testid="stExpander"] summary { padding: 12px 15px !important; }
div[data-testid="stExpander"] summary:hover { background: rgba(125,211,252,.045); }
div[data-testid="stForm"] {
    padding: 14px !important;
    border: 1px solid rgba(148,163,184,.13) !important;
    border-radius: 12px !important;
    background: rgba(2,6,23,.16);
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input,
div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
}
div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
    border-radius: 10px !important;
    font-weight: 700 !important;
}
.notif-rule { height: 1px; margin: 14px 0; background: rgba(148,163,184,.12); }
.notif-subhead { margin: 4px 0 10px; color: var(--ink, #e2e8f0); font-size: .78rem; font-weight: 700; letter-spacing: .02em; }
@media (max-width: 700px) {
    .notif-hero { padding: 24px 20px; border-radius: 16px; }
    .notif-hero::after { display: none; }
}
</style>
"""

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
            save_config(config, push=False)
    except Exception:
        pass

    return config


def save_config(config: dict, *, push: bool = True) -> None:
    """Save config dict to notification_config.json."""
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    if push:
        if not git_push_changes("Update notification settings via UI"):
            st.error(
                "設定は保存されましたが、Git pushに失敗しました。"
                "ネットワークまたはGit認証を確認してください。"
            )


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
    st.markdown(NOTIFICATION_CSS, unsafe_allow_html=True)
    config = load_config()
    favs = load_favorites()
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    is_enabled = config.get("enabled", True)
    ticker_settings = config.get("tickers", {})
    alert_count = sum(
        len(cfg.get("price_alerts", []))
        + len(cfg.get("date_alerts", []))
        + len(cfg.get("trendlines", []))
        for cfg in ticker_settings.values()
    )
    system_label = "稼働中" if is_enabled else "停止中"
    connection_label = "Discord 接続済み" if webhook_url else "Webhook 未設定"
    system_dot = "on" if is_enabled else "warn"
    connection_dot = "on" if webhook_url else "warn"
    st.markdown(
        f"""
        <section class="notif-hero">
          <div class="notif-eyebrow">Notification Control</div>
          <div class="notif-title">通知センター</div>
          <p class="notif-lead">
            お気に入り銘柄のシグナルと価格・日付・ライン条件を一か所で管理します。
            変更内容は保存後、自動的に通知環境へ反映されます。
          </p>
          <div class="notif-chip-row">
            <span class="notif-chip"><span class="notif-dot {system_dot}"></span>{system_label}</span>
            <span class="notif-chip"><span class="notif-dot {connection_dot}"></span>{connection_label}</span>
            <span class="notif-chip">対象 {len(favs)} 銘柄</span>
            <span class="notif-chip">設定中 {alert_count} 条件</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # Compact actions directly below the overview.
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
        '<div class="notif-section-head"><div class="notif-section-kicker">02 · DEFAULTS</div>'
        '<div class="notif-section-title">共通の検知条件</div>'
        '<div class="notif-section-copy">個別指定のない銘柄に適用される基本ルールです。</div></div>',
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
        '<div class="notif-section-head"><div class="notif-section-kicker">03 · ASSETS</div>'
        '<div class="notif-section-title">銘柄別の通知ルール</div>'
        '<div class="notif-section-copy">検索して銘柄を開き、条件を追加・調整できます。</div></div>',
        unsafe_allow_html=True,
    )

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

                trendlines: list[dict] = t_cfg.setdefault("trendlines", [])
                date_alerts: list[dict] = t_cfg.setdefault("date_alerts", [])
                notifications_enabled = t_cfg.get("notifications_enabled", True)

                ticker_alert_count = len(price_alerts) + len(trendlines) + len(date_alerts)
                state_icon = "🟢" if notifications_enabled else "⚫"
                label = f"{t}　{name}" if name else t
                expander_label = (
                    f"{state_icon}  {label}　｜ "
                    f"週足 {'ON' if w_val else 'OFF'} · 日足 {'ON' if d_val else 'OFF'}　｜ "
                    f"条件 {ticker_alert_count}"
                )

                with st.expander(expander_label, expanded=False):
                    new_notifications_enabled = st.toggle(
                        "🔔 通知を有効にする",
                        value=notifications_enabled,
                        key=f"notify_chk_{t}",
                        help="この銘柄のすべてのアラート通知を有効にします"
                    )
                    if new_notifications_enabled != notifications_enabled:
                        t_cfg["notifications_enabled"] = new_notifications_enabled
                        save_config(config)
                        st.rerun()

                    if not new_notifications_enabled:
                        st.info("この銘柄の通知はOFFになっています。設定内容は保持されますが、アラートは発火しません。")

                    # MACD Settings
                    st.markdown(
                        "<div class='notif-subhead'>MACD 自動検知</div>",
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
                        "<div class='notif-rule'></div>",
                        unsafe_allow_html=True,
                    )

                    # Price Alerts
                    st.markdown(
                        "<div class='notif-subhead'>価格アラート</div>",
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

                    st.markdown(
                        "<div class='notif-rule'></div>",
                        unsafe_allow_html=True,
                    )

                    # Date Alerts
                    st.markdown(
                        "<div class='notif-subhead'>日付アラート</div>",
                        unsafe_allow_html=True,
                    )
                    if date_alerts:
                        for i, da in enumerate(date_alerts):
                            col_info, col_del = st.columns([8, 2])
                            col_info.markdown(
                                f"<div style='font-size: 0.85rem; padding-top: 4px;'>"
                                f"`{da['date']}` ({da.get('label', '')})"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            if col_del.button(
                                "🗑️ 削除",
                                key=f"del_da_{t}_{i}",
                                use_container_width=True,
                            ):
                                date_alerts.pop(i)
                                save_config(config)
                                st.toast(f"{t} の日付アラートを削除しました。")
                                st.rerun()
                    else:
                        st.caption("設定されている日付アラートはありません。")

                    with st.form(key=f"add_date_alert_{t}", clear_on_submit=True):
                        col_f1, col_f2, col_f3 = st.columns([4, 5, 3])
                        new_da_date = col_f1.date_input("日付", key=f"inp_da_date_{t}")
                        new_da_label = col_f2.text_input("ラベル (決算日など)", key=f"inp_da_label_{t}")
                        submit_da = col_f3.form_submit_button("追加")

                        if submit_da:
                            date_alerts.append({"date": new_da_date.isoformat(), "label": new_da_label})
                            save_config(config)
                            st.toast(f"{t} に日付アラートを追加しました。")
                            st.rerun()

                    st.markdown(
                        "<div class='notif-rule'></div>",
                        unsafe_allow_html=True,
                    )

                    # Trendline Alerts
                    st.markdown(
                        "<div class='notif-subhead'>ラインアラート</div>",
                        unsafe_allow_html=True,
                    )
                    if trendlines:
                        for i, tl in enumerate(trendlines):
                            col_info, col_del = st.columns([8, 2])
                            col_info.markdown(
                                f"<div style='font-size: 0.85rem; padding-top: 4px;'>"
                                f"ライン {i+1}: `{tl.get('x0', '')[:10]}` から `{tl.get('x1', '')[:10]}`"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            if col_del.button(
                                "🗑️ 削除",
                                key=f"del_tl_{t}_{i}",
                                use_container_width=True,
                            ):
                                trendlines.pop(i)
                                save_config(config)
                                st.toast(f"{t} のラインアラートを削除しました。")
                                st.rerun()
                    else:
                        st.caption("チャート上に引かれたラインはありません。")

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
