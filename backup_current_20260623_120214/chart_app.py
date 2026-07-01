"""Streamlit chart analyzer — candles + indicators + auto-detected trendlines.

Launch with::

    /c/Users/matsu/anaconda3/python.exe -m streamlit run chart_app.py
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from data import get_ticker_name, load_ohlcv, normalise_ticker
from evaluator import EvalParams, evaluate_lines
from indicators import ichimoku, macd, rsi, sma, volume_ma
from git_utils import git_push_changes
from trendlines import (
    Line,
    TrendParams,
    detect_all_lines_multiscale,
)
from universe import INDICES, UNIVERSE, all_names

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
PARAM_PATH = ARTIFACTS / "trend_params.json"  # legacy daily params (back-compat)
FAVORITES_PATH = ARTIFACTS / "favorites.json"


def _param_path_for(interval: str) -> Path:
    """Resolve the tuned-params JSON file for a given bar interval.

    Daily keeps the legacy ``trend_params.json`` filename so any older
    tooling that doesn't know about intervals still finds it. Other
    intervals get a suffixed file written by ``tuner.py --interval ...``.
    """
    if interval == "1d":
        return PARAM_PATH
    return ARTIFACTS / f"trend_params_{interval}.json"


# Sidebar display labels for each supported interval, plus metadata that
# drives the scale-selector wording ("短期 (~4ヶ月, lookback 90)" means
# something different on 1wk vs 1d).
INTERVAL_LABELS: dict[str, str] = {
    "1d": "日足",
    "1wk": "週足",
}

# Human-readable lookback hints used by the scale checkboxes. Kept in
# sync with trendlines.INTERVAL_SCALE_PROFILES so the sidebar copy
# doesn't drift from the actual detection windows.
SCALE_HINTS: dict[str, dict[str, str]] = {
    "1d": {
        "short": "短期 (~4ヶ月, lookback 90)",
        "mid":   "中期 (~1年, lookback 240)",
        "long":  "長期 (~2年, lookback 500)",
    },
    "1wk": {
        "short": "短期 (~1年, lookback 52週)",
        "mid":   "中期 (~3年, lookback 156週)",
        "long":  "長期 (~5年, lookback 260週)",
    },
}

# ---------------------------------------------------------------------------
# Styling — 京都ターミナル (Kyoto Terminal) theme
# ---------------------------------------------------------------------------
# Direction chosen via the anthropics/frontend-design skill (see skills.md):
#   Japanese editorial × financial terminal. Washi paper, sumi ink,
#   shu vermilion accent. Distinctive — intentionally avoids generic
#   SaaS blue-on-white / Inter-on-white aesthetics.
THEME = {
    "paper":         "#0F172A",
    "paper_alt":     "#1E293B",
    "surface":       "rgba(30,41,59,0.7)",
    "ink":           "#F1F5F9",
    "ink_soft":      "#CBD5E1",
    "ink_muted":     "#64748B",
    "border":        "rgba(255,255,255,0.08)",
    "border_strong": "rgba(255,255,255,0.15)",
    "shu":           "#F472B6",
    "shu_deep":      "#BE185D",
    "forest":        "#34D399",
    "navy":          "#38BDF8",
    "copper":        "#A78BFA",
    "gold":          "#FBBF24",
}

LINE_COLORS = {
    "support":    THEME["forest"],  # horizontal floor
    "resistance": THEME["shu"],     # horizontal ceiling
    "trend_up":   THEME["navy"],    # 紺 rising
    "trend_down": THEME["copper"],  # aged gold falling
}
LINE_LABELS_JA = {
    "support": "サポート",
    "resistance": "レジスタンス",
    "trend_up": "上昇トレンド",
    "trend_down": "下降トレンド",
}
SCALE_LABELS_JA = {
    "short": "短期 (~4ヶ月)",
    "mid": "中期 (~1年)",
    "long": "長期 (~2年)",
}


# Inline CSS implementing the 京都ターミナル direction. The ``!important``
# flags override Streamlit's emotion-generated defaults that would
# otherwise leak back through the component tree.
#
# Type stack: Shippori Mincho (display serif) + IBM Plex Sans JP (body) +
# IBM Plex Mono (numerics), all loaded from Google Fonts.
THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
    --paper:         {THEME['paper']};
    --paper-alt:     {THEME['paper_alt']};
    --surface:       {THEME['surface']};
    --ink:           {THEME['ink']};
    --ink-soft:      {THEME['ink_soft']};
    --ink-muted:     {THEME['ink_muted']};
    --border:        {THEME['border']};
    --border-strong: {THEME['border_strong']};
    --shu:           {THEME['shu']};
    --shu-deep:      {THEME['shu_deep']};
    --forest:        {THEME['forest']};
    --navy:          {THEME['navy']};
    --copper:        {THEME['copper']};
    --gold:          {THEME['gold']};
    --font-display:  'Inter', sans-serif;
    --font-body:     'Inter', sans-serif;
    --font-mono:     ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}}

/* Base */
html, body, [class*="st-"] {{
    font-family: var(--font-body) !important;
    color: var(--ink) !important;
}}
.stApp {{
    background-color: var(--paper);
    background-image: 
      radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
      radial-gradient(at 50% 0%, hsla(225,39%,30%,0.15) 0, transparent 50%), 
      radial-gradient(at 100% 0%, hsla(339,49%,30%,0.1) 0, transparent 50%);
    background-attachment: fixed;
}}

h1, h2, h3, h4, h5, [data-testid="stMarkdownContainer"] h1 {{
    font-family: var(--font-display) !important;
    color: var(--ink);
    font-weight: 600;
}}
h1 {{ font-size: 2.25rem; font-weight: 700; letter-spacing: -0.025em; border: none !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: rgba(15,23,42,0.9) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] hr {{
    border-color: var(--border);
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: var(--navy) !important;
    font-size: 0.8rem !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-top: 1rem;
}}

/* Glassmorphism Cards */
[data-testid="stMetric"], .kt-metric-card, .kt-hero-card, [data-testid="stExpander"], [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{
    background: var(--surface) !important;
    backdrop-filter: blur(10px);
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06) !important;
    padding: 1.25rem !important;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    margin-bottom: 1rem;
}}
[data-testid="stMetric"]:hover, .kt-metric-card:hover, .kt-hero-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05) !important;
    border-color: var(--border-strong) !important;
}}

/* Typography inside cards */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p, .kt-metric-label {{ color: var(--ink-muted) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.05em; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div, .kt-metric-value {{ color: var(--ink) !important; font-size: 1.875rem !important; font-weight: 600 !important; font-family: var(--font-body) !important; }}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] div {{ font-family: var(--font-body) !important; }}
.kt-metric-note {{ color: var(--ink-muted); font-size: 0.75rem; margin-top: auto; padding-top: 0.5rem; }}

/* Hero Card Tweaks */
.kt-hero-card {{ padding: 1.5rem 1.8rem !important; }}
.kt-hero-title {{ font-size: 1.5rem; font-weight: 700; color: var(--ink); display: flex; align-items: baseline; gap: 0.6em; }}
.kt-hero-ticker {{ color: var(--navy); background: rgba(56,189,248,0.1); padding: 4px 10px; border-radius: 6px; font-size: 1.1rem; border: 1px solid rgba(56,189,248,0.2); font-family: var(--font-mono); }}
.kt-hero-price-row {{ display: flex; align-items: baseline; gap: 1em; margin-top: 1rem; }}
.kt-hero-price {{ font-size: 2.5rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.kt-hero-delta {{ padding: 2px 10px; border-radius: 6px; font-size: 1rem; font-weight: 500; font-variant-numeric: tabular-nums; }}
.kt-hero-delta.up {{ color: var(--forest); background: rgba(52,211,153,0.1); }}
.kt-hero-delta.down {{ color: var(--shu); background: rgba(244,114,182,0.1); }}
.kt-hero-delta.flat {{ color: var(--ink-muted); background: rgba(255,255,255,0.05); }}

/* Buttons */
.stButton > button, .stFormSubmitButton > button, button[kind="secondary"], button[kind="primary"] {{
    background: var(--surface) !important;
    background-color: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500;
    transition: all 0.2s;
    box-shadow: none !important;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover, button[kind="secondary"]:hover, button[kind="primary"]:hover {{
    background: rgba(56,189,248,0.15) !important;
    background-color: rgba(56,189,248,0.15) !important;
    border-color: var(--navy) !important;
    color: var(--navy) !important;
    transform: translateY(-1px);
}}
.stButton > button:active, .stFormSubmitButton > button:active, button[kind="secondary"]:active, button[kind="primary"]:active {{
    transform: translateY(1px);
}}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
div[data-baseweb="input"], div[data-baseweb="input"] > div, .stTextArea [data-baseweb="textarea"] {{
    background: rgba(15,23,42,0.6) !important;
    background-color: rgba(15,23,42,0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}}
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] div, .stSelectbox [data-baseweb="select"] span {{
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}}
.stTextInput input:focus, .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {{
    border-color: var(--navy) !important;
    box-shadow: 0 0 0 1px var(--navy) !important;
}}

/* Dropdowns */
[data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {{
    background-color: var(--paper-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}}
ul[role="listbox"] li, [role="option"] {{
    background-color: transparent !important;
    color: var(--ink) !important;
}}
ul[role="listbox"] li:hover, [role="option"]:hover {{
    background-color: rgba(56,189,248,0.15) !important;
    color: var(--navy) !important;
}}

/* Checkbox/Radio */
[data-testid="stCheckbox"] label p, div[role="radiogroup"] label p {{ color: var(--ink); font-weight: 500; }}

/* Expanders */
[data-testid="stExpander"] details > summary {{ font-weight: 600; color: var(--ink); }}

/* Alerts */
[data-testid="stAlert"] {{
    background-color: rgba(30,41,59,0.7) !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--navy) !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
}}

/* Hide Sidebar button */
[data-testid="stSidebarCollapseButton"], button[aria-label="Close sidebar"] {{ display: none !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border-strong); border-radius: 4px; border: none; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--ink-muted); }}

/* Cleanup artifacts of old theme */
.kt-masthead::before, .kt-masthead::after {{ display: none; }}
.kt-masthead-kicker, .kt-masthead-byline, .kt-masthead-meta, .kt-masthead-sep {{ display: none; }}
.kt-subtitle, .kt-subtitle-line, .kt-section-label {{ display: none; }}
hr {{ border-top: 1px solid var(--border); }}

/* Fix Light Gray Areas */
[data-testid="stHeader"] {{
    background-color: transparent !important;
}}
.stApp > header {{
    background-color: transparent !important;
}}
[data-testid="stToolbar"] {{
    right: 2rem;
}}
/* Ensure the main background completely covers everything */
.stApp, .main {{
    background-color: var(--paper) !important;
}}


/* Fix Tabs Visibility */
[data-baseweb="tab-list"] {{
    background-color: transparent !important;
    gap: 8px;
}}
button[data-baseweb="tab"] {{
    background-color: rgba(30,41,59,0.5) !important;
    border: 1px solid var(--border) !important;
    border-bottom: 0 !important;
    color: var(--ink-muted) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background-color: rgba(56,189,248,0.15) !important;
    border-color: var(--navy) !important;
    color: var(--navy) !important;
}}
button[data-baseweb="tab"]:hover {{
    color: var(--ink) !important;
    background-color: rgba(30,41,59,0.8) !important;
}}
[data-testid="stTab"] {{
    background-color: transparent !important;
}}


/* ---- Custom Styling for st.radio (Selection Tabs) ---- */
div[role="radiogroup"] {{
    display: flex;
    gap: 4px;
    background: rgba(15,23,42,0.6);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid var(--border);
    flex-wrap: wrap;
}}
div[role="radiogroup"] > label {{
    flex: 1;
    min-width: 60px;
    text-align: center;
    background: transparent;
    padding: 8px 4px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
    margin: 0 !important;
    display: flex;
    justify-content: center;
    align-items: center;
}}
div[role="radiogroup"] > label:hover {{
    background: rgba(56,189,248,0.1);
}}
div[role="radiogroup"] > label > div:first-child {{
    display: none !important; /* Hide native radio circles */
}}
div[role="radiogroup"] > label p {{
    color: var(--ink-muted);
    font-weight: 600;
    font-size: 0.8rem;
    margin: 0;
    transition: color 0.2s;
}}
div[role="radiogroup"] > label:has(input:checked) {{
    background: rgba(56,189,248,0.2) !important;
    border-radius: 6px;
}}
div[role="radiogroup"] > label:has(input:checked) p {{
    color: var(--navy) !important;
    text-shadow: 0 0 10px rgba(56,189,248,0.3);
}}

/* Also style selectboxes for dark theme properly */
[data-baseweb="select"] {{
    cursor: pointer;
}}


</style>
"""

# Plotly font stack — mirrors --font-body so charts typographically align
# with the surrounding Streamlit chrome.
PLOTLY_FONT_FAMILY = (
    "'IBM Plex Sans JP', 'Hiragino Sans', 'Noto Sans CJK JP', sans-serif"
)
PLOTLY_FONT_MONO = (
    "'IBM Plex Mono', 'Menlo', monospace"
)


# ---------------------------------------------------------------------------
# Favorites persistence (stored as {ticker: display_name})
# ---------------------------------------------------------------------------
def load_favorites() -> dict[str, str]:
    """Load favorites as ``{ticker: name}``. Migrates legacy list format
    and normalises 4-digit codes to ``XXXX.T`` so name lookups work."""
    if not FAVORITES_PATH.exists():
        return {}
    try:
        raw = json.loads(FAVORITES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, list):
        items = ((t, "") for t in raw)
    elif isinstance(raw, dict):
        items = ((str(k), str(v) if v is not None else "") for k, v in raw.items())
    else:
        return {}
    out: dict[str, str] = {}
    for t, name in items:
        nt = normalise_ticker(t)
        if nt and nt not in out:
            out[nt] = name
    return out


def save_favorites(favs: dict[str, str], *, push: bool = True) -> None:
    FAVORITES_PATH.parent.mkdir(exist_ok=True)
    FAVORITES_PATH.write_text(
        json.dumps(favs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if push:
        git_push_changes("Update favorites list via UI")


def add_favorite(ticker: str, name: str = "", *, push: bool = True) -> None:
    favs = load_favorites()
    ticker = normalise_ticker(ticker)
    if not ticker:
        return
    # Don't overwrite an existing non-empty name with a blank one
    favs[ticker] = name or favs.get(ticker, "")
    save_favorites(favs, push=push)


def remove_favorite(ticker: str, *, push: bool = True) -> None:
    favs = load_favorites()
    ticker = normalise_ticker(ticker)
    if ticker in favs:
        del favs[ticker]
        save_favorites(favs, push=push)


def sync_favorite_and_notifications(
    ticker: str, name: str, *, remove: bool = False,
) -> None:
    """Persist a favorite and its notification entry with one Git push."""
    from notification_ui import load_config, save_config

    ticker = normalise_ticker(ticker)
    if remove:
        # Keep existing alert settings so re-adding the ticker restores them.
        remove_favorite(ticker, push=False)
    else:
        add_favorite(ticker, name, push=False)
        config = load_config()
        defaults = config.get("global_defaults", {})
        ticker_cfg = config.setdefault("tickers", {}).setdefault(ticker, {})
        ticker_cfg.setdefault("notification_mode", "all")
        ticker_cfg.setdefault("notifications_enabled", True)
        ticker_cfg.setdefault("weekly_macd_cross", defaults.get("weekly_macd_cross", True))
        ticker_cfg.setdefault("daily_macd_cross", defaults.get("daily_macd_cross", False))
        ticker_cfg.setdefault("price_alert", False)
        ticker_cfg.setdefault("price_alerts", [])
        ticker_cfg.setdefault("date_alerts", [])
        ticker_cfg.setdefault("trendline_alert", defaults.get("trendline_alert", True))
        ticker_cfg.setdefault("trendlines", [])
        save_config(config, push=False)

    git_push_changes(
        "Remove favorite via UI" if remove
        else "Add favorite and notification settings via UI"
    )


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_tuned_params(interval: str = "1d") -> tuple[TrendParams, EvalParams]:
    """Load the tuned TrendParams/EvalParams for a bar interval.

    Each interval has its own JSON file produced by
    ``tuner.py --interval <x>``. Missing files (e.g. the user picks an
    interval we haven't tuned yet) fall through to the dataclass
    defaults, which at least keep the app functional.
    """
    path = _param_path_for(interval)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Filter out unknown fields in case the saved JSON pre-dates a
        # newer dataclass definition (forward/backward compatibility).
        tp_keys = {f.name for f in fields(TrendParams)}
        ep_keys = {f.name for f in fields(EvalParams)}
        return (
            TrendParams(**{k: v for k, v in raw["trend_params"].items() if k in tp_keys}),
            EvalParams(**{k: v for k, v in raw["eval_params"].items() if k in ep_keys}),
        )
    return TrendParams(), EvalParams()


@st.cache_data(ttl=3600, show_spinner=False)
def load_chart_data(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    return load_ohlcv(ticker, auto_download=True, interval=interval)


def get_default_ticker() -> str:
    """Default chart ticker used before the user picks one in the right panel."""
    return "7203.T"


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_last_prices(
    codes: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    """Return a startup snapshot of latest close and percentage change.

    Fetch all rows together so tickers without a local OHLCV cache are not
    left blank.  The caller stores this result in session state, therefore the
    network request runs only once per app session and not on ticker changes.
    """
    out: dict[str, dict[str, float | None]] = {}
    pending = set(codes)

    try:
        import yfinance as yf

        quotes = yf.download(
            list(codes),
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        for code in codes:
            try:
                if isinstance(quotes.columns, pd.MultiIndex):
                    close = pd.to_numeric(quotes[code]["Close"], errors="coerce").dropna()
                else:
                    close = pd.to_numeric(quotes["Close"], errors="coerce").dropna()
                if close.empty:
                    continue
                last = float(close.iloc[-1])
                pct = 0.0
                if len(close) >= 2 and float(close.iloc[-2]) != 0.0:
                    pct = (last / float(close.iloc[-2]) - 1.0) * 100.0
                out[code] = {"last": last, "pct": pct}
                pending.discard(code)
            except (KeyError, TypeError, ValueError):
                continue
    except Exception as exc:
        print(f"[chart_app] startup quote download failed: {exc}")

    # Fall back to the local cache for individual symbols the batch provider
    # could not return (market-specific indices, temporary API errors, etc.).
    for code in pending:
        path = ROOT / "cache" / (code.replace(".", "_").replace("^", "_") + ".parquet")
        try:
            if path.exists():
                try:
                    df = pd.read_parquet(path, columns=["Close"])
                except Exception:
                    df = pd.read_parquet(path)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df[["Close"]] if "Close" in df.columns else None
            else:
                df = None
        except Exception:
            df = None

        if df is None or df.empty or "Close" not in df.columns:
            out.setdefault(code, {"last": None, "pct": None})
            continue

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if close.empty:
            out.setdefault(code, {"last": None, "pct": None})
            continue

        last = float(close.iloc[-1])
        pct = 0.0
        if len(close) >= 2 and float(close.iloc[-2]) != 0.0:
            pct = (last / float(close.iloc[-2]) - 1.0) * 100.0
        out[code] = {"last": last, "pct": pct}
    return out


def _line_to_dict(line: Line) -> dict:
    return {
        "kind": line.kind,
        "start_date": line.start_date.isoformat(),
        "start_price": float(line.start_price),
        "end_date": line.end_date.isoformat(),
        "end_price": float(line.end_price),
        "touches": [t.isoformat() for t in line.touches],
        "score": float(line.score),
        "valid": line.valid,
        "scale": line.scale,
    }


def _dict_to_line(d: dict) -> Line:
    return Line(
        kind=d["kind"],
        start_date=pd.Timestamp(d["start_date"]),
        start_price=d["start_price"],
        end_date=pd.Timestamp(d["end_date"]),
        end_price=d["end_price"],
        touches=[pd.Timestamp(t) for t in d["touches"]],
        score=d["score"],
        valid=d["valid"],
        scale=d["scale"],
    )


def _df_hash(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return ""
    # Quick stable hash of df shape, last index timestamp and last close price
    return f"{df.shape}_{df.index[-1]}_{df['Close'].iloc[-1]}"


@st.cache_data(show_spinner="トレンドライン検出中...", ttl=3600)
def _cached_detect_and_evaluate(
    _df_full_hash: str,
    ticker: str,
    interval: str,
    pivot_window: int,
    tolerance_pct: float,
    min_touches: int,
    max_slope_annual: float,
    lookback_bars: int,
    max_lines_per_kind: int,
    min_span_days: int,
    max_last_touch_age_days: int,
    selected_scales: tuple[str, ...],
    forward_bars: int,
    tolerance_eval: float,
) -> list[dict]:
    """Cached trendline detection and evaluation."""
    from trendlines import TrendParams, detect_all_lines_multiscale
    from evaluator import EvalParams, evaluate_lines

    df = load_chart_data(ticker, interval)
    if df is None or df.empty:
        return []

    params = TrendParams(
        pivot_window=pivot_window,
        tolerance_pct=tolerance_pct,
        min_touches=min_touches,
        max_slope_annual=max_slope_annual,
        lookback_bars=lookback_bars,
        max_lines_per_kind=max_lines_per_kind,
        min_span_days=min_span_days,
        max_last_touch_age_days=max_last_touch_age_days,
    )
    eval_params = EvalParams(
        forward_bars=forward_bars,
        tolerance_pct=tolerance_eval,
    )

    lines = detect_all_lines_multiscale(
        df, params, scales=list(selected_scales), interval=interval
    )
    if lines:
        evaluate_lines(lines, df, eval_params)

    return [_line_to_dict(l) for l in lines]


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_historical_per_series(
    ticker: str, interval: str, n_bars: int
) -> tuple[pd.Series | None, str]:
    """Cached historical PER series calculation."""
    df = load_chart_data(ticker, interval)
    if df is None or df.empty:
        return None, ""
    return _historical_per_series(ticker, df)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker: str) -> dict:
    """Fetch trailing PER / PBR / dividend yield from yfinance.

    Cached for an hour to avoid hammering yfinance on every rerun. Returns
    an empty dict on any failure (so the metrics simply show as ``—``).

    We deliberately pull the raw inputs (``trailingEps``, ``dividendRate``)
    in addition to the pre-computed ratios so the caller can distinguish:

    * 赤字 (EPS ≤ 0) — PER is mathematically meaningless; yfinance
      sometimes still returns a stale ``trailingPE``, so we ignore it.
    * 無配 (dividendRate == 0 / None) — should render as "無配" rather
      than "0.00%" or "—", so the user knows it's a *confirmed* non-payer
      rather than missing data.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(normalise_ticker(ticker)).info or {}
    except Exception as e:
        print(f"[chart_app] fundamentals lookup failed for {ticker}: {e}")
        return {"fetched_at": pd.Timestamp.now()}
    return {
        "per": info.get("trailingPE"),
        "trailing_eps": info.get("trailingEps"),
        "forward_per": info.get("forwardPE"),
        "forward_eps": info.get("forwardEps"),
        "pbr": info.get("priceToBook"),
        "dividend_yield": info.get("dividendYield"),
        "dividend_rate": info.get("dividendRate"),
        "trailing_dividend_rate": info.get("trailingAnnualDividendRate"),
        "trailing_dividend_yield": info.get("trailingAnnualDividendYield"),
        "fetched_at": pd.Timestamp.now(),
    }


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _latest_forecast_eps(ticker: str) -> tuple[float, pd.Timestamp, str] | None:
    """Return the most recent company-forecast EPS for the current fiscal year.

    Walks ``/fins/summary`` from newest to oldest and picks the first
    usable forecast:

    - On a Q1/Q2/Q3 決算短信, ``FEPS`` is the current-year guidance (may
      be a mid-year revision of the number first issued at the prior FY).
    - On an FY 決算短信, ``NxFEPS`` is the next-year guidance published
      alongside the realised FY result.

    Values are split-adjusted to post-split units so dividing the current
    (split-adjusted) close by the returned EPS yields a PER aligned with
    Yahoo Finance Japan's 予想PER display. Returns ``None`` when no
    usable forecast is found (J-Quants unavailable, non-JP ticker, or
    every eligible filing carries a blank / unparseable forecast).

    The returned tuple is ``(eps, disclosure_date, period_label)`` where
    ``period_label`` describes which filing contributed the number, so
    the UI can surface e.g. "会社予想 (2Q 時点)".
    """
    try:
        import jquants
    except Exception as e:
        print(f"[chart_app] jquants import failed: {e}")
        return None
    if not jquants.is_configured():
        return None
    try:
        raw = jquants.get_fins_summary(ticker)
    except Exception as e:
        print(f"[chart_app] J-Quants summary fetch failed for {ticker}: {e}")
        return None
    if raw is None or raw.empty:
        return None

    df = raw.copy()
    df["DiscDate"] = pd.to_datetime(df.get("DiscDate"), errors="coerce")
    df = df.dropna(subset=["DiscDate"])
    # Restrict to 決算短信 filings (the FEPS/NxFEPS fields on standalone
    # forecast-revision notices are harder to pair with the right period).
    doc = df.get("DocType", pd.Series([""] * len(df))).astype(str)
    df = df[doc.str.contains("FinancialStatements", na=False)].copy()
    if df.empty:
        return None

    df["FEPS"] = pd.to_numeric(df.get("FEPS"), errors="coerce")
    df["NxFEPS"] = pd.to_numeric(df.get("NxFEPS"), errors="coerce")

    # Apply forward-split adjustment to both forecast columns so filings
    # dated before a split are rescaled into post-split share-count units.
    splits = _yf_splits(ticker)
    if splits is not None and not splits.empty:
        for ex_date, ratio in splits.items():
            mask = df["DiscDate"] < ex_date
            if mask.any():
                for col in ("FEPS", "NxFEPS"):
                    df.loc[mask, col] = df.loc[mask, col] / ratio

    df = df.sort_values("DiscDate")
    # Newest filing first — we want the freshest forecast available.
    for _, row in df.iloc[::-1].iterrows():
        cur_per = str(row.get("CurPerType", ""))
        if cur_per == "FY":
            val = row.get("NxFEPS")
            label = "会社予想 (次期)"
        elif cur_per in ("1Q", "2Q", "3Q"):
            val = row.get("FEPS")
            label = f"会社予想 ({cur_per} 時点)"
        else:
            continue
        if val is None or pd.isna(val):
            continue
        return float(val), pd.Timestamp(row["DiscDate"]), label
    return None


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _load_fy_eps_steps(ticker: str) -> tuple[pd.DataFrame | None, str]:
    """Return an annual EPS step-function frame for a ticker.

    Each row is one 本決算 (FY annual report) from J-Quants ``/fins/summary``:

    - ``DiscDate`` — the day the market learned both the realised FY EPS
      and the company's guidance for the next FY.
    - ``actual_eps`` — realised full-year EPS (Q1+Q2+Q3+Q4 sum,
      straight from the 決算短信 headline).
    - ``forecast_eps`` — ``NxFEPS`` from the same filing: the company's
      own guidance for the fiscal year that starts right after.

    Both EPS columns are pre-split-adjusted so they're in the same unit
    as yfinance's split-adjusted close prices. The returned label
    documents the data source for the chart subtitle.

    No yfinance fallback: if J-Quants isn't configured or the ticker
    is outside its coverage, returns ``(None, "")``.
    """
    try:
        import jquants
    except Exception as e:
        print(f"[chart_app] jquants import failed: {e}")
        return None, ""
    if not jquants.is_configured():
        return None, ""
    try:
        fy = jquants.fy_guidance(ticker)
    except Exception as e:
        print(f"[chart_app] J-Quants fy_guidance failed for {ticker}: {e}")
        return None, ""
    if fy is None or fy.empty:
        return None, ""
    fy = _split_adjust_fy_eps(fy, ticker)
    return fy, "J-Quants 本決算"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _yf_splits(ticker: str) -> pd.Series | None:
    """Historical split ratios for a ticker (yfinance ``Ticker.splits``).

    Returns a Series indexed by split *ex-date*, where each value is the
    ratio (e.g. 5.0 for a 1-for-5 forward split). Empty / None on failure.
    """
    try:
        import yfinance as yf
        s = yf.Ticker(normalise_ticker(ticker)).splits
    except Exception as e:
        print(f"[chart_app] yfinance splits lookup failed for {ticker}: {e}")
        return None
    if s is None or s.empty:
        return None
    s = s[s > 0].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index()


def _split_adjust_fy_eps(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Rescale pre-split EPS and NxFEPS on FY rows to post-split units.

    Both the realised full-year EPS and the next-FY company forecast
    are quoted in the share-count regime that existed *at disclosure
    time*. A filing dated before a forward split therefore needs to
    be divided by the split ratio so the denominator of PER lines up
    with yfinance's split-adjusted Close series. Applies to every
    split whose ex-date is after the filing date.
    """
    splits = _yf_splits(ticker)
    if splits is None or splits.empty or df is None or df.empty:
        return df
    out = df.copy()
    disc = pd.to_datetime(out["DiscDate"])
    for ex_date, ratio in splits.items():
        mask = disc < ex_date
        if mask.any():
            for col in ("EPS", "NxFEPS"):
                if col in out.columns:
                    out.loc[mask, col] = out.loc[mask, col] / ratio
    return out


def _historical_per_series(
    ticker: str, df: pd.DataFrame
) -> tuple[pd.Series | None, str]:
    """Return ``(per_series, source_label)`` aligned to ``df.index``.

    PER is computed as a step function that changes only at 本決算
    (annual report) disclosure dates — matching how Yahoo Finance
    Japan displays 予想PER / 実績PER. For each FY filing we know two
    numbers:

    - ``actual_eps``: the realised full-year EPS just reported
      (Q1+Q2+Q3+Q4 sum).
    - ``forecast_eps``: the company's own guidance for the *next*
      fiscal year, announced on the same day.

    The denominator for each daily bar is picked from the most recent
    FY filing at or before that date:

    - If there is a *later* FY filing, we use ``actual_eps`` from the
      current-segment filing — i.e. the fiscal year those results cover
      has already been fully reported; PER = price ÷ historical 実績 EPS.
    - If this is the *most recent* segment (no later FY filing exists),
      we switch to ``forecast_eps`` from that filing — the company's own
      guidance for the fiscal year currently in progress. This is the
      "直近1年は会社予想 EPS で計算" segment.

    Dates before the earliest FY filing have no denominator and are
    left NA. Non-positive denominators (赤字 or 無予想) are masked.
    """
    fy, source = _load_fy_eps_steps(ticker)
    if fy is None or fy.empty:
        return None, ""

    fy = fy.sort_values("DiscDate").reset_index(drop=True)
    # Pick the EPS denominator for each segment: realised EPS while a
    # newer FY exists, company guidance on the trailing edge. If the
    # company does not publish guidance (e.g. SoftBank Group) fall
    # back to realised EPS for the trailing segment — that mirrors
    # Yahoo Finance Japan's "実績PER" display for non-guiding issuers.
    denoms: list[tuple[pd.Timestamp, float]] = []
    for i, row in fy.iterrows():
        is_latest = (i == len(fy) - 1)
        val = row.get("NxFEPS") if is_latest else None
        if val is None or pd.isna(val):
            val = row.get("EPS")
        if pd.notna(val):
            denoms.append((pd.Timestamp(row["DiscDate"]), float(val)))

    if not denoms:
        return None, ""

    eps_step = pd.Series({d: v for d, v in denoms}).sort_index()
    combined_idx = df.index.union(eps_step.index)
    eps_ff = eps_step.reindex(combined_idx).ffill().reindex(df.index)
    per = df["Close"] / eps_ff
    per = per.where(eps_ff > 0)
    per = per.replace([float("inf"), float("-inf")], pd.NA)
    return per, source


def _format_financial_value(val) -> str:
    """Yen 生データを `兆円` または `億円` に変換"""
    if pd.isna(val) or val == "" or val is None:
        return "—"
    try:
        val_float = float(val)
        okuen = val_float / 100_000_000
        if abs(okuen) >= 10_000:
            return f"{okuen / 10_000:,.2f} 兆円"
        return f"{okuen:,.0f} 億円"
    except Exception:
        return "—"


def _format_eps_value(val) -> str:
    if pd.isna(val) or val == "" or val is None:
        return "—"
    try:
        return f"{float(val):,.2f} 円"
    except Exception:
        return "—"


def _format_pct_value(val) -> str:
    if pd.isna(val) or val == "" or val is None:
        return "—"
    try:
        return f"{float(val) * 100:.1f}%"
    except Exception:
        return "—"


def _is_valid_alert_trendline(line: dict) -> bool:
    """Reject Plotly helper shapes accidentally returned as user lines."""
    x0, x1 = line.get("x0"), line.get("x1")
    if not x0 or not x1 or str(x0) == str(x1):
        return False
    if pd.isna(pd.to_datetime(x0, errors="coerce")):
        return False
    if pd.isna(pd.to_datetime(x1, errors="coerce")):
        return False
    try:
        float(line.get("y0"))
        float(line.get("y1"))
    except (TypeError, ValueError):
        return False
    return True


def _render_fundamentals_section(
    ticker: str, last: float, interval: str, latest_bar: pd.Timestamp,
) -> None:
    """Render fundamentals after the chart has already reached the UI."""
    fund = load_fundamentals(ticker)
    bar_unit = "営業日" if interval == "1d" else "週"
    fetched_at = fund.get("fetched_at")
    fetched_str = (
        fetched_at.strftime("%Y年%m月%d日 %H:%M")
        if isinstance(fetched_at, pd.Timestamp) else "—"
    )
    st.caption(
        f"📅 チャート・テクニカル指標の最新{bar_unit}: "
        f"**{latest_bar:%Y年%m月%d日}** ／ "
        f"ファンダメンタルズ取得時刻: {fetched_str} (yfinance, 1時間キャッシュ)"
    )
    st.markdown(
        '<div class="kt-section-label">ファンダメンタルズ — 指標</div>',
        unsafe_allow_html=True,
    )
    f1, f2, f3 = st.columns(3)

    forecast = _latest_forecast_eps(ticker)
    per_val: float | None = None
    per_is_loss = False
    if forecast is not None:
        fcst_eps, _fcst_date, _fcst_label = forecast
        if fcst_eps <= 0:
            per_is_loss = True
        else:
            per_val = float(last) / fcst_eps
    else:
        yf_fwd_eps = fund.get("forward_eps")
        yf_fwd_per = fund.get("forward_per")
        if yf_fwd_eps is not None and yf_fwd_eps <= 0:
            per_is_loss = True
        elif yf_fwd_per and yf_fwd_per > 0:
            per_val = float(yf_fwd_per)
    if per_is_loss:
        per_value_html = "赤字"
    elif per_val is not None and per_val > 0:
        per_value_html = f"{per_val:.1f} 倍"
    else:
        per_value_html = "—"
    f1.markdown(
        f'''<div class="kt-metric-card">
        <div class="kt-metric-label">PER</div>
        <div class="kt-metric-value">{per_value_html}</div>
        <div class="kt-metric-note">※ 会社予想EPS基準のため他ツール (Yahoo等) と数値が異なる場合があります</div>
        </div>''',
        unsafe_allow_html=True,
    )

    pbr = fund.get("pbr")
    pbr_value_html = f"{pbr:.2f} 倍" if pbr and pbr > 0 else "—"
    f2.markdown(
        f'''<div class="kt-metric-card">
        <div class="kt-metric-label">PBR</div>
        <div class="kt-metric-value">{pbr_value_html}</div>
        </div>''',
        unsafe_allow_html=True,
    )

    annual_dps: float | None = None
    for candidate in (fund.get("dividend_rate"), fund.get("trailing_dividend_rate")):
        if candidate is not None:
            try:
                annual_dps = float(candidate)
            except (TypeError, ValueError):
                continue
            break
    if annual_dps is None:
        div_value_html = "—"
    elif annual_dps == 0:
        div_value_html = "無配"
    else:
        div_value_html = f"{annual_dps / float(last) * 100:.2f}%"
    f3.markdown(
        f'''<div class="kt-metric-card">
        <div class="kt-metric-label">配当利回り</div>
        <div class="kt-metric-value">{div_value_html}</div>
        </div>''',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------
def build_figure(
    df: pd.DataFrame,
    lines: list[Line],
    *,
    show_sma: bool,
    show_sma100: bool,
    show_sma200: bool,
    show_ichimoku: bool,
    show_support: bool,
    show_resistance: bool,
    show_trend_up: bool,
    show_trend_down: bool,
    show_invalid: bool,
    ath: tuple[float, pd.Timestamp] | None = None,
    atl: tuple[float, pd.Timestamp] | None = None,
    initial_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ui_revision: str | None = None,
    ticker_label: str,
    currency_symbol: str = "¥",
    show_volume: bool = True,
    show_macd: bool = True,
    show_rsi: bool = True,
    show_per: bool = False,
    per_series: pd.Series | None = None,
    per_source: str = "",
    interval: str = "1d",
) -> tuple[go.Figure, dict[str, int]]:
    """Build the chart figure with price panel + optional oscillator rows.

    ``df`` should be the plotted OHLCV window plus enough warmup bars for
    indicators. Keeping it bounded avoids sending years of unused points to
    Plotly on every rerun.

    The oscillator panels (出来高 / MACD / ヒストリカル PER) are each
    optional and rendered in the order enabled. Row heights are
    distributed so the price panel keeps ~58% when any oscillator is
    shown and 100% when none is.
    """
    close = df["Close"]

    # Calculate visible price range for setting y-axis range (visible window +/- 10%)
    visible_df = df
    if initial_range is not None:
        start_date, end_date = initial_range
        visible_df = df.loc[start_date:end_date]
        if visible_df.empty:
            visible_df = df

    visible_high = float(visible_df["High"].max())
    visible_low = float(visible_df["Low"].min())
    diff = visible_high - visible_low
    if diff <= 0:
        padding = visible_high * 0.1 if visible_high > 0 else 10.0
    else:
        padding = diff * 0.1

    y_min = max(0.0, visible_low - padding)
    y_max = visible_high + padding

    # --- Decide which oscillator rows to render ----------------------------
    has_per = (
        show_per
        and per_series is not None
        and not per_series.dropna().empty
    )
    oscillators: list[tuple[str, str]] = []
    if show_volume:
        oscillators.append(("volume", "出来高"))
    if show_macd:
        oscillators.append(("macd", "MACD"))
    if show_rsi:
        oscillators.append(("rsi", "RSI"))
    if has_per:
        per_title = "ヒストリカル PER"
        if per_source:
            per_title += f" ({per_source})"
        oscillators.append(("per", per_title))

    n_osc = len(oscillators)
    total_rows = 1 + n_osc
    if n_osc == 0:
        row_heights = [1.0]
    else:
        # Price keeps ~0.58; remaining height split evenly across oscillators.
        main_h = 0.58
        osc_h = (1.0 - main_h) / n_osc
        row_heights = [main_h] + [osc_h] * n_osc

    subplot_titles = tuple([""] + [label for _, label in oscillators])
    osc_rows: dict[str, int] = {
        name: i + 2 for i, (name, _) in enumerate(oscillators)
    }

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # --- Row 1: Candlestick + MAs + Trendlines + Ichimoku -------------------
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="ローソク足",
            # JP convention: 陽線 (up) = 朱 vermilion, 陰線 (down) = 深緑
            increasing_line_color=THEME["shu"],
            decreasing_line_color=THEME["forest"],
            increasing_fillcolor=THEME["shu"],
            decreasing_fillcolor=THEME["forest"],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # SMA overlays — short bundle (5/25/75) is one toggle; 100 and 200
    # are independent toggles so users can surface swing / long-term
    # reference lines without the near-term noise.
    sma_specs: list[tuple[int, str]] = []
    if show_sma:
        sma_specs.extend([
            (5, THEME["navy"]),       # 短期 = 紺 (short pulse)
            (25, THEME["copper"]),    # 中期 = 銅 (mid rhythm)
            (75, THEME["shu_deep"]),  # 長期 = 深朱 (long anchor)
        ])
    if show_sma100:
        sma_specs.append((100, THEME["forest"]))
    if show_sma200:
        sma_specs.append((200, THEME["gold"]))
    for window, color in sma_specs:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma(close, window),
                name=f"SMA{window}",
                line=dict(color=color, width=1),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    if show_ichimoku:
        ichi = ichimoku(df)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=ichi["tenkan"], name="転換線",
                line=dict(color="#e377c2", width=1), hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=ichi["kijun"], name="基準線",
                line=dict(color="#8c564b", width=1), hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=ichi["span_a"], name="先行スパンA",
                line=dict(color="rgba(0,200,0,0.3)", width=1),
                hoverinfo="skip",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=ichi["span_b"], name="先行スパンB",
                line=dict(color="rgba(200,0,0,0.3)", width=1),
                fill="tonexty", fillcolor="rgba(128,128,128,0.15)",
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Trendlines: draw each detected line at its full extent (start_date ..
    # latest data). Plotly clips to the current view automatically, so when
    # the user pans the line stays in place.
    kind_visibility = {
        "support": show_support,
        "resistance": show_resistance,
        "trend_up": show_trend_up,
        "trend_down": show_trend_down,
    }
    legend_shown: set[str] = set()
    data_end = df.index[-1]

    for line in lines:
        if not kind_visibility.get(line.kind, True):
            continue
        if line.valid is False and not show_invalid:
            continue

        x0 = line.start_date
        # Extend the line forward to the latest available bar so it projects
        # to the right edge of the chart (regardless of how the user has
        # panned/zoomed the visible range).
        x1 = max(line.end_date, data_end)
        y0 = line.price_at(x0)
        y1 = line.price_at(x1)
        color = LINE_COLORS[line.kind]

        # Line style by validation status. Plotly's short "dash"/"dot" styles
        # are hard to see against the candlestick background, so unjudged and
        # failed lines use longer-ink variants (longdash / longdashdot).
        #   valid=True  -> solid thick        (this line historically worked)
        #   valid=None  -> long-dash          (not enough forward data yet)
        #   valid=False -> long-dash-dot thin (historically failed)
        if line.valid is True:
            dash, width = "solid", 2.2
        elif line.valid is None:
            dash, width = "longdash", 1.8
        else:
            dash, width = "longdashdot", 1.4

        scale_label = SCALE_LABELS_JA.get(getattr(line, "scale", "long"), "")
        label = f"{LINE_LABELS_JA[line.kind]} ({scale_label})"
        legend_key = f"{line.kind}_{getattr(line, 'scale', 'long')}"
        show_in_legend = legend_key not in legend_shown
        if show_in_legend:
            legend_shown.add(legend_key)

        validity_text = {
            True: "検証 ✅",
            False: "検証 ❌",
            None: "検証 未判定",
        }[line.valid]

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                name=label,
                legendgroup=legend_key,
                showlegend=show_in_legend,
                line=dict(color=color, width=width, dash=dash),
                hovertemplate=(
                    f"{label}<br>"
                    f"開始: %{{x|%Y-%m-%d}}<br>"
                    f"価格: %{{y:.1f}}<br>"
                    f"タッチ: {len(line.touches)}回<br>"
                    f"スコア: {line.score:.1f}<br>"
                    f"{validity_text}"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        # Mark the pivot touches as dots. With an explicit initial x-range
        # set on the figure, off-screen markers no longer expand the axis,
        # so we can safely include every touch — they become visible as the
        # user pans through history.
        if line.touches:
            touch_prices = [line.price_at(t) for t in line.touches]
            fig.add_trace(
                go.Scatter(
                    x=list(line.touches),
                    y=touch_prices,
                    mode="markers",
                    marker=dict(color=color, size=6, symbol="circle-open"),
                    legendgroup=legend_key,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )

    # All-time high / low horizontal lines (gold), drawn from the *full*
    # history (not just the visible window). The y-axis will auto-extend
    # if the level is far above/below the current price action.
    def _fmt_price(p: float) -> str:
        return f"{currency_symbol}{p:,.0f}" if currency_symbol else f"{p:,.2f}"

    if ath is not None:
        ath_price, ath_date = ath
        fig.add_hline(
            y=ath_price,
            line_color=THEME["gold"],
            line_width=1.6,
            line_dash="dashdot",
            annotation_text=(
                f"上場来高値 {_fmt_price(ath_price)} "
                f"({ath_date.strftime('%Y-%m-%d')})"
            ),
            annotation_position="top left",
            annotation_font_color=THEME["gold"],
            annotation_font_size=11,
            row=1, col=1,
        )

    if atl is not None:
        atl_price, atl_date = atl
        fig.add_hline(
            y=atl_price,
            line_color=THEME["gold"],
            line_width=1.6,
            line_dash="dashdot",
            annotation_text=(
                f"上場来安値 {_fmt_price(atl_price)} "
                f"({atl_date.strftime('%Y-%m-%d')})"
            ),
            annotation_position="bottom left",
            annotation_font_color=THEME["gold"],
            annotation_font_size=11,
            row=1, col=1,
        )

    # --- Volume row ---------------------------------------------------------
    if "volume" in osc_rows:
        vrow = osc_rows["volume"]
        volume_colors = [
            THEME["shu"] if c >= o else THEME["forest"]
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df["Volume"], name="出来高",
                marker_color=volume_colors, showlegend=False,
            ),
            row=vrow, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=volume_ma(df["Volume"], 20),
                name="Vol MA20",
                line=dict(color=THEME["ink_muted"], width=1),
                showlegend=False, hoverinfo="skip",
            ),
            row=vrow, col=1,
        )

    # --- MACD row -----------------------------------------------------------
    if "macd" in osc_rows:
        mrow = osc_rows["macd"]
        macd_df = macd(close)
        hist_colors = [
            THEME["shu"] if v >= 0 else THEME["forest"]
            for v in macd_df["hist"]
        ]
        fig.add_trace(
            go.Bar(
                x=df.index, y=macd_df["hist"], name="Hist",
                marker_color=hist_colors, showlegend=False,
            ),
            row=mrow, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=macd_df["macd"], name="MACD",
                line=dict(color=THEME["navy"], width=1.2),
                showlegend=False,
            ),
            row=mrow, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=macd_df["signal"], name="Signal",
                line=dict(color=THEME["copper"], width=1.2),
                showlegend=False,
            ),
            row=mrow, col=1,
        )

    # --- RSI row -----------------------------------------------------------
    if "rsi" in osc_rows:
        rrow = osc_rows["rsi"]
        rsi_series = rsi(close)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=rsi_series, name="RSI",
                line=dict(color=THEME["shu_deep"], width=1.2),
                showlegend=False,
            ),
            row=rrow, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color=THEME["ink_muted"], line_width=1, row=rrow, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=THEME["ink_muted"], line_width=1, row=rrow, col=1)
        fig.update_yaxes(range=[0, 100], row=rrow, col=1)

    # --- Historical PER row -------------------------------------------------
    if "per" in osc_rows:
        prow = osc_rows["per"]
        fig.add_trace(
            go.Scatter(
                x=per_series.index, y=per_series,
                name="ヒストリカル PER",
                mode="lines",
                line=dict(color=THEME["shu_deep"], width=1.4),
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>PER %{y:.1f} 倍<extra></extra>",
                connectgaps=False,
            ),
            row=prow, col=1,
        )
        # Median reference line — a quiet anchor for "cheap vs expensive"
        median_per = float(per_series.dropna().median())
        if pd.notna(median_per):
            fig.add_hline(
                y=median_per,
                line_dash="dashdot",
                line_color=THEME["ink_muted"],
                line_width=0.9,
                annotation_text=f"中央値 {median_per:.1f} 倍",
                annotation_position="top left",
                annotation_font_color=THEME["ink_muted"],
                annotation_font_size=10,
                row=prow, col=1,
            )

    # --- Layout -------------------------------------------------------------
    fig.update_layout(
        # Title intentionally omitted: the ticker/name is rendered by
        # Streamlit as a subheader OUTSIDE the figure (see the caller).
        # Keeping it inside the Plotly layout caused collisions with the
        # horizontal legend once the legend wrapped onto multiple rows.
        title=None,
        # Height scales with oscillator count: base 560 + ~115px per panel,
        # so a bare candlestick view stays compact while a fully populated
        # 4-oscillator layout gets enough room to read each row.
        height=560 + 115 * n_osc,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=THEME["surface"],
            bordercolor=THEME["ink"],
            font=dict(family=PLOTLY_FONT_FAMILY, color=THEME["ink"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(
                family=PLOTLY_FONT_FAMILY,
                color=THEME["ink"],
                size=11,
            ),
        ),
        margin=dict(l=48, r=24, t=56, b=24),
        template="plotly_white",
        plot_bgcolor=THEME["surface"],
        paper_bgcolor=THEME["surface"],
        font=dict(
            family=PLOTLY_FONT_FAMILY,
            color=THEME["ink"],
            size=12,
        ),
        # Click-and-drag in the chart area = pan. Click-and-drag on an axis
        # = zoom that axis only (Plotly default behaviour). Combined with
        # scrollZoom in the renderer config, this gives free 2D navigation.
        dragmode="pan",
        clickmode="event+select",
        # uirevision preserves user pan/zoom state across Streamlit reruns
        # as long as the value stays constant. Changing tickers or the
        # initial-range slider invalidates it and resets the view.
        uirevision=ui_revision or ticker_label,
    )

    # Subplot titles (出来高 / MACD / ヒストリカル PER) — re-style to match melta
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(
            family=PLOTLY_FONT_FAMILY,
            color=THEME["ink_muted"],
            size=11,
        )

    # Gridlines use the melta border colour for quiet but legible rulers
    fig.update_yaxes(
        gridcolor=THEME["border"],
        zerolinecolor=THEME["border"],
        linecolor=THEME["border"],
        tickfont=dict(color=THEME["ink_muted"], size=10),
    )
    # Set the price panel y-axis range to visible high/low +/- 10%
    fig.update_yaxes(
        range=[y_min, y_max],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        gridcolor=THEME["border"],
        linecolor=THEME["border"],
        tickfont=dict(color=THEME["ink_muted"], size=10),
    )
    # Hide exchange-closed dates so each daily candle has equal horizontal
    # spacing. Weekends use Plotly's recurring rule; only missing weekdays
    # (Japanese holidays and exceptional closures) are listed explicitly.
    # The previous implementation enumerated every weekend over ten years,
    # producing thousands of range-break values and occasional malformed gaps.
    rangebreaks: list[dict] = []
    if interval == "1d":
        rangebreaks.append(dict(bounds=["sat", "mon"]))
        business_days = pd.date_range(df.index[0], df.index[-1], freq="B")
        trading_days = pd.DatetimeIndex(df.index).normalize()
        missing_business_days = business_days.difference(trading_days)
        if len(missing_business_days):
            rangebreaks.append(
                dict(
                    values=[d.strftime("%Y-%m-%d") for d in missing_business_days],
                    dvalue=24 * 60 * 60 * 1000,
                )
            )
    elif interval == "1wk":
        missing_weeks = []
        for i in range(1, len(df)):
            delta = df.index[i] - df.index[i-1]
            if delta.days > 7:
                gap_dates = pd.date_range(df.index[i-1] + pd.Timedelta(days=7), df.index[i] - pd.Timedelta(days=1), freq="7D")
                missing_weeks.extend([d.strftime("%Y-%m-%d") for d in gap_dates])
        if missing_weeks:
            rangebreaks.append(
                dict(values=missing_weeks, dvalue=7 * 86400000)
            )
    
    xaxis_kwargs: dict = {"rangebreaks": rangebreaks}
    if initial_range is not None:
        xaxis_kwargs["range"] = list(initial_range)
    fig.update_xaxes(**xaxis_kwargs)
    return fig, osc_rows


# ---------------------------------------------------------------------------
# Name resolution with yfinance fallback
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=False)
def _resolve_name_via_yfinance(ticker: str) -> str:
    """Look up a human-readable name for a ticker that isn't in the
    built-in UNIVERSE/INDICES map. Cached for a week so we don't hit
    yfinance on every rerun. Returns an empty string on failure."""
    try:
        return get_ticker_name(ticker) or ""
    except Exception:
        return ""


def _display_name_for(ticker: str, name_lookup: dict[str, str]) -> str:
    """Return a display name for ``ticker``. Prefers the built-in lookup
    (Nikkei 225 + INDICES), then falls back to yfinance. Falls back to
    the ticker itself if nothing is available."""
    name = name_lookup.get(ticker, "")
    if name:
        return name
    fetched = _resolve_name_via_yfinance(ticker)
    return fetched or ticker


def _render_ticker_panel(name_lookup: dict[str, str]) -> None:
    """Render the resizable right-side ticker picker."""
    import html as _html

    favs = load_favorites()
    selected = st.session_state.get("selected_ticker", get_default_ticker())
    all_codes = sorted(
        set([code for code, _, _ in UNIVERSE])
        | set([code for code, _ in INDICES])
        | set(favs.keys())
        | {selected}
    )
    # Quotes in the ticker panel are a startup snapshot.  Streamlit reruns the
    # script whenever the chart ticker changes, so keep the first values in
    # session state instead of rebuilding the quote set for the new ticker.
    if "_ticker_panel_price_snapshot" not in st.session_state:
        st.session_state["_ticker_panel_price_snapshot"] = _fetch_last_prices(
            tuple(all_codes)
        )
    prices = st.session_state["_ticker_panel_price_snapshot"]
    selected_name = _html.escape(_display_name_for(selected, name_lookup))
    safe_selected = _html.escape(selected)
    selected_is_fav = selected in favs
    fav_action = "remove" if selected_is_fav else "add"
    fav_label = "⭐ お気に入りから外す" if selected_is_fav else "☆ お気に入りに追加"

    def _row(code: str, name: str) -> str:
        p = prices.get(code, {})
        last = p.get("last")
        pct = p.get("pct")
        is_jp = code.endswith(".T") or code.isdigit()
        if last is None:
            price_html = "&mdash;"
            pct_html = '<span class="rp-flat">&mdash;</span>'
        else:
            price_html = f"&yen;{last:,.0f}" if is_jp else f"{last:,.2f}"
            pct_val = float(pct or 0.0)
            if pct_val > 0:
                pct_html = f'<span class="rp-up">▲{pct_val:.2f}%</span>'
            elif pct_val < 0:
                pct_html = f'<span class="rp-down">▼{abs(pct_val):.2f}%</span>'
            else:
                pct_html = '<span class="rp-flat">0.00%</span>'

        safe_code = _html.escape(code)
        safe_name = _html.escape(name or name_lookup.get(code, code))
        active = " is-active" if code == selected else ""
        return (
            f'<button type="button" class="rp-row{active}" onclick="pickTicker(\'{safe_code}\')">'
            f'<span class="rp-main"><span class="rp-code">{safe_code}</span>'
            f'<span class="rp-name">{safe_name}</span></span>'
            f'<span class="rp-quote"><span class="rp-price">{price_html}</span>{pct_html}</span>'
            f'</button>'
        )

    nikkei_rows: list[str] = []
    prev_sector: str | None = None
    for code, name, sector in UNIVERSE:
        if sector != prev_sector:
            nikkei_rows.append(f'<div class="rp-sector">{_html.escape(sector)}</div>')
            prev_sector = sector
        nikkei_rows.append(_row(code, name))

    index_rows = [_row(code, name) for code, name in INDICES]
    if favs:
        favorite_rows = [
            _row(code, stored_name or name_lookup.get(code, code))
            for code, stored_name in favs.items()
        ]
    else:
        favorite_rows = ['<div class="rp-empty">お気に入りはまだありません</div>']

    panel_html = f"""
<style>
#rightTickerPanel {{
    position: fixed;
    top: 0;
    right: 14px;
    width: 300px;
    min-width: 220px;
    max-width: 52vw;
    height: 100vh;
    z-index: 9999;
    display: flex;
    background: rgba(15, 23, 42, 0.98);
    border-left: 1px solid rgba(148, 163, 184, 0.22);
    box-shadow: -8px 0 24px rgba(0, 0, 0, 0.32);
    color: #e2e8f0;
    font-family: "IBM Plex Sans JP", "Hiragino Sans", sans-serif;
}}
#rightTickerHandle {{
    width: 7px;
    flex: 0 0 7px;
    cursor: ew-resize;
    position: relative;
}}
#rightTickerHandle:hover,
#rightTickerHandle.is-dragging {{
    background: rgba(56, 189, 248, 0.26);
}}
#rightTickerHandle::after {{
    content: "";
    position: absolute;
    left: 2px;
    top: calc(50% - 24px);
    width: 3px;
    height: 48px;
    border-radius: 999px;
    background: rgba(226, 232, 240, 0.24);
}}
.rp-inner {{
    min-width: 0;
    flex: 1;
    display: flex;
    flex-direction: column;
}}
.rp-title {{
    padding: 64px 12px 10px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #cbd5e1;
    border-bottom: 1px solid rgba(148, 163, 184, 0.16);
}}
.rp-favbar {{
    padding: 8px 10px 10px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.14);
    background: rgba(2, 6, 23, 0.18);
}}
.rp-selected {{
    min-width: 0;
    margin-bottom: 7px;
}}
.rp-selected-code {{
    display: block;
    color: #f8fafc;
    font-family: "IBM Plex Mono", Menlo, monospace;
    font-size: 0.78rem;
    font-weight: 700;
}}
.rp-selected-name {{
    display: block;
    color: #94a3b8;
    font-size: 0.68rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.rp-favbtn {{
    width: 100%;
    border: 1px solid rgba(251, 191, 36, 0.45);
    border-radius: 4px;
    background: rgba(251, 191, 36, 0.10);
    color: #fde68a;
    cursor: pointer;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 7px 8px;
    text-align: center;
}}
.rp-favbtn:hover {{
    background: rgba(251, 191, 36, 0.18);
    border-color: rgba(251, 191, 36, 0.75);
}}
.rp-tabs {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 4px;
    padding: 8px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.14);
}}
.rp-tab {{
    min-width: 0;
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 4px;
    background: rgba(30, 41, 59, 0.65);
    color: #94a3b8;
    cursor: pointer;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 6px 3px;
    white-space: nowrap;
}}
.rp-tab:hover {{
    color: #e2e8f0;
    border-color: rgba(56, 189, 248, 0.42);
}}
.rp-tab.is-active {{
    color: #0f172a;
    background: #38bdf8;
    border-color: #38bdf8;
}}
.rp-scroll {{
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    padding-bottom: 16px;
}}
.rp-scroll::-webkit-scrollbar {{ width: 6px; }}
.rp-scroll::-webkit-scrollbar-thumb {{
    background: rgba(148, 163, 184, 0.30);
    border-radius: 999px;
}}
.rp-sector {{
    padding: 10px 12px 4px;
    color: #64748b;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    background: rgba(2, 6, 23, 0.25);
}}
.rp-row {{
    width: 100%;
    border: 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.10);
    background: transparent;
    color: inherit;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 8px 10px;
    cursor: pointer;
    text-align: left;
}}
.rp-row:hover {{
    background: rgba(56, 189, 248, 0.08);
}}
.rp-row.is-active {{
    background: rgba(56, 189, 248, 0.16);
    box-shadow: inset 3px 0 0 #38bdf8;
}}
.rp-main {{
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
}}
.rp-code {{
    color: #f8fafc;
    font-family: "IBM Plex Mono", Menlo, monospace;
    font-size: 0.75rem;
    font-weight: 700;
}}
.rp-name {{
    max-width: 150px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: #94a3b8;
    font-size: 0.68rem;
}}
.rp-quote {{
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 2px;
    font-family: "IBM Plex Mono", Menlo, monospace;
}}
.rp-price {{
    color: #cbd5e1;
    font-size: 0.72rem;
}}
.rp-up, .rp-down, .rp-flat {{
    font-size: 0.66rem;
    font-weight: 700;
}}
.rp-up {{ color: #34d399; }}
.rp-down {{ color: #fb7185; }}
.rp-flat {{ color: #64748b; }}
.rp-empty {{
    color: #64748b;
    font-size: 0.78rem;
    padding: 24px 12px;
    text-align: center;
}}
.rp-direct {{
    padding: 10px;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 6px;
}}
.rp-direct input {{
    min-width: 0;
    border: 1px solid rgba(148, 163, 184, 0.26);
    border-radius: 4px;
    background: rgba(2, 6, 23, 0.34);
    color: #e2e8f0;
    padding: 7px 8px;
    font-size: 0.78rem;
}}
.rp-direct button {{
    border: 1px solid #38bdf8;
    border-radius: 4px;
    background: #38bdf8;
    color: #0f172a;
    padding: 0 10px;
    cursor: pointer;
    font-weight: 700;
}}
.rp-hint {{
    padding: 0 10px 8px;
    color: #64748b;
    font-size: 0.68rem;
}}
:root {{
    --rightTickerReserved: 334px;
}}
html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {{
    overflow-y: auto !important;
}}
[data-testid="stAppViewContainer"] {{
    padding-right: var(--rightTickerReserved) !important;
    box-sizing: border-box !important;
}}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"],
section.main {{
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}}
.main .block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {{
    max-width: 100% !important;
    width: 100% !important;
    box-sizing: border-box !important;
}}
@media (max-width: 900px) {{
    #rightTickerPanel {{
        position: relative;
        width: 100% !important;
        max-width: none;
        height: 48vh;
        margin-bottom: 16px;
    }}
    #rightTickerHandle {{ display: none; }}
    :root {{
        --rightTickerReserved: 0px;
    }}
    [data-testid="stAppViewContainer"] {{
        padding-right: 0 !important;
    }}
}}
</style>
<div id="rightTickerPanel" data-server-ticker="{safe_selected}" data-server-favorite="{int(selected_is_fav)}">
  <div id="rightTickerHandle"></div>
  <div class="rp-inner">
    <div class="rp-title">銘柄選択</div>
    <div class="rp-favbar">
      <div class="rp-selected">
        <span class="rp-selected-code">{safe_selected}</span>
        <span class="rp-selected-name">{selected_name}</span>
      </div>
      <button type="button" class="rp-favbtn" onclick="toggleFavorite('{safe_selected}', '{fav_action}', this)">{fav_label}</button>
    </div>
    <div class="rp-tabs">
      <button type="button" class="rp-tab is-active" onclick="switchTickerTab('nikkei', this)">日経225</button>
      <button type="button" class="rp-tab" onclick="switchTickerTab('index', this)">指数</button>
      <button type="button" class="rp-tab" onclick="switchTickerTab('favorite', this)">お気に入り</button>
      <button type="button" class="rp-tab" onclick="switchTickerTab('direct', this)">直接入力</button>
    </div>
    <div class="rp-scroll">
      <div id="rp-tab-nikkei">{''.join(nikkei_rows)}</div>
      <div id="rp-tab-index" style="display:none">{''.join(index_rows)}</div>
      <div id="rp-tab-favorite" style="display:none">{''.join(favorite_rows)}</div>
      <div id="rp-tab-direct" style="display:none">
        <div class="rp-direct">
          <input id="rpDirectInput" placeholder="7203, 6758.T, ^N225, AAPL" oninput="updateDirectResult()" onkeydown="directTickerKey(event)">
          <button type="button" onclick="pickDirectTicker()">表示</button>
        </div>
        <div class="rp-hint">4桁コードは .T を自動付与します。</div>
        <div id="rpDirectResult" class="rp-empty">ティッカーを入力してください</div>
      </div>
    </div>
  </div>
</div>
<script>
(function() {{
  function normalizeTicker(raw) {{
    var code = (raw || '').trim().toUpperCase();
    if (/^[0-9]{{4}}$/.test(code)) code = code + '.T';
    return code;
  }}

  window.pickTicker = function(code) {{
    code = normalizeTicker(code);
    if (!code) return;
    // Give immediate visual feedback while Streamlit prepares the new chart.
    var pickedName = code;
    document.querySelectorAll('.rp-row').forEach(function(row) {{
      var rowCode = row.querySelector('.rp-code');
      var active = !!rowCode && rowCode.textContent.trim() === code;
      row.classList.toggle('is-active', active);
      if (active) {{
        var rowName = row.querySelector('.rp-name');
        if (rowName) pickedName = rowName.textContent.trim();
      }}
    }});
    var selectedCode = document.querySelector('.rp-selected-code');
    var selectedName = document.querySelector('.rp-selected-name');
    if (selectedCode) selectedCode.textContent = code;
    if (selectedName) selectedName.textContent = pickedName;
    var url = new URL(window.parent.location.href);
    url.searchParams.set('ticker', code);
    window.parent.history.replaceState({{}}, '', url.toString());
    sendTickerEvent({{type: 'select', code: code}}, url.toString());
  }};

  window.toggleFavorite = function(code, action, button) {{
    code = normalizeTicker(code);
    if (!code) return;
    var favoriteTab = document.getElementById('rp-tab-favorite');
    var matchingRows = Array.from(document.querySelectorAll('.rp-row')).filter(function(row) {{
      var rowCode = row.querySelector('.rp-code');
      return !!rowCode && rowCode.textContent.trim() === code;
    }});
    if (action === 'add') {{
      if (button) {{
        button.textContent = '⭐ お気に入りから外す';
        button.setAttribute('onclick', "toggleFavorite('" + code + "', 'remove', this)");
      }}
      if (favoriteTab && !matchingRows.some(function(row) {{ return row.parentElement === favoriteTab; }})) {{
        var sourceRow = matchingRows.length ? matchingRows[0] : null;
        var emptyMessage = favoriteTab.querySelector('.rp-empty');
        if (emptyMessage) emptyMessage.remove();
        if (sourceRow) favoriteTab.appendChild(sourceRow.cloneNode(true));
      }}
    }} else if (action === 'remove') {{
      if (button) {{
        button.textContent = '☆ お気に入りに追加';
        button.setAttribute('onclick', "toggleFavorite('" + code + "', 'add', this)");
      }}
      if (favoriteTab) {{
        Array.from(favoriteTab.querySelectorAll('.rp-row')).forEach(function(row) {{
          var rowCode = row.querySelector('.rp-code');
          if (rowCode && rowCode.textContent.trim() === code) row.remove();
        }});
        if (!favoriteTab.querySelector('.rp-row')) {{
          favoriteTab.innerHTML = '<div class="rp-empty">お気に入りはまだありません</div>';
        }}
      }}
    }}
    var url = new URL(window.parent.location.href);
    url.searchParams.set('ticker', code);
    url.searchParams.set('fav_ticker', code);
    url.searchParams.set('fav_action', action);
    sendTickerEvent({{type: 'favorite', code: code, action: action}}, url.toString());
  }};

  function sendTickerEvent(payload, fallbackUrl) {{
    var input = window.parent.document.querySelector(
      'input[aria-label="ticker-panel-event"]'
    );
    if (!input) return;
    payload.nonce = Date.now();
    var value = JSON.stringify(payload);
    var setter = Object.getOwnPropertyDescriptor(
      window.parent.HTMLInputElement.prototype, 'value'
    ).set;
    input.focus();
    setter.call(input, value);
    input.dispatchEvent(new Event('input', {{bubbles: true}}));
    input.dispatchEvent(new Event('change', {{bubbles: true}}));
    input.dispatchEvent(new KeyboardEvent('keydown', {{
      key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true
    }}));
    input.blur();

    // Some browser/Streamlit combinations ignore programmatic key events on
    // hidden inputs. Fall back to navigation only if the in-app update did
    // not replace the selected ticker within a generous interval.
    setTimeout(function() {{
      var panel = document.getElementById('rightTickerPanel');
      var acknowledged = !!panel && panel.dataset.serverTicker === payload.code;
      if (acknowledged && payload.type === 'favorite') {{
        var expectedFavorite = payload.action === 'add' ? '1' : '0';
        acknowledged = panel.dataset.serverFavorite === expectedFavorite;
      }}
      if (!acknowledged) {{
        window.parent.location.href = fallbackUrl;
      }}
    }}, 5000);
  }}

  window.pickDirectTicker = function() {{
    var input = document.getElementById('rpDirectInput');
    window.pickTicker(input ? input.value : '');
  }};

  window.updateDirectResult = function() {{
    var input = document.getElementById('rpDirectInput');
    var result = document.getElementById('rpDirectResult');
    var code = normalizeTicker(input ? input.value : '');
    if (!result) return;
    if (!code) {{
      result.className = 'rp-empty';
      result.innerHTML = 'ティッカーを入力してください';
      return;
    }}
    result.className = 'rp-row';
    result.setAttribute('onclick', "pickTicker('" + code.replace(/'/g, "\\\\'") + "')");
    result.innerHTML =
      '<span class="rp-main"><span class="rp-code">' + code + '</span>' +
      '<span class="rp-name">直接入力</span></span>' +
      '<span class="rp-quote"><span class="rp-price">表示</span></span>';
  }};

  window.directTickerKey = function(event) {{
    if (event.key === 'Enter') {{
      event.preventDefault();
      window.pickDirectTicker();
    }}
  }};

  window.switchTickerTab = function(tab, btn) {{
    ['nikkei', 'index', 'favorite', 'direct'].forEach(function(name) {{
      var el = document.getElementById('rp-tab-' + name);
      if (el) el.style.display = name === tab ? 'block' : 'none';
    }});
    document.querySelectorAll('.rp-tab').forEach(function(el) {{
      el.classList.remove('is-active');
    }});
    btn.classList.add('is-active');
    try {{ localStorage.setItem('rightTickerActiveTab', tab); }} catch (e) {{}}
    if (tab === 'direct') {{
      setTimeout(function() {{
        var input = document.getElementById('rpDirectInput');
        if (input) input.focus();
      }}, 20);
    }}
  }};

  var panel = document.getElementById('rightTickerPanel');
  var handle = document.getElementById('rightTickerHandle');
  var panelRightGap = 14;
  var startX = 0;
  var startWidth = 0;
  var dragging = false;

  var scrollArea = panel.querySelector('.rp-scroll');
  if (scrollArea) {{
    try {{
      scrollArea.scrollTop = parseInt(
        localStorage.getItem('rightTickerScrollTop') || '0', 10
      );
    }} catch (e) {{}}
    scrollArea.addEventListener('scroll', function() {{
      try {{ localStorage.setItem('rightTickerScrollTop', String(scrollArea.scrollTop)); }} catch (e) {{}}
    }}, {{passive: true}});
  }}

  try {{
    var activeTab = localStorage.getItem('rightTickerActiveTab') || 'nikkei';
    var activeButton = Array.from(document.querySelectorAll('.rp-tab')).find(function(btn) {{
      return (btn.getAttribute('onclick') || '').indexOf("'" + activeTab + "'") !== -1;
    }});
    if (activeButton) window.switchTickerTab(activeTab, activeButton);
  }} catch (e) {{}}

  function applyWidth(width) {{
    var clamped = Math.min(Math.max(width, 220), window.innerWidth * 0.52);
    panel.style.width = clamped + 'px';
    panel.style.right = panelRightGap + 'px';
    var reserved = clamped + panelRightGap + 20;
    document.documentElement.style.setProperty('--rightTickerReserved', reserved + 'px');
    document.querySelectorAll('html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main').forEach(function(el) {{
      el.style.overflowY = 'auto';
    }});
    document.querySelectorAll('[data-testid="stAppViewContainer"]').forEach(function(el) {{
      el.style.paddingRight = reserved + 'px';
      el.style.boxSizing = 'border-box';
    }});
    document.querySelectorAll('[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"], section.main').forEach(function(el) {{
      el.style.width = '100%';
      el.style.maxWidth = '100%';
      el.style.boxSizing = 'border-box';
    }});
    document.querySelectorAll('.main .block-container, [data-testid="stMainBlockContainer"], [data-testid="stAppViewBlockContainer"]').forEach(function(el) {{
      el.style.maxWidth = '100%';
      el.style.width = '100%';
      el.style.boxSizing = 'border-box';
    }});
    if (window.innerWidth <= 900) {{
      document.documentElement.style.setProperty('--rightTickerReserved', '0px');
      document.querySelectorAll('[data-testid="stAppViewContainer"]').forEach(function(el) {{
        el.style.paddingRight = '0px';
      }});
    }}
    setTimeout(function() {{
      window.dispatchEvent(new Event('resize'));
    }}, 30);
    try {{ localStorage.setItem('rightTickerPanelWidth', String(clamped)); }} catch (e) {{}}
  }}

  try {{
    var saved = parseInt(localStorage.getItem('rightTickerPanelWidth') || '', 10);
    applyWidth(saved || panel.offsetWidth || 300);
  }} catch (e) {{}}

  handle.addEventListener('mousedown', function(event) {{
    dragging = true;
    startX = event.clientX;
    startWidth = panel.offsetWidth;
    handle.classList.add('is-dragging');
    document.body.style.userSelect = 'none';
    event.preventDefault();
  }});
  document.addEventListener('mousemove', function(event) {{
    if (!dragging) return;
    applyWidth(startWidth + (startX - event.clientX));
  }});
  document.addEventListener('mouseup', function() {{
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('is-dragging');
    document.body.style.userSelect = '';
  }});
}})();
</script>
"""
    html_part, script_part = panel_html.split("<script>", 1)
    script_body = script_part.rsplit("</script>", 1)[0]
    injector_html = f"""
<script>
(function() {{
  const doc = window.parent.document;
  const html = {json.dumps(html_part, ensure_ascii=False)};
  const scriptBody = {json.dumps(script_body, ensure_ascii=False)};

  const existingPanel = doc.getElementById('rightTickerPanel');
  const existingStyle = doc.getElementById('rightTickerPanelStyle');
  if (existingStyle) existingStyle.remove();

  const template = doc.createElement('template');
  template.innerHTML = html.trim();

  const style = template.content.querySelector('style');
  if (style) {{
    style.id = 'rightTickerPanelStyle';
    doc.head.appendChild(style);
  }}

  const panel = template.content.querySelector('#rightTickerPanel');
  if (existingPanel && panel) {{
    // Keep the tab DOM, scroll position, and listeners alive. Only the pieces
    // that can change after a ticker/favorite action are synchronized.
    const nextSelected = panel.querySelector('.rp-selected');
    const currentSelected = existingPanel.querySelector('.rp-selected');
    if (nextSelected && currentSelected) currentSelected.innerHTML = nextSelected.innerHTML;
    existingPanel.dataset.serverTicker = panel.dataset.serverTicker || '';
    existingPanel.dataset.serverFavorite = panel.dataset.serverFavorite || '0';

    const nextFavButton = panel.querySelector('.rp-favbtn');
    const currentFavButton = existingPanel.querySelector('.rp-favbtn');
    if (nextFavButton && currentFavButton) {{
      currentFavButton.textContent = nextFavButton.textContent;
      currentFavButton.setAttribute('onclick', nextFavButton.getAttribute('onclick'));
    }}

    const nextFavoriteRows = panel.querySelector('#rp-tab-favorite');
    const currentFavoriteRows = existingPanel.querySelector('#rp-tab-favorite');
    if (nextFavoriteRows && currentFavoriteRows) {{
      currentFavoriteRows.innerHTML = nextFavoriteRows.innerHTML;
    }}

    const selectedCode = {json.dumps(safe_selected)};
    existingPanel.querySelectorAll('.rp-row').forEach(function(row) {{
      const code = row.querySelector('.rp-code');
      row.classList.toggle('is-active', !!code && code.textContent.trim() === selectedCode);
    }});
  }} else if (panel) {{
    doc.body.appendChild(panel);
  }}

  if (!existingPanel) {{
    const script = doc.createElement('script');
    script.text = scriptBody;
    doc.body.appendChild(script);
    setTimeout(function() {{ script.remove(); }}, 0);
  }}
}})();
</script>
"""
    components.html(injector_html, height=0, width=0)


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="トレンドライン自動検出",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Inject the 京都ターミナル theme (see skills.md / frontend-design skill):
    # loads Google Fonts + CSS variables + Streamlit component overrides.
    # Must run before any other Streamlit widget so styles apply on first paint.
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    # Editorial masthead: oversized display title with a monospace
    # kicker strip (issue № / edition / date), a giant ghost 株 kanji
    # watermark sitting behind it, and an italic bilingual byline.
    _today = pd.Timestamp.now().strftime("%Y.%m.%d")
    st.markdown(
        f'''
        <div class="kt-masthead">
            <div class="kt-masthead-kicker">
                <span class="kt-kicker-issue">N<sup>o</sup> 01</span>
                <span class="kt-kicker-dot">·</span>
                <span class="kt-kicker-label">Kyoto Terminal — 株価解析</span>
                <span class="kt-kicker-date">{_today}</span>
            </div>
            <h1 class="kt-masthead-title">
                株価<span class="kt-title-accent">トレンドライン</span>自動検出
            </h1>
            <div class="kt-masthead-byline">
                日経 225・指数・任意ティッカーのローソク足に、自動検出したトレンドラインとサポート・レジスタンスを重ねて描画する研究ツール。
                <span class="kt-masthead-byline-en">Stock Trendline Auto-Detection · Research Build</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    name_lookup = all_names()

    # ----- Sidebar: App Mode Selector -----
    st.sidebar.header("機能選択")
    app_mode = st.sidebar.selectbox(
        "モード選択",
        ["チャート分析", "通知設定"],
        index=0,
        label_visibility="collapsed"
    )

    if app_mode == "通知設定":
        st.markdown(
            """
            <style>
            .kt-masthead { display: none !important; }
            [data-testid="stAppViewContainer"] {
                padding-right: 0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        components.html(
            """
            <script>
            (function() {
              const doc = window.parent.document;
              const panel = doc.getElementById('rightTickerPanel');
              const style = doc.getElementById('rightTickerPanelStyle');
              if (panel) panel.remove();
              if (style) style.remove();
              doc.documentElement.style.setProperty('--rightTickerReserved', '0px');
              doc.querySelectorAll('[data-testid="stAppViewContainer"]').forEach(function(el) {
                el.style.paddingRight = '0px';
              });
            })();
            </script>
            """,
            height=0,
            width=0,
        )
        from notification_ui import show_notification_ui
        show_notification_ui(name_lookup)
        return

    # ----- Right panel: ticker selection -----------------------------------
    qp_ticker = st.query_params.get("ticker", "")
    if qp_ticker and st.session_state.get("_applied_query_ticker") != qp_ticker:
        st.session_state["selected_ticker"] = normalise_ticker(qp_ticker)
        st.session_state["_applied_query_ticker"] = qp_ticker
    if "selected_ticker" not in st.session_state or not st.session_state["selected_ticker"]:
        st.session_state["selected_ticker"] = get_default_ticker()

    # The custom ticker panel writes to this hidden Streamlit input.  This
    # triggers an in-app rerun without reloading the browser page or rebuilding
    # the user's tab/navigation context.
    st.markdown(
        """
        <style>
        div[data-testid="stTextInput"]:has(input[aria-label="ticker-panel-event"]) {
            position: fixed;
            left: -10000px;
            top: 0;
            width: 1px;
            height: 1px;
            overflow: hidden;
            opacity: 0;
            pointer-events: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    panel_event_raw = st.text_input(
        "ticker-panel-event",
        key="_ticker_panel_event",
        label_visibility="collapsed",
    )
    if panel_event_raw:
        try:
            panel_event = json.loads(panel_event_raw)
        except (TypeError, json.JSONDecodeError):
            panel_event = {}
        event_nonce = panel_event.get("nonce")
        if event_nonce and event_nonce != st.session_state.get("_ticker_panel_event_nonce"):
            st.session_state["_ticker_panel_event_nonce"] = event_nonce
            event_ticker = normalise_ticker(panel_event.get("code", ""))
            if event_ticker:
                st.session_state["selected_ticker"] = event_ticker
                st.session_state["_applied_query_ticker"] = event_ticker
                if panel_event.get("type") == "favorite":
                    if panel_event.get("action") == "remove":
                        sync_favorite_and_notifications(
                            event_ticker, "", remove=True,
                        )
                    elif panel_event.get("action") == "add":
                        display_name = name_lookup.get(event_ticker, "")
                        if not display_name:
                            display_name = get_ticker_name(event_ticker) or ""
                        sync_favorite_and_notifications(
                            event_ticker, display_name,
                        )

    fav_action = st.query_params.get("fav_action", "")
    fav_ticker = normalise_ticker(st.query_params.get("fav_ticker", ""))
    if fav_action in {"add", "remove"} and fav_ticker:
        st.session_state["selected_ticker"] = fav_ticker
        if fav_action == "remove":
            sync_favorite_and_notifications(fav_ticker, "", remove=True)
        else:
            display_name = name_lookup.get(fav_ticker, "")
            if not display_name:
                display_name = get_ticker_name(fav_ticker) or ""
            sync_favorite_and_notifications(fav_ticker, display_name)
        st.query_params["ticker"] = fav_ticker
        if "fav_action" in st.query_params:
            del st.query_params["fav_action"]
        if "fav_ticker" in st.query_params:
            del st.query_params["fav_ticker"]
        st.rerun()

    _render_ticker_panel(name_lookup)

    ticker = st.session_state["selected_ticker"]
    st.sidebar.caption(f"選択中の銘柄: {ticker}")

    # ----- Sidebar: bar interval ------------------------------------------
    # Drives data-loader interval + which tuned-params file we read.
    # Each interval has its own lookback profiles and tuning, so switching
    # this rebuilds both the OHLCV cache lookup and the detection windows.
    interval = st.sidebar.radio(
        "時間足",
        options=list(INTERVAL_LABELS.keys()),
        format_func=lambda k: INTERVAL_LABELS[k],
        index=0,
        horizontal=True,
    )

    tuned_trend, tuned_eval = load_tuned_params(interval)

    # ----- Sidebar: scales -------------------------------------------------
    # Labels adjust to the selected interval because "短期" on 日足 and 5分足
    # mean very different calendar lengths.
    st.sidebar.header("検出スケール")
    hints = SCALE_HINTS[interval]
    scale_short = st.sidebar.checkbox(hints["short"], value=True)
    scale_mid = st.sidebar.checkbox(hints["mid"], value=True)
    scale_long = st.sidebar.checkbox(hints["long"], value=True)
    selected_scales = [
        s for s, ok in [("short", scale_short), ("mid", scale_mid), ("long", scale_long)]
        if ok
    ]

    # ----- Sidebar: display period ----------------------------------------
    # Slider range + label depends on interval: 日足 counts in 営業日,
    # 週足 counts in 週.
    period_slider_cfg = {
        "1d":  {"label": "初期表示の営業日数",  "min": 60,  "max": 1000, "value": 300,  "step": 20},
        "1wk": {"label": "初期表示の週数",      "min": 26,  "max": 520,  "value": 156,  "step": 4},
    }[interval]
    period_bars = st.sidebar.slider(
        period_slider_cfg["label"],
        min_value=period_slider_cfg["min"],
        max_value=period_slider_cfg["max"],
        value=period_slider_cfg["value"],
        step=period_slider_cfg["step"],
        help=(
            "チャート読み込み時の初期ズーム範囲。読み込み後はチャートの軸を"
            "ドラッグして自由に拡大・縮小・パンできます。"
        ),
    )

    # ----- Sidebar: detection params (expander) ---------------------------
    with st.sidebar.expander("検出パラメータ (上級)", expanded=False):
        tuned_path = _param_path_for(interval)
        if tuned_path.exists():
            precision = json.loads(tuned_path.read_text("utf-8"))["precision"]
            st.caption(
                f"チューニング済み ({INTERVAL_LABELS[interval]}): "
                f"precision ≈ {precision * 100:.1f}%"
            )
        else:
            st.caption(
                f"{INTERVAL_LABELS[interval]} はチューニング未実施 — "
                f"既定パラメータを使用します。"
            )
        # Slider ranges widened from the old daily-only values to cover
        # the 1wk tuning (tolerance up to ~5%, slope up to ~3/yr).
        tol_default = float(tuned_trend.tolerance_pct * 100)
        tolerance_pct = st.slider(
            "タッチ許容範囲 (%)", 0.3, 5.0,
            min(max(tol_default, 0.3), 5.0), step=0.1,
        ) / 100.0
        min_touches = st.slider(
            "最小タッチ数", 3, 8, tuned_trend.min_touches,
        )
        slope_default = float(tuned_trend.max_slope_annual)
        max_slope_annual = st.slider(
            "年率最大スロープ", 0.2, 3.0,
            min(max(slope_default, 0.2), 3.0), step=0.1,
        )
        max_lines_per_kind = st.slider(
            "種類別の最大線数", 1, 10, tuned_trend.max_lines_per_kind,
        )

    # ----- Sidebar: display toggles, split by category -------------------
    # テクニカル指標 (MA, 一目, トレンドライン, S/R) と その他 (ATH/ATL など)
    # を別エキスパンダに分離。検証 NG ラインは常に非表示なので UI から外す。
    with st.sidebar.expander("表示切替: テクニカル", expanded=True):
        show_sma = st.checkbox("移動平均線 (5/25/75)", value=True)
        show_sma100 = st.checkbox("SMA100", value=False)
        show_sma200 = st.checkbox("SMA200", value=False)
        show_ichimoku = st.checkbox("一目均衡表", value=False)
        # Master trendline toggle: when OFF, none of the detected lines are
        # drawn unless a specific kind is selected below. Turning it ON shows
        # every kind, while each child checkbox also works independently.
        show_trendlines = st.checkbox("🔺 トレンドラインを表示", value=False)
        show_support = st.checkbox("🟢 サポート", value=False) or show_trendlines
        show_resistance = st.checkbox("🔴 レジスタンス", value=False) or show_trendlines
        show_trend_up = st.checkbox("🔵 上昇トレンド", value=False) or show_trendlines
        show_trend_down = st.checkbox("🟠 下降トレンド", value=False) or show_trendlines
        detect_trendlines = any(
            (show_support, show_resistance, show_trend_up, show_trend_down)
        )

    with st.sidebar.expander("表示切替: その他", expanded=True):
        show_ath_atl = st.checkbox("🟡 上場来高値・安値", value=False)

    # Fundamentals and earnings are core content and are always displayed.
    show_fundamentals = True
    show_earnings = True

    # Validation-failed (検証NG) lines are always hidden — the UI toggle
    # used to live in the テクニカル expander but was removed per spec.
    show_invalid = False

    # ----- Sidebar: oscillator panels (expander) -------------------------
    with st.sidebar.expander("オシレーター", expanded=True):
        show_volume = st.checkbox("出来高", value=True)
        show_macd = st.checkbox("MACD", value=True)
        show_rsi = st.checkbox("RSI", value=True)
        show_per_hist = st.checkbox(
            "ヒストリカル PER",
            value=False,
            help=(
                "決算短信の会社予想／実績 EPS をステップ関数として"
                "終値に割り当てた推移 (J-Quants 決算短信) を"
                "オシレーターとして表示します。"
            ),
        )

    # ----- Data loading ----------------------------------------------------
    df_full = load_chart_data(ticker, interval=interval)
    if df_full is None or df_full.empty:
        st.error(
            f"`{ticker}` のデータが取得できませんでした。"
            " yfinance でダウンロードできない銘柄の可能性があります。"
        )
        return

    if not selected_scales:
        st.warning("少なくとも 1 つのスケールを選択してください。")
        return

    # Initial visible window: last N bars (the user can pan/zoom freely
    # afterwards, but this controls what's centred on first render).
    period_bars = min(period_bars, len(df_full))
    initial_start = df_full.index[-period_bars]
    initial_end = df_full.index[-1]
    warmup_bars = 260 if interval == "1d" else 220
    df_plot = df_full.tail(min(len(df_full), period_bars + warmup_bars))

    # ----- Detection + evaluation ------------------------------------------
    # Base params override tolerance/min_touches from sidebar but keep tuned
    # quality filters (min_span_days, max_last_touch_age_days).
    base_params = TrendParams(
        pivot_window=tuned_trend.pivot_window,  # overridden by scale profile
        tolerance_pct=tolerance_pct,
        min_touches=min_touches,
        max_slope_annual=max_slope_annual,
        lookback_bars=tuned_trend.lookback_bars,
        max_lines_per_kind=max_lines_per_kind,
        min_span_days=tuned_trend.min_span_days,
        max_last_touch_age_days=tuned_trend.max_last_touch_age_days,
    )
    lines: list[Line] = []
    if detect_trendlines:
        # Use cached detection to avoid O(n²) recalculation on every rerender.
        # When trendlines are hidden, skip this entirely; detection is the
        # heaviest CPU path in the chart screen.
        line_dicts = _cached_detect_and_evaluate(
            _df_full_hash=_df_hash(df_full),
            ticker=ticker,
            interval=interval,
            pivot_window=base_params.pivot_window,
            tolerance_pct=tolerance_pct,
            min_touches=min_touches,
            max_slope_annual=max_slope_annual,
            lookback_bars=base_params.lookback_bars,
            max_lines_per_kind=max_lines_per_kind,
            min_span_days=base_params.min_span_days,
            max_last_touch_age_days=base_params.max_last_touch_age_days,
            selected_scales=tuple(sorted(selected_scales)),
            forward_bars=tuned_eval.forward_bars,
            tolerance_eval=tuned_eval.tolerance_pct,
        )
        lines = [_dict_to_line(d) for d in line_dicts]

    # All-time high / low computed from the full cached history
    ath: tuple[float, pd.Timestamp] | None = None
    atl: tuple[float, pd.Timestamp] | None = None
    if show_ath_atl:
        ath = (float(df_full["High"].max()), df_full["High"].idxmax())
        atl = (float(df_full["Low"].min()), df_full["Low"].idxmin())

    # ----- Header metrics --------------------------------------------------
    label = _display_name_for(ticker, name_lookup)
    last = df_full["Close"].iloc[-1]
    prev = df_full["Close"].iloc[-2] if len(df_full) >= 2 else last
    chg = last - prev
    chg_pct = chg / prev * 100 if prev else 0

    # Detect currency: Japanese tickers use ¥, indices use raw numbers
    is_jp = ticker.endswith(".T") or ticker.isdigit()
    cur = "¥" if is_jp else ""

    # Hero card: big ticker/company name at top, price + delta underneath.
    # Replaces the default st.metric so the asset identity is prominent
    # and we don't need a duplicate "#### {ticker} {label}" heading above
    # the chart further down.
    price_str = (
        f"{cur}{last:,.0f}" if is_jp else f"{cur}{last:,.2f}"
    )
    chg_str = (
        f"{chg:+,.0f} ({chg_pct:+.2f}%)" if is_jp
        else f"{chg:+,.2f} ({chg_pct:+.2f}%)"
    )
    if chg > 0:
        delta_class = "up"
        delta_arrow = "↑"
    elif chg < 0:
        delta_class = "down"
        delta_arrow = "↓"
    else:
        delta_class = "flat"
        delta_arrow = "→"
    # html.escape the company name in case it contains characters that
    # would corrupt the markdown block (defensive — yfinance names are
    # usually clean, but names from direct-input tickers could have &).
    import html as _html
    safe_label = _html.escape(label)
    st.markdown(
        f'''
        <div class="kt-hero-card">
          <div class="kt-hero-title">
            <span class="kt-hero-ticker">{ticker}</span>
            <span>{safe_label}</span>
          </div>
          <div class="kt-hero-price-row">
            <span class="kt-hero-price">{price_str}</span>
            <span class="kt-hero-delta {delta_class}">{delta_arrow} {chg_str}</span>
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # Keep fundamentals in their original visual position, but fill this slot
    # only after the chart has been sent to the browser.
    fundamentals_slot = st.empty()

    # ----- Historical PER (optional oscillator) ---------------------------
    # Computed lazily only when the panel is enabled. The underlying
    # FY step-function data is cached for 12h inside ``_load_fy_eps_steps``.
    per_series: pd.Series | None = None
    per_source: str = ""
    if show_per_hist:
        per_series, per_source = _cached_historical_per_series(
            ticker, interval, len(df_plot)
        )
        if per_series is not None:
            per_series = per_series.reindex(df_plot.index)
        if per_series is None or per_series.dropna().empty:
            st.info(
                "ヒストリカル PER を算出できませんでした"
                " (対象銘柄の四半期 EPS が取得できません)。"
            )

    # ----- Chart -----------------------------------------------------------
    fig, osc_rows = build_figure(
        df_plot,
        lines,
        show_sma=show_sma,
        show_sma100=show_sma100,
        show_sma200=show_sma200,
        show_ichimoku=show_ichimoku,
        show_support=show_support,
        show_resistance=show_resistance,
        show_trend_up=show_trend_up,
        show_trend_down=show_trend_down,
        show_invalid=show_invalid,
        ath=ath,
        atl=atl,
        initial_range=(initial_start, initial_end),
        ui_revision=(
            f"{ticker}-{interval}-{period_bars}"
            f"-{int(show_volume)}{int(show_macd)}{int(show_rsi)}{int(show_per_hist)}"
            f"-{int(show_sma)}{int(show_sma100)}{int(show_sma200)}"
        ),
        ticker_label=f"{ticker} {label}",
        currency_symbol=cur,
        show_volume=show_volume,
        show_macd=show_macd,
        show_rsi=show_rsi,
        show_per=show_per_hist,
        per_series=per_series,
        per_source=per_source,
        interval=interval,
    )
    # Editorial section divider above the chart: a thin ink rule with a
    # shu "seal" stamp in the middle. Replaces the duplicate "#### {ticker}"
    # heading that used to sit here and purely exists to visually
    # separate the fundamentals row from the chart showpiece below.
    st.markdown(
        '''
        <div class="kt-chart-divider">
            <span class="kt-chart-divider-kanji">— チャート解析</span>
            <span class="kt-chart-divider-line"></span>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # ----- Price Alert configuration (interactive) -------------------------
    from notification_ui import load_config, save_config
    from components.price_line_chart import price_line_chart

    noti_config = load_config()
    ticker_cfg = noti_config.setdefault("tickers", {}).setdefault(ticker, {})
    price_alerts = ticker_cfg.setdefault("price_alerts", [])

    # Current price
    curr_p = float(df_full["Close"].iloc[-1])

    # Show existing price alerts as horizontal lines on the Plotly fig
    for pa in price_alerts:
        p = pa["price"]
        direction = pa["direction"]
        dir_label = "上抜け" if direction == "above" else "下抜け"
        color = "#B7362E" if direction == "above" else "#2E6B47"
        fig.add_hline(
            y=p,
            line_color=color,
            line_width=1.5,
            line_dash="dash",
            annotation_text=f"🔔 {dir_label} {p:,.0f}円",
            annotation_position="bottom right",
            annotation_font_color=color,
            annotation_font_size=10,
            row=1, col=1,
        )

    # Show existing date alerts as vertical lines on the Plotly fig
    date_alerts = ticker_cfg.setdefault("date_alerts", [])
    for da in date_alerts:
        da_date = da.get("date", "")
        da_label = da.get("label") or da_date
        if da_date:
            fig.add_vline(
                x=da_date,
                line_color="#2E6B47",
                line_width=1.5,
                line_dash="dash",
            )
            fig.add_annotation(
                x=da_date,
                y=1,
                yref="paper",
                text=f"📅 {da_label}",
                showarrow=False,
                font=dict(color="#2E6B47", size=10),
                xanchor="left",
                yanchor="bottom",
            )

    # Show existing trendline alerts as shapes on the Plotly fig
    trendlines = ticker_cfg.setdefault("trendlines", [])
    for tl in trendlines:
        if not _is_valid_alert_trendline(tl):
            continue
        target = tl.get("target", "price")
        if target == "price":
            trow = 1
        else:
            trow = osc_rows.get(target.lower(), 1)
            
        fig.add_shape(
            type="line",
            name="user-alert-trendline",
            x0=tl["x0"],
            y0=tl["y0"],
            x1=tl["x1"],
            y1=tl["y1"],
            line=dict(color="#d65a31", width=2.0, dash="solid"),
            editable=True,
            layer="above",
            row=trow, col=1,
        )

    # Configure the default style for new user-drawn trendlines
    fig.update_layout(
        newshape=dict(
            line=dict(color="#d65a31", width=2.0, dash="solid")
        )
    )

    # ---- Render chart via custom component (zero-rerun draggable line) ----
    # Track processed alert IDs to prevent duplicate registration
    _processed_ids = st.session_state.setdefault("_processed_alert_ids", set())
    _chart_key = f"chart_line_{ticker}"

    chart_result = price_line_chart(
        fig,
        current_price=curr_p,
        height=fig.layout.height or 700,
        key=_chart_key,
    )

    # Network-backed fundamentals are deliberately rendered after the chart.
    # The reserved slot keeps the visual order unchanged without delaying the
    # primary ticker-switch feedback.
    if show_fundamentals:
        with fundamentals_slot.container():
            _render_fundamentals_section(
                ticker, float(last), interval, pd.Timestamp(df_full.index[-1])
            )

    # Process alert registration from the component
    if chart_result is not None:
        alert_id = chart_result.get("_id", "")
        if alert_id and alert_id not in _processed_ids:
            _processed_ids.add(alert_id)
            if "price" in chart_result:
                # ---- Price alert from horizontal drag ----
                new_price = float(chart_result["price"])
                new_dir = chart_result["direction"]
                price_alerts.append({
                    "price": new_price,
                    "direction": new_dir,
                })
                ticker_cfg["price_alert"] = True
                save_config(noti_config)
                dir_lbl = "上抜け" if new_dir == "above" else "下抜け"
                st.toast(f"✅ 価格アラート {new_price:,.0f}円（{dir_lbl}）を登録しました")
                st.rerun()
            elif "date" in chart_result:
                # ---- Date alert from vertical drag ----
                new_date = chart_result["date"]
                new_label = chart_result.get("label") or new_date
                date_alerts.append({
                    "date": new_date,
                    "label": new_label,
                })
                save_config(noti_config)
                st.toast(f"✅ 日付アラート {new_date}（{new_label}）を登録しました")
                st.rerun()
            elif "trendlines" in chart_result:
                # ---- Trendlines alert from user drawings ----
                yref_to_target = {"y": "price", "y1": "price"}
                for name, row in osc_rows.items():
                    yref_to_target[f"y{row}"] = name.upper()
                
                new_trendlines = []
                for tl in chart_result["trendlines"]:
                    if not _is_valid_alert_trendline(tl):
                        continue
                    yref = tl.get("yref", "y")
                    target = yref_to_target.get(yref, "price")
                    if target not in {"price", "RSI"}:
                        continue
                    tl_copy = dict(tl)
                    tl_copy["target"] = target
                    new_trendlines.append(tl_copy)

                ticker_cfg["trendlines"] = new_trendlines
                ticker_cfg["trendline_alert"] = True
                save_config(noti_config)
                st.toast("✅ トレンドラインのアラート設定を更新しました")
                st.rerun()

    # ---- Price Alert settings panel (below chart) --------------------------
    with st.expander("🔔 価格アラートの設定", expanded=False):
        # --- Manual input (always available as fallback) ---
        st.caption("価格を直接入力して追加（チャート上の📏ボタンからもラインを引けます）:")
        with st.form(key=f"price_alert_form_{ticker}"):
            col_ui1, col_ui2 = st.columns([7, 3])
            with col_ui1:
                alert_price_input = st.number_input(
                    "アラート設定価格 (円)",
                    min_value=0,
                    max_value=10_000_000,
                    value=int(curr_p),
                    step=10,
                    key=f"num_preview_price_{ticker}",
                )
            with col_ui2:
                new_dir = st.selectbox(
                    "方向",
                    ["above", "below"],
                    format_func=lambda x: "上抜け" if x == "above" else "下抜け",
                    key=f"sel_dir_main_{ticker}",
                )
            submit_btn = st.form_submit_button("➕ アラート追加", use_container_width=True)
            if submit_btn:
                price_alerts.append({
                    "price": float(alert_price_input),
                    "direction": new_dir,
                })
                ticker_cfg["price_alert"] = True
                save_config(noti_config)
                st.toast(f"価格アラート ({alert_price_input:,.0f}円) を追加しました。")
                st.rerun()

        # --- Current alert list ---
        if price_alerts:
            st.markdown("**現在設定されている価格アラート一覧:**")
            for idx, pa in enumerate(price_alerts):
                col_lst1, col_lst2, col_lst3 = st.columns([4, 4, 2])
                dir_label = "📈 上抜け" if pa["direction"] == "above" else "📉 下抜け"
                col_lst1.write(f"価格: `{pa['price']:,.0f}` 円")
                col_lst2.write(f"判定: `{dir_label}`")
                if col_lst3.button("🗑️ 削除", key=f"del_pa_main_{ticker}_{idx}"):
                    price_alerts.pop(idx)
                    ticker_cfg["price_alert"] = len(price_alerts) > 0
                    save_config(noti_config)
                    st.toast("価格しきい値条件を削除しました。")
                    st.rerun()

    # ----- Date Alert settings panel (below chart) --------------------------
    with st.expander("📅 日付アラートの設定", expanded=False):
        st.caption("チャート上の📅ボタンから縦線を引いて登録するか、直接日付を入力して追加できます:")
        import datetime as _dt
        with st.form(key=f"date_alert_form_{ticker}"):
            col_d1, col_d2 = st.columns([5, 5])
            with col_d1:
                alert_date_input = st.date_input(
                    "通知日付",
                    value=_dt.date.today() + _dt.timedelta(days=7),
                    key=f"date_input_{ticker}",
                )
            with col_d2:
                alert_label_input = st.text_input(
                    "ラベル（任意）",
                    placeholder="例: 決算発表日",
                    key=f"label_input_{ticker}",
                )
            submit_date_btn = st.form_submit_button("➕ 日付アラート追加", use_container_width=True)
            if submit_date_btn:
                # 土日なら最寄り営業日へ自動補正
                chosen = alert_date_input
                if chosen.weekday() == 5:   # 土曜 → 金曜
                    chosen = chosen - _dt.timedelta(days=1)
                    st.toast("⚠️ 土曜日のため金曜日に変更しました。")
                elif chosen.weekday() == 6:  # 日曜 → 月曜
                    chosen = chosen + _dt.timedelta(days=1)
                    st.toast("⚠️ 日曜日のため月曜日に変更しました。")
                date_alerts.append({
                    "date": str(chosen),
                    "label": alert_label_input.strip() or str(chosen),
                })
                save_config(noti_config)
                st.toast(f"📅 日付アラート ({chosen}) を追加しました。")
                st.rerun()


        if date_alerts:
            st.markdown("**現在設定されている日付アラート一覧:**")
            for idx, da in enumerate(date_alerts):
                col_d1, col_d2, col_d3 = st.columns([4, 4, 2])
                col_d1.write(f"日付: `{da['date']}`")
                col_d2.write(f"ラベル: `{da.get('label', da['date'])}`")
                if col_d3.button("🗑️ 削除", key=f"del_da_{ticker}_{idx}"):
                    date_alerts.pop(idx)
                    save_config(noti_config)
                    st.toast("日付アラートを削除しました。")
                    st.rerun()
        else:
            st.info("日付アラートはまだ設定されていません。")

    # ----- Earnings details table (J-Quants V2 summary data) ---------------
    if show_earnings:
        # 日本株（末尾が .T または数値）の場合のみ業績表示を試みる
        is_jp_stock = ticker.endswith(".T") or ticker.isdigit()

        if is_jp_stock:
            with st.expander("📊 業績・決算短信データ", expanded=False):
                from jquants import cleaned_summary

                with st.spinner("決算データを取得中..."):
                    clean_df = cleaned_summary(ticker)

                if clean_df is not None and not clean_df.empty:
                    latest_row = clean_df.iloc[-1]
                    fy_end = pd.to_datetime(latest_row.get("CurFYEn"))
                    fy_str = f"{fy_end.strftime('%Y/%m')}期" if pd.notna(fy_end) else "今期"

                    st.markdown(f"##### 🔮 {fy_str} 通期会社予想")
                    forecast_data = {
                        "指標": ["売上高", "営業利益", "経常利益", "当期純利益", "1株当たり利益 (FEPS)"],
                        "会社予想値": [
                            _format_financial_value(latest_row.get("FSales")),
                            _format_financial_value(latest_row.get("FOP")),
                            _format_financial_value(latest_row.get("FOdP")),
                            _format_financial_value(latest_row.get("FNP")),
                            _format_eps_value(latest_row.get("FEPS")),
                        ]
                    }
                    st.table(pd.DataFrame(forecast_data))

                    st.markdown("##### 📈 四半期決算短信実績 (開示日順)")
                    clean_df_desc = clean_df.sort_values("DiscDate", ascending=False)

                    rows = []
                    for _, row in clean_df_desc.iterrows():
                        cur_fy_end = pd.to_datetime(row.get("CurFYEn"))
                        per_type = row.get("CurPerType", "")
                        period_str = f"{cur_fy_end.strftime('%Y/%m')}期 {per_type}" if pd.notna(cur_fy_end) else per_type

                        rows.append({
                            "決算期": period_str,
                            "発表日": pd.to_datetime(row.get("DiscDate")).strftime("%Y-%m-%d"),
                            "売上高": _format_financial_value(row.get("Sales")),
                            "営業利益": _format_financial_value(row.get("OP")),
                            "経常利益": _format_financial_value(row.get("OdP")),
                            "当期純利益": _format_financial_value(row.get("NP")),
                            "1株益 (EPS)": _format_eps_value(row.get("EPS")),
                            "自己資本比率": _format_pct_value(row.get("EqAR")),
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.info("決算短信データが見つかりませんでした。(J-Quants APIキー設定状況や通信状態を確認してください)")
        else:
            with st.expander("📊 業績・決算短信データ", expanded=False):
                st.info("業績データの表示は日本株のみサポートされています。(指数・米国株等は対象外です)")

    # ----- Line details table ---------------------------------------------
    if lines:
        with st.expander("検出された線の詳細", expanded=False):
            rows = []
            for line in sorted(lines, key=lambda l: -l.score):
                rows.append(
                    {
                        "スケール": SCALE_LABELS_JA.get(
                            getattr(line, "scale", "long"), ""
                        ),
                        "種類": LINE_LABELS_JA[line.kind],
                        "開始": line.start_date.strftime("%Y-%m-%d"),
                        "終了": line.end_date.strftime("%Y-%m-%d"),
                        "開始価格": round(line.start_price, 2),
                        "終了価格": round(line.end_price, 2),
                        "タッチ数": len(line.touches),
                        "スコア": round(line.score, 1),
                        "検証": {True: "✅", False: "❌", None: "—"}[line.valid],
                    }
                )
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, use_container_width=True,
            )

    with st.expander("このアプリについて", expanded=False):
        st.markdown(
            """
            - **マルチスケール検出**: 短期 (約4ヶ月) / 中期 (約1年) / 長期 (約2年)
              の 3 つの時間軸でそれぞれ独立に検出し、重複は自動マージします。
            - **検証 (定義B)**: 線が検出された後の将来 90 営業日で、価格が
              その線に 2 回以上タッチし、少なくとも 1 回は逆方向に 2% 以上
              反発したかを判定します。検証 NG の線は UI からは非表示です。
            - **線のスタイル**: 実線 = 検証パス、ダッシュ = 未判定 (フォワード
              データ不足・直近の線など)。
            - **品質フィルタ**: タッチが短期間に集中した線、最後のタッチから
              長期間経過した「古い」線は自動的に除外されます。
            - チューニング済みパラメータは `artifacts/trend_params.json` に
              保存されており、全銘柄平均で **80% 以上の有効率** を達成しています。
            - **投資助言ではありません**。過去データに対するアルゴリズム的な
              パターン検出ツールです。
            """
        )

    # ----- Colophon (editorial footer) ------------------------------------
    # Front-page colophon block: type stack + data sources + the signature
    # 印 hanko in mincho italic. Purely decorative, but it closes the page
    # with the same editorial tone the masthead opened with.


if __name__ == "__main__":
    main()
