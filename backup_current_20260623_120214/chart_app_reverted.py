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
from plotly.subplots import make_subplots

from data import get_ticker_name, load_ohlcv, normalise_ticker
from evaluator import EvalParams, evaluate_lines
from indicators import ichimoku, macd, rsi, sma, volume_ma
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


def save_favorites(favs: dict[str, str]) -> None:
    FAVORITES_PATH.parent.mkdir(exist_ok=True)
    FAVORITES_PATH.write_text(
        json.dumps(favs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        from git_utils import git_push_changes
        git_push_changes("Update favorites list via UI")
    except Exception:
        pass


def add_favorite(ticker: str, name: str = "") -> None:
    favs = load_favorites()
    ticker = normalise_ticker(ticker)
    if not ticker:
        return
    # Don't overwrite an existing non-empty name with a blank one
    favs[ticker] = name or favs.get(ticker, "")
    save_favorites(favs)


def remove_favorite(ticker: str) -> None:
    favs = load_favorites()
    ticker = normalise_ticker(ticker)
    if ticker in favs:
        del favs[ticker]
        save_favorites(favs)


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

    ``df`` should be the FULL OHLCV history; ``initial_range`` controls
    which slice is visible on first render. Users can drag the chart
    axes/area to zoom & pan freely beyond that initial range.

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
    # Hide gaps in the data so the chart reads as a continuous series.
    # Strategy depends on the bar interval:
    #   - 1d:  list every missing calendar day explicitly (weekends AND
    #          Japanese holidays AND individual suspensions). Bounds-based
    #          weekend hiding alone leaves visible holes on holidays.
    #   - 1wk: no rangebreaks — weekly bars are already contiguous.
    rangebreaks: list[dict] = []
    if interval == "1d":
        all_days = pd.date_range(df.index[0], df.index[-1], freq="D")
        missing = sorted(set(all_days) - set(df.index))
        if missing:
            rangebreaks.append(
                dict(values=[d.strftime("%Y-%m-%d") for d in missing])
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


# ---------------------------------------------------------------------------
# Ticker selection sidebar
# ---------------------------------------------------------------------------
def _select_ticker(name_lookup: dict[str, str]) -> str | None:
    """Sidebar UI for choosing a ticker. Returns ticker symbol or None."""
    mode = st.sidebar.radio(
        "銘柄ソース",
        ["日経225", "指数", "お気に入り", "直接入力"],
        horizontal=True,
    )

    ticker: str | None = None

    if mode == "日経225":
        # Build a flat selectbox list where each sector is introduced by
        # a visual "header" row (── 電機 ──) that the user cannot pick as a
        # ticker. UNIVERSE is already sector-ordered, so we can detect
        # sector boundaries by watching the sector field change.
        items: list[tuple[str, str | None]] = []  # (display_label, value-or-None)
        prev_sector: str | None = None
        for code, nm, sector in UNIVERSE:
            if sector != prev_sector:
                items.append((f"── {sector} ──", None))
                prev_sector = sector
            items.append((f"    {code}  {nm}", code))

        # Default to whichever row Toyota lives on
        default_idx = next(
            (i for i, (_lbl, val) in enumerate(items) if val == "7203.T"),
            1,  # fallback: first real ticker (0 is a header)
        )
        choice_idx = st.sidebar.selectbox(
            "銘柄",
            options=range(len(items)),
            format_func=lambda i: items[i][0],
            index=default_idx,
            key="nikkei225_pick_idx",
        )
        chosen = items[choice_idx][1]
        if chosen is None:
            # User clicked on a section header — these aren't real tickers.
            # Remember the last valid pick in session_state so the chart
            # doesn't blank out when this happens.
            st.sidebar.caption(
                "※ 業種名の行は選択できません。銘柄行を選んでください。"
            )
            ticker = st.session_state.get("nikkei225_last_ticker", "7203.T")
        else:
            ticker = chosen
            st.session_state["nikkei225_last_ticker"] = ticker

    elif mode == "指数":
        opts = {f"{code}  {n}": code for code, n in INDICES}
        label = st.sidebar.selectbox("指数", list(opts.keys()), index=0)
        ticker = opts[label]

    elif mode == "お気に入り":
        favs = load_favorites()
        if not favs:
            st.sidebar.info(
                "お気に入りはまだありません。他のモードで銘柄を選び、"
                "「⭐ お気に入りに追加」ボタンで登録してください。"
            )
        else:
            # Resolve a display name for every favorite. Order of preference:
            #   1) the name already stored in favorites.json
            #   2) built-in Nikkei225 / INDICES lookup
            #   3) yfinance (via _display_name_for → cached for a week)
            # When (3) fires we also write the result back to favorites.json
            # so subsequent sessions don't need the yfinance round-trip.
            opts: dict[str, str] = {}
            dirty = False
            for t, stored_name in favs.items():
                if stored_name:
                    display_name = stored_name
                else:
                    display_name = _display_name_for(t, name_lookup)
                    if display_name and display_name != t:
                        favs[t] = display_name
                        dirty = True
                label_text = (
                    f"{t}  {display_name}" if display_name and display_name != t
                    else t
                )
                opts[label_text] = t
            if dirty:
                save_favorites(favs)
            label = st.sidebar.selectbox("お気に入り", list(opts.keys()))
            ticker = opts[label]

    else:  # 直接入力
        raw = st.sidebar.text_input(
            "ティッカー (例: 7203, 6758.T, ^N225)",
            value="",
            help="4 桁コードだけ入力すると .T を自動付与します。"
            " 指数や米国株は yfinance 形式でそのまま入力してください。",
        )
        if raw.strip():
            ticker = normalise_ticker(raw)

    # Favorite add/remove button
    if ticker:
        st.sidebar.markdown("---")
        favs = load_favorites()
        if ticker in favs:
            if st.sidebar.button(f"⭐ お気に入りから外す ({ticker})"):
                remove_favorite(ticker)
                st.rerun()
        else:
            if st.sidebar.button(f"☆ お気に入りに追加 ({ticker})"):
                # Resolve name: built-in lookup first, then yfinance
                display_name = name_lookup.get(ticker, "")
                if not display_name:
                    with st.spinner(f"{ticker} の銘柄名を取得中..."):
                        display_name = get_ticker_name(ticker) or ""
                add_favorite(ticker, display_name)
                st.rerun()

    return ticker


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
def _render_watchlist_panel(name_lookup: dict[str, str]) -> None:
    """Render a fixed-position, resizable TradingView-style watchlist panel.

    The panel is injected as raw HTML via st.markdown so it sits outside
    Streamlit's normal column flow.  Resizing is handled by a JS drag-handle
    on the left edge of the panel.  Clicking a row calls pickTicker() which
    sets ?ticker=CODE in the URL; Python reads it on the next rerun.
    """
    import html as _html
    import json

    favs   = load_favorites()
    # Fetch prices for ALL tickers up front (unique sorted)
    all_codes_set = set(
        [code for code, _, _ in UNIVERSE]
        + [code for code, _ in INDICES]
        + list(favs.keys())
    )
    all_codes = sorted(list(all_codes_set))
    prices = _fetch_last_prices(tuple(all_codes))
    selected = st.session_state.get("selected_ticker", get_default_ticker())

    # ---- helper: build one HTML row ----
    def _row(code: str, nm: str) -> str:
        p     = prices.get(code, {})
        last_p = p.get("last")
        pct   = p.get("pct", 0.0)
        is_jp = code.endswith(".T") or code.isdigit()
        if last_p is not None:
            pr_str = f"&yen;{last_p:,.0f}" if is_jp else f"{last_p:,.2f}"
            if pct > 0:
                chg_span = f'<span class="wlup">&#9650;{pct:.2f}%</span>'
            elif pct < 0:
                chg_span = f'<span class="wldn">&#9660;{abs(pct):.2f}%</span>'
            else:
                chg_span = f'<span class="wlfl">0.00%</span>'
        else:
            pr_str   = "&mdash;"
            chg_span = '<span class="wlfl">&mdash;</span>'
        sel   = " wlsel" if code == selected else ""
        safe_nm = _html.escape(nm[:16] + ("…" if len(nm) > 16 else ""))
        safe_code = _html.escape(code)
        return (
            f'<div class="wlrow{sel}" onclick="pickTicker(\'{safe_code}\')">'
            f'<div class="wll"><span class="wlsym">{safe_code}</span>'
            f'<span class="wlnm">{safe_nm}</span></div>'
            f'<div class="wlr"><span class="wlpr">{pr_str}</span>'
            f'{chg_span}</div></div>'
        )

    # ---- build tab contents ----
    nk_rows = []
    prev_sec = None
    for code, nm, sec in UNIVERSE:
        if sec != prev_sec:
            nk_rows.append(
                f'<div class="wlsec">{_html.escape(sec)}</div>'
            )
            prev_sec = sec
        nk_rows.append(_row(code, nm))

    if favs:
        fav_rows = [_row(t, stored or name_lookup.get(t, t))
                    for t, stored in favs.items()]
    else:
        fav_rows = ['<div class="wlempty">お気に入りはまだありません</div>']

    idx_rows = [_row(code, nm) for code, nm in INDICES]

    # all items for search (rendered hidden, JS filters inline)
    srch_rows = (
        [_row(c, n) for c, n, _ in UNIVERSE]
        + [_row(c, n) for c, n in INDICES]
    )

    panel_html = f"""
<style>
/* Hide default Streamlit Header (Deploy and menu buttons) */
[data-testid="stHeader"] {{
    display: none !important;
}}
/* ===== Fixed resizable watchlist panel ===== */
#wlPanel {{
    position: fixed;
    right: 0;
    top: 0;
    height: 100vh;
    width: 280px;
    min-width: 160px;
    max-width: 50vw;
    background: rgba(10,18,36,0.97);
    border-left: 1px solid rgba(255,255,255,0.1);
    z-index: 9999;
    display: flex;
    flex-direction: row;
    font-family: 'Inter', sans-serif;
    box-shadow: -4px 0 20px rgba(0,0,0,0.4);
    backdrop-filter: blur(12px);
}}
#wlHandle {{
    width: 5px;
    cursor: ew-resize;
    background: transparent;
    flex-shrink: 0;
    transition: background 0.15s;
    position: relative;
}}
#wlHandle:hover, #wlHandle.dragging {{ background: rgba(56,189,248,0.5); }}
#wlHandle::after {{
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%,-50%);
    width: 3px;
    height: 40px;
    border-radius: 2px;
    background: rgba(255,255,255,0.15);
}}
#wlInner {{
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
}}
.wltitle {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94A3B8;
    padding: 36px 12px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    flex-shrink: 0;
}}
.wltabbar {{
    display: flex;
    gap: 2px;
    padding: 6px 8px 4px;
    flex-shrink: 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.wltab {{
    flex: 1;
    background: transparent;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 4px;
    color: #64748B;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 4px 2px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: center;
}}
.wltab:hover {{ color: #CBD5E1; border-color: rgba(255,255,255,0.2); }}
.wltab.active {{ background: rgba(56,189,248,0.15); border-color: rgba(56,189,248,0.4); color: #38BDF8; }}
#wlSearch {{
    margin: 6px 8px;
    padding: 5px 8px;
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 5px;
    color: #F1F5F9;
    font-size: 0.75rem;
    width: calc(100% - 16px);
    box-sizing: border-box;
    flex-shrink: 0;
}}
#wlSearch:focus {{ outline: none; border-color: rgba(56,189,248,0.5); }}
.wlscroll {{
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
}}
.wlscroll::-webkit-scrollbar {{ width: 4px; }}
.wlscroll::-webkit-scrollbar-track {{ background: transparent; }}
.wlscroll::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 2px; }}
.wlsec {{
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    padding: 8px 10px 3px;
    background: rgba(0,0,0,0.2);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}}
.wlrow {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.035);
    cursor: pointer;
    transition: background 0.1s;
}}
.wlrow:hover {{ background: rgba(56,189,248,0.07); }}
.wlsel {{ background: rgba(56,189,248,0.13) !important; border-left: 2px solid #38BDF8; }}
.wll {{ display: flex; flex-direction: column; min-width: 0; }}
.wlr {{ display: flex; flex-direction: column; align-items: flex-end; flex-shrink: 0; margin-left: 4px; }}
.wlsym {{ font-family: ui-monospace,monospace; font-size: 0.72rem; font-weight: 600; color: #E2E8F0; white-space: nowrap; }}
.wlnm  {{ font-size: 0.62rem; color: #475569; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 110px; }}
.wlpr  {{ font-family: ui-monospace,monospace; font-size: 0.72rem; font-weight: 600; color: #CBD5E1; }}
.wlup  {{ font-size: 0.65rem; color: #34D399; }}
.wldn  {{ font-size: 0.65rem; color: #F472B6; }}
.wlfl  {{ font-size: 0.65rem; color: #475569; }}
.wlempty {{ padding: 20px 12px; font-size: 0.75rem; color: #475569; text-align: center; }}
/* Push main content left so chart isn't hidden under panel */
.main .block-container {{ margin-right: 300px !important; }}
</style>

<div id="wlPanel">
  <div id="wlHandle"></div>
  <div id="wlInner">
    <div class="wltitle">&#x1F4CB; ウォッチリスト</div>
    <div class="wltabbar">
      <button class="wltab active" onclick="switchTab('nk',this)">日経225</button>
      <button class="wltab" onclick="switchTab('fav',this)">お気に入り</button>
      <button class="wltab" onclick="switchTab('idx',this)">指数</button>
      <button class="wltab" onclick="switchTab('srch',this)">検索</button>
    </div>
    <input type="text" id="wlSearch" placeholder="&#x1F50D; 銘柄を検索..." oninput="filterSearch()" style="display:none">
    <div class="wlscroll" id="wlBody">
      <div id="tab-nk">{''.join(nk_rows)}</div>
      <div id="tab-fav" style="display:none">{''.join(fav_rows)}</div>
      <div id="tab-idx" style="display:none">{''.join(idx_rows)}</div>
      <div id="tab-srch" style="display:none">{''.join(srch_rows)}</div>
    </div>
  </div>
</div>

<script>
(function() {{
  // Hide Streamlit hidden buttons cleanly
  var allCodes = {json.dumps(all_codes)};
  function hideButtons() {{
    var docs = [document];
    if (window.parent) docs.push(window.parent.document);
    docs.forEach(function(doc) {{
      var buttons = doc.querySelectorAll('button');
      buttons.forEach(function(b) {{
        var t = b.textContent.trim();
        if (t && allCodes.includes(t)) {{
            var container = b.closest('div[data-testid="element-container"]');
            if (container) {{
                container.style.display = 'none';
                container.style.position = 'absolute';
                container.style.left = '-9999px';
            }}
        }}
      }});
    }});
  }}
  // Run hiding repeatedly for a short time to catch Streamlit's async rendering
  hideButtons();
  setTimeout(hideButtons, 100);
  setTimeout(hideButtons, 500);
  setTimeout(hideButtons, 1000);

  // ---- Tab switching ----
  function switchTab(tab, btn) {{
    ['nk','fav','idx','srch'].forEach(function(t) {{
      document.getElementById('tab-'+t).style.display = t===tab ? 'block' : 'none';
    }});
    document.querySelectorAll('.wltab').forEach(function(b) {{ b.classList.remove('active'); }});
    btn.classList.add('active');
    var srch = document.getElementById('wlSearch');
    srch.style.display = tab==='srch' ? 'block' : 'none';
    if (tab === 'srch') {{ srch.focus(); }}
  }}
  window.switchTab = switchTab;

  // ---- Ticker pick (programmatically click Streamlit button to keep session state) ----
  window.pickTicker = function(code) {{
    // 1. Update the browser URL query parameter without page reload
    try {{
      var url = new URL(window.location.href);
      url.searchParams.set('ticker', code);
      window.history.pushState({{}}, '', url.toString());
      if (window.parent && window.parent.history) {{
        window.parent.history.pushState({{}}, '', url.toString());
      }}
    }} catch (e) {{
      console.warn("Failed to update history state:", e);
    }}

    // 2. Programmatically click the hidden Streamlit button matching the ticker text content
    var doc = window.parent ? window.parent.document : document;
    var buttons = doc.querySelectorAll('button');
    for (var i = 0; i < buttons.length; i++) {{
      if (buttons[i].textContent.trim() === code) {{
        buttons[i].click();
        return;
      }}
    }}
    // Try local document as fallback
    buttons = document.querySelectorAll('button');
    for (var i = 0; i < buttons.length; i++) {{
      if (buttons[i].textContent.trim() === code) {{
        buttons[i].click();
        return;
      }}
    }}

    // Fallback: if button click failed, do a full page reload
    var url = new URL(window.location.href);
    url.searchParams.set('ticker', code);
    window.location.href = url.toString();
  }};

  // ---- Search filter ----
  window.filterSearch = function() {{
    var q = document.getElementById('wlSearch').value.toLowerCase();
    var rows = document.querySelectorAll('#tab-srch .wlrow');
    rows.forEach(function(r) {{
      r.style.display = r.textContent.toLowerCase().includes(q) ? 'flex' : 'none';
    }});
  }};

  // ---- Drag-to-resize handle ----
  var panel  = document.getElementById('wlPanel');
  var handle = document.getElementById('wlHandle');
  var dragging = false;
  var startX, startW;

  handle.addEventListener('mousedown', function(e) {{
    dragging = true;
    startX = e.clientX;
    startW = panel.offsetWidth;
    handle.classList.add('dragging');
    document.body.style.userSelect = 'none';
    e.preventDefault();
  }});
  document.addEventListener('mousemove', function(e) {{
    if (!dragging) return;
    var dx = startX - e.clientX;   // drag left → bigger panel
    var newW = Math.min(Math.max(startW + dx, 160), window.innerWidth * 0.5);
    panel.style.width = newW + 'px';
    // Update main content margin
    var mc = document.querySelector('.main .block-container');
    if (mc) {{
      mc.style.marginRight = (newW + 10) + 'px';
      mc.style.maxWidth = 'calc(100% - ' + (newW + 10) + 'px)';
    }}
  }});
  document.addEventListener('mouseup', function() {{
    if (dragging) {{
      dragging = false;
      handle.classList.remove('dragging');
      document.body.style.userSelect = '';
    }}
  }});
}})();
</script>
"""
    st.markdown(panel_html, unsafe_allow_html=True)

    # Render hidden Streamlit buttons to receive clicks from JS (preserves session state)
    with st.container():
        st.markdown('<div class="wl-hidden-btn-trigger"></div>', unsafe_allow_html=True)
        for code in all_codes:
            if st.button(code, key=f"wl_hidden_btn_{code}"):
                st.session_state["selected_ticker"] = code
                st.rerun()



def main() -> None:
    st.set_page_config(
        page_title="トレンドライン自動検出",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Check if we should pull latest settings from Git repository (run once per session)
    if "git_pulled" not in st.session_state:
        try:
            from git_utils import git_pull
            git_pull()
        except Exception:
            pass
        st.session_state.git_pulled = True
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
        from notification_ui import show_notification_ui
        show_notification_ui(name_lookup)
        return

# ----- Fixed right watchlist panel (HTML/JS, no tabs) ---------------
    _render_watchlist_panel(name_lookup)

    # Read ticker from URL query param (set by watchlist JS click)
    _qp_ticker = st.query_params.get("ticker", "")
    if _qp_ticker and _qp_ticker != st.session_state.get("selected_ticker", ""):
        st.session_state["selected_ticker"] = _qp_ticker
    
    # Ensure selected_ticker is initialized to default if empty/not present
    if "selected_ticker" not in st.session_state or not st.session_state["selected_ticker"]:
        st.session_state["selected_ticker"] = get_default_ticker()
        
    ticker = st.session_state["selected_ticker"]
    st.caption(f"選択中の銘柄: {ticker}")


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
        # drawn regardless of the per-kind checkboxes below. Useful for
        # looking at a clean candlestick view without any overlays.
        show_trendlines = st.checkbox("🔺 トレンドラインを表示", value=False)
        show_support = (
            st.checkbox("🟢 サポート", value=False) and show_trendlines
        )
        show_resistance = (
            st.checkbox("🔴 レジスタンス", value=False) and show_trendlines
        )
        show_trend_up = (
            st.checkbox("🔵 上昇トレンド", value=False) and show_trendlines
        )
        show_trend_down = (
            st.checkbox("🟠 下降トレンド", value=False) and show_trendlines
        )

    with st.sidebar.expander("表示切替: その他", expanded=True):
        show_ath_atl = st.checkbox("🟡 上場来高値・安値", value=False)

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
    # Use cached detection to avoid O(n²) recalculation on every rerender
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

    # ----- 最新データ取得日の表示 -----------------------------------------
    # チャート / テクニカル指標は df_full の最終行 (=yfinance が返した最新
    # バー) が基準。ファンダメンタルズは load_fundamentals の TTL=1h キャッシュ
    # 充填時刻 (fund["fetched_at"]) が基準 — 再描画ごとに now を表示すると
    # 実データが古いままでも「今取得」に見えてしまう。
    fund = load_fundamentals(ticker)
    _bar_unit = "営業日" if interval == "1d" else "週"
    _latest_bar_str = f"{df_full.index[-1]:%Y年%m月%d日}"
    _fetched_at = fund.get("fetched_at")
    _fund_str = (
        _fetched_at.strftime("%Y年%m月%d日 %H:%M")
        if isinstance(_fetched_at, pd.Timestamp) else "—"
    )
    st.caption(
        f"📅 チャート・テクニカル指標の最新{_bar_unit}: **{_latest_bar_str}**"
        f" ／ ファンダメンタルズ取得時刻: {_fund_str} (yfinance, 1時間キャッシュ)"
    )

    # ----- Fundamentals row (PER / PBR / 配当利回り) ----------------------
    st.markdown(
        '''
        <div class="kt-section-label">
            ファンダメンタルズ — 指標
        </div>
        ''',
        unsafe_allow_html=True,
    )
    f1, f2, f3 = st.columns(3)

    # --- PER ---------------------------------------------------------------
    # Prefer J-Quants 会社予想 EPS (the number the company itself issued
    # in its latest 決算短信) for the displayed "PER". yfinance's
    # ``forwardPE`` is used as a fallback for non-JP tickers and for
    # issuers J-Quants doesn't cover. The direct yfinance number is
    # unreliable on post-split JP stocks — e.g. 7013 IHI after its
    # 2025-09-29 1:7 split shows ``forwardPE ≈ 5`` because yfinance
    # failed to adjust the forecast EPS — so we compute the PER
    # ourselves from the split-adjusted close ÷ split-adjusted forecast.
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
        # Fallback: yfinance forwardPE (non-JP or uncovered issuers).
        yf_fwd_eps = fund.get("forward_eps")
        yf_fwd_per = fund.get("forward_per")
        if yf_fwd_eps is not None and yf_fwd_eps <= 0:
            per_is_loss = True
        elif yf_fwd_per and yf_fwd_per > 0:
            per_val = float(yf_fwd_per)

    # Custom card so the 算出基準 note can live *inside* the same white
    # frame as the value (st.metric doesn't let us inject extra content).
    # The .kt-metric-card CSS class mirrors stMetric styling exactly.
    if per_is_loss:
        per_value_html = "赤字"
    elif per_val is not None and per_val > 0:
        per_value_html = f"{per_val:.1f} 倍"
    else:
        per_value_html = "—"
    f1.markdown(
        f'''
        <div class="kt-metric-card">
          <div class="kt-metric-label">PER</div>
          <div class="kt-metric-value">{per_value_html}</div>
          <div class="kt-metric-note">
            ※ 会社予想EPS基準のため他ツール (Yahoo等) と数値が異なる場合があります
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # --- PBR ---------------------------------------------------------------
    # Uses the same .kt-metric-card class as PER so the three fundamentals
    # cards stay height-aligned even when PER has its 算出基準 note.
    pbr = fund.get("pbr")
    pbr_value_html = f"{pbr:.2f} 倍" if pbr and pbr > 0 else "—"
    f2.markdown(
        f'''
        <div class="kt-metric-card">
          <div class="kt-metric-label">PBR</div>
          <div class="kt-metric-value">{pbr_value_html}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # --- 配当利回り --------------------------------------------------------
    # Compute yield ourselves from split-adjusted annual dividend ÷ current
    # close, rather than trusting yfinance's pre-formatted yield fields.
    # Rationale: yfinance is inconsistent about whether ``dividendYield``
    # is a fraction (0.0286 → 2.86%) or a percent (0.61 → 0.61%), and the
    # heuristic we used (v > 1 ⇒ percent) misfires on every sub-1% payer
    # and reports them as ~60%+. ``dividendRate`` is always in yen per
    # share and is split-adjusted, so dividing by the latest close is
    # unambiguous.
    #
    # Three display states:
    #   (a) 配当あり — rate > 0      → percent yield
    #   (b) 無配     — rate == 0     → "無配"
    #   (c) 不明     — rate is None  → "—"
    rate = fund.get("dividend_rate")
    trailing_rate = fund.get("trailing_dividend_rate")
    # Prefer forward rate (current-year company guidance); fall back to
    # the trailing-12m figure when the forecast rate is absent.
    annual_dps: float | None = None
    for candidate in (rate, trailing_rate):
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
        yield_pct = annual_dps / float(last) * 100
        div_value_html = f"{yield_pct:.2f}%"
    f3.markdown(
        f'''
        <div class="kt-metric-card">
          <div class="kt-metric-label">配当利回り</div>
          <div class="kt-metric-value">{div_value_html}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # ----- Historical PER (optional oscillator) ---------------------------
    # Computed lazily only when the panel is enabled. The underlying
    # FY step-function data is cached for 12h inside ``_load_fy_eps_steps``.
    per_series: pd.Series | None = None
    per_source: str = ""
    if show_per_hist:
        per_series, per_source = _cached_historical_per_series(
            ticker, interval, len(df_full)
        )
        if per_series is None or per_series.dropna().empty:
            st.info(
                "ヒストリカル PER を算出できませんでした"
                " (対象銘柄の四半期 EPS が取得できません)。"
            )

    # ----- Chart -----------------------------------------------------------
    fig, osc_rows = build_figure(
        df_full,  # pass full history; user pans/zooms within it
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
        target = tl.get("target", "price")
        if target == "price":
            trow = 1
        else:
            trow = osc_rows.get(target.lower(), 1)
            
        fig.add_shape(
            type="line",
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

    # Debug logging to trace shapes and result
    try:
        from datetime import datetime
        shapes_info = []
        if hasattr(fig.layout, "shapes") and fig.layout.shapes:
            for s in fig.layout.shapes:
                shapes_info.append({
                    "type": getattr(s, "type", None),
                    "x0": getattr(s, "x0", None),
                    "y0": getattr(s, "y0", None),
                    "x1": getattr(s, "x1", None),
                    "y1": getattr(s, "y1", None),
                    "editable": getattr(s, "editable", None),
                    "layer": getattr(s, "layer", None)
                })
        with open(r"c:\Users\matsu\OneDrive\claude\stock_future\debug_trendlines.txt", "a", encoding="utf-8") as debug_f:
            debug_f.write(f"\n--- RUN ticker={ticker} time={datetime.now().isoformat()} ---\n")
            debug_f.write(f"Config trendlines: {ticker_cfg.get('trendlines')}\n")
            debug_f.write(f"fig.layout.shapes: {shapes_info}\n")
            debug_f.write(f"chart_result: {chart_result}\n")
    except Exception as e:
        with open(r"c:\Users\matsu\OneDrive\claude\stock_future\debug_trendlines.txt", "a", encoding="utf-8") as debug_f:
            debug_f.write(f"Logging error: {str(e)}\n")

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
                    yref = tl.get("yref", "y")
                    target = yref_to_target.get(yref, "price")
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
    from jquants import cleaned_summary

    # 日本株（末尾が .T または数値）の場合のみ業績表示を試みる
    is_jp_stock = ticker.endswith(".T") or ticker.isdigit()

    if is_jp_stock:
        with st.expander("📊 業績・決算短信データ", expanded=False):
            with st.spinner("決算データを取得中..."):
                clean_df = cleaned_summary(ticker)

            if clean_df is not None and not clean_df.empty:
                # 1. 会社通期予想値の表示 (最新レコードから)
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

                # 2. 四半期実績推移の表示 (降順)
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
