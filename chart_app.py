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
from indicators import ichimoku, macd, sma, volume_ma
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
    "paper":         "#F4EFE3",  # washi page background
    "paper_alt":     "#ECE5D3",
    "surface":       "#FBF8EF",  # card / chart surface
    "ink":           "#14110E",  # sumi ink — text, borders, structure
    "ink_soft":      "#2A251E",
    "ink_muted":     "#5C554A",  # secondary labels, tick marks
    "border":        "#CFC6B0",
    "border_strong": "#A89B7A",
    "shu":           "#B7362E",  # 朱 vermilion — signature accent, rising candles
    "shu_deep":      "#8A231C",
    "forest":        "#2E6B47",  # deep green — falling candles, support
    "navy":          "#1E3A5F",  # 紺 — rising trend lines, indicators
    "copper":        "#8A6E3A",  # aged gold — falling trend lines, MACD signal
    "gold":          "#9B7421",  # ATH / ATL markers
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
@import url('https://fonts.googleapis.com/css2?family=Shippori+Mincho:wght@400;500;600;700&family=IBM+Plex+Sans+JP:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ---- Root tokens as CSS variables (match the Python THEME dict) ----- */
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
    --font-display:  'Shippori Mincho', 'Hiragino Mincho ProN',
                     'Yu Mincho', serif;
    --font-body:     'IBM Plex Sans JP', 'Hiragino Sans',
                     'Noto Sans CJK JP', sans-serif;
    --font-mono:     'IBM Plex Mono', 'Menlo', monospace;
}}

/* ---- Base typography + paper page background ------------------------ */
html, body, [class*="st-"] {{
    font-family: var(--font-body);
    font-weight: 400;
    letter-spacing: 0.01em;
    color: var(--ink);
}}
.stApp {{
    background-color: var(--paper);
    /* Subtle paper grain via inline SVG turbulence */
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0.08 0 0 0 0 0.07 0 0 0 0 0.05 0 0 0 0.05 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    color: var(--ink);
}}

/* ---- Headings use the mincho display serif ------------------------- */
h1, h2, h3, h4, h5,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {{
    font-family: var(--font-display) !important;
    color: var(--ink);
    font-weight: 600;
    letter-spacing: 0;
    line-height: 1.35;
}}
h1 {{ font-size: 2.375rem; font-weight: 700; }}
h2 {{ font-size: 1.5rem; }}
h3 {{ font-size: 1.125rem; }}

/* Hero title — hanko square accent + ink rule underneath */
.stApp h1:first-of-type {{
    position: relative;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--ink);
    display: inline-block;
    margin-bottom: 0.5rem;
}}

/* ---- Caption = muted italic mincho (editorial byline) ------------- */
[data-testid="stCaptionContainer"], .stCaption,
[data-testid="stCaptionContainer"] p {{
    font-family: var(--font-display) !important;
    font-style: italic;
    color: var(--ink-muted) !important;
    font-size: 0.9375rem;
    line-height: 1.75;
}}

/* ---- Sidebar — vertical paper strip + vermilion margin rule ------ */
[data-testid="stSidebar"] {{
    background: var(--surface);
    border-right: 1px solid var(--ink);
    box-shadow: 2px 0 0 var(--shu);
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    font-family: var(--font-body) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--shu) !important;
    font-weight: 600;
    margin-top: 1.75rem;
    padding-bottom: 0.375rem;
    border-bottom: 1px solid var(--ink);
}}

/* ---- Widget labels -------------------------------------------------- */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] {{
    color: var(--ink) !important;
    font-family: var(--font-body) !important;
    font-size: 0.8125rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}}

/* ---- Metric cards — square ink frame, hanko-stamp hover accent --- */
[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--ink);
    border-radius: 0;
    padding: 20px 24px;
    position: relative;
    transition: transform 200ms ease-out;
}}
[data-testid="stMetric"]::after {{
    content: "";
    position: absolute;
    left: -1px;
    right: -1px;
    bottom: -4px;
    height: 3px;
    background: var(--shu);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 280ms cubic-bezier(0.2, 0.7, 0.2, 1);
}}
[data-testid="stMetric"]:hover::after {{
    transform: scaleX(1);
}}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {{
    color: var(--ink-muted) !important;
    font-family: var(--font-body) !important;
    font-size: 0.6875rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {{
    font-family: var(--font-mono) !important;
    color: var(--ink) !important;
    font-weight: 500;
    font-size: 1.75rem;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
    margin-top: 0.375rem;
}}
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] div {{
    font-family: var(--font-mono) !important;
    font-variant-numeric: tabular-nums;
    font-size: 0.8125rem;
    font-weight: 500;
}}

/* ---- Custom metric card (used for PER/PBR/配当利回り so we can
       embed an inline note inside the same white frame as the value,
       AND keep all three fundamentals cards at matching heights).
       Structurally mirrors [data-testid="stMetric"] so the look
       matches if a real stMetric sits next to one. ---------------- */
.kt-metric-card {{
    background: var(--surface);
    border: 1px solid var(--ink);
    border-radius: 0;
    padding: 20px 24px;
    position: relative;
    transition: transform 200ms ease-out;
    /* Fill the column so sibling cards stay aligned even when one
       has an extra note underneath the value. */
    min-height: 148px;
    height: 100%;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
}}
.kt-metric-card::after {{
    content: "";
    position: absolute;
    left: -1px;
    right: -1px;
    bottom: -4px;
    height: 3px;
    background: var(--shu);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 280ms cubic-bezier(0.2, 0.7, 0.2, 1);
}}
.kt-metric-card:hover::after {{
    transform: scaleX(1);
}}
.kt-metric-card .kt-metric-label {{
    color: var(--ink-muted);
    font-family: var(--font-body);
    font-size: 0.6875rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}}
.kt-metric-card .kt-metric-value {{
    font-family: var(--font-mono);
    color: var(--ink);
    font-weight: 500;
    font-size: 1.75rem;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
    margin-top: 0.375rem;
}}
.kt-metric-card .kt-metric-note {{
    color: var(--ink-muted);
    font-family: var(--font-display);
    font-style: italic;
    font-size: 0.75rem;
    line-height: 1.45;
    margin-top: auto;
    padding-top: 0.75rem;
    opacity: 0.9;
}}

/* ---- Hero card (the prominent ticker/name + price block that sits
       above the fundamentals row). Uses a large mincho display title
       for the ticker+company name so the identity of the asset is
       instantly readable, with the price + delta underneath. ----- */
.kt-hero-card {{
    background: var(--surface);
    border: 1px solid var(--ink);
    border-radius: 0;
    padding: 22px 28px 24px 28px;
    position: relative;
    margin-bottom: 0.875rem;
}}
.kt-hero-card::after {{
    content: "";
    position: absolute;
    left: -1px;
    right: -1px;
    bottom: -4px;
    height: 3px;
    background: var(--shu);
}}
.kt-hero-card .kt-hero-title {{
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 1.75rem;
    color: var(--ink);
    line-height: 1.25;
    letter-spacing: 0.01em;
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.6em;
}}
.kt-hero-card .kt-hero-ticker {{
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: 1.3rem;
    color: var(--shu);
    letter-spacing: 0.04em;
    padding: 2px 10px;
    border: 1px solid var(--shu);
    background: var(--paper);
}}
.kt-hero-card .kt-hero-price-row {{
    margin-top: 0.875rem;
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 1.1em;
}}
.kt-hero-card .kt-hero-price {{
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 2.375rem;
    font-weight: 500;
    color: var(--ink);
    line-height: 1;
}}
.kt-hero-card .kt-hero-delta {{
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 1rem;
    font-weight: 500;
    padding: 3px 10px;
    border: 1px solid currentColor;
}}
.kt-hero-card .kt-hero-delta.up    {{ color: var(--shu);    }}
.kt-hero-card .kt-hero-delta.down  {{ color: var(--forest); }}
.kt-hero-card .kt-hero-delta.flat  {{ color: var(--ink-muted); }}

/* ---- Buttons — stamped ink block that shifts on press ------------ */
.stButton > button,
.stFormSubmitButton > button,
[data-testid="stFormSubmitButton"] button {{
    background: var(--surface);
    color: var(--ink);
    border: 1px solid var(--ink);
    border-radius: 0;
    font-family: var(--font-body);
    font-weight: 500;
    padding: 0.5rem 1.25rem;
    letter-spacing: 0.06em;
    box-shadow: 3px 3px 0 var(--ink);
    transition: transform 120ms ease-out, box-shadow 120ms ease-out,
                background 150ms ease-out, color 150ms ease-out;
}}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {{
    background: var(--shu);
    color: var(--paper);
    box-shadow: 5px 5px 0 var(--ink);
    transform: translate(-2px, -2px);
}}
.stButton > button:active,
.stFormSubmitButton > button:active,
[data-testid="stFormSubmitButton"] button:active {{
    box-shadow: 1px 1px 0 var(--ink);
    transform: translate(2px, 2px);
}}
.stButton > button:focus:not(:active),
.stFormSubmitButton > button:focus:not(:active),
[data-testid="stFormSubmitButton"] button:focus:not(:active) {{
    outline: 2px solid var(--shu);
    outline-offset: 2px;
}}

/* ---- Inputs / selectbox (closed state) ---------------------------- */
.stSelectbox [data-baseweb="select"] > div,
.stTextInput [data-baseweb="input"],
.stNumberInput [data-baseweb="input"] {{
    border-radius: 0 !important;
    border: 1px solid var(--ink) !important;
    background: #ffffff !important;
}}
.stSelectbox [data-baseweb="select"] div,
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] input {{
    color: var(--ink) !important;
    font-family: var(--font-body) !important;
}}

/* Force the inner <input> element of text/number inputs to a legible
   white-on-black-text combo. Baseweb nests the actual editable element
   a few levels deep, so the wrapper rule above doesn't reach it and
   the text comes out dark-on-dark under this theme. */
.stTextInput input,
.stTextInput [data-baseweb="input"] input,
.stNumberInput input,
.stNumberInput [data-baseweb="input"] input {{
    background: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: #000000 !important;
    font-family: var(--font-mono) !important;
}}
.stTextInput input::placeholder,
.stNumberInput input::placeholder {{
    color: #888888 !important;
    -webkit-text-fill-color: #888888 !important;
}}

/* ---- Selectbox dropdown menu (rendered in a React portal, so the
       selectors below intentionally target body-level baseweb nodes,
       not .stSelectbox descendants). Without this the open menu
       renders with baseweb's default dark-on-dark scheme. */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
ul[role="listbox"] {{
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 0 !important;
    box-shadow: 3px 3px 0 var(--ink) !important;
    font-family: var(--font-body) !important;
}}
ul[role="listbox"] li,
[data-baseweb="menu"] li,
[role="option"] {{
    background: var(--surface) !important;
    color: var(--ink) !important;
    font-family: var(--font-body) !important;
}}
ul[role="listbox"] li:hover,
[data-baseweb="menu"] li:hover,
[role="option"]:hover,
[role="option"][aria-selected="true"] {{
    background: var(--paper-alt) !important;
    color: var(--shu) !important;
}}

/* Radio — keep native layout, recolour */
div[role="radiogroup"] label p {{
    color: var(--ink);
    font-size: 0.875rem;
    font-weight: 500;
}}
div[role="radiogroup"] label:hover p {{
    color: var(--shu);
}}

/* Checkbox text */
[data-testid="stCheckbox"] label p {{
    color: var(--ink);
    font-family: var(--font-body);
}}

/* ---- Slider — vermilion handle on ink track --------------------- */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
    background: var(--shu);
    border: 1px solid var(--ink);
    border-radius: 0;
    width: 14px !important;
    height: 14px !important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
    background: var(--ink);
}}

/* ---- Expanders — square ink card + offset shadow ---------------- */
[data-testid="stExpander"] {{
    background: var(--surface);
    border: 1px solid var(--ink);
    border-radius: 0;
    box-shadow: 3px 3px 0 var(--border-strong);
    overflow: hidden;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {{
    font-family: var(--font-display);
    font-weight: 600;
    color: var(--ink);
    font-size: 1rem;
}}

/* ---- Plotly chart container — the showpiece card --------------- */
[data-testid="stPlotlyChart"] {{
    background: var(--surface);
    border: 1px solid var(--ink);
    border-radius: 0;
    padding: 4px;
    box-shadow: 5px 5px 0 var(--ink);
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}}

/* ---- Dataframe ------------------------------------------------- */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--ink);
    border-radius: 0;
    overflow: hidden;
    box-shadow: 3px 3px 0 var(--border-strong);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
}}

/* ---- Alerts --------------------------------------------------- */
[data-testid="stAlert"] {{
    border-radius: 0;
    border: 1px solid var(--ink);
    border-left: 4px solid var(--shu);
    box-shadow: 3px 3px 0 var(--border-strong);
    font-family: var(--font-body);
}}

/* ---- Editorial horizontal rules ----------------------------- */
hr {{
    border: 0;
    border-top: 1px solid var(--ink);
    margin: 1.5rem 0;
}}

/* ---- Main content breathing room + max width -------------- */
.main .block-container {{
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1480px;
}}

/* ---- Scrollbar in the sumi palette ------------------------- */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: var(--paper-alt); }}
::-webkit-scrollbar-thumb {{
    background: var(--ink-muted);
    border: 2px solid var(--paper-alt);
}}
::-webkit-scrollbar-thumb:hover {{ background: var(--ink); }}

/* =====================================================================
   POLISH PASS — additional refinements applied via the frontend-design
   skill. All rules below are additive; they layer on top of the base
   京都ターミナル theme defined above.
   ===================================================================== */

/* ---- Masthead: editorial front-page title block ------------------ */
.kt-masthead {{
    position: relative;
    margin-top: -0.5rem;
    margin-bottom: 2rem;
    padding-top: 0.75rem;
    padding-bottom: 1.25rem;
    border-top: 1px solid var(--ink);
    overflow: hidden;
    isolation: isolate;
}}
/* Giant ghost kanji watermark — sits behind everything, barely there */
.kt-masthead::before {{
    content: "株";
    position: absolute;
    right: -0.15em;
    top: 50%;
    transform: translateY(-52%);
    font-family: var(--font-display);
    font-size: clamp(10rem, 22vw, 18rem);
    font-weight: 700;
    line-height: 0.85;
    color: var(--shu);
    opacity: 0.055;
    letter-spacing: -0.04em;
    pointer-events: none;
    z-index: 0;
}}
/* Double rule at the bottom of the masthead — 1px ink over 4px shu */
.kt-masthead::after {{
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 6px;
    background:
        linear-gradient(to bottom,
            var(--ink) 0,
            var(--ink) 1px,
            transparent 1px,
            transparent 3px,
            var(--shu) 3px,
            var(--shu) 6px);
}}
.kt-masthead-kicker {{
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 0.85rem;
    align-items: baseline;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.7rem;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--ink-muted);
    padding-bottom: 0.75rem;
    margin-bottom: 0.85rem;
    border-bottom: 1px dashed var(--border-strong);
}}
.kt-kicker-issue {{
    color: var(--shu);
    font-weight: 700;
    letter-spacing: 0.18em;
}}
.kt-kicker-issue sup {{
    font-size: 0.65em;
    vertical-align: super;
    margin-left: 1px;
}}
.kt-kicker-dot {{
    color: var(--border-strong);
    letter-spacing: 0;
}}
.kt-kicker-label {{
    color: var(--ink);
    font-weight: 600;
}}
.kt-kicker-date {{
    margin-left: auto;
    color: var(--ink-soft);
}}
.kt-masthead h1.kt-masthead-title {{
    position: relative;
    z-index: 1;
    margin: 0;
    padding: 0;
    border: none !important;
    display: block;
    font-family: var(--font-display) !important;
    font-size: clamp(2.75rem, 5.6vw, 4.75rem) !important;
    font-weight: 800 !important;
    line-height: 1.02;
    letter-spacing: -0.035em;
    color: var(--ink);
    text-shadow: 0.5px 0 0 currentColor;  /* optical boost */
}}
/* Killswitch — the base rule adds an underline + inline-block to the
   first h1 on the page. Our masthead wraps h1 so we need to unset both. */
.stApp .kt-masthead h1.kt-masthead-title {{
    display: block !important;
    border-bottom: none !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}}
.kt-masthead-title .kt-title-accent {{
    color: var(--shu);
}}
.kt-masthead-byline {{
    position: relative;
    z-index: 1;
    margin-top: 0.85rem;
    font-family: var(--font-display);
    font-style: italic;
    color: var(--ink-muted);
    font-size: 1rem;
    line-height: 1.6;
    max-width: 62ch;
}}
.kt-masthead-byline-en {{
    display: block;
    font-family: var(--font-mono);
    font-style: normal;
    font-size: 0.7rem;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--shu);
    margin-top: 0.35rem;
    font-weight: 600;
}}
.kt-masthead-meta {{
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink-muted);
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: center;
}}
.kt-masthead-issue {{
    color: var(--shu);
    font-weight: 600;
    letter-spacing: 0.15em;
}}
.kt-masthead-issue sup {{
    font-size: 0.65em;
    vertical-align: super;
    margin-left: 1px;
}}
.kt-masthead-sep {{
    color: var(--border-strong);
    font-family: var(--font-display);
    letter-spacing: 0;
}}
.kt-subtitle {{
    font-family: var(--font-display);
    font-style: italic;
    color: var(--ink-muted);
    font-size: 0.95rem;
    line-height: 1.7;
    max-width: 62ch;
    padding-left: 3rem;
    margin-top: 0.125rem;
    margin-bottom: 1.5rem;
}}
.kt-subtitle-line {{
    display: block;
    font-style: normal;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--ink-soft);
    letter-spacing: 0.05em;
    margin-top: 0.35rem;
    padding-top: 0.35rem;
    border-top: 1px dashed var(--border-strong);
}}

/* ---- Section labels: tiny editorial headings for content groups ---- */
.kt-section-label {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 1.75rem;
    margin-bottom: 0.75rem;
    font-family: var(--font-mono);
    font-size: 0.6875rem;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    color: var(--shu);
    font-weight: 600;
}}
.kt-section-label::before {{
    content: "";
    display: inline-block;
    width: 18px;
    height: 2px;
    background: var(--shu);
    flex-shrink: 0;
}}
.kt-section-label::after {{
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
}}
.kt-section-label .kt-section-num {{
    font-family: var(--font-mono);
    color: var(--ink-muted);
    font-weight: 500;
    font-size: 0.625rem;
    padding: 1px 6px;
    border: 1px solid var(--border-strong);
    letter-spacing: 0.1em;
}}

/* ---- Radio buttons: horizontal pill-tab look ---------------------- */
/* Applies to every st.radio(horizontal=True). Individual radios
   become bordered rectangles; the selected one fills with shu ink. */
div[role="radiogroup"] {{
    display: flex !important;
    flex-wrap: wrap;
    gap: 6px !important;
    padding: 4px 0 !important;
}}
div[role="radiogroup"] > label {{
    margin: 0 !important;
    padding: 6px 14px !important;
    border: 1px solid var(--ink) !important;
    background: var(--surface) !important;
    cursor: pointer;
    transition: background 140ms ease-out, color 140ms ease-out,
                box-shadow 140ms ease-out, transform 140ms ease-out;
    box-shadow: 2px 2px 0 var(--ink);
}}
div[role="radiogroup"] > label:hover {{
    background: var(--paper-alt) !important;
    transform: translate(-1px, -1px);
    box-shadow: 3px 3px 0 var(--ink);
}}
/* Hide the native round radio dot inside the label */
div[role="radiogroup"] > label > div:first-child {{
    display: none !important;
}}
/* Text inside the pill */
div[role="radiogroup"] > label p {{
    font-family: var(--font-body) !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    margin: 0 !important;
    color: var(--ink) !important;
}}
/* Selected pill: shu fill, paper text, inset shadow */
div[role="radiogroup"] > label:has(input:checked) {{
    background: var(--shu) !important;
    box-shadow: 1px 1px 0 var(--ink), inset 0 0 0 1px var(--shu-deep);
    transform: translate(1px, 1px);
}}
div[role="radiogroup"] > label:has(input:checked) p {{
    color: var(--paper) !important;
}}
div[role="radiogroup"] > label:has(input:checked):hover p {{
    color: var(--paper) !important;
}}

/* ---- Custom checkbox: square ink box, shu check mark ------------- */
/* Baseweb renders the native checkbox inside a styled <span>; we
   hide the default visual and paint our own with a border + ::after
   pseudo-element for the checkmark. */
[data-testid="stCheckbox"] label {{
    display: flex !important;
    align-items: center;
    gap: 0.55rem !important;
    cursor: pointer;
}}
[data-testid="stCheckbox"] label > span:first-child,
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > span:first-child {{
    width: 16px !important;
    height: 16px !important;
    min-width: 16px !important;
    background: var(--surface) !important;
    border: 1.5px solid var(--ink) !important;
    border-radius: 0 !important;
    box-shadow: 1px 1px 0 var(--ink);
    position: relative;
    transition: background 140ms ease-out;
}}
/* Hide the inner baseweb check icon; we draw our own via ::after */
[data-testid="stCheckbox"] label > span:first-child > * {{
    opacity: 0 !important;
}}
[data-testid="stCheckbox"] label:has(input:checked) > span:first-child {{
    background: var(--shu) !important;
    border-color: var(--ink) !important;
}}
[data-testid="stCheckbox"] label:has(input:checked) > span:first-child::after {{
    content: "";
    position: absolute;
    left: 3px;
    top: -1px;
    width: 5px;
    height: 10px;
    border: solid var(--paper);
    border-width: 0 2px 2px 0;
    transform: rotate(42deg);
}}

/* ---- Dataframe header polish ---------------------------------- */
[data-testid="stDataFrame"] thead tr th,
[data-testid="stDataFrame"] [role="columnheader"] {{
    background: var(--paper-alt) !important;
    color: var(--ink) !important;
    font-family: var(--font-body) !important;
    font-size: 0.6875rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-bottom: 2px solid var(--shu) !important;
}}
[data-testid="stDataFrame"] tbody tr td,
[data-testid="stDataFrame"] [role="gridcell"] {{
    font-family: var(--font-mono) !important;
    font-size: 0.8125rem !important;
}}

/* ---- Alerts: stronger shu bar + stamp shadow ------------------ */
[data-testid="stAlert"] {{
    padding: 14px 18px !important;
    border-left-width: 6px !important;
}}
[data-testid="stAlert"] p {{
    font-family: var(--font-body) !important;
    color: var(--ink) !important;
    font-size: 0.875rem;
    line-height: 1.65;
}}

/* ---- Sidebar headers: numbered prefix via attr ---------------- */
/* Streamlit renders st.sidebar.header("X") as an h2. We can't easily
   inject a counter number without markdown, but we can add a small
   shu square as a ::before decoration so each section visually
   separates from the ones above. */
[data-testid="stSidebar"] h1::before,
[data-testid="stSidebar"] h2::before,
[data-testid="stSidebar"] h3::before {{
    content: "";
    display: inline-block;
    width: 8px;
    height: 8px;
    background: var(--shu);
    margin-right: 0.55rem;
    vertical-align: middle;
    transform: translateY(-1px) rotate(45deg);
}}

/* ---- Sidebar divider line between controls -------------------- */
[data-testid="stSidebar"] hr {{
    border-top: 1px solid var(--border);
    margin: 1rem 0;
}}

/* ---- Colophon (editorial footer) ------------------------------ */
.kt-colophon {{
    margin-top: 3.5rem;
    padding-top: 1.25rem;
    padding-bottom: 0.5rem;
    border-top: 1px solid var(--ink);
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1.5rem;
    position: relative;
}}
.kt-colophon::before {{
    content: "";
    position: absolute;
    left: 0;
    top: -1px;
    width: 140px;
    height: 3px;
    background: var(--shu);
}}
.kt-colophon-block {{
    font-family: var(--font-mono);
    font-size: 0.6875rem;
    letter-spacing: 0.1em;
    line-height: 1.7;
    color: var(--ink-muted);
    text-transform: uppercase;
}}
.kt-colophon-block strong {{
    display: block;
    color: var(--shu);
    font-weight: 600;
    margin-bottom: 0.25rem;
    letter-spacing: 0.18em;
}}
.kt-colophon-mark {{
    font-family: var(--font-display);
    font-style: italic;
    font-size: 0.95rem;
    color: var(--ink-soft);
    letter-spacing: 0.02em;
    text-transform: none;
}}

/* ---- Decorative chart-section divider: editorial rule + seal --- */
.kt-chart-divider {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.75rem;
    margin-bottom: 0.25rem;
}}
.kt-chart-divider .kt-chart-divider-line {{
    flex: 1;
    height: 1px;
    background: var(--ink);
}}
.kt-chart-divider .kt-chart-divider-seal {{
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 0.8125rem;
    color: var(--paper);
    background: var(--shu);
    padding: 2px 10px;
    letter-spacing: 0.2em;
    box-shadow: 2px 2px 0 var(--ink);
    transform: rotate(-1deg);
}}
.kt-chart-divider .kt-chart-divider-kanji {{
    font-family: var(--font-display);
    font-size: 0.75rem;
    color: var(--ink-muted);
    letter-spacing: 0.3em;
    text-transform: uppercase;
}}

/* ---- Code and inline code block styling (for price displays etc.) --- */
code {{
    background-color: var(--paper-alt) !important;
    color: var(--shu) !important;
    padding: 0.15rem 0.35rem !important;
    border-radius: 3px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.875em !important;
    border: 1px solid var(--border-strong) !important;
}}

/* Hide Streamlit's sidebar collapse button and expander button to keep it always open */
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"],
button[aria-label="Close sidebar"],
[data-testid="collapsedControl"] {{
    display: none !important;
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
    show_per: bool = False,
    per_series: pd.Series | None = None,
    per_source: str = "",
    interval: str = "1d",
) -> go.Figure:
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
    return fig


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

    # ----- Sidebar: ticker selection ---------------------------------------
    st.sidebar.header("銘柄選択")
    ticker = _select_ticker(name_lookup)
    if not ticker:
        st.info("サイドバーで銘柄を選択してください。")
        return

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
    fig = build_figure(
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
            f"-{int(show_volume)}{int(show_macd)}{int(show_per_hist)}"
            f"-{int(show_sma)}{int(show_sma100)}{int(show_sma200)}"
        ),
        ticker_label=f"{ticker} {label}",
        currency_symbol=cur,
        show_volume=show_volume,
        show_macd=show_macd,
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
                annotation_text=f"📅 {da_label}",
                annotation_position="top right",
                annotation_font_color="#2E6B47",
                annotation_font_size=10,
            )

    # ---- Render chart via custom component (zero-rerun draggable line) ----
    # Track processed alert IDs to prevent duplicate registration
    _processed_ids = st.session_state.setdefault("_processed_alert_ids", set())
    _chart_key = f"chart_line_{ticker}"

    chart_result = price_line_chart(
        fig,
        current_price=curr_p,
        height=700,
        key=_chart_key,
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
                date_alerts.append({
                    "date": str(alert_date_input),
                    "label": alert_label_input.strip() or str(alert_date_input),
                })
                save_config(noti_config)
                st.toast(f"📅 日付アラート ({alert_date_input}) を追加しました。")
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
