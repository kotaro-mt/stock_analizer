import re

with open("chart_app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Replace THEME dict
theme_pattern = re.compile(r'THEME = \{.*?\}', re.DOTALL)
new_theme = '''THEME = {
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
}'''
content = theme_pattern.sub(new_theme, content, count=1)

# 2. Replace THEME_CSS string
css_pattern = re.compile(r'THEME_CSS = f"""\n<style>.*?</style>\n"""', re.DOTALL)
new_css = '''THEME_CSS = f"""
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
</style>
"""'''
content = css_pattern.sub(new_css, content, count=1)

# Ensure the plot font config is changed
content = content.replace(
    '''PLOTLY_FONT_FAMILY = (
    "'IBM Plex Sans JP', 'Hiragino Sans', 'Noto Sans CJK JP', sans-serif"
)''',
    '''PLOTLY_FONT_FAMILY = (
    "'Inter', sans-serif"
)'''
)

with open("chart_app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched chart_app.py")
