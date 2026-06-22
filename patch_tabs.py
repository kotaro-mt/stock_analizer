import re

with open("chart_app.py", "r", encoding="utf-8") as f:
    content = f.read()

# CSS to fix Streamlit Tabs styling and Selectbox highlighting
css_addition = """
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
"""

if "/* Fix Tabs Visibility */" not in content:
    content = content.replace("</style>", css_addition + "\n</style>")

with open("chart_app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Patched Tabs styling in chart_app.py")
