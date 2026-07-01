import re

with open('chart_app.py', 'r', encoding='utf-8') as f:
    current = f.read()

with open('backup_pre_design_update/chart_app.py', 'r', encoding='utf-8') as f:
    backup = f.read()

# Extract THEME and THEME_CSS from current
theme_match = re.search(r'(THEME = \{.*?\n\})', current, re.DOTALL)
theme_css_match = re.search(r'(THEME_CSS = f\"\"\"\n<style>.*?</style>\n\"\"\")', current, re.DOTALL)

if theme_match and theme_css_match:
    backup = re.sub(r'THEME = \{.*?\n\}', theme_match.group(1), backup, flags=re.DOTALL)
    
    # We will also inject the custom CSS for selection tabs inside the CSS match before closing </style>
    css_content = theme_css_match.group(1)
    custom_tabs_css = """
/* ---- Custom Styling for st.radio (Selection Tabs) ---- */
div[role="radiogroup"] {
    display: flex;
    gap: 4px;
    background: rgba(15,23,42,0.6);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid var(--border);
    flex-wrap: wrap;
}
div[role="radiogroup"] > label {
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
}
div[role="radiogroup"] > label:hover {
    background: rgba(56,189,248,0.1);
}
div[role="radiogroup"] > label > div:first-child {
    display: none !important; /* Hide native radio circles */
}
div[role="radiogroup"] > label p {
    color: var(--ink-muted);
    font-weight: 600;
    font-size: 0.8rem;
    margin: 0;
    transition: color 0.2s;
}
div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(56,189,248,0.2) !important;
    border-radius: 6px;
}
div[role="radiogroup"] > label:has(input:checked) p {
    color: var(--navy) !important;
    text-shadow: 0 0 10px rgba(56,189,248,0.3);
}

/* Also style selectboxes for dark theme properly */
[data-baseweb="select"] {
    cursor: pointer;
}
"""
    # Insert custom CSS right before </style>
    css_content = css_content.replace('</style>', custom_tabs_css + '\n</style>')
    
    backup = re.sub(r'THEME_CSS = f\"\"\"\n<style>.*?</style>\n\"\"\"', css_content, backup, flags=re.DOTALL)

with open('chart_app.py', 'w', encoding='utf-8') as f:
    f.write(backup)

print("Strict merge completed! Watchlist reverted to sidebar, theme preserved and enhanced.")
