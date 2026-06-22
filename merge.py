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
    backup = re.sub(r'THEME_CSS = f\"\"\"\n<style>.*?</style>\n\"\"\"', theme_css_match.group(1), backup, flags=re.DOTALL)

# Extract _render_watchlist_panel from current
panel_match = re.search(r'(def _render_watchlist_panel.*?)(?=if __name__ ==)', current, re.DOTALL)

if panel_match:
    panel_code = panel_match.group(1)
    backup = backup.replace('def main() -> None:', panel_code + '\n\ndef main() -> None:')

# Extract the ticker selection logic from current
ticker_logic_match = re.search(r'(# ----- Fixed right watchlist panel.*?)(?=# ----- Sidebar: bar interval)', current, re.DOTALL)

if ticker_logic_match:
    ticker_logic = ticker_logic_match.group(1)
    old_ticker_logic = """    # ----- Sidebar: ticker selection ---------------------------------------
    st.sidebar.header("銘柄選択")
    ticker = _select_ticker(name_lookup)
    if not ticker:
        st.info("サイドバーで銘柄を選択してください。")
        return"""
    backup = backup.replace(old_ticker_logic, ticker_logic.strip() + '\n')

with open('chart_app_reverted.py', 'w', encoding='utf-8') as f:
    f.write(backup)

print("Merged successfully!")
