"""
Replace the st.tabs layout with:
1. A full-width main chart area (no tabs)
2. The watchlist rendered as a fixed-position right panel via st.markdown HTML/JS
3. Ticker picked via URL query param (?ticker=CODE)
"""

with open("chart_app.py", "r", encoding="utf-8") as f:
    content = f.read()

# ---- Step 1: replace tab layout block ----
old_layout = (
    '    # ----- Main area: tabs (chart | watchlist) ---------------------------\n'
    '    tab_chart, tab_wl = st.tabs(["\\U0001f4c8 チャート分析", "\\U0001f4cb ウォッチリスト"])\n'
    '\n'
    '    with tab_wl:\n'
    '        _render_watchlist_panel(name_lookup)\n'
    '\n'
    '    with tab_chart:\n'
    '        ticker = st.session_state.get("selected_ticker", "7203.T")\n'
    '        if not ticker:\n'
    '            st.info("ウォッチリストから銘柄を選択してください。")\n'
    '            return\n'
)
new_layout = (
    '    # ----- Fixed right watchlist panel (HTML/JS, no tabs) ---------------\n'
    '    _render_watchlist_panel(name_lookup)\n'
    '\n'
    '    # Read ticker from URL query param (set by watchlist JS click)\n'
    '    _qp_ticker = st.query_params.get("ticker", "")\n'
    '    if _qp_ticker and _qp_ticker != st.session_state.get("selected_ticker", ""):\n'
    '        st.session_state["selected_ticker"] = _qp_ticker\n'
    '    ticker = st.session_state.get("selected_ticker", "7203.T")\n'
    '    if not ticker:\n'
    '        st.info("右側のウォッチリストから銘柄を選択してください。")\n'
    '        return\n'
    '\n'
)

if old_layout in content:
    content = content.replace(old_layout, new_layout, 1)
    print("Step 1 OK: layout replaced")
else:
    print("ERROR: layout pattern not found")

with open("chart_app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
