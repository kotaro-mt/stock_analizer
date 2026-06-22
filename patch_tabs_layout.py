"""
Convert the 2-column layout back to a st.tabs() layout.

Current structure (lines 1605-1616):
    # 2-column layout
    col_chart, col_watchlist = st.columns([5, 2], gap="medium")
    with col_watchlist:
        _render_watchlist_panel(name_lookup)
    with col_chart:
        ticker = st.session_state.get("selected_ticker", "7203.T")
        if not ticker:
            ...
            return
        [rest of chart code, indented by 8 spaces]

Target structure:
    tab_chart, tab_wl = st.tabs(["📈 チャート分析", "📋 ウォッチリスト"])
    with tab_wl:
        _render_watchlist_panel(name_lookup)
    with tab_chart:
        ticker = st.session_state.get("selected_ticker", "7203.T")
        if not ticker:
            ...
            return
        [rest of chart code, still indented by 8 spaces - unchanged]

Steps:
1. Replace the 2-column header + col_watchlist block with st.tabs
2. Replace `with col_chart:` with `with tab_chart:`
"""

with open("chart_app.py", "r", encoding="utf-8") as f:
    content = f.read()

# ---- Step 1: replace 2-column setup + col_watchlist block ----
old_header = (
    "    # ----- Main area: 2-column layout (chart | watchlist) ----------------\n"
    "    # Left (chart): 5 parts  Right (watchlist): 2 parts\n"
    "    col_chart, col_watchlist = st.columns([5, 2], gap=\"medium\")\n"
    "\n"
    "    with col_watchlist:\n"
    "        _render_watchlist_panel(name_lookup)\n"
    "\n"
    "    with col_chart:\n"
)
new_header = (
    "    # ----- Main area: tabs (chart | watchlist) ---------------------------\n"
    "    tab_chart, tab_wl = st.tabs([\"\\U0001f4c8 チャート分析\", \"\\U0001f4cb ウォッチリスト\"])\n"
    "\n"
    "    with tab_wl:\n"
    "        _render_watchlist_panel(name_lookup)\n"
    "\n"
    "    with tab_chart:\n"
)

if old_header in content:
    content = content.replace(old_header, new_header, 1)
    print("Replaced column layout with tabs")
else:
    print("ERROR: old_header pattern not found")

with open("chart_app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
