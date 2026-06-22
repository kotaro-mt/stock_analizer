"""Fix: remove the misplaced watchlist comment block that got indented into main(),
and fix the _fetch_last_prices definition which lost its top-level indentation.
"""

with open("chart_app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Remove the indented comment block that snuck into main()
bad = (
    "\n\n    # ---------------------------------------------------------------------------\n"
    "    # Watchlist panel (TradingView-style right-side panel)\n"
    "    # ---------------------------------------------------------------------------\n"
    "    @st.cache_data(ttl=300, show_spinner=False)\n"
)
good = (
    "\n\n\n"
    "# ---------------------------------------------------------------------------\n"
    "# Watchlist panel (TradingView-style right-side panel)\n"
    "# ---------------------------------------------------------------------------\n"
    "@st.cache_data(ttl=300, show_spinner=False)\n"
)

if bad in content:
    content = content.replace(bad, good)
    print("Fixed misplaced comment+decorator")
else:
    print("Pattern not found – check manually")
    print(repr(content[content.find("# Watchlist panel")-200:content.find("# Watchlist panel")+200]))

with open("chart_app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Done")
