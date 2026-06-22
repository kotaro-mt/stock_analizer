"""
Move all lines from line 1621 up to (but not including) the blank line
before _fetch_last_prices (the watchlist helper) into the col_chart
`with` block by adding 4 spaces of indentation.

Line ranges (1-indexed):
  - col_chart `with` block starts at line 1612 (`    with col_chart:`)
  - The block currently ends at line 1619 (the `return` after ticker check)
  - The chart content (sidebar calls + data loading + fig rendering) spans
    lines 1621..end_of_main

Strategy: replace the closing of `with col_chart:` so it encompasses
lines 1621..end_of_main.  The end of main() is the blank line right before
the `# ---------------------` separator for _fetch_last_prices.
"""

with open("chart_app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the line that starts the watchlist helper (def _fetch_last_prices)
watchlist_helper_start = None
for i, line in enumerate(lines):
    if line.strip().startswith("@st.cache_data(ttl=300") and "_fetch_last_prices" not in line:
        continue
    if "def _fetch_last_prices(" in line:
        watchlist_helper_start = i
        break

print(f"watchlist helper starts at line {watchlist_helper_start + 1}")

# The with col_chart block (0-indexed) ends at what we inserted.
# We need to find where `    with col_chart:` is, then indent everything
# from the line AFTER the ticker check return, up to main()'s end.

# Find `with col_chart:` line
col_chart_with = None
for i, line in enumerate(lines):
    if "    with col_chart:" in line:
        col_chart_with = i
        break

print(f"col_chart with-block at line {col_chart_with + 1}")

# Find the `return` inside col_chart (after `if not ticker:`)
col_chart_return = None
for i in range(col_chart_with, col_chart_with + 20):
    if "            return" in lines[i] or "        return" in lines[i]:
        if "ウォッチリスト" in lines[i-1] or "ウォッチリスト" in lines[i-2] or "selected_ticker" in lines[i-1]:
            col_chart_return = i
            break

if col_chart_return is None:
    for i in range(col_chart_with, col_chart_with + 20):
        if "return" in lines[i] and "ticker" in lines[i-1]:
            col_chart_return = i
            break

print(f"ticker-check return at line {col_chart_return + 1}")

# The main() function ends just before _fetch_last_prices
# (there are a couple of blank/comment lines between them)
# Walk backward from watchlist_helper_start to find the last non-blank
# line of main()
main_end = watchlist_helper_start - 1
while main_end > 0 and lines[main_end].strip() == "":
    main_end -= 1
# main_end now points to the last non-blank line of main(), which should
# be the colophon comment. We indent everything from col_chart_return+1
# to main_end (inclusive) by 4 spaces.

print(f"main() content to indent: lines {col_chart_return + 2}..{main_end + 1}")

# Add 4 spaces to each line in that range
for i in range(col_chart_return + 1, main_end + 1):
    if lines[i].strip():  # skip truly blank lines
        lines[i] = "    " + lines[i]

with open("chart_app.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Done. Verify with py_compile.")
