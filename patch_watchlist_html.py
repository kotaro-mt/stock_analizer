"""Replace _render_watchlist_panel (line 2490..2747) with a pure HTML/JS
fixed-position resizable right panel implementation."""

with open("chart_app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find function start and end
start_line = None  # 0-indexed line of "def _render_watchlist_panel"
for i, l in enumerate(lines):
    if "def _render_watchlist_panel(" in l:
        start_line = i
        break

# Find end: next top-level def/class or "if __name__"
end_line = None
for i in range(start_line + 1, len(lines)):
    if lines[i].startswith("if __name__") or (lines[i].startswith("def ") and not lines[i].startswith("    ")):
        end_line = i
        break

print(f"Function occupies lines {start_line+1}..{end_line} (0-indexed {start_line}..{end_line-1})")

new_func = r'''def _render_watchlist_panel(name_lookup: dict[str, str]) -> None:
    """Render a fixed-position, resizable TradingView-style watchlist panel.

    The panel is injected as raw HTML via st.markdown so it sits outside
    Streamlit's normal column flow.  Resizing is handled by a JS drag-handle
    on the left edge of the panel.  Clicking a row calls pickTicker() which
    sets ?ticker=CODE in the URL; Python reads it on the next rerun.
    """
    import html as _html

    # Fetch prices for ALL tickers up front
    all_codes = (
        tuple(code for code, _, _ in UNIVERSE)
        + tuple(code for code, _ in INDICES)
    )
    prices = _fetch_last_prices(all_codes)
    favs   = load_favorites()
    selected = st.session_state.get("selected_ticker", "7203.T")

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
    padding: 10px 12px 6px;
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
.main .block-container {{ margin-right: 290px !important; max-width: calc(100% - 290px) !important; }}
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

  // ---- Ticker pick (navigate to ?ticker=CODE) ----
  window.pickTicker = function(code) {{
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

'''

# Replace lines start_line..end_line-1 with new function
lines[start_line:end_line] = [new_func]

with open("chart_app.py", "w", encoding="utf-8") as f:
    f.writelines(lines)
print(f"Replaced _render_watchlist_panel (was lines {start_line+1}..{end_line})")
print("Done")
