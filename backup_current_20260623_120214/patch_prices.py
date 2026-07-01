with open("chart_app.py", "r", encoding="utf-8") as f:
    content = f.read()

old = '''    import yfinance as yf
    results: dict[str, dict] = {}
    if not tickers:
        return results
    try:
        raw = yf.download(
            list(tickers), period="5d", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker"
        )
        for t in tickers:
            try:
                if len(tickers) == 1:
                    closes = raw["Close"]
                else:
                    closes = raw["Close"][t]
                closes = closes.dropna()
                if len(closes) >= 2:
                    last = float(closes.iloc[-1])
                    prev = float(closes.iloc[-2])
                    chg = last - prev
                    pct = chg / prev * 100 if prev else 0.0
                    results[t] = {"last": last, "chg": chg, "pct": pct}
                elif len(closes) == 1:
                    results[t] = {"last": float(closes.iloc[-1]), "chg": 0.0, "pct": 0.0}
            except Exception:
                pass
    except Exception:
        pass
    return results'''

new = '''    import yfinance as yf
    results: dict[str, dict] = {}
    if not tickers:
        return results

    # Batch 50 tickers at a time to avoid yfinance rate-limits
    BATCH = 50
    ticker_list = list(tickers)
    for start in range(0, len(ticker_list), BATCH):
        batch = ticker_list[start:start + BATCH]
        try:
            raw = yf.download(
                batch, period="7d", interval="1d",
                auto_adjust=True, progress=False,
            )
            if raw.empty:
                continue

            # Handle MultiIndex (multi-ticker) vs flat (single-ticker) columns
            if isinstance(raw.columns, pd.MultiIndex):
                # yfinance 0.2+: columns are ("Close", "7203.T"), etc.
                try:
                    close_df = raw["Close"]
                except KeyError:
                    continue
                for t in batch:
                    try:
                        if t in close_df.columns:
                            series = close_df[t].dropna()
                        else:
                            continue
                        if len(series) >= 2:
                            last = float(series.iloc[-1])
                            prev = float(series.iloc[-2])
                            chg = last - prev
                            pct = chg / prev * 100 if prev else 0.0
                            results[t] = {"last": last, "chg": chg, "pct": pct}
                        elif len(series) == 1:
                            results[t] = {"last": float(series.iloc[-1]), "chg": 0.0, "pct": 0.0}
                    except Exception:
                        pass
            else:
                # Single-ticker download: flat columns
                t = batch[0]
                try:
                    series = raw["Close"].dropna()
                    if len(series) >= 2:
                        last = float(series.iloc[-1])
                        prev = float(series.iloc[-2])
                        chg = last - prev
                        pct = chg / prev * 100 if prev else 0.0
                        results[t] = {"last": last, "chg": chg, "pct": pct}
                    elif len(series) == 1:
                        results[t] = {"last": float(series.iloc[-1]), "chg": 0.0, "pct": 0.0}
                except Exception:
                    pass
        except Exception:
            pass
    return results'''

if old in content:
    content = content.replace(old, new)
    print("Replaced successfully")
else:
    print("Pattern not found!")

with open("chart_app.py", "w", encoding="utf-8") as f:
    f.write(content)
