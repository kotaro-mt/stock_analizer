"""Cached OHLCV loader for stock_future chart analyzer.

The legacy ML pipeline already downloaded ~10 years of daily OHLCV for
every ticker in ``universe.py`` into ``cache/*.parquet``. This module
reads that cache so the new chart-analysis project does not re-download.

If a ticker's cache is missing, it falls back to yfinance and populates
the cache so the next lookup is fast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"

# Supported bar intervals. The daily cache predates this module and lives
# in ``cache/*.parquet`` at the top level; weekly gets a per-interval
# subdirectory so it doesn't collide with the legacy layout.
INTERVALS: dict[str, dict] = {
    "1d": {
        "period": "10y",
        "cache_subdir": None,         # top-level (legacy layout)
        "ttl_hours": 24.0,
    },
    "1wk": {
        "period": "10y",
        "cache_subdir": "1wk",
        "ttl_hours": 24.0,
    },
}


def _ticker_to_filename(ticker: str) -> str:
    # ^N225 -> _N225.parquet; 7203.T -> 7203_T.parquet
    return ticker.replace(".", "_").replace("^", "_") + ".parquet"


def _cache_path(ticker: str, interval: str) -> Path:
    """Resolve the on-disk cache path for a ticker + interval combination."""
    cfg = INTERVALS.get(interval)
    if cfg is None:
        raise ValueError(f"unknown interval: {interval!r}")
    sub = cfg["cache_subdir"]
    base = CACHE_DIR if sub is None else CACHE_DIR / sub
    return base / _ticker_to_filename(ticker)


def normalise_ticker(raw: str) -> str:
    """Accept 4-digit codes without .T suffix and add it automatically."""
    t = raw.strip().upper()
    if not t:
        return t
    # 4-digit Japanese stock -> append .T
    if t.isdigit() and len(t) == 4:
        return t + ".T"
    return t


# Backwards-compat alias (older callers used the underscore-prefixed name)
_normalise_ticker = normalise_ticker


def _download_from_yfinance(
    ticker: str, interval: str = "1d", period: str | None = None,
) -> pd.DataFrame | None:
    """Fetch OHLCV from yfinance at the requested interval.

    ``period`` defaults to ``INTERVALS[interval]["period"]`` (full history).
    Pass a shorter period (e.g. ``"1mo"``) for incremental refreshes.
    Returns ``None`` on any failure.
    """
    cfg = INTERVALS.get(interval)
    if cfg is None:
        raise ValueError(f"unknown interval: {interval!r}")
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.download(
            ticker,
            period=period or cfg["period"],
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"[data] yfinance download error for {ticker} @ {interval}: {e}")
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return None
    df = df[needed]
    # Daily/weekly bars from yfinance are usually tz-naive already; strip
    # if the provider ever attaches one so parquet round-trips cleanly.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    auto_download: bool = True,
    interval: str = "1d",
) -> pd.DataFrame | None:
    """Load OHLCV for a single ticker at the requested bar interval.

    If the cache is missing and ``auto_download`` is True, fetch from
    yfinance and populate the cache. Daily caches use the legacy flat
    ``cache/*.parquet`` layout; weekly gets a per-interval subdirectory.

    Returns a DataFrame with columns ``[Open, High, Low, Close, Volume]``
    indexed by datetime, or ``None`` if unavailable.
    """
    ticker = _normalise_ticker(ticker)
    cfg = INTERVALS.get(interval)
    if cfg is None:
        raise ValueError(f"unknown interval: {interval!r}")
    path = _cache_path(ticker, interval)
    ttl_hours = float(cfg.get("ttl_hours", 24.0))

    if not path.exists():
        # No cache at all — fetch full history.
        if not auto_download:
            return None
        df_new = _download_from_yfinance(ticker, interval=interval)
        if df_new is None or df_new.empty:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df_new.to_parquet(path)
        except Exception as e:
            print(f"[data] failed to write {interval} cache for {ticker}: {e}")

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[data] cache read error for {ticker} ({path.name}): {e}. Rebuilding cache...")
        if not auto_download:
            return None
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        df = _download_from_yfinance(ticker, interval=interval)
        if df is None or df.empty:
            return None
        try:
            df.to_parquet(path)
        except Exception as ex:
            print(f"[data] failed to write recovered {interval} cache for {ticker}: {ex}")
    # Some parquets have a MultiIndex column level from yfinance downloads;
    # flatten it defensively (also needed before the TTL check below so the
    # ``Close`` column is addressable consistently).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ----- Incremental refresh when the cache is stale --------------------
    # The cache used to be "write once, read forever", so technical
    # indicators (MA/RSI/MACD) were computed on whatever bars happened to
    # be in the parquet — typically days or weeks behind. Honor the
    # per-interval ``ttl_hours`` by downloading a short recent window
    # (``1mo``) and merging on index when the last cached bar is older
    # than the TTL. Full re-download is avoided to keep this fast.
    if auto_download and not df.empty:
        last_ts = df.index[-1]
        # df.index is tz-naive (see _download_from_yfinance), so compare
        # against a tz-naive "now".
        age_hours = (pd.Timestamp.now() - last_ts).total_seconds() / 3600.0
        if age_hours > ttl_hours:
            df_new = _download_from_yfinance(
                ticker, interval=interval, period="1mo",
            )
            if df_new is not None and not df_new.empty:
                merged = pd.concat([df, df_new])
                merged = merged[~merged.index.duplicated(keep="last")]
                merged = merged.sort_index()
                try:
                    merged.to_parquet(path)
                except Exception as e:
                    print(f"[data] failed to refresh {interval} cache for {ticker}: {e}")
                df = merged
    # Drop rows where any OHLCV is NaN (holidays with partial data, etc.)
    df = df.dropna(how="any", subset=["Open", "High", "Low", "Close", "Volume"])
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


def load_universe(
    tickers: Iterable[str] | None = None,
    min_bars: int = 100,
    *,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Load all (or selected) tickers as dict[ticker -> DataFrame].

    Tickers with fewer than ``min_bars`` rows at the requested interval
    are skipped.
    """
    if tickers is None:
        from universe import UNIVERSE
        tickers = [t for t, _n, _s in UNIVERSE]
    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = load_ohlcv(t, interval=interval)
        if df is not None and len(df) >= min_bars:
            result[t] = df
    return result


def available_tickers(interval: str = "1d") -> list[str]:
    """List all tickers currently present in the cache at an interval."""
    cfg = INTERVALS.get(interval)
    if cfg is None:
        raise ValueError(f"unknown interval: {interval!r}")
    sub = cfg["cache_subdir"]
    base = CACHE_DIR if sub is None else CACHE_DIR / sub
    if not base.exists():
        return []
    return sorted(
        p.stem.replace("_", ".") for p in base.glob("*.parquet")
    )


# ---------------------------------------------------------------------------
# Ticker name lookup (cached, falls back to yfinance.info)
# ---------------------------------------------------------------------------
_NAME_CACHE: dict[str, str] = {}


def get_ticker_name(ticker: str) -> str | None:
    """Return a human-readable name for a ticker, or None if unknown.

    Uses an in-process cache to avoid repeated yfinance.info hits, which
    are slow and rate-limited. Returns None on any failure so callers
    can fall back gracefully.
    """
    ticker = _normalise_ticker(ticker)
    if not ticker:
        return None
    if ticker in _NAME_CACHE:
        return _NAME_CACHE[ticker] or None
    name = ""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        name = info.get("longName") or info.get("shortName") or ""
    except Exception as e:
        print(f"[data] yfinance name lookup failed for {ticker}: {e}")
    _NAME_CACHE[ticker] = name
    return name or None
