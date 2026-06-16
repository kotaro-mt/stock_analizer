"""J-Quants V2 API client for actual (non-consensus) JP fundamentals.

Used by stock_future's chart analyzer to pull real quarterly earnings
reports (決算短信) straight from the JPX data feed, so the historical
PER curve reflects the EPS values the market actually priced on, not
yfinance's analyst aggregates.

J-Quants V2 uses a simple ``x-api-key`` header (no JWT refresh flow).
The key is issued on the dashboard at https://jpx-jquants.com/.

Credentials are read from (in order of preference):
  1. ``st.secrets["jquants"]["api_key"]``  (Streamlit)
  2. ``JQUANTS_API_KEY`` environment variable
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache" / "jquants"

API_BASE = "https://api.jquants.com/v2"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
def _read_api_key() -> str | None:
    """Look up the J-Quants API key from Streamlit secrets or env vars."""
    try:
        import streamlit as st  # type: ignore
        if "jquants" in st.secrets:
            sec = st.secrets["jquants"]
            for k in ("api_key", "refresh_token"):
                # ``refresh_token`` is accepted for backward compat with
                # an earlier config layout; treat it as an api_key.
                v = sec.get(k) if hasattr(sec, "get") else (sec[k] if k in sec else None)
                if v:
                    return str(v)
    except Exception:
        # Streamlit not installed / no secrets file / not in runtime context
        pass
    return os.environ.get("JQUANTS_API_KEY") or os.environ.get(
        "JQUANTS_REFRESH_TOKEN"
    )


def is_configured() -> bool:
    return bool(_read_api_key())


def _auth_headers() -> dict[str, str] | None:
    key = _read_api_key()
    if not key:
        return None
    return {"x-api-key": key}


# ---------------------------------------------------------------------------
# /fins/summary — 決算短信サマリー
# ---------------------------------------------------------------------------
def _code_to_jquants(code: str) -> str | None:
    """Normalise a ticker to J-Quants' 5-digit code format.

    J-Quants V2 uses 5-digit codes (4-digit tickers get a trailing ``0``
    appended, matching the new JPX unified system). Non-JP tickers
    return ``None``.
    """
    raw = code.strip().upper().replace(".T", "")
    if not raw.isdigit():
        return None
    if len(raw) == 4:
        return raw + "0"
    if len(raw) == 5:
        return raw
    return None


def _summary_cache_path(jq_code: str) -> Path:
    return CACHE_DIR / f"summary_{jq_code}.parquet"


def get_fins_summary(
    code: str,
    *,
    max_age_hours: float = 12.0,
) -> pd.DataFrame | None:
    """Fetch ``/fins/summary`` for a ticker (cached to parquet on disk).

    The summary endpoint contains one row per disclosed 決算短信 with
    fields such as ``EPS`` (cumulative actual EPS for the period),
    ``FEPS`` (company forecast for full year), ``Sales``, ``NP``, etc.

    Returns a DataFrame or None on failure. Stale cache is used as a
    fallback when the network call errors.
    """
    jq_code = _code_to_jquants(code)
    if jq_code is None:
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _summary_cache_path(jq_code)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < max_age_hours * 3600:
            try:
                return pd.read_parquet(path)
            except Exception:
                pass  # cache corrupt — re-fetch

    headers = _auth_headers()
    if headers is None:
        return None

    try:
        r = requests.get(
            f"{API_BASE}/fins/summary",
            headers=headers,
            params={"code": jq_code},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        print(f"[jquants] /fins/summary fetch failed for {jq_code}: {e}")
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
        return None

    if not data:
        return None

    df = pd.DataFrame(data)
    try:
        df.to_parquet(path)
    except Exception as e:
        print(f"[jquants] failed to cache summary for {jq_code}: {e}")
    return df


# ---------------------------------------------------------------------------
# Historical TTM EPS derivation
# ---------------------------------------------------------------------------
_PERIOD_NUM = {"1Q": 1, "2Q": 2, "3Q": 3, "FY": 4}


def _clean_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce and filter /fins/summary rows down to clean 決算短信 records.

    Drops forecast-revision / dividend-revision filings, keeps only rows
    with parseable EPS and period code. When both consolidated and
    non-consolidated variants exist for the same period, prefers the
    consolidated version (which is what the market tracks).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    doc = df.get("DocType", pd.Series([""] * len(df))).astype(str)
    mask = doc.str.contains("FinancialStatements", na=False)
    mask &= ~doc.str.contains("Forecast", na=False)
    df = df[mask].copy()
    if df.empty:
        return df

    df["DiscDate"] = pd.to_datetime(df.get("DiscDate"), errors="coerce")
    df["CurFYSt"] = pd.to_datetime(df.get("CurFYSt"), errors="coerce")
    df["EPS"] = pd.to_numeric(df.get("EPS"), errors="coerce")

    df = df.dropna(subset=["DiscDate", "CurFYSt", "EPS", "CurPerType"])
    df = df[df["CurPerType"].isin(_PERIOD_NUM.keys())]
    if df.empty:
        return df

    df["period_num"] = df["CurPerType"].map(_PERIOD_NUM).astype(int)
    df["is_consolidated"] = doc.reindex(df.index).str.contains(
        "_Consolidated_", na=False
    )

    # Prefer consolidated when duplicates exist for a (FY, period) pair
    df = df.sort_values(["CurFYSt", "period_num", "is_consolidated"])
    df = df.drop_duplicates(
        subset=["CurFYSt", "period_num"], keep="last"
    )
    return df.sort_values(["CurFYSt", "period_num"])


def cleaned_summary(code: str) -> pd.DataFrame | None:
    """Return the cleaned ``/fins/summary`` rows for a ticker.

    A thin wrapper around ``get_fins_summary`` + ``_clean_summary``.
    The returned frame has one row per 決算短信, sorted by
    ``(CurFYSt, period_num)``, with at minimum these columns:
    ``DiscDate``, ``CurFYSt``, ``EPS``, ``CurPerType``, ``period_num``.

    Callers that need split-adjusted EPS should rescale the ``EPS``
    column *before* passing the frame to ``ttm_from_cleaned`` — the
    quarter-over-quarter differencing is unit-sensitive, so pre-split
    and post-split values must be reconciled first.
    """
    raw = get_fins_summary(code)
    df = _clean_summary(raw) if raw is not None else pd.DataFrame()
    return df if not df.empty else None


def ttm_from_cleaned(df: pd.DataFrame) -> pd.Series | None:
    """Derive a trailing-12-month EPS series from a cleaned summary frame.

    Japanese quarterly reports disclose EPS as *fiscal-year-to-date*
    (1Q = Q1 only, 2Q = Q1+Q2 cumulative, 3Q = Q1+Q2+Q3, FY = full year).
    Single-quarter EPS is derived by differencing consecutive disclosures
    within the same fiscal year.

    A single-quarter value is only considered **clean** when the preceding
    quarter in the same fiscal year is also present (so the difference
    isolates one quarter). First-record-of-FY is clean iff period == 1Q.
    This matters at the boundary of the J-Quants 2-year window: the
    earliest record may be a mid-year cumulative (e.g. 3Q) and must NOT
    be treated as a single-quarter value — otherwise the first TTM point
    would be inflated by roughly 3× its true size.

    TTM EPS at each disclosure date = rolling sum of the previous 4
    *consecutive clean* single-quarter values. Windows that include any
    non-clean placeholder are dropped.

    The returned Series is indexed by ``DiscDate`` — the day the market
    actually learned the figure, not the period end — so the caller can
    forward-fill onto price bars without look-ahead bias.
    """
    if df is None or df.empty:
        return None

    records: list[dict] = []
    for _, g in df.groupby("CurFYSt"):
        g = g.sort_values("period_num")
        prev_cum = 0.0
        prev_period = 0  # resets each fiscal year
        for _, row in g.iterrows():
            cum = float(row["EPS"])
            period = int(row["period_num"])
            # Clean iff the immediately preceding quarter of the same FY
            # is present (or this is 1Q, which needs no predecessor).
            is_clean = (period == prev_period + 1)
            q_eps = cum - prev_cum if is_clean else float("nan")
            records.append(
                {
                    "DiscDate": row["DiscDate"],
                    "q_eps": q_eps,
                    "is_clean": is_clean,
                }
            )
            prev_cum = cum
            prev_period = period

    if not records:
        return None

    q_df = pd.DataFrame(records).sort_values("DiscDate")
    q_df = q_df.drop_duplicates("DiscDate", keep="last").set_index("DiscDate")

    # Manual 4-wide rolling window: emit a TTM value only when all 4
    # entries in the window are clean (consecutive single quarters).
    ttm_points: dict[pd.Timestamp, float] = {}
    window_vals: list[float] = []
    window_clean: list[bool] = []
    for idx, row in q_df.iterrows():
        window_vals.append(float(row["q_eps"]) if row["is_clean"] else 0.0)
        window_clean.append(bool(row["is_clean"]))
        if len(window_vals) > 4:
            window_vals.pop(0)
            window_clean.pop(0)
        if len(window_vals) == 4 and all(window_clean):
            ttm_points[idx] = sum(window_vals)

    if not ttm_points:
        return None
    return pd.Series(ttm_points).sort_index()


def historical_ttm_eps(code: str) -> pd.Series | None:
    """Convenience wrapper — fetch, clean, and derive TTM EPS in one call.

    Does **no** split adjustment. Callers that need split-adjusted EPS
    should use ``cleaned_summary`` → rescale ``EPS`` → ``ttm_from_cleaned``.
    """
    df = cleaned_summary(code)
    return ttm_from_cleaned(df) if df is not None else None


# ---------------------------------------------------------------------------
# FY-only 会社予想 EPS (本決算 guidance)
# ---------------------------------------------------------------------------
def fy_guidance(code: str) -> pd.DataFrame | None:
    """Return a frame of 本決算 (FY) filings with actual + next-FY guidance.

    Each row represents a single annual earnings disclosure (CurPerType ==
    "FY") and carries:

    - ``DiscDate``       — the day the market learned the figures
    - ``CurFYSt``        — start of the reported fiscal year
    - ``EPS``            — realised full-year EPS (sum of 4 quarters)
    - ``NxFEPS``         — company's own guidance for the *next* fiscal
                            year, announced on the same day as the actual

    Returned sorted by DiscDate. Values are left in their originally-
    disclosed units — callers must split-adjust before comparing with
    split-adjusted close prices. Returns ``None`` if there are no FY
    rows (e.g. non-JP ticker or < 1 year of coverage).
    """
    raw = get_fins_summary(code)
    df = _clean_summary(raw) if raw is not None else pd.DataFrame()
    if df.empty:
        return None
    fy = df[df["CurPerType"] == "FY"].copy()
    if fy.empty:
        return None
    fy["NxFEPS"] = pd.to_numeric(fy.get("NxFEPS"), errors="coerce")
    cols = ["DiscDate", "CurFYSt", "EPS", "NxFEPS"]
    return fy[cols].sort_values("DiscDate").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    tk = sys.argv[1] if len(sys.argv) > 1 else "7203"
    if not is_configured():
        print("No J-Quants credentials configured.")
        sys.exit(1)
    summary = get_fins_summary(tk)
    if summary is None:
        print(f"No summary data for {tk}")
        sys.exit(0)
    print(f"{tk} - {len(summary)} filings")
    print(summary[["DiscDate", "CurPerType", "DocType", "EPS", "FEPS"]].to_string())
    print()
    ttm = historical_ttm_eps(tk)
    if ttm is None:
        print("No TTM EPS series.")
    else:
        print(f"TTM EPS ({len(ttm)} points):")
        print(ttm.to_string())
