"""Trendline and Support/Resistance detection.

Pure algorithmic (not ML) detection of:
  - Horizontal support lines  (clusters of low pivots at the same price)
  - Horizontal resistance lines (clusters of high pivots at the same price)
  - Diagonal uptrend lines      (lines through multiple low pivots, +slope)
  - Diagonal downtrend lines    (lines through multiple high pivots, -slope)

Detection parameters live in ``TrendParams`` and are tuned by
``tuner.py`` to hit a precision target on past data.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

LineKind = Literal["support", "resistance", "trend_up", "trend_down"]
ScaleName = Literal["short", "mid", "long"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TrendParams:
    """Tunable parameters for trendline detection."""

    pivot_window: int = 10            # bars each side for local extrema
    tolerance_pct: float = 0.015      # price tolerance for "touching" a line
    min_touches: int = 3              # minimum touches to be a valid candidate
    max_slope_annual: float = 1.0     # cap diagonal slope (100% / year)
    lookback_bars: int = 500          # bars to analyse per chart
    max_lines_per_kind: int = 5       # keep top-K per category
    # Quality filters (prevent stale / instant clusters from passing)
    min_span_days: int = 30           # touches must span >= N calendar days
    max_last_touch_age_days: int = 180  # newest touch within N days of "now"


@dataclass
class Line:
    """A detected support/resistance/trend line."""

    kind: LineKind
    start_date: pd.Timestamp           # earliest pivot the line passes through
    start_price: float                 # line price at start_date
    end_date: pd.Timestamp             # latest pivot the line passes through
    end_price: float                   # line price at end_date
    touches: list[pd.Timestamp] = field(default_factory=list)
    score: float = 0.0                 # quality score (higher is better)
    valid: bool | None = None          # set later by evaluator
    scale: ScaleName = "long"          # which timeframe scale produced it

    # --- geometry helpers -------------------------------------------------
    def slope_per_day(self) -> float:
        """Price change per calendar day along the line (0 for horizontal)."""
        days = (self.end_date - self.start_date).days
        if days <= 0:
            return 0.0
        return (self.end_price - self.start_price) / days

    def price_at(self, date: pd.Timestamp) -> float:
        """Line's price at an arbitrary date (extrapolated if outside range)."""
        days = (date - self.start_date).days
        return self.start_price + self.slope_per_day() * days

    def is_support_like(self) -> bool:
        return self.kind in ("support", "trend_up")


# ---------------------------------------------------------------------------
# Pivot detection
# ---------------------------------------------------------------------------
def find_pivots(df: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """Return (high_pivots, low_pivots) as Series indexed by pivot date."""
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()

    high_idx = argrelextrema(highs, np.greater, order=window)[0]
    low_idx = argrelextrema(lows, np.less, order=window)[0]

    high_pivots = pd.Series(highs[high_idx], index=df.index[high_idx])
    low_pivots = pd.Series(lows[low_idx], index=df.index[low_idx])
    return high_pivots, low_pivots


# ---------------------------------------------------------------------------
# Horizontal line detection (support / resistance clusters)
# ---------------------------------------------------------------------------
def detect_horizontal_lines(
    pivots: pd.Series,
    tolerance_pct: float,
    min_touches: int,
    kind: LineKind,
) -> list[Line]:
    """Cluster pivots at similar prices into horizontal S/R lines."""
    if len(pivots) < min_touches:
        return []

    prices = pivots.to_numpy()
    dates = np.array(pivots.index)
    sort_idx = np.argsort(prices)
    sorted_prices = prices[sort_idx]
    sorted_dates = dates[sort_idx]

    lines: list[Line] = []
    i = 0
    n = len(sorted_prices)
    while i < n:
        # Greedy cluster expansion: take all prices within tolerance of the seed
        seed = sorted_prices[i]
        j = i + 1
        while j < n and (sorted_prices[j] - seed) / max(seed, 1e-9) <= tolerance_pct:
            j += 1
        cluster_prices = sorted_prices[i:j]
        cluster_dates = sorted_dates[i:j]

        if len(cluster_prices) >= min_touches:
            mean_price = float(cluster_prices.mean())
            touch_dates = sorted(pd.Timestamp(d) for d in cluster_dates)
            lines.append(
                Line(
                    kind=kind,
                    start_date=touch_dates[0],
                    start_price=mean_price,
                    end_date=touch_dates[-1],
                    end_price=mean_price,
                    touches=touch_dates,
                    # Score: touches, with bonus for recent activity
                    score=float(len(cluster_prices)),
                )
            )
        i = j
    return lines


# ---------------------------------------------------------------------------
# Diagonal line detection (trend lines)
# ---------------------------------------------------------------------------
def _dates_to_ordinals(dates: np.ndarray) -> np.ndarray:
    """Convert a datetime64 array to calendar-day integers (for slope math)."""
    return (dates.astype("datetime64[D]").astype(np.int64))


def detect_diagonal_lines(
    pivots: pd.Series,
    tolerance_pct: float,
    min_touches: int,
    max_slope_annual: float,
    kind: LineKind,
) -> list[Line]:
    """Find diagonal lines passing through multiple pivots.

    For each pair of pivots (i, j), define the line y = slope*x + intercept,
    then count how many pivots fall within ``tolerance_pct`` of that line.
    Keep lines with >= ``min_touches`` and reject impossibly steep slopes.
    """
    if len(pivots) < min_touches:
        return []

    dates = np.array(pivots.index)
    prices = pivots.to_numpy()
    ords = _dates_to_ordinals(dates)
    n = len(prices)

    candidates: list[Line] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = ords[j] - ords[i]
            if dx <= 0:
                continue
            slope = (prices[j] - prices[i]) / dx
            avg_price = (prices[i] + prices[j]) / 2.0
            # Slope sanity: skip >= X% per year moves
            annual_pct = abs(slope) * 365.0 / max(avg_price, 1e-9)
            if annual_pct > max_slope_annual:
                continue
            intercept = prices[i] - slope * ords[i]

            # Count touches: pivots within tolerance of the line
            predicted = slope * ords + intercept
            # Avoid division by zero for lines that cross zero
            denom = np.where(np.abs(predicted) > 1e-9, predicted, 1e-9)
            errors = np.abs(prices - predicted) / np.abs(denom)
            touched = np.where(errors <= tolerance_pct)[0]
            if len(touched) < min_touches:
                continue

            # Direction filter: support-ish lines only count if slope >= 0
            if kind == "trend_up" and slope < 0:
                continue
            if kind == "trend_down" and slope > 0:
                continue

            touched_dates = [pd.Timestamp(dates[k]) for k in touched]
            touched_dates.sort()
            start_date = touched_dates[0]
            end_date = touched_dates[-1]
            start_price = float(slope * ords[touched[0]] + intercept)
            end_price = float(slope * ords[touched[-1]] + intercept)
            candidates.append(
                Line(
                    kind=kind,
                    start_date=start_date,
                    start_price=start_price,
                    end_date=end_date,
                    end_price=end_price,
                    touches=touched_dates,
                    # Score: favour more touches AND longer span
                    score=float(len(touched) * 10 + (end_date - start_date).days / 30.0),
                )
            )

    # Deduplicate: greedy keep by score, reject near-duplicates
    candidates.sort(key=lambda l: -l.score)
    kept: list[Line] = []
    for c in candidates:
        is_dupe = False
        c_touches = set(c.touches)
        for k in kept:
            shared = len(c_touches & set(k.touches))
            # Two lines sharing >= 70% of touches are considered duplicates
            if shared / max(len(c.touches), 1) >= 0.7:
                is_dupe = True
                break
        if not is_dupe:
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Quality filters + scoring tweaks
# ---------------------------------------------------------------------------
def _filter_by_quality(
    lines: list[Line],
    now: pd.Timestamp,
    params: TrendParams,
) -> list[Line]:
    """Drop lines whose touches are clustered in time or too old."""
    out: list[Line] = []
    for line in lines:
        span_days = (line.end_date - line.start_date).days
        if span_days < params.min_span_days:
            continue
        age_days = (now - line.end_date).days
        if age_days < 0:  # shouldn't happen, but be safe
            age_days = 0
        if age_days > params.max_last_touch_age_days:
            continue
        out.append(line)
    return out


def _apply_recency_score(lines: list[Line], now: pd.Timestamp) -> None:
    """Boost the score of lines whose last touch is recent and span is long."""
    for line in lines:
        age = max(0, (now - line.end_date).days)
        span = max(0, (line.end_date - line.start_date).days)
        recency = max(0.0, 1.0 - age / 180.0)          # 0..1
        span_bonus = min(span / 365.0, 2.0)             # 0..2
        line.score = line.score + recency * 2.0 + span_bonus


# ---------------------------------------------------------------------------
# Top-level: detect everything (single scale)
# ---------------------------------------------------------------------------
def detect_all_lines(df: pd.DataFrame, params: TrendParams) -> list[Line]:
    """Run full detection (horizontal + diagonal) on an OHLCV DataFrame."""
    if len(df) > params.lookback_bars:
        df = df.iloc[-params.lookback_bars:]
    if len(df) < 50:
        return []

    now = df.index[-1]
    high_pivots, low_pivots = find_pivots(df, params.pivot_window)

    def _postprocess(raw: list[Line]) -> list[Line]:
        filtered = _filter_by_quality(raw, now, params)
        _apply_recency_score(filtered, now)
        return sorted(filtered, key=lambda l: -l.score)[: params.max_lines_per_kind]

    lines: list[Line] = []
    lines.extend(_postprocess(detect_horizontal_lines(
        high_pivots, params.tolerance_pct, params.min_touches, kind="resistance",
    )))
    lines.extend(_postprocess(detect_horizontal_lines(
        low_pivots, params.tolerance_pct, params.min_touches, kind="support",
    )))
    lines.extend(_postprocess(detect_diagonal_lines(
        low_pivots,
        params.tolerance_pct,
        params.min_touches,
        params.max_slope_annual,
        kind="trend_up",
    )))
    lines.extend(_postprocess(detect_diagonal_lines(
        high_pivots,
        params.tolerance_pct,
        params.min_touches,
        params.max_slope_annual,
        kind="trend_down",
    )))
    return lines


# ---------------------------------------------------------------------------
# Multi-scale detection (short / mid / long timeframes)
# ---------------------------------------------------------------------------
# Each scale overrides a subset of TrendParams. Universal params (tolerance,
# min_touches, max_slope_annual) come from the caller. Profiles are keyed
# first by bar interval and then by scale label so weekly and intraday
# charts don't inherit daily-sized lookback windows.
#
# Quick mental model for the numeric choices:
#
#   1d:  lookback 90 / 240 / 500 bars  ≈ 4mo / 1y / 2y
#        min_span in calendar days (15 / 30 / 60)
#        pivot_window 4 / 7 / 10
#   1wk: 1 weekly bar = 5 trading days
#        lookback 52 / 156 / 260 bars  ≈ 1y / 3y / 5y
#        min_span 60 / 180 / 365 days
#        pivot_window 3 / 4 / 6
INTERVAL_SCALE_PROFILES: dict[str, dict[str, dict]] = {
    "1d": {
        "short": {
            "lookback_bars": 90,
            "pivot_window": 4,
            "min_span_days": 15,
            "max_last_touch_age_days": 45,
        },
        "mid": {
            "lookback_bars": 240,
            "pivot_window": 7,
            "min_span_days": 30,
            "max_last_touch_age_days": 90,
        },
        "long": {
            "lookback_bars": 500,
            "pivot_window": 10,
            "min_span_days": 60,
            "max_last_touch_age_days": 180,
        },
    },
    "1wk": {
        "short": {
            "lookback_bars": 52,
            "pivot_window": 3,
            "min_span_days": 60,
            "max_last_touch_age_days": 120,
        },
        "mid": {
            "lookback_bars": 156,
            "pivot_window": 4,
            "min_span_days": 180,
            "max_last_touch_age_days": 240,
        },
        "long": {
            "lookback_bars": 260,
            "pivot_window": 6,
            "min_span_days": 365,
            "max_last_touch_age_days": 540,
        },
    },
}

# Back-compat alias — existing callers that import ``SCALE_PROFILES`` keep
# working against the daily profile.
SCALE_PROFILES = INTERVAL_SCALE_PROFILES["1d"]


def detect_all_lines_multiscale(
    df: pd.DataFrame,
    base_params: TrendParams,
    scales: list[str] | tuple[str, ...] = ("short", "mid", "long"),
    *,
    interval: str = "1d",
) -> list[Line]:
    """Detect lines at multiple time scales and merge near-duplicates.

    ``base_params`` supplies the "quality" knobs (tolerance_pct, min_touches,
    max_slope_annual, max_lines_per_kind). Each scale overrides the
    window-specific params via ``INTERVAL_SCALE_PROFILES[interval]`` so the
    same "short / mid / long" labels mean different lookback windows on
    daily vs weekly vs 5-min data.
    """
    profiles_for_interval = INTERVAL_SCALE_PROFILES.get(
        interval, INTERVAL_SCALE_PROFILES["1d"]
    )
    all_lines: list[Line] = []
    for scale in scales:
        profile = profiles_for_interval.get(scale)
        if profile is None:
            continue
        # Need enough bars to run detection at this scale
        if len(df) < profile["lookback_bars"] // 2:
            continue
        params = replace(base_params, **profile)
        lines = detect_all_lines(df, params)
        for line in lines:
            line.scale = scale  # tag
        all_lines.extend(lines)
    return _dedupe_across_scales(all_lines)


def _dedupe_across_scales(lines: list[Line]) -> list[Line]:
    """Remove near-duplicate lines detected at different scales.

    Two lines are duplicates if they're the same kind AND their prices
    at a common reference date are within 1% AND their annualised slopes
    differ by < 10% per year. Keeps the higher-scoring line.
    """
    lines_sorted = sorted(lines, key=lambda l: -l.score)
    kept: list[Line] = []
    for line in lines_sorted:
        is_dup = False
        for k in kept:
            if k.kind != line.kind:
                continue
            ref = max(line.end_date, k.end_date)
            p1 = line.price_at(ref)
            p2 = k.price_at(ref)
            if max(abs(p1), abs(p2)) < 1e-9:
                continue
            if abs(p1 - p2) / max(abs(p1), abs(p2)) > 0.01:
                continue
            # Annualised slope difference
            s1 = line.slope_per_day() * 365.0 / max(abs(p1), 1e-9)
            s2 = k.slope_per_day() * 365.0 / max(abs(p2), 1e-9)
            if abs(s1 - s2) > 0.10:
                continue
            is_dup = True
            break
        if not is_dup:
            kept.append(line)
    return kept
