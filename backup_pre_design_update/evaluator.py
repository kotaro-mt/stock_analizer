"""Validate detected lines against *forward* price action.

Definition B (standard) — a line is considered valid if, within
``forward_bars`` trading days AFTER its last pivot:

  1. price touched the line at least ``min_touches_forward`` more times
     (touch = bar's low/high comes within ``tolerance_pct`` of the line),
     with consecutive touches separated by >= ``cooldown_bars`` days so
     we don't double-count the same "press";
  2. at least ``min_rebounds`` of those touches are followed by a rebound
     of at least ``rebound_pct`` away from the line in the expected
     direction within ``rebound_bars`` days.

This is a strict enough rule that only lines that actually acted as
support/resistance — not coincidences — are counted as valid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trendlines import Line


@dataclass
class EvalParams:
    """Line validation parameters (should roughly match detection tolerance)."""

    forward_bars: int = 90          # how far forward we look
    tolerance_pct: float = 0.015    # touch tolerance (match detection)
    rebound_bars: int = 7           # days to realise a rebound after touch
    rebound_pct: float = 0.020      # rebound size as fraction of line price
    cooldown_bars: int = 3          # minimum gap between touches
    min_touches_forward: int = 2    # required number of forward touches
    min_rebounds: int = 1           # of which this many must actually rebound


def evaluate_line(line: Line, df: pd.DataFrame, params: EvalParams) -> bool | None:
    """Return True/False if the line (did not) function as S/R in the forward
    window. Returns ``None`` when there is not enough forward data to judge
    — e.g. the line was detected at today's end of data, so we can't tell yet.
    """
    # Only consider bars strictly AFTER the last pivot the line was built from
    after = df.loc[df.index > line.end_date]
    if len(after) < 10:  # not enough forward data to judge
        return None
    after = after.iloc[: params.forward_bars]

    highs = after["High"].to_numpy()
    lows = after["Low"].to_numpy()
    dates = after.index

    is_support = line.is_support_like()

    touches = 0
    rebounds = 0
    last_touch_i = -10_000

    for i in range(len(dates)):
        line_price = line.price_at(dates[i])
        if line_price <= 0:
            continue

        tol_abs = line_price * params.tolerance_pct

        if is_support:
            # Support: we consider "touched" if the day's low dips near the line
            touched = (lows[i] <= line_price + tol_abs) and (lows[i] >= line_price - tol_abs * 3)
        else:
            # Resistance: the day's high pokes near the line
            touched = (highs[i] >= line_price - tol_abs) and (highs[i] <= line_price + tol_abs * 3)

        if touched and (i - last_touch_i) >= params.cooldown_bars:
            touches += 1
            last_touch_i = i

            # Rebound: within rebound_bars, does price move away by rebound_pct?
            end_i = min(i + params.rebound_bars + 1, len(dates))
            if end_i > i + 1:
                if is_support:
                    max_high = highs[i + 1 : end_i].max()
                    if max_high >= line_price * (1 + params.rebound_pct):
                        rebounds += 1
                else:
                    min_low = lows[i + 1 : end_i].min()
                    if min_low <= line_price * (1 - params.rebound_pct):
                        rebounds += 1

    return (
        touches >= params.min_touches_forward
        and rebounds >= params.min_rebounds
    )


def evaluate_lines(
    lines: list[Line],
    df: pd.DataFrame,
    params: EvalParams,
) -> tuple[int, int]:
    """Batch-evaluate lines. Returns (valid_count, determined_count).

    ``determined_count`` excludes lines that could not be evaluated (not enough
    forward data). Mutates each line's ``.valid`` field with the result
    (``True`` / ``False`` / ``None``).
    """
    valid = 0
    determined = 0
    for line in lines:
        result = evaluate_line(line, df, params)
        line.valid = result
        if result is None:
            continue
        determined += 1
        if result:
            valid += 1
    return valid, determined
