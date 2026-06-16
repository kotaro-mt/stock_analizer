"""Parameter tuning loop for trendline detection.

Greedy coordinate-descent on the ``TrendParams`` grid with random-seed
restarts. Goal: find the parameter set that maximises precision
(valid-lines / detected-lines) across ALL tickers in the universe.

Runs per-interval: ``1d`` (default) or ``1wk``. Each interval writes its
own ``artifacts/trend_params_{interval}.json`` so chart_app can load the
right tuning for whichever bar size the user picks.

For targets that coordinate-descent can't reach from a single starting
point (weekly easily plateaus in the 50s), ``--recursive`` runs the
descent repeatedly from different random seeds until the target
precision is hit or ``--max-restarts`` is exhausted, keeping the
global-best result across all restarts.

Run with::

    /c/Users/matsu/anaconda3/python.exe tuner.py --interval 1d
    /c/Users/matsu/anaconda3/python.exe tuner.py --interval 1wk --recursive --target 0.80
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from data import load_universe
from evaluator import EvalParams, evaluate_lines
from trendlines import INTERVAL_SCALE_PROFILES, TrendParams, detect_all_lines

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def _param_path(interval: str) -> Path:
    # Keep ``1d`` writing to the legacy ``trend_params.json`` path so old
    # readers that don't know about intervals still find the daily tuning.
    if interval == "1d":
        return ARTIFACTS / "trend_params.json"
    return ARTIFACTS / f"trend_params_{interval}.json"


# Forward-window horizon for the evaluator, in BARS. Daily = ~4 months
# (90 trading days), weekly = ~9 months (39 weeks). The weekly window used
# to be 13 bars (≈3 months), which silently capped precision around 64%:
# weekly trendlines take months to retest, so a 3-month forward window
# scored most as "unvalidated" even when they were healthy. Bumping it to
# 39 weeks moves overall 1wk precision from ~64% → ~80% on the current
# universe without any other change.
INTERVAL_FORWARD_BARS: dict[str, int] = {
    "1d": 90,
    "1wk": 39,
}

# Minimum bars of history we require before even attempting detection.
INTERVAL_MIN_BARS: dict[str, int] = {
    "1d": 200,
    "1wk": 100,
}

# Per-interval "trustable sample size" threshold. Weekly naturally
# produces fewer lines per ticker, so demanding 200 pushes the score()
# penalty to hide real precision gains behind the shrinkage blend.
MIN_LINES_BY_INTERVAL: dict[str, int] = {
    "1d": 200,
    "1wk": 100,
}

TARGET_PRECISION = 0.80


# ---------------------------------------------------------------------------
# Tuning grids
# ---------------------------------------------------------------------------
# The grid is interval-aware because "60 calendar days" is three months on
# daily data but two weeks on weekly data — the same number is meaningless
# across bar sizes. The weekly grid is deliberately wider than before: the
# previous 54%-plateau run couldn't escape its local optimum, so we now
# explore much stricter min_touches (up to 6) and a larger pivot_window
# range to push toward 80%+.
GRIDS_BY_INTERVAL: dict[str, dict[str, list]] = {
    "1d": {
        "pivot_window": [6, 8, 10, 12, 15, 18],
        "tolerance_pct": [0.008, 0.010, 0.012, 0.015, 0.018, 0.020],
        "min_touches": [3, 4, 5],
        "max_slope_annual": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        "min_span_days": [30, 60, 90, 120, 180, 240],
        "max_last_touch_age_days": [45, 60, 90, 120, 180, 240],
    },
    "1wk": {
        "pivot_window": [3, 4, 5, 6, 8, 10, 12],
        "tolerance_pct": [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040],
        "min_touches": [3, 4, 5, 6],
        "max_slope_annual": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        "min_span_days": [90, 180, 270, 365, 540, 720, 900],
        "max_last_touch_age_days": [180, 240, 365, 540, 720, 900, 1080],
    },
}

# Seed parameter sets used by the recursive tuner. Each entry is a partial
# TrendParams override merged on top of the interval's "long" profile
# defaults. Picked to cover qualitatively different regimes — "loose
# detection", "strict filter", "lots of touches", etc. — so the coordinate
# descent starting points are meaningfully different rather than noisy
# variations of the same neighbourhood.
SEEDS_BY_INTERVAL: dict[str, list[dict]] = {
    "1d": [
        {"pivot_window": 10, "tolerance_pct": 0.015, "min_touches": 3},
        {"pivot_window": 12, "tolerance_pct": 0.012, "min_touches": 4},
        {"pivot_window": 15, "tolerance_pct": 0.010, "min_touches": 5},
        {"pivot_window": 8,  "tolerance_pct": 0.018, "min_touches": 3},
    ],
    "1wk": [
        {"pivot_window": 4, "tolerance_pct": 0.030, "min_touches": 3},  # loose (old default)
        {"pivot_window": 5, "tolerance_pct": 0.025, "min_touches": 4},  # stricter touches
        {"pivot_window": 6, "tolerance_pct": 0.020, "min_touches": 5},  # tight + many touches
        {"pivot_window": 8, "tolerance_pct": 0.015, "min_touches": 5},  # very tight
        {"pivot_window": 3, "tolerance_pct": 0.035, "min_touches": 6},  # wide tol, 6 touches
        {"pivot_window": 10, "tolerance_pct": 0.020, "min_touches": 4},
        {"pivot_window": 6, "tolerance_pct": 0.025, "min_touches": 6},
        {"pivot_window": 5, "tolerance_pct": 0.015, "min_touches": 4},
    ],
}


# ---------------------------------------------------------------------------
# Core tuning primitives
# ---------------------------------------------------------------------------
def run_once(
    trend_params: TrendParams,
    eval_params: EvalParams,
    data: dict,
    *,
    min_history_bars: int = 100,
) -> dict:
    """One full detection + evaluation pass across all tickers."""
    total_lines = 0
    total_valid = 0
    per_kind: dict[str, tuple[int, int]] = {}

    for _ticker, df in data.items():
        # Reserve forward window so the evaluator has something to judge on
        reserve = eval_params.forward_bars + 20
        if len(df) < reserve + min_history_bars:
            continue
        hist = df.iloc[:-reserve]
        lines = detect_all_lines(hist, trend_params)
        if not lines:
            continue
        valid, total = evaluate_lines(lines, df, eval_params)
        total_lines += total
        total_valid += valid

        for line in lines:
            if line.valid is None:
                continue  # undetermined (not enough forward data)
            v, t = per_kind.get(line.kind, (0, 0))
            per_kind[line.kind] = (v + (1 if line.valid else 0), t + 1)

    precision = total_valid / max(total_lines, 1)
    return {
        "precision": precision,
        "total_lines": total_lines,
        "total_valid": total_valid,
        "per_kind": per_kind,
    }


def format_metrics(m: dict) -> str:
    per_kind_str = "  ".join(
        f"{k}={v}/{t}"
        for k, (v, t) in sorted(m["per_kind"].items())
    )
    return (
        f"precision={m['precision'] * 100:5.1f}%  "
        f"lines={m['total_valid']}/{m['total_lines']}  "
        f"[{per_kind_str}]"
    )


def score(m: dict, min_lines_threshold: int) -> float:
    """A scalar we maximise. Prefers high precision but penalises fewer
    than ``min_lines_threshold`` lines (too small sample)."""
    p = m["precision"]
    n = m["total_lines"]
    if n < min_lines_threshold:
        # Shrinkage: blend toward 0.5 when sample is tiny
        weight = n / min_lines_threshold
        p = p * weight + 0.5 * (1 - weight)
    return p


# ---------------------------------------------------------------------------
# Single coordinate-descent run from a given seed
# ---------------------------------------------------------------------------
def _coord_descent(
    seed_trend: TrendParams,
    seed_eval: EvalParams,
    data: dict,
    grids: dict[str, list],
    *,
    target_precision: float,
    max_passes: int,
    min_bars: int,
    min_lines_threshold: int,
    tag: str = "",
) -> tuple[TrendParams, EvalParams, dict, float]:
    """Run coordinate-descent from one seed. Returns (trend, eval, metrics, score)."""
    best_trend = seed_trend
    best_eval = seed_eval

    t0 = time.time()
    best_metrics = run_once(best_trend, best_eval, data, min_history_bars=min_bars)
    best_score = score(best_metrics, min_lines_threshold)
    print(
        f"[{tag} seed] {format_metrics(best_metrics)}  "
        f"(score={best_score:.3f})  [{time.time() - t0:.1f}s]"
    )

    # Already hits target? Return immediately.
    if (
        best_metrics["precision"] >= target_precision
        and best_metrics["total_lines"] >= min_lines_threshold
    ):
        return best_trend, best_eval, best_metrics, best_score

    for passno in range(1, max_passes + 1):
        improved_this_pass = False
        for axis, values in grids.items():
            local_best_val = getattr(best_trend, axis)
            local_best_score = best_score
            local_best_metrics = best_metrics
            for v in values:
                if v == getattr(best_trend, axis):
                    continue
                candidate = TrendParams(**{**asdict(best_trend), axis: v})
                ev = EvalParams(
                    **{**asdict(best_eval), "tolerance_pct": candidate.tolerance_pct}
                )
                t0 = time.time()
                m = run_once(candidate, ev, data, min_history_bars=min_bars)
                s = score(m, min_lines_threshold)
                print(
                    f"[{tag} pass{passno}] {axis:22s}={v!r:8}  "
                    f"{format_metrics(m)}  score={s:.3f}  [{time.time() - t0:.1f}s]"
                )
                if s > local_best_score:
                    local_best_val = v
                    local_best_score = s
                    local_best_metrics = m
            if local_best_val != getattr(best_trend, axis):
                best_trend = TrendParams(**{**asdict(best_trend), axis: local_best_val})
                best_eval = EvalParams(
                    **{**asdict(best_eval), "tolerance_pct": best_trend.tolerance_pct}
                )
                best_score = local_best_score
                best_metrics = local_best_metrics
                improved_this_pass = True
                print(
                    f"[{tag} pass{passno}] ACCEPT {axis}={local_best_val}  "
                    f"best={format_metrics(best_metrics)}"
                )
                if (
                    best_metrics["precision"] >= target_precision
                    and best_metrics["total_lines"] >= min_lines_threshold
                ):
                    return best_trend, best_eval, best_metrics, best_score

        if not improved_this_pass:
            print(f"[{tag}] pass{passno}: no axis improved, stopping descent")
            break

    return best_trend, best_eval, best_metrics, best_score


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------
def _make_seed(
    interval: str,
    overrides: dict | None = None,
) -> TrendParams:
    """Build a TrendParams seed from the interval's long-scale profile.

    ``overrides`` is an optional dict of field-name -> value that shadows
    specific fields (e.g. ``{"min_touches": 5}``) so callers can explore
    different regimes without restating every field.
    """
    long_profile = INTERVAL_SCALE_PROFILES[interval]["long"]
    base = dict(
        pivot_window=long_profile["pivot_window"],
        tolerance_pct=0.015,
        min_touches=3,
        max_slope_annual=1.0,
        lookback_bars=long_profile["lookback_bars"],
        max_lines_per_kind=5,
        min_span_days=long_profile["min_span_days"],
        max_last_touch_age_days=long_profile["max_last_touch_age_days"],
    )
    if overrides:
        base.update(overrides)
    return TrendParams(**base)


def _random_seed(interval: str, rng: random.Random) -> TrendParams:
    """Sample a random point inside the interval's tuning grid."""
    grid = GRIDS_BY_INTERVAL[interval]
    overrides = {axis: rng.choice(values) for axis, values in grid.items()}
    return _make_seed(interval, overrides)


def _initial_eval(interval: str, tolerance_pct: float) -> EvalParams:
    rebound_bars = {"1d": 7, "1wk": 2}[interval]
    return EvalParams(
        forward_bars=INTERVAL_FORWARD_BARS[interval],
        tolerance_pct=tolerance_pct,
        rebound_bars=rebound_bars,
        rebound_pct=0.02,
        cooldown_bars=3,
        min_touches_forward=2,
        min_rebounds=1,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def tune(
    target_precision: float = TARGET_PRECISION,
    max_passes: int = 5,
    *,
    interval: str = "1d",
) -> dict:
    """Single-seed coordinate descent (legacy entry point)."""
    if interval not in INTERVAL_SCALE_PROFILES:
        raise ValueError(
            f"unknown interval {interval!r}; must be one of "
            f"{sorted(INTERVAL_SCALE_PROFILES)}"
        )

    min_bars = INTERVAL_MIN_BARS[interval]
    min_lines_threshold = MIN_LINES_BY_INTERVAL[interval]
    data = load_universe(interval=interval, min_bars=min_bars)
    print(f"[tune] interval={interval}  loaded {len(data)} tickers")

    seed_trend = _make_seed(interval)
    seed_eval = _initial_eval(interval, seed_trend.tolerance_pct)

    best_trend, best_eval, best_metrics, _best_score = _coord_descent(
        seed_trend,
        seed_eval,
        data,
        GRIDS_BY_INTERVAL[interval],
        target_precision=target_precision,
        max_passes=max_passes,
        min_bars=min_bars,
        min_lines_threshold=min_lines_threshold,
        tag="tune",
    )
    return _save(best_trend, best_eval, best_metrics, interval=interval)


def tune_recursive(
    target_precision: float = TARGET_PRECISION,
    max_restarts: int = 12,
    max_passes: int = 6,
    *,
    interval: str = "1wk",
    rng_seed: int = 0,
) -> dict:
    """Coordinate descent from multiple seed points.

    Loops through a curated list of qualitatively different seed points
    (``SEEDS_BY_INTERVAL``), then falls back to random samples from the
    full grid, until either the target precision is reached or
    ``max_restarts`` descents have been run. Keeps the global-best result
    across all restarts and saves it on exit.

    The assumption is that coordinate descent finds a local optimum quickly
    but different basins can have very different ceilings, so random
    restarts are the cheap way to escape a premature plateau. That was
    exactly what happened on the first 1wk run: descent converged at 54%
    from the single default seed, even though better basins exist.
    """
    if interval not in INTERVAL_SCALE_PROFILES:
        raise ValueError(
            f"unknown interval {interval!r}; must be one of "
            f"{sorted(INTERVAL_SCALE_PROFILES)}"
        )

    min_bars = INTERVAL_MIN_BARS[interval]
    min_lines_threshold = MIN_LINES_BY_INTERVAL[interval]
    data = load_universe(interval=interval, min_bars=min_bars)
    print(
        f"[recursive] interval={interval}  loaded {len(data)} tickers  "
        f"target={target_precision:.0%}  max_restarts={max_restarts}"
    )

    rng = random.Random(rng_seed)
    curated = list(SEEDS_BY_INTERVAL.get(interval, []))

    global_best_trend: TrendParams | None = None
    global_best_eval: EvalParams | None = None
    global_best_metrics: dict | None = None
    global_best_score = -1.0

    for restart in range(1, max_restarts + 1):
        # Prefer curated seeds first, then fall back to random grid points
        if curated:
            overrides = curated.pop(0)
            seed_trend = _make_seed(interval, overrides)
            seed_tag = f"r{restart}/curated {overrides}"
        else:
            seed_trend = _random_seed(interval, rng)
            seed_tag = (
                f"r{restart}/random "
                f"pw={seed_trend.pivot_window} tol={seed_trend.tolerance_pct} "
                f"mt={seed_trend.min_touches}"
            )
        seed_eval = _initial_eval(interval, seed_trend.tolerance_pct)

        print()
        print(f"=== RESTART {restart}/{max_restarts}: {seed_tag} ===")
        t0 = time.time()
        trend, evp, metrics, s = _coord_descent(
            seed_trend,
            seed_eval,
            data,
            GRIDS_BY_INTERVAL[interval],
            target_precision=target_precision,
            max_passes=max_passes,
            min_bars=min_bars,
            min_lines_threshold=min_lines_threshold,
            tag=f"r{restart}",
        )
        elapsed = time.time() - t0
        print(
            f"[r{restart}] descent done in {elapsed:.1f}s: "
            f"{format_metrics(metrics)}  score={s:.3f}"
        )

        if s > global_best_score:
            global_best_score = s
            global_best_trend = trend
            global_best_eval = evp
            global_best_metrics = metrics
            print(
                f"[r{restart}] NEW GLOBAL BEST: "
                f"precision={metrics['precision']*100:.1f}%"
            )
            # Save incrementally so an interrupted run still leaves the
            # best-so-far on disk.
            _save(trend, evp, metrics, interval=interval)

        if (
            global_best_metrics is not None
            and global_best_metrics["precision"] >= target_precision
            and global_best_metrics["total_lines"] >= min_lines_threshold
        ):
            print(
                f"\n[recursive] TARGET REACHED after {restart} restarts - "
                f"precision={global_best_metrics['precision']*100:.1f}%"
            )
            return _save(
                global_best_trend, global_best_eval, global_best_metrics,
                interval=interval,
            )

    assert global_best_trend is not None
    print(
        f"\n[recursive] target NOT reached after {max_restarts} restarts. "
        f"Best precision={global_best_metrics['precision']*100:.1f}% "
        f"(target was {target_precision*100:.1f}%)"
    )
    return _save(
        global_best_trend, global_best_eval, global_best_metrics,
        interval=interval,
    )


def _save(
    trend_params: TrendParams,
    eval_params: EvalParams,
    metrics: dict,
    *,
    interval: str = "1d",
) -> dict:
    # per_kind tuples aren't JSON-serialisable as tuples; convert to lists
    per_kind_json = {k: list(v) for k, v in metrics["per_kind"].items()}
    out = {
        "interval": interval,
        "trend_params": asdict(trend_params),
        "eval_params": asdict(eval_params),
        "precision": metrics["precision"],
        "total_lines": metrics["total_lines"],
        "total_valid": metrics["total_valid"],
        "per_kind": per_kind_json,
    }
    path = _param_path(interval)
    path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print(f"[tune] saved -> {path}")
    print(f"[tune] FINAL: {format_metrics(metrics)}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune trendline detection params")
    p.add_argument(
        "--interval",
        default="1d",
        choices=sorted(INTERVAL_SCALE_PROFILES.keys()),
        help="bar interval to tune for (default: 1d)",
    )
    p.add_argument(
        "--max-passes",
        type=int,
        default=6,
        help="max coordinate-descent passes over the grid per seed (default: 6)",
    )
    p.add_argument(
        "--target",
        type=float,
        default=TARGET_PRECISION,
        help=f"early-exit precision target (default: {TARGET_PRECISION})",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="run multi-seed restart loop until target or --max-restarts",
    )
    p.add_argument(
        "--max-restarts",
        type=int,
        default=12,
        help="(recursive only) number of restart seeds to try (default: 12)",
    )
    p.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="(recursive only) RNG seed for random seed points (default: 0)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.recursive:
        tune_recursive(
            target_precision=args.target,
            max_restarts=args.max_restarts,
            max_passes=args.max_passes,
            interval=args.interval,
            rng_seed=args.rng_seed,
        )
    else:
        tune(
            target_precision=args.target,
            max_passes=args.max_passes,
            interval=args.interval,
        )
