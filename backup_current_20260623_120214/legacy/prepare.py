"""Data prep + feature engineering + Sharpe evaluation for stock_future.

This file is READ-ONLY from the perspective of the self-improvement loop.
train.py may freely import from it, but must not modify it. The Sharpe
evaluation implemented here is the ground-truth metric the loop optimizes.

Pipeline (Phase A)
------------------
1. Download ~10 years of daily OHLCV for every ticker in universe.py via
   yfinance, caching one parquet per ticker under ``cache/``.
2. Build a universe-wide reference series (``compute_market_reference``):
   - ``mkt_logret``          : cross-sectional mean of 1-day log returns
   - ``mkt_total_turnover``  : sum of close*volume across all tickers
3. Compute ~13 engineered features per ticker (log returns, MA ratios,
   RSI, volatility, volume z-score, intraday range, plus trading-value
   features that use the universe turnover reference).
4. Build sliding windows of shape ``(LOOKBACK=60, num_features)`` with a
   target of the next ``HORIZON=20`` daily **residual** log returns
   (``individual − universe mean``) — plus the base date and ticker
   for evaluation grouping.
5. Time-split globally: train ≤ 2022, val 2023–2024, test 2025+.
6. Fit a per-feature z-score scaler on train only, save to ``artifacts/``.
7. Cache the whole prepared bundle as a single joblib pickle so train.py
   reloads it instantly on every iteration.

Sharpe evaluation
-----------------
``evaluate_sharpe`` builds a long-flat strategy from the model's predicted
20-day cumulative **residual** log return: go long tomorrow if the
prediction is > 0 (i.e. the model expects the ticker to beat the
universe mean). Daily portfolio return on date d is the mean 1-day
realized **residual** return across longs. Because labels are
market-neutralised, always-long has expected Sharpe ≈ 0, and any
positive Sharpe reflects real cross-sectional predictive skill rather
than bull-market β. Annualized by ``× √252``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset

from universe import tickers as _universe_tickers

# ---------------------------------------------------------------------------
# Paths and fixed constants — do NOT change in train.py
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
ARTIFACTS_DIR = ROOT / "artifacts"
CACHE_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
# v3 in Phase B: cross-sectional rank features added (17 features total)
# v4_h5: horizon shortened 20 -> 5 days (feature-horizon match experiment)
PREPARED_CACHE = CACHE_DIR / "prepared_v4_h5.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.npz"

LOOKBACK = 60               # days fed to the model
HORIZON = 5                 # prediction horizon (1 week of trading days)
HISTORY_START = "2015-01-01"
TRAIN_END = "2022-12-31"
VAL_END = "2024-12-31"

FEATURE_COLS: list[str] = [
    # --- price / momentum ---
    "logret_1d",
    "logret_5d",
    "logret_20d",
    "close_over_ma5",
    "close_over_ma25",
    "close_over_ma75",
    "rsi14_centered",
    # --- volatility / volume / range ---
    "vol20",
    "volzscore20",
    "high_low_range",
    # --- Phase A: trading-value / liquidity ---
    "turnover_z_60d",        # within-ticker liquidity anomaly
    "turnover_share_20d",    # this ticker's 20d share of universe turnover
    "big_move_count_20d",    # fraction of last 20 days with |logret_1d| > 3%
    # --- Phase B: cross-sectional rank features (per-date rank across
    #     the universe, centred to [-0.5, +0.5]). These make momentum
    #     vs. reversal explicitly cross-sectional instead of absolute.
    "rank_logret_20d",
    "rank_rsi14",
    "rank_vol20",
    "rank_turnover_z_60d",
]


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('.', '_')}.parquet"


def download_ticker(ticker: str, start: str = HISTORY_START) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, auto_adjust=True,
                         progress=False, threads=False)
    except Exception as e:  # pragma: no cover - network failure paths
        print(f"[warn] download failed {ticker}: {e}")
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df.to_parquet(path)
    return df


def download_all(tickers_list: Iterable[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    tickers_list = list(tickers_list)
    try:
        from tqdm import tqdm
        iterator = tqdm(tickers_list, desc="download")
    except Exception:
        iterator = tickers_list
    for t in iterator:
        df = download_ticker(t)
        if df is not None and len(df) > LOOKBACK + HORIZON + 50:
            out[t] = df
    return out


# ---------------------------------------------------------------------------
# Market reference (universe-wide aggregates used for residual labels,
# per-ticker trading-value features, and cross-sectional rank features)
# ---------------------------------------------------------------------------
def compute_market_reference(data: dict[str, pd.DataFrame]
                             ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build universe-wide reference panels.

    Returns ``(market_df, rank_panels)`` where:

    - ``market_df`` is a DataFrame indexed by business date with:
        - ``mkt_logret``         : cross-sectional mean of 1-day log
                                   returns across tickers with data on
                                   that date (for residual labels).
        - ``mkt_total_turnover`` : cross-sectional sum of ``close*volume``
                                   (for ``turnover_share_20d``).

    - ``rank_panels`` is a dict mapping feature name → DataFrame
      ``[date × ticker]`` of **per-date percentile ranks centred to
      ``[-0.5, +0.5]``**. Each panel answers the question "where does
      this ticker sit in the universe *today* on this indicator?" —
      which is the cross-sectional signal the model was blind to in
      Phase A. Supplied panels:

        * ``rank_logret_20d``
        * ``rank_rsi14``
        * ``rank_vol20``
        * ``rank_turnover_z_60d``
    """
    logret_cols: dict[str, pd.Series] = {}
    turnover_cols: dict[str, pd.Series] = {}
    logret20_cols: dict[str, pd.Series] = {}
    rsi_cols: dict[str, pd.Series] = {}
    vol20_cols: dict[str, pd.Series] = {}
    turnover_z_cols: dict[str, pd.Series] = {}

    for ticker, df in data.items():
        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float)
        logret_1d = np.log(close / close.shift(1))
        turnover = close * volume

        logret_cols[ticker] = logret_1d
        turnover_cols[ticker] = turnover

        logret20_cols[ticker] = np.log(close / close.shift(20))
        rsi_cols[ticker] = _rsi(close, 14) / 100.0 - 0.5
        vol20_cols[ticker] = logret_1d.rolling(20).std()
        t_mean60 = turnover.rolling(60).mean()
        t_std60 = turnover.rolling(60).std()
        turnover_z_cols[ticker] = (turnover - t_mean60) / (t_std60 + 1e-9)

    lr_df = pd.DataFrame(logret_cols)
    tv_df = pd.DataFrame(turnover_cols)
    market_df = pd.DataFrame({
        "mkt_logret": lr_df.mean(axis=1, skipna=True),
        "mkt_total_turnover": tv_df.sum(axis=1, skipna=True, min_count=1),
    })

    def _rank_panel(cols: dict[str, pd.Series]) -> pd.DataFrame:
        # [date × ticker] → percentile rank along axis=1 (per date),
        # then centre to [-0.5, +0.5]. NaNs are left as NaN so they
        # don't leak into the model.
        panel = pd.DataFrame(cols)
        ranks = panel.rank(axis=1, pct=True) - 0.5
        return ranks

    rank_panels: dict[str, pd.DataFrame] = {
        "rank_logret_20d":     _rank_panel(logret20_cols),
        "rank_rsi14":          _rank_panel(rsi_cols),
        "rank_vol20":          _rank_panel(vol20_cols),
        "rank_turnover_z_60d": _rank_panel(turnover_z_cols),
    }
    return market_df, rank_panels


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def build_features(df: pd.DataFrame,
                   market_df: pd.DataFrame | None = None,
                   rank_panels: dict[str, pd.DataFrame] | None = None,
                   ticker: str | None = None) -> pd.DataFrame:
    """Compute engineered features for a single ticker.

    If ``market_df`` is provided, Phase A trading-value features that
    require universe context (``turnover_share_20d``) are populated
    against it; otherwise they fall back to a neutral placeholder
    (single-ticker Streamlit inference path).

    If ``rank_panels`` and ``ticker`` are both provided, Phase B
    cross-sectional rank features are looked up from the pre-computed
    per-date rank panels. At inference time (no universe) they fall
    back to 0 (the centre of the rank scale).
    """
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    logret_1d = np.log(close / close.shift(1))
    feats = pd.DataFrame(index=df.index)
    feats["logret_1d"] = logret_1d
    feats["logret_5d"] = np.log(close / close.shift(5))
    feats["logret_20d"] = np.log(close / close.shift(20))
    feats["close_over_ma5"] = close / close.rolling(5).mean() - 1.0
    feats["close_over_ma25"] = close / close.rolling(25).mean() - 1.0
    feats["close_over_ma75"] = close / close.rolling(75).mean() - 1.0
    feats["rsi14_centered"] = _rsi(close, 14) / 100.0 - 0.5
    feats["vol20"] = logret_1d.rolling(20).std()
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    feats["volzscore20"] = (volume - vol_mean) / (vol_std + 1e-9)
    feats["high_low_range"] = (high - low) / close

    # --- Phase A trading-value / liquidity features ---
    turnover = close * volume
    t_mean60 = turnover.rolling(60).mean()
    t_std60 = turnover.rolling(60).std()
    feats["turnover_z_60d"] = (turnover - t_mean60) / (t_std60 + 1e-9)

    turnover_ma20 = turnover.rolling(20).mean()
    if market_df is not None:
        mkt_tv = market_df["mkt_total_turnover"].reindex(df.index)
        mkt_tv_ma20 = mkt_tv.rolling(20).mean()
        feats["turnover_share_20d"] = turnover_ma20 / (mkt_tv_ma20 + 1e-9)
    else:
        # Inference-time fallback (no universe context): use a small
        # constant so the scaler treats it as a flat, neutral feature.
        feats["turnover_share_20d"] = 0.01

    feats["big_move_count_20d"] = (
        logret_1d.abs().gt(0.03).rolling(20).sum() / 20.0
    )

    # --- Phase B cross-sectional rank features ---
    rank_feature_names = [
        "rank_logret_20d", "rank_rsi14",
        "rank_vol20", "rank_turnover_z_60d",
    ]
    if rank_panels is not None and ticker is not None:
        for rname in rank_feature_names:
            panel = rank_panels.get(rname)
            if panel is not None and ticker in panel.columns:
                series = panel[ticker].reindex(df.index)
                feats[rname] = series
            else:
                feats[rname] = 0.0
    else:
        # Inference-time fallback: neutral rank (0 = centre of [-0.5, +0.5]).
        for rname in rank_feature_names:
            feats[rname] = 0.0

    # clamp pathological values
    feats["volzscore20"] = feats["volzscore20"].clip(-10, 10)
    feats["turnover_z_60d"] = feats["turnover_z_60d"].clip(-10, 10)
    feats["turnover_share_20d"] = feats["turnover_share_20d"].clip(0.0, 0.5)
    return feats[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------
@dataclass
class Windows:
    X: np.ndarray          # [N, LOOKBACK, F]  float32
    y: np.ndarray          # [N, HORIZON]      float32
    dates: np.ndarray      # [N]  datetime64[ns]
    tickers: np.ndarray    # [N]  object (ticker string)

    def __len__(self) -> int:
        return int(len(self.X))


def _make_windows_single(feats: pd.DataFrame, close: pd.Series,
                         mkt_logret: pd.Series | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(X, y, dates)`` windows for a single ticker.

    When ``mkt_logret`` is supplied the labels are residualised against
    the universe mean (market-neutral Phase A labels). When it is
    ``None`` the labels are plain absolute log returns (kept for
    possible legacy / diagnostic use).
    """
    X_arr = feats.values.astype(np.float32)
    close_arr = close.values.astype(np.float64)
    # logret[i] = log(close[i+1]/close[i]) — length T-1.
    # ticker_logret[i] is the return *realised on* feats.index[i+1].
    with np.errstate(divide="ignore", invalid="ignore"):
        ticker_logret = np.log(close_arr[1:] / close_arr[:-1])
    if mkt_logret is not None:
        mkt_series = mkt_logret.reindex(feats.index).values.astype(np.float64)
        # Align market return to the same "realised on day i+1" convention.
        mkt_aligned = mkt_series[1:]
        y_series = ticker_logret - mkt_aligned
    else:
        y_series = ticker_logret

    n = len(X_arr)
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ds: list[np.datetime64] = []
    start = LOOKBACK - 1
    end = n - HORIZON - 1  # need y_series[i : i+HORIZON]
    for i in range(start, end):
        x = X_arr[i - LOOKBACK + 1 : i + 1]
        y = y_series[i : i + HORIZON]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            continue
        Xs.append(x)
        ys.append(y.astype(np.float32))
        ds.append(feats.index[i].to_datetime64())
    if not Xs:
        return (np.empty((0, LOOKBACK, len(FEATURE_COLS)), dtype=np.float32),
                np.empty((0, HORIZON), dtype=np.float32),
                np.empty((0,), dtype="datetime64[ns]"))
    return (np.stack(Xs),
            np.stack(ys),
            np.array(ds, dtype="datetime64[ns]"))


def build_windows_for_all(data: dict[str, pd.DataFrame]) -> Windows:
    market_df, rank_panels = compute_market_reference(data)
    mkt_logret = market_df["mkt_logret"]
    Xs, ys, ds, ts = [], [], [], []
    for ticker, df in data.items():
        feats = build_features(df, market_df,
                               rank_panels=rank_panels, ticker=ticker)
        close = df["Close"].astype(float)
        X, y, d = _make_windows_single(feats, close, mkt_logret)
        if len(X) == 0:
            continue
        Xs.append(X)
        ys.append(y)
        ds.append(d)
        ts.append(np.array([ticker] * len(X), dtype=object))
    if not Xs:
        raise RuntimeError("No windows built — data empty?")
    X_all = np.concatenate(Xs)
    y_all = np.concatenate(ys)
    d_all = np.concatenate(ds)
    t_all = np.concatenate(ts)

    # ------------------------------------------------------------------
    # Per-date double-centering
    # ------------------------------------------------------------------
    # ``_make_windows_single`` already subtracted the universe mean, but
    # the "universe" at label time includes tickers that may not have a
    # valid window on the same date (because of history/future gaps).
    # That leaves a small residual β baked into the labels: a strategy
    # that simply goes long every ticker-with-a-window each day does
    # *not* earn exactly zero.
    #
    # We fix this analytically by demeaning each window's 20-day label
    # vector against the cross-sectional mean of all windows that share
    # the same start date. After this pass, the cross-section of labels
    # on every window-start date sums to zero, so ``always_long`` has
    # precisely zero expected residual return → ``always_long_sharpe = 0``
    # by construction. Any positive Sharpe from a selective strategy is
    # therefore *purely* a subset-selection effect (variance reduction)
    # or real predictive skill — never market β leakage.
    # ------------------------------------------------------------------
    dates_ns = d_all.astype("datetime64[ns]")
    unique_dates, inverse = np.unique(dates_ns, return_inverse=True)
    n_unique = len(unique_dates)
    sum_per_date = np.zeros((n_unique, y_all.shape[1]), dtype=np.float64)
    np.add.at(sum_per_date, inverse, y_all.astype(np.float64))
    count_per_date = np.bincount(inverse, minlength=n_unique).astype(np.float64)
    mean_per_date = sum_per_date / count_per_date[:, None]
    y_centered = (y_all - mean_per_date[inverse]).astype(np.float32)

    return Windows(X=X_all, y=y_centered, dates=d_all, tickers=t_all)


def split_windows(w: Windows) -> tuple[Windows, Windows, Windows]:
    train_end = np.datetime64(TRAIN_END)
    val_end = np.datetime64(VAL_END)
    train_mask = w.dates <= train_end
    val_mask = (w.dates > train_end) & (w.dates <= val_end)
    test_mask = w.dates > val_end
    def sub(m: np.ndarray) -> Windows:
        return Windows(X=w.X[m], y=w.y[m], dates=w.dates[m], tickers=w.tickers[m])
    return sub(train_mask), sub(val_mask), sub(test_mask)


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------
class FeatureScaler:
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0).astype(np.float32)
        self.std = flat.std(axis=0).astype(np.float32) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        z = np.load(path)
        s = cls()
        s.mean = z["mean"].astype(np.float32)
        s.std = z["std"].astype(np.float32)
        return s


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------
class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def prepare_all(smoke: bool = False, force: bool = False) -> dict:
    """Return a dict with train/val/test Windows + scaler + config constants."""
    if (not smoke) and (not force) and PREPARED_CACHE.exists():
        try:
            bundle = joblib.load(PREPARED_CACHE)
            print(f"[prepare] loaded cached bundle: "
                  f"train={len(bundle['train'])} val={len(bundle['val'])} "
                  f"test={len(bundle['test'])}")
            return bundle
        except Exception as e:  # pragma: no cover
            print(f"[prepare] cache load failed ({e}); rebuilding")

    t0 = time.time()
    ticks = _universe_tickers()
    if smoke:
        ticks = ticks[:3]
    data = download_all(ticks)
    print(f"[prepare] loaded {len(data)} tickers in {time.time() - t0:.1f}s")

    windows = build_windows_for_all(data)
    print(f"[prepare] total windows: {len(windows)}  "
          f"(labels = residual log returns, {len(FEATURE_COLS)} features)")

    train, val, test = split_windows(windows)
    print(f"[prepare] train={len(train)}  val={len(val)}  test={len(test)}")

    scaler = FeatureScaler()
    train_X = scaler.fit_transform(train.X)
    val_X = scaler.transform(val.X)
    test_X = scaler.transform(test.X)
    scaler.save(SCALER_PATH)

    bundle = {
        "train": Windows(X=train_X, y=train.y, dates=train.dates, tickers=train.tickers),
        "val": Windows(X=val_X, y=val.y, dates=val.dates, tickers=val.tickers),
        "test": Windows(X=test_X, y=test.y, dates=test.dates, tickers=test.tickers),
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "feature_cols": FEATURE_COLS,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
    }
    if not smoke:
        joblib.dump(bundle, PREPARED_CACHE, compress=3)
        print(f"[prepare] cached prepared bundle -> {PREPARED_CACHE}")
    return bundle


# ---------------------------------------------------------------------------
# Sharpe evaluation  (the ground-truth metric)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_all_metrics(model: torch.nn.Module,
                         val_windows: Windows,
                         batch_size: int = 1024,
                         signal_threshold: float = 0.0,
                         device: str = "cpu") -> dict:
    """Compute the full bundle of market-neutral evaluation metrics.

    Labels in ``val_windows.y`` are **doubly-centred** residual log
    returns (individual − universe mean, then − per-date window-mean),
    so always-long has zero expected residual return by construction.

    Returns a dict with:
    - ``sharpe``             : annualised Sharpe of a long-only strategy
                               that buys every window with predicted 20d
                               cumulative > ``signal_threshold``.
    - ``always_long_sharpe`` : Sharpe of buying every window every day.
                               Should be ~0 (floating-point noise).
    - ``val_mse``            : MSE of the 20-vector prediction (or of the
                               scalar cum prediction if the model emits
                               one number).
    - ``ic_spearman``        : Spearman rank correlation of predicted 20d
                               cumulative return vs realised, pooled
                               across all windows. This is the primary
                               cross-sectional-skill metric.
    - ``ic_pearson``         : Pearson equivalent.
    - ``dir_acc_20d``        : sign(cum_pred) == sign(cum_true) accuracy.
    - ``frac_long``          : fraction of windows where cum_pred > 0.
    - ``n_days``             : number of unique evaluation dates.
    """
    model.eval()
    X = torch.from_numpy(val_windows.X).float()
    y = torch.from_numpy(val_windows.y).float()
    preds_cum: list[np.ndarray] = []
    mse_num = 0.0
    mse_den = 0
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        out = model(xb)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        if out.shape[-1] == HORIZON:
            cum = out.sum(dim=-1)
            mse_num += ((out - yb) ** 2).mean().item() * len(xb)
        else:
            cum = out.squeeze(-1)
            mse_num += ((cum - yb.sum(dim=-1)) ** 2).mean().item() * len(xb)
        mse_den += len(xb)
        preds_cum.append(cum.detach().cpu().numpy())

    cum_preds = np.concatenate(preds_cum).astype(np.float64)
    val_mse = mse_num / max(mse_den, 1)
    cum_true = val_windows.y.sum(axis=1).astype(np.float64)
    next_day_ret = val_windows.y[:, 0].astype(np.float64)

    # --- Information Coefficient (pooled, no grouping) ---
    # pandas's .corr handles NaN-tolerant ranking; scipy not required.
    s_pred = pd.Series(cum_preds)
    s_true = pd.Series(cum_true)
    ic_spearman = s_pred.corr(s_true, method="spearman")
    ic_pearson = s_pred.corr(s_true, method="pearson")
    if not np.isfinite(ic_spearman):
        ic_spearman = 0.0
    if not np.isfinite(ic_pearson):
        ic_pearson = 0.0
    dir_acc_20d = float((np.sign(cum_preds) == np.sign(cum_true)).mean())
    frac_long = float((cum_preds > signal_threshold).mean())

    # --- Daily PnL construction ---
    df = pd.DataFrame({
        "date": pd.to_datetime(val_windows.dates),
        "ret": next_day_ret,
    })
    daily_all = df.groupby("date")["ret"].mean()
    # After double-centering, daily_all[d] is exact-zero up to float
    # rounding (~1e-10). In that regime both mu and sigma are dominated
    # by rounding noise and their ratio becomes a meaningless O(1)
    # number, so clamp to 0 when both are that small.
    mu_all = float(daily_all.mean())
    sigma_all = float(daily_all.std(ddof=1))
    if (np.isfinite(sigma_all) and sigma_all > 1e-8
            and float(np.abs(daily_all).max()) > 1e-8):
        always_long_sharpe = mu_all / sigma_all * float(np.sqrt(252.0))
    else:
        always_long_sharpe = 0.0
    n_days = int(daily_all.shape[0])

    # --- Selective long (pred > threshold) ---
    signals = cum_preds > signal_threshold
    if signals.any():
        df_sel = df.loc[signals]
        daily_long = df_sel.groupby("date")["ret"].mean()
        all_dates = pd.Index(sorted(df["date"].unique()))
        daily = daily_long.reindex(all_dates, fill_value=0.0)
        sigma_sel = float(daily.std(ddof=1))
        if np.isfinite(sigma_sel) and sigma_sel > 1e-12:
            sharpe = float(daily.mean()) / sigma_sel * float(np.sqrt(252.0))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    return {
        "sharpe": float(sharpe),
        "always_long_sharpe": float(always_long_sharpe),
        "val_mse": float(val_mse),
        "ic_spearman": float(ic_spearman),
        "ic_pearson": float(ic_pearson),
        "dir_acc_20d": float(dir_acc_20d),
        "frac_long": float(frac_long),
        "n_days": n_days,
    }


def evaluate_sharpe(model: torch.nn.Module,
                    val_windows: Windows,
                    batch_size: int = 1024,
                    signal_threshold: float = 0.0,
                    device: str = "cpu") -> tuple[float, float]:
    """Backward-compat wrapper that returns only ``(sharpe, val_mse)``.

    New code should prefer :func:`evaluate_all_metrics` which returns a
    dict containing IC, directional accuracy, and the always-long
    baseline Sharpe alongside the legacy numbers.
    """
    m = evaluate_all_metrics(model, val_windows, batch_size,
                              signal_threshold, device)
    return m["sharpe"], m["val_mse"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Prepare stock_future training data")
    p.add_argument("--smoke", action="store_true",
                   help="Use only 3 tickers for a quick end-to-end check")
    p.add_argument("--force", action="store_true",
                   help="Rebuild the prepared bundle cache")
    args = p.parse_args()
    bundle = prepare_all(smoke=args.smoke, force=args.force)
    tr, va, te = bundle["train"], bundle["val"], bundle["test"]
    print("--- prepared summary ---")
    print(f"feature_cols : {bundle['feature_cols']}")
    print(f"lookback     : {bundle['lookback']}")
    print(f"horizon      : {bundle['horizon']}")
    print(f"train.X shape: {tr.X.shape}  y shape: {tr.y.shape}")
    print(f"val.X   shape: {va.X.shape}  y shape: {va.y.shape}")
    print(f"test.X  shape: {te.X.shape}  y shape: {te.y.shape}")
    if len(tr) > 0:
        print(f"train date range: {tr.dates.min()}  ..  {tr.dates.max()}")
    if len(va) > 0:
        print(f"val   date range: {va.dates.min()}  ..  {va.dates.max()}")
    if len(te) > 0:
        print(f"test  date range: {te.dates.min()}  ..  {te.dates.max()}")


if __name__ == "__main__":
    _cli()
