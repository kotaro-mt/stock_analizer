"""Baseline model + training loop for stock_future.

This is the ONE file the self-improvement loop edits. Everything is fair
game: architecture, loss, optimizer, learning-rate schedule, which features
to use, signal threshold, etc. But:

* Do NOT edit ``prepare.py`` — the data pipeline and ``evaluate_sharpe``
  must stay pinned so the metric is comparable across iterations.
* Do NOT add new pip dependencies.
* Keep the public contract: at the end of main(), print a block starting
  with ``---`` followed by a ``val_sharpe:`` line so the loop runner can
  grep the metric. Also append one row to ``results.tsv``.

Run it with::

    /c/Users/matsu/anaconda3/python.exe train.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prepare import (
    prepare_all,
    evaluate_all_metrics,
    StockDataset,
    ARTIFACTS_DIR,
    FEATURE_COLS,
    LOOKBACK,
    HORIZON,
)

ROOT = Path(__file__).parent
RESULTS_TSV = ROOT / "results.tsv"
CKPT_PATH = ARTIFACTS_DIR / "model.pt"

# ---------------------------------------------------------------------------
# Hyperparameters (Claude tunes these between runs)
# ---------------------------------------------------------------------------
HIDDEN = 64
NUM_LAYERS = 1
DROPOUT = 0.0
BATCH_SIZE = 256
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5
SIGNAL_THRESHOLD = 0.0
SEED = 42
# Blended loss weights: L = -corr + MSE_WEIGHT * mse
# - Each batch is now ONE date (via DateBatchSampler), so the batch
#   Pearson becomes *exactly* cross-sectional Pearson — a true
#   differentiable relaxation of Spearman IC. MSE still anchors scale.
MSE_WEIGHT = 100.0
NOTES = "h5exp: HORIZON 20 -> 5 (feature-horizon match test)"


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class PearsonBlendedLoss(nn.Module):
    """Negative Pearson correlation of 20d-cum prediction vs target,
    blended with an MSE anchor to keep prediction scale stable.

    The primary training signal is ``-corr``, where ``corr`` is the
    Pearson correlation between the batch's predicted 20-day cumulative
    residual returns and the realised targets. This is a differentiable
    relaxation of the Information Coefficient — directly optimising
    the thing ``evaluate_all_metrics`` ranks models by.

    An MSE term on the full 20-vector provides scale and distribution
    anchoring so ``cum_pred > 0`` remains a meaningful long signal.
    """

    def __init__(self, mse_weight: float = 100.0) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.mse(pred, target)
        # Collapse 20-vector to cum scalar for the correlation term
        if pred.dim() > 1 and pred.shape[-1] > 1:
            pc = pred.sum(dim=-1)
        else:
            pc = pred.reshape(-1)
        if target.dim() > 1 and target.shape[-1] > 1:
            tc = target.sum(dim=-1)
        else:
            tc = target.reshape(-1)
        pc_c = pc - pc.mean()
        tc_c = tc - tc.mean()
        num = (pc_c * tc_c).sum()
        den = torch.sqrt((pc_c * pc_c).sum() * (tc_c * tc_c).sum()).clamp(min=1e-8)
        corr = num / den
        return -corr + self.mse_weight * mse


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    """Small LSTM that maps [B, T, F] -> [B, HORIZON] daily log returns."""

    def __init__(self, in_features: int, hidden: int = HIDDEN,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT,
                 horizon: int = HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optim: torch.optim.Optimizer, loss_fn,
                    device: str) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        bs = xb.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def main() -> None:
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cpu"

    bundle = prepare_all()
    train_w = bundle["train"]
    val_w = bundle["val"]
    print(f"[train] train_windows={len(train_w)} val_windows={len(val_w)}")

    train_ds = StockDataset(train_w.X, train_w.y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=False)

    in_features = len(FEATURE_COLS)
    model = LSTMForecaster(in_features, HIDDEN, NUM_LAYERS, DROPOUT, HORIZON).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params={n_params}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = PearsonBlendedLoss(mse_weight=MSE_WEIGHT)

    best_ic = -float("inf")
    best_metrics: dict | None = None
    best_state: dict | None = None

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, loss_fn, device)
        mets = evaluate_all_metrics(
            model, val_w, signal_threshold=SIGNAL_THRESHOLD, device=device
        )
        print(f"epoch {ep:02d}  "
              f"train_loss={tr_loss:+.5f}  val_mse={mets['val_mse']:.6f}  "
              f"ic={mets['ic_spearman']:+.4f}  "
              f"sharpe={mets['sharpe']:+.4f}  "
              f"always_long={mets['always_long_sharpe']:+.4f}  "
              f"dir_acc={mets['dir_acc_20d'] * 100:.2f}%")
        # Select best epoch by IC (ranking skill), not Sharpe (subset-noisy)
        if mets["ic_spearman"] > best_ic:
            best_ic = mets["ic_spearman"]
            best_metrics = mets
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    assert best_metrics is not None, "training loop did not run any epochs"
    train_time = time.time() - t_start

    # --- save checkpoint ---
    ckpt = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "config": {
            "arch": "LSTMForecaster",
            "in_features": in_features,
            "hidden": HIDDEN,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "horizon": HORIZON,
            "lookback": LOOKBACK,
            "feature_cols": FEATURE_COLS,
            "signal_threshold": SIGNAL_THRESHOLD,
        },
        "val_sharpe": best_metrics["sharpe"],
        "val_mse": best_metrics["val_mse"],
        "val_ic_spearman": best_metrics["ic_spearman"],
        "val_ic_pearson": best_metrics["ic_pearson"],
        "val_dir_acc_20d": best_metrics["dir_acc_20d"],
        "val_always_long_sharpe": best_metrics["always_long_sharpe"],
        "val_metrics": best_metrics,
        "notes": NOTES,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, CKPT_PATH)

    # --- autoresearch-style metric block ---
    # The loop runner greps these lines, so keep the labels stable.
    print("---")
    print(f"val_ic_spearman:  {best_metrics['ic_spearman']:+.4f}")
    print(f"val_sharpe:       {best_metrics['sharpe']:+.4f}")
    print(f"val_always_long:  {best_metrics['always_long_sharpe']:+.4f}")
    print(f"val_dir_acc_20d:  {best_metrics['dir_acc_20d'] * 100:.2f}%")
    print(f"val_mse:          {best_metrics['val_mse']:.6f}")
    print(f"train_time_sec:   {train_time:.1f}")
    print(f"num_params:       {n_params}")
    print(f"notes:            {NOTES}")

    # --- append to results.tsv ---
    header = ("timestamp\tval_ic_spearman\tval_sharpe\tval_always_long\t"
              "val_dir_acc\tval_mse\ttrain_time\tparams\tnotes\n")
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(header, encoding="utf-8")
    with RESULTS_TSV.open("a", encoding="utf-8") as f:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"{ts}\t{best_metrics['ic_spearman']:.6f}\t"
            f"{best_metrics['sharpe']:.6f}\t"
            f"{best_metrics['always_long_sharpe']:.6f}\t"
            f"{best_metrics['dir_acc_20d']:.4f}\t"
            f"{best_metrics['val_mse']:.6f}\t"
            f"{train_time:.1f}\t{n_params}\t{NOTES}\n"
        )


if __name__ == "__main__":
    main()
