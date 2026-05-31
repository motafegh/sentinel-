"""
calibrate_temperature.py — Per-class temperature scaling for SENTINEL (Interp-1)

WHAT THIS DOES
──────────────
Fits one scalar temperature T_c per class (10 classes) by minimising BCE NLL on
the validation set.  Calibrated logit for class c = logit_c / T_c.

WHY PER-CLASS
─────────────
Run 4 ECE ranges from 0.205 to 0.310 across classes.  A single global
temperature cannot correct per-class miscalibration simultaneously.

USAGE
─────
    source ml/.venv/bin/activate
    python -m ml.scripts.calibrate_temperature \
        --checkpoint ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt \
        --out ml/calibration/temperatures_run4.json

OUTPUT
──────
    JSON file:  {class_name: float, ...}  — one T per class
    PNG:        ECE before/after per class bar chart
    JSON stats: ECE before/after + confidence improvement
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    add_common_args,
    load_model,
    load_val_split,
    collect_predictions,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

N_BINS = 15


# ── ECE computation ────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """Equal-width ECE for a single class."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc  = labels[mask].mean()
        ece += mask.mean() * abs(conf - acc)
    return float(ece)


def compute_all_ece(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Returns ECE array [NUM_CLASSES]."""
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    return np.array([compute_ece(probs[:, c], labels[:, c]) for c in range(NUM_CLASSES)])


# ── Temperature scaling ────────────────────────────────────────────────────────

class PerClassTemperature(nn.Module):
    """Learnable per-class temperature scaling — one scalar T_c per class."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # log-temperature initialised to 0 → T=1 (no scaling)
        self.log_T = nn.Parameter(torch.zeros(num_classes))

    @property
    def temperatures(self) -> torch.Tensor:
        return self.log_T.exp().clamp(min=0.05, max=20.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperatures


def fit_temperatures(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.05,
) -> np.ndarray:
    """
    Fit per-class temperatures via LBFGS minimising per-sample BCE NLL.

    Returns:
        temperatures: np.ndarray [NUM_CLASSES]
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    model = PerClassTemperature(NUM_CLASSES)
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    bce = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled = model(logits_t)
        loss = bce(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    temps = model.temperatures.detach().numpy()
    return temps


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_ece_comparison(ece_before: np.ndarray, ece_after: np.ndarray, out_path: Path) -> None:
    x = np.arange(NUM_CLASSES)
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    bars_b = ax.bar(x - width / 2, ece_before, width, label="Before", color="#e07070", alpha=0.85)
    bars_a = ax.bar(x + width / 2, ece_after,  width, label="After",  color="#70aae0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("ECE (lower = better)")
    ax.set_title("Per-class ECE before/after temperature scaling")
    ax.legend()
    ax.axhline(0.05, color="green", linestyle="--", linewidth=1, label="target 0.05")
    # Annotate bars with values
    for bar in bars_a:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved ECE plot → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class temperature scaling calibration")
    add_common_args(p)
    # Override --out default (add_common_args sets it to None)
    p.set_defaults(out="ml/calibration/temperatures.json")
    p.add_argument("--max-iter", type=int, default=200,
                   help="LBFGS max iterations (default 200)")
    p.add_argument("--lr", type=float, default=0.05,
                   help="LBFGS learning rate (default 0.05)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    log.info(f"Device: {device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    log.info("Loading val split...")
    stems, df_split, cache = load_val_split(args.cache, args.label_csv, args.splits_dir)
    log.info(f"Val split: {len(stems)} contracts")

    log.info("Collecting predictions (full val set)...")
    preds = collect_predictions(
        model=model,
        stems=stems,
        df_split=df_split,
        cache=cache,
        device=device,
        return_aux=False,  # main head logits only for calibration
    )

    logits: np.ndarray = preds["logits"]   # [N, 10]
    labels: np.ndarray = preds["labels"]   # [N, 10] int

    log.info(f"Collected {len(logits)} predictions")

    # ECE before calibration
    ece_before = compute_all_ece(logits, labels)
    log.info("ECE before calibration:")
    for c, name in enumerate(CLASS_NAMES):
        log.info(f"  {name:30s} {ece_before[c]:.4f}")
    log.info(f"  Mean ECE: {ece_before.mean():.4f}")

    # Fit temperatures
    log.info(f"Fitting per-class temperatures (LBFGS max_iter={args.max_iter})...")
    temperatures = fit_temperatures(logits, labels, max_iter=args.max_iter, lr=args.lr)

    log.info("Fitted temperatures:")
    for c, name in enumerate(CLASS_NAMES):
        log.info(f"  {name:30s} T={temperatures[c]:.4f}")

    # ECE after calibration
    scaled_logits = logits / temperatures[np.newaxis, :]
    ece_after = compute_all_ece(scaled_logits, labels)
    log.info("ECE after calibration:")
    for c, name in enumerate(CLASS_NAMES):
        delta = ece_after[c] - ece_before[c]
        log.info(f"  {name:30s} {ece_after[c]:.4f}  (Δ={delta:+.4f})")
    log.info(f"  Mean ECE after: {ece_after.mean():.4f}  (Δ={ece_after.mean()-ece_before.mean():+.4f})")

    # Save temperatures JSON
    temps_dict = {name: float(temperatures[c]) for c, name in enumerate(CLASS_NAMES)}
    with open(out_path, "w") as f:
        json.dump(temps_dict, f, indent=2)
    log.info(f"Temperatures saved → {out_path}")

    # Save stats JSON
    stats = {
        "ece_before": {name: float(ece_before[c]) for c, name in enumerate(CLASS_NAMES)},
        "ece_after":  {name: float(ece_after[c])  for c, name in enumerate(CLASS_NAMES)},
        "mean_ece_before": float(ece_before.mean()),
        "mean_ece_after":  float(ece_after.mean()),
        "temperatures":    temps_dict,
    }
    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats saved → {stats_path}")

    # Plot
    plot_path = out_path.with_name(out_path.stem + "_ece_comparison.png")
    _plot_ece_comparison(ece_before, ece_after, plot_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
