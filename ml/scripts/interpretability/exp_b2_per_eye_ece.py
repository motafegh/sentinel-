"""
exp_b2_per_eye_ece.py — B2: Per-Eye Calibration (GNN / Transformer / Fused)

PURPOSE
───────
Compute Expected Calibration Error (ECE) for EACH of the three classifier
eyes separately:
  - GNN eye:         logits from gnn_head(gnn_embed)
  - Transformer eye: logits from tf_head(tf_embed)
  - Fused eye:       logits from fused_head(fused_embed)
  - Main head:       combined Three-Eye logits (reference)

EXP-L7 measured mean ECE=0.252 across all 10 classes for the main head only.
This experiment reveals WHICH eye is miscalibrated, so temperature scaling
can be applied at the right level.

Key questions answered:
- Is the GNN eye more/less calibrated than the Transformer eye?
- Which eye drives the overall miscalibration?
- Does per-class ECE differ between eyes (e.g. Reentrancy worse in TF eye)?

APPROACH
─────────
1. Run collect_predictions() with return_aux=True → gets logits for all 4 heads.
2. For each head, convert logits to probabilities (sigmoid).
3. Compute ECE per class using 15-bin equal-width binning.
4. Report per-class ECE per eye as a table and heatmap.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_b2_per_eye_ece.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --cache ml/data/cached_dataset_v9.pkl \\
        --label-csv ml/data/processed/multilabel_index.csv \\
        --splits-dir ml/data/splits/v9_deduped \\
        --out ml/logs/interpretability/b2_per_eye_ece.json

OUTPUT
──────
    - Per-eye per-class ECE table (stdout)
    - Reliability diagram PNG per eye
    - ECE heatmap PNG: class × eye
    - JSON report at --out

EXIT CODES
──────────
    0  completed
    1  fatal error
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

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    load_model,
    load_val_split,
    add_common_args,
    collect_predictions,
    CLASS_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

N_BINS = 15


# ── ECE computation ───────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Compute Expected Calibration Error for binary predictions.

    Args:
        probs:  [N] float array, predicted probabilities in [0,1]
        labels: [N] int array, binary ground truth
        n_bins: number of equal-width bins

    Returns:
        ECE as a scalar float.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = labels[mask].astype(float).mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def compute_all_ece(preds: dict) -> dict:
    """
    Compute ECE for each eye × class.

    Args:
        preds: output of collect_predictions() — has keys
               "logits", "gnn", "transformer", "fused", "labels"

    Returns:
        {eye_name: {class_name: ece_float}}
    """
    eyes = {
        "gnn":         preds.get("gnn"),
        "transformer": preds.get("transformer"),
        "fused":       preds.get("fused"),
        "main":        preds["logits"],
    }
    labels = preds["labels"]  # [N, 10]
    result: dict = {}

    for eye_name, logits in eyes.items():
        if logits is None:
            log.warning(f"Eye '{eye_name}' not available — skipping")
            continue
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid [N, 10]
        result[eye_name] = {}
        for ci, cls in enumerate(CLASS_NAMES):
            ece = compute_ece(probs[:, ci], labels[:, ci])
            result[eye_name][cls] = round(ece, 4)

    return result


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_table(ece_results: dict) -> None:
    eye_names = list(ece_results.keys())
    print(f"\n{'═'*80}")
    print("  B2: Per-Eye ECE (Expected Calibration Error, lower is better)")
    print(f"{'═'*80}")
    header = f"  {'Class':26s}" + "".join(f"  {e:>12s}" for e in eye_names)
    print(header)
    print(f"  {'-'*78}")
    for cls in CLASS_NAMES:
        row = f"  {cls:26s}"
        for eye in eye_names:
            ece = ece_results.get(eye, {}).get(cls, 0.0)
            row += f"  {ece:>12.4f}"
        print(row)

    print(f"\n  {'Mean':26s}", end="")
    for eye in eye_names:
        vals = [ece_results[eye].get(cls, 0.0) for cls in CLASS_NAMES]
        print(f"  {np.mean(vals):>12.4f}", end="")
    print(f"\n{'═'*80}\n")


def _save_heatmap(ece_results: dict, out_dir: Path) -> None:
    eye_names = list(ece_results.keys())
    matrix = np.array([
        [ece_results[eye].get(cls, 0.0) for eye in eye_names]
        for cls in CLASS_NAMES
    ])
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=0.35)
    plt.colorbar(im, ax=ax, label="ECE (lower = better calibrated)")
    ax.set_xticks(range(len(eye_names)))
    ax.set_xticklabels([e.capitalize() + " eye" for e in eye_names], fontsize=10)
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    for i, cls in enumerate(CLASS_NAMES):
        for j, eye in enumerate(eye_names):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if val > 0.20 else "black")
    ax.set_title("B2: Per-Eye ECE by Class\n(ECE < 0.05 = well calibrated; > 0.15 = poor)")
    plt.tight_layout()
    out_path = out_dir / "b2_per_eye_ece.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"ECE heatmap saved: {out_path}")


def _save_reliability_diagrams(preds: dict, out_dir: Path) -> None:
    eyes = {
        "gnn":         preds.get("gnn"),
        "transformer": preds.get("transformer"),
        "fused":       preds.get("fused"),
        "main":        preds["logits"],
    }
    labels = preds["labels"]

    for eye_name, logits in eyes.items():
        if logits is None:
            continue
        probs = 1.0 / (1.0 + np.exp(-logits))  # [N, 10]
        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        axes = axes.flatten()
        bins = np.linspace(0, 1, N_BINS + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        for ci, (cls, ax) in enumerate(zip(CLASS_NAMES, axes)):
            p = probs[:, ci]
            y = labels[:, ci].astype(float)
            bin_acc = []
            bin_conf = []
            bin_counts = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (p >= lo) & (p < hi)
                if mask.sum() > 0:
                    bin_acc.append(y[mask].mean())
                    bin_conf.append(p[mask].mean())
                    bin_counts.append(mask.sum())
                else:
                    bin_acc.append(np.nan)
                    bin_conf.append(bin_centers[len(bin_acc) - 1])
                    bin_counts.append(0)

            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
            ax.scatter(bin_conf, bin_acc, s=30, zorder=3, label="Actual")
            ax.bar(bin_conf, np.array(bin_counts) / max(sum(bin_counts), 1),
                   width=1.0 / N_BINS, alpha=0.2, color="blue", label="Density")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(cls, fontsize=8)
            ax.set_xlabel("Confidence", fontsize=7)
            ax.set_ylabel("Accuracy", fontsize=7)
            ax.tick_params(labelsize=6)

        fig.suptitle(f"B2: Reliability Diagram — {eye_name} eye", fontsize=11)
        plt.tight_layout()
        out_path = out_dir / f"b2_reliability_{eye_name}.png"
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close()
        log.info(f"Reliability diagram saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B2: Per-Eye ECE Calibration Analysis")
    add_common_args(p)
    p.add_argument("--max-samples", type=int, default=None,
                   dest="max_samples",
                   help="Subsample val split for faster run (default: all)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out) if args.out else Path("ml/logs/interpretability/b2_per_eye_ece.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stems, df_split, cache = load_val_split(
            cache_path = Path(args.cache),
            label_csv  = Path(args.label_csv),
            splits_dir = Path(args.splits_dir),
            split      = args.split,
        )
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1

    try:
        model = load_model(checkpoint_path=Path(args.checkpoint), device=args.device)
    except Exception as exc:
        log.error(f"Model load failed: {exc}")
        return 1

    log.info("Collecting per-eye predictions...")
    preds = collect_predictions(
        model=model, stems=stems, df_split=df_split, cache=cache,
        device=args.device, return_aux=True, max_samples=args.max_samples,
    )

    ece_results = compute_all_ece(preds)
    _print_table(ece_results)
    _save_heatmap(ece_results, out_path.parent)
    _save_reliability_diagrams(preds, out_path.parent)

    report = {
        "experiment": "exp_b2_per_eye_ece",
        "checkpoint": str(args.checkpoint),
        "n_samples":  len(preds["stems"]),
        "n_bins":     N_BINS,
        "ece_by_eye": ece_results,
        "mean_ece": {
            eye: round(float(np.mean(list(cls_ece.values()))), 4)
            for eye, cls_ece in ece_results.items()
        },
    }
    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON report saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
