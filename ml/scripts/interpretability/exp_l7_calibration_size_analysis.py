"""
exp_l7_calibration_size_analysis.py — Layer 3, P2: Per-Class Calibration and
Decision Boundary Analysis

PURPOSE
───────
Two complementary analyses of model reliability:

1. Calibration: Does the model's predicted probability correspond to actual
   frequency? A perfectly calibrated model predicts 0.7 for a class when that
   class is positive 70% of the time among contracts assigned that probability.
   Poor calibration means the decision threshold needs tuning and confidence
   estimates are misleading.

2. Size-Stratified F1: Does model performance degrade on large contracts?
   Large contracts (>150 nodes) may exceed the GNN's effective receptive field
   or overflow the BERT token window, creating a size-dependent accuracy gap
   that must be understood before deployment.

LAYER / PRIORITY
─────────────────
Layer 3, Priority 2 — Model reliability and deployment safety.

CALIBRATION METHOD
───────────────────
Sigmoid output in [0,1] is binned into 10 equal-width bins [0.0, 0.1), ...,
[0.9, 1.0]. Per bin: mean predicted probability and fraction of true positives.
ECE = sum_bin (|bin| / N) * |mean_pred - frac_positive|.

SIZE STRATIFICATION
────────────────────
Small:  total_nodes < 30
Medium: 30 <= total_nodes <= 150
Large:  total_nodes > 150

PASS CRITERIA
─────────────
For each class with >= 20 large-contract examples: F1 on large contracts is
within 10 percentage points (0.10) of F1 on small contracts.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_l7_calibration_size_analysis.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --cache ml/data/cached_dataset_v10.pkl \\
        --label-csv ml/data/processed/multilabel_index.csv \\
        --splits-dir ml/data/splits/v9_deduped \\
        --n-contracts 2000 \\
        --out ml/logs/interpretability/l7_calibration_size

OUTPUT
──────
ml/logs/interpretability/l7_calibration_size/
  calibration_curves.png   — 10-panel calibration curves (one per class)
  size_stratified_f1.png   — bar chart: F1 by class × size stratum
  l7_results.json          — ECE, size-stratified F1, pass/fail per class

EXIT CODES
──────────
    0  all classes with sufficient data pass the size-gap criterion
    1  one or more classes fail (or load/inference error)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    load_model,
    load_val_split,
    collect_predictions,
    add_common_args,
    CLASS_NAMES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Size strata ───────────────────────────────────────────────────────────────

SIZE_STRATA = {
    "small":  (0, 30),       # total_nodes < 30
    "medium": (30, 151),     # 30 <= nodes <= 150
    "large":  (151, 10**7),  # nodes > 150
}

_N_BINS = 10
_MIN_BIN_SAMPLES = 3       # bins with fewer samples are skipped for ECE
_MIN_STRATUM_SAMPLES = 20  # strata with fewer samples are excluded from pass/fail


# ── Calibration ───────────────────────────────────────────────────────────────

def compute_calibration(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = _N_BINS,
) -> dict:
    """
    Compute calibration curve and ECE for a single class.

    Args:
        probs:  [N] float — predicted probabilities for this class.
        labels: [N] int   — binary ground-truth labels.
        n_bins: number of equal-width bins.

    Returns:
        dict with keys:
            bin_edges:      list of (lo, hi) tuples
            bin_mean_pred:  list of mean predicted prob per bin (None if empty)
            bin_frac_pos:   list of fraction positive per bin (None if empty)
            bin_counts:     list of sample counts per bin
            ece:            float
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mean_pred = []
    bin_frac_pos  = []
    bin_counts    = []
    bin_edges     = []
    ece_accum     = 0.0
    n_total       = len(probs)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Include upper edge in last bin
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        count = int(mask.sum())
        bin_counts.append(count)
        bin_edges.append((round(float(lo), 2), round(float(hi), 2)))

        if count < _MIN_BIN_SAMPLES:
            bin_mean_pred.append(None)
            bin_frac_pos.append(None)
        else:
            mp = float(probs[mask].mean())
            fp = float(labels[mask].mean())
            bin_mean_pred.append(round(mp, 4))
            bin_frac_pos.append(round(fp, 4))
            ece_accum += (count / n_total) * abs(mp - fp)

    return {
        "bin_edges":     bin_edges,
        "bin_mean_pred": bin_mean_pred,
        "bin_frac_pos":  bin_frac_pos,
        "bin_counts":    bin_counts,
        "ece":           round(float(ece_accum), 4),
    }


# ── Size-stratified F1 ────────────────────────────────────────────────────────

def f1_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """Binary F1 at a fixed threshold."""
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return float("nan")
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return round(2 * prec * rec / (prec + rec), 4)


def compute_size_stratified_f1(
    probs_all: np.ndarray,
    labels_all: np.ndarray,
    node_counts: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute per-class F1 for each size stratum.

    Args:
        probs_all:   [N, 10] predicted probabilities.
        labels_all:  [N, 10] binary labels.
        node_counts: [N] int  total node count per contract.
        threshold:   decision threshold.

    Returns:
        dict mapping class_name -> {stratum: {"f1": float, "n": int}}
    """
    result = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_result = {}
        for stratum, (lo, hi) in SIZE_STRATA.items():
            mask = (node_counts >= lo) & (node_counts < hi)
            n = int(mask.sum())
            if n == 0:
                cls_result[stratum] = {"f1": None, "n": 0}
            else:
                f1 = f1_at_threshold(
                    probs_all[mask, cls_idx],
                    labels_all[mask, cls_idx],
                    threshold,
                )
                cls_result[stratum] = {"f1": f1, "n": n}
        result[cls_name] = cls_result
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_calibration_curves(
    calibration_data: dict,
    output_path: Path,
) -> None:
    """
    Plot one calibration subplot per class (10 subplots, 2 rows × 5 cols).
    """
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("Per-Class Calibration Curves (SENTINEL GNN+TF)", fontsize=13)

    bin_centers = [(lo + hi) / 2 for lo, hi in calibration_data[CLASS_NAMES[0]]["bin_edges"]]

    for idx, cls_name in enumerate(CLASS_NAMES):
        ax = axes[idx // 5][idx % 5]
        cal = calibration_data[cls_name]

        xs, ys = [], []
        for mp, fp in zip(cal["bin_mean_pred"], cal["bin_frac_pos"]):
            if mp is not None and fp is not None:
                xs.append(mp)
                ys.append(fp)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="perfect")
        if xs:
            ax.plot(xs, ys, "o-", color="steelblue", markersize=5, linewidth=1.5,
                    label="model")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{cls_name}\nECE={cal['ece']:.3f}", fontsize=9)
        ax.set_xlabel("Mean predicted prob", fontsize=7)
        ax.set_ylabel("Fraction positive", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Calibration curves saved: {output_path}")


def plot_size_stratified_f1(
    size_f1: dict,
    output_path: Path,
) -> None:
    """
    Grouped bar chart: per-class F1 broken down by size stratum.
    """
    strata_list  = list(SIZE_STRATA.keys())
    n_classes    = len(CLASS_NAMES)
    bar_width    = 0.22
    x            = np.arange(n_classes)
    colors       = {"small": "#4C72B0", "medium": "#DD8452", "large": "#55A868"}

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, stratum in enumerate(strata_list):
        vals = []
        for cls_name in CLASS_NAMES:
            v = size_f1[cls_name][stratum]["f1"]
            vals.append(v if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0)
        offset = (i - 1) * bar_width
        ax.bar(x + offset, vals, bar_width, label=stratum, color=colors[stratum], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Size-Stratified F1 per Class (small=<30 nodes, medium=30-150, large=>150)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Size-stratified F1 chart saved: {output_path}")


# ── Pass/fail evaluation ──────────────────────────────────────────────────────

def evaluate_pass_fail(size_f1: dict) -> tuple[list[dict], bool]:
    """
    For each class with >= _MIN_STRATUM_SAMPLES large examples: check that
    F1_large is within 0.10 of F1_small.

    Returns:
        (per_class_results, overall_pass)
    """
    per_class = []
    overall   = True

    for cls_name in CLASS_NAMES:
        strata = size_f1[cls_name]
        n_large = strata["large"]["n"]
        n_small = strata["small"]["n"]

        if n_large < _MIN_STRATUM_SAMPLES or n_small < _MIN_STRATUM_SAMPLES:
            per_class.append({
                "class": cls_name,
                "evaluated": False,
                "reason": f"insufficient large ({n_large}) or small ({n_small}) samples",
                "pass": None,
            })
            continue

        f1_large = strata["large"]["f1"]
        f1_small = strata["small"]["f1"]

        if f1_large is None or f1_small is None or np.isnan(f1_large) or np.isnan(f1_small):
            per_class.append({
                "class": cls_name,
                "evaluated": False,
                "reason": "F1 undefined (no positive samples in stratum)",
                "pass": None,
            })
            continue

        gap  = abs(f1_large - f1_small)
        ok   = bool(gap <= 0.10)
        if not ok:
            overall = False

        per_class.append({
            "class":     cls_name,
            "evaluated": True,
            "f1_small":  f1_small,
            "f1_large":  f1_large,
            "gap":       round(float(gap), 4),
            "pass":      ok,
        })

    return per_class, overall


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-class calibration and size-stratified F1 — Layer 3, P2"
    )
    add_common_args(parser, require_checkpoint=True)
    parser.set_defaults(n_contracts=2000)
    args = parser.parse_args()

    # Output directory
    out_dir: Optional[Path] = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        model = load_model(
            Path(args.checkpoint),
            device=args.device,
            phase2_edge_types=args.phase2_edge_types,
        )
    except Exception as exc:
        log.error(f"Failed to load model: {exc}")
        return 1

    # Load data
    try:
        stems, df_split, cache = load_val_split(
            Path(args.cache),
            Path(args.label_csv),
            Path(args.splits_dir),
            split=args.split,
        )
    except Exception as exc:
        log.error(f"Failed to load data: {exc}")
        return 1

    # Collect predictions (full pipeline)
    try:
        preds = collect_predictions(
            model,
            stems,
            df_split,
            cache,
            device=args.device,
            return_aux=False,
            max_samples=args.n_contracts,
            seed=args.seed,
        )
    except Exception as exc:
        log.error(f"Inference failed: {exc}")
        return 1

    probs_all  = 1.0 / (1.0 + np.exp(-preds["logits"]))   # sigmoid [N, 10]
    labels_all = preds["labels"].astype(int)               # [N, 10]
    valid_stems = preds["stems"]

    # Extract node counts from cache
    node_counts = np.array([
        cache[s][0].num_nodes if s in cache and isinstance(cache[s], tuple)
        else 0
        for s in valid_stems
    ], dtype=np.int32)

    log.info(
        f"Node count stats: min={node_counts.min()} median={np.median(node_counts):.0f} "
        f"max={node_counts.max()} mean={node_counts.mean():.1f}"
    )

    # ── Calibration ────────────────────────────────────────────────────────────
    log.info("Computing calibration curves...")
    calibration_data = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        calibration_data[cls_name] = compute_calibration(
            probs_all[:, cls_idx],
            labels_all[:, cls_idx],
        )
        log.info(f"  {cls_name}: ECE={calibration_data[cls_name]['ece']:.4f}")

    # ── Size-stratified F1 ─────────────────────────────────────────────────────
    log.info("Computing size-stratified F1...")
    size_f1 = compute_size_stratified_f1(probs_all, labels_all, node_counts)

    for cls_name in CLASS_NAMES:
        row = size_f1[cls_name]
        log.info(
            f"  {cls_name}: "
            f"small(n={row['small']['n']}) F1={row['small']['f1']}  "
            f"medium(n={row['medium']['n']}) F1={row['medium']['f1']}  "
            f"large(n={row['large']['n']}) F1={row['large']['f1']}"
        )

    # ── Pass/fail ──────────────────────────────────────────────────────────────
    per_class_pass, overall_pass = evaluate_pass_fail(size_f1)
    n_eval = sum(1 for r in per_class_pass if r["evaluated"])
    n_pass = sum(1 for r in per_class_pass if r.get("pass") is True)
    log.info(f"Pass/fail: {n_pass}/{n_eval} evaluated classes pass size-gap criterion")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if out_dir:
        plot_calibration_curves(calibration_data, out_dir / "calibration_curves.png")
        plot_size_stratified_f1(size_f1, out_dir / "size_stratified_f1.png")

    # ── JSON output ────────────────────────────────────────────────────────────
    report = {
        "experiment":    "exp_l7_calibration_size_analysis",
        "layer":         3,
        "priority":      2,
        "n_samples":     len(valid_stems),
        "split":         args.split,
        "pass_criteria": "F1_large within 0.10 of F1_small per class (min 20 samples each)",
        "overall_pass":  overall_pass,
        "per_class_size_pass": per_class_pass,
        "ece_per_class": {cls_name: calibration_data[cls_name]["ece"]
                          for cls_name in CLASS_NAMES},
        "calibration":   calibration_data,
        "size_f1":       size_f1,
        "node_count_stats": {
            "min":    int(node_counts.min()),
            "max":    int(node_counts.max()),
            "mean":   round(float(node_counts.mean()), 1),
            "median": float(np.median(node_counts)),
        },
    }

    if out_dir:
        json_path = out_dir / "l7_results.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Results written to: {json_path}")
    else:
        print(json.dumps(report, indent=2))

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
