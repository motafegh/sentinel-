"""
exp_a4_aux_contribution.py — Cross-Attention Fusion Contribution Analysis

PURPOSE
───────
Layer 3, P0 experiment.  Compares the four prediction heads in SentinelModel
(main logits, GNN auxiliary, Transformer auxiliary, Fused auxiliary) on both
F1 and AUC-ROC across all 10 vulnerability classes.  Quantifies how much the
GNN eye contributes beyond a random baseline, validating that the GNN pathway
is learning useful representations and not just riding the Transformer.

LAYER
─────
Layer 3 — inference-time head comparison (no gradient, no retraining).

PASS CRITERIA
─────────────
GNN auxiliary head F1 ≥ (random baseline + 5 pp) for ≥5 of 10 classes.
Random baseline F1 for class c = 2*p*0.5 / (p + 0.5) where p = positive rate.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/interpretability/exp_a4_aux_contribution.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --out ml/interpretability_results/exp_a4 \\
        --n-contracts 1000

OUTPUT
──────
  <out>/exp_a4_aux_contribution.csv   — F1 and AUCROC per head × class
  <out>/exp_a4_aux_contribution.json  — same data + pass/fail flags
  <out>/exp_a4_f1_heatmap.png         — head × class F1 heatmap
  <out>/exp_a4_aucroc_heatmap.png     — head × class AUCROC heatmap
  stdout                              — tables + pass/fail
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    load_model,
    load_val_split,
    add_common_args,
    CLASS_NAMES,
    collect_predictions,
    plot_class_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEAD_NAMES = ["main", "gnn", "transformer", "fused"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _compute_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Compute per-class F1 score.

    Args:
        probs:     [N, C] sigmoid probabilities
        labels:    [N, C] binary labels
        threshold: classification threshold

    Returns:
        [C] F1 scores
    """
    preds  = (probs >= threshold).astype(np.int32)
    f1s    = np.zeros(probs.shape[1])
    for c in range(probs.shape[1]):
        tp = int(((preds[:, c] == 1) & (labels[:, c] == 1)).sum())
        fp = int(((preds[:, c] == 1) & (labels[:, c] == 0)).sum())
        fn = int(((preds[:, c] == 0) & (labels[:, c] == 1)).sum())
        denom = 2 * tp + fp + fn
        f1s[c] = (2 * tp / denom) if denom > 0 else 0.0
    return f1s


def _compute_aucroc(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute per-class AUC-ROC. Returns 0.5 (random) if a class has no positives.

    Args:
        probs:  [N, C] sigmoid probabilities
        labels: [N, C] binary labels

    Returns:
        [C] AUC-ROC scores
    """
    from sklearn.metrics import roc_auc_score
    aucs = np.full(probs.shape[1], 0.5)
    for c in range(probs.shape[1]):
        if labels[:, c].sum() == 0 or labels[:, c].sum() == labels.shape[0]:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                aucs[c] = float(roc_auc_score(labels[:, c], probs[:, c]))
            except ValueError:
                pass
    return aucs


def _random_baseline_f1(pos_rate: float) -> float:
    """Compute F1 of a random classifier at 0.5 threshold given positive rate p."""
    p = pos_rate
    if p == 0.0:
        return 0.0
    # precision = p (random at 0.5 threshold), recall = 0.5
    # F1 = 2*p*0.5 / (p + 0.5)
    return 2.0 * p * 0.5 / (p + 0.5)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Auxiliary head contribution analysis for SENTINEL.",
    )
    add_common_args(p, require_checkpoint=True)
    p.set_defaults(n_contracts=1000)
    return p


def run(args: argparse.Namespace) -> int:
    device  = args.device
    out_dir = Path(args.out) if args.out else Path("ml/interpretability_results/exp_a4")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────────
    model = load_model(
        Path(args.checkpoint),
        device=device,
        phase2_edge_types=args.phase2_edge_types,
    )
    model.eval()

    # ── Load data ───────────────────────────────────────────────────────────────
    stems, df_split, cache = load_val_split(
        cache_path=Path(args.cache),
        label_csv=Path(args.label_csv),
        splits_dir=Path(args.splits_dir),
        split=args.split,
    )

    # ── Collect predictions from all heads ──────────────────────────────────────
    log.info("Collecting predictions from all 4 heads (main + gnn + transformer + fused)...")
    preds = collect_predictions(
        model=model,
        stems=stems,
        df_split=df_split,
        cache=cache,
        device=device,
        return_aux=True,
        max_samples=args.n_contracts,
        seed=args.seed,
    )

    labels_arr = preds["labels"]  # [N, 10]
    N = labels_arr.shape[0]
    log.info(f"Predictions collected for {N:,} samples.")

    # Build probability arrays per head (sigmoid of logits)
    head_probs: dict[str, np.ndarray] = {
        "main":        _sigmoid(preds["logits"]),
    }
    if "gnn" in preds:
        head_probs["gnn"]         = _sigmoid(preds["gnn"])
        head_probs["transformer"] = _sigmoid(preds["transformer"])
        head_probs["fused"]       = _sigmoid(preds["fused"])
    else:
        log.warning("Auxiliary head outputs not found in predictions. "
                    "Only main head will be evaluated.")

    available_heads = list(head_probs.keys())

    # ── Compute F1 and AUC-ROC per head × class ─────────────────────────────────
    pos_rates = labels_arr.mean(axis=0)  # [10]
    baselines = np.array([_random_baseline_f1(p) for p in pos_rates])  # [10]

    f1_matrix   = np.zeros((len(available_heads), len(CLASS_NAMES)))  # [H, C]
    auc_matrix  = np.zeros((len(available_heads), len(CLASS_NAMES)))  # [H, C]

    for h_idx, head_name in enumerate(available_heads):
        probs = head_probs[head_name]
        f1_matrix[h_idx]  = _compute_f1(probs, labels_arr, threshold=0.5)
        auc_matrix[h_idx] = _compute_aucroc(probs, labels_arr)

    # ── Print F1 table ───────────────────────────────────────────────────────────
    def _print_table(matrix: np.ndarray, title: str, heads: list[str], fmt: str = ".3f") -> None:
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
        col_h  = "  ".join(f"{c[:8]:>8}" for c in CLASS_NAMES)
        print(f"{'Head':<14}  {col_h}")
        print("-" * 110)
        for h_idx, head_name in enumerate(heads):
            row = "  ".join(format(matrix[h_idx, c], fmt) for c in range(len(CLASS_NAMES)))
            print(f"{head_name:<14}  {row}")
        if title.startswith("F1"):
            print(f"{'Baseline':14}  {'  '.join(format(baselines[c], fmt) for c in range(len(CLASS_NAMES)))}")
        print("=" * 110)

    _print_table(f1_matrix,  "F1 SCORES by Head × Class (threshold=0.5)", available_heads)
    _print_table(auc_matrix, "AUC-ROC by Head × Class",                   available_heads)

    # ── GNN contribution analysis ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GNN HEAD CONTRIBUTION vs. RANDOM BASELINE")
    print("=" * 80)
    print(f"{'Class':<25}  {'Pos rate':>9}  {'Baseline F1':>11}  {'GNN F1':>8}  {'Delta':>8}  {'Status':>20}")
    print("-" * 85)

    gnn_useful_classes = []
    gnn_weak_classes   = []

    if "gnn" in available_heads:
        gnn_idx = available_heads.index("gnn")
        gnn_f1  = f1_matrix[gnn_idx]

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            pos_rate  = float(pos_rates[cls_idx])
            baseline  = float(baselines[cls_idx])
            gnn_score = float(gnn_f1[cls_idx])
            delta     = gnn_score - baseline
            useful    = delta >= 0.05
            status    = "GNN useful (≥+5pp)" if useful else "GNN contributing nothing"
            if useful:
                gnn_useful_classes.append(cls_name)
            else:
                gnn_weak_classes.append(cls_name)

            print(f"{cls_name:<25}  {pos_rate:>9.4f}  {baseline:>11.4f}  "
                  f"{gnn_score:>8.4f}  {delta:>+8.4f}  {status:>20}")

        n_useful   = len(gnn_useful_classes)
        overall    = "PASS" if n_useful >= 5 else "FAIL"

        print(f"\nGNN useful for {n_useful}/10 classes (F1 ≥ baseline+5pp) → {overall}")
        print(f"Useful classes:   {', '.join(gnn_useful_classes) if gnn_useful_classes else '(none)'}")
        print(f"Weak classes:     {', '.join(gnn_weak_classes) if gnn_weak_classes else '(none)'}")
    else:
        log.warning("GNN aux head not available; skipping GNN contribution analysis.")
        n_useful = 0
        overall  = "FAIL"

    print("=" * 80 + "\n")

    # ── CSV output ───────────────────────────────────────────────────────────────
    rows = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        row: dict = {
            "class":         cls_name,
            "positive_rate": float(pos_rates[cls_idx]),
            "baseline_f1":   float(baselines[cls_idx]),
        }
        for h_idx, head_name in enumerate(available_heads):
            row[f"f1_{head_name}"]    = float(f1_matrix[h_idx, cls_idx])
            row[f"aucroc_{head_name}"]= float(auc_matrix[h_idx, cls_idx])
        if "gnn" in available_heads:
            gnn_idx = available_heads.index("gnn")
            row["gnn_delta_vs_baseline"] = float(f1_matrix[gnn_idx, cls_idx] - baselines[cls_idx])
        rows.append(row)

    csv_path = out_dir / "exp_a4_aux_contribution.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info(f"CSV saved: {csv_path}")

    # ── JSON output ──────────────────────────────────────────────────────────────
    json_data = {
        "experiment":         "exp_a4_aux_contribution",
        "n_samples":          N,
        "available_heads":    available_heads,
        "pass_criteria":      "GNN F1 >= baseline+5pp for >=5 of 10 classes",
        "n_gnn_useful":       n_useful if "gnn" in available_heads else None,
        "gnn_useful_classes": gnn_useful_classes if "gnn" in available_heads else [],
        "gnn_weak_classes":   gnn_weak_classes   if "gnn" in available_heads else [],
        "overall":            overall,
        "positive_rates":     {c: float(pos_rates[i]) for i, c in enumerate(CLASS_NAMES)},
        "baseline_f1":        {c: float(baselines[i])  for i, c in enumerate(CLASS_NAMES)},
        "f1_by_head": {
            head_name: {c: float(f1_matrix[h_idx, ci]) for ci, c in enumerate(CLASS_NAMES)}
            for h_idx, head_name in enumerate(available_heads)
        },
        "aucroc_by_head": {
            head_name: {c: float(auc_matrix[h_idx, ci]) for ci, c in enumerate(CLASS_NAMES)}
            for h_idx, head_name in enumerate(available_heads)
        },
    }
    json_path = out_dir / "exp_a4_aux_contribution.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    log.info(f"JSON saved: {json_path}")

    # ── Heatmap PNGs ─────────────────────────────────────────────────────────────
    if len(available_heads) > 1:
        # Pad to include baseline row in F1 heatmap
        f1_with_baseline = np.vstack([f1_matrix, baselines[np.newaxis, :]])
        f1_row_labels    = available_heads + ["baseline"]
        plot_class_heatmap(
            matrix=f1_with_baseline,
            row_labels=f1_row_labels,
            col_labels=CLASS_NAMES,
            title="F1 Score by Head × Vulnerability Class (threshold=0.5, incl. baseline)",
            output_path=out_dir / "exp_a4_f1_heatmap.png",
            fmt=".3f",
            cmap="Greens",
            figsize=(16, 6),
        )

        plot_class_heatmap(
            matrix=auc_matrix,
            row_labels=available_heads,
            col_labels=CLASS_NAMES,
            title="AUC-ROC by Head × Vulnerability Class",
            output_path=out_dir / "exp_a4_aucroc_heatmap.png",
            fmt=".3f",
            cmap="Blues",
            figsize=(16, 5),
        )
    else:
        log.info("Only main head available — skipping multi-head heatmaps.")

    return 0 if overall == "PASS" else 1


def main() -> None:
    parser = _build_argparser()
    args   = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
