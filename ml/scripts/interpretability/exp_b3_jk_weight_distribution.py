"""
exp_b3_jk_weight_distribution.py — B3: JK Weight Distribution per Contract

PURPOSE
───────
EXP-L1 reported MEAN phase weights across all val contracts (Phase1=33.4%,
Phase2=33.3%, Phase3=33.3%).  The near-equal means mask per-contract variance:
if some Reentrancy contracts have Phase2 weight=0.6 and others=0.1, the model
IS selective but the mean obscures it.

This experiment measures:
- Standard deviation of Phase 2 weights per class (high std = selective use)
- Histogram of Phase 2 weights for each class (bimodal = model routes selectively)
- Mean Phase 2 weight per class (class-specific check of EXP-L1 finding)

Key questions answered:
- Is Phase 2 weight high and low variance for Reentrancy (good) or uniform (bad)?
- Does any class have a bimodal distribution of Phase 2 weights?
- Is Phase 2 systematically lower than Phase 1/3 for any class?

APPROACH
─────────
1. Run GNNEncoder (no transformer) for each val-split contract in eval mode.
2. Read gnn.jk.last_node_weights [N_nodes, 3] after each forward pass.
3. Aggregate node weights to contract level: mean per phase across all nodes.
4. Bucket by class membership: for each class, collect contract-level phase weights
   for positive contracts.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_b3_jk_weight_distribution.py \\
        --checkpoint ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt \\
        --cache ml/data/cached_dataset_v8.pkl \\
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \\
        --splits-dir ml/data/splits/deduped \\
        --out ml/logs/interpretability/b3_jk_weight_distribution.json

OUTPUT
──────
    - Per-class Phase 2 mean ± std table (stdout)
    - Histogram grid PNG: Phase 2 weight distribution per class
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
    CLASS_NAMES,
    PHASE_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

MAX_CONTRACTS = 500  # per-class cap to keep runtime reasonable


# ── JK weight collection ──────────────────────────────────────────────────────

def collect_jk_weights(
    model,
    stems: list[str],
    df_split,
    cache: dict,
    device: str,
) -> dict:
    """
    For each class, collect per-contract mean JK weights for positive contracts.

    Returns:
        {class_name: {"phase1": [float, ...], "phase2": [...], "phase3": [...]}}
    """
    from torch_geometric.data import Batch

    gnn = model.gnn
    if not hasattr(gnn, "jk") or gnn.jk is None:
        raise RuntimeError("GNNEncoder has no JK aggregator (use_jk=False)")

    stem_to_labels = {
        row["md5_stem"]: [int(row[c]) for c in CLASS_NAMES]
        for _, row in df_split.iterrows()
    }

    results: dict = {cls: {"phase1": [], "phase2": [], "phase3": []} for cls in CLASS_NAMES}

    # Collect for ALL val stems, then bucket by class
    log.info(f"Running GNN forward pass on up to {MAX_CONTRACTS} val contracts...")
    rng = np.random.default_rng(42)
    sample_stems = list(stems)
    if len(sample_stems) > MAX_CONTRACTS:
        sample_stems = rng.choice(sample_stems, size=MAX_CONTRACTS, replace=False).tolist()

    stem_weights: dict[str, list[float]] = {}  # stem -> [p1, p2, p3]

    model.eval()
    with torch.no_grad():
        for stem in sample_stems:
            if stem not in cache:
                continue
            entry = cache[stem]
            if not isinstance(entry, tuple):
                continue
            graph, _ = entry
            try:
                batch     = Batch.from_data_list([graph]).to(device)
                x         = batch.x.float()
                edge_index = batch.edge_index
                batch_vec  = batch.batch
                edge_attr  = getattr(batch, "edge_attr", None)

                _ = gnn(x, edge_index, batch_vec, edge_attr)

                w = gnn.jk.last_node_weights  # [N, 3] or None
                if w is None:
                    continue
                w_mean = w.mean(dim=0).tolist()  # [3] mean across nodes
                stem_weights[stem] = w_mean
            except Exception as exc:
                log.debug(f"  Skipping {stem}: {exc}")

    log.info(f"Collected JK weights for {len(stem_weights):,} contracts")

    # Bucket by class
    for cls_idx, cls in enumerate(CLASS_NAMES):
        for stem, (p1, p2, p3) in stem_weights.items():
            if stem in stem_to_labels and stem_to_labels[stem][cls_idx] == 1:
                results[cls]["phase1"].append(p1)
                results[cls]["phase2"].append(p2)
                results[cls]["phase3"].append(p3)

    return results, stem_weights


# ── Aggregation and reporting ─────────────────────────────────────────────────

def summarise(results: dict) -> dict:
    summary = {}
    for cls, phases in results.items():
        summary[cls] = {}
        for phase, vals in phases.items():
            if vals:
                summary[cls][phase] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std":  round(float(np.std(vals)),  4),
                    "n":    len(vals),
                }
            else:
                summary[cls][phase] = {"mean": 0.0, "std": 0.0, "n": 0}
    return summary


def _print_table(summary: dict) -> None:
    print(f"\n{'═'*80}")
    print("  B3: JK Weight Distribution per Class (positive contracts only)")
    print(f"{'═'*80}")
    header = f"  {'Class':26s}  {'P1 mean±std':>18s}  {'P2 mean±std':>18s}  {'P3 mean±std':>18s}"
    print(header)
    print(f"  {'-'*78}")
    for cls in CLASS_NAMES:
        s = summary.get(cls, {})
        row = f"  {cls:26s}"
        for ph in ("phase1", "phase2", "phase3"):
            m = s.get(ph, {}).get("mean", 0.0)
            sd = s.get(ph, {}).get("std", 0.0)
            row += f"  {m:.3f}±{sd:.3f}     "
        print(row)
    print(f"{'═'*80}\n")


def _save_phase2_histograms(results: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()
    for ci, (cls, ax) in enumerate(zip(CLASS_NAMES, axes)):
        vals = results[cls]["phase2"]
        if vals:
            ax.hist(vals, bins=20, range=(0.0, 1.0), color="steelblue",
                    alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(vals), color="red", linestyle="--",
                       linewidth=1.2, label=f"mean={np.mean(vals):.3f}")
            ax.legend(fontsize=7)
        ax.set_title(cls, fontsize=8)
        ax.set_xlabel("Phase 2 weight", fontsize=7)
        ax.set_ylabel("# contracts", fontsize=7)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=6)
    fig.suptitle("B3: Phase 2 JK Weight Distribution per Class (positive contracts)", fontsize=11)
    plt.tight_layout()
    out_path = out_dir / "b3_phase2_weight_histograms.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Histogram grid saved: {out_path}")


def _save_phase_means_bar(summary: dict, out_dir: Path) -> None:
    phases = ["phase1", "phase2", "phase3"]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (ph, col) in enumerate(zip(phases, colors)):
        means = [summary.get(cls, {}).get(ph, {}).get("mean", 0.0) for cls in CLASS_NAMES]
        ax.bar(x + i * width, means, width, label=ph.replace("phase", "Phase "), color=col, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean JK phase weight")
    ax.set_title("B3: Mean JK Phase Weights per Class (positive contracts)")
    ax.legend()
    ax.axhline(1/3, color="black", linestyle=":", linewidth=0.8, label="Uniform (1/3)")
    plt.tight_layout()
    out_path = out_dir / "b3_phase_weight_means.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Bar chart saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B3: JK Weight Distribution per Contract")
    add_common_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out) if args.out else Path("ml/logs/interpretability/b3_jk_weight_distribution.json")
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

    results, stem_weights = collect_jk_weights(model, stems, df_split, cache, args.device)
    summary = summarise(results)

    _print_table(summary)
    _save_phase2_histograms(results, out_path.parent)
    _save_phase_means_bar(summary, out_path.parent)

    report = {
        "experiment": "exp_b3_jk_weight_distribution",
        "checkpoint": str(args.checkpoint),
        "n_contracts_processed": len(stem_weights),
        "summary": summary,
    }
    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON report saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
