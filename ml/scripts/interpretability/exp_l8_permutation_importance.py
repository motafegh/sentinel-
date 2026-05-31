"""
exp_l8_permutation_importance.py — Layer 3, P2: Feature Importance via Node
Feature Permutation

PURPOSE
───────
For each of the 11 node feature dimensions, shuffle that dimension's values
randomly within each graph (preserving graph structure but destroying the
feature's signal), and measure the mean absolute change in the model's
per-class predictions.

A large change means the feature carries important signal. A near-zero change
means the model does not use that feature (either because it's genuinely
uninformative, or because the model has found alternative signal paths).

This uses GNN-only inference for efficiency (skips the transformer tokeniser).
Transformer-side feature importance requires attention analysis (see exp_l9).

LAYER / PRIORITY
─────────────────
Layer 3, Priority 2 — Feature attribution and model transparency.

METHOD
──────
For each feature dimension d in [0..10]:
  1. Compute baseline GNN-only predictions for N sample graphs.
  2. For each graph, permute g.x[:, d] within that graph (in-place on a clone).
  3. Re-run GNN-only inference.
  4. importance[d] = mean_over_graphs( abs(permuted_pred - baseline_pred) )
     Result: [11, 10] matrix — importance per feature per class.

Note: permutation is WITHIN each graph (not across graphs) to preserve the
marginal distribution of the feature while destroying graph-local correlations.

FEATURE NAMES
─────────────
Dim 0:  type_id_norm        — node type normalised
Dim 1:  visibility_norm     — function visibility (public/external/internal/private)
Dim 2:  uses_block_globals  — reads block.timestamp, block.number, etc.
Dim 3:  has_loop            — contains a loop construct
Dim 4:  payable             — function marked payable
Dim 5:  is_modifier         — node is a modifier
Dim 6:  return_ignored      — return value of call not captured
Dim 7:  call_depth_norm     — call stack depth normalised
Dim 8:  ext_call_count_norm — external call count normalised
Dim 9:  has_state_write     — writes to contract state
Dim 10: ext_call_count_raw  — raw external call count

PASS CRITERIA
─────────────
Qualitative only. Expected ranking (based on vulnerability semantics):
- type_id_norm (dim 0) and return_ignored (dim 6) should rank in top-4 overall.
- uses_block_globals (dim 2) and has_state_write (dim 9) should rank in top-6.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_l8_permutation_importance.py \\
        --checkpoint ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt \\
        --cache ml/data/cached_dataset_v8.pkl \\
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \\
        --splits-dir ml/data/splits/deduped \\
        --n-contracts 200 \\
        --out ml/logs/interpretability/l8_permutation

OUTPUT
──────
ml/logs/interpretability/l8_permutation/
  importance_heatmap.png    — 11×10 heatmap: feature × class importance
  importance_bar.png        — bar chart: mean importance per feature (across classes)
  l8_results.json           — importance matrix + rankings

EXIT CODES
──────────
    0  completed successfully
    1  load or inference error
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
    get_node_type_tensor,
    plot_class_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Feature names (v8 schema — authoritative source) ───────────────────────────
# BUG FIX (2026-05-31): The previous hard-coded list was a stale pre-v8 schema
# that misidentified 9 of 11 features. Replaced with the canonical import from
# graph_schema.py so this script always matches the training schema.

from ml.src.preprocessing.graph_schema import FEATURE_NAMES  # type: ignore

_NUM_FEATURES = 11

# ── GNN-only inference helper ─────────────────────────────────────────────────

def gnn_predict_single(model, graph, device: str) -> np.ndarray:
    """
    GNN-only prediction for a single graph.

    Returns:
        np.ndarray [10] — sigmoid probabilities.
    """
    from torch_geometric.data import Batch
    from torch_geometric.nn import global_mean_pool, global_max_pool

    batch = Batch.from_data_list([graph]).to(device)
    edge_attr = getattr(batch, "edge_attr", None)

    with torch.no_grad():
        x_out, b, _ = model.gnn(
            batch.x.float(),
            batch.edge_index,
            batch.batch,
            edge_attr,
        )

        # Function-level pool (matches SentinelModel forward)
        node_types = get_node_type_tensor(batch)
        func_ids   = torch.tensor([1, 2, 4, 5, 6], device=device)
        func_mask  = torch.isin(node_types, func_ids)

        if func_mask.any():
            pool_embs  = x_out[func_mask]
            pool_batch = b[func_mask]
        else:
            pool_embs  = x_out
            pool_batch = b

        num_graphs = int(b.max().item()) + 1
        gnn_max    = global_max_pool(pool_embs, pool_batch, size=num_graphs)
        gnn_mean   = global_mean_pool(pool_embs, pool_batch, size=num_graphs)
        gnn_eye    = model.gnn_eye_proj(torch.cat([gnn_max, gnn_mean], dim=1))
        logits     = model.aux_gnn(gnn_eye).squeeze(0)  # [10]
        probs      = torch.sigmoid(logits).cpu().numpy()

    return probs


# ── Permutation importance ────────────────────────────────────────────────────

def permutation_importance_single_feature(
    model,
    graphs: list,
    feat_dim: int,
    device: str,
    baseline_preds: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Permute feature `feat_dim` within each graph and measure mean absolute
    prediction change.

    Args:
        model:           SentinelModel in eval mode.
        graphs:          List of PyG Data objects.
        feat_dim:        Feature dimension to permute (0-10).
        device:          Torch device string.
        baseline_preds:  [N, 10] baseline predictions (precomputed).
        seed:            RNG seed for reproducible permutation.

    Returns:
        np.ndarray [10] — mean absolute prediction change per class.
    """
    rng = np.random.default_rng(seed + feat_dim)
    permuted_preds = []

    for graph in graphs:
        g_perm = graph.clone()
        N = g_perm.x.shape[0]
        # Permute this feature's values within the graph
        perm_idx = torch.from_numpy(rng.permutation(N).astype(np.int64))
        g_perm.x[:, feat_dim] = g_perm.x[perm_idx, feat_dim]
        try:
            pred = gnn_predict_single(model, g_perm, device)
        except Exception as exc:
            log.debug(f"Skipping permuted graph (feat {feat_dim}): {exc}")
            pred = np.zeros(len(CLASS_NAMES))
        permuted_preds.append(pred)

    permuted_arr = np.array(permuted_preds)  # [N, 10]
    importance   = np.abs(baseline_preds - permuted_arr).mean(axis=0)  # [10]
    return importance


def compute_permutation_importance(
    model,
    graphs: list,
    device: str,
    seed: int = 42,
) -> np.ndarray:
    """
    Run permutation importance for all 11 feature dimensions.

    Returns:
        np.ndarray [11, 10] — importance[feat_dim, class_idx].
    """
    log.info(f"Computing baseline predictions for {len(graphs)} graphs...")
    baseline_preds = []
    for graph in graphs:
        try:
            pred = gnn_predict_single(model, graph, device)
        except Exception as exc:
            log.debug(f"Skipping graph at baseline: {exc}")
            pred = np.zeros(len(CLASS_NAMES))
        baseline_preds.append(pred)
    baseline_arr = np.array(baseline_preds)  # [N, 10]

    importance_matrix = np.zeros((_NUM_FEATURES, len(CLASS_NAMES)), dtype=np.float32)

    for feat_dim in range(_NUM_FEATURES):
        fname = FEATURE_NAMES[feat_dim]
        log.info(f"  Permuting feature {feat_dim}: {fname}")
        imp = permutation_importance_single_feature(
            model, graphs, feat_dim, device, baseline_arr, seed=seed
        )
        importance_matrix[feat_dim] = imp
        log.info(
            f"    mean={imp.mean():.4f} "
            f"top-class={CLASS_NAMES[int(imp.argmax())]}({imp.max():.4f})"
        )

    return importance_matrix


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_importance_bar(
    importance_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Bar chart of mean importance per feature (averaged across classes)."""
    mean_imp = importance_matrix.mean(axis=1)  # [11]
    order    = np.argsort(mean_imp)[::-1]
    names    = [FEATURE_NAMES[i] for i in order]
    vals     = mean_imp[order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(names)), vals[::-1], color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Mean Absolute Prediction Change (across 10 classes)")
    ax.set_title("Node Feature Permutation Importance (GNN-only)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Importance bar chart saved: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Node feature permutation importance — Layer 3, P2"
    )
    add_common_args(parser, require_checkpoint=True)
    parser.set_defaults(n_contracts=200)
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

    model.eval()

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

    # Subsample graphs
    rng = np.random.default_rng(args.seed)
    if len(stems) > args.n_contracts:
        chosen_indices = rng.choice(len(stems), size=args.n_contracts, replace=False)
        stems = [stems[i] for i in chosen_indices]

    graphs = []
    for stem in stems:
        if stem not in cache:
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple) or len(entry) < 2:
            continue
        graph = entry[0]
        graphs.append(graph)

    log.info(f"Loaded {len(graphs)} graphs for permutation analysis.")
    if not graphs:
        log.error("No graphs available.")
        return 1

    # Run permutation importance
    importance_matrix = compute_permutation_importance(
        model, graphs, args.device, seed=args.seed
    )

    # Rankings
    mean_imp  = importance_matrix.mean(axis=1)
    rank_order = np.argsort(mean_imp)[::-1]

    log.info("Feature importance ranking (by mean across classes):")
    for rank, feat_idx in enumerate(rank_order):
        log.info(f"  #{rank+1:2d}  {FEATURE_NAMES[feat_idx]:<24}  mean_imp={mean_imp[feat_idx]:.5f}")

    # Qualitative pass/fail
    top4 = set(rank_order[:4])
    top6 = set(rank_order[:6])
    # BUG FIX (2026-05-31): corrected indices to match v8 FEATURE_NAMES from graph_schema.py
    # v8 dims: 0=type_id, 7=return_ignored, 2=uses_block_globals, 10=external_call_count
    # The old criteria used stale schema indices (6=return_ignored, 9=has_state_write) which
    # in v8 are actually loc and has_loop respectively.
    expected_top4 = {0, 7}    # type_id [0], return_ignored [7]
    expected_top6 = {0, 2, 7, 10}  # + uses_block_globals [2], external_call_count [10]

    pass_top4 = expected_top4.issubset(top4)
    pass_top6 = expected_top6.issubset(top6)
    overall_pass = pass_top4 and pass_top6

    log.info(
        f"Pass criteria — top-4 contains {{type_id[0], return_ignored[7]}}: {pass_top4}  "
        f"top-6 contains expected: {pass_top6}"
    )

    # Plots
    if out_dir:
        plot_class_heatmap(
            importance_matrix,
            row_labels=FEATURE_NAMES,
            col_labels=CLASS_NAMES,
            title="Node Feature Permutation Importance (GNN-only)\n[11 features × 10 classes]",
            output_path=out_dir / "importance_heatmap.png",
            fmt=".4f",
            cmap="YlOrRd",
            figsize=(14, 7),
        )
        plot_importance_bar(importance_matrix, out_dir / "importance_bar.png")

    # JSON output
    imp_list = importance_matrix.tolist()
    per_feature = [
        {
            "feature":      FEATURE_NAMES[i],
            "dim":          i,
            "mean_imp":     round(float(mean_imp[i]), 5),
            "rank":         int(np.where(rank_order == i)[0][0]) + 1,
            "per_class":    {CLASS_NAMES[j]: round(float(importance_matrix[i, j]), 5)
                             for j in range(len(CLASS_NAMES))},
        }
        for i in range(_NUM_FEATURES)
    ]
    per_feature_sorted = sorted(per_feature, key=lambda x: x["rank"])

    report = {
        "experiment":   "exp_l8_permutation_importance",
        "layer":        3,
        "priority":     2,
        "n_graphs":     len(graphs),
        "split":        args.split,
        "pass_criteria": (
            "top-4 features include type_id[0] + return_ignored[7]; "
            "top-6 include + uses_block_globals[2] + external_call_count[10] (v8 schema)"
        ),
        "pass_top4":    pass_top4,
        "pass_top6":    pass_top6,
        "overall_pass": overall_pass,
        "importance_matrix_11x10": imp_list,
        "feature_rankings":        per_feature_sorted,
        "feature_names":           FEATURE_NAMES,
        "class_names":             CLASS_NAMES,
    }

    if out_dir:
        json_path = out_dir / "l8_results.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Results written to: {json_path}")
    else:
        print(json.dumps(report, indent=2))

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
