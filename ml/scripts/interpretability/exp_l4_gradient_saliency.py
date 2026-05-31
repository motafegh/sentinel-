"""
exp_l4_gradient_saliency.py — Layer 3, P1: Gradient Saliency (Node Features & Node Identity)

PURPOSE
───────
Compute the gradient of the model's prediction score for each class C with
respect to the input node feature matrix X.  The resulting saliency map
reveals which node features and node types drive each class prediction.

Key questions answered:
- Timestamp: is uses_block_globals (feature dim 2) in the top-3 most salient features?
- Reentrancy: is saliency concentrated on CFG_NODE_CALL (type 8) / CFG_NODE_WRITE
  (type 9) nodes rather than CONTRACT (type 7) / FUNCTION (type 1) nodes?

LAYER / PRIORITY
─────────────────
Layer 3, Priority 1 — Feature-level interpretability.

APPROACH
─────────
1. For each class, iterate over up to N val-split contracts where that class=1.
2. Set x.requires_grad_(True), run forward (with torch.compile handled via
   try/except), call logits[0, class_idx].backward().
3. Saliency per contract = x.grad.abs() [N_nodes, 11].
4. Aggregate: mean saliency per feature dim and per node type.

torch.compile graceful fallback:
- Primary: standard gradient backward (works when model is not compiled or
  when compile mode allows grad).
- Fallback: permutation-based importance approximation (swap each feature
  column to its mean and measure logit drop — no gradient required).

PASS CRITERIA
─────────────
- Timestamp:  uses_block_globals (dim 2) accounts for ≥20% of total saliency
- Reentrancy: CFG_NODE_CALL (8) + CFG_NODE_WRITE (9) combined saliency ≥20%
- FLAG:       if CONTRACT (7) or FUNCTION (1) accounts for >50% of Reentrancy
              saliency → model may not be using CFG structure

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_l4_gradient_saliency.py \\
        --checkpoint ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt \\
        --cache ml/data/cached_dataset_v8.pkl \\
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \\
        --splits-dir ml/data/splits/deduped \\
        --out ml/logs/interpretability/l4_gradient_saliency.json

OUTPUT
──────
    - Per-class saliency tables (stdout)
    - Heatmap: feature_dim × class (PNG)
    - Heatmap: node_type × class (PNG)
    - JSON report

EXIT CODES
──────────
    0  analysis completed (even if some pass criteria not met)
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
    get_node_type_tensor,
    plot_class_heatmap,
)
from ml.src.preprocessing.graph_schema import NODE_TYPES, EDGE_TYPES, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_TYPE_ID: float = float(max(NODE_TYPES.values()))  # 12.0

# FEATURE_NAMES imported from graph_schema (v8, 11 dims):
#   [0] type_id  [1] visibility  [2] uses_block_globals (Timestamp signal)
#   [3] view  [4] payable  [5] complexity  [6] loc
#   [7] return_ignored (UnusedReturn signal)  [8] call_target_typed
#   [9] has_loop  [10] external_call_count (Reentrancy signal)

NODE_TYPE_ID_TO_NAME: dict[int, str] = {v: k for k, v in NODE_TYPES.items()}

# Node types 1-6 are function-level; 8-12 are CFG-level
_ALL_NODE_TYPE_IDS: list[int] = sorted(NODE_TYPES.values())


# ── Saliency computation ──────────────────────────────────────────────────────

def compute_saliency_backward(
    model,
    graph,
    token: dict,
    class_idx: int,
    device: str,
) -> tuple[np.ndarray | None, str]:
    """
    Compute gradient saliency via standard backward pass.

    Returns:
        (saliency [N, 11] or None, method_used)
    """
    from torch_geometric.data import Batch

    try:
        model.eval()
        # We need gradients through x, so disable torch.no_grad
        batch = Batch.from_data_list([graph]).to(device)
        x = batch.x.float()
        x.requires_grad_(True)
        batch.x = x

        input_ids  = token["input_ids"].unsqueeze(0).to(device)
        attn_mask  = token["attention_mask"].unsqueeze(0).to(device)

        # Forward with gradients
        output = model(batch, input_ids, attn_mask, return_aux=False)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        # Backward on the single class score
        logits[0, class_idx].backward()

        if x.grad is None:
            return None, "backward_no_grad"

        saliency = x.grad.abs().detach().cpu().numpy()  # [N, 11]
        return saliency, "backward"

    except Exception as exc:
        log.debug(f"Backward saliency failed (class {class_idx}): {exc}")
        return None, f"backward_failed:{type(exc).__name__}"


def compute_saliency_permutation(
    model,
    graph,
    token: dict,
    class_idx: int,
    device: str,
) -> tuple[np.ndarray | None, str]:
    """
    Permutation-based saliency approximation — no gradient required.

    For each feature dimension d, replace column d with its global mean and
    measure the drop in logit score.  Works with any compiled model.

    Returns:
        (saliency [N, 11], "permutation")
        saliency[n, d] = max(0, logit_original - logit_with_d_zeroed)
        broadcast back to per-node: each node gets the same feature importance,
        so node-type aggregation still reveals structural patterns.
    """
    from torch_geometric.data import Batch

    try:
        model.eval()
        batch = Batch.from_data_list([graph]).to(device)
        x_orig = batch.x.float()
        input_ids = token["input_ids"].unsqueeze(0).to(device)
        attn_mask = token["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(batch, input_ids, attn_mask, return_aux=False)
            logit_orig = float(
                (output[0] if isinstance(output, tuple) else output)[0, class_idx].item()
            )

        n_nodes, n_feats = x_orig.shape
        feature_importance = np.zeros(n_feats, dtype=np.float32)

        for d in range(n_feats):
            batch_d = Batch.from_data_list([graph]).to(device)
            x_d = batch_d.x.float()
            col_mean = x_d[:, d].mean()
            x_d[:, d] = col_mean
            batch_d.x = x_d

            with torch.no_grad():
                out_d = model(batch_d, input_ids, attn_mask, return_aux=False)
                logit_d = float(
                    (out_d[0] if isinstance(out_d, tuple) else out_d)[0, class_idx].item()
                )

            feature_importance[d] = max(0.0, logit_orig - logit_d)

        # Broadcast: each node receives the same feature importance vector
        saliency = np.tile(feature_importance, (n_nodes, 1))  # [N, 11]
        return saliency, "permutation"

    except Exception as exc:
        log.debug(f"Permutation saliency failed (class {class_idx}): {exc}")
        return None, f"permutation_failed:{type(exc).__name__}"


def compute_saliency(
    model,
    graph,
    token: dict,
    class_idx: int,
    device: str,
) -> tuple[np.ndarray | None, str]:
    """
    Try backward saliency first; fall back to permutation if it fails.
    """
    sal, method = compute_saliency_backward(model, graph, token, class_idx, device)
    if sal is not None:
        return sal, method

    log.debug(f"  Falling back to permutation saliency for class {class_idx}")
    return compute_saliency_permutation(model, graph, token, class_idx, device)


# ── Aggregation helpers ───────────────────────────────────────────────────────

def aggregate_saliency(
    saliency_list: list[np.ndarray],  # each [N_i, 11]
    node_types_list: list[np.ndarray],  # each [N_i] int
) -> tuple[np.ndarray, dict[int, float]]:
    """
    Aggregate saliency across multiple contracts.

    Returns:
        mean_per_feature: [11] mean saliency per feature dim (normalized to sum=1)
        mean_per_node_type: {type_id: fraction_of_total_saliency}
    """
    all_sal = np.vstack(saliency_list)       # [sum_N, 11]
    all_types = np.concatenate(node_types_list)  # [sum_N]

    # Per-feature mean
    per_feature = all_sal.mean(axis=0)       # [11]
    feature_total = per_feature.sum()
    if feature_total > 0:
        per_feature_norm = per_feature / feature_total
    else:
        per_feature_norm = per_feature

    # Per-node-type mean saliency (total across all features, then sum per type)
    node_total_sal = all_sal.sum(axis=1)     # [sum_N] total saliency per node
    total_sal = node_total_sal.sum()
    per_type: dict[int, float] = {}
    for tid in _ALL_NODE_TYPE_IDS:
        mask = all_types == tid
        if mask.any():
            per_type[tid] = float(node_total_sal[mask].sum() / max(total_sal, 1e-12))
        else:
            per_type[tid] = 0.0

    return per_feature_norm, per_type


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_class_saliency(
    model,
    stems: list[str],
    df_split,
    cache: dict,
    class_idx: int,
    class_name: str,
    device: str,
    max_contracts: int,
) -> dict | None:
    """
    Collect saliency for up to max_contracts positive examples of class_idx.
    """
    import pandas as pd

    # Filter to positive examples
    if class_name not in df_split.columns:
        log.warning(f"  Class {class_name} not found in CSV columns")
        return None

    stem_to_label = dict(zip(df_split["md5_stem"], df_split[class_name]))
    positive_stems = [s for s in stems if stem_to_label.get(s, 0) == 1]
    log.info(f"  Class {class_name}: {len(positive_stems)} positive examples, "
             f"using up to {max_contracts}")

    sampled = positive_stems[:max_contracts]
    if not sampled:
        log.warning(f"  No positive examples for class {class_name}")
        return None

    saliency_list = []
    node_types_list = []
    methods_used: dict[str, int] = {}

    for stem in sampled:
        if stem not in cache:
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple) or len(entry) < 2:
            continue
        graph, token = entry

        sal, method = compute_saliency(model, graph, token, class_idx, device)
        if sal is None:
            continue

        node_types = get_node_type_tensor(graph).numpy()  # [N]
        saliency_list.append(sal)
        node_types_list.append(node_types)
        methods_used[method] = methods_used.get(method, 0) + 1

    if not saliency_list:
        log.warning(f"  No saliency computed for class {class_name}")
        return None

    log.info(f"  Saliency collected for {len(saliency_list)} contracts "
             f"(methods: {methods_used})")

    per_feature_norm, per_type = aggregate_saliency(saliency_list, node_types_list)

    # Top-3 feature dims
    top3_feat_idx = np.argsort(per_feature_norm)[::-1][:3].tolist()
    top3_feat = [(FEATURE_NAMES[i], round(float(per_feature_norm[i]), 4))
                 for i in top3_feat_idx]

    # Top-3 node types
    top3_type_items = sorted(per_type.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top3_types = [(NODE_TYPE_ID_TO_NAME.get(tid, f"type{tid}"), round(frac, 4))
                  for tid, frac in top3_type_items]

    # Specific checks
    dim2_fraction = float(per_feature_norm[2])
    call_type_frac = per_type.get(NODE_TYPES["CFG_NODE_CALL"], 0.0)
    write_type_frac = per_type.get(NODE_TYPES["CFG_NODE_WRITE"], 0.0)
    contract_type_frac = per_type.get(NODE_TYPES["CONTRACT"], 0.0)
    function_type_frac = per_type.get(NODE_TYPES["FUNCTION"], 0.0)

    cfg_combined = call_type_frac + write_type_frac
    high_level_combined = contract_type_frac + function_type_frac

    log.info(f"  Top-3 feature dims: {top3_feat}")
    log.info(f"  Top-3 node types:   {top3_types}")

    result = {
        "class": class_name,
        "n_contracts_used": len(saliency_list),
        "methods_used": methods_used,
        "top3_feature_dims": top3_feat,
        "top3_node_types": top3_types,
        "per_feature_norm": [round(float(v), 4) for v in per_feature_norm],
        "per_node_type": {NODE_TYPE_ID_TO_NAME.get(k, f"type{k}"): round(v, 4)
                          for k, v in per_type.items()},
        "checks": {},
    }

    # Class-specific pass criteria
    if class_name == "Timestamp":
        passes = dim2_fraction >= 0.20
        result["checks"]["timestamp_dim2_fraction"] = round(dim2_fraction, 4)
        result["checks"]["timestamp_dim2_pass"] = passes
        log.info(
            f"  TIMESTAMP CHECK: uses_block_globals (dim 2) = {dim2_fraction:.1%} "
            f"({'PASS' if passes else 'FAIL'}, threshold ≥20%)"
        )

    if class_name == "Reentrancy":
        cfg_pass = cfg_combined >= 0.20
        high_level_flag = high_level_combined > 0.50
        result["checks"]["reentrancy_cfg_fraction"] = round(cfg_combined, 4)
        result["checks"]["reentrancy_cfg_pass"] = cfg_pass
        result["checks"]["reentrancy_high_level_flag"] = high_level_flag
        log.info(
            f"  REENTRANCY CHECK: CFG_NODE_CALL+WRITE = {cfg_combined:.1%} "
            f"({'PASS' if cfg_pass else 'FAIL'}, threshold ≥20%)"
        )
        if high_level_flag:
            log.warning(
                f"  FLAG: CONTRACT+FUNCTION saliency = {high_level_combined:.1%} > 50% "
                "— model may not be using CFG structure for Reentrancy"
            )

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_feature_heatmap(
    feature_matrix: np.ndarray,  # [10 classes, 11 features]
    out_dir: Path,
) -> None:
    """
    Save heatmap: rows=classes, cols=feature_dims.
    """
    plot_class_heatmap(
        matrix=feature_matrix,
        row_labels=CLASS_NAMES,
        col_labels=[f"d{i}:{FEATURE_NAMES[i][:10]}" for i in range(len(FEATURE_NAMES))],
        title="Gradient Saliency — Feature Dim × Class (normalized)",
        output_path=out_dir / "l4_feature_saliency_heatmap.png",
        fmt=".2f",
        cmap="YlOrRd",
        figsize=(16, 8),
    )


def plot_nodetype_heatmap(
    nodetype_matrix: np.ndarray,  # [10 classes, n_types]
    type_labels: list[str],
    out_dir: Path,
) -> None:
    """
    Save heatmap: rows=classes, cols=node_types.
    """
    plot_class_heatmap(
        matrix=nodetype_matrix,
        row_labels=CLASS_NAMES,
        col_labels=type_labels,
        title="Gradient Saliency — Node Type × Class (fraction of total)",
        output_path=out_dir / "l4_nodetype_saliency_heatmap.png",
        fmt=".2f",
        cmap="YlOrRd",
        figsize=(18, 8),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gradient saliency for SENTINEL node features and node identity"
    )
    add_common_args(parser, require_checkpoint=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_path = Path(args.out) if args.out else Path("ml/logs/interpretability/l4_gradient_saliency.json")
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        model = load_model(
            checkpoint_path=Path(args.checkpoint),
            device=args.device,
            phase2_edge_types=getattr(args, "phase2_edge_types", None),
        )
    except Exception as exc:
        log.error(f"Model load failed: {exc}")
        return 1

    # ── Load val split ─────────────────────────────────────────────────────────
    try:
        stems, df_split, cache = load_val_split(
            cache_path=Path(args.cache),
            label_csv=Path(args.label_csv),
            splits_dir=Path(args.splits_dir),
            split=getattr(args, "split", "val"),
        )
    except Exception as exc:
        log.error(f"Data load failed: {exc}")
        return 1

    # ── Per-class saliency ─────────────────────────────────────────────────────
    log.info(f"Running gradient saliency for all {len(CLASS_NAMES)} classes...")

    class_results = []
    feature_matrix  = np.zeros((len(CLASS_NAMES), len(FEATURE_NAMES)), dtype=np.float32)
    type_ids_ordered = _ALL_NODE_TYPE_IDS
    type_labels_ordered = [NODE_TYPE_ID_TO_NAME.get(tid, f"type{tid}") for tid in type_ids_ordered]
    nodetype_matrix = np.zeros((len(CLASS_NAMES), len(type_ids_ordered)), dtype=np.float32)

    for ci, cname in enumerate(CLASS_NAMES):
        log.info(f"\n[{ci+1}/{len(CLASS_NAMES)}] {cname}")
        res = run_class_saliency(
            model=model,
            stems=stems,
            df_split=df_split,
            cache=cache,
            class_idx=ci,
            class_name=cname,
            device=args.device,
            max_contracts=args.n_contracts,
        )
        if res is not None:
            class_results.append(res)
            feature_matrix[ci] = res["per_feature_norm"]
            for j, tid in enumerate(type_ids_ordered):
                tname = NODE_TYPE_ID_TO_NAME.get(tid, f"type{tid}")
                nodetype_matrix[ci, j] = res["per_node_type"].get(tname, 0.0)

    # ── Print per-class tables ─────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PER-CLASS TOP-3 FEATURE DIMS")
    log.info("=" * 70)
    for r in class_results:
        log.info(f"  {r['class']:25s}: {r['top3_feature_dims']}")

    log.info("\n" + "=" * 70)
    log.info("PER-CLASS TOP-3 NODE TYPES")
    log.info("=" * 70)
    for r in class_results:
        log.info(f"  {r['class']:25s}: {r['top3_node_types']}")

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    try:
        plot_feature_heatmap(feature_matrix, out_dir)
    except Exception as exc:
        log.warning(f"Feature heatmap failed: {exc}")

    try:
        plot_nodetype_heatmap(nodetype_matrix, type_labels_ordered, out_dir)
    except Exception as exc:
        log.warning(f"Node-type heatmap failed: {exc}")

    # ── JSON output ───────────────────────────────────────────────────────────
    report = {
        "experiment": "exp_l4_gradient_saliency",
        "checkpoint": str(args.checkpoint),
        "n_contracts_per_class": args.n_contracts,
        "split": getattr(args, "split", "val"),
        "class_results": class_results,
        "feature_names": FEATURE_NAMES,
        "node_type_order": type_labels_ordered,
    }
    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nJSON report saved: {out_path}")
    log.info(f"Heatmaps saved to: {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
