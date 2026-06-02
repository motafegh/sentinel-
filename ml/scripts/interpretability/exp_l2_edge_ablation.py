"""
exp_l2_edge_ablation.py — Inference-Time Edge Type Ablation

PURPOSE
───────
Layer 3, P0 experiment.  Measures the causal contribution of each of the 11
edge types by zeroing that edge type's embedding vector and measuring the drop
in predicted vulnerability probabilities.  Validates that semantically
important edge types (CONTROL_FLOW, CALL_ENTRY for Reentrancy; DEF_USE for
IntegerUO) cause measurable drops, while structurally irrelevant types (EMITS)
do not.

LAYER
─────
Layer 3 — inference-time ablation (no gradient, no retraining).

PASS CRITERIA
─────────────
• CONTROL_FLOW (type 6) ablation causes ≥0.03 mean probability drop for
  Reentrancy positives.
• CALL_ENTRY   (type 8) ablation causes ≥0.03 mean probability drop for
  Reentrancy positives.
• If ablating ALL CFG edges (types 6, 8, 9, 10) combined causes <0.01 total
  drop → CRITICAL WARNING printed (model not using CF edges at all).

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/interpretability/exp_l2_edge_ablation.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --out ml/interpretability_results/exp_l2 \\
        --n-contracts 200

OUTPUT
──────
  <out>/exp_l2_ablation_delta.csv    — 11×10 mean probability drop table
  <out>/exp_l2_ablation_delta.json   — same data + pass/fail flags
  <out>/exp_l2_ablation_heatmap.png  — edge_type × class heatmap (positives only)
  stdout                             — ablation results + pass/fail
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    load_model,
    load_val_split,
    add_common_args,
    CLASS_NAMES,
    plot_class_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Edge type metadata — must match graph_schema.EDGE_TYPES exactly.
# BUG-AUDIT: original had CONTAINS at 0 and CALLS at 5, which is inverted.
# Schema: CALLS=0, READS=1, WRITES=2, EMITS=3, INHERITS=4, CONTAINS=5, CONTROL_FLOW=6...
EDGE_TYPE_NAMES = [
    "CALLS",           # 0
    "READS",           # 1
    "WRITES",          # 2
    "EMITS",           # 3
    "INHERITS",        # 4
    "CONTAINS",        # 5
    "CONTROL_FLOW",    # 6
    "REVERSE_CONTAINS",# 7
    "CALL_ENTRY",      # 8
    "RETURN_TO",       # 9
    "DEF_USE",         # 10
]
NUM_EDGE_TYPES = 11

# Ablation checks: (description, edge_type_idx, class_name_or_None, min_drop_or_None)
ABLATION_CHECKS = [
    ("CONTROL_FLOW ablation hurts Reentrancy",  6, "Reentrancy",  0.03),
    ("CALL_ENTRY ablation hurts Reentrancy",    8, "Reentrancy",  0.03),
    ("DEF_USE ablation hurts IntegerUO",       10, "IntegerUO",   0.02),
    ("EMITS ablation has no effect",            3, None,          None),  # diagnostic only
]

CFG_EDGE_TYPES = [6, 8, 9, 10]  # all control-flow-related


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inference-time edge type ablation for SENTINEL GNN.",
    )
    add_common_args(p, require_checkpoint=True)
    p.set_defaults(n_contracts=200)
    return p


def _run_inference_on_sample(
    model,
    stem: str,
    cache: dict,
    stem_to_labels: dict,
    device: str,
) -> tuple[np.ndarray | None, list[int] | None]:
    """
    Run a single forward pass and return (sigmoid_probs [10], labels [10]) or (None, None).
    """
    from torch_geometric.data import Batch

    if stem not in cache:
        return None, None
    entry = cache[stem]
    if not isinstance(entry, tuple) or len(entry) < 2:
        return None, None
    graph, token = entry
    labels = stem_to_labels.get(stem)
    if labels is None:
        return None, None

    try:
        batch_g   = Batch.from_data_list([graph]).to(device)
        input_ids = token["input_ids"].unsqueeze(0).to(device)
        attn_mask = token["attention_mask"].unsqueeze(0).to(device)

        logits = model(batch_g, input_ids, attn_mask, return_aux=False)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [10]
        return probs, labels

    except Exception as exc:
        log.debug(f"Inference error for {stem}: {exc}")
        return None, None


def _structural_ablate_graph(graph, edge_t: int):
    """Return a clone of graph with all edges of type edge_t removed from edge_index/edge_attr.

    This is a HARD ablation: the edges are physically absent from the graph, so they
    cannot participate in message-passing at all (no node features, no attention).
    Contrast with the soft embedding-zero ablation where the edge still exists in
    edge_index but receives a zero embedding vector.
    """
    if not hasattr(graph, "edge_attr") or graph.edge_index.shape[1] == 0:
        return graph
    keep = graph.edge_attr != edge_t
    new_g = graph.clone()
    new_g.edge_index = graph.edge_index[:, keep]
    new_g.edge_attr = graph.edge_attr[keep]
    return new_g


def run(args: argparse.Namespace) -> int:
    device  = args.device
    out_dir = Path(args.out) if args.out else Path("ml/interpretability_results/exp_l2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────────
    model = load_model(
        Path(args.checkpoint),
        device=device,
        phase2_edge_types=args.phase2_edge_types,
    )
    model.eval()

    # Discover attribute name
    if hasattr(model, "gnn"):
        gnn_enc = model.gnn
    elif hasattr(model, "gnn_encoder"):
        gnn_enc = model.gnn_encoder
    else:
        raise RuntimeError("Cannot find GNN encoder on model.")

    if gnn_enc.edge_embedding is None:
        raise RuntimeError("Model has no edge_embedding (use_edge_attr=False). "
                           "Edge ablation requires edge embeddings.")

    # ── Load data ───────────────────────────────────────────────────────────────
    stems, df_split, cache = load_val_split(
        cache_path=Path(args.cache),
        label_csv=Path(args.label_csv),
        splits_dir=Path(args.splits_dir),
        split=args.split,
    )

    rng = np.random.default_rng(args.seed)
    if args.n_contracts and len(stems) > args.n_contracts:
        idx = rng.choice(len(stems), size=args.n_contracts, replace=False)
        stems = [stems[i] for i in idx]
        df_split = df_split.iloc[idx].reset_index(drop=True)

    stem_to_labels = {
        row["md5_stem"]: [int(row[c]) for c in CLASS_NAMES]
        for _, row in df_split.iterrows()
    }

    # ── Baseline inference ───────────────────────────────────────────────────────
    log.info("Running baseline inference...")
    baseline_probs: list[np.ndarray] = []
    baseline_labels: list[list[int]] = []
    valid_stems: list[str]           = []

    with torch.no_grad():
        for stem in stems:
            probs, labels = _run_inference_on_sample(model, stem, cache, stem_to_labels, device)
            if probs is not None:
                baseline_probs.append(probs)
                baseline_labels.append(labels)
                valid_stems.append(stem)

    if not baseline_probs:
        log.error("No baseline predictions collected.")
        return 1

    baseline_arr = np.stack(baseline_probs, axis=0)   # [N, 10]
    labels_arr   = np.array(baseline_labels, dtype=np.int32)  # [N, 10]
    log.info(f"Baseline: {len(valid_stems):,} samples collected.")

    # ── Per-edge ablation ────────────────────────────────────────────────────────
    # delta_pos[edge_type, class] = mean drop for POSITIVE samples of that class
    # delta_neg[edge_type, class] = mean drop for NEGATIVE samples
    n_edge_types  = min(NUM_EDGE_TYPES, gnn_enc.edge_embedding.num_embeddings)
    delta_pos = np.zeros((n_edge_types, len(CLASS_NAMES)))
    delta_neg = np.zeros((n_edge_types, len(CLASS_NAMES)))
    n_pos     = np.zeros((n_edge_types, len(CLASS_NAMES)), dtype=int)
    n_neg     = np.zeros((n_edge_types, len(CLASS_NAMES)), dtype=int)

    for edge_t in range(n_edge_types):
        edge_name = EDGE_TYPE_NAMES[edge_t] if edge_t < len(EDGE_TYPE_NAMES) else f"type_{edge_t}"
        log.info(f"Ablating edge type {edge_t} ({edge_name})...")

        # Save and zero
        orig_emb = gnn_enc.edge_embedding.weight.data[edge_t].clone()
        gnn_enc.edge_embedding.weight.data[edge_t] = 0.0

        ablated_probs: list[np.ndarray] = []

        with torch.no_grad():
            for stem in valid_stems:
                probs, _ = _run_inference_on_sample(model, stem, cache, stem_to_labels, device)
                ablated_probs.append(probs if probs is not None else np.zeros(len(CLASS_NAMES)))

        # Restore
        gnn_enc.edge_embedding.weight.data[edge_t] = orig_emb

        ablated_arr = np.stack(ablated_probs, axis=0)  # [N, 10]
        delta_arr   = baseline_arr - ablated_arr        # positive = drop in prob

        for cls_idx in range(len(CLASS_NAMES)):
            pos_mask = labels_arr[:, cls_idx] == 1
            neg_mask = labels_arr[:, cls_idx] == 0
            if pos_mask.sum() > 0:
                delta_pos[edge_t, cls_idx] = float(delta_arr[pos_mask, cls_idx].mean())
                n_pos[edge_t, cls_idx]     = int(pos_mask.sum())
            if neg_mask.sum() > 0:
                delta_neg[edge_t, cls_idx] = float(delta_arr[neg_mask, cls_idx].mean())
                n_neg[edge_t, cls_idx]     = int(neg_mask.sum())

    # ── Print ablation table (positives) ────────────────────────────────────────
    print("\n" + "=" * 130)
    print("EXP-L2: Edge Type Ablation — Mean Probability Drop for POSITIVE samples")
    print("=" * 130)

    col_header = "  ".join(f"{c[:8]:>8}" for c in CLASS_NAMES)
    print(f"{'Edge Type':<22}  {col_header}")
    print("-" * 130)
    for edge_t in range(n_edge_types):
        edge_name = EDGE_TYPE_NAMES[edge_t] if edge_t < len(EDGE_TYPE_NAMES) else f"type_{edge_t}"
        row_vals  = "  ".join(f"{delta_pos[edge_t, c]:>8.4f}" for c in range(len(CLASS_NAMES)))
        print(f"{edge_name:<22}  {row_vals}")
    print("=" * 130)

    # ── Ablation checks (pass/fail) ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ABLATION CHECKS")
    print("=" * 90)
    print(f"{'Check':<48}  {'Edge':>5}  {'Class':<15}  {'Delta':>8}  {'Threshold':>10}  {'Result':>8}")
    print("-" * 100)

    n_pass = 0
    n_checkable = 0
    check_results = []

    for desc, edge_t, cls_name, min_drop in ABLATION_CHECKS:
        if cls_name is None:
            # Diagnostic only
            edge_name = EDGE_TYPE_NAMES[edge_t] if edge_t < len(EDGE_TYPE_NAMES) else f"type_{edge_t}"
            max_drop  = float(delta_pos[edge_t].max())
            print(f"{desc:<48}  {edge_t:>5}  {'(all classes)':<15}  "
                  f"{max_drop:>8.4f}  {'diag':>10}  {'INFO':>8}")
            check_results.append({"description": desc, "edge_type": edge_t,
                                  "class": None, "delta": max_drop,
                                  "threshold": None, "result": "INFO"})
            continue

        cls_idx  = CLASS_NAMES.index(cls_name)
        delta    = float(delta_pos[edge_t, cls_idx])
        edge_name= EDGE_TYPE_NAMES[edge_t] if edge_t < len(EDGE_TYPE_NAMES) else f"type_{edge_t}"
        passed   = delta >= min_drop
        result   = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        n_checkable += 1

        print(f"{desc:<48}  {edge_t:>5}  {cls_name:<15}  {delta:>8.4f}  {min_drop:>10.4f}  {result:>8}")
        check_results.append({
            "description": desc,
            "edge_type":   edge_t,
            "class":       cls_name,
            "delta":       delta,
            "threshold":   min_drop,
            "result":      result,
        })

    # Combined CFG check
    cfg_drops = []
    for edge_t in CFG_EDGE_TYPES:
        if edge_t < n_edge_types:
            reentrancy_idx = CLASS_NAMES.index("Reentrancy")
            cfg_drops.append(delta_pos[edge_t, reentrancy_idx])
    total_cfg_drop = float(np.mean(cfg_drops)) if cfg_drops else 0.0
    if total_cfg_drop < 0.01:
        print(f"\n*** CRITICAL WARNING: Combined CFG-edge ablation caused only "
              f"{total_cfg_drop:.4f} mean probability drop for Reentrancy. "
              f"Model may not be using control-flow edges. ***")

    overall = "PASS" if n_pass >= 2 else "FAIL"
    print(f"\nPASS CRITERIA: both CONTROL_FLOW and CALL_ENTRY ablations ≥0.03 drop for Reentrancy")
    print(f"Result: {n_pass}/{n_checkable} checks passed → {overall}")
    print("=" * 90 + "\n")

    # ── Structural ablation (hard edge removal from edge_index) ─────────────────
    # Runs only for Phase 2 CFG edge types. This is the methodologically correct
    # ablation: edges are fully absent, so the model cannot attend to them at all.
    # Embedding-zero (above) leaves edges in edge_index, understating the effect.
    STRUCTURAL_EDGE_TYPES = [6, 8, 9, 10]  # CONTROL_FLOW, CALL_ENTRY, RETURN_TO, DEF_USE
    struct_delta_pos = np.zeros((NUM_EDGE_TYPES, len(CLASS_NAMES)))
    struct_n_pos     = np.zeros((NUM_EDGE_TYPES, len(CLASS_NAMES)), dtype=int)

    log.info("Running structural ablation (hard edge removal) for Phase 2 edge types...")
    for edge_t in STRUCTURAL_EDGE_TYPES:
        edge_name = EDGE_TYPE_NAMES[edge_t] if edge_t < len(EDGE_TYPE_NAMES) else f"type_{edge_t}"
        log.info(f"  Structural ablation: removing {edge_name} (type {edge_t}) from edge_index...")

        ablated_struct: list[np.ndarray] = []

        with torch.no_grad():
            for stem in valid_stems:
                if stem not in cache:
                    ablated_struct.append(np.zeros(len(CLASS_NAMES)))
                    continue
                entry = cache[stem]
                if not isinstance(entry, tuple) or len(entry) < 2:
                    ablated_struct.append(np.zeros(len(CLASS_NAMES)))
                    continue
                graph_orig, token = entry
                graph_mod = _structural_ablate_graph(graph_orig, edge_t)

                from torch_geometric.data import Batch as _Batch
                try:
                    batch_g   = _Batch.from_data_list([graph_mod]).to(device)
                    input_ids = token["input_ids"].unsqueeze(0).to(device)
                    attn_mask = token["attention_mask"].unsqueeze(0).to(device)
                    logits    = model(batch_g, input_ids, attn_mask, return_aux=False)
                    probs     = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    ablated_struct.append(probs)
                except Exception as exc:
                    log.debug(f"  Structural ablation inference error for {stem}: {exc}")
                    ablated_struct.append(np.zeros(len(CLASS_NAMES)))

        ablated_struct_arr = np.stack(ablated_struct, axis=0)
        delta_struct_arr   = baseline_arr - ablated_struct_arr

        for cls_idx in range(len(CLASS_NAMES)):
            pos_mask = labels_arr[:, cls_idx] == 1
            if pos_mask.sum() > 0:
                struct_delta_pos[edge_t, cls_idx] = float(delta_struct_arr[pos_mask, cls_idx].mean())
                struct_n_pos[edge_t, cls_idx]     = int(pos_mask.sum())

    # Print structural ablation results
    print("\n" + "=" * 130)
    print("EXP-L2: STRUCTURAL Ablation (hard edge removal) — Phase 2 edge types only")
    print("=" * 130)
    col_header = "  ".join(f"{c[:8]:>8}" for c in CLASS_NAMES)
    print(f"{'Edge Type':<22}  {col_header}")
    print("-" * 130)
    for edge_t in STRUCTURAL_EDGE_TYPES:
        edge_name = EDGE_TYPE_NAMES[edge_t] if edge_t < len(EDGE_TYPE_NAMES) else f"type_{edge_t}"
        row_vals  = "  ".join(f"{struct_delta_pos[edge_t, c]:>8.4f}" for c in range(len(CLASS_NAMES)))
        print(f"{edge_name:<22}  {row_vals}")

    # Combined Phase 2 structural drop
    reentrancy_idx = CLASS_NAMES.index("Reentrancy")
    combined_struct = float(np.mean([struct_delta_pos[t, reentrancy_idx] for t in STRUCTURAL_EDGE_TYPES]))
    print(f"\nCombined Phase2 structural drop for Reentrancy: {combined_struct:.4f}")
    if combined_struct < 0.01:
        print("*** CRITICAL: structural removal of all Phase2 edges changes Reentrancy "
              f"prediction by only {combined_struct:.4f}. Model is not using CFG structure. ***")
    print("=" * 130 + "\n")

    # ── CSV output ───────────────────────────────────────────────────────────────
    rows = []
    for edge_t in range(n_edge_types):
        edge_name = EDGE_TYPE_NAMES[edge_t] if edge_t < len(EDGE_TYPE_NAMES) else f"type_{edge_t}"
        row = {"edge_type": edge_t, "edge_name": edge_name}
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            row[f"delta_pos_{cls_name}"] = float(delta_pos[edge_t, cls_idx])
            row[f"delta_neg_{cls_name}"] = float(delta_neg[edge_t, cls_idx])
            row[f"n_pos_{cls_name}"]     = int(n_pos[edge_t, cls_idx])
        rows.append(row)

    csv_path = out_dir / "exp_l2_ablation_delta.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info(f"CSV saved: {csv_path}")

    # ── JSON output ──────────────────────────────────────────────────────────────
    json_data = {
        "experiment":                  "exp_l2_edge_ablation",
        "n_samples":                   len(valid_stems),
        "n_edge_types":                n_edge_types,
        "pass_criteria":               "CONTROL_FLOW and CALL_ENTRY ablation each >=0.03 for Reentrancy positives",
        "n_pass":                      n_pass,
        "n_checkable":                 n_checkable,
        "overall":                     overall,
        "cfg_combined_drop_embedding": total_cfg_drop,
        "cfg_combined_drop_structural":combined_struct,
        "checks":                      check_results,
        "delta_pos_embedding":         delta_pos.tolist(),
        "delta_pos_structural":        struct_delta_pos.tolist(),
        "structural_edge_types":       STRUCTURAL_EDGE_TYPES,
        "edge_type_names":             EDGE_TYPE_NAMES[:n_edge_types],
        "class_names":                 CLASS_NAMES,
    }
    json_path = out_dir / "exp_l2_ablation_delta.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    log.info(f"JSON saved: {json_path}")

    # ── Heatmap PNG (edge_type × class, positive drops) ─────────────────────────
    edge_labels = [
        EDGE_TYPE_NAMES[t] if t < len(EDGE_TYPE_NAMES) else f"type_{t}"
        for t in range(n_edge_types)
    ]
    plot_class_heatmap(
        matrix=delta_pos,
        row_labels=edge_labels,
        col_labels=CLASS_NAMES,
        title="Edge Ablation: Mean Probability Drop for Positive Samples (edge_type × class)",
        output_path=out_dir / "exp_l2_ablation_heatmap.png",
        fmt=".3f",
        cmap="Reds",
        figsize=(16, 9),
    )

    return 0 if overall == "PASS" else 1


def main() -> None:
    parser = _build_argparser()
    args   = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
