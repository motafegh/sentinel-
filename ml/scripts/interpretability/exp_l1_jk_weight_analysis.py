"""
exp_l1_jk_weight_analysis.py — JK Attention Weight Distribution Per Vulnerability Class

PURPOSE
───────
Layer 3, P0 experiment.  For each contract in the val split, extracts the
per-node JK attention weights (phase1 / phase2 / phase3) stored by _JKAttention
after a forward pass in eval mode.  Aggregates by ground-truth vulnerability class
and tests the hypothesis that inter-procedural vulnerability classes (Reentrancy,
IntegerUO) are dominated by Phase 2 weights while simpler single-node classes
(Timestamp, UnusedReturn) are dominated by Phase 1.

LAYER
─────
Layer 3 — inference-time interpretability (no gradient computation required).

PASS CRITERIA
─────────────
≥3 of [Reentrancy, IntegerUO, Timestamp, UnusedReturn] match their expected
dominant phase (Phase 2 for Reentrancy/IntegerUO, Phase 1 for Timestamp/UnusedReturn).

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/interpretability/exp_l1_jk_weight_analysis.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --out ml/interpretability_results/exp_l1 \\
        --n-contracts 1000

OUTPUT
──────
  <out>/exp_l1_jk_weights.csv        — per-class mean/std weights + entropy
  <out>/exp_l1_jk_weights.json       — same data as JSON
  <out>/exp_l1_jk_heatmap.png        — phase × class heatmap (mean weights)
  stdout                             — hypothesis check table + pass/fail
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    PHASE_NAMES,
    get_node_type_tensor,
    plot_class_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Hypotheses: (class_name, expected_dominant_phase_index 0/1/2)
HYPOTHESES: list[tuple[str, int]] = [
    ("Reentrancy",   1),  # Phase 2 — cross-function CF patterns
    ("IntegerUO",    1),  # Phase 2 — cross-function CF / DFG
    ("Timestamp",    0),  # Phase 1 — local struct / containment
    ("UnusedReturn", 0),  # Phase 1 — local struct
]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="JK Attention Weight Distribution per vulnerability class.",
    )
    add_common_args(p, require_checkpoint=True)
    p.set_defaults(n_contracts=1000)
    return p


def run(args: argparse.Namespace) -> int:
    device = args.device
    out_dir = Path(args.out) if args.out else Path("ml/interpretability_results/exp_l1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────────
    model = load_model(
        Path(args.checkpoint),
        device=device,
        phase2_edge_types=args.phase2_edge_types,
    )
    model.eval()

    # Discover attribute name (gnn vs gnn_encoder)
    if hasattr(model, "gnn"):
        gnn_enc = model.gnn
        log.info("Using model.gnn as GNN encoder attribute.")
    elif hasattr(model, "gnn_encoder"):
        gnn_enc = model.gnn_encoder
        log.info("Using model.gnn_encoder as GNN encoder attribute.")
    else:
        raise RuntimeError("Cannot find GNN encoder on model (tried 'gnn' and 'gnn_encoder').")

    if gnn_enc.jk is None:
        raise RuntimeError("JK aggregation is disabled (model.gnn.jk is None). "
                           "This experiment requires gnn_use_jk=True.")

    # ── Load data ───────────────────────────────────────────────────────────────
    stems, df_split, cache = load_val_split(
        cache_path=Path(args.cache),
        label_csv=Path(args.label_csv),
        splits_dir=Path(args.splits_dir),
        split=args.split,
    )

    # Subsample
    rng = np.random.default_rng(args.seed)
    if args.n_contracts and len(stems) > args.n_contracts:
        idx = rng.choice(len(stems), size=args.n_contracts, replace=False)
        stems = [stems[i] for i in idx]
        df_split = df_split.iloc[idx].reset_index(drop=True)

    stem_to_labels = {
        row["md5_stem"]: [int(row[c]) for c in CLASS_NAMES]
        for _, row in df_split.iterrows()
    }

    # ── Run inference — one sample at a time to access last_node_weights ────────
    from torch_geometric.data import Batch

    # per_class_weights[c] → list of [3] mean weight arrays (one per contract positive for class c)
    per_class_weights: list[list[np.ndarray]] = [[] for _ in range(len(CLASS_NAMES))]
    per_class_entropy: list[list[float]]      = [[] for _ in range(len(CLASS_NAMES))]

    n_processed = 0
    n_skipped   = 0

    with torch.no_grad():
        for stem in stems:
            if stem not in cache:
                n_skipped += 1
                continue
            entry = cache[stem]
            if not isinstance(entry, tuple) or len(entry) < 2:
                n_skipped += 1
                continue

            graph, token = entry
            labels = stem_to_labels.get(stem)
            if labels is None:
                n_skipped += 1
                continue

            try:
                batch_g    = Batch.from_data_list([graph]).to(device)
                input_ids  = token["input_ids"].unsqueeze(0).to(device)
                attn_mask  = token["attention_mask"].unsqueeze(0).to(device)

                # Forward — in eval mode, jk.last_node_weights is populated
                _logits, _aux = model(batch_g, input_ids, attn_mask, return_aux=True)

                jk_node_weights = gnn_enc.jk.last_node_weights  # [N, 3] or None
                # BUG FIX (2026-05-31): previously stored mean weight (~0.333), not entropy.
                # Shannon entropy H = -sum(w * log(w)) over the 3 phases.
                # Max H = log(3) ≈ 1.099 when weights are uniform.
                _w = gnn_enc.jk.last_weights.clamp(min=1e-8)
                jk_entropy_val  = float(-(_w * _w.log()).sum().item())

                if jk_node_weights is None:
                    n_skipped += 1
                    continue

                # Mean weights across all nodes in this graph → [3]
                mean_w = jk_node_weights.mean(0).numpy()  # [3]

                # Record for each class where this contract is a positive
                for cls_idx, is_pos in enumerate(labels):
                    if is_pos:
                        per_class_weights[cls_idx].append(mean_w)
                        per_class_entropy[cls_idx].append(jk_entropy_val)

                n_processed += 1

            except Exception as exc:
                log.debug(f"Skipping {stem}: {exc}")
                n_skipped += 1
                continue

    log.info(f"Processed: {n_processed:,}  |  Skipped: {n_skipped:,}")

    # ── Per-class aggregation ────────────────────────────────────────────────────
    num_phases = 3
    # mean_weights[class, phase]
    mean_weights  = np.full((len(CLASS_NAMES), num_phases), np.nan)
    std_weights   = np.full((len(CLASS_NAMES), num_phases), np.nan)
    mean_entropies= np.full(len(CLASS_NAMES), np.nan)
    counts        = np.zeros(len(CLASS_NAMES), dtype=int)

    for cls_idx in range(len(CLASS_NAMES)):
        ws = per_class_weights[cls_idx]
        if not ws:
            continue
        arr = np.stack(ws, axis=0)  # [M, 3]
        mean_weights[cls_idx]   = arr.mean(0)
        std_weights[cls_idx]    = arr.std(0)
        mean_entropies[cls_idx] = float(np.mean(per_class_entropy[cls_idx]))
        counts[cls_idx]         = len(ws)

    # ── Print per-class table ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("EXP-L1: JK Attention Weight Distribution per Vulnerability Class")
    print("=" * 90)

    header = f"{'Class':<25} {'N':>5}  {'Ph1 mean±std':>18}  {'Ph2 mean±std':>18}  {'Ph3 mean±std':>18}  {'Entropy':>8}  {'Dominant':>10}"
    print(header)
    print("-" * 110)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if counts[cls_idx] == 0:
            print(f"{cls_name:<25} {'0':>5}  {'—':>18}  {'—':>18}  {'—':>18}  {'—':>8}  {'—':>10}")
            continue
        mw = mean_weights[cls_idx]
        sw = std_weights[cls_idx]
        dominant_idx = int(np.argmax(mw))
        dominant_name = f"Phase{dominant_idx+1}"
        ph_strs = [f"{mw[i]:.3f}±{sw[i]:.3f}" for i in range(num_phases)]
        print(
            f"{cls_name:<25} {counts[cls_idx]:>5}  "
            f"{ph_strs[0]:>18}  {ph_strs[1]:>18}  {ph_strs[2]:>18}  "
            f"{mean_entropies[cls_idx]:>8.4f}  {dominant_name:>10}"
        )

    # ── Hypothesis check table ───────────────────────────────────────────────────
    expected_phase_names = {0: "Phase1", 1: "Phase2", 2: "Phase3"}
    print("\n" + "=" * 90)
    print("HYPOTHESIS CHECK")
    print("=" * 90)
    print(f"{'Hypothesis':<42}  {'Expected':>10}  {'Actual':>10}  {'Ph2 mean':>10}  {'Entropy':>8}  {'Result':>8}")
    print("-" * 96)

    n_pass = 0
    hypothesis_results = []
    for cls_name, expected_phase_idx in HYPOTHESES:
        cls_idx = CLASS_NAMES.index(cls_name)
        if counts[cls_idx] == 0:
            result_str = "NO DATA"
            actual_name = "—"
            ph2_mean = float("nan")
            ent_val  = float("nan")
        else:
            mw = mean_weights[cls_idx]
            actual_phase_idx = int(np.argmax(mw))
            actual_name  = expected_phase_names[actual_phase_idx]
            ph2_mean     = float(mw[1])
            ent_val      = float(mean_entropies[cls_idx])
            passed       = (actual_phase_idx == expected_phase_idx)
            result_str   = "PASS" if passed else "FAIL"
            if passed:
                n_pass += 1

        desc = f"{cls_name} dominant={expected_phase_names[expected_phase_idx]}"
        print(
            f"{desc:<42}  {expected_phase_names[expected_phase_idx]:>10}  "
            f"{actual_name:>10}  {ph2_mean:>10.4f}  {ent_val:>8.4f}  {result_str:>8}"
        )
        hypothesis_results.append({
            "class": cls_name,
            "expected_phase": expected_phase_names[expected_phase_idx],
            "actual_phase":   actual_name,
            "ph2_mean":       ph2_mean,
            "entropy":        ent_val,
            "result":         result_str,
        })

    print("-" * 96)
    overall = "PASS" if n_pass >= 3 else "FAIL"
    print(f"\nPASS CRITERIA: ≥3/4 hypotheses correct → {n_pass}/4 → {overall}")
    print("=" * 90 + "\n")

    # ── CSV output ───────────────────────────────────────────────────────────────
    rows = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        row = {"class": cls_name, "n_positives": int(counts[cls_idx])}
        for ph in range(num_phases):
            row[f"phase{ph+1}_mean"] = float(mean_weights[cls_idx, ph]) if not np.isnan(mean_weights[cls_idx, ph]) else None
            row[f"phase{ph+1}_std"]  = float(std_weights[cls_idx, ph])  if not np.isnan(std_weights[cls_idx, ph])  else None
        row["mean_entropy"] = float(mean_entropies[cls_idx]) if not np.isnan(mean_entropies[cls_idx]) else None
        row["dominant_phase"] = (
            f"Phase{int(np.argmax(mean_weights[cls_idx]))+1}"
            if not np.all(np.isnan(mean_weights[cls_idx])) else None
        )
        rows.append(row)

    csv_path = out_dir / "exp_l1_jk_weights.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log.info(f"CSV saved: {csv_path}")

    # ── JSON output ──────────────────────────────────────────────────────────────
    json_data = {
        "experiment":       "exp_l1_jk_weight_analysis",
        "n_processed":      n_processed,
        "n_skipped":        n_skipped,
        "pass_criteria":    ">=3 of 4 hypotheses correct",
        "n_pass":           n_pass,
        "overall":          overall,
        "per_class":        rows,
        "hypothesis_check": hypothesis_results,
    }
    json_path = out_dir / "exp_l1_jk_weights.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=lambda x: None if x != x else x)
    log.info(f"JSON saved: {json_path}")

    # ── Heatmap PNG (phase × class) ──────────────────────────────────────────────
    # matrix shape: [3 phases, 10 classes]
    heatmap_matrix = np.full((num_phases, len(CLASS_NAMES)), np.nan)
    for cls_idx in range(len(CLASS_NAMES)):
        if not np.all(np.isnan(mean_weights[cls_idx])):
            heatmap_matrix[:, cls_idx] = mean_weights[cls_idx]

    # Replace NaN with 0 for display
    heatmap_display = np.nan_to_num(heatmap_matrix, nan=0.0)

    short_phase_names = ["Phase1\n(struct)", "Phase2\n(CF/DFG)", "Phase3\n(rev-CONT)"]
    plot_class_heatmap(
        matrix=heatmap_display,
        row_labels=short_phase_names,
        col_labels=CLASS_NAMES,
        title="JK Attention Mean Weight by Phase × Vulnerability Class",
        output_path=out_dir / "exp_l1_jk_heatmap.png",
        fmt=".3f",
        cmap="YlOrRd",
        figsize=(16, 5),
    )

    return 0 if overall == "PASS" else 1


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
