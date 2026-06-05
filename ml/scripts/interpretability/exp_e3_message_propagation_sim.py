"""
exp_e3_message_propagation_sim.py — Layer 2, P1: Message-Passing Information
Propagation Simulation (Random Weights)

PURPOSE
───────
With randomly initialised GNNEncoder weights (no checkpoint), run a forward
pass and measure how information flows between neighbouring nodes via cosine
similarity.  This tests whether the graph TOPOLOGY (not learned weights) enables
CEI-pattern-related node pairs to develop similar representations after message
passing.

LAYER / PRIORITY
─────────────────
Layer 2, Priority 1 — Topology-driven information-flow verification.

HOW IT WORKS
────────────
1. Build a GNNEncoder with random weights (same architecture as production).
2. Run with return_intermediates=True to get embeddings after each phase.
3. For each edge in the graph, compute cosine similarity between source and
   destination node embeddings at each phase.
4. Track two edge-type slices:
     CALL_ENTRY edges (type 8): CFG_NODE_CALL -> callee
     CONTROL_FLOW edges (type 6): consecutive CFG nodes
5. Compare reentrancy-positive vs reentrancy-negative contracts.

Pass criterion:
  For CALL_ENTRY edges, reentrancy-positive contracts should show >=0.02 higher
  mean cosine similarity after Phase 2 than Phase 1 (phase 2 propagation helps).
  If Phase 1 and Phase 2 cosine similarities are within 0.01 of each other for
  any tracked edge type, warn that phase 2 edges are not creating new information
  pathways.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_e3_message_propagation_sim.py \\
        --cache ml/data/cached_dataset_v10.pkl \\
        --label-csv ml/data/processed/multilabel_index.csv \\
        --splits-dir ml/data/splits/v9_deduped \\
        --out ml/logs/interpretability/e3_message_propagation_sim.json \\
        --n-contracts 100

    # No --checkpoint needed — uses random weights by design.

OUTPUT
──────
Per-phase mean cosine similarity for CALL_ENTRY and CONTROL_FLOW edges,
comparison table between reentrancy-positive and negative, JSON report, PNG.

EXIT CODES
──────────
    0  pass criterion met
    1  pass criterion not met
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
import pandas as pd
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    add_common_args,
    load_val_split,
    CLASS_NAMES,
    get_node_type_tensor,
)
from ml.src.preprocessing.graph_schema import NODE_TYPES, EDGE_TYPES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Edge types of interest
_CALL_ENTRY   = EDGE_TYPES["CALL_ENTRY"]    # 8
_CONTROL_FLOW = EDGE_TYPES["CONTROL_FLOW"]  # 6

PHASE_NAMES = ["phase1", "phase2", "phase3"]
TRACKED_EDGE_TYPES = {
    "CALL_ENTRY":   _CALL_ENTRY,
    "CONTROL_FLOW": _CONTROL_FLOW,
}


# ── Model construction ────────────────────────────────────────────────────────

def _build_random_gnn(device: str = "cpu"):
    """Construct a GNNEncoder with random weights (no checkpoint loaded).

    Matches the production architecture (Run 4): 8-layer GAT, hidden_dim=256,
    phase2_edge_types=[6,8,9], no dropout (diagnostic mode).
    node_feature_dim is the module-level constant NODE_FEATURE_DIM=11; it is
    NOT a constructor argument — GNNEncoder always uses the schema constant.
    """
    from ml.src.models.gnn_encoder import GNNEncoder
    model = GNNEncoder(
        hidden_dim        = 256,
        num_layers        = 8,
        heads             = 8,
        dropout           = 0.0,   # no dropout for diagnostic
        use_edge_attr     = True,
        edge_emb_dim      = 64,
        use_jk            = True,
        jk_mode           = "attention",
        phase2_edge_types = [6, 8, 9],
    ).to(device)
    model.eval()
    return model


# ── Cosine similarity per edge type ──────────────────────────────────────────

def _edge_cosine_by_type(
    embeddings: torch.Tensor,
    edge_index:  torch.Tensor,
    edge_attr:   torch.Tensor,
    edge_type_id: int,
) -> float:
    """
    Mean cosine similarity between source and destination embeddings for all
    edges of a given type.

    Returns float('nan') if no edges of the requested type exist.
    """
    if edge_index.numel() == 0 or edge_attr is None:
        return float("nan")

    mask = (edge_attr == edge_type_id)
    if not mask.any():
        return float("nan")

    ei_sub = edge_index[:, mask]     # [2, E_sub]
    src_emb = embeddings[ei_sub[0]]  # [E_sub, D]
    dst_emb = embeddings[ei_sub[1]]  # [E_sub, D]

    # F.cosine_similarity over dim=1
    cos_sim = F.cosine_similarity(src_emb, dst_emb, dim=1)  # [E_sub]
    return float(cos_sim.mean().item())


# ── Per-contract forward pass ─────────────────────────────────────────────────

@torch.no_grad()
def _process_contract(graph, gnn_model, device: str) -> dict | None:
    """
    Run GNNEncoder with return_intermediates=True on one contract graph.

    Returns dict with per-phase cosine similarities for tracked edge types,
    or None if the graph is invalid.
    """
    if graph is None or graph.x is None or graph.x.shape[0] == 0:
        return None
    if graph.edge_attr is None or graph.edge_index.numel() == 0:
        return None

    x          = graph.x.float().to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr  = graph.edge_attr.to(device)
    batch      = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    try:
        # GNNEncoder returns (embeddings, batch, jk_entropy, intermediates_dict)
        # when return_intermediates=True (4-tuple).
        fwd = gnn_model(x, edge_index, batch, edge_attr, return_intermediates=True)
        if len(fwd) == 4:
            out, _, _jk_entropy, intermediates = fwd
        elif len(fwd) == 3:
            out, _, intermediates = fwd
        else:
            log.debug(f"Unexpected GNNEncoder return length: {len(fwd)}")
            return None
    except Exception as exc:
        log.debug(f"GNNEncoder forward error: {exc}")
        return None

    result: dict = {}
    for phase_name, phase_key in zip(
        PHASE_NAMES, ["after_phase1", "after_phase2", "after_phase3"]
    ):
        phase_emb = intermediates[phase_key].float()  # [N, D]
        result[phase_name] = {}
        for et_name, et_id in TRACKED_EDGE_TYPES.items():
            result[phase_name][et_name] = _edge_cosine_by_type(
                phase_emb, edge_index, edge_attr, et_id
            )

    return result


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(
    stems: list[str],
    df_split: pd.DataFrame,
    cache: dict,
    n_contracts: int,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Run message-propagation simulation and return aggregated results."""
    gnn_model = _build_random_gnn(device)

    rng = np.random.default_rng(seed)
    stem_to_row = {row["md5_stem"]: row for _, row in df_split.iterrows()}

    pos_stems = [s for s in stems if s in stem_to_row and int(stem_to_row[s].get("Reentrancy", 0)) == 1]
    neg_stems = [s for s in stems if s in stem_to_row and int(stem_to_row[s].get("Reentrancy", 0)) == 0]

    if len(pos_stems) > n_contracts:
        pos_stems = rng.choice(pos_stems, size=n_contracts, replace=False).tolist()
    if len(neg_stems) > n_contracts:
        neg_stems = rng.choice(neg_stems, size=n_contracts, replace=False).tolist()

    log.info(f"Processing {len(pos_stems)} positive + {len(neg_stems)} negative contracts")

    def _aggregate_group(stem_list: list[str]) -> dict:
        # Accumulate sum and count per (phase, edge_type)
        sums:   dict[str, dict[str, float]] = {p: {e: 0.0 for e in TRACKED_EDGE_TYPES} for p in PHASE_NAMES}
        counts: dict[str, dict[str, int]]   = {p: {e: 0   for e in TRACKED_EDGE_TYPES} for p in PHASE_NAMES}

        for stem in stem_list:
            if stem not in cache:
                continue
            entry = cache[stem]
            if not isinstance(entry, tuple):
                continue
            graph, _ = entry
            res = _process_contract(graph, gnn_model, device)
            if res is None:
                continue
            for phase in PHASE_NAMES:
                for et_name in TRACKED_EDGE_TYPES:
                    val = res[phase][et_name]
                    if not np.isnan(val):
                        sums[phase][et_name]   += val
                        counts[phase][et_name] += 1

        means: dict = {}
        for phase in PHASE_NAMES:
            means[phase] = {}
            for et_name in TRACKED_EDGE_TYPES:
                c = counts[phase][et_name]
                means[phase][et_name] = (
                    round(sums[phase][et_name] / c, 6) if c > 0 else None
                )
                means[phase][f"{et_name}_n_contracts"] = c
        return means

    log.info("Processing reentrancy-positive contracts...")
    pos_means = _aggregate_group(pos_stems)

    log.info("Processing reentrancy-negative contracts...")
    neg_means = _aggregate_group(neg_stems)

    # Pass criterion: CALL_ENTRY cosine(Phase2) - cosine(Phase1) >= 0.02 for positives
    ce_p1_pos = pos_means.get("phase1", {}).get("CALL_ENTRY")
    ce_p2_pos = pos_means.get("phase2", {}).get("CALL_ENTRY")
    if ce_p1_pos is not None and ce_p2_pos is not None:
        delta_pos = ce_p2_pos - ce_p1_pos
        pass_call_entry = delta_pos >= 0.02
    else:
        delta_pos       = None
        pass_call_entry = False

    # Warning: if phase1 and phase2 are within 0.01 for any tracked edge type
    warnings: list[str] = []
    for et_name in TRACKED_EDGE_TYPES:
        p1 = pos_means.get("phase1", {}).get(et_name)
        p2 = pos_means.get("phase2", {}).get(et_name)
        if p1 is not None and p2 is not None and abs(p2 - p1) < 0.01:
            warnings.append(
                f"{et_name}: Phase1 and Phase2 cosine similarity within 0.01 "
                f"(p1={p1:.4f}, p2={p2:.4f}) — phase2 edge types are not creating "
                f"new information pathways for this edge type"
            )

    return {
        "reentrancy_positive": {
            "n_stems":  len(pos_stems),
            "means":    pos_means,
        },
        "reentrancy_negative": {
            "n_stems":  len(neg_stems),
            "means":    neg_means,
        },
        "call_entry_phase2_minus_phase1_pos": delta_pos,
        "pass_call_entry_criterion": pass_call_entry,
        "pass_criterion_description": (
            "CALL_ENTRY cosine(Phase2) - cosine(Phase1) >= 0.02 for reentrancy positives"
        ),
        "warnings": warnings,
        "overall_pass": pass_call_entry,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(results: dict) -> None:
    print(f"\n{'═'*72}")
    print("  EXP-E3: Message Propagation Simulation (Random Weights)")
    print(f"{'═'*72}")

    for group_name, group_key in [
        ("Reentrancy POSITIVE", "reentrancy_positive"),
        ("Reentrancy NEGATIVE", "reentrancy_negative"),
    ]:
        group = results[group_key]
        print(f"\n  {group_name} (n={group['n_stems']}):")
        print(f"  {'Edge Type':<16} {'Phase1':>10} {'Phase2':>10} {'Phase3':>10}  {'delta P2-P1':>12}")
        print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}  {'-'*12}")
        means = group["means"]
        for et_name in TRACKED_EDGE_TYPES:
            p1 = means.get("phase1", {}).get(et_name)
            p2 = means.get("phase2", {}).get(et_name)
            p3 = means.get("phase3", {}).get(et_name)
            delta = (p2 - p1) if (p1 is not None and p2 is not None) else None
            fmt = lambda v: f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
            dfmt = lambda v: f"{v:>12.4f}" if v is not None else f"{'N/A':>12}"
            print(f"  {et_name:<16} {fmt(p1)} {fmt(p2)} {fmt(p3)}  {dfmt(delta)}")

    delta = results.get("call_entry_phase2_minus_phase1_pos")
    passed = results["overall_pass"]
    print(f"\n  Pass criterion: CALL_ENTRY delta(P2-P1) >= 0.02 for positives")
    print(f"  Observed delta: {delta:.4f}" if delta is not None else "  Observed delta: N/A")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    for w in results.get("warnings", []):
        print(f"\n  WARNING: {w}")

    print(f"\n{'─'*72}")
    print(f"  EXP-E3 RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'═'*72}\n")


def _save_plot(results: dict, out_dir: Path) -> None:
    phases = PHASE_NAMES
    x = np.arange(len(phases))
    width = 0.35

    fig, axes = plt.subplots(1, len(TRACKED_EDGE_TYPES), figsize=(11, 5))
    if len(TRACKED_EDGE_TYPES) == 1:
        axes = [axes]

    for ax, et_name in zip(axes, TRACKED_EDGE_TYPES):
        pos_vals = [results["reentrancy_positive"]["means"].get(p, {}).get(et_name) or 0.0 for p in phases]
        neg_vals = [results["reentrancy_negative"]["means"].get(p, {}).get(et_name) or 0.0 for p in phases]
        ax.bar(x - width / 2, pos_vals, width, label="Pos (Reentrancy)")
        ax.bar(x + width / 2, neg_vals, width, label="Neg")
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title(f"{et_name} edges")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("E3: Mean Cosine Similarity per Phase (Random GNN Weights)")
    plt.tight_layout()
    plot_path = out_dir / "e3_message_propagation_sim.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved: {plot_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP-E3: Message Propagation Simulation (Random Weights)"
    )
    add_common_args(p, require_checkpoint=False)
    p.set_defaults(n_contracts=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_path = (
        Path(args.out)
        if args.out
        else Path("ml/logs/interpretability/e3_message_propagation_sim.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if hasattr(args, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    # Use CPU for random-weight diagnostic — avoids unnecessary VRAM usage
    device = "cpu"

    try:
        stems, df_split, cache = load_val_split(
            cache_path  = Path(args.cache),
            label_csv   = Path(args.label_csv),
            splits_dir  = Path(args.splits_dir),
            split       = args.split,
        )
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1

    results = run_analysis(
        stems       = stems,
        df_split    = df_split,
        cache       = cache,
        n_contracts = args.n_contracts,
        device      = device,
        seed        = args.seed,
    )

    _print_report(results)
    _save_plot(results, out_path.parent)

    report = {
        "args": {
            "cache":       str(args.cache),
            "label_csv":   str(args.label_csv),
            "splits_dir":  str(args.splits_dir),
            "split":       args.split,
            "n_contracts": args.n_contracts,
            "seed":        args.seed,
            "device":      device,
        },
        "results":      results,
        "overall_pass": results["overall_pass"],
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved: {out_path}")

    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
