"""
exp_l9_attention_rollout.py — Layer 3, P2: Attention Rollout (Layer-Collapsed
Attribution)

PURPOSE
───────
Propagate GAT attention weights backward through the GNN encoder layers to
produce a per-original-node attribution score. Nodes that contribute most to
the final pooled embedding are assigned the highest scores.

This answers: "Which specific CFG nodes does the model focus on when predicting
a vulnerability?" For reentrancy, we expect CFG_NODE_CALL (type 8) and
CFG_NODE_WRITE (type 9) nodes to rank highly.

LAYER / PRIORITY
─────────────────
Layer 3, Priority 2 — Node-level attribution and interpretability.

ATTENTION ROLLOUT ALGORITHM
────────────────────────────
For a sequence of attention layers L_1, ..., L_K:
  1. For each layer L_k, build attention matrix A_k [N×N]:
       A_k[dst, src] = mean attention weight across heads for edge (src→dst).
  2. Add identity (residual approximation):
       A_k_aug = 0.5 * A_k + 0.5 * I
  3. Normalise rows to sum to 1.
  4. Rollout: A_roll = A_1 @ A_2 @ ... @ A_K
  5. Attribution for node v relative to pool node p:
       score[v] = A_roll[p, v]

LIMITATION NOTE
────────────────
Full 8-layer rollout requires model modification (GATConv return_attention_weights
cannot easily be hooked post-hoc without re-running each layer manually).

This script captures Phase 2 (conv3 — CONTROL_FLOW) and Phase 3 (conv4 —
REVERSE_CONTAINS up) only. These are the two phases most relevant to CEI
violations: Phase 2 captures CALL→WRITE ordering, Phase 3 propagates the
signal from CFG to FUNCTION node.

A future extension could instrument GNNEncoder to return attention per layer.

PASS CRITERIA
─────────────
Relative-rank: for each vulnerable/safe pair, the mean attribution score of
ALL CFG_NODE_CALL (type 8) and CFG_NODE_WRITE (type 9) nodes is higher in
the vulnerable contract than in the matched safe contract.

This replaces the previous "≥2 CALL/WRITE in top-10" criterion, which was
satisfied by both vulnerable and safe contracts and therefore non-discriminative.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_l9_attention_rollout.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --contracts-dir ml/scripts/interpretability/test_contracts \\
        --out ml/logs/interpretability/l9_attention_rollout

OUTPUT
──────
ml/logs/interpretability/l9_attention_rollout/
  attention_rollout_report.txt  — human-readable per-contract top-10 nodes
  l9_results.json               — full attribution scores per contract

EXIT CODES
──────────
    0  pass criteria met for reentrancy contracts
    1  criteria not met or load error
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

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    load_model,
    add_common_args,
    CLASS_NAMES,
    get_node_type_tensor,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Node type labels ───────────────────────────────────────────────────────────

_NODE_TYPE_LABELS = {
    0:  "UNKNOWN",
    1:  "FUNCTION",
    2:  "MODIFIER",
    3:  "STATE_VAR",
    4:  "FALLBACK",
    5:  "RECEIVE",
    6:  "CONSTRUCTOR",
    7:  "EVENT",
    8:  "CFG_NODE_CALL",
    9:  "CFG_NODE_WRITE",
    10: "CFG_NODE_OTHER",
    11: "CFG_NODE_CHECK",
    12: "CONTRACT",
}

_CALL_TYPE  = 8
_WRITE_TYPE = 9

# Contracts to analyse
_DEFAULT_CONTRACTS = [
    "reentrancy_vulnerable.sol",
    "reentrancy_safe.sol",
    "inheritance_propagation.sol",
]

_DEFAULT_CONTRACTS_DIR = Path(__file__).parent / "test_contracts"


# ── Graph extraction ───────────────────────────────────────────────────────────

def extract_graph(sol_path: Path):
    from ml.src.preprocessing.graph_extractor import (
        extract_contract_graph,
        GraphExtractionConfig,
    )
    config = GraphExtractionConfig(include_edge_attr=True)
    return extract_contract_graph(sol_path, config=config)


# ── Attention rollout ─────────────────────────────────────────────────────────

def _build_attention_matrix(
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """
    Build a dense [N, N] attention matrix from GATConv output.

    GATConv returns alpha as [E, H] (H heads) or [E] (1 head).
    We average over heads to get [E] scalar weights.

    A[dst, src] = attention weight for src→dst.
    """
    if alpha.dim() == 2:
        alpha_mean = alpha.mean(dim=1)  # [E]
    else:
        alpha_mean = alpha              # [E]

    # Normalise within each destination (softmax already applied by GATConv,
    # but numerical noise can accumulate — re-normalise for safety)
    src, dst = edge_index[0], edge_index[1]
    A = torch.zeros(n_nodes, n_nodes, device=alpha.device)
    A[dst, src] = alpha_mean.float()

    # Row-normalise: each dst's incoming weights sum to 1
    row_sums = A.sum(dim=1, keepdim=True).clamp(min=1e-9)
    A = A / row_sums

    return A  # [N, N]


def rollout_two_layers(
    A1: torch.Tensor,
    A2: torch.Tensor,
    residual_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combine two attention matrices with residual identity and compute rollout.

    A_aug = (1-w)*A + w*I for each layer, then rollout = A1_aug @ A2_aug.
    """
    n = A1.shape[0]
    I = torch.eye(n, device=A1.device)

    A1_aug = (1.0 - residual_weight) * A1 + residual_weight * I
    A2_aug = (1.0 - residual_weight) * A2 + residual_weight * I

    # Re-normalise rows after adding identity
    def _row_norm(M):
        s = M.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return M / s

    A1_aug = _row_norm(A1_aug)
    A2_aug = _row_norm(A2_aug)

    rollout = A1_aug @ A2_aug  # [N, N]
    return rollout


def compute_rollout_attribution(
    model,
    graph,
    device: str,
) -> dict:
    """
    Run Phase 2 (conv3) + Phase 3 (conv4) with return_attention_weights=True.

    Returns:
        dict with keys:
            attribution:     np.ndarray [N] — attribution score per node
            node_types:      np.ndarray [N] — integer node type IDs
            top10_indices:   list[int] — top-10 node indices by attribution
            top10_types:     list[str] — type labels for top-10 nodes
            n_call_write_in_top10: int
    """
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([graph]).to(device)
    edge_attr = getattr(batch, "edge_attr", None)
    x_in  = batch.x.float()
    ei    = batch.edge_index
    ea    = edge_attr
    n     = x_in.shape[0]

    gnn = model.gnn

    # ── Forward through Phase 1 (no attention capture) ───────────────────────
    with torch.no_grad():
        # Embed edge types for all layers
        edge_emb = None
        if gnn.use_edge_attr and ea is not None:
            edge_emb = gnn.edge_embedding(ea.clamp(0, gnn.edge_embedding.num_embeddings - 1))

        # Phase 1 — conv1 + skip + conv2
        x_h = x_in
        if edge_emb is not None:
            x1, (ei1, alpha1) = gnn.conv1(x_h, ei, edge_attr=edge_emb,
                                           return_attention_weights=True)
        else:
            x1, (ei1, alpha1) = gnn.conv1(x_h, ei, return_attention_weights=True)

        # IMP-G2 skip connection (Linear(input_dim, hidden_dim))
        skip1 = gnn.phase1_skip(x_h) if hasattr(gnn, "phase1_skip") else 0
        x1_activated = torch.relu(x1 + skip1)

        if edge_emb is not None:
            x2, _ = gnn.conv2(x1_activated, ei, edge_attr=edge_emb,
                              return_attention_weights=True)
        else:
            x2, _ = gnn.conv2(x1_activated, ei, return_attention_weights=True)

        x2 = gnn.phase_norm[0](x2)  # Phase 1 LayerNorm

        # ── Phase 2 — conv3 (CF only) — CAPTURE ATTENTION ────────────────────
        # Build CF edge mask
        cf_mask = torch.zeros(ea.shape[0], dtype=torch.bool, device=device)
        for etype in (gnn.phase2_edge_types or [6, 8, 9]):
            cf_mask |= (ea == etype)

        ei_cf  = ei[:, cf_mask]
        ea_cf  = edge_emb[cf_mask] if edge_emb is not None else None

        if ei_cf.shape[1] > 0:
            if ea_cf is not None:
                x3, (ei3, alpha3) = gnn.conv3(x2, ei_cf, edge_attr=ea_cf,
                                               return_attention_weights=True)
            else:
                x3, (ei3, alpha3) = gnn.conv3(x2, ei_cf, return_attention_weights=True)
            A_phase2 = _build_attention_matrix(ei3, alpha3, n)
        else:
            log.warning("No CF edges found — using identity for Phase 2 attention.")
            A_phase2 = torch.eye(n, device=device)
            x3 = x2

        # Run remaining Phase 2 layers (conv3b, conv3c) without attention capture
        icfg_mask = (ea == 8) | (ea == 9)
        ei_icfg   = ei[:, icfg_mask]
        ea_icfg   = edge_emb[icfg_mask] if edge_emb is not None else None

        if ei_icfg.shape[1] > 0:
            if ea_icfg is not None:
                x3b, _ = gnn.conv3b(x3, ei_icfg, edge_attr=ea_icfg,
                                    return_attention_weights=True)
            else:
                x3b, _ = gnn.conv3b(x3, ei_icfg, return_attention_weights=True)
        else:
            x3b = x3

        ph2_ei_mask = cf_mask | icfg_mask
        ei_ph2      = ei[:, ph2_ei_mask]
        ea_ph2      = edge_emb[ph2_ei_mask] if edge_emb is not None else None
        x3b_in      = x3 + x3b  # simple residual

        if ei_ph2.shape[1] > 0:
            if ea_ph2 is not None:
                x3c, _ = gnn.conv3c(x3b_in, ei_ph2, edge_attr=ea_ph2,
                                    return_attention_weights=True)
            else:
                x3c, _ = gnn.conv3c(x3b_in, ei_ph2, return_attention_weights=True)
        else:
            x3c = x3b_in

        x3c = gnn.phase_norm[1](x3c)  # Phase 2 LayerNorm

        # ── Phase 3 — conv4 (REVERSE_CONTAINS up) — CAPTURE ATTENTION ────────
        rc_type = None
        for name, val in vars(gnn).items():
            if "reverse_contains" in name.lower() or (
                hasattr(gnn, "_phase3_ei") or hasattr(gnn, "phase3_contains_edge_type")
            ):
                pass

        # Build REVERSE_CONTAINS mask (edge type 5 = CONTAINS; reversed → CFG→FUNCTION)
        # In v8 schema, CONTAINS is type 5; REVERSE_CONTAINS is built by flipping.
        # The GNNEncoder uses a pre-built reverse edge index stored as a buffer or
        # computed on-the-fly. We replicate the same logic:
        contains_mask = (ea == 5)
        if contains_mask.any():
            ei_rev = ei[:, contains_mask].flip(0)  # flip src↔dst
            ea_rev = edge_emb[contains_mask] if edge_emb is not None else None

            if ea_rev is not None:
                x4, (ei4, alpha4) = gnn.conv4(x3c, ei_rev, edge_attr=ea_rev,
                                               return_attention_weights=True)
            else:
                x4, (ei4, alpha4) = gnn.conv4(x3c, ei_rev, return_attention_weights=True)
            A_phase3 = _build_attention_matrix(ei4, alpha4, n)
        else:
            log.warning("No CONTAINS edges found — using identity for Phase 3 attention.")
            A_phase3 = torch.eye(n, device=device)
            x4 = x3c

    # ── Rollout ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        rollout = rollout_two_layers(A_phase2, A_phase3)  # [N, N]

    # Identify pool nodes (FUNCTION-like)
    node_type_ids = get_node_type_tensor(batch).cpu().numpy()
    func_types    = {1, 2, 4, 5, 6}
    pool_nodes    = [i for i, t in enumerate(node_type_ids) if t in func_types]

    if not pool_nodes:
        log.warning("No function-level nodes — using node 0 as pool node.")
        pool_nodes = [0]

    # Average rollout attribution over all pool nodes
    rollout_np = rollout.cpu().numpy()
    attribution = rollout_np[pool_nodes, :].mean(axis=0)  # [N]

    # Top-10
    top10_idx   = np.argsort(attribution)[::-1][:10].tolist()
    top10_types = [_NODE_TYPE_LABELS.get(int(node_type_ids[i]), "UNKNOWN")
                   for i in top10_idx]

    n_call_write = sum(
        1 for t in node_type_ids[top10_idx]
        if int(t) in (_CALL_TYPE, _WRITE_TYPE)
    )

    # Mean attribution score across ALL CALL/WRITE nodes (not just top-10).
    # Used by the relative-rank criterion: vulnerable should score higher than safe.
    cw_mask = np.isin(node_type_ids, [_CALL_TYPE, _WRITE_TYPE])
    mean_cw_score = float(attribution[cw_mask].mean()) if cw_mask.any() else 0.0

    return {
        "attribution":             attribution.tolist(),
        "node_types":              node_type_ids.tolist(),
        "top10_indices":           top10_idx,
        "top10_types":             top10_types,
        "top10_scores":            [round(float(attribution[i]), 5) for i in top10_idx],
        "n_call_write_in_top10":   n_call_write,
        "mean_cw_attribution":     round(mean_cw_score, 6),
        "n_nodes":                 n,
        "n_pool_nodes":            len(pool_nodes),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attention rollout attribution — Layer 3, P2"
    )
    add_common_args(parser, require_checkpoint=True)
    parser.add_argument(
        "--contracts-dir",
        default=str(_DEFAULT_CONTRACTS_DIR),
        dest="contracts_dir",
        help="Directory containing test .sol contracts.",
    )
    parser.add_argument(
        "--contracts",
        nargs="+",
        default=_DEFAULT_CONTRACTS,
        help="List of .sol filenames to analyse (looked up in contracts-dir).",
    )
    # Remove n_contracts from common args — not used here
    args = parser.parse_args()

    contracts_dir = Path(args.contracts_dir)
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

    results = []
    text_lines = []
    text_lines.append("=" * 70)
    text_lines.append("ATTENTION ROLLOUT REPORT — SENTINEL GNN (Phase2 + Phase3)")
    text_lines.append("Note: Full 8-layer rollout requires model modification.")
    text_lines.append("      This script captures Phase 2 (conv3) and Phase 3 (conv4) only.")
    text_lines.append("=" * 70)

    reentrancy_pass_list = []

    for sol_file in args.contracts:
        sol_path = contracts_dir / sol_file
        if not sol_path.exists():
            log.warning(f"Contract not found: {sol_path}")
            results.append({"file": sol_file, "error": "file not found"})
            continue

        log.info(f"Processing: {sol_file}")
        try:
            graph = extract_graph(sol_path)
        except Exception as exc:
            log.error(f"Extraction failed for {sol_file}: {exc}")
            results.append({"file": sol_file, "error": str(exc)})
            continue

        try:
            rollout_result = compute_rollout_attribution(model, graph, args.device)
        except Exception as exc:
            log.error(f"Rollout failed for {sol_file}: {exc}")
            results.append({"file": sol_file, "error": str(exc)})
            continue

        r = {"file": sol_file, **rollout_result}
        results.append(r)

        # Format text report
        text_lines.append(f"\nContract: {sol_file}")
        text_lines.append(f"  Nodes: {rollout_result['n_nodes']}  Pool nodes: {rollout_result['n_pool_nodes']}")
        text_lines.append("  Top-10 nodes by rollout attribution:")
        for rank, (idx, typ, score) in enumerate(zip(
            rollout_result["top10_indices"],
            rollout_result["top10_types"],
            rollout_result["top10_scores"],
        )):
            text_lines.append(f"    #{rank+1:2d}  node={idx:4d}  type={typ:<20}  score={score:.5f}")

        n_cw = rollout_result["n_call_write_in_top10"]
        mean_cw = rollout_result["mean_cw_attribution"]
        text_lines.append(
            f"  CFG_NODE_CALL/WRITE in top-10: {n_cw}/10  "
            f"  mean CW attribution: {mean_cw:.5f}"
        )

    text_lines.append("\n" + "=" * 70)

    # ── Relative-rank criterion (replaces absolute count) ──────────────────
    # For each vulnerable/safe pair, PASS = vulnerable's mean CALL/WRITE
    # attribution is strictly higher than the paired safe contract's.
    # Pairing: strip "_vulnerable" / "_safe" suffix to find matching base name.
    result_by_file = {r["file"]: r for r in results if "error" not in r}
    pair_results: list[dict] = []

    for sol_file, r in result_by_file.items():
        stem = sol_file.replace(".sol", "")
        if "_vulnerable" not in stem:
            continue
        base   = stem.replace("_vulnerable", "")
        safe_f = f"{base}_safe.sol"
        if safe_f not in result_by_file:
            continue
        rs = result_by_file[safe_f]
        vuln_score = r["mean_cw_attribution"]
        safe_score = rs["mean_cw_attribution"]
        pair_pass  = vuln_score > safe_score
        pair_results.append({
            "base":       base,
            "vuln_file":  sol_file,
            "safe_file":  safe_f,
            "vuln_mean_cw": vuln_score,
            "safe_mean_cw": safe_score,
            "delta":      round(vuln_score - safe_score, 6),
            "pair_pass":  pair_pass,
        })
        text_lines.append(
            f"  Pair [{base}]: vuln_CW={vuln_score:.5f}  safe_CW={safe_score:.5f}"
            f"  delta={vuln_score - safe_score:+.5f}  "
            + ("PASS" if pair_pass else "FAIL")
        )

    if pair_results:
        overall_pass = all(p["pair_pass"] for p in pair_results)
        n_pass = sum(p["pair_pass"] for p in pair_results)
        text_lines.append(
            f"\nOVERALL PASS: {overall_pass}  "
            f"({n_pass}/{len(pair_results)} vuln/safe pairs: vuln CW attribution > safe)"
        )
    else:
        overall_pass = False
        log.warning("No vulnerable/safe pairs processed — cannot evaluate relative-rank criterion.")
        text_lines.append("OVERALL PASS: False  (no vuln/safe pairs found)")

    text_lines.append("=" * 70)

    report_text = "\n".join(text_lines)
    print(report_text)

    if out_dir:
        txt_path = out_dir / "attention_rollout_report.txt"
        txt_path.write_text(report_text)
        log.info(f"Text report saved: {txt_path}")

    # JSON output (exclude full attribution arrays to keep file small)
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != "attribution" and k != "node_types"}
        json_results.append(jr)

    report = {
        "experiment":   "exp_l9_attention_rollout",
        "layer":        3,
        "priority":     2,
        "pass_criteria": (
            "For each vulnerable/safe contract pair: mean attribution of CALL/WRITE nodes "
            "is higher in the vulnerable contract than in the safe contract."
        ),
        "note": (
            "Phase 2 (conv3 CF-only) and Phase 3 (conv4 REVERSE_CONTAINS) only. "
            "Full 8-layer rollout requires model instrumentation."
        ),
        "overall_pass":  overall_pass,
        "pair_results":  pair_results,
        "contracts":     json_results,
    }

    if out_dir:
        json_path = out_dir / "l9_results.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"JSON results saved: {json_path}")
    else:
        print(json.dumps(report, indent=2))

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
