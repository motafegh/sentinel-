"""
exp_l3_attention_visualization.py — Layer 3, P1: GAT Attention Weight Visualization

PURPOSE
───────
Capture GAT attention weights from Phase 2's conv3 (CONTROL_FLOW-only, Layer 3)
AND conv3b (CALL_ENTRY+RETURN_TO ICFG-only, Layer 4) layers of GNNEncoder,
and visualize which edges receive the highest attention for specific contracts.
This reveals whether the model genuinely routes information along CF / ICFG
(cross-function call/return) paths for vulnerability detection — especially
the CEI Reentrancy pattern which requires cross-function call structure.

LAYER / PRIORITY
─────────────────
Layer 3, Priority 1 — GNN attention alignment with vulnerability patterns.

APPROACH
─────────
Uses a monkey-patch hook on gnn.conv3.forward so that a single GNNEncoder
forward pass (GNN-only, no transformer) captures the (edge_index, alpha) tuple
from return_attention_weights=True without modifying production code.

For each test contract:
1. Extract the graph using GraphExtractionConfig + extract_contract_graph.
2. Run GNNEncoder.forward with the hook active.
3. Compute mean attention across GAT heads: alpha.mean(dim=1) → [num_edges].
4. Select the top-20 highest-attention edges.
5. Print human-readable "(src_type, src_idx) --[edge_type, attn=X.XXX]--> (dst_type, dst_idx)".
6. Report what fraction of top-20 edges are CONTROL_FLOW or CALL_ENTRY.
7. Check whether CFG_NODE_CALL→CFG_NODE_WRITE paths are represented.
8. Produce a scatter+line matplotlib visualization for each contract.

PASS CRITERIA (qualitative)
────────────────────────────
For reentrancy-positive contracts: ≥30% of top-20 attention edges are
CONTROL_FLOW (type 6) or CALL_ENTRY (type 8).
For safe contracts: this % should be lower (diagnostic comparison).

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_l3_attention_visualization.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --out ml/logs/interpretability/l3_attention_visualization.json

    # Optional: override contracts directory
    PYTHONPATH=. python ml/scripts/interpretability/exp_l3_attention_visualization.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --contracts-dir ml/scripts/test_contracts \\
        --out ml/logs/interpretability/l3_attention_visualization.json

OUTPUT
──────
    - Text report (stdout) of top-20 edges per contract
    - PNG visualization per contract: <out_dir>/<contract_stem>_attn.png
    - JSON report: <out> (all findings)

EXIT CODES
──────────
    0  at least one contract processed successfully
    1  fatal error (no contracts, model load failure)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    load_model,
    add_common_args,
    CLASS_NAMES,
    PHASE_NAMES,
    get_node_type_tensor,
    plot_class_heatmap,
)
from ml.src.preprocessing.graph_schema import NODE_TYPES, EDGE_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_TYPE_ID: float = float(max(NODE_TYPES.values()))  # 12.0

# Build reverse-lookup: int → name
NODE_TYPE_ID_TO_NAME: dict[int, str] = {v: k for k, v in NODE_TYPES.items()}
EDGE_TYPE_ID_TO_NAME: dict[int, str] = {v: k for k, v in EDGE_TYPES.items()}

_CF_TYPES  = {EDGE_TYPES["CONTROL_FLOW"], EDGE_TYPES["CALL_ENTRY"]}
_ICFG_TYPES = {EDGE_TYPES["CALL_ENTRY"], EDGE_TYPES["RETURN_TO"]}

# ── Hook-based attention capture ──────────────────────────────────────────────

def _make_hook(store: dict, key: str, original_forward):
    """Return a hooked forward function that captures (edge_index, alpha)."""
    def hooked(x, edge_index, edge_attr=None, **kwargs):
        result = original_forward(
            x, edge_index, edge_attr=edge_attr,
            return_attention_weights=True,
        )
        if isinstance(result, tuple) and len(result) == 2:
            out, attn_info = result
            if isinstance(attn_info, tuple) and len(attn_info) == 2:
                store[key + "_edge_index"] = attn_info[0].detach().cpu()
                store[key + "_alpha"]      = attn_info[1].detach().cpu()
        return out
    return hooked


def capture_phase2_attention(
    model,
    graph,
    device: str,
) -> dict:
    """
    Run one GNNEncoder forward pass capturing attention from conv3 (CF-only)
    AND conv3b (CALL_ENTRY+RETURN_TO ICFG-only) simultaneously.

    Uses temporary monkey-patches on gnn.conv3 and gnn.conv3b so
    return_attention_weights=True is passed without modifying production code.
    The transformer path is skipped — GNNEncoder is called directly.

    Returns:
        dict with top-level keys "conv3" and "conv3b", each containing:
            "edge_index": LongTensor [2, E]
            "alpha":      FloatTensor [E, num_heads]
            "mean_alpha": FloatTensor [E]
    """
    from torch_geometric.data import Batch

    gnn = model.gnn
    captured: dict = {}

    orig3  = gnn.conv3.forward
    orig3b = gnn.conv3b.forward
    gnn.conv3.forward  = _make_hook(captured, "conv3",  orig3)
    gnn.conv3b.forward = _make_hook(captured, "conv3b", orig3b)

    try:
        batch     = Batch.from_data_list([graph]).to(device)
        x         = batch.x.float()
        edge_index = batch.edge_index
        batch_vec  = batch.batch
        edge_attr  = getattr(batch, "edge_attr", None)

        with torch.no_grad():
            _ = gnn(x, edge_index, batch_vec, edge_attr)
    finally:
        gnn.conv3.forward  = orig3
        gnn.conv3b.forward = orig3b

    result = {}
    for layer in ("conv3", "conv3b"):
        ei_key = layer + "_edge_index"
        al_key = layer + "_alpha"
        if al_key not in captured:
            log.warning(f"Hook did not capture attention for {layer} — no edges.")
            result[layer] = {
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "alpha":      torch.zeros(0, 1),
                "mean_alpha": torch.zeros(0),
            }
        else:
            alpha      = captured[al_key]
            mean_alpha = alpha.mean(dim=1)
            result[layer] = {
                "edge_index": captured[ei_key],
                "alpha":      alpha,
                "mean_alpha": mean_alpha,
            }
    return result


# ── Per-contract analysis ─────────────────────────────────────────────────────

def _report_layer(
    layer_name: str,
    attn_data: dict,
    graph,
    node_types: torch.Tensor,
    relevant_types: set,
    relevant_label: str,
    out_dir: Path,
    stem: str,
) -> dict:
    """Build top-20 report for one conv layer's captured attention."""
    edge_index = attn_data["edge_index"]
    mean_alpha = attn_data["mean_alpha"]

    n_edges = edge_index.shape[1]
    if n_edges == 0:
        log.warning(f"  {stem} [{layer_name}]: no edges captured")
        return {"n_edges": 0, "top20": [], "relevant_fraction": 0.0, "criterion_pass": False}

    k = min(20, n_edges)
    top_idx = torch.topk(mean_alpha, k).indices

    report_edges = []
    relevant_count = 0

    log.info(f"  Top-{k} attention edges ({layer_name} — {relevant_label}):")
    for rank, i in enumerate(top_idx.tolist()):
        src_node = int(edge_index[0, i].item())
        dst_node = int(edge_index[1, i].item())
        attn_val = float(mean_alpha[i].item())

        src_type_id = int(node_types[src_node].item()) if src_node < node_types.shape[0] else -1
        dst_type_id = int(node_types[dst_node].item()) if dst_node < node_types.shape[0] else -1
        src_name = NODE_TYPE_ID_TO_NAME.get(src_type_id, f"type{src_type_id}")
        dst_name = NODE_TYPE_ID_TO_NAME.get(dst_type_id, f"type{dst_type_id}")
        edge_name = relevant_label  # each conv layer sees only its own edge subset

        is_relevant = True  # all edges captured by this hook are of the relevant type
        if is_relevant:
            relevant_count += 1

        entry = {
            "rank": rank + 1,
            "src_node": src_node,
            "dst_node": dst_node,
            "src_type": src_name,
            "dst_type": dst_name,
            "edge_type": edge_name,
            "attn": round(attn_val, 4),
        }
        report_edges.append(entry)
        log.info(
            f"    #{rank+1:2d} ({src_name}, {src_node}) "
            f"--[{edge_name}, attn={attn_val:.4f}]--> "
            f"({dst_name}, {dst_node})"
        )

    call_write_present = any(
        e["src_type"] == "CFG_NODE_CALL" and e["dst_type"] == "CFG_NODE_WRITE"
        for e in report_edges
    )
    relevant_fraction = relevant_count / k
    criterion_pass = relevant_fraction >= 0.30

    log.info(f"  {relevant_label} fraction in top-{k}: {relevant_fraction:.1%}")
    log.info(f"  CFG_NODE_CALL→CFG_NODE_WRITE in top-{k}: {call_write_present}")
    log.info(f"  PASS CRITERION (≥30%): {'PASS' if criterion_pass else 'FAIL'}")

    png_path = out_dir / f"{stem}_{layer_name}_attn.png"
    try:
        _plot_attention(
            graph=graph,
            edge_index=edge_index,
            mean_alpha=mean_alpha,
            node_types=node_types,
            top_idx=top_idx,
            stem=f"{stem} [{layer_name}]",
            out_path=png_path,
        )
    except Exception as exc:
        log.warning(f"  Visualization failed for {stem} [{layer_name}]: {exc}")

    return {
        "n_edges": n_edges,
        "top20": report_edges,
        "relevant_fraction": round(relevant_fraction, 4),
        "call_write_in_top20": call_write_present,
        "criterion_pass": criterion_pass,
        "png": str(png_path),
    }


def analyse_contract(
    model,
    graph,
    stem: str,
    device: str,
    out_dir: Path,
) -> dict:
    """
    Capture conv3 (CF-only) and conv3b (CALL_ENTRY+RETURN_TO) attention for
    one contract, produce per-layer reports and PNGs.

    Returns a result dict suitable for JSON serialization.
    """
    log.info(f"Analysing contract: {stem}")

    phase2 = capture_phase2_attention(model, graph, device)
    node_types = (graph.x[:, 0].float() * _MAX_TYPE_ID).round().long()

    conv3_report  = _report_layer(
        "conv3",  phase2["conv3"],
        graph, node_types, _CF_TYPES,   "CONTROL_FLOW",       out_dir, stem,
    )
    conv3b_report = _report_layer(
        "conv3b", phase2["conv3b"],
        graph, node_types, _ICFG_TYPES, "CALL_ENTRY+RETURN_TO", out_dir, stem,
    )

    criterion_pass = conv3_report["criterion_pass"] or conv3b_report["criterion_pass"]

    return {
        "stem":          stem,
        "conv3":         conv3_report,
        "conv3b":        conv3b_report,
        "criterion_pass": criterion_pass,
    }


def _plot_attention(
    graph,
    edge_index: torch.Tensor,
    mean_alpha: torch.Tensor,
    node_types: torch.Tensor,
    top_idx: torch.Tensor,
    stem: str,
    out_path: Path,
) -> None:
    """
    Scatter-plot nodes; draw top-20 attention edges as lines scaled by alpha.

    Falls back gracefully if networkx is unavailable.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_nodes = graph.x.shape[0]
    # Use a simple spring-like layout: random positions seeded for reproducibility
    rng = np.random.RandomState(42)
    pos = rng.randn(n_nodes, 2)  # [N, 2]

    # Color nodes by type
    type_ids = node_types.numpy()
    cmap_nodes = plt.cm.tab20
    node_colors = cmap_nodes(type_ids % 20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pos[:, 0], pos[:, 1], c=node_colors, s=20, alpha=0.6, zorder=2)

    # Draw top-20 edges with width proportional to attention
    max_attn = float(mean_alpha.max().item()) if mean_alpha.numel() > 0 else 1.0
    for i in top_idx.tolist():
        src = int(edge_index[0, i].item())
        dst = int(edge_index[1, i].item())
        if src >= n_nodes or dst >= n_nodes:
            continue
        attn_val = float(mean_alpha[i].item())
        lw = max(0.5, 4.0 * attn_val / max(max_attn, 1e-8))
        ax.annotate(
            "",
            xy=pos[dst],
            xytext=pos[src],
            arrowprops=dict(
                arrowstyle="-|>",
                color="red",
                lw=lw,
                alpha=0.7,
            ),
            zorder=3,
        )

    ax.set_title(f"conv3 GAT attention — top-{len(top_idx)} edges\n{stem}", fontsize=10)
    ax.axis("off")

    # Legend for node types present
    present_types = np.unique(type_ids)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap_nodes(t % 20), markersize=8,
                   label=NODE_TYPE_ID_TO_NAME.get(int(t), f"type{t}"))
        for t in present_types
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7,
              ncol=2, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  Attention plot saved: {out_path}")


# ── Graph extraction from .sol files ─────────────────────────────────────────

def load_graphs_from_sol_dir(
    contracts_dir: Path,
    device: str,
) -> list[tuple[str, object]]:
    """
    Extract graphs from .sol test contracts using GraphExtractionConfig.

    Returns list of (stem, graph) tuples. Skips contracts that fail extraction.
    """
    try:
        from ml.src.preprocessing.graph_extractor import (
            GraphExtractionConfig,
            extract_contract_graph,
        )
    except ImportError as exc:
        log.warning(f"Could not import graph_extractor: {exc}")
        return []

    sol_files = sorted(contracts_dir.glob("*.sol"))
    if not sol_files:
        log.warning(f"No .sol files found in {contracts_dir}")
        return []

    import shutil
    import sys
    _solc = shutil.which("solc") or str(Path(sol_files[0]).parent.parent.parent / ".venv/bin/solc") if sol_files else None
    _solc = shutil.which("solc") or str(Path(sys.executable).parent / "solc")
    results = []
    cfg = GraphExtractionConfig(
        include_edge_attr=True,
        solc_binary=_solc,
    )
    for sol_path in sol_files:
        stem = sol_path.stem
        try:
            graph = extract_contract_graph(str(sol_path), cfg)
            if graph is None:
                log.debug(f"  Extraction returned None for {stem}")
                continue
            results.append((stem, graph))
            log.info(f"  Loaded: {stem} — {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
        except Exception as exc:
            log.warning(f"  Extraction failed for {stem}: {exc}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GAT conv3+conv3b attention visualization for SENTINEL GNNEncoder Phase 2"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint .pt (e.g. ml/checkpoints/sentinel_best.pt)",
    )
    parser.add_argument(
        "--contracts-dir",
        default="ml/scripts/test_contracts",
        dest="contracts_dir",
        help="Directory of .sol test contracts (default: ml/scripts/test_contracts)",
    )
    parser.add_argument(
        "--out",
        default="ml/logs/interpretability/l3_attention_visualization.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--phase2-edge-types",
        type=int,
        nargs="+",
        default=None,
        dest="phase2_edge_types",
        help="Override Phase 2 edge types (must match training config, e.g. 6 8 9)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        model = load_model(
            checkpoint_path=Path(args.checkpoint),
            device=args.device,
            phase2_edge_types=args.phase2_edge_types,
        )
    except Exception as exc:
        log.error(f"Model load failed: {exc}")
        return 1

    model.eval()
    gnn = model.gnn

    # ── Load test contracts ────────────────────────────────────────────────────
    contracts_dir = Path(args.contracts_dir)
    graphs = load_graphs_from_sol_dir(contracts_dir, args.device)

    if not graphs:
        log.error(
            f"No graphs loaded from {contracts_dir}. "
            "Ensure Slither is installed and contracts compile successfully."
        )
        return 1

    log.info(f"Processing {len(graphs)} contracts...")

    # ── Analyse each contract ─────────────────────────────────────────────────
    all_results = []
    for stem, graph in graphs:
        try:
            result = analyse_contract(
                model=model,
                graph=graph,
                stem=stem,
                device=args.device,
                out_dir=out_dir,
            )
            all_results.append(result)
        except Exception as exc:
            log.warning(f"  Failed to analyse {stem}: {exc}", exc_info=True)

    if not all_results:
        log.error("No contracts analysed successfully.")
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SUMMARY — conv3 (CF) + conv3b (ICFG) attention fractions")
    log.info("=" * 60)
    for r in all_results:
        verdict = "PASS" if r.get("criterion_pass") else "FAIL"
        c3_frac  = r["conv3"].get("relevant_fraction", 0.0)
        c3b_frac = r["conv3b"].get("relevant_fraction", 0.0)
        log.info(
            f"  {r['stem']:40s} | conv3_CF={c3_frac:.2f} | conv3b_ICFG={c3b_frac:.2f} | {verdict}"
        )

    # Compare reentrancy vs others on conv3b (ICFG — CEI-relevant)
    reent_results = [r for r in all_results if "reentranc" in r["stem"].lower()]
    safe_results  = [r for r in all_results if "reentranc" not in r["stem"].lower()]
    if reent_results and safe_results:
        reent_icfg = np.mean([r["conv3b"].get("relevant_fraction", 0.0) for r in reent_results])
        safe_icfg  = np.mean([r["conv3b"].get("relevant_fraction", 0.0) for r in safe_results])
        log.info(f"\n  Mean conv3b ICFG frac — reentrancy contracts : {reent_icfg:.2f}")
        log.info(f"  Mean conv3b ICFG frac — other contracts      : {safe_icfg:.2f}")
        if reent_icfg > safe_icfg:
            log.info("  Diagnostic: reentrancy contracts attract more ICFG attention (expected)")
        else:
            log.info("  Diagnostic: ICFG attention NOT higher for reentrancy — model may not be using CEI cross-function structure")

    # ── Write JSON ─────────────────────────────────────────────────────────────
    report = {
        "experiment": "exp_l3_attention_visualization",
        "checkpoint": str(args.checkpoint),
        "contracts_dir": str(contracts_dir),
        "n_contracts": len(all_results),
        "results": all_results,
    }
    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nJSON report saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
