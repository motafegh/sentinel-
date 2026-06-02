"""
exp_l6_counterfactual_contracts.py — Layer 3, P1: Counterfactual Contract Testing

PURPOSE
───────
Run paired vulnerable/safe minimal Solidity contracts through the full GNN
pipeline and verify that the model assigns higher probability to the expected
vulnerability class for the vulnerable contract than for the structurally
equivalent safe contract.

This is the strongest model-level sanity check: if the model cannot distinguish
a minimal CEI violation from a correct CEI implementation, the feature extraction
or training signal has a fundamental defect.

LAYER / PRIORITY
─────────────────
Layer 3, Priority 1 — Counterfactual model validity.

APPROACH: GNN-only inference
─────────────────────────────
Full pipeline inference requires tokenisation (GraphCodeBERT window tokeniser).
To keep this script self-contained and runnable without the full BERT dependency,
we use GNN-only inference via model.gnn + model.gnn_eye_proj + model.aux_gnn.
This mirrors what aux_gnn measures during training.

GNN-only inference path:
  graph → GNNEncoder → function-level pool → gnn_eye_proj → aux_gnn → sigmoid

PASS CRITERIA
─────────────
For each test pair, the vulnerable contract must score HIGHER than the safe
contract on the expected vulnerability class index (sigmoid probability).
All 4 pairs must pass (4/4).

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_l6_counterfactual_contracts.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --out ml/logs/interpretability/l6_counterfactual.json

    # Optional: point to a different contracts directory
    PYTHONPATH=. python ml/scripts/interpretability/exp_l6_counterfactual_contracts.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --contracts-dir ml/scripts/interpretability/test_contracts \\
        --out ml/logs/interpretability/l6_counterfactual.json

OUTPUT
──────
Per-pair comparison table printed to stdout.
JSON report with scores and pass/fail per pair.
Exit 0 if all pairs pass, exit 1 if any pair fails.

EXIT CODES
──────────
    0  all 4 pairs pass
    1  one or more pairs fail (or extraction error)
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

# ── Test pair definitions ──────────────────────────────────────────────────────

TEST_PAIRS = [
    {
        "name": "CEI_reentrancy",
        "vulnerable": "reentrancy_vulnerable.sol",
        "safe": "reentrancy_safe.sol",
        "expected_class_idx": 6,
        "expected_class_name": "Reentrancy",
    },
    {
        "name": "integer_uo",
        "vulnerable": "integer_uo_vulnerable.sol",
        "safe": "integer_uo_safe.sol",
        "expected_class_idx": 4,
        "expected_class_name": "IntegerUO",
    },
    {
        "name": "timestamp",
        "vulnerable": "timestamp_vulnerable.sol",
        "safe": "timestamp_safe.sol",
        "expected_class_idx": 7,
        "expected_class_name": "Timestamp",
    },
    {
        "name": "unused_return",
        "vulnerable": "unused_return_vulnerable.sol",
        "safe": "unused_return_safe.sol",
        "expected_class_idx": 9,
        "expected_class_name": "UnusedReturn",
    },
]

# ── Contract directories to search ────────────────────────────────────────────

_DEFAULT_CONTRACTS_DIRS = [
    Path(__file__).parent / "test_contracts",
    Path(__file__).parent.parent / "test_contracts",
]

# ── Graph extraction ───────────────────────────────────────────────────────────

def extract_graph(sol_path: Path):
    """
    Extract a PyG Data graph from a Solidity file using the SENTINEL graph extractor.

    Returns:
        PyG Data object with .x, .edge_index, .edge_attr

    Raises:
        RuntimeError: if Slither is not available or compilation fails.
    """
    from ml.src.preprocessing.graph_extractor import (
        extract_contract_graph,
        GraphExtractionConfig,
    )
    import shutil
    import sys
    _solc = shutil.which("solc") or str(Path(sys.executable).parent / "solc")
    config = GraphExtractionConfig(include_edge_attr=True, solc_binary=_solc)
    graph = extract_contract_graph(sol_path, config=config)
    return graph


# ── GNN-only inference ─────────────────────────────────────────────────────────

def gnn_only_predict(model, graph, device: str) -> torch.Tensor:
    """
    Run GNN-only inference: GNNEncoder → function-level pool → gnn_eye_proj → aux_gnn.

    This replicates the GNN eye path from SentinelModel.forward() without requiring
    the transformer tokeniser.

    Returns:
        Tensor [10] — sigmoid probabilities per class.
    """
    from torch_geometric.data import Batch
    from torch_geometric.nn import global_mean_pool, global_max_pool

    gnn = model.gnn
    gnn_eye_proj = model.gnn_eye_proj
    aux_gnn = model.aux_gnn

    batch = Batch.from_data_list([graph]).to(device)
    edge_attr = getattr(batch, "edge_attr", None)

    with torch.no_grad():
        x_out, b, _ = gnn(
            batch.x.float(),
            batch.edge_index,
            batch.batch,
            edge_attr,
        )
        # Pool over FUNCTION-like nodes (type IDs 1,2,4,5,6 = FUNCTION/MODIFIER/
        # FALLBACK/RECEIVE/CONSTRUCTOR), matching SentinelModel forward logic.
        node_types = get_node_type_tensor(batch)
        func_ids = torch.tensor([1, 2, 4, 5, 6], device=device)
        func_mask = torch.isin(node_types, func_ids)

        if func_mask.any():
            pool_embs = x_out[func_mask]
            pool_batch = b[func_mask]
        else:
            # Ghost graph fallback: no function-level nodes
            log.warning("No function-level nodes found — using all-node pool (ghost graph).")
            pool_embs = x_out
            pool_batch = b

        num_graphs = int(b.max().item()) + 1
        gnn_max  = global_max_pool(pool_embs, pool_batch, size=num_graphs)   # [1, H]
        gnn_mean = global_mean_pool(pool_embs, pool_batch, size=num_graphs)  # [1, H]
        gnn_eye  = gnn_eye_proj(torch.cat([gnn_max, gnn_mean], dim=1))       # [1, eye_dim]
        logits   = aux_gnn(gnn_eye)                                           # [1, 10]
        probs    = torch.sigmoid(logits).squeeze(0)                           # [10]

    return probs.cpu()


# ── Main logic ────────────────────────────────────────────────────────────────

def find_contract(filename: str, contracts_dirs: list[Path]) -> Optional[Path]:
    """Search contract directories in order and return the first match."""
    for d in contracts_dirs:
        p = d / filename
        if p.exists():
            return p
    return None


def run_pair(pair: dict, model, device: str, contracts_dirs: list[Path]) -> dict:
    """
    Extract graphs and run GNN-only inference for one vulnerable/safe pair.

    Returns a result dict with scores, delta, and pass/fail status.
    """
    name        = pair["name"]
    cls_idx     = pair["expected_class_idx"]
    cls_name    = pair["expected_class_name"]

    result = {
        "name":               name,
        "expected_class_idx": cls_idx,
        "expected_class_name": cls_name,
        "vulnerable_score":   None,
        "safe_score":         None,
        "delta":              None,
        "pass":               False,
        "error":              None,
    }

    vuln_path = find_contract(pair["vulnerable"], contracts_dirs)
    safe_path = find_contract(pair["safe"], contracts_dirs)

    if vuln_path is None or safe_path is None:
        missing = []
        if vuln_path is None:
            missing.append(pair["vulnerable"])
        if safe_path is None:
            missing.append(pair["safe"])
        result["error"] = f"Contract files not found: {missing}"
        log.error(result["error"])
        return result

    log.info(f"[{name}] extracting vulnerable: {vuln_path.name}")
    try:
        vuln_graph = extract_graph(vuln_path)
    except Exception as exc:
        result["error"] = f"Extraction failed for {vuln_path.name}: {exc}"
        log.error(result["error"])
        return result

    log.info(f"[{name}] extracting safe: {safe_path.name}")
    try:
        safe_graph = extract_graph(safe_path)
    except Exception as exc:
        result["error"] = f"Extraction failed for {safe_path.name}: {exc}"
        log.error(result["error"])
        return result

    log.info(f"[{name}] running GNN-only inference")
    try:
        vuln_probs = gnn_only_predict(model, vuln_graph, device)
        safe_probs = gnn_only_predict(model, safe_graph, device)
    except Exception as exc:
        result["error"] = f"Inference failed for {name}: {exc}"
        log.error(result["error"])
        return result

    v_score = float(vuln_probs[cls_idx])
    s_score = float(safe_probs[cls_idx])
    delta   = v_score - s_score

    result["vulnerable_score"] = round(v_score, 4)
    result["safe_score"]       = round(s_score, 4)
    result["delta"]            = round(delta, 4)
    result["pass"]             = bool(v_score > s_score)

    # Also store all class scores for debugging
    result["vuln_all_scores"] = {CLASS_NAMES[i]: round(float(vuln_probs[i]), 4)
                                 for i in range(len(CLASS_NAMES))}
    result["safe_all_scores"] = {CLASS_NAMES[i]: round(float(safe_probs[i]), 4)
                                 for i in range(len(CLASS_NAMES))}

    status = "PASS" if result["pass"] else "FAIL"
    log.info(
        f"[{name}] {cls_name}: vuln={v_score:.4f} safe={s_score:.4f} "
        f"delta={delta:+.4f}  [{status}]"
    )
    return result


def print_summary(results: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Pair':<22} {'Class':<16} {'Vuln':>7} {'Safe':>7} {'Delta':>8} {'Status':>6}"
    )
    print()
    print("=" * len(header))
    print("  COUNTERFACTUAL CONTRACT TEST RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        if r["error"]:
            status = "ERROR"
            vuln_s = safe_s = delta_s = "N/A"
        else:
            status = "PASS" if r["pass"] else "FAIL"
            vuln_s  = f"{r['vulnerable_score']:.4f}"
            safe_s  = f"{r['safe_score']:.4f}"
            delta_s = f"{r['delta']:+.4f}"

        print(
            f"  {r['name']:<20} {r['expected_class_name']:<16} "
            f"{vuln_s:>7} {safe_s:>7} {delta_s:>8} {status:>6}"
        )

    print("-" * len(header))
    n_pass  = sum(1 for r in results if r["pass"])
    n_total = len(results)
    n_error = sum(1 for r in results if r["error"])
    print(
        f"  SUMMARY: {n_pass}/{n_total} pairs pass"
        + (f"  ({n_error} extraction error(s))" if n_error else "")
    )
    print("=" * len(header))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Counterfactual contract testing — Layer 3, P1"
    )
    add_common_args(parser, require_checkpoint=True)
    parser.add_argument(
        "--contracts-dir",
        default=None,
        dest="contracts_dir",
        help=(
            "Directory containing test .sol contracts. "
            "Defaults to ml/scripts/interpretability/test_contracts/, "
            "then ml/scripts/test_contracts/ as fallback."
        ),
    )
    args = parser.parse_args()

    # Build search path for contracts
    contracts_dirs: list[Path] = []
    if args.contracts_dir:
        contracts_dirs.append(Path(args.contracts_dir))
    contracts_dirs.extend(_DEFAULT_CONTRACTS_DIRS)

    log.info(f"Contract search path: {[str(d) for d in contracts_dirs]}")

    # Determine output path
    out_path: Optional[Path] = None
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Run all pairs
    results = []
    for pair in TEST_PAIRS:
        r = run_pair(pair, model, args.device, contracts_dirs)
        results.append(r)

    print_summary(results)

    # Write JSON
    n_pass  = sum(1 for r in results if r["pass"])
    n_total = len(results)
    all_pass = (n_pass == n_total) and all(r["error"] is None for r in results)

    report = {
        "experiment":   "exp_l6_counterfactual_contracts",
        "layer":        3,
        "priority":     1,
        "pass_criteria": "vulnerable_score > safe_score for all pairs",
        "n_pass":       n_pass,
        "n_total":      n_total,
        "overall_pass": all_pass,
        "pairs":        results,
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Results written to: {out_path}")
    else:
        print(json.dumps(report, indent=2))

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
