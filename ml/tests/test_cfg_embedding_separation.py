"""
test_cfg_embedding_separation.py — Pre-flight GNN architecture validation.

NON-NEGOTIABLE GATE: This test must pass before any production re-extraction,
training, or evaluation begins. If it fails, the GNN architecture cannot
structurally distinguish call-before-write from write-before-call — the core
reentrancy detection task. Fix the extractor or GNN before proceeding.

WHAT IS TESTED
──────────────
Contract A: call BEFORE write  (vulnerable — classic reentrancy).
Contract B: write BEFORE call  (safe — CEI pattern).

These are the exact contracts from the original root cause analysis.

With the correct three-phase GNN architecture:
  - CFG_NODE_CALL (type 8) and CFG_NODE_WRITE (type 9) have distinct type_id
    → different initial embeddings before any message passing.
  - Phase 2 CONTROL_FLOW edges (type 6) encode "call precedes write" vs
    "write precedes call" in the CFG node embeddings.
  - Phase 3 reverse-CONTAINS edges aggregate this order signal UP into the
    FUNCTION node.
→ The `withdraw` function node must have meaningfully different embeddings.

WHY FUNCTION NODE, NOT MEAN POOL
─────────────────────────────────
Both contracts share ~13 identical nodes (CONTRACT, STATE_VAR, require node,
etc.). After global_mean_pool over all nodes, the 2-node difference is diluted
to ~0.92 cosine similarity — easily passing a 0.95 threshold even with useless
Phase 2 layers. Comparing the function-level node directly tests exactly what
Phase 3 was designed to do: propagate order information INTO the function node.

WHY FIXED SEED + THRESHOLD 0.85
─────────────────────────────────
torch.manual_seed(42) makes the test deterministic and reproducible across runs.
A threshold of 0.85 is tight enough to require meaningful structural separation
(random 128-dim vectors have cosine ~0.0 ± 0.1; structurally similar graphs may
correlate more by chance). Do NOT change the seed — if the test fails with
seed 42, the architecture is broken; fix the architecture.

DIAGNOSE IN THIS ORDER IF THE TEST FAILS
─────────────────────────────────────────
1. Verify CFG_NODE_CALL (type 8) and CFG_NODE_WRITE (type 9) are correctly
   assigned by _cfg_node_type() in graph_extractor.py.
2. Verify CONTROL_FLOW edges (type 6) exist in both graphs.
3. Verify Phase 2 GATConv (conv3) uses add_self_loops=False — self-loops
   destroy the directional signal CONTROL_FLOW is meant to encode.
4. Verify Phase 3 reverse-CONTAINS edges are built correctly (flip(0) on
   CONTAINS edge_index) and conv4 also uses add_self_loops=False.
5. Do NOT change the seed to make the test pass — that masks the problem.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# _find_function_node helper
# ─────────────────────────────────────────────────────────────────────────────

def _find_function_node(graph, func_name: str) -> int:
    """
    Return the graph node index of the named function node.

    Requires graph.node_metadata to be set (populated during extraction per §2.2).
    Uses suffix-matching so both "withdraw" and "A.withdraw" match "withdraw".

    Raises ValueError (with available names) if not found.
    Raises AttributeError if graph.node_metadata is missing — fix extraction first.
    """
    from ml.src.preprocessing.graph_schema import NODE_TYPES

    for i, type_id in enumerate(graph.x[:, 0].tolist()):
        if int(type_id) == NODE_TYPES["FUNCTION"]:
            meta_name = graph.node_metadata[i].get("name", "")
            if meta_name == func_name or meta_name.split(".")[-1] == func_name:
                return i

    # Build error message before raising (avoids f-string SyntaxError on
    # list comprehensions split across string literals).
    available_names = [
        graph.node_metadata[i]["name"]
        for i in range(graph.x.shape[0])
        if int(graph.x[i, 0]) == NODE_TYPES["FUNCTION"]
    ]
    raise ValueError(
        f"Function '{func_name}' not found in graph. "
        f"Available function nodes: {available_names}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_reentrancy_embedding_separation():
    """
    The GNN must produce meaningfully different embeddings for the `withdraw`
    function node in a call-before-write contract vs a write-before-call contract,
    even with a randomly-initialised model (no trained weights).

    Cosine similarity < 0.85 is required. A value ≥ 0.85 means the GNN cannot
    structurally distinguish the order — the architecture is broken.
    """
    slither_pkg = pytest.importorskip("slither", reason="requires slither-analyzer")

    import tempfile
    from pathlib import Path
    from ml.src.preprocessing.graph_extractor import extract_contract_graph
    from ml.src.models.gnn_encoder import GNNEncoder

    contract_a = """
    pragma solidity ^0.8.0;
    contract A {
        mapping(address => uint) public balances;
        function withdraw(uint amount) external {
            require(balances[msg.sender] >= amount);
            (bool ok,) = msg.sender.call{value: amount}("");
            balances[msg.sender] -= amount;
        }
    }
    """

    contract_b = """
    pragma solidity ^0.8.0;
    contract B {
        mapping(address => uint) public balances;
        function withdraw(uint amount) external {
            require(balances[msg.sender] >= amount);
            balances[msg.sender] -= amount;
            (bool ok,) = msg.sender.call{value: amount}("");
        }
    }
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = Path(tmpdir) / "contract_a.sol"
        path_b = Path(tmpdir) / "contract_b.sol"
        path_a.write_text(contract_a)
        path_b.write_text(contract_b)

        graph_a = extract_contract_graph(path_a)
        graph_b = extract_contract_graph(path_b)

    # Verify extraction produced CFG nodes
    type_ids_a = graph_a.x[:, 0].int().tolist()
    type_ids_b = graph_b.x[:, 0].int().tolist()

    from ml.src.preprocessing.graph_schema import NODE_TYPES, EDGE_TYPES
    assert NODE_TYPES["CFG_NODE_CALL"] in type_ids_a, (
        "Contract A (call-before-write) must have a CFG_NODE_CALL (type 8) node. "
        "Check _cfg_node_type() in graph_extractor.py."
    )
    assert NODE_TYPES["CFG_NODE_WRITE"] in type_ids_a, (
        "Contract A must have a CFG_NODE_WRITE (type 9) node."
    )
    assert (graph_a.edge_attr == EDGE_TYPES["CONTROL_FLOW"]).any(), (
        "Contract A must have CONTROL_FLOW edges (type 6). "
        "Check _build_control_flow_edges() in graph_extractor.py."
    )
    assert (graph_b.edge_attr == EDGE_TYPES["CONTROL_FLOW"]).any(), (
        "Contract B must have CONTROL_FLOW edges (type 6)."
    )

    # Fixed seed — do not change; fix the architecture if the test fails
    torch.manual_seed(42)
    gnn = GNNEncoder()
    gnn.eval()

    with torch.no_grad():
        node_embs_a, _ = gnn(
            graph_a.x, graph_a.edge_index, graph_a.batch, graph_a.edge_attr
        )
        node_embs_b, _ = gnn(
            graph_b.x, graph_b.edge_index, graph_b.batch, graph_b.edge_attr
        )

    withdraw_a_idx = _find_function_node(graph_a, "withdraw")
    withdraw_b_idx = _find_function_node(graph_b, "withdraw")

    emb_a = node_embs_a[withdraw_a_idx].unsqueeze(0)   # [1, 128]
    emb_b = node_embs_b[withdraw_b_idx].unsqueeze(0)   # [1, 128]

    cosine_sim = F.cosine_similarity(emb_a, emb_b).item()

    assert cosine_sim < 0.85, (
        f"\n{'!'*70}\n"
        f"PRE-FLIGHT TEST FAILED\n"
        f"{'!'*70}\n"
        f"cosine_similarity(withdraw_A_emb, withdraw_B_emb) = {cosine_sim:.4f}\n"
        f"Threshold: < 0.85\n\n"
        f"The GNN cannot distinguish call-before-write (Contract A) from\n"
        f"write-before-call (Contract B) at the function-node level.\n"
        f"Do NOT proceed to re-extraction or training.\n\n"
        f"Diagnose in this order:\n"
        f"  1. Check CFG_NODE_CALL (8) and CFG_NODE_WRITE (9) assignment in\n"
        f"     _cfg_node_type() — Contract A must have call node BEFORE write node.\n"
        f"  2. Check CONTROL_FLOW edges (type 6) exist in both graphs.\n"
        f"  3. Check Phase 2 GATConv (conv3): add_self_loops must be False.\n"
        f"  4. Check Phase 3 reverse-CONTAINS: edge_index must be flipped (0).\n"
        f"     conv4 must also have add_self_loops=False.\n"
        f"  5. Do NOT change the seed (42) — fix the architecture.\n"
        f"{'!'*70}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 intermediate test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_phase3_changes_function_node_embedding():
    """
    After Phase 3 (reverse-CONTAINS), the withdraw function node embedding
    must differ from its Phase 2 value — i.e., Phase 3 messages are non-zero
    and are integrated into the function node.

    This verifies that the reverse aggregation path is wired correctly:
    if after_phase3[func_idx] == after_phase2[func_idx] for ALL contracts,
    Phase 3 is a no-op and the architecture is broken.
    """
    slither_pkg = pytest.importorskip("slither", reason="requires slither-analyzer")

    import tempfile
    from pathlib import Path
    from ml.src.preprocessing.graph_extractor import extract_contract_graph
    from ml.src.models.gnn_encoder import GNNEncoder

    contract = """
    pragma solidity ^0.8.0;
    contract A {
        mapping(address => uint) public balances;
        function withdraw(uint amount) external {
            require(balances[msg.sender] >= amount);
            (bool ok,) = msg.sender.call{value: amount}("");
            balances[msg.sender] -= amount;
        }
    }
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "a.sol"
        path.write_text(contract)
        graph = extract_contract_graph(path)

    torch.manual_seed(42)
    gnn = GNNEncoder()
    gnn.eval()

    with torch.no_grad():
        _, _, intermediates = gnn(
            graph.x, graph.edge_index, graph.batch, graph.edge_attr,
            return_intermediates=True
        )

    func_idx = _find_function_node(graph, "withdraw")

    emb_after_phase2 = intermediates["after_phase2"][func_idx]
    emb_after_phase3 = intermediates["after_phase3"][func_idx]

    changed = (emb_after_phase2 != emb_after_phase3).any().item()
    assert changed, (
        "Phase 3 (reverse-CONTAINS) did not change the 'withdraw' function node "
        "embedding — Phase 3 messages are zero or not reaching the function node. "
        "Check: (1) rev_contains_ei is built correctly with flip(0); "
        "(2) conv4 uses add_self_loops=False; "
        "(3) the function node appears as a target in rev_contains_ei."
    )
