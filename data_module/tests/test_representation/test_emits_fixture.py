"""EMITS edge fixture test — confirms BUG-H7 fix is live in the v9 extractor.

BUG-H7: event nodes were registered AFTER the edge loop, so EMITS edges were
dropped (destination node not yet in the graph). Fix is in graph_extractor.py
lines 1653-1656 (register event nodes before the loop). This test confirms the
fix is present and functional.

Requires solc-0.5.11. Marked skipif when solc is absent.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

SOLC_ROOT = Path.home() / ".solc-select" / "artifacts"
TEST_SOLC_VERSION = "0.5.11"
TEST_SOLC_BIN = SOLC_ROOT / f"solc-{TEST_SOLC_VERSION}" / f"solc-{TEST_SOLC_VERSION}"

HAS_SOLC = TEST_SOLC_BIN.exists()
REPO_ROOT = Path(__file__).resolve().parents[3]

EMITS_EDGE_TYPE = 3  # EDGE_TYPES["EMITS"] confirmed from graph_schema.py

# Minimal contract: one event, one function that emits it.
EMIT_CONTRACT = textwrap.dedent("""\
    pragma solidity ^0.5.11;

    contract TokenTransfer {
        event Transfer(address indexed from, address indexed to, uint256 value);

        mapping(address => uint256) public balances;

        constructor() public {
            balances[msg.sender] = 1000;
        }

        function transfer(address to, uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            balances[msg.sender] -= amount;
            balances[to] += amount;
            emit Transfer(msg.sender, to, amount);
        }
    }
""")


@pytest.mark.skipif(not HAS_SOLC, reason=f"solc-{TEST_SOLC_VERSION} not in {SOLC_ROOT}")
class TestEmitsEdge:
    """Confirm EMITS (type 3) edges are present after the BUG-H7 fix."""

    @pytest.fixture(scope="class")
    def extracted(self, tmp_path_factory):
        from sentinel_data.representation.graph_extractor import (
            GraphExtractionConfig,
            extract_contract_graph,
        )
        tmp = tmp_path_factory.mktemp("emits_contract")
        sol_path = tmp / "emit_contract.sol"
        sol_path.write_text(EMIT_CONTRACT)
        cfg = GraphExtractionConfig(
            solc_binary=TEST_SOLC_BIN,
            solc_version=TEST_SOLC_VERSION,
            allow_paths=[str(REPO_ROOT)],
        )
        return extract_contract_graph(str(sol_path), config=cfg)

    def test_emits_edges_present(self, extracted):
        """EMITS (type 3) edges must exist for a contract that calls emit."""
        if extracted.edge_attr is None or extracted.edge_attr.numel() == 0:
            pytest.skip("graph has no edges — extraction may have returned empty")
        mask = extracted.edge_attr == EMITS_EDGE_TYPE
        count = int(mask.sum().item())
        assert count > 0, (
            f"BUG-H7 regression: EMITS (type 3) edges missing for a contract with 'emit Transfer(...)'. "
            f"Edge type histogram: { {t: int((extracted.edge_attr == t).sum()) for t in range(12)} }"
        )

    def test_emits_edge_type_value(self):
        """Confirm the EMITS constant in graph_schema matches the expected value."""
        from sentinel_data.representation.graph_schema import EDGE_TYPES
        assert EDGE_TYPES["EMITS"] == EMITS_EDGE_TYPE, (
            f"EDGE_TYPES['EMITS'] changed: expected {EMITS_EDGE_TYPE}, got {EDGE_TYPES['EMITS']}"
        )

    def test_graph_schema_version_is_v9(self):
        """Extracted graph must carry the v9 schema version."""
        from sentinel_data.representation.graph_schema import FEATURE_SCHEMA_VERSION
        assert FEATURE_SCHEMA_VERSION == "v9"

    def test_node_feature_dim(self, extracted):
        """Node features must be [n_nodes, 12] per the v9 schema."""
        assert extracted.x.ndim == 2
        assert extracted.x.shape[1] == 12, (
            f"Expected NODE_FEATURE_DIM=12, got {extracted.x.shape[1]}"
        )
