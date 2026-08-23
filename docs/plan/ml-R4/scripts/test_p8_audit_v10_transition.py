"""Focused tests for V9-to-V10 transition-audit diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).with_name("p8_audit_v10_transition.py")
SPEC = importlib.util.spec_from_file_location("p8_audit_v10_transition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PROBE_SCRIPT = Path(__file__).with_name("p8_probe_v10_structural_drift.py")
PROBE_SPEC = importlib.util.spec_from_file_location(
    "p8_probe_v10_structural_drift", PROBE_SCRIPT
)
assert PROBE_SPEC is not None and PROBE_SPEC.loader is not None
PROBE = importlib.util.module_from_spec(PROBE_SPEC)
PROBE_SPEC.loader.exec_module(PROBE)


def _graph(
    features: list[list[float]],
    metadata: list[dict[str, object]],
    edge_index: list[list[int]],
    edge_types: list[int],
) -> SimpleNamespace:
    return SimpleNamespace(
        x=torch.tensor(features, dtype=torch.float32),
        node_metadata=metadata,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_types, dtype=torch.long),
    )


def test_parse_only_source_syntax_hits_are_explicitly_diagnostic(tmp_path: Path) -> None:
    source = tmp_path / "fixture.sol"
    source.write_text(
        """contract Fixture {
    function f(address payable target) public {
        target.call("");
        target.call.value(1)("");
        target.delegatecall("");
        target.transfer(1);
        target.send(1);
        new Fixture();
    }
}
""",
        encoding="utf-8",
    )

    assert MODULE._parse_only_source_syntax_hits(source) == {
        "raw_low_level": 3,
        "ether_transfer": 1,
        "ether_send": 1,
        "contract_creation": 1,
    }


def test_historical_v9_normal_mode_defaults_to_full_analysis() -> None:
    assert MODULE._historical_v9_extraction_mode({}) == "slither_full_analysis"
    assert (
        MODULE._historical_v9_extraction_mode(
            {"graph_extraction_mode": "slither_parse_only"}
        )
        == "slither_parse_only"
    )


def test_unchanged_edge_topology_comparison_is_order_independent() -> None:
    left = SimpleNamespace(
        edge_attr=torch.tensor([8, 6, 8, 12]),
        edge_index=torch.tensor([[1, 0, 1, 3], [2, 1, 2, 3]]),
    )
    right = SimpleNamespace(
        edge_attr=torch.tensor([12, 8, 6, 8]),
        edge_index=torch.tensor([[3, 1, 0, 1], [3, 2, 1, 2]]),
    )

    assert MODULE._edge_topology_equal_through(left, right, 10) is True
    right.edge_index[1, 3] = 4
    assert MODULE._edge_topology_equal_through(left, right, 10) is False


def test_node_aware_probe_proves_only_exact_index_permutation_equivalence() -> None:
    metadata = [
        {"name": "Fixture.f()", "type": "FUNCTION", "source_lines": [1]},
        {"name": "EXPRESSION x", "type": "CFG_NODE_OTHER", "source_lines": [2]},
        {"name": "EXPRESSION x", "type": "CFG_NODE_OTHER", "source_lines": [2]},
    ]
    left = _graph(
        [[0.1], [0.2], [0.2]],
        metadata,
        [[0, 0, 1], [1, 2, 2]],
        [5, 5, 6],
    )
    right = _graph(
        [[0.1], [0.2], [0.2]],
        list(metadata),
        [[0, 0, 2], [1, 2, 1]],
        [5, 5, 6],
    )

    comparison = PROBE.compare_graphs(left, right)

    assert comparison["raw_node_features_equal"] is True
    assert comparison["raw_node_metadata_equal"] is True
    assert comparison["raw_unchanged_edge_topology_equal"] is False
    assert comparison["exact_node_index_invariant_equivalent"] is True
    assert comparison["classification"] == "NODE_ORDER_INDEX_NONDETERMINISM_PROVEN"


def test_node_aware_probe_rejects_feature_classification_drift() -> None:
    left = _graph(
        [[0.1], [12.0 / 13.0]],
        [
            {"name": "Fixture.f()", "type": "FUNCTION", "source_lines": [1]},
            {
                "name": "EXPRESSION balance = value",
                "type": "CFG_NODE_OTHER",
                "source_lines": [2],
            },
        ],
        [[0], [1]],
        [5],
    )
    right = _graph(
        [[0.1], [9.0 / 13.0]],
        [
            {"name": "Fixture.f()", "type": "FUNCTION", "source_lines": [1]},
            {
                "name": "EXPRESSION balance = value",
                "type": "CFG_NODE_WRITE",
                "source_lines": [2],
            },
        ],
        [[0], [1]],
        [5],
    )

    comparison = PROBE.compare_graphs(left, right)

    assert comparison["raw_unchanged_edge_topology_equal"] is True
    assert comparison["exact_node_index_invariant_equivalent"] is False
    assert comparison["classification"] == "FEATURE_OR_METADATA_CLASSIFICATION_DRIFT"
    diffs = comparison["unique_identity_semantic_diffs"]
    assert len(diffs) == 1
    assert diffs[0]["identity"] == {
        "name": "EXPRESSION balance = value",
        "source_lines": [2],
        "coarse_type": "CFG_NODE",
    }
    assert diffs[0]["left_type"] == "CFG_NODE_OTHER"
    assert diffs[0]["right_type"] == "CFG_NODE_WRITE"


def test_node_aware_probe_rejects_real_unchanged_edge_topology_change() -> None:
    metadata = [
        {"name": "Fixture.f()", "type": "FUNCTION", "source_lines": [1]},
        {"name": "EXPRESSION a", "type": "CFG_NODE_OTHER", "source_lines": [2]},
        {"name": "EXPRESSION b", "type": "CFG_NODE_OTHER", "source_lines": [3]},
    ]
    left = _graph(
        [[0.1], [0.2], [0.3]],
        metadata,
        [[0, 1], [1, 2]],
        [5, 6],
    )
    right = _graph(
        [[0.1], [0.2], [0.3]],
        list(metadata),
        [[0, 2], [1, 1]],
        [5, 6],
    )

    comparison = PROBE.compare_graphs(left, right)

    assert comparison["exact_node_index_invariant_equivalent"] is False
    assert comparison["classification"] == "SEMANTIC_STRUCTURE_DRIFT"
