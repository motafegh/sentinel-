from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data


SCRIPT = Path(__file__).with_name("p8_probe_v10_v25_reproducibility.py")
SCRIPT_DIR = str(SCRIPT.parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
SPEC = importlib.util.spec_from_file_location("p8_probe_v10_v25_reproducibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _graph(node_type: str, type_id: int, *, other_feature: float = 0.0) -> Data:
    x = torch.zeros((2, 12), dtype=torch.float)
    x[0, 0] = type_id / 13.0
    x[0, 1] = other_feature
    graph = Data(
        x=x,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([6], dtype=torch.long),
    )
    graph.node_metadata = [
        {"name": "EXPRESSION alias.field = 1", "type": node_type, "source_lines": [10]},
        {"name": "END_IF", "type": "CFG_NODE_OTHER", "source_lines": [11]},
    ]
    return graph


def test_expected_storage_write_canonicalization_restores_exact_equivalence() -> None:
    target = {("EXPRESSION alias.field = 1", (10,))}
    reference = _graph("CFG_NODE_OTHER", 12)
    v25 = _graph("CFG_NODE_WRITE", 9)

    canonical_reference = probe._canonicalize_expected_writes(
        reference, "dive/fixture", target
    )
    canonical_v25 = probe._canonicalize_expected_writes(v25, "dive/fixture", target)

    comparison = probe.compare_graphs(canonical_reference, canonical_v25)
    assert comparison["exact_node_index_invariant_equivalent"] is True
    passed, failures = probe._all_targets_are_write(v25, "dive/fixture", target)
    assert passed is True
    assert failures == []


def test_unrelated_feature_difference_remains_blocking_after_canonicalization() -> None:
    target = {("EXPRESSION alias.field = 1", (10,))}
    reference = _graph("CFG_NODE_OTHER", 12, other_feature=0.0)
    v25 = _graph("CFG_NODE_WRITE", 9, other_feature=1.0)

    canonical_reference = probe._canonicalize_expected_writes(
        reference, "dive/fixture", target
    )
    canonical_v25 = probe._canonicalize_expected_writes(v25, "dive/fixture", target)

    comparison = probe.compare_graphs(canonical_reference, canonical_v25)
    assert comparison["exact_node_index_invariant_equivalent"] is False


def test_non_write_target_is_reported_as_failure() -> None:
    target = {("EXPRESSION alias.field = 1", (10,))}
    graph = _graph("CFG_NODE_READ", 10)

    passed, failures = probe._all_targets_are_write(graph, "dive/fixture", target)
    assert passed is False
    assert failures == [
        {
            "name": "EXPRESSION alias.field = 1",
            "source_lines": [10],
            "observed_types": ["CFG_NODE_READ"],
        }
    ]
