"""Focused fail-closed tests for the evidence-bound V10 transition audit V3."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data


SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPT = SCRIPT_DIR / "p8_audit_v10_transition_v3.py"
SPEC = importlib.util.spec_from_file_location("p8_audit_v10_transition_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _graph(
    node_types: list[str],
    type_ids: list[int],
    *,
    metadata_order: list[int] | None = None,
    extra_feature: float = 0.0,
) -> Data:
    metadata = [
        {
            "name": f"NODE_{index}",
            "type": node_type,
            "source_lines": [index + 1],
        }
        for index, node_type in enumerate(node_types)
    ]
    x = torch.zeros((len(node_types), 12), dtype=torch.float)
    for index, type_id in enumerate(type_ids):
        x[index, 0] = type_id / 13.0
    x[0, 1] = extra_feature

    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.tensor([5, 6], dtype=torch.long)

    if metadata_order is not None:
        order = torch.tensor(metadata_order, dtype=torch.long)
        inverse = {old: new for new, old in enumerate(metadata_order)}
        metadata = [metadata[index] for index in metadata_order]
        x = x[order]
        remapped = edge_index.clone()
        for column in range(remapped.shape[1]):
            remapped[0, column] = inverse[int(edge_index[0, column])]
            remapped[1, column] = inverse[int(edge_index[1, column])]
        edge_index = remapped

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.node_metadata = metadata
    return graph


def _bounded_report(semantic_sha: str) -> dict:
    rows = []
    for index in range(8):
        rows.append(
            {
                "contract": f"dive/index-{index}",
                "decision": AUDIT.INDEX_DECISION,
                "passed": True,
            }
        )
    for index in range(12):
        rows.append(
            {
                "contract": f"dive/write-{index}",
                "decision": AUDIT.WRITE_DECISION,
                "passed": True,
            }
        )
    return {
        "schema": AUDIT.BOUNDED_SCHEMA,
        "extractor_version": AUDIT.V10_REPRESENTATION_EXTRACTOR_VERSION,
        "slither_analyzer": AUDIT.PRIMARY_SLITHER_VERSION,
        "bounded_v25_reproducibility_passed": True,
        "zero_unexplained_drift": True,
        "physical_acceptance": False,
        "training_authorized": False,
        "blocking_identities": [],
        "unexpected_identities": 20,
        "index_equivalence_identities": 8,
        "semantic_correction_identities": 12,
        "repeat_generations": 3,
        "semantic_evidence_sha256": semantic_sha,
        "contracts": rows,
    }


def test_bounded_evidence_is_sha_bound_and_exactly_split(tmp_path: Path) -> None:
    semantic = tmp_path / "semantic.json"
    semantic.write_text('{"fixture": true}\n', encoding="utf-8")
    report = _bounded_report(AUDIT._sha256(semantic))

    index_set, write_set = AUDIT._validate_bounded_evidence(
        report,
        semantic_evidence_path=semantic,
    )

    assert len(index_set) == 8
    assert len(write_set) == 12
    assert not (index_set & write_set)

    report["semantic_evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not bound"):
        AUDIT._validate_bounded_evidence(
            report,
            semantic_evidence_path=semantic,
        )


def test_index_evidence_requires_exact_labelled_isomorphism() -> None:
    left = _graph(
        ["FUNCTION", "CFG_NODE_OTHER", "CFG_NODE_OTHER"],
        [1, 12, 12],
    )
    right = _graph(
        ["FUNCTION", "CFG_NODE_OTHER", "CFG_NODE_OTHER"],
        [1, 12, 12],
        metadata_order=[0, 2, 1],
    )

    result = AUDIT._reconcile_index_equivalence(left, right)
    assert result["passed"] is True
    assert result["comparison"]["exact_node_index_invariant_equivalent"] is True

    right.node_metadata[1] = dict(right.node_metadata[1])
    right.node_metadata[1]["name"] = "DIFFERENT"
    result = AUDIT._reconcile_index_equivalence(left, right)
    assert result["passed"] is False


def test_storage_write_reconciliation_accepts_only_targeted_correction() -> None:
    reference = _graph(
        ["FUNCTION", "CFG_NODE_OTHER", "CFG_NODE_OTHER"],
        [1, 12, 12],
    )
    candidate = _graph(
        ["FUNCTION", "CFG_NODE_WRITE", "CFG_NODE_OTHER"],
        [1, 9, 12],
    )
    targets = {("NODE_1", (2,))}

    result = AUDIT._reconcile_storage_write(
        reference,
        candidate,
        logical="dive/fixture",
        targets=targets,
    )
    assert result["passed"] is True
    assert result["semantic_write_failures"] == []
    assert (
        result["canonical_comparison"]["exact_node_index_invariant_equivalent"]
        is True
    )

    # An unrelated feature change must remain blocking after canonicalizing
    # the approved WRITE target.
    candidate.x[0, 1] = 1.0
    result = AUDIT._reconcile_storage_write(
        reference,
        candidate,
        logical="dive/fixture",
        targets=targets,
    )
    assert result["passed"] is False
    assert (
        result["canonical_comparison"]["exact_node_index_invariant_equivalent"]
        is False
    )


def test_storage_write_reconciliation_rejects_target_not_emitted_as_write() -> None:
    reference = _graph(
        ["FUNCTION", "CFG_NODE_OTHER", "CFG_NODE_OTHER"],
        [1, 12, 12],
    )
    candidate = _graph(
        ["FUNCTION", "CFG_NODE_READ", "CFG_NODE_OTHER"],
        [1, 10, 12],
    )

    result = AUDIT._reconcile_storage_write(
        reference,
        candidate,
        logical="dive/fixture",
        targets={("NODE_1", (2,))},
    )

    assert result["passed"] is False
    assert result["semantic_write_failures"] == [
        {
            "name": "NODE_1",
            "source_lines": [2],
            "observed_types": ["CFG_NODE_READ"],
        }
    ]
