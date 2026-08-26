from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCRIPT = SCRIPT_DIR / "p8_probe_v10_v25_blocker_write_evidence.py"
SPEC = importlib.util.spec_from_file_location(
    "p8_probe_v10_v25_blocker_write_evidence",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _diff(right_type: str = "CFG_NODE_WRITE") -> dict:
    return {
        "identity": {
            "name": "EXPRESSION alias.field = 1",
            "source_lines": [10],
            "coarse_type": "CFG_NODE",
        },
        "left_type": "CFG_NODE_OTHER",
        "right_type": right_type,
        "left_features": [0.9, 0.0],
        "right_features": [0.7, 0.0],
    }


def _report(*, repeat_equivalent: bool = True, right_type: str = "CFG_NODE_WRITE") -> dict:
    diff = _diff(right_type)
    comparison = {
        "exact_node_index_invariant_equivalent": False,
        "unique_identity_semantic_diffs": [diff],
    }
    return {
        "schema": probe.SOURCE_SCHEMA,
        "blocking_identities": ["dive/fixture"],
        "contracts": [
            {
                "contract": "dive/fixture",
                "decision": probe.BLOCKER_DECISION,
                "semantic_write_failures": {},
                "repeat_comparisons": {
                    "repeat_1__vs__repeat_2": {
                        "exact_node_index_invariant_equivalent": repeat_equivalent
                    },
                    "repeat_1__vs__repeat_3": {
                        "exact_node_index_invariant_equivalent": True
                    },
                    "repeat_2__vs__repeat_3": {
                        "exact_node_index_invariant_equivalent": True
                    },
                },
                "reference_comparisons": {
                    "reference_canonical__vs__repeat_1": comparison,
                    "reference_canonical__vs__repeat_2": comparison,
                    "reference_canonical__vs__repeat_3": comparison,
                },
            }
        ],
    }


def test_adapter_accepts_only_repeat_stable_lower_class_to_write_diff() -> None:
    adapter, provenance = probe._derive_adapter_report(_report())

    assert provenance == {
        "blocking_identities": ["dive/fixture"],
        "requested_nodes": 1,
    }
    comparison = adapter["contracts"][0]["comparisons"][
        "reference__vs__candidate"
    ]
    assert comparison["exact_node_index_invariant_equivalent"] is False
    assert comparison["unique_identity_semantic_diffs"][0]["right_type"] == (
        "CFG_NODE_WRITE"
    )


def test_adapter_rejects_repeat_instability() -> None:
    with pytest.raises(ValueError, match="not repeat-deterministic"):
        probe._derive_adapter_report(_report(repeat_equivalent=False))


def test_adapter_rejects_difference_outside_write_correction() -> None:
    with pytest.raises(ValueError, match="outside lower-class -> WRITE"):
        probe._derive_adapter_report(_report(right_type="CFG_NODE_READ"))


def test_storage_proof_requires_positive_storage_root() -> None:
    report = {
        "contracts": [
            {
                "contract": "dive/fixture",
                "nodes": [
                    {
                        "name": "EXPRESSION alias.field = 1",
                        "source_lines": [10],
                        "expression_writes": [
                            {
                                "root_variable": {
                                    "location": "storage",
                                    "is_storage": True,
                                }
                            }
                        ],
                    }
                ],
            }
        ]
    }
    passed, failures = probe._storage_write_proof(report)
    assert passed is True
    assert failures == []

    report["contracts"][0]["nodes"][0]["expression_writes"][0][
        "root_variable"
    ] = {"location": "memory", "is_storage": False}
    passed, failures = probe._storage_write_proof(report)
    assert passed is False
    assert failures == [
        {
            "contract": "dive/fixture",
            "name": "EXPRESSION alias.field = 1",
            "source_lines": [10],
        }
    ]
