"""Tests for corrected Phase-8 logical lineage V3."""

from __future__ import annotations

from sentinel_data.vnext.policy import CLASS_NAMES
from sentinel_data.vnext.r4_logical_v3 import _freeze_roles_v3
from sentinel_data.vnext.r4_v3_versions import (
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)


def _policy():
    return {
        "class_supervision": {
            name: {"status": "ENABLED" if index < 8 else "DISABLED"}
            for index, name in enumerate(CLASS_NAMES)
        }
    }


def _semantic_rows(contract_id):
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        enabled = index < 8
        rows.append(
            {
                "contract_id": contract_id,
                "class_index": index,
                "class_name": name,
                "training_strength": "STRONG" if enabled else "NONE",
                "target_value": 1 if enabled else None,
            }
        )
    return rows


def test_v3_role_freeze_is_group_atomic_and_versioned():
    contracts = ["a" * 64, "b" * 64, "c" * 64]
    semantic = [row for contract in contracts for row in _semantic_rows(contract)]
    artifact_info = {
        contract: {"representation_available": True}
        for contract in contracts
    }
    groups = [
        {
            "group_id": f"g{index}",
            "members": [contract],
            "sources": ["solidifi"],
        }
        for index, contract in enumerate(contracts)
    ]
    grouping = {"grouping_version": GROUPING_VERSION_V3, "groups": groups}

    group_rows, contract_rows, manifest = _freeze_roles_v3(
        semantic,
        artifact_info,
        grouping,
        _policy(),
        grouping_sha="1" * 64,
        policy_sha="2" * 64,
    )

    assert manifest["partition_version"] == ROLE_PARTITION_VERSION_V3
    assert manifest["grouping_version"] == GROUPING_VERSION_V3
    assert manifest["address_literal_grouping_authority"] is False
    assert set(row["role"] for row in group_rows) == {
        "TRAIN_STRONG",
        "MODEL_SELECTION",
        "INTERNAL_AUDIT",
    }
    assert len(contract_rows) == 3
    assert all(row["partition_version"] == ROLE_PARTITION_VERSION_V3 for row in contract_rows)
    assert all(row["schema"] == "r4-repaired-contract-role-row-v3" for row in contract_rows)
