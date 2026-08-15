"""Repository-safe tests for the repaired dynamic vNext builder."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sentinel_data.vnext.policy import CLASS_NAMES
from sentinel_data.vnext.r4_builder import (
    build_repaired_publication,
    build_semantic_cells,
    freeze_roles,
)


@pytest.fixture
def policy():
    root = Path(__file__).resolve().parents[3]
    return json.loads((root / "docs/plan/ml-R4/specs/data_vnext_policy_v1.json").read_text())


def _rep_triple(root: Path, source: str, contract_id: str) -> None:
    directory = root / source
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in (".pt", ".tokens.pt", ".rep.json"):
        (directory / f"{contract_id}{suffix}").write_bytes(b"fixture")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claim(
    artifact: str,
    source: str,
    mapped_class: str | None,
    strength: str,
    target: int | None,
    record: str,
):
    return {
        "artifact_id": artifact,
        "source": source,
        "source_record_id": record,
        "native_category": "fixture",
        "mapped_class_name": mapped_class,
        "training_strength": strength,
        "target_value": target,
        "outcome_state": "CONFIRMED_POSITIVE" if strength == "STRONG" else "NOT_REVIEWED",
        "reason_code": "FIXTURE",
    }


def test_cross_source_claims_are_aggregated_without_negative_synthesis(tmp_path, policy):
    contract_id = "a" * 64
    _rep_triple(tmp_path, "dive", contract_id)
    grouping = {
        "artifact_sources": {contract_id: ["dive", "smartbugs_curated"]},
        "artifact_to_group": {contract_id: "g1"},
        "groups": [{"group_id": "g1", "members": [contract_id], "sources": ["dive", "smartbugs_curated"]}],
    }
    claims = [
        _claim(contract_id, "smartbugs_curated", "Timestamp", "STRONG", 1, "sb-1"),
        _claim(contract_id, "dive", "TransactionOrderDependence", "WEAK", 1, "dive-1"),
    ]

    rows, info = build_semantic_cells(claims, grouping, policy, tmp_path)
    by_class = {row["class_name"]: row for row in rows}

    assert len(rows) == len(CLASS_NAMES)
    assert by_class["Timestamp"]["target_value"] == 1
    assert by_class["Timestamp"]["training_strength"] == "STRONG"
    assert by_class["Timestamp"]["outcome_state"] == "CONFIRMED_POSITIVE"
    assert by_class["TransactionOrderDependence"]["target_value"] == 1
    assert by_class["TransactionOrderDependence"]["training_strength"] == "WEAK"
    assert all(row["target_value"] != 0 for row in rows)
    assert info[contract_id]["representation_source"] == "dive"
    assert info[contract_id]["sources"] == ["dive", "smartbugs_curated"]


def test_disabled_class_never_gets_supervised_target(tmp_path, policy):
    contract_id = "b" * 64
    _rep_triple(tmp_path, "solidifi", contract_id)
    grouping = {
        "artifact_sources": {contract_id: ["solidifi"]},
        "artifact_to_group": {contract_id: "g1"},
        "groups": [{"group_id": "g1", "members": [contract_id], "sources": ["solidifi"]}],
    }
    claims = [_claim(contract_id, "solidifi", "GasException", "STRONG", 1, "s-1")]
    rows, _ = build_semantic_cells(claims, grouping, policy, tmp_path)
    gas = next(row for row in rows if row["class_name"] == "GasException")
    assert gas["target_value"] is None
    assert gas["training_strength"] == "NONE"
    assert not gas["source_policy_loss_eligible"]


def _role_fixture(policy, *, copies_per_class: int = 3):
    enabled = [
        name for name in CLASS_NAMES
        if policy["class_supervision"][name]["status"] == "ENABLED"
    ]
    semantic_rows = []
    artifact_info = {}
    groups = []
    counter = 0
    for class_name in enabled:
        for _ in range(copies_per_class):
            counter += 1
            contract_id = f"{counter:064x}"
            group_id = f"group-{counter:03d}"
            semantic_rows.append(
                {
                    "contract_id": contract_id,
                    "class_name": class_name,
                    "training_strength": "STRONG",
                    "target_value": 1,
                }
            )
            artifact_info[contract_id] = {"representation_available": True}
            groups.append(
                {
                    "group_id": group_id,
                    "members": [contract_id],
                    "sources": ["fixture"],
                }
            )
    return semantic_rows, artifact_info, {"groups": groups}, enabled


def test_role_freeze_reserves_all_three_strong_roles_per_enabled_class(policy):
    semantic_rows, artifact_info, grouping, enabled = _role_fixture(policy)
    group_rows, contract_rows, manifest = freeze_roles(
        semantic_rows,
        artifact_info,
        grouping,
        policy,
        grouping_sha="g" * 64,
        policy_sha="p" * 64,
    )
    by_role_class = {}
    cell_class = {row["contract_id"]: row["class_name"] for row in semantic_rows}
    for row in contract_rows:
        by_role_class.setdefault(row["role"], set()).add(cell_class[row["contract_id"]])
    for role in ("TRAIN_STRONG", "MODEL_SELECTION", "INTERNAL_AUDIT"):
        assert set(enabled).issubset(by_role_class[role])
    assert manifest["confirmed_negative_rows"] == 0
    assert len(contract_rows) == len(group_rows)


def test_role_freeze_is_deterministic_under_group_input_reordering(policy):
    semantic_rows, artifact_info, grouping, _ = _role_fixture(policy)
    first = freeze_roles(
        semantic_rows,
        artifact_info,
        grouping,
        policy,
        grouping_sha="g" * 64,
        policy_sha="p" * 64,
    )[1]
    reversed_grouping = {"groups": list(reversed(grouping["groups"]))}
    second = freeze_roles(
        list(reversed(semantic_rows)),
        artifact_info,
        reversed_grouping,
        policy,
        grouping_sha="g" * 64,
        policy_sha="p" * 64,
    )[1]
    assert first == second


def test_role_freeze_fails_if_three_role_class_support_is_impossible(policy):
    semantic_rows, artifact_info, grouping, _ = _role_fixture(
        policy, copies_per_class=2
    )
    with pytest.raises(RuntimeError, match="lacks three represented strong"):
        freeze_roles(
            semantic_rows,
            artifact_info,
            grouping,
            policy,
            grouping_sha="g" * 64,
            policy_sha="p" * 64,
        )


def test_all_members_of_a_group_inherit_one_role(policy):
    semantic_rows, artifact_info, grouping, _ = _role_fixture(policy)
    # Add a sibling with no signal to the first strong group. It must inherit the
    # exact same group role rather than being independently partitioned.
    sibling = "f" * 64
    first_contract = grouping["groups"][0]["members"][0]
    grouping["groups"][0]["members"].append(sibling)
    artifact_info[sibling] = {"representation_available": True}
    semantic_rows.append(
        {
            "contract_id": sibling,
            "class_name": "CallToUnknown",
            "training_strength": "NONE",
            "target_value": None,
        }
    )
    contract_rows = freeze_roles(
        semantic_rows,
        artifact_info,
        grouping,
        policy,
        grouping_sha="g" * 64,
        policy_sha="p" * 64,
    )[1]
    roles = {
        row["contract_id"]: row["role"]
        for row in contract_rows
        if row["contract_id"] in {first_contract, sibling}
    }
    assert roles[first_contract] == roles[sibling]


def test_publication_consumes_hash_bound_materialized_ledger(tmp_path, policy):
    policy_path = Path(__file__).resolve().parents[3] / "docs/plan/ml-R4/specs/data_vnext_policy_v1.json"
    claims = []
    artifact_sources = {}
    artifact_to_group = {}
    groups = []
    counter = 0
    enabled = [
        name for name in CLASS_NAMES
        if policy["class_supervision"][name]["status"] == "ENABLED"
    ]
    reps = tmp_path / "representations"
    for class_name in enabled:
        for copy in range(3):
            counter += 1
            contract_id = f"{counter:064x}"
            group_id = f"group-{counter:03d}"
            claims.append(
                _claim(
                    contract_id,
                    "solidifi",
                    class_name,
                    "STRONG",
                    1,
                    f"record-{counter}",
                )
            )
            artifact_sources[contract_id] = ["solidifi"]
            artifact_to_group[contract_id] = group_id
            groups.append(
                {"group_id": group_id, "members": [contract_id], "sources": ["solidifi"]}
            )
            _rep_triple(reps, "solidifi", contract_id)

    claims_path = tmp_path / "source_claims.jsonl"
    claims_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in claims))
    grouping_path = tmp_path / "grouping.json"
    grouping = {
        "artifact_sources": artifact_sources,
        "artifact_to_group": artifact_to_group,
        "groups": groups,
    }
    grouping_path.write_text(json.dumps(grouping, sort_keys=True))
    ledger_rows, _ = build_semantic_cells(claims, grouping, policy, reps)
    ledger_rows.sort(key=lambda row: (row["contract_id"], row["class_index"]))
    ledger_path = tmp_path / "evidence_ledger_v2.parquet"
    pq.write_table(pa.Table.from_pylist(ledger_rows), ledger_path)
    ledger_manifest_path = tmp_path / "evidence_ledger_v2_manifest.json"
    ledger_manifest_path.write_text(
        json.dumps(
            {
                "ledger_version": "evidence-ledger-r4-v2",
                "artifacts": {
                    "ledger": {"sha256": _sha(ledger_path)},
                    "source_claims": {"sha256": _sha(claims_path)},
                    "grouping": {"sha256": _sha(grouping_path)},
                    "policy": {"sha256": _sha(policy_path)},
                },
            }
        )
    )

    manifest = build_repaired_publication(
        claims_path=claims_path,
        grouping_path=grouping_path,
        policy_path=policy_path,
        representation_root=reps,
        output_dir=tmp_path / "publication",
        ledger_path=ledger_path,
        ledger_manifest_path=ledger_manifest_path,
    )
    assert manifest["artifacts"]["evidence_ledger"]["sha256"] == _sha(ledger_path)
    assert manifest["artifacts"]["evidence_ledger_manifest"]["sha256"] == _sha(
        ledger_manifest_path
    )
