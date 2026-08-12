#!/usr/bin/env python3
"""Validate R4 Phase-6 frozen role/acceptance artifacts."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("docs/plan/ml-R4")
M = ROOT / "manifests"
EXPECTED_CONTRACTS = 22493
EXPECTED_GROUPS = 13509
EXPECTED_LEDGER_SHA = "3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7"
EXPECTED_ROLES = {"TRAIN_STRONG", "MODEL_SELECTION", "INTERNAL_AUDIT", "TRAIN_WEAK", "TRAIN_UNLABELED", "EXCLUDED"}
ENABLED_CLASSES = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "IntegerUO",
    "MishandledException", "Reentrancy", "Timestamp", "TransactionOrderDependence"
]
DISABLED_CLASSES = {"GasException", "UnusedReturn"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> int:
    partition_path = M / "p6_partition_manifest.json"
    group_path = M / "p6_role_group_manifest.jsonl"
    contract_path = M / "p6_contract_role_manifest.jsonl"
    support_path = M / "p6_role_support_table.json"
    unsupported_path = M / "p6_unsupported_roles.json"
    acceptance_path = M / "p6_untouched_acceptance_manifest.json"
    inventory_path = M / "p6_role_support_inventory.json"
    group_inventory_path = M / "p6_group_eligibility_inventory.jsonl"
    policy_path = ROOT / "specs/data_vnext_policy_v1.json"

    for p in [partition_path, group_path, contract_path, support_path, unsupported_path, acceptance_path, inventory_path, group_inventory_path, policy_path]:
        assert p.is_file(), p

    partition = load_json(partition_path)
    policy = load_json(policy_path)
    inventory = load_json(inventory_path)
    groups = load_jsonl(group_path)
    contracts = load_jsonl(contract_path)
    eligibility = {x["group_id"]: x for x in load_jsonl(group_inventory_path)}
    support = load_json(support_path)
    unsupported = load_json(unsupported_path)
    acceptance = load_json(acceptance_path)

    assert partition["schema"] == "r4-phase6-partition-manifest-v1"
    assert partition["partition_version"] == "r4-vnext-roles-v1"
    assert partition["status"] in {"FROZEN_CANDIDATE_G6", "FROZEN_G6"}
    assert partition["ledger_sha256"] == EXPECTED_LEDGER_SHA
    assert partition["population_contracts"] == EXPECTED_CONTRACTS
    assert partition["population_groups"] == EXPECTED_GROUPS
    assert policy["status"] == "ACCEPTED_G5"
    assert partition["policy_sha256"] == sha256(policy_path)
    assert inventory["ledger_sha256"] == EXPECTED_LEDGER_SHA
    assert inventory["groups"] == EXPECTED_GROUPS

    artifact_map = {
        "group_manifest": group_path,
        "contract_manifest": contract_path,
        "support_table": support_path,
        "unsupported_roles": unsupported_path,
        "acceptance_manifest": acceptance_path,
    }
    for key, path in artifact_map.items():
        assert partition["artifacts"][key]["sha256"] == sha256(path), key

    assert len(groups) == EXPECTED_GROUPS
    assert len({g["group_id"] for g in groups}) == EXPECTED_GROUPS
    assert len(eligibility) == EXPECTED_GROUPS
    assert len(contracts) == EXPECTED_CONTRACTS
    assert len({c["contract_id"] for c in contracts}) == EXPECTED_CONTRACTS

    role_group_counts = Counter(g["role"] for g in groups)
    role_contract_counts = Counter(c["role"] for c in contracts)
    assert set(role_group_counts) == EXPECTED_ROLES
    assert dict(sorted(role_group_counts.items())) == partition["role_group_counts"]
    assert dict(sorted(role_contract_counts.items())) == partition["role_contract_counts"]
    assert sum(role_contract_counts.values()) == EXPECTED_CONTRACTS

    group_by_id = {g["group_id"]: g for g in groups}
    contracts_by_group: dict[str, set[str]] = defaultdict(set)
    for c in contracts:
        gid = c["group_id"]
        assert gid in group_by_id
        assert c["role"] == group_by_id[gid]["role"]
        contracts_by_group[gid].add(c["contract_id"])
    for gid, g in group_by_id.items():
        assert contracts_by_group[gid] == set(g["contract_ids"]), gid
        assert g["contract_count"] == len(g["contract_ids"])
        e = eligibility[gid]
        assert set(g["contract_ids"]) == set(e["contract_ids"])
        assert g["sources"] == e["sources"]
        assert g["strong_classes"] == e["strong_classes"]
        assert g["weak_classes"] == e["weak_classes"]
        assert g["represented_contracts"] == e["represented_contracts"]
        if g["role"] == "EXCLUDED":
            assert g["represented_contracts"] != g["contract_count"]
            assert g["reason"] == "EXCLUDED_NO_COMPLETE_REPRESENTATION_GROUP"
        else:
            assert g["represented_contracts"] == g["contract_count"]
            assert g["reason"] is None
        if g["role"] == "TRAIN_WEAK":
            assert not g["strong_classes"]
            assert set(g["weak_classes"]) == {"TransactionOrderDependence"}
        if g["role"] == "TRAIN_UNLABELED":
            assert not g["strong_classes"] and not g["weak_classes"]
        if g["role"] in {"TRAIN_STRONG", "MODEL_SELECTION", "INTERNAL_AUDIT"}:
            assert g["strong_classes"]

    # Exact no-representation population remains excluded at group-safe granularity.
    assert partition["role_contract_counts"]["EXCLUDED"] == 836
    assert partition["role_group_counts"]["EXCLUDED"] == 835

    # Strong class coverage remains in each permitted role.
    for cls in ENABLED_CLASSES:
        assert partition["represented_strong_groups_by_class"][cls] >= 3
        for role in ("TRAIN_STRONG", "MODEL_SELECTION", "INTERNAL_AUDIT"):
            n = sum(cls in g["strong_classes"] and g["role"] == role for g in groups)
            assert n >= 1, (cls, role)
    assert set(partition["represented_strong_groups_by_class"]) == set(ENABLED_CLASSES)
    for cls in DISABLED_CLASSES:
        assert policy["class_supervision"][cls]["status"] == "SUPERVISION_DISABLED_PENDING_EVIDENCE"

    # Support table must contain no invented negatives anywhere.
    assert support["ledger_sha256"] == EXPECTED_LEDGER_SHA
    for role, rdata in support["role_support"].items():
        for cls, c in rdata["by_class"].items():
            assert c["confirmed_negative_rows"] == 0, (role, cls)
            if cls in DISABLED_CLASSES:
                assert c["confirmed_positive_rows"] == 0
                assert c["weak_positive_rows"] == 0
    assert support["limitations"]["THRESHOLD_FIT"] == "Unsupported; empty."
    assert support["limitations"]["CALIBRATION_FIT"] == "Unsupported; empty."

    # Unsupported roles must be truly empty and acceptance must be frozen empty.
    expected_status = {
        "THRESHOLD_FIT": "UNSUPPORTED_EMPTY",
        "CALIBRATION_FIT": "UNSUPPORTED_EMPTY",
        "UNTOUCHED_ACCEPTANCE": "UNSUPPORTED_EMPTY_FROZEN",
    }
    for role, status in expected_status.items():
        item = unsupported["roles"][role]
        assert item["status"] == status
        assert item["groups"] == [] and item["contracts"] == []
    assert acceptance["status"] == "UNSUPPORTED_EMPTY_FROZEN"
    assert acceptance["frozen"] is True
    assert acceptance["contract_ids"] == [] and acceptance["group_ids"] == []

    # Governance/evidence record must preserve the limitation visibly.
    finding = (ROOT / "findings/08_phase6_role_partition_and_acceptance_freeze.md").read_text()
    adr = (ROOT / "adrs/ADR-R4-006-role-partition-and-acceptance-freeze.md").read_text()
    decisions = (ROOT / "DECISION_REGISTER.md").read_text()
    risks = (ROOT / "RISK_AND_BLOCKER_REGISTER.md").read_text()
    assert "UNSUPPORTED_EMPTY_FROZEN" in finding
    assert "**Status:** Accepted" in adr
    assert "| R4-D-006 |" in decisions and "| ACCEPTED |" in decisions
    assert "R4-R015" in risks and "R4-R016" in risks

    result = {
        "passed": True,
        "partition_status": partition["status"],
        "contracts": EXPECTED_CONTRACTS,
        "groups": EXPECTED_GROUPS,
        "role_group_counts": dict(sorted(role_group_counts.items())),
        "role_contract_counts": dict(sorted(role_contract_counts.items())),
        "confirmed_negative_rows": 0,
        "threshold_fit": "UNSUPPORTED_EMPTY",
        "calibration_fit": "UNSUPPORTED_EMPTY",
        "untouched_acceptance": "UNSUPPORTED_EMPTY_FROZEN",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
