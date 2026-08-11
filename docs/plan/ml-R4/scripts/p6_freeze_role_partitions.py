#!/usr/bin/env python3
"""Freeze deterministic, leakage-group-safe DATA vNext role partitions for R4 Phase 6."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from p6_inventory_role_support import (
    EXPECTED_CLASSES,
    EXPECTED_CONTRACTS,
    EXPECTED_LEDGER_SHA,
    EXPECTED_ROWS,
    group_id,
    require_pyarrow,
    sha256_file,
    signal_for,
)

PARTITION_VERSION = "r4-vnext-roles-v1"
MODEL_FRACTION = 0.15
AUDIT_FRACTION = 0.15


def ranking(policy_sha: str, gid: str, salt: str = "") -> str:
    payload = f"{PARTITION_VERSION}|{EXPECTED_LEDGER_SHA}|{policy_sha}|{salt}|{gid}".encode()
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pick_coverage_groups(
    candidates: list[dict[str, Any]],
    supported_classes: list[str],
    assigned: dict[str, str],
    role: str,
    policy_sha: str,
) -> None:
    """Assign at least one distinct group per supported class to role."""
    for cls in supported_classes:
        choices = [
            g for g in candidates
            if g["group_id"] not in assigned and cls in g["strong_classes"]
        ]
        choices.sort(key=lambda g: ranking(policy_sha, g["group_id"], f"coverage:{role}:{cls}"))
        if not choices:
            raise RuntimeError(f"cannot reserve {role} coverage for {cls}")
        assigned[choices[0]["group_id"]] = role


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path("docs/plan/ml-R4")
    ap.add_argument("--ledger", type=Path, default=root / "ledger/evidence_ledger_v1.parquet")
    ap.add_argument("--policy", type=Path, default=root / "specs/data_vnext_policy_v1.json")
    ap.add_argument("--group-inventory", type=Path, default=root / "manifests/p6_group_eligibility_inventory.jsonl")
    ap.add_argument("--group-output", type=Path, default=root / "manifests/p6_role_group_manifest.jsonl")
    ap.add_argument("--contract-output", type=Path, default=root / "manifests/p6_contract_role_manifest.jsonl")
    ap.add_argument("--support-output", type=Path, default=root / "manifests/p6_role_support_table.json")
    ap.add_argument("--unsupported-output", type=Path, default=root / "manifests/p6_unsupported_roles.json")
    ap.add_argument("--acceptance-output", type=Path, default=root / "manifests/p6_untouched_acceptance_manifest.json")
    ap.add_argument("--manifest-output", type=Path, default=root / "manifests/p6_partition_manifest.json")
    args = ap.parse_args()

    pq = require_pyarrow()
    if sha256_file(args.ledger) != EXPECTED_LEDGER_SHA:
        raise RuntimeError("ledger SHA mismatch")
    policy = json.loads(args.policy.read_text())
    if policy["status"] != "ACCEPTED_G5":
        raise RuntimeError("policy not ACCEPTED_G5")
    policy_sha = sha256_file(args.policy)

    groups = load_jsonl(args.group_inventory)
    if len({g["group_id"] for g in groups}) != len(groups):
        raise RuntimeError("duplicate group id in eligibility inventory")

    excluded: dict[str, str] = {}
    strong_candidates = []
    weak_candidates = []
    unlabeled_candidates = []
    for g in groups:
        if int(g["represented_contracts"]) != int(g["contract_count"]):
            excluded[g["group_id"]] = "EXCLUDED_NO_COMPLETE_REPRESENTATION_GROUP"
            continue
        if g["classification"] == "STRONG_ELIGIBLE_GROUP":
            strong_candidates.append(g)
        elif g["classification"] == "WEAK_ELIGIBLE_GROUP":
            weak_candidates.append(g)
        elif g["classification"] == "UNLABELED_GROUP":
            unlabeled_candidates.append(g)
        else:
            raise RuntimeError(f"unknown group classification {g['classification']}")

    supported_classes = [
        cls for cls in EXPECTED_CLASSES
        if any(cls in g["strong_classes"] for g in strong_candidates)
    ]
    expected_enabled_with_support = [
        cls for cls in EXPECTED_CLASSES
        if policy["class_supervision"][cls]["status"] == "ENABLED"
    ]
    if supported_classes != expected_enabled_with_support:
        raise RuntimeError(
            f"enabled classes without represented strong group support: expected {expected_enabled_with_support}, got {supported_classes}"
        )

    # Every supported class must have at least three represented strong groups so
    # TRAIN_STRONG, MODEL_SELECTION, and INTERNAL_AUDIT can each receive one.
    represented_strong_groups_by_class = {
        cls: sum(cls in g["strong_classes"] for g in strong_candidates)
        for cls in supported_classes
    }
    scarce = {cls: n for cls, n in represented_strong_groups_by_class.items() if n < 3}
    if scarce:
        raise RuntimeError(f"insufficient represented strong groups for three-role coverage: {scarce}")

    assigned: dict[str, str] = {}
    pick_coverage_groups(strong_candidates, supported_classes, assigned, "MODEL_SELECTION", policy_sha)
    pick_coverage_groups(strong_candidates, supported_classes, assigned, "INTERNAL_AUDIT", policy_sha)

    n_strong = len(strong_candidates)
    target_model = max(len(supported_classes), round(n_strong * MODEL_FRACTION))
    target_audit = max(len(supported_classes), round(n_strong * AUDIT_FRACTION))

    remaining = [g for g in strong_candidates if g["group_id"] not in assigned]
    remaining.sort(key=lambda g: ranking(policy_sha, g["group_id"], "strong-fill"))

    def count_role(role: str) -> int:
        return sum(r == role for r in assigned.values())

    for g in remaining:
        if count_role("MODEL_SELECTION") < target_model:
            assigned[g["group_id"]] = "MODEL_SELECTION"
        elif count_role("INTERNAL_AUDIT") < target_audit:
            assigned[g["group_id"]] = "INTERNAL_AUDIT"
        else:
            assigned[g["group_id"]] = "TRAIN_STRONG"

    # Coverage-first assignment could fill target sizes early; all strong groups
    # not explicitly assigned above must become training groups.
    for g in strong_candidates:
        assigned.setdefault(g["group_id"], "TRAIN_STRONG")

    for g in weak_candidates:
        assigned[g["group_id"]] = "TRAIN_WEAK"
    for g in unlabeled_candidates:
        assigned[g["group_id"]] = "TRAIN_UNLABELED"
    for gid in excluded:
        assigned[gid] = "EXCLUDED"

    if len(assigned) != len(groups):
        raise RuntimeError(f"assigned groups {len(assigned)} != inventory groups {len(groups)}")

    # Ensure every represented strong class remains in all three permitted
    # strong-evidence roles; model selection remains positive-only/limited.
    role_class_group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_gid = {g["group_id"]: g for g in groups}
    for gid, role in assigned.items():
        for cls in by_gid[gid]["strong_classes"]:
            role_class_group_counts[role][cls] += 1
    for cls in supported_classes:
        for role in ("TRAIN_STRONG", "MODEL_SELECTION", "INTERNAL_AUDIT"):
            if role_class_group_counts[role][cls] < 1:
                raise RuntimeError(f"{role} lacks strong group support for {cls}")

    group_rows = []
    contract_rows = []
    role_group_counts = Counter()
    role_contract_counts = Counter()
    for g in sorted(groups, key=lambda x: x["group_id"]):
        gid = g["group_id"]
        role = assigned[gid]
        role_group_counts[role] += 1
        role_contract_counts[role] += int(g["contract_count"])
        group_rows.append({
            "schema": "r4-phase6-role-group-row-v1",
            "partition_version": PARTITION_VERSION,
            "group_id": gid,
            "role": role,
            "reason": excluded.get(gid),
            "contract_ids": g["contract_ids"],
            "contract_count": g["contract_count"],
            "sources": g["sources"],
            "historical_splits": g["historical_splits"],
            "strong_classes": g["strong_classes"],
            "weak_classes": g["weak_classes"],
            "represented_contracts": g["represented_contracts"],
            "assignment_rank_sha256": ranking(policy_sha, gid, f"role:{role}"),
        })
        for cid in sorted(g["contract_ids"]):
            contract_rows.append({
                "schema": "r4-phase6-contract-role-row-v1",
                "partition_version": PARTITION_VERSION,
                "contract_id": cid,
                "group_id": gid,
                "role": role,
            })

    if len(contract_rows) != EXPECTED_CONTRACTS or len({r["contract_id"] for r in contract_rows}) != EXPECTED_CONTRACTS:
        raise RuntimeError("contract role manifest does not cover the exact population once")

    write_jsonl(args.group_output, group_rows)
    write_jsonl(args.contract_output, sorted(contract_rows, key=lambda x: x["contract_id"]))

    role_by_group = assigned
    table = pq.read_table(args.ledger)
    rows = table.to_pylist()
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("ledger row count changed")

    support: dict[str, dict[str, Any]] = {}
    for role in sorted(set(assigned.values())):
        support[role] = {
            "groups": role_group_counts[role],
            "contracts": role_contract_counts[role],
            "by_class": {
                cls: {
                    "confirmed_positive_rows": 0,
                    "confirmed_negative_rows": 0,
                    "weak_positive_rows": 0,
                    "unlabeled_or_masked_rows": 0,
                    "excluded_rows": 0,
                } for cls in EXPECTED_CLASSES
            },
            "source_contracts": Counter(),
        }

    seen_source_contract: set[tuple[str, str, str]] = set()
    for row in rows:
        gid = group_id(row)
        role = role_by_group[gid]
        cls = str(row["class_name"])
        source = str(row["primary_source"])
        cid = str(row["contract_id"])
        key = (role, source, cid)
        if key not in seen_source_contract:
            support[role]["source_contracts"][source] += 1
            seen_source_contract.add(key)
        cell = support[role]["by_class"][cls]
        if role == "EXCLUDED":
            cell["excluded_rows"] += 1
            continue
        strength, signal_cls = signal_for(row, policy)
        if strength == "STRONG" and signal_cls == cls:
            cell["confirmed_positive_rows"] += 1
        elif strength == "WEAK" and signal_cls == cls:
            cell["weak_positive_rows"] += 1
        else:
            cell["unlabeled_or_masked_rows"] += 1
        # Policy v1 intentionally has no confirmed-negative rows.

    for role in support:
        support[role]["source_contracts"] = dict(sorted(support[role]["source_contracts"].items()))

    unsupported = {
        "schema": "r4-phase6-unsupported-roles-v1",
        "partition_version": PARTITION_VERSION,
        "roles": {
            "THRESHOLD_FIT": {
                "status": "UNSUPPORTED_EMPTY",
                "reason": "No trustworthy class-specific confirmed-negative support exists under data-vnext-policy-v1.",
                "groups": [],
                "contracts": []
            },
            "CALIBRATION_FIT": {
                "status": "UNSUPPORTED_EMPTY",
                "reason": "Calibration requires outcome-labeled discrimination support; current evidence is strong/weak positive plus unlabeled with no confirmed negatives.",
                "groups": [],
                "contracts": []
            },
            "UNTOUCHED_ACCEPTANCE": {
                "status": "UNSUPPORTED_EMPTY_FROZEN",
                "reason": "No trustworthy unexposed labeled corpus exists in the recovered repository evidence; historical/manual/quickstart corpora are exposed, semantically invalid for negatives, unavailable, or deferred.",
                "groups": [],
                "contracts": []
            }
        }
    }
    write_json(args.unsupported_output, unsupported)

    acceptance = {
        "schema": "r4-phase6-untouched-acceptance-manifest-v1",
        "partition_version": PARTITION_VERSION,
        "status": "UNSUPPORTED_EMPTY_FROZEN",
        "frozen": True,
        "contract_ids": [],
        "group_ids": [],
        "prior_exposure": "No candidate corpus qualified as both semantically trustworthy and unexposed.",
        "reason": unsupported["roles"]["UNTOUCHED_ACCEPTANCE"]["reason"],
        "prohibited_uses": [
            "Do not relabel historical test data as untouched acceptance.",
            "Do not use manual_hand_written_contracts as untouched acceptance; it was explicitly used for model/agent validation.",
            "Do not use benchmark_v0.1_quickstart NonVulnerable labels as negatives; Tier-A builder contains invalid access_control/tx.origin -> NonVulnerable mappings.",
            "Do not use Tier-E BCCC/tool-silence design as confirmed negative evidence."
        ]
    }
    write_json(args.acceptance_output, acceptance)

    support_report = {
        "schema": "r4-phase6-role-support-table-v1",
        "partition_version": PARTITION_VERSION,
        "ledger_sha256": EXPECTED_LEDGER_SHA,
        "policy_sha256": policy_sha,
        "role_support": support,
        "limitations": {
            "MODEL_SELECTION": "Positive-only strong evidence. Useful for positive loss/recall diagnostics but not full discrimination/F1/AUC.",
            "THRESHOLD_FIT": "Unsupported; empty.",
            "CALIBRATION_FIT": "Unsupported; empty.",
            "UNTOUCHED_ACCEPTANCE": "Unsupported; empty/frozen.",
            "CONFIRMED_NEGATIVES": "Zero rows in policy v1."
        }
    }
    write_json(args.support_output, support_report)

    manifest = {
        "schema": "r4-phase6-partition-manifest-v1",
        "partition_version": PARTITION_VERSION,
        "status": "FROZEN_CANDIDATE_G6",
        "ledger_sha256": EXPECTED_LEDGER_SHA,
        "policy_version": policy["policy_version"],
        "policy_sha256": policy_sha,
        "population_contracts": EXPECTED_CONTRACTS,
        "population_groups": len(groups),
        "role_group_counts": dict(sorted(role_group_counts.items())),
        "role_contract_counts": dict(sorted(role_contract_counts.items())),
        "strong_role_targets": {
            "model_selection_fraction": MODEL_FRACTION,
            "internal_audit_fraction": AUDIT_FRACTION,
            "training_fraction_remainder": 1.0 - MODEL_FRACTION - AUDIT_FRACTION,
            "coverage_rule": "At least one represented strong group per supported class in TRAIN_STRONG, MODEL_SELECTION, and INTERNAL_AUDIT."
        },
        "represented_strong_groups_by_class": represented_strong_groups_by_class,
        "artifacts": {
            "group_manifest": {"path": str(args.group_output), "sha256": sha256_file(args.group_output)},
            "contract_manifest": {"path": str(args.contract_output), "sha256": sha256_file(args.contract_output)},
            "support_table": {"path": str(args.support_output), "sha256": sha256_file(args.support_output)},
            "unsupported_roles": {"path": str(args.unsupported_output), "sha256": sha256_file(args.unsupported_output)},
            "acceptance_manifest": {"path": str(args.acceptance_output), "sha256": sha256_file(args.acceptance_output)},
        },
        "invariants": [
            "one_role_per_leakage_group",
            "one_role_per_contract",
            "group_not_split",
            "no_unrepresented_contract_in_training_or_model_selection_roles",
            "no_confirmed_negative_synthesis",
            "dive_tod_only_weak_signal",
            "threshold_calibration_acceptance_empty_unsupported",
            "phase5_authority_not_strengthened"
        ]
    }
    write_json(args.manifest_output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
