#!/usr/bin/env python3
"""Audit corrected logical lineage V3 against accepted repaired V2.

The gate proves that the grouping/partition correction changed only logical
boundaries while preserving role-independent supervision and all accepted
physical representation bytes. It also derives the new active training/sample
counts; it does not authorize training.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

from sentinel_data.vnext.policy import CLASS_NAMES
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)

DATA_ROOT = REPO_ROOT / "data_module/data"
V2_BUILD = DATA_ROOT / "r4-v2-build"
V3_BUILD = DATA_ROOT / "r4-v3-logical-build"
V2_PUBLICATION = DATA_ROOT / "exports/sentinel-r4-vnext-v2"
V3_PUBLICATION = DATA_ROOT / "exports/sentinel-r4-vnext-v3"
OUTPUT = V3_BUILD / "logical_v3_acceptance.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _group_stats(payload):
    sizes = sorted(len(group["members"]) for group in payload["groups"])
    return {
        "groups": len(sizes),
        "max": max(sizes),
        "multi_member": sum(value > 1 for value in sizes),
        "singletons": sum(value == 1 for value in sizes),
    }


def main() -> int:
    import pyarrow.parquet as pq

    v2_manifest = _load(V2_PUBLICATION / "manifest.json")
    v3_manifest = _load(V3_PUBLICATION / "manifest.json")
    v2_grouping = _load(V2_BUILD / "grouping.json")
    v3_grouping = _load(V3_BUILD / "grouping.json")
    v3_partition = _load(V3_PUBLICATION / "partition_manifest.json")
    v3_binding = _load(V3_PUBLICATION / "representation_binding_report.json")
    rows = pq.read_table(V3_PUBLICATION / "ml_targets.parquet").to_pylist()

    active_train_rows = []
    active_train_groups = set()
    active_train_roles = Counter()

    outcome_metric_rows = []
    outcome_metric_groups = set()
    outcome_metric_roles = Counter()
    model_selection_rows = []
    model_selection_groups = set()
    internal_audit_rows = []
    internal_audit_groups = set()

    for row in rows:
        role = str(row["role"])
        loss_active = any(
            bool(row.get(f"effective_loss_mask_{index}"))
            for index in range(len(CLASS_NAMES))
        )
        metric_active = any(
            bool(row.get(f"outcome_metric_mask_{index}"))
            for index in range(len(CLASS_NAMES))
        )
        if loss_active:
            active_train_rows.append(row)
            active_train_groups.add(str(row["group_id"]))
            active_train_roles[role] += 1
        if metric_active:
            outcome_metric_rows.append(row)
            outcome_metric_groups.add(str(row["group_id"]))
            outcome_metric_roles[role] += 1
            if role == "MODEL_SELECTION":
                model_selection_rows.append(row)
                model_selection_groups.add(str(row["group_id"]))
            elif role == "INTERNAL_AUDIT":
                internal_audit_rows.append(row)
                internal_audit_groups.add(str(row["group_id"]))

    batch_size = 8
    accumulation = 8
    micro_batches = math.ceil(len(active_train_groups) / batch_size)
    optimizer_steps = math.ceil(micro_batches / accumulation)

    v2_stats = _group_stats(v2_grouping)
    v3_stats = _group_stats(v3_grouping)
    address_edges = [
        edge
        for edge in v3_grouping.get("evidence_edges") or []
        if edge.get("reason") == "same_source_shared_address_candidate"
    ]
    parent_digest = (v2_manifest.get("representation_binding_report") or {}).get(
        "binding_digest_sha256"
    )

    metric_roles = set(outcome_metric_roles)
    expected_metric_roles = {"MODEL_SELECTION", "INTERNAL_AUDIT"}
    checks = {
        "dataset_version_v3": v3_manifest.get("dataset_version") == DATASET_VERSION_V3,
        "grouping_version_v3": v3_manifest.get("grouping_version") == GROUPING_VERSION_V3,
        "partition_version_v3": v3_manifest.get("partition_version") == ROLE_PARTITION_VERSION_V3,
        "population_contracts_unchanged": v3_manifest.get("population", {}).get("contracts")
        == v2_manifest.get("population", {}).get("contracts"),
        "population_cells_unchanged": v3_manifest.get("population", {}).get(
            "contract_class_rows"
        )
        == v2_manifest.get("population", {}).get("contract_class_rows"),
        "target_counts_unchanged": v3_manifest.get("target_counts")
        == v2_manifest.get("target_counts"),
        "training_strength_counts_unchanged": v3_manifest.get("training_strength_counts")
        == v2_manifest.get("training_strength_counts"),
        "confirmed_negative_rows_zero": v3_manifest.get("confirmed_negative_rows") == 0,
        "address_authority_disabled": v3_manifest.get("address_literal_grouping_authority")
        is False,
        "address_union_edges_zero": len(address_edges) == 0,
        "giant_v2_group_removed": v3_stats["max"] < v2_stats["max"],
        "binding_passed": v3_binding.get("passed") is True,
        "physical_binding_digest_unchanged": v3_binding.get("binding_digest_sha256")
        == parent_digest,
        "all_contracts_physically_checked": v3_binding.get("checked_contracts")
        == v3_manifest.get("population", {}).get("contracts"),
        "all_representation_files_checked": v3_binding.get("checked_files")
        == 3 * v3_manifest.get("population", {}).get("contracts", 0),
        "partition_reports_no_confirmed_negatives": v3_partition.get(
            "confirmed_negative_rows"
        )
        == 0,
        "outcome_metric_roles_are_selection_or_audit_only": metric_roles
        <= expected_metric_roles,
    }

    report = {
        "schema": "sentinel-r4-logical-v3-acceptance-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "physical_rebuild_performed": False,
        "training_authorized": False,
        "versions": {
            "dataset": v3_manifest.get("dataset_version"),
            "grouping": v3_manifest.get("grouping_version"),
            "partition": v3_manifest.get("partition_version"),
            "physical_parent_dataset": v2_manifest.get("dataset_version"),
        },
        "grouping_comparison": {
            "v2": v2_stats,
            "v3": v3_stats,
            "v3_address_edges": len(address_edges),
        },
        "role_contract_counts": v3_manifest.get("role_contract_counts"),
        "role_group_counts": v3_partition.get("role_group_counts"),
        "active_supervision": {
            "optimizer_contracts": len(active_train_rows),
            "optimizer_groups": len(active_train_groups),
            "optimizer_contracts_by_role": dict(sorted(active_train_roles.items())),
            "outcome_metric_contracts": len(outcome_metric_rows),
            "outcome_metric_groups": len(outcome_metric_groups),
            "outcome_metric_contracts_by_role": dict(sorted(outcome_metric_roles.items())),
            "model_selection_contracts": len(model_selection_rows),
            "model_selection_groups": len(model_selection_groups),
            "internal_audit_contracts": len(internal_audit_rows),
            "internal_audit_groups": len(internal_audit_groups),
            "effective_loss_cells": v3_manifest.get("effective_loss_cells"),
            "outcome_metric_cells": v3_manifest.get("outcome_metric_cells"),
        },
        "planning_arithmetic_if_batch8_accum8": {
            "micro_batches_per_epoch": micro_batches,
            "optimizer_steps_per_epoch": optimizer_steps,
            "hundred_epoch_steps_if_later_authorized": optimizer_steps * 100,
            "authorized_horizon": False,
        },
        "decision_boundary": (
            "PASS accepts the corrected logical V3 grouping/partition over the existing "
            "physical repaired artifacts. MODEL_SELECTION and INTERNAL_AUDIT outcome-metric "
            "populations are reported separately. PASS does not promote the target-aware "
            "selector, create confirmed-negative truth, select a PU objective, or authorize "
            "full training."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
