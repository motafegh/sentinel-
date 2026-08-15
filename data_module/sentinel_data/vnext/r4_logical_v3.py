"""Corrected logical-lineage publication for repaired Phase-8 DATA.

V3 does not regenerate the accepted repaired source or representation bytes.
It replaces only the leakage grouping and all logical artifacts derived from
that grouping: role assignment, publication rows, representation binding, and
later evaluation reservations.

The accepted role-independent ``evidence-ledger-r4-v2`` is reused byte-for-byte
only after its source-claim/policy hashes are verified and its semantic rows are
recomputed against the V3 artifact population.  The V2 ledger's old grouping
hash is intentionally *not* inherited as V3 authority because grouping is the
thing being corrected.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sentinel_data.vnext.policy import CLASS_NAMES, validate_policy_surface
from sentinel_data.vnext.r4_builder import build_semantic_cells
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    LOGICAL_BUILD_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
    SOURCE_EVIDENCE_LEDGER_VERSION,
)

MODEL_FRACTION = 0.15
AUDIT_FRACTION = 0.15
EXPORT_SCHEMA_VERSION = "v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - local build dependency
        raise RuntimeError("logical-v3 publication requires pyarrow") from exc
    return pa, pq


def _ranking(grouping_sha: str, policy_sha: str, group_id: str, salt: str) -> str:
    payload = (
        f"{ROLE_PARTITION_VERSION_V3}|{grouping_sha}|{policy_sha}|{salt}|{group_id}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _freeze_roles_v3(
    semantic_rows: list[dict[str, Any]],
    artifact_info: dict[str, dict[str, Any]],
    grouping: dict[str, Any],
    policy: dict[str, Any],
    *,
    grouping_sha: str,
    policy_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Assign every corrected V3 leakage group exactly one role."""

    cells_by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in semantic_rows:
        cells_by_contract[str(row["contract_id"])].append(row)

    groups: list[dict[str, Any]] = []
    for raw_group in grouping.get("groups") or []:
        gid = str(raw_group["group_id"])
        members = [str(value) for value in raw_group["members"]]
        missing = [cid for cid in members if cid not in cells_by_contract]
        if missing:
            raise ValueError(f"v3 group contains contracts without semantic cells: {missing[:5]}")
        complete_rep = all(artifact_info[cid]["representation_available"] for cid in members)
        strong_classes = sorted(
            {
                row["class_name"]
                for cid in members
                for row in cells_by_contract[cid]
                if row["training_strength"] == "STRONG" and row["target_value"] == 1
            }
        )
        weak_classes = sorted(
            {
                row["class_name"]
                for cid in members
                for row in cells_by_contract[cid]
                if row["training_strength"] == "WEAK" and row["target_value"] == 1
            }
        )
        classification = (
            "STRONG_ELIGIBLE_GROUP"
            if strong_classes
            else "WEAK_ELIGIBLE_GROUP"
            if weak_classes
            else "UNLABELED_GROUP"
        )
        groups.append(
            {
                "group_id": gid,
                "contract_ids": sorted(members),
                "contract_count": len(members),
                "sources": sorted(raw_group.get("sources") or []),
                "strong_classes": strong_classes,
                "weak_classes": weak_classes,
                "represented_contracts": sum(
                    artifact_info[cid]["representation_available"] for cid in members
                ),
                "classification": classification,
                "complete_representation": complete_rep,
            }
        )

    strong = [g for g in groups if g["complete_representation"] and g["strong_classes"]]
    weak = [
        g
        for g in groups
        if g["complete_representation"] and not g["strong_classes"] and g["weak_classes"]
    ]
    unlabeled = [
        g
        for g in groups
        if g["complete_representation"]
        and not g["strong_classes"]
        and not g["weak_classes"]
    ]
    excluded = [g for g in groups if not g["complete_representation"]]

    enabled_classes = [
        name
        for name in CLASS_NAMES
        if policy["class_supervision"][name]["status"] == "ENABLED"
    ]
    support_counts = {
        name: sum(name in group["strong_classes"] for group in strong)
        for name in enabled_classes
    }
    scarce = {name: count for name, count in support_counts.items() if count < 3}
    if scarce:
        raise RuntimeError(
            "v3 population lacks three represented strong leakage groups for "
            f"TRAIN/MODEL_SELECTION/INTERNAL_AUDIT coverage: {scarce}"
        )

    assigned: dict[str, str] = {}

    def pick_coverage(role: str) -> None:
        for class_name in enabled_classes:
            choices = [
                group
                for group in strong
                if group["group_id"] not in assigned
                and class_name in group["strong_classes"]
            ]
            choices.sort(
                key=lambda group: _ranking(
                    grouping_sha,
                    policy_sha,
                    group["group_id"],
                    f"coverage:{role}:{class_name}",
                )
            )
            if not choices:
                raise RuntimeError(f"cannot reserve v3 {role} coverage for {class_name}")
            assigned[choices[0]["group_id"]] = role

    pick_coverage("MODEL_SELECTION")
    pick_coverage("INTERNAL_AUDIT")

    target_model = max(len(enabled_classes), round(len(strong) * MODEL_FRACTION))
    target_audit = max(len(enabled_classes), round(len(strong) * AUDIT_FRACTION))

    def role_count(role: str) -> int:
        return sum(value == role for value in assigned.values())

    remaining = [group for group in strong if group["group_id"] not in assigned]
    remaining.sort(
        key=lambda group: _ranking(
            grouping_sha, policy_sha, group["group_id"], "strong-fill"
        )
    )
    for group in remaining:
        gid = group["group_id"]
        if role_count("MODEL_SELECTION") < target_model:
            assigned[gid] = "MODEL_SELECTION"
        elif role_count("INTERNAL_AUDIT") < target_audit:
            assigned[gid] = "INTERNAL_AUDIT"
        else:
            assigned[gid] = "TRAIN_STRONG"

    for group in strong:
        assigned.setdefault(group["group_id"], "TRAIN_STRONG")
    for group in weak:
        assigned[group["group_id"]] = "TRAIN_WEAK"
    for group in unlabeled:
        assigned[group["group_id"]] = "TRAIN_UNLABELED"
    for group in excluded:
        assigned[group["group_id"]] = "EXCLUDED"

    if len(assigned) != len(groups):
        raise AssertionError("not every corrected v3 group received exactly one role")

    strong_coverage_by_role = {
        role: {
            class_name: sum(
                assigned[group["group_id"]] == role
                and class_name in group["strong_classes"]
                for group in strong
            )
            for class_name in enabled_classes
        }
        for role in ("TRAIN_STRONG", "MODEL_SELECTION", "INTERNAL_AUDIT")
    }
    missing_role_coverage = {
        role: [name for name, count in counts.items() if count < 1]
        for role, counts in strong_coverage_by_role.items()
        if any(count < 1 for count in counts.values())
    }
    if missing_role_coverage:
        raise RuntimeError(
            "enabled-class strong coverage is missing after v3 role freeze: "
            f"{missing_role_coverage}"
        )

    group_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: item["group_id"]):
        role = assigned[group["group_id"]]
        group_rows.append(
            {
                "schema": "r4-repaired-role-group-row-v3",
                "partition_version": ROLE_PARTITION_VERSION_V3,
                **group,
                "role": role,
                "assignment_rank_sha256": _ranking(
                    grouping_sha, policy_sha, group["group_id"], f"role:{role}"
                ),
            }
        )
        for contract_id in group["contract_ids"]:
            contract_rows.append(
                {
                    "schema": "r4-repaired-contract-role-row-v3",
                    "partition_version": ROLE_PARTITION_VERSION_V3,
                    "contract_id": contract_id,
                    "group_id": group["group_id"],
                    "role": role,
                }
            )

    role_group_counts = Counter(row["role"] for row in group_rows)
    role_contract_counts = Counter(row["role"] for row in contract_rows)
    manifest = {
        "schema": "r4-repaired-partition-manifest-v3",
        "partition_version": ROLE_PARTITION_VERSION_V3,
        "grouping_version": GROUPING_VERSION_V3,
        "status": "LOCAL_LOGICAL_REBUILD_CANDIDATE_NOT_G8_AUTHORIZED",
        "population_contracts": len(contract_rows),
        "population_groups": len(group_rows),
        "role_group_counts": dict(sorted(role_group_counts.items())),
        "role_contract_counts": dict(sorted(role_contract_counts.items())),
        "represented_strong_groups_by_class": support_counts,
        "strong_group_coverage_by_role_and_class": strong_coverage_by_role,
        "threshold_fit": "UNSUPPORTED_EMPTY",
        "calibration_fit": "UNSUPPORTED_EMPTY",
        "untouched_acceptance": "UNSUPPORTED_EMPTY_FROZEN",
        "confirmed_negative_rows": 0,
        "model_fraction": MODEL_FRACTION,
        "internal_audit_fraction": AUDIT_FRACTION,
        "address_literal_grouping_authority": False,
    }
    return group_rows, sorted(contract_rows, key=lambda row: row["contract_id"]), manifest


def _role_semantic_row_v3(row: dict[str, Any], role: str) -> dict[str, Any]:
    strength = str(row["training_strength"])
    source_loss = bool(row["source_policy_loss_eligible"])
    metric = strength == "STRONG" and role in {"MODEL_SELECTION", "INTERNAL_AUDIT"}
    effective = source_loss and (
        (strength == "STRONG" and role == "TRAIN_STRONG")
        or (strength == "WEAK" and role == "TRAIN_WEAK")
    )
    return {
        "policy_version": row["policy_version"],
        "contract_id": row["contract_id"],
        "class_index": row["class_index"],
        "class_name": row["class_name"],
        "historical_state": row["historical_state"],
        "source_claims": row["source_claims"],
        "outcome_state": row["outcome_state"],
        "target_value": row["target_value"],
        "training_signal": row["training_signal"],
        "training_strength": strength,
        "loss_eligible": source_loss,
        "outcome_metric_eligible": metric,
        "role_eligibility": [role] + ([] if metric else ["EXCLUDE_OUTCOME_METRICS"]),
        "policy_decision_id": row["policy_decision_id"],
        "evidence_ids": row["evidence_ids"],
        "limitations": sorted(set([*row["limitations"], row["reason_code"]])),
        "effective_loss_mask": effective,
    }


def build_logical_v3_publication(
    *,
    claims_path: Path,
    grouping_path: Path,
    policy_path: Path,
    representation_root: Path,
    source_ledger_path: Path,
    source_ledger_manifest_path: Path,
    source_v2_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build V3 logical publication over accepted V2 physical artifacts."""

    pa, pq = _require_pyarrow()
    claims = _load_jsonl(claims_path)
    grouping = json.loads(grouping_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    validate_policy_surface(policy)

    if grouping.get("grouping_version") != GROUPING_VERSION_V3:
        raise ValueError(
            f"logical-v3 requires {GROUPING_VERSION_V3}, got {grouping.get('grouping_version')!r}"
        )
    if (grouping.get("address_diagnostics") or {}).get("used_as_grouping_authority") is not False:
        raise ValueError("logical-v3 grouping must explicitly disable address authority")

    from sentinel_data.preprocessing.r4_completeness import (
        require_complete_representation_sources,
    )

    representation_manifests = require_complete_representation_sources(
        representation_root,
        grouping.get("preprocessing_manifests") or {},
    )

    grouping_sha = _sha256_file(grouping_path)
    policy_sha = _sha256_file(policy_path)
    claims_sha = _sha256_file(claims_path)

    ledger_manifest = json.loads(source_ledger_manifest_path.read_text(encoding="utf-8"))
    if ledger_manifest.get("ledger_version") != SOURCE_EVIDENCE_LEDGER_VERSION:
        raise ValueError("logical-v3 source evidence-ledger version mismatch")
    ledger_artifacts = ledger_manifest.get("artifacts") or {}
    required_source_inputs = {
        "ledger": _sha256_file(source_ledger_path),
        "source_claims": claims_sha,
        "policy": policy_sha,
    }
    for name, actual_sha in required_source_inputs.items():
        if (ledger_artifacts.get(name) or {}).get("sha256") != actual_sha:
            raise ValueError(f"logical-v3 source evidence-ledger {name} hash mismatch")

    # Recompute semantic rows against V3 population/representation availability.
    # Group ids do not appear in the role-independent semantic ledger, so this
    # equality proves the grouping correction did not rewrite label semantics.
    recomputed_semantics, artifact_info = build_semantic_cells(
        claims, grouping, policy, representation_root
    )
    ledger_semantics = pq.read_table(source_ledger_path).to_pylist()
    sort_key = lambda row: (str(row["contract_id"]), int(row["class_index"]))
    recomputed_semantics.sort(key=sort_key)
    ledger_semantics.sort(key=sort_key)
    if json.loads(json.dumps(recomputed_semantics, sort_keys=True)) != json.loads(
        json.dumps(ledger_semantics, sort_keys=True)
    ):
        raise ValueError(
            "logical-v3 semantic recomputation diverges from accepted V2 evidence ledger"
        )

    parent_manifest = json.loads(source_v2_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("dataset_version") != "sentinel-r4-vnext-v2":
        raise ValueError("logical-v3 parent publication must be accepted repaired-v2")
    parent_binding = parent_manifest.get("representation_binding_report") or {}
    if not parent_binding.get("binding_digest_sha256"):
        raise ValueError("logical-v3 parent publication lacks physical binding digest")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"logical-v3 publication output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    group_rows, contract_roles, partition = _freeze_roles_v3(
        ledger_semantics,
        artifact_info,
        grouping,
        policy,
        grouping_sha=grouping_sha,
        policy_sha=policy_sha,
    )
    role_by_contract = {
        row["contract_id"]: (row["role"], row["group_id"])
        for row in contract_roles
    }

    semantic_rows: list[dict[str, Any]] = []
    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger_semantics:
        role, _ = role_by_contract[row["contract_id"]]
        final = _role_semantic_row_v3(row, role)
        semantic_rows.append(final)
        by_contract[row["contract_id"]].append(final)

    ml_rows: list[dict[str, Any]] = []
    for contract_id in sorted(by_contract):
        role, group_id = role_by_contract[contract_id]
        info = artifact_info[contract_id]
        ordered = sorted(by_contract[contract_id], key=lambda row: row["class_index"])
        row: dict[str, Any] = {
            "contract_id": contract_id,
            "source": info["representation_source"] or sorted(info["sources"])[0],
            "source_claim_sources": sorted(info["sources"]),
            "group_id": group_id,
            "role": role,
            "representation_required": role != "EXCLUDED",
        }
        for cell in ordered:
            idx = int(cell["class_index"])
            row[f"target_{idx}"] = cell["target_value"]
            row[f"strength_{idx}"] = cell["training_strength"]
            row[f"source_loss_eligible_{idx}"] = bool(cell["loss_eligible"])
            row[f"effective_loss_mask_{idx}"] = bool(cell["effective_loss_mask"])
            row[f"outcome_metric_mask_{idx}"] = bool(cell["outcome_metric_eligible"])
            row[f"outcome_state_{idx}"] = cell["outcome_state"]
            row[f"policy_decision_id_{idx}"] = cell["policy_decision_id"]
        ml_rows.append(row)

    if any(
        row.get(f"target_{idx}") == 0
        for row in ml_rows
        for idx in range(len(CLASS_NAMES))
    ):
        raise AssertionError("logical-v3 publication contains target 0")

    label_states_path = output_dir / "label_states.parquet"
    ml_targets_path = output_dir / "ml_targets.parquet"
    pq.write_table(pa.Table.from_pylist(semantic_rows), label_states_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(ml_rows), ml_targets_path, compression="zstd")
    _write_jsonl(output_dir / "group_roles.jsonl", group_rows)
    _write_jsonl(output_dir / "contract_roles.jsonl", contract_roles)
    _write_json(output_dir / "partition_manifest.json", partition)
    _write_json(
        output_dir / "unsupported_roles.json",
        {
            "partition_version": ROLE_PARTITION_VERSION_V3,
            "THRESHOLD_FIT": "UNSUPPORTED_EMPTY",
            "CALIBRATION_FIT": "UNSUPPORTED_EMPTY",
            "UNTOUCHED_ACCEPTANCE": "UNSUPPORTED_EMPTY_FROZEN",
        },
    )
    _write_json(
        output_dir / "untouched_acceptance.json",
        {
            "partition_version": ROLE_PARTITION_VERSION_V3,
            "status": "UNSUPPORTED_EMPTY_FROZEN",
            "frozen": True,
            "contract_ids": [],
            "group_ids": [],
        },
    )
    _write_json(
        output_dir / "source_registry.json",
        {
            "policy_version": policy["policy_version"],
            "sources": policy["sources"],
            "claim_index_sha256": claims_sha,
        },
    )
    _write_json(
        output_dir / "crosswalk_registry.json",
        {
            "policy_version": policy["policy_version"],
            "smartbugs_curated": policy["sources"]["smartbugs_curated"],
            "solidifi": policy["sources"]["solidifi"],
            "dive": policy["sources"]["dive"],
        },
    )

    target_counts = Counter(
        "None" if cell["target_value"] is None else str(cell["target_value"])
        for cell in semantic_rows
    )
    strength_counts = Counter(cell["training_strength"] for cell in semantic_rows)
    role_counts = Counter(row["role"] for row in ml_rows)

    manifest = {
        "dataset_version": DATASET_VERSION_V3,
        "logical_build_version": LOGICAL_BUILD_VERSION_V3,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "status": "LOGICAL_V3_CANDIDATE_LOCAL_BINDING_REQUIRED",
        "policy_version": policy["policy_version"],
        "ledger_version": SOURCE_EVIDENCE_LEDGER_VERSION,
        "partition_version": ROLE_PARTITION_VERSION_V3,
        "grouping_version": GROUPING_VERSION_V3,
        "population": {
            "contracts": len(ml_rows),
            "contract_class_rows": len(semantic_rows),
            "classes": len(CLASS_NAMES),
        },
        "class_names": list(CLASS_NAMES),
        "target_counts": dict(sorted(target_counts.items())),
        "training_strength_counts": dict(sorted(strength_counts.items())),
        "effective_loss_cells": sum(cell["effective_loss_mask"] for cell in semantic_rows),
        "outcome_metric_cells": sum(cell["outcome_metric_eligible"] for cell in semantic_rows),
        "role_contract_counts": dict(sorted(role_counts.items())),
        "confirmed_negative_rows": 0,
        "address_literal_grouping_authority": False,
        "physical_artifacts_reused": True,
        "parent_v2": {
            "dataset_version": parent_manifest["dataset_version"],
            "manifest_sha256": _sha256_file(source_v2_manifest_path),
            "representation_binding_digest_sha256": parent_binding[
                "binding_digest_sha256"
            ],
        },
        "artifacts": {
            "label_states": {
                "path": "label_states.parquet",
                "sha256": _sha256_file(label_states_path),
            },
            "ml_targets": {
                "path": "ml_targets.parquet",
                "sha256": _sha256_file(ml_targets_path),
            },
            "claims": {
                "path": "r4-v2-build/source_claims.jsonl",
                "sha256": claims_sha,
            },
            "grouping": {
                "path": "r4-v3-logical-build/grouping.json",
                "sha256": grouping_sha,
            },
            "policy": {
                "path": "docs/plan/ml-R4/specs/data_vnext_policy_v1.json",
                "sha256": policy_sha,
            },
            "source_evidence_ledger": {
                "path": "r4-v2-build/evidence_ledger_v2.parquet",
                "sha256": _sha256_file(source_ledger_path),
            },
            "source_evidence_ledger_manifest": {
                "path": "r4-v2-build/evidence_ledger_v2_manifest.json",
                "sha256": _sha256_file(source_ledger_manifest_path),
            },
            "representation_manifests": {
                "path": "representations-r4-v2/*/repaired_representation_manifest.json",
                "sha256_by_source": {
                    source: value["manifest_sha256"]
                    for source, value in sorted(representation_manifests.items())
                },
            },
        },
        "representation_root_recorded": False,
        "representation_binding_report": None,
        "limitations": [
            "V3 changes grouping/roles only; physical repaired-v2 source and representation bytes are reused.",
            "No confirmed-negative evidence exists; threshold/calibration/acceptance roles remain unsupported.",
            "Selector and GPU research must be regenerated against V3 roles before promotion decisions.",
            "The previous V2 negative-review queue is obsolete because group reservations changed.",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def bind_logical_v3_publication(
    *,
    publication_dir: Path,
    representations_root: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Reuse the accepted physical V2 validator against a V3 logical manifest.

    The physical representation contract is unchanged.  A temporary manifest
    shim changes only the dataset-version discriminator so the existing V2
    physical validator can perform the exact same graph/token/sidecar checks.
    The resulting report is then rebound to the V3 manifest with the V3 dataset
    identity restored.  No physical artifact is copied or modified.
    """

    from sentinel_data.preprocessing.r4_versions import REPAIRED_DATA_PUBLICATION_ID
    from sentinel_data.vnext.r4_binding import bind_repaired_publication

    publication_dir = Path(publication_dir)
    manifest_path = publication_dir / "manifest.json"
    ml_targets_path = publication_dir / "ml_targets.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != DATASET_VERSION_V3:
        raise ValueError("logical-v3 binder requires sentinel-r4-vnext-v3")
    if manifest.get("address_literal_grouping_authority") is not False:
        raise ValueError("logical-v3 manifest does not disable address grouping authority")

    with tempfile.TemporaryDirectory(prefix="sentinel-r4-v3-bind-") as temp_name:
        temp = Path(temp_name)
        shim_manifest = json.loads(json.dumps(manifest))
        shim_manifest["dataset_version"] = REPAIRED_DATA_PUBLICATION_ID
        shim_manifest["representation_binding_report"] = None
        (temp / "manifest.json").write_text(
            json.dumps(shim_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.symlink(ml_targets_path.resolve(), temp / "ml_targets.parquet")
        shim_report_path = temp / "representation_binding_report.json"
        report = bind_repaired_publication(
            publication_dir=temp,
            representations_root=representations_root,
            report_path=shim_report_path,
        )

    report = json.loads(json.dumps(report))
    report["schema"] = "sentinel-r4-logical-v3-representation-binding-v1"
    report["dataset_version"] = DATASET_VERSION_V3
    report["physical_validator_contract_reused_from"] = REPAIRED_DATA_PUBLICATION_ID
    report["logical_grouping_version"] = GROUPING_VERSION_V3
    report["logical_partition_version"] = ROLE_PARTITION_VERSION_V3
    report["address_literal_grouping_authority"] = False

    if report_path is None:
        report_path = publication_dir / "representation_binding_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if report.get("passed"):
        manifest["representation_binding_report"] = {
            "path": report_path.name,
            "sha256": _sha256_file(report_path),
            "binding_digest_sha256": report["binding_digest_sha256"],
        }
        manifest["status"] = "LOGICAL_V3_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


__all__ = ["bind_logical_v3_publication", "build_logical_v3_publication"]
