"""Build the repaired R4-v2 semantic overlay and leakage-safe role freeze.

This is a *new* lineage builder.  It does not modify or reinterpret the frozen
Phase-3/Phase-6/vNext-v1 artifacts.  Population counts are derived from the
physical repaired preprocessing/representation outputs at local rebuild time.

Permanent invariants remain policy-v1 invariants:

* no target ``0`` without confirmed-negative evidence (there is none here);
* strong and weak positive evidence remain distinct;
* role assignment is group-atomic after final repaired grouping;
* threshold-fit, calibration-fit, and untouched acceptance remain empty;
* disabled classes cannot receive supervised signal.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sentinel_data.preprocessing.r4_versions import (
    REPAIRED_DATA_PUBLICATION_ID,
    REPAIRED_EVIDENCE_LEDGER_ID,
    REPAIRED_ROLE_PARTITION_ID,
)
from sentinel_data.vnext.policy import CLASS_NAMES, validate_policy_surface

MODEL_FRACTION = 0.15
AUDIT_FRACTION = 0.15
EXPORT_SCHEMA_VERSION = "v2"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        raise RuntimeError("repaired DATA publication requires pyarrow") from exc
    return pa, pq


def _representation_sources(
    artifact: str,
    candidate_sources: list[str],
    representation_root: Path,
) -> list[str]:
    available: list[str] = []
    for source in sorted(set(candidate_sources)):
        root = representation_root / source
        required = (
            root / f"{artifact}.pt",
            root / f"{artifact}.tokens.pt",
            root / f"{artifact}.rep.json",
        )
        if all(path.is_file() for path in required):
            available.append(source)
    return available


def build_semantic_cells(
    claims: list[dict[str, Any]],
    grouping: dict[str, Any],
    policy: dict[str, Any],
    representation_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build role-independent contract×class semantic cells."""

    validate_policy_surface(policy)
    claims_by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        if claim.get("target_value") == 0:
            raise ValueError("repaired source claim illegally contains target 0")
        claims_by_artifact[str(claim["artifact_id"])].append(claim)

    artifact_sources = {
        str(k): [str(v) for v in values]
        for k, values in (grouping.get("artifact_sources") or {}).items()
    }
    artifact_to_group = {
        str(k): str(v)
        for k, v in (grouping.get("artifact_to_group") or {}).items()
    }
    if set(claims_by_artifact) != set(artifact_to_group):
        missing_claims = sorted(set(artifact_to_group) - set(claims_by_artifact))[:10]
        missing_groups = sorted(set(claims_by_artifact) - set(artifact_to_group))[:10]
        raise ValueError(
            "claims/grouping population mismatch: "
            f"without_claims={missing_claims}, without_groups={missing_groups}"
        )

    artifact_info: dict[str, dict[str, Any]] = {}
    semantic_rows: list[dict[str, Any]] = []
    for artifact in sorted(claims_by_artifact):
        source_claims = sorted(
            claims_by_artifact[artifact],
            key=lambda row: (
                str(row.get("source") or ""),
                str(row.get("source_record_id") or ""),
                str(row.get("native_category") or ""),
            ),
        )
        candidates = artifact_sources.get(artifact) or sorted(
            {str(row["source"]) for row in source_claims}
        )
        available_sources = _representation_sources(
            artifact, candidates, representation_root
        )
        artifact_info[artifact] = {
            "group_id": artifact_to_group[artifact],
            "sources": candidates,
            "representation_sources": available_sources,
            "representation_available": bool(available_sources),
            "representation_source": available_sources[0] if available_sources else None,
        }

        for class_index, class_name in enumerate(CLASS_NAMES):
            class_cfg = policy["class_supervision"][class_name]
            enabled = class_cfg["status"] == "ENABLED"
            mapped = [
                row for row in source_claims
                if row.get("mapped_class_name") == class_name
            ]
            strong = [
                row for row in mapped
                if row.get("training_strength") == "STRONG"
                and row.get("target_value") == 1
            ]
            weak = [
                row for row in mapped
                if row.get("training_strength") == "WEAK"
                and row.get("target_value") == 1
            ]
            if not enabled:
                target = None
                strength = "NONE"
                signal = "NONE"
                outcome = "NOT_REVIEWED" if mapped else "UNKNOWN"
                reason = "SUPERVISION_DISABLED_PENDING_EVIDENCE"
            elif strong:
                target = 1
                strength = "STRONG"
                signal = "POSITIVE"
                outcome = "CONFIRMED_POSITIVE"
                reason = "REPAIRED_STRONG_SOURCE_CLAIM"
            elif weak:
                target = 1
                strength = "WEAK"
                signal = "POSITIVE"
                outcome = "NOT_REVIEWED"
                reason = "REPAIRED_WEAK_SOURCE_CLAIM"
            else:
                target = None
                strength = "NONE"
                signal = "NONE"
                outcome = "NOT_REVIEWED" if mapped else "UNKNOWN"
                reason = "NO_AUTHORIZED_CLASS_TARGET"

            semantic_rows.append(
                {
                    "ledger_version": REPAIRED_EVIDENCE_LEDGER_ID,
                    "policy_version": policy["policy_version"],
                    "contract_id": artifact,
                    "class_index": class_index,
                    "class_name": class_name,
                    "historical_state": "HISTORICAL_MISSING",
                    "source_claims": source_claims,
                    "outcome_state": outcome,
                    "target_value": target,
                    "training_signal": signal,
                    "training_strength": strength,
                    "source_policy_loss_eligible": strength in {"STRONG", "WEAK"},
                    "policy_decision_id": "R4-D-002",
                    "reason_code": reason,
                    "evidence_ids": sorted(
                        {
                            f"source-record:{row.get('source_record_id')}"
                            for row in mapped
                            if row.get("source_record_id")
                        }
                    ),
                    "limitations": sorted(
                        {
                            str(row.get("reason_code"))
                            for row in source_claims
                            if row.get("reason_code")
                        }
                    ),
                }
            )

    if any(row["target_value"] == 0 for row in semantic_rows):
        raise AssertionError("repaired semantic builder synthesized target 0")
    return semantic_rows, artifact_info


def _ranking(
    grouping_sha: str,
    policy_sha: str,
    group_id: str,
    salt: str,
) -> str:
    payload = (
        f"{REPAIRED_ROLE_PARTITION_ID}|{grouping_sha}|{policy_sha}|{salt}|{group_id}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_roles(
    semantic_rows: list[dict[str, Any]],
    artifact_info: dict[str, dict[str, Any]],
    grouping: dict[str, Any],
    policy: dict[str, Any],
    *,
    grouping_sha: str,
    policy_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Assign every final leakage group exactly one repaired Phase-8 role."""

    cells_by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in semantic_rows:
        cells_by_contract[str(row["contract_id"])].append(row)

    groups: list[dict[str, Any]] = []
    for raw_group in grouping.get("groups") or []:
        gid = str(raw_group["group_id"])
        members = [str(x) for x in raw_group["members"]]
        missing = [cid for cid in members if cid not in cells_by_contract]
        if missing:
            raise ValueError(f"group contains contracts without semantic cells: {missing[:5]}")
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
        g for g in groups
        if g["complete_representation"]
        and not g["strong_classes"]
        and g["weak_classes"]
    ]
    unlabeled = [
        g for g in groups
        if g["complete_representation"]
        and not g["strong_classes"]
        and not g["weak_classes"]
    ]
    excluded = [g for g in groups if not g["complete_representation"]]

    enabled_classes = [
        name for name in CLASS_NAMES
        if policy["class_supervision"][name]["status"] == "ENABLED"
    ]
    support_counts = {
        name: sum(name in g["strong_classes"] for g in strong)
        for name in enabled_classes
    }
    scarce = {name: count for name, count in support_counts.items() if count < 3}
    if scarce:
        raise RuntimeError(
            "repaired population lacks three represented strong leakage groups "
            f"for TRAIN/MODEL_SELECTION/INTERNAL_AUDIT coverage: {scarce}"
        )

    assigned: dict[str, str] = {}

    def pick_coverage(role: str) -> None:
        for class_name in enabled_classes:
            choices = [
                g for g in strong
                if g["group_id"] not in assigned
                and class_name in g["strong_classes"]
            ]
            choices.sort(
                key=lambda g: _ranking(
                    grouping_sha,
                    policy_sha,
                    g["group_id"],
                    f"coverage:{role}:{class_name}",
                )
            )
            if not choices:
                raise RuntimeError(f"cannot reserve {role} coverage for {class_name}")
            assigned[choices[0]["group_id"]] = role

    pick_coverage("MODEL_SELECTION")
    pick_coverage("INTERNAL_AUDIT")

    target_model = max(len(enabled_classes), round(len(strong) * MODEL_FRACTION))
    target_audit = max(len(enabled_classes), round(len(strong) * AUDIT_FRACTION))

    def role_count(role: str) -> int:
        return sum(value == role for value in assigned.values())

    remaining = [g for g in strong if g["group_id"] not in assigned]
    remaining.sort(
        key=lambda g: _ranking(
            grouping_sha, policy_sha, g["group_id"], "strong-fill"
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
        raise AssertionError("not every repaired group received exactly one role")

    group_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: item["group_id"]):
        role = assigned[group["group_id"]]
        group_rows.append(
            {
                "schema": "r4-repaired-role-group-row-v2",
                "partition_version": REPAIRED_ROLE_PARTITION_ID,
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
                    "schema": "r4-repaired-contract-role-row-v2",
                    "partition_version": REPAIRED_ROLE_PARTITION_ID,
                    "contract_id": contract_id,
                    "group_id": group["group_id"],
                    "role": role,
                }
            )

    role_group_counts = Counter(row["role"] for row in group_rows)
    role_contract_counts = Counter(row["role"] for row in contract_rows)
    manifest = {
        "schema": "r4-repaired-partition-manifest-v2",
        "partition_version": REPAIRED_ROLE_PARTITION_ID,
        "status": "LOCAL_REBUILD_CANDIDATE_NOT_G8_AUTHORIZED",
        "population_contracts": len(contract_rows),
        "population_groups": len(group_rows),
        "role_group_counts": dict(sorted(role_group_counts.items())),
        "role_contract_counts": dict(sorted(role_contract_counts.items())),
        "represented_strong_groups_by_class": support_counts,
        "threshold_fit": "UNSUPPORTED_EMPTY",
        "calibration_fit": "UNSUPPORTED_EMPTY",
        "untouched_acceptance": "UNSUPPORTED_EMPTY_FROZEN",
        "confirmed_negative_rows": 0,
        "model_fraction": MODEL_FRACTION,
        "internal_audit_fraction": AUDIT_FRACTION,
    }
    return group_rows, sorted(contract_rows, key=lambda r: r["contract_id"]), manifest


def _role_semantic_row(row: dict[str, Any], role: str) -> dict[str, Any]:
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


def build_repaired_publication(
    *,
    claims_path: Path,
    grouping_path: Path,
    policy_path: Path,
    representation_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the complete local repaired-v2 candidate publication."""

    pa, pq = _require_pyarrow()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"repaired publication output is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    claims = _load_jsonl(claims_path)
    grouping = json.loads(grouping_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    validate_policy_surface(policy)

    semantic_base, artifact_info = build_semantic_cells(
        claims, grouping, policy, representation_root
    )
    grouping_sha = _sha256_file(grouping_path)
    policy_sha = _sha256_file(policy_path)
    group_rows, contract_roles, partition = freeze_roles(
        semantic_base,
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
    for row in semantic_base:
        role, _ = role_by_contract[row["contract_id"]]
        final = _role_semantic_row(row, role)
        semantic_rows.append(final)
        by_contract[row["contract_id"]].append(final)

    ml_rows: list[dict[str, Any]] = []
    for contract_id in sorted(by_contract):
        role, group_id = role_by_contract[contract_id]
        info = artifact_info[contract_id]
        ordered = sorted(by_contract[contract_id], key=lambda r: r["class_index"])
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
        raise AssertionError("repaired publication contains target 0")

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
            "partition_version": REPAIRED_ROLE_PARTITION_ID,
            "THRESHOLD_FIT": "UNSUPPORTED_EMPTY",
            "CALIBRATION_FIT": "UNSUPPORTED_EMPTY",
            "UNTOUCHED_ACCEPTANCE": "UNSUPPORTED_EMPTY_FROZEN",
        },
    )
    _write_json(
        output_dir / "untouched_acceptance.json",
        {
            "partition_version": REPAIRED_ROLE_PARTITION_ID,
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
            "claim_index_sha256": _sha256_file(claims_path),
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
        "dataset_version": REPAIRED_DATA_PUBLICATION_ID,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "status": "REPAIRED_CANDIDATE_LOCAL_ACCEPTANCE_REQUIRED",
        "policy_version": policy["policy_version"],
        "ledger_version": REPAIRED_EVIDENCE_LEDGER_ID,
        "partition_version": REPAIRED_ROLE_PARTITION_ID,
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
        "artifacts": {
            "label_states": {"path": "label_states.parquet", "sha256": _sha256_file(label_states_path)},
            "ml_targets": {"path": "ml_targets.parquet", "sha256": _sha256_file(ml_targets_path)},
            "claims": {"path": str(claims_path), "sha256": _sha256_file(claims_path)},
            "grouping": {"path": str(grouping_path), "sha256": grouping_sha},
            "policy": {"path": str(policy_path), "sha256": policy_sha},
        },
        "representation_root_recorded": False,
        "representation_binding_report": None,
        "limitations": [
            "Physical representation binding/acceptance must pass before this candidate can replace historical sentinel-r4-vnext-v1.",
            "No confirmed-negative evidence exists; threshold/calibration/acceptance roles remain unsupported.",
            "Token coverage metadata is diagnostic only and does not establish four-window adequacy."
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
