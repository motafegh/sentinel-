"""Source-native claim reconstruction for the repaired R4-v2 lineage.

The v1 evidence ledger is immutable and cannot describe source records that were
physically lost during historical preprocessing.  This module reconstructs
*source claims* from repaired preprocessing provenance and the already-accepted
``data-vnext-policy-v1``.  It never invents negative evidence.

Notably, SmartBugs ``time_manipulation`` remains distinct from
``bad_randomness`` before policy mapping, closing the historical Timestamp
ambiguity identified by the Phase-8 real-data audit.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from sentinel_data.vnext.policy import CLASS_NAMES, validate_policy_surface

_DIVE_CANONICAL = {
    "Reentrancy": "Reentrancy",
    "Access Control": "ExternalBug",
    "Arithmetic": "IntegerUO",
    "Unchecked Return Values": "UnusedReturn",
    "DoS": "DenialOfService",
    "Front Running": "TransactionOrderDependence",
    "Time manipulation": "Timestamp",
    "Bad Randomness": None,
}


def _path_category(path: str, candidates: set[str]) -> str | None:
    parts = Path(path).parts
    matches = [part for part in parts if part in candidates]
    if len(matches) > 1:
        raise ValueError(f"ambiguous source category in {path!r}: {matches}")
    return matches[0] if matches else None


def load_dive_labels(path: Path) -> dict[str, set[str]]:
    """Load the protected DIVE CSV as contract-id -> positive native categories."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "contractID" not in rows[0]:
        raise ValueError(f"DIVE labels CSV missing contractID: {path}")
    category_columns = [key for key in rows[0] if key != "contractID"]
    result: dict[str, set[str]] = {}
    for row in rows:
        contract_id = str(row["contractID"]).strip()
        if not contract_id or contract_id in result:
            raise ValueError(f"invalid/duplicate DIVE contractID: {contract_id!r}")
        positives: set[str] = set()
        for category in category_columns:
            value = str(row.get(category, "")).strip()
            if value not in {"0", "1"}:
                raise ValueError(
                    f"DIVE label must be binary: {contract_id}:{category}={value!r}"
                )
            if value == "1":
                positives.add(category)
        result[contract_id] = positives
    return result


def _claim(
    *,
    artifact_id: str,
    source: str,
    record: dict[str, Any],
    category: str | None,
    mapped_class: str | None,
    training_strength: str,
    target_value: int | None,
    outcome_state: str,
    reason: str,
    policy_version: str,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "source": source,
        "source_record_id": record.get("source_record_id"),
        "original_path": record.get("original_path"),
        "native_category": category,
        "source_claim_state": "POSITIVE" if category else "NO_ASSERTION",
        "mapped_class_name": mapped_class,
        "target_value": target_value,
        "training_strength": training_strength,
        "outcome_state": outcome_state,
        "policy_version": policy_version,
        "reason_code": reason,
        "evidence_ref": {
            "provenance_schema": "r4-provenance-v1",
            "raw_sha256": record.get("raw_sha256"),
            "flattened_sha256": record.get("flattened_sha256"),
            "ingestion_entry": record.get("ingestion_entry") or {},
        },
    }


def claims_for_meta(
    source: str,
    meta: dict[str, Any],
    policy: dict[str, Any],
    *,
    dive_labels: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return all source-native claims carried by one repaired artifact."""

    validate_policy_surface(policy)
    artifact_id = str(meta["sha256"])
    records = list(meta.get("source_records") or [])
    if not records:
        raise ValueError(f"repaired artifact has no source_records: {artifact_id}")
    rows: list[dict[str, Any]] = []

    if source == "smartbugs_curated":
        cfg = policy["sources"]["smartbugs_curated"]
        approved = dict(cfg["approved_mappings"])
        no_target = dict(cfg["no_target_categories"])
        candidates = set(approved) | set(no_target)
        for record in records:
            category = _path_category(str(record.get("original_path") or ""), candidates)
            if category is None:
                rows.append(
                    _claim(
                        artifact_id=artifact_id,
                        source=source,
                        record=record,
                        category=None,
                        mapped_class=None,
                        training_strength="NONE",
                        target_value=None,
                        outcome_state="UNKNOWN",
                        reason="SMARTBUGS_CATEGORY_NOT_BOUND_IN_PROVENANCE",
                        policy_version=policy["policy_version"],
                    )
                )
                continue
            if category in approved:
                mapped = str(approved[category])
                rows.append(
                    _claim(
                        artifact_id=artifact_id,
                        source=source,
                        record=record,
                        category=category,
                        mapped_class=mapped,
                        training_strength="STRONG",
                        target_value=1,
                        outcome_state="CONFIRMED_POSITIVE",
                        reason="SMARTBUGS_APPROVED_NATIVE_CATEGORY_STRONG_POSITIVE",
                        policy_version=policy["policy_version"],
                    )
                )
            else:
                rows.append(
                    _claim(
                        artifact_id=artifact_id,
                        source=source,
                        record=record,
                        category=category,
                        mapped_class=None,
                        training_strength="NONE",
                        target_value=None,
                        outcome_state="NOT_REVIEWED",
                        reason=f"SMARTBUGS_{no_target[category]}",
                        policy_version=policy["policy_version"],
                    )
                )
        return rows

    if source == "solidifi":
        approved = dict(policy["sources"]["solidifi"]["direct_or_approved_mappings"])
        candidates = set(approved)
        for record in records:
            category = _path_category(str(record.get("original_path") or ""), candidates)
            if category is None:
                rows.append(
                    _claim(
                        artifact_id=artifact_id,
                        source=source,
                        record=record,
                        category=None,
                        mapped_class=None,
                        training_strength="NONE",
                        target_value=None,
                        outcome_state="UNKNOWN",
                        reason="SOLIDIFI_INJECTION_CATEGORY_NOT_BOUND_IN_PROVENANCE",
                        policy_version=policy["policy_version"],
                    )
                )
            else:
                rows.append(
                    _claim(
                        artifact_id=artifact_id,
                        source=source,
                        record=record,
                        category=category,
                        mapped_class=str(approved[category]),
                        training_strength="STRONG",
                        target_value=1,
                        outcome_state="CONFIRMED_POSITIVE",
                        reason="SOLIDIFI_INJECTED_CLASS_STRONG_POSITIVE",
                        policy_version=policy["policy_version"],
                    )
                )
        return rows

    if source == "dive":
        if dive_labels is None:
            raise ValueError("DIVE repaired source claims require the protected labels CSV")
        mapped_policy = policy["sources"]["dive"]["mapped_category_policy"]
        for record in records:
            contract_id = Path(str(record.get("original_path") or "")).stem
            if contract_id not in dive_labels:
                raise ValueError(
                    f"DIVE source record absent from labels CSV: {contract_id}"
                )
            positives = sorted(dive_labels[contract_id])
            if not positives:
                rows.append(
                    _claim(
                        artifact_id=artifact_id,
                        source=source,
                        record=record,
                        category=None,
                        mapped_class=None,
                        training_strength="NONE",
                        target_value=None,
                        outcome_state="UNKNOWN",
                        reason="DIVE_NO_POSITIVE_SOURCE_CLAIM",
                        policy_version=policy["policy_version"],
                    )
                )
                continue
            for category in positives:
                mapped = _DIVE_CANONICAL.get(category)
                if category == "Front Running":
                    cfg = mapped_policy[category]
                    if cfg.get("training_strength") != "WEAK" or cfg.get("target_value") != 1:
                        raise ValueError("accepted DIVE Front Running policy changed unexpectedly")
                    strength = "WEAK"
                    target = 1
                    reason = "DIVE_TOD_WEAK_POSITIVE_ONLY"
                else:
                    strength = "NONE"
                    target = None
                    reason = "DIVE_NATIVE_POSITIVE_MASKED_BY_ACCEPTED_POLICY"
                rows.append(
                    _claim(
                        artifact_id=artifact_id,
                        source=source,
                        record=record,
                        category=category,
                        mapped_class=mapped,
                        training_strength=strength,
                        target_value=target,
                        outcome_state="NOT_REVIEWED",
                        reason=reason,
                        policy_version=policy["policy_version"],
                    )
                )
        return rows

    raise ValueError(f"source is not active in repaired baseline: {source}")


def build_claim_index(
    source_dirs: dict[str, Path],
    policy_path: Path,
    output_path: Path,
    *,
    dive_labels_csv: Path | None = None,
    verify_completeness: bool = True,
) -> dict[str, Any]:
    """Materialize deterministic JSONL source claims from repaired meta files."""

    from sentinel_data.preprocessing.r4_completeness import (
        require_complete_preprocessed_sources,
    )

    preprocessing_manifests = (
        require_complete_preprocessed_sources(source_dirs)
        if verify_completeness
        else {}
    )

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    validate_policy_surface(policy)
    dive_labels = load_dive_labels(dive_labels_csv) if dive_labels_csv else None
    rows: list[dict[str, Any]] = []
    artifact_ids: set[str] = set()
    for source, directory in sorted(source_dirs.items()):
        for meta_path in sorted(directory.glob("*.meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            artifact_id = str(meta["sha256"])
            artifact_ids.add(artifact_id)
            rows.extend(
                claims_for_meta(
                    source,
                    meta,
                    policy,
                    dive_labels=dive_labels,
                )
            )
    rows.sort(
        key=lambda row: (
            row["artifact_id"],
            str(row.get("source_record_id") or ""),
            str(row.get("native_category") or ""),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "policy_version": policy["policy_version"],
        "artifacts": len(artifact_ids),
        "claims": len(rows),
        "strong_positive_claims": sum(r["training_strength"] == "STRONG" for r in rows),
        "weak_positive_claims": sum(r["training_strength"] == "WEAK" for r in rows),
        "target_zero_claims": sum(r["target_value"] == 0 for r in rows),
        "class_names": list(CLASS_NAMES),
        "preprocessing_manifest_sha256": {
            source: value["manifest_sha256"]
            for source, value in sorted(preprocessing_manifests.items())
        },
        "output": str(output_path),
    }
