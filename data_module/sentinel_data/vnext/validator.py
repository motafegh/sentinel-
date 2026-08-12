"""Independent semantic/hash validator for DATA vNext v2 overlays."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .builder import (
    DATASET_VERSION,
    EXPECTED_CONTRACTS,
    EXPECTED_EFFECTIVE_LOSS_CELLS,
    EXPECTED_EXCLUDED,
    EXPECTED_OUTCOME_METRIC_CELLS,
    EXPECTED_ROWS,
    EXPECTED_STRENGTH_COUNTS,
    EXPECTED_TARGET_COUNTS,
)
from .policy import CLASS_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_ROLE_COUNTS = {
    "EXCLUDED": 836,
    "INTERNAL_AUDIT": 62,
    "MODEL_SELECTION": 56,
    "TRAIN_STRONG": 275,
    "TRAIN_UNLABELED": 20491,
    "TRAIN_WEAK": 773,
}
EXPECTED_STRONG_BY_SOURCE = {"smartbugs_curated": 120, "solidifi": 283}
EXPECTED_WEAK_BY_SOURCE_CLASS = {("dive", "TransactionOrderDependence"): 604}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DATA vNext validation requires pyarrow") from exc
    return pq


def _validate_input_bindings(manifest: dict[str, Any], errors: list[str]) -> None:
    inputs = manifest.get("inputs") or {}
    required = {
        "ledger",
        "policy",
        "label_schema",
        "partition_manifest",
        "contract_roles",
        "unsupported_roles",
        "untouched_acceptance",
    }
    if set(inputs) - {"schema"} != required:
        errors.append("input_binding_set_mismatch")
        return
    for name in sorted(required):
        meta = inputs.get(name) or {}
        raw = str(meta.get("path") or "")
        path = Path(raw)
        if path.is_absolute():
            errors.append(f"input_path_not_repo_relative:{name}")
            continue
        actual_path = _REPO_ROOT / path
        if not actual_path.is_file():
            errors.append(f"bound_input_missing:{name}")
            continue
        if _sha256(actual_path) != meta.get("sha256"):
            errors.append(f"bound_input_hash_mismatch:{name}")


def _validate_bound_semantic_report(output_dir: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    meta = manifest.get("semantic_validation_report")
    if meta is None:
        return
    if not isinstance(meta, dict):
        errors.append("semantic_validation_report_metadata_invalid")
        return
    path = output_dir / str(meta.get("path") or "")
    if not path.is_file():
        errors.append("semantic_validation_report_missing")
        return
    if _sha256(path) != meta.get("sha256"):
        errors.append("semantic_validation_report_hash_mismatch")
    if path.stat().st_size != int(meta.get("bytes", -1)):
        errors.append("semantic_validation_report_size_mismatch")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append("semantic_validation_report_invalid_json")
        return
    if report.get("passed") is not True or report.get("require_representation_binding") is not False:
        errors.append("semantic_validation_report_not_prelocal_pass")


def validate_vnext_overlay(
    output_dir: Path,
    *,
    require_representation_binding: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one DATA vNext overlay without trusting builder-side counts."""
    pq = _require_pyarrow()
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    if manifest.get("dataset_version") != DATASET_VERSION:
        errors.append("dataset_version_mismatch")
    if manifest.get("export_schema_version") != "v2":
        errors.append("export_schema_not_v2")
    if tuple(manifest.get("class_order") or ()) != CLASS_NAMES:
        errors.append("class_order_mismatch")
    if manifest.get("graph_schema_version") != "v9":
        errors.append("graph_schema_version_mismatch")
    if manifest.get("historical_artifacts_mutated") is not False:
        errors.append("historical_mutation_flag_not_false")
    generation_commit = str(manifest.get("generation_commit") or "")
    if len(generation_commit) != 40 or any(c not in "0123456789abcdef" for c in generation_commit.lower()):
        errors.append("generation_commit_not_full_sha")

    _validate_input_bindings(manifest, errors)

    expected_artifacts = {
        "label_states",
        "ml_targets",
        "source_registry",
        "crosswalk_registry",
        "evidence_snapshot",
        "representation_requirements",
    }
    artifacts = manifest.get("artifacts") or {}
    if set(artifacts) != expected_artifacts:
        errors.append("artifact_set_mismatch")
    for name in sorted(expected_artifacts & set(artifacts)):
        meta = artifacts[name]
        path = output_dir / str(meta.get("path", ""))
        if not path.is_file():
            errors.append(f"artifact_missing:{name}")
            continue
        if path.name != meta.get("path"):
            errors.append(f"artifact_path_not_local_filename:{name}")
        if _sha256(path) != meta.get("sha256"):
            errors.append(f"artifact_hash_mismatch:{name}")
        if path.stat().st_size != int(meta.get("bytes", -1)):
            errors.append(f"artifact_size_mismatch:{name}")

    _validate_bound_semantic_report(output_dir, manifest, errors)

    label_states_path = output_dir / "label_states.parquet"
    ml_targets_path = output_dir / "ml_targets.parquet"
    semantic_rows = pq.read_table(label_states_path).to_pylist() if label_states_path.is_file() else []
    ml_rows = pq.read_table(ml_targets_path).to_pylist() if ml_targets_path.is_file() else []

    if len(semantic_rows) != EXPECTED_ROWS:
        errors.append(f"semantic_row_count:{len(semantic_rows)}")
    if len(ml_rows) != EXPECTED_CONTRACTS:
        errors.append(f"ml_contract_count:{len(ml_rows)}")

    unique_keys = {(str(r.get("contract_id")), int(r.get("class_index", -1))) for r in semantic_rows}
    if len(unique_keys) != len(semantic_rows):
        errors.append("duplicate_contract_class_key")
    contract_ids = [str(r.get("contract_id")) for r in ml_rows]
    if len(set(contract_ids)) != len(contract_ids):
        errors.append("duplicate_ml_contract")

    by_contract: dict[str, list[dict[str, Any]]] = {}
    for row in semantic_rows:
        by_contract.setdefault(str(row["contract_id"]), []).append(row)
    if len(by_contract) != EXPECTED_CONTRACTS:
        errors.append(f"semantic_contract_count:{len(by_contract)}")
    for cid, rows in by_contract.items():
        ordered = sorted(rows, key=lambda r: int(r["class_index"]))
        if len(ordered) != 10 or tuple(r["class_name"] for r in ordered) != CLASS_NAMES:
            errors.append(f"contract_class_shape:{cid}")
            if len(errors) > 100:
                break

    strong_by_source: Counter[str] = Counter()
    weak_by_source_class: Counter[tuple[str, str]] = Counter()
    target_counts: Counter[str] = Counter()
    strength_counts: Counter[str] = Counter()
    for row in semantic_rows:
        target = row.get("target_value")
        target_counts[str(target)] += 1
        strength = str(row.get("training_strength"))
        strength_counts[strength] += 1
        claims = row.get("source_claims") or []
        claim = claims[0] if claims else {}
        source = str(claim.get("source") or "")
        cls = str(row.get("class_name"))

        if target == 0:
            errors.append(f"target_zero_present:{row['contract_id']}:{cls}")
        if cls in {"GasException", "UnusedReturn"}:
            if target is not None or strength != "NONE" or bool(row.get("loss_eligible")):
                errors.append(f"disabled_class_supervised:{row['contract_id']}:{cls}")
        if strength == "WEAK":
            weak_by_source_class[(source, cls)] += 1
            if source != "dive" or cls != "TransactionOrderDependence" or target != 1:
                errors.append(f"invalid_weak_signal:{row['contract_id']}:{source}:{cls}")
            if bool(row.get("outcome_metric_eligible")):
                errors.append(f"weak_metric_eligible:{row['contract_id']}:{cls}")
            if row.get("outcome_state") not in {"UNKNOWN", "NOT_REVIEWED"}:
                errors.append(f"weak_promoted_to_outcome_truth:{row['contract_id']}:{cls}")
            if claim.get("source_claim_state") != "POSITIVE" or claim.get("mapped_class_name") != cls:
                errors.append(f"weak_claim_provenance_invalid:{row['contract_id']}:{cls}")
        if strength == "STRONG":
            strong_by_source[source] += 1
            if source not in {"solidifi", "smartbugs_curated"} or target != 1:
                errors.append(f"invalid_strong_source:{row['contract_id']}:{source}:{cls}")
            if source == "smartbugs_curated" and cls == "Timestamp":
                errors.append(f"ambiguous_smartbugs_timestamp_strong:{row['contract_id']}")
            if row.get("outcome_state") != "CONFIRMED_POSITIVE":
                errors.append(f"strong_not_confirmed_positive:{row['contract_id']}:{cls}")
            if not row.get("evidence_ids") or not claim.get("evidence_ids"):
                errors.append(f"confirmed_positive_missing_evidence:{row['contract_id']}:{cls}")
            if claim.get("source_claim_state") != "POSITIVE" or claim.get("mapped_class_name") != cls:
                errors.append(f"strong_claim_provenance_invalid:{row['contract_id']}:{cls}")
            if source == "smartbugs_curated" and claim.get("crosswalk_action") != "DIRECT":
                errors.append(f"smartbugs_strong_crosswalk_not_direct:{row['contract_id']}:{cls}")
        if strength == "NONE":
            if target is not None or bool(row.get("loss_eligible")):
                errors.append(f"masked_row_has_target:{row['contract_id']}:{cls}")
        if row.get("outcome_state") in {"UNKNOWN", "NOT_REVIEWED", "CONFLICTING_EVIDENCE", "NOT_APPLICABLE", "INVALID_RECORD"}:
            if bool(row.get("outcome_metric_eligible")):
                errors.append(f"unresolved_metric_eligible:{row['contract_id']}:{cls}")

    if dict(sorted(target_counts.items())) != EXPECTED_TARGET_COUNTS:
        errors.append(f"target_counts_mismatch:{dict(target_counts)}")
    if dict(sorted(strength_counts.items())) != EXPECTED_STRENGTH_COUNTS:
        errors.append(f"strength_counts_mismatch:{dict(strength_counts)}")
    if dict(sorted(strong_by_source.items())) != EXPECTED_STRONG_BY_SOURCE:
        errors.append(f"strong_source_counts_mismatch:{dict(strong_by_source)}")
    if dict(sorted(weak_by_source_class.items())) != EXPECTED_WEAK_BY_SOURCE_CLASS:
        errors.append(f"weak_source_class_counts_mismatch:{dict(weak_by_source_class)}")

    role_counts: Counter[str] = Counter()
    excluded_count = 0
    effective_loss_cells = 0
    outcome_metric_cells = 0
    for row in ml_rows:
        role = str(row.get("role"))
        role_counts[role] += 1
        required = bool(row.get("representation_required"))
        if role == "EXCLUDED":
            excluded_count += 1
            if required:
                errors.append(f"excluded_requires_representation:{row['contract_id']}")
        elif not required:
            errors.append(f"active_role_not_representation_required:{row['contract_id']}:{role}")

        for i, cls in enumerate(CLASS_NAMES):
            target = row.get(f"target_{i}")
            strength = str(row.get(f"strength_{i}"))
            source_eligible = bool(row.get(f"source_loss_eligible_{i}"))
            effective = bool(row.get(f"effective_loss_mask_{i}"))
            metric = bool(row.get(f"outcome_metric_mask_{i}"))
            state = str(row.get(f"outcome_state_{i}"))
            if target == 0:
                errors.append(f"ml_target_zero:{row['contract_id']}:{cls}")
            expected_effective = source_eligible and (
                (strength == "STRONG" and role == "TRAIN_STRONG")
                or (strength == "WEAK" and role == "TRAIN_WEAK")
            )
            if effective != expected_effective:
                errors.append(f"effective_loss_role_mismatch:{row['contract_id']}:{cls}:{role}")
            if effective:
                effective_loss_cells += 1
            if metric:
                outcome_metric_cells += 1
                if role not in {"MODEL_SELECTION", "INTERNAL_AUDIT"}:
                    errors.append(f"metric_role_mismatch:{row['contract_id']}:{cls}:{role}")
                if strength != "STRONG" or state != "CONFIRMED_POSITIVE":
                    errors.append(f"metric_not_strong_positive:{row['contract_id']}:{cls}")
            if cls in {"GasException", "UnusedReturn"} and (target is not None or effective or source_eligible):
                errors.append(f"disabled_ml_cell_active:{row['contract_id']}:{cls}")

    if excluded_count != EXPECTED_EXCLUDED:
        errors.append(f"excluded_count:{excluded_count}")
    if dict(sorted(role_counts.items())) != EXPECTED_ROLE_COUNTS:
        errors.append(f"role_counts_frozen_mismatch:{dict(role_counts)}")
    if dict(sorted(role_counts.items())) != dict(sorted((manifest.get("role_contract_counts") or {}).items())):
        errors.append("role_counts_manifest_mismatch")
    if effective_loss_cells != EXPECTED_EFFECTIVE_LOSS_CELLS:
        errors.append(f"effective_loss_cells_mismatch:{effective_loss_cells}")
    if outcome_metric_cells != EXPECTED_OUTCOME_METRIC_CELLS:
        errors.append(f"outcome_metric_cells_mismatch:{outcome_metric_cells}")
    if manifest.get("unsupported_roles") != {
        "THRESHOLD_FIT": "UNSUPPORTED_EMPTY",
        "CALIBRATION_FIT": "UNSUPPORTED_EMPTY",
        "UNTOUCHED_ACCEPTANCE": "UNSUPPORTED_EMPTY_FROZEN",
    }:
        errors.append("unsupported_roles_changed")

    requirements_path = output_dir / "representation_requirements.json"
    requirements = json.loads(requirements_path.read_text()) if requirements_path.is_file() else {}
    expected_required = EXPECTED_CONTRACTS - EXPECTED_EXCLUDED
    if requirements.get("required_contracts") != expected_required:
        errors.append("representation_required_count_mismatch")
    if requirements.get("physical_binding_status") not in {"PENDING_LOCAL_G7_GATE", "VALIDATED_LOCAL_G7"}:
        errors.append("representation_binding_status_invalid")

    binding_meta = manifest.get("representation_binding_report")
    if require_representation_binding:
        allowed_states = {
            "REPRESENTATIONS_VALIDATED_G7_PENDING_FINAL",
            "VALIDATED_G7_CANDIDATE",
        }
        if manifest.get("status") not in allowed_states:
            errors.append("manifest_not_representation_bound_for_g7")
        if not isinstance(binding_meta, dict):
            errors.append("representation_binding_report_not_bound")
        else:
            binding_path = output_dir / str(binding_meta.get("path", ""))
            if not binding_path.is_file():
                errors.append("representation_binding_report_missing")
            else:
                if _sha256(binding_path) != binding_meta.get("sha256"):
                    errors.append("representation_binding_report_hash_mismatch")
                report = json.loads(binding_path.read_text())
                if report.get("status") != "VALIDATED_LOCAL_G7":
                    errors.append("representation_binding_report_not_validated")
                if report.get("missing_files_total") != 0 or report.get("mismatch_total") != 0:
                    errors.append("representation_binding_report_has_failures")
                if report.get("required_contracts") != expected_required:
                    errors.append("representation_binding_required_count_mismatch")
                if report.get("checked_contracts") != expected_required or report.get("checked_files") != expected_required * 3:
                    errors.append("representation_binding_checked_population_mismatch")
                if report.get("binding_digest_sha256") != binding_meta.get("binding_digest_sha256"):
                    errors.append("representation_binding_digest_mismatch")

    report = {
        "schema": "sentinel-data-vnext-validation-report-v1",
        "passed": not errors,
        "require_representation_binding": require_representation_binding,
        "errors": errors[:250],
        "contracts": len(ml_rows),
        "contract_class_rows": len(semantic_rows),
        "unique_contract_class_keys": len(unique_keys),
        "role_contract_counts": dict(sorted(role_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "training_strength_counts": dict(sorted(strength_counts.items())),
        "strong_rows_by_source": dict(sorted(strong_by_source.items())),
        "weak_rows_by_source_class": {f"{s}:{c}": n for (s, c), n in sorted(weak_by_source_class.items())},
        "effective_loss_cells": effective_loss_cells,
        "outcome_metric_cells": outcome_metric_cells,
        "excluded_contracts": excluded_count,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["validate_vnext_overlay"]
