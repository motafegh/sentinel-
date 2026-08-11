#!/usr/bin/env python3
"""Strict schema-surface + semantic validation for the R4 Phase-3 ledger.

``p3_validate_evidence_ledger.py`` owns cross-row/evidence semantics. This
companion fails closed on the rest of the versioned v1 schema surface:
additional properties, primitive types, enum domains, uniqueness constraints,
and manifest artifact references. No external jsonschema dependency is needed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import p3_validate_evidence_ledger as semantic

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
CLASS_NAMES = semantic.CLASS_NAMES

ROW_REQUIRED = set(semantic.ROW_REQUIRED)
ROW_ALLOWED = ROW_REQUIRED | {
    "source_record_id", "source_tier", "dedup_group_id", "project_group_id",
    "leakage_group_id", "source_native_label", "parser_id", "crosswalk_id",
    "phase2_trace_ids",
}
ROW_ENUMS = {
    "source_tier": {"T0", "T1", "T2", "T3", "T4", None},
    "historical_state": set(semantic.HISTORICAL_STATES),
    "historical_split": {"train", "val", "test", None},
    "source_native_state": {
        "EXPLICIT_POSITIVE", "EXPLICIT_NEGATIVE", "UNKNOWN", "ABSENT",
        "UNSUPPORTED", "DROPPED_CATEGORY", "MAPPED_NONVULNERABLE",
        "UNAVAILABLE", "MIXED", "NOT_RECONSTRUCTED",
    },
    "crosswalk_action": {
        "DIRECT", "LOSSY_MAP", "DROP", "MAP_NONVULNERABLE", "UNSUPPORTED",
        "NONE", "UNKNOWN",
    },
    "merger_action": {
        "SINGLE_SOURCE", "POSITIVE_PRECEDENCE", "ALL_ZERO_SELECTION",
        "CONFLICT", "NONE", "UNKNOWN",
    },
    "verification_action": {"GATE_ONLY", "HISTORICAL_DIRECT_PATCH", "NONE", "UNKNOWN"},
    "prior_review_state": {
        "NONE", "CONFIRMED_POSITIVE", "CONFIRMED_NEGATIVE", "UNKNOWN",
        "CONFLICTING_EVIDENCE", "NOT_REVIEWED", "INVALID_RECORD",
    },
    "outcome_state": {
        "CONFIRMED_POSITIVE", "CONFIRMED_NEGATIVE", "UNKNOWN",
        "NOT_APPLICABLE", "CONFLICTING_EVIDENCE", "NOT_REVIEWED",
        "INVALID_RECORD",
    },
    "partition": set(semantic.PARTITIONS),
    "provenance_kind": {
        "HISTORICAL_RECOVERED", "HISTORICAL_CONCLUSION_ONLY",
        "NEW_REPRODUCTION", "NEW_GAP_REVIEW", "TRANSFORMATION_RECONSTRUCTION",
        "NO_EVIDENCE",
    },
}
ZERO_ORIGINS = {
    "EXPLICIT_SOURCE_ZERO", "SOURCE_NATIVE_UNKNOWN", "SOURCE_ABSENCE",
    "CLASS_UNSUPPORTED", "DROPPED_CATEGORY", "MAPPED_NONVULNERABLE",
    "PARSER_DEFAULT", "MERGER_PRESERVED_ZERO", "SYNTHETIC_NONVULNERABLE",
    "HISTORICAL_POST_EXPORT_SUPPRESSION", "HISTORICAL_MISSING",
    "UNRESOLVED_WITHIN_KNOWN_MECHANISMS", "NONE",
}

EVIDENCE_REQUIRED = set(semantic.EVIDENCE_REQUIRED)
EVIDENCE_ALLOWED = EVIDENCE_REQUIRED | {
    "contract_id", "class_index", "source", "producer_version", "code_locations",
}
EVIDENCE_ENUMS = {
    "scope_type": {
        "CONTRACT_CLASS", "CONTRACT", "SOURCE_CLASS", "CORPUS_CLASS",
        "TRANSFORMATION_CATEGORY",
    },
    "evidence_type": {
        "EXPLOIT_OR_INJECTION_VERIFIED", "EXPERT_MANUAL_REVIEW",
        "REPRODUCIBLE_STATIC_REASONING", "DYNAMIC_TOOL_SUPPORT",
        "STATIC_TOOL_SUPPORT", "SOURCE_ASSERTION", "TRANSFORMATION_DEFAULT",
        "NO_EVIDENCE",
    },
    "polarity": {"SUPPORTS_POSITIVE", "SUPPORTS_NEGATIVE", "NEUTRAL", "CONFLICTING"},
    "historical_or_new": {
        "HISTORICAL_RECOVERED", "HISTORICAL_CONCLUSION_ONLY",
        "NEW_REPRODUCTION", "NEW_GAP_REVIEW", "TRANSFORMATION_RECONSTRUCTION",
    },
}

MANIFEST_REQUIRED = {
    "ledger_version", "schema_version", "source_export_artifact_id",
    "source_export_sha256", "expected_contracts", "expected_classes",
    "expected_rows", "actual_contracts", "actual_rows", "ledger_parquet",
    "evidence_jsonl", "validation_report", "class_names", "generation_commit",
    "status", "limitations",
}
MANIFEST_ALLOWED = MANIFEST_REQUIRED | {
    "created_at", "source_counts", "outcome_state_counts",
    "historical_state_counts", "role_counts",
}
MANIFEST_STATUSES = {"DRAFT", "MATERIALIZED", "VALIDATED", "FAILED"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unique_strings(value: Any, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(v, str) and (bool(v) or not nonempty) for v in value)
        and len(value) == len(set(value))
    )


def _extra(obj: dict[str, Any], allowed: set[str], prefix: str, errors: list[str]) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        errors.append(f"{prefix}: additional properties not allowed: {extra}")


def _nullable_string(value: Any) -> bool:
    return value is None or isinstance(value, str)


def validate_row_surface(row: dict[str, Any], idx: int, errors: list[str]) -> None:
    ref = f"row[{idx}] {row.get('contract_id', '?')}:{row.get('class_index', '?')}"
    _extra(row, ROW_ALLOWED, ref, errors)
    missing = sorted(ROW_REQUIRED - set(row))
    if missing:
        errors.append(f"{ref}: missing required fields {missing}")
        return

    if row.get("ledger_version") != "r4-ledger-v1":
        errors.append(f"{ref}: invalid ledger_version")
    if not isinstance(row.get("contract_id"), str) or not row["contract_id"]:
        errors.append(f"{ref}: contract_id must be non-empty string")
    ci = row.get("class_index")
    if not _is_int(ci) or not 0 <= ci <= 9:
        errors.append(f"{ref}: class_index must be integer 0..9")
    elif row.get("class_name") != CLASS_NAMES[ci]:
        errors.append(f"{ref}: class_name does not match locked class order")
    if not isinstance(row.get("primary_source"), str) or not row["primary_source"]:
        errors.append(f"{ref}: primary_source must be non-empty string")

    for field, allowed in ROW_ENUMS.items():
        if row.get(field) not in allowed:
            errors.append(f"{ref}: invalid {field} {row.get(field)!r}")

    for field in (
        "source_record_id", "dedup_group_id", "project_group_id",
        "leakage_group_id", "source_native_label", "parser_id", "crosswalk_id",
    ):
        if field in row and not _nullable_string(row.get(field)):
            errors.append(f"{ref}: {field} must be string or null")

    target = row.get("historical_target")
    if target not in (0, 1, None) or isinstance(target, bool):
        errors.append(f"{ref}: historical_target must be 0, 1, or null")
    sha = row.get("historical_export_sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        errors.append(f"{ref}: historical_export_sha256 must be 64 hex chars")
    if not isinstance(row.get("historical_export_artifact_id"), str) or not row["historical_export_artifact_id"]:
        errors.append(f"{ref}: historical_export_artifact_id must be non-empty string")
    if not isinstance(row.get("representation_available"), bool):
        errors.append(f"{ref}: representation_available must be boolean")
    if not isinstance(row.get("supervised_loss_masked"), bool):
        errors.append(f"{ref}: supervised_loss_masked must be boolean")
    if not isinstance(row.get("outcome_metrics_masked"), bool):
        errors.append(f"{ref}: outcome_metrics_masked must be boolean")

    zero = row.get("zero_origin_categories")
    if not _unique_strings(zero) or any(v not in ZERO_ORIGINS for v in (zero or [])):
        errors.append(f"{ref}: zero_origin_categories must contain unique valid values")
    if "phase2_trace_ids" in row and not _unique_strings(row.get("phase2_trace_ids")):
        errors.append(f"{ref}: phase2_trace_ids must be a unique string list")
    for field in ("evidence_ids", "independence_groups", "artifact_ids"):
        value = row.get(field)
        if not _unique_strings(value, nonempty=True) or (field == "artifact_ids" and not value):
            errors.append(f"{ref}: {field} must contain unique non-empty strings")
    if not _unique_strings(row.get("limitations")):
        errors.append(f"{ref}: limitations must be a unique string list")
    roles = row.get("role_eligibility")
    if not _unique_strings(roles) or any(v not in semantic.ROLE_VALUES for v in (roles or [])):
        errors.append(f"{ref}: role_eligibility must contain unique valid values")


def validate_evidence_surface(item: dict[str, Any], idx: int, errors: list[str]) -> None:
    ref = f"evidence[{idx}] {item.get('evidence_id', '?')}"
    _extra(item, EVIDENCE_ALLOWED, ref, errors)
    missing = sorted(EVIDENCE_REQUIRED - set(item))
    if missing:
        errors.append(f"{ref}: missing required fields {missing}")
        return
    if not isinstance(item.get("evidence_id"), str) or not item["evidence_id"]:
        errors.append(f"{ref}: evidence_id must be non-empty string")
    for field, allowed in EVIDENCE_ENUMS.items():
        if item.get(field) not in allowed:
            errors.append(f"{ref}: invalid {field} {item.get(field)!r}")
    ci = item.get("class_index")
    if ci is not None and (not _is_int(ci) or not 0 <= ci <= 9):
        errors.append(f"{ref}: class_index must be integer 0..9 or null")
    for field in ("contract_id", "source", "producer_version"):
        if field in item and not _nullable_string(item.get(field)):
            errors.append(f"{ref}: {field} must be string or null")
    for field in ("producer", "independence_group", "finding_summary", "artifact_id"):
        if not isinstance(item.get(field), str) or not item[field]:
            errors.append(f"{ref}: {field} must be non-empty string")
    if "code_locations" in item and not _unique_strings(item.get("code_locations")):
        errors.append(f"{ref}: code_locations must be a unique string list")
    sha = item.get("artifact_sha256")
    if sha is not None and (not isinstance(sha, str) or not SHA256_RE.fullmatch(sha)):
        errors.append(f"{ref}: artifact_sha256 must be 64 hex chars or null")
    for field in ("raw_evidence_available", "tool_only"):
        if not isinstance(item.get(field), bool):
            errors.append(f"{ref}: {field} must be boolean")
    if not _unique_strings(item.get("limitations")):
        errors.append(f"{ref}: limitations must be a unique string list")


def _artifact_ref(value: Any, ref: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{ref}: must be object with path + sha256")
        return
    if set(value) != {"path", "sha256"}:
        errors.append(f"{ref}: requires exactly path + sha256")
    if not isinstance(value.get("path"), str) or not value.get("path"):
        errors.append(f"{ref}: path must be non-empty string")
    sha = value.get("sha256")
    if sha is not None and (not isinstance(sha, str) or not SHA256_RE.fullmatch(sha)):
        errors.append(f"{ref}: sha256 must be 64 hex chars or null")


def validate_manifest_surface(manifest: dict[str, Any], errors: list[str]) -> None:
    _extra(manifest, MANIFEST_ALLOWED, "manifest", errors)
    missing = sorted(MANIFEST_REQUIRED - set(manifest))
    if missing:
        errors.append(f"manifest: missing required fields {missing}")
        return
    if manifest.get("ledger_version") != "r4-ledger-v1" or manifest.get("schema_version") != "1":
        errors.append("manifest: invalid ledger/schema version")
    if "created_at" in manifest and manifest.get("created_at") is not None and not isinstance(manifest.get("created_at"), str):
        errors.append("manifest: created_at must be string or null")
    if not isinstance(manifest.get("source_export_artifact_id"), str) or not manifest["source_export_artifact_id"]:
        errors.append("manifest: source_export_artifact_id must be non-empty string")
    sha = manifest.get("source_export_sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        errors.append("manifest: source_export_sha256 must be 64 hex chars")
    for field in ("expected_contracts", "expected_rows"):
        if not _is_int(manifest.get(field)) or manifest[field] < 1:
            errors.append(f"manifest: {field} must be positive integer")
    if manifest.get("expected_classes") != 10:
        errors.append("manifest: expected_classes must equal 10")
    for field in ("actual_contracts", "actual_rows"):
        if not _is_int(manifest.get(field)) or manifest[field] < 0:
            errors.append(f"manifest: {field} must be non-negative integer")
    for field in ("ledger_parquet", "evidence_jsonl", "validation_report"):
        _artifact_ref(manifest.get(field), f"manifest.{field}", errors)
    if manifest.get("class_names") != CLASS_NAMES:
        errors.append("manifest: class_names must match locked order")
    if not isinstance(manifest.get("generation_commit"), str) or len(manifest["generation_commit"]) < 7:
        errors.append("manifest: generation_commit must have length >= 7")
    if manifest.get("status") not in MANIFEST_STATUSES:
        errors.append(f"manifest: invalid status {manifest.get('status')!r}")
    if not isinstance(manifest.get("limitations"), list) or any(not isinstance(v, str) for v in manifest.get("limitations", [])):
        errors.append("manifest: limitations must be string list")
    for field in ("source_counts", "outcome_state_counts", "historical_state_counts", "role_counts"):
        if field in manifest:
            value = manifest[field]
            if not isinstance(value, dict) or any(
                not isinstance(k, str) or not _is_int(v) or v < 0 for k, v in value.items()
            ):
                errors.append(f"manifest: {field} must map strings to non-negative integers")
    if manifest.get("status") in {"MATERIALIZED", "VALIDATED"}:
        if manifest.get("actual_contracts", 0) < 1 or manifest.get("actual_rows", 0) < 10:
            errors.append(f"manifest: {manifest.get('status')} requires materialized population")


def validate_strict(rows, evidence_items, manifest, *, allow_partial_population=False):
    surface_errors: list[str] = []
    for idx, row in enumerate(rows):
        validate_row_surface(row, idx, surface_errors)
    for idx, item in enumerate(evidence_items):
        validate_evidence_surface(item, idx, surface_errors)
    if manifest is not None:
        validate_manifest_surface(manifest, surface_errors)
    semantic_report = semantic.validate_ledger(
        rows, evidence_items, manifest, allow_partial_population=allow_partial_population
    )
    semantic_errors = list(semantic_report.get("errors") or [])
    errors = surface_errors + semantic_errors
    return {
        **semantic_report,
        "schema": "r4-ledger-strict-validation-report-v1",
        "passed": not errors,
        "surface_errors": surface_errors,
        "semantic_errors": semantic_errors,
        "errors": errors,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-partial-population", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        rows = semantic.load_ledger(args.ledger)
        evidence_items = semantic._read_jsonl(args.evidence)
        manifest = semantic.load_json(args.manifest)
        report = validate_strict(
            rows, evidence_items, manifest,
            allow_partial_population=args.allow_partial_population,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"STRICT VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
