#!/usr/bin/env python3
"""Strict Phase-3 evidence-ledger validation.

This is the fail-closed companion to ``p3_validate_evidence_ledger.py``.
The original validator enforces cross-row and evidence semantics; this module
adds the schema-surface guarantees that are easy to miss in hand-written
semantic checks: allowed properties, field types, enum domains, uniqueness,
manifest artifact references, and evidence-item field contracts.

No external JSON-schema package is required. The implementation mirrors the
versioned R4 v1 schemas in ``docs/plan/ml-R4/schemas`` and then delegates to the
semantic validator. A production G3 decision requires this strict validator to
pass on the materialized ledger.
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
ROW_OPTIONAL = {
    "source_record_id",
    "source_tier",
    "dedup_group_id",
    "project_group_id",
    "leakage_group_id",
    "source_native_label",
    "parser_id",
    "crosswalk_id",
    "phase2_trace_ids",
}
ROW_ALLOWED = ROW_REQUIRED | ROW_OPTIONAL

SOURCE_TIERS = {"T0", "T1", "T2", "T3", "T4", None}
HISTORICAL_STATES = set(semantic.HISTORICAL_STATES)
HISTORICAL_SPLITS = {"train", "val", "test", None}
SOURCE_NATIVE_STATES = {
    "EXPLICIT_POSITIVE",
    "EXPLICIT_NEGATIVE",
    "UNKNOWN",
    "ABSENT",
    "UNSUPPORTED",
    "DROPPED_CATEGORY",
    "MAPPED_NONVULNERABLE",
    "UNAVAILABLE",
    "MIXED",
    "NOT_RECONSTRUCTED",
}
CROSSWALK_ACTIONS = {
    "DIRECT",
    "LOSSY_MAP",
    "DROP",
    "MAP_NONVULNERABLE",
    "UNSUPPORTED",
    "NONE",
    "UNKNOWN",
}
MERGER_ACTIONS = {
    "SINGLE_SOURCE",
    "POSITIVE_PRECEDENCE",
    "ALL_ZERO_SELECTION",
    "CONFLICT",
    "NONE",
    "UNKNOWN",
}
VERIFICATION_ACTIONS = {"GATE_ONLY", "HISTORICAL_DIRECT_PATCH", "NONE", "UNKNOWN"}
ZERO_ORIGINS = {
    "EXPLICIT_SOURCE_ZERO",
    "SOURCE_NATIVE_UNKNOWN",
    "SOURCE_ABSENCE",
    "CLASS_UNSUPPORTED",
    "DROPPED_CATEGORY",
    "MAPPED_NONVULNERABLE",
    "PARSER_DEFAULT",
    "MERGER_PRESERVED_ZERO",
    "SYNTHETIC_NONVULNERABLE",
    "HISTORICAL_POST_EXPORT_SUPPRESSION",
    "HISTORICAL_MISSING",
    "UNRESOLVED_WITHIN_KNOWN_MECHANISMS",
    "NONE",
}
PRIOR_REVIEW_STATES = {
    "NONE",
    "CONFIRMED_POSITIVE",
    "CONFIRMED_NEGATIVE",
    "UNKNOWN",
    "CONFLICTING_EVIDENCE",
    "NOT_REVIEWED",
    "INVALID_RECORD",
}
OUTCOME_STATES = PRIOR_REVIEW_STATES | {"NOT_APPLICABLE"} - {"NONE"}
PROVENANCE_KINDS = {
    "HISTORICAL_RECOVERED",
    "HISTORICAL_CONCLUSION_ONLY",
    "NEW_REPRODUCTION",
    "NEW_GAP_REVIEW",
    "TRANSFORMATION_RECONSTRUCTION",
    "NO_EVIDENCE",
}

EVIDENCE_REQUIRED = set(semantic.EVIDENCE_REQUIRED)
EVIDENCE_OPTIONAL = {"contract_id", "class_index", "source", "producer_version", "code_locations"}
EVIDENCE_ALLOWED = EVIDENCE_REQUIRED | EVIDENCE_OPTIONAL
EVIDENCE_SCOPES = {
    "CONTRACT_CLASS",
    "CONTRACT",
    "SOURCE_CLASS",
    "CORPUS_CLASS",
    "TRANSFORMATION_CATEGORY",
}
EVIDENCE_TYPES = {
    "EXPLOIT_OR_INJECTION_VERIFIED",
    "EXPERT_MANUAL_REVIEW",
    "REPRODUCIBLE_STATIC_REASONING",
    "DYNAMIC_TOOL_SUPPORT",
    "STATIC_TOOL_SUPPORT",
    "SOURCE_ASSERTION",
    "TRANSFORMATION_DEFAULT",
    "NO_EVIDENCE",
}
EVIDENCE_POLARITIES = {"SUPPORTS_POSITIVE", "SUPPORTS_NEGATIVE", "NEUTRAL", "CONFLICTING"}
EVIDENCE_PROVENANCE = {
    "HISTORICAL_RECOVERED",
    "HISTORICAL_CONCLUSION_ONLY",
    "NEW_REPRODUCTION",
    "NEW_GAP_REVIEW",
    "TRANSFORMATION_RECONSTRUCTION",
}

MANIFEST_REQUIRED = {
    "ledger_version",
    "schema_version",
    "source_export_artifact_id",
    "source_export_sha256",
    "expected_contracts",
    "expected_classes",
    "expected_rows",
    "actual_contracts",
    "actual_rows",
    "ledger_parquet",
    "evidence_jsonl",
    "validation_report",
    "class_names",
    "generation_commit",
    "status",
    "limitations",
}
MANIFEST_OPTIONAL = {
    "created_at",
    "source_counts",
    "outcome_state_counts",
    "historical_state_counts",
    "role_counts",
}
MANIFEST_ALLOWED = MANIFEST_REQUIRED | MANIFEST_OPTIONAL
MANIFEST_STATUSES = {"DRAFT", "MATERIALIZED", "VALIDATED", "FAILED"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unique_string_list(value: Any, *, nonempty: bool = False) -> bool:
    if not isinstance(value, list):
        return False
    if any(not isinstance(v, str) or (nonempty and not v) for v in value):
        return False
    return len(value) == len(set(value))


def _check_nullable_string(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _check_additional_properties(obj: dict[str, Any], allowed: set[str], prefix: str, errors: list[str]) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        errors.append(f"{prefix}: additional properties are not allowed: {extra}")


def validate_row_surface(row: dict[str, Any], idx: int, errors: list[str]) -> None:
    ref = f"row[{idx}] {row.get('contract_id', '?')}:{row.get('class_index', '?')}"
    _check_additional_properties(row, ROW_ALLOWED, ref, errors)
    missing = sorted(ROW_REQUIRED - set(row))
    if missing:
        errors.append(f"{ref}: missing required fields {missing}")
        return

    if row.get("ledger_version") != "r4-ledger-v1":
        errors.append(f"{ref}: ledger_version must equal r4-ledger-v1")
    if not isinstance(row.get("contract_id"), str) or not row["contract_id"]:
        errors.append(f"{ref}: contract_id must be a non-empty string")
    ci = row.get("class_index")
    if not _is_int(ci) or not 0 <= ci <= 9:
        errors.append(f"{ref}: class_index must be integer 0..9")
    elif row.get("class_name") != CLASS_NAMES[ci]:
        errors.append(f"{ref}: class_name does not match locked class order")
    if not isinstance(row.get("primary_source"), str) or not row["primary_source"]:
        errors.append(f"{ref}: primary_source must be a non-empty string")

    for field in (
        "source_record_id",
        "dedup_group_id",
        "project_group_id",
        "leakage_group_id",
        "source_native_label",
        "parser_id",
        "crosswalk_id",
    ):
        if field in row and not _check_nullable_string(row.get(field)):
            errors.append(f"{ref}: {field} must be string or null")

    if row.get("source_tier") not in SOURCE_TIERS:
        errors.append(f"{ref}: invalid source_tier {row.get('source_tier')!r}")
    if row.get("historical_state") not in HISTORICAL_STATES:
        errors.append(f"{ref}: invalid historical_state {row.get('historical_state')!r}")
    if row.get("historical_target") not in (0, 1, None) or isinstance(row.get("historical_target"), bool):
        errors.append(f"{ref}: historical_target must be 0, 1, or null")
    if not isinstance(row.get("historical_export_artifact_id"), str) or not row["historical_export_artifact_id"]:
        errors.append(f"{ref}: historical_export_artifact_id must be non-empty string")
    export_sha = row.get("historical_export_sha256")
    if not isinstance(export_sha, str) or not SHA256_RE.fullmatch(export_sha):
        errors.append(f"{ref}: historical_export_sha256 must be 64 hex chars")
    if row.get("historical_split") not in HISTORICAL_SPLITS:
        errors.append(f"{ref}: invalid historical_split {row.get('historical_split')!r}")
    if not isinstance(row.get("representation_available"), bool):
        errors.append(f"{ref}: representation_available must be boolean")

    if row.get("source_native_state") not in SOURCE_NATIVE_STATES:
        errors.append(f"{ref}: invalid source_native_state {row.get('source_native_state')!r}")
    if row.get("crosswalk_action") not in CROSSWALK_ACTIONS:
        errors.append(f"{ref}: invalid crosswalk_action {row.get('crosswalk_action')!r}")
    if row.get("merger_action") not in MERGER_ACTIONS:
        errors.append(f"{ref}: invalid merger_action {row.get('merger_action')!r}")
    if row.get("verification_action") not in VERIFICATION_ACTIONS:
        errors.append(f"{ref}: invalid verification_action {row.get('verification_action')!r}")

    zero_origins = row.get("zero_origin_categories")
    if not _unique_string_list(zero_origins) or any(v not in ZERO_ORIGINS for v in (zero_origins or [])):
        errors.append(f"{ref}: zero_origin_categories must be unique valid enum values")
    if "phase2_trace_ids" in row and not _unique_string_list(row.get("phase2_trace_ids")):
        errors.append(f"{ref}: phase2_trace_ids must be a unique string list")
    if not _unique_string_list(row.get("evidence_ids"), nonempty=True):
        errors.append(f"{ref}: evidence_ids must be a unique non-empty-string list")
    if not _unique_string_list(row.get("independence_groups"), nonempty=True):
        errors.append(f"{ref}: independence_groups must be a unique non-empty-string list")
    if row.get("prior_review_state") not in PRIOR_REVIEW_STATES:
        errors.append(f"{ref}: invalid prior_review_state {row.get('prior_review_state')!r}")
    if row.get("outcome_state") not in OUTCOME_STATES:
        errors.append(f"{ref}: invalid outcome_state {row.get('outcome_state')!r}")
    if not _unique_string_list(row.get("limitations")):
        errors.append(f"{ref}: limitations must be a unique string list")
    if not isinstance(row.get("supervised_loss_masked"), bool):
        errors.append(f"{ref}: supervised_loss_masked must be boolean")
    if not isinstance(row.get("outcome_metrics_masked"), bool):
        errors.append(f"{ref}: outcome_metrics_masked must be boolean")

    roles = row.get("role_eligibility")
    if not _unique_string_list(roles) or any(v not in semantic.ROLE_VALUES for v in (roles or [])):
        errors.append(f"{ref}: role_eligibility must be unique valid enum values")
    if row.get("partition") not in semantic.PARTITIONS:
        errors.append(f"{ref}: invalid partition {row.get('partition')!r}")
    if not _unique_string_list(row.get("artifact_ids"), nonempty=True) or not row.get("artifact_ids"):
        errors.append(f"{ref}: artifact_ids must contain unique non-empty strings")
    if row.get("provenance_kind") not in PROVENANCE_KINDS:
        errors.append(f"{ref}: invalid provenance_kind {row.get('provenance_kind')!r}")


def validate_evidence_surface(item: dict[str, Any], idx: int, errors: list[str]) -> None:
    prefix = f"evidence[{idx}] {item.get('evidence_id', '?')}"
    _check_additional_properties(item, EVIDENCE_ALLOWED, prefix, errors)
    missing = sorted(EVIDENCE_REQUIRED - set(item))
    if missing:
        errors.append(f"{prefix}: missing required fields {missing}")
        return

    if not isinstance(item.get("evidence_id"), str) or not item["evidence_id"]:
        errors.append(f"{prefix}: evidence_id must be non-empty string")
    if item.get("scope_type") not in EVIDENCE_SCOPES:
        errors.append(f"{prefix}: invalid scope_type {item.get('scope_type')!r}")
    if "contract_id" in item and not _check_nullable_string(item.get("contract_id")):
        errors.append(f"{prefix}: contract_id must be string or null")
    ci = item.get("class_index")
    if ci is not None and (not _is_int(ci) or not 0 <= ci <= 9):
        errors.append(f"{prefix}: class_index must be integer 0..9 or null")
    if "source" in item and not _check_nullable_string(item.get("source")):
        errors.append(f"{prefix}: source must be string or null")
    if item.get("evidence_type") not in EVIDENCE_TYPES:
        errors.append(f"{prefix}: invalid evidence_type {item.get('evidence_type')!r}")
    if not isinstance(item.get("producer"), str) or not item["producer"]:
        errors.append(f"{prefix}: producer must be non-empty string")
    if "producer_version" in item and not _check_nullable_string(item.get("producer_version")):
        errors.append(f"{prefix}: producer_version must be string or null")
    if not isinstance(item.get("independence_group"), str) or not item["independence_group"]:
        errors.append(f"{prefix}: independence_group must be non-empty string")
    if item.get("polarity") not in EVIDENCE_POLARITIES:
        errors.append(f"{prefix}: invalid polarity {item.get('polarity')!r}")
    if not isinstance(item.get("finding_summary"), str) or not item["finding_summary"]:
        errors.append(f"{prefix}: finding_summary must be non-empty string")
    if "code_locations" in item and not _unique_string_list(item.get("code_locations")):
        errors.append(f"{prefix}: code_locations must be a unique string list")
    if not isinstance(item.get("artifact_id"), str) or not item["artifact_id"]:
        errors.append(f"{prefix}: artifact_id must be non-empty string")
    artifact_sha = item.get("artifact_sha256")
    if artifact_sha is not None and (not isinstance(artifact_sha, str) or not SHA256_RE.fullmatch(artifact_sha)):
        errors.append(f"{prefix}: artifact_sha256 must be 64 hex chars or null")
    if item.get("historical_or_new") not in EVIDENCE_PROVENANCE:
        errors.append(f"{prefix}: invalid historical_or_new {item.get('historical_or_new')!r}")
    if not isinstance(item.get("raw_evidence_available"), bool):
        errors.append(f"{prefix}: raw_evidence_available must be boolean")
    if not isinstance(item.get("tool_only"), bool):
        errors.append(f"{prefix}: tool_only must be boolean")
    if not _unique_string_list(item.get("limitations")):
        errors.append(f"{prefix}: limitations must be a unique string list")


def _validate_artifact_ref(value: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix}: artifact reference must be an object")
        return
    extra = sorted(set(value) - {"path", "sha256"})
    if extra:
        errors.append(f"{prefix}: additional artifact-ref properties not allowed: {extra}")
    if set(value) != {"path", "sha256"}:
        errors.append(f"{prefix}: artifact reference requires exactly path + sha256")
    path = value.get("path")
    sha = value.get("sha256")
    if not isinstance(path, str) or not path:
        errors.append(f"{prefix}: path must be non-empty string")
    if sha is not None and (not isinstance(sha, str) or not SHA256_RE.fullmatch(sha)):
        errors.append(f"{prefix}: sha256 must be 64 hex chars or null")


def _validate_count_map(value: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix}: must be an object")
        return
    for key, count in value.items():
        if not isinstance(key, str) or not _is_int(count) or count < 0:
            errors.append(f"{prefix}: counts must be non-negative integers keyed by strings")
            break


def validate_manifest_surface(manifest: dict[str, Any], errors: list[str]) -> None:
    prefix = "manifest"
    _check_additional_properties(manifest, MANIFEST_ALLOWED, prefix, errors)
    missing = sorted(MANIFEST_REQUIRED - set(manifest))
    if missing:
        errors.append(f"manifest: missing required fields {missing}")
        return

    if manifest.get("ledger_version") != "r4-ledger-v1":
        errors.append("manifest: ledger_version must equal r4-ledger-v1")
    if manifest.get("schema_version") != "1":
        errors.append("manifest: schema_version must equal '1'")
    if "created_at" in manifest and manifest.get("created_at") is not None and not isinstance(manifest.get("created_at"), str):
        errors.append("manifest: created_at must be string or null")
    if not isinstance(manifest.get("source_export_artifact_id"), str) or not manifest["source_export_artifact_id"]:
        errors.append("manifest: source_export_artifact_id must be non-empty string")
    sha = manifest.get("source_export_sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        errors.append("manifest: source_export_sha256 must be 64 hex chars")

    for field in ("expected_contracts", "expected_rows"):
        value = manifest.get(field)
        if not _is_int(value) or value < 1:
            errors.append(f"manifest: {field} must be positive integer")
    if manifest.get("expected_classes") != 10:
        errors.append("manifest: expected_classes must equal 10")
    for field in ("actual_contracts", "actual_rows"):
        value = manifest.get(field)
        if not _is_int(value) or value < 0:
            errors.append(f"manifest: {field} must be non-negative integer")

    for field in ("ledger_parquet", "evidence_jsonl", "validation_report"):
        _validate_artifact_ref(manifest.get(field), f"manifest.{field}", errors)
    if manifest.get("class_names") != CLASS_NAMES:
        errors.append("manifest: class_names must match locked canonical order")
    generation_commit = manifest.get("generation_commit")
    if not isinstance(generation_commit, str) or len(generation_commit) < 7:
        errors.append("manifest: generation_commit must be string length >= 7")
    if manifest.get("status") not in MANIFEST_STATUSES:
        errors.append(f"manifest: invalid status {manifest.get('status')!r}")
    if not isinstance(manifest.get("limitations"), list) or any(not isinstance(v, str) for v in manifest.get("limitations", [])):
        errors.append("manifest: limitations must be a string list")
    for field in ("source_counts", "outcome_state_counts", "historical_state_counts", "role_counts"):
        if field in manifest:
            _validate_count_map(manifest.get(field), f"manifest.{field}", errors)

    status = manifest.get("status")
    if status in {"MATERIALIZED", "VALIDATED"}:
        if not _is_int(manifest.get("actual_contracts")) or manifest["actual_contracts"] < 1:
            errors.append(f"manifest: {status} requires actual_contracts >= 1")
        if not _is_int(manifest.get("actual_rows")) or manifest["actual_rows"] < 10:
            errors.append(f"manifest: {status} requires actual_rows >= 10")


def validate_strict(
    rows: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    *,
    allow_partial_population: bool = False,
) -> dict[str, Any]:
    surface_errors: list[str] = []
    for idx, row in enumerate(rows):
        validate_row_surface(row, idx, surface_errors)
    for idx, item in enumerate(evidence_items):
        validate_evidence_surface(item, idx, surface_errors)
    if manifest is not None:
        validate_manifest_surface(manifest, surface_errors)

    semantic_report = semantic.validate_ledger(
        rows,
        evidence_items,
        manifest,
        allow_partial_population=allow_partial_population,
    )
    semantic_errors = list(semantic_report.get("errors") or [])
    all_errors = surface_errors + semantic_errors
    return {
        **semantic_report,
        "schema": "r4-ledger-strict-validation-report-v1",
        "passed": not all_errors,
        "surface_errors": surface_errors,
        "semantic_errors": semantic_errors,
        "errors": all_errors,
    }


def _parse_args() -> argparse.Namespace:
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
            rows,
            evidence_items,
            manifest,
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
