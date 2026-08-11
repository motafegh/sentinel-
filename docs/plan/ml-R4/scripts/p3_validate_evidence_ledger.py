#!/usr/bin/env python3
"""Validate SENTINEL R4 evidence-ledger artifacts.

The validator intentionally uses the Python standard library for JSON/JSONL.
Parquet input is supported only when pyarrow is already available; the script
never silently substitutes another population if Parquet cannot be read.

Usage:
    python docs/plan/ml-R4/scripts/p3_validate_evidence_ledger.py \
        --ledger path/to/ledger.jsonl \
        --evidence path/to/evidence.jsonl \
        --manifest path/to/manifest.json \
        --report path/to/report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

CLASS_NAMES = [
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
]

MASKED_OUTCOME_STATES = {
    "UNKNOWN",
    "NOT_APPLICABLE",
    "CONFLICTING_EVIDENCE",
    "NOT_REVIEWED",
    "INVALID_RECORD",
}
CONFIRMED_OUTCOME_STATES = {"CONFIRMED_POSITIVE", "CONFIRMED_NEGATIVE"}
TOOL_EVIDENCE_TYPES = {"DYNAMIC_TOOL_SUPPORT", "STATIC_TOOL_SUPPORT"}
HISTORICAL_STATES = {
    "HISTORICAL_POSITIVE": 1,
    "HISTORICAL_ZERO": 0,
    "HISTORICAL_MISSING": None,
}
ROLE_VALUES = {
    "TRAIN_STRONG",
    "TRAIN_WEAK",
    "TRAIN_UNLABELED",
    "MODEL_SELECTION",
    "THRESHOLD_FIT",
    "CALIBRATION_FIT",
    "INTERNAL_AUDIT",
    "UNTOUCHED_ACCEPTANCE",
    "CASE_STUDY",
    "EXCLUDE_OUTCOME_METRICS",
}
PARTITIONS = {
    "UNASSIGNED",
    "TRAIN",
    "MODEL_SELECTION",
    "THRESHOLD_FIT",
    "CALIBRATION_FIT",
    "INTERNAL_AUDIT",
    "UNTOUCHED_ACCEPTANCE",
    "CASE_STUDY",
    "EXCLUDED",
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

ROW_REQUIRED = {
    "ledger_version",
    "contract_id",
    "class_index",
    "class_name",
    "primary_source",
    "historical_state",
    "historical_target",
    "historical_export_artifact_id",
    "historical_export_sha256",
    "historical_split",
    "representation_available",
    "source_native_state",
    "crosswalk_action",
    "merger_action",
    "verification_action",
    "zero_origin_categories",
    "evidence_ids",
    "independence_groups",
    "prior_review_state",
    "outcome_state",
    "limitations",
    "supervised_loss_masked",
    "outcome_metrics_masked",
    "role_eligibility",
    "partition",
    "artifact_ids",
    "provenance_kind",
}

EVIDENCE_REQUIRED = {
    "evidence_id",
    "scope_type",
    "evidence_type",
    "producer",
    "independence_group",
    "polarity",
    "finding_summary",
    "artifact_id",
    "artifact_sha256",
    "historical_or_new",
    "raw_evidence_available",
    "tool_only",
    "limitations",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            rows.append(value)
    return rows


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Parquet validation requires pyarrow in the execution environment. "
            "Install/use the existing ML/data environment that already carries "
            "pyarrow; do not substitute a different ledger population."
        ) from exc
    return pq.read_table(path).to_pylist()


def load_ledger(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".parquet":
        return _read_parquet(path)
    raise ValueError(f"Unsupported ledger format: {path}; expected .jsonl or .parquet")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _missing_keys(row: dict[str, Any], required: Iterable[str]) -> list[str]:
    return sorted(set(required) - set(row))


def _row_ref(row: dict[str, Any]) -> str:
    return f"{row.get('contract_id', '?')}:{row.get('class_index', '?')}"


def validate_evidence_items(
    items: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(items):
        prefix = f"evidence[{idx}]"
        missing = _missing_keys(item, EVIDENCE_REQUIRED)
        if missing:
            errors.append(f"{prefix}: missing required fields {missing}")
            continue

        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            errors.append(f"{prefix}: evidence_id must be non-empty string")
            continue
        if evidence_id in by_id:
            errors.append(f"duplicate evidence_id: {evidence_id}")
            continue
        by_id[evidence_id] = item

        scope = item.get("scope_type")
        if scope == "CONTRACT_CLASS":
            if not item.get("contract_id") or not isinstance(item.get("class_index"), int):
                errors.append(f"{evidence_id}: CONTRACT_CLASS scope requires contract_id + class_index")
        elif scope == "CONTRACT":
            if not item.get("contract_id"):
                errors.append(f"{evidence_id}: CONTRACT scope requires contract_id")
        elif scope in {"SOURCE_CLASS", "CORPUS_CLASS"}:
            if not item.get("source") or not isinstance(item.get("class_index"), int):
                errors.append(f"{evidence_id}: {scope} scope requires source + class_index")
        elif scope != "TRANSFORMATION_CATEGORY":
            errors.append(f"{evidence_id}: invalid scope_type {scope!r}")

        class_index = item.get("class_index")
        if class_index is not None and (not isinstance(class_index, int) or not 0 <= class_index <= 9):
            errors.append(f"{evidence_id}: class_index must be null or 0..9")

        evidence_type = item.get("evidence_type")
        tool_only = item.get("tool_only")
        if evidence_type in TOOL_EVIDENCE_TYPES and tool_only is not True:
            errors.append(f"{evidence_id}: tool evidence must set tool_only=true")

        historical_or_new = item.get("historical_or_new")
        raw_available = item.get("raw_evidence_available")
        artifact_sha = item.get("artifact_sha256")
        if historical_or_new == "HISTORICAL_CONCLUSION_ONLY" and raw_available is True:
            errors.append(
                f"{evidence_id}: HISTORICAL_CONCLUSION_ONLY cannot claim raw_evidence_available=true"
            )
        if raw_available is True and (not isinstance(artifact_sha, str) or not SHA256_RE.fullmatch(artifact_sha)):
            errors.append(f"{evidence_id}: retained raw evidence requires artifact_sha256")
        if artifact_sha is not None and (
            not isinstance(artifact_sha, str) or not SHA256_RE.fullmatch(artifact_sha)
        ):
            errors.append(f"{evidence_id}: invalid artifact_sha256")

        if not item.get("independence_group"):
            errors.append(f"{evidence_id}: independence_group is required")
        if not item.get("producer"):
            errors.append(f"{evidence_id}: producer is required")

    if not items:
        warnings.append("evidence item set is empty")
    return by_id


def validate_ledger(
    rows: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
    *,
    allow_partial_population: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    evidence_by_id = validate_evidence_items(evidence_items, errors, warnings)

    keys_seen: set[tuple[str, int]] = set()
    contract_classes: dict[str, set[int]] = defaultdict(set)
    contract_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    leakage_partitions: dict[str, set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    historical_counts: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        ref = _row_ref(row)
        missing = _missing_keys(row, ROW_REQUIRED)
        if missing:
            errors.append(f"row[{idx}] {ref}: missing required fields {missing}")
            continue

        contract_id = row.get("contract_id")
        class_index = row.get("class_index")
        class_name = row.get("class_name")

        if not isinstance(contract_id, str) or not contract_id:
            errors.append(f"row[{idx}] {ref}: contract_id must be non-empty string")
            continue
        if not isinstance(class_index, int) or not 0 <= class_index <= 9:
            errors.append(f"row[{idx}] {ref}: class_index must be 0..9")
            continue
        if class_name != CLASS_NAMES[class_index]:
            errors.append(
                f"row[{idx}] {ref}: class_name={class_name!r} does not match "
                f"locked class {CLASS_NAMES[class_index]!r}"
            )

        key = (contract_id, class_index)
        if key in keys_seen:
            errors.append(f"duplicate canonical ledger key: {contract_id}:{class_index}")
        keys_seen.add(key)
        contract_classes[contract_id].add(class_index)
        contract_rows[contract_id].append(row)

        historical_state = row.get("historical_state")
        historical_target = row.get("historical_target")
        if historical_state not in HISTORICAL_STATES:
            errors.append(f"{ref}: invalid historical_state {historical_state!r}")
        elif historical_target != HISTORICAL_STATES[historical_state]:
            errors.append(
                f"{ref}: {historical_state} requires historical_target="
                f"{HISTORICAL_STATES[historical_state]!r}, got {historical_target!r}"
            )

        export_id = row.get("historical_export_artifact_id")
        export_sha = row.get("historical_export_sha256")
        if not isinstance(export_id, str) or not export_id:
            errors.append(f"{ref}: missing historical_export_artifact_id")
        if not isinstance(export_sha, str) or not SHA256_RE.fullmatch(export_sha):
            errors.append(f"{ref}: historical_export_sha256 must be 64 hex chars")

        if row.get("ledger_version") != "r4-ledger-v1":
            errors.append(f"{ref}: unsupported ledger_version {row.get('ledger_version')!r}")

        outcome = row.get("outcome_state")
        evidence_ids = row.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            errors.append(f"{ref}: evidence_ids must be a list")
            evidence_ids = []
        if outcome in CONFIRMED_OUTCOME_STATES and not evidence_ids:
            errors.append(f"{ref}: {outcome} requires at least one evidence reference")

        if outcome in MASKED_OUTCOME_STATES:
            if row.get("supervised_loss_masked") is not True:
                errors.append(f"{ref}: {outcome} must be masked from supervised loss")
            if row.get("outcome_metrics_masked") is not True:
                errors.append(f"{ref}: {outcome} must be masked from outcome metrics")

        roles = row.get("role_eligibility")
        if not isinstance(roles, list) or any(role not in ROLE_VALUES for role in roles):
            errors.append(f"{ref}: role_eligibility contains invalid value(s)")
            roles = []
        if "EXCLUDE_OUTCOME_METRICS" in roles and row.get("outcome_metrics_masked") is not True:
            errors.append(f"{ref}: EXCLUDE_OUTCOME_METRICS requires outcome_metrics_masked=true")

        partition = row.get("partition")
        if partition not in PARTITIONS:
            errors.append(f"{ref}: invalid partition {partition!r}")

        zero_origins = row.get("zero_origin_categories")
        if not isinstance(zero_origins, list):
            errors.append(f"{ref}: zero_origin_categories must be a list")
            zero_origins = []
        if historical_state == "HISTORICAL_ZERO" and not zero_origins:
            errors.append(f"{ref}: historical zero requires at least one zero-origin category")
        if historical_state == "HISTORICAL_POSITIVE":
            non_none = [z for z in zero_origins if z != "NONE"]
            if non_none:
                errors.append(f"{ref}: historical positive cannot carry zero origins {non_none}")

        artifact_ids = row.get("artifact_ids")
        if not isinstance(artifact_ids, list) or not artifact_ids:
            errors.append(f"{ref}: artifact_ids must contain at least one artifact identity")

        row_independence = set(row.get("independence_groups") or [])
        resolved_evidence: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                errors.append(f"{ref}: unresolved evidence_id {evidence_id}")
                continue
            resolved_evidence.append(evidence)
            group = evidence.get("independence_group")
            if group and group not in row_independence:
                errors.append(
                    f"{ref}: evidence {evidence_id} independence_group={group!r} "
                    "missing from row.independence_groups"
                )

            scope = evidence.get("scope_type")
            if scope == "CONTRACT_CLASS" and (
                evidence.get("contract_id") != contract_id
                or evidence.get("class_index") != class_index
            ):
                errors.append(f"{ref}: evidence {evidence_id} CONTRACT_CLASS scope mismatch")
            elif scope == "CONTRACT" and evidence.get("contract_id") != contract_id:
                errors.append(f"{ref}: evidence {evidence_id} CONTRACT scope mismatch")
            elif scope in {"SOURCE_CLASS", "CORPUS_CLASS"}:
                if evidence.get("class_index") != class_index:
                    errors.append(f"{ref}: evidence {evidence_id} class scope mismatch")
                if evidence.get("source") not in {None, row.get("primary_source")}:
                    errors.append(f"{ref}: evidence {evidence_id} source scope mismatch")

        if "UNTOUCHED_ACCEPTANCE" in roles:
            if not resolved_evidence:
                errors.append(f"{ref}: UNTOUCHED_ACCEPTANCE requires evidence")
            elif all(bool(ev.get("tool_only")) for ev in resolved_evidence):
                errors.append(
                    f"{ref}: UNTOUCHED_ACCEPTANCE cannot be supported only by tool evidence"
                )

        leakage_group = row.get("leakage_group_id")
        if leakage_group and partition not in {"UNASSIGNED", "EXCLUDED"}:
            leakage_partitions[str(leakage_group)].add(str(partition))

        source_counts[str(row.get("primary_source"))] += 1
        outcome_counts[str(outcome)] += 1
        historical_counts[str(historical_state)] += 1

    expected_class_set = set(range(10))
    for contract_id, indexes in contract_classes.items():
        if indexes != expected_class_set:
            missing = sorted(expected_class_set - indexes)
            extra = sorted(indexes - expected_class_set)
            errors.append(
                f"contract {contract_id}: class coverage must be exactly 0..9; "
                f"missing={missing}, extra={extra}"
            )

    for leakage_group, partitions in leakage_partitions.items():
        if len(partitions) > 1:
            errors.append(
                f"leakage_group {leakage_group}: incompatible partitions {sorted(partitions)}"
            )

    actual_contracts = len(contract_classes)
    actual_rows = len(rows)

    if manifest is not None:
        expected_contracts = manifest.get("expected_contracts")
        expected_classes = manifest.get("expected_classes")
        expected_rows = manifest.get("expected_rows")
        manifest_actual_contracts = manifest.get("actual_contracts")
        manifest_actual_rows = manifest.get("actual_rows")

        if expected_classes != 10:
            errors.append(f"manifest: expected_classes must be 10, got {expected_classes!r}")
        if isinstance(expected_contracts, int) and expected_rows != expected_contracts * 10:
            errors.append(
                f"manifest: expected_rows must equal expected_contracts*10; "
                f"got {expected_rows!r} vs {expected_contracts * 10}"
            )
        if manifest_actual_contracts != actual_contracts:
            errors.append(
                f"manifest: actual_contracts={manifest_actual_contracts!r} "
                f"but ledger has {actual_contracts}"
            )
        if manifest_actual_rows != actual_rows:
            errors.append(
                f"manifest: actual_rows={manifest_actual_rows!r} but ledger has {actual_rows}"
            )

        if not allow_partial_population:
            if expected_contracts != actual_contracts:
                errors.append(
                    f"population incomplete: expected {expected_contracts} contracts, "
                    f"loaded {actual_contracts}"
                )
            if expected_rows != actual_rows:
                errors.append(
                    f"population incomplete: expected {expected_rows} rows, loaded {actual_rows}"
                )

        manifest_classes = manifest.get("class_names")
        if manifest_classes != CLASS_NAMES:
            errors.append("manifest: class_names do not match locked canonical order")

        manifest_export_sha = manifest.get("source_export_sha256")
        for row in rows:
            if row.get("historical_export_sha256") != manifest_export_sha:
                errors.append(
                    f"{_row_ref(row)}: row export hash does not match manifest source_export_sha256"
                )
                break

    report = {
        "schema": "r4-ledger-validation-report-v1",
        "passed": not errors,
        "actual_contracts": actual_contracts,
        "actual_rows": actual_rows,
        "unique_keys": len(keys_seen),
        "evidence_items": len(evidence_items),
        "source_row_counts": dict(sorted(source_counts.items())),
        "historical_state_counts": dict(sorted(historical_counts.items())),
        "outcome_state_counts": dict(sorted(outcome_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-partial-population",
        action="store_true",
        help="Validate schema/semantics without requiring manifest expected population counts.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        rows = load_ledger(args.ledger)
        evidence_items = _read_jsonl(args.evidence)
        manifest = load_json(args.manifest) if args.manifest else None
        report = validate_ledger(
            rows,
            evidence_items,
            manifest,
            allow_partial_population=args.allow_partial_population,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
