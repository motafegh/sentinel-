"""Evidence record validation, redaction, and R0 closure coverage rules."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from scripts.r0_evidence.matrix import MATRIX_ROWS, MatrixRow

_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(authorization|password|private[_-]?key|secret|token)(\s*[:=]\s*)([^\s,;]+)"
)
_URL_CREDENTIALS = re.compile(r"(?P<scheme>https?://)[^/@\s]+:[^/@\s]+@")
_OUTCOME_STATUSES = frozenset({"pass", "fail", "blocked", "unavailable"})
_REVIEW_STATUSES = frozenset({"pending", "accepted", "rejected"})


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_bundle_digest(manifest: Mapping[str, Any]) -> str:
    """Digest a probe manifest with the single canonical bundle algorithm."""
    return sha256_bytes(canonical_json_bytes(manifest))


def redact_text(text: str, *, workspace: Path | None = None) -> str:
    """Redact credential-shaped values and normalize local absolute paths."""

    redacted = _SECRET_ASSIGNMENT.sub(r"\1\2<REDACTED>", text)
    redacted = _URL_CREDENTIALS.sub(r"\g<scheme><REDACTED>@", redacted)
    if workspace is not None:
        redacted = redacted.replace(str(workspace.resolve()), "<WORKSPACE>")
    redacted = redacted.replace(str(Path.home().resolve()), "<HOME>")
    return redacted


def validate_record(record: Mapping[str, Any]) -> list[str]:
    """Return every structural/semantic error in one evidence record."""

    errors: list[str] = []
    required = {
        "schema_version",
        "kind",
        "record_id",
        "matrix_row_id",
        "phase",
        "baseline_commit",
        "candidate_commit",
        "comparison_key",
        "probe",
        "environment_manifest",
        "execution",
        "outcome",
        "test_references",
        "review",
    }
    missing = sorted(required - set(record))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
        return errors

    if record["schema_version"] not in {"1", "2"} or record["kind"] != "r0_evidence_record":
        errors.append("unsupported evidence schema/kind")
    if not isinstance(record["record_id"], str) or not record["record_id"]:
        errors.append("record_id must be a non-empty string")
    if not isinstance(record["matrix_row_id"], str) or not record["matrix_row_id"]:
        errors.append("matrix_row_id must be a non-empty string")
    if record["phase"] not in {"before", "after"}:
        errors.append("phase must be before or after")
    if not _FULL_GIT_SHA.fullmatch(str(record["baseline_commit"])):
        errors.append("baseline_commit must be a full lowercase Git SHA")
    if record["phase"] == "after" and not _FULL_GIT_SHA.fullmatch(str(record["candidate_commit"])):
        errors.append("after record requires a full candidate_commit")
    if record["phase"] == "before" and record["candidate_commit"] is not None:
        errors.append("before record candidate_commit must be null")
    if not _SHA256.fullmatch(str(record["comparison_key"])):
        errors.append("comparison_key must be a lowercase SHA-256")
    if record["schema_version"] == "2":
        if not _SHA256.fullmatch(str(record.get("probe_bundle_sha256", ""))):
            errors.append("probe_bundle_sha256 must be a lowercase SHA-256")
        if not _FULL_GIT_SHA.fullmatch(str(record.get("probe_bundle_commit", ""))):
            errors.append("probe_bundle_commit must be a full lowercase Git SHA")
    if (
        not isinstance(record["test_references"], list)
        or not record["test_references"]
        or not all(isinstance(item, str) and item for item in record["test_references"])
    ):
        errors.append("test_references must be a non-empty string list")

    probe = record["probe"]
    if not isinstance(probe, Mapping):
        errors.append("probe must be an object")
    else:
        for field in (
            "probe_id",
            "contract_version",
            "argv_template",
            "resolved_argv",
            "cwd",
        ):
            if not probe.get(field):
                errors.append(f"probe.{field} is required")
        for field in ("argv_template", "resolved_argv"):
            value = probe.get(field)
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) and item for item in value)
            ):
                errors.append(f"probe.{field} must be a non-empty string list")

    environment = record["environment_manifest"]
    if not isinstance(environment, Mapping):
        errors.append("environment_manifest must be an object")
    else:
        for field in ("path", "sha256", "environment_contract"):
            if not environment.get(field):
                errors.append(f"environment_manifest.{field} is required")
        if not _SHA256.fullmatch(str(environment.get("sha256", ""))):
            errors.append("environment_manifest.sha256 must be a lowercase SHA-256")
        fingerprint = environment.get("comparison_fingerprint")
        if fingerprint is not None and not _SHA256.fullmatch(str(fingerprint)):
            errors.append("environment_manifest.comparison_fingerprint must be a lowercase SHA-256")

    execution = record["execution"]
    if not isinstance(execution, Mapping):
        errors.append("execution must be an object")
    else:
        for field in (
            "started_at",
            "finished_at",
            "exit_code",
            "stdout_sha256",
            "stderr_sha256",
            "stdout",
            "stderr",
        ):
            if field not in execution:
                errors.append(f"execution.{field} is required")
        if not isinstance(execution.get("exit_code"), int):
            errors.append("execution.exit_code must be an integer")
        for field in ("stdout_sha256", "stderr_sha256"):
            if not _SHA256.fullmatch(str(execution.get(field, ""))):
                errors.append(f"execution.{field} must be a lowercase SHA-256")

    outcome = record["outcome"]
    if not isinstance(outcome, Mapping):
        errors.append("outcome must be an object")
    else:
        if outcome.get("status") not in _OUTCOME_STATUSES:
            errors.append("outcome.status is invalid")
        if not isinstance(outcome.get("invariant_passed"), bool):
            errors.append("outcome.invariant_passed must be boolean")
        assertions = outcome.get("assertions")
        if not isinstance(assertions, list) or not assertions:
            errors.append("outcome.assertions must be a non-empty list")
        else:
            for index, assertion in enumerate(assertions):
                if not isinstance(assertion, Mapping):
                    errors.append(f"outcome.assertions[{index}] must be an object")
                    continue
                if not isinstance(assertion.get("name"), str) or not assertion["name"]:
                    errors.append(f"outcome.assertions[{index}].name is required")
                if not isinstance(assertion.get("passed"), bool):
                    errors.append(f"outcome.assertions[{index}].passed must be boolean")
                if not isinstance(assertion.get("detail"), str):
                    errors.append(f"outcome.assertions[{index}].detail must be a string")
            if outcome.get("invariant_passed") is True and any(
                not a.get("passed") for a in assertions if isinstance(a, Mapping)
            ):
                errors.append(
                    "outcome.invariant_passed=true but some required assertions have passed=false"
                )
        if outcome.get("status") == "pass" and outcome.get("invariant_passed") is not True:
            errors.append("outcome.status=pass requires invariant_passed=true")
        if outcome.get("status") == "fail" and outcome.get("invariant_passed") is not False:
            errors.append("outcome.status=fail requires invariant_passed=false")

    review = record["review"]
    if not isinstance(review, Mapping):
        errors.append("review must be an object")
    else:
        if review.get("status") not in _REVIEW_STATUSES:
            errors.append("review.status is invalid")
        if review.get("status") in {"accepted", "rejected"} and (
            not review.get("reviewer") or not review.get("decided_at")
        ):
            errors.append("decided review requires reviewer and decided_at")
    return errors


_APPROVED_BASELINE = "1256d9aab45add9cf2d23fe33aaa944303259012"


def validate_coverage(
    records: Iterable[Mapping[str, Any]],
    *,
    rows: Iterable[MatrixRow] = MATRIX_ROWS,
    invalid_artifacts: Iterable[Mapping[str, Any]] = (),
    expected_baseline: str | None = _APPROVED_BASELINE,
    expected_candidate: str | None = None,
    expected_probe_bundle_commit: str | None = None,
    expected_probe_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """R0 closure validator — single-tier exact comparison_key matching.

    Requires:
    - before records at approved baseline commit
    - after records at final candidate commit
    - identical probe bundle (same comparison_key means same env + probe)
    - approved reviewer decisions on both records

    No metadata-only fallback. If comparison_keys differ, the row is not comparable.
    """
    by_row: dict[str, list[Mapping[str, Any]]] = {}
    malformed: list[dict[str, Any]] = []
    for record in records:
        errors = validate_record(record)
        if errors:
            malformed.append({"record_id": record.get("record_id"), "errors": errors})
            continue
        by_row.setdefault(str(record["matrix_row_id"]), []).append(record)

    row_reports: list[dict[str, Any]] = []
    for row in rows:
        candidates = by_row.get(row.row_id, [])
        before_recs = [r for r in candidates if r["phase"] == "before"]
        after_recs = [r for r in candidates if r["phase"] == "after"]
        issues: list[str] = []
        if not before_recs:
            issues.append("missing before record")
        if not after_recs:
            issues.append("missing after record")

        comparable_pair = None
        for left in before_recs:
            for right in after_recs:
                if left["comparison_key"] == right["comparison_key"]:
                    comparable_pair = (left, right)
                    break
            if comparable_pair:
                break

        if before_recs and after_recs and comparable_pair is None:
            issues.append("before/after comparison_key mismatch — not comparable")

        if comparable_pair:
            left, right = comparable_pair

            # Baseline check
            if expected_baseline:
                actual = left.get("baseline_commit", "")
                if actual != expected_baseline:
                    issues.append(
                        f"before baseline_commit {actual[:12] if actual else 'MISSING'} "
                        f"!= approved {expected_baseline[:12]}"
                    )

            # Candidate check
            if expected_candidate:
                actual = right.get("candidate_commit", "")
                if actual != expected_candidate:
                    issues.append(
                        f"after candidate_commit {actual[:12] if actual else 'MISSING'} "
                        f"!= expected {expected_candidate[:12]}"
                    )

            # R0-F1: probe bundle check — both before and after must match
            if expected_probe_bundle_sha256:
                for label, record in (("before", left), ("after", right)):
                    actual = record.get("probe_bundle_sha256", "")
                    if actual != expected_probe_bundle_sha256:
                        issues.append(
                            f"{label} probe_bundle_sha256 "
                            f"{actual[:16] if actual else 'MISSING'} "
                            f"!= expected {expected_probe_bundle_sha256[:16]}"
                        )
            if expected_probe_bundle_commit:
                for label, record in (("before", left), ("after", right)):
                    actual = record.get("probe_bundle_commit", "")
                    if actual != expected_probe_bundle_commit:
                        issues.append(
                            f"{label} probe_bundle_commit "
                            f"{actual[:12] if actual else 'MISSING'} "
                            f"!= expected {expected_probe_bundle_commit[:12]}"
                        )
            if left.get("probe_bundle_sha256") != right.get("probe_bundle_sha256"):
                issues.append("before/after probe bundle digest mismatch")
            if left.get("probe_bundle_commit") != right.get("probe_bundle_commit"):
                issues.append("before/after probe bundle commit mismatch")

            if left.get("baseline_commit") == right.get("candidate_commit"):
                issues.append("before and after records must be from different commits")

            # Outcome checks
            if left["outcome"]["invariant_passed"] is not False:
                issues.append("before record does not demonstrate the failing baseline")
            if right["outcome"]["status"] != "pass" or not right["outcome"]["invariant_passed"]:
                issues.append("after record does not prove the invariant")
            if right["execution"]["exit_code"] != 0:
                issues.append("after probe exited nonzero")

            for label, record in (("before", left), ("after", right)):
                review = record["review"]
                if (
                    review["status"] != "accepted"
                    or not review.get("reviewer")
                    or not review.get("decided_at")
                ):
                    issues.append(f"{label} record lacks accepted reviewer decision")

        row_reports.append(
            {
                "row_id": row.row_id,
                "owner_package": row.owner_package,
                "complete": not issues,
                "issues": issues,
            }
        )

    unknown_rows = sorted(set(by_row) - {row.row_id for row in rows})
    invalid_artifact_list = [dict(item) for item in invalid_artifacts]
    complete = (
        not malformed
        and not invalid_artifact_list
        and not unknown_rows
        and all(row["complete"] for row in row_reports)
    )
    return {
        "schema_version": "1",
        "kind": "r0_coverage_report",
        "complete": complete,
        "rows": row_reports,
        "malformed_records": malformed,
        "invalid_artifacts": invalid_artifact_list,
        "unknown_matrix_rows": unknown_rows,
    }


def load_evidence_artifacts(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Load canonical records and surface invalid after-like evidence files.

    Acceptance ledgers and manifests intentionally share the evidence directory, so
    arbitrary JSON cannot be treated as a record. Files whose names claim to be an
    ``after`` artifact are different: silently ignoring one can manufacture a false
    closure narrative even though coverage remains open.
    """

    records: list[dict[str, Any]] = []
    invalid_artifacts: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            invalid_artifacts.append(
                {"path": str(path.relative_to(root)), "error": f"unreadable JSON: {exc}"}
            )
            continue
        if isinstance(payload, dict) and payload.get("kind") == "r0_evidence_record":
            records.append(payload)
        elif "_after_" in path.name:
            invalid_artifacts.append(
                {
                    "path": str(path.relative_to(root)),
                    "error": "after-like artifact is not a canonical r0_evidence_record",
                }
            )
    return records, invalid_artifacts


def load_evidence_records(root: Path) -> list[dict[str, Any]]:
    """Load canonical evidence records; prefer ``load_evidence_artifacts`` for closure."""

    records, _ = load_evidence_artifacts(root)
    return records


__all__ = [
    "canonical_json_bytes",
    "load_evidence_artifacts",
    "load_evidence_records",
    "redact_text",
    "sha256_bytes",
    "sha256_file",
    "validate_coverage",
    "validate_record",
]
