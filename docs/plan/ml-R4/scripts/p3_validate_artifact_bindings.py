#!/usr/bin/env python3
"""Validate that a Phase-3 manifest is cryptographically bound to its files.

The schema/semantic validators check the contents of the ledger. This validator
checks the publication boundary: every manifest artifact reference must resolve
inside the repository, exist, and hash to the SHA-256 recorded in the manifest.
A VALIDATED manifest must also point at a validation report whose own
``passed`` field is true.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_FIELDS = ("ledger_parquet", "evidence_jsonl", "validation_report")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(root: Path, raw_path: Any, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"manifest.{field}.path must be a non-empty repository-relative path")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError(f"manifest.{field}.path must not be absolute: {raw_path}")
    root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest.{field}.path escapes repository root: {raw_path}") from exc
    return resolved


def validate_bindings(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    errors: list[str] = []
    checked: dict[str, dict[str, Any]] = {}

    status = manifest.get("status")
    if status not in {"MATERIALIZED", "VALIDATED", "FAILED", "DRAFT"}:
        errors.append(f"manifest.status is invalid: {status!r}")

    for field in ARTIFACT_FIELDS:
        ref = manifest.get(field)
        if not isinstance(ref, dict):
            errors.append(f"manifest.{field} must be an object")
            continue
        try:
            path = _resolve_repo_path(root, ref.get("path"), field)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        expected_sha = ref.get("sha256")
        if status == "VALIDATED" and (
            not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha)
        ):
            errors.append(f"manifest.{field}.sha256 must be populated for VALIDATED state")
            continue
        if expected_sha is not None and (
            not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha)
        ):
            errors.append(f"manifest.{field}.sha256 must be 64 lowercase hex chars or null")
            continue

        if not path.is_file():
            errors.append(f"manifest.{field} file is missing: {path}")
            continue
        actual_sha = sha256_file(path)
        if expected_sha is not None and actual_sha != expected_sha:
            errors.append(
                f"manifest.{field} SHA-256 mismatch: {actual_sha} != {expected_sha}"
            )
        checked[field] = {
            "path": str(path.relative_to(root.resolve())),
            "sha256": actual_sha,
            "bytes": path.stat().st_size,
        }

    if status == "VALIDATED" and "validation_report" in checked:
        report_path = root.resolve() / checked["validation_report"]["path"]
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"validation report is unreadable JSON: {exc}")
        else:
            if report.get("passed") is not True:
                errors.append("VALIDATED manifest points to a report that did not pass")

    generation_commit = manifest.get("generation_commit")
    if status == "VALIDATED" and (
        not isinstance(generation_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", generation_commit)
    ):
        errors.append("VALIDATED manifest generation_commit must be an exact 40-hex Git commit")

    return {
        "schema": "r4-ledger-artifact-binding-report-v1",
        "passed": not errors,
        "status": status,
        "checked": checked,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        report = validate_bindings(manifest, args.root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ARTIFACT BINDING ERROR: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
