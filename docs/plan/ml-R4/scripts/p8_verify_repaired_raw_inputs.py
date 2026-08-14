#!/usr/bin/env python3
"""Verify the protected Phase-8 raw manifests before repaired preprocessing.

The script is read-only.  It verifies every manifest-recorded byte length and
SHA-256 for the three active sources and checks that no manifest path escapes
its source root.  It intentionally records no machine-specific path in output.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module" / "data"
ACTIVE_SOURCES = ("dive", "smartbugs_curated", "solidifi")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def verify_source(source: str) -> dict[str, Any]:
    root = DATA_ROOT / "raw" / source
    manifest_path = root / "ingestion_manifest.json"
    if not manifest_path.is_file():
        return {
            "source": source,
            "passed": False,
            "reason": "missing_ingestion_manifest",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "source": source,
            "passed": False,
            "reason": "invalid_ingestion_manifest",
            "detail": str(exc),
        }

    files = list(manifest.get("files") or [])
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_bytes = 0
    for entry in sorted(files, key=lambda item: str(item.get("path") or "")):
        relative = str(entry.get("path") or "")
        if not relative or relative in seen:
            errors.append(
                {
                    "path": relative,
                    "reason": "blank_or_duplicate_manifest_path",
                }
            )
            continue
        seen.add(relative)
        candidate = root / relative
        if not _inside(root, candidate):
            errors.append({"path": relative, "reason": "path_escapes_source_root"})
            continue
        if not candidate.is_file():
            errors.append({"path": relative, "reason": "missing_file"})
            continue
        actual_size = candidate.stat().st_size
        total_bytes += actual_size
        expected_size = int(entry.get("size_bytes", -1))
        if actual_size != expected_size:
            errors.append(
                {
                    "path": relative,
                    "reason": "size_mismatch",
                    "expected": expected_size,
                    "actual": actual_size,
                }
            )
            continue
        actual_sha = _sha256(candidate)
        expected_sha = str(entry.get("sha256") or "")
        if actual_sha != expected_sha:
            errors.append(
                {
                    "path": relative,
                    "reason": "sha256_mismatch",
                    "expected": expected_sha,
                    "actual": actual_sha,
                }
            )

    declared_count = manifest.get("contract_count")
    if declared_count is not None and int(declared_count) != len(files):
        errors.append(
            {
                "reason": "declared_contract_count_mismatch",
                "declared": int(declared_count),
                "manifest_files": len(files),
            }
        )

    return {
        "source": source,
        "passed": not errors,
        "connector": manifest.get("connector"),
        "pin": manifest.get("pin"),
        "resolved_pin": manifest.get("resolved_pin"),
        "manifest_records": len(files),
        "unique_paths": len(seen),
        "total_bytes": total_bytes,
        "errors": errors[:200],
        "errors_total": len(errors),
        "physical_root_recorded": False,
    }


def main() -> int:
    results = [verify_source(source) for source in ACTIVE_SOURCES]
    passed = all(row.get("passed") for row in results)
    print(
        json.dumps(
            {
                "schema": "sentinel-r4-repaired-raw-provenance-gate-v1",
                "passed": passed,
                "sources": results,
                "expected_historical_manifest_records": {
                    "dive": 22330,
                    "smartbugs_curated": 143,
                    "solidifi": 350,
                    "total": 22823,
                },
                "claim_scope": (
                    "byte-level agreement with existing ingestion manifests only; "
                    "this does not prove portable reacquisition"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
