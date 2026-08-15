"""Byte-exact raw-manifest verification for repaired R4 DATA builds.

The audited local layout intentionally uses ``repo`` symlinks for DIVE and
SmartBugs.  Security checks therefore distinguish lexical path traversal from
an allowed source-root symlink target: ``..`` and absolute manifest paths are
rejected, while the source's explicit ``repo`` entry may resolve to its pinned
local staging tree.  Every accepted file is still bound by size and SHA-256
from its ingestion manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_candidate(
    root: Path,
    relative: str,
    allowed_resolved_roots: Iterable[Path],
) -> tuple[Path | None, str | None]:
    """Resolve one manifest path without confusing safe symlinks with traversal."""

    logical = PurePosixPath(relative)
    if logical.is_absolute() or not logical.parts or ".." in logical.parts:
        return None, "invalid_or_traversing_manifest_path"
    candidate = root.joinpath(*logical.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None, "missing_or_unresolvable_file"
    allowed = [Path(item).resolve() for item in allowed_resolved_roots]
    if not any(resolved.is_relative_to(base) for base in allowed):
        return None, "path_escapes_allowed_repository_roots"
    if not resolved.is_file():
        return None, "not_a_regular_file"
    return resolved, None


def verify_manifest_source(
    source: str,
    root: Path,
    manifest_path: Path,
    *,
    allowed_resolved_roots: Iterable[Path] | None = None,
    max_reported_errors: int = 200,
) -> dict[str, Any]:
    """Verify every manifest record against the current physical bytes."""

    root = Path(root)
    manifest_path = Path(manifest_path)
    if allowed_resolved_roots is None:
        allowed_resolved_roots = (root.resolve(), (root / "repo").resolve())
    if not manifest_path.is_file():
        return {
            "source": source,
            "passed": False,
            "reason": "missing_ingestion_manifest",
            "errors_total": 1,
            "errors": [{"reason": "missing_ingestion_manifest"}],
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "source": source,
            "passed": False,
            "reason": "invalid_ingestion_manifest",
            "detail": str(exc),
            "errors_total": 1,
            "errors": [{"reason": "invalid_ingestion_manifest", "detail": str(exc)}],
        }

    files = manifest.get("files")
    if not isinstance(files, list):
        return {
            "source": source,
            "passed": False,
            "reason": "manifest_files_not_a_list",
            "errors_total": 1,
            "errors": [{"reason": "manifest_files_not_a_list"}],
        }

    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_bytes = 0
    for raw_entry in sorted(files, key=lambda item: str((item or {}).get("path") or "")):
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        relative = str(entry.get("path") or "")
        if not relative or relative in seen:
            errors.append({"path": relative, "reason": "blank_or_duplicate_manifest_path"})
            continue
        seen.add(relative)
        candidate, path_error = _safe_candidate(
            root,
            relative,
            allowed_resolved_roots,
        )
        if path_error:
            errors.append({"path": relative, "reason": path_error})
            continue
        assert candidate is not None
        actual_size = candidate.stat().st_size
        total_bytes += actual_size
        try:
            expected_size = int(entry.get("size_bytes", -1))
        except (TypeError, ValueError):
            expected_size = -1
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
    if declared_count is not None:
        try:
            declared = int(declared_count)
        except (TypeError, ValueError):
            declared = -1
        if declared != len(files):
            errors.append(
                {
                    "reason": "declared_contract_count_mismatch",
                    "declared": declared,
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
        "manifest_sha256": _sha256(manifest_path),
        "unique_paths": len(seen),
        "total_bytes": total_bytes,
        "errors": errors[:max_reported_errors],
        "errors_total": len(errors),
        "physical_root_recorded": False,
        "path_policy": "lexical_no_traversal; resolved_target_within_source_or_explicit_repo_root",
    }


def require_manifest_source(
    source: str,
    root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Return a passing report or fail before repaired output is created."""

    report = verify_manifest_source(source, root, manifest_path)
    if not report.get("passed"):
        preview = report.get("errors") or []
        raise ValueError(
            f"raw manifest verification failed for {source}: "
            f"errors_total={report.get('errors_total')} preview={preview[:3]}"
        )
    return report


__all__ = ["require_manifest_source", "verify_manifest_source"]
