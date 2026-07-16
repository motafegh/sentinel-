"""R0.5: Authenticated closed-inventory release descriptor.

A ``release_descriptor.json`` file written alongside ``manifest.json`` that
independently authenticates the full file set — including manifest.json itself.
This breaks the circular-hash problem (Fix A): the manifest is excluded from
``artifact_hash`` (because it contains that hash), but the release descriptor's
``manifest_hash`` field locks the manifest content, so tampering either file
is detectable.

Output layout (same dir as manifest.json)::

    <export_dir>/
      release_descriptor.json   ← written by write_release_descriptor()
      manifest.json
      ...
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

RELEASE_DESCRIPTOR_FILENAME = "release_descriptor.json"

_FILES_EXCLUDED_FROM_DESCRIPTOR = frozenset({RELEASE_DESCRIPTOR_FILENAME})


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_release_descriptor(
    export_dir: Path,
    manifest_hash: str,
    artifact_hash: str,
    data_files: list[Path] | None = None,
) -> dict[str, Any]:
    """Write ``release_descriptor.json`` into *export_dir*.

    Args:
        export_dir: Root of the export artifact (must contain manifest.json).
        manifest_hash: SHA-256 hex digest of ``manifest.json`` content.
        artifact_hash: Same value stored in ``manifest["artifact_hash"]``.
        data_files: Optional explicit list of data files.  If omitted, the
            entire *export_dir* is scanned (excluding the descriptor itself).

    Returns:
        The full descriptor dict (also written to disk).
    """
    if data_files is None:
        data_files = sorted(
            p for p in export_dir.rglob("*")
            if p.is_file()
               and p.name != RELEASE_DESCRIPTOR_FILENAME
        )

    per_file_hashes: dict[str, str] = {}
    for p in data_files:
        rel = str(p.relative_to(export_dir))
        per_file_hashes[rel] = _sha256_file(p)

    descriptor = {
        "descriptor_version": "1",
        "manifest_hash": manifest_hash,
        "artifact_hash": artifact_hash,
        "hash_algorithm": "sha256",
        "files": per_file_hashes,
    }

    release_id_body = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    descriptor["release_id"] = hashlib.sha256(release_id_body.encode()).hexdigest()

    dest = export_dir / RELEASE_DESCRIPTOR_FILENAME
    dest.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n"
    )

    return descriptor


_VERIFY_OK = "ok"
_VERIFY_MISSING = "file_missing"
_VERIFY_EXTRA = "file_extra"
_VERIFY_HASH_MISMATCH = "hash_mismatch"
_VERIFY_NO_DESCRIPTOR = "no_descriptor"
_VERIFY_RELEASE_ID = "release_id_mismatch"


def verify_release(export_dir: Path) -> dict[str, Any]:
    """Verify on-disk file set against ``release_descriptor.json``.

    Returns a structured dict::

        {
            "verified": bool,
            "reason": str,          # "ok" | one of the _VERIFY_* constants
            "files_checked": int,
            "mismatches": [{"path": str, "expected": str, "actual": str}],
            "missing": list[str],
            "extra": list[str],
        }
    """
    descriptor_path = export_dir / RELEASE_DESCRIPTOR_FILENAME
    if not descriptor_path.exists():
        return {
            "verified": False,
            "reason": _VERIFY_NO_DESCRIPTOR,
            "files_checked": 0,
            "mismatches": [],
            "missing": [],
            "extra": [],
        }

    descriptor: dict = json.loads(descriptor_path.read_text())
    expected_release_id = descriptor.get("release_id", "")
    computed_release_id_body = json.dumps(
        {k: v for k, v in descriptor.items() if k != "release_id"},
        sort_keys=True,
        separators=(",", ":"),
    )
    computed_release_id = hashlib.sha256(computed_release_id_body.encode()).hexdigest()

    if computed_release_id != expected_release_id:
        return {
            "verified": False,
            "reason": _VERIFY_RELEASE_ID,
            "files_checked": 0,
            "mismatches": [{
                "path": RELEASE_DESCRIPTOR_FILENAME,
                "expected": expected_release_id,
                "actual": computed_release_id,
            }],
            "missing": [],
            "extra": [],
        }

    expected_files: dict[str, str] = descriptor.get("files", {})
    on_disk_files: set[str] = set()
    mismatches: list[dict[str, str]] = []

    for p in export_dir.rglob("*"):
        if p.is_file() and p.name != RELEASE_DESCRIPTOR_FILENAME:
            rel = str(p.relative_to(export_dir))
            on_disk_files.add(rel)
            if rel in expected_files:
                actual_hex = _sha256_file(p)
                if actual_hex != expected_files[rel]:
                    mismatches.append({
                        "path": rel,
                        "expected": expected_files[rel],
                        "actual": actual_hex,
                    })

    expected_set = set(expected_files)
    missing = sorted(expected_set - on_disk_files)
    extra = sorted(on_disk_files - expected_set)

    verified = (
        not mismatches
        and not missing
        and not extra
    )

    if not verified:
        reason = (
            _VERIFY_HASH_MISMATCH if mismatches
            else _VERIFY_MISSING if missing
            else _VERIFY_EXTRA
        )
    else:
        reason = _VERIFY_OK

    return {
        "verified": verified,
        "reason": reason,
        "files_checked": len(on_disk_files),
        "mismatches": mismatches,
        "missing": missing,
        "extra": extra,
    }


__all__ = [
    "RELEASE_DESCRIPTOR_FILENAME",
    "write_release_descriptor",
    "verify_release",
]
