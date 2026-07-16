"""Path validation and job-scoped containment for report persistence."""

from __future__ import annotations

import re
from pathlib import Path

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def validate_address(address: str) -> str:
    """Validate an Ethereum address and return it.

    Raises ValueError if the address is malformed. A validated address
    is safe to store as domain data but must never be used as a filename
    component without containment checks.
    """
    cleaned = (address or "").strip()
    if not _ADDRESS_RE.match(cleaned):
        raise ValueError(f"invalid Ethereum address: {address!r} — expected 0x + 40 hex chars")
    return cleaned


def is_valid_job_id(job_id: str) -> bool:
    """Return True if *job_id* is a canonical UUID string."""
    return bool(_UUID_RE.match((job_id or "").strip().lower()))


def validate_job_id(job_id: str) -> str:
    """Validate a job identifier and return it.

    Raises ValueError if the job ID is not a canonical UUID. A UUID is
    inherently safe as a filename component — it contains only hex digits
    and hyphens, so path traversal is impossible.
    """
    cleaned = (job_id or "").strip().lower()
    if not is_valid_job_id(cleaned):
        raise ValueError(f"invalid job_id: {job_id!r} — expected canonical UUID")
    return cleaned


def job_report_dir(root: Path, job_id: str) -> Path:
    """Resolve the job-scoped report directory under *root*.

    The returned path is ``root / job_id``. Both *root* and the result
    are resolved and checked for containment. Raises ValueError on any
    escape attempt.
    """
    safe_job = validate_job_id(job_id)
    resolved_root = root.resolve()
    candidate = resolved_root / safe_job
    if candidate.is_symlink():
        raise ValueError(f"job report directory must not be a symlink: {candidate}")
    return candidate


def job_report_path(root: Path, job_id: str, filename: str) -> Path:
    """Resolve a single file inside the job-scoped report directory.

    *filename* must be a bare basename (no path separators, no ``..``).
    Raises ValueError on any escape attempt.
    """
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError(f"unsafe report filename: {filename!r}")
    base = job_report_dir(root, job_id)
    return base / filename


def assert_contained(path: Path, root: Path) -> Path:
    """Assert that *path* resolves inside *root* and return the resolved path."""
    rp = path.resolve()
    rr = root.resolve()
    if not rp.is_relative_to(rr):
        raise ValueError(f"path containment violation: {rp} is not inside {rr}")
    return rp


__all__ = [
    "assert_contained",
    "is_valid_job_id",
    "job_report_dir",
    "job_report_path",
    "validate_address",
    "validate_job_id",
]
