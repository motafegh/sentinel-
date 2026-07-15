"""Atomic report and hotspot persistence with structured status (Rule 5C)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from .paths import is_valid_job_id, job_report_path

REPORT_PERSISTENCE_TOOL_KEY = "report_persistence"
HOTSPOT_PERSISTENCE_TOOL_KEY = "hotspot_persistence"
PERSISTENCE_TOOL_KEY = REPORT_PERSISTENCE_TOOL_KEY


def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write *content* to *path* atomically via temp + os.replace.

    The temp file is created in the same directory as *path* to guarantee
    same-filesystem atomicity. Raises on any I/O failure — the caller
    must surface a structured status.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception as write_error:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError as cleanup_error:
                raise RuntimeError(
                    f"atomic write failed for {path} and temp cleanup failed for "
                    f"{tmp}: write={write_error}; cleanup={cleanup_error}"
                ) from cleanup_error
        raise


def persist_report(
    state: dict[str, Any],
    report: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Persist a JSON report to ``root / {job_id} / report.json``.

    Returns a tool-status dict suitable for ``state["tool_status"]``.
    Never raises — failures are carried in the returned status (Rule 5C).
    """
    job_id = (state.get("job_id") or "").strip()
    if not job_id or not is_valid_job_id(job_id):
        return {
            REPORT_PERSISTENCE_TOOL_KEY: {
                "ran": False,
                "reason": "missing_or_invalid_job_id",
                "detail": f"job_id={job_id!r} is not a canonical UUID; report not persisted",
            }
        }

    try:
        path = job_report_path(root, job_id, "report.json")
        success_status = {
            "ran": True,
            "reason": "ok",
            "detail": str(path.relative_to(root.resolve())),
        }
        published_report = dict(report)
        published_tool_status = dict(report.get("tool_status", {}) or {})
        published_tool_status[REPORT_PERSISTENCE_TOOL_KEY] = success_status
        published_report["tool_status"] = published_tool_status
        _atomic_write(path, json.dumps(published_report, indent=2))
        logger.debug("persist_report | written → {}", path)
        return {REPORT_PERSISTENCE_TOOL_KEY: success_status}
    except Exception as exc:
        logger.warning("persist_report | failed (non-fatal to graph): {}", exc)
        return {
            REPORT_PERSISTENCE_TOOL_KEY: {
                "ran": False,
                "reason": "write_failure",
                "detail": str(exc),
            }
        }


def persist_hotspot(
    state: dict[str, Any],
    html_str: str,
    root: Path,
) -> dict[str, Any]:
    """Persist a hotspot HTML to ``root / {job_id} / hotspot.html``.

    Returns a tool-status dict (same contract as persist_report).
    """
    job_id = (state.get("job_id") or "").strip()
    if not job_id or not is_valid_job_id(job_id):
        return {
            HOTSPOT_PERSISTENCE_TOOL_KEY: {
                "ran": False,
                "reason": "missing_or_invalid_job_id",
                "detail": f"job_id={job_id!r} is not a canonical UUID; hotspot not persisted",
            }
        }

    try:
        path = job_report_path(root, job_id, "hotspot.html")
        _atomic_write(path, html_str)
        logger.info("persist_hotspot | written → {}", path)
        return {
            HOTSPOT_PERSISTENCE_TOOL_KEY: {
                "ran": True,
                "reason": "ok",
                "detail": str(path.relative_to(root.resolve())),
            }
        }
    except Exception as exc:
        logger.warning("persist_hotspot | failed (non-fatal): {}", exc)
        return {
            HOTSPOT_PERSISTENCE_TOOL_KEY: {
                "ran": False,
                "reason": "write_failure",
                "detail": str(exc),
            }
        }


__all__ = [
    "PERSISTENCE_TOOL_KEY",
    "HOTSPOT_PERSISTENCE_TOOL_KEY",
    "REPORT_PERSISTENCE_TOOL_KEY",
    "persist_hotspot",
    "persist_report",
]
