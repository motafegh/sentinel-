# agents/src/rag/build_index/_io.py
"""
Atomic-write + durability helpers for the RAG index build.

All artifact writes go through temp-file + atomic-replace so a crash
mid-write never leaves a half-written file. Each artifact is fsync'd
before replace and the parent directory is fsync'd after, so the rename
is durable on POSIX. A pre-write snapshot of existing artifacts is kept
and restored if any write fails.

The shape is shared by build_index (full rebuild) and ingestion/pipeline
(incremental update) so both obey the same durability contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import faiss
from loguru import logger

from ._paths import _REQUIRED_ARTIFACTS, INDEX_DIR


# ── Atomic write helpers ──────────────────────────────────────────────────────

def _tmp_path(path: Path, build_id: str) -> Path:
    """
    Return a temp path in the same directory as the final artifact.

    Same-directory temp files are required because Path.replace() is atomic
    only within the same filesystem.
    """
    return path.with_name(f".{path.name}.{build_id}.tmp")


def _fsync_directory(directory: Path) -> None:
    """
    Best-effort fsync for the parent directory after atomic replace.

    On POSIX filesystems this makes the rename durable. On platforms where
    directory fsync is unsupported, this silently degrades to normal replace.
    """
    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        return

    fd: int | None = None
    try:
        fd = os.open(str(directory), os.O_DIRECTORY)
        os.fsync(fd)
    except OSError:
        # Not all filesystems allow directory fsync. Atomic replace still happened.
        return
    finally:
        if fd is not None:
            os.close(fd)


def _atomic_write_json(path: Path, payload: dict[str, Any], build_id: str) -> None:
    """Write JSON atomically using temp-file + replace."""
    tmp = _tmp_path(path, build_id)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        tmp.replace(path)
        _fsync_directory(path.parent)

    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_pickle(path: Path, payload: Any, build_id: str) -> None:
    """Write a pickle artifact atomically using temp-file + replace."""
    tmp = _tmp_path(path, build_id)
    try:
        with tmp.open("wb") as f:
            pickle.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())

        tmp.replace(path)
        _fsync_directory(path.parent)

    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_faiss(path: Path, index: faiss.Index, build_id: str) -> None:
    """
    Write a FAISS index atomically.

    FAISS writes to a filesystem path, so we write to a same-directory temp
    file first, then atomically replace the final artifact.
    """
    tmp = _tmp_path(path, build_id)
    try:
        faiss.write_index(index, str(tmp))
        tmp.replace(path)
        _fsync_directory(path.parent)

    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Backup / rollback helpers ────────────────────────────────────────────────

def _snapshot_existing_artifacts(build_id: str) -> dict[Path, Path | None]:
    """
    Copy existing artifacts before replacing them.

    This protects against partial replacement if an exception occurs while
    writing the new artifact set. It is not a substitute for atomic writes;
    it is a rollback safety net.
    """
    backup_dir = INDEX_DIR / "backups" / build_id
    snapshot: dict[Path, Path | None] = {}

    for final_path in _REQUIRED_ARTIFACTS:
        if final_path.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / final_path.name
            shutil.copy2(final_path, backup_path)
            snapshot[final_path] = backup_path
        else:
            snapshot[final_path] = None

    if backup_dir.exists():
        logger.info("Existing index artifacts backed up to {}", backup_dir)

    return snapshot


def _restore_snapshot(snapshot: dict[Path, Path | None]) -> None:
    """
    Restore artifacts from a pre-write snapshot.

    If an artifact did not exist before the failed write, remove any newly
    created artifact at that path.
    """
    logger.warning("Restoring previous index artifacts from rollback snapshot")

    for final_path, backup_path in snapshot.items():
        try:
            if backup_path is not None and backup_path.exists():
                shutil.copy2(backup_path, final_path)
            elif final_path.exists():
                final_path.unlink()
        except Exception as exc:
            logger.error("Rollback failed for {}: {}", final_path, exc)


# ── Checksums ─────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    """Return SHA256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()