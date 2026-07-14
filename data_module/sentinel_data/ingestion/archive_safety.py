"""Safe ZIP extraction with path containment and attack rejection.

Replaces the broken ``str.startswith`` containment check in
``manual_connector._extract_zip`` (D2-DATA-001 / R0-ARCHIVE-CONTAINMENT).

Defenses:
    - ``Path.is_relative_to`` containment (not string prefix)
    - Symlink rejection via mode bits
    - Absolute path rejection (POSIX and Windows drive letters)
    - Special file rejection (devices, FIFOs, sockets)
    - NUL and control character rejection in member names
    - Per-file size, member count, total size, compression ratio limits
    - Atomic extraction: temp workspace → atomic promote
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


class ArchiveSafetyError(Exception):
    """Base exception for all archive safety violations."""


class ArchiveTraversalError(ArchiveSafetyError):
    """A ZIP member attempts to escape the extraction root."""


class ArchiveSymlinkError(ArchiveSafetyError):
    """A ZIP member is a symlink that could escape the extraction root."""


class ArchiveLimitError(ArchiveSafetyError):
    """A ZIP member or archive exceeds a configured safety limit."""


class ArchiveBadNameError(ArchiveSafetyError):
    """A ZIP member has an invalid or dangerous name."""


@dataclass(frozen=True)
class ArchiveLimits:
    """Safety limits for ZIP extraction.

    L2 measured values based on the largest real archive in the project
    (smartbugs-results-master_2.zip: 1.1M members, 7.8 GB uncompressed,
    max ratio 160.5x, max depth 6). Each limit has >=1.25x headroom above
    the observed maximum. Approved by Ali 2026-07-14.
    """

    max_members: int = 2_000_000
    max_total_uncompressed_bytes: int = 16 * 1024 * 1024 * 1024  # 16 GiB
    max_per_file_bytes: int = 512 * 1024 * 1024  # 512 MiB
    max_compression_ratio: float = 200.0
    max_path_depth: int = 32


DEFAULT_LIMITS = ArchiveLimits()


def _check_member_name(name: str) -> None:
    """Reject names with NUL, control characters, or Windows drive letters."""
    if not name:
        raise ArchiveBadNameError("empty member name")
    if "\x00" in name:
        raise ArchiveBadNameError(f"NUL byte in member name: {name!r}")
    if any(ord(c) < 0x20 for c in name):
        raise ArchiveBadNameError(f"control character in member name: {name!r}")
    if len(name) == 1 and name[0] == "/":
        raise ArchiveBadNameError("root directory member")
    if len(name) >= 2 and name[1] == ":":
        raise ArchiveBadNameError(f"Windows drive letter in member name: {name!r}")


def _is_symlink_mode(external_attr: int) -> bool:
    """Check if the ZIP external_attr indicates a symlink."""
    unix_mode = external_attr >> 16
    return stat.S_ISLNK(unix_mode)


def _is_special_mode(external_attr: int) -> bool:
    """Check if the ZIP external_attr indicates a device, FIFO, or socket."""
    unix_mode = external_attr >> 16
    return stat.S_ISCHR(unix_mode) or stat.S_ISBLK(unix_mode) or stat.S_ISFIFO(unix_mode) or stat.S_ISSOCK(unix_mode)


def _check_path_depth(name: str, max_depth: int) -> None:
    """Reject paths exceeding the configured depth limit."""
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if len(parts) > max_depth:
        raise ArchiveLimitError(
            f"path depth {len(parts)} exceeds limit {max_depth}: {name!r}"
        )


def extract_zip_safe(
    zip_path: Path,
    dest: Path,
    *,
    source_name: str = "archive",
    limits: ArchiveLimits | None = None,
) -> Path:
    """Extract *zip_path* into *dest* safely.

    Returns the destination Path on success. Raises ArchiveSafetyError
    (or a subclass) on any violation. The extraction is atomic: a
    temporary workspace is used, and only promoted to *dest* on full
    success.
    """
    if limits is None:
        limits = DEFAULT_LIMITS

    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    tmp_root = dest.parent / f".{dest.name}__extraction_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    member_count = 0

    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()

            if len(infos) > limits.max_members:
                raise ArchiveLimitError(
                    f"[{source_name}] archive has {len(infos)} members, "
                    f"limit is {limits.max_members}"
                )

            for info in infos:
                name = info.filename
                member_count += 1

                if "__MACOSX/" in name or name.endswith(".DS_Store"):
                    continue

                _check_member_name(name)

                if _is_symlink_mode(info.external_attr):
                    raise ArchiveSymlinkError(
                        f"[{source_name}] symlink member rejected: {name!r}"
                    )

                if _is_special_mode(info.external_attr):
                    raise ArchiveSafetyError(
                        f"[{source_name}] special file member rejected: {name!r}"
                    )

                _check_path_depth(name, limits.max_path_depth)

                if name.endswith("/"):
                    target_dir = (tmp_root / name).resolve()
                    if not target_dir.is_relative_to(tmp_root):
                        raise ArchiveTraversalError(
                            f"[{source_name}] directory traversal: {name!r}"
                        )
                    target_dir.mkdir(parents=True, exist_ok=True)
                    continue

                file_size = info.file_size
                if file_size > limits.max_per_file_bytes:
                    raise ArchiveLimitError(
                        f"[{source_name}] member {name!r} size {file_size} "
                        f"exceeds per-file limit {limits.max_per_file_bytes}"
                    )

                if info.compress_size > 0:
                    ratio = file_size / info.compress_size
                    if ratio > limits.max_compression_ratio:
                        raise ArchiveLimitError(
                            f"[{source_name}] member {name!r} compression ratio "
                            f"{ratio:.1f} exceeds limit {limits.max_compression_ratio}"
                        )

                total_bytes += file_size
                if total_bytes > limits.max_total_uncompressed_bytes:
                    raise ArchiveLimitError(
                        f"[{source_name}] total uncompressed size exceeds limit "
                        f"{limits.max_total_uncompressed_bytes}"
                    )

                target = (tmp_root / name).resolve()
                if not target.is_relative_to(tmp_root):
                    raise ArchiveTraversalError(
                        f"[{source_name}] path traversal: {name!r} -> {target}"
                    )

                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    except zipfile.BadZipFile as e:
        raise ArchiveSafetyError(
            f"[{source_name}] not a valid zip: {zip_path} ({e})"
        ) from e
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        raise

    if dest.exists() and any(dest.iterdir()):
        shutil.rmtree(dest)
    tmp_root.replace(dest)
    return dest


__all__ = [
    "ArchiveLimitError",
    "ArchiveLimits",
    "ArchiveBadNameError",
    "ArchiveSafetyError",
    "ArchiveSymlinkError",
    "ArchiveTraversalError",
    "DEFAULT_LIMITS",
    "extract_zip_safe",
]
