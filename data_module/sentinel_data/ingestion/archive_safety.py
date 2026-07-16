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

import shutil
import stat
import tempfile
import unicodedata
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fcntl


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


class ArchiveCollisionError(ArchiveSafetyError):
    """Two members resolve to the same portable extraction path."""


class ArchivePromotionError(ArchiveSafetyError):
    """The staged archive could not be promoted without losing prior state."""


class ArchiveCleanupError(ArchiveSafetyError):
    """Extraction or promotion cleanup failed and left a named artifact."""


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
    if name.startswith(("/", "\\")):
        raise ArchiveBadNameError(f"absolute member name: {name!r}")
    if "\\" in name:
        raise ArchiveBadNameError(f"backslash separator in member name: {name!r}")
    if any(part in {".", ".."} for part in name.split("/")):
        raise ArchiveTraversalError(f"path traversal component in member name: {name!r}")


def _is_symlink_mode(external_attr: int) -> bool:
    """Check if the ZIP external_attr indicates a symlink."""
    unix_mode = external_attr >> 16
    return stat.S_ISLNK(unix_mode)


def _is_special_mode(external_attr: int) -> bool:
    """Check if the ZIP external_attr indicates a device, FIFO, or socket."""
    unix_mode = external_attr >> 16
    return (
        stat.S_ISCHR(unix_mode)
        or stat.S_ISBLK(unix_mode)
        or stat.S_ISFIFO(unix_mode)
        or stat.S_ISSOCK(unix_mode)
    )


def _check_path_depth(name: str, max_depth: int) -> None:
    """Reject paths exceeding the configured depth limit."""
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if len(parts) > max_depth:
        raise ArchiveLimitError(f"path depth {len(parts)} exceeds limit {max_depth}: {name!r}")


def _portable_member_key(name: str) -> tuple[str, bool]:
    """Return a cross-platform collision key and whether the member is a directory."""

    portable = name.replace("\\", "/")
    is_directory = portable.endswith("/")
    parts = [unicodedata.normalize("NFC", part).casefold() for part in portable.split("/") if part]
    return "/".join(parts), is_directory


def _validate_member_inventory(
    infos: list[zipfile.ZipInfo], *, source_name: str, limits: ArchiveLimits
) -> None:
    """Reject ambiguous archives before writing a single member."""

    seen: dict[str, tuple[str, bool]] = {}
    required_directories: set[str] = set()
    for info in infos:
        name = info.filename
        if "__MACOSX/" in name or name.endswith(".DS_Store"):
            continue
        _check_member_name(name)
        _check_path_depth(name, limits.max_path_depth)
        key, is_directory = _portable_member_key(name)
        previous = seen.get(key)
        if previous is not None:
            raise ArchiveCollisionError(
                f"[{source_name}] colliding members {previous[0]!r} and {name!r}"
            )
        if not is_directory and key in required_directories:
            raise ArchiveCollisionError(
                f"[{source_name}] file member {name!r} collides with an existing directory path"
            )
        for parent in Path(key).parents:
            parent_key = parent.as_posix()
            if parent_key == ".":
                break
            parent_entry = seen.get(parent_key)
            if parent_entry is not None and not parent_entry[1]:
                raise ArchiveCollisionError(
                    f"[{source_name}] member {name!r} descends from file {parent_entry[0]!r}"
                )
            required_directories.add(parent_key)
        seen[key] = (name, is_directory)


@contextmanager
def _destination_lock(dest: Path) -> Iterator[None]:
    lock_path = dest.parent / f".{dest.name}__extraction.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _remove_staging_after_failure(tmp_root: Path, original: BaseException) -> None:
    try:
        shutil.rmtree(tmp_root)
    except OSError as cleanup_error:
        raise ArchiveCleanupError(
            f"failed to clean staging directory {tmp_root} after {type(original).__name__}: "
            f"{cleanup_error}"
        ) from cleanup_error


def _promote_with_rollback(tmp_root: Path, dest: Path) -> None:
    backup = Path(tempfile.mkdtemp(prefix=f".{dest.name}__backup_", dir=dest.parent))
    backup.rmdir()
    previous_moved = False
    try:
        if dest.exists():
            dest.replace(backup)
            previous_moved = True
        tmp_root.replace(dest)
    except OSError as promotion_error:
        if previous_moved:
            try:
                backup.replace(dest)
            except OSError as rollback_error:
                raise ArchivePromotionError(
                    f"promotion failed for {dest} and rollback failed; prior state is at "
                    f"{backup}: promotion={promotion_error}; rollback={rollback_error}"
                ) from rollback_error
        _remove_staging_after_failure(tmp_root, promotion_error)
        raise ArchivePromotionError(
            f"promotion failed for {dest}; prior destination restored: {promotion_error}"
        ) from promotion_error

    if previous_moved:
        try:
            shutil.rmtree(backup)
        except OSError as cleanup_error:
            raise ArchiveCleanupError(
                f"new destination installed at {dest}, but previous-state cleanup failed "
                f"for {backup}: {cleanup_error}"
            ) from cleanup_error


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

    dest = dest.absolute()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        raise ArchiveSymlinkError(f"[{source_name}] destination is a symlink: {dest}")

    tmp_root = Path(tempfile.mkdtemp(prefix=f".{dest.name}__extraction_", dir=dest.parent))

    total_bytes = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()

            if len(infos) > limits.max_members:
                raise ArchiveLimitError(
                    f"[{source_name}] archive has {len(infos)} members, "
                    f"limit is {limits.max_members}"
                )

            _validate_member_inventory(infos, source_name=source_name, limits=limits)

            for info in infos:
                name = info.filename
                if "__MACOSX/" in name or name.endswith(".DS_Store"):
                    continue

                if _is_symlink_mode(info.external_attr):
                    raise ArchiveSymlinkError(f"[{source_name}] symlink member rejected: {name!r}")

                if _is_special_mode(info.external_attr):
                    raise ArchiveSafetyError(
                        f"[{source_name}] special file member rejected: {name!r}"
                    )

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
        error = ArchiveSafetyError(f"[{source_name}] not a valid zip: {zip_path} ({e})")
        if tmp_root.exists():
            _remove_staging_after_failure(tmp_root, error)
        raise error from e
    except Exception as error:
        if tmp_root.exists():
            _remove_staging_after_failure(tmp_root, error)
        raise

    with _destination_lock(dest):
        if dest.is_symlink():
            _remove_staging_after_failure(
                tmp_root,
                ArchiveSymlinkError(f"destination became a symlink during extraction: {dest}"),
            )
            raise ArchiveSymlinkError(
                f"[{source_name}] destination became a symlink during extraction: {dest}"
            )
        _promote_with_rollback(tmp_root, dest)
    return dest


__all__ = [
    "ArchiveLimitError",
    "ArchiveLimits",
    "ArchiveBadNameError",
    "ArchiveCleanupError",
    "ArchiveCollisionError",
    "ArchivePromotionError",
    "ArchiveSafetyError",
    "ArchiveSymlinkError",
    "ArchiveTraversalError",
    "DEFAULT_LIMITS",
    "extract_zip_safe",
]
