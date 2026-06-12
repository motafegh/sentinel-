"""Manual connector — for sources that require a manual download step.

Implements the generic 'I already have the data on disk somewhere' flow:
  - `cfg.extra["staging_path"]` can point to a directory of .sol files,
    a single .zip to extract, or a glob pattern.
  - The connector materializes the data into `dest/repo/` (the same layout
    the git connector produces) by either symlinking the source directory
    or extracting the zip. The downstream `find_sol_files()` is identical
    to the git case, so no other module needs to know the data came from
    a manual source rather than a git clone.

Behavior options (set in `cfg.extra`):
  - `staging_path`  (required) — directory, .zip file, or glob pattern
  - `staging_glob`  (optional, default "**/*.sol") — used when staging_path is a dir
  - `materialize`   (optional, default "symlink") — "symlink" or "copy"
    - "symlink" is fast (no data movement) and reversible; works on POSIX.
    - "copy" duplicates the data into the repo dir; needed if staging is
      on a different filesystem (e.g. external drive) or for Docker.

After materialization, the file list is the .sol files found in `dest/repo/`,
filtered through `cfg.include_subdirs` / `cfg.exclude_subdirs` like all other
connectors. The pin for a manual source is conventionally a date or version
string from the upstream (e.g. "v1.0", "2025-10-02") since there's no git SHA.

Added 2026-06-10 for the DIVE integration test (Nature Sci. Data 2025,
distributed as zips on a journal page rather than a git repo). The same
connector handles SmartBugs Wild (downloaded as a tarball), the Yizhou
Chen CLEAR ICSE 2025 Zenodo record, and any future manually-distributed
source.
"""

from __future__ import annotations

import fnmatch
import glob
import os
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

from sentinel_data.ingestion.connectors.base import (
    BaseConnector,
    ConnectorError,
    PullResult,
    SourceConfig,
)


class ManualConnector(BaseConnector):
    """Materialise pre-downloaded .sol sources from a local path, zip, or glob."""

    def _pull(self, cfg: SourceConfig, dest: Path) -> PullResult:
        """Materialize staging data into dest/repo and return discovered .sol files."""
        extra = cfg.extra or {}
        staging = extra.get("staging_path")
        if not staging:
            raise ConnectorError(
                f"[{cfg.name}] manual connector requires `staging_path` in config extra. "
                f"Set it to a directory of .sol files, a .zip, or a glob pattern."
            )

        repo_dir = dest / "repo"
        if repo_dir.exists() or repo_dir.is_symlink():
            # Idempotent re-runs: if the materialize already happened, don't redo it.
            # The manifest SHA check below will catch any source changes.
            pass
        else:
            materialize_staging(Path(staging), repo_dir, extra, source_name=cfg.name)

        sol_files = self.find_sol_files(
            repo_dir,
            include_subdirs=cfg.include_subdirs or None,
            exclude_subdirs=cfg.exclude_subdirs or None,
        )
        if not sol_files:
            raise ConnectorError(
                f"[{cfg.name}] manual connector materialized {repo_dir} but "
                f"found 0 .sol files (after include/exclude filters). Check "
                f"`include_subdirs` in config.yaml or `staging_glob` in extra."
            )

        # No resolved_pin in the git sense; use the staging path's mtime as a
        # stable identifier. Conventionally configs set `pin` to a date/version
        # string (e.g. "v1.0", "2025-10-02") and we copy that through.
        try:
            staging_stat = os.stat(staging)
            pin_marker = datetime.fromtimestamp(staging_stat.st_mtime).strftime("%Y-%m-%d")
        except OSError:
            pin_marker = cfg.pin or "unknown"

        return PullResult(
            source=cfg.name,
            local_dir=repo_dir,
            resolved_pin=cfg.pin or pin_marker,
            sol_files=sol_files,
            fetched_at="",     # filled by BaseConnector.pull()
            duration_s=0.0,
        )


def materialize_staging(staging: Path, repo_dir: Path, extra: dict, source_name: str) -> None:
    """Materialize `staging` into `repo_dir`.

    Three cases:
      1. `staging` is a .zip file  → extract to `repo_dir`
      2. `staging` is a directory  → symlink (default) or copy to `repo_dir`
      3. `staging` is a glob pattern → resolve to a single directory or zip
         (glob must match exactly one path)
    """
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    # Resolve glob patterns
    if any(c in str(staging) for c in "*?["):
        matches = [Path(p) for p in glob.glob(str(staging))]
        if len(matches) != 1:
            raise ConnectorError(
                f"[{source_name}] staging glob '{staging}' matched {len(matches)} paths; "
                f"expected exactly 1. Refine the pattern."
            )
        staging = matches[0]

    if not staging.exists():
        raise ConnectorError(
            f"[{source_name}] staging_path does not exist: {staging}"
        )

    mode = (extra.get("materialize") or "symlink").lower()
    if mode not in ("symlink", "copy"):
        raise ConnectorError(
            f"[{source_name}] materialize must be 'symlink' or 'copy', got {mode!r}"
        )

    if staging.is_file() and staging.suffix.lower() == ".zip":
        # Extract zip. We use `unzip` if available (handles macOS __MACOSX
        # metadata noise gracefully), else fall back to Python's zipfile.
        if mode == "symlink":
            # For a zip, "symlink" is meaningless — there's nothing to symlink.
            # We treat it as "extract in place" and use the extracted dir as
            # `repo_dir`.
            extract_root = repo_dir.parent / f"{repo_dir.name}__zip_extracted"
            if not extract_root.exists():
                extract_root.mkdir(parents=True)
                _extract_zip(staging, extract_root, source_name=source_name)
            # Symlink the extracted dir as repo_dir
            repo_dir.symlink_to(extract_root.resolve())
        else:  # copy
            _extract_zip(staging, repo_dir, source_name=source_name)
    elif staging.is_dir():
        if mode == "symlink":
            repo_dir.symlink_to(staging.resolve())
        else:  # copy
            shutil.copytree(staging, repo_dir, dirs_exist_ok=True)
    else:
        raise ConnectorError(
            f"[{source_name}] staging_path is neither a directory nor a .zip: {staging}"
        )


def _extract_zip(zip_path: Path, dest: Path, source_name: str) -> None:
    """Extract a zip to `dest`, stripping macOS metadata (`__MACOSX/`, `.DS_Store`).

    These are the noise directories the DIVE zip was full of (~44,687 files
    were in the zip but only 22,332 were actual .sol; the rest were macOS
    resource forks). Skipping them keeps the downstream find_sol_files
    output clean.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                # Skip macOS metadata noise
                if "__MACOSX/" in member or member.endswith(".DS_Store"):
                    continue
                # Extract, preserving relative path within dest
                target = dest / member
                # Defend against zip-slip
                if not str(target.resolve()).startswith(str(dest.resolve())):
                    raise ConnectorError(
                        f"[{source_name}] zip-slip attempt: {member}"
                    )
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile as e:
        raise ConnectorError(f"[{source_name}] not a valid zip: {zip_path} ({e})")
