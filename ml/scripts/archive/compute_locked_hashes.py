#!/usr/bin/env python3
"""
compute_locked_hashes.py — Pin the v4-sprint architecture freeze.

WHY THIS EXISTS
───────────────
The v4 retrain holds the model architecture, graph schema, and the eval
split constant so v4 numbers are directly comparable to v3's tuned
F1-macro 0.5069. The autoresearch harness (round 2) reads the sidecar
written by this script and aborts any run that would silently change
one of those files.

This is a discipline tool, not a security control. The lock is scoped
to the v4 sprint — Phase B (Architecture Playground) explicitly lifts
it for everything except the eval split (see
`docs/changes/2026-05-09-M1-autoresearch-integration-plan.md` §7).

WHAT IT WRITES
──────────────
`ml/locked_files.sha256` — one line per file, in the
standard `sha256sum` format:

    <sha256_hex>  <relative_path>

The file is verifiable with the system tool:

    sha256sum -c ml/locked_files.sha256

Round 2's `auto_experiment.py` will parse the same file natively.

OPERATOR WORKFLOW
─────────────────
The hashes for source files (sentinel_model.py, gnn_encoder.py,
graph_schema.py, graph_extractor.py) can be computed anywhere — they
live in the repo. The hash for `ml/data/splits/val_indices.npy` can
only be computed on a machine that has the data pulled (i.e. your
laptop after `dvc pull`).

Recommended sequence:

    # On your RTX 3070 laptop, with data pulled:
    poetry run python ml/scripts/compute_locked_hashes.py --write
    git add ml/locked_files.sha256
    git commit -m "data: pin v4-sprint locked-file hashes"

    # Anywhere afterwards, to verify:
    poetry run python ml/scripts/compute_locked_hashes.py --check

The script tolerates missing files in `--write` mode (it warns and
omits the entry) so it can be partially populated from a clean
checkout, but the sidecar is only useful once `val_indices.npy` is
included.

Usage:
    python ml/scripts/compute_locked_hashes.py [--write | --check]
                                               [--sidecar PATH]
                                               [--repo-root PATH]
                                               [--allow-missing]

Exit codes:
    0   write succeeded; or --check matched all entries
    1   --check found a mismatch, missing file, or could not read sidecar
    2   --write was asked to record nothing (all files missing) without
        --allow-missing
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# v4-sprint locked file set
# ---------------------------------------------------------------------------
# These paths are RELATIVE to the repo root and are written that way into the
# sidecar so `sha256sum -c` works when invoked from the repo root.
#
# Source-file entries pin the architecture for the duration of the v4 sprint.
# val_indices.npy is the eval gate and is permanently locked — Phase B
# (Architecture Playground) lifts the source-file locks but keeps this one.
LOCKED_V4_SPRINT: tuple[str, ...] = (
    "ml/src/models/sentinel_model.py",
    "ml/src/models/gnn_encoder.py",
    "ml/src/preprocessing/graph_schema.py",
    "ml/src/preprocessing/graph_extractor.py",
    "ml/data/splits/val_indices.npy",
)

DEFAULT_SIDECAR = Path("ml/locked_files.sha256")

# Streaming buffer — large enough to keep IO efficient on the .npy file
# (~few MB), small enough to not balloon RAM for source files.
_HASH_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the lowercase hex sha256 of a file, streamed in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sidecar(text: str) -> dict[str, str]:
    """
    Parse a `sha256sum`-format sidecar.

    Each non-blank, non-comment line is `<hex>  <path>` (two spaces).
    Lines starting with `#` are comments and are ignored. The relative
    path may contain spaces; only the first whitespace-block separates
    hash from path.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # sha256sum uses two spaces; we accept any run of whitespace to be
        # forgiving about hand-edited sidecars.
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"malformed sidecar line: {raw!r}")
        digest, rel_path = parts
        digest = digest.strip()
        rel_path = rel_path.strip()
        # `sha256sum` prefixes the path with `*` for binary mode; strip it.
        if rel_path.startswith("*"):
            rel_path = rel_path[1:]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"not a sha256 digest: {digest!r}")
        out[rel_path] = digest
    return out


def render_sidecar(entries: dict[str, str]) -> str:
    """Render entries in deterministic order — stable across runs and OSes."""
    header = (
        "# v4-sprint locked-file hashes (see docs/changes/"
        "2026-05-09-M1-autoresearch-integration-plan.md §7).\n"
        "# Format: sha256sum-compatible. Verify with:\n"
        "#   sha256sum -c ml/locked_files.sha256\n"
        "# Regenerate with:\n"
        "#   poetry run python ml/scripts/compute_locked_hashes.py --write\n"
    )
    body_lines = [f"{entries[p]}  {p}" for p in sorted(entries)]
    return header + "\n".join(body_lines) + "\n"


def write_mode(
    repo_root: Path, sidecar: Path, allow_missing: bool
) -> int:
    """Compute hashes and write the sidecar. Tolerates missing files when allowed."""
    entries: dict[str, str] = {}
    missing: list[str] = []
    for rel in LOCKED_V4_SPRINT:
        full = repo_root / rel
        if not full.is_file():
            missing.append(rel)
            continue
        entries[rel] = sha256_file(full)

    for rel in missing:
        print(f"WARNING: skipping (file not present): {rel}", file=sys.stderr)

    if not entries:
        print(
            "ERROR: no locked files were found. Are you running from the "
            "repo root and on a machine with the data pulled (dvc pull)?",
            file=sys.stderr,
        )
        return 2
    if missing and not allow_missing:
        print(
            "ERROR: some locked files are missing and --allow-missing was "
            "not passed. The sidecar would be incomplete; refusing to "
            "write.\n"
            "Pass --allow-missing only when partially populating from a "
            "clean checkout (the data file's hash will be added later by "
            "running this script on the laptop).",
            file=sys.stderr,
        )
        return 2

    sidecar_full = repo_root / sidecar
    sidecar_full.parent.mkdir(parents=True, exist_ok=True)
    sidecar_full.write_text(render_sidecar(entries), encoding="utf-8")

    print(f"Wrote {len(entries)} entries to {sidecar}", file=sys.stderr)
    if missing:
        print(
            f"  ({len(missing)} file(s) skipped; rerun this script on a "
            f"machine that has them, then commit the updated sidecar.)",
            file=sys.stderr,
        )
    return 0


def check_mode(repo_root: Path, sidecar: Path) -> int:
    """Verify all sidecar entries match the current files. Exit 0/1."""
    sidecar_full = repo_root / sidecar
    if not sidecar_full.is_file():
        print(f"ERROR: sidecar not found: {sidecar}", file=sys.stderr)
        return 1
    try:
        expected = parse_sidecar(sidecar_full.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"ERROR: sidecar is malformed: {exc}", file=sys.stderr)
        return 1
    if not expected:
        print(f"ERROR: sidecar has no entries: {sidecar}", file=sys.stderr)
        return 1

    bad: list[str] = []
    for rel, want in sorted(expected.items()):
        full = repo_root / rel
        if not full.is_file():
            print(f"MISMATCH: {rel} — file missing", file=sys.stderr)
            bad.append(rel)
            continue
        got = sha256_file(full)
        if got != want:
            print(
                f"MISMATCH: {rel}\n  expected {want}\n  got      {got}",
                file=sys.stderr,
            )
            bad.append(rel)

    if bad:
        print(
            f"FAIL: {len(bad)} of {len(expected)} entries did not match.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: all {len(expected)} entries match.", file=sys.stderr)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute or verify v4-sprint locked-file hashes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Compute and write the sidecar:\n"
            "    python ml/scripts/compute_locked_hashes.py --write\n"
            "  Compute partially (CI / clean checkout — data not pulled):\n"
            "    python ml/scripts/compute_locked_hashes.py --write "
            "--allow-missing\n"
            "  Verify the sidecar against current files:\n"
            "    python ml/scripts/compute_locked_hashes.py --check\n"
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--write",
        action="store_true",
        help="Compute hashes and write the sidecar (default).",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify the sidecar against current files; exit 1 on mismatch.",
    )
    p.add_argument(
        "--sidecar",
        type=Path,
        default=DEFAULT_SIDECAR,
        help=f"Sidecar path relative to repo root (default: {DEFAULT_SIDECAR}).",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Project root. Defaults to the directory containing .git/ above "
             "this script.",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="In --write mode, allow the sidecar to be partially populated "
             "(skip files that are not on disk). Useful on a fresh checkout "
             "where data has not yet been pulled.",
    )
    args = p.parse_args()
    if not args.write and not args.check:
        args.write = True  # default
    return args


def find_repo_root(script_path: Path) -> Path:
    """Walk up from this script until we find the .git directory (true repo root).

    The repo has nested pyproject.toml files (ml/, agents/) so .git is the
    only unambiguous marker.
    """
    here = script_path.resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise SystemExit(
        "Could not locate repo root (no .git found above this script). "
        "Pass --repo-root explicitly."
    )


def main() -> int:
    args = parse_args()
    repo_root = (args.repo_root or find_repo_root(Path(__file__))).resolve()
    if args.check:
        return check_mode(repo_root, args.sidecar)
    return write_mode(repo_root, args.sidecar, args.allow_missing)


if __name__ == "__main__":
    sys.exit(main())
