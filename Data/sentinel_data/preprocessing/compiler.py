"""Compiler — resolve pragma solidity version, invoke solc binary, two-pass on failure.

Two-pass compile (per Stage 1 plan D-1.4 + AUDIT_PATCHES 1-P1, 1-P2):
  Pass 1: extract pragma exactly as written, clean whitespace, try that version.
  Pass 2: relax — try the nearest available version if exact isn't installed.

Bugs fixed vs. Phase 5 Session 3 retry script:
  (a) spaced pragmas like `^ 0.4 .9` — strip whitespace before regex
  (b) exact-version pragmas like `0.4.25` compiled with wrong version — try requested first
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Root where solc-select stores binaries
_SOLC_ARTIFACTS = Path.home() / ".solc-select" / "artifacts"

_PRAGMA_RE = re.compile(
    r"pragma\s+solidity\s+([^;]+);",
    re.MULTILINE,
)


@dataclass
class CompileResult:
    success: bool
    solc_version: str        # version actually used
    pragma_raw: str          # original pragma string from file
    error: str = ""          # compile error message on failure
    attempted_versions: list[str] = None

    def __post_init__(self):
        if self.attempted_versions is None:
            self.attempted_versions = []


def compile_contract(sol_path: Path) -> CompileResult:
    """Try to compile `sol_path` with the appropriate solc version.

    Returns CompileResult with success=True/False and the version used.
    Does NOT raise — all errors are captured in CompileResult.error.

    On failure, CompileResult.error contains the LAST version's stderr
    (the most recent attempted version is usually the most informative —
    the candidates are tried newest-first so the last one tried is the
    most likely to match the file's pragma).
    """
    source = sol_path.read_text(errors="replace")
    pragma_raw = _extract_pragma(source)

    if not pragma_raw:
        return CompileResult(
            success=False,
            solc_version="",
            pragma_raw="",
            error="no pragma solidity found",
        )

    # Pass 1: try exact requested version
    requested = _parse_version(pragma_raw)
    attempted: list[str] = []
    last_err = ""

    if requested:
        bin_path = _solc_binary(requested)
        if bin_path:
            attempted.append(requested)
            ok, err = _run_solc(bin_path, sol_path)
            if ok:
                return CompileResult(True, requested, pragma_raw, attempted_versions=attempted)
            last_err = err

    # Pass 2: try nearest available version that satisfies the constraint
    available = _available_versions()
    candidates = _satisfying_versions(pragma_raw, available)

    for ver in candidates:
        if ver in attempted:
            continue
        bin_path = _solc_binary(ver)
        if not bin_path:
            continue
        attempted.append(ver)
        ok, err = _run_solc(bin_path, sol_path)
        if ok:
            return CompileResult(True, ver, pragma_raw, attempted_versions=attempted)
        last_err = err

    return CompileResult(
        success=False,
        solc_version="",
        pragma_raw=pragma_raw,
        error=f"all versions failed; last error: {last_err[:300]}",
        attempted_versions=attempted,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_pragma(source: str) -> str:
    m = _PRAGMA_RE.search(source)
    if not m:
        return ""
    # strip internal whitespace (fixes `^ 0.4 .9` → `^0.4.9`)
    return re.sub(r"\s+", "", m.group(1))


def _parse_version(pragma: str) -> str:
    """Extract a single clean version string from a pragma constraint, or '' if ambiguous."""
    # exact version: `0.4.25` or `=0.4.25`
    m = re.fullmatch(r"=?(\d+\.\d+\.\d+)", pragma)
    if m:
        return m.group(1)
    # caret / tilde with single version: `^0.8.0`, `~0.6.12`
    m = re.fullmatch(r"[\^~>=]*(\d+\.\d+\.\d+)", pragma)
    if m:
        return m.group(1)
    return ""


def _satisfying_versions(pragma: str, available: list[str]) -> list[str]:
    """Return versions from `available` that could satisfy `pragma`, newest first."""
    # Extract the floor version from the pragma
    m = re.search(r"(\d+\.\d+\.\d+)", pragma)
    if not m:
        return available  # no version found — try all, newest first

    floor = tuple(int(x) for x in m.group(1).split("."))

    # Determine ceiling from double-bound range pragmas like `>=0.4.0 <0.9.0`
    ceiling_m = re.search(r"<(\d+\.\d+\.\d+)", pragma)
    ceiling = tuple(int(x) for x in ceiling_m.group(1).split(".")) if ceiling_m else None

    def satisfies(ver: str) -> bool:
        v = tuple(int(x) for x in ver.split("."))
        if v < floor:
            return False
        if ceiling and v >= ceiling:
            return False
        return True

    return [v for v in reversed(available) if satisfies(v)]


def _available_versions() -> list[str]:
    """Return all installed solc versions, oldest first."""
    if not _SOLC_ARTIFACTS.exists():
        return []
    vers = []
    for d in _SOLC_ARTIFACTS.iterdir():
        if d.is_dir() and d.name.startswith("solc-"):
            ver = d.name[len("solc-"):]
            if re.fullmatch(r"\d+\.\d+\.\d+", ver):
                vers.append(ver)
    return sorted(vers, key=lambda v: tuple(int(x) for x in v.split(".")))


def _solc_binary(version: str) -> Path | None:
    p = _SOLC_ARTIFACTS / f"solc-{version}" / f"solc-{version}"
    return p if p.exists() else None


def _run_solc(bin_path: Path, sol_path: Path) -> tuple[bool, str]:
    # Pass --allow-paths so relative imports like `../interface.sol` from files
    # in deep subdirs (e.g. DeFiHackLabs' 2018-04/ -> ../interface.sol) resolve
    # without solc 0.8.x's "File outside of allowed directories" security check
    # blocking. The default allowed scope is the source file's directory and
    # below; we expand it to the entire source repo (sol_path's parent's parent
    # is a reasonable upper bound — going beyond the cloned repo is a sign
    # of misconfiguration, not a legitimate need).
    allow_root = sol_path.parent.parent
    result = subprocess.run(
        [str(bin_path), "--bin", str(sol_path),
         "--allow-paths", str(allow_root)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        return True, ""
    return False, (result.stderr or result.stdout)[-500:]
