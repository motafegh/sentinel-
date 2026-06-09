"""Flattener — resolve import chains to a single .sol file using solc --flatten.

Falls back gracefully: if solc --flatten fails (e.g. forge-std imports),
the original file is passed through unchanged with flatten_status=skipped.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sentinel_data.preprocessing.compiler import _available_versions, _extract_pragma, _solc_binary


@dataclass
class FlattenResult:
    content: str           # flattened source (or original if skipped)
    flatten_status: str    # "flattened" | "skipped_no_imports" | "skipped_error"
    error: str = ""


_IMPORT_RE = re.compile(r'^\s*import\s+', re.MULTILINE)


def flatten_contract(sol_path: Path) -> FlattenResult:
    """Flatten `sol_path`. Returns FlattenResult with content ready for next step."""
    source = sol_path.read_text(errors="replace")

    # Skip flattening if there are no import statements (most contracts in DeFiHackLabs)
    if not _IMPORT_RE.search(source):
        return FlattenResult(content=source, flatten_status="skipped_no_imports")

    # Try solc --flatten with the version that matches this file's pragma
    pragma = _extract_pragma(source)
    solc_bin = _pick_solc(pragma)

    if solc_bin:
        result = subprocess.run(
            [str(solc_bin), "--flatten", str(sol_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return FlattenResult(content=result.stdout, flatten_status="flattened")

    # Flatten failed (common for forge-std / hardhat imports) — pass through
    return FlattenResult(
        content=source,
        flatten_status="skipped_error",
        error="solc --flatten failed; using original source",
    )


def _pick_solc(pragma: str):
    """Return a solc binary path that matches `pragma`, or None."""
    from sentinel_data.preprocessing.compiler import _parse_version, _satisfying_versions
    available = _available_versions()
    requested = _parse_version(pragma)
    if requested:
        from sentinel_data.preprocessing.compiler import _solc_binary
        b = _solc_binary(requested)
        if b:
            return b
    candidates = _satisfying_versions(pragma, available)
    for ver in reversed(candidates):
        from sentinel_data.preprocessing.compiler import _solc_binary
        b = _solc_binary(ver)
        if b:
            return b
    return None
