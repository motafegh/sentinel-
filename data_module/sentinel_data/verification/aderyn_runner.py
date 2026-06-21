"""Aderyn runner with content-addressed cache — uses Aderyn CLI (Rust, Cyfrin).

Aderyn uses one workspace at a time. We run it per-file (each .sol in its own
temp dir) to handle mixed solc versions safely, mirroring the BCCC audit pattern
(2026-06-14, `~/.claude/scratch/bccc_aderyn_audit_20260614.md`).

Cache location: data_dir/aderyn_cache/<source>/<sha256>.aderyn.json
Cache key: sha256 of the .sol content (same as the label/rep sha256).

Detector mapping mirrors slither_runner.CLASS_TO_DETECTORS but uses Aderyn's
own detector names (different vocabulary).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("sentinel_data.verification.aderyn_runner")

# Canonical ADERYN_TO_DETECTORS list (mapped from the same class semantics as
# the Slither version, but using Aderyn's detector vocabulary — see
# `aderyn registry` for the full list).
ADERYN_DETECTORS: dict[str, list[str]] = {
    "Reentrancy": [
        "reentrancy-state-change",
        "non-reentrant-not-first",
        "unchecked-send",  # sent ETH without checking return value (often reentrancy-adjacent)
    ],
    "ExternalBug": [
        "tx-origin-used-for-auth",
        "eth-send-unchecked-address",
        "delegate-call-unchecked-address",
        "arbitrary-transfer-from",
        "state-no-address-check",
        "incorrect-erc20-interface",
        "constant-function-changes-state",
    ],
    # Other classes not relevant to current DIVE crosswalk investigation:
    # ("CallToUnknown", "Timestamp", "IntegerUO", "MishandledException",
    #  "UnusedReturn", "DenialOfService", "GasException",
    #  "TransactionOrderDependence") are mapped to empty lists here — out of
    #  scope for the EB+RE plan; add later if needed.
}

# All Aderyn detectors we care about across any class
ALL_ADERYN_DETECTORS: set[str] = set(
    d for detectors in ADERYN_DETECTORS.values() for d in detectors
)

# Try to find the aderyn binary in common locations
_ADERYN_BIN_CANDIDATES = [
    "/home/motafeq/.cargo/bin/aderyn",
    os.path.expanduser("~/.cargo/bin/aderyn"),
    "/usr/local/bin/aderyn",
]


def _find_aderyn_bin() -> Optional[str]:
    import shutil
    p = shutil.which("aderyn")
    if p:
        return p
    for c in _ADERYN_BIN_CANDIDATES:
        if Path(c).exists() and os.access(c, os.X_OK):
            return c
    return None


def _check_aderyn_available() -> None:
    if _find_aderyn_bin() is None:
        raise FileNotFoundError(
            "Aderyn binary not found. Install via `cargo install aderyn` "
            "(from https://github.com/cyfrin/aderyn) and ensure ~/.cargo/bin "
            "is in PATH."
        )


@dataclass
class AderynFindings:
    """Result of running Aderyn on a single contract."""

    sha256: str
    source: str
    detectors_run: list[str]
    findings: list[dict]        # [{detector_name, title, line_count, ...}]
    error: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def checks_fired(self) -> set[str]:
        return {f["detector_name"] for f in self.findings}

    def agrees_with_class(self, class_name: str) -> bool:
        """Return True if at least one Aderyn detector for class_name fired."""
        detectors = ADERYN_DETECTORS.get(class_name, [])
        if not detectors:
            return False
        return bool(self.checks_fired & set(detectors))

    def to_cache_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "source": self.source,
            "aderyn_version": "0.6.8",
            "detectors_run": self.detectors_run,
            "findings": self.findings,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_cache_dict(cls, d: dict) -> "AderynFindings":
        return cls(
            sha256=d["sha256"],
            source=d["source"],
            detectors_run=d.get("detectors_run", []),
            findings=d.get("findings", []),
            error=d.get("error"),
            duration_ms=d.get("duration_ms", 0.0),
        )


def _run_aderyn_on_file(
    sol_path: Path,
    sha256: str,
    source: str,
) -> AderynFindings:
    """Run Aderyn on a single .sol file (in its own temp workspace)."""
    _check_aderyn_available()
    aderyn_bin = _find_aderyn_bin()

    with tempfile.TemporaryDirectory(prefix="aderyn_") as workdir:
        workdir = Path(workdir)
        # Always copy (not symlink) — Aderyn panics with "File Not Found in
        # Ignore stats" when given a symlinked source (verified 2026-06-18).
        target = workdir / sol_path.name
        import shutil
        shutil.copy(sol_path, target)
        out_path = workdir / "report.json"
        t0 = time.monotonic()
        findings = []
        error = None
        try:
            result = subprocess.run(
                [aderyn_bin, str(workdir), "--src", str(workdir), "-o", str(out_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if out_path.exists():
                try:
                    rpt = json.loads(out_path.read_text())
                    for issue in rpt.get("high_issues", {}).get("issues", []):
                        findings.append({
                            "detector_name": issue.get("detector_name"),
                            "title": issue.get("title"),
                            "instances": len(issue.get("instances", [])),
                        })
                    for issue in rpt.get("low_issues", {}).get("issues", []):
                        findings.append({
                            "detector_name": issue.get("detector_name"),
                            "title": issue.get("title"),
                            "instances": len(issue.get("instances", [])),
                            "severity": "low",
                        })
                except (json.JSONDecodeError, OSError) as e:
                    error = f"parse error: {str(e)[:200]}"
            else:
                error = (result.stderr or result.stdout or "")[:200]
        except subprocess.TimeoutExpired:
            error = "aderyn timeout (60s)"
        except Exception as e:
            error = str(e)[:200]

        duration_ms = (time.monotonic() - t0) * 1000

    return AderynFindings(
        sha256=sha256,
        source=source,
        detectors_run=sorted(ALL_ADERYN_DETECTORS),
        findings=findings,
        error=error,
        duration_ms=duration_ms,
    )


def run_on_contract(
    sha256: str,
    source: str,
    data_dir: Path,
    *,
    detectors: Optional[list[str]] = None,
    force: bool = False,
) -> Optional[AderynFindings]:
    """Run Aderyn on a contract identified by sha256 + source.

    Looks up the preprocessed .sol from
    data_dir/preprocessed/<source>/<sha256>.sol.

    Results are cached in data_dir/aderyn_cache/<source>/<sha256>.aderyn.json.

    Returns:
        AderynFindings or None if the .sol file is not found.
    """
    if detectors is None:
        detectors = sorted(ALL_ADERYN_DETECTORS)

    cache_dir = data_dir / "aderyn_cache" / source
    cache_path = cache_dir / f"{sha256}.aderyn.json"

    # Cache hit
    if not force and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            return AderynFindings.from_cache_dict(cached)
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt cache → re-run

    sol_path = data_dir / "preprocessed" / source / f"{sha256}.sol"
    if not sol_path.exists():
        return None

    result = _run_aderyn_on_file(sol_path, sha256, source)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result.to_cache_dict(), indent=2))

    if result.error:
        log.debug(f"  Aderyn error for {sha256[:12]} ({source}): {result.error[:80]}")
    return result
