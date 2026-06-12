"""Shared Slither runner with content-addressed cache — Stage 4.

Runs Slither's Python API directly (not CLI subprocess) against
preprocessed .sol files. Caches results to avoid re-running.

Cache location: data_dir/slither_cache/<source>/<sha256>.slither.json
Cache key: sha256 of the .sol content (same as the label/rep sha256).

All three tool_validator, fp_estimator, and negative_checker use this
module so Slither runs are never duplicated.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("sentinel_data.verification.slither_runner")

# Canonical CLASS_TO_DETECTORS list (from project_agents.md).
# Only detectors available in installed Slither 0.10 are included.
# integer-overflow was removed in Slither 0.10 (Solidity 0.8+ handles it natively).
CLASS_TO_DETECTORS: dict[str, list[str]] = {
    "Reentrancy": [
        "reentrancy-eth", "reentrancy-no-eth", "reentrancy-benign", "reentrancy-events",
    ],
    "CallToUnknown": [
        "low-level-calls", "controlled-delegatecall", "delegatecall-loop",
    ],
    "Timestamp": ["timestamp"],
    "IntegerUO": [],  # no dedicated Slither detector in v0.10 for integer overflow
    "MishandledException": [
        "unchecked-send", "unchecked-lowlevel", "unchecked-transfer", "return-bomb",
    ],
    "UnusedReturn": ["unused-return"],
    "ExternalBug": [
        "arbitrary-send-eth", "low-level-calls", "tx-origin", "controlled-delegatecall",
    ],
    "DenialOfService": ["calls-loop", "costly-loop", "msg-value-loop"],
    "GasException": ["calls-loop", "costly-loop", "low-level-calls"],
    "TransactionOrderDependence": ["tx-origin", "controlled-delegatecall"],
}

# All detectors used across any class (union)
ALL_DETECTORS: set[str] = set(d for detectors in CLASS_TO_DETECTORS.values() for d in detectors)

# Slither's Python API: run_detectors returns list[list[OrderedDict]]
_SLITHER_VERSION: Optional[str] = None
_DET_REGISTRY: Optional[dict[str, type]] = None  # argument → detector class


def _get_slither_version() -> str:
    global _SLITHER_VERSION
    if _SLITHER_VERSION is None:
        try:
            from slither import Slither
            _SLITHER_VERSION = "0.10"
        except Exception:
            _SLITHER_VERSION = "unknown"
    return _SLITHER_VERSION


def _get_detector_registry() -> dict[str, type]:
    """Build argument→class map from all installed Slither detectors."""
    global _DET_REGISTRY
    if _DET_REGISTRY is not None:
        return _DET_REGISTRY
    import inspect
    import slither.detectors.all_detectors as ad
    from slither.detectors.abstract_detector import AbstractDetector
    registry: dict[str, type] = {}
    for name, obj in inspect.getmembers(ad, inspect.isclass):
        if issubclass(obj, AbstractDetector) and hasattr(obj, "ARGUMENT") and obj.ARGUMENT:
            registry[obj.ARGUMENT] = obj
    _DET_REGISTRY = registry
    return registry


def _resolve_solc_binary(solc_version: str) -> Optional[Path]:
    """Resolve solc binary from ~/.solc-select/artifacts/."""
    if not solc_version:
        return None
    bin_path = Path.home() / ".solc-select" / "artifacts" / f"solc-{solc_version}" / f"solc-{solc_version}"
    return bin_path if bin_path.exists() else None


@dataclass
class SlitherFindings:
    sha256: str
    source: str
    detectors_run: list[str]
    findings: list[dict]        # [{check, impact, confidence}]
    error: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def checks_fired(self) -> set[str]:
        return {f["check"] for f in self.findings}

    def agrees_with_class(self, class_name: str) -> bool:
        """Return True if at least one detector for class_name fired."""
        detectors = CLASS_TO_DETECTORS.get(class_name, [])
        if not detectors:
            return False  # no detector for this class
        return bool(self.checks_fired & set(detectors))

    def to_cache_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "source": self.source,
            "slither_version": _get_slither_version(),
            "detectors_run": self.detectors_run,
            "findings": self.findings,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_cache_dict(cls, d: dict) -> "SlitherFindings":
        return cls(
            sha256=d["sha256"],
            source=d["source"],
            detectors_run=d.get("detectors_run", []),
            findings=d.get("findings", []),
            error=d.get("error"),
            duration_ms=d.get("duration_ms", 0.0),
        )


def _run_slither_api(
    sol_path: Path,
    sha256: str,
    source: str,
    solc_version: str,
    detectors_to_run: list[str],
) -> SlitherFindings:
    """Run Slither's Python API on a single .sol file."""
    from slither import Slither
    from slither.exceptions import SlitherError

    registry = _get_detector_registry()
    det_classes = [registry[d] for d in detectors_to_run if d in registry]

    solc_bin = _resolve_solc_binary(solc_version)
    kwargs = {"disable_color": True}
    if solc_bin:
        kwargs["solc"] = str(solc_bin)

    t0 = time.monotonic()
    try:
        sl = Slither(str(sol_path), **kwargs)
        for DetCls in det_classes:
            sl.register_detector(DetCls)
        raw_results = sl.run_detectors()  # list[list[OrderedDict]]
        findings = []
        for result_group in raw_results:
            for finding in result_group:
                findings.append({
                    "check": finding["check"],
                    "impact": finding["impact"],
                    "confidence": finding["confidence"],
                })
        error = None
    except (SlitherError, Exception) as e:
        findings = []
        error = str(e)[:200]

    duration_ms = (time.monotonic() - t0) * 1000
    return SlitherFindings(
        sha256=sha256,
        source=source,
        detectors_run=detectors_to_run,
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
) -> Optional[SlitherFindings]:
    """Run Slither on a contract identified by sha256 + source.

    Looks up the preprocessed .sol and meta.json from
    data_dir/preprocessed/<source>/<sha256>.sol.

    Results are cached in data_dir/slither_cache/<source>/<sha256>.slither.json.

    Args:
        sha256: Contract SHA256 (matches preprocessed filename).
        source: Source name (e.g. "solidifi", "dive").
        data_dir: Path to data/ directory.
        detectors: List of detector arguments to run. Default: ALL_DETECTORS.
        force: Re-run even if cache hit exists.

    Returns:
        SlitherFindings or None if the .sol file is not found.
    """
    if detectors is None:
        detectors = sorted(ALL_DETECTORS)

    cache_dir = data_dir / "slither_cache" / source
    cache_path = cache_dir / f"{sha256}.slither.json"

    # Cache hit
    if not force and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            # Check that the cached run included all requested detectors
            cached_dets = set(cached.get("detectors_run", []))
            if set(detectors) <= cached_dets:
                return SlitherFindings.from_cache_dict(cached)
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt cache → re-run

    # Find .sol file
    sol_path = data_dir / "preprocessed" / source / f"{sha256}.sol"
    if not sol_path.exists():
        return None

    # Find solc version
    meta_path = data_dir / "preprocessed" / source / f"{sha256}.meta.json"
    solc_version = ""
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            solc_version = meta.get("solc_version", "")
        except (json.JSONDecodeError, OSError):
            pass

    result = _run_slither_api(sol_path, sha256, source, solc_version, detectors)

    # Write cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result.to_cache_dict(), indent=2))

    if result.error:
        log.debug(f"  Slither error for {sha256[:12]} ({source}): {result.error[:80]}")
    return result
