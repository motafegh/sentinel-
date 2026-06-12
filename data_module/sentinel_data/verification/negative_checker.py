"""Negative checker — Stage 4 Task 4.5.

For every contract labeled `NonVulnerable` (i.e. none of the 10 sentinel
classes is positive), runs Slither and reports what fraction has at least
one tool hit. This is the contamination check that catches the BCCC
"41% of NonVulnerable had Slither hits" pattern.

Design decisions (per plan D-4.6 and AUDIT_PATCHES 4-P4, 4-P10):
  - NonVulnerable = all 10 sentinel classes have value=0.
  - We use the canonical CLASS_TO_DETECTORS list (the union of detectors
    across all classes) — NOT a generic Slither run. The point is to
    catch OZ-flagged patterns that should be false positives on clean
    code (SafeMath, Reentrancy in `nonReentrant`, etc.). A generic
    Slither run would flag too many false positives on clean code.
  - The default hit threshold is 5% (per AUDIT_PATCHES 4-P10). 10% is
    too lax — by the time 10% of NonVulnerable contracts have tool
    hits, the class is already heavily contaminated.

Run-time contract:
  - For 7K contracts, this is a full corpus run. Slither cache makes
  subsequent runs near-instant.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.slither_runner import (
    ALL_DETECTORS, run_on_contract,
)

log = logging.getLogger("sentinel_data.verification.negative_checker")

# Default hit-rate threshold (per AUDIT_PATCHES 4-P10).
# 5% = WARN; 10% = FAIL (BCCC had 41%).
DEFAULT_WARN_THRESHOLD = 0.05
DEFAULT_FAIL_THRESHOLD = 0.10


@dataclass
class NonVulnContractCheck:
    """One NonVulnerable contract's contamination check."""
    sha256: str
    source: Optional[str]
    has_hit: bool                # True if at least one detector fired
    detectors_fired: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class NonVulnSourceStats:
    """Per-source breakdown of hit rate."""
    source: str
    total: int = 0
    hits: int = 0
    errored: int = 0

    @property
    def hit_rate(self) -> Optional[float]:
        denom = self.total - self.errored
        if denom <= 0:
            return None
        return self.hits / denom


@dataclass
class NonVulnResult:
    """Top-level result container for the NonVulnerable contamination check."""

    total_checked: int = 0            # NonVulnerable contracts found
    total_hits: int = 0               # had at least one detector fire
    total_errored: int = 0
    total_skipped: int = 0            # .sol not found
    detector_hit_counts: Counter = field(default_factory=Counter)
    by_source: dict[str, NonVulnSourceStats] = field(default_factory=dict)
    hit_contracts: list[NonVulnContractCheck] = field(default_factory=list)
    duration_s: float = 0.0
    warn_threshold: float = DEFAULT_WARN_THRESHOLD
    fail_threshold: float = DEFAULT_FAIL_THRESHOLD

    @property
    def hit_rate(self) -> Optional[float]:
        denom = self.total_checked - self.total_errored
        if denom <= 0:
            return None
        return self.total_hits / denom

    @property
    def status(self) -> str:
        """WARN / FAIL / OK based on hit rate and thresholds."""
        rate = self.hit_rate
        if rate is None:
            return "OK"  # no data — assume ok
        if rate > self.fail_threshold:
            return "FAIL"
        if rate > self.warn_threshold:
            return "WARN"
        return "OK"

    def summary_lines(self) -> list[str]:
        lines = [
            "── Negative Checker Report ───────────────────────",
            f"  NonVulnerable contracts checked: {self.total_checked}",
            f"  With at least one Slither hit:  {self.total_hits}",
            f"  Skipped (no .sol):              {self.total_skipped}",
            f"  Errored:                        {self.total_errored}",
            f"  Hit rate:                       "
            f"{self.hit_rate:.1%}" if self.hit_rate is not None else
            "  Hit rate:                       —",
            f"  Status:                         {self.status}  "
            f"(warn>{self.warn_threshold:.0%}, fail>{self.fail_threshold:.0%})",
        ]
        if self.by_source:
            lines += ["", "  Per-source breakdown:"]
            for src, st in sorted(self.by_source.items()):
                rate = f"{st.hit_rate:.1%}" if st.hit_rate is not None else "—"
                lines.append(
                    f"    {src:<25} {rate:>7}  "
                    f"({st.hits}/{st.total - st.errored} hit, {st.errored} err)"
                )
        if self.detector_hit_counts:
            lines += ["", "  Top detectors that fired on NonVulnerable:"]
            for det, n in self.detector_hit_counts.most_common(10):
                lines.append(f"    {det:<35} {n} hits")
        lines.append("─────────────────────────────────────────────────")
        return lines

    def __str__(self) -> str:
        return "\n".join(self.summary_lines())


def _is_nonvulnerable(merged: dict) -> bool:
    """Return True if the contract has no positive label in any of the 10 classes."""
    classes = merged.get("classes", {})
    for cls in class_names():
        if classes.get(cls, {}).get("value") == 1:
            return False
    return True


def run_negative_check(
    data_dir: Path,
    *,
    warn_threshold: float = DEFAULT_WARN_THRESHOLD,
    fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
    detectors: Optional[list[str]] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> NonVulnResult:
    """Run Slither on every NonVulnerable contract; report the hit rate.

    Args:
        data_dir: Path to data/ directory.
        warn_threshold: Hit rate above this = WARN (default 0.05).
        fail_threshold: Hit rate above this = FAIL (default 0.10).
        detectors: Detectors to run (default: union of CLASS_TO_DETECTORS).
        limit: If set, check at most this many NonVulnerable contracts
               (fast smoke tests).
        force: Re-run Slither even if cache hit exists.

    Returns:
        NonVulnResult with hit rate, per-source breakdown, and status.
    """
    if detectors is None:
        detectors = sorted(ALL_DETECTORS)

    merged_dir = data_dir / "labels" / "merged"
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged labels dir not found: {merged_dir}")

    result = NonVulnResult(
        warn_threshold=warn_threshold, fail_threshold=fail_threshold,
    )
    t0 = time.monotonic()

    for lf in merged_dir.glob("*.labels.json"):
        try:
            lj = json.loads(lf.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        if not _is_nonvulnerable(lj):
            continue

        if limit is not None and result.total_checked >= limit:
            break

        sha = lj["sha256"]
        sources = lj.get("sources") or [None]
        source = sources[0]
        if source is None:
            result.total_skipped += 1
            continue

        findings = run_on_contract(sha, source, data_dir, force=force, detectors=detectors)
        if findings is None:
            result.total_skipped += 1
            continue

        result.total_checked += 1
        src_stats = result.by_source.setdefault(
            source, NonVulnSourceStats(source=source)
        )
        src_stats.total += 1
        if findings.error:
            result.total_errored += 1
            src_stats.errored += 1
            continue

        has_hit = bool(findings.findings)
        if has_hit:
            result.total_hits += 1
            src_stats.hits += 1
            fired = sorted(findings.checks_fired)
            for det in fired:
                result.detector_hit_counts[det] += 1
            result.hit_contracts.append(NonVulnContractCheck(
                sha256=sha, source=source, has_hit=True, detectors_fired=fired,
            ))
            log.debug(f"  HIT on {sha[:12]} ({source}): {fired}")

    result.duration_s = time.monotonic() - t0
    rate = f"{result.hit_rate:.1%}" if result.hit_rate is not None else "—"
    log.info(
        f"  NonVuln hit rate: {rate} "
        f"({result.total_hits}/{result.total_checked - result.total_errored} hit)  "
        f"status={result.status}"
    )
    return result
