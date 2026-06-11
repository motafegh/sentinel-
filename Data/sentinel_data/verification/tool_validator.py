"""Tool validator — Stage 4 Task 4.4.

Runs Slither (default) on every labeled positive and reports the per-class
agreement rate: of 100 Reentrancy positives, how many does Slither also
flag for reentrancy?

Design decisions (per plan D-4.3):
  - Tool agreement is CORROBORATIVE, not authoritative. Slither has known
    FPs and FNs; high agreement reinforces a class, low agreement is
    suspicious but not conclusive.
  - The default tool is Slither. Mythril and Semgrep are deferred to v2.1
    (require additional install steps per the plan).
  - Per-class detector lists come from CLASS_TO_DETECTORS in slither_runner,
    which mirrors the canonical detector mapping in project_agents.md.
  - The slither_runner content-addressed cache (data/slither_cache/<source>/)
    makes repeated runs free; first run is slow (5–30s/contract).

Run-time contract:
  - 7K positives × ~5s = ~10 hours first run, ~minutes on cache hit.
  - For fast smoke tests, use `limit_per_class=N`.
  - For BCCC regression test (Task 4.7), the limit defaults to 50/class
    to match the Phase 5 sample size.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.slither_runner import (
    CLASS_TO_DETECTORS, SlitherFindings, run_on_contract,
)

log = logging.getLogger("sentinel_data.verification.tool_validator")


class AgreementVerdict(str, Enum):
    AGREE = "AGREE"             # at least one detector for this class fired
    DISAGREE = "DISAGREE"       # Slither ran cleanly; no class detectors fired
    NO_DETECTOR = "NO_DETECTOR" # class has no Slither detector in v0.10
    SKIP = "SKIP"               # .sol file not found (not preprocessed yet)
    ERROR = "ERROR"             # Slither run errored


@dataclass
class ContractAgreement:
    sha256: str
    class_name: str
    source: Optional[str]      # source the label came from (drives .sol lookup)
    verdict: AgreementVerdict
    detectors_fired: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ClassAgreementStats:
    class_name: str
    positives_total: int = 0          # labeled positives seen
    agree: int = 0                    # Slither agreed
    disagree: int = 0                 # Slither disagreed
    no_detector: int = 0              # class has no detector — skipped
    skipped: int = 0                  # .sol not found
    errored: int = 0                  # Slither run errored
    checkable: int = 0                # agree + disagree (the rate denominator)
    details: list[ContractAgreement] = field(default_factory=list)

    @property
    def agreement_rate(self) -> Optional[float]:
        """Fraction of checkable positives Slither agreed with. None if 0."""
        if self.checkable == 0:
            return None
        return self.agree / self.checkable

    @property
    def coverage(self) -> float:
        """Fraction of positives that were actually checked (not skip/error)."""
        denom = self.positives_total
        return self.checkable / denom if denom > 0 else 0.0


@dataclass
class ToolValidationResult:
    by_class: dict[str, ClassAgreementStats] = field(default_factory=dict)
    total_positives: int = 0
    total_agrees: int = 0
    total_checkable: int = 0
    total_skipped: int = 0
    total_errored: int = 0
    duration_s: float = 0.0

    def summary_lines(self) -> list[str]:
        lines = [
            "── Tool Validation Report (Slither) ──────────────",
            f"  Total positives checked: {self.total_checkable}",
            f"  Total agrees:             {self.total_agrees}  "
            f"({self.total_agrees / self.total_checkable:.1%})"
            if self.total_checkable else "  Total agrees:             0",
            "",
            "  Per-class Slither agreement:",
        ]
        for cls in class_names():
            s = self.by_class.get(cls)
            if s is None or s.positives_total == 0:
                continue
            rate = f"{s.agreement_rate:.1%}" if s.agreement_rate is not None else "—"
            det = "  (no detector)" if s.no_detector == s.positives_total else ""
            lines.append(
                f"    {cls:<30} {rate:>7}  ({s.agree}/{s.checkable} agreed)  "
                f"[{s.errored} err, {s.skipped} skip]{det}"
            )
        lines.append("─────────────────────────────────────────────────")
        return lines

    def __str__(self) -> str:
        return "\n".join(self.summary_lines())


def _find_source_for_sha(merged_entry: dict) -> Optional[str]:
    """Pick the first source in the merged label for .sol lookup."""
    sources = merged_entry.get("sources") or []
    if not sources:
        return None
    return sources[0]


def run_tool_validation(
    data_dir: Path,
    *,
    limit_per_class: Optional[int] = None,
    force: bool = False,
    only_classes: Optional[list[str]] = None,
) -> ToolValidationResult:
    """Run Slither on every labeled positive and compute per-class agreement.

    For each (class, contract) pair where the merged label value == 1, runs
    Slither and checks whether at least one detector for `class` fired.

    Args:
        data_dir: Path to data/ directory (contains labels/merged/ and preprocessed/).
        limit_per_class: If set, check at most this many positives per class
                         (fast smoke tests). Default: check all.
        force: Re-run Slither even if cache hit exists.
        only_classes: Restrict to these class names (default: all 10).

    Returns:
        ToolValidationResult with per-class agreement stats.
    """
    merged_dir = data_dir / "labels" / "merged"
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged labels dir not found: {merged_dir}")

    classes = only_classes if only_classes else class_names()
    result = ToolValidationResult(
        by_class={c: ClassAgreementStats(class_name=c) for c in classes},
    )
    t0 = time.monotonic()

    for lf in merged_dir.glob("*.labels.json"):
        try:
            lj = json.loads(lf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        sha = lj["sha256"]
        source = _find_source_for_sha(lj)
        merged_classes = lj.get("classes", {})

        for cls in classes:
            entry = merged_classes.get(cls, {})
            if entry.get("value") != 1:
                continue
            stats = result.by_class[cls]
            stats.positives_total += 1
            result.total_positives += 1

            # Class with no detector in v0.10 (e.g. IntegerUO) — skip.
            if not CLASS_TO_DETECTORS.get(cls):
                stats.no_detector += 1
                stats.details.append(ContractAgreement(
                    sha256=sha, class_name=cls, source=source,
                    verdict=AgreementVerdict.NO_DETECTOR,
                ))
                continue

            if source is None:
                stats.skipped += 1
                result.total_skipped += 1
                stats.details.append(ContractAgreement(
                    sha256=sha, class_name=cls, source=None,
                    verdict=AgreementVerdict.SKIP,
                    error="no source in merged label",
                ))
                continue

            findings = run_on_contract(
                sha, source, data_dir, force=force,
                detectors=CLASS_TO_DETECTORS[cls],
            )
            if findings is None:
                stats.skipped += 1
                result.total_skipped += 1
                stats.details.append(ContractAgreement(
                    sha256=sha, class_name=cls, source=source,
                    verdict=AgreementVerdict.SKIP,
                    error="preprocessed .sol not found",
                ))
                continue

            if findings.error:
                stats.errored += 1
                result.total_errored += 1
                stats.details.append(ContractAgreement(
                    sha256=sha, class_name=cls, source=source,
                    verdict=AgreementVerdict.ERROR,
                    error=findings.error,
                ))
                continue

            # Successful Slither run — decide AGREE / DISAGREE
            agreed = findings.agrees_with_class(cls)
            if agreed:
                stats.agree += 1
                stats.checkable += 1
                result.total_agrees += 1
                result.total_checkable += 1
                stats.details.append(ContractAgreement(
                    sha256=sha, class_name=cls, source=source,
                    verdict=AgreementVerdict.AGREE,
                    detectors_fired=sorted(findings.checks_fired),
                ))
            else:
                stats.disagree += 1
                stats.checkable += 1
                result.total_checkable += 1
                stats.details.append(ContractAgreement(
                    sha256=sha, class_name=cls, source=source,
                    verdict=AgreementVerdict.DISAGREE,
                    detectors_fired=sorted(findings.checks_fired),
                ))

            if limit_per_class is not None and stats.checkable >= limit_per_class:
                continue

        # If we just want to early-exit the outer loop, break when ALL classes hit the limit
        if limit_per_class is not None and all(
            result.by_class[c].checkable >= limit_per_class for c in classes
            if CLASS_TO_DETECTORS.get(c)  # only count classes that can be checked
        ):
            log.info(f"  limit_per_class={limit_per_class} reached for all classes; stopping early")
            break

    result.duration_s = time.monotonic() - t0
    for cls, stats in result.by_class.items():
        if stats.positives_total:
            rate = f"{stats.agreement_rate:.1%}" if stats.agreement_rate is not None else "—"
            log.info(
                f"  {cls}: {rate} agreement "
                f"({stats.agree}/{stats.checkable} agree, "
                f"{stats.no_detector} no-detector, {stats.errored} err, {stats.skipped} skip)"
            )
    return result
