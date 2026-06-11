"""FP estimator — Stage 4 Task 4.5 (stratified by source AND tier).

Samples N positives per class (default 50), runs all enabled tools on the
sample, and reports an empirical false-positive rate per (class, source, tier)
stratum.

Design decision (per AUDIT_PATCHES 4-P9):
  Sampling is stratified by SOURCE and CONFIDENCE TIER (not just by class).
  A class with 90% T3 labels and 10% T0 labels has a very different FP rate
  by tier — the per-tier per-class breakdown is the operational signal.

FP definition (v1 — corroborated-tool disagreement):
  A sampled positive is flagged as a "likely FP" if NO tool (currently only
  Slither) fires for that class on the contract. This is an upper-bound
  estimate (a positive that Slither misses is not necessarily a wrong label
  — Slither has known FNs), but a high per-class rate IS diagnostic of
  noise. The v2.1 enhancement is to compound Slither-disagreement with
  semantic-checker FAIL (the compound rate is the ground truth).

Run-time contract:
  - N=50/class × 10 classes × ~5s/Slither = ~40 minutes first run
  - Subsequent runs are fast (slither_runner content-addressed cache)
  - For BCCC regression test (Task 4.7) the limit defaults to 50/class
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.slither_runner import (
    CLASS_TO_DETECTORS, run_on_contract,
)

log = logging.getLogger("sentinel_data.verification.fp_estimator")

# Hard ceiling above which a class is FAIL in the gate (per plan D-4.5)
FP_RATE_FAIL_THRESHOLD = 0.30


@dataclass
class FPRecord:
    """One sampled positive's FP estimation result."""
    sha256: str
    class_name: str
    source: Optional[str]
    tier: Optional[str]
    likely_fp: bool        # True if no tool agreed
    error: Optional[str] = None   # set if Slither run errored


@dataclass
class StratumStats:
    """Per (class, source, tier) stratum."""
    class_name: str
    source: str
    tier: str
    sampled: int = 0
    likely_fp: int = 0
    errored: int = 0

    @property
    def fp_rate(self) -> Optional[float]:
        denom = self.sampled - self.errored
        if denom <= 0:
            return None
        return self.likely_fp / denom


@dataclass
class ClassFPStats:
    """Per-class aggregate over all strata."""
    class_name: str
    total_positives_in_corpus: int = 0   # labeled positives available
    sampled: int = 0                     # total drawn into the sample
    likely_fp: int = 0
    errored: int = 0
    strata: dict[tuple[str, str], StratumStats] = field(default_factory=dict)
    # strata key = (source, tier)

    @property
    def fp_rate(self) -> Optional[float]:
        denom = self.sampled - self.errored
        if denom <= 0:
            return None
        return self.likely_fp / denom

    @property
    def failed(self) -> bool:
        """True if FP rate is above the FAIL threshold (plan D-4.5)."""
        rate = self.fp_rate
        return rate is not None and rate > FP_RATE_FAIL_THRESHOLD


@dataclass
class FPEstimationResult:
    by_class: dict[str, ClassFPStats] = field(default_factory=dict)
    total_sampled: int = 0
    total_likely_fp: int = 0
    total_errored: int = 0
    sample_size_per_class: int = 0       # the N used (echoed for the report)
    duration_s: float = 0.0

    def summary_lines(self) -> list[str]:
        lines = [
            "── FP Estimation Report (Slither disagreement) ─────",
            f"  Sample size per class: {self.sample_size_per_class}",
            f"  Total sampled:         {self.total_sampled}",
            f"  Total likely FP:       {self.total_likely_fp}",
            f"  Total errored:         {self.total_errored}",
            f"  FAIL threshold:        >{FP_RATE_FAIL_THRESHOLD:.0%}",
            "",
            "  Per-class FP rate (with per-stratum breakdown):",
        ]
        for cls in class_names():
            s = self.by_class.get(cls)
            if s is None or s.sampled == 0:
                continue
            rate = f"{s.fp_rate:.1%}" if s.fp_rate is not None else "—"
            fail = "  ✗ FAIL" if s.failed else ""
            lines.append(
                f"    {cls:<30} {rate:>7}  "
                f"({s.likely_fp}/{s.sampled - s.errored} likely FP, "
                f"{s.errored} err){fail}"
            )
            for (src, tier), st in sorted(s.strata.items()):
                st_rate = f"{st.fp_rate:.1%}" if st.fp_rate is not None else "—"
                lines.append(
                    f"        · {src:<15} {tier:<5}  {st_rate:>7}  "
                    f"({st.likely_fp}/{st.sampled - st.errored} likely FP)"
                )
        lines.append("─────────────────────────────────────────────────")
        return lines

    def __str__(self) -> str:
        return "\n".join(self.summary_lines())


def _stratified_sample(
    positives: list[tuple[str, dict]],  # (sha, merged_label) for the class
    n: int,
    seed: int,
) -> list[tuple[str, dict]]:
    """Stratified sample of N from the positive set, one per (source, tier) cell.

    Proportional allocation: each (source, tier) cell contributes
    `n * (cell_size / total)` (rounded), with the remainder distributed to
    the largest cells (deterministic). If n exceeds the number of strata,
    we sample more than 1 per stratum; if a stratum is exhausted, we
    sample with replacement from the rest of the corpus.
    """
    if not positives or n <= 0:
        return []

    rng = random.Random(seed)

    # Group by (source, tier)
    cells: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    for sha, lj in positives:
        cls_entry = lj.get("classes", {})
        # Find the source from the merged-level 'sources' list (Stage 3
        # sets this to the source that contributed the positive for this class)
        sources = lj.get("sources") or [None]
        source = sources[0] or "unknown"
        # Find the tier: prefer the per-class entry's tier, fall back to None
        # Iterate classes to find which one is positive
        tier = None
        for c_name, entry in cls_entry.items():
            if entry.get("value") == 1 and c_name in lj.get("classes", {}):
                tier = entry.get("tier") or "none"
                break
        cells[(source, tier)].append((sha, lj))

    total = sum(len(v) for v in cells.values())
    if total == 0:
        return []

    # Proportional allocation with deterministic remainder distribution
    raw_alloc: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    allocated = 0
    for key, items in cells.items():
        share = n * len(items) / total
        whole = int(share)
        raw_alloc[key] = whole
        allocated += whole
        remainders.append((share - whole, key))
    # Distribute the leftover (n - allocated) to the highest remainders
    remainders.sort(reverse=True)
    i = 0
    while allocated < n and remainders:
        frac, key = remainders[i % len(remainders)]
        raw_alloc[key] += 1
        allocated += 1
        i += 1

    # Sample (without replacement) from each cell up to its allocation
    sampled: list[tuple[str, dict]] = []
    for key, items in cells.items():
        items_copy = list(items)            # don't mutate the cell list
        rng.shuffle(items_copy)
        alloc = min(raw_alloc[key], len(items_copy))
        sampled.extend(items_copy[:alloc])

    # If any cell was underfilled (because len(items) < alloc), top up
    # from the global pool with a stable deterministic order
    if len(sampled) < n:
        sampled_shas = {s for s, _ in sampled}
        # Build the global pool from the cells (already grouped), not from
        # `positives` — that way we don't depend on its order.
        pool: list[tuple[str, dict]] = []
        for items in cells.values():
            pool.extend(items)
        rng.shuffle(pool)
        for s, lj in pool:
            if len(sampled) >= n:
                break
            if s not in sampled_shas:
                sampled.append((s, lj))
                sampled_shas.add(s)

    return sampled[:n]


def run_fp_estimation(
    data_dir: Path,
    *,
    sample_size: int = 50,
    seed: int = 42,
    only_classes: Optional[list[str]] = None,
    force: bool = False,
) -> FPEstimationResult:
    """Run stratified FP estimation across all classes.

    For each class, samples `sample_size` positives (proportional to
    (source, tier) cell size), runs Slither on each, and reports the
    per-class empirical FP rate (Slither disagreement rate).

    Args:
        data_dir: Path to data/ directory.
        sample_size: Per-class sample size (default 50, per plan D-4.4).
        seed: RNG seed for reproducibility.
        only_classes: Restrict to these classes (default: all 10).
        force: Re-run Slither even if cache hit exists.

    Returns:
        FPEstimationResult with per-class and per-stratum stats.
    """
    merged_dir = data_dir / "labels" / "merged"
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged labels dir not found: {merged_dir}")

    classes = only_classes if only_classes else class_names()
    result = FPEstimationResult(
        by_class={c: ClassFPStats(class_name=c) for c in classes},
        sample_size_per_class=sample_size,
    )
    t0 = time.monotonic()

    # Load all merged labels once
    all_labels: list[dict] = []
    for lf in merged_dir.glob("*.labels.json"):
        try:
            all_labels.append(json.loads(lf.read_text()))
        except (json.JSONDecodeError, OSError):
            continue

    for cls in classes:
        if not CLASS_TO_DETECTORS.get(cls):
            # Class has no detector — Slither cannot corroborate; skip
            log.info(f"  {cls}: no Slither detector — FP estimation skipped")
            continue

        # Collect all positives for this class
        positives: list[tuple[str, dict]] = []
        for lj in all_labels:
            entry = lj.get("classes", {}).get(cls, {})
            if entry.get("value") == 1:
                positives.append((lj["sha256"], lj))

        stats = result.by_class[cls]
        stats.total_positives_in_corpus = len(positives)
        if not positives:
            log.info(f"  {cls}: no positives in corpus")
            continue

        # Sample
        sample = _stratified_sample(positives, sample_size, seed + hash(cls) % 10000)
        stats.sampled = len(sample)

        # Run Slither on each sampled positive
        for sha, lj in sample:
            sources = lj.get("sources") or [None]
            source = sources[0]
            # Get tier for this contract's positive class
            entry = lj.get("classes", {}).get(cls, {})
            tier = entry.get("tier") or "none"

            if source is None:
                stats.errored += 1
                result.total_errored += 1
                continue

            findings = run_on_contract(
                sha, source, data_dir, force=force,
                detectors=CLASS_TO_DETECTORS[cls],
            )
            if findings is None or findings.error:
                stats.errored += 1
                result.total_errored += 1
                continue

            # Decision: likely FP = Slither did not agree
            likely_fp = not findings.agrees_with_class(cls)
            stats.likely_fp += int(likely_fp)
            result.total_likely_fp += int(likely_fp)

            # Per-stratum update
            stratum_key = (source, tier)
            if stratum_key not in stats.strata:
                stats.strata[stratum_key] = StratumStats(
                    class_name=cls, source=source, tier=tier,
                )
            st = stats.strata[stratum_key]
            st.sampled += 1
            st.likely_fp += int(likely_fp)

        result.total_sampled += stats.sampled
        rate = f"{stats.fp_rate:.1%}" if stats.fp_rate is not None else "—"
        log.info(f"  {cls}: FP rate {rate}  ({stats.likely_fp}/{stats.sampled} sampled)")

    result.duration_s = time.monotonic() - t0
    return result
