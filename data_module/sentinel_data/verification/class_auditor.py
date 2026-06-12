"""Class auditor — Stage 4 Task 4.A.

Reads merged .labels.json files and produces:
  1. Per-class positive counts (with per-source and per-tier breakdowns)
  2. 10×10 co-occurrence matrix (conditional probabilities)
  3. Co-occurrence flags for pairs that exceed the noise threshold

The co-occurrence matrix is the primary artifact that catches the BCCC
99% DoS↔Reentrancy co-occurrence pattern automatically.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentinel_data.labeling.schema import class_names

log = logging.getLogger("sentinel_data.verification.class_auditor")

# Conditional co-occurrence rate above which a pair is flagged as suspicious.
# BCCC had 99% DoS+Reentrancy; 50% is a very conservative threshold.
CO_OCCUR_FLAG_THRESHOLD = 0.50

_CLASSES = class_names()
_N = len(_CLASSES)
_IDX = {c: i for i, c in enumerate(_CLASSES)}


@dataclass
class ClassStats:
    """Per-class aggregate stats."""
    class_name: str
    total_positives: int = 0
    total_contracts: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    by_tier: dict[str, int] = field(default_factory=dict)

    @property
    def prevalence(self) -> float:
        return self.total_positives / self.total_contracts if self.total_contracts else 0.0


@dataclass
class CoOccurrencePair:
    class_a: str
    class_b: str
    count: int           # contracts where both are positive
    count_a: int         # contracts where class_a is positive
    rate: float          # count / count_a  (P(b=1 | a=1))
    flagged: bool        # rate > CO_OCCUR_FLAG_THRESHOLD


@dataclass
class AuditResult:
    per_class: dict[str, ClassStats] = field(default_factory=dict)
    co_occurrence: list[CoOccurrencePair] = field(default_factory=list)
    flagged_pairs: list[CoOccurrencePair] = field(default_factory=list)
    total_contracts: int = 0
    duration_s: float = 0.0

    def summary_lines(self) -> list[str]:
        lines = [
            "── Class Audit Report ─────────────────────────────",
            f"  Total contracts: {self.total_contracts}",
            "",
            "  Class breakdown (positives / total, prevalence):",
        ]
        for cls in _CLASSES:
            s = self.per_class.get(cls)
            if s:
                src_str = ", ".join(f"{k}:{v}" for k, v in sorted(s.by_source.items()))
                lines.append(
                    f"    {cls:<30} {s.total_positives:>6} / {s.total_contracts}"
                    f"  ({s.prevalence:.1%})  [{src_str}]"
                )
        if self.flagged_pairs:
            lines += ["", "  ⚠ Co-occurrence flags (P(B|A) > {:.0%}):".format(CO_OCCUR_FLAG_THRESHOLD)]
            for p in self.flagged_pairs:
                lines.append(
                    f"    P({p.class_b}=1 | {p.class_a}=1) = "
                    f"{p.rate:.1%}  ({p.count}/{p.count_a})"
                )
        else:
            lines += ["", "  Co-occurrence: no suspicious pairs found."]
        lines.append("─────────────────────────────────────────────────")
        return lines

    def __str__(self) -> str:
        return "\n".join(self.summary_lines())


def run_audit(data_dir: Path) -> AuditResult:
    """Read merged labels and produce class audit report.

    Args:
        data_dir: Path to data/ directory (contains labels/merged/).

    Returns:
        AuditResult with per-class stats and co-occurrence matrix.
    """
    merged_dir = data_dir / "labels" / "merged"
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged labels dir not found: {merged_dir}")

    t0 = time.monotonic()

    # Accumulate: per-class positive count, by-source, by-tier
    per_class: dict[str, ClassStats] = {c: ClassStats(class_name=c) for c in _CLASSES}

    # Co-occurrence: count_pos[i][j] = contracts where class_i AND class_j are positive
    count_pos = [[0] * _N for _ in range(_N)]
    # count_any[i] = contracts where class_i is positive
    count_any = [0] * _N

    total = 0
    for lf in merged_dir.glob("*.labels.json"):
        try:
            lj: dict[str, Any] = json.loads(lf.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Cannot read {lf}: {e}")
            continue
        total += 1
        classes = lj.get("classes", {})

        # Collect which classes are positive for this contract
        positive_indices: list[int] = []
        for cls in _CLASSES:
            stats = per_class[cls]
            stats.total_contracts += 1
            entry = classes.get(cls, {})
            val = entry.get("value", 0)
            if val == 1:
                stats.total_positives += 1
                src = entry.get("source") or "unknown"
                stats.by_source[src] = stats.by_source.get(src, 0) + 1
                tier = entry.get("tier") or "none"
                stats.by_tier[tier] = stats.by_tier.get(tier, 0) + 1
                positive_indices.append(_IDX[cls])

        # Update co-occurrence counts
        for i in positive_indices:
            count_any[i] += 1
        for i in positive_indices:
            for j in positive_indices:
                count_pos[i][j] += 1

    # Build co-occurrence pairs (upper triangle + flagged diagonal cross)
    co_occurrence: list[CoOccurrencePair] = []
    flagged: list[CoOccurrencePair] = []
    for i in range(_N):
        if count_any[i] == 0:
            continue
        for j in range(_N):
            if i == j:
                continue
            if count_pos[i][j] == 0:
                continue
            rate = count_pos[i][j] / count_any[i]
            pair = CoOccurrencePair(
                class_a=_CLASSES[i],
                class_b=_CLASSES[j],
                count=count_pos[i][j],
                count_a=count_any[i],
                rate=rate,
                flagged=rate > CO_OCCUR_FLAG_THRESHOLD,
            )
            co_occurrence.append(pair)
            if pair.flagged:
                flagged.append(pair)
                log.info(
                    f"Co-occurrence flag: P({_CLASSES[j]}=1 | {_CLASSES[i]}=1) = "
                    f"{rate:.1%} ({count_pos[i][j]}/{count_any[i]})"
                )

    return AuditResult(
        per_class=per_class,
        co_occurrence=co_occurrence,
        flagged_pairs=flagged,
        total_contracts=total,
        duration_s=time.monotonic() - t0,
    )
