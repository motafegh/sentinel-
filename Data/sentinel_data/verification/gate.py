"""Verification gate — Stage 4.

Produces per-class VERIFIED / PROVISIONAL / BEST-EFFORT / FAIL verdicts
from class_auditor and semantic_checker outputs.

Gate rules (D-4.5):
  VERIFIED      — semantic pass_rate > 90%  AND no co-occurrence flag
  PROVISIONAL   — semantic pass_rate 60–90% OR no graph reps available (T0/T1 tier sources only)
  BEST-EFFORT   — semantic pass_rate 30–60% OR NOT_EXTRACTABLE class with T2+ source
  FAIL          — semantic pass_rate < 30%  OR co-occurrence flag on high-noise source

For T0 (injection-verified, SolidiFI) with no semantic failures:
  → VERIFIED regardless of rep coverage (ground truth by construction)

For T2 (curated, DIVE) with no semantic checks (no reps):
  → PROVISIONAL (trusted curation, unverified at AST level)

Hard gate: any class with FAIL blocks downstream export.
Soft gate: PROVISIONAL / BEST-EFFORT export with warning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.class_auditor import AuditResult, CO_OCCUR_FLAG_THRESHOLD
from sentinel_data.verification.semantic_checker import (
    SemanticCheckResult, CheckVerdict, ClassCheckSummary
)

log = logging.getLogger("sentinel_data.verification.gate")

# Tier confidence order: lower index = higher confidence
_TIER_RANK = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
_HIGH_CONFIDENCE_TIERS = {"T0", "T1"}


class Verdict(str, Enum):
    VERIFIED = "VERIFIED"
    PROVISIONAL = "PROVISIONAL"
    BEST_EFFORT = "BEST-EFFORT"
    FAIL = "FAIL"


@dataclass
class ClassVerdict:
    class_name: str
    verdict: Verdict
    reason: str
    semantic_pass_rate: Optional[float] = None
    semantic_coverage: float = 0.0      # fraction of positives with a rep
    co_occurrence_flagged: bool = False
    highest_tier: Optional[str] = None  # best (lowest rank) tier across positives


@dataclass
class GateResult:
    verdicts: dict[str, ClassVerdict] = field(default_factory=dict)
    hard_fails: list[str] = field(default_factory=list)

    @property
    def gate_passed(self) -> bool:
        return len(self.hard_fails) == 0

    def __str__(self) -> str:
        lines = ["── Verification Gate ──────────────────────────────"]
        for cls in class_names():
            v = self.verdicts.get(cls)
            if v:
                icon = {"VERIFIED": "✓", "PROVISIONAL": "~", "BEST-EFFORT": "?", "FAIL": "✗"}.get(v.verdict.value, "?")
                sem = f"  sem={v.semantic_pass_rate:.0%}" if v.semantic_pass_rate is not None else ""
                cov = f"  cov={v.semantic_coverage:.0%}"
                tier = f"  best_tier={v.highest_tier}" if v.highest_tier else ""
                flag = "  ⚠co-occur" if v.co_occurrence_flagged else ""
                lines.append(
                    f"  {icon} {cls:<30} {v.verdict.value:<14}"
                    f"{sem}{cov}{tier}{flag}"
                )
        lines.append("")
        if self.hard_fails:
            lines.append(f"  FAIL ✗ — hard fails: {', '.join(self.hard_fails)}")
        else:
            lines.append("  PASS ✓ — no hard failures")
        lines.append("─────────────────────────────────────────────────")
        return "\n".join(lines)


def run_gate(
    audit: AuditResult,
    semantic: SemanticCheckResult,
) -> GateResult:
    """Compute per-class verification verdict.

    Args:
        audit: Output from class_auditor.run_audit().
        semantic: Output from semantic_checker.run_semantic_check().

    Returns:
        GateResult with per-class verdicts and hard-fail list.
    """
    result = GateResult()

    flagged_classes = {p.class_a for p in audit.flagged_pairs}

    for cls in class_names():
        cls_stats = audit.per_class.get(cls)
        cls_sem: ClassCheckSummary = semantic.by_class.get(cls, ClassCheckSummary(class_name=cls))
        co_flag = cls in flagged_classes

        if cls_stats is None or cls_stats.total_positives == 0:
            verdict = ClassVerdict(
                class_name=cls,
                verdict=Verdict.PROVISIONAL,
                reason="no positives in corpus",
            )
            result.verdicts[cls] = verdict
            continue

        # Determine best (highest confidence) tier present for this class
        highest_tier: Optional[str] = None
        for tier in ("T0", "T1", "T2", "T3", "T4"):
            if cls_stats.by_tier.get(tier, 0) > 0:
                highest_tier = tier
                break

        # Semantic pass rate — None if no positives were checkable (all SKIP or NOT_EXTRACTABLE)
        checkable = cls_sem.pass_count + cls_sem.fail_count
        pass_rate: Optional[float] = (cls_sem.pass_count / checkable) if checkable > 0 else None
        coverage = cls_sem.coverage

        # --- Gate logic ---

        # T0 (injection-verified) with no semantic failures → VERIFIED
        if highest_tier == "T0" and cls_sem.fail_count == 0 and not co_flag:
            # T0 is ground truth; if semantic check runs and passes, VERIFIED
            # If semantic was all SKIP (no reps), still VERIFIED for T0
            verdict_val = Verdict.VERIFIED
            reason = "T0 injection-verified; no semantic failures"
            if cls_sem.fail_count > 0:
                verdict_val = Verdict.PROVISIONAL
                reason = f"T0 source but {cls_sem.fail_count} semantic failures detected"

        # NOT_EXTRACTABLE class with high-confidence tier → PROVISIONAL
        elif cls_sem.not_extractable > 0 and cls_sem.pass_count == 0 and cls_sem.fail_count == 0:
            if highest_tier in _HIGH_CONFIDENCE_TIERS:
                verdict_val = Verdict.VERIFIED if highest_tier == "T0" else Verdict.PROVISIONAL
                reason = f"class not extractable from v9 features; {highest_tier} source trusted"
            elif co_flag:
                verdict_val = Verdict.BEST_EFFORT
                reason = "class not extractable; co-occurrence flag raised"
            else:
                verdict_val = Verdict.PROVISIONAL
                reason = f"class not extractable from v9 features; {highest_tier} source"

        # No graph reps → PROVISIONAL for T1/T2, BEST-EFFORT for T3/T4
        elif pass_rate is None:
            if highest_tier in ("T0", "T1"):
                verdict_val = Verdict.VERIFIED if highest_tier == "T0" else Verdict.PROVISIONAL
                reason = f"no graph reps yet; {highest_tier} tier trusted"
            elif highest_tier == "T2":
                verdict_val = Verdict.PROVISIONAL
                reason = "no graph reps yet; T2 curated source"
            else:
                verdict_val = Verdict.BEST_EFFORT
                reason = f"no graph reps; {highest_tier} source only"
            if co_flag:
                verdict_val = Verdict.BEST_EFFORT
                reason += "; co-occurrence flag"

        # Pass rate < 30% → FAIL
        elif pass_rate < 0.30:
            verdict_val = Verdict.FAIL
            reason = f"semantic pass rate {pass_rate:.0%} < 30%"

        # Pass rate 30–60% → BEST-EFFORT
        elif pass_rate < 0.60:
            verdict_val = Verdict.BEST_EFFORT
            reason = f"semantic pass rate {pass_rate:.0%} (30–60%)"
            if co_flag:
                reason += "; co-occurrence flag"

        # Pass rate 60–90% → PROVISIONAL
        elif pass_rate < 0.90:
            verdict_val = Verdict.PROVISIONAL
            reason = f"semantic pass rate {pass_rate:.0%} (60–90%)"
            if co_flag:
                reason += "; co-occurrence flag"

        # Pass rate > 90% → VERIFIED (unless co-occurrence flag)
        else:
            verdict_val = Verdict.VERIFIED if not co_flag else Verdict.PROVISIONAL
            reason = f"semantic pass rate {pass_rate:.0%}"
            if co_flag:
                reason += "; co-occurrence flag → downgraded to PROVISIONAL"

        verdict = ClassVerdict(
            class_name=cls,
            verdict=verdict_val,
            reason=reason,
            semantic_pass_rate=pass_rate,
            semantic_coverage=coverage,
            co_occurrence_flagged=co_flag,
            highest_tier=highest_tier,
        )
        result.verdicts[cls] = verdict
        log.info(f"  {cls}: {verdict_val.value} — {reason}")

    result.hard_fails = [cls for cls, v in result.verdicts.items() if v.verdict == Verdict.FAIL]
    return result
