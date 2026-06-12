"""Verification gate — Stage 4.

Produces per-class VERIFIED / PROVISIONAL / BEST-EFFORT / FAIL verdicts
from class_auditor, semantic_checker, and (optionally) tool_validator,
fp_estimator, and negative_checker outputs.

Gate rules (D-4.5):
  VERIFIED      — semantic pass_rate > 90%  AND no co-occurrence flag
  PROVISIONAL   — semantic pass_rate 60–90% OR no graph reps available (T0/T1 tier sources only)
  BEST-EFFORT   — semantic pass_rate 30–60% OR NOT_EXTRACTABLE class with T2+ source
  FAIL          — semantic pass_rate < 30%
                  OR co-occurrence flag on high-noise source
                  OR fp_rate > 30% (from fp_estimator, when provided)
                  OR (tool_agreement < 30% AND co-flag, from tool_validator)

For T0 (injection-verified, SolidiFI) with no semantic failures:
  → VERIFIED regardless of rep coverage (ground truth by construction)

For T2 (curated, DIVE) with no semantic checks (no reps):
  → PROVISIONAL (trusted curation, unverified at AST level)

Hard gate: any class with FAIL blocks downstream export.
Soft gate: PROVISIONAL / BEST-EFFORT export with warning.

The negative_checker result (when provided) is a corpus-level signal —
its status (OK / WARN / FAIL) is added to the gate result. A FAIL adds
a special entry to `hard_fails` (keyed as `__neg_check__`) which blocks
export.
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
    SemanticCheckResult, CheckVerdict, ClassCheckSummary,
)
from sentinel_data.verification.tool_validator import ToolValidationResult
from sentinel_data.verification.fp_estimator import (
    FPEstimationResult, FP_RATE_FAIL_THRESHOLD,
)
from sentinel_data.verification.negative_checker import NonVulnResult

log = logging.getLogger("sentinel_data.verification.gate")

# Tier confidence order: lower index = higher confidence
_TIER_RANK = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
_HIGH_CONFIDENCE_TIERS = {"T0", "T1"}

# Tool agreement below this AND co-flag → downgrade to PROVISIONAL.
# (signals the labels may be noise even if semantic pass rate is high)
TOOL_AGREEMENT_DOWNGRADE_THRESHOLD = 0.30


class Verdict(str, Enum):
    """Per-class verification verdict from the gate logic."""

    VERIFIED = "VERIFIED"
    PROVISIONAL = "PROVISIONAL"
    BEST_EFFORT = "BEST-EFFORT"
    FAIL = "FAIL"


@dataclass
class ClassVerdict:
    """Gate verdict for a single vulnerability class, with supporting evidence."""

    class_name: str
    verdict: Verdict
    reason: str
    semantic_pass_rate: Optional[float] = None
    semantic_coverage: float = 0.0
    co_occurrence_flagged: bool = False
    highest_tier: Optional[str] = None
    tool_agreement: Optional[float] = None  # from tool_validator (if provided)
    fp_rate: Optional[float] = None         # from fp_estimator (if provided)


@dataclass
class GateResult:
    """Top-level result container for the verification gate run."""

    verdicts: dict[str, ClassVerdict] = field(default_factory=dict)
    hard_fails: list[str] = field(default_factory=list)
    negative_check_status: Optional[str] = None  # OK / WARN / FAIL / None (not run)

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
                tool = f"  tool={v.tool_agreement:.0%}" if v.tool_agreement is not None else ""
                fp = f"  fp={v.fp_rate:.0%}" if v.fp_rate is not None else ""
                lines.append(
                    f"  {icon} {cls:<30} {v.verdict.value:<14}"
                    f"{sem}{cov}{tier}{flag}{tool}{fp}"
                )
        if self.negative_check_status:
            lines.append(f"\n  NonVulnerable checker: {self.negative_check_status}")
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
    *,
    tool_validation: Optional[ToolValidationResult] = None,
    fp_estimation: Optional[FPEstimationResult] = None,
    negative_check: Optional[NonVulnResult] = None,
) -> GateResult:
    """Compute per-class verification verdict.

    Args:
        audit: Output from class_auditor.run_audit().
        semantic: Output from semantic_checker.run_semantic_check().
        tool_validation: Optional — if provided, gates use the per-class
                         Slither agreement rate as a corroboration signal.
                         A class with very low agreement AND a co-occurrence
                         flag is downgraded.
        fp_estimation: Optional — if provided, classes with fp_rate above
                       FP_RATE_FAIL_THRESHOLD (30%) are FAIL.
        negative_check: Optional — if provided, the corpus-level status
                        (OK / WARN / FAIL) is added to the gate result.
                        A FAIL adds a special `__neg_check__` entry to
                        hard_fails (which blocks export).

    Returns:
        GateResult with per-class verdicts and hard-fail list.
    """
    result = GateResult()

    flagged_classes = {p.class_a for p in audit.flagged_pairs} | {p.class_b for p in audit.flagged_pairs}

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

        highest_tier: Optional[str] = None
        for tier in ("T0", "T1", "T2", "T3", "T4"):
            if cls_stats.by_tier.get(tier, 0) > 0:
                highest_tier = tier
                break

        checkable = cls_sem.pass_count + cls_sem.fail_count
        pass_rate: Optional[float] = (cls_sem.pass_count / checkable) if checkable > 0 else None
        coverage = cls_sem.coverage

        # Optional inputs (None if not provided)
        tool_agree: Optional[float] = None
        if tool_validation is not None:
            cls_tool = tool_validation.by_class.get(cls)
            if cls_tool is not None:
                tool_agree = cls_tool.agreement_rate

        fp_rate: Optional[float] = None
        if fp_estimation is not None:
            cls_fp = fp_estimation.by_class.get(cls)
            if cls_fp is not None and cls_fp.fp_rate is not None:
                fp_rate = cls_fp.fp_rate

        # --- Gate logic ---

        # FP rate > threshold → FAIL (independent of semantic pass rate)
        if fp_rate is not None and fp_rate > FP_RATE_FAIL_THRESHOLD:
            verdict_val = Verdict.FAIL
            reason = f"FP rate {fp_rate:.0%} > {FP_RATE_FAIL_THRESHOLD:.0%} (empirical)"

        # T0 (injection-verified) with no semantic failures → VERIFIED
        elif highest_tier == "T0" and cls_sem.fail_count == 0 and not co_flag:
            verdict_val = Verdict.VERIFIED
            reason = "T0 injection-verified; no semantic failures"

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

        # No graph reps → tier-based verdict
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

        # Pass rate > 90% → VERIFIED (unless co-occurrence flag or low tool agreement)
        else:
            verdict_val = Verdict.VERIFIED if not co_flag else Verdict.PROVISIONAL
            reason = f"semantic pass rate {pass_rate:.0%}"
            if co_flag:
                reason += "; co-occurrence flag → downgraded to PROVISIONAL"
            if (tool_agree is not None
                    and tool_agree < TOOL_AGREEMENT_DOWNGRADE_THRESHOLD
                    and co_flag):
                verdict_val = Verdict.PROVISIONAL
                reason += f"; tool agreement {tool_agree:.0%} (downgraded with co-flag)"

        verdict = ClassVerdict(
            class_name=cls,
            verdict=verdict_val,
            reason=reason,
            semantic_pass_rate=pass_rate,
            semantic_coverage=coverage,
            co_occurrence_flagged=co_flag,
            highest_tier=highest_tier,
            tool_agreement=tool_agree,
            fp_rate=fp_rate,
        )
        result.verdicts[cls] = verdict
        log.info(f"  {cls}: {verdict_val.value} — {reason}")

    result.hard_fails = [cls for cls, v in result.verdicts.items() if v.verdict == Verdict.FAIL]

    if negative_check is not None:
        result.negative_check_status = negative_check.status
        if negative_check.status == "FAIL":
            result.hard_fails.append("__neg_check__")
            log.warning(
                f"  NonVulnerable checker FAIL "
                f"(hit rate {negative_check.hit_rate:.1%}); export blocked"
            )

    return result
