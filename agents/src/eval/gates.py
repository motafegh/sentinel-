"""
gates.py — The 9 workstream gate assertions (WS1a/b/c/d/e, WS2, WS3, D4, macro_f1).

Extracted from scripts/eval_benchmark.py (P0.1 T0.1.2, 2026-06-23) so the
module runner and other callers can import them directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ContractEval:
    """One contract's evaluation row (used by all gate functions)."""
    stem: str
    report_path: str
    sidecar_path: str | None
    labels: list[str]
    ground_truth: str
    verdicts: dict[str, str]
    probabilities: dict[str, float]
    quick_screen_hits: dict[str, list]
    static_findings_count: int
    rag_results_count: int
    path_taken: str
    overall_label: str | None
    overall_verdict: str | None
    narrative: str | None
    error: str | None
    consensus_verdict: dict[str, dict]
    vulnerability_verdicts_classes: set
    eye_predictions: dict | None = None
    predicted_positive_classes: list[str] = field(default_factory=list)
    true_positive_classes: list[str] = field(default_factory=list)
    false_positive_classes: list[str] = field(default_factory=list)
    false_negative_classes: list[str] = field(default_factory=list)
    contract_correct: bool = False
    contract_exact: bool = False


@dataclass
class GateResult:
    gate_id: str
    description: str
    passed: bool
    detail: str
    value: Any = None


def _positive_classes(verdicts: dict[str, str], positive_set: set[str]) -> list[str]:
    return [c for c, v in verdicts.items() if v in positive_set]


# ---------------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------------


def gate_ws1a_silent_safe_on_flagged(rows: list[ContractEval]) -> GateResult:
    """No consensus-flagged class ends in verdict SAFE when consensus voted non-SAFE."""
    violations: list[str] = []
    for row in rows:
        for cls, cv in row.consensus_verdict.items():
            cv_verdict = cv.get("verdict", "")
            final_verdict = row.verdicts.get(cls, "MISSING")
            if final_verdict == "SAFE" and cv_verdict != "SAFE":
                violations.append(
                    f"{row.stem}/{cls} (consensus={cv_verdict} conf={cv.get('confidence', 0):.2f} \u2192 final=SAFE)"
                )
    passed = len(violations) == 0
    return GateResult(
        gate_id="WS1a_silent_safe_on_flagged",
        description="No consensus-flagged class ends SAFE (debate cannot clear consensus verdict)",
        passed=passed,
        detail=f"{len(violations)} violation(s)" + (f": {'; '.join(violations[:5])}" if violations else ""),
        value=len(violations),
    )


def gate_ws1b_inconclusive_on_timeout(rows: list[ContractEval]) -> GateResult:
    """edge_debate_timeout must emit INCONCLUSIVE (LLM-on mode only)."""
    edge = next((r for r in rows if r.stem == "edge_debate_timeout"), None)
    has_inconclusive = any(v == "INCONCLUSIVE" for v in (edge.verdicts.values() if edge else []))
    has_llm = any(r.narrative for r in rows)
    if not has_llm:
        return GateResult(
            gate_id="WS1b_inconclusive_on_timeout",
            description="edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only)",
            passed=True, detail="N/A in --no-llm mode", value="skipped",
        )
    passed = edge is not None and has_inconclusive
    return GateResult(
        gate_id="WS1b_inconclusive_on_timeout",
        description="edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only)",
        passed=passed,
        detail=f"edge_debate_timeout present={edge is not None}, INCONCLUSIVE emitted={has_inconclusive}",
        value=passed,
    )


def gate_ws1c_no_missing_consensus_votes(rows: list[ContractEval]) -> GateResult:
    """Every consensus_engine vote must appear in final vulnerability_verdicts."""
    missing: list[str] = []
    for row in rows:
        for cls in row.consensus_verdict:
            if cls not in row.vulnerability_verdicts_classes:
                missing.append(f"{row.stem}/{cls}")
    passed = len(missing) == 0
    return GateResult(
        gate_id="WS1c_no_missing_consensus_votes",
        description="No consensus vote is missing from final verdicts (Finding #15)",
        passed=passed,
        detail=f"{len(missing)} missing vote(s)" + (f": {'; '.join(missing[:5])}" if missing else ""),
        value=len(missing),
    )


def gate_ws1d_confidence_1_0_not_downgraded(rows: list[ContractEval]) -> GateResult:
    """No class with consensus confidence=1.0 ends in SAFE."""
    violations: list[str] = []
    for row in rows:
        for cls, cv in row.consensus_verdict.items():
            if cv.get("confidence", 0) >= 1.0 and row.verdicts.get(cls) == "SAFE":
                violations.append(f"{row.stem}/{cls} (conf=1.0 \u2192 SAFE)")
    passed = len(violations) == 0
    return GateResult(
        gate_id="WS1d_confidence_1_0_not_downgraded",
        description="No consensus confidence=1.0 class ends in SAFE (Finding #14 worst case)",
        passed=passed,
        detail=f"{len(violations)} violation(s)" + (f": {'; '.join(violations[:5])}" if violations else ""),
        value=len(violations),
    )


def gate_ws1e_no_vulnerable_label_with_safe_verdict(rows: list[ContractEval]) -> GateResult:
    """No contract with overall_label=vulnerable has overall_verdict=SAFE."""
    violations: list[str] = []
    for row in rows:
        if row.overall_label in ("confirmed_vulnerable", "vulnerable") and row.overall_verdict == "SAFE":
            violations.append(f"{row.stem} (label={row.overall_label}, verdict={row.overall_verdict})")
    passed = len(violations) == 0
    return GateResult(
        gate_id="WS1e_no_vulnerable_label_with_safe_verdict",
        description="No contract has overall_label=vulnerable + overall_verdict=SAFE (Finding #19)",
        passed=passed,
        detail=f"{len(violations)} violation(s)" + (f": {'; '.join(violations[:5])}" if violations else ""),
        value=len(violations),
    )


def gate_ws2_false_positives_on_safe(rows: list[ContractEval], positive_set: set[str]) -> GateResult:
    """Zero false-positive verdicts on the safe subset."""
    safe_rows = [r for r in rows if r.ground_truth == "safe"]
    fps: list[str] = []
    for row in safe_rows:
        flagged = _positive_classes(row.verdicts, positive_set)
        if flagged:
            fps.append(f"{row.stem}: {flagged}")
    passed = len(fps) == 0
    return GateResult(
        gate_id="WS2_false_positives_on_safe",
        description="Zero false-positive verdicts on the safe subset",
        passed=passed,
        detail=f"{len(fps)} FP(s) on {len(safe_rows)} safe contract(s)" + (f": {'; '.join(fps[:5])}" if fps else ""),
        value=len(fps),
    )


def gate_ws3_long_contract_bug_detected(rows: list[ContractEval], positive_set: set[str]) -> GateResult:
    """edge_long_contract_truncated Reentrancy must be flagged (bug past 2000-char cutoff)."""
    edge = next((r for r in rows if r.stem == "edge_long_contract_truncated"), None)
    if edge is None:
        return GateResult(
            gate_id="WS3_long_contract_bug_detected",
            description="edge_long_contract_truncated Reentrancy flagged (bug is past 2000-char cutoff)",
            passed=False, detail="edge_long_contract_truncated not in corpus", value="missing",
        )
    re_verdict = edge.verdicts.get("Reentrancy", "MISSING")
    passed = re_verdict in positive_set
    return GateResult(
        gate_id="WS3_long_contract_bug_detected",
        description="edge_long_contract_truncated Reentrancy flagged (bug is past 2000-char cutoff)",
        passed=passed,
        detail=f"Reentrancy verdict = {re_verdict} (positive set = {sorted(positive_set)})",
        value=re_verdict,
    )


def gate_d4_eye_predictions_present(rows: list[ContractEval]) -> GateResult:
    """Every contract's ml_result should include eye_predictions."""
    present = sum(1 for r in rows if r.eye_predictions)
    passed = present == len(rows)
    return GateResult(
        gate_id="D4_eye_predictions_present",
        description="eye_predictions field present in all ml_results",
        passed=passed,
        detail=f"eye_predictions present in {present}/{len(rows)} reports",
        value=present,
    )


def gate_macro_f1_vs_baseline(macro_f1: float, baseline: dict | None) -> GateResult:
    """macro_F1 must not drop vs the stored baseline."""
    if baseline is None:
        return GateResult(
            gate_id="macro_f1_vs_baseline",
            description="macro_F1 >= baseline (no baseline given — informational)",
            passed=True, detail=f"macro_F1 = {macro_f1:.4f} (no baseline to compare)", value=macro_f1,
        )
    base_f1 = float(baseline.get("macro_f1", 0.0))
    passed = macro_f1 >= base_f1
    delta = macro_f1 - base_f1
    return GateResult(
        gate_id="macro_f1_vs_baseline",
        description=f"macro_F1 >= baseline ({base_f1:.4f})",
        passed=passed,
        detail=f"macro_F1 = {macro_f1:.4f} (delta {delta:+.4f} vs baseline {base_f1:.4f})",
        value=macro_f1,
    )
