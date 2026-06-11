"""Verification report generator — Stage 4.

Produces a human-readable verification_report.md from the outputs of
class_auditor, semantic_checker, and gate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.class_auditor import AuditResult
from sentinel_data.verification.gate import GateResult, Verdict
from sentinel_data.verification.semantic_checker import SemanticCheckResult

log = logging.getLogger("sentinel_data.verification.report_generator")

_VERDICT_ICON = {
    Verdict.VERIFIED: "✅",
    Verdict.PROVISIONAL: "⚠️",
    Verdict.BEST_EFFORT: "🔶",
    Verdict.FAIL: "❌",
}


@dataclass
class VerificationReport:
    audit: AuditResult
    semantic: SemanticCheckResult
    gate: GateResult
    corpus_tag: str = ""    # e.g. "SolidiFI+DIVE-2026-06-11"

    def to_markdown(self) -> str:
        lines = [
            "# SENTINEL v2 Verification Report",
            "",
            f"**Corpus:** {self.corpus_tag or 'unknown'}  ",
            f"**Total contracts:** {self.audit.total_contracts}  ",
            f"**Gate:** {'PASS ✓' if self.gate.gate_passed else 'FAIL ✗'}",
            "",
            "---",
            "",
            "## 1. Per-Class Gate",
            "",
            "| Class | Verdict | Semantic pass rate | Coverage | Best tier | Co-occur flag |",
            "|---|---|---|---|---|---|",
        ]
        for cls in class_names():
            v = self.gate.verdicts.get(cls)
            if v is None:
                continue
            icon = _VERDICT_ICON.get(v.verdict, "?")
            sem = f"{v.semantic_pass_rate:.0%}" if v.semantic_pass_rate is not None else "—"
            cov = f"{v.semantic_coverage:.0%}"
            tier = v.highest_tier or "—"
            flag = "⚠ yes" if v.co_occurrence_flagged else "no"
            lines.append(
                f"| {cls} | {icon} {v.verdict.value} | {sem} | {cov} | {tier} | {flag} |"
            )

        lines += [
            "",
            "---",
            "",
            "## 2. Per-Class Corpus Stats",
            "",
            "| Class | Positives | Prevalence | By source | By tier |",
            "|---|---|---|---|---|",
        ]
        for cls in class_names():
            s = self.audit.per_class.get(cls)
            if s is None:
                continue
            src_str = ", ".join(f"{k}:{v}" for k, v in sorted(s.by_source.items()))
            tier_str = ", ".join(f"{k}:{v}" for k, v in sorted(s.by_tier.items()))
            lines.append(
                f"| {cls} | {s.total_positives} | {s.prevalence:.1%} "
                f"| {src_str or '—'} | {tier_str or '—'} |"
            )

        lines += [
            "",
            "---",
            "",
            "## 3. Co-occurrence Matrix (flagged pairs)",
            "",
        ]
        if self.audit.flagged_pairs:
            lines += [
                "| P(B=1 | A=1) | Class A | Class B | Rate | Count |",
                "|---|---|---|---|---|",
            ]
            for p in sorted(self.audit.flagged_pairs, key=lambda x: -x.rate):
                lines.append(
                    f"| ⚠ | {p.class_a} | {p.class_b} | {p.rate:.1%} | {p.count}/{p.count_a} |"
                )
        else:
            lines.append("_No flagged co-occurrence pairs (all below threshold)._")

        lines += [
            "",
            "---",
            "",
            "## 4. Semantic Check Summary",
            "",
            "| Class | Pass | Fail | Skip | Not extractable | Coverage |",
            "|---|---|---|---|---|---|",
        ]
        for cls in class_names():
            sem = self.semantic.by_class.get(cls)
            if sem is None:
                continue
            total = sem.pass_count + sem.fail_count + sem.positives_skipped + sem.not_extractable
            lines.append(
                f"| {cls} | {sem.pass_count} | {sem.fail_count} | {sem.positives_skipped} "
                f"| {sem.not_extractable} | {sem.coverage:.0%} |"
            )

        lines += [
            "",
            "---",
            "",
            "## 5. Hard Failures",
            "",
        ]
        if self.gate.hard_fails:
            lines += [f"- **{cls}** is FAIL — export blocked" for cls in self.gate.hard_fails]
        else:
            lines.append("_No hard failures. All classes export-safe._")

        lines += [
            "",
            "---",
            "",
            "## 6. Known Limitations",
            "",
            "- **Stage 2 representation coverage**: Full DIVE corpus (22,073 contracts) has not yet",
            "  had Stage 2 graph extraction run. Semantic checks for DIVE are marked SKIP until",
            "  `sentinel-data represent --source dive` is run on the full corpus.",
            "- **NOT_EXTRACTABLE classes**: DoS, GasException, TransactionOrderDependence have no",
            "  dedicated feature in the v9 graph schema. Full verification requires Slither-based",
            "  AST analysis (deferred to Stage 4 v2 with tool_validator).",
            "- **tool_validator, fp_estimator, negative_checker**: deferred; require Slither batch",
            "  runs on the full corpus.",
        ]
        return "\n".join(lines)

    def write(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown())
        log.info(f"Wrote verification report to {output_path}")


def generate_report(
    audit: AuditResult,
    semantic: SemanticCheckResult,
    gate: GateResult,
    *,
    corpus_tag: str = "",
    output_path: Path | None = None,
) -> VerificationReport:
    """Build and optionally write a VerificationReport.

    Args:
        audit: Output from class_auditor.run_audit().
        semantic: Output from semantic_checker.run_semantic_check().
        gate: Output from gate.run_gate().
        corpus_tag: Human-readable corpus identifier for the report header.
        output_path: If provided, write report.md here.

    Returns:
        VerificationReport with .to_markdown() method.
    """
    report = VerificationReport(
        audit=audit,
        semantic=semantic,
        gate=gate,
        corpus_tag=corpus_tag,
    )
    if output_path:
        report.write(output_path)
    return report
