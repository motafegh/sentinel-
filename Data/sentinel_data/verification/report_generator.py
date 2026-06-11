"""Verification report generator — Stage 4.

Produces a human-readable verification_report.md from the outputs of
class_auditor, semantic_checker, tool_validator, fp_estimator,
negative_checker, and gate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sentinel_data.labeling.schema import class_names
from sentinel_data.verification.class_auditor import AuditResult
from sentinel_data.verification.fp_estimator import FPEstimationResult
from sentinel_data.verification.gate import GateResult, Verdict
from sentinel_data.verification.negative_checker import NonVulnResult
from sentinel_data.verification.semantic_checker import SemanticCheckResult
from sentinel_data.verification.tool_validator import ToolValidationResult

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
    tool_validation: Optional[ToolValidationResult] = None
    fp_estimation: Optional[FPEstimationResult] = None
    negative_check: Optional[NonVulnResult] = None
    corpus_tag: str = ""

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
            "| Class | Verdict | Semantic | Coverage | Best tier | Tool | FP | Co-occur |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for cls in class_names():
            v = self.gate.verdicts.get(cls)
            if v is None:
                continue
            icon = _VERDICT_ICON.get(v.verdict, "?")
            sem = f"{v.semantic_pass_rate:.0%}" if v.semantic_pass_rate is not None else "—"
            cov = f"{v.semantic_coverage:.0%}"
            tier = v.highest_tier or "—"
            tool = f"{v.tool_agreement:.0%}" if v.tool_agreement is not None else "—"
            fp = f"{v.fp_rate:.0%}" if v.fp_rate is not None else "—"
            flag = "⚠" if v.co_occurrence_flagged else ""
            lines.append(
                f"| {cls} | {icon} {v.verdict.value} | {sem} | {cov} "
                f"| {tier} | {tool} | {fp} | {flag} |"
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
                "| Class A | Class B | P(B=1 \\| A=1) | Count |",
                "|---|---|---|---|",
            ]
            for p in sorted(self.audit.flagged_pairs, key=lambda x: -x.rate):
                lines.append(
                    f"| {p.class_a} | {p.class_b} | {p.rate:.1%} | {p.count}/{p.count_a} |"
                )
        else:
            lines.append("_No flagged co-occurrence pairs (all below threshold)._")

        lines += ["", "---", "", "## 4. Semantic Check Summary", "",
                  "| Class | Pass | Fail | Skip | Not extractable | Coverage |",
                  "|---|---|---|---|---|---|"]
        for cls in class_names():
            sem = self.semantic.by_class.get(cls)
            if sem is None:
                continue
            lines.append(
                f"| {cls} | {sem.pass_count} | {sem.fail_count} | {sem.positives_skipped} "
                f"| {sem.not_extractable} | {sem.coverage:.0%} |"
            )

        if self.tool_validation is not None:
            lines += [
                "", "---", "",
                "## 5. Tool Validation (Slither agreement)",
                "",
                "| Class | Agreement | Checked | Agree | Disagree | No-detector | Errored |",
                "|---|---|---|---|---|---|---|",
            ]
            for cls in class_names():
                s = self.tool_validation.by_class.get(cls)
                if s is None or s.positives_total == 0:
                    continue
                rate = f"{s.agreement_rate:.1%}" if s.agreement_rate is not None else "—"
                lines.append(
                    f"| {cls} | {rate} | {s.checkable} | {s.agree} | {s.disagree} "
                    f"| {s.no_detector} | {s.errored} |"
                )

        if self.fp_estimation is not None:
            lines += [
                "", "---", "",
                f"## 6. FP Estimation (stratified, N={self.fp_estimation.sample_size_per_class}/class)",
                "",
                f"**Total sampled:** {self.fp_estimation.total_sampled}  ",
                f"**Total likely FP:** {self.fp_estimation.total_likely_fp}  ",
                f"**Total errored:** {self.fp_estimation.total_errored}  ",
                f"**FAIL threshold:** >30% per class",
                "",
                "| Class | FP rate | Sampled | Likely FP | Strata |",
                "|---|---|---|---|---|",
            ]
            for cls in class_names():
                s = self.fp_estimation.by_class.get(cls)
                if s is None or s.sampled == 0:
                    continue
                rate = f"{s.fp_rate:.1%}" if s.fp_rate is not None else "—"
                fail = "  ✗" if s.failed else ""
                strata_str = ", ".join(
                    f"{src}/{tier}: {st.fp_rate:.0%}"
                    for (src, tier), st in sorted(s.strata.items())
                    if st.fp_rate is not None
                )
                lines.append(
                    f"| {cls} | {rate}{fail} | {s.sampled} | {s.likely_fp} | {strata_str} |"
                )

        if self.negative_check is not None:
            nc = self.negative_check
            status_icon = {"OK": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(nc.status, "?")
            lines += [
                "", "---", "",
                "## 7. Negative Checker (NonVulnerable contamination)",
                "",
                f"**Status:** {status_icon} {nc.status}  ",
                f"**NonVulnerable contracts checked:** {nc.total_checked}  ",
                f"**With at least one Slither hit:** {nc.total_hits}  ",
                f"**Hit rate:** {nc.hit_rate:.1%}  " if nc.hit_rate is not None
                else "**Hit rate:** —  ",
                f"**Thresholds:** WARN > {nc.warn_threshold:.0%}, FAIL > {nc.fail_threshold:.0%}",
                "",
            ]
            if nc.by_source:
                lines += ["### Per-source breakdown", "",
                          "| Source | Hit rate | Hits | Total | Errored |",
                          "|---|---|---|---|---|"]
                for src, st in sorted(nc.by_source.items()):
                    rate = f"{st.hit_rate:.1%}" if st.hit_rate is not None else "—"
                    lines.append(
                        f"| {src} | {rate} | {st.hits} | {st.total} | {st.errored} |"
                    )
            if nc.detector_hit_counts:
                lines += ["", "### Top detectors that fired on NonVulnerable", ""]
                for det, n in nc.detector_hit_counts.most_common(10):
                    lines.append(f"- `{det}`: {n} hits")

        lines += [
            "", "---", "",
            "## 8. Hard Failures",
            "",
        ]
        if self.gate.hard_fails:
            lines += [f"- **{cls}** — export blocked" for cls in self.gate.hard_fails]
        else:
            lines.append("_No hard failures. All classes export-safe._")

        lines += [
            "", "---", "",
            "## 9. Known Limitations",
            "",
            "- **Stage 2 representation coverage**: Full DIVE corpus (22,073 contracts) may not",
            "  have had Stage 2 graph extraction run for all contracts. Semantic checks for",
            "  un-represented contracts are marked SKIP.",
            "- **NOT_EXTRACTABLE classes (v9 schema)**: DenialOfService, GasException,",
            "  TransactionOrderDependence have no dedicated feature in the v9 graph schema.",
            "  Full verification requires Slither-based AST analysis (the tool_validator).",
            "- **Tool agreement ≠ label quality**: Slither's per-class precision is variable",
            "  (e.g. ~52% on reentrancy). A high disagreement rate is suspicious but not",
            "  conclusive; the FP estimator (Section 6) provides the empirical rate.",
            "- **FP estimator (v1)**: uses Slither-disagreement as the upper bound on the",
            "  false-positive rate. v2.1 enhancement: compound Slither + semantic check.",
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
    tool_validation: Optional[ToolValidationResult] = None,
    fp_estimation: Optional[FPEstimationResult] = None,
    negative_check: Optional[NonVulnResult] = None,
    corpus_tag: str = "",
    output_path: Path | None = None,
) -> VerificationReport:
    """Build and optionally write a VerificationReport.

    Args:
        audit: Output from class_auditor.run_audit().
        semantic: Output from semantic_checker.run_semantic_check().
        gate: Output from gate.run_gate().
        tool_validation: Optional — output from tool_validator.run_tool_validation().
        fp_estimation: Optional — output from fp_estimator.run_fp_estimation().
        negative_check: Optional — output from negative_checker.run_negative_check().
        corpus_tag: Human-readable corpus identifier for the report header.
        output_path: If provided, write report.md here.

    Returns:
        VerificationReport with .to_markdown() method.
    """
    report = VerificationReport(
        audit=audit,
        semantic=semantic,
        gate=gate,
        tool_validation=tool_validation,
        fp_estimation=fp_estimation,
        negative_check=negative_check,
        corpus_tag=corpus_tag,
    )
    if output_path:
        report.write(output_path)
    return report
