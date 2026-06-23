#!/usr/bin/env python3
"""
SENTINEL agents pipeline evaluation comparator (WS0 gate infrastructure, refactored 2026-06-22 to use src.eval).

Reads per-contract audit reports produced by `scripts/run_real_audit.py
--corpus DIR --output-dir OUT`, matches each to its labelled sidecar in the
corpus, computes per-class precision/recall/F1 + contract-level accuracy +
the per-workstream gate assertions defined in
`docs/plan/agents/2026-06-21-agents-redesign/03_GATE_INFRASTRUCTURE_PLAN.md`,
and writes a markdown report + a JSON metrics file. Exits nonzero on regression
vs a stored baseline.

WS6a (2026-06-22): aggregation logic (per-class metrics, macro/micro, per-contract
TP/FP/FN) now delegates to the new `src.eval.PipelineMetrics` library — this
script is now a thin CLI wrapper. Other callers (notebooks, the C.1 FastAPI
gateway, CI) can use the same library directly.

USAGE
    cd ~/projects/sentinel/agents
    poetry run python scripts/eval_benchmark.py \\
        --reports  eval/runs/<timestamp> \\
        --corpus   ../data_module/benchmarks/benchmark_v0.1_quickstart/contracts/by_class \\
        [--baseline eval/baselines/pre_redesign.json] \\
        [--output   eval/runs/<timestamp>/eval_report.md] \\
        [--metrics  eval/runs/<timestamp>/eval_metrics.json]

EXIT CODES
    0  all gates pass AND macro_F1 >= baseline (or no baseline given)
    1  one or more gates failed OR macro_F1 regressed vs baseline
    2  bad inputs (reports dir empty, no sidecars matched, etc.)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# WS6a (2026-06-22): delegate per-class + aggregate metrics to the new
# eval library. Behaviour is identical to the previous inline implementation.
from src.eval import (
    ClassMetrics,
    ContractMetrics,
    PipelineMetrics,
)

# ── Verdict semantics ───────────────────────────────────────────────────────
# A verdict is "positive" (flagged vulnerable) if it's in this set. Tunable
# via --positive-verdicts (comma-separated). Default: CONFIRMED + LIKELY are
# the only "we think this is a real bug" verdicts; DISPUTED/WATCH/SAFE/
# INCONCLUSIVE are all "not flagged" (INCONCLUSIVE is deliberately NOT positive
# — it means "we couldn't check", which is a different failure mode than a
# false positive and must not be counted as a detected bug).
DEFAULT_POSITIVE_VERDICTS = {"CONFIRMED", "LIKELY"}

# The borderline band for WS1's silent-SAFE gate: a class flagged by ML
# (prob in this range, i.e. crossed DEEP_THRESHOLD ~0.35 but below the 0.50
# consensus cutoff) that ends in verdict SAFE with no corroboration is the
# exact failure mode Finding #9/#10 describes.
BORDERLINE_BAND = (0.35, 0.50)


# ══════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ContractEval:
    """One contract's evaluation row."""
    stem: str
    report_path: str
    sidecar_path: str | None
    labels: list[str]                 # vulnerability classes (from sidecar)
    ground_truth: str                 # "vulnerable" | "safe"
    verdicts: dict[str, str]          # class -> verdict string (from report)
    probabilities: dict[str, float]   # class -> ML prob (from report)
    quick_screen_hits: dict[str, list]  # {slither: [...], aderyn: [...]}
    static_findings_count: int
    rag_results_count: int
    path_taken: str                   # "deep" | "fast"
    overall_label: str | None
    overall_verdict: str | None       # from final_report (for WS1e)
    narrative: str | None
    error: str | None
    consensus_verdict: dict[str, dict]  # class -> {verdict, confidence, ...} (WS1.5 gates)
    vulnerability_verdicts_classes: set  # classes in final_report.vulnerability_verdicts (WS1c)
    eye_predictions: dict | None = None  # D4 (WS3): per-eye predictions from ml_result
    # Derived
    predicted_positive_classes: list[str] = field(default_factory=list)
    true_positive_classes: list[str] = field(default_factory=list)
    false_positive_classes: list[str] = field(default_factory=list)
    false_negative_classes: list[str] = field(default_factory=list)
    contract_correct: bool = False      # loose: safe→no positive OR vuln→≥1 positive
    contract_exact: bool = False        # strict: predicted set == label set


# Note: ClassMetrics is now imported from src.eval.pipeline_metrics (line 47).
# The previous inline dataclass has been replaced — see WS6a (2026-06-22).


@dataclass
class GateResult:
    gate_id: str          # e.g. "WS1a_silent_safe_on_flagged"
    description: str
    passed: bool
    detail: str           # human-readable pass/fail detail
    value: Any = None     # the metric value (count, ratio, etc.)


# ══════════════════════════════════════════════════════════════════════════
# Loading + matching
# ══════════════════════════════════════════════════════════════════════════

def _build_sidecar_index(corpus_dir: Path) -> dict[str, Path]:
    """Index every <stem>.json sidecar under corpus_dir by stem.
    Returns {stem: sidecar_path}. Stems are unique within the benchmark
    (filenames are designed to be unique across class dirs).

    Sidecars are optional — the manual_hand_written_contracts/ corpus uses
    `// expect:` headers in the .sol file itself instead. _parse_expect_header
    handles that fallback at load time. Uses os.walk(followlinks=True) so
    symlinked corpus dirs (the combined corpus) work.
    """
    import os as _os
    index: dict[str, Path] = {}
    for root, _dirs, files in _os.walk(corpus_dir, followlinks=True):
        for f in files:
            if not f.endswith(".json"):
                continue
            if f in ("tier_a_manifest.json", "contamination_check.json"):
                continue
            index[Path(f).stem] = Path(root) / f
    return index


def _parse_expect_header(sol_path: Path) -> tuple[list[str], str] | None:
    """Parse `// expect:` labels from the first few lines of a .sol file.

    The manual_hand_written_contracts/ corpus (see its README.md) puts ground
    truth in an `// expect:` header as a comma-separated class list, e.g.:
        // expect: Reentrancy
        // expect: Reentrancy,Timestamp,UnusedReturn
        // expect:                      (empty = Safe contract)

    Returns (labels_list, ground_truth_string) or None if no `// expect:` line
    found in the first 20 lines.
    """
    try:
        text = sol_path.read_text()
    except OSError:
        return None
    for line in text.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("// expect:"):
            payload = stripped[len("// expect:"):].strip()
            labels = [c.strip() for c in payload.split(",") if c.strip()]
            gt = "safe" if not labels else "vulnerable"
            return labels, gt
    return None


def _derive_labels_for_stem(
    stem: str,
    sol_path: Path | None,
    sidecar_index: dict[str, Path],
) -> tuple[list[str], str, Path | None, str] | None:
    """Resolve labels for one contract stem. Tries (in order):
      1. .json sidecar (benchmark v0.1 + my edge cases)
      2. `// expect:` header in the .sol (manual_hand_written_contracts/)
    Returns (labels, ground_truth, label_source_path, label_format) or None
    if neither source is available.
    """
    sidecar_path = sidecar_index.get(stem)
    if sidecar_path is not None:
        sidecar = json.loads(sidecar_path.read_text())
        labels = list(sidecar.get("labels", []) or [])
        gt = sidecar.get("ground_truth", "safe" if not labels else "vulnerable")
        return labels, gt, sidecar_path, "json_sidecar"
    if sol_path is not None and sol_path.exists():
        parsed = _parse_expect_header(sol_path)
        if parsed is not None:
            labels, gt = parsed
            return labels, gt, sol_path, "expect_header"
    return None


def _load_contract_eval(
    report_path: Path,
    sidecar_index: dict[str, Path],
    sol_index: dict[str, Path],
) -> ContractEval | None:
    """Load one report + its matching labels. Returns None if no label source.

    Label resolution order (per _derive_labels_for_stem):
      1. .json sidecar alongside the .sol (benchmark v0.1, my edge cases)
      2. `// expect:` header in the .sol itself (manual_hand_written_contracts/)
    """
    stem = report_path.stem.removesuffix("_report")
    raw = json.loads(report_path.read_text())
    sol_path = sol_index.get(stem)
    label_info = _derive_labels_for_stem(stem, sol_path, sidecar_index)
    if label_info is None:
        return None  # no sidecar AND no // expect: header — can't evaluate
    labels, gt, label_source_path, _label_format = label_info
    verdicts = raw.get("verdicts") or (raw.get("final_report", {}) or {}).get("verdicts") or {}
    ml_result = raw.get("ml_result", {}) or {}
    probabilities = ml_result.get("probabilities", {}) or {}
    final_report = raw.get("final_report", {}) or {}
    return ContractEval(
        stem=stem,
        report_path=str(report_path),
        sidecar_path=str(label_source_path) if label_source_path else None,
        labels=labels,
        ground_truth=gt,
        verdicts={str(k): str(v) for k, v in verdicts.items()},
        probabilities={str(k): float(v) for k, v in probabilities.items()},
        quick_screen_hits=raw.get("quick_screen_hits", {}) or {},
        static_findings_count=int(raw.get("static_findings_count", 0) or 0),
        rag_results_count=int(raw.get("rag_results_count", 0) or 0),
        path_taken=final_report.get("path_taken", "unknown"),
        overall_label=final_report.get("overall_label"),
        overall_verdict=final_report.get("overall_verdict"),
        narrative=raw.get("narrative"),
        error=raw.get("error"),
        consensus_verdict=raw.get("consensus_verdict", {}) or {},
        vulnerability_verdicts_classes={
            vv.get("vulnerability_class", "") for vv in final_report.get("vulnerability_verdicts", []) or []
        },
        eye_predictions=ml_result.get("eye_predictions"),
    )


def _build_sol_index(corpus_dir: Path) -> dict[str, Path]:
    """Index every <stem>.sol under corpus_dir by stem. For the
    manual_hand_written_contracts/ corpus this is the label source (via
    `// expect:` header parsing); for benchmark v0.1 it's just a reference.
    Uses os.walk(followlinks=True) so symlinked corpus dirs work."""
    import os as _os
    index: dict[str, Path] = {}
    for root, _dirs, files in _os.walk(corpus_dir, followlinks=True):
        for f in files:
            if f.endswith(".sol"):
                index[Path(f).stem] = Path(root) / f
    return index


def load_corpus(reports_dir: Path, corpus_dir: Path) -> list[ContractEval]:
    """Load every <stem>_report.json + match to labels. Returns the eval rows."""
    sidecar_index = _build_sidecar_index(corpus_dir)
    sol_index = _build_sol_index(corpus_dir)
    reports = sorted(reports_dir.rglob("*_report.json"))
    rows: list[ContractEval] = []
    skipped: list[str] = []
    for rp in reports:
        row = _load_contract_eval(rp, sidecar_index, sol_index)
        if row is None:
            skipped.append(rp.stem.removesuffix("_report"))
        else:
            rows.append(row)
    if skipped:
        print(f"WARN: {len(skipped)} report(s) had no matching sidecar or // expect: header "
              f"(skipped): {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}",
              file=sys.stderr)
    if not rows:
        print(f"ERROR: no reports with matching labels found in {reports_dir} "
              f"(corpus: {corpus_dir})", file=sys.stderr)
        sys.exit(2)
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Metric computation
# ══════════════════════════════════════════════════════════════════════════

def _positive_classes(verdicts: dict[str, str], positive_set: set[str]) -> list[str]:
    return [c for c, v in verdicts.items() if v in positive_set]


def compute_per_contract(rows: list[ContractEval], positive_set: set[str]) -> None:
    """Derive per-contract TP/FP/FN lists + correctness flags."""
    for row in rows:
        pred = set(_positive_classes(row.verdicts, positive_set))
        labels = set(row.labels)
        row.predicted_positive_classes = sorted(pred)
        row.true_positive_classes = sorted(pred & labels)
        row.false_positive_classes = sorted(pred - labels)
        row.false_negative_classes = sorted(labels - pred)
        # Loose: a vulnerable contract is "correct" if we flagged at least one
        # of its real classes; a safe contract is "correct" if we flagged none.
        if row.ground_truth == "safe":
            row.contract_correct = len(pred) == 0
        elif row.ground_truth == "vulnerable":
            row.contract_correct = len(pred & labels) > 0
        # Strict: predicted positive set exactly equals label set.
        row.contract_exact = pred == labels


def compute_class_metrics(rows: list[ContractEval], positive_set: set[str]) -> dict[str, ClassMetrics]:
    """
    Per-class TP/FP/FN/TN/P/R/F1 over the whole corpus.

    WS6a (2026-06-22): this used to be the inline aggregation. Now it
    converts each ContractEval row to the new minimal ContractMetrics
    type and delegates to `PipelineMetrics` for the actual computation.
    Behaviour is identical to the previous inline implementation — same
    semantics, same field names, same JSON shape (because the new
    `ClassMetrics.as_dict()` matches the old one).

    Note: we call `pm.compute()` (which runs derive_per_contract +
    compute_class_metrics + compute_aggregates) — NOT just
    `pm.compute_class_metrics()` directly, because the latter requires
    derive_per_contract to have already populated
    `row.predicted_positive_classes`. `compute()` is the safe entry point.
    """
    minimal_rows = [
        ContractMetrics(
            stem=row.stem,
            report_path=row.report_path,
            labels=row.labels,
            ground_truth=row.ground_truth,
            verdicts=row.verdicts,
            probabilities=row.probabilities,
        )
        for row in rows
    ]
    pm = PipelineMetrics(minimal_rows, positive_verdicts=positive_set)
    pm.compute()
    return pm.class_metrics


def compute_macro_micro(class_metrics: dict[str, ClassMetrics]) -> tuple[float, float]:
    """
    Macro-F1 + micro-F1 over the ClassMetrics dict.

    WS6a (2026-06-22): delegates to PipelineMetrics. The dict input is
    the one produced by `compute_class_metrics` (above), which already
    ran `pm.compute()` so macro/micro are populated on a sidecar pm
    reachable through this function.

    Implementation note: we rebuild a fresh `PipelineMetrics` from the
    class_metrics dict and call `compute_aggregates()`. This avoids
    re-walking the original contracts (which the new `compute_class_metrics`
    already did internally).
    """
    if not class_metrics:
        return 0.0, 0.0
    pm = PipelineMetrics.__new__(PipelineMetrics)
    pm.class_metrics = class_metrics
    pm.contracts = []   # compute_aggregates uses len(self.contracts)
    pm.compute_aggregates()
    return pm.macro_f1, pm.micro_f1


# ══════════════════════════════════════════════════════════════════════════
# Per-workstream gate assertions
# ══════════════════════════════════════════════════════════════════════════

def gate_ws1a_silent_safe_on_flagged(rows: list[ContractEval]) -> GateResult:
    """WS1(a) / WS1.5: no flagged class ends in verdict SAFE when consensus_engine
    voted non-SAFE. This is the core FN/FP asymmetry enforcement: the debate can
    only downgrade to DISPUTED, never to SAFE, when consensus flagged the class.

    A violation is: final_verdict == SAFE AND consensus_verdict[cls] exists AND
    consensus_verdict[cls].verdict != SAFE. This means the debate silently
    cleared a class that consensus flagged — the exact Finding #14 failure mode.
    """
    violations: list[str] = []
    for row in rows:
        for cls, cv in row.consensus_verdict.items():
            cv_verdict = cv.get("verdict", "")
            final_verdict = row.verdicts.get(cls, "MISSING")
            if final_verdict == "SAFE" and cv_verdict != "SAFE":
                violations.append(
                    f"{row.stem}/{cls} (consensus={cv_verdict} conf={cv.get('confidence',0):.2f} → final=SAFE)"
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
    """WS1(b): the edge_debate_timeout case must emit INCONCLUSIVE, not SAFE.
    Only meaningful in LLM-on mode (in --no-llm there is no debate to time out)."""
    edge = next((r for r in rows if r.stem == "edge_debate_timeout"), None)
    has_inconclusive = any(v == "INCONCLUSIVE" for v in (edge.verdicts.values() if edge else []))
    has_llm = any(r.narrative for r in rows)  # crude LLM-on proxy: narrative is LLM-generated
    if not has_llm:
        return GateResult(
            gate_id="WS1b_inconclusive_on_timeout",
            description="edge_debate_timeout emits INCONCLUSIVE (LLM-on mode only)",
            passed=True, detail="N/A in --no-llm mode (no debate to time out)", value="skipped",
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
    """WS1.5(c) / Finding #15: every class consensus_engine voted on must appear
    in the final vulnerability_verdicts. Previously, classes with ML < 0.25 +
    tool hits were voted on by consensus but silently dropped because the
    synthesizer loop only iterated all_flagged (ML ≥ 0.25)."""
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
    """WS1.5(d) / Finding #14 worst case: no class with consensus confidence=1.0
    (all tools agreed) ends in SAFE. This is the most egregious FN — unanimous
    tool agreement silently cleared by the debate."""
    violations: list[str] = []
    for row in rows:
        for cls, cv in row.consensus_verdict.items():
            if cv.get("confidence", 0) >= 1.0 and row.verdicts.get(cls) == "SAFE":
                violations.append(f"{row.stem}/{cls} (conf=1.0 → SAFE)")
    passed = len(violations) == 0
    return GateResult(
        gate_id="WS1d_confidence_1_0_not_downgraded",
        description="No consensus confidence=1.0 class ends in SAFE (Finding #14 worst case)",
        passed=passed,
        detail=f"{len(violations)} violation(s)" + (f": {'; '.join(violations[:5])}" if violations else ""),
        value=len(violations),
    )


def gate_ws1e_no_vulnerable_label_with_safe_verdict(rows: list[ContractEval]) -> GateResult:
    """WS1.5(e) / Finding #19: no contract has overall_label=confirmed_vulnerable
    but overall_verdict=SAFE. This is the systematic-downgrade symptom — the ML
    says vulnerable, the final verdict says nothing confirmed."""
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
    """WS2: zero false-positive verdicts on the safe subset (the measurable
    harm from a RAG hallucination is a false-positive verdict on a safe
    contract). Covers NonVulnerable + edge_safe_rag_resembles_exploit."""
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
    """WS3: the edge_long_contract_truncated contract's Reentrancy bug (which
    sits past the 2000-char cutoff) must be flagged. With current raw
    truncation this is expected to FAIL on the baseline — that's the point."""
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
    """D4 (WS3, 2026-06-22): every contract's ml_result should include
    eye_predictions (per-eye auxiliary predictions) as discountable clues."""
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
    """The regression net: macro_F1 must not drop vs the stored baseline."""
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


# ══════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════

def render_markdown(
    rows: list[ContractEval],
    class_metrics: dict[str, ClassMetrics],
    macro_f1: float,
    micro_f1: float,
    gates: list[GateResult],
    positive_set: set[str],
    baseline: dict | None,
) -> str:
    lines: list[str] = []
    lines.append("# SENTINEL Agents Pipeline Evaluation Report\n")
    lines.append(f"**Contracts evaluated:** {len(rows)}  ")
    lines.append(f"**Positive verdict set:** {sorted(positive_set)}  ")
    lines.append(f"**Macro-F1:** {macro_f1:.4f}  |  **Micro-F1:** {micro_f1:.4f}\n")

    # ── Gate summary ───────────────────────────────────────────────────────
    lines.append("## Gate assertions\n")
    lines.append("| Gate | Description | Passed | Detail |")
    lines.append("|---|---|---|---|")
    for g in gates:
        mark = "PASS" if g.passed else "FAIL"
        lines.append(f"| `{g.gate_id}` | {g.description} | {mark} | {g.detail} |")
    all_pass = all(g.passed for g in gates)
    lines.append(f"\n**Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}**\n")

    # ── Per-class metrics ──────────────────────────────────────────────────
    lines.append("## Per-class metrics\n")
    lines.append("| Class | Support | TP | FP | FN | TN | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in sorted(class_metrics.values(), key=lambda x: -x.support):
        p = f"{m.precision:.4f}" if not math.isnan(m.precision) else "nan"
        r = f"{m.recall:.4f}" if not math.isnan(m.recall) else "nan"
        lines.append(f"| {m.cls} | {m.support} | {m.tp} | {m.fp} | {m.fn} | {m.tn} | {p} | {r} | {m.f1:.4f} |")

    # ── Contract-level accuracy ────────────────────────────────────────────
    correct = sum(1 for r in rows if r.contract_correct)
    exact = sum(1 for r in rows if r.contract_exact)
    lines.append("\n## Contract-level accuracy\n")
    lines.append(f"- **Loose** (safe→no flag OR vuln→≥1 correct flag): {correct}/{len(rows)} = {correct/len(rows):.2%}")
    lines.append(f"- **Strict exact-match** (predicted set == label set): {exact}/{len(rows)} = {exact/len(rows):.2%}\n")

    # ── Per-contract detail ────────────────────────────────────────────────
    lines.append("## Per-contract detail\n")
    lines.append("| Contract | GT | Labels | Predicted | TP | FP | FN | Correct |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in sorted(rows, key=lambda r: r.stem):
        gt = row.ground_truth
        labels = ",".join(row.labels) or "—"
        pred = ",".join(row.predicted_positive_classes) or "—"
        tp = ",".join(row.true_positive_classes) or "—"
        fp = ",".join(row.false_positive_classes) or "—"
        fn = ",".join(row.false_negative_classes) or "—"
        mark = "✓" if row.contract_correct else "✗"
        lines.append(f"| {row.stem} | {gt} | {labels} | {pred} | {tp} | {fp} | {fn} | {mark} |")

    # ── Baseline comparison ────────────────────────────────────────────────
    if baseline is not None:
        lines.append("\n## Baseline comparison\n")
        lines.append(f"| Metric | Baseline | Current | Delta |")
        lines.append(f"|---|---|---|---|")
        lines.append(f"| macro_F1 | {baseline.get('macro_f1', 0):.4f} | {macro_f1:.4f} | {macro_f1 - baseline.get('macro_f1', 0):+.4f} |")
        lines.append(f"| micro_F1 | {baseline.get('micro_f1', 0):.4f} | {micro_f1:.4f} | {micro_f1 - baseline.get('micro_f1', 0):+.4f} |")
        lines.append(f"| contract_accuracy_loose | {baseline.get('contract_accuracy_loose', 0):.4f} | {correct/len(rows):.4f} | {(correct/len(rows)) - baseline.get('contract_accuracy_loose', 0):+.4f} |")

    return "\n".join(lines) + "\n"


def build_metrics_json(
    rows: list[ContractEval],
    class_metrics: dict[str, ClassMetrics],
    macro_f1: float,
    micro_f1: float,
    gates: list[GateResult],
    positive_set: set[str],
) -> dict:
    correct = sum(1 for r in rows if r.contract_correct)
    exact = sum(1 for r in rows if r.contract_exact)
    return {
        "contract_count": len(rows),
        "positive_verdicts": sorted(positive_set),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "contract_accuracy_loose": correct / len(rows) if rows else 0.0,
        "contract_accuracy_exact": exact / len(rows) if rows else 0.0,
        "per_class": {m.cls: asdict(m) for m in class_metrics.values()},
        "gates": [
            {"gate_id": g.gate_id, "description": g.description, "passed": g.passed, "detail": g.detail, "value": g.value}
            for g in gates
        ],
        "per_contract": [
            {
                "stem": r.stem,
                "ground_truth": r.ground_truth,
                "labels": r.labels,
                "predicted_positive": r.predicted_positive_classes,
                "true_positive": r.true_positive_classes,
                "false_positive": r.false_positive_classes,
                "false_negative": r.false_negative_classes,
                "contract_correct": r.contract_correct,
                "contract_exact": r.contract_exact,
                "path_taken": r.path_taken,
                "error": r.error,
            }
            for r in rows
        ],
    }


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        prog="eval_benchmark",
        description="SENTINEL agents pipeline evaluation comparator (WS0 gate infra).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--reports", type=Path, required=True, metavar="DIR",
                   help="Dir containing <stem>_report.json files (output of run_real_audit.py --corpus).")
    p.add_argument("--corpus", type=Path, required=True, metavar="DIR",
                   help="Corpus root containing <stem>.sol + <stem>.json sidecars (recursive search).")
    p.add_argument("--baseline", type=Path, default=None, metavar="JSON",
                   help="Prior eval_metrics.json to compare against (regression check).")
    p.add_argument("--output", type=Path, default=None, metavar="MD",
                   help="Where to write the markdown report (default: <reports>/eval_report.md).")
    p.add_argument("--metrics", type=Path, default=None, metavar="JSON",
                   help="Where to write the JSON metrics (default: <reports>/eval_metrics.json).")
    p.add_argument("--positive-verdicts", default=",".join(sorted(DEFAULT_POSITIVE_VERDICTS)), metavar="CSV",
                   help="Comma-separated verdicts counted as positive (flagged vulnerable).")
    args = p.parse_args()

    positive_set = {v.strip() for v in args.positive_verdicts.split(",") if v.strip()}

    if not args.reports.is_dir():
        print(f"ERROR: --reports dir not found: {args.reports}", file=sys.stderr)
        sys.exit(2)
    if not args.corpus.is_dir():
        print(f"ERROR: --corpus dir not found: {args.corpus}", file=sys.stderr)
        sys.exit(2)

    rows = load_corpus(args.reports, args.corpus)
    compute_per_contract(rows, positive_set)
    class_metrics = compute_class_metrics(rows, positive_set)
    macro_f1, micro_f1 = compute_macro_micro(class_metrics)

    baseline = None
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text())

    gates = [
        gate_ws1a_silent_safe_on_flagged(rows),
        gate_ws1b_inconclusive_on_timeout(rows),
        gate_ws1c_no_missing_consensus_votes(rows),
        gate_ws1d_confidence_1_0_not_downgraded(rows),
        gate_ws1e_no_vulnerable_label_with_safe_verdict(rows),
        gate_ws2_false_positives_on_safe(rows, positive_set),
        gate_ws3_long_contract_bug_detected(rows, positive_set),
        gate_d4_eye_predictions_present(rows),
        gate_macro_f1_vs_baseline(macro_f1, baseline),
    ]

    md = render_markdown(rows, class_metrics, macro_f1, micro_f1, gates, positive_set, baseline)
    out_md = args.output or (args.reports / "eval_report.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    print(f"markdown report -> {out_md}")

    metrics = build_metrics_json(rows, class_metrics, macro_f1, micro_f1, gates, positive_set)
    out_json = args.metrics or (args.reports / "eval_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"metrics JSON    -> {out_json}")

    # Console summary
    print()
    print("=" * 72)
    print(f"Contracts: {len(rows)}  |  Macro-F1: {macro_f1:.4f}  |  Micro-F1: {micro_f1:.4f}")
    print("=" * 72)
    for g in gates:
        mark = "PASS" if g.passed else "FAIL"
        print(f"  [{mark}] {g.gate_id}: {g.detail}")
    print("=" * 72)

    all_pass = all(g.passed for g in gates)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
