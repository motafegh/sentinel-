"""
run_benchmark.py — CLI runner for the SENTINEL eval framework (P0.1 T0.1.3).

Usage:
    python -m src.eval.run_benchmark \\
        --name <id> --config <yaml> --reports <dir> --corpus <dir> \\
        [--baseline <json>]

Loads a named config (P1 loader), runs gates + PipelineMetrics (with macro_fbeta),
writes eval/runs/<timestamp>_<name>/eval_metrics.json + eval_report.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_config
from src.config.loader import load_config as load_sentinel_config
from src.eval import (
    ClassMetrics,
    ContractEval,
    ContractMetrics,
    GateResult,
    PipelineMetrics,
    gate_d4_eye_predictions_present,
    gate_macro_f1_vs_baseline,
    gate_ws1a_silent_safe_on_flagged,
    gate_ws1b_inconclusive_on_timeout,
    gate_ws1c_no_missing_consensus_votes,
    gate_ws1d_confidence_1_0_not_downgraded,
    gate_ws1e_no_vulnerable_label_with_safe_verdict,
    gate_ws2_false_positives_on_safe,
    gate_ws3_long_contract_bug_detected,
)
from src.eval.regression import RegressionBaseline


# ---------------------------------------------------------------------------
# Corpus loading (adapted from scripts/eval_benchmark.py)
# ---------------------------------------------------------------------------

def _build_sidecar_index(corpus_dir: Path) -> dict[str, Path]:
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


def _build_sol_index(corpus_dir: Path) -> dict[str, Path]:
    import os as _os
    index: dict[str, Path] = {}
    for root, _dirs, files in _os.walk(corpus_dir, followlinks=True):
        for f in files:
            if f.endswith(".sol"):
                index[Path(f).stem] = Path(root) / f
    return index


def _parse_expect_header(sol_path: Path) -> tuple[list[str], str] | None:
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
    stem = report_path.stem.removesuffix("_report")
    raw = json.loads(report_path.read_text())
    sol_path = sol_index.get(stem)
    label_info = _derive_labels_for_stem(stem, sol_path, sidecar_index)
    if label_info is None:
        return None
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


def load_corpus(reports_dir: Path, corpus_dir: Path) -> list[ContractEval]:
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
    if not rows:
        print(f"ERROR: no reports with matching labels found in {reports_dir}", file=sys.stderr)
        sys.exit(2)
    return rows


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _positive_classes(verdicts: dict[str, str], positive_set: set[str]) -> list[str]:
    return [c for c, v in verdicts.items() if v in positive_set]


def compute_per_contract(rows: list[ContractEval], positive_set: set[str]) -> None:
    for row in rows:
        pred = set(_positive_classes(row.verdicts, positive_set))
        labels = set(row.labels)
        row.predicted_positive_classes = sorted(pred)
        row.true_positive_classes = sorted(pred & labels)
        row.false_positive_classes = sorted(pred - labels)
        row.false_negative_classes = sorted(labels - pred)
        if row.ground_truth == "safe":
            row.contract_correct = len(pred) == 0
        elif row.ground_truth == "vulnerable":
            row.contract_correct = len(pred & labels) > 0
        row.contract_exact = pred == labels


def compute_metrics(rows: list[ContractEval], positive_set: set[str]) -> PipelineMetrics:
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
    return pm


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_markdown(
    rows: list[ContractEval],
    pm: PipelineMetrics,
    gates: list[GateResult],
    baseline: dict | None,
) -> str:
    lines: list[str] = []
    lines.append("# SENTINEL Agents Pipeline Evaluation Report\n")
    lines.append(f"**Contracts evaluated:** {len(rows)}  \n")
    lines.append(f"**Macro-F1:** {pm.macro_f1:.4f}  |  **Macro-Fbeta:** {pm.macro_fbeta:.4f}  |  **Micro-F1:** {pm.micro_f1:.4f}\n")

    lines.append("## Gate assertions\n")
    lines.append("| Gate | Description | Passed | Detail |")
    lines.append("|---|---|---|---|")
    for g in gates:
        mark = "PASS" if g.passed else "FAIL"
        lines.append(f"| `{g.gate_id}` | {g.description} | {mark} | {g.detail} |")
    all_pass = all(g.passed for g in gates)
    lines.append(f"\n**Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}**\n")

    lines.append("## Per-class metrics\n")
    lines.append("| Class | Support | TP | FP | FN | TN | Precision | Recall | F1 | F-beta |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in sorted(pm.class_metrics.values(), key=lambda x: -x.support):
        p = f"{m.precision:.4f}" if not math.isnan(m.precision) else "nan"
        r = f"{m.recall:.4f}" if not math.isnan(m.recall) else "nan"
        lines.append(f"| {m.cls} | {m.support} | {m.tp} | {m.fp} | {m.fn} | {m.tn} | {p} | {r} | {m.f1:.4f} | {m.fbeta:.4f} |")

    correct = sum(1 for r in rows if r.contract_correct)
    exact = sum(1 for r in rows if r.contract_exact)
    lines.append("\n## Contract-level accuracy\n")
    lines.append(f"- **Loose**: {correct}/{len(rows)} = {correct / len(rows):.2%}")
    lines.append(f"- **Exact**: {exact}/{len(rows)} = {exact / len(rows):.2%}\n")

    if baseline is not None:
        lines.append("## Baseline comparison\n")
        lines.append("| Metric | Baseline | Current | Delta |")
        lines.append("|---|---|---|---|")
        lines.append(f"| macro_F1 | {baseline.get('macro_f1', 0):.4f} | {pm.macro_f1:.4f} | {pm.macro_f1 - baseline.get('macro_f1', 0):+.4f} |")
        lines.append(f"| macro_Fbeta | {baseline.get('macro_fbeta', 0):.4f} | {pm.macro_fbeta:.4f} | {pm.macro_fbeta - baseline.get('macro_fbeta', 0):+.4f} |")

    return "\n".join(lines) + "\n"


def build_metrics_json(
    rows: list[ContractEval],
    pm: PipelineMetrics,
    gates: list[GateResult],
) -> dict:
    correct = sum(1 for r in rows if r.contract_correct)
    exact = sum(1 for r in rows if r.contract_exact)
    d = pm.as_dict()
    d["gates"] = [
        {"gate_id": g.gate_id, "description": g.description, "passed": g.passed, "detail": g.detail, "value": g.value}
        for g in gates
    ]
    d["per_contract"] = [
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
    ]
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.eval.run_benchmark",
        description="SENTINEL eval framework runner (P0.1 T0.1.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name", type=str, required=True, help="Run identifier (suffix for output dir)")
    p.add_argument("--config", type=Path, default=None, help="Path to verdicts YAML config (default: configs/verdicts_default.yaml)")
    p.add_argument("--reports", type=Path, required=True, help="Dir containing <stem>_report.json files")
    p.add_argument("--corpus", type=Path, required=True, help="Corpus root with .sol + .json sidecars")
    p.add_argument("--baseline", type=Path, default=None, help="Prior eval_metrics.json for regression check")
    p.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: eval/runs/<ts>_<name>)")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1. Load config
    if args.config:
        cfg = load_sentinel_config(args.config)
    else:
        cfg = get_config()

    positive_set = frozenset(cfg.eval.positive_verdicts)

    # 2. Validate inputs
    if not args.reports.is_dir():
        print(f"ERROR: --reports dir not found: {args.reports}", file=sys.stderr)
        sys.exit(2)
    if not args.corpus.is_dir():
        print(f"ERROR: --corpus dir not found: {args.corpus}", file=sys.stderr)
        sys.exit(2)

    # 3. Load and score
    rows = load_corpus(args.reports, args.corpus)
    compute_per_contract(rows, positive_set)
    pm = compute_metrics(rows, positive_set)

    # 4. Baseline
    baseline = None
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text())

    # 5. Gates
    gates = [
        gate_ws1a_silent_safe_on_flagged(rows),
        gate_ws1b_inconclusive_on_timeout(rows),
        gate_ws1c_no_missing_consensus_votes(rows),
        gate_ws1d_confidence_1_0_not_downgraded(rows),
        gate_ws1e_no_vulnerable_label_with_safe_verdict(rows),
        gate_ws2_false_positives_on_safe(rows, positive_set),
        gate_ws3_long_contract_bug_detected(rows, positive_set),
        gate_d4_eye_predictions_present(rows),
        gate_macro_f1_vs_baseline(pm.macro_f1, baseline),
    ]

    # 6. Output dir
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (Path("eval") / "runs" / f"{ts}_{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 7. Write report
    md = render_markdown(rows, pm, gates, baseline)
    report_path = out_dir / "eval_report.md"
    report_path.write_text(md)
    print(f"Report -> {report_path}")

    # 8. Write metrics JSON
    metrics = build_metrics_json(rows, pm, gates)
    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Metrics -> {metrics_path}")

    # 9. Console summary
    print()
    print("=" * 72)
    print(f"Contracts: {len(rows)}  |  Macro-F1: {pm.macro_f1:.4f}  |  Macro-Fbeta: {pm.macro_fbeta:.4f}")
    print("=" * 72)
    for g in gates:
        mark = "PASS" if g.passed else "FAIL"
        print(f"  [{mark}] {g.gate_id}: {g.detail}")
    print("=" * 72)

    all_pass = all(g.passed for g in gates)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
