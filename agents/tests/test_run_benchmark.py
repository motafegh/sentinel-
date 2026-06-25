from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.pipeline_metrics import ContractMetrics, PipelineMetrics
from src.eval.run_benchmark import (
    build_arg_parser,
    build_metrics_json,
    compute_metrics,
    compute_per_contract,
    load_corpus,
    render_markdown,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def contract_eval() -> dict:
    """Minimal ContractEval-like dict fixture (matches the module's ContractEval dataclass)."""
    return {
        "stem": "test_contract",
        "report_path": "/fake/test_contract_report.json",
        "sidecar_path": None,
        "labels": ["Reentrancy"],
        "ground_truth": "vulnerable",
        "verdicts": {"Reentrancy": "CONFIRMED"},
        "probabilities": {"Reentrancy": 0.95},
        "quick_screen_hits": {},
        "static_findings_count": 0,
        "rag_results_count": 0,
        "path_taken": "deep",
        "overall_label": "vulnerable",
        "overall_verdict": "CONFIRMED",
        "narrative": None,
        "error": None,
        "consensus_verdict": {},
        "vulnerability_verdicts_classes": set(),
        "eye_predictions": None,
        "predicted_positive_classes": [],  # will be set by compute_per_contract
        "true_positive_classes": [],
        "false_positive_classes": [],
        "false_negative_classes": [],
        "contract_correct": False,
        "contract_exact": False,
    }


def make_contract_eval(stem: str, labels: list[str], verdicts: dict[str, str],
                       gt: str = "vulnerable") -> dict:
    return {
        "stem": stem,
        "report_path": f"/fake/{stem}_report.json",
        "sidecar_path": None,
        "labels": labels,
        "ground_truth": gt,
        "verdicts": verdicts,
        "probabilities": {k: 0.9 for k in verdicts},
        "quick_screen_hits": {},
        "static_findings_count": 0,
        "rag_results_count": 0,
        "path_taken": "deep",
        "overall_label": gt,
        "overall_verdict": list(verdicts.values())[0] if verdicts else "SAFE",
        "narrative": None,
        "error": None,
        "consensus_verdict": {},
        "vulnerability_verdicts_classes": set(),
        "eye_predictions": None,
        "predicted_positive_classes": [],
        "true_positive_classes": [],
        "false_positive_classes": [],
        "false_negative_classes": [],
        "contract_correct": False,
        "contract_exact": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# build_arg_parser
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildArgParser:
    def test_parser_created(self):
        p = build_arg_parser()
        assert p is not None
        assert p.prog == "python -m src.eval.run_benchmark"

    def test_required_args(self):
        p = build_arg_parser()
        # --name and --reports and --corpus are required
        args = p.parse_args(["--name", "foo", "--reports", "/r", "--corpus", "/c"])
        assert args.name == "foo"
        assert str(args.reports) == "/r"
        assert str(args.corpus) == "/c"

    def test_optional_args(self):
        p = build_arg_parser()
        args = p.parse_args([
            "--name", "bar", "--reports", "/r", "--corpus", "/c",
            "--config", "/cfg.yaml", "--baseline", "/bl.json",
            "--output-dir", "/out",
        ])
        assert str(args.config) == "/cfg.yaml"
        assert str(args.baseline) == "/bl.json"
        assert str(args.output_dir) == "/out"


# ═══════════════════════════════════════════════════════════════════════════
# load_corpus
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadCorpus:
    def test_loads_contract_from_json_sidecar(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # Sidecar JSON
        (corpus / "c1.json").write_text(json.dumps({
            "labels": ["Reentrancy"], "ground_truth": "vulnerable",
        }))
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "c1_report.json").write_text(json.dumps({
            "verdicts": {"Reentrancy": "CONFIRMED"},
            "ml_result": {"probabilities": {"Reentrancy": 0.95}},
            "final_report": {"path_taken": "deep"},
        }))
        rows = load_corpus(reports, corpus)
        assert len(rows) == 1
        assert rows[0].stem == "c1"
        assert rows[0].ground_truth == "vulnerable"
        assert rows[0].labels == ["Reentrancy"]

    def test_loads_contract_from_expect_header(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "c2.sol").write_text(
            "pragma solidity ^0.8.0;\n// expect: Reentrancy\ncontract C {}\n"
        )
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "c2_report.json").write_text(json.dumps({
            "verdicts": {"Reentrancy": "CONFIRMED"},
            "ml_result": {},
            "final_report": {"path_taken": "deep"},
        }))
        rows = load_corpus(reports, corpus)
        assert len(rows) == 1
        assert rows[0].stem == "c2"
        assert rows[0].ground_truth == "vulnerable"

    def test_skips_report_without_label_source(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # No .json sidecar, no .sol with // expect: header
        (corpus / "c3.sol").write_text("pragma solidity ^0.8.0;\ncontract C {}\n")
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "c3_report.json").write_text(json.dumps({
            "verdicts": {"X": "CONFIRMED"},
        }))
        with pytest.raises(SystemExit):
            load_corpus(reports, corpus)

    def test_empty_report_dir_exits(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        reports = tmp_path / "reports"
        reports.mkdir()
        with pytest.raises(SystemExit):
            load_corpus(reports, corpus)

    def test_sidecar_preferred_over_expect_header(self, tmp_path):
        """JSON sidecar takes precedence over // expect: header."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "c4.json").write_text(json.dumps({
            "labels": ["Reentrancy"], "ground_truth": "vulnerable",
        }))
        (corpus / "c4.sol").write_text("// expect: IntegerOverflow\ncontract C {}\n")
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "c4_report.json").write_text(json.dumps({
            "verdicts": {"Reentrancy": "CONFIRMED"},
            "ml_result": {},
            "final_report": {},
        }))
        rows = load_corpus(reports, corpus)
        assert len(rows) == 1
        assert rows[0].labels == ["Reentrancy"]  # from sidecar, not expect header


# ═══════════════════════════════════════════════════════════════════════════
# compute_per_contract
# ═══════════════════════════════════════════════════════════════════════════

def _as_dataclass(d: dict):
    """Convert a dict fixture back to ContractEval."""
    from dataclasses import make_dataclass
    from src.eval import ContractEval
    d2 = {k: v for k, v in d.items()}
    return ContractEval(**d2)


class TestComputePerContract:
    def test_vulnerable_correct(self, contract_eval):
        row = _as_dataclass(contract_eval)
        compute_per_contract([row], {"CONFIRMED", "LIKELY"})
        assert row.contract_correct is True
        assert row.contract_exact is True
        assert row.true_positive_classes == ["Reentrancy"]
        assert row.false_positive_classes == []
        assert row.false_negative_classes == []

    def test_vulnerable_miss(self, contract_eval):
        d = dict(contract_eval)
        d["verdicts"] = {}
        row = _as_dataclass(d)
        compute_per_contract([row], {"CONFIRMED", "LIKELY"})
        assert row.contract_correct is False
        assert row.contract_exact is False
        assert row.true_positive_classes == []
        assert row.false_positive_classes == []

    def test_safe_correct(self, contract_eval):
        d = dict(contract_eval)
        d["labels"] = []
        d["ground_truth"] = "safe"
        d["verdicts"] = {}
        row = _as_dataclass(d)
        compute_per_contract([row], {"CONFIRMED", "LIKELY"})
        assert row.contract_correct is True
        assert row.contract_exact is True

    def test_safe_wrong_false_positive(self, contract_eval):
        d = dict(contract_eval)
        d["labels"] = []
        d["ground_truth"] = "safe"
        d["verdicts"] = {"Reentrancy": "CONFIRMED"}
        row = _as_dataclass(d)
        compute_per_contract([row], {"CONFIRMED", "LIKELY"})
        assert row.contract_correct is False
        assert row.contract_exact is False
        assert row.false_positive_classes == ["Reentrancy"]

    def test_partial_match(self, contract_eval):
        d = dict(contract_eval)
        d["labels"] = ["Reentrancy", "IntegerOverflow"]
        d["verdicts"] = {"Reentrancy": "CONFIRMED"}
        row = _as_dataclass(d)
        compute_per_contract([row], {"CONFIRMED", "LIKELY"})
        assert row.contract_correct is True  # at least one label found
        assert row.contract_exact is False   # not all labels found
        assert row.true_positive_classes == ["Reentrancy"]
        assert row.false_negative_classes == ["IntegerOverflow"]


# ═══════════════════════════════════════════════════════════════════════════
# compute_metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    def test_returns_pipeline_metrics(self, contract_eval):
        row = _as_dataclass(contract_eval)
        compute_per_contract([row], {"CONFIRMED"})
        pm = compute_metrics([row], {"CONFIRMED"})
        assert isinstance(pm, PipelineMetrics)

    def test_metrics_values(self, contract_eval):
        row = _as_dataclass(contract_eval)
        compute_per_contract([row], {"CONFIRMED"})
        pm = compute_metrics([row], {"CONFIRMED"})
        assert pm.macro_f1 == pytest.approx(1.0)
        assert pm.macro_fbeta == pytest.approx(1.0)
        assert pm.contract_accuracy_loose == 1.0

    def test_safe_contract_not_affecting_class_support(self, tmp_path):
        rows = [
            _as_dataclass(make_contract_eval("c1", ["A"], {"A": "CONFIRMED"})),
            _as_dataclass(make_contract_eval("c2", [], {}, gt="safe")),
        ]
        compute_per_contract(rows, {"CONFIRMED"})
        pm = compute_metrics(rows, {"CONFIRMED"})
        assert pm.class_metrics["A"].support == 1
        assert pm.contract_accuracy_loose == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# render_markdown
# ═══════════════════════════════════════════════════════════════════════════

class TestRenderMarkdown:
    def test_contains_expected_sections(self, contract_eval):
        row = _as_dataclass(contract_eval)
        compute_per_contract([row], {"CONFIRMED"})
        pm = compute_metrics([row], {"CONFIRMED"})
        md = render_markdown([row], pm, [], {})
        assert "Macro-F1" in md
        assert "Macro-Fbeta" in md
        assert "Per-class metrics" in md
        assert "Contract-level accuracy" in md
        assert "ALL GATES PASS" in md

    def test_contains_baseline_section(self, contract_eval):
        row = _as_dataclass(contract_eval)
        compute_per_contract([row], {"CONFIRMED"})
        pm = compute_metrics([row], {"CONFIRMED"})
        baseline = pm.as_dict()
        md = render_markdown([row], pm, [], baseline)
        assert "Baseline comparison" in md
        assert "macro_F1" in md
        assert "macro_Fbeta" in md


# ═══════════════════════════════════════════════════════════════════════════
# build_metrics_json
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildMetricsJson:
    def test_structure(self, contract_eval):
        row = _as_dataclass(contract_eval)
        compute_per_contract([row], {"CONFIRMED"})
        pm = compute_metrics([row], {"CONFIRMED"})
        d = build_metrics_json([row], pm, [])
        assert "macro_f1" in d
        assert "macro_fbeta" in d
        assert "gates" in d
        assert "per_contract" in d
        assert len(d["per_contract"]) == 1
        assert d["per_contract"][0]["stem"] == "test_contract"

    def test_json_serialisable(self, contract_eval):
        row = _as_dataclass(contract_eval)
        compute_per_contract([row], {"CONFIRMED"})
        pm = compute_metrics([row], {"CONFIRMED"})
        d = build_metrics_json([row], pm, [])
        json.dumps(d)  # must not raise
