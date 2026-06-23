"""
Tests for the WS6a Phase C.2 eval framework (2026-06-22).

Coverage:
  - ClassMetrics: precision/recall/F1 derivation, NaN handling, as_dict
  - PipelineMetrics: per-class aggregation, macro/micro, contract accuracy,
                     positive_verdicts customisation
  - Benchmark: corpus loading with json sidecar + // expect: header fallback,
               missing label handling, path resolution
  - RegressionBaseline: load/save round-trip, compare() pass/fail semantics,
                        per-class regression detection
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import (
    ClassMetrics,
    ContractMetrics,
    PipelineMetrics,
    DEFAULT_POSITIVE_VERDICTS,
    BORDERLINE_BAND,
    metrics_from_contracts,
    Benchmark,
    RegressionBaseline,
    RegressionResult,
)


# ═══════════════════════════════════════════════════════════════════════════
# ClassMetrics
# ═══════════════════════════════════════════════════════════════════════════

class TestClassMetrics:
    def test_perfect_precision_recall_f1(self):
        m = ClassMetrics(cls="Reentrancy", tp=10, fp=0, fn=0, tn=80)
        m.compute()
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_zero_precision(self):
        m = ClassMetrics(cls="X", tp=0, fp=10, fn=0, tn=90)
        m.compute()
        assert m.precision == 0.0
        # recall is NaN (no positive ground truth) — but compute() should
        # not crash, and F1 should be 0.
        assert math.isnan(m.recall)
        assert m.f1 == 0.0

    def test_zero_recall(self):
        m = ClassMetrics(cls="X", tp=0, fp=0, fn=10, tn=90)
        m.compute()
        # precision is NaN (no positive predictions)
        assert math.isnan(m.precision)
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_balanced_f1(self):
        m = ClassMetrics(cls="X", tp=5, fp=5, fn=5, tn=85)
        m.compute()
        # P=R=0.5, F1=0.5
        assert m.precision == 0.5
        assert m.recall == 0.5
        assert m.f1 == 0.5

    def test_as_dict_serialises_nan_as_zero(self):
        m = ClassMetrics(cls="X", tp=0, fp=0, fn=0, tn=10)
        m.compute()
        d = m.as_dict()
        assert d["precision"] == 0.0   # not NaN
        assert d["recall"]    == 0.0
        assert d["f1"]        == 0.0
        # JSON-serialisable (NaN is not valid JSON).
        json.dumps(d)


# ═══════════════════════════════════════════════════════════════════════════
# PipelineMetrics — small synthetic benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def _contract(stem, labels, verdicts, gt="vulnerable"):
    return ContractMetrics(
        stem=stem, report_path=f"/fake/{stem}.json",
        labels=labels, ground_truth=gt, verdicts=verdicts, probabilities={},
    )


class TestPipelineMetrics:
    def test_perfect_run(self):
        contracts = [
            _contract("c1", ["Reentrancy"],       {"Reentrancy": "CONFIRMED"}),
            _contract("c2", ["Reentrancy"],       {"Reentrancy": "CONFIRMED"}),
            _contract("c3", ["ExternalBug"],      {"ExternalBug": "LIKELY"}),
            _contract("c4", [],                   {}, gt="safe"),  # safe baseline
        ]
        pm = metrics_from_contracts(contracts)
        assert pm.macro_f1 == 1.0
        assert pm.micro_f1 == 1.0
        assert pm.contract_accuracy_loose == 1.0
        assert pm.contract_accuracy_exact == 1.0
        assert "Reentrancy" in pm.class_metrics
        assert "ExternalBug" in pm.class_metrics
        assert pm.class_metrics["Reentrancy"].support == 2

    def test_all_missed(self):
        contracts = [
            _contract("c1", ["Reentrancy"], {}),
            _contract("c2", ["Reentrancy"], {}),
        ]
        pm = metrics_from_contracts(contracts)
        # Macro-F1 over Reentrancy: 0/2 → F1=0, over the "always-0" Reentrancy
        # support is 2, but the safe contract is TN... actually for Reentrancy:
        # tp=0, fp=0, fn=2, tn=0 → precision=NaN, recall=0, F1=0.
        assert pm.class_metrics["Reentrancy"].f1 == 0.0
        # Macro averages all supported classes.
        assert pm.macro_f1 == 0.0

    def test_custom_positive_verdicts(self):
        """A run that only counts CONFIRMED (not LIKELY) as positive."""
        contracts = [
            _contract("c1", ["X"], {"X": "LIKELY"}),     # LIKELY ignored
            _contract("c2", ["X"], {"X": "CONFIRMED"}),
        ]
        # With default positives (CONFIRMED+LIKELY): both predicted → 1 TP, 0 FN.
        pm_default = metrics_from_contracts(contracts)
        assert pm_default.class_metrics["X"].tp == 2
        # With CONFIRMED-only: c1 is now a miss → 1 TP, 1 FN.
        pm_strict = metrics_from_contracts(
            contracts, positive_verdicts=frozenset({"CONFIRMED"})
        )
        assert pm_strict.class_metrics["X"].tp == 1
        assert pm_strict.class_metrics["X"].fn == 1
        assert pm_strict.class_metrics["X"].precision == 1.0   # 1 TP / (1 TP + 0 FP)
        assert pm_strict.class_metrics["X"].recall    == 0.5   # 1 TP / (1 TP + 1 FN)

    def test_micro_vs_macro(self):
        """Micro-F1 weights by class support; macro-F1 is unweighted mean."""
        # 10 contracts with class A: 9 hit, 1 miss. support=10.
        # 10 contracts with class B: 1 hit, 9 false alarms. support=1.
        # Class A: 9 TP, 0 FP, 1 FN → P=1.0, R=0.9, F1=2*1*0.9/1.9 ≈ 0.947
        # Class B: 1 TP, 9 FP, 0 FN → P=0.1, R=1.0, F1=2*0.1*1.0/1.1 ≈ 0.182
        # Macro-F1 = (0.947 + 0.182) / 2 ≈ 0.564
        # Micro-F1 = total TP=10, FP=9, FN=1 → P=10/19≈0.526, R=10/11≈0.909
        #   F1 = 2*0.526*0.909/(0.526+0.909) ≈ 0.667
        contracts = []
        for i in range(9):
            contracts.append(_contract(f"a_hit{i}", ["A"], {"A": "CONFIRMED"}))
        contracts.append(_contract("a_missed", ["A"], {}))
        for i in range(9):
            contracts.append(_contract(f"b_fp{i}", [], {"B": "LIKELY"}))
        contracts.append(_contract("b_tp", ["B"], {"B": "CONFIRMED"}))
        pm = metrics_from_contracts(contracts)
        macro = pm.macro_f1
        micro = pm.micro_f1
        # Hand-computed: macro ≈ 0.564
        expected_macro = (2*1.0*0.9/1.9 + 2*0.1*1.0/1.1) / 2
        assert abs(macro - expected_macro) < 0.01
        # Micro ≈ 0.667
        expected_micro = 2*(10/19)*(10/11) / ((10/19) + (10/11))
        assert abs(micro - expected_micro) < 0.01
        # And the relationship: micro > macro when one small class dominates FPs
        assert micro > macro

    def test_as_dict_round_trip(self):
        contracts = [
            _contract("c1", ["A"], {"A": "CONFIRMED"}),
            _contract("c2", [], {}),
        ]
        pm = metrics_from_contracts(contracts)
        d = pm.as_dict()
        # All keys present
        for k in ("contract_count", "macro_f1", "micro_f1", "per_class"):
            assert k in d
        # JSON-serialisable.
        json.dumps(d)

    def test_derive_per_contract_loose_vs_exact(self):
        """Loose: vuln→≥1 correct flag. Strict: predicted set == label set."""
        contracts = [
            _contract("c1", ["A", "B"], {"A": "CONFIRMED"}),     # 1/2 correct
            _contract("c2", ["A"],     {"A": "CONFIRMED"}),     # exact
            _contract("c3", ["A"],     {"A": "LIKELY"}),        # exact
        ]
        pm = PipelineMetrics(contracts)
        pm.derive_per_contract()
        # c1: loose=True (got A), exact=False (missing B)
        assert pm.contracts[0].contract_correct is True
        assert pm.contracts[0].contract_exact is False
        # c2: both true
        assert pm.contracts[1].contract_correct is True
        assert pm.contracts[1].contract_exact is True
        # c3: both true
        assert pm.contracts[2].contract_correct is True
        assert pm.contracts[2].contract_exact is True

    def test_safe_contract_with_no_flag_is_correct(self):
        contracts = [
            _contract("safe1", [], {}, gt="safe"),
            _contract("safe2", [], {"A": "DISPUTED"}, gt="safe"),  # not a positive
        ]
        pm = metrics_from_contracts(contracts)
        assert pm.contract_accuracy_loose == 1.0
        assert pm.contracts[0].contract_correct is True
        assert pm.contracts[1].contract_correct is True

    def test_inconclusive_is_not_positive(self):
        """INCONCLUSIVE must not count as a flag (per WS1.5)."""
        contracts = [
            _contract("c1", ["A"], {"A": "INCONCLUSIVE"}),
        ]
        pm = metrics_from_contracts(contracts)
        # Predicted positive = [] (INCONCLUSIVE not in DEFAULT_POSITIVE_VERDICTS)
        # Class A: tp=0, fp=0, fn=1, tn=0 → F1=0
        assert pm.class_metrics["A"].tp == 0
        assert pm.class_metrics["A"].fn == 1
        assert pm.contracts[0].contract_correct is False  # missed a vuln

    def test_default_positive_verdicts_includes_confirmed_and_likely(self):
        assert "CONFIRMED" in DEFAULT_POSITIVE_VERDICTS
        assert "LIKELY" in DEFAULT_POSITIVE_VERDICTS
        assert "DISPUTED" not in DEFAULT_POSITIVE_VERDICTS
        assert "WATCH" not in DEFAULT_POSITIVE_VERDICTS
        assert "SAFE" not in DEFAULT_POSITIVE_VERDICTS
        assert "INCONCLUSIVE" not in DEFAULT_POSITIVE_VERDICTS

    def test_borderline_band_constant(self):
        # The gate machinery references this; lock in the values.
        assert BORDERLINE_BAND == (0.35, 0.50)


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark — corpus loading
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmark:
    def test_load_with_json_sidecar(self, tmp_path):
        # Create a minimal corpus: one .sol + one .json sidecar
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "vuln.sol").write_text("pragma solidity ^0.8.0;\n")
        (corpus / "vuln.json").write_text(json.dumps({
            "vulnerabilities": ["Reentrancy"],
        }))
        (corpus / "safe.sol").write_text("pragma solidity ^0.8.0;\n")
        (corpus / "safe.json").write_text(json.dumps({
            "vulnerabilities": [],
        }))

        # Create matching reports
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "vuln_report.json").write_text(json.dumps({
            "verdicts": {"Reentrancy": "CONFIRMED"},
            "ml_result": {"probabilities": {"Reentrancy": 0.85}},
            "final_report": {"path_taken": "deep", "verdicts": {"Reentrancy": "CONFIRMED"}},
        }))
        (reports / "safe_report.json").write_text(json.dumps({
            "verdicts": {},
            "ml_result": {"probabilities": {}},
            "final_report": {"path_taken": "fast", "verdicts": {}},
        }))

        bench = Benchmark(name="test", corpus_dir=corpus, reports_dir=reports)
        contracts = bench.load()
        assert len(contracts) == 2
        assert contracts[0].stem in ("vuln", "safe")
        assert contracts[0].ground_truth in ("vulnerable", "safe")

    def test_load_with_expect_header_fallback(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # // expect: header in the .sol, no .json sidecar
        (corpus / "vuln.sol").write_text(
            "// expect: Reentrancy\n"
            "pragma solidity ^0.8.0;\n"
            "contract C {}\n"
        )
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "vuln_report.json").write_text(json.dumps({
            "verdicts": {"Reentrancy": "CONFIRMED"},
            "ml_result": {"probabilities": {"Reentrancy": 0.85}},
            "final_report": {"path_taken": "deep", "verdicts": {"Reentrancy": "CONFIRMED"}},
        }))
        bench = Benchmark(name="test", corpus_dir=corpus, reports_dir=reports)
        contracts = bench.load()
        assert len(contracts) == 1
        assert contracts[0].labels == ["Reentrancy"]
        assert contracts[0].ground_truth == "vulnerable"

    def test_skips_reports_without_labels(self, tmp_path):
        """A report with no matching .json sidecar AND no `// expect:` header
        is silently dropped (can't evaluate it)."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "no_labels.sol").write_text(
            "pragma solidity ^0.8.0;\n"
            "contract C {}\n"  # no // expect: header
        )
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "no_labels_report.json").write_text(json.dumps({
            "verdicts": {"X": "CONFIRMED"},
            "ml_result": {},
            "final_report": {"path_taken": "deep"},
        }))
        bench = Benchmark(name="test", corpus_dir=corpus, reports_dir=reports)
        contracts = bench.load()
        assert len(contracts) == 0

    def test_excludes_manifest_json(self, tmp_path):
        """Top-level manifest files (tier_a_manifest.json, etc.) aren't labels."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "tier_a_manifest.json").write_text(json.dumps({"meta": True}))
        (corpus / "contamination_check.json").write_text(json.dumps({}))
        (corpus / "real.sol").write_text("pragma solidity ^0.8.0;\n")
        (corpus / "real.json").write_text(json.dumps({"vulnerabilities": ["X"]}))
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "real_report.json").write_text(json.dumps({
            "verdicts": {"X": "CONFIRMED"},
            "ml_result": {},
            "final_report": {"path_taken": "deep"},
        }))
        bench = Benchmark(name="test", corpus_dir=corpus, reports_dir=reports)
        # Manifest files don't have matching reports, so 0 contracts.
        # But also doesn't crash on the manifest sidecar present.
        contracts = bench.load()
        assert len(contracts) == 1
        assert contracts[0].stem == "real"

    def test_missing_corpus_dir_raises(self, tmp_path):
        bench = Benchmark(
            name="test",
            corpus_dir=tmp_path / "nonexistent",
            reports_dir=tmp_path / "reports",
        )
        (tmp_path / "reports").mkdir()
        with pytest.raises(FileNotFoundError):
            bench.load()

    def test_missing_reports_dir_raises(self, tmp_path):
        bench = Benchmark(
            name="test",
            corpus_dir=tmp_path / "corpus",
            reports_dir=tmp_path / "nonexistent",
        )
        (tmp_path / "corpus").mkdir()
        with pytest.raises(FileNotFoundError):
            bench.load()


# ═══════════════════════════════════════════════════════════════════════════
# RegressionBaseline
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressionBaseline:
    def test_load_existing_pre_redesign(self):
        """The existing pre_redesign.json must be loadable as-is."""
        # The baseline was written by the WS0 script and matches the format
        # PipelineMetrics.as_dict() produces. Verify round-trip.
        path = Path("eval/baselines/pre_redesign.json")
        if not path.is_file():
            pytest.skip("pre_redesign.json not present in this checkout")
        baseline = RegressionBaseline.load(path)
        assert baseline.macro_f1 > 0.0
        assert baseline.contract_count == 88
        assert "Reentrancy" in baseline.per_class
        # Per-class F1 values are valid floats.
        for cls, m in baseline.per_class.items():
            assert isinstance(m.get("f1"), (int, float))
            assert 0.0 <= m["f1"] <= 1.0

    def test_save_then_load_round_trip(self, tmp_path):
        contracts = [
            _contract("c1", ["A"], {"A": "CONFIRMED"}),
            _contract("c2", ["A"], {"A": "LIKELY"}),
        ]
        pm = metrics_from_contracts(contracts)
        baseline = RegressionBaseline.from_metrics(pm)
        out_path = tmp_path / "baseline.json"
        baseline.save(out_path)
        loaded = RegressionBaseline.load(out_path)
        assert loaded.macro_f1 == pm.macro_f1
        assert loaded.contract_count == 2

    def test_compare_regression(self, tmp_path):
        """A run with worse macro-F1 is flagged as regressed."""
        # Baseline: 2 perfect predictions
        base_contracts = [
            _contract("c1", ["A"], {"A": "CONFIRMED"}),
            _contract("c2", ["A"], {"A": "CONFIRMED"}),
        ]
        base_pm = metrics_from_contracts(base_contracts)
        baseline = RegressionBaseline.from_metrics(base_pm)
        # Current: same contracts but both miss A → macro F1 = 0
        cur_contracts = [
            _contract("c1", ["A"], {}),
            _contract("c2", ["A"], {}),
        ]
        cur_pm = metrics_from_contracts(cur_contracts)
        result = baseline.compare(cur_pm)
        assert result.regressed is True
        assert result.current_macro_f1 < result.baseline_macro_f1
        assert "A" in result.regressed_classes

    def test_compare_improvement(self, tmp_path):
        """A run with better macro-F1 is NOT regressed; class is in improved_classes."""
        base_contracts = [
            _contract("c1", ["A"], {}),                 # miss
            _contract("c2", ["A"], {"A": "CONFIRMED"}),  # hit
        ]
        baseline = RegressionBaseline.from_metrics(metrics_from_contracts(base_contracts))
        cur_contracts = [
            _contract("c1", ["A"], {"A": "CONFIRMED"}),  # now hit
            _contract("c2", ["A"], {"A": "CONFIRMED"}),  # still hit
        ]
        result = baseline.compare(metrics_from_contracts(cur_contracts))
        assert result.regressed is False
        assert "A" in result.improved_classes

    def test_compare_no_change(self, tmp_path):
        contracts = [
            _contract("c1", ["A"], {"A": "CONFIRMED"}),
        ]
        baseline = RegressionBaseline.from_metrics(metrics_from_contracts(contracts))
        result = baseline.compare(metrics_from_contracts(contracts))
        assert result.regressed is False
        assert result.regressed_classes == []
        assert result.improved_classes == []
        assert abs(result.baseline_macro_f1 - result.current_macro_f1) < 1e-9

    def test_compare_class_added_in_current(self, tmp_path):
        """A class appearing only in the current run shows up with baseline=0."""
        base_contracts = [
            _contract("c1", ["A"], {"A": "CONFIRMED"}),
        ]
        baseline = RegressionBaseline.from_metrics(metrics_from_contracts(base_contracts))
        cur_contracts = [
            _contract("c1", ["A"], {"A": "CONFIRMED"}),
            _contract("c2", ["B"], {"B": "CONFIRMED"}),
        ]
        result = baseline.compare(metrics_from_contracts(cur_contracts))
        # B didn't exist in baseline → per_class_deltas has it with baseline=0
        assert "B" in result.per_class_deltas
        assert result.per_class_deltas["B"]["baseline_f1"] == 0.0
        assert result.per_class_deltas["B"]["current_f1"] == 1.0

    def test_compare_min_delta_threshold(self, tmp_path):
        """Below `min_delta_pp`, a class isn't reported as regressed/improved."""
        base_contracts = [
            _contract("c1", ["A"], {"A": "CONFIRMED"}),
        ]
        baseline = RegressionBaseline.from_metrics(metrics_from_contracts(base_contracts))
        # Same prediction → delta=0
        result = baseline.compare(
            metrics_from_contracts(base_contracts), min_delta_pp=0.01
        )
        assert result.regressed_classes == []
        assert result.improved_classes == []
