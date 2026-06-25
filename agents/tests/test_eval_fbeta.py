"""
Tests for F-beta (beta=2) in PipelineMetrics (P0.1 T0.1.1).

Asserts:
  - F-beta formula correct on a known confusion matrix
  - beta=1 equals F1 (regression)
  - beta=2 weights recall
  - macro_fbeta = NaN-aware mean over classes with support>0
  - as_dict() includes fbeta + macro_fbeta
"""

from __future__ import annotations

import math

import pytest

from src.eval.pipeline_metrics import (
    ClassMetrics,
    ContractMetrics,
    PipelineMetrics,
    metrics_from_contracts,
)


def fbeta_formula(p: float, r: float, beta: float) -> float:
    """Reference F-beta implementation."""
    if math.isnan(p) or math.isnan(r) or (p + r) == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0


class TestClassMetricsFbeta:
    def test_fbeta_beta1_equals_f1(self):
        """F-beta with beta=1 should equal F1 exactly."""
        cm = ClassMetrics(cls="Test", tp=10, fp=2, fn=3, tn=100)
        cm.compute(beta=1.0)
        assert cm.fbeta == pytest.approx(cm.f1, abs=1e-10)
        assert cm.fbeta == pytest.approx(fbeta_formula(cm.precision, cm.recall, 1.0), abs=1e-6)

    def test_fbeta_beta2_weights_recall(self):
        """F2 weights recall 4x more than precision. Lower precision should
        hurt less than lower recall."""
        cm_high_precision = ClassMetrics(cls="A", tp=8, fp=1, fn=2, tn=100)
        cm_high_precision.compute(beta=2.0)
        cm_high_recall = ClassMetrics(cls="B", tp=8, fp=2, fn=1, tn=100)
        cm_high_recall.compute(beta=2.0)

        # When recall is higher (fn=1 vs fn=2), F2 should be higher even if precision is slightly lower
        assert cm_high_recall.fbeta > cm_high_precision.fbeta

    def test_fbeta_formula_known_values(self):
        """Known confusion matrix: tp=15, fp=5, fn=10, tn=50
        precision = 15/20 = 0.75, recall = 15/25 = 0.60
        F1 = 2*0.75*0.60/(0.75+0.60) = 0.6666...
        F2 = 5*0.75*0.60/(4*0.75+0.60) = 2.25/(3.0+0.60) = 0.625
        """
        cm = ClassMetrics(cls="Known", tp=15, fp=5, fn=10, tn=50)
        cm.compute(beta=2.0)
        assert cm.precision == pytest.approx(0.75)
        assert cm.recall == pytest.approx(0.60)
        assert cm.f1 == pytest.approx(2 * 0.75 * 0.60 / (0.75 + 0.60), abs=1e-6)
        expected_f2 = 5 * 0.75 * 0.60 / (4 * 0.75 + 0.60)
        assert cm.fbeta == pytest.approx(expected_f2, abs=1e-6)

    def test_fbeta_no_positives_defaults_zero(self):
        """No tp/fp/fn should give precision=NaN, recall=NaN, fbeta=0.0"""
        cm = ClassMetrics(cls="Empty", tp=0, fp=0, fn=0, tn=100)
        cm.compute(beta=2.0)
        assert math.isnan(cm.precision)
        assert math.isnan(cm.recall)
        assert cm.fbeta == 0.0

    def test_fbeta_no_support_returns_zero(self):
        """support=0, no tp/fp/fn/tn — all zeros."""
        cm = ClassMetrics(cls="NoSupport", tp=0, fp=0, fn=0, tn=0)
        cm.compute(beta=2.0)
        assert cm.fbeta == 0.0

    def test_as_dict_includes_fbeta(self):
        cm = ClassMetrics(cls="DictTest", tp=5, fp=2, fn=3, tn=50)
        cm.compute(beta=2.0)
        d = cm.as_dict()
        assert "fbeta" in d
        assert d["fbeta"] > 0.0


class TestPipelineMetricsFbeta:
    def test_macro_fbeta_is_nan_aware(self):
        """macro_fbeta should only average classes with support>0."""
        contracts = [
            ContractMetrics(
                stem="c1", report_path="", labels=["A"],
                ground_truth="vulnerable", verdicts={"A": "CONFIRMED"},
                probabilities={},
            ),
            ContractMetrics(
                stem="c2", report_path="", labels=["B"],
                ground_truth="vulnerable", verdicts={"B": "CONFIRMED"},
                probabilities={},
            ),
            ContractMetrics(
                stem="c3", report_path="", labels=[],  # safe — no support for any class
                ground_truth="safe", verdicts={},
                probabilities={},
            ),
        ]
        pm = metrics_from_contracts(contracts)
        assert "macro_fbeta" in pm.as_dict()
        assert pm.macro_fbeta > 0.0
        # Classes A and B both have support > 0, but there is no class C with support=0
        # so macro_fbeta should be (F2_A + F2_B) / 2
        assert pm.macro_fbeta == pytest.approx(
            (pm.class_metrics["A"].fbeta + pm.class_metrics["B"].fbeta) / 2, abs=1e-6
        )

    def test_as_dict_includes_macro_fbeta(self):
        contracts = [
            ContractMetrics(
                stem="c1", report_path="", labels=["A"],
                ground_truth="vulnerable", verdicts={"A": "CONFIRMED"},
                probabilities={},
            ),
        ]
        pm = metrics_from_contracts(contracts)
        d = pm.as_dict()
        assert "macro_fbeta" in d
        assert d["macro_fbeta"] > 0.0

    def test_perfect_predictions(self):
        """Perfect predictions: all vulnerable contracts flagged correctly,
        safe contracts not flagged."""
        contracts = [
            ContractMetrics(
                stem="c1", report_path="", labels=["Reentrancy"],
                ground_truth="vulnerable", verdicts={"Reentrancy": "CONFIRMED"},
                probabilities={},
            ),
            ContractMetrics(
                stem="c2", report_path="", labels=["IntegerUO"],
                ground_truth="vulnerable", verdicts={"IntegerUO": "CONFIRMED"},
                probabilities={},
            ),
            ContractMetrics(
                stem="c3", report_path="", labels=[],
                ground_truth="safe", verdicts={},
                probabilities={},
            ),
        ]
        pm = metrics_from_contracts(contracts)
        assert pm.macro_f1 == pytest.approx(1.0)
        assert pm.macro_fbeta == pytest.approx(1.0)

    def test_zero_predictions(self):
        """No positive predictions at all: P=NaN → F1=0, Fbeta=0"""
        contracts = [
            ContractMetrics(
                stem="c1", report_path="", labels=["A"],
                ground_truth="vulnerable", verdicts={},
                probabilities={},
            ),
        ]
        pm = metrics_from_contracts(contracts)
        assert pm.macro_f1 == 0.0
        assert pm.macro_fbeta == 0.0
