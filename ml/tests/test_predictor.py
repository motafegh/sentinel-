"""Regression tests for the predictor tier-threshold fix (F8/F10).

Per audit finding: _format_result() was using scalar tier_confirmed_threshold
(0.55) instead of per-class self.thresholds[cls_idx] for the CONFIRMED tier.
This meant a class tuned to 0.90 would trigger CONFIRMED at 0.56 (wrong),
and a class tuned to 0.35 would NOT trigger at 0.36 (wrong in the other dir).

These tests use a mock predictor (no GPU, no real model weights) to verify
the fix.
"""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data
from ml.src.inference.predictor import Predictor as VulnerabilityPredictor


def _stub_graph(n_nodes: int = 5, n_edges: int = 4) -> Data:
    return Data(
        x=torch.randn(n_nodes, 12),
        edge_index=torch.zeros(2, n_edges, dtype=torch.long),
        edge_attr=torch.zeros(n_edges, dtype=torch.long),
    )


def _make_mock_predictor(per_class_thresholds: list[float]) -> VulnerabilityPredictor:
    """Build a VulnerabilityPredictor with mocked model + thresholds.

    Bypasses the checkpoint-loading __init__ by constructing the object
    directly and injecting the minimal state needed by _format_result().
    """
    obj = object.__new__(VulnerabilityPredictor)

    # Minimal state that _format_result uses
    obj.device = "cpu"
    obj.threshold = 0.50
    obj.tier_confirmed_threshold = VulnerabilityPredictor.TIER_CONFIRMED_THRESHOLD  # 0.55 (old scalar)
    obj.tier_suspicious_threshold = VulnerabilityPredictor.TIER_SUSPICIOUS_THRESHOLD  # 0.25
    obj.thresholds = torch.tensor(per_class_thresholds, dtype=torch.float32)
    obj.thresholds_loaded = True
    obj.num_classes = len(per_class_thresholds)

    # Class names matching the 10-class locked order
    obj._class_names = [
        "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
        "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
        "TransactionOrderDependence", "UnusedReturn",
    ]
    assert len(obj._class_names) == len(per_class_thresholds)

    return obj


def test_high_tuned_class_does_not_trigger_at_old_scalar():
    """A class with tuned threshold 0.90 must NOT trigger CONFIRMED at 0.56.

    With the OLD scalar (0.55), prob=0.56 would incorrectly trigger CONFIRMED.
    With the fix (per-class), it should be SUSPICIOUS (0.56 >= 0.25 susp_thr).
    """
    # CallToUnknown = class 0, tuned to 0.90
    thresholds = [0.90] + [0.50] * 9
    pred = _make_mock_predictor(thresholds)

    # prob[0]=0.56: above old scalar (0.55) but below tuned (0.90)
    probs = torch.tensor([0.56] + [0.0] * 9)
    result = pred._format_result(graph=_stub_graph(), probs=probs, tokens={}, windows_used=1)

    confirmed_classes = [e["vulnerability_class"] for e in result["confirmed"]]
    suspicious_classes = [e["vulnerability_class"] for e in result["suspicious"]]

    assert "CallToUnknown" not in confirmed_classes, (
        "CallToUnknown (tuned=0.90) must NOT be CONFIRMED at prob=0.56"
    )
    assert "CallToUnknown" in suspicious_classes, (
        "CallToUnknown should be SUSPICIOUS (0.56 >= susp_thr 0.25)"
    )


def test_high_tuned_class_triggers_above_tuned_threshold():
    """A class with tuned threshold 0.90 MUST trigger CONFIRMED at prob=0.92."""
    thresholds = [0.90] + [0.50] * 9
    pred = _make_mock_predictor(thresholds)

    probs = torch.tensor([0.92] + [0.0] * 9)
    result = pred._format_result(graph=_stub_graph(), probs=probs, tokens={}, windows_used=1)

    confirmed_classes = [e["vulnerability_class"] for e in result["confirmed"]]
    assert "CallToUnknown" in confirmed_classes, (
        "CallToUnknown (tuned=0.90) must be CONFIRMED at prob=0.92"
    )


def test_low_tuned_class_triggers_below_old_scalar():
    """A class with tuned threshold 0.35 MUST trigger CONFIRMED at prob=0.36.

    With the OLD scalar (0.55), prob=0.36 would be SUSPICIOUS only.
    With the fix (per-class), it should be CONFIRMED (0.36 >= 0.35).
    """
    # Reentrancy = class 6, tuned to 0.35
    thresholds = [0.50] * 6 + [0.35] + [0.50] * 3
    pred = _make_mock_predictor(thresholds)

    probs = torch.tensor([0.0] * 6 + [0.36] + [0.0] * 3)
    result = pred._format_result(graph=_stub_graph(), probs=probs, tokens={}, windows_used=1)

    confirmed_classes = [e["vulnerability_class"] for e in result["confirmed"]]
    assert "Reentrancy" in confirmed_classes, (
        "Reentrancy (tuned=0.35) must be CONFIRMED at prob=0.36"
    )


def test_different_thresholds_per_class():
    """Each class uses its own threshold independently."""
    # Class 0 (CallToUnknown): 0.80 — will NOT trigger at 0.70
    # Class 6 (Reentrancy):    0.40 — WILL trigger at 0.45
    thresholds = [0.80] + [0.50] * 5 + [0.40] + [0.50] * 3
    pred = _make_mock_predictor(thresholds)

    probs = torch.tensor([0.70] + [0.0] * 5 + [0.45] + [0.0] * 3)
    result = pred._format_result(graph=_stub_graph(), probs=probs, tokens={}, windows_used=1)

    confirmed_classes = {e["vulnerability_class"] for e in result["confirmed"]}
    suspicious_classes = {e["vulnerability_class"] for e in result["suspicious"]}

    assert "CallToUnknown" not in confirmed_classes
    assert "CallToUnknown" in suspicious_classes   # 0.70 >= 0.25 susp_thr
    assert "Reentrancy" in confirmed_classes


def test_no_confirmed_when_all_below_tuned():
    """No CONFIRMED entries when all probs are below their tuned thresholds."""
    thresholds = [0.90] * 10
    pred = _make_mock_predictor(thresholds)

    probs = torch.tensor([0.60] * 10)  # all below 0.90
    result = pred._format_result(graph=_stub_graph(), probs=probs, tokens={}, windows_used=1)

    assert result["confirmed"] == []
    assert len(result["suspicious"]) == 10  # all above 0.25 susp_thr


def test_probabilities_always_present():
    """The full 10-class probabilities dict is always in the result."""
    thresholds = [0.50] * 10
    pred = _make_mock_predictor(thresholds)
    probs = torch.zeros(10)
    result = pred._format_result(graph=_stub_graph(), probs=probs, tokens={}, windows_used=1)
    assert set(result["probabilities"].keys()) == set(pred._class_names)
