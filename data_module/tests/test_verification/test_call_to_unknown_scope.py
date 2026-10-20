"""Regression contract for the R4-D-016 CallToUnknown semantic clarification."""

from unittest.mock import MagicMock

import torch

from sentinel_data.verification.semantic_checker import CheckVerdict, _check_class


def _v10_graph(edge_id: int):
    graph = MagicMock()
    graph.graph_schema_version = "v10"
    graph.edge_attr = torch.tensor([edge_id], dtype=torch.long)
    graph.x = torch.zeros((1, 12), dtype=torch.float32)
    return graph


def test_low_level_call_is_call_to_unknown_signal():
    verdict, _ = _check_class("CallToUnknown", _v10_graph(12), None)
    assert verdict == CheckVerdict.PASS


def test_send_is_call_to_unknown_signal():
    verdict, _ = _check_class("CallToUnknown", _v10_graph(14), None)
    assert verdict == CheckVerdict.PASS


def test_transfer_is_not_call_to_unknown_signal():
    verdict, _ = _check_class("CallToUnknown", _v10_graph(13), None)
    assert verdict == CheckVerdict.FAIL


def test_typed_high_level_call_is_not_call_to_unknown_signal():
    verdict, _ = _check_class("CallToUnknown", _v10_graph(11), None)
    assert verdict == CheckVerdict.FAIL


def test_library_call_is_not_call_to_unknown_signal():
    verdict, _ = _check_class("CallToUnknown", _v10_graph(15), None)
    assert verdict == CheckVerdict.FAIL


def test_contract_creation_is_not_call_to_unknown_signal():
    verdict, _ = _check_class("CallToUnknown", _v10_graph(16), None)
    assert verdict == CheckVerdict.FAIL
