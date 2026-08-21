"""Tests for feature-based semantic_checker (Stage 4)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentinel_data.verification.semantic_checker import (
    CheckVerdict, run_semantic_check, _check_class, _is_pre_08,
    _has_external_call_edge, SemanticCheckResult,
)

_DATA_DIR = Path("data_module/data")
_MERGED_DIR = _DATA_DIR / "labels" / "merged"


def _skip_if_no_merged():
    if not _MERGED_DIR.exists() or not any(_MERGED_DIR.glob("*.labels.json")):
        pytest.skip("Merged labels not found — run merger first")


def _mock_graph(
    has_cei=1,
    feat2=0.0,
    feat7=0.0,
    feat11=0.0,
    has_ext_call=False,
    graph_schema_version="v9",
    edge_attr=None,
):
    """Create a minimal mock graph."""
    import torch
    g = MagicMock()
    g.has_cei_path = has_cei
    x = torch.zeros(5, 12)
    x[:, 2] = feat2
    x[:, 7] = feat7
    x[:, 11] = feat11
    g.x = x
    g.graph_schema_version = graph_schema_version
    if edge_attr is not None:
        g.edge_attr = torch.tensor(edge_attr)
    elif has_ext_call:
        g.edge_attr = torch.tensor([0, 11, 5])
    else:
        g.edge_attr = torch.tensor([0, 5, 6])
    return g


class TestCheckClass:
    def test_reentrancy_pass(self):
        g = _mock_graph(has_cei=1)
        verdict, _ = _check_class("Reentrancy", g, None)
        assert verdict == CheckVerdict.PASS

    def test_reentrancy_fail(self):
        g = _mock_graph(has_cei=0)
        verdict, _ = _check_class("Reentrancy", g, None)
        assert verdict == CheckVerdict.FAIL

    def test_reentrancy_skip_no_graph(self):
        verdict, _ = _check_class("Reentrancy", None, None)
        assert verdict == CheckVerdict.SKIP

    def test_timestamp_pass(self):
        g = _mock_graph(feat2=1.0)
        verdict, _ = _check_class("Timestamp", g, None)
        assert verdict == CheckVerdict.PASS

    def test_timestamp_fail(self):
        g = _mock_graph(feat2=0.0)
        verdict, _ = _check_class("Timestamp", g, None)
        assert verdict == CheckVerdict.FAIL

    def test_integer_uo_pass_via_unchecked_block(self):
        g = _mock_graph(feat11=1.0)
        verdict, _ = _check_class("IntegerUO", g, {"solc_version": "0.8.0"})
        assert verdict == CheckVerdict.PASS

    def test_integer_uo_pass_via_pre08_pragma(self):
        verdict, note = _check_class("IntegerUO", None, {"solc_version": "0.5.10"})
        assert verdict == CheckVerdict.PASS
        assert "pre-0.8" in note

    def test_integer_uo_fail_08_no_unchecked(self):
        g = _mock_graph(feat11=0.0)
        verdict, _ = _check_class("IntegerUO", g, {"solc_version": "0.8.0"})
        assert verdict == CheckVerdict.FAIL

    def test_unused_return_pass(self):
        g = _mock_graph(feat7=1.0)
        verdict, _ = _check_class("UnusedReturn", g, None)
        assert verdict == CheckVerdict.PASS

    def test_mishandled_exception_pass(self):
        g = _mock_graph(feat7=1.0)
        verdict, _ = _check_class("MishandledException", g, None)
        assert verdict == CheckVerdict.PASS

    def test_call_to_unknown_v9_not_extractable(self):
        g = _mock_graph(has_ext_call=True)
        verdict, _ = _check_class("CallToUnknown", g, None)
        assert verdict == CheckVerdict.NOT_EXTRACTABLE

    def test_call_to_unknown_v10_low_level_pass(self):
        g = _mock_graph(graph_schema_version="v10", edge_attr=[0, 12, 5])
        verdict, _ = _check_class("CallToUnknown", g, None)
        assert verdict == CheckVerdict.PASS

    def test_call_to_unknown_v10_library_only_fails(self):
        g = _mock_graph(graph_schema_version="v10", edge_attr=[0, 15, 5])
        verdict, _ = _check_class("CallToUnknown", g, None)
        assert verdict == CheckVerdict.FAIL

    def test_external_bug_not_extractable(self):
        g = _mock_graph(graph_schema_version="v10", edge_attr=[11, 12, 13, 14])
        verdict, _ = _check_class("ExternalBug", g, None)
        assert verdict == CheckVerdict.NOT_EXTRACTABLE

    def test_graph_sidecar_schema_mismatch_fails_closed(self):
        g = _mock_graph(graph_schema_version="v10", edge_attr=[12])
        with pytest.raises(ValueError, match="graph/sidecar schema mismatch"):
            _check_class("CallToUnknown", g, {"schema_version": "v9"})

    @pytest.mark.parametrize("class_name", ["CallToUnknown", "Reentrancy"])
    def test_v10_parse_only_is_not_misread_as_absent_signal(self, class_name):
        g = _mock_graph(graph_schema_version="v10", edge_attr=[])
        verdict, note = _check_class(
            class_name,
            g,
            {
                "schema_version": "v10",
                "graph_analysis_degraded": True,
                "graph_extraction_mode": "slither_parse_only",
            },
        )
        assert verdict == CheckVerdict.NOT_EXTRACTABLE
        assert "parse-only" in note

    def test_dos_not_extractable(self):
        g = _mock_graph()
        verdict, _ = _check_class("DenialOfService", g, None)
        assert verdict == CheckVerdict.NOT_EXTRACTABLE

    def test_gas_exception_not_extractable(self):
        g = _mock_graph()
        verdict, _ = _check_class("GasException", g, None)
        assert verdict == CheckVerdict.NOT_EXTRACTABLE

    def test_tod_not_extractable(self):
        g = _mock_graph()
        verdict, _ = _check_class("TransactionOrderDependence", g, None)
        assert verdict == CheckVerdict.NOT_EXTRACTABLE


class TestHelpers:
    def test_is_pre_08_true(self):
        assert _is_pre_08({"solc_version": "0.5.10"}) is True
        assert _is_pre_08({"solc_version": "0.7.6"}) is True
        assert _is_pre_08({"solc_version": "^0.4.21"}) is True

    def test_is_pre_08_false(self):
        assert _is_pre_08({"solc_version": "0.8.0"}) is False
        assert _is_pre_08({"solc_version": "0.8.17"}) is False

    def test_is_pre_08_empty(self):
        assert _is_pre_08({}) is False

    def test_has_external_call_edge_true(self):
        import torch
        g = MagicMock()
        g.edge_attr = torch.tensor([0, 11, 5])
        assert _has_external_call_edge(g) is True

    def test_has_external_call_edge_false(self):
        import torch
        g = MagicMock()
        g.edge_attr = torch.tensor([0, 5, 6])
        assert _has_external_call_edge(g) is False


class TestRunSemanticCheck:
    def test_smoke_limit1(self):
        _skip_if_no_merged()
        result = run_semantic_check(_DATA_DIR, limit_per_class=1)
        assert isinstance(result, SemanticCheckResult)
        assert len(result.by_class) == 10

    def test_all_classes_in_result(self):
        _skip_if_no_merged()
        from sentinel_data.labeling.schema import class_names
        result = run_semantic_check(_DATA_DIR, limit_per_class=2)
        for cls in class_names():
            assert cls in result.by_class

    def test_solidifi_reentrancy_mostly_pass(self):
        """SolidiFI Reentrancy is T0 injection-verified — should have high pass rate."""
        _skip_if_no_merged()
        solidifi_reps = _DATA_DIR / "representations" / "solidifi"
        if not solidifi_reps.exists():
            pytest.skip("SolidiFI reps not found")
        result = run_semantic_check(_DATA_DIR, limit_per_class=50)
        reen = result.by_class["Reentrancy"]
        checkable = reen.pass_count + reen.fail_count
        if checkable > 0:
            # SolidiFI injection-verified: expect high pass rate
            assert reen.pass_count / checkable >= 0.60, (
                f"SolidiFI Reentrancy pass rate too low: {reen.pass_count}/{checkable}"
            )
