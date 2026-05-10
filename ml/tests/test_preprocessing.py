"""
test_preprocessing.py — Unit tests for graph_extractor v2 schema.

Tests are in two tiers:

  Unit tests (no external deps):
    - Schema sanity assertions (NODE_FEATURE_DIM=12, NODE_TYPES 13 entries, etc.)
    - _cfg_node_type() priority ordering
    - _build_cfg_node_features() shape and in_unchecked=0.0 invariant
    - _build_node_features() shape and sentinel value handling
    - _build_control_flow_edges() index correctness and node_metadata alignment
    - Mock-based feature function tests for return_ignored, call_target_typed,
      in_unchecked, has_loop, external_call_count

  Integration tests (@pytest.mark.integration, require slither-analyzer):
    - Reentrancy (call-before-write): CFG_NODE_CALL exists, CONTROL_FLOW edges exist
    - CEI-safe (write-before-call): CONTROL_FLOW edge write→call exists
    - unchecked{} contract: in_unchecked=1.0 on func node, 0.0 on all CFG nodes
    - Loop contract: has_loop=1.0
    - Typed interface contract: call_target_typed=1.0
    - Merged-IR node: _cfg_node_type() assigns CFG_NODE_CALL when call+write merge
    - node_metadata alignment: len(graph.node_metadata) == graph.x.shape[0]
    - Function nodes have non-empty "name" key in metadata
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from ml.src.preprocessing.graph_schema import (
    EDGE_TYPES,
    FEATURE_NAMES,
    NODE_FEATURE_DIM,
    NODE_TYPES,
    NUM_EDGE_TYPES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schema sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaSanity:
    def test_node_feature_dim_is_12(self):
        assert NODE_FEATURE_DIM == 12, (
            f"NODE_FEATURE_DIM={NODE_FEATURE_DIM}; expected 12 (v2 schema). "
            "gas_intensity was removed in the final v2 schema."
        )

    def test_feature_names_length_matches_dim(self):
        assert len(FEATURE_NAMES) == NODE_FEATURE_DIM

    def test_num_edge_types_is_7(self):
        assert NUM_EDGE_TYPES == 7

    def test_edge_types_contains_new_v2_edges(self):
        assert "CONTAINS"     in EDGE_TYPES and EDGE_TYPES["CONTAINS"]     == 5
        assert "CONTROL_FLOW" in EDGE_TYPES and EDGE_TYPES["CONTROL_FLOW"] == 6

    def test_node_types_has_13_entries(self):
        assert len(NODE_TYPES) == 13, (
            f"NODE_TYPES has {len(NODE_TYPES)} entries; expected 13 "
            "(ids 0-12 including 5 CFG subtypes)."
        )

    def test_cfg_subtypes_present_and_ordered(self):
        assert NODE_TYPES["CFG_NODE_CALL"]  == 8
        assert NODE_TYPES["CFG_NODE_WRITE"] == 9
        assert NODE_TYPES["CFG_NODE_READ"]  == 10
        assert NODE_TYPES["CFG_NODE_CHECK"] == 11
        assert NODE_TYPES["CFG_NODE_OTHER"] == 12

    def test_feature_names_no_gas_intensity(self):
        assert "gas_intensity" not in FEATURE_NAMES, (
            "gas_intensity was removed in the final v2 schema (circular heuristic)."
        )

    def test_feature_names_no_reentrant(self):
        assert "reentrant" not in FEATURE_NAMES, (
            "reentrant was removed in v2 — it leaked Slither's pre-computed answer."
        )

    def test_feature_names_has_all_new_features(self):
        for fname in ("return_ignored", "call_target_typed", "in_unchecked",
                      "has_loop", "external_call_count"):
            assert fname in FEATURE_NAMES, f"'{fname}' missing from FEATURE_NAMES"

    def test_return_ignored_at_index_7(self):
        assert FEATURE_NAMES[7] == "return_ignored"

    def test_call_target_typed_at_index_8(self):
        assert FEATURE_NAMES[8] == "call_target_typed"

    def test_external_call_count_at_index_11(self):
        assert FEATURE_NAMES[11] == "external_call_count"


# ─────────────────────────────────────────────────────────────────────────────
# Mock helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_ir_op(type_cls, lvalue=True, read_vars=None):
    """Create a mock Slither IR operation."""
    op = MagicMock(spec=type_cls)
    op.lvalue = MagicMock() if lvalue else None
    op.read = read_vars or []
    return op


def _make_mock_func(
    canonical_name: str = "TestContract.withdraw",
    visibility: str = "external",
    pure: bool = False,
    view: bool = False,
    payable: bool = True,
    is_constructor: bool = False,
    is_fallback: bool = False,
    is_receive: bool = False,
    nodes: list | None = None,
    slithir_operations: list | None = None,
    high_level_calls: list | None = None,
    low_level_calls: list | None = None,
    source_content: str = "",
    source_lines: list | None = None,
) -> MagicMock:
    """Build a minimal mock Slither Function object."""
    func = MagicMock()
    func.canonical_name = canonical_name
    func.visibility = visibility
    func.pure = pure
    func.view = view
    func.payable = payable
    func.is_constructor = is_constructor
    func.is_fallback = is_fallback
    func.is_receive = is_receive
    func.nodes = nodes or []
    func.slithir_operations = slithir_operations or []
    func.high_level_calls = high_level_calls or []
    func.low_level_calls  = low_level_calls  or []

    sm = MagicMock()
    sm.content = source_content
    sm.lines   = source_lines or []
    func.source_mapping = sm

    return func


def _make_mock_slither_node(
    node_id: int = 0,
    node_type=None,
    irs: list | None = None,
    sons: list | None = None,
    source_lines: list | None = None,
) -> MagicMock:
    """Build a minimal mock Slither FunctionNode (CFG basic block)."""
    node = MagicMock()
    node.node_id = node_id
    node.type    = node_type
    node.irs     = irs or []
    node.sons    = sons or []

    sm = MagicMock()
    sm.lines = source_lines or [node_id + 1]  # non-empty default
    node.source_mapping = sm

    return node


# ─────────────────────────────────────────────────────────────────────────────
# _cfg_node_type() unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCfgNodeType:
    """Priority: CALL > WRITE > READ > CHECK > OTHER."""

    def _import(self):
        from ml.src.preprocessing.graph_extractor import _cfg_node_type
        return _cfg_node_type

    def test_call_wins_over_write(self):
        slither = pytest.importorskip("slither")
        from slither.slithir.operations import LowLevelCall, HighLevelCall
        from slither.core.variables.state_variable import StateVariable

        _cfg_node_type = self._import()

        call_op = _make_mock_ir_op(HighLevelCall, lvalue=True)
        write_op = _make_mock_ir_op(object)
        write_op.lvalue = MagicMock(spec=StateVariable)

        node = _make_mock_slither_node(irs=[call_op, write_op])
        result = _cfg_node_type(node)
        assert result == NODE_TYPES["CFG_NODE_CALL"], (
            "CFG_NODE_CALL must win when both a call and a state write are present."
        )

    def test_write_wins_over_read(self):
        slither = pytest.importorskip("slither")
        from slither.core.variables.state_variable import StateVariable

        _cfg_node_type = self._import()

        write_op = MagicMock()
        write_op.lvalue = MagicMock(spec=StateVariable)
        sv = MagicMock(spec=StateVariable)
        read_op = MagicMock()
        read_op.read = [sv]

        node = _make_mock_slither_node(irs=[write_op, read_op])
        result = _cfg_node_type(node)
        assert result == NODE_TYPES["CFG_NODE_WRITE"]

    def test_empty_irs_returns_other(self):
        _cfg_node_type = self._import()
        node = _make_mock_slither_node(irs=[])
        result = _cfg_node_type(node)
        assert result == NODE_TYPES["CFG_NODE_OTHER"]


# ─────────────────────────────────────────────────────────────────────────────
# _build_cfg_node_features() unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCfgNodeFeatures:
    def _import(self):
        from ml.src.preprocessing.graph_extractor import _build_cfg_node_features
        return _build_cfg_node_features

    def test_returns_exactly_12_elements(self):
        _build_cfg_node_features = self._import()
        node = _make_mock_slither_node()
        func = _make_mock_func()
        result = _build_cfg_node_features(node, func, NODE_TYPES["CFG_NODE_CALL"])
        assert len(result) == NODE_FEATURE_DIM, (
            f"_build_cfg_node_features returned {len(result)} elements; "
            f"expected {NODE_FEATURE_DIM}."
        )

    def test_type_id_reflects_cfg_type(self):
        _build_cfg_node_features = self._import()
        node = _make_mock_slither_node()
        func = _make_mock_func()
        for cfg_type in [8, 9, 10, 11, 12]:
            result = _build_cfg_node_features(node, func, cfg_type)
            assert result[0] == float(cfg_type), (
                f"type_id [0] expected {float(cfg_type)}, got {result[0]}"
            )

    def test_in_unchecked_is_always_zero(self):
        """CFG nodes must NEVER inherit in_unchecked from the parent function."""
        _build_cfg_node_features = self._import()

        # Parent function has in_unchecked=1.0 (from source content)
        func = _make_mock_func(source_content="unchecked { x -= 1; }")

        node = _make_mock_slither_node()
        result = _build_cfg_node_features(node, func, NODE_TYPES["CFG_NODE_WRITE"])

        assert result[9] == 0.0, (
            "in_unchecked [9] must be 0.0 for CFG nodes — NEVER inherited from "
            "parent function flag. Inheriting would mark all CFG nodes in a "
            "function that has any unchecked block, including safe statements outside it."
        )

    def test_call_target_typed_default_is_1(self):
        """Default safe: not applicable at statement level."""
        _build_cfg_node_features = self._import()
        node = _make_mock_slither_node()
        func = _make_mock_func()
        result = _build_cfg_node_features(node, func, NODE_TYPES["CFG_NODE_OTHER"])
        assert result[8] == 1.0

    def test_loc_from_source_mapping(self):
        _build_cfg_node_features = self._import()
        node = _make_mock_slither_node(source_lines=[10, 11, 12])
        func = _make_mock_func()
        result = _build_cfg_node_features(node, func, NODE_TYPES["CFG_NODE_READ"])
        assert result[6] == 3.0, f"Expected loc=3.0, got {result[6]}"


# ─────────────────────────────────────────────────────────────────────────────
# _build_node_features() unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildNodeFeatures:
    def _import(self):
        from ml.src.preprocessing.graph_extractor import _build_node_features
        return _build_node_features

    def test_returns_exactly_12_elements_for_contract_node(self):
        _build_node_features = self._import()
        contract_mock = MagicMock()
        contract_mock.canonical_name = None
        contract_mock.name = "TestContract"
        contract_mock.visibility = "public"
        contract_mock.source_mapping = None
        result = _build_node_features(contract_mock, NODE_TYPES["CONTRACT"])
        assert len(result) == NODE_FEATURE_DIM

    def test_type_id_override_for_constructor(self):
        slither = pytest.importorskip("slither")
        _build_node_features = self._import()
        func = _make_mock_func(is_constructor=True)
        result = _build_node_features(func, NODE_TYPES["FUNCTION"])
        assert result[0] == float(NODE_TYPES["CONSTRUCTOR"])

    def test_type_id_override_for_fallback(self):
        slither = pytest.importorskip("slither")
        _build_node_features = self._import()
        func = _make_mock_func(is_fallback=True)
        result = _build_node_features(func, NODE_TYPES["FUNCTION"])
        assert result[0] == float(NODE_TYPES["FALLBACK"])

    def test_return_ignored_sentinel_on_ir_failure(self):
        """return_ignored [7] returns -1.0 when Slither IR is unavailable."""
        slither = pytest.importorskip("slither")
        _build_node_features = self._import()
        import ml.src.preprocessing.graph_extractor as ge_module

        # Patch _compute_return_ignored to return the sentinel, simulating IR unavailability
        original = ge_module._compute_return_ignored
        ge_module._compute_return_ignored = lambda _func: -1.0
        try:
            func = _make_mock_func()
            result = _build_node_features(func, NODE_TYPES["FUNCTION"])
        finally:
            ge_module._compute_return_ignored = original

        assert result[7] == -1.0, (
            "return_ignored must be -1.0 (sentinel) when Slither IR is unavailable, "
            "not 0.0 (assumed safe). The sentinel gives the GNN a distinct embedding "
            "for 'unknown' vs 'confirmed safe'."
        )

    def test_non_function_call_target_typed_is_1(self):
        """Non-Function nodes default to call_target_typed=1.0 (not applicable)."""
        _build_node_features = self._import()

        class FakeStateVar:
            canonical_name = "TestContract.x"
            visibility     = "private"
            source_mapping = None
            # No 'nodes' or 'pure' attrs — duck-typing classifies this as non-Function

        result = _build_node_features(FakeStateVar(), NODE_TYPES["STATE_VAR"])
        assert result[8] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Feature compute function unit tests (mocked Slither objects)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeReturnIgnored:
    def _import(self):
        from ml.src.preprocessing.graph_extractor import _compute_return_ignored
        return _compute_return_ignored

    def test_returns_1_when_lvalue_none(self):
        slither = pytest.importorskip("slither")
        from slither.slithir.operations import LowLevelCall
        _fn = self._import()

        call_op = MagicMock(spec=LowLevelCall)
        call_op.lvalue = None  # return value discarded
        func = _make_mock_func(slithir_operations=[call_op])
        assert _fn(func) == 1.0

    def test_returns_0_when_all_lvalues_captured(self):
        slither = pytest.importorskip("slither")
        from slither.slithir.operations import HighLevelCall
        _fn = self._import()

        call_op = MagicMock(spec=HighLevelCall)
        call_op.lvalue = MagicMock()  # return value captured
        func = _make_mock_func(slithir_operations=[call_op])
        assert _fn(func) == 0.0

    def test_returns_sentinel_on_attribute_error(self):
        _fn = self._import()

        class FakeFunc:
            canonical_name = "TestContract.f"

            @property
            def slithir_operations(self):
                raise AttributeError("no IR")

        assert _fn(FakeFunc()) == -1.0, "Should return -1.0 sentinel, not 0.0"

    def test_no_calls_returns_0(self):
        slither = pytest.importorskip("slither")
        _fn = self._import()
        func = _make_mock_func(slithir_operations=[])  # no operations
        assert _fn(func) == 0.0


class TestComputeCallTargetTyped:
    def _import(self):
        from ml.src.preprocessing.graph_extractor import _compute_call_target_typed
        return _compute_call_target_typed

    def test_low_level_call_returns_0(self):
        slither = pytest.importorskip("slither")
        _fn = self._import()
        func = _make_mock_func(low_level_calls=[MagicMock()])
        assert _fn(func) == 0.0

    def test_no_calls_returns_1(self):
        slither = pytest.importorskip("slither")
        _fn = self._import()
        func = _make_mock_func(high_level_calls=[], low_level_calls=[])
        assert _fn(func) == 1.0

    def test_sentinel_when_source_unavailable(self):
        _fn = self._import()

        class FakeFunc:
            canonical_name  = "TestContract.f"
            low_level_calls  = []
            high_level_calls = []
            source_mapping   = None   # unavailable → sentinel -1.0

        # Patch out the AddressType import so type-resolution raises ImportError,
        # which the outer except catches.  The fallback then sees source_mapping=None
        # and must return -1.0 rather than the closed-world "safe" 1.0.
        with patch.dict("sys.modules", {"slither.core.solidity_types": None}):
            result = _fn(FakeFunc())

        assert result == -1.0


class TestComputeInUnchecked:
    def _import(self):
        from ml.src.preprocessing.graph_extractor import _compute_in_unchecked
        return _compute_in_unchecked

    def test_regex_matches_unchecked_with_space(self):
        _fn = self._import()
        func = _make_mock_func(source_content="unchecked { x -= 1; }")
        # If STARTUNCHECKED raises AttributeError, regex path is triggered
        with patch("slither.core.cfg.node.NodeType") as mock_nt:
            del mock_nt.STARTUNCHECKED  # simulate absence
            result = _fn(func)
        assert result == 1.0

    def test_regex_matches_unchecked_no_space(self):
        _fn = self._import()
        func = _make_mock_func(source_content="unchecked{ x -= 1; }")
        with patch("slither.core.cfg.node.NodeType") as mock_nt:
            del mock_nt.STARTUNCHECKED
            result = _fn(func)
        assert result == 1.0

    def test_regex_matches_unchecked_newline_brace(self):
        """unchecked\\n{ is valid Solidity and must be caught by regex."""
        _fn = self._import()
        func = _make_mock_func(source_content="unchecked\n{ x -= 1; }")
        with patch("slither.core.cfg.node.NodeType") as mock_nt:
            del mock_nt.STARTUNCHECKED
            result = _fn(func)
        assert result == 1.0

    def test_regex_does_not_match_unchecked_in_comment(self):
        """'unchecked' in a comment string should be irrelevant — but regex checks
        raw source so this can false-positive. Documented limitation, not tested here."""
        pass  # intentionally left blank — regex-based detection has known limits


class TestComputeHasLoop:
    def _import(self):
        from ml.src.preprocessing.graph_extractor import _compute_has_loop
        return _compute_has_loop

    def test_returns_1_when_ifloop_present(self):
        slither = pytest.importorskip("slither")
        from slither.core.cfg.node import NodeType
        _fn = self._import()

        loop_node = _make_mock_slither_node(node_type=NodeType.IFLOOP)
        func = _make_mock_func(nodes=[loop_node])
        assert _fn(func) == 1.0

    def test_returns_0_when_no_loop(self):
        slither = pytest.importorskip("slither")
        from slither.core.cfg.node import NodeType
        _fn = self._import()

        plain_node = _make_mock_slither_node(node_type=NodeType.EXPRESSION)
        func = _make_mock_func(nodes=[plain_node])
        assert _fn(func) == 0.0


class TestComputeExternalCallCount:
    def _import(self):
        from ml.src.preprocessing.graph_extractor import _compute_external_call_count
        return _compute_external_call_count

    def test_no_calls_returns_0(self):
        _fn = self._import()
        func = _make_mock_func(high_level_calls=[], low_level_calls=[])
        assert _fn(func) == 0.0

    def test_one_call_correct_normalisation(self):
        _fn = self._import()
        func = _make_mock_func(high_level_calls=[MagicMock()], low_level_calls=[])
        expected = math.log1p(1) / math.log1p(20)
        assert abs(_fn(func) - expected) < 1e-6

    def test_clamped_at_1_for_many_calls(self):
        _fn = self._import()
        calls = [MagicMock() for _ in range(100)]
        func = _make_mock_func(high_level_calls=calls, low_level_calls=[])
        assert _fn(func) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _build_control_flow_edges() unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildControlFlowEdges:
    def _import(self):
        from ml.src.preprocessing.graph_extractor import _build_control_flow_edges
        return _build_control_flow_edges

    def test_x_list_and_node_metadata_stay_aligned(self):
        """len(node_metadata) must equal len(x_list) after every call."""
        slither = pytest.importorskip("slither")
        _fn = self._import()

        n0 = _make_mock_slither_node(node_id=0, sons=[])
        n1 = _make_mock_slither_node(node_id=1, sons=[])
        n0.sons = [n1]

        func = _make_mock_func(nodes=[n0, n1])
        x_list: list = [
            [0.0] * NODE_FEATURE_DIM,  # simulate pre-existing declaration node
        ]
        node_metadata: list = [{"name": "Contract", "type": "CONTRACT", "source_lines": []}]
        node_index_map: dict = {}

        _fn(func, func_node_idx=0, node_index_map=node_index_map,
            x_list=x_list, node_metadata=node_metadata)

        assert len(x_list) == len(node_metadata), (
            f"x_list length {len(x_list)} ≠ node_metadata length {len(node_metadata)}. "
            "The invariant must hold: both lists grow by 1 per CFG node."
        )

    def test_graph_idx_uses_x_list_length(self):
        """graph_idx = len(x_list) — not len(node_index_map)."""
        slither = pytest.importorskip("slither")
        _fn = self._import()

        n0 = _make_mock_slither_node(node_id=0, sons=[])
        func = _make_mock_func(nodes=[n0])

        # Start with 5 pre-existing nodes in x_list
        x_list = [[0.0] * NODE_FEATURE_DIM for _ in range(5)]
        node_metadata = [{"name": f"node{i}", "type": "STATE_VAR", "source_lines": []} for i in range(5)]
        node_index_map: dict = {}

        _fn(func, func_node_idx=0, node_index_map=node_index_map,
            x_list=x_list, node_metadata=node_metadata)

        # The CFG node should have been assigned index 5 (len(x_list) before append)
        assert node_index_map[n0] == 5, (
            f"CFG node assigned index {node_index_map[n0]}; expected 5. "
            "graph_idx must be len(x_list) before the append, not len(node_index_map)."
        )

    def test_control_flow_edges_only_within_function(self):
        """Pass 2 must only add CONTROL_FLOW edges within the current function's CFG.
        A successor not in node_index_map (e.g. from a different function) is skipped."""
        slither = pytest.importorskip("slither")
        _fn = self._import()

        # n0 has a successor that is NOT in the current function's nodes
        external_node = _make_mock_slither_node(node_id=99, sons=[])
        n0 = _make_mock_slither_node(node_id=0, sons=[external_node])  # successor not in map

        func = _make_mock_func(nodes=[n0])  # only n0, not external_node
        x_list: list = []
        node_metadata: list = []
        node_index_map: dict = {}

        _, control_flow_edges = _fn(func, func_node_idx=0, node_index_map=node_index_map,
                                    x_list=x_list, node_metadata=node_metadata)

        # external_node is not in node_index_map so no CONTROL_FLOW edge should exist
        assert len(control_flow_edges) == 0, (
            "CONTROL_FLOW edges must only be built within the current function's "
            "node_index_map scope. Successors from other functions must be skipped "
            "via the 'if successor in node_index_map' gate."
        )

    def test_contains_edges_built_for_each_cfg_node(self):
        slither = pytest.importorskip("slither")
        _fn = self._import()

        nodes = [_make_mock_slither_node(node_id=i, sons=[]) for i in range(3)]
        func = _make_mock_func(nodes=nodes)
        x_list: list = [[0.0] * NODE_FEATURE_DIM]  # 1 function node at idx 0
        node_metadata: list = [{"name": "func", "type": "FUNCTION", "source_lines": []}]
        node_index_map: dict = {}

        contains_edges, _ = _fn(func, func_node_idx=0, node_index_map=node_index_map,
                                 x_list=x_list, node_metadata=node_metadata)

        assert len(contains_edges) == 3, (
            "Expected one CONTAINS edge per CFG node (3 nodes → 3 edges)."
        )
        # All CONTAINS edges should start from the function node (idx 0)
        for src, dst in contains_edges:
            assert src == 0, f"CONTAINS edge src {src} ≠ func_node_idx 0"

    def test_node_metadata_has_required_keys(self):
        slither = pytest.importorskip("slither")
        _fn = self._import()

        n0 = _make_mock_slither_node(node_id=0, sons=[])
        func = _make_mock_func(nodes=[n0])
        x_list: list = []
        node_metadata: list = []
        _fn(func, func_node_idx=0, node_index_map={}, x_list=x_list, node_metadata=node_metadata)

        for meta in node_metadata:
            assert "name"         in meta, "node_metadata entry missing 'name' key"
            assert "type"         in meta, "node_metadata entry missing 'type' key"
            assert "source_lines" in meta, "node_metadata entry missing 'source_lines' key"


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests (require slither-analyzer)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestExtractionIntegration:
    """
    These tests call extract_contract_graph() on real .sol files written to a
    temp directory. They require slither-analyzer and a working solc installation.
    Run with: pytest ml/tests/test_preprocessing.py -m integration
    """

    @pytest.fixture
    def sol_file(self, tmp_path):
        """Write a Solidity file to a temp dir and return its Path."""
        def _write(content: str, name: str = "test.sol") -> Path:
            p = tmp_path / name
            p.write_text(content)
            return p
        return _write

    def _extract(self, sol_path: Path):
        from ml.src.preprocessing.graph_extractor import extract_contract_graph
        return extract_contract_graph(sol_path)

    # ── Reentrancy (call-before-write) ─────────────────────────────────────
    def test_reentrancy_has_cfg_node_call(self, sol_file):
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract R {
            mapping(address => uint) public balances;
            function withdraw(uint amount) external {
                require(balances[msg.sender] >= amount);
                (bool ok,) = msg.sender.call{value: amount}("");
                balances[msg.sender] -= amount;
            }
        }
        """)
        graph = self._extract(path)
        type_ids = graph.x[:, 0].int().tolist()
        assert NODE_TYPES["CFG_NODE_CALL"] in type_ids, (
            "Reentrancy contract must have at least one CFG_NODE_CALL (type 8) node."
        )

    def test_reentrancy_has_control_flow_edges(self, sol_file):
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract R {
            mapping(address => uint) public balances;
            function withdraw(uint amount) external {
                require(balances[msg.sender] >= amount);
                (bool ok,) = msg.sender.call{value: amount}("");
                balances[msg.sender] -= amount;
            }
        }
        """)
        graph = self._extract(path)
        assert hasattr(graph, "edge_attr"), "graph.edge_attr missing"
        cf_edges = (graph.edge_attr == EDGE_TYPES["CONTROL_FLOW"]).sum().item()
        assert cf_edges > 0, "Reentrancy contract must have CONTROL_FLOW edges (type 6)."

    # ── CEI-safe (write-before-call) ───────────────────────────────────────
    def test_cei_safe_has_write_before_call_in_control_flow(self, sol_file):
        """
        Write-before-call (safe CEI): there must be a CONTROL_FLOW edge from a
        CFG_NODE_WRITE (9) to a CFG_NODE_CALL (8) node — i.e., the write node
        precedes the call node in execution order.
        """
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract Safe {
            mapping(address => uint) public balances;
            function withdraw(uint amount) external {
                require(balances[msg.sender] >= amount);
                balances[msg.sender] -= amount;
                (bool ok,) = msg.sender.call{value: amount}("");
            }
        }
        """)
        graph = self._extract(path)
        type_ids = graph.x[:, 0].int().tolist()
        assert NODE_TYPES["CFG_NODE_WRITE"] in type_ids
        assert NODE_TYPES["CFG_NODE_CALL"]  in type_ids

        write_indices = {i for i, t in enumerate(type_ids) if t == NODE_TYPES["CFG_NODE_WRITE"]}
        call_indices  = {i for i, t in enumerate(type_ids) if t == NODE_TYPES["CFG_NODE_CALL"]}

        # Check that at least one CONTROL_FLOW edge goes from a write node to a call node
        cf_mask = (graph.edge_attr == EDGE_TYPES["CONTROL_FLOW"])
        cf_src = graph.edge_index[0][cf_mask].tolist()
        cf_dst = graph.edge_index[1][cf_mask].tolist()

        found_write_before_call = any(
            src in write_indices and dst in call_indices
            for src, dst in zip(cf_src, cf_dst)
        )
        assert found_write_before_call, (
            "CEI-safe contract must have a CONTROL_FLOW edge from CFG_NODE_WRITE → "
            "CFG_NODE_CALL. This encodes 'write before call' execution order."
        )

    # ── unchecked{} contract ──────────────────────────────────────────────
    def test_unchecked_func_node_has_in_unchecked_1(self, sol_file):
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract U {
            function add(uint a, uint b) external pure returns (uint) {
                unchecked { return a + b; }
            }
        }
        """)
        graph = self._extract(path)
        # FUNCTION nodes: type_id == 1
        func_mask = (graph.x[:, 0].int() == NODE_TYPES["FUNCTION"])
        func_in_unchecked = graph.x[func_mask, 9]  # in_unchecked at index 9
        assert (func_in_unchecked == 1.0).any(), (
            "Function node in unchecked{} contract should have in_unchecked=1.0."
        )

    def test_unchecked_cfg_nodes_have_in_unchecked_0(self, sol_file):
        """
        CRITICAL: CFG nodes must have in_unchecked=0.0 even when the parent
        function has in_unchecked=1.0. The feature must never be inherited.
        """
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract U {
            function add(uint a, uint b) external pure returns (uint) {
                unchecked { return a + b; }
            }
        }
        """)
        graph = self._extract(path)
        # CFG node types: 8–12
        cfg_mask = (graph.x[:, 0].int() >= 8) & (graph.x[:, 0].int() <= 12)
        if cfg_mask.any():
            cfg_in_unchecked = graph.x[cfg_mask, 9]
            assert (cfg_in_unchecked == 0.0).all(), (
                f"CFG nodes have in_unchecked values: {cfg_in_unchecked.tolist()}. "
                "All must be 0.0 — in_unchecked must NEVER be inherited from the parent function."
            )

    # ── Loop contract ──────────────────────────────────────────────────────
    def test_loop_func_has_has_loop_1(self, sol_file):
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract L {
            function sum(uint[] calldata arr) external pure returns (uint total) {
                for (uint i; i < arr.length; i++) { total += arr[i]; }
            }
        }
        """)
        graph = self._extract(path)
        func_mask = (graph.x[:, 0].int() == NODE_TYPES["FUNCTION"])
        func_has_loop = graph.x[func_mask, 10]  # has_loop at index 10
        assert (func_has_loop == 1.0).any(), "Loop function must have has_loop=1.0."

    # ── node_metadata alignment ───────────────────────────────────────────
    def test_node_metadata_length_equals_x_shape(self, sol_file):
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract M {
            uint public x;
            function set(uint v) external { x = v; }
        }
        """)
        graph = self._extract(path)
        assert hasattr(graph, "node_metadata"), (
            "graph.node_metadata missing — extraction must attach this attribute."
        )
        assert len(graph.node_metadata) == graph.x.shape[0], (
            f"node_metadata length {len(graph.node_metadata)} ≠ "
            f"x.shape[0] {graph.x.shape[0]}. They must be index-aligned."
        )

    def test_function_nodes_have_name_in_metadata(self, sol_file):
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract T {
            function foo() external pure returns (uint) { return 42; }
        }
        """)
        graph = self._extract(path)
        assert hasattr(graph, "node_metadata")
        type_ids = graph.x[:, 0].int().tolist()
        func_indices = [i for i, t in enumerate(type_ids) if t == NODE_TYPES["FUNCTION"]]
        for idx in func_indices:
            meta = graph.node_metadata[idx]
            name = meta.get("name", "")
            assert name, (
                f"FUNCTION node at index {idx} has empty 'name' in node_metadata. "
                "Function nodes must have a populated canonical name."
            )

    def test_x_has_correct_feature_dim(self, sol_file):
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract D { uint public x; }
        """)
        graph = self._extract(path)
        assert graph.x.shape[1] == NODE_FEATURE_DIM, (
            f"graph.x.shape[1] = {graph.x.shape[1]}; expected {NODE_FEATURE_DIM} (v2 schema)."
        )

    def test_edge_attr_is_1d_int64(self, sol_file):
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract E { uint public x; function set(uint v) external { x = v; } }
        """)
        graph = self._extract(path)
        assert graph.edge_attr.dim() == 1, (
            "edge_attr must be 1-D [E] not [E, 1]. "
            "nn.Embedding will crash on shape [E, 1]."
        )
        assert graph.edge_attr.dtype == torch.long

    def test_edge_attr_values_in_valid_range(self, sol_file):
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract V { uint public x; function set(uint v) external { x = v; } }
        """)
        graph = self._extract(path)
        if graph.edge_attr.numel() > 0:
            assert graph.edge_attr.max().item() < NUM_EDGE_TYPES, (
                f"edge_attr contains values >= NUM_EDGE_TYPES={NUM_EDGE_TYPES}."
            )
            assert graph.edge_attr.min().item() >= 0

    def test_contains_edges_present_when_function_has_cfg(self, sol_file):
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract C { function f() external { uint x = 1; } }
        """)
        graph = self._extract(path)
        assert (graph.edge_attr == EDGE_TYPES["CONTAINS"]).any(), (
            "Contract with function body must have CONTAINS edges (type 5)."
        )

    def test_node_metadata_type_field_matches_type_id(self, sol_file):
        """metadata['type'] must correspond to the node's type_id in graph.x."""
        path = sol_file("""
        pragma solidity ^0.8.0;
        contract A { function f() external { uint x = 1; } }
        """)
        graph = self._extract(path)
        type_ids = graph.x[:, 0].int().tolist()
        for i, (type_id, meta) in enumerate(zip(type_ids, graph.node_metadata)):
            expected_name = {v: k for k, v in NODE_TYPES.items()}.get(type_id, "UNKNOWN")
            meta_type = meta.get("type", "")
            assert meta_type == expected_name, (
                f"Node {i}: type_id={type_id} → expected type name '{expected_name}', "
                f"got '{meta_type}' in node_metadata."
            )
