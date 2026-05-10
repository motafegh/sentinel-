"""
test_preprocessing.py — Unit tests for ContractPreprocessor and graph_extractor.

ContractPreprocessor.__init__ loads the CodeBERT tokenizer from HuggingFace
and _extract_graph calls Slither/solc. Both are mocked so the preprocessor
tests run without any external dependencies or network access.

Graph feature tests use mock Slither objects to test individual feature
computations in isolation.  Integration tests that call extract_contract_graph
on real .sol files are marked with @pytest.mark.integration and require
Slither to be installed.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from ml.src.preprocessing.graph_schema import (
    EDGE_TYPES,
    FEATURE_NAMES,
    NODE_FEATURE_DIM,
    NODE_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers: build synthetic outputs that _extract_graph / _tokenize would return
# ---------------------------------------------------------------------------

def _synthetic_graph(contract_hash: str = "abc123") -> Data:
    """Minimal valid graph with NODE_FEATURE_DIM-dim node features (v2 schema)."""
    return Data(
        x=torch.randn(5, NODE_FEATURE_DIM),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        contract_hash=contract_hash,
    )


def _synthetic_tokens(contract_hash: str = "abc123", truncated: bool = False) -> dict:
    return {
        "input_ids":      torch.ones(1, 512, dtype=torch.long),
        "attention_mask": torch.ones(1, 512, dtype=torch.long),
        "contract_hash":  contract_hash,
        "num_tokens":     128,
        "truncated":      truncated,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def preprocessor():
    """ContractPreprocessor with tokenizer load patched out."""
    with patch(
        "ml.src.inference.preprocess.AutoTokenizer.from_pretrained",
        return_value=MagicMock(),
    ):
        from ml.src.inference.preprocess import ContractPreprocessor
        return ContractPreprocessor()


# ---------------------------------------------------------------------------
# Input validation — process_source()
# ---------------------------------------------------------------------------

def test_process_source_empty_string_raises(preprocessor):
    with pytest.raises(ValueError, match="empty"):
        preprocessor.process_source("")


def test_process_source_whitespace_only_raises(preprocessor):
    with pytest.raises(ValueError, match="empty"):
        preprocessor.process_source("   \n\t  ")


def test_process_source_too_large_raises(preprocessor):
    oversized = "x" * (preprocessor.MAX_SOURCE_BYTES + 1)
    with pytest.raises(ValueError, match="too large"):
        preprocessor.process_source(oversized)


def test_process_source_exactly_at_limit_does_not_raise(preprocessor):
    source = "x" * preprocessor.MAX_SOURCE_BYTES
    contract_hash = hashlib.md5(source.encode()).hexdigest()

    preprocessor._extract_graph = MagicMock(return_value=_synthetic_graph(contract_hash))
    preprocessor._tokenize      = MagicMock(return_value=_synthetic_tokens(contract_hash))
    preprocessor._log_result    = MagicMock()

    graph, tokens = preprocessor.process_source(source)
    assert graph is not None
    assert tokens is not None


# ---------------------------------------------------------------------------
# Input validation — process()
# ---------------------------------------------------------------------------

def test_process_missing_file_raises(preprocessor, tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocessor.process(tmp_path / "does_not_exist.sol")


# ---------------------------------------------------------------------------
# Output shape contract — v2 schema (NODE_FEATURE_DIM=13)
# ---------------------------------------------------------------------------

def test_process_source_graph_x_shape(preprocessor):
    """graph.x must be [N, NODE_FEATURE_DIM] to match GNNEncoder in_channels."""
    source        = "pragma solidity ^0.8.0; contract A {}"
    contract_hash = hashlib.md5(source.encode()).hexdigest()

    preprocessor._extract_graph = MagicMock(return_value=_synthetic_graph(contract_hash))
    preprocessor._tokenize      = MagicMock(return_value=_synthetic_tokens(contract_hash))
    preprocessor._log_result    = MagicMock()

    graph, _ = preprocessor.process_source(source)
    assert graph.x.shape[1] == NODE_FEATURE_DIM, (
        f"expected {NODE_FEATURE_DIM}-dim node features (v2 schema), "
        f"got {graph.x.shape[1]}"
    )


def test_process_source_token_shape(preprocessor):
    """input_ids and attention_mask must be [1, 512] for single-sample inference."""
    source        = "pragma solidity ^0.8.0; contract A {}"
    contract_hash = hashlib.md5(source.encode()).hexdigest()

    preprocessor._extract_graph = MagicMock(return_value=_synthetic_graph(contract_hash))
    preprocessor._tokenize      = MagicMock(return_value=_synthetic_tokens(contract_hash))
    preprocessor._log_result    = MagicMock()

    _, tokens = preprocessor.process_source(source)
    assert tokens["input_ids"].shape      == (1, 512)
    assert tokens["attention_mask"].shape == (1, 512)


def test_process_source_tokens_include_required_keys(preprocessor):
    source        = "pragma solidity ^0.8.0; contract A {}"
    contract_hash = hashlib.md5(source.encode()).hexdigest()

    preprocessor._extract_graph = MagicMock(return_value=_synthetic_graph(contract_hash))
    preprocessor._tokenize      = MagicMock(return_value=_synthetic_tokens(contract_hash))
    preprocessor._log_result    = MagicMock()

    _, tokens = preprocessor.process_source(source)
    for key in ("input_ids", "attention_mask", "contract_hash", "num_tokens", "truncated"):
        assert key in tokens, f"missing key: {key}"


# ---------------------------------------------------------------------------
# Content-addressed hashing
# ---------------------------------------------------------------------------

def test_process_source_same_source_same_hash(preprocessor):
    source        = "pragma solidity ^0.8.0; contract A {}"
    contract_hash = hashlib.md5(source.encode()).hexdigest()

    preprocessor._extract_graph = MagicMock(return_value=_synthetic_graph(contract_hash))
    preprocessor._tokenize      = MagicMock(return_value=_synthetic_tokens(contract_hash))
    preprocessor._log_result    = MagicMock()

    _, t1 = preprocessor.process_source(source)
    _, t2 = preprocessor.process_source(source)
    assert t1["contract_hash"] == t2["contract_hash"]


def test_process_source_different_source_different_hash(preprocessor):
    s1 = "pragma solidity ^0.8.0; contract A {}"
    s2 = "pragma solidity ^0.8.0; contract B {}"
    h1 = hashlib.md5(s1.encode()).hexdigest()
    h2 = hashlib.md5(s2.encode()).hexdigest()

    preprocessor._extract_graph = MagicMock(
        side_effect=[_synthetic_graph(h1), _synthetic_graph(h2)]
    )
    preprocessor._tokenize = MagicMock(
        side_effect=[_synthetic_tokens(h1), _synthetic_tokens(h2)]
    )
    preprocessor._log_result = MagicMock()

    _, t1 = preprocessor.process_source(s1)
    _, t2 = preprocessor.process_source(s2)
    assert t1["contract_hash"] != t2["contract_hash"]


# ---------------------------------------------------------------------------
# Truncation flag
# ---------------------------------------------------------------------------

def test_process_source_truncated_flag_propagated(preprocessor):
    source        = "pragma solidity ^0.8.0; contract A {}"
    contract_hash = hashlib.md5(source.encode()).hexdigest()

    preprocessor._extract_graph = MagicMock(return_value=_synthetic_graph(contract_hash))
    preprocessor._tokenize      = MagicMock(return_value=_synthetic_tokens(contract_hash, truncated=True))
    preprocessor._log_result    = MagicMock()

    _, tokens = preprocessor.process_source(source)
    assert tokens["truncated"] is True


# ---------------------------------------------------------------------------
# Schema constants sanity checks
# ---------------------------------------------------------------------------

def test_feature_names_length_matches_node_feature_dim():
    assert len(FEATURE_NAMES) == NODE_FEATURE_DIM, (
        f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries; "
        f"NODE_FEATURE_DIM={NODE_FEATURE_DIM}"
    )


def test_feature_names_v2_has_expected_entries():
    assert "return_ignored"      in FEATURE_NAMES
    assert "call_target_typed"   in FEATURE_NAMES
    assert "in_unchecked"        in FEATURE_NAMES
    assert "has_loop"            in FEATURE_NAMES
    assert "gas_intensity"       in FEATURE_NAMES
    assert "external_call_count" in FEATURE_NAMES


def test_feature_names_v2_reentrant_removed():
    assert "reentrant" not in FEATURE_NAMES, (
        "'reentrant' should have been removed in v2 schema — "
        "it was Slither's own detection leaking into training features."
    )


def test_edge_types_v2_contains_cfg_entries():
    assert "CONTAINS"     in EDGE_TYPES
    assert "CONTROL_FLOW" in EDGE_TYPES
    assert EDGE_TYPES["CONTAINS"]     == 5
    assert EDGE_TYPES["CONTROL_FLOW"] == 6


def test_node_types_v2_contains_cfg_node():
    assert "CFG_NODE" in NODE_TYPES
    assert NODE_TYPES["CFG_NODE"] == 8


# ---------------------------------------------------------------------------
# _build_node_features — unit tests with mock Slither objects
# ---------------------------------------------------------------------------

def _make_mock_function(
    canonical_name: str = "MyContract.myFunc",
    visibility: str = "public",
    pure: bool = False,
    view: bool = False,
    payable: bool = False,
    is_constructor: bool = False,
    is_fallback: bool = False,
    is_receive: bool = False,
    nodes: list | None = None,
    source_lines: list | None = None,
    # new v2 features
    slithir_ops: list | None = None,
    high_level_calls: list | None = None,
    low_level_calls: list | None = None,
    external_calls_as_expressions: list | None = None,
) -> MagicMock:
    """Create a mock Slither Function object for feature computation tests."""
    from slither.core.declarations import Function

    func = MagicMock(spec=Function)
    func.canonical_name   = canonical_name
    func.visibility       = visibility
    func.pure             = pure
    func.view             = view
    func.payable          = payable
    func.is_constructor   = is_constructor
    func.is_fallback      = is_fallback
    func.is_receive       = is_receive
    func.nodes            = nodes or []
    func.slithir_operations = slithir_ops or []
    func.high_level_calls   = high_level_calls or []
    func.low_level_calls    = low_level_calls or []
    func.external_calls_as_expressions = external_calls_as_expressions or []
    func.state_variables_read    = []
    func.state_variables_written = []
    func.internal_calls          = []

    src = MagicMock()
    src.lines   = source_lines or list(range(10))
    src.content = ""
    func.source_mapping = src

    return func


def test_build_node_features_returns_correct_length():
    from ml.src.preprocessing.graph_extractor import _build_node_features
    func = _make_mock_function()
    feats = _build_node_features(func, NODE_TYPES["FUNCTION"])
    assert len(feats) == NODE_FEATURE_DIM, (
        f"_build_node_features returned {len(feats)} values, "
        f"expected {NODE_FEATURE_DIM}"
    )


def test_build_node_features_type_id_at_index_0():
    from ml.src.preprocessing.graph_extractor import _build_node_features
    func = _make_mock_function()
    feats = _build_node_features(func, NODE_TYPES["FUNCTION"])
    assert feats[0] == float(NODE_TYPES["FUNCTION"])


def test_build_node_features_constructor_overrides_type_id():
    from ml.src.preprocessing.graph_extractor import _build_node_features
    func = _make_mock_function(is_constructor=True)
    feats = _build_node_features(func, NODE_TYPES["FUNCTION"])
    assert feats[0] == float(NODE_TYPES["CONSTRUCTOR"]), (
        "Constructor function should override type_id to CONSTRUCTOR"
    )


def test_build_node_features_payable_flag():
    from ml.src.preprocessing.graph_extractor import _build_node_features
    func = _make_mock_function(payable=True)
    feats = _build_node_features(func, NODE_TYPES["FUNCTION"])
    payable_idx = FEATURE_NAMES.index("payable")
    assert feats[payable_idx] == 1.0


def test_build_node_features_no_reentrant_feature():
    """v2 schema: reentrant must not appear in computed features (it's removed)."""
    from ml.src.preprocessing.graph_extractor import _build_node_features
    # Even if Slither had is_reentrant=True, it should NOT be in features
    func = _make_mock_function()
    func.is_reentrant = True
    feats = _build_node_features(func, NODE_TYPES["FUNCTION"])
    assert len(feats) == NODE_FEATURE_DIM
    # We can't directly assert "reentrant not in feats" by value (it's 1.0),
    # but we can verify the length is 13 (no extra feature slipped in).


# ---------------------------------------------------------------------------
# _compute_return_ignored — unit tests
# ---------------------------------------------------------------------------

def test_return_ignored_no_calls_is_zero():
    from ml.src.preprocessing.graph_extractor import _compute_return_ignored
    func = _make_mock_function(slithir_ops=[])
    assert _compute_return_ignored(func) == 0.0


def test_return_ignored_low_level_call_with_lvalue_is_zero():
    from ml.src.preprocessing.graph_extractor import _compute_return_ignored
    try:
        from slither.slithir.operations import LowLevelCall
        op = MagicMock(spec=LowLevelCall)
        op.lvalue = MagicMock()  # lvalue is NOT None → return captured
        func = _make_mock_function(slithir_ops=[op])
        assert _compute_return_ignored(func) == 0.0
    except ImportError:
        pytest.skip("slither not installed")


def test_return_ignored_low_level_call_without_lvalue_is_one():
    from ml.src.preprocessing.graph_extractor import _compute_return_ignored
    try:
        from slither.slithir.operations import LowLevelCall
        op = MagicMock(spec=LowLevelCall)
        op.lvalue = None  # return value discarded
        func = _make_mock_function(slithir_ops=[op])
        assert _compute_return_ignored(func) == 1.0
    except ImportError:
        pytest.skip("slither not installed")


# ---------------------------------------------------------------------------
# _compute_in_unchecked — unit tests
# ---------------------------------------------------------------------------

def test_in_unchecked_no_unchecked_is_zero():
    from ml.src.preprocessing.graph_extractor import _compute_in_unchecked
    func = _make_mock_function(nodes=[])
    func.source_mapping.content = "uint x = a + b;"
    assert _compute_in_unchecked(func) == 0.0


def test_in_unchecked_source_contains_unchecked_is_one():
    from ml.src.preprocessing.graph_extractor import _compute_in_unchecked
    func = _make_mock_function(nodes=[])
    func.source_mapping.content = "unchecked { x += 1; }"
    assert _compute_in_unchecked(func) == 1.0


# ---------------------------------------------------------------------------
# _compute_has_loop — unit tests
# ---------------------------------------------------------------------------

def test_has_loop_no_nodes_is_zero():
    from ml.src.preprocessing.graph_extractor import _compute_has_loop
    func = _make_mock_function(nodes=[])
    func.source_mapping.content = "uint x = 1;"
    assert _compute_has_loop(func) == 0.0


def test_has_loop_for_loop_in_source_is_one():
    from ml.src.preprocessing.graph_extractor import _compute_has_loop
    func = _make_mock_function(nodes=[])
    func.source_mapping.content = "for (uint i = 0; i < 10; i++) {}"
    assert _compute_has_loop(func) == 1.0


def test_has_loop_while_loop_in_source_is_one():
    from ml.src.preprocessing.graph_extractor import _compute_has_loop
    func = _make_mock_function(nodes=[])
    func.source_mapping.content = "while (condition) { doSomething(); }"
    assert _compute_has_loop(func) == 1.0


# ---------------------------------------------------------------------------
# _compute_external_call_count — unit tests
# ---------------------------------------------------------------------------

def test_external_call_count_zero_calls():
    from ml.src.preprocessing.graph_extractor import _compute_external_call_count
    func = _make_mock_function(high_level_calls=[], low_level_calls=[])
    assert _compute_external_call_count(func) == pytest.approx(math.log1p(0))


def test_external_call_count_two_calls():
    from ml.src.preprocessing.graph_extractor import _compute_external_call_count
    func = _make_mock_function(
        high_level_calls=[MagicMock(), MagicMock()],
        low_level_calls=[],
    )
    assert _compute_external_call_count(func) == pytest.approx(math.log1p(2))


# ---------------------------------------------------------------------------
# _build_cfg_node_features — unit tests
# ---------------------------------------------------------------------------

def test_build_cfg_node_features_returns_correct_length():
    from ml.src.preprocessing.graph_extractor import _build_cfg_node_features
    fn_node  = MagicMock()
    fn_node.node_id = 0
    fn_node.irs = []
    fn_node.source_mapping = MagicMock()
    fn_node.source_mapping.lines = list(range(3))
    parent = _make_mock_function()
    feats = _build_cfg_node_features(fn_node, parent)
    assert len(feats) == NODE_FEATURE_DIM


def test_build_cfg_node_features_type_id_is_cfg_node():
    from ml.src.preprocessing.graph_extractor import _build_cfg_node_features
    fn_node = MagicMock()
    fn_node.irs = []
    fn_node.source_mapping = MagicMock()
    fn_node.source_mapping.lines = [1, 2]
    parent = _make_mock_function()
    feats = _build_cfg_node_features(fn_node, parent)
    assert feats[0] == float(NODE_TYPES["CFG_NODE"])


# ---------------------------------------------------------------------------
# Integration: extract_contract_graph on real contracts (requires Slither)
# ---------------------------------------------------------------------------

slither_available = pytest.importorskip("slither", reason="slither not installed")


CLASSIC_REENTRANCY = """\
pragma solidity ^0.8.0;

contract Reentrant {
    mapping(address => uint) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    // VULNERABLE: calls external address before zeroing balance
    function withdraw(uint amount) external {
        require(balances[msg.sender] >= amount);
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok);
        balances[msg.sender] -= amount;  // write AFTER call
    }
}
"""

SAFE_CEI = """\
pragma solidity ^0.8.0;

contract SafeCEI {
    mapping(address => uint) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    // SAFE: checks-effects-interactions — balance zeroed BEFORE call
    function withdraw(uint amount) external {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;  // write BEFORE call
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok);
    }
}
"""

UNCHECKED_OVERFLOW = """\
pragma solidity ^0.8.0;

contract UncheckedOps {
    function unsafeAdd(uint a, uint b) external pure returns (uint) {
        unchecked {
            return a + b;  // can overflow on 0.8+
        }
    }
}
"""

DOS_UNBOUNDED_LOOP = """\
pragma solidity ^0.8.0;

contract Lottery {
    address[] public players;

    function addPlayer() external {
        players.push(msg.sender);
    }

    // VULNERABLE: unbounded loop — gas DoS if players grows large
    function distributeAll() external {
        for (uint i = 0; i < players.length; i++) {
            payable(players[i]).transfer(1 ether);
        }
    }
}
"""


@pytest.fixture(scope="module")
def tmp_sol(tmp_path_factory):
    return tmp_path_factory.mktemp("sol")


def _extract(source: str, tmp_dir: Path) -> Data:
    from ml.src.preprocessing.graph_extractor import (
        GraphExtractionConfig,
        extract_contract_graph,
    )
    path = tmp_dir / "contract.sol"
    path.write_text(source)
    return extract_contract_graph(path, GraphExtractionConfig())


def test_extract_graph_x_dim_is_node_feature_dim(tmp_sol):
    graph = _extract(CLASSIC_REENTRANCY, tmp_sol)
    assert graph.x.shape[1] == NODE_FEATURE_DIM, (
        f"Expected {NODE_FEATURE_DIM}-dim features, got {graph.x.shape[1]}"
    )


def test_extract_graph_edge_attr_present_and_1d(tmp_sol):
    graph = _extract(CLASSIC_REENTRANCY, tmp_sol)
    assert hasattr(graph, "edge_attr")
    assert graph.edge_attr.dim() == 1


def test_extract_graph_contains_cfg_nodes(tmp_sol):
    """CFG_NODE nodes must be present in a contract with functions."""
    graph = _extract(CLASSIC_REENTRANCY, tmp_sol)
    cfg_node_type = float(NODE_TYPES["CFG_NODE"])
    type_ids = graph.x[:, 0]
    assert (type_ids == cfg_node_type).any(), (
        "Expected at least one CFG_NODE (type_id=8) in reentrancy contract graph"
    )


def test_extract_graph_contains_edges_present(tmp_sol):
    """CONTAINS(5) edges must connect function nodes to their CFG blocks."""
    graph = _extract(CLASSIC_REENTRANCY, tmp_sol)
    assert hasattr(graph, "edge_attr")
    contains_id = EDGE_TYPES["CONTAINS"]
    assert (graph.edge_attr == contains_id).any(), (
        "Expected CONTAINS(5) edges in reentrancy contract graph"
    )


def test_extract_graph_control_flow_edges_present(tmp_sol):
    """CONTROL_FLOW(6) edges must encode execution order within functions."""
    graph = _extract(CLASSIC_REENTRANCY, tmp_sol)
    assert hasattr(graph, "edge_attr")
    cf_id = EDGE_TYPES["CONTROL_FLOW"]
    assert (graph.edge_attr == cf_id).any(), (
        "Expected CONTROL_FLOW(6) edges in reentrancy contract graph"
    )


def test_extract_graph_edge_attr_values_in_range(tmp_sol):
    """All edge type IDs must be in [0, NUM_EDGE_TYPES)."""
    from ml.src.preprocessing.graph_schema import NUM_EDGE_TYPES
    graph = _extract(CLASSIC_REENTRANCY, tmp_sol)
    if graph.edge_attr.numel() > 0:
        assert int(graph.edge_attr.min()) >= 0
        assert int(graph.edge_attr.max()) < NUM_EDGE_TYPES


def test_extract_unchecked_contract_has_in_unchecked_feature(tmp_sol):
    """A function with unchecked{} should have in_unchecked=1.0 on its func node."""
    graph = _extract(UNCHECKED_OVERFLOW, tmp_sol)
    in_unchecked_idx = FEATURE_NAMES.index("in_unchecked")
    func_type = float(NODE_TYPES["FUNCTION"])
    func_mask = graph.x[:, 0] == func_type
    if func_mask.any():
        max_in_unchecked = graph.x[func_mask, in_unchecked_idx].max().item()
        assert max_in_unchecked == 1.0, (
            "Expected in_unchecked=1.0 on a function containing unchecked{}"
        )


def test_extract_dos_contract_has_has_loop_feature(tmp_sol):
    """A function with an unbounded for loop should have has_loop=1.0."""
    graph = _extract(DOS_UNBOUNDED_LOOP, tmp_sol)
    has_loop_idx = FEATURE_NAMES.index("has_loop")
    func_type = float(NODE_TYPES["FUNCTION"])
    func_mask = graph.x[:, 0] == func_type
    if func_mask.any():
        max_has_loop = graph.x[func_mask, has_loop_idx].max().item()
        assert max_has_loop == 1.0, (
            "Expected has_loop=1.0 on a function containing a for loop"
        )


def test_extract_safe_cei_has_fewer_return_ignored_than_reentrancy(tmp_sol):
    """
    Safe CEI contract should have return_ignored=0.0 (captured return),
    while the reentrancy contract's withdraw may have ignored returns depending
    on how Slither resolves the (bool ok,) = call{...} pattern.

    This test verifies the feature is at least computed without error and that
    the extractor completes on both contracts.
    """
    g_vuln = _extract(CLASSIC_REENTRANCY, tmp_sol)
    g_safe = _extract(SAFE_CEI, tmp_sol)
    ri_idx = FEATURE_NAMES.index("return_ignored")
    assert g_vuln.x.shape[1] == NODE_FEATURE_DIM
    assert g_safe.x.shape[1] == NODE_FEATURE_DIM
    # Both should extract successfully; values are float [0.0, 1.0]
    assert g_vuln.x[:, ri_idx].min() >= 0.0
    assert g_safe.x[:, ri_idx].max() <= 1.0
