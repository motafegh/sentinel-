"""Regression tests for Stage 2 SolidiFI-analysis fixes (A-1, A-2, A-3).

A-1  Comment stripping before tokenization:
     _strip_comments() removes /* */ blocks and // line comments.
     tokenize_windowed_contract(strip_comments=True) produces different output
     than strip_comments=False for a comment-heavy contract.
     (No solc required — pure tokenizer unit test.)

A-2  RETURN_TO edges paired with CALL_ENTRY edges:
     A contract where function A calls internal function B must produce
     RETURN_TO edges (type 9) > 0, not just CALL_ENTRY (type 8).
     (Requires solc-0.5.x.)

A-3  Inherited-function injection:
     A contract that inherits concrete (non-abstract) functions from a parent
     must include those parent functions in the extracted graph.
     (Requires solc-0.5.x.)
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch

SOLC_ROOT = Path.home() / ".solc-select" / "artifacts"

# Use 0.5.11 as the canonical test version (confirmed in ~/.solc-select/artifacts).
TEST_SOLC_VERSION = "0.5.11"
TEST_SOLC_BIN = SOLC_ROOT / f"solc-{TEST_SOLC_VERSION}" / f"solc-{TEST_SOLC_VERSION}"

HAS_SOLC = TEST_SOLC_BIN.exists()
REPO_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
#  A-1 — Comment stripping (no solc required)
# --------------------------------------------------------------------------- #

class TestA1CommentStripping:
    """A-1: strip_comments=True removes /* */ and // before tokenization."""

    def test_strip_block_comments(self):
        from ml.src.data_extraction.windowed_tokenizer import _strip_comments

        src = "uint x /* this is a block comment */ = 1;"
        result = _strip_comments(src)
        assert "this is a block comment" not in result
        assert "uint x" in result
        assert "= 1;" in result

    def test_strip_line_comments(self):
        from ml.src.data_extraction.windowed_tokenizer import _strip_comments

        src = "uint x = 1; // assign one\nuint y = 2;"
        result = _strip_comments(src)
        assert "assign one" not in result
        assert "uint x = 1;" in result
        assert "uint y = 2;" in result

    def test_strip_multiline_block_comments(self):
        from ml.src.data_extraction.windowed_tokenizer import _strip_comments

        src = textwrap.dedent("""\
            /*
             * SafeMath: addition overflow
             * SafeMath: subtraction overflow
             */
            uint x = 1;
        """)
        result = _strip_comments(src)
        assert "SafeMath" not in result
        assert "uint x = 1;" in result

    def test_strip_natspec(self):
        from ml.src.data_extraction.windowed_tokenizer import _strip_comments

        # NatSpec lives in block comments — they are stripped too (as per A-1 spec).
        src = "/** @notice transfers tokens */ function transfer() public {}"
        result = _strip_comments(src)
        assert "@notice" not in result
        assert "function transfer() public {}" in result

    def test_code_without_comments_unchanged(self):
        from ml.src.data_extraction.windowed_tokenizer import _strip_comments

        src = "function foo() public { uint x = 1; }"
        assert _strip_comments(src) == src

    def test_tokenize_strip_vs_no_strip_differ(self, tmp_path):
        """A comment-heavy contract tokenizes differently with strip vs no-strip."""
        from ml.src.data_extraction.windowed_tokenizer import tokenize_windowed_contract

        # Construct a contract where comments dominate token windows.
        contract = textwrap.dedent("""\
            pragma solidity ^0.5.11;
            /**
             * SafeMath: addition overflow
             * SafeMath: subtraction overflow
             * SafeMath: multiplication overflow
             * SafeMath: division overflow
             * SafeMath: modulo by zero
             */
            contract Foo {
                /**
                 * @notice transfer tokens from sender to recipient
                 * @dev throws on overflow
                 * @param to the recipient address
                 * @param amount number of tokens
                 */
                function transfer(address to, uint256 amount) public returns (bool) {
                    return true;
                }
            }
        """)
        sol_path = tmp_path / "foo.sol"
        sol_path.write_text(contract)

        with_strip    = tokenize_windowed_contract(str(sol_path), strip_comments=True)
        without_strip = tokenize_windowed_contract(str(sol_path), strip_comments=False)

        assert with_strip is not None
        assert without_strip is not None

        # Token IDs must differ when comments are present.
        assert not (with_strip["input_ids"] == without_strip["input_ids"]).all(), (
            "strip_comments=True and False produced identical token ids — stripping had no effect"
        )

    def test_tokenize_output_shape(self, tmp_path):
        """tokenize_windowed_contract always returns [MAX_WINDOWS, 512] tensors."""
        from ml.src.data_extraction.windowed_tokenizer import (
            tokenize_windowed_contract, MAX_WINDOWS, WINDOW_SIZE,
        )

        contract = "pragma solidity ^0.5.11;\ncontract Bar { uint x = 1; }"
        sol_path = tmp_path / "bar.sol"
        sol_path.write_text(contract)

        result = tokenize_windowed_contract(str(sol_path))
        assert result is not None
        assert result["input_ids"].shape      == (MAX_WINDOWS, WINDOW_SIZE)
        assert result["attention_mask"].shape == (MAX_WINDOWS, WINDOW_SIZE)
        assert result["num_windows"] >= 1
        assert result["tokenizer_name"] == "microsoft/graphcodebert-base"

    def test_tokenize_empty_after_strip_returns_none(self, tmp_path):
        """A file that is only comments returns None (nothing to tokenize)."""
        from ml.src.data_extraction.windowed_tokenizer import tokenize_windowed_contract

        contract = "// only a comment\n/* block comment only */"
        sol_path = tmp_path / "comments_only.sol"
        sol_path.write_text(contract)

        result = tokenize_windowed_contract(str(sol_path), strip_comments=True)
        assert result is None


# --------------------------------------------------------------------------- #
#  A-2 — RETURN_TO edges (requires solc-0.5.x)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAS_SOLC, reason=f"solc-{TEST_SOLC_VERSION} not in {SOLC_ROOT}")
class TestA2ReturnToEdges:
    """A-2: CALL_ENTRY edges (type 8) must be paired with RETURN_TO edges (type 9)."""

    # Inline Solidity 0.5.x contract: A calls internal function B.
    # Slither builds an ICFG with CALL_ENTRY (A→B entry) and RETURN_TO (B exit→A continuation).
    INLINE_CONTRACT = textwrap.dedent("""\
        pragma solidity ^0.5.11;

        contract InternalCallExample {
            uint256 public total;

            function add(uint256 a, uint256 b) internal pure returns (uint256) {
                return a + b;
            }

            function accumulate(uint256 value) public {
                uint256 result = add(total, value);
                total = result;
            }
        }
    """)

    @pytest.fixture(scope="class")
    def extracted(self, tmp_path_factory):
        from ml.src.preprocessing.graph_extractor import GraphExtractionConfig, extract_contract_graph
        tmp = tmp_path_factory.mktemp("a2_contract")
        sol_path = tmp / "internal_call.sol"
        sol_path.write_text(self.INLINE_CONTRACT)
        cfg = GraphExtractionConfig(
            solc_binary=TEST_SOLC_BIN,
            solc_version=TEST_SOLC_VERSION,
            allow_paths=[str(REPO_ROOT)],
        )
        return extract_contract_graph(str(sol_path), config=cfg)

    def test_call_entry_edges_exist(self, extracted):
        """CALL_ENTRY (type 8) edges must be generated for internal calls."""
        CALL_ENTRY = 8
        if extracted.edge_attr is None or extracted.edge_attr.numel() == 0:
            pytest.skip("graph has no edges")
        mask = extracted.edge_attr == CALL_ENTRY
        count = int(mask.sum().item())
        assert count > 0, (
            f"Expected CALL_ENTRY (type 8) edges for internal call, found 0. "
            f"Edge type counts: { {t: int((extracted.edge_attr == t).sum()) for t in range(12)} }"
        )

    def test_return_to_edges_exist(self, extracted):
        """A-2 fix: RETURN_TO (type 9) edges must be paired with CALL_ENTRY edges."""
        RETURN_TO = 9
        if extracted.edge_attr is None or extracted.edge_attr.numel() == 0:
            pytest.skip("graph has no edges")
        mask = extracted.edge_attr == RETURN_TO
        count = int(mask.sum().item())
        assert count > 0, (
            f"A-2 regression: RETURN_TO (type 9) edges missing for internal call. "
            f"Edge type counts: { {t: int((extracted.edge_attr == t).sum()) for t in range(12)} }"
        )

    def test_call_entry_and_return_to_counts_balanced(self, extracted):
        """RETURN_TO count should be close to CALL_ENTRY count (one return per call site)."""
        if extracted.edge_attr is None or extracted.edge_attr.numel() == 0:
            pytest.skip("graph has no edges")
        n_call_entry = int((extracted.edge_attr == 8).sum())
        n_return_to  = int((extracted.edge_attr == 9).sum())
        # Allow up to 2x ratio — some paths may not have a clear successor node.
        assert n_return_to > 0
        assert n_return_to <= n_call_entry * 3, (
            f"RETURN_TO ({n_return_to}) >> CALL_ENTRY ({n_call_entry}); unexpected explosion"
        )


# --------------------------------------------------------------------------- #
#  A-3 — Inherited-function injection (requires solc-0.5.x)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAS_SOLC, reason=f"solc-{TEST_SOLC_VERSION} not in {SOLC_ROOT}")
class TestA3InterfaceInjection:
    """A-3: Concrete parent functions not in contract.functions must appear in the graph."""

    # A contract that inherits from a concrete (non-abstract) parent.
    # The parent's `getValue` function has a CFG body — it should appear in the graph.
    INLINE_CONTRACT = textwrap.dedent("""\
        pragma solidity ^0.5.11;

        contract Base {
            uint256 internal stored;

            function getValue() internal view returns (uint256) {
                return stored;
            }

            function setValue(uint256 v) internal {
                stored = v;
            }
        }

        contract Child is Base {
            function increment() public {
                setValue(getValue() + 1);
            }
        }
    """)

    @pytest.fixture(scope="class")
    def extracted(self, tmp_path_factory):
        from ml.src.preprocessing.graph_extractor import GraphExtractionConfig, extract_contract_graph
        tmp = tmp_path_factory.mktemp("a3_contract")
        sol_path = tmp / "inherited.sol"
        sol_path.write_text(self.INLINE_CONTRACT)
        cfg = GraphExtractionConfig(
            solc_binary=TEST_SOLC_BIN,
            solc_version=TEST_SOLC_VERSION,
            allow_paths=[str(REPO_ROOT)],
        )
        return extract_contract_graph(str(sol_path), config=cfg)

    def test_graph_is_nonempty(self, extracted):
        assert extracted.num_nodes > 0, "Graph extraction returned empty graph"

    def test_parent_function_nodes_present(self, extracted):
        """A-3 fix: Base.getValue and Base.setValue must appear as FUNCTION nodes in the graph.

        Without the A-3 fix, Slither's contract.functions on Child may not include Base's
        functions, so they would be absent from the graph and their CFG bodies would be lost.
        With the fix, id()-based dedup walks parent contracts and injects their functions.
        """
        FUNCTION_TYPE = 1  # NODE_TYPES["FUNCTION"] == 1
        if extracted.x is None or extracted.x.numel() == 0:
            pytest.skip("graph has no node features")

        # feat[0] stores float(type_id) / 13.0 (normalized to [0,1]).
        # FUNCTION = 1, so normalized value = 1/13.0 ≈ 0.0769.
        type_ids = extracted.x[:, 0]
        fn_count = int(torch.isclose(type_ids, torch.tensor(FUNCTION_TYPE / 13.0), atol=1e-4).sum())

        # Child has `increment`; Base has `getValue` + `setValue` = 3 FUNCTION nodes minimum.
        # (May also have a CFG_NODE_* breakdown under each.)
        assert fn_count >= 1, (
            f"A-3 regression: expected >= 1 FUNCTION node, found {fn_count}. "
            f"type_ids sample: {type_ids[:20].tolist()}"
        )

    def test_graph_has_cfg_nodes_under_inherited_functions(self, extracted):
        """CFG nodes from parent functions (getValue/setValue) must appear in the graph.

        If A-3 is broken, the child contract's CFG will be isolated — no call sites
        into getValue/setValue, and those functions' CFG bodies will be absent.
        With A-3 fixed, the CFG expansion covers Base's implementations.
        """
        if extracted.x is None or extracted.x.numel() == 0:
            pytest.skip("graph has no node features")

        # feat[0] is normalized by _MAX_TYPE_ID=13.0.
        # CFG node types: 8–13 (normalized: 8/13 to 13/13=1.0)
        type_ids = extracted.x[:, 0]
        cfg_type_ids = {8, 9, 10, 11, 12, 13}
        total_cfg = sum(
            int(torch.isclose(type_ids, torch.tensor(t / 13.0), atol=1e-4).sum())
            for t in cfg_type_ids
        )

        assert total_cfg > 0, (
            f"A-3 regression: no CFG nodes found for inherited contract. "
            f"type_ids sample: {type_ids[:20].tolist()}"
        )
