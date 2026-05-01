"""
test_preprocessing.py — Unit tests for ContractPreprocessor.

ContractPreprocessor.__init__ loads the CodeBERT tokenizer from HuggingFace
and _extract_graph calls Slither/solc. Both are mocked so tests run without
any external dependencies or network access.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Helpers: build synthetic outputs that _extract_graph / _tokenize would return
# ---------------------------------------------------------------------------

def _synthetic_graph(contract_hash: str = "abc123") -> Data:
    return Data(
        x=torch.randn(5, 8),
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
    """
    ContractPreprocessor with tokenizer load patched out.
    Returned instance has _extract_graph and _tokenize patched separately
    per test as needed.
    """
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
    # Replace _extract_graph and _tokenize so we don't call Slither / tokenizer
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
# Output shape contract
# ---------------------------------------------------------------------------

def test_process_source_graph_x_shape(preprocessor, tmp_path):
    """graph.x must be [N, 8] — GNNEncoder has in_channels=8 hardcoded."""
    source        = "pragma solidity ^0.8.0; contract A {}"
    contract_hash = hashlib.md5(source.encode()).hexdigest()

    preprocessor._extract_graph = MagicMock(return_value=_synthetic_graph(contract_hash))
    preprocessor._tokenize      = MagicMock(return_value=_synthetic_tokens(contract_hash))
    preprocessor._log_result    = MagicMock()

    graph, _ = preprocessor.process_source(source)
    assert graph.x.shape[1] == 8, f"expected 8-dim node features, got {graph.x.shape[1]}"


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
    """Same source → same contract_hash — enables response caching."""
    source        = "pragma solidity ^0.8.0; contract A {}"
    contract_hash = hashlib.md5(source.encode()).hexdigest()

    synthetic_graph  = _synthetic_graph(contract_hash)
    synthetic_tokens = _synthetic_tokens(contract_hash)
    preprocessor._extract_graph = MagicMock(return_value=synthetic_graph)
    preprocessor._tokenize      = MagicMock(return_value=synthetic_tokens)
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
