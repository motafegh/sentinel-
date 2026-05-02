"""
test_cache.py — Unit tests for InferenceCache (T1-A).

All tests use temporary directories; no real model files are needed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from ml.src.inference.cache import InferenceCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n_nodes: int = 3) -> Data:
    x          = torch.randn(n_nodes, 8)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_attr  = torch.zeros(0, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _make_tokens() -> dict:
    return {
        "input_ids":      torch.ones(512, dtype=torch.long),
        "attention_mask": torch.ones(512, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cache_miss_writes_files(tmp_path):
    """Cache miss: put() writes two .pt files under cache_dir."""
    cache = InferenceCache(cache_dir=tmp_path, ttl_seconds=3600)
    key   = "abc123_v1"

    assert cache.get(key) is None, "fresh cache should be a miss"

    cache.put(key, _make_graph(), _make_tokens())

    assert (tmp_path / f"{key}_graph.pt").exists()
    assert (tmp_path / f"{key}_tokens.pt").exists()


def test_cache_hit_returns_same_object(tmp_path):
    """Cache hit: get() returns (graph, tokens) that match what was stored."""
    cache  = InferenceCache(cache_dir=tmp_path, ttl_seconds=3600)
    key    = "abc123_v1"
    graph  = _make_graph()
    tokens = _make_tokens()

    cache.put(key, graph, tokens)
    result = cache.get(key)

    assert result is not None, "should be a cache hit after put()"
    cached_graph, cached_tokens = result
    assert torch.equal(cached_graph.x, graph.x)
    assert torch.equal(cached_tokens["input_ids"], tokens["input_ids"])


def test_ttl_expiry_evicts_entry(tmp_path):
    """TTL expiry: an entry older than ttl_seconds is treated as a miss and deleted."""
    cache = InferenceCache(cache_dir=tmp_path, ttl_seconds=100)
    key   = "abc123_v1"
    cache.put(key, _make_graph(), _make_tokens())

    graph_file = tmp_path / f"{key}_graph.pt"
    mtime = graph_file.stat().st_mtime

    # Advance clock so the entry looks 200 s old (> 100 s TTL).
    with patch("ml.src.inference.cache.time") as mock_time:
        mock_time.time.return_value = mtime + 200
        result = cache.get(key)

    assert result is None, "expired entry should be evicted and return None"
    assert not graph_file.exists(), "eviction should delete the graph file"


def test_cache_key_includes_schema_version(tmp_path):
    """Schema version in key: a v1-keyed entry is a miss when looked up under v2."""
    cache = InferenceCache(cache_dir=tmp_path, ttl_seconds=3600)
    md5   = "deadbeef"

    cache.put(f"{md5}_v1", _make_graph(), _make_tokens())

    assert cache.get(f"{md5}_v1") is not None, "v1 key should hit"
    assert cache.get(f"{md5}_v2") is None,     "v2 key should miss — schema changed"
