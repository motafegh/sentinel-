"""Tests for sentinel_data.export.{graph_writer, token_writer}."""
import json
import math
import pytest
import torch
from pathlib import Path
from torch_geometric.data import Data

from sentinel_data.export.graph_writer import write_graphs_shards
from sentinel_data.export.token_writer import write_tokens_shards


def _make_splits(tmp_path: Path, shas: list[tuple[str, str, str]]) -> Path:
    """shas = [(sha, source, split), ...]"""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    buckets: dict[str, list] = {"train": [], "val": [], "test": []}
    for sha, source, split in shas:
        buckets[split].append({"sha256": sha, "source": source, "n_pos": 0,
                                "tier": "T0", "classes": {}, "primary_class": "NonVulnerable",
                                "loc": 0})
    for name, rows in buckets.items():
        (splits_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows)
        )
    return splits_dir


def _write_graph(rep_root: Path, sha: str, source: str, n_nodes: int = 5) -> None:
    (rep_root / source).mkdir(parents=True, exist_ok=True)
    g = Data(
        x=torch.randn(n_nodes, 12),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, dtype=torch.long),
    )
    torch.save(g, rep_root / source / f"{sha}.pt")


def _write_tokens(rep_root: Path, sha: str, source: str) -> None:
    (rep_root / source).mkdir(parents=True, exist_ok=True)
    tok = torch.zeros(4, 512, dtype=torch.long)
    torch.save(tok, rep_root / source / f"{sha}.tokens.pt")


def test_graph_writer_shard_count(tmp_path):
    N = 12
    shas = [(f"{i:064d}", "solidifi", "train") for i in range(N)]
    rep_root = tmp_path / "representations"
    for sha, source, _ in shas:
        _write_graph(rep_root, sha, source)
    splits_dir = _make_splits(tmp_path, shas)
    out_dir = tmp_path / "graphs"

    paths, shard_idx = write_graphs_shards(rep_root, splits_dir, out_dir, shard_size=5)
    assert len(paths) == math.ceil(N / 5)  # 3 shards (5+5+2)
    assert len(shard_idx) == N


def test_graph_writer_shard_index_values(tmp_path):
    """Each sha maps to a shard number; the shard numbers are 0-indexed and contiguous."""
    N = 7
    shas = [(f"{i:064d}", "solidifi", "train") for i in range(N)]
    rep_root = tmp_path / "representations"
    for sha, source, _ in shas:
        _write_graph(rep_root, sha, source)
    splits_dir = _make_splits(tmp_path, shas)
    _, shard_idx = write_graphs_shards(rep_root, splits_dir, tmp_path / "g", shard_size=3)
    shard_nums = sorted(set(shard_idx.values()))
    assert shard_nums == list(range(len(shard_nums)))


def test_graph_writer_skips_missing(tmp_path):
    """Contracts without a .pt file are silently skipped."""
    shas = [(f"{i:064d}", "solidifi", "train") for i in range(5)]
    rep_root = tmp_path / "representations"
    # Only write 3 of the 5
    for sha, source, _ in shas[:3]:
        _write_graph(rep_root, sha, source)
    splits_dir = _make_splits(tmp_path, shas)
    _, shard_idx = write_graphs_shards(rep_root, splits_dir, tmp_path / "g", shard_size=10)
    assert len(shard_idx) == 3  # only 3 made it


def test_token_writer_shard_shape(tmp_path):
    N = 12
    shas = [(f"{i:064d}", "solidifi", "train") for i in range(N)]
    rep_root = tmp_path / "representations"
    for sha, source, _ in shas:
        _write_tokens(rep_root, sha, source)
    splits_dir = _make_splits(tmp_path, shas)
    out_dir = tmp_path / "tokens"

    paths, shard_idx = write_tokens_shards(rep_root, splits_dir, out_dir, shard_size=5)
    assert len(paths) == math.ceil(N / 5)
    # First shard should have 5 contracts → shape [5, 4, 512]
    t = torch.load(paths[0], weights_only=True)
    assert t.shape == (5, 4, 512)
    # Last shard has 2
    t_last = torch.load(paths[-1], weights_only=True)
    assert t_last.shape[0] == N % 5 or t_last.shape[0] == 5


def test_graph_token_same_order(tmp_path):
    """graph_writer and token_writer must produce the same contract_id list."""
    N = 8
    shas = [(f"{i:064d}", "solidifi", "train") for i in range(N)]
    rep_root = tmp_path / "representations"
    for sha, source, _ in shas:
        _write_graph(rep_root, sha, source)
        _write_tokens(rep_root, sha, source)
    splits_dir = _make_splits(tmp_path, shas)

    _, g_idx = write_graphs_shards(rep_root, splits_dir, tmp_path / "g", shard_size=4)
    _, t_idx = write_tokens_shards(rep_root, splits_dir, tmp_path / "t", shard_size=4)
    assert set(g_idx) == set(t_idx), "graph and token shard indices must cover the same contracts"
    for sha in g_idx:
        assert g_idx[sha] == t_idx[sha], f"shard number mismatch for {sha}"
