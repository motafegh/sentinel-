"""
test_dataset.py — Unit tests for DualPathDataset and dual_path_collate_fn.

All tests use temporary directories with synthetic .pt files so no real
training data is required.
"""

from __future__ import annotations

import torch
import pytest
from pathlib import Path
from torch_geometric.data import Data

from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_graph(path: Path, stem: str, n_nodes: int = 5) -> None:
    x          = torch.randn(n_nodes, 8)
    src        = torch.arange(n_nodes - 1)
    dst        = torch.arange(1, n_nodes)
    edge_index = torch.stack([src, dst])
    graph      = Data(x=x, edge_index=edge_index, y=torch.tensor(0))
    torch.save(graph, path / f"{stem}.pt")


def _write_tokens(path: Path, stem: str) -> None:
    tokens = {
        "input_ids":      torch.ones(512, dtype=torch.long),
        "attention_mask": torch.ones(512, dtype=torch.long),
    }
    torch.save(tokens, path / f"{stem}.pt")


@pytest.fixture()
def paired_dirs(tmp_path):
    """Three paired graph+token files in temporary directories."""
    graphs_dir = tmp_path / "graphs"
    tokens_dir = tmp_path / "tokens"
    graphs_dir.mkdir()
    tokens_dir.mkdir()

    stems = ["aaa", "bbb", "ccc"]
    for stem in stems:
        _write_graph(graphs_dir, stem)
        _write_tokens(tokens_dir, stem)

    return graphs_dir, tokens_dir, stems


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def test_dataset_length(paired_dirs):
    graphs_dir, tokens_dir, stems = paired_dirs
    dataset = DualPathDataset(
        graphs_dir=str(graphs_dir),
        tokens_dir=str(tokens_dir),
        validate=False,
    )
    assert len(dataset) == len(stems)


def test_dataset_getitem_returns_three_items(paired_dirs):
    graphs_dir, tokens_dir, _ = paired_dirs
    dataset = DualPathDataset(
        graphs_dir=str(graphs_dir),
        tokens_dir=str(tokens_dir),
        validate=False,
    )
    item = dataset[0]
    assert len(item) == 3, "expected (graph, tokens, label)"


def test_dataset_graph_shape(paired_dirs):
    graphs_dir, tokens_dir, _ = paired_dirs
    dataset = DualPathDataset(
        graphs_dir=str(graphs_dir),
        tokens_dir=str(tokens_dir),
        validate=False,
    )
    graph, tokens, _ = dataset[0]
    assert graph.x.shape[1] == 8, "node features must be 8-dim"


def test_dataset_token_shape(paired_dirs):
    graphs_dir, tokens_dir, _ = paired_dirs
    dataset = DualPathDataset(
        graphs_dir=str(graphs_dir),
        tokens_dir=str(tokens_dir),
        validate=False,
    )
    _, tokens, _ = dataset[0]
    assert tokens["input_ids"].shape == (512,)
    assert tokens["attention_mask"].shape == (512,)


# ---------------------------------------------------------------------------
# Split indices
# ---------------------------------------------------------------------------

def test_dataset_indices_subset(paired_dirs):
    graphs_dir, tokens_dir, _ = paired_dirs
    dataset = DualPathDataset(
        graphs_dir=str(graphs_dir),
        tokens_dir=str(tokens_dir),
        indices=[0, 2],
        validate=False,
    )
    assert len(dataset) == 2


def test_dataset_empty_indices_raises(paired_dirs):
    graphs_dir, tokens_dir, _ = paired_dirs
    with pytest.raises(ValueError, match="empty"):
        DualPathDataset(
            graphs_dir=str(graphs_dir),
            tokens_dir=str(tokens_dir),
            indices=[],
            validate=False,
        )


def test_dataset_out_of_range_index_raises(paired_dirs):
    graphs_dir, tokens_dir, _ = paired_dirs
    with pytest.raises(ValueError, match="out of range"):
        DualPathDataset(
            graphs_dir=str(graphs_dir),
            tokens_dir=str(tokens_dir),
            indices=[999],
            validate=False,
        )


# ---------------------------------------------------------------------------
# Unpaired files — only paired hashes used
# ---------------------------------------------------------------------------

def test_dataset_skips_unpaired_files(tmp_path):
    graphs_dir = tmp_path / "graphs"
    tokens_dir = tmp_path / "tokens"
    graphs_dir.mkdir()
    tokens_dir.mkdir()

    # One paired, one graph-only, one token-only
    _write_graph(graphs_dir, "paired")
    _write_tokens(tokens_dir, "paired")
    _write_graph(graphs_dir, "graph_only")
    _write_tokens(tokens_dir, "token_only")

    dataset = DualPathDataset(
        graphs_dir=str(graphs_dir),
        tokens_dir=str(tokens_dir),
        validate=False,
    )
    assert len(dataset) == 1


# ---------------------------------------------------------------------------
# Binary collation via dual_path_collate_fn
# ---------------------------------------------------------------------------

def test_collate_binary_label_shape(paired_dirs):
    graphs_dir, tokens_dir, _ = paired_dirs
    dataset = DualPathDataset(
        graphs_dir=str(graphs_dir),
        tokens_dir=str(tokens_dir),
        validate=False,
    )
    samples = [dataset[i] for i in range(3)]
    graphs_batch, tokens_batch, labels = dual_path_collate_fn(samples)

    assert labels.shape == (3,), f"binary labels should be [B], got {labels.shape}"
    assert tokens_batch["input_ids"].shape == (3, 512)
    assert tokens_batch["attention_mask"].shape == (3, 512)


def test_collate_graphs_batch_size(paired_dirs):
    graphs_dir, tokens_dir, _ = paired_dirs
    dataset = DualPathDataset(
        graphs_dir=str(graphs_dir),
        tokens_dir=str(tokens_dir),
        validate=False,
    )
    samples = [dataset[i] for i in range(3)]
    graphs_batch, _, _ = dual_path_collate_fn(samples)
    # PyG Batch.num_graphs gives the number of graphs in the batch
    assert graphs_batch.num_graphs == 3


# ---------------------------------------------------------------------------
# Multi-label collation
# ---------------------------------------------------------------------------

def test_collate_multilabel_label_shape(tmp_path):
    graphs_dir = tmp_path / "graphs"
    tokens_dir = tmp_path / "tokens"
    label_csv  = tmp_path / "labels.csv"
    graphs_dir.mkdir()
    tokens_dir.mkdir()

    stems = ["aaa", "bbb", "ccc"]
    for stem in stems:
        _write_graph(graphs_dir, stem)
        _write_tokens(tokens_dir, stem)

    # Write a minimal multilabel CSV
    header = "md5_stem," + ",".join(f"c{i}" for i in range(10))
    rows   = [f"{stem}," + ",".join(["0"] * 10) for stem in stems]
    label_csv.write_text("\n".join([header] + rows))

    dataset = DualPathDataset(
        graphs_dir=str(graphs_dir),
        tokens_dir=str(tokens_dir),
        label_csv=label_csv,
        validate=False,
    )
    samples = [dataset[i] for i in range(3)]
    _, _, labels = dual_path_collate_fn(samples)
    assert labels.shape == (3, 10), f"multi-label labels should be [B, 10], got {labels.shape}"
    assert labels.dtype == torch.float32
