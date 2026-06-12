"""Tests for sentinel_data.export.chunker (chunk_export end-to-end)."""
import json
import os
import time
import pytest
import torch
from pathlib import Path
from torch_geometric.data import Data

from sentinel_data.export.chunker import chunk_export, _hash_export_data
from sentinel_data.labeling.schema import class_names


def _make_splits(tmp_path: Path, n_train: int = 20, n_val: int = 5, n_test: int = 5,
                 source: str = "solidifi") -> Path:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    names = class_names()
    i = 0
    for split_name, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
        rows = []
        for _ in range(count):
            sha = f"{i:064d}"
            rows.append({
                "sha256": sha, "source": source, "n_pos": 0,
                "tier": "T0", "classes": {n: 0 for n in names},
                "primary_class": "NonVulnerable", "loc": 0,
            })
            i += 1
        (splits_dir / f"{split_name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows)
        )
    return splits_dir


def _make_reps(rep_root: Path, preproc_root: Path, splits_dir: Path,
               source: str = "solidifi") -> None:
    """Write .pt, .tokens.pt, .rep.json, .meta.json, .sol for every split row."""
    (rep_root / source).mkdir(parents=True, exist_ok=True)
    (preproc_root / source).mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        for line in (splits_dir / f"{split_name}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            sha = c["sha256"]
            g = Data(x=torch.randn(4, 12),
                     edge_index=torch.zeros(2, 0, dtype=torch.long),
                     edge_attr=torch.zeros(0, dtype=torch.long))
            torch.save(g, rep_root / source / f"{sha}.pt")
            torch.save(torch.zeros(4, 512, dtype=torch.long),
                       rep_root / source / f"{sha}.tokens.pt")
            (rep_root / source / f"{sha}.rep.json").write_text(json.dumps({
                "sha256": sha, "source": source, "node_count": 4, "edge_count": 0,
                "solc_version": "0.8.4", "schema_version": "v9",
            }))
            (preproc_root / source / f"{sha}.meta.json").write_text(json.dumps({
                "sha256": sha, "version_bucket": "modern",
                "has_unchecked_block": False, "dedup_group_id": None,
            }))
            (preproc_root / source / f"{sha}.sol").write_text(
                "pragma solidity ^0.8.0;\nfunction foo() public {}\n"
            )


def test_chunk_export_produces_all_files(tmp_path):
    splits_dir = _make_splits(tmp_path, n_train=10, n_val=2, n_test=2)
    rep_root = tmp_path / "representations"
    preproc_root = tmp_path / "preprocessed"
    _make_reps(rep_root, preproc_root, splits_dir)
    out_dir = tmp_path / "export"

    manifest = chunk_export(rep_root, preproc_root, splits_dir, out_dir)

    assert (out_dir / "labels.parquet").exists()
    assert (out_dir / "metadata.parquet").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "graphs" / "_shard_index.json").exists()
    assert (out_dir / "tokens" / "_shard_index.json").exists()
    assert manifest.n_contracts == 14
    assert manifest.n_contracts_with_reps == 14


def test_chunk_export_manifest_written_last(tmp_path):
    """manifest.json must have the latest mtime (Fix A regression test)."""
    splits_dir = _make_splits(tmp_path, n_train=5, n_val=1, n_test=1)
    rep_root = tmp_path / "representations"
    preproc_root = tmp_path / "preprocessed"
    _make_reps(rep_root, preproc_root, splits_dir)
    out_dir = tmp_path / "export"

    chunk_export(rep_root, preproc_root, splits_dir, out_dir)

    manifest_mtime = os.path.getmtime(out_dir / "manifest.json")
    data_mtimes = [
        os.path.getmtime(out_dir / "labels.parquet"),
        os.path.getmtime(out_dir / "metadata.parquet"),
    ]
    for shard in (out_dir / "graphs").glob("graphs-*.pt"):
        data_mtimes.append(os.path.getmtime(shard))
    # manifest must be at least as recent as all data files
    assert manifest_mtime >= max(data_mtimes), "manifest.json must be written LAST"


def test_chunk_export_artifact_hash_excludes_manifest(tmp_path):
    """Modifying manifest.json must NOT change the artifact_hash (Fix A)."""
    splits_dir = _make_splits(tmp_path, n_train=5, n_val=1, n_test=1)
    rep_root = tmp_path / "representations"
    preproc_root = tmp_path / "preprocessed"
    _make_reps(rep_root, preproc_root, splits_dir)
    out_dir = tmp_path / "export"

    manifest = chunk_export(rep_root, preproc_root, splits_dir, out_dir)
    original_hash = manifest.artifact_hash

    # Tamper with manifest.json
    manifest_path = out_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    raw["tampered"] = True
    manifest_path.write_text(json.dumps(raw))

    # Recompute hash — must be unchanged (manifest.json excluded)
    recomputed = _hash_export_data(out_dir)
    assert recomputed == original_hash, "manifest.json must not affect the artifact_hash"


def test_chunk_export_hash_changes_on_data_tamper(tmp_path):
    """Modifying a data file MUST change the artifact_hash."""
    splits_dir = _make_splits(tmp_path, n_train=5, n_val=1, n_test=1)
    rep_root = tmp_path / "representations"
    preproc_root = tmp_path / "preprocessed"
    _make_reps(rep_root, preproc_root, splits_dir)
    out_dir = tmp_path / "export"

    manifest = chunk_export(rep_root, preproc_root, splits_dir, out_dir)
    original_hash = manifest.artifact_hash

    # Tamper with labels.parquet
    labels_path = out_dir / "labels.parquet"
    labels_path.write_bytes(labels_path.read_bytes() + b"\x00")

    recomputed = _hash_export_data(out_dir)
    assert recomputed != original_hash, "Tampering with a data file must change the hash"


def test_chunk_export_split_counts(tmp_path):
    splits_dir = _make_splits(tmp_path, n_train=10, n_val=3, n_test=3)
    rep_root = tmp_path / "representations"
    preproc_root = tmp_path / "preprocessed"
    _make_reps(rep_root, preproc_root, splits_dir)
    out_dir = tmp_path / "export"

    manifest = chunk_export(rep_root, preproc_root, splits_dir, out_dir)
    assert len(manifest.splits["train"]) == 10
    assert len(manifest.splits["val"]) == 3
    assert len(manifest.splits["test"]) == 3
