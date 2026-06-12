"""Tests for sentinel_data.export.export.SentinelDatasetExport."""
import json
import pytest
import torch
from pathlib import Path
from torch_geometric.data import Data

from sentinel_data.export.chunker import chunk_export
from sentinel_data.export.export import SentinelDatasetExport
from sentinel_data.labeling.schema import class_names


def _make_splits(tmp_path: Path, n: int = 10, source: str = "solidifi") -> Path:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    names = class_names()
    rows = [{"sha256": f"{i:064d}", "source": source, "n_pos": 0,
              "tier": "T0", "classes": {n: 0 for n in names},
              "primary_class": "NonVulnerable", "loc": 0} for i in range(n)]
    (splits_dir / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (splits_dir / "val.jsonl").write_text("")
    (splits_dir / "test.jsonl").write_text("")
    return splits_dir


def _make_reps(rep_root: Path, preproc_root: Path, splits_dir: Path,
               source: str = "solidifi") -> None:
    (rep_root / source).mkdir(parents=True, exist_ok=True)
    (preproc_root / source).mkdir(parents=True, exist_ok=True)
    for line in (splits_dir / "train.jsonl").read_text().splitlines():
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


def _build_export(tmp_path: Path, n: int = 10) -> Path:
    splits_dir = _make_splits(tmp_path, n=n)
    rep_root = tmp_path / "representations"
    preproc_root = tmp_path / "preprocessed"
    _make_reps(rep_root, preproc_root, splits_dir)
    out_dir = tmp_path / "export"
    chunk_export(rep_root, preproc_root, splits_dir, out_dir)
    return out_dir


def test_sentinel_dataset_export_loads(tmp_path):
    out_dir = _build_export(tmp_path)
    export = SentinelDatasetExport(out_dir)
    assert export.manifest.n_contracts == 10
    assert export.manifest.schema_version == "v1"


def test_sentinel_dataset_export_verify_hash_true(tmp_path):
    out_dir = _build_export(tmp_path)
    export = SentinelDatasetExport(out_dir)
    assert export.verify_artifact_hash() is True


def test_sentinel_dataset_export_verify_hash_false_on_tamper(tmp_path):
    out_dir = _build_export(tmp_path)
    # Tamper with labels.parquet
    labels_path = out_dir / "labels.parquet"
    labels_path.write_bytes(labels_path.read_bytes() + b"\xff")
    export = SentinelDatasetExport(out_dir)
    assert export.verify_artifact_hash() is False


def test_sentinel_dataset_export_manifest_tamper_does_not_affect_hash(tmp_path):
    """Fix A: modifying manifest.json must not break verify_artifact_hash."""
    out_dir = _build_export(tmp_path)
    export = SentinelDatasetExport(out_dir)
    assert export.verify_artifact_hash() is True

    # Tamper manifest
    raw = json.loads((out_dir / "manifest.json").read_text())
    raw["extra"] = "injected"
    (out_dir / "manifest.json").write_text(json.dumps(raw))

    # reload — still verifies OK because manifest is excluded from hash
    export2 = SentinelDatasetExport(out_dir)
    assert export2.verify_artifact_hash() is True


def test_sentinel_dataset_export_get_split_ids(tmp_path):
    out_dir = _build_export(tmp_path, n=10)
    export = SentinelDatasetExport(out_dir)
    train_ids = export.get_split_contract_ids("train")
    assert len(train_ids) == 10
    assert all(len(s) == 64 for s in train_ids)


def test_sentinel_dataset_export_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        SentinelDatasetExport(tmp_path / "nonexistent")


def test_sentinel_dataset_export_repr(tmp_path):
    out_dir = _build_export(tmp_path)
    export = SentinelDatasetExport(out_dir)
    r = repr(export)
    assert "SentinelDatasetExport" in r
    assert "n_contracts=10" in r


def test_import_from_export_module():
    """Fix C: public API is importable from sentinel_data.export."""
    from sentinel_data.export import (
        SentinelDatasetExport, ExportManifest, chunk_export,
        write_labels_parquet, write_metadata_parquet,
        write_graphs_shards, write_tokens_shards,
    )
