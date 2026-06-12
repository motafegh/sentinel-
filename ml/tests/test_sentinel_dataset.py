"""Tests for SentinelDataset and sentinel_collate_fn (Stage 7B).

Tests requiring the real export artifact are guarded with skipif.
The export dir is produced by `sentinel-data export --dataset-version sentinel-v2-baseline-2026-06-12`.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = REPO_ROOT / "data_module" / "data" / "exports" / "sentinel-v2-baseline-2026-06-12"

HAS_EXPORT = EXPORT_DIR.exists() and (EXPORT_DIR / "manifest.json").exists()

skip_no_export = pytest.mark.skipif(
    not HAS_EXPORT,
    reason=f"Export dir not found at {EXPORT_DIR} — run `sentinel-data export` first",
)


@skip_no_export
class TestSentinelDatasetLoads:
    """Basic load and shape tests — require the real export artifact."""

    @pytest.fixture(scope="class")
    def ds_train(self):
        from ml.src.datasets.sentinel_dataset import SentinelDataset
        return SentinelDataset("train", EXPORT_DIR)

    def test_len_positive(self, ds_train):
        assert len(ds_train) > 0

    def test_loads_and_iterates(self, ds_train):
        for i in range(min(10, len(ds_train))):
            graph, tokens, y, contract_id, tier = ds_train[i]
            assert graph is not None
            assert tokens is not None
            assert y is not None
            assert isinstance(contract_id, str)

    def test_graph_shape(self, ds_train):
        graph, _, _, _, _ = ds_train[0]
        assert graph.x.ndim == 2
        assert graph.x.shape[1] == 12, f"Expected NODE_FEATURE_DIM=12, got {graph.x.shape[1]}"
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_attr.ndim == 1

    def test_token_shape(self, ds_train):
        _, tokens, _, _, _ = ds_train[0]
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape == (4, 512)
        assert tokens["attention_mask"].shape == (4, 512)
        assert tokens["input_ids"].dtype == torch.int64
        assert tokens["attention_mask"].dtype == torch.int64

    def test_attention_mask_reconstructed(self, ds_train):
        # attention_mask must be 1 where input_ids != pad_id (1), 0 elsewhere
        _, tokens, _, _, _ = ds_train[0]
        expected = (tokens["input_ids"] != 1).long()
        assert (tokens["attention_mask"] == expected).all()

    def test_label_shape(self, ds_train):
        _, _, y, _, _ = ds_train[0]
        assert y.shape == (10,)
        assert y.dtype == torch.float32
        assert ((y == 0.0) | (y == 1.0)).all(), f"Labels must be binary, got: {y}"

    def test_confidence_tier_str_or_none(self, ds_train):
        tier = ds_train[0][4]
        assert tier is None or isinstance(tier, str)

    def test_confidence_tier_has_none_entries(self, ds_train):
        # NonVulnerable contracts have confidence_tier=None.
        # None entries appear late in JSONL order, so check label_lookup directly.
        none_count = sum(1 for _, tier in ds_train._label_lookup.values() if tier is None)
        assert none_count > 0, "Expected at least one contract with confidence_tier=None in label lookup"

    def test_val_and_test_splits_load(self):
        from ml.src.datasets.sentinel_dataset import SentinelDataset
        for split in ("val", "test"):
            ds = SentinelDataset(split, EXPORT_DIR)
            assert len(ds) > 0


@skip_no_export
class TestSentinelDatasetGates:
    """Schema and artifact hash gate tests."""

    def test_schema_version_gate_raises(self):
        from ml.src.datasets.sentinel_dataset import SentinelDataset
        with pytest.raises(ValueError, match="Graph schema version mismatch"):
            SentinelDataset("train", EXPORT_DIR, model_graph_schema_version="v8")

    def test_artifact_hash_gate_raises(self):
        from ml.src.datasets.sentinel_dataset import SentinelDataset
        from sentinel_data.export.export import SentinelDatasetExport
        # Mock verify_artifact_hash to simulate tampering without touching files
        with patch.object(SentinelDatasetExport, "verify_artifact_hash", return_value=False):
            with pytest.raises(ValueError, match="artifact hash mismatch"):
                SentinelDataset("train", EXPORT_DIR)


@skip_no_export
class TestSentinelCollate:
    """Collate function shape and type tests."""

    @pytest.fixture(scope="class")
    def loader(self):
        from ml.src.datasets.sentinel_dataset import SentinelDataset
        from ml.src.datasets.collate import sentinel_collate_fn
        ds = SentinelDataset("val", EXPORT_DIR)
        return DataLoader(ds, batch_size=4, collate_fn=sentinel_collate_fn, num_workers=0)

    def test_collate_graph_batch(self, loader):
        g_batch, _, _, _, _ = next(iter(loader))
        assert g_batch.x.ndim == 2
        assert g_batch.x.shape[1] == 12

    def test_collate_token_shapes(self, loader):
        _, tokens, _, _, _ = next(iter(loader))
        assert tokens["input_ids"].shape == (4, 4, 512)
        assert tokens["attention_mask"].shape == (4, 4, 512)

    def test_collate_label_shape(self, loader):
        _, _, y_batch, _, _ = next(iter(loader))
        assert y_batch.shape == (4, 10)
        assert y_batch.dtype == torch.float32

    def test_collate_contract_ids_are_list(self, loader):
        _, _, _, contract_ids, _ = next(iter(loader))
        assert isinstance(contract_ids, list)
        assert len(contract_ids) == 4
        assert all(isinstance(cid, str) for cid in contract_ids)

    def test_collate_tiers_are_list(self, loader):
        _, _, _, _, tiers = next(iter(loader))
        assert isinstance(tiers, list)
        assert len(tiers) == 4
        assert all(t is None or isinstance(t, str) for t in tiers)
