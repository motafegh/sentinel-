"""Manifest-boundary tests for the logical-v3 ML adapter."""

from __future__ import annotations

import hashlib
import json

import pytest

from ml.src.datasets.vnext_logical_v3_dataset import LogicalV3TrainingDataset
from sentinel_data.preprocessing.r4_versions import GRAPH_SCHEMA_VERSION
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adapter(tmp_path, *, partition=ROLE_PARTITION_VERSION_V3):
    digest = "a" * 64
    report = {
        "passed": True,
        "dataset_version": DATASET_VERSION_V3,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "binding_digest_sha256": digest,
        "address_literal_grouping_authority": False,
    }
    report_path = tmp_path / "representation_binding_report.json"
    report_path.write_text(json.dumps(report))

    obj = LogicalV3TrainingDataset.__new__(LogicalV3TrainingDataset)
    obj.overlay_dir = tmp_path
    obj.manifest = {
        "dataset_version": DATASET_VERSION_V3,
        "export_schema_version": "v2",
        "partition_version": partition,
        "grouping_version": GROUPING_VERSION_V3,
        "address_literal_grouping_authority": False,
        "confirmed_negative_rows": 0,
        "status": "LOGICAL_V3_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED",
        "representation_binding_report": {
            "sha256": _sha256(report_path),
            "binding_digest_sha256": digest,
        },
    }
    return obj, digest


def test_logical_v3_adapter_accepts_bound_v3_manifest(tmp_path):
    obj, digest = _adapter(tmp_path)
    obj._validate_manifest(digest)
    assert obj.binding_digest == digest


def test_logical_v3_adapter_rejects_v2_partition(tmp_path):
    obj, digest = _adapter(tmp_path, partition="r4-vnext-roles-v2")
    with pytest.raises(ValueError, match="role partition"):
        obj._validate_manifest(digest)
