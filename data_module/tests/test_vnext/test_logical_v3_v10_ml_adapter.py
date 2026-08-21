"""Training stop-line tests for the logical-V3/v10 adapter."""

from __future__ import annotations

import hashlib
import json

import pytest

from ml.src.datasets.vnext_logical_v3_v10_dataset import (
    LogicalV3V10TrainingDataset,
    V10_PHYSICAL_ACCEPTANCE_SCHEMA,
    V10_TRAINING_AUTHORIZED_STATUS,
)
from sentinel_data.preprocessing.r4_versions import (
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)
from sentinel_data.vnext.policy import CLASS_NAMES


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adapter(tmp_path, *, physical_acceptance: bool, training_authorized: bool):
    digest = "a" * 64
    report = {
        "schema": V10_PHYSICAL_ACCEPTANCE_SCHEMA,
        "passed": physical_acceptance,
        "physical_acceptance": physical_acceptance,
        "training_authorized": False,
        "graph_schema_version": "v10",
        "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "binding_digest_sha256": digest,
    }
    report_path = tmp_path / "v10_physical_acceptance.json"
    report_path.write_text(json.dumps(report))
    obj = LogicalV3V10TrainingDataset.__new__(LogicalV3V10TrainingDataset)
    obj.overlay_dir = tmp_path
    obj.manifest = {
        "dataset_version": DATASET_VERSION_V3,
        "export_schema_version": "v2",
        "partition_version": ROLE_PARTITION_VERSION_V3,
        "grouping_version": GROUPING_VERSION_V3,
        "address_literal_grouping_authority": False,
        "confirmed_negative_rows": 0,
        "class_order": list(CLASS_NAMES),
        "graph_schema_version": "v10",
        "representation_extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
        "representation_root_version": V10_REPRESENTATION_ROOT_NAME,
        "status": V10_TRAINING_AUTHORIZED_STATUS,
        "representation_binding_report": {
            "path": report_path.name,
            "sha256": _sha(report_path),
            "binding_digest_sha256": digest,
        },
        "training_authorization": {
            "authorized": training_authorized,
            "decision_id": "R4-D-FUTURE",
            "binding_digest_sha256": digest,
        },
    }
    return obj, digest


def test_v10_adapter_requires_physical_acceptance(tmp_path):
    obj, digest = _adapter(tmp_path, physical_acceptance=False, training_authorized=True)
    with pytest.raises(ValueError, match="physically accepted"):
        obj._validate_manifest(digest)


def test_v10_adapter_requires_separate_training_authorization(tmp_path):
    obj, digest = _adapter(tmp_path, physical_acceptance=True, training_authorized=False)
    with pytest.raises(ValueError, match="no explicit training authorization"):
        obj._validate_manifest(digest)


def test_v10_adapter_accepts_only_both_stop_lines(tmp_path):
    obj, digest = _adapter(tmp_path, physical_acceptance=True, training_authorized=True)
    obj._validate_manifest(digest)
    assert obj.binding_digest == digest
