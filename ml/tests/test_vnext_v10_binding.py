"""Run-identity stop-line tests for a future authorized v10 lineage."""

from __future__ import annotations

import hashlib
import json

import pytest

from ml.src.datasets.vnext_logical_v3_v10_dataset import (
    V10_PHYSICAL_ACCEPTANCE_SCHEMA,
    V10_TRAINING_AUTHORIZED_STATUS,
)
from ml.src.training import vnext_binding
from sentinel_data.preprocessing.r4_versions import (
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.vnext.policy import CLASS_NAMES
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path, *, physical_acceptance: bool):
    digest = "a" * 64
    report_path = tmp_path / "v10_physical_acceptance.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": V10_PHYSICAL_ACCEPTANCE_SCHEMA,
                "passed": physical_acceptance,
                "physical_acceptance": physical_acceptance,
                "training_authorized": False,
                "graph_schema_version": "v10",
                "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
                "binding_digest_sha256": digest,
            }
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
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
                    "authorized": True,
                    "decision_id": "R4-D-FUTURE",
                    "binding_digest_sha256": digest,
                },
                "inputs": {},
            }
        )
    )
    return manifest_path, digest


def _build(manifest_path, digest):
    return vnext_binding.build_v10_run_binding(
        source_commit="f" * 40,
        manifest_path=manifest_path,
        expected_representation_digest=digest,
        seed=42,
        weak_positive_weight=0.5,
        optimizer_config={"epochs": 100},
        train_contracts=10,
        train_groups=9,
        selection_contracts=2,
        selection_groups=2,
    )


def test_v10_run_binding_rejects_diagnostic_candidate(tmp_path):
    manifest_path, digest = _manifest(tmp_path, physical_acceptance=False)
    with pytest.raises(ValueError, match="physically accepted"):
        _build(manifest_path, digest)


def test_v10_run_binding_binds_exact_versions_and_no_checkpoint_reuse(
    tmp_path, monkeypatch
):
    manifest_path, digest = _manifest(tmp_path, physical_acceptance=True)
    monkeypatch.setattr(vnext_binding, "runtime_binding_metadata", lambda: {"fixture": True})
    binding = _build(manifest_path, digest)
    assert binding["data"]["graph_schema_version"] == "v10"
    assert (
        binding["data"]["representation_extractor_version"]
        == V10_REPRESENTATION_EXTRACTOR_VERSION
    )
    assert binding["limits"]["historical_checkpoint_reuse"] is False
    assert binding["training_authorization"]["decision_id"] == "R4-D-FUTURE"
