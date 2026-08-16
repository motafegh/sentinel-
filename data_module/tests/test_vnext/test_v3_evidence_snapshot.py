from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "docs/plan/ml-R4/scripts/p8_snapshot_logical_v3_evidence.py"
spec = importlib.util.spec_from_file_location("p8_snapshot_logical_v3_evidence", SCRIPT)
assert spec is not None and spec.loader is not None
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)

MANIFEST_SHA = "a" * 64
BINDING_SHA = "b" * 64
BINDING_DIGEST = "c" * 64
SENSITIVITY_SHA = "d" * 64
SELECTOR_SHA = "e" * 64
SOURCE_COMMIT = "f" * 40


def _payloads():
    manifest = {
        "dataset_version": snapshot.DATASET_VERSION,
        "grouping_version": snapshot.GROUPING_VERSION,
        "partition_version": snapshot.PARTITION_VERSION,
        "status": snapshot.BOUND_STATUS,
        "confirmed_negative_rows": 0,
        "address_literal_grouping_authority": False,
        "role_contract_counts": {"MODEL_SELECTION": 2, "INTERNAL_AUDIT": 2},
        "representation_binding_report": {
            "sha256": BINDING_SHA,
            "binding_digest_sha256": BINDING_DIGEST,
        },
    }
    partition = {
        "partition_version": snapshot.PARTITION_VERSION,
        "grouping_version": snapshot.GROUPING_VERSION,
        "confirmed_negative_rows": 0,
        "role_contract_counts": manifest["role_contract_counts"],
    }
    binding = {
        "passed": True,
        "dataset_version": snapshot.DATASET_VERSION,
        "logical_grouping_version": snapshot.GROUPING_VERSION,
        "logical_partition_version": snapshot.PARTITION_VERSION,
        "binding_digest_sha256": BINDING_DIGEST,
    }
    summary = {
        "versions": {
            "dataset": snapshot.DATASET_VERSION,
            "grouping": snapshot.GROUPING_VERSION,
            "partition": snapshot.PARTITION_VERSION,
        },
        "physical_binding": {
            "passed": True,
            "binding_digest_sha256": BINDING_DIGEST,
        },
        "semantic_invariants": {
            "target_counts_unchanged_from_v2": True,
            "training_strength_counts_unchanged_from_v2": True,
            "confirmed_negative_rows": 0,
        },
    }
    lineage = {
        "dataset_version": snapshot.DATASET_VERSION,
        "grouping_version": snapshot.GROUPING_VERSION,
        "partition_version": snapshot.PARTITION_VERSION,
        "publication_manifest_sha256": MANIFEST_SHA,
        "representation_binding_digest_sha256": BINDING_DIGEST,
        "source_commit": SOURCE_COMMIT,
    }
    acceptance = {
        "status": "PASS",
        "checks": {"one": True, "two": True},
        "lineage": dict(lineage),
        "training_authorized": False,
    }
    grouping_audit = {
        "v3_policy_check": {"passed": True, "address_edge_count": 0}
    }
    sensitivity = {
        "lineage": dict(lineage),
        "full_training_authorized": False,
    }
    queue = {
        "dataset_version": snapshot.DATASET_VERSION,
        "partition_version": snapshot.PARTITION_VERSION,
        "publication_manifest_sha256": MANIFEST_SHA,
        "source_commit": SOURCE_COMMIT,
        "negative_truth_claim": False,
        "group_uniqueness_scope": "GLOBAL_ACROSS_ENABLED_CLASSES",
        "queued_cells": 2,
        "reserved_group_ids": ["g1", "g2"],
        "candidates": [
            {
                "group_id": "g1",
                "candidate_status": "PENDING_REVIEW",
                "current_target_value": None,
                "negative_truth_claim": False,
                "role_at_queue_creation": "TRAIN_UNLABELED",
                "publication_manifest_sha256": MANIFEST_SHA,
            },
            {
                "group_id": "g2",
                "candidate_status": "PENDING_REVIEW",
                "current_target_value": None,
                "negative_truth_claim": False,
                "role_at_queue_creation": "TRAIN_UNLABELED",
                "publication_manifest_sha256": MANIFEST_SHA,
            },
        ],
    }
    selector = {
        "lineage": dict(lineage),
        "experiment_only": True,
        "promotion_authorized": False,
        "changes_bound_representations": False,
        "failures_total": 0,
        "guarded_target_coverage_regressed_records": 0,
    }
    gpu = {
        "status": snapshot.GPU_STATUS,
        "source_commit": SOURCE_COMMIT,
        "dataset_version": snapshot.DATASET_VERSION,
        "grouping_version": snapshot.GROUPING_VERSION,
        "partition_version": snapshot.PARTITION_VERSION,
        "publication_manifest_sha256": MANIFEST_SHA,
        "representation_binding_digest_sha256": BINDING_DIGEST,
        "sensitivity_report_sha256": SENSITIVITY_SHA,
        "identical_initialization_verified": True,
        "runtime_scope": {
            "worst_case_probes_required": 4,
            "worst_case_probes_completed": 4,
            "checkpoint_written": False,
            "run12_weights_loaded": False,
        },
        "full_training_authorized": False,
        "selector_promotion_authorized": False,
    }
    return {
        "manifest": manifest,
        "manifest_sha256": MANIFEST_SHA,
        "partition": partition,
        "binding": binding,
        "binding_sha256": BINDING_SHA,
        "summary": summary,
        "acceptance": acceptance,
        "grouping_audit": grouping_audit,
        "sensitivity": sensitivity,
        "sensitivity_sha256": SENSITIVITY_SHA,
        "queue": queue,
        "selector": selector,
        "selector_sha256": SELECTOR_SHA,
        "gpu": gpu,
        "current_source_commit": SOURCE_COMMIT,
    }


def test_snapshot_coherence_accepts_one_bound_v3_lineage():
    report = snapshot.validate_snapshot_coherence(**_payloads())
    assert report["status"] == "PASS"
    assert all(report["checks"].values())


def test_snapshot_coherence_rejects_stale_queue_manifest_binding():
    payloads = _payloads()
    payloads["queue"]["publication_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="queue_manifest_matches"):
        snapshot.validate_snapshot_coherence(**payloads)


def test_snapshot_coherence_rejects_incomplete_gpu_probes():
    payloads = _payloads()
    payloads["gpu"]["runtime_scope"]["worst_case_probes_completed"] = 3
    with pytest.raises(ValueError, match="gpu_worst_case_probes_complete"):
        snapshot.validate_snapshot_coherence(**payloads)


def test_snapshot_coherence_rejects_mixed_research_source_commit():
    payloads = _payloads()
    payloads["queue"]["source_commit"] = "0" * 40
    with pytest.raises(ValueError, match="research_source_commit_consistent"):
        snapshot.validate_snapshot_coherence(**payloads)
