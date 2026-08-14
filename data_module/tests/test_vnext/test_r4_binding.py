"""Tests for repaired physical representation binding."""

from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from sentinel_data.preprocessing.r4_versions import (
    REPAIRED_DATA_PUBLICATION_ID,
    REPAIRED_REPRESENTATION_EXTRACTOR_VERSION,
)
from sentinel_data.vnext.r4_binding import bind_repaired_publication


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publication(tmp_path, *, contract_id="a" * 64, source="fixture"):
    publication = tmp_path / "publication"
    publication.mkdir()
    rows = [
        {
            "contract_id": contract_id,
            "source": source,
            "group_id": "g1",
            "role": "TRAIN_STRONG",
            "representation_required": True,
        }
    ]
    ml = publication / "ml_targets.parquet"
    pq.write_table(pa.Table.from_pylist(rows), ml)
    manifest = {
        "dataset_version": REPAIRED_DATA_PUBLICATION_ID,
        "artifacts": {"ml_targets": {"sha256": _sha(ml)}},
        "representation_binding_report": None,
        "status": "REPAIRED_CANDIDATE_LOCAL_ACCEPTANCE_REQUIRED",
    }
    (publication / "manifest.json").write_text(json.dumps(manifest))
    return publication


def _representation(tmp_path, *, contract_id="a" * 64, source="fixture", target="Vault"):
    root = tmp_path / "representations"
    directory = root / source
    directory.mkdir(parents=True)
    (directory / f"{contract_id}.pt").write_bytes(b"graph")
    token_payload = {
        "input_ids": torch.zeros((4, 512), dtype=torch.long),
        "attention_mask": torch.ones((4, 512), dtype=torch.long),
        "sha256": contract_id,
        "source": source,
    }
    torch.save(token_payload, directory / f"{contract_id}.tokens.pt")
    sidecar = {
        "sha256": contract_id,
        "source": source,
        "schema_version": "v9",
        "extractor_version": REPAIRED_REPRESENTATION_EXTRACTOR_VERSION,
        "graph_target_policy": "explicit_contract_fail_closed_v1",
        "requested_contract_name": target,
        "actual_contract_name": target,
        "coverage_interpretation": "diagnostic_only_no_adequacy_threshold",
        "retained_token_ratio": 0.4,
        "pre_subsampling_window_count": 10,
        "retained_unique_code_tokens": 2000,
        "pre_subsampling_code_tokens": 5000,
    }
    (directory / f"{contract_id}.rep.json").write_text(json.dumps(sidecar))
    return root


def test_binding_passes_and_binds_manifest_without_physical_root(tmp_path):
    publication = _publication(tmp_path)
    root = _representation(tmp_path)
    report = bind_repaired_publication(
        publication_dir=publication,
        representations_root=root,
    )
    assert report["passed"] is True
    assert report["checked_contracts"] == 1
    assert report["checked_files"] == 3
    assert report["physical_root_recorded"] is False
    assert report["binding_digest_sha256"]
    assert report["token_coverage"][
        "contracts_with_more_than_four_pre_subsampling_windows"
    ] == 1
    manifest = json.loads((publication / "manifest.json").read_text())
    assert manifest["status"] == "REPAIRED_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED"
    assert manifest["representation_binding_report"]["binding_digest_sha256"] == report[
        "binding_digest_sha256"
    ]
    persisted = (publication / "representation_binding_report.json").read_text()
    assert str(root) not in persisted


def test_wrong_graph_target_fails_and_does_not_promote_manifest(tmp_path):
    publication = _publication(tmp_path)
    root = _representation(tmp_path)
    sidecar = root / "fixture" / f"{'a' * 64}.rep.json"
    payload = json.loads(sidecar.read_text())
    payload["actual_contract_name"] = "SafeMath"
    sidecar.write_text(json.dumps(payload))

    report = bind_repaired_publication(
        publication_dir=publication,
        representations_root=root,
    )
    assert report["passed"] is False
    assert report["missing_or_invalid_total"] == 1
    manifest = json.loads((publication / "manifest.json").read_text())
    assert manifest["status"] == "REPAIRED_CANDIDATE_LOCAL_ACCEPTANCE_REQUIRED"
    assert manifest["representation_binding_report"] is None


def test_frozen_token_shape_drift_fails_closed(tmp_path):
    publication = _publication(tmp_path)
    root = _representation(tmp_path)
    token_path = root / "fixture" / f"{'a' * 64}.tokens.pt"
    payload = torch.load(token_path, weights_only=True)
    payload["input_ids"] = torch.zeros((5, 512), dtype=torch.long)
    payload["attention_mask"] = torch.ones((5, 512), dtype=torch.long)
    torch.save(payload, token_path)

    report = bind_repaired_publication(
        publication_dir=publication,
        representations_root=root,
    )
    assert report["passed"] is False
    assert "frozen token shape mismatch" in report["errors"][0]["detail"]
