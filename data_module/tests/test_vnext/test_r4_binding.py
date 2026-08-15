"""Tests for repaired physical representation binding."""

from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data

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
    graph = Data(
        x=torch.zeros((2, 12), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([5], dtype=torch.long),
    )
    graph.node_metadata = [{"name": "Vault", "type": "CONTRACT"}, {"name": "f", "type": "FUNCTION"}]
    graph.contract_name = target
    graph.has_cei_path = 0
    graph.num_nodes = 2
    graph.num_edges = 1
    torch.save(graph, directory / f"{contract_id}.pt")
    token_payload = {
        "input_ids": torch.zeros((4, 512), dtype=torch.long),
        "attention_mask": torch.ones((4, 512), dtype=torch.long),
        "sha256": contract_id,
        "source": source,
        "coverage_schema_version": "r4-token-coverage-v1",
        "pre_subsampling_window_count": 10,
        "pre_subsampling_code_tokens": 5000,
        "selected_window_indices": [0, 3, 6, 9],
        "selected_code_token_ranges": [[0, 510], [1530, 2040], [3060, 3570], [4590, 5000]],
        "retained_unique_code_tokens": 2000,
        "retained_token_ratio": 0.4,
        "content_tokens_per_window": 510,
        "coverage_interpretation": "diagnostic_only_no_adequacy_threshold",
    }
    torch.save(token_payload, directory / f"{contract_id}.tokens.pt")
    sidecar = {
        "sha256": contract_id,
        "source": source,
        "schema_version": "v9",
        "extractor_version": REPAIRED_REPRESENTATION_EXTRACTOR_VERSION,
        "graph_target_policy": "file_level_inheritance_leaf_union_v1",
        "requested_contract_names": [target],
        "actual_contract_names": [target],
        "requested_contract_name": target,
        "actual_contract_name": target,
        "node_count": 2,
        "edge_count": 1,
        "graph_component_count": 1,
        "coverage_schema_version": "r4-token-coverage-v1",
        "coverage_interpretation": "diagnostic_only_no_adequacy_threshold",
        "retained_token_ratio": 0.4,
        "pre_subsampling_window_count": 10,
        "retained_unique_code_tokens": 2000,
        "pre_subsampling_code_tokens": 5000,
        "selected_window_indices": [0, 3, 6, 9],
        "selected_code_token_ranges": [[0, 510], [1530, 2040], [3060, 3570], [4590, 5000]],
        "content_tokens_per_window": 510,
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
    payload["actual_contract_names"] = ["SafeMath"]
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


def test_graph_bytes_must_deserialize_and_match_schema(tmp_path):
    publication = _publication(tmp_path)
    root = _representation(tmp_path)
    graph_path = root / "fixture" / f"{'a' * 64}.pt"
    graph_path.write_bytes(b"not a graph")

    report = bind_repaired_publication(
        publication_dir=publication,
        representations_root=root,
    )
    assert report["passed"] is False
    assert report["missing_or_invalid_total"] == 1
