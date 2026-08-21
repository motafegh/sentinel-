"""Tests for fail-closed v10 candidate population binding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from sentinel_data.preprocessing.r4_versions import (
    V10_REPRESENTATION_EXTRACTOR_VERSION,
    V10_REPRESENTATION_ROOT_NAME,
)
from sentinel_data.vnext.r4_v10_binding import bind_v10_candidate


def _write_representation(root: Path, *, schema: str, token_bytes_from: Path | None = None) -> None:
    source = "fixture"
    contract_id = "a" * 64
    directory = root / source
    directory.mkdir(parents=True)
    graph = Data(
        x=torch.zeros((2, 12), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([12 if schema == "v10" else 11], dtype=torch.long),
    )
    graph.node_metadata = [{"name": "Vault"}, {"name": "f"}]
    graph.contract_name = "Vault"
    graph.num_nodes = 2
    graph.num_edges = 1
    if schema == "v10":
        graph.graph_schema_version = "v10"
        graph.representation_extractor_version = V10_REPRESENTATION_EXTRACTOR_VERSION
        graph.unclassified_call_ir = []
        graph.classified_call_ir_counts = {
            "HIGH_LEVEL_CALL": 0,
            "LOW_LEVEL_CALL": int(schema == "v10"),
            "ETHER_TRANSFER": 0,
            "ETHER_SEND": 0,
            "LIBRARY_CALL": 0,
            "CONTRACT_CREATION": 0,
        }
        graph.emitted_call_edge_counts = dict(graph.classified_call_ir_counts)
        graph.call_mapping_errors = []
    torch.save(graph, directory / f"{contract_id}.pt")

    token_path = directory / f"{contract_id}.tokens.pt"
    if token_bytes_from is not None:
        token_path.write_bytes(token_bytes_from.read_bytes())
    else:
        tokens = {
            "input_ids": torch.zeros((4, 512), dtype=torch.long),
            "attention_mask": torch.zeros((4, 512), dtype=torch.long),
            "sha256": contract_id,
            "source": source,
            "coverage_schema_version": "fixture-v1",
            "pre_subsampling_window_count": 1,
            "pre_subsampling_code_tokens": 0,
            "selected_window_indices": [0],
            "selected_code_token_ranges": [],
            "retained_unique_code_tokens": 0,
            "retained_token_ratio": 0.0,
            "content_tokens_per_window": 510,
            "coverage_interpretation": "diagnostic_only_no_adequacy_threshold",
        }
        torch.save(tokens, token_path)

    sidecar = {
        "sha256": contract_id,
        "source": source,
        "schema_version": schema,
        "extractor_version": (
            V10_REPRESENTATION_EXTRACTOR_VERSION if schema == "v10" else "v2.2-r4-repaired"
        ),
        "graph_target_policy": "file_level_inheritance_leaf_union_v1",
        "requested_contract_names": ["Vault"],
        "actual_contract_names": ["Vault"],
        "graph_component_count": 1,
        "node_count": 2,
        "edge_count": 1,
        "coverage_schema_version": "fixture-v1",
        "pre_subsampling_window_count": 1,
        "pre_subsampling_code_tokens": 0,
        "selected_window_indices": [0],
        "selected_code_token_ranges": [],
        "retained_unique_code_tokens": 0,
        "retained_token_ratio": 0.0,
        "content_tokens_per_window": 510,
        "coverage_interpretation": "diagnostic_only_no_adequacy_threshold",
    }
    if schema == "v10":
        sidecar["token_lineage"] = "accepted_v9_byte_copy"
        sidecar["unclassified_call_ir"] = []
        sidecar["unclassified_call_ir_count"] = 0
        sidecar["classified_call_ir_counts"] = dict(graph.classified_call_ir_counts)
        sidecar["emitted_call_edge_counts"] = dict(graph.emitted_call_edge_counts)
        sidecar["call_mapping_errors"] = []
    (directory / f"{contract_id}.rep.json").write_text(json.dumps(sidecar))


def test_v10_binding_passes_but_does_not_authorize_training(tmp_path: Path) -> None:
    accepted = tmp_path / "representations-r4-v2"
    candidate = tmp_path / V10_REPRESENTATION_ROOT_NAME
    _write_representation(accepted, schema="v9")
    accepted_token = accepted / "fixture" / f"{'a' * 64}.tokens.pt"
    _write_representation(candidate, schema="v10", token_bytes_from=accepted_token)

    report_path = tmp_path / "report.json"
    report = bind_v10_candidate(
        candidate_root=candidate,
        accepted_v9_root=accepted,
        report_path=report_path,
    )

    assert report["passed"] is True
    assert report["token_byte_identical_contracts"] == 1
    assert report["physical_acceptance"] is False
    assert report["training_authorized"] is False
    assert str(tmp_path) not in report_path.read_text()


def test_v10_binding_rejects_token_drift(tmp_path: Path) -> None:
    accepted = tmp_path / "representations-r4-v2"
    candidate = tmp_path / V10_REPRESENTATION_ROOT_NAME
    _write_representation(accepted, schema="v9")
    accepted_token = accepted / "fixture" / f"{'a' * 64}.tokens.pt"
    _write_representation(candidate, schema="v10", token_bytes_from=accepted_token)
    with (candidate / "fixture" / f"{'a' * 64}.tokens.pt").open("ab") as handle:
        handle.write(b"drift")

    report = bind_v10_candidate(candidate_root=candidate, accepted_v9_root=accepted)
    assert report["passed"] is False
    assert "token bytes differ" in report["errors"][0]["detail"]


def test_v10_binding_rejects_wrong_candidate_root_name(tmp_path: Path) -> None:
    accepted = tmp_path / "representations-r4-v2"
    candidate = tmp_path / "wrong"
    accepted.mkdir()
    candidate.mkdir()
    with pytest.raises(ValueError, match="candidate root must be named"):
        bind_v10_candidate(candidate_root=candidate, accepted_v9_root=accepted)
