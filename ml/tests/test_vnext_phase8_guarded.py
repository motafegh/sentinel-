from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch_geometric.data import Data

import ml.src.datasets.vnext_logical_v3_guarded_dataset as guarded_dataset
import ml.src.training.vnext_guarded_binding as guarded_binding
from ml.src.datasets.vnext_logical_v3_guarded_dataset import (
    LogicalV3GuardedTrainingDataset,
    R4_D011_BINDING_DIGEST,
    R4_D013_BINDING_DIGEST,
)
from ml.src.training.vnext_guarded_run_control import (
    R4_D014_OBJECTIVE_ID,
    guarded_optimizer_binding_config,
    validate_guarded_phase8_populations,
)
from ml.src.training.vnext_phase8_config import Phase8Settings
from sentinel_data.preprocessing.r4_versions import (
    GUARDED_REPRESENTATION_ROOT_NAME,
    GUARDED_TOKEN_LINEAGE_VERSION,
    GUARDED_TOKEN_SELECTOR_VERSION,
    GUARDED_TOKEN_TRANSFORMERS_VERSION,
    HISTORICAL_TOKEN_SELECTOR_VERSION,
    V10_REPRESENTATION_EXTRACTOR_VERSION,
)
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    GROUPING_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(cid: str, role: str, group: str, index: int, *, loss: bool, metric: bool) -> dict:
    strength = "STRONG"
    row = {
        "contract_id": cid,
        "source": "fixture",
        "group_id": group,
        "role": role,
        "representation_required": True,
    }
    for class_index in range(10):
        active = class_index == index
        row[f"target_{class_index}"] = 1.0 if active else None
        row[f"strength_{class_index}"] = strength if active else "NONE"
        row[f"source_loss_eligible_{class_index}"] = bool(loss and active)
        row[f"effective_loss_mask_{class_index}"] = bool(loss and active)
        row[f"outcome_metric_mask_{class_index}"] = bool(metric and active)
        row[f"outcome_state_{class_index}"] = "CONFIRMED_POSITIVE" if active else "UNKNOWN"
        row[f"policy_decision_id_{class_index}"] = "fixture"
    return row


def _write_guarded_rep(root: Path, cid: str, *, effective_selector: str) -> dict:
    source_dir = root / "fixture"
    source_dir.mkdir(parents=True, exist_ok=True)
    graph_path = source_dir / f"{cid}.pt"
    token_path = source_dir / f"{cid}.tokens.pt"
    sidecar_path = source_dir / f"{cid}.rep.json"

    graph = Data(
        x=torch.zeros((2, 12)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0,), dtype=torch.long),
    )
    graph.graph_schema_version = "v10"
    graph.representation_extractor_version = V10_REPRESENTATION_EXTRACTOR_VERSION
    torch.save(graph, graph_path)

    fallback = effective_selector == HISTORICAL_TOKEN_SELECTOR_VERSION
    decision = {
        "requested_selector": GUARDED_TOKEN_SELECTOR_VERSION,
        "control_selector": HISTORICAL_TOKEN_SELECTOR_VERSION,
        "effective_selector": effective_selector,
        "used_control_fallback": fallback,
        "selected_indices": [0],
        "candidate_indices": [0],
        "control_indices": [0],
        "candidate_target_coverage_tokens": 1,
        "control_target_coverage_tokens": 1 if fallback else 0,
        "selected_target_coverage_tokens": 1,
        "pre_subsampling_window_count": 1,
    }
    torch.save(
        {
            "sha256": cid,
            "source": "fixture",
            "token_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
            "token_selector_policy": GUARDED_TOKEN_SELECTOR_VERSION,
            "transformers_version": GUARDED_TOKEN_TRANSFORMERS_VERSION,
            "selector_decision": decision,
            "selected_window_indices": [0],
            "input_ids": torch.zeros((4, 512), dtype=torch.long),
            "attention_mask": torch.ones((4, 512), dtype=torch.long),
        },
        token_path,
    )
    sidecar_path.write_text(json.dumps({"sha256": cid, "source": "fixture"}) + "\n")
    return {
        "source": "fixture",
        "contract_id": cid,
        "effective_selector": effective_selector,
        "used_control_fallback": fallback,
        "selected_window_indices": [0],
        "graph_sha256": _sha(graph_path),
        "tokens_sha256": _sha(token_path),
        "sidecar_sha256": _sha(sidecar_path),
    }


def _fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(guarded_dataset, "EXPECTED_GUARDED_CONTRACTS", 2)
    repo_root = tmp_path / "repo"
    overlay = repo_root / "overlay"
    overlay.mkdir(parents=True)
    physical_root = repo_root / "data_module/data/d5" / GUARDED_REPRESENTATION_ROOT_NAME
    physical_root.mkdir(parents=True)

    train_cid = "a" * 64
    selection_cid = "b" * 64
    rows = [
        _row(train_cid, "TRAIN_STRONG", "g1", 0, loss=True, metric=False),
        _row(selection_cid, "MODEL_SELECTION", "g2", 1, loss=False, metric=True),
    ]
    ml_targets = overlay / "ml_targets.parquet"
    pq.write_table(pa.Table.from_pylist(rows), ml_targets)

    logical_rep_report = overlay / "representation_binding_report.json"
    logical_rep_report.write_text(json.dumps({
        "passed": True,
        "dataset_version": DATASET_VERSION_V3,
        "graph_schema_version": "v9",
        "binding_digest_sha256": "1" * 64,
        "address_literal_grouping_authority": False,
    }) + "\n")
    manifest = {
        "dataset_version": DATASET_VERSION_V3,
        "export_schema_version": "v2",
        "partition_version": ROLE_PARTITION_VERSION_V3,
        "grouping_version": GROUPING_VERSION_V3,
        "address_literal_grouping_authority": False,
        "confirmed_negative_rows": 0,
        "status": "LOGICAL_V3_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED",
        "representation_binding_report": {
            "sha256": _sha(logical_rep_report),
            "binding_digest_sha256": "1" * 64,
        },
        "artifacts": {"ml_targets": {"sha256": _sha(ml_targets)}},
        "role_contract_counts": {"TRAIN_STRONG": 1, "MODEL_SELECTION": 1},
    }
    manifest_path = overlay / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    logical_acceptance = repo_root / "logical_acceptance.json"
    logical_acceptance.write_text(json.dumps({
        "schema": "sentinel-r4-logical-v3-acceptance-v1",
        "status": "PASS",
        "training_authorized": False,
        "versions": {
            "dataset": DATASET_VERSION_V3,
            "grouping": GROUPING_VERSION_V3,
            "partition": ROLE_PARTITION_VERSION_V3,
        },
        "checks": {"confirmed_negative_rows_zero": True},
        "lineage": {"publication_manifest_sha256": _sha(manifest_path)},
        "role_contract_counts": {"TRAIN_STRONG": 1, "TRAIN_WEAK": 0, "MODEL_SELECTION": 1},
        "role_group_counts": {"TRAIN_STRONG": 1, "TRAIN_WEAK": 0, "MODEL_SELECTION": 1},
        "active_supervision": {
            "optimizer_contracts_by_role": {"TRAIN_STRONG": 1, "TRAIN_WEAK": 0},
            "optimizer_contracts": 1,
            "optimizer_groups": 1,
            "model_selection_contracts": 1,
            "model_selection_groups": 1,
        },
    }, sort_keys=True) + "\n")

    records = [
        _write_guarded_rep(physical_root, train_cid, effective_selector=GUARDED_TOKEN_SELECTOR_VERSION),
        _write_guarded_rep(physical_root, selection_cid, effective_selector=HISTORICAL_TOKEN_SELECTOR_VERSION),
    ]
    candidate_manifest = {
        "schema": "sentinel-r4-guarded-token-candidate-manifest-v1",
        "status": "FULL_GUARDED_TOKEN_CANDIDATE",
        "full_population": True,
        "physical_acceptance": False,
        "training_authorized": False,
        "source_commit": "c" * 40,
        "contracts_written": 2,
        "binding_digest_sha256": R4_D013_BINDING_DIGEST,
        "representation_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
        "selector_policy": GUARDED_TOKEN_SELECTOR_VERSION,
        "control_selector": HISTORICAL_TOKEN_SELECTOR_VERSION,
        "records": records,
    }
    candidate_manifest_path = physical_root / "guarded_candidate_manifest.json"
    candidate_manifest_path.write_text(json.dumps(candidate_manifest, sort_keys=True) + "\n")

    relative_root = physical_root.relative_to(repo_root).as_posix()
    physical_acceptance = repo_root / "physical_acceptance.json"
    physical_acceptance.write_text(json.dumps({
        "schema": "sentinel-r4-guarded-token-physical-acceptance-v1",
        "decision_id": "R4-D-013",
        "status": "PASS",
        "decision": "ACCEPTED_IMMUTABLE_LOCAL_GUARDED_TOKEN_REPRESENTATION",
        "physical_acceptance": True,
        "training_authorized": False,
        "accepted_lineage": {
            "binding_digest_sha256": R4_D013_BINDING_DIGEST,
            "candidate_manifest_sha256": _sha(candidate_manifest_path),
            "contracts": 2,
            "representation_lineage": GUARDED_TOKEN_LINEAGE_VERSION,
            "selector_policy": GUARDED_TOKEN_SELECTOR_VERSION,
            "control_selector": HISTORICAL_TOKEN_SELECTOR_VERSION,
            "transformers_version": GUARDED_TOKEN_TRANSFORMERS_VERSION,
            "graph_schema_version": "v10",
            "extractor_version": V10_REPRESENTATION_EXTRACTOR_VERSION,
            "token_shape": [4, 512],
            "graph_parent_decision_id": "R4-D-011",
            "graph_parent_binding_digest_sha256": R4_D011_BINDING_DIGEST,
            "physical_root": relative_root,
        },
    }, sort_keys=True) + "\n")
    return {
        "repo_root": repo_root,
        "overlay": overlay,
        "physical_root": physical_root,
        "logical_acceptance": logical_acceptance,
        "physical_acceptance": physical_acceptance,
        "train_cid": train_cid,
    }


def test_guarded_dataset_binds_v3_and_r4_d013(tmp_path: Path, monkeypatch):
    fx = _fixture(tmp_path, monkeypatch)
    ds = LogicalV3GuardedTrainingDataset(
        repo_root=fx["repo_root"],
        logical_acceptance_path=fx["logical_acceptance"],
        physical_acceptance_path=fx["physical_acceptance"],
        overlay_dir=fx["overlay"],
        representations_root=fx["physical_root"],
        roles=("TRAIN_STRONG",),
    )
    assert ds.binding_digest == R4_D013_BINDING_DIGEST
    assert ds.logical_v3_parent_binding_digest == "1" * 64
    graph, tokens, supervision, cid, role, group = ds[0]
    assert graph.graph_schema_version == "v10"
    assert tokens["input_ids"].shape == (4, 512)
    assert cid == fx["train_cid"]
    assert role == "TRAIN_STRONG"
    assert group == "g1"
    assert supervision["effective_loss_mask"].sum().item() == 1


def test_guarded_dataset_detects_mutated_active_token(tmp_path: Path, monkeypatch):
    fx = _fixture(tmp_path, monkeypatch)
    token_path = fx["physical_root"] / "fixture" / f"{fx['train_cid']}.tokens.pt"
    token_path.write_bytes(token_path.read_bytes() + b"mutated")
    with pytest.raises(ValueError, match="active artifact hash mismatch"):
        LogicalV3GuardedTrainingDataset(
            repo_root=fx["repo_root"],
            logical_acceptance_path=fx["logical_acceptance"],
            physical_acceptance_path=fx["physical_acceptance"],
            overlay_dir=fx["overlay"],
            representations_root=fx["physical_root"],
            roles=("TRAIN_STRONG",),
        )


def test_guarded_optimizer_config_enforces_bounded_scope():
    settings = Phase8Settings(epochs=2)
    config = guarded_optimizer_binding_config(
        settings=settings,
        parameter_groups=[{"name": "toy", "weight_decay": settings.weight_decay}],
        scheduler_metadata={"max_lrs": [settings.lr], "steps_per_epoch": 1, "total_optimizer_steps": 2},
        num_workers=0,
        milestone_interval_epochs=1,
    )
    assert config["objective"] == R4_D014_OBJECTIVE_ID
    assert config["execution_scope"] == "bounded_pilot_only"
    assert config["full_training_authorized"] is False
    with pytest.raises(ValueError, match="bounded pilot"):
        guarded_optimizer_binding_config(
            settings=Phase8Settings(epochs=100),
            parameter_groups=[{"name": "toy", "weight_decay": settings.weight_decay}],
            scheduler_metadata={"max_lrs": [settings.lr]},
            num_workers=0,
            milestone_interval_epochs=10,
        )


def test_guarded_population_validation_uses_logical_v3_acceptance(tmp_path: Path):
    acceptance = tmp_path / "logical.json"
    acceptance.write_text(json.dumps({
        "status": "PASS",
        "role_contract_counts": {"TRAIN_STRONG": 3, "TRAIN_WEAK": 2, "MODEL_SELECTION": 2},
        "role_group_counts": {"TRAIN_STRONG": 2, "TRAIN_WEAK": 2, "MODEL_SELECTION": 2},
        "active_supervision": {
            "optimizer_contracts_by_role": {"TRAIN_STRONG": 2, "TRAIN_WEAK": 2},
            "optimizer_contracts": 4,
            "optimizer_groups": 4,
            "model_selection_contracts": 1,
            "model_selection_groups": 1,
        },
    }) + "\n")

    class TrainDS:
        def __len__(self):
            return 4

    class SelectionDS:
        def __len__(self):
            return 1

    train = TrainDS()
    train.frozen_role_counts = {"TRAIN_STRONG": 3, "TRAIN_WEAK": 2}
    train.role_counts = {"TRAIN_STRONG": 2, "TRAIN_WEAK": 2}
    train.frozen_group_count = 4
    train.group_count = 4
    train.skipped_no_signal_counts = {"TRAIN_STRONG": 1}

    selection = SelectionDS()
    selection.frozen_role_counts = {"MODEL_SELECTION": 2}
    selection.role_counts = {"MODEL_SELECTION": 1}
    selection.frozen_group_count = 2
    selection.group_count = 1
    selection.skipped_no_signal_counts = {"MODEL_SELECTION": 1}

    validate_guarded_phase8_populations(train, selection, logical_acceptance_path=acceptance)


def test_guarded_run_binding_contains_all_authorities(tmp_path: Path, monkeypatch):
    fx = _fixture(tmp_path, monkeypatch)
    decision_path = Path(__file__).resolve().parents[2] / "docs/plan/ml-R4/evidence/2026-10-01_phase8_objective_evaluation/decision.json"
    settings = Phase8Settings(epochs=2)
    optimizer_config = guarded_optimizer_binding_config(
        settings=settings,
        parameter_groups=[{"name": "toy", "weight_decay": settings.weight_decay}],
        scheduler_metadata={"max_lrs": [settings.lr], "steps_per_epoch": 1, "total_optimizer_steps": 2},
        num_workers=0,
        milestone_interval_epochs=1,
    )
    monkeypatch.setattr(guarded_binding, "runtime_binding_metadata", lambda: {"fixture": True})
    payload = guarded_binding.build_guarded_run_binding(
        source_commit="d" * 40,
        repo_root=fx["repo_root"],
        manifest_path=fx["overlay"] / "manifest.json",
        logical_acceptance_path=fx["logical_acceptance"],
        physical_acceptance_path=fx["physical_acceptance"],
        objective_decision_path=decision_path,
        representations_root=fx["physical_root"],
        seed=settings.seed,
        weak_positive_weight=settings.weak_positive_weight,
        optimizer_config=optimizer_config,
        train_contracts=1,
        train_groups=1,
        selection_contracts=1,
        selection_groups=1,
    )
    assert payload["schema"] == "sentinel-r4-phase8-guarded-run-binding-v1"
    assert payload["data"]["graph_parent"]["decision_id"] == "R4-D-011"
    assert payload["data"]["guarded_tokens"]["decision_id"] == "R4-D-013"
    assert payload["objective_evaluation"]["decision_id"] == "R4-D-014"
    assert payload["architecture_config"]["graph_schema_version"] == "v10"
    assert payload["limits"]["full_training_authorized"] is False
    assert len(payload["binding_digest_sha256"]) == 64
