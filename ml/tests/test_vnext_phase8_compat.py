from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch_geometric.data import Data

from ml.src.datasets.vnext_dataset import VNextTrainingDataset, vnext_collate_fn
from ml.src.training.group_sampler import DeterministicGroupSampler
from ml.src.training.vnext_losses import masked_bce_positive_loss, positive_selection_metrics, strength_weights

CLASS_COUNT = 10


def _row(cid: str, role: str, group: str, index: int, strength: str, loss: bool, metric: bool, target=1.0):
    out = {"contract_id": cid, "source": "fixture", "group_id": group, "role": role, "representation_required": True}
    for i in range(CLASS_COUNT):
        active = i == index
        out[f"target_{i}"] = target if active else None
        out[f"strength_{i}"] = strength if active else "NONE"
        out[f"source_loss_eligible_{i}"] = bool(loss and active)
        out[f"effective_loss_mask_{i}"] = bool(loss and active)
        out[f"outcome_metric_mask_{i}"] = bool(metric and active)
        out[f"outcome_state_{i}"] = "CONFIRMED_POSITIVE" if active else "UNKNOWN"
        out[f"policy_decision_id_{i}"] = "fixture"
    return out


def _write_rep(root: Path, cid: str):
    d = root / "fixture"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(Data(x=torch.zeros((2, 12)), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0,), dtype=torch.long)), d / f"{cid}.pt")
    torch.save({"input_ids": torch.zeros((4, 512), dtype=torch.long), "attention_mask": torch.ones((4, 512), dtype=torch.long)}, d / f"{cid}.tokens.pt")


def _fixture(tmp_path: Path, zero=False):
    overlay = tmp_path / "overlay"
    reps = tmp_path / "representations"
    overlay.mkdir()
    rows = [
        _row("a" * 64, "TRAIN_STRONG", "g1", 0, "STRONG", True, False),
        _row("b" * 64, "TRAIN_WEAK", "g2", 8, "WEAK", True, False),
        _row("c" * 64, "MODEL_SELECTION", "g3", 6, "STRONG", False, True),
    ]
    if zero:
        rows[0]["target_0"] = 0.0
    pq.write_table(pa.Table.from_pylist(rows), overlay / "ml_targets.parquet")
    (overlay / "manifest.json").write_text('{"dataset_version":"sentinel-r4-vnext-v1","export_schema_version":"v2","graph_schema_version":"v9","historical_artifacts_mutated":false,"status":"VALIDATED_G7_CANDIDATE","representation_binding_report":{"binding_digest_sha256":"fixture"},"role_contract_counts":{"TRAIN_STRONG":1,"TRAIN_WEAK":1,"MODEL_SELECTION":1}}\n')
    for cid in ("a" * 64, "b" * 64, "c" * 64):
        _write_rep(reps, cid)
    return overlay, reps


def test_adapter_preserves_nulls_and_roles(tmp_path: Path):
    overlay, reps = _fixture(tmp_path)
    ds = VNextTrainingDataset(overlay_dir=overlay, representations_root=reps, roles=("TRAIN_STRONG", "TRAIN_WEAK"), expected_binding_digest=None, verify_publication=False)
    assert len(ds) == 2
    _, _, supervision, _, role, _ = ds[0]
    assert role in {"TRAIN_STRONG", "TRAIN_WEAK"}
    assert torch.isnan(supervision["targets"]).sum().item() == 9
    assert supervision["effective_loss_mask"].sum().item() == 1
    assert not torch.any(supervision["targets"] == 0.0)


def test_model_selection_is_metric_only(tmp_path: Path):
    overlay, reps = _fixture(tmp_path)
    ds = VNextTrainingDataset(overlay_dir=overlay, representations_root=reps, roles=("MODEL_SELECTION",), expected_binding_digest=None, verify_publication=False)
    _, _, supervision, _, role, _ = ds[0]
    assert role == "MODEL_SELECTION"
    assert not supervision["effective_loss_mask"].any()
    assert supervision["outcome_metric_mask"].sum().item() == 1


def test_adapter_rejects_target_zero(tmp_path: Path):
    overlay, reps = _fixture(tmp_path, zero=True)
    with pytest.raises(ValueError, match="refuses target 0"):
        VNextTrainingDataset(overlay_dir=overlay, representations_root=reps, roles=("TRAIN_STRONG",), expected_binding_digest=None, verify_publication=False)


def test_collate_keeps_masks_explicit(tmp_path: Path):
    overlay, reps = _fixture(tmp_path)
    ds = VNextTrainingDataset(overlay_dir=overlay, representations_root=reps, roles=("TRAIN_STRONG", "TRAIN_WEAK"), expected_binding_digest=None, verify_publication=False)
    graphs, tokens, supervision, cids, roles, groups = vnext_collate_fn([ds[0], ds[1]])
    assert graphs.num_graphs == 2
    assert tokens["input_ids"].shape == (2, 4, 512)
    assert supervision["targets"].shape == (2, 10)
    assert supervision["effective_loss_mask"].dtype is torch.bool
    assert len(cids) == len(roles) == len(groups) == 2


def test_masked_loss_ignores_unknown_cells():
    targets = torch.tensor([[1.0, float("nan")]])
    mask = torch.tensor([[True, False]])
    strengths = torch.tensor([[2, 0]], dtype=torch.uint8)
    a = torch.tensor([[0.0, -50.0]], requires_grad=True)
    b = torch.tensor([[0.0, 50.0]], requires_grad=True)
    la = masked_bce_positive_loss(a, targets, mask, strengths, weak_positive_weight=0.25)
    lb = masked_bce_positive_loss(b, targets, mask, strengths, weak_positive_weight=0.25)
    assert torch.allclose(la, lb)
    la.backward()
    assert a.grad[0, 1].item() == 0.0


def test_weak_weight_is_explicit():
    strengths = torch.tensor([[2, 1]], dtype=torch.uint8)
    mask = torch.tensor([[True, True]])
    assert strength_weights(strengths, mask, 0.25).tolist() == [[1.0, 0.25]]


def test_positive_selection_ignores_unmasked_cells():
    targets = torch.tensor([[1.0, float("nan")]])
    mask = torch.tensor([[True, False]])
    a = positive_selection_metrics(torch.tensor([[0.0, -50.0]]), targets, mask)
    b = positive_selection_metrics(torch.tensor([[0.0, 50.0]]), targets, mask)
    assert a["positive_nll"] == pytest.approx(b["positive_nll"])
    assert a["metric_cells"] == 1


def test_group_sampler_is_deterministic_and_group_balanced():
    groups = {"g1": (0, 1, 2), "g2": (3,), "g3": (4, 5)}
    a = DeterministicGroupSampler(groups, seed=123)
    b = DeterministicGroupSampler(groups, seed=123)
    a.set_epoch(4)
    b.set_epoch(4)
    seq = list(a)
    assert seq == list(b)
    assert len(seq) == 3
    assert sum(i in groups["g1"] for i in seq) == 1
    assert sum(i in groups["g2"] for i in seq) == 1
    assert sum(i in groups["g3"] for i in seq) == 1


def test_committed_projection_has_no_negative_targets_and_disabled_masks():
    repo = Path(__file__).resolve().parents[2]
    rows = pq.read_table(repo / "data_module/data/exports/sentinel-r4-vnext-v1/ml_targets.parquet").to_pylist()
    assert len(rows) == 22493
    effective = 0
    for row in rows:
        for i in range(CLASS_COUNT):
            assert row[f"target_{i}"] in (None, 1)
        assert row["effective_loss_mask_3"] is False
        assert row["effective_loss_mask_9"] is False
        effective += sum(bool(row[f"effective_loss_mask_{i}"]) for i in range(CLASS_COUNT))
    assert effective == 852
