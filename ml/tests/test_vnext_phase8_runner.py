from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim import AdamW

from ml.src.training.vnext_checkpoint import (
    assert_checkpoint_binding,
    atomic_torch_save,
    build_checkpoint_payload,
    load_checkpoint,
    restore_checkpoint,
)
from ml.src.training.vnext_phase8_config import Phase8Settings
from ml.src.training.vnext_run_control import (
    build_phase8_scheduler,
    is_better_positive_nll,
    optimizer_steps_per_epoch,
)
from ml.src.training.vnext_run_io import (
    RunPaths,
    append_epoch_jsonl,
    initial_checkpoint_index,
    reconcile_resume_index,
)


def _binding(digest: str = "a" * 64):
    return {
        "schema": "sentinel-r4-phase8-run-binding-v1",
        "binding_digest_sha256": digest,
        "source_commit": "deadbeef",
    }


def _toy_stack(settings: Phase8Settings):
    model = torch.nn.Linear(3, 1)
    optimizer = AdamW(
        [{"params": list(model.parameters()), "lr": settings.lr, "name": "toy"}],
        weight_decay=settings.weight_decay,
    )
    scheduler, metadata = build_phase8_scheduler(
        optimizer=optimizer,
        max_lrs=[settings.lr],
        settings=settings,
        loader_batches=88,
    )
    return model, optimizer, scheduler, metadata


def test_scheduler_horizon_uses_grouped_sampler_steps():
    settings = Phase8Settings()
    model, optimizer, scheduler, metadata = _toy_stack(settings)
    assert optimizer_steps_per_epoch(88, 8) == 11
    assert metadata["steps_per_epoch"] == 11
    assert metadata["total_optimizer_steps"] == 1100
    assert scheduler.total_steps == 1100
    del model, optimizer


def test_positive_nll_checkpoint_rule_is_strict_and_finite():
    assert is_better_positive_nll(0.9, None)
    assert is_better_positive_nll(0.8, 0.9)
    assert not is_better_positive_nll(0.9, 0.9)
    assert not is_better_positive_nll(1.0, 0.9)
    with pytest.raises(ValueError):
        is_better_positive_nll(float("nan"), 0.9)
    with pytest.raises(ValueError):
        is_better_positive_nll(0.9, float("inf"))


def test_full_checkpoint_roundtrip_restores_state_and_rng(tmp_path: Path):
    settings = Phase8Settings(epochs=2)
    binding = _binding()
    model, optimizer, scheduler, _ = _toy_stack(settings)

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    x = torch.tensor([[1.0, 2.0, 3.0]])
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    payload = build_checkpoint_payload(
        kind="latest",
        epoch=1,
        global_optimizer_step=1,
        run_binding=binding,
        settings=settings,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        best_positive_nll=0.7,
        best_positive_nll_epoch=1,
        epoch_event={"epoch": 1, "value": 123},
        selection_records=[{"contract_id": "abc", "probability": 0.5}],
    )
    path = tmp_path / "latest.pt"
    identity = atomic_torch_save(payload, path)
    assert identity["epoch"] == 1
    assert identity["kind"] == "latest"
    assert len(identity["sha256"]) == 64

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = float(torch.rand(()))

    with torch.no_grad():
        for param in model.parameters():
            param.add_(100.0)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    checkpoint = load_checkpoint(path, map_location="cpu")
    restored = restore_checkpoint(
        checkpoint,
        expected_run_binding=binding,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert restored == {
        "completed_epoch": 1,
        "next_epoch": 2,
        "global_optimizer_step": 1,
        "best_positive_nll": 0.7,
        "best_positive_nll_epoch": 1,
    }
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert float(torch.rand(())) == expected_torch

    for name, tensor in checkpoint["model_state_dict"].items():
        assert torch.equal(model.state_dict()[name], tensor)


def test_checkpoint_binding_mismatch_fails_closed(tmp_path: Path):
    settings = Phase8Settings(epochs=2)
    model, optimizer, scheduler, _ = _toy_stack(settings)
    payload = build_checkpoint_payload(
        kind="latest",
        epoch=1,
        global_optimizer_step=1,
        run_binding=_binding("a" * 64),
        settings=settings,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        best_positive_nll=None,
        best_positive_nll_epoch=None,
        epoch_event={"epoch": 1},
        selection_records=[],
    )
    path = tmp_path / "latest.pt"
    atomic_torch_save(payload, path)
    checkpoint = load_checkpoint(path, map_location="cpu")
    with pytest.raises(ValueError, match="binding mismatch"):
        assert_checkpoint_binding(checkpoint, _binding("b" * 64))


def test_epoch_jsonl_is_contiguous_idempotent_and_conflict_safe(tmp_path: Path):
    path = tmp_path / "epoch_metrics.jsonl"
    first = {"epoch": 1, "loss": 0.5}
    assert append_epoch_jsonl(path, first)
    assert not append_epoch_jsonl(path, first)

    with pytest.raises(ValueError, match="conflicting epoch 1"):
        append_epoch_jsonl(path, {"epoch": 1, "loss": 0.4})
    with pytest.raises(ValueError, match="expected 2"):
        append_epoch_jsonl(path, {"epoch": 3, "loss": 0.3})

    assert append_epoch_jsonl(path, {"epoch": 2, "loss": 0.4})
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_resume_reconciles_checkpoint_index_after_latest_crash_window(tmp_path: Path):
    settings = Phase8Settings(epochs=20)
    binding = _binding()
    model, optimizer, scheduler, _ = _toy_stack(settings)
    paths = RunPaths.from_root(tmp_path / "run")
    paths.checkpoints.mkdir(parents=True)

    base = build_checkpoint_payload(
        kind="latest",
        epoch=10,
        global_optimizer_step=110,
        run_binding=binding,
        settings=settings,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        best_positive_nll=0.6,
        best_positive_nll_epoch=10,
        epoch_event={"epoch": 10},
        selection_records=[],
    )
    for kind, path in (
        ("best_positive_nll", paths.best_checkpoint),
        ("milestone", paths.milestone_checkpoint(10)),
        ("latest", paths.latest_checkpoint),
    ):
        payload = dict(base)
        payload["kind"] = kind
        atomic_torch_save(payload, path)

    checkpoint = load_checkpoint(paths.latest_checkpoint, map_location="cpu")
    index = reconcile_resume_index(
        index=initial_checkpoint_index(binding),
        paths=paths,
        checkpoint=checkpoint,
        run_binding=binding,
        total_epochs=settings.epochs,
        milestone_interval_epochs=10,
    )
    assert index["latest"]["epoch"] == 10
    assert index["best_positive_nll"]["epoch"] == 10
    assert [item["epoch"] for item in index["milestones"]] == [10]
