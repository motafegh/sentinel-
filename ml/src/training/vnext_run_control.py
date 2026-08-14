"""Run-control policy and scheduler construction for R4 Phase 8."""
from __future__ import annotations

import math
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from ml.src.training.vnext_phase8_config import Phase8Settings

DEFAULT_MILESTONE_INTERVAL_EPOCHS = 10

EXPECTED_TRAIN_FROZEN_ROLES = {
    "TRAIN_STRONG": 275,
    "TRAIN_WEAK": 773,
}
EXPECTED_TRAIN_ACTIVE_ROLES = {
    "TRAIN_STRONG": 275,
    "TRAIN_WEAK": 577,
}
EXPECTED_TRAIN_FROZEN_GROUPS = 703
EXPECTED_TRAIN_ACTIVE_CONTRACTS = 852
EXPECTED_SKIPPED_NO_SIGNAL = {"TRAIN_WEAK": 196}
EXPECTED_SELECTION_ROLES = {"MODEL_SELECTION": 56}
EXPECTED_SELECTION_GROUPS = 51


def seed_phase8(seed: int) -> None:
    """Seed the controlled stochastic streams used by Phase 8."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def optimizer_steps_per_epoch(
    loader_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    if loader_batches <= 0:
        raise ValueError("loader_batches must be > 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    return math.ceil(loader_batches / gradient_accumulation_steps)


def build_phase8_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    max_lrs: list[float],
    settings: Phase8Settings,
    loader_batches: int,
) -> tuple[OneCycleLR, dict[str, Any]]:
    """Retain historical OneCycleLR shape on the actual grouped step horizon."""
    steps_per_epoch = optimizer_steps_per_epoch(
        loader_batches,
        settings.gradient_accumulation_steps,
    )
    total_steps = int(settings.epochs) * steps_per_epoch
    if len(max_lrs) != len(optimizer.param_groups):
        raise ValueError(
            "OneCycleLR max_lrs/optimizer parameter-group count mismatch"
        )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=list(max_lrs),
        epochs=int(settings.epochs),
        steps_per_epoch=steps_per_epoch,
        pct_start=float(settings.warmup_pct),
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1.0e4,
        three_phase=False,
    )
    metadata = {
        "name": "one_cycle_lr",
        "anneal_strategy": "cos",
        "cycle_momentum": True,
        "base_momentum": 0.85,
        "max_momentum": 0.95,
        "div_factor": 25.0,
        "final_div_factor": 1.0e4,
        "three_phase": False,
        "pct_start": float(settings.warmup_pct),
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "max_lrs": [float(v) for v in max_lrs],
    }
    return scheduler, metadata


def is_better_positive_nll(
    candidate: float,
    best: float | None,
) -> bool:
    """Lower NLL is only a MODEL_SELECTION positive-fit diagnostic."""
    value = float(candidate)
    if not math.isfinite(value):
        raise ValueError(f"MODEL_SELECTION positive_nll is not finite: {value}")
    if best is None:
        return True
    best_value = float(best)
    if not math.isfinite(best_value):
        raise ValueError(f"stored best_positive_nll is not finite: {best_value}")
    return value < best_value


def git_source_commit(repo_root: Path) -> str:
    """Return HEAD only when tracked source/config has no local modifications."""
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            "Phase-8 full training requires a clean tracked worktree so "
            "source_commit binds the executed code"
        )
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def validate_phase8_populations(train_ds: Any, selection_ds: Any) -> None:
    checks = [
        (
            train_ds.frozen_role_counts == EXPECTED_TRAIN_FROZEN_ROLES,
            f"unexpected frozen training roles: {train_ds.frozen_role_counts}",
        ),
        (
            train_ds.frozen_group_count == EXPECTED_TRAIN_FROZEN_GROUPS,
            f"unexpected frozen training groups: {train_ds.frozen_group_count}",
        ),
        (
            train_ds.role_counts == EXPECTED_TRAIN_ACTIVE_ROLES,
            f"unexpected active training roles: {train_ds.role_counts}",
        ),
        (
            len(train_ds) == EXPECTED_TRAIN_ACTIVE_CONTRACTS,
            f"unexpected active training contracts: {len(train_ds)}",
        ),
        (
            train_ds.group_count == EXPECTED_TRAIN_FROZEN_GROUPS,
            f"unexpected active training groups: {train_ds.group_count}",
        ),
        (
            train_ds.skipped_no_signal_counts == EXPECTED_SKIPPED_NO_SIGNAL,
            f"unexpected no-signal training siblings: "
            f"{train_ds.skipped_no_signal_counts}",
        ),
        (
            selection_ds.frozen_role_counts == EXPECTED_SELECTION_ROLES,
            f"unexpected frozen MODEL_SELECTION roles: "
            f"{selection_ds.frozen_role_counts}",
        ),
        (
            selection_ds.role_counts == EXPECTED_SELECTION_ROLES,
            f"unexpected active MODEL_SELECTION roles: {selection_ds.role_counts}",
        ),
        (
            len(selection_ds) == 56,
            f"unexpected MODEL_SELECTION contracts: {len(selection_ds)}",
        ),
        (
            selection_ds.group_count == EXPECTED_SELECTION_GROUPS,
            f"unexpected MODEL_SELECTION groups: {selection_ds.group_count}",
        ),
    ]
    for passed, message in checks:
        if not passed:
            raise RuntimeError(message)


def optimizer_binding_config(
    *,
    settings: Phase8Settings,
    parameter_groups: list[dict[str, Any]],
    scheduler_metadata: Mapping[str, Any],
    num_workers: int,
    milestone_interval_epochs: int,
) -> dict[str, Any]:
    max_lrs = list(scheduler_metadata.get("max_lrs") or [])
    if len(max_lrs) != len(parameter_groups):
        raise ValueError("scheduler max_lrs/parameter-group count mismatch")
    return {
        "objective": "masked_positive_bce",
        "model_initialization": "fresh_phase8_factory_no_run12_learned_weights",
        "run12_learned_weights_loaded": False,
        "torch_compile": False,
        "optimizer": "adamw",
        "scheduler": dict(scheduler_metadata),
        "epochs": int(settings.epochs),
        "batch_size": int(settings.batch_size),
        "gradient_accumulation_steps": int(
            settings.gradient_accumulation_steps
        ),
        "base_lr": float(settings.lr),
        "weight_decay": float(settings.weight_decay),
        "parameter_groups": [
            {
                "name": str(group["name"]),
                "max_lr": float(max_lr),
                "weight_decay": float(
                    group.get("weight_decay", settings.weight_decay)
                ),
            }
            for group, max_lr in zip(parameter_groups, max_lrs)
        ],
        "weak_positive_weight": float(settings.weak_positive_weight),
        "aux_loss_weight": float(settings.aux_loss_weight),
        "aux_loss_warmup_epochs": int(settings.aux_loss_warmup_epochs),
        "aux_phase2_loss_weight": float(settings.aux_phase2_loss_weight),
        "jk_entropy_reg_lambda": float(settings.jk_entropy_reg_lambda),
        "grad_clip": float(settings.grad_clip),
        "mixed_precision": "bf16_autocast",
        "num_workers": int(num_workers),
        "persistent_workers": bool(num_workers > 0),
        "model_selection_interval_epochs": 1,
        "fixed_diagnostic_threshold": float(
            settings.fixed_diagnostic_threshold
        ),
        "early_stopping": False,
        "label_smoothing": 0.0,
        "legacy_label_sampler": False,
        "threshold_tuning": False,
        "calibration_fit": False,
        "untouched_acceptance": False,
        "checkpoint_interval_epochs": int(milestone_interval_epochs),
        "primary_completion_checkpoint": "final",
        "limited_selection_checkpoint": "best_positive_nll",
    }


def build_phase8_loaders(
    train_ds: Any,
    selection_ds: Any,
    settings: Phase8Settings,
    num_workers: int,
    sampler: Any,
    collate_fn: Any,
) -> tuple[DataLoader, DataLoader]:
    kwargs = {
        "batch_size": settings.batch_size,
        "num_workers": num_workers,
        "persistent_workers": bool(num_workers > 0),
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=False,
        **kwargs,
    )
    selection_loader = DataLoader(
        selection_ds,
        shuffle=False,
        **kwargs,
    )
    return train_loader, selection_loader


def resolve_output_root(
    *,
    repo_root: Path,
    run_binding: Mapping[str, Any],
    output_dir: Path | None,
    resume_path: Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    if resume_path is not None:
        if resume_path.parent.name != "checkpoints":
            raise ValueError(
                "resume checkpoint must live under <run>/checkpoints "
                "when --output-dir is omitted"
            )
        return resume_path.parent.parent
    return (
        repo_root
        / "ml/logs/r4-phase8"
        / f"run-{run_binding['binding_digest_sha256'][:12]}"
    )


__all__ = [
    "DEFAULT_MILESTONE_INTERVAL_EPOCHS",
    "build_phase8_loaders",
    "build_phase8_scheduler",
    "git_source_commit",
    "is_better_positive_nll",
    "optimizer_binding_config",
    "optimizer_steps_per_epoch",
    "resolve_output_root",
    "seed_phase8",
    "validate_phase8_populations",
]
