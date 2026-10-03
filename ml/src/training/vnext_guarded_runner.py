"""Durable bounded runner for the accepted R4-D-013 guarded Phase-8 seam.

This runner is intentionally separate from the historical G7/v9 full-run
orchestrator. It supports only the governed M4-B recovery proof (2 epochs) and
M4-C bounded dynamics pilot (8 epochs). It can never launch the 100-epoch
Phase-8 horizon.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.optim import AdamW

from ml.src.datasets.vnext_logical_v3_guarded_dataset import (
    LogicalV3GuardedTrainingDataset,
    vnext_collate_fn,
)
from ml.src.training.group_sampler import DeterministicGroupSampler
from ml.src.training.vnext_checkpoint import (
    atomic_torch_save,
    build_checkpoint_payload,
    load_checkpoint,
    restore_checkpoint,
)
from ml.src.training.vnext_epoch import (
    evaluate_positive_selection,
    train_masked_epoch,
)
from ml.src.training.vnext_guarded_binding import build_guarded_run_binding
from ml.src.training.vnext_guarded_run_control import (
    guarded_optimizer_binding_config,
    validate_guarded_phase8_populations,
)
from ml.src.training.vnext_model_factory import build_phase8_v10_model
from ml.src.training.vnext_param_groups import build_parameter_groups
from ml.src.training.vnext_phase8_config import Phase8Settings
from ml.src.training.vnext_run_control import (
    build_phase8_loaders,
    build_phase8_scheduler,
    git_source_commit,
    is_better_positive_nll,
    seed_phase8,
)
from ml.src.training.vnext_run_io import (
    RunPaths,
    append_epoch_jsonl,
    atomic_write_json,
    build_run_manifest,
    initial_checkpoint_index,
    population_payload,
    read_json,
    reconcile_resume_index,
    relative_artifact,
    relative_checkpoint_identity,
    validate_checkpoint_index,
    validate_run_manifest,
)

M4B_RECOVERY_EPOCHS = 2
M4C_PILOT_EPOCHS = 8
_ALLOWED_GUARDED_HORIZONS = frozenset({M4B_RECOVERY_EPOCHS, M4C_PILOT_EPOCHS})


def _finite(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{name} is not finite: {numeric}")
    return numeric


def _validate_guarded_settings(settings: Phase8Settings) -> None:
    if int(settings.epochs) not in _ALLOWED_GUARDED_HORIZONS:
        raise ValueError(
            "guarded Phase-8 runner permits only governed M4 horizons "
            f"{sorted(_ALLOWED_GUARDED_HORIZONS)}; got {settings.epochs}"
        )
    if int(settings.batch_size) != 8:
        raise ValueError("guarded M4 runner requires batch_size=8")
    if int(settings.gradient_accumulation_steps) != 8:
        raise ValueError("guarded M4 runner requires gradient_accumulation_steps=8")


def _resolve_guarded_output_root(
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
                "guarded resume checkpoint must live under <run>/checkpoints "
                "when --output-dir is omitted"
            )
        return resume_path.parent.parent
    return (
        Path(repo_root).resolve()
        / "ml/logs/r4-phase8-guarded"
        / f"run-{run_binding['binding_digest_sha256'][:12]}"
    )


def _persist_guarded_manifest(
    *,
    paths: RunPaths,
    state: str,
    run_binding: Mapping[str, Any],
    settings: Phase8Settings,
    scheduler_metadata: Mapping[str, Any],
    train_population: Mapping[str, Any],
    selection_population: Mapping[str, Any],
    started_from: str,
    completed_epoch: int,
    global_optimizer_step: int,
    best_positive_nll: float | None,
    best_positive_nll_epoch: int | None,
    error: Mapping[str, Any] | None = None,
) -> None:
    payload = build_run_manifest(
        state=state,
        run_binding=run_binding,
        settings=settings,
        scheduler_metadata=scheduler_metadata,
        output_root=paths.root,
        train_population=train_population,
        selection_population=selection_population,
        started_from=started_from,
        completed_epoch=completed_epoch,
        global_optimizer_step=global_optimizer_step,
        best_positive_nll=best_positive_nll,
        best_positive_nll_epoch=best_positive_nll_epoch,
        checkpoint_index_path=paths.checkpoint_index,
        error=error,
    )
    payload["execution_scope"] = "bounded_pilot_only"
    payload["completion_policy"] = {
        "bounded_final_checkpoint": "final",
        "fixed_horizon_epochs": int(settings.epochs),
        "full_training_authorized": False,
        "g8_passed": False,
        "acceptance_access": False,
    }
    atomic_write_json(paths.manifest, payload)


def run_guarded_phase8_bounded(
    *,
    overlay_dir: Path,
    representations_root: Path,
    logical_acceptance_path: Path,
    physical_acceptance_path: Path,
    objective_decision_path: Path,
    settings: Phase8Settings,
    output_dir: Path | None = None,
    resume: Path | None = None,
    num_workers: int = 0,
    milestone_interval_epochs: int = 1,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    """Run/resume a governed M4 bounded guarded training horizon.

    stop_after_epoch is an invocation-level controlled-pause mechanism for
    M4-B recovery proof. It does not alter the immutable run binding or horizon.
    """

    _validate_guarded_settings(settings)
    if not torch.cuda.is_available():
        raise RuntimeError("guarded M4 execution requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("guarded M4 execution requires CUDA BF16 support")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if milestone_interval_epochs <= 0:
        raise ValueError("milestone_interval_epochs must be > 0")
    if stop_after_epoch is not None:
        stop_after_epoch = int(stop_after_epoch)
        if stop_after_epoch < 1 or stop_after_epoch >= int(settings.epochs):
            raise ValueError(
                "controlled stop must be within the run and before final epoch"
            )

    repo_root = Path(__file__).resolve().parents[3]
    overlay_dir = Path(overlay_dir).expanduser().resolve()
    representations_root = Path(representations_root).expanduser().resolve()
    logical_acceptance_path = Path(logical_acceptance_path).expanduser().resolve()
    physical_acceptance_path = Path(physical_acceptance_path).expanduser().resolve()
    objective_decision_path = Path(objective_decision_path).expanduser().resolve()
    resume_path = None if resume is None else Path(resume).expanduser().resolve()

    if resume_path is not None and resume_path.name != "latest.pt":
        raise ValueError(
            "guarded same-run resume requires <run>/checkpoints/latest.pt"
        )

    seed_phase8(settings.seed)
    source_commit = git_source_commit(repo_root)

    train_ds = LogicalV3GuardedTrainingDataset(
        repo_root=repo_root,
        logical_acceptance_path=logical_acceptance_path,
        physical_acceptance_path=physical_acceptance_path,
        overlay_dir=overlay_dir,
        representations_root=representations_root,
        roles=("TRAIN_STRONG", "TRAIN_WEAK"),
    )
    selection_ds = LogicalV3GuardedTrainingDataset(
        repo_root=repo_root,
        logical_acceptance_path=logical_acceptance_path,
        physical_acceptance_path=physical_acceptance_path,
        overlay_dir=overlay_dir,
        representations_root=representations_root,
        roles=("MODEL_SELECTION",),
    )
    validate_guarded_phase8_populations(
        train_ds,
        selection_ds,
        logical_acceptance_path=logical_acceptance_path,
    )
    train_population, selection_population = population_payload(
        train_ds, selection_ds
    )

    sampler = DeterministicGroupSampler(
        train_ds.group_to_indices,
        seed=settings.seed,
    )
    train_loader, selection_loader = build_phase8_loaders(
        train_ds,
        selection_ds,
        settings,
        num_workers,
        sampler,
        vnext_collate_fn,
    )

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_phase8_v10_model(device)
    param_groups, max_lrs = build_parameter_groups(model, settings)
    optimizer = AdamW(param_groups, weight_decay=settings.weight_decay)
    scheduler, scheduler_metadata = build_phase8_scheduler(
        optimizer=optimizer,
        max_lrs=max_lrs,
        settings=settings,
        loader_batches=len(train_loader),
    )
    optimizer_config = guarded_optimizer_binding_config(
        settings=settings,
        parameter_groups=param_groups,
        scheduler_metadata=scheduler_metadata,
        num_workers=num_workers,
        milestone_interval_epochs=milestone_interval_epochs,
    )
    run_binding = build_guarded_run_binding(
        source_commit=source_commit,
        repo_root=repo_root,
        manifest_path=overlay_dir / "manifest.json",
        logical_acceptance_path=logical_acceptance_path,
        physical_acceptance_path=physical_acceptance_path,
        objective_decision_path=objective_decision_path,
        representations_root=representations_root,
        seed=settings.seed,
        weak_positive_weight=settings.weak_positive_weight,
        optimizer_config=optimizer_config,
        train_contracts=len(train_ds),
        train_groups=train_ds.group_count,
        selection_contracts=len(selection_ds),
        selection_groups=selection_ds.group_count,
    )
    if run_binding.get("scope") != "bounded_pilot_only":
        raise RuntimeError("guarded run binding lost bounded-pilot scope")
    if (run_binding.get("limits") or {}).get("full_training_authorized") is not False:
        raise RuntimeError("guarded run binding unexpectedly authorizes full training")

    root = _resolve_guarded_output_root(
        repo_root=repo_root,
        run_binding=run_binding,
        output_dir=output_dir,
        resume_path=resume_path,
    )
    paths = RunPaths.from_root(root)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    last_completed_epoch = 0
    global_optimizer_step = 0
    best_positive_nll: float | None = None
    best_positive_nll_epoch: int | None = None
    started_from = "fresh"

    if resume_path is None:
        occupied = [
            item
            for item in (
                paths.manifest,
                paths.checkpoint_index,
                paths.latest_checkpoint,
            )
            if item.exists()
        ]
        if occupied:
            raise FileExistsError(
                "guarded output already contains durable run state; "
                f"resume latest.pt instead: {occupied}"
            )
        checkpoint_index = initial_checkpoint_index(run_binding)
        atomic_write_json(paths.checkpoint_index, checkpoint_index)
    else:
        if resume_path != paths.latest_checkpoint:
            raise ValueError(
                f"resume checkpoint must be this run's latest.pt: "
                f"{paths.latest_checkpoint}"
            )
        manifest = read_json(paths.manifest)
        validate_run_manifest(manifest, run_binding)
        checkpoint_index = read_json(paths.checkpoint_index)
        validate_checkpoint_index(checkpoint_index, run_binding)

        checkpoint = load_checkpoint(resume_path, map_location=device)
        restored = restore_checkpoint(
            checkpoint,
            expected_run_binding=run_binding,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(restored["next_epoch"])
        last_completed_epoch = int(restored["completed_epoch"])
        global_optimizer_step = int(restored["global_optimizer_step"])
        best_positive_nll = restored["best_positive_nll"]
        best_positive_nll_epoch = restored["best_positive_nll_epoch"]
        started_from = relative_artifact(resume_path, paths.root)

        checkpoint_index = reconcile_resume_index(
            index=checkpoint_index,
            paths=paths,
            checkpoint=checkpoint,
            run_binding=run_binding,
            total_epochs=settings.epochs,
            milestone_interval_epochs=milestone_interval_epochs,
        )
        atomic_write_json(paths.checkpoint_index, checkpoint_index)
        append_epoch_jsonl(paths.epoch_metrics, checkpoint["epoch_event"])
        append_epoch_jsonl(
            paths.selection_records,
            {
                "epoch": int(checkpoint["epoch"]),
                "records": list(checkpoint["selection_records"]),
            },
        )
        del checkpoint

    if start_epoch > settings.epochs:
        if last_completed_epoch != settings.epochs:
            raise RuntimeError("guarded resume lies beyond configured horizon")
        if not paths.final_checkpoint.is_file():
            raise RuntimeError("completed guarded run is missing final.pt")
        return {
            "status": "GUARDED_BOUNDED_ALREADY_COMPLETE",
            "source_commit": source_commit,
            "binding_digest_sha256": run_binding["binding_digest_sha256"],
            "output_root": str(paths.root),
            "epochs_completed": int(last_completed_epoch),
            "global_optimizer_steps": int(global_optimizer_step),
            "bounded_final_checkpoint": str(paths.final_checkpoint),
            "full_training_authorized": False,
            "g8_passed": False,
        }

    _persist_guarded_manifest(
        paths=paths,
        state="RUNNING",
        run_binding=run_binding,
        settings=settings,
        scheduler_metadata=scheduler_metadata,
        train_population=train_population,
        selection_population=selection_population,
        started_from=started_from,
        completed_epoch=last_completed_epoch,
        global_optimizer_step=global_optimizer_step,
        best_positive_nll=best_positive_nll,
        best_positive_nll_epoch=best_positive_nll_epoch,
    )

    try:
        for epoch in range(start_epoch, settings.epochs + 1):
            train_metrics = train_masked_epoch(
                model=model,
                loader=train_loader,
                sampler=sampler,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                settings=settings,
                epoch=epoch,
                use_amp=True,
            )
            selection_metrics, selection_records = evaluate_positive_selection(
                model=model,
                loader=selection_loader,
                device=device,
                settings=settings,
                epoch=epoch,
                use_amp=True,
            )

            epoch_steps = int(train_metrics["optimizer_steps"])
            if epoch_steps != int(scheduler_metadata["steps_per_epoch"]):
                raise RuntimeError(
                    "guarded optimizer-step count drift: "
                    f"{epoch_steps} != {scheduler_metadata['steps_per_epoch']}"
                )
            global_optimizer_step += epoch_steps
            expected_global = epoch * int(scheduler_metadata["steps_per_epoch"])
            if global_optimizer_step != expected_global:
                raise RuntimeError(
                    "guarded global optimizer-step drift: "
                    f"{global_optimizer_step} != {expected_global}"
                )

            positive_nll = _finite(
                selection_metrics["positive_nll"],
                "guarded MODEL_SELECTION positive_nll",
            )
            improved = is_better_positive_nll(positive_nll, best_positive_nll)
            if improved:
                best_positive_nll = positive_nll
                best_positive_nll_epoch = epoch

            lr_by_group = {
                str(group.get("name", f"group_{idx}")): float(group["lr"])
                for idx, group in enumerate(optimizer.param_groups)
            }
            epoch_event = {
                "epoch": int(epoch),
                "global_optimizer_step": int(global_optimizer_step),
                "train": dict(train_metrics),
                "model_selection": dict(selection_metrics),
                "learning_rates": lr_by_group,
                "best_positive_nll": float(best_positive_nll),
                "best_positive_nll_epoch": int(best_positive_nll_epoch),
                "new_best_positive_nll": bool(improved),
                "interpretation": "positive_fit_diagnostic_only",
            }
            base_checkpoint = build_checkpoint_payload(
                kind="latest",
                epoch=epoch,
                global_optimizer_step=global_optimizer_step,
                run_binding=run_binding,
                settings=settings,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_positive_nll=best_positive_nll,
                best_positive_nll_epoch=best_positive_nll_epoch,
                epoch_event=epoch_event,
                selection_records=selection_records,
            )

            if improved:
                payload = dict(base_checkpoint)
                payload["kind"] = "best_positive_nll"
                checkpoint_index["best_positive_nll"] = (
                    relative_checkpoint_identity(
                        atomic_torch_save(payload, paths.best_checkpoint),
                        paths.root,
                    )
                )

            if (
                epoch % milestone_interval_epochs == 0
                and epoch < settings.epochs
            ):
                payload = dict(base_checkpoint)
                payload["kind"] = "milestone"
                identity = relative_checkpoint_identity(
                    atomic_torch_save(
                        payload,
                        paths.milestone_checkpoint(epoch),
                    ),
                    paths.root,
                )
                milestones = {
                    int(item["epoch"]): item
                    for item in checkpoint_index.get("milestones", [])
                }
                milestones[epoch] = identity
                checkpoint_index["milestones"] = [
                    milestones[key] for key in sorted(milestones)
                ]

            if epoch == settings.epochs:
                payload = dict(base_checkpoint)
                payload["kind"] = "final"
                checkpoint_index["final"] = relative_checkpoint_identity(
                    atomic_torch_save(payload, paths.final_checkpoint),
                    paths.root,
                )

            checkpoint_index["latest"] = relative_checkpoint_identity(
                atomic_torch_save(base_checkpoint, paths.latest_checkpoint),
                paths.root,
            )
            atomic_write_json(paths.checkpoint_index, checkpoint_index)

            append_epoch_jsonl(paths.epoch_metrics, epoch_event)
            append_epoch_jsonl(
                paths.selection_records,
                {"epoch": int(epoch), "records": list(selection_records)},
            )
            last_completed_epoch = epoch

            _persist_guarded_manifest(
                paths=paths,
                state="COMPLETE" if epoch == settings.epochs else "RUNNING",
                run_binding=run_binding,
                settings=settings,
                scheduler_metadata=scheduler_metadata,
                train_population=train_population,
                selection_population=selection_population,
                started_from=started_from,
                completed_epoch=last_completed_epoch,
                global_optimizer_step=global_optimizer_step,
                best_positive_nll=best_positive_nll,
                best_positive_nll_epoch=best_positive_nll_epoch,
            )

            if stop_after_epoch is not None and epoch == stop_after_epoch:
                return {
                    "status": "GUARDED_BOUNDED_CONTROLLED_PAUSE",
                    "source_commit": source_commit,
                    "binding_digest_sha256": run_binding[
                        "binding_digest_sha256"
                    ],
                    "output_root": str(paths.root),
                    "epochs_completed": int(last_completed_epoch),
                    "next_epoch": int(epoch) + 1,
                    "global_optimizer_steps": int(global_optimizer_step),
                    "latest_checkpoint": str(paths.latest_checkpoint),
                    "full_training_authorized": False,
                    "g8_passed": False,
                }

    except BaseException as exc:
        state = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        try:
            _persist_guarded_manifest(
                paths=paths,
                state=state,
                run_binding=run_binding,
                settings=settings,
                scheduler_metadata=scheduler_metadata,
                train_population=train_population,
                selection_population=selection_population,
                started_from=started_from,
                completed_epoch=last_completed_epoch,
                global_optimizer_step=global_optimizer_step,
                best_positive_nll=best_positive_nll,
                best_positive_nll_epoch=best_positive_nll_epoch,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        except Exception as manifest_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "guarded failure manifest could not be written: "
                    f"{type(manifest_exc).__name__}: {manifest_exc}"
                )
        raise

    if last_completed_epoch != settings.epochs:
        raise RuntimeError("guarded runner exited before bounded horizon complete")
    if not paths.final_checkpoint.is_file():
        raise RuntimeError("guarded bounded horizon completed without final.pt")

    return {
        "status": "GUARDED_BOUNDED_TRAINING_COMPLETE",
        "source_commit": source_commit,
        "binding_digest_sha256": run_binding["binding_digest_sha256"],
        "output_root": str(paths.root),
        "epochs_completed": int(last_completed_epoch),
        "global_optimizer_steps": int(global_optimizer_step),
        "optimizer_steps_per_epoch": int(
            scheduler_metadata["steps_per_epoch"]
        ),
        "planned_optimizer_steps": int(
            scheduler_metadata["total_optimizer_steps"]
        ),
        "best_positive_nll": float(best_positive_nll),
        "best_positive_nll_epoch": int(best_positive_nll_epoch),
        "best_positive_nll_scope": "positive_fit_diagnostic_only",
        "bounded_final_checkpoint": str(paths.final_checkpoint),
        "checkpoint_index": str(paths.checkpoint_index),
        "cuda_peak_allocated_mb": round(
            torch.cuda.max_memory_allocated() / 1024**2, 2
        ),
        "full_training_authorized": False,
        "g8_passed": False,
    }


__all__ = [
    "M4B_RECOVERY_EPOCHS",
    "M4C_PILOT_EPOCHS",
    "run_guarded_phase8_bounded",
]
