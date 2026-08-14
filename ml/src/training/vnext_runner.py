"""Canonical durable full-run orchestrator for R4 Phase 8."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.optim import AdamW

from ml.src.training.group_sampler import DeterministicGroupSampler
from ml.src.training.vnext_binding import build_run_binding
from ml.src.training.vnext_checkpoint import (
    assert_checkpoint_binding,
    atomic_torch_save,
    build_checkpoint_payload,
    checkpoint_artifact_identity,
    load_checkpoint,
    restore_checkpoint,
)
from ml.src.training.vnext_epoch import (
    evaluate_positive_selection,
    train_masked_epoch,
)
from ml.src.training.vnext_phase8_config import Phase8Settings
from ml.src.training.vnext_run_control import (
    DEFAULT_MILESTONE_INTERVAL_EPOCHS,
    build_phase8_loaders,
    build_phase8_scheduler,
    git_source_commit,
    is_better_positive_nll,
    optimizer_binding_config,
    resolve_output_root,
    seed_phase8,
    validate_phase8_populations,
)
from ml.src.training.vnext_run_io import (
    RunPaths,
    append_epoch_jsonl,
    atomic_write_json,
    build_run_manifest,
    initial_checkpoint_index,
    read_json,
    population_payload,
    reconcile_resume_index,
    relative_artifact,
    relative_checkpoint_identity,
    validate_checkpoint_index,
    validate_run_manifest,
)


def _finite(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{name} is not finite: {numeric}")
    return numeric


def _persist_manifest(
    *, paths: RunPaths, state: str, run_binding: Mapping[str, Any],
    settings: Phase8Settings, scheduler_metadata: Mapping[str, Any],
    train_population: Mapping[str, Any], selection_population: Mapping[str, Any],
    started_from: str, completed_epoch: int, global_optimizer_step: int,
    best_positive_nll: float | None, best_positive_nll_epoch: int | None,
    error: Mapping[str, Any] | None = None,
) -> None:
    atomic_write_json(
        paths.manifest,
        build_run_manifest(
            state=state, run_binding=run_binding, settings=settings,
            scheduler_metadata=scheduler_metadata, output_root=paths.root,
            train_population=train_population, selection_population=selection_population,
            started_from=started_from, completed_epoch=completed_epoch,
            global_optimizer_step=global_optimizer_step,
            best_positive_nll=best_positive_nll,
            best_positive_nll_epoch=best_positive_nll_epoch,
            checkpoint_index_path=paths.checkpoint_index, error=error,
        ),
    )


def run_phase8_training(
    *,
    overlay_dir: Path,
    representations_root: Path,
    output_dir: Path | None = None,
    resume: Path | None = None,
    num_workers: int = 4,
    milestone_interval_epochs: int = DEFAULT_MILESTONE_INTERVAL_EPOCHS,
) -> dict[str, Any]:
    """Run or resume the canonical fixed-horizon R4 Phase-8 retrain."""
    if not torch.cuda.is_available():
        raise RuntimeError("R4 Phase-8 full training requires CUDA")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if milestone_interval_epochs <= 0:
        raise ValueError("milestone_interval_epochs must be > 0")

    repo_root = Path(__file__).resolve().parents[3]
    overlay_dir = Path(overlay_dir).expanduser().resolve()
    representations_root = Path(representations_root).expanduser().resolve()
    resume_path = None if resume is None else Path(resume).expanduser().resolve()
    if resume_path is not None and resume_path.name != "latest.pt":
        raise ValueError(
            "canonical same-run resume requires <run>/checkpoints/latest.pt; "
            "older checkpoints require a separate future fork workflow"
        )

    settings = Phase8Settings()
    seed_phase8(settings.seed)
    source_commit = git_source_commit(repo_root)

    # Heavy DATA/model imports stay out of module import so CPU control tests remain small.
    from ml.src.datasets.vnext_dataset import (
        CANONICAL_G7_BINDING_DIGEST,
        VNextTrainingDataset,
        vnext_collate_fn,
    )
    from ml.src.training.vnext_model_factory import build_phase8_model
    from ml.src.training.vnext_param_groups import build_parameter_groups

    train_ds = VNextTrainingDataset(
        overlay_dir=overlay_dir,
        representations_root=representations_root,
        roles=("TRAIN_STRONG", "TRAIN_WEAK"),
    )
    selection_ds = VNextTrainingDataset(
        overlay_dir=overlay_dir,
        representations_root=representations_root,
        roles=("MODEL_SELECTION",),
    )
    validate_phase8_populations(train_ds, selection_ds)
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

    model = build_phase8_model(device)
    param_groups, max_lrs = build_parameter_groups(model, settings)
    optimizer = AdamW(param_groups, weight_decay=settings.weight_decay)
    scheduler, scheduler_metadata = build_phase8_scheduler(
        optimizer=optimizer,
        max_lrs=max_lrs,
        settings=settings,
        loader_batches=len(train_loader),
    )
    optimizer_config = optimizer_binding_config(
        settings=settings,
        parameter_groups=param_groups,
        scheduler_metadata=scheduler_metadata,
        num_workers=num_workers,
        milestone_interval_epochs=milestone_interval_epochs,
    )
    run_binding = build_run_binding(
        source_commit=source_commit,
        manifest_path=overlay_dir / "manifest.json",
        expected_representation_digest=CANONICAL_G7_BINDING_DIGEST,
        seed=settings.seed,
        weak_positive_weight=settings.weak_positive_weight,
        optimizer_config=optimizer_config,
        train_contracts=len(train_ds),
        train_groups=train_ds.group_count,
        selection_contracts=len(selection_ds),
        selection_groups=selection_ds.group_count,
    )

    root = resolve_output_root(
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
            p for p in (paths.manifest, paths.checkpoint_index, paths.latest_checkpoint)
            if p.exists()
        ]
        if occupied:
            raise FileExistsError(
                "Phase-8 output already contains durable run state; use --resume "
                f"with latest.pt instead of overwriting: {occupied}"
            )
        checkpoint_index = initial_checkpoint_index(run_binding)
        atomic_write_json(paths.checkpoint_index, checkpoint_index)
    else:
        if resume_path != paths.latest_checkpoint:
            raise ValueError(
                f"resume checkpoint must be this run's latest.pt: {paths.latest_checkpoint}"
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
            raise RuntimeError(
                "resume checkpoint lies beyond the configured Phase-8 horizon"
            )
        if not paths.final_checkpoint.is_file():
            raise RuntimeError("completed Phase-8 run is missing final.pt")
        _persist_manifest(
            paths=paths, state="COMPLETE", run_binding=run_binding,
            settings=settings, scheduler_metadata=scheduler_metadata,
            train_population=train_population,
            selection_population=selection_population,
            started_from=started_from, completed_epoch=last_completed_epoch,
            global_optimizer_step=global_optimizer_step,
            best_positive_nll=best_positive_nll,
            best_positive_nll_epoch=best_positive_nll_epoch,
        )
        return {
            "status": "PHASE8_TRAINING_ALREADY_COMPLETE",
            "source_commit": source_commit,
            "binding_digest_sha256": run_binding["binding_digest_sha256"],
            "output_root": str(paths.root),
            "epochs_completed": int(last_completed_epoch),
            "global_optimizer_steps": int(global_optimizer_step),
            "primary_g8_checkpoint": str(paths.final_checkpoint),
        }

    _persist_manifest(
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
                    "Phase-8 optimizer-step count drift: "
                    f"{epoch_steps} != {scheduler_metadata['steps_per_epoch']}"
                )
            global_optimizer_step += epoch_steps
            expected_global = epoch * int(scheduler_metadata["steps_per_epoch"])
            if global_optimizer_step != expected_global:
                raise RuntimeError(
                    "Phase-8 global optimizer-step drift: "
                    f"{global_optimizer_step} != {expected_global}"
                )

            positive_nll = _finite(
                selection_metrics["positive_nll"],
                "MODEL_SELECTION positive_nll",
            )
            improved = is_better_positive_nll(
                positive_nll, best_positive_nll
            )
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

            # Companion files are written before latest.pt. latest.pt is the
            # transaction marker for the epoch; resume reconciliation can repair
            # metadata/logs if a crash occurs after it is promoted.
            if improved:
                payload = dict(base_checkpoint)
                payload["kind"] = "best_positive_nll"
                checkpoint_index["best_positive_nll"] = relative_checkpoint_identity(
                    atomic_torch_save(payload, paths.best_checkpoint), paths.root
                )

            if (
                epoch % milestone_interval_epochs == 0
                and epoch < settings.epochs
            ):
                payload = dict(base_checkpoint)
                payload["kind"] = "milestone"
                identity = relative_checkpoint_identity(
                    atomic_torch_save(
                        payload, paths.milestone_checkpoint(epoch)
                    ),
                    paths.root,
                )
                milestones = {
                    int(item["epoch"]): item
                    for item in checkpoint_index.get("milestones", [])
                }
                milestones[epoch] = identity
                checkpoint_index["milestones"] = [
                    milestones[k] for k in sorted(milestones)
                ]

            if epoch == settings.epochs:
                payload = dict(base_checkpoint)
                payload["kind"] = "final"
                checkpoint_index["final"] = relative_checkpoint_identity(
                    atomic_torch_save(payload, paths.final_checkpoint), paths.root
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

            _persist_manifest(
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
    except BaseException as exc:
        state = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        try:
            _persist_manifest(
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
                    "Phase-8 failure manifest could not be written: "
                    f"{type(manifest_exc).__name__}: {manifest_exc}"
                )
        raise

    if last_completed_epoch != settings.epochs:
        raise RuntimeError(
            "Phase-8 runner exited without completing the fixed horizon"
        )
    if not paths.final_checkpoint.is_file():
        raise RuntimeError("Phase-8 fixed horizon completed without final.pt")

    return {
        "status": "PHASE8_TRAINING_COMPLETE",
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
        "best_positive_nll_scope": "positive_only_limited_diagnostic",
        "primary_g8_checkpoint": str(paths.final_checkpoint),
        "checkpoint_index": str(paths.checkpoint_index),
        "cuda_peak_allocated_mb": round(
            torch.cuda.max_memory_allocated() / 1024**2, 2
        ),
    }


__all__ = ["run_phase8_training"]
