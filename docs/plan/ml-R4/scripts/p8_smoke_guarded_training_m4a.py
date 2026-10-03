#!/usr/bin/env python3
"""Protected-local M4-A guarded CUDA mechanics smoke.

Executes exactly eight train micro-batches (one optimizer step under the frozen
8x accumulation contract) and exactly one MODEL_SELECTION batch on the accepted
R4-D-009/R4-D-011/R4-D-013/R4-D-014 seam.

This command writes only a JSON report. It writes no training checkpoint and
cannot authorize M4-B/M4-C or full training by itself.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

import torch
from torch.optim import AdamW

from ml.src.datasets.vnext_logical_v3_guarded_dataset import (
    LogicalV3GuardedTrainingDataset,
    R4_D011_BINDING_DIGEST,
    R4_D013_BINDING_DIGEST,
    vnext_collate_fn,
)
from ml.src.training.group_sampler import DeterministicGroupSampler
from ml.src.training.vnext_epoch import (
    evaluate_positive_selection,
    train_masked_epoch,
)
from ml.src.training.vnext_guarded_binding import build_guarded_run_binding
from ml.src.training.vnext_guarded_run_control import (
    R4_D014_DECISION_ID,
    R4_D014_OBJECTIVE_ID,
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
    seed_phase8,
)
from ml.src.training.vnext_run_io import population_payload
from sentinel_data.representation.graph_schema_versions import get_graph_schema

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_OVERLAY = DATA_ROOT / "exports/sentinel-r4-vnext-v3"
DEFAULT_REPRESENTATIONS = (
    DATA_ROOT
    / "r4-guarded-d5-full-2026-09-29-a"
    / "representations-r4-v10-v26-guarded-v1-candidate"
)
DEFAULT_LOGICAL_ACCEPTANCE = (
    REPO_ROOT
    / "docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/logical_v3_acceptance.json"
)
DEFAULT_PHYSICAL_ACCEPTANCE = (
    REPO_ROOT
    / "docs/plan/ml-R4/evidence/2026-09-29_guarded_token_physical_acceptance/acceptance.json"
)
DEFAULT_OBJECTIVE_DECISION = (
    REPO_ROOT
    / "docs/plan/ml-R4/evidence/2026-10-01_phase8_objective_evaluation/decision.json"
)

REPORT_SCHEMA = "sentinel-r4-phase8-m4a-guarded-cuda-smoke-v1"
TRAIN_MICRO_BATCHES = 8
SELECTION_BATCHES = 1
SMOKE_EPOCHS = 1


def _write_fresh(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"M4-A smoke output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _finite_metrics(payload: dict[str, Any], *, prefix: str) -> None:
    for key, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            raise RuntimeError(f"{prefix} metric is not finite: {key}={value!r}")


def _cuda_memory() -> dict[str, float]:
    return {
        "max_allocated_mb": float(torch.cuda.max_memory_allocated()) / (1024.0**2),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved()) / (1024.0**2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--representations-root",
        type=Path,
        default=DEFAULT_REPRESENTATIONS,
    )
    parser.add_argument(
        "--logical-acceptance",
        type=Path,
        default=DEFAULT_LOGICAL_ACCEPTANCE,
    )
    parser.add_argument(
        "--physical-acceptance",
        type=Path,
        default=DEFAULT_PHYSICAL_ACCEPTANCE,
    )
    parser.add_argument(
        "--objective-decision",
        type=Path,
        default=DEFAULT_OBJECTIVE_DECISION,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M4-A guarded mechanics smoke requires CUDA")
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("M4-A requires CUDA BF16 support")

    source_commit = git_source_commit(REPO_ROOT)
    overlay = args.overlay.expanduser().resolve()
    representations = args.representations_root.expanduser().resolve()
    logical_acceptance = args.logical_acceptance.expanduser().resolve()
    physical_acceptance = args.physical_acceptance.expanduser().resolve()
    objective_decision = args.objective_decision.expanduser().resolve()

    seed_phase8(20260813)
    train_ds = LogicalV3GuardedTrainingDataset(
        repo_root=REPO_ROOT,
        logical_acceptance_path=logical_acceptance,
        physical_acceptance_path=physical_acceptance,
        overlay_dir=overlay,
        representations_root=representations,
        roles=("TRAIN_STRONG", "TRAIN_WEAK"),
    )
    selection_ds = LogicalV3GuardedTrainingDataset(
        repo_root=REPO_ROOT,
        logical_acceptance_path=logical_acceptance,
        physical_acceptance_path=physical_acceptance,
        overlay_dir=overlay,
        representations_root=representations,
        roles=("MODEL_SELECTION",),
    )
    validate_guarded_phase8_populations(
        train_ds,
        selection_ds,
        logical_acceptance_path=logical_acceptance,
    )
    train_population, selection_population = population_payload(
        train_ds,
        selection_ds,
    )

    settings = Phase8Settings(
        epochs=SMOKE_EPOCHS,
        batch_size=8,
        gradient_accumulation_steps=8,
    )
    sampler = DeterministicGroupSampler(
        train_ds.group_to_indices,
        seed=settings.seed,
    )
    train_loader, selection_loader = build_phase8_loaders(
        train_ds,
        selection_ds,
        settings,
        0,
        sampler,
        vnext_collate_fn,
    )
    if len(train_loader) < TRAIN_MICRO_BATCHES:
        raise RuntimeError("M4-A train loader is smaller than the required smoke window")
    if len(selection_loader) < SELECTION_BATCHES:
        raise RuntimeError("M4-A selection loader is empty")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_phase8_v10_model(device)
    schema = get_graph_schema("v10")
    if model.gnn.graph_schema_version != "v10":
        raise RuntimeError("M4-A V10 model factory returned the wrong graph schema")
    if model.gnn.edge_embedding is None:
        raise RuntimeError("M4-A V10 model disabled edge embeddings")
    if int(model.gnn.edge_embedding.num_embeddings) != int(schema.num_edge_types):
        raise RuntimeError("M4-A V10 edge vocabulary mismatch")

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
        num_workers=0,
        milestone_interval_epochs=1,
    )
    run_binding = build_guarded_run_binding(
        source_commit=source_commit,
        repo_root=REPO_ROOT,
        manifest_path=overlay / "manifest.json",
        logical_acceptance_path=logical_acceptance,
        physical_acceptance_path=physical_acceptance,
        objective_decision_path=objective_decision,
        representations_root=representations,
        seed=settings.seed,
        weak_positive_weight=settings.weak_positive_weight,
        optimizer_config=optimizer_config,
        train_contracts=len(train_ds),
        train_groups=train_ds.group_count,
        selection_contracts=len(selection_ds),
        selection_groups=selection_ds.group_count,
    )

    scheduler_before = int(scheduler.last_epoch)
    started = time.perf_counter()
    train_metrics = train_masked_epoch(
        model=model,
        loader=train_loader,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        settings=settings,
        epoch=1,
        use_amp=True,
        max_batches=TRAIN_MICRO_BATCHES,
    )
    scheduler_after = int(scheduler.last_epoch)
    if int(train_metrics["optimizer_steps"]) != 1:
        raise RuntimeError(
            "M4-A must execute exactly one optimizer step: "
            f"{train_metrics['optimizer_steps']}"
        )
    if scheduler_after != scheduler_before + 1:
        raise RuntimeError(
            "M4-A scheduler did not advance exactly once: "
            f"{scheduler_before} -> {scheduler_after}"
        )
    _finite_metrics(train_metrics, prefix="train")

    selection_metrics, selection_records = evaluate_positive_selection(
        model=model,
        loader=selection_loader,
        device=device,
        settings=settings,
        epoch=1,
        use_amp=True,
        max_batches=SELECTION_BATCHES,
    )
    _finite_metrics(selection_metrics, prefix="selection")
    if int(selection_metrics.get("metric_cells", 0)) <= 0:
        raise RuntimeError("M4-A selection smoke produced zero metric cells")
    elapsed = time.perf_counter() - started

    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS_M4A_GUARDED_CUDA_SMOKE_REVIEW_REQUIRED",
        "source_commit": source_commit,
        "authorities": {
            "logical": "R4-D-009",
            "graph_parent": "R4-D-011",
            "graph_parent_binding_digest_sha256": R4_D011_BINDING_DIGEST,
            "guarded_tokens": "R4-D-013",
            "guarded_binding_digest_sha256": R4_D013_BINDING_DIGEST,
            "objective_evaluation": R4_D014_DECISION_ID,
            "objective_id": R4_D014_OBJECTIVE_ID,
        },
        "run_binding_digest_sha256": run_binding["binding_digest_sha256"],
        "run_binding_scope": run_binding["scope"],
        "train_population": train_population,
        "selection_population": selection_population,
        "smoke_contract": {
            "train_micro_batches": TRAIN_MICRO_BATCHES,
            "gradient_accumulation_steps": settings.gradient_accumulation_steps,
            "optimizer_steps_executed": int(train_metrics["optimizer_steps"]),
            "selection_batches": SELECTION_BATCHES,
            "scheduler_last_epoch_before": scheduler_before,
            "scheduler_last_epoch_after": scheduler_after,
            "durable_checkpoint_written": False,
            "historical_checkpoint_loaded": False,
            "mixed_precision": "bf16_autocast",
        },
        "train_metrics": train_metrics,
        "selection_metrics": selection_metrics,
        "selection_records_emitted": len(selection_records),
        "model": {
            "graph_schema_version": model.gnn.graph_schema_version,
            "edge_types": int(model.gnn.edge_embedding.num_embeddings),
            "expected_edge_types": int(schema.num_edge_types),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "cuda_device": torch.cuda.get_device_name(device),
            **_cuda_memory(),
        },
        "m4b_execution_authorized": False,
        "m4c_execution_authorized": False,
        "full_training_authorized": False,
        "g8_passed": False,
        "review_required": True,
        "decision_boundary": (
            "This smoke proves exactly one guarded optimizer step plus one "
            "positive-only selection batch. It writes no checkpoint and does "
            "not itself authorize M4-B, the 8-epoch M4-C pilot, or full training."
        ),
    }
    _write_fresh(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
