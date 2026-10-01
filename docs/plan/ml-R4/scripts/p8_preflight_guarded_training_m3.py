#!/usr/bin/env python3
"""Protected-local M3 preflight for the accepted guarded Phase-8 training seam.

This command performs no training and no optimizer step. It proves that the
actual logical-V3 publication and R4-D-013 physical root can be composed under
R4-D-014 into one deterministic bounded-pilot run binding and explicit V10
frozen model configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

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
REPORT_SCHEMA = "sentinel-r4-phase8-m3-guarded-preflight-v1"
BOUNDED_PREFLIGHT_EPOCHS = 2


def _write_fresh(path: Path, payload: dict) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"M3 preflight output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--representations-root", type=Path, default=DEFAULT_REPRESENTATIONS)
    parser.add_argument("--logical-acceptance", type=Path, default=DEFAULT_LOGICAL_ACCEPTANCE)
    parser.add_argument("--physical-acceptance", type=Path, default=DEFAULT_PHYSICAL_ACCEPTANCE)
    parser.add_argument("--objective-decision", type=Path, default=DEFAULT_OBJECTIVE_DECISION)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_commit = git_source_commit(REPO_ROOT)
    overlay = args.overlay.expanduser().resolve()
    representations = args.representations_root.expanduser().resolve()
    logical_acceptance = args.logical_acceptance.expanduser().resolve()
    physical_acceptance = args.physical_acceptance.expanduser().resolve()
    objective_decision = args.objective_decision.expanduser().resolve()

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
    train_population, selection_population = population_payload(train_ds, selection_ds)

    # Exercise actual payload validation on both authority roles without
    # iterating or optimizing the datasets.
    train_sample = train_ds[0]
    selection_sample = selection_ds[0]

    settings = Phase8Settings(
        epochs=BOUNDED_PREFLIGHT_EPOCHS,
        batch_size=8,
        gradient_accumulation_steps=8,
    )
    sampler = DeterministicGroupSampler(train_ds.group_to_indices, seed=settings.seed)
    train_loader, _ = build_phase8_loaders(
        train_ds,
        selection_ds,
        settings,
        0,
        sampler,
        vnext_collate_fn,
    )

    device = torch.device("cpu")
    model = build_phase8_v10_model(device)
    schema = get_graph_schema("v10")
    if model.gnn.graph_schema_version != "v10":
        raise RuntimeError("M3 V10 model factory did not construct graph schema v10")
    if model.gnn.edge_embedding is None:
        raise RuntimeError("M3 V10 model unexpectedly disabled edge embeddings")
    if int(model.gnn.edge_embedding.num_embeddings) != int(schema.num_edge_types):
        raise RuntimeError("M3 V10 model edge vocabulary does not match graph schema")

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

    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS_M3_GUARDED_INTEGRATION_REVIEW_REQUIRED",
        "source_commit": source_commit,
        "logical_overlay": str(overlay.relative_to(REPO_ROOT)),
        "physical_root": str(representations.relative_to(REPO_ROOT)),
        "authorities": {
            "logical": "R4-D-009",
            "graph_parent": "R4-D-011",
            "graph_parent_binding_digest_sha256": R4_D011_BINDING_DIGEST,
            "guarded_tokens": "R4-D-013",
            "guarded_binding_digest_sha256": R4_D013_BINDING_DIGEST,
            "objective_evaluation": R4_D014_DECISION_ID,
            "objective_id": R4_D014_OBJECTIVE_ID,
        },
        "train_population": train_population,
        "selection_population": selection_population,
        "payload_probe": {
            "train_contract_id": train_sample[3],
            "train_role": train_sample[4],
            "selection_contract_id": selection_sample[3],
            "selection_role": selection_sample[4],
            "token_shape": list(train_sample[1]["input_ids"].shape),
            "graph_schema_version": getattr(train_sample[0], "graph_schema_version", None),
        },
        "model": {
            "graph_schema_version": model.gnn.graph_schema_version,
            "edge_types": int(model.gnn.edge_embedding.num_embeddings),
            "expected_edge_types": int(schema.num_edge_types),
        },
        "bounded_contract": {
            "epochs": settings.epochs,
            "batch_size": settings.batch_size,
            "gradient_accumulation_steps": settings.gradient_accumulation_steps,
            "loader_batches_per_epoch": len(train_loader),
            "optimizer_steps_per_epoch": scheduler_metadata["steps_per_epoch"],
            "planned_optimizer_steps": scheduler_metadata["total_optimizer_steps"],
            "optimizer_steps_executed": 0,
        },
        "run_binding_digest_sha256": run_binding["binding_digest_sha256"],
        "run_binding_scope": run_binding["scope"],
        "full_training_authorized": False,
        "g8_passed": False,
        "m4_execution_authorized": False,
        "review_required": True,
        "decision_boundary": (
            "This preflight proves M3 integration only. It performs no optimizer "
            "step and does not itself authorize the M4 pilot or full training."
        ),
    }
    _write_fresh(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))

    # Explicitly drop the heavy stack after the no-step proof.
    del scheduler, optimizer, model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
