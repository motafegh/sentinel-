#!/usr/bin/env python3
"""
promote_model.py — Promote a SENTINEL checkpoint to the MLflow Model Registry.

WHY THIS EXISTS
───────────────
Manual model promotion (copying .pt files) leaves no audit trail and no
staged-rollout mechanism. This script logs the checkpoint as an MLflow
artifact, registers it in the Model Registry, and transitions it to the
requested stage in one atomic operation.

MLflow tracking URI defaults to the project-local SQLite DB:
    sqlite:///mlruns.db
Override via MLFLOW_TRACKING_URI env var.

Usage examples
──────────────
Promote to Staging:
    python ml/scripts/promote_model.py \\
        --checkpoint ml/checkpoints/multilabel_crossattn_best.pt \\
        --stage Staging \\
        --val-f1-macro 0.4679 \\
        --note "P0-B retrain with edge_attr embeddings"

Promote to Production:
    python ml/scripts/promote_model.py \\
        --checkpoint ml/checkpoints/multilabel_crossattn_best.pt \\
        --stage Production \\
        --val-f1-macro 0.4812 \\
        --note "Validated; F1 > 0.4679 gate passed"

Dry run (no MLflow writes):
    python ml/scripts/promote_model.py \\
        --checkpoint ml/checkpoints/multilabel_crossattn_best.pt \\
        --stage Staging \\
        --val-f1-macro 0.4679 \\
        --dry-run

Exit codes:
    0  success (or dry-run complete)
    1  checkpoint not found, unknown stage, MLflow error
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

VALID_STAGES = {"Staging", "Production"}
MODEL_NAME   = "sentinel-vulnerability-detector"
DEFAULT_EXPERIMENT = "sentinel-retrain-v2"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_checkpoint_meta(checkpoint: Path) -> dict:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        cfg = raw.get("config", {})
        return {
            "architecture": cfg.get("architecture", "unknown"),
            "num_classes":  cfg.get("num_classes", "unknown"),
            "epoch":        raw.get("epoch", "unknown"),
        }
    return {"architecture": "unknown", "num_classes": "unknown", "epoch": "unknown"}


def promote(
    checkpoint: Path,
    stage: str,
    val_f1_macro: float,
    note: str,
    experiment_name: str,
    dry_run: bool,
) -> int:
    import mlflow
    from mlflow.tracking import MlflowClient

    meta = _load_checkpoint_meta(checkpoint)
    git_sha = _git_commit()

    print(f"\nSENTINEL Model Promotion")
    print(f"  Checkpoint   : {checkpoint}")
    print(f"  Architecture : {meta['architecture']}")
    print(f"  Epoch        : {meta['epoch']}")
    print(f"  val_f1_macro : {val_f1_macro:.4f}")
    print(f"  Stage        : {stage}")
    print(f"  Git commit   : {git_sha}")
    print(f"  Note         : {note!r}")
    print(f"  Dry run      : {dry_run}\n")

    if dry_run:
        print("DRY RUN — no MLflow writes performed.")
        return 0

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"promote-{stage.lower()}") as run:
        # Log the checkpoint as an artifact
        mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")

        # Log promotion metrics and tags
        mlflow.log_metric("val_f1_macro", val_f1_macro)
        mlflow.set_tags({
            "architecture": meta["architecture"],
            "num_classes":  str(meta["num_classes"]),
            "epoch":        str(meta["epoch"]),
            "git_commit":   git_sha,
            "stage":        stage,
            "note":         note,
        })

        artifact_uri = f"runs:/{run.info.run_id}/checkpoints/{checkpoint.name}"

    # Register and transition
    client = MlflowClient()
    mv = mlflow.register_model(model_uri=artifact_uri, name=MODEL_NAME)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )

    print(f"Registered '{MODEL_NAME}' version {mv.version} → {stage}")
    print(f"Run ID: {run.info.run_id}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a SENTINEL checkpoint to the MLflow Model Registry."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pt checkpoint file to register.",
    )
    parser.add_argument(
        "--stage", required=True, choices=sorted(VALID_STAGES),
        help="Target registry stage: Staging or Production.",
    )
    parser.add_argument(
        "--val-f1-macro", type=float, required=True,
        help="Validation F1-macro score to record as a metric tag.",
    )
    parser.add_argument(
        "--note", default="",
        help="Free-text description for this model version.",
    )
    parser.add_argument(
        "--experiment", default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing to MLflow.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    if args.stage not in VALID_STAGES:
        print(f"ERROR: unknown stage '{args.stage}'. Valid: {sorted(VALID_STAGES)}", file=sys.stderr)
        sys.exit(1)

    sys.exit(
        promote(
            checkpoint    = checkpoint,
            stage         = args.stage,
            val_f1_macro  = args.val_f1_macro,
            note          = args.note,
            experiment_name = args.experiment,
            dry_run       = args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
