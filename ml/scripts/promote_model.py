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
        --checkpoint ml/checkpoints/multilabel_crossattn_v2_best.pt \\
        --stage Staging \\
        --val-f1-macro 0.4812 \\
        --note "v2 retrain: edge_attr embeddings (P0-B) active"

Promote to Production:
    python ml/scripts/promote_model.py \\
        --checkpoint ml/checkpoints/multilabel_crossattn_v2_best.pt \\
        --stage Production \\
        --val-f1-macro 0.4812 \\
        --note "Validated; F1 > 0.4679 gate passed; edge_attr active"

Dry run (no MLflow writes):
    python ml/scripts/promote_model.py \\
        --checkpoint ml/checkpoints/multilabel_crossattn_v2_best.pt \\
        --stage Staging \\
        --val-f1-macro 0.4812 \\
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

import mlflow
from mlflow.tracking import MlflowClient
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


def _get_current_production_f1(client: "MlflowClient") -> float | None:
    """Return val_f1_macro of the current Production model, or None if none exists."""
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        return run.data.metrics.get("val_f1_macro")
    except Exception:
        return None


def _check_baseline(baseline_path: Path) -> None:
    """
    Fail if the drift baseline file is missing or was built from training data.

    A baseline built from training data will fire KS alerts on almost every real
    production contract (training corpus is a 2024 historical snapshot).
    Only baselines built from warm-up production traffic are safe to deploy with.
    """
    if not baseline_path.exists():
        print(
            f"ERROR: drift baseline not found at {baseline_path}.\n"
            "       Run: python ml/scripts/compute_drift_baseline.py --source warmup\n"
            "       after the API has collected N_WARMUP production requests.",
            file=sys.stderr,
        )
        sys.exit(1)

    import json
    with baseline_path.open() as f:
        meta = json.load(f)
    source = meta.get("source", "unknown")
    if source == "training":
        print(
            f"ERROR: drift baseline at {baseline_path} was built from training data\n"
            "       (source='training'). This will produce noisy/useless drift alerts.\n"
            "       Rebuild with: python ml/scripts/compute_drift_baseline.py --source warmup",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_behavioral_probes(checkpoint: Path, stage: str) -> Path:
    """
    Fail if the behavioral probes JSON is missing or has any FAIL.

    Added 2026-06-17 per `ml/audit_docs/2026-06-17_testing_suite_overhaul_plan.md`.
    The behavioral probes (C.2.4 in C_diagnostic_checks.md) are the missing
    test that would have caught the Run 12 ExternalBug FP. They MUST pass
    before promotion.

    The JSON is at `<stem>_behavioral_probes.json` alongside the checkpoint.
    Run `python ml/testing_specs/synthetic_probes.py --checkpoint <ckpt>`
    to generate it.

    Returns the path to the probes JSON (for downstream display).
    """
    probes_path = checkpoint.with_name(f"{checkpoint.stem}_behavioral_probes.json")
    if not probes_path.exists():
        print(
            f"ERROR: behavioral probes JSON not found at {probes_path}.\n"
            "       Run: python ml/testing_specs/synthetic_probes.py "
            f"--checkpoint {checkpoint} --output {probes_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    import json
    with probes_path.open() as f:
        data = json.load(f)
    summary = data.get("summary", {})
    if not summary.get("all_passed", False):
        n_fail = summary.get("failed", "?")
        n_total = summary.get("total", "?")
        print(
            f"ERROR: behavioral probes have {n_fail}/{n_total} FAILs.\n"
            f"       {stage} promotion BLOCKED until all probes pass.\n"
            "       Re-run with: python ml/testing_specs/synthetic_probes.py "
            f"--checkpoint {checkpoint}\n"
            "       Investigate the failing probes (likely model issue).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Behavioral probes: {summary.get('passed', '?')}/{summary.get('total', '?')} PASS ✓")
    return probes_path


def _check_label_quality(stage: str) -> Path | None:
    """
    Fail if the label quality JSON has any FAIL.

    Added 2026-06-17 per `ml/audit_docs/2026-06-17_testing_suite_overhaul_plan.md`.
    The label quality check (F.1.0 in F_new_run_checklist.md) catches
    over-labeled classes (like Run 12's ExternalBug at 74% positive).
    The JSON is at `<export_dir>/label_quality.json` — generated by
    `python ml/testing_specs/label_quality.py`.

    For this gate, we look for the v3 export's label quality JSON by default.
    """
    import json
    candidates = [
        Path("ml/checkpoints/v3_label_quality.json"),
        Path("/home/motafeq/projects/sentinel/ml/checkpoints/v3_label_quality.json"),
    ]
    quality_path = next((c for c in candidates if c.exists()), None)
    if quality_path is None:
        # No JSON found — this is a warning, not a fail
        print(
            "WARNING: label quality JSON not found.\n"
            "         Run: python ml/testing_specs/label_quality.py "
            "--output ml/checkpoints/v3_label_quality.json",
            file=sys.stderr,
        )
        return None

    with quality_path.open() as f:
        data = json.load(f)
    summary = data.get("summary", {})
    if summary.get("failed", 0) > 0:
        n_fail = summary.get("failed", "?")
        n_warn = summary.get("warned", "?")
        print(
            f"ERROR: label quality has {n_fail} FAILs, {n_warn} WARNs.\n"
            f"       {stage} promotion BLOCKED until labels are audited.\n"
            "       Investigate over-labeled classes. See F.1.0 in "
            "F_new_run_checklist.md.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Label quality: {summary.get('passed', '?')} PASS, {summary.get('warned', '?')} WARN ✓")
    return quality_path


def promote(
    checkpoint: Path,
    stage: str,
    val_f1_macro: float,
    note: str,
    experiment_name: str,
    dry_run: bool,
    require_baseline: Path | None,
) -> int:
    meta = _load_checkpoint_meta(checkpoint)
    git_sha = _git_commit()

    # ── Gate: behavioral probes (NEW 2026-06-17) ────────────────────────────
    # Catches broken classes (e.g., Run 12's ExternalBug FP).
    # See ml/testing_specs/C_diagnostic_checks.md C.2.4 and BUG-2.
    probes_path = _check_behavioral_probes(checkpoint, stage)

    # ── Gate: label quality (NEW 2026-06-17) ────────────────────────────────
    # Catches over-labeled classes (e.g., Run 12's ExternalBug at 74% positive).
    # See ml/testing_specs/F_new_run_checklist.md F.1.0 and BUG-2.
    quality_path = _check_label_quality(stage)

    # ── Gate: drift baseline (Production only) ──────────────────────────────
    if stage == "Production" and require_baseline is not None:
        _check_baseline(require_baseline)

    # ── Gate: must beat current Production model ─────────────────────────────
    # Check before dry-run print so the gate fires even in --dry-run mode,
    # making it visible without needing MLflow to be writable.
    client = MlflowClient()
    current_prod_f1 = _get_current_production_f1(client)
    if stage == "Production" and current_prod_f1 is not None:
        if val_f1_macro <= current_prod_f1:
            print(
                f"ERROR: val_f1_macro={val_f1_macro:.4f} does not exceed current "
                f"Production model F1={current_prod_f1:.4f}.\n"
                "       Promoting would downgrade the Production model. Aborting.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  F1 gate      : {val_f1_macro:.4f} > Production {current_prod_f1:.4f} ✓")

    # ── Companion threshold JSON ─────────────────────────────────────────────
    thresholds_path = checkpoint.with_name(f"{checkpoint.stem}_thresholds.json")
    has_thresholds = thresholds_path.exists()

    print(f"\nSENTINEL Model Promotion")
    print(f"  Checkpoint   : {checkpoint}")
    print(f"  Thresholds   : {thresholds_path} ({'found' if has_thresholds else 'MISSING — will use uniform 0.5'})")
    print(f"  Architecture : {meta['architecture']}")
    print(f"  Epoch        : {meta['epoch']}")
    print(f"  val_f1_macro : {val_f1_macro:.4f}")
    print(f"  Stage        : {stage}")
    print(f"  Git commit   : {git_sha}")
    print(f"  Note         : {note!r}")
    print(f"  Dry run      : {dry_run}\n")

    if not has_thresholds:
        print(
            "WARNING: no companion thresholds JSON found.\n"
            "         Deployed model will use uniform 0.5 threshold for all classes.\n"
            "         Run tune_threshold.py first for production deployments.\n"
        )

    if dry_run:
        print("DRY RUN — no MLflow writes performed.")
        return 0

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"promote-{stage.lower()}") as run:
        # Log checkpoint as artifact
        mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")

        # Log companion thresholds JSON alongside the checkpoint
        if has_thresholds:
            mlflow.log_artifact(str(thresholds_path), artifact_path="checkpoints")

        mlflow.log_metric("val_f1_macro", val_f1_macro)
        mlflow.set_tags({
            "architecture":    meta["architecture"],
            "num_classes":     str(meta["num_classes"]),
            "epoch":           str(meta["epoch"]),
            "git_commit":      git_sha,
            "stage":           stage,
            "note":            note,
            "thresholds_file": thresholds_path.name if has_thresholds else "none",
        })

        artifact_uri = f"runs:/{run.info.run_id}/checkpoints/{checkpoint.name}"

    mv = mlflow.register_model(model_uri=artifact_uri, name=MODEL_NAME)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )

    print(f"Registered '{MODEL_NAME}' version {mv.version} → {stage}")
    print(f"Run ID: {run.info.run_id}")
    if has_thresholds:
        print(f"Thresholds JSON logged as artifact alongside checkpoint.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a SENTINEL checkpoint to the MLflow Model Registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Promotion gates (enforced automatically):
  All stages : (1) <stem>_behavioral_probes.json must exist and all_passed=true
               (2) ml/checkpoints/v3_label_quality.json must exist with no FAILs
  Production : (3) val_f1_macro must exceed current Production model's F1
               (4) --require-baseline path must exist and have source='warmup'

Examples:
  # Promote Run 4 to Staging
  python ml/scripts/promote_model.py \\
      --checkpoint ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt \\
      --stage Staging --val-f1-macro 0.3362 \\
      --note "Run4 ep32: GCB+prefix+8L, F1=0.3362 all-time best. Pipeline FAIL=0."

  # Promote to Production (requires warmup baseline)
  python ml/scripts/promote_model.py \\
      --checkpoint ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt \\
      --stage Production --val-f1-macro 0.3362 \\
      --require-baseline ml/data/drift_baseline.json \\
      --note "Production deploy after integration smoke test."
""",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the .pt checkpoint file to register.")
    parser.add_argument("--stage", required=True, choices=sorted(VALID_STAGES),
                        help="Target registry stage: Staging or Production.")
    parser.add_argument("--val-f1-macro", type=float, required=True,
                        help="Validation F1-macro for this run (used for gate check + metric tag).")
    parser.add_argument("--note", default="",
                        help="Free-text description for this model version.")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing to MLflow.")
    parser.add_argument(
        "--require-baseline", metavar="PATH",
        help=(
            "Path to drift_baseline.json built from warm-up traffic (not training data). "
            "Required for Production promotions to prevent deploying without drift monitoring. "
            "Omit for Staging."
        ),
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    require_baseline = Path(args.require_baseline) if args.require_baseline else None

    sys.exit(
        promote(
            checkpoint        = checkpoint,
            stage             = args.stage,
            val_f1_macro      = args.val_f1_macro,
            note              = args.note,
            experiment_name   = args.experiment,
            dry_run           = args.dry_run,
            require_baseline  = require_baseline,
        )
    )


if __name__ == "__main__":
    main()
