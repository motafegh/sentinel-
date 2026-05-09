#!/usr/bin/env python3
"""
auto_experiment.py — SENTINEL v4-sprint autoresearch single-run wrapper.

Each invocation runs one experiment (smoke or confirm), tunes thresholds, and
emits a machine-readable score line that the autoresearch agent can grep.

Smoke  (--regime smoke):   1 epoch · 10 % stratified subsample · ~3–5 min on RTX 3070 8 GB
Confirm (--regime confirm): 5 epochs · full train split · ~30–60 min on RTX 3070 8 GB

stdout contract (last lines — agent greps these):
    SENTINEL_SCORE=<tuned_f1_macro float>
    PEAK_VRAM_MB=<int>
    REGIME=<smoke|confirm>

Exit codes:
    0  clean run; score emitted
    1  pre-flight failed; no training started
    2  OOM or runtime error during training
    3  training succeeded but threshold tuning failed (score emitted as 0.0)

Usage:
    poetry run python ml/scripts/auto_experiment.py \\
        --regime smoke \\
        --run-name auto-001 \\
        --experiment-name sentinel-retrain-v4 \\
        --loss-fn focal --gamma 2.0 --alpha 0.25 \\
        --lora-r 8 --lora-alpha 16 \\
        --batch-size 16 --lr 3e-4

    # Arbitrary TrainConfig field overrides (forwarded via parse_known_args):
    poetry run python ml/scripts/auto_experiment.py \\
        --regime confirm --run-name auto-002 \\
        --early-stop-patience 5 --warmup-pct 0.10

Phase-B playground mode (post-v4 — relaxes architecture lock):
    poetry run python ml/scripts/auto_experiment.py \\
        --regime smoke --mode playground --run-name playground-001 ...
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

# Make repo root importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import mlflow
import numpy as np
import torch
from loguru import logger

from ml.scripts.compute_locked_hashes import DEFAULT_SIDECAR, check_mode as _check_hashes
from ml.scripts.tune_threshold import (
    apply_thresholds,
    build_threshold_grid,
    build_val_loader,
    collect_probabilities,
    evaluate_overall,
    load_model_from_checkpoint,
    save_results,
    sweep_one_class,
)
from ml.src.training.trainer import CLASS_NAMES, TrainConfig, train

logger.remove()
logger.add(sys.stderr, level="INFO")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# v3 tuned F1 gate and per-class floors (from SENTINEL-EVAL-BACKLOG.md).
GATE_F1 = 0.5069
V3_FLOORS: dict[str, float] = {
    "CallToUnknown":              0.3436,
    "DenialOfService":            0.3500,
    "ExternalBug":                0.3845,
    "GasException":               0.5001,
    "IntegerUO":                  0.7714,
    "MishandledException":        0.4416,
    "Reentrancy":                 0.4862,
    "Timestamp":                  0.4289,
    "TransactionOrderDependence": 0.4270,
    "UnusedReturn":               0.4360,
}

# Smoke: train on 10 % subsample, 1 epoch.
SMOKE_SUBSAMPLE = 0.10
SMOKE_EPOCHS    = 1

# Confirm: full training set, 5 epochs (overrideable with --max-epochs).
CONFIRM_EPOCHS  = 5

# Minimum smoke F1 that warrants escalating to a confirm run (heuristic).
SMOKE_PROMOTE_THRESHOLD = 0.42

# Minimum GPU free memory required before starting a run.
MIN_VRAM_GB = 5.5  # display always holds ~1.6 GB on RTX 3070 Laptop

# Autoresearch experiment name — the only valid value in v4-sprint mode.
AUTORESEARCH_EXPERIMENT = "sentinel-retrain-v4"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[argparse.Namespace, dict]:
    """Parse known CLI flags; forward unknown --key value pairs into TrainConfig."""
    p = argparse.ArgumentParser(
        description="SENTINEL autoresearch single-run wrapper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Any extra --flag value pairs whose names match a TrainConfig field "
            "(underscores, not hyphens) are forwarded directly into TrainConfig."
        ),
    )

    # Regime
    p.add_argument(
        "--regime",
        choices=["smoke", "confirm"],
        default="smoke",
        help="smoke=1 epoch 10%% subsample; confirm=full data 5 epochs.",
    )
    p.add_argument(
        "--mode",
        choices=["v4-sprint", "playground"],
        default="v4-sprint",
        help="v4-sprint enforces all locked-file hashes; playground only locks val_indices.npy.",
    )

    # Identity
    p.add_argument("--run-name",        required=True, help="MLflow run name.")
    p.add_argument(
        "--experiment-name",
        default=AUTORESEARCH_EXPERIMENT,
        help="MLflow experiment (must be sentinel-retrain-v4 in v4-sprint mode).",
    )

    # Searchable hyperparameters
    p.add_argument("--loss-fn",  default="focal",  choices=["bce", "focal"])
    p.add_argument("--gamma",    type=float, default=2.0,  help="focal_gamma")
    p.add_argument("--alpha",    type=float, default=0.25, help="focal_alpha")
    p.add_argument("--lora-r",   type=int,   default=8,    choices=[8, 16],
                   help="LoRA rank (32 forbidden on 8 GB).")
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument(
        "--batch-size", type=int, default=16, choices=[8, 16],
        help="Batch size (32 OOMs on 8 GB with LoRA r≥8).",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--weighted-sampler",
        default="none",
        choices=["none", "DoS-only", "all-rare"],
        help="Weighted sampler strategy for rare-class upsampling.",
    )

    # Base checkpoint — all runs fine-tune from v3 weights (model-only, fresh optimizer)
    p.add_argument(
        "--base-checkpoint",
        default="ml/checkpoints/multilabel-v3-fresh-60ep_best.pt",
        help=(
            "Checkpoint to resume model weights from before each run. "
            "Always uses model-only resume (fresh optimizer/scheduler). "
            "Set to empty string to train from scratch (not recommended for v4 search)."
        ),
    )

    # Regime overrides
    p.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override default epoch count (1 for smoke, 5 for confirm).",
    )

    # Diagnostics
    p.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip GPU VRAM check (useful for CPU debugging).",
    )

    args, extra_argv = p.parse_known_args()

    # Convert remaining ['--some-field', 'value', ...] into a dict.
    extra: dict = {}
    it = iter(extra_argv)
    for token in it:
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            try:
                raw = next(it)
            except StopIteration:
                raw = "true"
            extra[key] = _coerce(raw)
        else:
            logger.warning(f"Ignoring unrecognised token in extra args: {token!r}")

    return args, extra


def _coerce(s: str) -> int | float | bool | str:
    """Try int → float → bool → str."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    return s


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def _preflight(mode: str, skip_vram: bool) -> None:
    """Run all pre-flight checks. Raises SystemExit(1) on any failure."""
    ok = True

    # 1. Locked-file hash guard.
    sidecar_rel = DEFAULT_SIDECAR  # already a relative Path from compute_locked_hashes
    rc = _check_hashes(repo_root=_REPO_ROOT, sidecar=sidecar_rel)
    if rc != 0:
        logger.error(
            "PREFLIGHT FAILED: locked-file hash mismatch. "
            "A v4-sprint-locked file was modified. "
            "Run `python ml/scripts/compute_locked_hashes.py --check` for details."
        )
        ok = False
    else:
        logger.info("Hash guard: OK (all locked files match)")

    # 2. GPU VRAM check.
    if not skip_vram:
        if not torch.cuda.is_available():
            logger.error(
                "PREFLIGHT FAILED: no CUDA GPU detected. "
                "Training requires a CUDA GPU (RTX 3070 8 GB target)."
            )
            ok = False
        else:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb  = free_bytes  / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            if free_gb < MIN_VRAM_GB:
                logger.error(
                    f"PREFLIGHT FAILED: GPU has {free_gb:.1f} GB free "
                    f"(need ≥ {MIN_VRAM_GB:.0f} GB). "
                    "Close other GPU processes and retry."
                )
                ok = False
            else:
                logger.info(f"GPU VRAM: {free_gb:.1f} / {total_gb:.1f} GB free")

    # 3. v4-sprint mode: experiment name must be the designated one.
    #    (Prevents polluting other experiments during the sprint.)
    # Checked at config-build time — nothing to do here.

    # 4. MLflow connectivity (sqlite URI needs no network; just confirm init works).
    try:
        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        logger.info(f"MLflow URI: {mlflow.get_tracking_uri()}")
    except Exception as exc:
        logger.error(f"PREFLIGHT FAILED: MLflow init error: {exc}")
        ok = False

    if not ok:
        raise SystemExit(1)

    logger.info("Pre-flight checks passed.")


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

_TRAINCONFIG_FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


def _build_config(args: argparse.Namespace, extra: dict) -> TrainConfig:
    """Build a TrainConfig from CLI args and arbitrary extra overrides."""
    smoke = args.regime == "smoke"

    epochs = args.max_epochs or (SMOKE_EPOCHS if smoke else CONFIRM_EPOCHS)

    # Base checkpoint: fine-tune from v3 weights (model-only, fresh optimizer).
    base_ckpt = args.base_checkpoint.strip() if args.base_checkpoint else ""
    resume_from = base_ckpt if base_ckpt else None
    if resume_from:
        logger.info(f"Base checkpoint: {resume_from} (model-only resume)")
    else:
        logger.warning("No base checkpoint — training from scratch (not recommended)")

    # Known-arg overrides (highest priority).
    overrides: dict = {
        "experiment_name":          args.experiment_name,
        "run_name":                 args.run_name,
        "loss_fn":                  args.loss_fn,
        "focal_gamma":              args.gamma,
        "focal_alpha":              args.alpha,
        "lora_r":                   args.lora_r,
        "lora_alpha":               args.lora_alpha,
        "batch_size":               args.batch_size,
        "lr":                       args.lr,
        "use_weighted_sampler":     args.weighted_sampler,
        "epochs":                   epochs,
        "smoke_subsample_fraction": SMOKE_SUBSAMPLE if smoke else 1.0,
        # Fine-tune from v3; always model-only so optimizer re-calibrates to new knobs.
        "resume_from":              resume_from,
        "resume_model_only":        True,
        # Smoke skips the RAM cache to avoid polluting a fresh cache with
        # subsample artifacts; confirm uses the cache normally.
        "cache_path":               None if smoke else "ml/data/cached_dataset.pkl",
        # AMP is non-negotiable on 8 GB.
        "use_amp": True,
    }

    # Extra forwarded args fill in TrainConfig fields not covered above.
    # Known-arg overrides win; extras only set fields that weren't set explicitly.
    for key, val in extra.items():
        if key in _TRAINCONFIG_FIELDS and key not in overrides:
            overrides[key] = val
        elif key not in _TRAINCONFIG_FIELDS:
            logger.warning(
                f"Extra arg '{key}' is not a TrainConfig field — ignored. "
                "Check spelling (use underscores, not hyphens)."
            )

    # Drop any keys that aren't valid TrainConfig fields.
    cfg_kwargs = {k: v for k, v in overrides.items() if k in _TRAINCONFIG_FIELDS}
    return TrainConfig(**cfg_kwargs)


# ---------------------------------------------------------------------------
# Threshold tuning (programmatic — no subprocess)
# ---------------------------------------------------------------------------

def _tune_checkpoint(
    checkpoint_path: Path,
    config: TrainConfig,
) -> tuple[float, dict[str, float], Path]:
    """
    Run per-class threshold tuning on val split.

    Returns:
        tuned_f1_macro  — the authoritative SENTINEL_SCORE
        per_class_f1    — {class_name: f1} for floor checking
        thresholds_path — path to the saved JSON
    """
    device = config.device
    label_csv = Path(config.label_csv)
    output_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_thresholds.json")

    model, ckpt_config = load_model_from_checkpoint(checkpoint_path, device)
    val_loader = build_val_loader(
        config=config,
        label_csv=label_csv,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    probs, labels = collect_probabilities(model, val_loader, device)

    grid = build_threshold_grid(0.05, 0.95, 0.05)
    best_thresholds = []
    thresholds_dict: dict[str, float] = {}
    for i, class_name in enumerate(CLASS_NAMES):
        best, rows = sweep_one_class(class_name, probs[:, i], labels[:, i], grid)
        best_thresholds.append(best)
        thresholds_dict[class_name] = best.threshold

    preds   = apply_thresholds(probs, thresholds_dict)
    overall = evaluate_overall(labels, preds)

    save_results(
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        thresholds=thresholds_dict,
        best_thresholds=best_thresholds,
        overall_metrics=overall,
        ckpt_config=ckpt_config,
        thresholds_grid=grid,
    )

    per_class_f1 = {bt.class_name: bt.f1 for bt in best_thresholds}
    return float(overall["f1_macro"]), per_class_f1, output_path


# ---------------------------------------------------------------------------
# Floor check
# ---------------------------------------------------------------------------

def _check_floors(per_class_f1: dict[str, float]) -> list[str]:
    """Return list of classes that breach the v3 per-class floor."""
    breaches = []
    for cls, floor in V3_FLOORS.items():
        actual = per_class_f1.get(cls, 0.0)
        if actual < floor:
            breaches.append(
                f"{cls}: {actual:.4f} < floor {floor:.4f} "
                f"(gap {floor - actual:.4f})"
            )
    return breaches


# ---------------------------------------------------------------------------
# MLflow supplement logging
# ---------------------------------------------------------------------------

def _log_tuned_metrics(
    run_id: str,
    experiment_name: str,
    tuned_f1: float,
    per_class_f1: dict[str, float],
    peak_vram_mb: int,
    regime: str,
    breaches: list[str],
) -> None:
    """Log tuned metrics to the finished training run."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("tuned_f1_macro", tuned_f1)
        mlflow.log_metric("peak_vram_mb", peak_vram_mb)
        for cls, f1 in per_class_f1.items():
            mlflow.log_metric(f"tuned_f1_{cls}", f1)
        mlflow.set_tag("SENTINEL_SCORE",  f"{tuned_f1:.6f}")
        mlflow.set_tag("regime",          regime)
        mlflow.set_tag("gate_passed",     str(tuned_f1 > GATE_F1))
        mlflow.set_tag("floor_breaches",  "; ".join(breaches) if breaches else "none")
        if tuned_f1 > GATE_F1:
            mlflow.set_tag("smoke_promote_hint",
                           "yes" if regime == "smoke" and tuned_f1 > SMOKE_PROMOTE_THRESHOLD
                           else "n/a")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args, extra = _parse_args()

    # v4-sprint: enforce the designated MLflow experiment name.
    if args.mode == "v4-sprint" and args.experiment_name != AUTORESEARCH_EXPERIMENT:
        logger.error(
            f"In v4-sprint mode the experiment must be '{AUTORESEARCH_EXPERIMENT}', "
            f"got '{args.experiment_name}'. "
            "Pass --mode playground to lift this restriction."
        )
        raise SystemExit(1)

    # Pre-flight.
    try:
        _preflight(args.mode, args.skip_preflight)
    except SystemExit:
        raise  # already logged

    # Build config.
    config = _build_config(args, extra)
    logger.info(
        f"Regime={args.regime} | mode={args.mode} | "
        f"run={config.run_name} | experiment={config.experiment_name} | "
        f"loss={config.loss_fn} | lora_r={config.lora_r} | "
        f"batch={config.batch_size} | lr={config.lr} | "
        f"sampler={config.use_weighted_sampler}"
    )

    # Training.
    peak_vram_mb = 0
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.monotonic()
        result = train(config)
        elapsed = time.monotonic() - t0

        if torch.cuda.is_available():
            peak_vram_mb = int(torch.cuda.max_memory_allocated() / (1024 ** 2))

        logger.info(
            f"Training complete | elapsed={elapsed:.0f}s | "
            f"best_raw_f1={result['best_f1_macro']:.4f} | "
            f"epoch={result['final_epoch']} | "
            f"early_stopped={result['early_stopped']} | "
            f"peak_vram={peak_vram_mb} MB"
        )
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "OOM during training. "
            "Try --batch-size 8 or --lora-r 8. Exit 2."
        )
        raise SystemExit(2)
    except Exception:
        logger.exception("Runtime failure during training. Exit 2.")
        raise SystemExit(2)

    # Retrieve the MLflow run ID for supplemental logging.
    last_run = mlflow.last_active_run()
    run_id = last_run.info.run_id if last_run else None

    # Threshold tuning.
    checkpoint_path = Path(result["checkpoint_path"])
    try:
        tuned_f1, per_class_f1, thresholds_path = _tune_checkpoint(checkpoint_path, config)
        logger.info(f"Tuned F1-macro: {tuned_f1:.4f} (gate={GATE_F1})")
    except Exception:
        logger.exception(
            "Threshold tuning failed — emitting SENTINEL_SCORE=0.0 (exit 3)."
        )
        # Still emit the protocol lines so the agent can log this as a discard.
        print(f"SENTINEL_SCORE=0.0")
        print(f"PEAK_VRAM_MB={peak_vram_mb}")
        print(f"REGIME={args.regime}")
        raise SystemExit(3)

    # Floor check and gate report.
    breaches = _check_floors(per_class_f1)
    gate_pass = tuned_f1 > GATE_F1 and not breaches

    logger.info(
        f"Gate check: {'PASS' if gate_pass else 'FAIL'} | "
        f"tuned_F1={tuned_f1:.4f} vs gate={GATE_F1} | "
        f"floor_breaches={len(breaches)}"
    )
    if breaches:
        logger.warning("Per-class floor breaches (do NOT promote this run):")
        for b in breaches:
            logger.warning(f"  {b}")
    if args.regime == "smoke" and tuned_f1 > SMOKE_PROMOTE_THRESHOLD:
        logger.info(
            f"Smoke F1 {tuned_f1:.4f} > promote threshold {SMOKE_PROMOTE_THRESHOLD} "
            "→ consider escalating to --regime confirm."
        )

    # Supplement MLflow with tuned metrics.
    if run_id:
        try:
            _log_tuned_metrics(
                run_id=run_id,
                experiment_name=config.experiment_name,
                tuned_f1=tuned_f1,
                per_class_f1=per_class_f1,
                peak_vram_mb=peak_vram_mb,
                regime=args.regime,
                breaches=breaches,
            )
        except Exception:
            logger.warning("Failed to log tuned metrics to MLflow (non-fatal).")

    # Machine-readable output (agent greps these exact prefixes).
    print(f"SENTINEL_SCORE={tuned_f1:.6f}")
    print(f"PEAK_VRAM_MB={peak_vram_mb}")
    print(f"REGIME={args.regime}")

    raise SystemExit(0)


if __name__ == "__main__":
    main()
