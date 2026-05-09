"""
train.py — SENTINEL Training Entry Point (Track 3, 2026-04-17)

Thin wrapper around ml.src.training.trainer that provides a clean
command-line interface for starting and resuming training runs.

Usage:
    # Run with defaults (multi-label Track 3, 40 epochs)
    poetry run python ml/scripts/train.py

    # Override common hyperparameters
    poetry run python ml/scripts/train.py \\
        --run-name multilabel-v2 \\
        --epochs 40 \\
        --lr 3e-4 \\
        --batch-size 32

    # Resume from a checkpoint (model weights only, fresh optimizer/scheduler)
    # Use this when batch_size or any training hyperparameter has changed.
    poetry run python ml/scripts/train.py \\
        --resume ml/checkpoints/multilabel-v1_best.pt \\
        --run-name multilabel-v1-resumed

    # Full resume (model + optimizer + scheduler state restored exactly)
    # Only use this when batch_size is IDENTICAL to the checkpoint.
    poetry run python ml/scripts/train.py \\
        --resume ml/checkpoints/multilabel-v1_best.pt \\
        --run-name multilabel-v1-resumed \\
        --no-resume-model-only

    # Full resume + force optimizer reset (model weights + patience_counter
    # preserved, but stale Adam moments discarded). Use when batch_size changed
    # but you still want the exact epoch counter from the checkpoint.
    poetry run python ml/scripts/train.py \\
        --resume ml/checkpoints/multilabel-v1_best.pt \\
        --run-name multilabel-v1-resumed \\
        --no-resume-model-only \\
        --resume-reset-optimizer

    # Binary legacy run (for comparison)
    poetry run python ml/scripts/train.py \\
        --num-classes 1 \\
        --label-csv "" \\
        --experiment-name sentinel-training \\
        --run-name binary-compare

All commands must be run from the project root (~/projects/sentinel),
not from ml/ — the ml.src.* import chain requires the project root on sys.path.

RESUME GUIDE
────────────
Choose the right resume mode based on what has changed:

  batch_size SAME, epochs extended:
    → --no-resume-model-only                           (full resume, Fix #10 handles scheduler)

  batch_size CHANGED (e.g. 16→32):
    → (default, no flag)                               (model-only, cleanest option)
    → --no-resume-model-only --resume-reset-optimizer  (force-reset, keeps epoch counter)

  Any hyperparameter change:
    → (default, no flag)                               (model-only is always safe)

  Exact continuation of an interrupted run (same config):
    → --no-resume-model-only                           (full resume)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
# Make ml/ importable when running as a script from the repo root.
# parents[2] resolves from ml/scripts/ → ml/ → project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.training.trainer import TrainConfig, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SENTINEL — GNN + CodeBERT multi-label vulnerability detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Run identity ---
    p.add_argument(
        "--run-name",
        default="multilabel-crossattn-v1",
        help="MLflow run name. Also used as checkpoint prefix: <run-name>_best.pt",
    )
    p.add_argument(
        "--experiment-name",
        default="sentinel-multilabel",
        help="MLflow experiment name. Use 'sentinel-training' for binary comparison runs.",
    )

    # --- Model ---
    p.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help=(
            "Number of output classes. "
            "10 = multi-label Track 3 (default). "
            "1 = binary legacy mode (set label-csv to empty string)."
        ),
    )

    # --- Label source ---
    p.add_argument(
        "--label-csv",
        default="ml/data/processed/multilabel_index.csv",
        help=(
            "Path to multilabel_index.csv for multi-label label loading. "
            "Set to empty string for binary mode (uses graph.y labels)."
        ),
    )

    # --- Training hyperparameters ---
    p.add_argument("--epochs",       type=int,   default=40,   help="Number of training epochs")
    p.add_argument("--batch-size",   type=int,   default=32,   help="Batch size for train + val loaders")
    p.add_argument("--lr",           type=float, default=3e-4, help="AdamW learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Decision threshold applied to sigmoid(logits) during evaluate(). "
            "Used for F1 computation. Run tune_threshold.py after retrain to find "
            "optimal per-class or shared threshold."
        ),
    )

    # --- Paths ---
    p.add_argument("--graphs-dir",      default="ml/data/graphs",  help="Graph .pt files directory")
    p.add_argument("--tokens-dir",      default="ml/data/tokens",  help="Token .pt files directory")
    p.add_argument("--splits-dir",      default="ml/data/splits",  help="Split index .npy files directory")
    p.add_argument("--checkpoint-dir",  default="ml/checkpoints",  help="Directory to save checkpoints")
    p.add_argument(
        "--checkpoint-name",
        default=None,
        help="Checkpoint filename. Defaults to <run-name>_best.pt",
    )

    # --- Loss and regularisation ---
    p.add_argument(
        "--loss-fn",
        choices=["bce", "focal"],
        default="bce",
        help="Loss function: 'bce' (BCEWithLogitsLoss) or 'focal' (FocalLoss).",
    )
    p.add_argument("--early-stop-patience", type=int,   default=7,    help="Early stopping patience (epochs without improvement)")
    p.add_argument("--grad-clip",           type=float, default=1.0,  help="Max gradient norm for clip_grad_norm_")
    p.add_argument("--warmup-pct",          type=float, default=0.05, help="Fraction of steps used for OneCycleLR warm-up")
    p.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (default: enabled)",
    )
    p.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable AMP (useful for debugging NaN losses)",
    )
    p.add_argument("--num-workers",  type=int, default=2,   help="DataLoader worker processes")
    p.add_argument("--log-interval", type=int, default=100, help="Log loss every N batches")
    p.add_argument("--focal-gamma", type=float, default=2.0,  help="FocalLoss focusing exponent (used when --loss-fn focal)")
    p.add_argument("--focal-alpha", type=float, default=0.25, help="FocalLoss class-balance weight (used when --loss-fn focal)")
    p.add_argument(
        "--weighted-sampler",
        choices=["none", "DoS-only", "all-rare"],
        default="none",
        help=(
            "Weighted random sampler strategy for rare-class upsampling. "
            "DoS-only: upsample DenialOfService class 39× (137 vs 5343 IntegerUO samples). "
            "all-rare: inverse class-count weighting (singletons get highest weight)."
        ),
    )

    # --- Resume ---
    p.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Path to a checkpoint saved by a previous run. "
            "Loads model weights, optimizer state, and epoch counter. "
            "Checkpoint must be in new dict format with 'model', 'optimizer', "
            "'epoch', 'best_f1', 'config' keys. "
            "Old plain state_dict checkpoints are NOT resumable."
        ),
    )
    p.add_argument(
        "--no-resume-model-only",
        dest="resume_model_only",
        action="store_false",
        default=True,
        help=(
            "When set, restores optimizer state from checkpoint in addition to "
            "model weights (full resume). Scheduler state is only restored if "
            "total_steps matches — on epoch extension it is skipped automatically "
            "(Fix #10). WARNING: Only use this when batch_size is IDENTICAL to "
            "the checkpoint. If batch_size changed, use --resume-reset-optimizer "
            "together with this flag, or omit this flag entirely (model-only resume). "
            "Default is model-weights-only resume (fresh optimizer/scheduler)."
        ),
    )
    p.add_argument(
        "--resume-reset-optimizer",
        dest="force_optimizer_reset",
        action="store_true",
        default=False,
        help=(
            "When set alongside --no-resume-model-only, the optimizer and scheduler "
            "state from the checkpoint are DISCARDED even though a full resume was "
            "requested. Model weights and patience_counter are still restored from "
            "the checkpoint. This is the correct flag to use when batch_size has "
            "changed since the checkpoint was saved — it gives AdamW a clean start "
            "calibrated to the new gradient noise level while keeping the exact "
            "epoch counter and early-stopping state from the checkpoint. "
            "Has no effect if --no-resume-model-only is not also set."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Empty string from CLI → None for label_csv (triggers binary mode in TrainConfig)
    label_csv = args.label_csv if args.label_csv else None

    config = TrainConfig(
        run_name              = args.run_name,
        experiment_name       = args.experiment_name,
        num_classes           = args.num_classes,
        label_csv             = label_csv,
        epochs                = args.epochs,
        batch_size            = args.batch_size,
        lr                    = args.lr,
        weight_decay          = args.weight_decay,
        threshold             = args.threshold,
        loss_fn               = args.loss_fn,
        focal_gamma           = args.focal_gamma,
        focal_alpha           = args.focal_alpha,
        early_stop_patience   = args.early_stop_patience,
        grad_clip             = args.grad_clip,
        warmup_pct            = args.warmup_pct,
        use_amp               = args.use_amp,
        num_workers           = args.num_workers,
        log_interval          = args.log_interval,
        graphs_dir            = args.graphs_dir,
        tokens_dir            = args.tokens_dir,
        splits_dir            = args.splits_dir,
        checkpoint_dir        = args.checkpoint_dir,
        checkpoint_name       = args.checkpoint_name or f"{args.run_name}_best.pt",
        resume_from           = args.resume,
        resume_model_only     = args.resume_model_only,
        force_optimizer_reset = args.force_optimizer_reset,
        use_weighted_sampler  = args.weighted_sampler,
    )

    train(config)


if __name__ == "__main__":
    main()
