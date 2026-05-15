"""
train.py — SENTINEL Training Entry Point (v5.2 three-eye + JK architecture)

NAMING CONVENTION
-----------------
Always include the date in --run-name so checkpoints, log files, and MLflow
runs are uniquely named and never overwrite each other:

    --run-name v5.2-smoke-20260514
    --run-name v5.2-jk-20260514

Each run produces:
    ml/checkpoints/<run-name>_best.pt          ← checkpoint
    ml/checkpoints/<run-name>_best.state.json  ← patience/epoch sidecar
    ml/logs/<run-name>.log                     ← append-mode per-run log

RESUMING
--------
The resume command is printed at training startup. General form:

    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \\
        --resume ml/checkpoints/<run-name>_best.pt \\
        --run-name <run-name>-resumed \\
        --experiment-name sentinel-v5.2 \\
        --epochs 60 \\
        --gradient-accumulation-steps 4

Usage examples:

    # v5.2 smoke run (2 epochs, 10% data — run this first to clear Phase 4 gates)
    # Note: log_interval auto-adjusts to ~12 steps for 10% smoke data + accum=4
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \\
        --run-name v5.2-smoke-20260514 \\
        --experiment-name sentinel-v5.2 \\
        --epochs 2 \\
        --smoke-subsample-fraction 0.1 \\
        --gradient-accumulation-steps 4
    # Phase 4 gates: GNN share ≥ 15%; JK all phases > 5%; no NaN after step 50

    # v5.2 full 60-epoch run (after smoke gates pass)
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \\
        --run-name v5.2-jk-20260514 \\
        --experiment-name sentinel-v5.2 \\
        --epochs 60 \\
        --gradient-accumulation-steps 4

    # Disable JK (ablation)
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \\
        --run-name v5.2-no-jk-20260514 \\
        --no-jk \\
        --epochs 60 \\
        --gradient-accumulation-steps 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch.multiprocessing as mp
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.training.trainer import TrainConfig, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SENTINEL — GNN + CodeBERT multi-label vulnerability detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Run identity ---
    p.add_argument("--run-name",        default="multilabel-v5-fresh")
    p.add_argument("--experiment-name", default="sentinel-multilabel")

    # --- Model ---
    p.add_argument("--num-classes", type=int, default=10)

    # --- Label source ---
    p.add_argument("--label-csv", default="ml/data/processed/multilabel_index_deduped.csv")

    # --- Training hyperparameters ---
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--threshold",      type=float, default=0.5,
                   help="Inference decision threshold (also used by tune_threshold.py).")
    p.add_argument("--eval-threshold", type=float, default=0.35,
                   help=(
                       "Training-time evaluation threshold for early stopping / patience. "
                       "Intentionally lower than --threshold (0.5) so minority classes are "
                       "not flipping above/below the boundary every epoch, which would "
                       "inject ±0.04 macro-F1 noise into the patience counter. "
                       "Default 0.35 based on observed minority-class probability clustering."
                   ))

    # --- Gradient accumulation ---
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Accumulate gradients over N micro-batches before stepping the optimizer. "
            "Effective batch size = batch_size × N. "
            "Use 4 on RTX 3070 8 GB (effective batch=64) to avoid VRAM fragmentation "
            "slowdown that appears from epoch 2 onward. Default: 1 (disabled)."
        ),
    )

    # --- Paths ---
    p.add_argument("--graphs-dir",      default="ml/data/graphs")
    p.add_argument("--tokens-dir",      default="ml/data/tokens")
    p.add_argument("--splits-dir",      default="ml/data/splits/deduped")
    p.add_argument("--checkpoint-dir",  default="ml/checkpoints")
    p.add_argument("--checkpoint-name", default=None)

    # --- Loss and regularisation ---
    p.add_argument("--loss-fn",              choices=["bce", "focal"], default="bce")
    p.add_argument("--early-stop-patience",  type=int,   default=10)
    p.add_argument("--grad-clip",            type=float, default=1.0)
    p.add_argument("--warmup-pct",           type=float, default=0.10)
    p.add_argument("--use-amp",              action="store_true", default=True)
    p.add_argument("--no-amp",               dest="use_amp", action="store_false")
    p.add_argument("--num-workers",          type=int, default=2)
    p.add_argument("--log-interval",         type=int, default=100)
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--focal-alpha",          type=float, default=0.25)

    # --- GNN architecture (v5.2) ---
    p.add_argument("--gnn-hidden-dim",   type=int,   default=128)
    p.add_argument("--gnn-layers",       type=int,   default=4)
    p.add_argument("--gnn-heads",        type=int,   default=8)
    p.add_argument("--gnn-dropout",      type=float, default=0.2)
    p.add_argument("--gnn-edge-emb-dim", type=int,   default=32)
    p.add_argument("--no-edge-attr",     dest="use_edge_attr", action="store_false", default=True)
    p.add_argument("--no-jk",            dest="gnn_use_jk",   action="store_false", default=True,
                   help="Disable JK attention aggregation (v5.2 default: enabled)")
    p.add_argument("--gnn-lr-multiplier",  type=float, default=2.5,
                   help="GNN LR = lr × this (default 2.5 — counteracts GNN gradient collapse)")
    p.add_argument("--lora-lr-multiplier", type=float, default=0.5,
                   help="LoRA LR = lr × this (default 0.5 — prevents CodeBERT forgetting)")

    # --- Auxiliary loss (v5 three-eye) ---
    p.add_argument("--aux-loss-weight", type=float, default=0.3)

    # --- LoRA architecture (v5) ---
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    p.add_argument("--smoke-subsample-fraction", type=float, default=1.0)
    p.add_argument(
        "--weighted-sampler",
        choices=["none", "DoS-only", "all-rare"],
        default="none",
    )

    # --- Resume ---
    p.add_argument("--resume", default=None, metavar="CHECKPOINT")
    p.add_argument(
        "--no-resume-model-only",
        dest="resume_model_only",
        action="store_false",
        default=True,
    )
    p.add_argument(
        "--resume-reset-optimizer",
        dest="force_optimizer_reset",
        action="store_true",
        default=False,
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
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
        eval_threshold        = args.eval_threshold,
        loss_fn               = args.loss_fn,
        focal_gamma           = args.focal_gamma,
        focal_alpha           = args.focal_alpha,
        gnn_hidden_dim        = args.gnn_hidden_dim,
        gnn_layers            = args.gnn_layers,
        gnn_heads             = args.gnn_heads,
        gnn_dropout           = args.gnn_dropout,
        gnn_edge_emb_dim      = args.gnn_edge_emb_dim,
        use_edge_attr         = args.use_edge_attr,
        gnn_use_jk            = args.gnn_use_jk,
        gnn_lr_multiplier     = args.gnn_lr_multiplier,
        lora_lr_multiplier    = args.lora_lr_multiplier,
        aux_loss_weight       = args.aux_loss_weight,
        lora_r                = args.lora_r,
        lora_alpha            = args.lora_alpha,
        lora_dropout          = args.lora_dropout,
        smoke_subsample_fraction      = args.smoke_subsample_fraction,
        gradient_accumulation_steps   = args.gradient_accumulation_steps,
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
    mp.set_start_method('spawn', force=True)
    main()
