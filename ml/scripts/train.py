"""
train.py — SENTINEL Training Entry Point (v7 three-eye + JK, 7-layer GNN, LoRA)

NAMING CONVENTION
-----------------
Always include the date in --run-name so checkpoints, log files, and MLflow
runs are uniquely named and never overwrite each other:

    --run-name v7.0-20260518
    --run-name v7.0-smoke-20260518

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
        --experiment-name sentinel-v7 \\
        --epochs 100

Usage examples:

    # v7.0 smoke run (2 epochs, 10% data)
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \\
        --run-name v7.0-smoke-20260518 \\
        --experiment-name sentinel-v7 \\
        --epochs 2 \\
        --smoke-subsample-fraction 0.1
    # Gates: GNN share ≥ 15%; JK all phases > 5%; no NaN after step 50

    # v7.0 full run
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \\
        --run-name v7.0-20260518 \\
        --experiment-name sentinel-v7 \\
        --epochs 100

    # Disable JK (ablation)
    TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \\
        --run-name v7.0-no-jk-20260518 \\
        --no-jk \\
        --epochs 100
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
    p.add_argument("--run-name",        default="sentinel-v8")
    p.add_argument("--experiment-name", default="sentinel-multilabel")

    # --- Model ---
    p.add_argument("--num-classes", type=int, default=10)

    # --- Label source ---
    p.add_argument("--label-csv", default="ml/data/processed/multilabel_index_cleaned.csv")

    # --- Training hyperparameters ---
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=8)
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
        default=8,
        metavar="N",
        help=(
            "Accumulate gradients over N micro-batches before stepping the optimizer. "
            "Effective batch size = batch_size × N. "
            "v7 default: 8 (batch=8 × 8 = effective 64) on RTX 3070 8 GB."
        ),
    )

    # --- Paths ---
    p.add_argument("--graphs-dir",      default="ml/data/graphs")
    p.add_argument("--tokens-dir",      default="ml/data/tokens_windowed",
                   help="Token .pt directory. Multi-window [W,512] tensors (v7 default).")
    p.add_argument("--splits-dir",      default="ml/data/splits/deduped")
    p.add_argument("--checkpoint-dir",  default="ml/checkpoints")
    p.add_argument("--checkpoint-name", default=None)
    p.add_argument("--cache-path",      default="ml/data/cached_dataset_v8.pkl",
                   help="RAM cache pickle (v8 — schema v8 graphs with CALL_ENTRY/RETURN_TO/DEF_USE).")

    # --- Loss and regularisation ---
    p.add_argument("--loss-fn",              choices=["bce", "focal", "asl"], default="asl")
    p.add_argument("--early-stop-patience",  type=int,   default=30)
    p.add_argument("--grad-clip",            type=float, default=1.0)
    p.add_argument("--warmup-pct",           type=float, default=0.10)
    p.add_argument("--use-amp",              action="store_true", default=True)
    p.add_argument("--no-amp",               dest="use_amp", action="store_false")
    p.add_argument("--num-workers",          type=int, default=4,
                   help="DataLoader workers. Uses fork workers (CoW cache, no RAM copy). 0=main-process fallback.")
    p.add_argument("--compile",              action="store_true", default=True,
                   help="torch.compile(model, dynamic=True) — ~20-40%% speedup. Set TRITON_CACHE_DIR=/tmp/triton_cache to avoid WSL p9io crash.")
    p.add_argument("--no-compile",           dest="compile", action="store_false")
    p.add_argument("--log-interval",         type=int, default=100)
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--focal-alpha",          type=float, default=0.25)
    p.add_argument("--asl-gamma-neg",        type=float, default=2.0,
                   help="ASL focus exponent for negatives. v7: 2.0 (4.0 caused all-zeros collapse). Only used when --loss-fn=asl.")
    p.add_argument("--asl-gamma-pos",        type=float, default=1.0,
                   help="ASL focus exponent for positives (default 1.0). Only used when --loss-fn=asl.")
    p.add_argument("--asl-clip",             type=float, default=0.01,
                   help="ASL probability margin. v7: 0.01 (0.05 caused oscillation at p≈0.03–0.06). Negatives with p<clip→zero gradient.")
    p.add_argument("--label-smoothing",      type=float, default=0.0,
                   help="Uniform label smoothing ε (0=off). v7: replaced by per-class class_label_smoothing in TrainConfig.")
    p.add_argument("--fusion-lr-multiplier", type=float, default=0.5,
                   help="LR multiplier for fusion+classifier params (RC1 fix).")
    p.add_argument("--pos-weight-min-samples", type=int, default=3000,
                   help=(
                       "Classes with >= this many training positives get pos_weight=1.0 "
                       "(no amplification). v7 default 3000: caps Reentrancy (4498 samples) at 1.0× "
                       "to prevent the 2.82× FN penalty that dominated gradients in v5.2. "
                       "0=disabled (all classes amplified)."
                   ))

    # --- GNN architecture (v6) ---
    p.add_argument("--gnn-hidden-dim",   type=int,   default=256)
    p.add_argument("--gnn-layers",       type=int,   default=7)
    p.add_argument("--gnn-heads",        type=int,   default=8)
    p.add_argument("--gnn-dropout",      type=float, default=0.2)
    p.add_argument("--gnn-edge-emb-dim", type=int,   default=64)
    p.add_argument("--no-edge-attr",     dest="use_edge_attr", action="store_false", default=True)
    p.add_argument("--no-jk",            dest="gnn_use_jk",   action="store_false", default=True,
                   help="Disable JK attention aggregation (v5.2 default: enabled)")
    p.add_argument("--phase2-edge-types", type=int, nargs="+", default=None,
                   dest="gnn_phase2_edge_types",
                   help=(
                       "Edge type IDs for Phase 2 cfg_mask. None=all v8 types (6,8,9,10). "
                       "Ablation examples: ICFG-only=6 8 9  DFG-only=6 10"
                   ))
    p.add_argument("--gnn-lr-multiplier",  type=float, default=2.5,
                   help="GNN LR = lr × this (default 2.5 — counteracts GNN gradient collapse)")
    p.add_argument("--lora-lr-multiplier", type=float, default=0.3,
                   help="LoRA LR = lr × this (default 0.3 — v6: tighter than 0.5 with wider GNN)")

    # --- Auxiliary loss (v5 three-eye) ---
    p.add_argument("--aux-loss-weight", type=float, default=0.3)
    p.add_argument("--dos-loss-weight", type=float, default=0.5,
                   help="DoS gradient weight (0.0=no gradient, 0.5=half, 1.0=full; 243 training positives as of 2026-05-23)")

    # --- LoRA architecture (v5) ---
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    # --- GNN prefix injection (Phase 1) ---
    p.add_argument("--gnn-prefix-k",             type=int,   default=0,
                   help="Number of GNN prefix tokens injected before code (0=disabled; 48 for Phase 1)")
    p.add_argument("--gnn-prefix-warmup-epochs", type=int,   default=15,
                   help="Epochs to suppress prefix injection (projection trains from random init after warmup)")
    p.add_argument("--gnn-prefix-proj-lr-mult",  type=float, default=5.0,
                   help="LR multiplier for gnn_to_bert_proj and prefix_type_embedding (NH-5: raised from 1.0 for cold-start)")
    p.add_argument("--no-prefix-proj-reset", action="store_true",
                   help="Disable NC-1 Adam state reset for prefix_proj at warmup transition")
    p.add_argument("--jk-entropy-reg-lambda", type=float, default=0.01,
                   help="C-3: JK entropy regularizer weight (0=disabled; 0.01 default penalises Phase3 collapse)")
    p.add_argument("--pos-weight-cap", type=float, default=10.0,
                   help="M-1: cap on sqrt-scaled pos_weight per class (was 20.0)")

    p.add_argument("--smoke-subsample-fraction", type=float, default=1.0)
    p.add_argument(
        "--weighted-sampler",
        choices=["none", "positive", "DoS-only", "all-rare"],
        default="positive",
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
        asl_gamma_neg         = args.asl_gamma_neg,
        asl_gamma_pos         = args.asl_gamma_pos,
        asl_clip              = args.asl_clip,
        gnn_hidden_dim        = args.gnn_hidden_dim,
        gnn_layers            = args.gnn_layers,
        gnn_heads             = args.gnn_heads,
        gnn_dropout           = args.gnn_dropout,
        gnn_edge_emb_dim      = args.gnn_edge_emb_dim,
        use_edge_attr         = args.use_edge_attr,
        gnn_use_jk            = args.gnn_use_jk,
        gnn_phase2_edge_types = args.gnn_phase2_edge_types,
        gnn_lr_multiplier     = args.gnn_lr_multiplier,
        lora_lr_multiplier    = args.lora_lr_multiplier,
        fusion_lr_multiplier  = args.fusion_lr_multiplier,
        label_smoothing       = args.label_smoothing,
        pos_weight_min_samples = args.pos_weight_min_samples,
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
        use_compile           = args.compile,
        log_interval          = args.log_interval,
        graphs_dir            = args.graphs_dir,
        tokens_dir            = args.tokens_dir,
        splits_dir            = args.splits_dir,
        checkpoint_dir        = args.checkpoint_dir,
        checkpoint_name       = args.checkpoint_name or f"{args.run_name}_best.pt",
        cache_path            = args.cache_path,
        resume_from           = args.resume,
        resume_model_only     = args.resume_model_only,
        force_optimizer_reset = args.force_optimizer_reset,
        use_weighted_sampler  = args.weighted_sampler,
        dos_loss_weight       = args.dos_loss_weight,
        gnn_prefix_k                    = args.gnn_prefix_k,
        gnn_prefix_warmup_epochs        = args.gnn_prefix_warmup_epochs,
        gnn_prefix_proj_lr_mult         = args.gnn_prefix_proj_lr_mult,
        gnn_prefix_proj_reset_on_warmup = not args.no_prefix_proj_reset,
        jk_entropy_reg_lambda           = args.jk_entropy_reg_lambda,
        pos_weight_cap                  = args.pos_weight_cap,
    )

    train(config)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
