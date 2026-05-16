"""
trainer.py — SENTINEL Training Loop (Cross-Attention + LoRA Upgrade)
FINAL STANDALONE VERSION – with tqdm progress bars, safe resume, offline mode,
and EARLY STOPPING.

SPEED OPTIMISATIONS APPLIED (vs original):
────────────────────────────────────────────────────────────────────────────
1. AMP (Automatic Mixed Precision)
2. TF32 matmuls enabled
3. num_workers raised to 2
4. pin_memory consistent with num_workers
5. zero_grad(set_to_none=True)
6. MLflow log_artifact moved outside epoch loop
7. EARLY STOPPING
8. GRADIENT ACCUMULATION (2026-05-12)
     gradient_accumulation_steps=N accumulates gradients over N micro-batches
     before calling optimizer.step(). Effective batch = batch_size × N.

AUDIT FIXES (2026-05-01 through 2026-05-13): see inline comments.
────────────────────────────────────────────────────────────────────────────
Fix #26 — need_weights=False on MHA in fusion_layer.py (see that file)
Fix #27 — gc.collect() + torch.cuda.empty_cache() between epochs
Fix #28 — batch_size default 16→8 (8 GB GPU compatibility)
           Also: grad norm logging moved inside is_accum_step block, before
           zero_grad(). Previously _grad_norm() fired AFTER zero_grad
           (set_to_none=True), so .grad was always None → always 0.000.
           Now grads are read after scaler.unscale_() but before zero_grad,
           i.e. in fp32 and post-clip — the correct moment to measure them.
           Logging fires on optimizer steps that cross log_interval, not on
           every micro-batch.
Fix #29 — Mid-epoch VRAM cleanup: when reserved VRAM > 90 %, call
           gc.collect() + torch.cuda.empty_cache() to release fragmented
           blocks.  Checked every log_interval optimizer steps.
Fix #30 — train.py CLI --batch-size default 32→8 (see that file)
Fix #31 — dual_path_dataset.py improved diagnostics (see that file)
Fix #32 — Scheduler resume bug: OneCycleLR was created with
           epochs=remaining_epochs instead of epochs=config.epochs, causing
           total_steps mismatch on resume and the scheduler was silently
           discarded.  Now always created with the full epoch count so
           checkpoint state_dict loads correctly.
Fix #33 — Aux loss warmup: aux_loss_weight ramps linearly from 0 to its
           configured value over the first aux_loss_warmup_epochs epochs.
           This prevents the three auxiliary classification heads from
           dominating gradients while the main classifier is still learning
           the basics (observed: aux loss 2-4× main loss early on).
Fix #34 — VRAM usage logged every epoch so OOM risk is visible.
"""

from __future__ import annotations
import gc
import json
import os
# ---------------------------------------------------------------------------
import sys
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score, hamming_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn
from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.focalloss import FocalLoss
from ml.src.training.losses import AsymmetricLoss

# ---------------------------------------------------------------------------
# Logging setup — module level only (handlers added per-run inside train())
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stderr, level="INFO")

# ---------------------------------------------------------------------------
# Class names – single source of truth
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "CallToUnknown",               # 0
    "DenialOfService",             # 1
    "ExternalBug",                 # 2
    "GasException",                # 3
    "IntegerUO",                   # 4
    "MishandledException",         # 5
    "Reentrancy",                  # 6
    "Timestamp",                   # 7
    "TransactionOrderDependence",  # 8
    "UnusedReturn",                # 9
]
NUM_CLASSES = len(CLASS_NAMES)

ARCHITECTURE = "three_eye_v5"

# Phase 1-A6 (2026-05-14): version string written to every saved checkpoint.
# Allows the resume path to detect and warn about architecture mismatches when
# loading a pre-v5.2 checkpoint (which lacks JK/REVERSE_CONTAINS) into a v5.2
# model.  Use _parse_version() for tuple comparison (not string sort).
MODEL_VERSION = "v6.0"

_VALID_LOSS_FNS: frozenset[str] = frozenset({"bce", "focal", "asl"})

# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------
def _vram_pct() -> float:
    """Return reserved VRAM as a fraction of total GPU memory (0.0–1.0).

    Uses ``torch.cuda.memory_reserved()`` (the caching allocator's pool)
    rather than ``memory_allocated()`` so that fragmented-but-unused blocks
    are counted — those are exactly what ``empty_cache()`` can release.
    Returns 0.0 on CPU-only systems.
    """
    if not torch.cuda.is_available():
        return 0.0
    reserved = torch.cuda.memory_reserved()
    total    = torch.cuda.get_device_properties(0).total_memory
    return reserved / total if total > 0 else 0.0


def _vram_str() -> str:
    """Human-readable VRAM string, e.g. ``6.2/7.8 GiB (79.5%)``."""
    if not torch.cuda.is_available():
        return "N/A"
    alloc = torch.cuda.memory_allocated()  / (1024**3)
    reser = torch.cuda.memory_reserved()   / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"{reser:.1f}/{total:.1f} GiB ({_vram_pct():.1%})"


def _parse_version(v: str) -> tuple[int, ...]:
    """
    Parse a version string like 'v5.2' or '5.1.3' into a comparable tuple.

    Phase 1-A6 (2026-05-14): used to detect when a checkpoint was saved by an
    older model version (e.g. pre-v5.2, no JK/REVERSE_CONTAINS) and log a
    warning so operators know why strict=False may silently discard keys.

    Examples:
        _parse_version("v5.2")   → (5, 2)
        _parse_version("5.1.3")  → (5, 1, 3)
        _parse_version("v4")     → (4,)
    """
    return tuple(int(x) for x in v.lstrip("v").split(".") if x.isdigit())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # --- Paths ---
    graphs_dir:      str = "ml/data/graphs"
    tokens_dir:      str = "ml/data/tokens"
    splits_dir:      str = "ml/data/splits/deduped"
    checkpoint_dir:  str = "ml/checkpoints"
    checkpoint_name: str = "multilabel-v5-fresh_best.pt"

    # --- Model ---
    num_classes:       int   = NUM_CLASSES
    fusion_output_dim: int   = 128
    fusion_dropout:    float = 0.3

    # --- GNN architecture (v6) ---
    gnn_hidden_dim:   int   = 256
    gnn_layers:       int   = 6
    gnn_heads:        int   = 8
    gnn_dropout:      float = 0.2
    use_edge_attr:    bool  = True
    gnn_edge_emb_dim: int   = 64
    # JK connections (Phase 1-A1, 2026-05-14)
    gnn_use_jk:       bool  = True
    gnn_jk_mode:      str   = 'attention'

    # --- LoRA architecture (v5) ---
    lora_r:               int        = 16
    lora_alpha:           int        = 32
    lora_dropout:         float      = 0.1
    lora_target_modules:  list[str]  = field(default_factory=lambda: ["query", "value"])

    # --- Label source ---
    label_csv: str = "ml/data/processed/multilabel_index_deduped.csv"

    # --- Training ---
    epochs:              int   = 100         # v6: 100 epochs (was 60); more data + harder loss need more steps
    batch_size:          int   = 16          # Fix #28: was 16, reduced for 8 GB GPU
    lr:                  float = 2e-4
    weight_decay:        float = 1e-2
    # Phase 2-B1 (2026-05-14): per-group LR multipliers.
    # GNN collapsed to ~10% gradient share by epoch 8 in v5.1-fix28 — boosting
    # its LR relative to LoRA counteracts the imbalance without changing the
    # global schedule.  LoRA at 0.5× prevents catastrophic forgetting of
    # CodeBERT features that took many epochs to adapt.
    gnn_lr_multiplier:     float = 2.5      # effective GNN LR = lr * 2.5
    lora_lr_multiplier:    float = 0.3      # effective LoRA LR = lr * 0.3 (v6: tighter than 0.5 — GNN is wider)
    # RC1 fix (2026-05-16): CrossAttentionFusion (821K params) was running at
    # full base LR and producing 4-5× higher gradient norms than the GNN.
    # CodeBERT's "external-call = Reentrancy" signal propagates through fusion
    # and overwhelms the classifier when fusion learns too fast.  0.5× matches
    # the LoRA rate and lets the GNN signal catch up.
    fusion_lr_multiplier:  float = 0.5      # effective fusion LR = lr * 0.5
    threshold:           float = 0.5   # inference threshold (used by predictor/tune_threshold)
    # Training-time evaluation threshold — intentionally lower than inference threshold.
    # At threshold=0.5 minority classes (UnusedReturn, TOD, ExternalBug, Timestamp) have
    # predicted probabilities clustering at 0.35–0.50.  A ±0.03 probability shift near
    # the boundary flips them between F1=0 and F1=0.15, causing ±0.04 macro-F1 swings
    # per epoch that are pure measurement noise.  The patience counter then fires early
    # while training loss is still declining (observed: stopped at ep30, loss=0.8855,
    # still improving).  Setting eval_threshold=0.35 moves minority classes away from
    # the boundary so patience receives a real learning signal, not threshold noise.
    # This only affects training-time F1 logging and early stopping — inference still
    # uses per-class thresholds from tune_threshold.py.
    eval_threshold:      float = 0.35
    early_stop_patience: int   = 30         # v6: 30 epochs patience (was 10; 100-ep run needs room)
    aux_loss_weight:     float = 0.3

    # --- Aux loss warmup (Fix #33) ---
    # aux_loss_weight ramps from 0 → aux_loss_weight linearly over this many
    # epochs.  Set to 0 to disable warmup (always full aux weight).
    aux_loss_warmup_epochs: int = 3

    # --- Gradient accumulation (2026-05-12) ---
    gradient_accumulation_steps: int = 1

    # --- Stability ---
    grad_clip:  float = 1.0
    warmup_pct: float = 0.10

    # --- Speed: AMP ---
    use_amp: bool = True

    # --- Speed: DataLoader ---
    num_workers:         int  = 2
    persistent_workers:  bool = True

    # --- Loss function ---
    # v4/v5 used BCE. ASL (Ridnik et al. ICCV 2021) is recommended for v6:
    # gamma_neg=4 down-weights easy negatives (vast majority of 44K×10 cells),
    # freeing gradient budget for rare positives like DoS (377 train samples).
    loss_fn:        str   = "bce"
    focal_gamma:    float = 2.0
    focal_alpha:    float = 0.25
    # ASL hyperparameters (only used when loss_fn="asl")
    asl_gamma_neg:  float = 4.0   # focus exponent for negatives (hard negative mining)
    asl_gamma_pos:  float = 1.0   # focus exponent for positives (mild — less than gamma_neg)
    asl_clip:       float = 0.05  # probability margin; negatives with p<clip → zero gradient
    # RC3 fix (2026-05-16): label smoothing prevents extreme overconfidence.
    # Without smoothing the model pushes Reentrancy → 0.97 on safe contracts
    # with zero penalty.  ε=0.05 sets soft targets: positive→0.95, negative→0.05.
    label_smoothing: float = 0.05

    # --- pos_weight cap ---
    # Classes with >= pos_weight_min_samples training positives are NOT amplified.
    # Reentrancy has ~3500 train positives and is not actually rare; giving it a
    # 2.82× FN penalty + BCCC external-call co-occurrence is the primary driver of
    # the behavioral collapse seen in v5.2. Setting min_samples=3000 clamps
    # Reentrancy, GasException, and MishandledException to 1.0 while leaving
    # DoS (257), Timestamp (~1500), and minority classes at their sqrt-scaled weights.
    pos_weight_min_samples: int = 0    # 0 = disabled (all classes get sqrt weight)

    # --- Cache ---
    cache_path: str | None = "ml/data/cached_dataset_deduped.pkl"

    # --- Logging ---
    log_interval: int = 100

    # --- MLflow ---
    experiment_name: str = "sentinel-multilabel"
    run_name:        str = "multilabel-v5-fresh"

    # --- Device ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # --- Resume ---
    resume_from:           str | None = None
    resume_model_only:     bool       = True
    force_optimizer_reset: bool       = False

    # --- Autoresearch harness knobs ---
    smoke_subsample_fraction: float = 1.0
    use_weighted_sampler:     str   = "none"

    def __post_init__(self) -> None:
        # Phase 0-A4 (2026-05-14): Relaxed hard raise to conditional guard.
        # gnn_layers < 4 is a hard error — three-phase architecture needs at least 4 layers.
        # gnn_layers > 4 is experimental (second CONTROL_FLOW hop, v5.3+); warn but allow,
        # so smoke runs with gnn_layers=5 can be launched without config changes.
        if self.gnn_layers < 4:
            raise ValueError(
                f"gnn_layers={self.gnn_layers} is not supported — minimum is 4 "
                "(three-phase architecture requires layers 1+2 for Phase 1, "
                "layer 3 for Phase 2 CONTROL_FLOW, layer 4 for Phase 3 CONTAINS)."
            )
        if self.gnn_layers > 6:
            logger.warning(
                f"gnn_layers={self.gnn_layers} is non-standard. "
                "v6 uses gnn_layers=6 (2 per phase). "
                "Extra layers beyond 6 receive Phase 1 (structural) edge masking by default."
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps={self.gradient_accumulation_steps} must be >= 1."
            )


# ---------------------------------------------------------------------------
# pos_weight computation
# ---------------------------------------------------------------------------
def compute_pos_weight(
    label_csv:            str,
    train_indices:        np.ndarray,
    num_classes:          int,
    device:               str,
    pos_weight_min_samples: int = 0,
) -> torch.Tensor:
    import pandas as pd

    df = pd.read_csv(label_csv)
    class_cols   = CLASS_NAMES[:num_classes]
    label_matrix = df[class_cols].values
    train_labels = label_matrix[train_indices]

    N = len(train_labels)
    pos_counts = train_labels.sum(axis=0)

    pos_weight_vals = []
    for c, pos in enumerate(pos_counts):
        if pos == 0:
            logger.warning(
                f"Class '{CLASS_NAMES[c]}' (index {c}) has zero positives in training split. "
                "Using pos=1 to avoid division by zero."
            )
            pos = 1
        if pos_weight_min_samples > 0 and pos >= pos_weight_min_samples:
            # Well-represented class: no amplification needed. A 2.82× FN penalty
            # on Reentrancy (3500 train samples) combined with BCCC external-call
            # co-occurrence causes the model to associate any external call with
            # Reentrancy regardless of CEI pattern (observed behavioral failure v5.2).
            pos_weight_vals.append(1.0)
        else:
            raw_ratio = float(N - pos) / float(pos)
            # Cap at 20.0: without a ceiling, severely data-starved classes (e.g. DoS
            # with 257 samples → raw_ratio≈120) produce unchecked gradient spikes that
            # destabilise the loss scale for the entire batch.
            pos_weight_vals.append(min(float(raw_ratio ** 0.5), 20.0))

    logger.info(f"pos_weight sqrt-scaled (min_samples={pos_weight_min_samples}) — training split only:")
    for name, pw in zip(CLASS_NAMES[:num_classes], pos_weight_vals):
        capped = " [capped=1.0]" if pw == 1.0 and pos_weight_min_samples > 0 else ""
        logger.info(f"  {name:<32} {pw:.2f}{capped}")

    return torch.tensor(pos_weight_vals, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model:     SentinelModel,
    loader:    DataLoader,
    device:    str,
    threshold: float = 0.5,
    use_amp:   bool  = True,
) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_true  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            graphs, tokens, labels = batch

            graphs         = graphs.to(device)
            input_ids      = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            labels         = labels.to(device).float()

            with torch.amp.autocast(device, enabled=use_amp):
                logits = model(graphs, input_ids, attention_mask)

            probs = torch.sigmoid(logits.float())
            preds = (probs >= threshold).long()

            all_preds.append(preds.cpu().numpy())
            all_true.append(labels.long().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)

    f1_macro     = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    f1_micro     = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    hamming      = hamming_loss(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None,     zero_division=0)

    metrics = {"f1_macro": f1_macro, "f1_micro": f1_micro, "hamming": hamming}
    for i, name in enumerate(CLASS_NAMES[:y_true.shape[1]]):
        metrics[f"f1_{name}"] = float(f1_per_class[i])

    return metrics


# ---------------------------------------------------------------------------
# Training one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model:                       SentinelModel,
    loader:                      DataLoader,
    optimizer:                   AdamW,
    loss_fn:                     nn.Module,
    aux_loss_fn:                 nn.Module,
    scheduler:                   OneCycleLR,
    scaler:                      torch.amp.GradScaler,
    device:                      str,
    grad_clip:                   float,
    log_interval:                int,
    use_amp:                     bool,
    aux_loss_weight:             float = 0.3,
    gradient_accumulation_steps: int   = 1,
    label_smoothing:             float = 0.0,
) -> tuple[float, int, float]:
    """Returns (avg_loss, nan_batch_count, last_gnn_share)."""
    model.train()
    total_loss = 0.0

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    accum_steps = max(1, gradient_accumulation_steps)

    optimizer_step = 0
    _gnn_collapse_streak = 0
    nan_loss_count = 0
    last_gnn_share = 0.0

    # Running sums for per-interval averaged loss logging (reset every log_interval).
    _run_main  = 0.0
    _run_gnn_a = 0.0
    _run_tf_a  = 0.0
    _run_fus_a = 0.0
    _run_n     = 0

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    for batch_idx, batch in enumerate(pbar):
        graphs, tokens, labels = batch

        graphs         = graphs.to(device)
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        labels         = labels.to(device).float()

        if label_smoothing > 0.0:
            labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

        with torch.amp.autocast(device, enabled=use_amp):
            logits, aux = model(graphs, input_ids, attention_mask, return_aux=True)
            main_loss   = loss_fn(logits, labels)
            # aux_loss_fn has no pos_weight — pathway heads give supervision signal
            # without amplifying rare-class imbalance through struggling aux heads.
            loss_gnn_a  = aux_loss_fn(aux["gnn"],         labels)
            loss_tf_a   = aux_loss_fn(aux["transformer"], labels)
            loss_fus_a  = aux_loss_fn(aux["fused"],       labels)
            aux_loss    = loss_gnn_a + loss_tf_a + loss_fus_a
            # Divide by actual window size, not the fixed accum_steps.
            # When len(loader) % accum_steps != 0 the last window has fewer
            # micro-batches; dividing by accum_steps under-scales that gradient
            # by (actual / accum_steps). Using actual_window_size keeps gradients
            # correctly normalised across all windows including the tail.
            _window_start      = (batch_idx // accum_steps) * accum_steps
            _actual_window     = min(accum_steps, len(loader) - _window_start)
            loss = (main_loss + aux_loss_weight * aux_loss) / _actual_window

        # Accumulate per-eye loss for the upcoming log line.
        _run_main  += main_loss.item()
        _run_gnn_a += loss_gnn_a.item()
        _run_tf_a  += loss_tf_a.item()
        _run_fus_a += loss_fus_a.item()
        _run_n     += 1

        scaler.scale(loss).backward()

        is_last_batch = (batch_idx + 1 == len(loader))
        is_accum_step = ((batch_idx + 1) % accum_steps == 0) or is_last_batch

        if is_accum_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)

            # Fix #28: read grad norms after unscale_(), before zero_grad().
            optimizer_step += 1
            should_log = (optimizer_step % log_interval == 0)
            if should_log:
                gnn_norm   = _grad_norm(model.gnn_eye_proj)
                tf_norm    = _grad_norm(model.transformer_eye_proj)
                fused_norm = _grad_norm(model.fusion)
                _total_norm = (gnn_norm**2 + tf_norm**2 + fused_norm**2) ** 0.5
                _gnn_share  = gnn_norm / _total_norm if _total_norm > 1e-8 else 0.0
                last_gnn_share = _gnn_share

                n = max(1, _run_n)
                logger.info(
                    f"  Step {optimizer_step}/{(len(loader) + accum_steps - 1) // accum_steps} "
                    f"(batch {batch_idx+1}/{len(loader)}) | "
                    f"loss={_run_main/n:.4f} "
                    f"[eyes: gnn={_run_gnn_a/n:.4f} tf={_run_tf_a/n:.4f} fused={_run_fus_a/n:.4f}] | "
                    f"grad: gnn={gnn_norm:.3f} tf={tf_norm:.3f} fused={fused_norm:.3f} | "
                    f"GNN share={_gnn_share:.1%}"
                )
                # Reset running sums for next interval.
                _run_main = _run_gnn_a = _run_tf_a = _run_fus_a = 0.0
                _run_n = 0

                # Phase 2-C2: GNN collapse detection.
                if _gnn_share < 0.10:
                    _gnn_collapse_streak += 1
                    if _gnn_collapse_streak >= 3:
                        logger.warning(
                            f"  ⚠ GNN collapse: share={_gnn_share:.1%} for "
                            f"{_gnn_collapse_streak} consecutive intervals. "
                            "Consider aborting and increasing gnn_lr_multiplier."
                        )
                    else:
                        logger.info(f"  GNN share below 10%: {_gnn_share:.1%} [streak {_gnn_collapse_streak}/3]")
                else:
                    _gnn_collapse_streak = 0

                if device == "cuda":
                    vpct = _vram_pct()
                    if vpct > 0.90:
                        gc.collect()
                        torch.cuda.empty_cache()
                        logger.info(f"  VRAM cleanup triggered: {_vram_str()} (was {vpct:.1%} reserved)")

            _scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() == _scale_before:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_for_log = loss.item() * accum_steps
        if not torch.isfinite(loss).item():
            nan_loss_count += 1
        else:
            total_loss += loss_for_log
        pbar.set_postfix({"loss": f"{loss_for_log:.4f}", "nan": nan_loss_count})

    n_batches = len(loader)
    if n_batches == 0:
        logger.warning("Empty train loader — returning 0.0 loss")
        return 0.0, 0, 0.0

    if nan_loss_count > 0:
        logger.warning(
            f"NaN/Inf loss in {nan_loss_count}/{n_batches} batches this epoch. "
            f"GradScaler skipped those steps silently. "
            f"If > 5% of batches, reduce lr or inspect inputs."
        )

    return total_loss / max(1, n_batches - nan_loss_count), nan_loss_count, last_gnn_share


def _grad_norm(module: nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return total_sq ** 0.5


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------
def _build_weighted_sampler(
    dataset: DualPathDataset,
    label_csv_path: Path,
    mode: str,
) -> "torch.utils.data.WeightedRandomSampler":
    import pandas as pd
    from torch.utils.data import WeightedRandomSampler

    df = pd.read_csv(label_csv_path).set_index("md5_stem")
    weights: list[float] = []
    for md5 in dataset.paired_hashes:
        if md5 not in df.index:
            weights.append(1.0)
            continue
        row = df.loc[md5]
        if mode == "DoS-only":
            w = 39.0 if float(row.get("DenialOfService", 0)) == 1.0 else 1.0
        elif mode == "all-rare":
            n_pos = max(1, sum(float(row.get(cls, 0)) for cls in CLASS_NAMES))
            w = 1.0 / n_pos
        else:
            w = 1.0
        weights.append(w)

    logger.info(
        f"WeightedRandomSampler | mode={mode} | samples={len(weights)} | "
        f"weight_range=[{min(weights):.1f}, {max(weights):.1f}]"
    )
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(config: TrainConfig) -> dict:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    device = config.device

    # ── Environment + backend configuration (done here, not at module level) ──
    # Module-level mutation fires at import time and affects the entire process.
    # Doing it here is safe: train() is only called intentionally.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"]       = "1"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True

    # ── Per-run file log (append mode — never overwrites previous runs) ──────
    # trainer.py uses loguru, not stdlib logging — use logger.add() not
    # logging.FileHandler(). Append mode (mode="a") is safe for resume.
    # Each run gets its own file named after run_name (which must include a
    # date suffix so repeated runs never collide, e.g. v5.2-jk-20260514).
    _log_dir = Path("ml/logs")
    _log_dir.mkdir(parents=True, exist_ok=True)
    _run_log_path = _log_dir / f"{config.run_name}.log"
    logger.add(
        str(_run_log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}",
        mode="a",
        encoding="utf-8",
    )
    logger.info(f"Run log: {_run_log_path}  (append mode — safe to resume)")

    logger.info(f"Training on: {device} | classes: {config.num_classes}")

    if config.loss_fn not in _VALID_LOSS_FNS:
        raise ValueError(
            f"Unknown loss_fn='{config.loss_fn}'. Valid options: {sorted(_VALID_LOSS_FNS)}."
        )

    if config.use_amp and device == "cpu":
        logger.warning("use_amp=True but device=cpu — AMP disabled (CUDA only)")
        config.use_amp = False

    if not (0.0 < config.smoke_subsample_fraction <= 1.0):
        raise ValueError(f"smoke_subsample_fraction must be in (0, 1], got {config.smoke_subsample_fraction}")

    train_indices = np.load(Path(config.splits_dir) / "train_indices.npy")
    val_indices   = np.load(Path(config.splits_dir) / "val_indices.npy")

    if config.smoke_subsample_fraction < 1.0:
        original_n = len(train_indices)
        keep_n = max(1, int(original_n * config.smoke_subsample_fraction))
        rng = np.random.default_rng(42)
        train_indices = np.sort(rng.choice(train_indices, size=keep_n, replace=False))
        logger.info(f"Smoke subsample: {keep_n}/{original_n} train samples ({100*config.smoke_subsample_fraction:.0f}%)")

    label_csv_path = Path(config.label_csv) if config.label_csv else None
    if label_csv_path is not None and not label_csv_path.exists():
        raise FileNotFoundError(f"label_csv not found: {label_csv_path}.")
    cache_path = Path(config.cache_path) if config.cache_path else None

    logger.info("Creating training dataset...")
    train_dataset = DualPathDataset(
        graphs_dir=config.graphs_dir,
        tokens_dir=config.tokens_dir,
        indices=train_indices.tolist(),
        label_csv=label_csv_path,
        cache_path=cache_path,
    )
    logger.info(f"Train dataset cache loaded: {train_dataset.cached_data is not None}")

    logger.info("Creating validation dataset...")
    val_dataset = DualPathDataset(
        graphs_dir=config.graphs_dir,
        tokens_dir=config.tokens_dir,
        indices=val_indices.tolist(),
        label_csv=label_csv_path,
        cache_path=cache_path,
    )

    _use_workers = config.num_workers > 0
    _loader_kwargs: dict = dict(
        batch_size=config.batch_size,
        collate_fn=dual_path_collate_fn,
        num_workers=config.num_workers,
    )
    if _use_workers:
        _loader_kwargs.update(
            pin_memory=True,
            persistent_workers=config.persistent_workers,
            prefetch_factor=2,
        )

    _sampler = None
    if config.use_weighted_sampler != "none" and label_csv_path is not None:
        _sampler = _build_weighted_sampler(train_dataset, label_csv_path, config.use_weighted_sampler)

    if _sampler is not None:
        train_loader = DataLoader(train_dataset, sampler=_sampler, shuffle=False, **_loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **_loader_kwargs)

    accum_steps  = config.gradient_accumulation_steps
    effective_bs = config.batch_size * accum_steps
    logger.info(
        f"DataLoader — workers: {config.num_workers} | pin_memory: {_use_workers} | "
        f"AMP: {config.use_amp} | TF32: {torch.backends.cuda.matmul.allow_tf32} | "
        f"grad_accum: {accum_steps} | effective_batch: {effective_bs}"
    )

    if label_csv_path is not None:
        pos_weight = compute_pos_weight(str(label_csv_path), train_indices, config.num_classes, device,
                                        pos_weight_min_samples=config.pos_weight_min_samples)
    else:
        logger.info("Binary mode (label_csv=None) — pos_weight not computed.")
        pos_weight = None

    model = SentinelModel(
        num_classes=config.num_classes,
        fusion_output_dim=config.fusion_output_dim,
        dropout=config.fusion_dropout,
        gnn_hidden_dim=config.gnn_hidden_dim,
        gnn_num_layers=config.gnn_layers,
        gnn_heads=config.gnn_heads,
        gnn_dropout=config.gnn_dropout,
        use_edge_attr=config.use_edge_attr,
        gnn_edge_emb_dim=config.gnn_edge_emb_dim,
        gnn_use_jk=config.gnn_use_jk,         # Phase 1-A5
        gnn_jk_mode=config.gnn_jk_mode,       # Phase 1-A5
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
    ).to(device)

    start_epoch      = 1
    best_f1          = 0.0
    patience_counter = 0
    _ckpt_state: dict | None = None

    if config.resume_from:
        logger.info(f"Resuming from: {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=True)
        _ckpt_state = ckpt

        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"Invalid checkpoint format: {config.resume_from}")

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        lora_skipped  = [k for k in missing if "lora_" in k]
        other_missing = [k for k in missing if "lora_" not in k]
        if lora_skipped:
            logger.warning(f"Resume: {len(lora_skipped)} LoRA keys not loaded (lora_r mismatch).")
        if other_missing:
            raise RuntimeError(f"Resume: {len(other_missing)} non-LoRA keys missing: {other_missing[:5]}")

        if config.resume_model_only:
            logger.info(
                f"Model-only resume — weights from epoch {ckpt.get('epoch', 0)} "
                f"(F1={ckpt.get('best_f1', 0.0):.4f}). Counter reset to fresh start."
            )
        else:
            start_epoch      = ckpt.get("epoch", 0) + 1
            best_f1          = ckpt.get("best_f1", 0.0)
            patience_counter = ckpt.get("patience_counter", 0)

            _resume_state_path = Path(config.resume_from).with_suffix(".state.json")
            if _resume_state_path.exists():
                try:
                    _saved_state = json.loads(_resume_state_path.read_text())
                except json.JSONDecodeError as _e:
                    logger.warning(f"Corrupt .state.json ({_e}) — patience_counter from checkpoint.")
                    _saved_state = {}
                _state_epoch = _saved_state.get("epoch", 0)
                if _state_epoch >= start_epoch - 1:
                    patience_counter = _saved_state.get("patience_counter", patience_counter)
                    logger.info(f"patience_counter from state file: {patience_counter}/{config.early_stop_patience}")
            else:
                logger.warning("No .state.json sidecar — patience_counter from checkpoint value.")

            logger.info(
                f"Full resume from epoch {start_epoch-1} | "
                f"best_f1={best_f1:.4f} | patience={patience_counter}/{config.early_stop_patience}"
            )

        # Phase 1-A6: version gate — warn when resuming a pre-v5.2 checkpoint.
        # Pre-v5.2 checkpoints lack JK parameters (gnn.jk.*) and the new
        # REVERSE_CONTAINS embedding row — strict=False above silently ignores
        # these missing keys (JK starts from random init, which is fine for a
        # fresh training run but NOT for a true resume of an identical model).
        ckpt_version_str = ckpt.get("model_version", "v0.0")
        ckpt_ver  = _parse_version(ckpt_version_str)
        model_ver = _parse_version(MODEL_VERSION)
        if ckpt_ver < model_ver:
            logger.warning(
                f"Checkpoint model_version='{ckpt_version_str}' is older than "
                f"current MODEL_VERSION='{MODEL_VERSION}'. "
                f"New parameters (JK attention, REVERSE_CONTAINS embedding row) "
                f"will be randomly initialised — this is expected for a fresh "
                f"v5.2 training run, but NOT for a true resume of the same model."
            )

        ckpt_cfg = ckpt.get("config", {})
        ckpt_num_classes = ckpt_cfg.get("num_classes")
        if ckpt_num_classes is not None and ckpt_num_classes != config.num_classes:
            raise ValueError(f"Checkpoint num_classes={ckpt_num_classes} ≠ config {config.num_classes}.")
        ckpt_arch = ckpt_cfg.get("architecture")
        if ckpt_arch is not None and ckpt_arch != ARCHITECTURE:
            raise ValueError(f"Checkpoint architecture='{ckpt_arch}' ≠ expected '{ARCHITECTURE}'.")

        ckpt_batch_size = ckpt_cfg.get("batch_size")
        if ckpt_batch_size is not None and ckpt_batch_size != config.batch_size and not config.resume_model_only:
            if config.force_optimizer_reset:
                logger.warning(f"Batch size changed {ckpt_batch_size}→{config.batch_size}. Optimizer will be reset.")
            else:
                logger.warning(f"BATCH SIZE MISMATCH (Fix #12): ckpt={ckpt_batch_size} current={config.batch_size}.")

    # Auxiliary heads always use plain BCE (no pos_weight, no focal reweighting).
    # pos_weight in aux losses amplifies rare-class gradients through the already-
    # struggling GNN/TF/fused heads, exacerbating instability.  The main loss
    # carries the class-balance signal; aux heads provide pathway-level supervision.
    aux_loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    if config.loss_fn == "focal":
        _focal = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
        class _FocalFromLogits(nn.Module):
            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return _focal(torch.sigmoid(logits.float()), targets)
        loss_fn: nn.Module = _FocalFromLogits()
        logger.info(f"Loss: FocalLoss(gamma={config.focal_gamma}, alpha={config.focal_alpha})")
    elif config.loss_fn == "asl":
        loss_fn = AsymmetricLoss(
            gamma_neg=config.asl_gamma_neg,
            gamma_pos=config.asl_gamma_pos,
            clip=config.asl_clip,
        )
        logger.info(
            f"Loss: AsymmetricLoss(gamma_neg={config.asl_gamma_neg}, "
            f"gamma_pos={config.asl_gamma_pos}, clip={config.asl_clip})"
        )
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("Loss: BCEWithLogitsLoss with class-balanced pos_weight")
        if pos_weight is not None and config.resume_from and not config.resume_model_only:
            logger.warning("Fix #13: pos_weight recomputed from current training split.")
    logger.info("Aux loss: BCEWithLogitsLoss without pos_weight (pathway supervision only)")

    # Phase 2-B1 (2026-05-14): separate LR groups to counteract GNN gradient collapse.
    # GNN share dropped to ~10% by epoch 8 in v5.1-fix28; boosting its LR × 2.5
    # gives it a larger update relative to the LoRA adapter without changing the
    # global schedule.  LoRA at × 0.5 prevents CodeBERT forgetting.
    #
    # Groups (in order — OneCycleLR max_lr list must match this order):
    #   [0] GNN        (model.gnn.*)      lr * gnn_lr_multiplier
    #   [1] LoRA       (any "lora_" key)  lr * lora_lr_multiplier
    #   [2] Other      (everything else)  lr
    #
    # Parameters are assigned to exactly one group; iteration order is stable
    # (dict preserves insertion order in Python 3.7+, same here). seen_ids
    # prevents double-counting if a parameter is reachable via multiple paths.
    _gnn_params:    list = []
    _lora_params:   list = []
    _fusion_params: list = []
    _other_params:  list = []
    _seen_param_ids: set = set()
    for _pname, _p in model.named_parameters():
        if not _p.requires_grad or id(_p) in _seen_param_ids:
            continue
        _seen_param_ids.add(id(_p))
        if _pname.startswith("gnn.") or _pname.startswith("gnn_eye_proj."):
            _gnn_params.append(_p)
        elif "lora_" in _pname:
            _lora_params.append(_p)
        elif (
            _pname.startswith("fusion.")
            or _pname.startswith("transformer_eye_proj.")
            or _pname.startswith("classifier.")
            or _pname.startswith("aux_")
        ):
            # RC1 fix: fusion + classifier at reduced LR to prevent CodeBERT's
            # Reentrancy bias from overwhelming the GNN signal via high-gradient
            # cross-attention (821K params running at full LR produced 4-5× the
            # GNN gradient norm throughout all of r3).
            _fusion_params.append(_p)
        else:
            _other_params.append(_p)

    _gnn_lr    = config.lr * config.gnn_lr_multiplier
    _lora_lr   = config.lr * config.lora_lr_multiplier
    _fusion_lr = config.lr * config.fusion_lr_multiplier
    _other_lr  = config.lr

    # Only include non-empty groups — empty param groups cause OneCycleLR to
    # misalign its max_lr list.
    _param_groups = []
    _max_lrs      = []
    if _gnn_params:
        _param_groups.append({"params": _gnn_params,    "lr": _gnn_lr})
        _max_lrs.append(_gnn_lr)
    if _lora_params:
        # LoRA matrices are low-rank adaptation updates; L2 weight decay competes
        # directly with the adaptation signal. Standard PEFT practice is weight_decay=0
        # for LoRA parameters while keeping decay on the rest of the network.
        _param_groups.append({"params": _lora_params,  "lr": _lora_lr,   "weight_decay": 0.0})
        _max_lrs.append(_lora_lr)
    if _fusion_params:
        _param_groups.append({"params": _fusion_params, "lr": _fusion_lr})
        _max_lrs.append(_fusion_lr)
    if _other_params:
        _param_groups.append({"params": _other_params,  "lr": _other_lr})
        _max_lrs.append(_other_lr)

    logger.info(
        f"Optimizer param groups: "
        f"GNN={len(_gnn_params)} params (lr×{config.gnn_lr_multiplier}) | "
        f"LoRA={len(_lora_params)} params (lr×{config.lora_lr_multiplier}) | "
        f"Fusion={len(_fusion_params)} params (lr×{config.fusion_lr_multiplier}) | "
        f"Other={len(_other_params)} params (lr×1.0)"
    )
    optimizer = AdamW(_param_groups, weight_decay=config.weight_decay)

    remaining_epochs = config.epochs - start_epoch + 1
    if remaining_epochs <= 0:
        logger.warning(f"start_epoch={start_epoch} >= config.epochs={config.epochs}: nothing to train.")
        return {
            "best_f1_macro": best_f1,
            "final_epoch": start_epoch - 1,
            "early_stopped": False,
            "checkpoint_path": str(Path(config.checkpoint_dir) / config.checkpoint_name),
        }

    steps_per_epoch = (len(train_loader) + accum_steps - 1) // accum_steps

    # ── Smoke-run auto log_interval ───────────────────────────────────────────
    # Default log_interval=100 assumes ~1,950 optimizer steps/epoch (full 60ep).
    # With 10% smoke data + accum=4, there are only ~49 steps/epoch → step 100
    # is never reached and the GNN-share gate cannot be observed.
    # Auto-lower to steps_per_epoch // 4 (capped at config.log_interval) so at
    # least 4 log points appear per epoch in any smoke configuration.
    _auto_log_interval = config.log_interval
    if steps_per_epoch < config.log_interval:
        _auto_log_interval = max(5, steps_per_epoch // 4)
        logger.info(
            f"Auto log_interval: {config.log_interval}→{_auto_log_interval} "
            f"(steps_per_epoch={steps_per_epoch} < log_interval={config.log_interval})"
        )

    # ── Resume command ────────────────────────────────────────────────────────
    # Logged at startup so the operator can copy-paste it without digging
    # through config files. The run-name gets a -resumed suffix to keep
    # MLflow runs separate; checkpoint_name is derived from that new name.
    _checkpoint_path = Path(config.checkpoint_dir) / config.checkpoint_name
    _resume_cmd = (
        f"TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \\\n"
        f"    --resume {_checkpoint_path} \\\n"
        f"    --run-name {config.run_name}-resumed \\\n"
        f"    --experiment-name {config.experiment_name} \\\n"
        f"    --epochs {config.epochs} \\\n"
        f"    --gradient-accumulation-steps {accum_steps}"
    )
    logger.info(f"Resume command (if needed):\n  {_resume_cmd}")

    # ── Fix #32: Scheduler must be created with the FULL epoch count so that
    #    total_steps matches the value from the original run.  Previously this
    #    used `remaining_epochs`, causing a total_steps mismatch on resume and
    #    the scheduler was silently discarded.
    #    On a fresh run (start_epoch=1) this is identical to the old behaviour
    #    because remaining_epochs == config.epochs.  On resume, the scheduler
    #    is created with the original total_steps, then its state_dict is
    #    loaded (which includes the correct last_epoch / step counter), so it
    #    picks up exactly where the original run left off.
    scheduler = OneCycleLR(
        optimizer,
        max_lr=_max_lrs,          # per-group max LR (list matches param_groups order)
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.warmup_pct,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    if _ckpt_state is not None and not config.resume_model_only:
        ckpt = _ckpt_state
        _skip_optimizer = config.force_optimizer_reset
        _optimizer_restored = False
        if not _skip_optimizer and "optimizer" in ckpt:
            # Phase 2-B1: guard against param-group count mismatch.
            # Pre-v5.2 checkpoints have 1 group; v5.2 has up to 3 (GNN/LoRA/other).
            # Loading mismatched optimizer state silently corrupts momentum buffers.
            ckpt_n_groups   = len(ckpt["optimizer"].get("param_groups", []))
            current_n_groups = len(optimizer.param_groups)
            if ckpt_n_groups != current_n_groups:
                logger.warning(
                    f"Optimizer param_groups count mismatch: "
                    f"checkpoint has {ckpt_n_groups}, current optimizer has {current_n_groups}. "
                    f"Skipping optimizer restore — optimizer starts fresh (Phase 2-B1 change)."
                )
            else:
                optimizer.load_state_dict(ckpt["optimizer"])
                logger.info("Optimizer state restored.")
                _optimizer_restored = True
        elif _skip_optimizer:
            logger.info("Optimizer state skipped — force_optimizer_reset=True.")
        else:
            logger.warning("Full resume requested but no 'optimizer' key in checkpoint. Fresh AdamW.")

        # ── GradScaler state restoration ──
        # Restoring the loss scale prevents the default scale=65536 from triggering
        # a NaN calibration wave (several optimizer.step() skips) at the start of
        # resumed training. Only restored on full resume, not model-only.
        if "scaler" in ckpt and _optimizer_restored:
            scaler.load_state_dict(ckpt["scaler"])
            logger.info(f"GradScaler state restored (scale={scaler.get_scale():.0f}).")

        # ── Fix #32 (continued): scheduler state restoration ──
        # Compare checkpoint total_steps against the full-run total_steps
        # (config.epochs × steps_per_epoch).  If they match, restore state;
        # otherwise the scheduler was created with the correct full-run
        # horizon but starts from step 0 — which is still better than the
        # old behaviour where the scheduler was discarded entirely.
        # Guard: only restore scheduler if optimizer was also restored — a
        # group-count mismatch skips the optimizer but left _skip_optimizer=False,
        # which previously caused base_lrs list-length mismatch on load.
        if "scheduler" in ckpt and _optimizer_restored:
            full_total_steps = config.epochs * steps_per_epoch
            if "total_steps" not in ckpt["scheduler"]:
                logger.warning("Scheduler state skipped — no 'total_steps' in checkpoint.")
            else:
                ckpt_total_steps = ckpt["scheduler"]["total_steps"]
                if ckpt_total_steps == full_total_steps:
                    scheduler.load_state_dict(ckpt["scheduler"])
                    logger.info(
                        f"Scheduler state restored (total_steps={ckpt_total_steps})."
                    )
                else:
                    logger.warning(
                        f"Scheduler total_steps mismatch "
                        f"(ckpt={ckpt_total_steps} vs current={full_total_steps}). "
                        "Scheduler starts fresh from step 0 — LR schedule may differ "
                        "from original run. To avoid this, resume with the same "
                        "batch_size and gradient_accumulation_steps."
                    )
        elif _skip_optimizer:
            logger.info("Scheduler state skipped — force_optimizer_reset=True.")
    elif _ckpt_state is not None:
        logger.info("Resumed model weights only (fresh optimizer/scheduler).")

    # ── Fix #34: log VRAM at training start ──
    if device == "cuda":
        logger.info(f"VRAM at training start: {_vram_str()}")

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        params = {
            "num_classes":                 config.num_classes,
            "epochs":                      config.epochs,
            "remaining_epochs":            remaining_epochs,
            "batch_size":                  config.batch_size,
            "gradient_accumulation_steps": accum_steps,
            "effective_batch_size":        effective_bs,
            "lr":                          config.lr,
            "weight_decay":                config.weight_decay,
            "threshold":                   config.threshold,
            "eval_threshold":              config.eval_threshold,
            "grad_clip":                   config.grad_clip,
            "warmup_pct":                  config.warmup_pct,
            "num_workers":                 config.num_workers,
            "use_amp":                     config.use_amp,
            "loss_fn":                     config.loss_fn,
            "focal_gamma":                 config.focal_gamma,
            "focal_alpha":                 config.focal_alpha,
            "device":                      device,
            "architecture":                ARCHITECTURE,
            "label_csv":                   config.label_csv,
            "resume_from":                 config.resume_from or "none",
            "resume_model_only":           config.resume_model_only,
            "force_optimizer_reset":       config.force_optimizer_reset,
            "early_stop_patience":         config.early_stop_patience,
            "gnn_hidden_dim":              config.gnn_hidden_dim,
            "gnn_layers":                  config.gnn_layers,
            "gnn_heads":                   config.gnn_heads,
            "gnn_dropout":                 config.gnn_dropout,
            "use_edge_attr":               config.use_edge_attr,
            "gnn_edge_emb_dim":            config.gnn_edge_emb_dim,
            "aux_loss_weight":             config.aux_loss_weight,
            "aux_loss_warmup_epochs":      config.aux_loss_warmup_epochs,
            "lora_r":                      config.lora_r,
            "lora_alpha":                  config.lora_alpha,
            "lora_dropout":                config.lora_dropout,
            "lora_target_modules":         ",".join(config.lora_target_modules),
            "fusion_output_dim":           config.fusion_output_dim,
            "fusion_lr_multiplier":        config.fusion_lr_multiplier,
            "label_smoothing":             config.label_smoothing,
            "pos_weight_min_samples":      config.pos_weight_min_samples,
        }
        if pos_weight is not None:
            for name, pw in zip(CLASS_NAMES[:config.num_classes], pos_weight.cpu().tolist()):
                params[f"pos_weight_{name}"] = round(pw, 3)
        mlflow.log_params(params)

        checkpoint_path = _checkpoint_path   # already computed above
        final_epoch = start_epoch - 1

        for epoch in range(start_epoch, config.epochs + 1):
            final_epoch = epoch
            logger.info(f"\n{'='*60}\nEpoch {epoch}/{config.epochs}\n{'='*60}")

            # ── Fix #33: aux loss warmup ──
            # Ramp aux_loss_weight from 0 → config.aux_loss_weight over the
            # first aux_loss_warmup_epochs epochs.  This prevents the three
            # auxiliary classification heads from dominating early gradients
            # (observed: aux loss 2-4× main loss at epoch 1, causing the main
            # classifier to learn slowly and rare classes to stay at F1=0).
            if config.aux_loss_warmup_epochs > 0 and epoch <= config.aux_loss_warmup_epochs:
                # (epoch-1) so the ramp starts at 0 on epoch 1, not at 1/warmup_epochs.
                # Previous formula started at 1/3 of target on epoch 1, never at 0.
                warmup_frac = (epoch - 1) / config.aux_loss_warmup_epochs
                effective_aux_weight = config.aux_loss_weight * warmup_frac
                logger.info(
                    f"  Aux warmup: epoch {epoch}/{config.aux_loss_warmup_epochs} "
                    f"→ aux_weight={effective_aux_weight:.4f} "
                    f"(target={config.aux_loss_weight:.4f})"
                )
            else:
                effective_aux_weight = config.aux_loss_weight

            train_loss, nan_batch_count, last_gnn_share = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                aux_loss_fn=aux_loss_fn,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                grad_clip=config.grad_clip,
                log_interval=_auto_log_interval,
                use_amp=config.use_amp,
                aux_loss_weight=effective_aux_weight,
                gradient_accumulation_steps=accum_steps,
                label_smoothing=config.label_smoothing,
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                threshold=config.eval_threshold,   # training-time threshold (lower = stable signal)
                use_amp=config.use_amp,
            )

            # Fix #27: release CUDA caching allocator free-blocks between epochs.
            gc.collect()
            torch.cuda.empty_cache()

            mlflow.log_metric("train_loss",       train_loss,       step=epoch)
            mlflow.log_metric("nan_batch_count",  nan_batch_count,  step=epoch)  # Phase 2-B3
            mlflow.log_metric("gnn_grad_share",   last_gnn_share,   step=epoch)

            # Phase 2-C1 (2026-05-14): JK attention weight logging.
            # Reads cached per-phase mean weights from the last training batch.
            # Metric names: jk_phase1_weight, jk_phase2_weight, jk_phase3_weight.
            # Gate: all three should be > 5% after smoke run (v5.2 success criterion).
            if config.gnn_use_jk and hasattr(model.gnn, "jk") and model.gnn.jk is not None:
                _jk_cache = getattr(model.gnn.jk, "last_weights", None)
                if _jk_cache is not None:
                    _jk_w = _jk_cache.cpu().tolist()   # [3] mean weights
                    for _pi, _w in enumerate(_jk_w, start=1):
                        mlflow.log_metric(f"jk_phase{_pi}_weight", _w, step=epoch)
                    logger.info(
                        f"  JK attention weights — "
                        f"Phase1={_jk_w[0]:.3f} Phase2={_jk_w[1]:.3f} Phase3={_jk_w[2]:.3f}"
                    )
                    # Phase 2-C3: phase dominance alert (> 80% = JK learned to ignore phases).
                    _max_w = max(_jk_w)
                    if _max_w > 0.80:
                        _dominant = _jk_w.index(_max_w) + 1
                        logger.warning(
                            f"  ⚠ JK phase dominance: Phase {_dominant} has "
                            f"{_max_w:.1%} attention weight. "
                            "Other phases are underutilised — consider checking LayerNorm or LR."
                        )

            mlflow.log_metric("val_f1_macro",  val_metrics["f1_macro"],  step=epoch)
            mlflow.log_metric("val_f1_micro",  val_metrics["f1_micro"],  step=epoch)
            mlflow.log_metric("val_hamming",   val_metrics["hamming"],   step=epoch)
            mlflow.log_metric("aux_loss_weight_effective", effective_aux_weight, step=epoch)
            for name in CLASS_NAMES[:config.num_classes]:
                mlflow.log_metric(f"val_f1_{name}", val_metrics[f"f1_{name}"], step=epoch)

            class_f1s = [(n, val_metrics[f"f1_{n}"]) for n in CLASS_NAMES[:config.num_classes]]
            class_f1s.sort(key=lambda x: x[1], reverse=True)
            top3    = " | ".join(f"{n}={v:.3f}" for n, v in class_f1s[:3])
            bottom3 = " | ".join(f"{n}={v:.3f}" for n, v in class_f1s[-3:])

            # ── Fix #34: log VRAM every epoch ──
            vram_info = f" | VRAM: {_vram_str()}" if device == "cuda" else ""

            logger.info(
                f"Epoch {epoch:>2}/{config.epochs} | "
                f"Loss={train_loss:.4f} | F1-macro={val_metrics['f1_macro']:.4f} | "
                f"Hamming={val_metrics['hamming']:.4f}{vram_info}\n"
                f"  Top3:    {top3}\n"
                f"  Bottom3: {bottom3}"
            )

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                patience_counter = 0
                _tmp_path = checkpoint_path.with_suffix(".tmp")
                torch.save(
                    {
                        "model":            model.state_dict(),
                        "optimizer":        optimizer.state_dict(),
                        "scheduler":        scheduler.state_dict(),
                        # GradScaler state: saves the current loss scale so resumed
                        # training doesn't start with the default scale=65536 and
                        # trigger a NaN calibration wave on the first few steps.
                        "scaler":           scaler.state_dict(),
                        "epoch":            epoch,
                        "best_f1":          best_f1,
                        "patience_counter": patience_counter,
                        # Phase 1-A6: model_version enables version-aware resume.
                        # _parse_version() converts this to a comparable tuple.
                        "model_version":    MODEL_VERSION,
                        "config": {
                            **dataclasses.asdict(config),
                            "num_classes":  config.num_classes,
                            "class_names":  CLASS_NAMES[:config.num_classes],
                            "architecture": ARCHITECTURE,
                        },
                    },
                    _tmp_path,
                )
                _tmp_path.replace(checkpoint_path)  # atomic on POSIX
                logger.info(f"  ★ New best F1-macro: {best_f1:.4f} — checkpoint saved")
            else:
                patience_counter += 1
                logger.info(f"  No improvement for {patience_counter}/{config.early_stop_patience} epochs")
                if patience_counter >= config.early_stop_patience:
                    logger.info(f"  Early stopping after {epoch} epochs (best F1={best_f1:.4f})")
                    break

            _state_path = checkpoint_path.with_suffix(".state.json")
            _state_path.write_text(
                json.dumps({"epoch": epoch, "patience_counter": patience_counter, "best_f1": best_f1})
            )

        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path))

        logger.info(f"\n✅ Training complete. Best val F1-macro: {best_f1:.4f}")

    return {
        "best_f1_macro":  best_f1,
        "final_epoch":    final_epoch,
        "early_stopped":  patience_counter >= config.early_stop_patience,
        "checkpoint_path": str(checkpoint_path),
    }


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
