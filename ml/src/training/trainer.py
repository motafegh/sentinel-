"""
trainer.py — SENTINEL Training Loop (Cross-Attention + LoRA Upgrade)
FINAL STANDALONE VERSION – with tqdm progress bars, safe resume, offline mode,
and EARLY STOPPING.

SPEED OPTIMISATIONS APPLIED (vs original):
────────────────────────────────────────────────────────────────────────────
1. AMP (Automatic Mixed Precision)
     torch.amp.autocast("cuda") wraps every forward pass.
     GradScaler handles the scaled backward pass and optimizer step.
     On Ampere GPUs (RTX 30xx/40xx/A-series) this alone is 2–3× faster
     because FP16/BF16 tensor-core matmuls replace FP32 everywhere
     except where precision matters (loss, normalisations — PyTorch handles
     this automatically). BF16 preferred over FP16 on Ampere; no inf/nan risk.

2. TF32 matmuls enabled
     Two flags tell cuBLAS and cuDNN to use TF32 units on Ampere+.
     Free ~1.5× speedup with negligible precision loss (10 mantissa bits
     vs 23 for FP32, but more than enough for gradients).
     Has zero effect on pre-Ampere GPUs (silently ignored).

3. num_workers raised to 2 (was 0)
     With num_workers=0 the GPU stalls after every batch while the main
     thread collates the next one. PyG's Batch.from_data_list() is real
     CPU work — it needs a background thread.
     With num_workers=2 + persistent_workers=True, two worker processes
     pre-fetch and collate while the GPU trains, hiding almost all CPU
     overhead. The RAM cache is safe to use with workers because each
     worker gets a read-only fork of the dict (no file I/O race).
     persistent_workers=True avoids re-spawning workers every epoch.

4. pin_memory now consistent with num_workers
     pin_memory=True is only meaningful when workers exist (they do the
     pinning asynchronously). With num_workers=0 it was a no-op/overhead.
     Now tied to num_workers > 0 automatically.

5. zero_grad(set_to_none=True)
     Frees gradient tensors entirely instead of writing zeros.
     Saves a full-model memory write per step — small but free.

6. MLflow log_artifact moved outside the epoch loop
     Previously called on every best checkpoint, which copies the file
     synchronously. Now logged once at the end of training (the last-saved
     best checkpoint). If you want per-epoch artifact logging re-enable it.

7. EARLY STOPPING
     Monitors validation F1‑macro, stops training when no improvement
     for `early_stop_patience` consecutive epochs. Restores best model
     automatically (implicitly because the best checkpoint is saved).

AUDIT FIXES (2026-05-01):
────────────────────────────────────────────────────────────────────────────
Fix #4 — Unknown loss_fn now raises ValueError (was silent BCE fallthrough).
Fix #7 — clip_grad_norm_ scans only trainable params (not frozen CodeBERT).
Fix #8 — OneCycleLR uses remaining_epochs=config.epochs-start_epoch+1 on resume.
Fix #2 — FocalLoss sigmoid called on logits.float() to prevent BF16 underflow.
Fix #9 — Resume arch check uses ARCHITECTURE constant, not missing config.architecture.
Fix #10 — Full resume (--no-resume-model-only) skips scheduler state when total_steps
          mismatch is detected (epoch extension). load_state_dict restores total_steps
          from the checkpoint, silently overwriting the new remaining_epochs schedule
          and causing OneCycleLR overflow after the original run's remaining steps.

AUDIT FIXES (2026-05-04, batch 1):
────────────────────────────────────────────────────────────────────────────
Fix #11 — patience_counter was never saved in the checkpoint dict and never
           restored on resume. Every resume silently reset early-stopping to
           zero, allowing a stagnating model to train far longer than
           `early_stop_patience` would permit across the full run. Now:
             - checkpoint dict includes `patience_counter`
             - resume block reads `ckpt.get("patience_counter", 0)`
             - Both the model-only and full-resume paths restore it.

Fix #12 — No guard when batch_size differs between checkpoint and resume.
           When batch_size changes (e.g. 16 → 32), steps_per_epoch halves.
           On full resume (--no-resume-model-only) the Adam first/second-
           moment vectors (m, v) are calibrated to gradient noise from the
           OLD batch size. Loading that stale optimizer state into a
           training loop with a different batch size causes loss spikes
           and declining F1 in the first several epochs while Adam
           re-calibrates its moments. Now:
             - ckpt `config.batch_size` is compared to `config.batch_size`
               on resume.
             - If they differ AND the resume is a FULL resume (optimizer
               state would be loaded), a prominent WARNING is logged
               explaining the stale-moment risk.
             - If `config.force_optimizer_reset=True` (or
               `--resume-reset-optimizer` CLI flag), the optimizer state
               is skipped even in full-resume mode, giving a clean AdamW
               calibrated to the new batch size while keeping model weights.
             - The warning recommends either:
               (a) Using model-only resume (omit --no-resume-model-only)
               (b) Using --resume-reset-optimizer
               (c) Keeping batch_size identical to the checkpoint.

Fix #13 — pos_weight recomputed fresh on every resume but optimizer state
           warning is now guarded so it only fires in multi-label mode where
           pos_weight is actually computed (binary mode sets pos_weight=None).

AUDIT FIXES (2026-05-04, batch 2):
────────────────────────────────────────────────────────────────────────────
Fix #14 — Binary mode crash: str(None) → pd.read_csv("None") → FileNotFoundError.
           compute_pos_weight is now skipped when label_csv=None; pos_weight=None
           is passed to BCEWithLogitsLoss (unweighted binary training).
           MLflow pos_weight param logging is guarded by `if pos_weight is not None`.

Fix #15 — Double checkpoint load: torch.load() was called twice for full resume
           (once for model weights, once for optimizer/scheduler). The loaded dict
           is now cached in _ckpt_state and reused, eliminating redundant I/O.

Fix #16 — Hardcoded magic number 47966 in batch-size mismatch warning replaced
           with len(train_dataset) so the steps_per_epoch estimate stays correct
           as the dataset grows.

Fix #17 — TrainConfig.batch_size default was 16 while train.py --batch-size
           defaulted to 32. Both now default to 16 (correct for RTX 3070 8GB VRAM).

Fix #18 — ZeroDivisionError in train_one_epoch when the DataLoader is empty
           (e.g. drop_last=True with very small dataset). Now returns 0.0 with
           a warning instead of crashing.

Fix #19 — prefetch_factor=None passed to DataLoader when num_workers=0. PyTorch
           accepts it today but documents it as only meaningful with workers.
           DataLoader kwargs are now built conditionally: prefetch_factor, pin_memory,
           and persistent_workers are only included when num_workers > 0.

Fix #20 — Missing CLI arguments for grad_clip, warmup_pct, loss_fn, use_amp,
           early_stop_patience, num_workers, and log_interval. All seven are now
           exposed via train.py and wired into TrainConfig.

Fix #21 — train() returned None, making programmatic use (testing, sweep scripts)
           require MLflow inspection to retrieve results. Now returns a dict with
           best_f1_macro, final_epoch, early_stopped, and checkpoint_path.

Fix #22 — FocalLoss gamma/alpha were hardcoded to 2.0/0.25 with no way to change
           them without editing source. Added focal_gamma and focal_alpha fields to
           TrainConfig (defaults preserve existing behaviour). Also added
           --focal-gamma and --focal-alpha CLI flags to train.py.

AUDIT FIXES (2026-05-04, batch 3):
────────────────────────────────────────────────────────────────────────────
Fix #23 — patience_counter always read as 0 when resuming from the best
           checkpoint (it is saved as 0 whenever a new best is found). Any
           non-improvement epochs before an interruption were lost, allowing
           the model to train far beyond early_stop_patience across resumes.
           Fix: write a tiny JSON sidecar ({checkpoint}.state.json) after
           every epoch containing epoch, patience_counter, and best_f1. On
           resume the sidecar overrides the checkpoint value; if the sidecar
           is absent a warning is logged explaining the limitation.

Fix #24 — No warning when --no-resume-model-only is used but the checkpoint
           contains no 'optimizer' key. The code silently initialised a fresh
           optimizer with no indication to the user. Added an else branch that
           logs a WARNING in this case.

Fix #25 — Scheduler state restore used ckpt["scheduler"].get("total_steps"),
           which returns None if the key is absent (older PyTorch checkpoints).
           None == new_total_steps is always False, so the fallthrough warning
           incorrectly attributed the skip to an epoch-extension mismatch.
           Now the key's existence is checked explicitly; if absent, a clear
           "missing total_steps" warning is emitted instead.

AUDIT FIXES (2026-05-04, external review):
────────────────────────────────────────────────────────────────────────────
Fix #9 (MLflow) — focal_gamma and focal_alpha were set in TrainConfig (Fix #22)
           but were never included in the MLflow params dict. Every Focal Loss
           sweep run was indistinguishable in the UI: impossible to sort/filter
           by gamma or alpha. Both params are now logged unconditionally.
           When loss_fn='bce' they appear but are irrelevant; when loss_fn='focal'
           they are the primary hyperparameters under sweep.
"""

from __future__ import annotations

import json
import os
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

# ---------------------------------------------------------------------------
# Force offline mode for HuggingFace models (no internet attempts)
# ---------------------------------------------------------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

# ---------------------------------------------------------------------------
# Optimisation #2: TF32 matmuls
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Logging setup
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

# ---------------------------------------------------------------------------
# Architecture constant — single source of truth (Fix #9)
# TrainConfig has no 'architecture' field; using config.architecture caused
# AttributeError on every resume. This constant is used wherever the
# architecture name needs to be recorded (checkpoint, MLflow, resume check).
# ---------------------------------------------------------------------------
ARCHITECTURE = "three_eye_v5"

# ---------------------------------------------------------------------------
# Valid loss function names — Audit fix #4
# Centralised here so the ValueError message and dispatch logic stay in sync.
# ---------------------------------------------------------------------------
_VALID_LOSS_FNS: frozenset[str] = frozenset({"bce", "focal"})


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

    # --- GNN architecture (v5) ---
    gnn_hidden_dim:   int   = 128
    gnn_layers:       int   = 4    # validated in __post_init__; only 4 supported in v5.0
    gnn_heads:        int   = 8
    gnn_dropout:      float = 0.2
    use_edge_attr:    bool  = True
    gnn_edge_emb_dim: int   = 32

    # --- LoRA architecture (v5) ---
    lora_r:               int        = 16
    lora_alpha:           int        = 32
    lora_dropout:         float      = 0.1
    lora_target_modules:  list[str]  = field(default_factory=lambda: ["query", "value"])

    # --- Label source ---
    label_csv: str = "ml/data/processed/multilabel_index_deduped.csv"

    # --- Training ---
    epochs:              int   = 60
    batch_size:          int   = 16
    lr:                  float = 2e-4
    weight_decay:        float = 1e-2
    threshold:           float = 0.5
    early_stop_patience: int   = 10
    aux_loss_weight:     float = 0.3   # λ in: main + λ*(aux_gnn + aux_tf + aux_fused)

    # --- Stability ---
    grad_clip:  float = 1.0
    warmup_pct: float = 0.10

    # --- Speed: AMP ---
    use_amp: bool = True

    # --- Speed: DataLoader ---
    num_workers:         int  = 2
    persistent_workers:  bool = True

    # --- Loss function ---
    # Valid values: "bce" | "focal"
    loss_fn:     str   = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # --- Cache ---
    cache_path: str | None = "ml/data/cached_dataset.pkl"

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
        """Validate config at startup — fires before data loading or GPU allocation."""
        if self.gnn_layers != 4:
            raise ValueError(
                f"gnn_layers={self.gnn_layers} is not supported in v5.0. "
                "Only gnn_layers=4 is implemented (3-phase: 2 structural + CONTAINS, "
                "1 CONTROL_FLOW directed, 1 reverse-CONTAINS). "
                "v5.1 target: gnn_layers=5 for 2 CONTROL_FLOW hops."
            )


# ---------------------------------------------------------------------------
# pos_weight computation
# ---------------------------------------------------------------------------
def compute_pos_weight(
    label_csv:     str,
    train_indices: np.ndarray,
    num_classes:   int,
    device:        str,
) -> torch.Tensor:
    """
    Compute per-class pos_weight for BCEWithLogitsLoss using sqrt scaling.

    Formula: pos_weight[i] = sqrt((N - n_pos[i]) / n_pos[i])

    Rationale for sqrt vs raw ratio:
      Raw ratio for DoS (~437 samples in 68K): (68K-437)/437 ≈ 155.
      Raw ratio for IntegerUO (~5343 samples): (68K-5343)/5343 ≈ 11.7.
      A 13× ratio between classes causes training instability.

      sqrt scaling: sqrt(155) ≈ 12.4 vs sqrt(11.7) ≈ 3.4 — a 3.6× ratio.
      This preserves ordering (DoS gets proportionally more weight) without
      extreme values that destabilise training. No arbitrary cap needed.

    IMPORTANT: recompute after data augmentation — the distribution changes.
    """
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
                "sqrt pos_weight undefined (division by zero) — using sqrt((N-1)/1) ≈ sqrt(N)."
            )
            pos = 1  # avoid division by zero; gives maximum weight

        raw_ratio = float(N - pos) / float(pos)
        pos_weight_vals.append(float(raw_ratio ** 0.5))

    logger.info("pos_weight sqrt-scaled (training split only):")
    for name, pw in zip(CLASS_NAMES[:num_classes], pos_weight_vals):
        logger.info(f"  {name:<32} {pw:.2f}")

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

            with torch.amp.autocast("cuda", enabled=use_amp):
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
    model:            SentinelModel,
    loader:           DataLoader,
    optimizer:        AdamW,
    loss_fn:          nn.Module,
    scheduler:        OneCycleLR,
    scaler:           torch.amp.GradScaler,
    device:           str,
    grad_clip:        float,
    log_interval:     int,
    use_amp:          bool,
    aux_loss_weight:  float = 0.3,
) -> float:
    model.train()
    total_loss = 0.0

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    for batch_idx, batch in enumerate(pbar):
        graphs, tokens, labels = batch

        graphs         = graphs.to(device)
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        labels         = labels.to(device).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, aux = model(graphs, input_ids, attention_mask, return_aux=True)
            main_loss   = loss_fn(logits, labels)
            aux_loss    = (
                loss_fn(aux["gnn"],         labels) +
                loss_fn(aux["transformer"], labels) +
                loss_fn(aux["fused"],       labels)
            )
            loss = main_loss + aux_loss_weight * aux_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if (batch_idx + 1) % log_interval == 0:
            # Per-eye gradient norm — unscale_ already called above.
            gnn_norm = _grad_norm(model.gnn_eye_proj)
            tf_norm  = _grad_norm(model.transformer_eye_proj)
            fused_norm = _grad_norm(model.fusion)
            logger.info(
                f"  Batch {batch_idx+1}/{len(loader)} | "
                f"loss={loss.item():.4f} (main={main_loss.item():.4f} "
                f"aux={aux_loss.item():.4f}) | "
                f"grad_norm gnn_eye={gnn_norm:.3f} tf_eye={tf_norm:.3f} "
                f"fused_eye={fused_norm:.3f}"
            )
            # Gradient collapse detection: GNN eye gradient < 5% of transformer
            if tf_norm > 1e-8 and gnn_norm / tf_norm < 0.05:
                logger.warning(
                    f"  ⚠ GNN eye gradient collapse: gnn={gnn_norm:.6f} "
                    f"tf={tf_norm:.6f} ratio={gnn_norm/tf_norm:.3f} "
                    "— consider increasing aux_loss_weight"
                )

    n_batches = len(loader)
    if n_batches == 0:
        logger.warning("Empty train loader — returning 0.0 loss")
        return 0.0
    return total_loss / n_batches


def _grad_norm(module: nn.Module) -> float:
    """Return the L2 norm of gradients across all parameters of a module.

    Returns 0.0 if all gradients are None (e.g. module not in backward path).
    Called after scaler.unscale_() so the scale factor is already removed.
    """
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return total_sq ** 0.5


# ---------------------------------------------------------------------------
# Weighted sampler (autoresearch harness — DoS upsampling / rare-class boost)
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
            # DenialOfService (CLASS_NAMES[1]) has 137 support vs 5343 for IntegerUO.
            # 39× boost compensates; value matched to pos_weight ratio in v3 runs.
            w = 39.0 if float(row.get("DenialOfService", 0)) == 1.0 else 1.0
        elif mode == "all-rare":
            # Inverse class count: samples with fewer positive labels get higher weight.
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
    logger.info(f"Training on: {device} | classes: {config.num_classes}")

    if config.loss_fn not in _VALID_LOSS_FNS:
        raise ValueError(
            f"Unknown loss_fn='{config.loss_fn}'. "
            f"Valid options: {sorted(_VALID_LOSS_FNS)}. "
            "Check your TrainConfig or the --loss_fn argument."
        )

    if config.use_amp and device == "cpu":
        logger.warning("use_amp=True but device=cpu — AMP disabled (CUDA only)")
        config.use_amp = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if not (0.0 < config.smoke_subsample_fraction <= 1.0):
        raise ValueError(
            f"smoke_subsample_fraction must be in (0, 1], "
            f"got {config.smoke_subsample_fraction}"
        )

    train_indices = np.load(Path(config.splits_dir) / "train_indices.npy")
    val_indices   = np.load(Path(config.splits_dir) / "val_indices.npy")

    if config.smoke_subsample_fraction < 1.0:
        original_n = len(train_indices)
        keep_n = max(1, int(original_n * config.smoke_subsample_fraction))
        rng = np.random.default_rng(42)
        train_indices = np.sort(rng.choice(train_indices, size=keep_n, replace=False))
        logger.info(
            f"Smoke subsample: {keep_n}/{original_n} train samples "
            f"({100 * config.smoke_subsample_fraction:.0f}%)"
        )

    label_csv_path = Path(config.label_csv) if config.label_csv else None
    if label_csv_path is not None and not label_csv_path.exists():
        raise FileNotFoundError(
            f"label_csv not found: {label_csv_path}. "
            "Check TrainConfig.label_csv or run build_multilabel_index.py first."
        )
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
        _sampler = _build_weighted_sampler(
            train_dataset, label_csv_path, config.use_weighted_sampler
        )

    if _sampler is not None:
        # WeightedRandomSampler is mutually exclusive with shuffle=True
        train_loader = DataLoader(train_dataset, sampler=_sampler, shuffle=False, **_loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **_loader_kwargs)

    logger.info(
        f"DataLoader — workers: {config.num_workers} | "
        f"pin_memory: {_use_workers} | "
        f"AMP: {config.use_amp} | "
        f"TF32: {torch.backends.cuda.matmul.allow_tf32}"
    )

    # ------------------------------------------------------------------
    # pos_weight  (multi-label only; binary mode skips CSV entirely)
    # ------------------------------------------------------------------
    if label_csv_path is not None:
        pos_weight = compute_pos_weight(
            str(label_csv_path), train_indices, config.num_classes, device
        )
    else:
        logger.info(
            "Binary mode (label_csv=None) — pos_weight not computed; "
            "BCEWithLogitsLoss will run unweighted."
        )
        pos_weight = None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
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
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
    ).to(device)

    # ------------------------------------------------------------------
    # Resume — model weights (always) + patience_counter (Fix #11)
    # ------------------------------------------------------------------
    start_epoch      = 1
    best_f1          = 0.0
    patience_counter = 0
    _ckpt_state: dict | None = None  # populated once, reused for optimizer restore

    if config.resume_from:
        logger.info(f"Resuming from: {config.resume_from}")
        # weights_only=True: checkpoint format is confirmed safe (plain dicts +
        # tensors from state_dict() — no custom pickled objects).
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=True)
        _ckpt_state = ckpt

        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"Invalid checkpoint format: {config.resume_from}")

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        lora_skipped = [k for k in missing if "lora_" in k]
        other_missing = [k for k in missing if "lora_" not in k]
        if lora_skipped:
            logger.warning(
                f"Resume: {len(lora_skipped)} LoRA keys not loaded "
                f"(lora_r mismatch — LoRA adapters re-initialised fresh). "
                f"GNN/fusion/classifier weights loaded from checkpoint."
            )
        if other_missing:
            raise RuntimeError(
                f"Resume: {len(other_missing)} non-LoRA keys missing from checkpoint: "
                f"{other_missing[:5]} ..."
            )

        if config.resume_model_only:
            # ── Fine-tune mode: weights loaded, everything else starts fresh ──
            # start_epoch / best_f1 / patience_counter stay at their defaults
            # (1 / 0.0 / 0) so the epoch counter, early stopping, and checkpoint
            # logic all behave as if this is a brand-new run.
            logger.info(
                f"Model-only resume (fine-tune) — weights from checkpoint epoch "
                f"{ckpt.get('epoch', 0)} (raw F1={ckpt.get('best_f1', 0.0):.4f}). "
                f"Epoch counter, patience, and best_f1 reset to fresh start."
            )
        else:
            # ── Full resume: restore epoch counter and early-stopping state ──
            start_epoch      = ckpt.get("epoch", 0) + 1
            best_f1          = ckpt.get("best_f1", 0.0)
            # Fix #11: restore patience counter
            patience_counter = ckpt.get("patience_counter", 0)

            # Fix #23: override patience_counter from per-epoch state file
            _resume_state_path = Path(config.resume_from).with_suffix(".state.json")
            if _resume_state_path.exists():
                _saved_state = json.loads(_resume_state_path.read_text())
                _state_epoch = _saved_state.get("epoch", 0)
                if _state_epoch >= start_epoch - 1:
                    patience_counter = _saved_state.get("patience_counter", patience_counter)
                    logger.info(
                        f"patience_counter overridden from state file "
                        f"(epoch {_state_epoch}): "
                        f"{patience_counter}/{config.early_stop_patience}"
                    )
            else:
                logger.warning(
                    "No .state.json sidecar found alongside the resume checkpoint — "
                    f"patience_counter initialised from checkpoint value ({patience_counter}). "
                    "Non-improvement epochs before the interruption are NOT counted. "
                    "State sidecars are written after every epoch starting with this run."
                )

            logger.info(
                f"Full resume from epoch {start_epoch-1} | "
                f"best_f1={best_f1:.4f} | "
                f"patience_counter={patience_counter}/{config.early_stop_patience}"
            )

        ckpt_cfg = ckpt.get("config", {})

        ckpt_num_classes = ckpt_cfg.get("num_classes")
        if ckpt_num_classes is not None and ckpt_num_classes != config.num_classes:
            raise ValueError(
                f"Checkpoint num_classes={ckpt_num_classes} does not match "
                f"config.num_classes={config.num_classes}. "
                "Fix TrainConfig or remove resume_from."
            )
        # ── Fix #9: use ARCHITECTURE constant, not config.architecture ──────
        ckpt_arch = ckpt_cfg.get("architecture")
        if ckpt_arch is not None and ckpt_arch != ARCHITECTURE:
            raise ValueError(
                f"Checkpoint architecture='{ckpt_arch}' does not match "
                f"expected '{ARCHITECTURE}'. "
                "Mismatched architectures will corrupt the state_dict load."
            )

        # ── Fix #12: batch_size change guard ────────────────────────────────
        ckpt_batch_size = ckpt_cfg.get("batch_size")
        _batch_size_changed = (
            ckpt_batch_size is not None
            and ckpt_batch_size != config.batch_size
        )
        if _batch_size_changed and not config.resume_model_only:
            if config.force_optimizer_reset:
                logger.warning(
                    f"\n{'!'*70}\n"
                    f"BATCH SIZE CHANGED: checkpoint={ckpt_batch_size} → "
                    f"current={config.batch_size}.\n"
                    f"force_optimizer_reset=True — optimizer/scheduler state will "
                    f"be DISCARDED even though --no-resume-model-only was set.\n"
                    f"Model weights + patience_counter are preserved.\n"
                    f"AdamW will start fresh, calibrated to the new batch size.\n"
                    f"{'!'*70}"
                )
            else:
                logger.warning(
                    f"\n{'!'*70}\n"
                    f"BATCH SIZE MISMATCH DETECTED (Fix #12):\n"
                    f"  Checkpoint batch_size : {ckpt_batch_size}\n"
                    f"  Current   batch_size : {config.batch_size}\n"
                    f"  steps_per_epoch changed: "
                    f"~{round(len(train_dataset)/ckpt_batch_size)} → "
                    f"~{round(len(train_dataset)/config.batch_size)}\n\n"
                    f"You are doing a FULL RESUME (--no-resume-model-only). "
                    f"The Adam m/v moments in the checkpoint were accumulated "
                    f"under batch_size={ckpt_batch_size}. Loading them into a "
                    f"loop with batch_size={config.batch_size} will likely cause "
                    f"loss spikes and degraded F1 for several epochs while Adam "
                    f"re-calibrates its running statistics.\n\n"
                    f"Recommended actions (pick ONE):\n"
                    f"  (a) Use model-only resume (omit --no-resume-model-only) — "
                    f"fresh AdamW + fresh OneCycleLR aligned to the new batch_size. "
                    f"This is the safest option when batch_size changes.\n"
                    f"  (b) Add --resume-reset-optimizer — restores model weights + "
                    f"patience_counter but discards stale optimizer/scheduler state.\n"
                    f"  (c) Keep batch_size={ckpt_batch_size} to match the checkpoint.\n\n"
                    f"Training will proceed with the stale optimizer state. "
                    f"Monitor loss for spikes. If F1 does not recover within "
                    f"5 epochs, consider restarting with option (a) or (b).\n"
                    f"{'!'*70}"
                )

    # ------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    if config.loss_fn == "focal":
        _focal = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)

        class _FocalFromLogits(nn.Module):
            """Thin wrapper so FocalLoss receives FP32 probabilities, not raw logits."""
            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return _focal(torch.sigmoid(logits.float()), targets)

        loss_fn: nn.Module = _FocalFromLogits()
        logger.info(f"Loss: FocalLoss(gamma={config.focal_gamma}, alpha={config.focal_alpha}) with FP32 sigmoid guard")
    else:  # "bce"
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("Loss: BCEWithLogitsLoss with class-balanced pos_weight")
        # ── Fix #13: pos_weight / optimizer state inconsistency warning ─────
        if pos_weight is not None and config.resume_from and not config.resume_model_only:
            logger.warning(
                "Fix #13: BCEWithLogitsLoss pos_weight has been recomputed "
                "from the current training split. The optimizer state being "
                "restored from the checkpoint was accumulated under the "
                "checkpoint's pos_weight values. If the training split indices "
                "have NOT changed since the checkpoint was saved (the normal "
                "case), this is a no-op difference. If splits were regenerated, "
                "the effective loss scale has changed and the restored Adam "
                "moments are calibrated to a different gradient magnitude. "
                "Consider using model-only resume in that case."
            )

    # ------------------------------------------------------------------
    # Trainable parameter set
    # ------------------------------------------------------------------
    # GNN, CrossAttentionFusion, and Classifier are ALL trained — they are
    # NOT frozen.  Only the CodeBERT backbone (125M params) is frozen; LoRA
    # adapters injected into CodeBERT's Q+V matrices by peft carry the
    # trainable transformer signal (~295K params at r=8, ~590K at r=16).
    #
    # filter(requires_grad) therefore includes:
    #   • GNNEncoder weights (conv1/2/3, edge_emb)
    #   • CrossAttentionFusion weights (node_proj, token_proj, MHA, out_proj)
    #   • Classifier Linear(fusion_output_dim → num_classes)
    #   • LoRA adapter matrices (injected by peft into CodeBERT Q+V)
    #
    # It excludes:
    #   • CodeBERT base weights (bert.embeddings.*, bert.encoder.layer.*.weight, …)
    #     — frozen by peft when LoRA is applied
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # ── Fix #8: OneCycleLR uses remaining_epochs on resume ────────────────
    remaining_epochs = config.epochs - start_epoch + 1
    if remaining_epochs <= 0:
        logger.warning(
            f"start_epoch={start_epoch} ≥ config.epochs={config.epochs}: "
            "nothing left to train. Increase config.epochs before resuming."
        )
        return {
            "best_f1_macro": best_f1,
            "final_epoch": start_epoch - 1,
            "early_stopped": False,
            "checkpoint_path": str(Path(config.checkpoint_dir) / config.checkpoint_name),
        }

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        epochs=remaining_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.warmup_pct,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    # ------------------------------------------------------------------
    # Restore optimizer + scheduler state if doing a full resume
    # ------------------------------------------------------------------
    if _ckpt_state is not None and not config.resume_model_only:
        ckpt = _ckpt_state

        _skip_optimizer = config.force_optimizer_reset

        if not _skip_optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            logger.info("Optimizer state restored.")
        elif _skip_optimizer:
            logger.info(
                "Optimizer state skipped — force_optimizer_reset=True. "
                "Fresh AdamW in use."
            )
        else:
            logger.warning(
                "Full resume requested (--no-resume-model-only) but the checkpoint "
                "contains no 'optimizer' key. Fresh AdamW initialised. "
                "Ensure your checkpoint was saved with optimizer state included."
            )

        if "scheduler" in ckpt and not _skip_optimizer:
            new_total_steps = remaining_epochs * len(train_loader)
            if "total_steps" not in ckpt["scheduler"]:
                logger.warning(
                    "Scheduler state skipped — checkpoint scheduler dict does not "
                    "contain 'total_steps' (checkpoint may have been created with an "
                    "older PyTorch version). Fresh OneCycleLR in use; optimizer "
                    "momentum state was restored."
                )
            else:
                ckpt_total_steps = ckpt["scheduler"]["total_steps"]
                if ckpt_total_steps == new_total_steps:
                    scheduler.load_state_dict(ckpt["scheduler"])
                    logger.info("Scheduler state restored.")
                else:
                    logger.warning(
                        f"Scheduler state skipped — checkpoint total_steps={ckpt_total_steps} "
                        f"≠ new total_steps={new_total_steps} (epoch extension or batch_size "
                        f"change detected). Fresh scheduler in use; optimizer momentum state "
                        f"was restored."
                    )
        elif _skip_optimizer:
            logger.info(
                "Scheduler state skipped — force_optimizer_reset=True. "
                "Fresh OneCycleLR in use."
            )
    elif _ckpt_state is not None:
        logger.info("Resumed model weights only (fresh optimizer/scheduler).")

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        params = {
            "num_classes":           config.num_classes,
            "epochs":                config.epochs,
            "remaining_epochs":      remaining_epochs,
            "batch_size":            config.batch_size,
            "lr":                    config.lr,
            "weight_decay":          config.weight_decay,
            "threshold":             config.threshold,
            "grad_clip":             config.grad_clip,
            "warmup_pct":            config.warmup_pct,
            "num_workers":           config.num_workers,
            "use_amp":               config.use_amp,
            "loss_fn":               config.loss_fn,
            # Fix #9 (external review): log focal params so Focal Loss sweeps
            # are distinguishable in the MLflow UI. Present for every run;
            # irrelevant when loss_fn='bce' but harmless to log.
            "focal_gamma":           config.focal_gamma,
            "focal_alpha":           config.focal_alpha,
            "device":                device,
            "architecture":          ARCHITECTURE,
            "label_csv":             config.label_csv,
            "resume_from":           config.resume_from or "none",
            "resume_model_only":     config.resume_model_only,
            "force_optimizer_reset": config.force_optimizer_reset,
            "early_stop_patience":   config.early_stop_patience,
            "gnn_hidden_dim":        config.gnn_hidden_dim,
            "gnn_layers":            config.gnn_layers,
            "gnn_heads":             config.gnn_heads,
            "gnn_dropout":           config.gnn_dropout,
            "use_edge_attr":         config.use_edge_attr,
            "gnn_edge_emb_dim":      config.gnn_edge_emb_dim,
            "aux_loss_weight":       config.aux_loss_weight,
            "lora_r":                config.lora_r,
            "lora_alpha":            config.lora_alpha,
            "lora_dropout":          config.lora_dropout,
            "lora_target_modules":   ",".join(config.lora_target_modules),
            "fusion_output_dim":     config.fusion_output_dim,
        }
        if pos_weight is not None:
            for name, pw in zip(CLASS_NAMES[:config.num_classes], pos_weight.cpu().tolist()):
                params[f"pos_weight_{name}"] = round(pw, 3)
        mlflow.log_params(params)

        checkpoint_path = Path(config.checkpoint_dir) / config.checkpoint_name
        final_epoch = start_epoch - 1

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        for epoch in range(start_epoch, config.epochs + 1):
            final_epoch = epoch
            logger.info(f"\n{'='*60}\nEpoch {epoch}/{config.epochs}\n{'='*60}")

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                grad_clip=config.grad_clip,
                log_interval=config.log_interval,
                use_amp=config.use_amp,
                aux_loss_weight=config.aux_loss_weight,
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                threshold=config.threshold,
                use_amp=config.use_amp,
            )

            mlflow.log_metric("train_loss",    train_loss,               step=epoch)
            mlflow.log_metric("val_f1_macro",  val_metrics["f1_macro"],  step=epoch)
            mlflow.log_metric("val_f1_micro",  val_metrics["f1_micro"],  step=epoch)
            mlflow.log_metric("val_hamming",   val_metrics["hamming"],   step=epoch)
            for name in CLASS_NAMES[:config.num_classes]:
                mlflow.log_metric(f"val_f1_{name}", val_metrics[f"f1_{name}"], step=epoch)

            class_f1s = [
                (n, val_metrics[f"f1_{n}"]) for n in CLASS_NAMES[:config.num_classes]
            ]
            class_f1s.sort(key=lambda x: x[1], reverse=True)
            top3    = " | ".join(f"{n}={v:.3f}" for n, v in class_f1s[:3])
            bottom3 = " | ".join(f"{n}={v:.3f}" for n, v in class_f1s[-3:])

            logger.info(
                f"Epoch {epoch:>2}/{config.epochs} | "
                f"Loss={train_loss:.4f} | F1-macro={val_metrics['f1_macro']:.4f} | "
                f"Hamming={val_metrics['hamming']:.4f}\n"
                f"  Top3:    {top3}\n"
                f"  Bottom3: {bottom3}"
            )

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                patience_counter = 0
                torch.save(
                    {
                        "model":           model.state_dict(),
                        "optimizer":       optimizer.state_dict(),
                        "scheduler":       scheduler.state_dict(),
                        "epoch":           epoch,
                        "best_f1":         best_f1,
                        "patience_counter": patience_counter,
                        "config": {
                            **dataclasses.asdict(config),
                            "num_classes":  config.num_classes,
                            "class_names":  CLASS_NAMES[:config.num_classes],
                            "architecture": ARCHITECTURE,
                        },
                    },
                    checkpoint_path,
                )
                logger.info(f"  ★ New best F1-macro: {best_f1:.4f} — checkpoint saved")
            else:
                patience_counter += 1
                logger.info(
                    f"  No improvement for "
                    f"{patience_counter} / {config.early_stop_patience} epochs"
                )
                if patience_counter >= config.early_stop_patience:
                    logger.info(
                        f"  Early stopping triggered after {epoch} epochs "
                        f"(best F1-macro = {best_f1:.4f})"
                    )
                    break

            # ── Fix #23: write per-epoch state sidecar ──────────────────────
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
