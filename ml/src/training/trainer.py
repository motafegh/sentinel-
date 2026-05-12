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

AUDIT FIXES (2026-05-01 through 2026-05-12): see inline comments.

Fix #26 — need_weights=False on MHA in fusion_layer.py (see that file)
Fix #27 — gc.collect() + torch.cuda.empty_cache() between epochs
         The CUDA caching allocator holds freed blocks from evaluate()
         across epoch boundaries, contributing to allocator fragmentation
         in long runs. Explicit cache release after each epoch gives the
         allocator a clean slate for epoch N+1's training loop.
         NOTE: This does NOT free tensors still in scope — Python's GC
         already handles those at function return. empty_cache() only
         releases the allocator's internal free-block pool back to CUDA.
"""

from __future__ import annotations

import gc
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

ARCHITECTURE = "three_eye_v5"

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
    gnn_layers:       int   = 4
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
    aux_loss_weight:     float = 0.3

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
        if self.gnn_layers != 4:
            raise ValueError(
                f"gnn_layers={self.gnn_layers} is not supported in v5.0. "
                "Only gnn_layers=4 is implemented."
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps={self.gradient_accumulation_steps} must be >= 1."
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
    model:                       SentinelModel,
    loader:                      DataLoader,
    optimizer:                   AdamW,
    loss_fn:                     nn.Module,
    scheduler:                   OneCycleLR,
    scaler:                      torch.amp.GradScaler,
    device:                      str,
    grad_clip:                   float,
    log_interval:                int,
    use_amp:                     bool,
    aux_loss_weight:             float = 0.3,
    gradient_accumulation_steps: int   = 1,
) -> float:
    model.train()
    total_loss = 0.0

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    accum_steps = max(1, gradient_accumulation_steps)

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    for batch_idx, batch in enumerate(pbar):
        graphs, tokens, labels = batch

        graphs         = graphs.to(device)
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        labels         = labels.to(device).float()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, aux = model(graphs, input_ids, attention_mask, return_aux=True)
            main_loss   = loss_fn(logits, labels)
            aux_loss    = (
                loss_fn(aux["gnn"],         labels) +
                loss_fn(aux["transformer"], labels) +
                loss_fn(aux["fused"],       labels)
            )
            loss = (main_loss + aux_loss_weight * aux_loss) / accum_steps

        scaler.scale(loss).backward()

        is_last_batch = (batch_idx + 1 == len(loader))
        is_accum_step = ((batch_idx + 1) % accum_steps == 0) or is_last_batch

        if is_accum_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_for_log = loss.item() * accum_steps
        total_loss  += loss_for_log
        pbar.set_postfix({"loss": f"{loss_for_log:.4f}"})

        if (batch_idx + 1) % log_interval == 0:
            gnn_norm   = _grad_norm(model.gnn_eye_proj)
            tf_norm    = _grad_norm(model.transformer_eye_proj)
            fused_norm = _grad_norm(model.fusion)
            logger.info(
                f"  Batch {batch_idx+1}/{len(loader)} | "
                f"loss={loss_for_log:.4f} (main={main_loss.item():.4f} "
                f"aux={aux_loss.item():.4f}) | "
                f"grad_norm gnn_eye={gnn_norm:.3f} tf_eye={tf_norm:.3f} "
                f"fused_eye={fused_norm:.3f}"
            )
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
        pos_weight = compute_pos_weight(str(label_csv_path), train_indices, config.num_classes, device)
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
                _saved_state = json.loads(_resume_state_path.read_text())
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

    if config.loss_fn == "focal":
        _focal = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
        class _FocalFromLogits(nn.Module):
            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return _focal(torch.sigmoid(logits.float()), targets)
        loss_fn: nn.Module = _FocalFromLogits()
        logger.info(f"Loss: FocalLoss(gamma={config.focal_gamma}, alpha={config.focal_alpha})")
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("Loss: BCEWithLogitsLoss with class-balanced pos_weight")
        if pos_weight is not None and config.resume_from and not config.resume_model_only:
            logger.warning("Fix #13: pos_weight recomputed from current training split.")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

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
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        epochs=remaining_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.warmup_pct,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    if _ckpt_state is not None and not config.resume_model_only:
        ckpt = _ckpt_state
        _skip_optimizer = config.force_optimizer_reset
        if not _skip_optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            logger.info("Optimizer state restored.")
        elif _skip_optimizer:
            logger.info("Optimizer state skipped — force_optimizer_reset=True.")
        else:
            logger.warning("Full resume requested but no 'optimizer' key in checkpoint. Fresh AdamW.")

        if "scheduler" in ckpt and not _skip_optimizer:
            new_total_steps = remaining_epochs * steps_per_epoch
            if "total_steps" not in ckpt["scheduler"]:
                logger.warning("Scheduler state skipped — no 'total_steps' in checkpoint.")
            else:
                ckpt_total_steps = ckpt["scheduler"]["total_steps"]
                if ckpt_total_steps == new_total_steps:
                    scheduler.load_state_dict(ckpt["scheduler"])
                    logger.info("Scheduler state restored.")
                else:
                    logger.warning(f"Scheduler skipped — total_steps mismatch ({ckpt_total_steps} vs {new_total_steps}).")
        elif _skip_optimizer:
            logger.info("Scheduler state skipped — force_optimizer_reset=True.")
    elif _ckpt_state is not None:
        logger.info("Resumed model weights only (fresh optimizer/scheduler).")

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
            "lora_r":                      config.lora_r,
            "lora_alpha":                  config.lora_alpha,
            "lora_dropout":                config.lora_dropout,
            "lora_target_modules":         ",".join(config.lora_target_modules),
            "fusion_output_dim":           config.fusion_output_dim,
        }
        if pos_weight is not None:
            for name, pw in zip(CLASS_NAMES[:config.num_classes], pos_weight.cpu().tolist()):
                params[f"pos_weight_{name}"] = round(pw, 3)
        mlflow.log_params(params)

        checkpoint_path = Path(config.checkpoint_dir) / config.checkpoint_name
        final_epoch = start_epoch - 1

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
                gradient_accumulation_steps=accum_steps,
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                threshold=config.threshold,
                use_amp=config.use_amp,
            )

            # Fix #27: release CUDA caching allocator free-blocks between epochs.
            # evaluate() returns with all its tensors out of scope; gc.collect()
            # ensures CPython finalises any lingering reference cycles before
            # empty_cache() returns pages to CUDA. Reduces allocator fragmentation
            # entering the next epoch's training loop.
            gc.collect()
            torch.cuda.empty_cache()

            mlflow.log_metric("train_loss",    train_loss,               step=epoch)
            mlflow.log_metric("val_f1_macro",  val_metrics["f1_macro"],  step=epoch)
            mlflow.log_metric("val_f1_micro",  val_metrics["f1_micro"],  step=epoch)
            mlflow.log_metric("val_hamming",   val_metrics["hamming"],   step=epoch)
            for name in CLASS_NAMES[:config.num_classes]:
                mlflow.log_metric(f"val_f1_{name}", val_metrics[f"f1_{name}"], step=epoch)

            class_f1s = [(n, val_metrics[f"f1_{n}"]) for n in CLASS_NAMES[:config.num_classes]]
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
                        "model":            model.state_dict(),
                        "optimizer":        optimizer.state_dict(),
                        "scheduler":        scheduler.state_dict(),
                        "epoch":            epoch,
                        "best_f1":          best_f1,
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
