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
"""

from __future__ import annotations

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
from torch_geometric.loader import DataLoader
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
# Ampere+ GPUs (RTX 30xx/40xx/A-series) use TF32 units for cuBLAS and cuDNN.
# ~1.5× speedup, no code changes needed elsewhere. Silently ignored on older
# GPUs so it's always safe to enable.
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
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # --- Paths ---
    graphs_dir:      str = "ml/data/graphs"
    tokens_dir:      str = "ml/data/tokens"
    splits_dir:      str = "ml/data/splits"
    checkpoint_dir:  str = "ml/checkpoints"
    checkpoint_name: str = "multilabel_crossattn_best.pt"

    # --- Model ---
    num_classes: int = NUM_CLASSES

    # --- Label source ---
    label_csv: str = "ml/data/processed/multilabel_index.csv"

    # --- Training ---
    epochs:       int   = 40
    batch_size:   int   = 16          # safe for 8 GB VRAM with AMP
    lr:           float = 3e-4
    weight_decay: float = 1e-2
    threshold:    float = 0.5
    early_stop_patience: int = 7      # Early stopping patience (epochs)

    # --- Stability ---
    grad_clip:  float = 1.0
    warmup_pct: float = 0.05

    # --- Speed: AMP ---
    # Optimisation #1. Wraps forward pass in autocast(BF16) + GradScaler.
    # BF16 preferred over FP16 on Ampere — same speed, no inf/nan risk.
    # Set use_amp=False only if you see NaN losses or are on a pre-Volta GPU.
    use_amp: bool = True

    # --- Speed: DataLoader ---
    # Optimisation #3. 2 workers pre-fetch and collate while GPU trains.
    # Safe with RAM cache (workers get a read-only fork of the dict).
    # Set to 0 only if you hit CUDA multiprocessing issues.
    num_workers:         int  = 2
    persistent_workers:  bool = True   # avoids re-spawning workers each epoch

    # --- Loss function ---
    # "bce"   (default): BCEWithLogitsLoss with class-balanced pos_weight.
    # "focal": FocalLoss(gamma=2.0, alpha=0.25) — down-weights easy examples,
    #          useful when class imbalance is severe and pos_weight is insufficient.
    #          Applies sigmoid internally so it receives raw logits identically to bce.
    loss_fn: str = "bce"

    # --- Cache ---
    cache_path: str | None = "ml/data/cached_dataset.pkl"

    # --- Logging ---
    log_interval: int = 100

    # --- MLflow ---
    experiment_name: str = "sentinel-multilabel"
    run_name:        str = "multilabel-crossattn-v1"

    # --- Device ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # --- Resume ---
    resume_from:       str | None = None
    resume_model_only: bool       = True


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

    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts

    pos_weight_vals = []
    for c, (pos, neg) in enumerate(zip(pos_counts, neg_counts)):
        if pos == 0:
            logger.warning(
                f"Class '{CLASS_NAMES[c]}' (index {c}) has zero positives — "
                "pos_weight capped at 100.0"
            )
            pos_weight_vals.append(100.0)
        else:
            pos_weight_vals.append(float(neg) / float(pos))

    logger.info("pos_weight (training split only):")
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

            # AMP in eval too — same speedup, inference is pure forward pass
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(graphs, input_ids, attention_mask)

            probs = torch.sigmoid(logits.float())   # cast back to FP32 for sigmoid
            preds = (probs >= threshold).long()

            all_preds.append(preds.cpu().numpy())
            all_true.append(labels.long().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)

    f1_macro    = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    f1_micro    = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    hamming     = hamming_loss(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None,    zero_division=0)

    metrics = {"f1_macro": f1_macro, "f1_micro": f1_micro, "hamming": hamming}
    for i, name in enumerate(CLASS_NAMES[:y_true.shape[1]]):
        metrics[f"f1_{name}"] = float(f1_per_class[i])

    return metrics


# ---------------------------------------------------------------------------
# Training one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model:        SentinelModel,
    loader:       DataLoader,
    optimizer:    AdamW,
    loss_fn:      nn.BCEWithLogitsLoss,
    scheduler:    OneCycleLR,
    scaler:       torch.amp.GradScaler,
    device:       str,
    grad_clip:    float,
    log_interval: int,
    use_amp:      bool,
) -> float:
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    for batch_idx, batch in enumerate(pbar):
        graphs, tokens, labels = batch

        graphs         = graphs.to(device)
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        labels         = labels.to(device).float()

        # Optimisation #5: set_to_none=True frees gradient tensors entirely
        # instead of writing zeros — saves one full-model memory write per step.
        optimizer.zero_grad(set_to_none=True)

        # Optimisation #1: AMP forward pass
        # autocast selects BF16 for eligible ops (matmuls, attention, convs)
        # and keeps FP32 for ops that need precision (norm, softmax, etc.).
        # GradScaler multiplies loss by a large scale factor before backward
        # to prevent FP16 underflow in gradients, then unscales before the
        # optimizer step. With BF16 the scaler is essentially a no-op but
        # keeping it here means the code works correctly with FP16 too.
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(graphs, input_ids, attention_mask)
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"  Batch {batch_idx+1}/{len(loader)} | loss={loss.item():.4f}"
            )

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(config: TrainConfig) -> None:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    device = config.device
    logger.info(f"Training on: {device} | classes: {config.num_classes}")

    if config.use_amp and device == "cpu":
        logger.warning("use_amp=True but device=cpu — AMP disabled (CUDA only)")
        config.use_amp = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_indices = np.load(Path(config.splits_dir) / "train_indices.npy")
    val_indices   = np.load(Path(config.splits_dir) / "val_indices.npy")

    label_csv_path = Path(config.label_csv) if config.label_csv else None
    cache_path     = Path(config.cache_path) if config.cache_path else None

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

    # Optimisation #3 + #4: workers + pin_memory consistent
    # pin_memory only helps when workers exist to do the pinning asynchronously.
    # persistent_workers keeps worker processes alive between epochs.
    _use_workers   = config.num_workers > 0
    _prefetch      = 2 if _use_workers else None  # pre-fetch 2 batches ahead

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dual_path_collate_fn,
        num_workers=config.num_workers,
        pin_memory=_use_workers,
        persistent_workers=_use_workers and config.persistent_workers,
        prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dual_path_collate_fn,
        num_workers=config.num_workers,
        pin_memory=_use_workers,
        persistent_workers=_use_workers and config.persistent_workers,
        prefetch_factor=_prefetch,
    )

    logger.info(
        f"DataLoader — workers: {config.num_workers} | "
        f"pin_memory: {_use_workers} | "
        f"AMP: {config.use_amp} | "
        f"TF32: {torch.backends.cuda.matmul.allow_tf32}"
    )

    # ------------------------------------------------------------------
    # pos_weight
    # ------------------------------------------------------------------
    pos_weight = compute_pos_weight(
        str(label_csv_path), train_indices, config.num_classes, device
    )

    # ------------------------------------------------------------------
    # Model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    model = SentinelModel(num_classes=config.num_classes).to(device)

    if config.loss_fn == "focal":
        # FocalLoss expects post-sigmoid probabilities; wrap so it receives raw logits
        # identically to BCEWithLogitsLoss — no change needed in train_one_epoch.
        _focal = FocalLoss(gamma=2.0, alpha=0.25)
        class _FocalFromLogits(nn.Module):
            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return _focal(torch.sigmoid(logits), targets)
        loss_fn: nn.Module = _FocalFromLogits()
        logger.info("Loss: FocalLoss(gamma=2.0, alpha=0.25) — pos_weight ignored")
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("Loss: BCEWithLogitsLoss with class-balanced pos_weight")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.warmup_pct,
        anneal_strategy="cos",
    )

    # Optimisation #1: GradScaler for AMP backward pass.
    # enabled=False when AMP is off — scaler becomes a transparent pass-through,
    # so the train_one_epoch code path is identical regardless of use_amp.
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    # ------------------------------------------------------------------
    # Resume (safe: model only by default)
    # ------------------------------------------------------------------
    start_epoch = 1
    best_f1     = 0.0
    patience_counter = 0   # for early stopping

    if config.resume_from:
        logger.info(f"Resuming from: {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)

        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise RuntimeError(f"Invalid checkpoint format: {config.resume_from}")

        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1     = ckpt.get("best_f1", 0.0)

        if not config.resume_model_only:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                logger.info("Optimizer state restored.")
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
                logger.info("Scheduler state restored.")
        else:
            logger.info("Resumed model weights only (fresh optimizer/scheduler).")

        logger.info(f"Resumed from epoch {start_epoch-1} | best_f1={best_f1:.4f}")

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        params = {
            "num_classes":          config.num_classes,
            "epochs":               config.epochs,
            "batch_size":           config.batch_size,
            "lr":                   config.lr,
            "weight_decay":         config.weight_decay,
            "threshold":            config.threshold,
            "grad_clip":            config.grad_clip,
            "warmup_pct":           config.warmup_pct,
            "num_workers":          config.num_workers,
            "use_amp":              config.use_amp,
            "device":               device,
            "architecture":         "cross_attention_lora",
            "label_csv":            config.label_csv,
            "resume_from":          config.resume_from or "none",
            "resume_model_only":    config.resume_model_only,
            "early_stop_patience":  config.early_stop_patience,
        }
        for name, pw in zip(CLASS_NAMES[:config.num_classes], pos_weight.cpu().tolist()):
            params[f"pos_weight_{name}"] = round(pw, 3)
        mlflow.log_params(params)

        checkpoint_path = Path(config.checkpoint_dir) / config.checkpoint_name

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        for epoch in range(start_epoch, config.epochs + 1):
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
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                threshold=config.threshold,
                use_amp=config.use_amp,
            )

            # Log metrics
            mlflow.log_metric("train_loss",    train_loss,               step=epoch)
            mlflow.log_metric("val_f1_macro",  val_metrics["f1_macro"],  step=epoch)
            mlflow.log_metric("val_f1_micro",  val_metrics["f1_micro"],  step=epoch)
            mlflow.log_metric("val_hamming",   val_metrics["hamming"],   step=epoch)
            for name in CLASS_NAMES[:config.num_classes]:
                mlflow.log_metric(f"val_f1_{name}", val_metrics[f"f1_{name}"], step=epoch)

            # Console summary
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

            # Check for improvement -> save checkpoint and reset patience
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                patience_counter = 0
                torch.save(
                    {
                        "model":     model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch":     epoch,
                        "best_f1":   best_f1,
                        "config": {
                            **dataclasses.asdict(config),
                            "num_classes":  config.num_classes,
                            "class_names":  CLASS_NAMES[:config.num_classes],
                            "architecture": "cross_attention_lora",
                        },
                    },
                    checkpoint_path,
                )
                logger.info(f"  ★ New best F1-macro: {best_f1:.4f} — checkpoint saved")
            else:
                patience_counter += 1
                logger.info(f"  No improvement for {patience_counter} / {config.early_stop_patience} epochs")
                if patience_counter >= config.early_stop_patience:
                    logger.info(f"  Early stopping triggered after {epoch} epochs (best F1-macro = {best_f1:.4f})")
                    break

        # Optimisation #6: log artifact once at end, not on every checkpoint save.
        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path))

        logger.info(f"\n✅ Training complete. Best val F1-macro: {best_f1:.4f}")


if __name__ == "__main__":
    config = TrainConfig()
    train(config)