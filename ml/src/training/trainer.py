"""
trainer.py — SENTINEL Training Loop (v8 — Three-Eye GNN+CodeBERT+LoRA)
With tqdm progress bars, safe resume, offline mode, and EARLY STOPPING.

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
           Now grads are read after grad_clip but before zero_grad,
           i.e. post-clip — the correct moment to measure them.
           Logging fires on optimizer steps that cross log_interval, not on
           every micro-batch.
Fix #29 — VRAM monitoring: when reserved VRAM > 90 %, log a warning.
           Mid-epoch empty_cache() removed — it forced a CUDA sync every
           log_interval steps. Between-epoch cleanup is sufficient.
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
import time
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
from ml.src.training.training_logger import (
    KILL,
    StructuredLogger,
    TrainingAbortError,
    compute_grad_stats,
    label_dist_from_tensor,
)

# ---------------------------------------------------------------------------
# Logging setup — module level only (handlers added per-run inside train())
# ---------------------------------------------------------------------------
# Use handler ID 0 (loguru's default stderr sink) specifically rather than
# logger.remove() which would destroy any caller-configured handlers.
try:
    logger.remove(0)
except ValueError:
    pass  # already removed (e.g. in test environments that pre-configure loguru)
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

ARCHITECTURE = "three_eye_v8"

MODEL_VERSION = "v8.0"

_VALID_LOSS_FNS: frozenset[str] = frozenset({"bce", "focal", "asl"})


# [A35] Module-level class — was a local class inside train(), making it
# unpicklable and incompatible with DDP. Moved here so it can be pickled.
class _FocalFromLogits(nn.Module):
    """Wraps FocalLoss to accept logits (applies sigmoid internally)."""

    def __init__(self, focal: "FocalLoss") -> None:
        super().__init__()
        self._focal = focal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._focal(torch.sigmoid(logits.float()), targets)


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
    tokens_dir:      str = "ml/data/tokens_windowed"
    splits_dir:      str = "ml/data/splits/deduped"
    checkpoint_dir:  str = "ml/checkpoints"
    checkpoint_name: str = "sentinel_best.pt"

    # --- Model ---
    num_classes:       int   = NUM_CLASSES
    fusion_output_dim: int   = 128
    fusion_dropout:    float = 0.3
    # IMP-D1: raise to 2048 after re-extraction with max_nodes=2048.
    # At 1024 the 227 contracts >1024 nodes are truncated in fusion attention.
    fusion_max_nodes:  int   = 1024

    # --- GNN architecture (v8) ---
    gnn_hidden_dim:   int   = 256
    gnn_layers:       int   = 8
    gnn_heads:        int   = 8
    gnn_dropout:      float = 0.2
    use_edge_attr:    bool  = True
    gnn_edge_emb_dim: int   = 64
    # JK connections (Phase 1-A1, 2026-05-14)
    gnn_use_jk:       bool  = True
    gnn_jk_mode:      str   = 'attention'
    # Phase 2 ablation: list of edge type IDs in Phase 2 cfg_mask; None = all v8 types
    gnn_phase2_edge_types: list[int]|None = None

    # --- LoRA architecture (v8) ---
    lora_r:               int        = 16
    lora_alpha:           int        = 32
    lora_dropout:         float      = 0.1
    lora_target_modules:  list[str]  = field(default_factory=lambda: ["query", "value"])

    # --- Label source ---
    label_csv: str = "ml/data/processed/multilabel_index_cleaned.csv"

    # --- Training ---
    epochs:              int   = 100         # 100 epochs; increased from v6's 60 for more data + harder ASL loss
    batch_size:          int   = 8           # Fix #28: 8 fits 8 GB GPU with MAX_WINDOWS=4 (16 saturated 7.9/8.0 GB)
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
    # Phase 2 CEI auxiliary loss (Interp-2): direct supervision on Phase 2 embeddings.
    # CEI classes (Reentrancy, ExternalBug, TOD) get 3× weighting.
    # Set to 0.0 to disable.
    aux_phase2_loss_weight: float = 0.10

    # --- Aux loss warmup (Fix #33) ---
    # aux_loss_weight ramps from 0 → aux_loss_weight linearly over this many
    # epochs.  Set to 0 to disable warmup (always full aux weight).
    aux_loss_warmup_epochs: int = 8

    # --- Gradient accumulation (2026-05-12) ---
    gradient_accumulation_steps: int = 8  # v7: batch=8 × 8 = effective 64

    # --- Stability ---
    grad_clip:  float = 1.0
    warmup_pct: float = 0.10

    # --- Speed: AMP ---
    use_amp: bool = True

    # --- Speed: DataLoader + compilation ---
    # num_workers=4: fork workers inherit shared cache via copy-on-write — zero extra RAM.
    # prefetch_factor=4 keeps the GPU fed; workers never call CUDA so fork is safe.
    # use_compile=True: submodule-level compile (GNN+fusion+classifier, NOT transformer).
    # CodeBERT+LoRA has HuggingFace control flow that breaks dynamo's compile graph when
    # the whole model is compiled; compiling submodules separately isolates those breaks.
    # TRITON_CACHE_DIR=/tmp/triton_cache routes Triton JIT to tmpfs, avoiding WSL2 p9io crash.
    num_workers:         int  = 4
    persistent_workers:  bool = True
    use_compile:         bool = True

    # --- Loss function ---
    # ASL (Ridnik et al. ICCV 2021): gamma_neg=4 down-weights easy negatives
    # (vast majority of 44K×10 cells), freeing gradient budget for rare positives
    # like DoS (377 train samples). Default since v6; BCE was used before that.
    loss_fn:        str   = "asl"
    focal_gamma:    float = 2.0
    focal_alpha:    float = 0.25
    # ASL hyperparameters (only used when loss_fn="asl")
    asl_gamma_neg:  float = 2.0   # focus exponent for negatives; reduced from 4.0 (BUG-C4: γ⁻=4 caused all-zeros collapse with 60% zero-label rows)
    asl_gamma_pos:  float = 1.0   # focus exponent for positives (mild — less than gamma_neg)
    asl_clip:       float = 0.01  # probability margin; negatives with p<clip → zero gradient; reduced from 0.05 (BUG-M2: hard boundary caused oscillation at p≈0.03–0.06)
    # RC3 fix (2026-05-16): label smoothing prevents extreme overconfidence.
    # Without smoothing the model pushes Reentrancy → 0.97 on safe contracts
    # with zero penalty.  ε=0.05 sets soft targets: positive→0.95, negative→0.05.
    label_smoothing: float = 0.0   # replaced by per-class smoothing below (BUG-M9)

    # --- Per-class label smoothing (BUG-M9) ---
    # Calibrated to confirmed/estimated noise rates per class.
    # Applied as: labels[:,c] = labels[:,c]*(1-eps[c]) + 0.5*eps[c]
    # Replaces uniform label_smoothing=0.05.
    class_label_smoothing: dict = field(default_factory=lambda: {
        "CallToUnknown":               0.10,
        "DenialOfService":             0.18,
        "ExternalBug":                 0.10,
        "GasException":                0.12,
        "IntegerUO":                   0.08,
        "MishandledException":         0.12,
        "Reentrancy":                  0.14,  # confirmed 14% noise (no external calls)
        "Timestamp":                   0.05,  # structural check exists; lower noise
        "TransactionOrderDependence":  0.10,
        "UnusedReturn":                0.10,
    })

    # --- DoS loss weight (BUG-H6) ---
    # 0.0 = no gradient for DoS column (original setting when DoS had 3 samples).
    # 0.5 = half gradient — safe starting point with ~243 training positives.
    # 1.0 = full gradient (normal).
    # Fractional values scale the gradient contribution, not a hard mask.
    # NUM_CLASSES stays 10 (ZKML proxy MLP is hardcoded to 10 outputs — LOCKED).
    dos_loss_weight: float = 0.5

    # --- pos_weight cap ---
    # Classes with >= pos_weight_min_samples training positives are NOT amplified.
    # Reentrancy has ~3500 train positives and is not actually rare; giving it a
    # 2.82× FN penalty + BCCC external-call co-occurrence is the primary driver of
    # the behavioral collapse seen in v5.2. Setting min_samples=3000 clamps
    # Reentrancy, GasException, and MishandledException to 1.0 while leaving
    # DoS (257), Timestamp (~1500), and minority classes at their sqrt-scaled weights.
    pos_weight_cap: float = 10.0  # M-1/H-4: cap on sqrt-scaled pos_weight; was 20.0
    pos_weight_min_samples: int = 3000  # classes with ≥3000 train positives get pos_weight=1.0 (BUG-H3: Reentrancy 2.82× amplification dominated gradient)

    # --- GNN prefix injection (Phase 1) ---
    # gnn_prefix_k=0 disables prefix entirely — identical to original model.
    # During warmup epochs the prefix is suppressed (None path); projection trains
    # from random init starting at epoch gnn_prefix_warmup_epochs.
    gnn_prefix_k:             int   = 0     # 0 = disabled; 48 for Phase 1
    gnn_prefix_warmup_epochs: int   = 15   # epochs without prefix
    gnn_prefix_proj_lr_mult:       float = 5.0   # NH-5: raised from 1.0; cold-start needs faster LR
    gnn_prefix_proj_reset_on_warmup: bool = True  # NC-1: reset Adam state for proj at ep=warmup_epoch
    jk_entropy_reg_lambda:         float = 0.005  # C-3: JK entropy regularizer weight (0=disabled); 0.01 forced uniform 33/33/33 in Run 3

    # --- Cache ---
    cache_path: str | None = "ml/data/cached_dataset_deduped.pkl"

    # --- Logging ---
    log_interval: int = 100
    # [A37] Threshold sweep is expensive (~19×num_classes evals). Sweep every N epochs
    # and always on the final epoch; reuse cached thresholds in between.
    threshold_tune_interval: int = 10

    # --- Structured logging (Phase 4.6) ---
    # Directory for step_metrics.jsonl / epoch_summary.jsonl / alerts.jsonl.
    # Defaults to ml/logs/<run_name>. Gate 4.6: all three files must exist before ep 1.
    log_dir: str | None = None

    # --- MLflow ---
    experiment_name: str = "sentinel-multilabel"
    run_name:        str = "sentinel-run"

    # --- Device ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # --- Resume ---
    resume_from:           str | None = None
    resume_model_only:     bool       = True
    force_optimizer_reset: bool       = False

    # --- Autoresearch harness knobs ---
    smoke_subsample_fraction: float = 1.0
    use_weighted_sampler:     str   = "positive"  # "positive"=3× weight on any-vuln rows; "DoS-only"; "all-rare"; "none" (BUG-H10: 60% zero-label rows trained at natural frequency)

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
        if self.gnn_layers > 8:
            logger.warning(
                f"gnn_layers={self.gnn_layers} is non-standard. "
                "v8+IMP uses gnn_layers=8 (2+3+3 per phase — IMP-G3 downward CONTAINS pass). "
                "Extra layers beyond 7 receive Phase 1 (structural) edge masking by default."
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps={self.gradient_accumulation_steps} must be >= 1."
            )
        unknown_cls = set(self.class_label_smoothing) - set(CLASS_NAMES)
        if unknown_cls:
            raise ValueError(
                f"NH-2: class_label_smoothing contains unknown class names: {unknown_cls}. "
                f"Valid classes: {CLASS_NAMES}"
            )
        invalid_eps = {k: v for k, v in self.class_label_smoothing.items() if not (0.0 <= v < 1.0)}
        if invalid_eps:
            raise ValueError(
                f"NH-2: class_label_smoothing values must be in [0, 1): invalid entries: {invalid_eps}"
            )


# ---------------------------------------------------------------------------
# pos_weight computation
# ---------------------------------------------------------------------------
def compute_pos_weight(
    train_dataset:          "DualPathDataset",
    num_classes:            int,
    device:                 str,
    pos_weight_min_samples: int   = 0,
    pos_weight_cap:         float = 20.0,
) -> torch.Tensor:
    # [A36] Compute from in-memory label map — no CSV I/O on every call.
    # train_dataset.paired_hashes is already filtered to training indices.
    if train_dataset._label_map is None:
        raise RuntimeError(
            "compute_pos_weight requires a dataset in multi-label mode (label_csv must be set)."
        )
    train_labels = np.stack([
        train_dataset._label_map[h].numpy()
        for h in train_dataset.paired_hashes
        if h in train_dataset._label_map
    ])

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
            # Cap at pos_weight_cap: without a ceiling, severely data-starved classes (e.g. DoS
            # with 257 samples → raw_ratio≈120) produce unchecked gradient spikes that
            # destabilise the loss scale for the entire batch.
            pos_weight_vals.append(min(float(raw_ratio ** 0.5), pos_weight_cap))

    logger.info(f"pos_weight sqrt-scaled (min_samples={pos_weight_min_samples}) — training split only:")
    for name, pw in zip(CLASS_NAMES[:num_classes], pos_weight_vals):
        capped = " [capped=1.0]" if pw == 1.0 and pos_weight_min_samples > 0 else ""
        logger.info(f"  {name:<32} {pw:.2f}{capped}")

    return torch.tensor(pos_weight_vals, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model:           SentinelModel,
    loader:          DataLoader,
    device:          str,
    threshold:       float = 0.5,
    use_amp:         bool  = True,
    tune_thresholds: bool  = False,
) -> dict[str, float]:
    """Evaluate model on loader.

    When tune_thresholds=True (BUG-M8), sweeps 19 thresholds per class over
    [0.1, 0.9] and picks the one maximising each class's F1. Reports both
    fixed-threshold and tuned-threshold macro F1. Tuned thresholds are stored
    in metrics["tuned_thresholds"] (list[float] of length num_classes).
    """
    model.eval()
    all_probs      = []
    all_true       = []
    all_num_nodes  = []  # [5.3] per-sample node counts for size-stratified Timestamp eval

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False,
                          disable=not sys.stdout.isatty()):
            graphs, tokens, labels = batch

            graphs         = graphs.to(device)
            input_ids      = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            labels         = labels.to(device).float()

            # [5.3] Collect per-graph node counts before moving graphs to GPU output.
            _nn = torch.bincount(graphs.batch, minlength=graphs.num_graphs).cpu().numpy()
            all_num_nodes.append(_nn)

            with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(graphs, input_ids, attention_mask)

            probs = torch.sigmoid(logits.float())
            all_probs.append(probs.cpu().numpy())
            all_true.append(labels.long().cpu().numpy())

    y_true  = np.concatenate(all_true)
    y_probs = np.concatenate(all_probs)
    y_pred  = (y_probs >= threshold).astype(int)

    f1_macro     = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    f1_micro     = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    hamming      = hamming_loss(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None,     zero_division=0)

    metrics = {"f1_macro": f1_macro, "f1_micro": f1_micro, "hamming": hamming}
    for i, name in enumerate(CLASS_NAMES[:y_true.shape[1]]):
        metrics[f"f1_{name}"] = float(f1_per_class[i])
    # [Phase 4.6] Pass raw arrays to caller for AUC/Brier/ECE computation.
    metrics["_y_true"]  = y_true
    metrics["_y_probs"] = y_probs

    # [5.3] Size-stratified Timestamp F1 (EXP-L7 stratum boundaries: <100, 100–300, >300).
    _ts_idx = CLASS_NAMES.index("Timestamp") if "Timestamp" in CLASS_NAMES else None
    if _ts_idx is not None and _ts_idx < y_true.shape[1]:
        _node_counts = np.concatenate(all_num_nodes)
        _strata = {
            "small":  _node_counts < 100,
            "medium": (_node_counts >= 100) & (_node_counts <= 300),
            "large":  _node_counts > 300,
        }
        for _stratum, _mask in _strata.items():
            if _mask.sum() >= 2:
                _f1 = f1_score(
                    y_true[_mask, _ts_idx], y_pred[_mask, _ts_idx], zero_division=0
                )
                metrics[f"f1_Timestamp_{_stratum}"] = float(_f1)
                metrics[f"n_Timestamp_{_stratum}"]  = int(_mask.sum())

    if tune_thresholds:
        # BUG-M8: sweep 19 candidate thresholds per class; pick best per-class F1.
        _candidates = np.linspace(0.1, 0.9, 19)
        num_classes = y_true.shape[1]
        tuned = []
        for c in range(num_classes):
            best_t, best_f1 = threshold, 0.0
            for t in _candidates:
                preds_c = (y_probs[:, c] >= t).astype(int)
                f1_c = f1_score(y_true[:, c], preds_c, zero_division=0)
                if f1_c > best_f1:
                    best_f1, best_t = f1_c, t
            tuned.append(float(best_t))

        # Re-evaluate with per-class tuned thresholds
        y_pred_tuned = np.stack(
            [(y_probs[:, c] >= tuned[c]).astype(int) for c in range(num_classes)],
            axis=1,
        )
        f1_tuned = f1_score(y_true, y_pred_tuned, average="macro", zero_division=0)
        metrics["f1_macro_tuned"]  = float(f1_tuned)
        metrics["tuned_thresholds"] = tuned

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
    device:                      str,
    grad_clip:                   float,
    log_interval:                int,
    use_amp:                     bool,
    aux_loss_weight:             float = 0.3,
    gradient_accumulation_steps: int   = 1,
    label_smoothing:             float = 0.0,   # kept for backward compat; overridden by class_eps
    class_eps:                   "torch.Tensor | None" = None,  # [NUM_CLASSES] per-class smoothing (BUG-M9)
    dos_loss_weight:             float = 0.0,   # 0.0 = zero DoS gradient (BUG-H6)
    jk_entropy_reg_lambda:       float = 0.0,   # C-3: JK entropy regularizer weight
    aux_phase2_loss_weight:      float = 0.0,   # Interp-2: CEI-weighted Phase 2 aux loss
    slog: "StructuredLogger | None" = None,     # Phase 4.6: structured logger
    epoch: int = 0,                             # current epoch for structured logging
) -> dict:
    """Returns dict: avg_loss, nan_batch_count, last_gnn_share,
    epoch_main_loss, epoch_aux_loss, epoch_ph2_loss,
    last_grad_norm_total, grad_norm_max_layer, loss_spike_count, grad_zero_count."""
    model.train()
    total_loss = 0.0

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    accum_steps = max(1, gradient_accumulation_steps)

    optimizer_step = 0
    _gnn_collapse_streak = 0
    nan_loss_count = 0
    last_gnn_share = 0.0

    # Epoch-level loss accumulators (not reset at log_interval; accumulate full epoch).
    _epoch_main_sum = 0.0
    _epoch_aux_sum  = 0.0
    _epoch_ph2_sum  = 0.0
    _epoch_n        = 0
    _last_grad_norm_total: float = 0.0
    _last_grad_max_layer: tuple[str, float] = ("", 0.0)
    _last_grad_zero_count: int = 0
    _last_ph2_ph1_ratio: float = 0.0

    # Running sums for per-interval averaged loss logging (reset every log_interval).
    _run_main   = 0.0
    _run_gnn_a  = 0.0
    _run_tf_a   = 0.0
    _run_fus_a  = 0.0
    _run_ph2_a  = 0.0
    _run_n      = 0

    # CEI class indices — ExternalBug=2, Reentrancy=6, TOD=8
    _CEI_INDICES = [2, 6, 8]
    _interval_t0 = time.perf_counter()   # wall-clock start of current log interval

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False,
                disable=not sys.stdout.isatty())
    for batch_idx, batch in enumerate(pbar):
        graphs, tokens, labels = batch

        graphs         = graphs.to(device)
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        labels         = labels.to(device).float()

        if slog is not None:
            _skip = slog.check_batch(labels, graphs.x, batch_idx, epoch)
            if not _skip:
                _skip = slog.check_inputs(graphs.x, graphs.edge_index, batch_idx, epoch)
            if _skip:
                continue

        # Per-class label smoothing (BUG-M9): class_eps overrides uniform label_smoothing.
        if class_eps is not None:
            labels = labels * (1.0 - class_eps) + 0.5 * class_eps
        elif label_smoothing > 0.0:
            labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
            logits, aux = model(graphs, input_ids, attention_mask, return_aux=True)

            # DoS gradient scaling (BUG-H6): blend between detached (no gradient)
            # and full logit to scale the gradient contribution by dos_loss_weight.
            # w=0.0 → fully detached (no DoS gradient); w=1.0 → full gradient;
            # w=0.5 → 50% gradient. Predictions are unaffected (inference uses
            # the original logits, not _logits_for_loss).
            if dos_loss_weight < 1.0:
                _dos_idx = CLASS_NAMES.index("DenialOfService")
                _logits_for_loss = logits.clone()
                _logits_for_loss[:, _dos_idx] = (
                    dos_loss_weight * logits[:, _dos_idx]
                    + (1.0 - dos_loss_weight) * logits[:, _dos_idx].detach()
                )
            else:
                _logits_for_loss = logits

            main_loss   = loss_fn(_logits_for_loss, labels)
            # aux_loss_fn has no pos_weight — pathway heads give supervision signal
            # without amplifying rare-class imbalance through struggling aux heads.
            # Apply same DoS scaling to aux heads.
            if dos_loss_weight < 1.0:
                _aux_masked = {}
                for _k, _v in aux.items():
                    if _k == "jk_entropy":
                        _aux_masked[_k] = _v  # scalar — no DoS column masking
                        continue
                    _vv = _v.clone()
                    _vv[:, _dos_idx] = (
                        dos_loss_weight * _v[:, _dos_idx]
                        + (1.0 - dos_loss_weight) * _v[:, _dos_idx].detach()
                    )
                    _aux_masked[_k] = _vv
            else:
                _aux_masked = aux
            loss_gnn_a  = aux_loss_fn(_aux_masked["gnn"],         labels)
            loss_tf_a   = aux_loss_fn(_aux_masked["transformer"], labels)
            loss_fus_a  = aux_loss_fn(_aux_masked["fused"],       labels)
            aux_loss    = loss_gnn_a + loss_tf_a + loss_fus_a
            # Interp-2: CEI-weighted Phase 2 auxiliary loss.
            # ExternalBug/Reentrancy/TOD get 3× weight to force Phase 2 (control-flow)
            # embeddings to encode CEI-pattern information directly.
            # Only computed when aux_phase2_loss_weight > 0 and "phase2" key present.
            loss_phase2_a = torch.tensor(0.0, device=labels.device)
            if aux_phase2_loss_weight > 0.0 and "phase2" in _aux_masked:
                _cei_weight = torch.ones(labels.shape[-1], device=labels.device)
                _cei_weight[_CEI_INDICES] = 3.0
                _raw = aux_loss_fn(_aux_masked["phase2"], labels)  # scalar BCE
                # Recompute per-class BCE to apply CEI weighting.
                _per_class = torch.nn.functional.binary_cross_entropy_with_logits(
                    _aux_masked["phase2"], labels, reduction="none"
                ).mean(0)  # [NUM_CLASSES]
                loss_phase2_a = (_per_class * _cei_weight).mean()
            # Divide by actual window size, not the fixed accum_steps.
            # When len(loader) % accum_steps != 0 the last window has fewer
            # micro-batches; dividing by accum_steps under-scales that gradient
            # by (actual / accum_steps). Using actual_window_size keeps gradients
            # correctly normalised across all windows including the tail.
            _window_start      = (batch_idx // accum_steps) * accum_steps
            _actual_window     = min(accum_steps, len(loader) - _window_start)
            loss = (
                main_loss
                + aux_loss_weight * aux_loss
                + aux_phase2_loss_weight * loss_phase2_a
            ) / _actual_window
            # C-3: JK entropy regularizer — penalizes Phase 3 JK attention collapse.
            # Adds to loss: lambda * (log(K) - H) where H is the mean per-node entropy.
            # When one phase dominates (H≈0), penalty ≈ lambda*log(3)≈0.011; at uniform H=log(3), penalty=0.
            if jk_entropy_reg_lambda > 0.0:
                _jk_ent = aux.get("jk_entropy") if isinstance(aux, dict) else None
                if _jk_ent is not None:
                    import math
                    _H_max = math.log(3)
                    _jk_reg = jk_entropy_reg_lambda * (_H_max - _jk_ent.clamp(max=_H_max))
                    loss = loss + _jk_reg / _actual_window

        # A38: NaN guard — check finiteness BEFORE backward() to prevent Adam state
        # corruption from NaN/Inf gradients propagating into the optimizer's running
        # mean/variance estimates.  Previously the check was AFTER backward(), so
        # corrupted gradients had already updated optimizer state.
        loss_for_log = loss.item() * accum_steps
        if not torch.isfinite(loss).item():
            nan_loss_count += 1
            # Zero any stale gradients from earlier micro-batches in this accumulation
            # window so the next window starts from a clean state.
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix({"loss": f"{loss_for_log:.4f}", "nan": nan_loss_count})
            continue

        # §2.1/2.7 — check finite loss for spike detection (NaN already filtered above).
        if slog is not None:
            slog.check_loss(loss_for_log, batch_idx, epoch)

        # Accumulate per-eye loss for the upcoming log line (finite batches only).
        _run_main  += main_loss.item()
        _run_gnn_a += loss_gnn_a.item()
        _run_tf_a  += loss_tf_a.item()
        _run_fus_a += loss_fus_a.item()
        _run_ph2_a += loss_phase2_a.item()
        _run_n     += 1

        # Epoch-level accumulation for structured logger (full epoch averages).
        _epoch_main_sum += main_loss.item()
        _epoch_aux_sum  += (loss_gnn_a + loss_tf_a + loss_fus_a).item()
        _epoch_ph2_sum  += loss_phase2_a.item()
        _epoch_n        += 1

        loss.backward()

        is_last_batch = (batch_idx + 1 == len(loader))
        is_accum_step = ((batch_idx + 1) % accum_steps == 0) or is_last_batch

        if is_accum_step:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)

            # A38: Post-clip guard — BF16 overflow can produce non-finite gradients
            # even from a finite loss value.  Detect and skip optimizer.step() so
            # Adam's running statistics stay uncorrupted.
            _has_bad_grads = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in trainable_params
            )
            if _has_bad_grads:
                logger.warning(
                    "[A38] Non-finite gradients after grad-clip (likely BF16 overflow)"
                    " — skipping optimizer.step() and zeroing gradients."
                )
                optimizer.zero_grad(set_to_none=True)
            else:
                # Fix #28: read grad norms after unscale_(), before zero_grad().
                optimizer_step += 1
                should_log = (optimizer_step % log_interval == 0)
                if should_log:
                    gnn_norm     = _grad_norm(model.gnn_eye_proj)
                    gnn_enc_norm = _grad_norm(model.gnn)  # C1: full GNN encoder backbone
                    tf_norm      = _grad_norm(model.transformer_eye_proj)
                    fused_norm   = _grad_norm(model.fusion)
                    _total_norm  = (gnn_norm**2 + tf_norm**2 + fused_norm**2) ** 0.5
                    _gnn_share   = gnn_norm / _total_norm if _total_norm > 1e-8 else 0.0
                    last_gnn_share = _gnn_share

                    # Phase 5.6: Phase 2 / Phase 1 gradient norm ratio.
                    # Phase 1 = conv1+conv2 (structural); Phase 2 = conv3+conv3b+conv3c (CFG/ICFG).
                    # Ratio near 0 → Phase 2 not receiving gradient signal (CEI patterns not learning).
                    _gnn_mod = getattr(model, "gnn", None)
                    if _gnn_mod is not None:
                        _ph1_sq = sum(
                            _grad_norm(getattr(_gnn_mod, _n)) ** 2
                            for _n in ("conv1", "conv2")
                            if hasattr(_gnn_mod, _n)
                        )
                        _ph2_sq = sum(
                            _grad_norm(getattr(_gnn_mod, _n)) ** 2
                            for _n in ("conv3", "conv3b", "conv3c")
                            if hasattr(_gnn_mod, _n)
                        )
                        _ph1_n = _ph1_sq ** 0.5
                        _ph2_n = _ph2_sq ** 0.5
                        _last_ph2_ph1_ratio = _ph2_n / _ph1_n if _ph1_n > 1e-8 else 0.0
                    else:
                        _ph2_n = _ph1_n = 0.0
                        _last_ph2_ph1_ratio = 0.0

                    n = max(1, _run_n)
                    _elapsed = time.perf_counter() - _interval_t0
                    _steps_in_interval = log_interval  # optimizer steps logged
                    _sps = _steps_in_interval / _elapsed if _elapsed > 0 else 0.0
                    _ph2_str = f" ph2={_run_ph2_a/n:.4f}" if aux_phase2_loss_weight > 0.0 else ""
                    logger.info(
                        f"  Step {optimizer_step}/{(len(loader) + accum_steps - 1) // accum_steps} "
                        f"(batch {batch_idx+1}/{len(loader)}) | "
                        f"loss={_run_main/n:.4f} "
                        f"[eyes: gnn={_run_gnn_a/n:.4f} tf={_run_tf_a/n:.4f} fused={_run_fus_a/n:.4f}{_ph2_str}] | "
                        f"grad: gnn={gnn_norm:.3f} gnn_enc={gnn_enc_norm:.3f} tf={tf_norm:.3f} fused={fused_norm:.3f} | "
                        f"GNN share={_gnn_share:.1%} Ph2/Ph1={_last_ph2_ph1_ratio:.2f} | "
                        f"{_sps:.2f} step/s ({100/_sps/60:.1f} min/100steps)"
                    )
                    # Reset running sums and interval timer for next interval.
                    _run_main = _run_gnn_a = _run_tf_a = _run_fus_a = _run_ph2_a = 0.0
                    _run_n = 0
                    _interval_t0 = time.perf_counter()

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
                            logger.info(f"  VRAM high: {_vram_str()} ({vpct:.1%} reserved)")

                    if slog is not None:
                        _gs_total, _gs_max_layer, _gs_zeros = compute_grad_stats(model)
                        _last_grad_norm_total = _gs_total
                        _last_grad_max_layer  = _gs_max_layer
                        _last_grad_zero_count = _gs_zeros
                        slog.check_grad_norm(_gs_total, optimizer_step, epoch)
                        slog.check_vram(optimizer_step, epoch)
                        _lr_step   = optimizer.param_groups[0]["lr"]
                        _vram_step = (torch.cuda.memory_allocated() / 1024**2) if device == "cuda" else 0.0
                        slog.log_step({
                            "step":              optimizer_step,
                            "epoch":             epoch,
                            "loss":              loss_for_log,
                            "main_loss":         main_loss.item(),
                            "aux_loss":          (loss_gnn_a + loss_tf_a + loss_fus_a).item(),
                            "phase2_loss":       loss_phase2_a.item(),
                            "grad_norm_total":   _gs_total,
                            "gnn_share":         last_gnn_share,
                            "ph2_ph1_grad_ratio": _last_ph2_ph1_ratio,
                            "lr":                _lr_step,
                            "vram_mb":           _vram_step,
                        })

                optimizer.step()
                if slog is not None and optimizer_step % 50 == 0:
                    slog.check_parameters(model, optimizer_step, epoch)
                    slog.check_adam_state(optimizer, optimizer_step, epoch)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss_for_log
        pbar.set_postfix({"loss": f"{loss_for_log:.4f}", "nan": nan_loss_count})

    n_batches = len(loader)
    if n_batches == 0:
        logger.warning("Empty train loader — returning 0.0 loss")
        return {
            "avg_loss": 0.0, "nan_batch_count": 0, "last_gnn_share": 0.0,
            "epoch_main_loss": 0.0, "epoch_aux_loss": 0.0, "epoch_ph2_loss": 0.0,
            "last_grad_norm_total": 0.0, "grad_norm_max_layer": ("", 0.0),
            "loss_spike_count": 0, "grad_zero_count": 0, "ph2_ph1_grad_ratio": 0.0,
        }

    if nan_loss_count > 0:
        _nan_rate = nan_loss_count / max(1, n_batches)
        logger.warning(
            f"[A38] NaN/Inf loss or bad-grad skips: {nan_loss_count}/{n_batches} "
            f"batches ({_nan_rate:.1%}) this epoch."
        )
        if _nan_rate > 0.005:
            logger.error(
                f"[A38] NaN rate {_nan_rate:.1%} exceeds Gate 0.2 threshold (0.5%) — "
                "HALT training immediately and investigate. "
                "Check LR, BF16 overflow, and data quality. "
                "Restart from the last clean checkpoint, NOT the current one."
            )
            if slog is not None:
                slog.alert(KILL,
                    f"[A38] NaN rate {_nan_rate:.1%} > 0.5% — aborting training.",
                    {"nan_rate": _nan_rate, "nan_count": nan_loss_count, "n_batches": n_batches},
                )

    _n_valid     = max(1, n_batches - nan_loss_count)
    _loss_spikes = slog._loss_spike_count if slog is not None else 0
    return {
        "avg_loss":             total_loss / _n_valid,
        "nan_batch_count":      nan_loss_count,
        "last_gnn_share":       last_gnn_share,
        "epoch_main_loss":      _epoch_main_sum / max(1, _epoch_n),
        "epoch_aux_loss":       _epoch_aux_sum  / max(1, _epoch_n),
        "epoch_ph2_loss":       _epoch_ph2_sum  / max(1, _epoch_n),
        "last_grad_norm_total": _last_grad_norm_total,
        "grad_norm_max_layer":  _last_grad_max_layer,
        "loss_spike_count":     _loss_spikes,
        "grad_zero_count":      _last_grad_zero_count,
        "ph2_ph1_grad_ratio":   _last_ph2_ph1_ratio,
    }


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
        if mode == "positive":
            # 3× weight for any row with at least one positive label.
            # Shifts effective positive/negative ratio from ~40/60 to ~60/40
            # without modifying labels. (BUG-H10 fix)
            has_vuln = any(float(row.get(cls, 0)) == 1.0 for cls in CLASS_NAMES)
            w = 3.0 if has_vuln else 1.0
        elif mode == "DoS-only":
            w = 39.0 if float(row.get("DenialOfService", 0)) == 1.0 else 1.0
        elif mode == "all-rare":
            n_pos = sum(float(row.get(cls, 0)) for cls in CLASS_NAMES)
            w = float(n_pos) if n_pos > 0 else 1.0
        elif mode == "timestamp-size":
            # E2 (Interp-3): Timestamp size shortcut mitigation.
            # Timestamp-positive contracts with >150 nodes get 4× weight;
            # large negatives get 0.5× weight. Encourages model to learn
            # genuine block.timestamp-in-branch patterns on large contracts.
            # Threshold 150 matches EXP-L7 "large" stratum boundary.
            _TIMESTAMP_IDX = CLASS_NAMES.index("Timestamp")
            _LARGE_THRESHOLD = 150
            is_ts_pos = float(row.get("Timestamp", 0)) == 1.0
            num_nodes = 0
            if dataset.cached_data is not None and md5 in dataset.cached_data:
                _entry = dataset.cached_data[md5]
                if isinstance(_entry, tuple) and len(_entry) >= 1:
                    _graph = _entry[0]
                    num_nodes = int(getattr(_graph, "num_nodes", 0))
            is_large = num_nodes > _LARGE_THRESHOLD
            if is_ts_pos and is_large:
                w = 4.0  # oversample hard Timestamp+ cases
            elif not is_ts_pos and is_large:
                w = 0.5  # undersample large negatives that model already predicts correctly
            else:
                w = 1.0
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

    # Load the cache once and share the dict between train and val datasets.
    # Both datasets cover disjoint subsets of the same cache — there is no
    # correctness reason to load it twice. Sharing halves cache RAM usage
    # (2.28 GB → once instead of twice = saves ~2.28 GB in the main process).
    _shared_cache: dict | None = None
    if cache_path is not None and cache_path.exists():
        import pickle
        logger.info(f"Loading shared cache: {cache_path} ...")
        with open(cache_path, "rb") as _f:
            _shared_cache = pickle.load(_f)
        logger.info(f"Shared cache loaded: {len(_shared_cache)} entries")

    logger.info("Creating training dataset...")
    train_dataset = DualPathDataset(
        graphs_dir=config.graphs_dir,
        tokens_dir=config.tokens_dir,
        indices=train_indices.tolist(),
        label_csv=label_csv_path,
        cache_path=None,  # shared below
    )
    if _shared_cache is not None:
        train_dataset.cached_data = _shared_cache
    logger.info(f"Train dataset cache loaded: {train_dataset.cached_data is not None}")

    logger.info("Creating validation dataset...")
    val_dataset = DualPathDataset(
        graphs_dir=config.graphs_dir,
        tokens_dir=config.tokens_dir,
        indices=val_indices.tolist(),
        label_csv=label_csv_path,
        cache_path=None,  # shared below
    )
    if _shared_cache is not None:
        val_dataset.cached_data = _shared_cache

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
            prefetch_factor=4,
            # fork workers inherit the parent's shared cache via copy-on-write —
            # no 2.28 GB copy per worker (safe: workers never call CUDA)
            multiprocessing_context="fork",
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
        # [A36] Use in-memory dataset labels — no redundant CSV read.
        pos_weight = compute_pos_weight(train_dataset, config.num_classes, device,
                                        pos_weight_min_samples=config.pos_weight_min_samples,
                                        pos_weight_cap=config.pos_weight_cap)
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
        gnn_use_jk=config.gnn_use_jk,
        gnn_jk_mode=config.gnn_jk_mode,
        gnn_phase2_edge_types=config.gnn_phase2_edge_types,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        gnn_prefix_k=config.gnn_prefix_k,
        gnn_prefix_warmup_epochs=config.gnn_prefix_warmup_epochs,
        fusion_max_nodes=config.fusion_max_nodes,
    ).to(device)

    # C-1: Verify GNN conv layers are float32 (BF16 global dtype pollution check).
    # The DTYPE FIX in transformer_encoder.py restores torch.default_dtype after BERT load.
    # This assertion catches any regression where GNN parameters are created as BF16.
    _gnn_dtype = next(model.gnn.conv1.parameters()).dtype
    if _gnn_dtype != torch.float32:
        raise RuntimeError(
            f"C-1: GNN conv1 parameters are {_gnn_dtype} (expected float32). "
            "BF16 global dtype pollution likely — check transformer_encoder.py DTYPE FIX."
        )
    logger.info(f"C-1: GNN dtype check passed — conv1 params are {_gnn_dtype}.")

    start_epoch      = 1
    best_f1          = 0.0
    patience_counter = 0
    _ckpt_state: dict | None = None

    if config.resume_from:
        logger.info(f"Resuming from: {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)  # LoRA peft objects not in safe globals
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
                f"New parameters (v7: conv3c 3rd CF hop, 11-dim schema) "
                f"will be randomly initialised — OK for fresh v7.0 training start, "
                f"NOT for a true resume of the same model."
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
        # [A35] _FocalFromLogits now defined at module level (picklable, DDP-safe).
        loss_fn: nn.Module = _FocalFromLogits(FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha))
        logger.info(f"Loss: FocalLoss(gamma={config.focal_gamma}, alpha={config.focal_alpha})")
    elif config.loss_fn == "asl":
        # pos_weight intentionally NOT passed to ASL. ASL handles class imbalance
        # via asymmetric gamma (gamma_neg > gamma_pos). Adding pos_weight on top
        # creates double-amplification: DoS pos_weight=10× combined with
        # ASL's asymmetric gradient scaling produced ~20,000× signal for 243 DoS
        # positives vs easy negatives — GNN share collapsed to 24% by ep16 (Run 3).
        loss_fn = AsymmetricLoss(
            gamma_neg=config.asl_gamma_neg,
            gamma_pos=config.asl_gamma_pos,
            clip=config.asl_clip,
        )
        logger.info(
            f"Loss: AsymmetricLoss(gamma_neg={config.asl_gamma_neg}, "
            f"gamma_pos={config.asl_gamma_pos}, clip={config.asl_clip}) — "
            f"pos_weight NOT applied (ASL is self-contained for imbalance)"
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
    _gnn_params:        list = []
    _lora_params:       list = []
    _fusion_params:     list = []
    _prefix_proj_params: list = []
    _other_params:      list = []
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
        elif (
            _pname.startswith("gnn_to_bert_proj.")
            or _pname.startswith("prefix_type_embedding.")
        ):
            _prefix_proj_params.append(_p)
        else:
            _other_params.append(_p)

    _gnn_lr         = config.lr * config.gnn_lr_multiplier
    _lora_lr        = config.lr * config.lora_lr_multiplier
    _fusion_lr      = config.lr * config.fusion_lr_multiplier
    _prefix_proj_lr = config.lr * config.gnn_prefix_proj_lr_mult
    _other_lr       = config.lr

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
    if _prefix_proj_params:
        _param_groups.append({"params": _prefix_proj_params, "lr": _prefix_proj_lr, "name": "prefix_proj"})
        _max_lrs.append(_prefix_proj_lr)
    if _other_params:
        _param_groups.append({"params": _other_params,  "lr": _other_lr})
        _max_lrs.append(_other_lr)

    logger.info(
        f"Optimizer param groups: "
        f"GNN={len(_gnn_params)} params (lr×{config.gnn_lr_multiplier}) | "
        f"LoRA={len(_lora_params)} params (lr×{config.lora_lr_multiplier}) | "
        f"Fusion={len(_fusion_params)} params (lr×{config.fusion_lr_multiplier}) | "
        f"PrefixProj={len(_prefix_proj_params)} params (lr×{config.gnn_prefix_proj_lr_mult}) | "
        f"Other={len(_other_params)} params (lr×1.0)"
    )
    # [NF-9] fused=True crashes on CPU; gate it to CUDA-only.
    _use_fused = device == "cuda" or device.startswith("cuda:")
    optimizer = AdamW(_param_groups, weight_decay=config.weight_decay, fused=_use_fused)

    # torch.compile is applied AFTER the optimizer so that param group name
    # matching ("gnn.", "lora_", "fusion.") uses the original uncompiled model
    # names. torch.compile wraps only the forward pass — the underlying parameter
    # tensors are the same objects, so the optimizer's references remain valid.
    if config.use_compile:
        try:
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.capture_scalar_outputs = True
            # Raise the per-module cache limit so dynamo can hold shape variants
            # for the GNN (variable node/edge counts per batch) without falling
            # back to eager after the default 8 compilations.
            torch._dynamo.config.cache_size_limit = 256
            torch._dynamo.config.accumulated_cache_size_limit = 256
            # Compile submodules individually, skipping model.transformer (CodeBERT+LoRA).
            # CodeBERT's HuggingFace forward has Python-level control flow that causes
            # graph breaks contaminating the GNN/fusion compile context when the whole
            # model is compiled. Submodule compilation isolates those breaks.
            for name in ("gnn", "fusion", "classifier",
                         "gnn_eye_proj", "transformer_eye_proj", "window_pooler",
                         "aux_gnn", "aux_transformer", "aux_fused"):  # H5: aux_fused was missing
                sub = getattr(model, name, None)
                if sub is not None:
                    setattr(model, name, torch.compile(sub, dynamic=True))
            logger.info(
                "torch.compile enabled on submodules (GNN/fusion/classifier/aux; "
                "transformer skipped — isolates CodeBERT graph breaks)"
            )
        except Exception as e:
            logger.warning(f"torch.compile failed, running eager: {e}")

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
    del _ckpt_state

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
            "jk_entropy_reg_lambda":       config.jk_entropy_reg_lambda,
        }
        if pos_weight is not None:
            for name, pw in zip(CLASS_NAMES[:config.num_classes], pos_weight.cpu().tolist()):
                params[f"pos_weight_{name}"] = round(pw, 3)
        mlflow.log_params(params)

        checkpoint_path = _checkpoint_path   # already computed above
        final_epoch = start_epoch - 1

        # [Phase 4.6] Create StructuredLogger — Gate 4.6 requires all three
        # JSONL files to exist before epoch 1 begins.
        _slog_dir = config.log_dir or f"ml/logs/{config.run_name}"
        _slog = StructuredLogger(_slog_dir)
        _slog.log_startup(
            dataset_paths=[
                Path(config.graphs_dir),
                Path(config.tokens_dir),
                Path(config.label_csv) if config.label_csv else Path("."),
            ],
            archive_dir=Path("ml/data/archive") if Path("ml/data/archive").exists() else None,
        )
        logger.info(f"[Phase 4.6] StructuredLogger active → {_slog_dir}")

        # Build per-class label smoothing tensor once (BUG-M9).
        _class_eps = torch.tensor(
            [config.class_label_smoothing.get(c, 0.05) for c in CLASS_NAMES[:config.num_classes]],
            dtype=torch.float32, device=device,
        )

        # Guardrail counters (BUG-M10).
        _consecutive_allzeros  = 0
        _consecutive_gnn_coll  = 0
        _class_death_counter   = [0] * config.num_classes

        # [A37] Cache per-class tuned thresholds between sweep epochs.
        _cached_tuned_thresholds: list[float] | None = None

        for epoch in range(start_epoch, config.epochs + 1):
            final_epoch = epoch
            logger.info(f"\n{'='*60}\nEpoch {epoch}/{config.epochs}\n{'='*60}")
            _slog.reset_epoch_counters()

            # Inform the model of the current epoch so it can apply the prefix
            # warmup suppression (gnn_prefix_k > 0 only; no-op otherwise).
            model._current_epoch = epoch

            # ── GNN prefix injection status logging ──
            if config.gnn_prefix_k > 0:
                _prefix_active = (epoch >= config.gnn_prefix_warmup_epochs)
                _prefix_status = "ACTIVE" if _prefix_active else f"WARMUP (starts ep{config.gnn_prefix_warmup_epochs})"
                logger.info(f"  GNN prefix K={config.gnn_prefix_k}: {_prefix_status}")
                mlflow.log_metric("prefix_active", int(_prefix_active), step=epoch)

                # Weight norm of gnn_to_bert_proj: constant during warmup (proj not called),
                # begins drifting from random init at epoch gnn_prefix_warmup_epochs.
                _proj = getattr(model, "gnn_to_bert_proj", None)
                if _proj is not None:
                    _proj_norm = _proj.weight.data.norm().item()
                    logger.info(f"  gnn_to_bert_proj weight norm: {_proj_norm:.4f}")
                    mlflow.log_metric("prefix_proj_weight_norm", _proj_norm, step=epoch)

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

            # NC-1: Reset Adam optimizer state for prefix_proj at warmup transition.
            # gnn_to_bert_proj received no gradient during warmup (prefix suppressed);
            # its Adam m1/m2 are stale near-zero. Fresh state gives a clean cold start.
            # NOTE: params with zero gradient during warmup never get Adam state
            # initialized — optimizer.state[p] won't exist. We initialize them to {}
            # regardless (clearing existing state OR ensuring a fresh entry at step 0).
            if (
                epoch == config.gnn_prefix_warmup_epochs
                and config.gnn_prefix_k > 0
                and config.gnn_prefix_proj_reset_on_warmup
            ):
                for _pg in optimizer.param_groups:
                    if _pg.get("name") == "prefix_proj":
                        _n_params = len(_pg["params"])
                        _n_had_state = sum(
                            1 for _p in _pg["params"] if _p in optimizer.state
                        )
                        for _p in _pg["params"]:
                            optimizer.state[_p] = {}  # clear or init fresh
                        logger.info(
                            f"NC-1: Reset Adam state for {_n_params} prefix_proj params "
                            f"at warmup transition (ep{epoch}). "
                            f"{_n_had_state} had existing state (rest were zero-grad during warmup)."
                        )
                        mlflow.log_metric("prefix_proj_adam_reset", 1, step=epoch)
                        break

            _epoch_t0 = time.perf_counter()
            try:
                _epoch_stats = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    aux_loss_fn=aux_loss_fn,
                    scheduler=scheduler,
                    device=device,
                    grad_clip=config.grad_clip,
                    log_interval=_auto_log_interval,
                    use_amp=config.use_amp,
                    aux_loss_weight=effective_aux_weight,
                    gradient_accumulation_steps=accum_steps,
                    label_smoothing=config.label_smoothing,
                    class_eps=_class_eps,
                    dos_loss_weight=config.dos_loss_weight,
                    jk_entropy_reg_lambda=config.jk_entropy_reg_lambda,
                    aux_phase2_loss_weight=config.aux_phase2_loss_weight,
                    slog=_slog,
                    epoch=epoch,
                )
            except TrainingAbortError:
                _slog.close()
                raise
            train_loss      = _epoch_stats["avg_loss"]
            nan_batch_count = _epoch_stats["nan_batch_count"]
            last_gnn_share  = _epoch_stats["last_gnn_share"]

            # [A37] Threshold sweep every threshold_tune_interval epochs + always at the final epoch.
            # Between sweeps, reuse cached thresholds to save the 19×C eval cost.
            _is_final_epoch = (epoch == config.epochs) or (patience_counter + 1 >= config.early_stop_patience)
            _should_tune = (epoch % config.threshold_tune_interval == 0) or _is_final_epoch or (_cached_tuned_thresholds is None)
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                threshold=config.eval_threshold,   # training-time threshold (lower = stable signal)
                use_amp=config.use_amp,
                tune_thresholds=_should_tune,
            )
            if _should_tune and "tuned_thresholds" in val_metrics:
                _cached_tuned_thresholds = val_metrics["tuned_thresholds"]
            elif not _should_tune and _cached_tuned_thresholds is not None:
                # Re-inject cached thresholds so downstream code sees them every epoch.
                val_metrics["tuned_thresholds"] = _cached_tuned_thresholds

            # Fix #27: release CUDA caching allocator free-blocks between epochs.
            gc.collect()
            torch.cuda.empty_cache()

            mlflow.log_metric("train_loss",         train_loss,                              step=epoch)
            mlflow.log_metric("nan_batch_count",    nan_batch_count,                         step=epoch)  # Phase 2-B3
            mlflow.log_metric("gnn_grad_share",     last_gnn_share,                          step=epoch)
            mlflow.log_metric("ph2_ph1_grad_ratio", _epoch_stats["ph2_ph1_grad_ratio"],      step=epoch)  # Phase 5.6

            # Phase 2-C1 (2026-05-14): JK attention weight logging.
            # Reads cached per-phase mean weights from the last training batch.
            # Metric names: jk_phase1_weight, jk_phase2_weight, jk_phase3_weight.
            # Gate: all three should be > 5% after smoke run (v5.2 success criterion).
            if config.gnn_use_jk and hasattr(model.gnn, "jk") and model.gnn.jk is not None:
                _jk_cache = getattr(model.gnn.jk, "last_weights", None)
                _jk_std_cache = getattr(model.gnn.jk, "last_weight_stds", None)
                if _jk_cache is not None:
                    _jk_w = _jk_cache.cpu().tolist()   # [3] mean weights
                    _jk_s = _jk_std_cache.cpu().tolist() if _jk_std_cache is not None else [0.0] * len(_jk_w)
                    for _pi, (_w, _s) in enumerate(zip(_jk_w, _jk_s), start=1):
                        mlflow.log_metric(f"jk_phase{_pi}_weight", _w, step=epoch)
                        mlflow.log_metric(f"jk_phase{_pi}_std",    _s, step=epoch)
                    logger.info(
                        f"  JK attention weights — "
                        f"Phase1={_jk_w[0]:.3f}±{_jk_s[0]:.3f} "
                        f"Phase2={_jk_w[1]:.3f}±{_jk_s[1]:.3f} "
                        f"Phase3={_jk_w[2]:.3f}±{_jk_s[2]:.3f}"
                    )
                    # std collapse alert: all stds < 0.015 → per-node routing has collapsed.
                    # Threshold was 0.05 but Run4 STDs (0.020–0.044) are healthy and fired every epoch.
                    if max(_jk_s) < 0.015 and epoch >= 3:
                        logger.warning(
                            f"  ⚠ JK STD COLLAPSE: all per-phase stds < 0.015 "
                            f"(max={max(_jk_s):.3f}). Per-node routing has collapsed to "
                            "a global weight — the JK mean is the full story."
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

            # IMP-M2 Tier 2: prefix_attention_mean diagnostic — once per epoch when
            # prefix is active. Runs on the first val batch only (eval mode + no_grad).
            # Near-zero (< 0.002) for 5+ consecutive post-warmup epochs = transformer
            # ignoring prefix → investigate projection LR or reduce K.
            if (
                config.gnn_prefix_k > 0
                and epoch >= config.gnn_prefix_warmup_epochs
                and hasattr(model, "compute_prefix_attention_mean")
            ):
                try:
                    _diag_graphs, _diag_tokens, _ = next(iter(val_loader))
                    _diag_graphs = _diag_graphs.to(device)
                    _diag_ids    = _diag_tokens["input_ids"].to(device)
                    _diag_mask   = _diag_tokens["attention_mask"].to(device)
                    model.eval()
                    _prefix_attn = model.compute_prefix_attention_mean(
                        _diag_graphs, _diag_ids, _diag_mask
                    )
                    model.train()
                    if _prefix_attn is not None:
                        mlflow.log_metric("prefix_attention_mean", _prefix_attn, step=epoch)
                        logger.info(f"  prefix_attention_mean: {_prefix_attn:.6f}")
                        if _prefix_attn < 0.002:
                            logger.warning(
                                f"  ⚠ prefix_attention_mean={_prefix_attn:.6f} < 0.002 "
                                "— transformer may be ignoring GNN prefix tokens. "
                                "Monitor for 5+ consecutive epochs; consider reducing K or "
                                "increasing gnn_prefix_proj_lr_mult."
                            )
                except Exception as _e:
                    logger.warning(f"  prefix_attention_mean diagnostic failed: {_e}")

            mlflow.log_metric("val_f1_macro",       val_metrics["f1_macro"],       step=epoch)
            mlflow.log_metric("val_f1_micro",       val_metrics["f1_micro"],       step=epoch)
            mlflow.log_metric("val_hamming",        val_metrics["hamming"],        step=epoch)
            mlflow.log_metric("aux_loss_weight_effective", effective_aux_weight,   step=epoch)
            mlflow.log_metric("aux_phase2_loss_weight",    config.aux_phase2_loss_weight, step=epoch)
            # BUG-M8: log tuned macro F1 (per-class threshold sweep)
            if "f1_macro_tuned" in val_metrics:
                mlflow.log_metric("val_f1_macro_tuned", val_metrics["f1_macro_tuned"], step=epoch)
            for name in CLASS_NAMES[:config.num_classes]:
                mlflow.log_metric(f"val_f1_{name}", val_metrics[f"f1_{name}"], step=epoch)

            class_f1s = [(n, val_metrics[f"f1_{n}"]) for n in CLASS_NAMES[:config.num_classes]]
            class_f1s.sort(key=lambda x: x[1], reverse=True)
            top3    = " | ".join(f"{n}={v:.3f}" for n, v in class_f1s[:3])
            bottom3 = " | ".join(f"{n}={v:.3f}" for n, v in class_f1s[-3:])

            # ── Fix #34: log VRAM every epoch ──
            vram_info = f" | VRAM: {_vram_str()}" if device == "cuda" else ""

            _epoch_elapsed = time.perf_counter() - _epoch_t0
            logger.info(
                f"Epoch {epoch:>2}/{config.epochs} | "
                f"Loss={train_loss:.4f} | F1-macro={val_metrics['f1_macro']:.4f} | "
                f"Hamming={val_metrics['hamming']:.4f} | "
                f"{_epoch_elapsed/60:.1f} min/ep{vram_info}\n"
                f"  Top3:    {top3}\n"
                f"  Bottom3: {bottom3}"
            )

            # ── Training guardrails (BUG-M10) ────────────────────────────────
            # All-zeros collapse: Hamming >0.85 for 3+ epochs means the model
            # predicts all-zeros on everything (maximizes Hamming by being always
            # wrong in the worst possible way for this task).
            _hamming = val_metrics["hamming"]
            if _hamming > 0.85:
                _consecutive_allzeros += 1
                if _consecutive_allzeros >= 3:
                    logger.critical(
                        f"ALL-ZEROS COLLAPSE DETECTED: Hamming={_hamming:.4f} for "
                        f"{_consecutive_allzeros} consecutive epochs. "
                        "Model is predicting all-zeros. Consider reducing gamma_neg, "
                        "increasing dos_loss_weight, or checking the weighted sampler."
                    )
            else:
                _consecutive_allzeros = 0

            # Class death: any class with F1=0.0 for 5+ epochs.
            for _ci, _cname in enumerate(CLASS_NAMES[:config.num_classes]):
                _cf1 = val_metrics.get(f"f1_{_cname}", 0.0)
                if _cf1 == 0.0:
                    _class_death_counter[_ci] += 1
                    if _class_death_counter[_ci] >= 5:
                        logger.warning(
                            f"CLASS DEATH: {_cname} F1=0.0 for "
                            f"{_class_death_counter[_ci]} consecutive epochs."
                        )
                else:
                    _class_death_counter[_ci] = 0

            # GNN collapse: gnn_grad_share <10% sustained.
            if last_gnn_share < 0.10:
                _consecutive_gnn_coll += 1
                if _consecutive_gnn_coll >= 5:
                    logger.critical(
                        f"GNN COLLAPSE: gnn_grad_share={last_gnn_share:.3f} for "
                        f"{_consecutive_gnn_coll} consecutive epochs. "
                        "GNN is not contributing. Check LayerNorm, GNN LR multiplier."
                    )
            else:
                _consecutive_gnn_coll = 0

            # ── [Phase 4.6] Structured epoch logging ─────────────────────────
            try:
                _y_true  = val_metrics.get("_y_true")
                _y_probs = val_metrics.get("_y_probs")
                _cn      = CLASS_NAMES[:config.num_classes]

                _auc_metrics   = {}
                _brier_metrics = {}
                _ece_val       = 0.0
                _prob_stats    = {}
                _jk_ent        = 0.0
                _f1_auc_div    = False

                if _y_true is not None and _y_probs is not None:
                    _auc_metrics   = _slog.compute_auc_metrics(_y_true, _y_probs, _cn)
                    _brier_metrics = _slog.compute_brier(_y_true, _y_probs, _cn)
                    _ece_val       = _slog.compute_ece(_y_true, _y_probs)
                    _prob_stats    = _slog.compute_prob_stats(_y_probs, _cn)
                    _per_f1        = {n: val_metrics.get(f"f1_{n}", 0.0) for n in _cn}
                    _f1_auc_div    = _slog.check_f1_auc_divergence(
                        _per_f1, _auc_metrics.get("auc_roc_per_label", {}),
                        _auc_metrics.get("auc_roc_delta", {}), epoch,
                    )

                _jk_cache = getattr(getattr(model, "gnn", None), "jk", None)
                _jk_w_list = (
                    getattr(_jk_cache, "last_weights", None).cpu().tolist()
                    if _jk_cache is not None and getattr(_jk_cache, "last_weights", None) is not None
                    else []
                )
                _jk_ent = _slog.check_jk_entropy(_jk_w_list, epoch)

                _aux_norms = _slog.check_aux_head(model, epoch)
                _vram_peak = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
                _vram_cur  = (torch.cuda.memory_allocated()     / 1024**2) if torch.cuda.is_available() else 0.0

                _lr_now = optimizer.param_groups[0]["lr"]
                _per_f1_all = {n: val_metrics.get(f"f1_{n}", 0.0) for n in _cn}

                # Compute prediction entropy from val probabilities (binary entropy).
                _pred_entropy = 0.0
                if _y_probs is not None:
                    _p = np.clip(_y_probs, 1e-7, 1 - 1e-7)
                    _pred_entropy = float(np.mean(-_p * np.log(_p) - (1 - _p) * np.log(1 - _p)))

                # Compute val label distribution from y_true collected during evaluate().
                _label_dist_val = (
                    label_dist_from_tensor(torch.from_numpy(_y_true.astype(int)), _cn)
                    if _y_true is not None else {}
                )

                _summary = _slog.build_epoch_summary(
                    epoch              = epoch,
                    train_loss         = train_loss,
                    val_loss           = None,
                    main_loss          = _epoch_stats["epoch_main_loss"],
                    aux_loss           = _epoch_stats["epoch_aux_loss"],
                    total_loss         = train_loss,
                    lr                 = _lr_now,
                    grad_norm_total    = _epoch_stats["last_grad_norm_total"],
                    grad_norm_max_layer = _epoch_stats["grad_norm_max_layer"],
                    param_nan_count    = nan_batch_count,
                    grad_nan_count     = 0,
                    vram_peak_mb       = _vram_peak,
                    vram_current_mb    = _vram_cur,
                    label_dist_train   = {},
                    label_dist_val     = _label_dist_val,
                    aux_weight_norm    = _aux_norms.get("aux_weight_norm", 0.0),
                    aux_bias_norm      = _aux_norms.get("aux_bias_norm",   0.0),
                    jk_weight_entropy  = _jk_ent,
                    prediction_entropy = _pred_entropy,
                    per_class_f1       = _per_f1_all,
                    auc_metrics        = _auc_metrics,
                    brier_metrics      = _brier_metrics,
                    ece                = _ece_val,
                    temperature        = 1.0,
                    prob_dist_stats    = _prob_stats,
                    f1_auc_divergence  = _f1_auc_div,
                    epoch_duration_sec = _epoch_elapsed,
                    steps_per_epoch    = steps_per_epoch,
                    gpu_util_mean_pct  = 0.0,
                    loss_spike_count   = _epoch_stats["loss_spike_count"],
                    grad_zero_count    = _epoch_stats["grad_zero_count"],
                )
                _slog.log_epoch(_summary)
                _slog.check_vram(step=0, epoch=epoch)
            except Exception as _sl_err:
                logger.warning(f"[Phase 4.6] StructuredLogger epoch logging failed: {_sl_err}")
            # ─────────────────────────────────────────────────────────────────

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                patience_counter = 0
                _tmp_path = checkpoint_path.with_suffix(".tmp")
                # Strip torch.compile's ._orig_mod. infix from state dict keys so
                # every load path works identically whether the model was compiled or not.
                _sd = model.state_dict()
                if any("._orig_mod." in k for k in _sd):
                    _sd = {k.replace("._orig_mod.", "."): v for k, v in _sd.items()}
                torch.save(
                    {
                        "model":            _sd,
                        "optimizer":        optimizer.state_dict(),
                        "scheduler":        scheduler.state_dict(),
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
        _slog.close()

    return {
        "best_f1_macro":  best_f1,
        "final_epoch":    final_epoch,
        "early_stopped":  patience_counter >= config.early_stop_patience,
        "checkpoint_path": str(checkpoint_path),
    }


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
