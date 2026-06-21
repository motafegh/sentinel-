# ml/src/training — SENTINEL Training Loop

Training infrastructure: the main training loop, loss functions, and structured logging.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `trainer.py` | 1300+ | `TrainConfig`, `train()`, `train_one_epoch()`, `evaluate()`, weighted sampler |
| `training_logger.py` | 729 | `StructuredLogger` — three-stream JSONL logging with alert tiers |
| `losses.py` | 126 | `AsymmetricLoss` (ASL) — default loss function |
| `focalloss.py` | 143 | `FocalLoss`, `MultiLabelFocalLoss` — alternative loss functions |
| `__init__.py` | 0 | Empty |

---

## trainer.py

### TrainConfig (dataclass)

All training hyperparameters in a single dataclass. Key groups:

**Paths:** `export_dir`, `checkpoint_dir`, `checkpoint_name`

**Model:** `num_classes=10`, `fusion_output_dim=128`, `fusion_max_nodes=2048`

**GNN:** `gnn_hidden_dim=256`, `gnn_layers=8`, `gnn_heads=8`, `gnn_dropout=0.2`, `use_edge_attr=True`, `gnn_edge_emb_dim=64`, `gnn_use_jk=True`, `gnn_jk_mode='attention'`

**LoRA:** `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.1`, `lora_target_modules=["query","value"]`

**Training:** `epochs=100`, `batch_size=8`, `lr=2e-4`, `weight_decay=1e-2`, `grad_clip=1.0`, `warmup_pct=0.10`

**LR multipliers:** `gnn_lr_multiplier=2.5`, `lora_lr_multiplier=0.3`, `fusion_lr_multiplier=0.5`

**Loss:** `loss_fn="asl"`, `asl_gamma_neg=2.0`, `asl_gamma_pos=1.0`, `asl_clip=0.01`

**Auxiliary:** `aux_loss_weight=0.3`, `aux_phase2_loss_weight=0.20`, `aux_loss_warmup_epochs=8`

**Regularization:** `jk_entropy_reg_lambda=0.005`, `dos_loss_weight=0.5`

**Per-class label smoothing:** `class_label_smoothing` dict with calibrated noise rates per class.

**Gradient accumulation:** `gradient_accumulation_steps=8` (effective batch = 64)

**Prefix:** `gnn_prefix_k=0`, `gnn_prefix_warmup_epochs=15`, `gnn_prefix_proj_lr_mult=5.0`

**Validation:** `eval_threshold=0.35`, `early_stop_patience=30`, `threshold_tune_interval=10`

**Performance:** `use_amp=True`, `use_compile=True`, `num_workers=4`, `persistent_workers=True`

### train() function

Main training entry point. Handles:
- Checkpoint resume (full or model-only)
- OneCycleLR scheduler with warmup
- WeightedRandomSampler
- Per-class label smoothing tensor construction
- MLflow experiment tracking
- Early stopping on `f1_macro_tuned`
- Threshold tuning every `threshold_tune_interval` epochs
- NaN rate monitoring with KILL alert at >0.5%

### train_one_epoch()

Single epoch training with:
- AMP (BF16) autocast
- Gradient accumulation
- NaN loss guard (A38) — checks finiteness BEFORE backward
- Post-clip gradient guard — skips optimizer.step on non-finite gradients
- Per-interval logging (loss, grad norms, GNN share, Phase2/Phase1 ratio)
- GNN collapse detection (warns at <10% share for 3 intervals)
- Structured logging via `StructuredLogger`

### evaluate()

Evaluation with optional threshold tuning:
- Returns f1_macro, f1_micro, hamming_loss, per-class F1
- When `tune_thresholds=True`: sweeps 19 thresholds per class over [0.1, 0.9]
- Size-stratified Timestamp F1 (small/medium/large by node count)

### compute_pos_weight()

Computes sqrt-scaled pos_weight for BCE/ASL loss:
- Classes with >= `pos_weight_min_samples` (3000) positives get weight=1.0
- Others: `min(sqrt(N/pos), pos_weight_cap)` where cap=10.0

### _build_weighted_sampler()

WeightedRandomSampler with modes:
- `"positive"`: 3x weight for any-vuln rows
- `"timestamp-size"`: 4x for large Timestamp+ contracts, 0.5x for large negatives
- `"DoS-only"`: 39x for DoS+ rows
- `"all-rare"`: weight = number of positive labels

---

## training_logger.py

### StructuredLogger

Three-stream JSONL logger for training monitoring.

**Streams:**
- `step_metrics.jsonl` — per-step data (loss, lr, grad_norm, vram)
- `epoch_summary.jsonl` — 37-field epoch summary
- `alerts.jsonl` — WARN/KILL alerts with timestamps

**Alert tiers:**
- `KILL` — raises `TrainingAbortError` immediately (NaN loss/params/Adam state)
- `WARN_SKIP` — returns skip=True to caller (poisoned batch)
- `WARN` — log alert, continue training

**Per-step checks:**
- `check_batch()` — label distribution, NaN/Inf in inputs
- `check_inputs()` — feature dim, negative edge_index
- `check_loss()` — NaN/Inf, spike detection (>5x rolling mean)
- `check_parameters()` — NaN/Inf in model params
- `check_adam_state()` — NaN/Inf in exp_avg/exp_avg_sq
- `check_vram()` — warns at >7500 MB
- `check_grad_norm()` — warns on >100x rolling mean

**Per-epoch checks:**
- `check_aux_head()` — aux_phase2 weight/bias norms
- `check_jk_entropy()` — Shannon entropy of JK attention weights

**Calibration metrics:**
- `compute_auc_metrics()` — per-label and macro/micro AUC-ROC + AUC-PR
- `compute_brier()` — per-label Brier Score
- `compute_ece()` — Expected Calibration Error (10 bins)
- `check_f1_auc_divergence()` — flags F1 improving while AUC degrading

---

## losses.py

### AsymmetricLoss (ASL)

Default loss function. Applies different gamma exponents to positives and negatives independently.

**Parameters:**
- `gamma_neg=4.0` (focus on hard negatives; reduced to 2.0 in Run 12 config)
- `gamma_pos=1.0` (mild positive focus)
- `clip=0.05` (probability margin; reduced to 0.01 in Run 12 config)
- `pos_weight` optional tensor

**AMP safety:** Explicit `.float()` casts to prevent BF16 precision loss.

Supports per-class gamma/clip via Tensor inputs registered as buffers.

---

## focalloss.py

### FocalLoss

Standard focal loss. Expects POST-SIGMOID probabilities (not raw logits). Used with `_FocalFromLogits` wrapper in trainer.py.

**Parameters:** `gamma=2.0`, `alpha=0.25`

### MultiLabelFocalLoss

Accepts RAW LOGITS (applies sigmoid internally). Per-class alpha weights for imbalance correction.

**Note:** FocalLoss is an alternative to ASL, not used by default in Run 12.
