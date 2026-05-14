# ML-T2: Training Pipeline — Technical Reference

This document specifies the v5.2 training pipeline for the SENTINEL model. It covers every
configuration field, component, and monitoring mechanism required to understand or reproduce
a training run. For architecture details see `ML-T1-ARCHITECTURE.md`.

---

## Table of Contents

1. [Entry Points](#1-entry-points)
2. [TrainConfig — All Fields](#2-trainconfig--all-fields)
3. [Optimizer](#3-optimizer)
4. [Scheduler](#4-scheduler)
5. [Loss Function](#5-loss-function)
6. [Gradient Accumulation and AMP](#6-gradient-accumulation-and-amp)
7. [Dataset and DataLoader](#7-dataset-and-dataloader)
8. [Checkpoint Save and Resume](#8-checkpoint-save-and-resume)
9. [Per-Epoch and Per-Step Monitoring](#9-per-epoch-and-per-step-monitoring)
10. [Collapse and Dominance Detection](#10-collapse-and-dominance-detection)
11. [Run Commands](#11-run-commands)
12. [Smoke and Full-Run Gates](#12-smoke-and-full-run-gates)

---

## 1. Entry Points

| Item | Path |
|---|---|
| CLI entry point | `ml/scripts/train.py` |
| Config dataclass | `ml/src/training/trainer.py` — `TrainConfig` |
| Main training function | `train()` in `ml/src/training/trainer.py` |

`train.py` parses CLI arguments, constructs a `TrainConfig`, then calls `train(config)`.
Every CLI flag maps directly to a `TrainConfig` field via `argparse`; no separate config
file is required.

---

## 2. TrainConfig — All Fields

All fields shown with their default values. Overrides are passed as CLI flags using
`--kebab-case` equivalents (e.g. `--gradient-accumulation-steps 4`).

### 2.1 Paths

| Field | Default | Notes |
|---|---|---|
| `graphs_dir` | `ml/data/graphs` | Root directory of `.pt` graph files |
| `tokens_dir` | `ml/data/tokens` | Root directory of `.pt` token files |
| `splits_dir` | `ml/data/splits/deduped` | Directory containing `train.csv`, `val.csv`, `test.csv` |
| `checkpoint_dir` | `ml/checkpoints` | Where checkpoints are written |
| `checkpoint_name` | `multilabel-v5-fresh_best.pt` | Filename for best checkpoint |
| `label_csv` | `ml/data/processed/multilabel_index_deduped.csv` | 44,420-row canonical label index |
| `cache_path` | `ml/data/cached_dataset_deduped.pkl` | Pre-built RAM cache (built by `ml/scripts/create_cache.py`) |

### 2.2 Model

| Field | Default | Notes |
|---|---|---|
| `num_classes` | `10` | LOCKED — append-only |
| `fusion_output_dim` | `128` | LOCKED — ZKML proxy depends on this dimension |
| `fusion_dropout` | `0.3` | Dropout in `CrossAttentionFusion` |

### 2.3 GNN Architecture

| Field | Default | Notes |
|---|---|---|
| `gnn_hidden_dim` | `128` | Hidden and output dimension per phase |
| `gnn_layers` | `4` | Total GAT layers across 3 phases (2+1+1) |
| `gnn_heads` | `8` | Attention heads for phase-1 layers; phases 2 and 3 use 1 head |
| `gnn_dropout` | `0.2` | Dropout within GAT layers |
| `use_edge_attr` | `True` | Pass edge-type embeddings into GAT |
| `gnn_edge_emb_dim` | `32` | Output dimension of `Embedding(8, 32)` |
| `gnn_use_jk` | `True` | Enable Jumping Knowledge aggregation (v5.2 new) |
| `gnn_jk_mode` | `'attention'` | JK aggregation mode — softmax attention over 3 phase outputs (v5.2 new) |

### 2.4 LoRA

| Field | Default | Notes |
|---|---|---|
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA scaling factor |
| `lora_dropout` | `0.1` | Dropout on LoRA adapters |
| `lora_target_modules` | `["query", "value"]` | Applied to all 12 CodeBERT layers |

### 2.5 Training Hyperparameters

| Field | Default | Notes |
|---|---|---|
| `epochs` | `60` | Total training epochs |
| `batch_size` | `16` | Per-GPU micro-batch size |
| `lr` | `2e-4` | Base learning rate (applies to "other" param group) |
| `weight_decay` | `1e-2` | AdamW weight decay — applied to all param groups |
| `gnn_lr_multiplier` | `2.5` | Multiplier for GNN param group (v5.2 new); effective LR = 5e-4 |
| `lora_lr_multiplier` | `0.5` | Multiplier for LoRA param group (v5.2 new); effective LR = 1e-4 |
| `threshold` | `0.5` | Default classification threshold at validation time |
| `early_stop_patience` | `10` | Epochs without val F1-macro improvement before stopping |
| `aux_loss_weight` | `0.3` | Target weight for auxiliary head losses (λ = 0.3 in v5.1+) |
| `aux_loss_warmup_epochs` | `3` | Ramp auxiliary weight from 0 to 0.3 linearly over first 3 epochs |

### 2.6 Gradient and AMP

| Field | Default | Notes |
|---|---|---|
| `gradient_accumulation_steps` | `1` | Set to `4` on RTX 3070 → effective batch = 16 × 4 = 64 |
| `grad_clip` | `1.0` | Max gradient norm; clipped after `scaler.unscale_()` |
| `warmup_pct` | `0.10` | Fraction of total steps used as OneCycleLR warm-up (`pct_start`) |
| `use_amp` | `True` | Enable `torch.amp.GradScaler("cuda")` |

### 2.7 DataLoader

| Field | Default | Notes |
|---|---|---|
| `num_workers` | `2` | DataLoader worker processes |
| `persistent_workers` | `True` | Keep workers alive between epochs |

### 2.8 Loss Function

| Field | Default | Notes |
|---|---|---|
| `loss_fn` | `"bce"` | Primary loss; alternative: `"focal"` |
| `focal_gamma` | `2.0` | Focal loss focusing exponent |
| `focal_alpha` | `0.25` | Focal loss balance factor |

### 2.9 MLflow

| Field | Default | Notes |
|---|---|---|
| `experiment_name` | `"sentinel-multilabel"` | MLflow experiment; v5.2 runs use `sentinel-v5.2` |
| `run_name` | `"multilabel-v5-fresh"` | MLflow run name |

MLflow tracking URI: `sqlite:///mlruns.db` (relative to project root). Do not use
`file:///mlruns` — experiments 1, 2, 3 in that store are corrupt.

### 2.10 Resume

| Field | Default | Notes |
|---|---|---|
| `resume_from` | `None` | Path to checkpoint file to resume from |
| `resume_model_only` | `True` | If `True`, only model weights restored; optimizer/scheduler reset |
| `force_optimizer_reset` | `False` | Force optimizer reset even on full resume |

### 2.11 Miscellaneous

| Field | Default | Notes |
|---|---|---|
| `smoke_subsample_fraction` | `1.0` | Fraction of dataset to use; set `0.1` for smoke runs |
| `use_weighted_sampler` | `"none"` | Options: `"none"`, `"DoS-only"`, `"all-rare"` |
| `log_interval` | `100` | Gradient-norm logging fires every N optimizer steps |
| `device` | auto-detected | `"cuda"` if available, else `"cpu"` |

---

## 3. Optimizer

Three AdamW parameter groups are constructed by iterating `model.named_parameters()`:

| Group | Filter | LR Multiplier | Effective LR (base=2e-4) |
|---|---|---|---|
| GNN | `name.startswith("gnn.")` | `gnn_lr_multiplier = 2.5` | **5e-4** |
| LoRA | `"lora_" in name` | `lora_lr_multiplier = 0.5` | **1e-4** |
| Other | all other trainable params | `1.0` | **2e-4** |

`weight_decay = 1e-2` is applied to all groups. A group is only added to the optimizer if
it contains at least one parameter — this prevents a `OneCycleLR` `max_lr` list length
mismatch when a group happens to be empty (e.g. LoRA disabled).

Rationale: The GNN received a higher LR multiplier (2.5×) to counteract the gradient
collapse observed in v5.0/v5.1 runs where GNN gradient share fell to < 5% by epoch 8.
The LoRA adapters receive a lower LR (0.5×) because CodeBERT is pre-trained and
benefits from conservative updates.

---

## 4. Scheduler

```python
OneCycleLR(
    optimizer,
    max_lr=[gnn_lr, lora_lr, other_lr],  # one entry per non-empty param group
    epochs=config.epochs,                 # always config.epochs — never remaining_epochs
    steps_per_epoch=steps_per_epoch,
    pct_start=0.10,
    anneal_strategy="cos"
)
```

`steps_per_epoch` is computed as `ceil(len(train_loader) / gradient_accumulation_steps)`.

**Critical invariant:** `epochs=config.epochs` must always be the full epoch count, not the
number of remaining epochs on a resume. This ensures `total_steps = epochs × steps_per_epoch`
matches the scheduler state restored from the checkpoint, allowing correct LR interpolation
on resume.

**Resume behaviour:** If the checkpoint's `optimizer.param_groups` count does not match the
current optimizer (e.g. pre-v5.2 checkpoint has 1 group; v5.2 has 3), optimizer and
scheduler restoration are both skipped with a WARNING. Training continues from the correct
epoch with a freshly initialised scheduler at step 0.

---

## 5. Loss Function

### 5.1 Primary Loss — BCEWithLogitsLoss

```python
BCEWithLogitsLoss(pos_weight=pos_weight)
```

`pos_weight` is a per-class tensor of shape `[10]` computed from the **train split only**
via `compute_pos_weight()`. The scaling formula is:

```
pos_weight[c] = (neg_count[c] / pos_count[c]) ** 0.5
```

Using the square-root (exponent 0.5) rather than the full ratio moderates the penalty on
very rare classes (e.g. DoS with only ~257 training examples) without over-correcting. The
tensor is saved to `ml/data/processed/pos_weights_v5.1.pt` for inspection.

`pos_weight` is **recomputed fresh** on every run (including resume) from the current train
split — Fix #13 ensures stale cached weights are never used.

### 5.2 Auxiliary Head Losses

Three auxiliary heads produce logits during the forward pass: `aux["gnn"]`,
`aux["transformer"]`, and `aux["fused"]`. Each receives the same multi-label targets.

```python
aux_loss = loss_fn(aux["gnn"], labels) \
         + loss_fn(aux["transformer"], labels) \
         + loss_fn(aux["fused"], labels)

effective_weight = min(epoch / aux_loss_warmup_epochs, 1.0) * aux_loss_weight
# epoch=0 → 0.0, epoch=1 → 0.1, epoch=2 → 0.2, epoch≥3 → 0.3

total_loss = (main_loss + effective_weight * aux_loss) / gradient_accumulation_steps
```

Auxiliary heads are **training-only** — they are detached and not evaluated at validation.

### 5.3 Alternative Loss — FocalLoss

When `loss_fn="focal"`, `FocalLoss(gamma=2.0, alpha=0.25)` is substituted for
`BCEWithLogitsLoss`. The focal kernel expects sigmoid-transformed probabilities, so
`sigmoid(logits)` is applied externally before passing to the loss. This path is available
for experimentation but is not the default for v5.2.

---

## 6. Gradient Accumulation and AMP

### 6.1 Accumulation Loop

Each call to `train()` iterates micro-batches. The optimizer step fires every
`gradient_accumulation_steps` micro-batches:

```
for step, batch in enumerate(train_loader):
    loss = compute_loss(batch) / gradient_accumulation_steps  # scale before backward
    scaler.scale(loss).backward()

    if (step + 1) % gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # grad norms read HERE — after unscale, before step (Fix #28)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
```

Dividing by `gradient_accumulation_steps` before `backward()` ensures gradient magnitude
is equivalent to a single forward pass over the full effective batch.

### 6.2 AMP GradScaler

```python
scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
```

The scaler silently skips `optimizer.step()` on any micro-batch where loss is NaN or Inf
(dynamic loss scaling). Skipped steps are counted by `nan_loss_count` (see Section 9).

### 6.3 Grad Clip Timing (Fix #28)

Gradient norms are read **after** `scaler.unscale_()` and **before** `optimizer.zero_grad()`.
Previous versions (pre-fix28) read norms before unscaling (artificially small) or after
`zero_grad()` (always zero). Fix #28 corrects both.

---

## 7. Dataset and DataLoader

### 7.1 Split Files

Training uses the **deduplicated** splits exclusively:

| Split | File | Rows |
|---|---|---|
| Train | `ml/data/splits/deduped/train.csv` | 31,092 |
| Val | `ml/data/splits/deduped/val.csv` | 6,661 |
| Test | `ml/data/splits/deduped/test.csv` | 6,667 |

The original `ml/data/splits/` directory contains leaky splits (34.9% of content groups
span multiple splits) and must not be used for training or evaluation.

### 7.2 RAM Cache

`ml/data/cached_dataset_deduped.pkl` holds pre-loaded graph and token data for the
44,420 canonical contracts. Loading from the cache avoids repeated disk I/O per epoch.
Verify the cache exists before training:

```bash
ls -lh ml/data/cached_dataset_deduped.pkl
```

If missing, rebuild with:

```bash
PYTHONPATH=. python ml/scripts/create_cache.py
```

### 7.3 Weighted Sampler

Controlled by `use_weighted_sampler`:

- `"none"` — standard random shuffle (default)
- `"DoS-only"` — up-sample the 377 DoS-positive contracts to increase their frequency
- `"all-rare"` — up-sample all classes with fewer than 1,000 positive examples

---

## 8. Checkpoint Save and Resume

### 8.1 Checkpoint Structure

A checkpoint is saved whenever `val_f1_macro` improves over `best_f1`:

```python
torch.save({
    "model":           model.state_dict(),
    "optimizer":       optimizer.state_dict(),
    "scheduler":       scheduler.state_dict(),
    "epoch":           epoch,
    "best_f1":         best_f1,
    "patience_counter": patience_counter,
    "model_version":   "v5.2",
    "config": {
        **dataclasses.asdict(config),
        "num_classes":  num_classes,
        "class_names":  class_names,
        "architecture": "three_eye_v5"
    }
}, checkpoint_path)
```

A sidecar JSON file is written alongside the checkpoint:

```
<checkpoint_name>.state.json → {"epoch": int, "patience_counter": int, "best_f1": float}
```

The sidecar is authoritative for `epoch`, `patience_counter`, and `best_f1` on resume —
the values inside the `.pt` file may lag by one save cycle in edge cases.

### 8.2 Checkpoint Version Gate (Phase 1-A6)

```python
MODEL_VERSION = "v5.2"
# _parse_version("v5.2") → (5, 2)
```

On resume: if `ckpt["model_version"]` parses to a tuple less than `(5, 2)`, a WARNING is
emitted noting that JK attention and per-phase LayerNorm parameters will be
randomly initialised. `load_state_dict(strict=False)` is always used to accommodate LoRA
key mismatches and new v5.2 parameters.

### 8.3 Resume Sequence

When `resume_from` is set and `resume_model_only=False`:

1. Load checkpoint with `weights_only=False` (LoRA state dict contains PEFT objects).
2. Emit version warning if pre-v5.2.
3. `model.load_state_dict(ckpt["model"], strict=False)`.
4. Read `epoch`, `best_f1`, `patience_counter` from `.state.json` sidecar.
5. Restore optimizer: **skip** if `param_groups` count mismatch.
6. Restore scheduler: check `total_steps` matches; **skip with WARNING** if mismatch.
7. Recompute `pos_weight` fresh from the current train split (Fix #13).

When `resume_model_only=True` (default): only step 3 executes. Optimizer, scheduler,
and training counters reset to initial state.

---

## 9. Per-Epoch and Per-Step Monitoring

### 9.1 MLflow Metrics — Logged Per Epoch

| Metric | Description |
|---|---|
| `train_loss` | Mean loss over non-NaN micro-batches |
| `nan_batch_count` | Number of NaN/Inf loss micro-batches skipped by scaler |
| `val_f1_macro` | Macro-averaged F1 across 10 classes at `threshold=0.5` |
| `val_f1_micro` | Micro-averaged F1 |
| `val_hamming` | Hamming loss (fraction of incorrect label assignments) |
| `aux_loss_weight_effective` | Actual `effective_weight` used in `total_loss` |
| `val_f1_{class}` × 10 | Per-class F1; class names from `class_names` list |
| `jk_phase1_weight` | JK attention weight for phase-1 output (if `use_jk=True`) |
| `jk_phase2_weight` | JK attention weight for phase-2 output (if `use_jk=True`) |
| `jk_phase3_weight` | JK attention weight for phase-3 output (if `use_jk=True`) |

JK weights sum to 1.0 per forward pass; the per-epoch logged value is the mean over all
validation batches.

### 9.2 Per-Step Gradient Norms (every `log_interval` optimizer steps)

```python
gnn_norm   = L2 norm of gradients on model.gnn_eye_proj parameters
tf_norm    = L2 norm of gradients on model.transformer_eye_proj parameters
fused_norm = L2 norm of gradients on model.fusion parameters
total_norm = gnn_norm + tf_norm + fused_norm
gnn_share  = gnn_norm / total_norm
```

These are logged to MLflow as step-level metrics. `gnn_share` is the primary signal for
GNN collapse detection (Section 10).

---

## 10. Collapse and Dominance Detection

### 10.1 GNN Collapse Detection (Phase 2-C2)

```
if gnn_share < 0.10:
    _gnn_collapse_streak += 1
else:
    _gnn_collapse_streak = 0

if _gnn_collapse_streak >= 3:
    WARNING: "GNN collapse detected — gnn_share below 0.10 for 3 consecutive intervals"
```

This pattern — GNN contribution dropping below 10% of total gradient norm for 3 consecutive
`log_interval` windows — was the root cause of silent GNN death in the v5.0 fix28 run
(collapsed at epoch 8, undetected until post-hoc analysis). The streak counter ensures
single transient dips do not trigger false alarms.

### 10.2 JK Phase Dominance Alert (Phase 2-C3)

```
if any(jk_phase_weight > 0.80):
    WARNING: "JK phase dominance — one phase weight exceeds 0.80"
```

If one JK phase consistently captures > 80% of attention, the JK mechanism has degenerated
to a single-phase pass-through. This alert fires once per epoch when the condition is met.

### 10.3 NaN Loss Counter (Phase 2-B3)

`nan_loss_count` is incremented each time `torch.isnan(loss) or torch.isinf(loss)` is
detected before the backward pass. At epoch end:

```
if nan_loss_count > 0:
    WARNING: f"Epoch {epoch}: {nan_loss_count} NaN/Inf batches ({fraction:.1%} of steps)"
```

Only non-NaN batches contribute to `train_loss`. A nonzero count at epoch 1 typically
indicates AMP scale misconfiguration or a data issue in a specific contract graph.

---

## 11. Run Commands

All commands assume the project root is the working directory.

### 11.1 Activate Environment

```bash
source ml/.venv/bin/activate
export TRANSFORMERS_OFFLINE=1
```

`TRANSFORMERS_OFFLINE=1` must be set at the shell level before any import of
`transformers`. Setting it inside Python after import has no effect.

### 11.2 Smoke Run (Phase 4 validation)

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-smoke \
    --experiment-name sentinel-v5.2 \
    --epochs 2 \
    --smoke-subsample-fraction 0.1 \
    --gradient-accumulation-steps 4
```

Uses 10% of the deduplicated dataset (≈ 4,400 contracts). Runs 2 epochs. No checkpoint
is promoted from smoke runs.

### 11.3 Full Training Run (Phase 5)

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-jk \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4
```

Expected wall time on RTX 3070 (8 GB VRAM): approximately 8–12 hours for 60 epochs at
effective batch size 64.

### 11.4 Resume Example

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-jk \
    --experiment-name sentinel-v5.2 \
    --epochs 60 \
    --gradient-accumulation-steps 4 \
    --resume-from ml/checkpoints/multilabel-v5-fresh_best.pt \
    --no-resume-model-only
```

`--no-resume-model-only` triggers full resume (optimizer + scheduler + counters).

---

## 12. Smoke and Full-Run Gates

### 12.1 Phase 4 Smoke Gates

Check these after the 2-epoch smoke run completes:

| Gate | Threshold | Where to Check |
|---|---|---|
| GNN share | ≥ 15% at optimizer step 100 | MLflow step metric `gnn_share` |
| JK phase1 weight | ≥ 5% | MLflow epoch metric `jk_phase1_weight` (epoch 1) |
| JK phase2 weight | ≥ 5% | MLflow epoch metric `jk_phase2_weight` (epoch 1) |
| JK phase3 weight | ≥ 5% | MLflow epoch metric `jk_phase3_weight` (epoch 1) |
| NaN batches | 0 after step 50 | MLflow metric `nan_batch_count` |
| Loss direction | decreasing | eyeball `train_loss` curve in MLflow UI |

All 6 gates must pass before launching the full 60-epoch run.

### 12.2 v5.2 Full-Run Gates

These gates are evaluated after the full training run completes. All are required before
promoting the checkpoint to production.

| Gate | Threshold | Type |
|---|---|---|
| JK gradient flow test | All params non-zero grad | Code (already PASSED — `test_jk_gradient_flow`) |
| Val F1-macro (tuned threshold) | > any valid prior run (TBD — no valid pre-dedup baseline) | Training metric |
| Detection rate on behavioral contracts | ≥ 70% | Behavioral (`ml/scripts/manual_test.py`) |
| Safe specificity on clean contracts | ≥ 66% | Behavioral |
| CEI-A contract fires (reentrancy present) | Absolute requirement | Behavioral |
| CEI-B contract silent (reentrancy absent) | Absolute requirement | Behavioral |
| v4 floor per-class | F1 ≥ (v4_F1 − 0.05) for each class | Training metric |

#### v4 Per-Class F1 Floors

| Class | v4 F1 | Floor (v4 − 0.05) |
|---|---|---|
| CallToUnknown | 0.397 | 0.347 |
| DoS | 0.384 | 0.334 |
| ExternalBug | 0.434 | 0.384 |
| GasException | 0.507 | 0.457 |
| IntegerUO | 0.776 | 0.726 |
| MishandledException | 0.459 | 0.409 |
| Reentrancy | 0.519 | 0.469 |
| Timestamp | 0.478 | 0.428 |
| TOD | 0.472 | 0.422 |
| UnusedReturn | 0.495 | 0.445 |

The DoS floor is intentionally lenient (0.334) given the severe data starvation
(~257 training examples). A per-class floor below the v4 value for any class means the
new architecture degraded on that vulnerability type.

---

*Last updated: 2026-05-14. Applies to model_version = "v5.2".*
