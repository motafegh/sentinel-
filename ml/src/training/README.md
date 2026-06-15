# training — Training Loop

> **Status:** ✅ Current — v9 schema, four-eye v8.1 architecture, verified 2026-06-14

Training loop, loss function, and configuration for SENTINEL v8.1 + GraphCodeBERT.

**Current architecture:** 8-layer GNN (2+3+3 phases), Flash Attention 2, IMP improvements (G1, G2, G3, M1, M3, C2, #26)

---

## Files

| File | Contents |
|------|----------|
| `trainer.py` | `TrainConfig`, `train_one_epoch()`, `evaluate()`, `train()` |
| `losses.py` | `AsymmetricLoss` — multi-label focal loss for class-imbalanced training |
| `focalloss.py` | `FocalLoss` and `MultiLabelFocalLoss` — focal loss implementations |

---

## TrainConfig (`trainer.py`)

Single source of truth for all hyperparameters and paths. Serialized into each checkpoint as `saved_cfg` via `dataclasses.asdict(config)` so any run can be fully reconstructed from its checkpoint.

```python
@dataclass
class TrainConfig:
    # Paths
    cache_path:       str   = "ml/data/cached_dataset_v9.pkl"
    splits_dir:       str   = "ml/data/splits/deduped"
    checkpoint_dir:   str   = "ml/checkpoints"
    run_name:         str   = "gcb-run"
    experiment_name:  str   = "sentinel-gcb"

    # Architecture
    gnn_hidden_dim:   int   = 256
    gnn_layers:       int   = 8      # 2+3+3 phases (IMP-G3 added conv4c)
    phase2_edge_types: list = field(default_factory=lambda: [6, 8, 9, 10])  # CF+CE+RT+DU (v8+, v9 adds EXTERNAL_CALL=11)
    lora_r:           int   = 16
    lora_alpha:       int   = 32

    # GNN Prefix Injection
    gnn_prefix_k:               int   = 0     # 0 = disabled
    gnn_prefix_warmup_epochs:   int   = 15
    gnn_prefix_proj_lr_mult:    float = 1.0

    # Training
    epochs:                     int   = 100
    batch_size:                 int   = 8
    gradient_accumulation_steps: int  = 8     # effective batch = 64
    lr:                         float = 2e-4

    # Per-group LR multipliers
    lora_lr_mult:    float = 0.3    # LoRA adapter LR
    gnn_lr_mult:     float = 2.5    # GNN encoder LR
    fusion_lr_mult:  float = 0.5    # CrossAttentionFusion LR

    # Loss
    loss_fn:         str   = "asl"
    asl_gamma_neg:   float = 2.0
    asl_gamma_pos:   float = 1.0
    asl_clip:        float = 0.01
    dos_loss_weight: float = 0.5    # DoS auxiliary loss weight

    # Sampling
    use_weighted_sampler: str = "positive"  # 3× weight for any-vuln rows
    pos_weight_min_samples: int = 3000

    # Performance
    num_workers:   int  = 4
    use_compile:   bool = True    # submodule-level; transformer excluded
    use_amp:       bool = True    # BF16; no GradScaler
    early_stop_patience: int = 30
```

**Per-group learning rates:** AdamW uses separate param groups with LR multipliers relative to the base `lr`:
- LoRA adapters: `lr * lora_lr_mult`
- GNNEncoder: `lr * gnn_lr_mult`
- CrossAttentionFusion: `lr * fusion_lr_mult`
- `gnn_to_bert_proj`: `lr * gnn_prefix_proj_lr_mult`
- Everything else (classifier, eye projectors): `lr`

---

## AsymmetricLoss (`losses.py`)

Multi-label focal loss with separate γ values for positives and negatives. Designed for class-imbalanced multi-label classification.

**Formula:**
```
ASL(p, y) = alpha_t * (1 - pt)^gamma_t * BCE(p, y)

For positive labels (y=1): gamma_t = gamma_pos, pt = p
For negative labels (y=0): gamma_t = gamma_neg, pt = 1-p (after probability shift)
Clip: p_neg clamped to [clip, 1-clip] to prevent hard negative dominance
```

**SENTINEL configuration:**
```
asl_gamma_neg = 2.0  — strong down-weighting of easy negatives (majority)
asl_gamma_pos = 1.0  — mild down-weighting of easy positives
asl_clip      = 0.01 — prevent log(0) and hard-negative collapse
```

**Interface:**
```python
loss_fn = AsymmetricLoss(gamma_neg=2.0, gamma_pos=1.0, clip=0.01)
# logits:  [B, 10]  float  — raw model output (pre-sigmoid)
# targets: [B, 10]  float  — multi-hot labels cast to float
loss = loss_fn(logits, targets)   # scalar
```

Loss takes **raw logits**. `SentinelModel.forward()` returns logits; apply sigmoid only at inference time.

---

## train_one_epoch() (`trainer.py`)

Runs one full pass over the training loader with gradient accumulation.

```python
def train_one_epoch(model, loader, optimizer, scaler, loss_fn, config, epoch) -> float:
    # Returns: mean ASL over all effective steps (float)
```

- `model._current_epoch = epoch` set before the loop — controls prefix warmup suppression
- Gradient accumulation: `loss / gradient_accumulation_steps` before backward
- BF16 autocast via `torch.amp.autocast("cuda", dtype=torch.bfloat16)`
- GradScaler disabled for BF16 (no underflow risk)
- Optimizer step every `gradient_accumulation_steps` mini-batches

---

## evaluate() (`trainer.py`)

Runs inference over val or test set, returns multi-label metrics.

```python
def evaluate(model, loader, config, device) -> dict[str, float]:
```

**Returns:**
```python
{
    "f1_macro":   float,   # primary checkpoint criterion — macro-averaged across 10 classes
    "f1_micro":   float,
    "precision":  float,
    "recall":     float,
    # per-class F1 for each of the 10 vulnerability classes
    "f1_CallToUnknown":            float,
    "f1_DenialOfService":          float,
    "f1_ExternalBug":              float,
    "f1_GasException":             float,
    "f1_IntegerUO":                float,
    "f1_MishandledException":      float,
    "f1_Reentrancy":               float,
    "f1_Timestamp":                float,
    "f1_TransactionOrderDependence": float,
    "f1_UnusedReturn":             float,
}
```

**Threshold:** fixed at 0.5 during training evaluation. Use `ml/scripts/tune_threshold.py` post-training for per-class optimal thresholds.

---

## train() (`trainer.py`)

Full training loop. Builds everything from `config`, trains for N epochs, logs to MLflow, saves best checkpoint.

```python
train(config: TrainConfig) -> None
```

**Checkpointing:** saves whenever `val_f1_macro` improves. Checkpoint dict:
```python
{
    "model_state_dict": ...,    # stripped of ._orig_mod. torch.compile infix
    "optimizer_state_dict": ...,
    "epoch": int,
    "val_f1": float,
    "saved_cfg": dataclasses.asdict(config),   # full config for reconstruction
    "class_thresholds": [...],                 # 10 floats, updated each epoch
}
```

**torch.compile strategy:** submodule-level — `gnn`, `fusion`, `classifier`, eye projectors, aux heads are compiled. `model.transformer` (GraphCodeBERT+LoRA) is excluded to avoid HuggingFace control-flow graph breaks. `cache_size_limit=256` prevents dynamo fallback on unique graph shapes.

**MLflow metrics logged per epoch:**

| Metric | Description |
|--------|-------------|
| `train_loss` | Mean ASL over training set |
| `val_f1_macro` | Macro-averaged F1 (checkpoint criterion) |
| `val_f1_{ClassName}` | Per-class F1 for all 10 classes |
| `prefix_active` | 1 when epoch ≥ warmup, 0 otherwise |
| `prefix_proj_weight_norm` | `gnn_to_bert_proj` weight L2 norm |
| `learning_rate` | Current LR (after scheduler) |

**Prefix logging (per epoch when gnn_prefix_k > 0):**
```
GNN prefix K=48: WARMUP (starts ep15)    ← epochs 0–14
GNN prefix K=48: ACTIVE                  ← epochs 15+
gnn_to_bert_proj weight norm: 16.0000    ← constant during warmup (zero gradient confirmed)
```

---

## Training History

| Run | Phase 2 | Best ep | Tuned F1 | Notes |
|-----|---------|---------|----------|-------|
| v7.0 | CF | 23 | 0.2875 | CodeBERT baseline |
| PLAN-3A | CF+CE+RT | 41 | **0.2877** | Best v8 checkpoint |
| v8.0-B | PLAN-3A + labels | 10 | killed | ceiling confirmed at ~0.287 |
| Run 7 | CF+CE+RT+DU | 39 | **0.3329** | Four-eye + type embedding + Phase2 heads=4 |

**Ceiling conclusion:** All v7/v8 CodeBERT runs converge to ~0.287 tuned F1. Run 7 with four-eye architecture (IMP-R7-2), type embedding (BUG-R7-2), and Phase 2 multi-head (IMP-R7-1) achieved 0.3329 tuned F1 — a 16% relative improvement.

MLflow backend: `sqlite:///mlruns.db`. View: `poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db`
