# training — Training Loop

Training loop, loss function, and configuration for SENTINEL.

---

## Files

| File | Contents |
|---|---|
| `trainer.py` | `TrainConfig`, `train_one_epoch()`, `evaluate()`, `train()` |
| `focalloss.py` | `FocalLoss` — binary focal loss for imbalanced classification |

---

## TrainConfig (`trainer.py`)

Single source of truth for all hyperparameters and paths. Pass a modified instance to `train()` to run experiments.

```python
@dataclass
class TrainConfig:
    # Paths
    graphs_dir:      str   = "ml/data/graphs"
    tokens_dir:      str   = "ml/data/tokens"
    splits_dir:      str   = "ml/data/splits"
    checkpoint_dir:  str   = "ml/checkpoints"
    checkpoint_name: str   = "sentinel_best.pt"

    # Training
    epochs:       int   = 20
    batch_size:   int   = 32
    lr:           float = 1e-4
    weight_decay: float = 1e-2    # AdamW

    # FocalLoss
    focal_gamma:  float = 2.0     # original paper default
    focal_alpha:  float = 0.25    # down-weights vulnerable (majority, 64%)

    # MLflow
    experiment_name: str = "sentinel-training"
    run_name:        str = "baseline"

    # Device (auto-detected)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

**Checkpoint naming:** each run saves to `{checkpoint_dir}/{run_name}_best.pt`, so experiments never overwrite each other.

**Overriding for an experiment:**
```python
config = TrainConfig(
    focal_alpha=0.35,
    lr=3e-5,
    epochs=30,
    run_name="run-combined",
)
train(config)
```

---

## FocalLoss (`focalloss.py`)

Binary Focal Loss — designed for imbalanced datasets. Multiplies standard BCE by `(1 - pt)^gamma` to crush the gradient contribution of easy examples, forcing the model to focus on hard ones.

**Formula:**
```
FL(p, y) = alpha_t * (1 - pt)^gamma * BCE(p, y)

where:
  pt      = p   if y == 1   (predicted prob for positive class)
           = 1-p if y == 0
  alpha_t = alpha     if y == 1
           = 1-alpha  if y == 0
```

**SENTINEL configuration (confirmed correct):**
```
focal_alpha = 0.25   →  label=1 (vulnerable, 64% majority) weight = 0.25
                        label=0 (safe,       36% minority) weight = 0.75
focal_gamma = 2.0    →  original paper default
```

Alpha 0.25 down-weights the vulnerable majority and up-weights the safe minority.
This is correct — vulnerable is the majority class (64.33%). Do not change it.

**Interface:**
```python
loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
# predictions: [B]  float  — model output AFTER sigmoid (values in [0,1])
# targets:     [B]  float  — labels cast to float (0.0 or 1.0)
loss = loss_fn(predictions, targets)   # scalar
```

**Critical:** predictions must be post-sigmoid. `SentinelModel` outputs are already sigmoid-activated.
Do not use `BCEWithLogitsLoss` — that applies sigmoid internally and would double-apply it.

---

## train_one_epoch() (`trainer.py`)

Runs one full pass over the training loader.

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    # Returns: mean Focal Loss over all batches (float)
```

- Calls `model.train()` to enable dropout (FusionLayer has `Dropout(0.3)`)
- Labels cast to `float` and reshaped with `.view(-1)` for BCE compatibility
- Optimizer: AdamW with `filter(requires_grad)` — excludes frozen CodeBERT params

---

## evaluate() (`trainer.py`)

Runs inference over val or test set, returns classification metrics.

```python
def evaluate(model, loader, device) -> dict[str, float]:
```

**Returns:**
```python
{
    "f1_macro":             float,  # primary checkpoint metric
    "f1_safe":              float,  # F1 for label=0
    "f1_vulnerable":        float,  # F1 for label=1
    "precision_vulnerable": float,  # TP / (TP + FP) for vulnerable predictions
    "recall_vulnerable":    float,  # TP / (TP + FN) — critical security metric
}
```

**Threshold:** fixed at 0.5 during training evaluation.
Use `ml/scripts/tune_threshold.py` to find the optimal inference threshold post-training.

---

## train() (`trainer.py`)

Full training loop. Builds everything from `config`, trains for N epochs, logs to MLflow, saves best checkpoint.

```python
train(config: TrainConfig) -> None
```

**Checkpointing strategy:** saves whenever `val_f1_macro` improves.
Checkpoint = plain `model.state_dict()` (tensors only, `weights_only=True`-safe).

**MLflow metrics logged per epoch:**

| Metric | Description |
|---|---|
| `train_loss` | Mean Focal Loss over training set |
| `val_f1_macro` | F1 averaged across both classes (checkpoint criterion) |
| `val_f1_safe` | F1 for safe class (label=0) |
| `val_f1_vulnerable` | F1 for vulnerable class (label=1) |
| `val_precision_vulnerable` | Precision for vulnerable predictions |
| `val_recall_vulnerable` | Recall for vulnerable — the security-critical metric |

**MLflow backend:** `sqlite:///mlruns.db` in the project root.
View with: `poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db`

---

## Baseline Training Results

| Metric | Value | Epoch |
|---|---|---|
| Best val F1-macro | 0.6515 | 16 (saved) |
| Best val F1-vuln | 0.7133 | 8 (not saved — checkpoint only tracks F1-macro) |
| MLflow run | `baseline` | ID: `6201a32250e94c47ae1d3daa7a10a989` |

Overnight experiments (4 runs) are tracked as separate named runs in the same `sentinel-training` experiment.  
