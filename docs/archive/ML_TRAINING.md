# SENTINEL ML Training — Technical Reference

## Overview

Three modules compose the training pipeline:

```
DualPathDataset      (datasets/dual_path_dataset.py) — disk → (graph, tokens, label)
FocalLoss            (training/focalloss.py)          — imbalance-aware loss
trainer.py           (training/trainer.py)            — full training loop + MLflow + checkpointing
```

---

## Data Context (Important — Read Before Training)

### What the labels mean

The training labels are **binary**: `1 = vulnerable`, `0 = safe`. Each label was derived from the
BCCC-SCsVul-2024 dataset by collapsing 12 vulnerability classes to binary. The model learns "does
this contract have any vulnerability?" — not which type.

### Why the class distribution is what it is

```
Vulnerable (label=1): 44,099  (64.33%)   ← majority class
Safe       (label=0): 24,456  (35.67%)   ← minority class
```

The BCCC dataset has 12 subdirectories. When the preprocessing script scanned them, it assigned the
label from the first folder where each unique hash appeared. Because the vulnerability folders
collectively contain more contracts than the NonVulnerable folder (and because the same contract
may appear in multiple vulnerability folders), vulnerable ends up as the majority. This is the
opposite of most security-related intuitions — `alpha=0.25` for the vulnerable class is **correct**
because it is the majority, not a bug.

### What was lost during preprocessing

41.2% of contracts in BCCC appear in 2–9 vulnerability folders simultaneously. The preprocessing
kept only one label per hash. If you retrain and see unexpected results, or need per-type detection,
the ground-truth multi-label information is preserved in `contract_labels_correct.csv` (columns
Class01–Class12). See [ML_DATASET_PIPELINE.md](ML_DATASET_PIPELINE.md) for the full picture.

---

## DualPathDataset (`ml/src/datasets/dual_path_dataset.py`)

### Responsibility

Loads paired graph + token `.pt` files for training, validation, and testing. Each sample is a smart contract represented two ways:
- **Graph** (`.pt`): PyG `Data` object — AST/CFG structure for the GNNEncoder
- **Tokens** (`.pt`): dict with CodeBERT token tensors for the TransformerEncoder

### File layout on disk

```
ml/data/graphs/<md5_hash>.pt  →  PyG Data(x=[N,8], edge_index=[2,E], y=[label])
ml/data/tokens/<md5_hash>.pt  →  dict(input_ids=[512], attention_mask=[512])
```

The MD5 hash is the pairing key. Files with the same stem belong to the same contract. 68,570 token
files and 68,556 graph files exist — 14 unpaired files are silently skipped at init time.

**68,555 paired samples** are used for training.

Note: the 68,555 paired graph+token files were extracted from 44,442 unique source contracts in
`contract_labels_correct.csv`. The difference (68,555 vs 44,442) is because contracts appearing in
multiple BCCC vulnerability folders were sometimes extracted multiple times under different path-
based MD5 hashes. The label baked into each `graph.y` is always binary (0 or 1).

### Initialisation flow

```
1. Glob *.pt in graphs_dir and tokens_dir
2. Compute set intersection of stems → paired hashes
3. Sort paired hashes alphabetically (deterministic across runs and machines)
4. If indices provided: filter to those positions (train/val/test split enforcement)
5. Load sample[0] as validation (validate=True by default)
```

**Sorting is critical.** Without sorting, `set` iteration order is non-deterministic — `dataset[0]` would return a different contract on every run, making splits non-reproducible.

**Eager validation** (`validate=True`): Loads one sample during `__init__`. A broken `.pt` file is caught immediately rather than 3 hours into training when the DataLoader trips on it.

### `__getitem__` returns

```python
graph, tokens, label = dataset[0]

graph.x               # [N, 8]  float32 — node features
graph.edge_index      # [2, E]  int64   — directed edges
tokens["input_ids"]   # [512]   long    — CodeBERT token IDs
tokens["attention_mask"] # [512] long   — 1=real, 0=padding
label                 # torch.tensor([0]) or torch.tensor([1]) — shape [1]
```

Labels are baked into `graph.y` during preprocessing. The CSV file is not read at training time.

### Loading flags: `weights_only`

```python
graph_data = torch.load(graph_path, weights_only=False)  # PyG Data objects
token_data = torch.load(token_path, weights_only=True)   # plain tensor dicts
```

**Why `weights_only=False` for graphs?** Graph `.pt` files contain PyG `Data` objects, which require unpickling custom classes (`DataEdgeAttr`, `DataTensorAttr`). These classes are explicitly allowlisted:

```python
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
```

**Why `weights_only=True` for tokens?** Token files contain only plain tensor dicts — stricter loading is safe and preferred.

### Integrity check

Each `__getitem__` call verifies that the `contract_hash` embedded in the graph and token files match. Hash mismatch indicates data corruption or wrong file pairing:

```python
if graph_hash and token_hash and graph_hash != token_hash:
    raise ValueError(f"Contract hash mismatch — ...")
```

This check only fires when both files actually stored the hash field. Older preprocessing files without the field are silently accepted.

### Label normalisation

Labels are stored inconsistently across preprocessing versions. `__getitem__` normalises:
```python
label = label.view(1).long()  # Always [1] shape, int64
```

This ensures `torch.stack(labels)` in the collate function always produces `[B, 1]` before `squeeze(1)`.

---

## `dual_path_collate_fn`

**Must be at module level** — not inside a class or function. DataLoader multiprocessing uses `pickle` to pass the collate function to worker processes. Closures and instance methods are not picklable.

### What it does

```python
def dual_path_collate_fn(batch):
    graphs = [item[0] for item in batch]
    tokens = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    batched_graphs  = Batch.from_data_list(graphs)      # variable-size PyG merge
    batched_tokens  = {
        "input_ids":      torch.stack([t["input_ids"]      for t in tokens]),  # [B, 512]
        "attention_mask": torch.stack([t["attention_mask"] for t in tokens]),  # [B, 512]
    }
    batched_labels  = torch.stack(labels).squeeze(1)    # [B, 1] → [B]
    return batched_graphs, batched_tokens, batched_labels
```

**Why `Batch.from_data_list(graphs)`?**

PyG graphs have variable numbers of nodes and edges — you cannot `torch.stack` them. `Batch.from_data_list()` merges them into a single disconnected super-graph. The `.batch` attribute maps each node back to its contract index:
```
4 nodes + 3 nodes + 2 nodes → batch = [0,0,0,0, 1,1,1, 2,2]
```
`global_mean_pool(x, batch)` uses this to average per-contract, not over the entire super-graph.

**Why `.squeeze(1)` and not `.squeeze()`?**

`.squeeze()` collapses **all** dimensions of size 1. When `batch_size=1`, labels are `[1, 1]` — bare `.squeeze()` would turn this into a scalar `[]`, which breaks subsequent operations expecting a 1D tensor. `.squeeze(1)` only removes dimension 1, leaving `[B]` intact regardless of B.

---

## FocalLoss (`ml/src/training/focalloss.py`)

### Formula

```python
bce     = F.binary_cross_entropy(predictions, targets, reduction="none")   # [B]
pt      = where(targets == 1, predictions, 1 - predictions)                # probability of true class
alpha_t = where(targets == 1, alpha, 1 - alpha)                            # class weight
focal   = alpha_t * (1 - pt)^gamma * bce                                   # [B]
loss    = focal.mean()                                                      # scalar
```

### Parameters

| Parameter | Value | Effect |
|---|---|---|
| `gamma` | 2.0 | Multiplies loss by `(1-pt)²`. For easy examples (pt≈1), `(1-pt)²≈0` — near-zero loss. For hard examples (pt≈0.5), loss is multiplied by 0.25. Forces the model to focus on hard, uncertain samples. |
| `alpha` | 0.25 | Weight for `label=1` (vulnerable, majority at 64.33%). `1-alpha=0.75` for `label=0` (safe, minority). Weight ratio 0.75/0.25 = 3×. Deliberately stronger than class imbalance ratio of 1.8× to compensate. |

### Why `alpha=0.25` is correct (not a bug)

Higher `alpha` means higher weight. Vulnerable is the majority class (64.33%) — it should be **down-weighted**. Safe is the minority (35.67%) — it should be **up-weighted**. So:
- `vulnerable (label=1): weight = alpha = 0.25` ← down-weighted ✓
- `safe (label=0): weight = 1 - alpha = 0.75` ← up-weighted ✓

This is confirmed correct. An earlier handover incorrectly called this a bug. The 10:18 PM final handover confirmed the sign is right — do not change it.

### Critical: input must be sigmoid-activated

```python
predictions: [B] float in [0, 1]   # ALREADY sigmoid-activated
```

`SentinelModel` applies sigmoid internally. `FocalLoss` uses `F.binary_cross_entropy` (not `BCEWithLogitsLoss`). **Never apply sigmoid twice.** Applying sigmoid to already-sigmoid outputs (which are in `[0,1]`) produces wrong gradients:
- Values near 0.5 get pushed toward 0.5 (sigmoid(0.5)≈0.622)
- Values near 1.0 map to sigmoid(1.0)≈0.731 instead of ≈1.0
- The loss landscape becomes distorted

---

## `trainer.py` — Training Loop

### `TrainConfig` dataclass

Single source of truth for all hyperparameters and paths. Everything lives in one place — no magic numbers scattered through the code.

```python
@dataclass
class TrainConfig:
    # Paths
    graphs_dir:      str = "ml/data/graphs"
    tokens_dir:      str = "ml/data/tokens"
    splits_dir:      str = "ml/data/splits"
    checkpoint_dir:  str = "ml/checkpoints"
    checkpoint_name: str = "sentinel_best.pt"

    # Training
    epochs:       int   = 20
    batch_size:   int   = 32
    lr:           float = 1e-4
    weight_decay: float = 1e-2

    # FocalLoss
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # MLflow
    experiment_name: str = "sentinel-training"
    run_name:        str = "baseline"

    # Device (auto-detected)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Resume (new-format checkpoints only)
    resume_from: str | None = None
```

To run a sweep, subclass `TrainConfig` or pass a modified instance to `train()`.

### Optimizer: AdamW with frozen parameters

```python
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config.lr,
    weight_decay=config.weight_decay,
)
```

`filter(requires_grad)` is essential. AdamW maintains momentum and variance statistics per parameter. The 124.6M frozen CodeBERT parameters never update — we don't want to waste memory or compute maintaining their optimizer state. This filter passes only the 239,041 trainable parameters to AdamW.

### Training loop

```python
for epoch in range(start_epoch, config.epochs + 1):
    # Train
    model.train()   # Enables dropout (FusionLayer 0.3, GNNEncoder 0.2)
    for graphs, tokens, labels in train_loader:
        labels = labels.to(device).float().view(-1)          # [B] float
        predictions = model(graphs, input_ids, attention_mask)  # [B] in [0,1]
        loss = focal_loss(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()   # Disables dropout
    val_metrics = evaluate(model, val_loader, device)

    # Checkpoint: save only when F1-macro improves
    if val_metrics["f1_macro"] > best_f1:
        best_f1 = val_metrics["f1_macro"]
        torch.save(checkpoint_dict, checkpoint_path)
```

**`labels.float().view(-1)`** — labels from the collate function are `long`. FocalLoss requires `float`. `view(-1)` (not `squeeze()`) always produces `[B]` — `squeeze()` would collapse `[1]` to `[]` when `B=1`.

**Shuffle:** `train_loader` uses `shuffle=True` each epoch to prevent ordering bias. `val_loader` uses `shuffle=False` for deterministic, reproducible metrics.

### Evaluation (`evaluate`)

```python
def evaluate(model, loader, device) -> dict[str, float]:
    """Returns: f1_macro, f1_safe, f1_vulnerable, precision_vulnerable, recall_vulnerable"""
```

Hard threshold at 0.5 to convert sigmoid probabilities → binary predictions:
```python
preds_binary = (predictions >= 0.5).long()
```

Note: this 0.5 evaluation threshold is hardcoded during training (for checkpoint selection). The production inference threshold is determined separately by `tune_threshold.py` after training.

**Metrics collected:**

| Metric | Sklearn call | Purpose |
|---|---|---|
| `f1_macro` | `f1_score(average="macro")` | Primary — unweighted average F1. Not gameable by mass-flagging. |
| `f1_safe` | `f1_score(average=None)[0]` | F1 for safe class (label=0) |
| `f1_vulnerable` | `f1_score(average=None)[1]` | F1 for vulnerable class (label=1) |
| `precision_vulnerable` | `precision_score(pos_label=1)` | Of flagged contracts, how many truly vulnerable? |
| `recall_vulnerable` | `recall_score(pos_label=1)` | Of truly vulnerable contracts, how many caught? ← critical security metric |

**Why F1-macro as checkpoint criterion (not recall_vulnerable)?**

Recall_vulnerable is gameable: a model that predicts everything as vulnerable achieves recall=1.0. F1-macro averages F1 over both classes — mass-flagging collapses F1-safe, pulling down F1-macro. The optimal threshold under F1-macro is genuinely balanced.

That said: if missing a vulnerability is far worse than a false alarm (likely in production), consider switching to `recall_vulnerable` as the checkpoint criterion. Both metrics are logged to MLflow for post-hoc comparison.

### Checkpointing

**New format (April 2026+):**

```python
torch.save({
    "model":     model.state_dict(),    # weights — inference use
    "optimizer": optimizer.state_dict(), # AdamW state — resume use
    "epoch":     epoch,                  # resume continues from epoch+1
    "best_f1":   best_f1,               # restored on resume
    "config":    dataclasses.asdict(config),  # hyperparameter record
}, checkpoint_path)
```

**Old format (pre-April 2026):** Plain `state_dict()`. Only model weights, no optimizer state or epoch. Supported by `predictor.py` and `tune_threshold.py` for inference but **cannot be resumed**.

### Resume logic

```python
if config.resume_from:
    ckpt = torch.load(config.resume_from, map_location=device, weights_only=True)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Cannot resume from old-format checkpoint ...")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0) + 1   # continue from next epoch
    best_f1     = ckpt.get("best_f1", 0.0)   # keep the previous best
```

The optimizer state includes:
- First moment (momentum) per parameter — preserves gradient direction memory
- Second moment (variance) per parameter — preserves adaptive learning rate state

Restoring these prevents the optimizer from behaving as if training just started, which would cause an initial loss spike and several "warm-up" epochs before stabilising.

### MLflow integration

```python
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment(config.experiment_name)

with mlflow.start_run(run_name=config.run_name):
    mlflow.log_params({...})               # once at start
    for epoch in ...:
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_f1_macro", ..., step=epoch)
        # ... 4 more metrics
        mlflow.log_artifact(checkpoint_path)   # when best improves
```

View results: `poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db` → `http://localhost:5000`

### Checkpoint naming convention

```python
checkpoint_path = Path(config.checkpoint_dir) / f"{config.run_name}_best.pt"
```

`baseline` → `sentinel_best.pt`... no, actually each run uses `{run_name}_best.pt`:
- `run-alpha-tune` → `run-alpha-tune_best.pt`
- `run-more-epochs` → `run-more-epochs_best.pt`

This means experiments never overwrite each other's checkpoints.

---

## Data splits

Splits are pre-computed index arrays (not boolean masks):

```
ml/data/splits/train_indices.npy  →  int64 array of 47,988 positions
ml/data/splits/val_indices.npy    →  int64 array of 10,283 positions
ml/data/splits/test_indices.npy   →  int64 array of 10,284 positions
```

Splits were created with stratified sampling (seed=42) to preserve the 64.33%/35.67% class ratio in all three splits.

**The test set has never been used during training or threshold selection.** It is reserved for the final, single holdout evaluation of the production model.

---

## Practical training reference

### Start a new run

```bash
poetry run python ml/scripts/train.py \
    --run-name run-alpha-tune \
    --epochs 30 \
    --focal-alpha 0.25
```

### Resume an interrupted run

```bash
poetry run python ml/scripts/train.py \
    --resume ml/checkpoints/run-alpha-tune_best.pt \
    --run-name run-alpha-tune-resumed \
    --epochs 40
```

**Note:** Only new-format checkpoints (dict with `"model"` key) can be resumed.

### Monitor training

```bash
# Check live progress
tail -f ml/logs/overnight.log

# Open MLflow dashboard
poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db
# → http://localhost:5000
```

### What to look for in MLflow

| Signal | Interpretation |
|---|---|
| `val_f1_macro` still rising at final epoch | Model undertrained — run more epochs |
| `train_loss` falling but `val_f1_macro` flat | Overfitting — reduce LR or add regularisation |
| `val_recall_vulnerable` << `val_f1_macro` | Model biased toward safe predictions — lower threshold or reduce alpha |
| `val_precision_vulnerable` << 0.7 | Too many false alarms — raise threshold |
| Oscillating `val_f1_macro` | Learning rate too high — try 3e-5 |
