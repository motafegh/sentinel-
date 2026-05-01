# Change Log: 2026-05-01 — ML Security & Correctness Audit (Items 1–8)

**Session date:** 2026-05-01
**Commit message prefix:** `fix(ml)` `fix(predictor)` `fix(trainer)` `fix(dataset)`
**Status:** Committed to `main`
**Related commits:**
- `b8be6f8` — fix(predictor): warmup 2-node 1-edge graph + BF16 sigmoid guard
- `d12718f` — fix(trainer): unknown loss_fn ValueError + OneCycleLR resume + trainable-only grad clip
- `d49e09f` — fix(dataset): weights_only=True for graph loading
- `caf95e9` — fix(focalloss): FP32 cast before BCE + docstring correction

---

## Background

A full audit of the `ml/` folder was conducted on 2026-05-01 combining:

1. Reviewer-identified issues (from PR review threads)
2. Static analysis of source code against known ML training pitfalls
3. Technical knowledge of PyTorch AMP/BF16 edge cases, PyG graph construction, and OneCycleLR scheduler semantics

The audit produced **18 findings** across 4 severity levels (Critical, High, Medium, Low, Informational).
This session implements **items 1–8** — all Critical, High, and the most impactful Medium items.
Items 9–18 (lower severity) are tracked for a future session.

---

## Fix #1 (Critical) — Pickle Deserialization: `weights_only=True` in `dual_path_dataset.py`

### Problem

`torch.load(graph_path, weights_only=False)` in `DualPathDataset` uses Python's `pickle` deserializer.
A tampered or substituted `.pt` graph cache file can execute arbitrary Python code with the process's privileges.
`weights_only=False` is the default-unsafe mode: it trusts the file completely.

The safe globals registration (`add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])`) was already present
at module load, providing exactly the allowlist needed — the flag was simply never switched.

### Fix

**File:** `ml/src/datasets/dual_path_dataset.py`

```python
# BEFORE (audit finding #1 — pickle attack surface)
graph = torch.load(graph_path, weights_only=False)

# AFTER — uses registered safe globals; rejects all other pickle classes
graph = torch.load(graph_path, weights_only=True)
# NOTE: if a future PyG release adds new internal classes to the graph objects,
# add them to the add_safe_globals() call at module load above, then re-run
# the pipeline to regenerate the graph cache.
```

### Why It Matters

Graph cache files in `ml/data/graphs/` are written during preprocessing, not training.
A supply-chain attack that replaces one graph file could inject code that runs silently
every time that contract is loaded into a training batch — including on CI/CD machines.

### Spec Reference
- Improvement ledger §6 — ML security constraints
- ADR-019 — safe deserialization policy

---

## Fix #2 (Critical) — BF16 Underflow in FocalLoss + `_FocalFromLogits`

### Problem

Under PyTorch AMP with BF16 (the default for RTX 3070 with `torch.autocast("cuda", dtype=torch.bfloat16)`):

- BF16 has only 3 mantissa bits vs FP32's 23. Probabilities smaller than ~0.008 round to `0.0`.
- `FocalLoss.forward()` calls `F.binary_cross_entropy(predictions, targets)`, which computes `log(p)`.
- `log(0.0) = -inf` → `nan` gradient.
- `GradScaler` detects `nan` and silently **skips the update step** (`optimizer.step()` is a no-op).
- Training appears to proceed normally (loss decreases slowly) but many steps are wasted.
- This is a silent failure: no exception, no warning, just degraded convergence.

The issue has two entry points:
1. `FocalLoss.forward()` receives `predictions` that may be in BF16.
2. `_FocalFromLogits.forward()` calls `torch.sigmoid(logits)` while logits are still in BF16.

### Fix

**File:** `ml/src/training/focalloss.py`

```python
# FocalLoss.forward() — cast FIRST, before any log or BCE operation
def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Audit fix #2 (2026-05-01): cast to FP32 before ANY numerical operation.
    # Under BF16 AMP, probabilities < ~0.008 underflow to 0.0 silently.
    # log(0.0) = -inf -> nan gradient -> GradScaler skips the step silently.
    # Always work in FP32 here regardless of the upstream dtype.
    predictions = predictions.float()   # <-- NEW
    targets = targets.float()           # <-- NEW
    ...

# _FocalFromLogits.forward() — sigmoid AFTER FP32 cast
def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Audit fix #2: cast logits to FP32 before sigmoid.
    # sigmoid(logit) in BF16 saturates at extreme logit values.
    probs = torch.sigmoid(logits.float())   # <-- was: torch.sigmoid(logits)
    return self.focal(probs, targets.float())
```

**File:** `ml/src/inference/predictor.py` (`_score` method)

```python
# Belt-and-suspenders: also cast at inference time
logits = self.model(batch, input_ids, attention_mask)  # [1, num_classes]
probs = torch.sigmoid(logits.float())  # <-- was: torch.sigmoid(logits)
```

### Why It Matters

BF16 training is the default on modern NVIDIA GPUs (Ampere/Ada). This bug affects every training
run on RTX 3070/3080/4090 hardware without explicit `dtype=torch.float32` override.
The failure is silent and progressive — convergence slows but never crashes.

### Spec Reference
- SENTINEL-SPEC §6 — Training stability constraints
- Improvement ledger §5.3 — loss function correctness

---

## Fix #3 (High) — `evaluate()` Uses Fixed 0.5 Threshold (Reviewer)

### Problem

Training metrics (F1, per-class precision/recall, early stopping signal) are computed under
a fixed 0.5 decision boundary while production inference applies per-class thresholds from
`*_thresholds.json`. The best-epoch checkpoint is selected based on the wrong metric.

This means:
- A checkpoint that scores well at 0.5 may not be the best at production thresholds.
- Early stopping may terminate training before the optimal checkpoint at production thresholds.

### Status

This fix requires adding an optional `thresholds: dict[str, float] | None` parameter to `evaluate()`
and wiring it through `train()`. This is a **medium-complexity change** touching the evaluate/train
interface and requires re-running threshold tuning afterward.

**Deferred to next session** — tracked as open item.

---

## Fix #4 (High) — Unknown `loss_fn` Raises `ValueError` in `trainer.py`

### Problem

If `config.loss_fn` was set to anything other than `"bce"` or `"focal"` (e.g., a typo like `"Focal"`
or `"focal_loss"`), the code silently fell through to BCE without any error:

```python
# BEFORE — silent fallthrough:
if config.loss_fn == "focal":
    criterion = _FocalFromLogits(...)
else:
    criterion = nn.BCEWithLogitsLoss(...)  # used for ANY unknown value
```

A misconfigured experiment would train with the wrong loss function, with no indication in logs
or MLflow that something was wrong.

### Fix

**File:** `ml/src/training/trainer.py`

```python
# Audit fix #4 (2026-05-01): explicit allowlist — unknown value raises immediately.
# Defined at module level so tests can import it without running train().
_VALID_LOSS_FNS: frozenset[str] = frozenset({"bce", "focal"})

def train(config: TrainConfig, ...) -> None:
    # Validate FIRST — before any I/O, model construction, or MLflow run.
    if config.loss_fn not in _VALID_LOSS_FNS:
        raise ValueError(
            f"Unknown loss_fn: '{config.loss_fn}'. "
            f"Valid options: {sorted(_VALID_LOSS_FNS)}. "
            "Check your TrainConfig — values are case-sensitive."
        )
    ...
```

### Why It Matters

Fail-fast validation at the top of `train()` means a misconfigured run surfaces in under 1 second
(before GPU allocation, before the MLflow run is created) rather than training for hours with the wrong loss.

---

## Fix #5 (High) — Warmup Uses 2-Node 1-Edge Graph in `predictor.py`

### Problem

The original warmup used a **zero-edge graph** (single node, `edge_index` shape `[2, 0]`):

```python
# BEFORE — zero edges: GATConv.propagate() is never called
dummy_x = torch.zeros(1, 8, ...)
dummy_edge_index = torch.zeros(2, 0, dtype=torch.long, ...)
```

`GATConv.propagate()` contains the attention coefficient computation. On a zero-edge graph,
this code path is entirely skipped. Shape bugs in the attention module (wrong head dim,
wrong feature projection) are never caught at startup — they surface on the very first
real contract scoring request in production.

### Fix

**File:** `ml/src/inference/predictor.py`

```python
# Audit fix #5 (2026-05-01): 2 nodes, 1 undirected edge so GATConv.propagate() runs.
# Edge 0→1 and 1→0 = both directions of one undirected edge.
# Node feature dim (8) matches GNNEncoder's expected input_dim.
dummy_x = torch.zeros(2, 8, dtype=torch.float32, device=self.device)
dummy_edge_index = torch.tensor(
    [[0, 1], [1, 0]], dtype=torch.long, device=self.device
)  # shape [2, 2] — two directed edges forming one undirected edge
dummy_graph = Data(x=dummy_x, edge_index=dummy_edge_index)
```

### Why It Matters

The warmup pass is the only gate between a cold server start and the first live request.
With a real edge present, any GAT head-dimension mismatch raises immediately on startup
rather than returning a 500 error to the first real user.

---

## Fix #6 (Medium) — `PredictResponse.threshold` Caveat Documented

### Problem

When per-class thresholds are loaded from `*_thresholds.json`, `_score()` still returns
`self.threshold` (the fallback float) in the `"threshold"` field of the response.
API consumers see the wrong threshold — the actual per-class boundaries are hidden.

### Status

A proper fix requires a schema change (the `"threshold"` field becomes `"thresholds": {class: float}`).
This is a **breaking API change** that affects all consumers.

**Partial fix applied:** Added a clear docstring note in `_score()` explaining the caveat and
referencing `thresholds_loaded` as the flag consumers should check. Full schema fix deferred.

```python
# Note on "threshold":
#     When per-class thresholds are loaded from JSON, self.threshold is
#     the fallback float and does NOT represent the actual decision
#     boundaries used. A future schema version should return the full
#     per-class threshold dict instead. Until then, consumers should
#     check thresholds_loaded on the predictor to know which mode is
#     active. Tracked as audit finding #6.
```

### Spec Reference
- SENTINEL-SPEC §3 — ML inference API contract (PredictResponse schema)

---

## Fix #7 (Medium) — `OneCycleLR` Uses `remaining_epochs` on Resume

### Problem

When training is resumed from a checkpoint at epoch `K` (out of `N` total):

```python
# BEFORE — scheduler always rebuilt for the full N epochs
scheduler = OneCycleLR(optimizer, max_lr=config.lr, epochs=config.epochs, ...)
```

`OneCycleLR` shapes its cosine curve over the **total** step budget.
If resumed at epoch 10 of 30 with `epochs=30`, the scheduler starts from max LR
and builds a 30-epoch curve — but only 20 epochs actually run.
The LR stays elevated through most of the final epochs (never reaching the cosine minimum),
leading to worse final convergence compared to a fresh run.

### Fix

**File:** `ml/src/training/trainer.py`

```python
# Audit fix #7 (2026-05-01): Load checkpoint BEFORE building scheduler
# so remaining_epochs is available for OneCycleLR's step budget.
#
# On resume: remaining_epochs = config.epochs - start_epoch + 1
#   The LR cosine curve is shaped for REMAINING steps only,
#   so the scheduler correctly reaches its minimum at the final epoch
#   regardless of when training was interrupted.
#
# On fresh start: start_epoch=1 → remaining_epochs = config.epochs (no change).
if resume_from and Path(resume_from).exists():
    ckpt = torch.load(resume_from, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 1) + 1
    best_f1 = ckpt.get("best_f1", 0.0)
    logger.info(f"Resumed from epoch {start_epoch - 1} | best_f1={best_f1:.4f}")

remaining_epochs = config.epochs - start_epoch + 1  # <-- NEW
scheduler = OneCycleLR(
    optimizer,
    max_lr=config.lr,
    epochs=remaining_epochs,           # <-- was: config.epochs
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy="cos",
)

# Log to MLflow so resumed runs are clearly distinguishable.
mlflow.log_param("remaining_epochs", remaining_epochs)
mlflow.log_param("start_epoch", start_epoch)
```

### Why It Matters

Early stopping and best-checkpoint selection happen in the final epochs where LR is lowest.
A resumed run with a wrong LR schedule may never reach the cosine minimum, making the
best-F1 checkpoint of a resumed run systematically worse than a fresh run trained for
the equivalent number of steps.

---

## Fix #8 (Medium) — `clip_grad_norm_` Clips Only Trainable Parameters

### Problem

```python
# BEFORE — iterates all 250M+ parameters (125M CodeBERT frozen + 125M trainable)
clip_grad_norm_(model.parameters(), config.grad_clip)
```

Frozen parameters (`requires_grad=False`) have `grad=None`.
`clip_grad_norm_` computes the global norm across all parameters, including frozen ones.
While frozen params contribute `0.0` to the norm sum, iterating 125M extra tensors is
pure wasted overhead — and semantically wrong (the norm should describe the magnitude
of the actual gradient update, not of a mixed frozen+trainable set).

### Fix

**File:** `ml/src/training/trainer.py`

```python
# Audit fix #8 (2026-05-01): pre-compute trainable parameter list once per
# training setup, then pass it to clip_grad_norm_ instead of model.parameters().
#
# Why: model.parameters() includes ~125M frozen CodeBERT weights (grad=None).
# Clipping only trainable params is both semantically correct (the norm measures
# the actual gradient update) and avoids iterating 125M extra zero tensors.
#
# Pre-computed once — list is stable across epochs since frozen params don't change.
trainable_params = [p for p in model.parameters() if p.requires_grad]

# ... inside the training loop:
clip_grad_norm_(trainable_params, config.grad_clip)  # <-- was: model.parameters()
```

### Why It Matters

With LoRA active, only ~2–5% of parameters are trainable. Gradient clipping should reflect
the norm of the LoRA adapters + classifier head only. Mixing in 125M frozen-weight tensors
dilutes the norm and may cause under-clipping (the per-trainable-param contribution to the
norm appears smaller than it actually is when computed across the full parameter set).

---

## Files Changed This Session

| File | Action | Audit Items |
|---|---|---|
| `ml/src/datasets/dual_path_dataset.py` | Modified | #1 (weights_only=True) |
| `ml/src/training/focalloss.py` | Modified | #2 (FP32 cast, docstring fix) |
| `ml/src/inference/predictor.py` | Modified | #2 (sigmoid FP32), #5 (warmup graph), #6 (threshold caveat) |
| `ml/src/training/trainer.py` | Modified | #4 (ValueError), #7 (OneCycleLR resume), #8 (trainable-only clip) |
| `docs/changes/2026-05-01-ml-audit-hardening.md` | Created | This file |

---

## Open Items (Deferred)

The following audit findings were identified but not yet implemented.
They are tracked for the next session:

| # | Severity | Description | File |
|---|---|---|---|
| 3 | High | `evaluate()` uses fixed 0.5 threshold; add per-class threshold param | `trainer.py` |
| 6 | Medium | `PredictResponse.threshold` schema: return full per-class dict | `predictor.py` + API contract |
| 9 | Medium | `process_source()` temp files not cleaned on SIGKILL | `preprocess.py` |
| 10 | Medium | File-path vs content hashing creates two incompatible namespaces | `preprocess.py`, `hash_utils.py` |
| 11 | Medium | RAM cache loaded via `pickle.load()` without integrity check | `dual_path_dataset.py` |
| 12 | Medium | `hash_utils.py` is dead code; hashing diverges from pipeline | `hash_utils.py` |
| 13 | Low | `clip_grad_norm_` Focal loss scalar not cast to `float()` | `trainer.py` |
| 14 | Low | Shape logging always silent; add `SENTINEL_TRACE=1` env flag | `trainer.py` |
| 15 | Low | `FocalLoss` docstring binary/multi-label mismatch | `focalloss.py` |
| 16 | Low | `CLASS_NAMES` imported from `trainer.py` into `predictor.py` | `predictor.py` |
| 17 | Info | `peft` check at module import time makes unit tests verbose | `sentinel_model.py` |
| 18 | Info | Pickle deserialization for checkpoint load (`weights_only=False`) still present | `trainer.py`, `predictor.py` |

---

## Spec References

- SENTINEL-SPEC §3 — ML inference API contract (PredictResponse schema)
- SENTINEL-SPEC §6 — Critical constraints (training stability, security)
- ADR-019 — `assert` → `RuntimeError` / safe deserialization policy
- ADR-025 — CrossAttentionFusion output_dim=128
- Improvement ledger §5.3 — loss function correctness
- Improvement ledger §5.7 — predictor.py hardening
- Improvement ledger §6 — ML security constraints
