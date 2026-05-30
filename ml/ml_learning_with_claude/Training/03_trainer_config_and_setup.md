# Training — Chunk 3: TrainConfig & Setup Infrastructure
**File:** `ml/src/training/trainer.py` (lines 1–415)
**Covers:** Module-level constants, VRAM helpers, `TrainConfig` dataclass, `compute_pos_weight`

---

## Warm-Up Recall (from Chunk 2 — AsymmetricLoss)

Answer from memory. One sentence each.

1. What does the `clip` parameter in `AsymmetricLoss` do, and what happens to the gradient when `prob < clip`?
2. Why is `gamma_neg` registered as a `buffer` rather than stored as a plain Python float?
3. ASL uses `focal_neg = prob_neg ** gamma_neg` rather than `(1 - pt) ** gamma`. Why does using `prob_neg` (not `1 - prob_neg`) correctly suppress easy negatives?

---

## P5 — Big Picture: What `trainer.py` Is

Every file studied so far (`gnn_encoder.py`, `transformer_encoder.py`, `fusion_layer.py`, `sentinel_model.py`, `losses.py`, `focalloss.py`) built the model and its loss functions. None of them runs the training. `trainer.py` is the **conductor** — it owns the complete training lifecycle:

```
trainer.py
├── TrainConfig          ← all hyperparameters in one place
├── compute_pos_weight   ← class imbalance weights from data
├── evaluate()           ← val loop, F1/Hamming, threshold sweep
├── train_one_epoch()    ← inner training loop
├── _build_weighted_sampler() ← dataset-level rebalancing
└── train()              ← full orchestrator: datasets, model, optimizer,
                            scheduler, checkpoint, MLflow, epoch loop
```

At 1,645 lines it is the largest file in the system. The complexity is not algorithmic — it's **operational**: getting all the pieces to work together reliably across 100 epochs, resuming from checkpoints, diagnosing failures, preventing the half-dozen ways a training run can silently go wrong.

**Cross-file relationships (already taught):**
- `DualPathDataset` + `dual_path_collate_fn` — Module 3 (Datasets)
- `SentinelModel` — Module 4 (Models, chunks 7–8)
- `AsymmetricLoss` + `FocalLoss` — Training chunks 1–2

**Cross-file relationships (not yet taught):**
- `MLflow` — experiment tracking library. Logs parameters + metrics to a SQLite database. Chunk 6 (epoch loop) goes deep on this.
- `train.py` (scripts) — the CLI wrapper that constructs a `TrainConfig` from command-line arguments and calls `train()`. Module 8.

---

## Section 1 — The Audit Log at the Top (lines 1–46)

```python
"""
trainer.py — SENTINEL Training Loop (v7 — Three-Eye GNN+CodeBERT+LoRA)
...
Fix #26 — need_weights=False on MHA in fusion_layer.py
Fix #27 — gc.collect() + torch.cuda.empty_cache() between epochs
Fix #28 — batch_size default 16→8 (8 GB GPU compatibility)
...
Fix #34 — VRAM usage logged every epoch so OOM risk is visible.
"""
```

> **Learning mode: Awareness only** — don't memorize the fix list. Know it exists.

This docstring is a production MLOps pattern: **every non-obvious fix is numbered and explained at the top of the file, not buried in the code**. When a training run produces strange behavior, you scan this list first. Fix numbers appear as inline comments throughout the code: `# Fix #28`, `# Fix #32`, etc. They cross-reference the cause-and-effect chain without requiring you to reconstruct it from git history.

> **[AUDIT] A1 — Fix #28 describes a cross-file change (batch_size in `train.py`)**

Fix #28 says "Fix #30 — train.py CLI --batch-size default 32→8 (see that file)". Two separate files changed for the same root cause (OOM on 8 GB GPU). The docstring names both. This is correct maintenance practice — without cross-file references, it's easy to revert one fix and miss the other.

---

## Section 2 — Module-Level Constants (lines 89–107)

```python
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
```

> **Learning mode: Understand the pattern** — the single-source-of-truth design.

`CLASS_NAMES` is defined once here and used everywhere: `compute_pos_weight`, `evaluate`, `train_one_epoch`, checkpoint saves, MLflow logging. This is the **single source of truth** pattern — if a class is renamed or the order changes, there is exactly one place to edit.

`ARCHITECTURE = "three_eye_v8"` is written into every checkpoint file. On resume, the checkpoint's architecture string is compared to the current `ARCHITECTURE` constant. If they differ (e.g., loading a two-eye v5 checkpoint into a three-eye v8 trainer), training fails fast with a clear error.

`_VALID_LOSS_FNS: frozenset` — the leading underscore marks it as module-private. `frozenset` (immutable set) is used instead of `list` because:
1. `in` checks are O(1) vs O(n) for lists (negligible at size 3, but the intent is clear)
2. Immutable — no accidental `.add()` mutation from test code

---

## Section 3 — VRAM Helpers (lines 112–150)

```python
def _vram_pct() -> float:
    if not torch.cuda.is_available():
        return 0.0
    reserved = torch.cuda.memory_reserved()
    total    = torch.cuda.get_device_properties(0).total_memory
    return reserved / total if total > 0 else 0.0

def _vram_str() -> str:
    alloc = torch.cuda.memory_allocated()  / (1024**3)
    reser = torch.cuda.memory_reserved()   / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"{reser:.1f}/{total:.1f} GiB ({_vram_pct():.1%})"
```

> **Learning mode: Understand the pattern** — the `reserved` vs `allocated` distinction is interview-relevant.

**`memory_reserved()` vs `memory_allocated()`:**

PyTorch's CUDA memory allocator works in two layers:

```
GPU Physical Memory
    └── Reserved (torch.cuda.memory_reserved)
            └── Allocated (torch.cuda.memory_allocated)
            └── Fragmented-but-unused (reserved - allocated)
```

- `memory_allocated()` = tensors actively in use
- `memory_reserved()` = all memory PyTorch has claimed from the OS, including freed-but-cached blocks
- `torch.cuda.empty_cache()` releases reserved-but-unused blocks back to the OS

The VRAM monitor uses `memory_reserved()` because that's what determines whether the next `.to(device)` call will OOM — PyTorch can reuse its cache, but if the cache is 90% full and you need a large allocation, the OS allocation may fail.

```python
def _parse_version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.lstrip("v").split(".") if x.isdigit())
```

> **Learning mode: Understand the pattern** — used for checkpoint version comparison.

Parses `"v8.0"` → `(8, 0)`. Tuples compare element-by-element in Python: `(5, 2) < (8, 0)` is `True`. This lets the resume logic detect when a checkpoint was saved by an older model version (`ckpt_ver < model_ver`) and warn that some parameters (e.g., JK weights) will be randomly initialized.

---

## Section 4 — `TrainConfig` Dataclass (lines 157–365)

```python
@dataclass
class TrainConfig:
    graphs_dir:      str = "ml/data/graphs"
    tokens_dir:      str = "ml/data/tokens_windowed"
    ...
    epochs:          int   = 100
    batch_size:      int   = 8
    lr:              float = 2e-4
    ...
```

> **Learning mode: Understand the pattern** — the `@dataclass` configuration object pattern in ML systems.

A `@dataclass` auto-generates `__init__`, `__repr__`, `__eq__` from field annotations. Every field gets a default value so `TrainConfig()` gives a runnable baseline. The CLI wrapper (`train.py`) overrides fields from command-line arguments.

**Why `@dataclass` over a plain `dict`?**

| Approach | IDE support | Type checking | Validation | Serializable |
|----------|------------|---------------|-----------|-------------|
| `dict` | No autocomplete | No | Manual | Yes (JSON) |
| `@dataclass` | Full autocomplete | Yes (mypy) | `__post_init__` | `dataclasses.asdict()` |
| `argparse.Namespace` | Partial | No | Via actions | Manual |

`dataclasses.asdict(config)` converts the entire config to a nested dict — used when saving it inside the checkpoint and logging to MLflow.

### Hyperparameter Groups

The config fields fall into natural groups. Understanding the groups is more important than memorizing individual values.

**Paths group:**
```python
graphs_dir, tokens_dir, splits_dir, checkpoint_dir, label_csv, cache_path
```
All file system locations the trainer needs. Overridable for different machines/environments.

**Model architecture group:**
```python
num_classes, fusion_output_dim, fusion_dropout,
gnn_hidden_dim, gnn_layers, gnn_heads, gnn_dropout,
gnn_use_jk, gnn_jk_mode, gnn_phase2_edge_types,
lora_r, lora_alpha, lora_dropout, lora_target_modules,
gnn_prefix_k, gnn_prefix_warmup_epochs
```
These are passed directly to `SentinelModel(...)`. If they change between runs, checkpoint resume may fail.

**Training dynamics group:**
```python
epochs, batch_size, lr, weight_decay,
gnn_lr_multiplier, lora_lr_multiplier, fusion_lr_multiplier,
grad_clip, warmup_pct, use_amp,
gradient_accumulation_steps
```

> ⚠️ **CRITICAL** — Four separate learning rates for one model. This is not over-engineering.
>
> The GNN, LoRA adapters, fusion layer, and prefix projection all learn at fundamentally different rates:
> - GNN was getting <10% gradient share at base LR → `gnn_lr_multiplier=2.5`
> - LoRA must not catastrophically forget CodeBERT's pretrained weights → `lora_lr_multiplier=0.3`
> - Fusion at full LR produced 4–5× the GNN gradient norm → `fusion_lr_multiplier=0.5`
> - Prefix projection is cold-started after warmup → `gnn_prefix_proj_lr_mult=5.0`

**Loss function group:**
```python
loss_fn, focal_gamma, focal_alpha,
asl_gamma_neg, asl_gamma_pos, asl_clip,
label_smoothing, class_label_smoothing,
dos_loss_weight, pos_weight_cap, pos_weight_min_samples
```

Note: `asl_gamma_neg=2.0` in the config (not 4.0 as in the ASL defaults). The comment explains: `γ⁻=4` caused all-zeros collapse with 60% zero-label rows (BUG-C4). Per-class label smoothing (`class_label_smoothing`) replaces uniform `label_smoothing` (BUG-M9) — each class gets its own ε calibrated to its estimated noise rate.

**MLOps group:**
```python
experiment_name, run_name, log_interval,
num_workers, persistent_workers, use_compile,
smoke_subsample_fraction, use_weighted_sampler
```

`smoke_subsample_fraction` enables fast "smoke runs" — training on 5–10% of data to check for crashes and basic convergence before committing to a full 100-epoch run.

**Resume group:**
```python
resume_from, resume_model_only, force_optimizer_reset
```

Three flavors of resume:
- `resume_from=None` → fresh run
- `resume_from=path, resume_model_only=True` → load weights only, fresh optimizer/scheduler/patience
- `resume_from=path, resume_model_only=False` → full resume: weights + optimizer + scheduler + patience counter

### `__post_init__` Validation (lines 334–365)

```python
def __post_init__(self) -> None:
    if self.gnn_layers < 4:
        raise ValueError("gnn_layers < 4 not supported ...")
    if self.gnn_layers > 8:
        logger.warning("gnn_layers > 8 is non-standard ...")
    if self.gradient_accumulation_steps < 1:
        raise ValueError(...)
    unknown_cls = set(self.class_label_smoothing) - set(CLASS_NAMES)
    if unknown_cls:
        raise ValueError(f"NH-2: unknown class names: {unknown_cls}")
    invalid_eps = {k: v for k, v in self.class_label_smoothing.items() if not (0.0 <= v < 1.0)}
    if invalid_eps:
        raise ValueError(...)
```

> **Learning mode: Understand the pattern** — fail-fast validation at configuration time, not at runtime.

`__post_init__` is called automatically after `__init__` in a `@dataclass`. It's the right place for validation because:
1. Errors appear at construction (`TrainConfig(gnn_layers=2)`), not after 30 minutes of data loading
2. The error message can explain *why* the constraint exists (three-phase architecture requires ≥4 layers)

**Asymmetric validation for `gnn_layers`:**
- `< 4` → hard `raise ValueError` (breaks the three-phase architecture invariant)
- `> 8` → soft `logger.warning` (non-standard but allowed for experiments)

This is correct severity calibration. Fewer layers breaks the architecture. More layers is just untested.

---

## Section 5 — `compute_pos_weight` (lines 371–415)

```python
def compute_pos_weight(
    label_csv, train_indices, num_classes, device,
    pos_weight_min_samples=0, pos_weight_cap=20.0,
) -> torch.Tensor:
    df = pd.read_csv(label_csv)
    train_labels = df[CLASS_NAMES].values[train_indices]
    N = len(train_labels)
    pos_counts = train_labels.sum(axis=0)

    pos_weight_vals = []
    for c, pos in enumerate(pos_counts):
        if pos == 0:
            logger.warning(...)
            pos = 1
        if pos_weight_min_samples > 0 and pos >= pos_weight_min_samples:
            pos_weight_vals.append(1.0)                        # well-represented: no amplification
        else:
            raw_ratio = float(N - pos) / float(pos)
            pos_weight_vals.append(min(float(raw_ratio ** 0.5), pos_weight_cap))

    return torch.tensor(pos_weight_vals, dtype=torch.float32, device=device)
```

> **Learning mode: Master the detail** — three design decisions here that each have a specific rationale.

**Decision 1 — sqrt scaling (`raw_ratio ** 0.5`):**

The naive inverse-frequency weight would be `(N - pos) / pos`. For DoS with 257 positives out of 44,000:
- Raw ratio: `(44000 - 257) / 257 ≈ 170`
- A 170× weight means one DoS positive gets 170× the gradient signal of a negative

This causes gradient spikes that destabilize the loss scale for the entire batch. Sqrt scaling: `sqrt(170) ≈ 13` — still amplifies rare classes, but within a workable range.

**Decision 2 — `pos_weight_min_samples` cap (set to 3000):**

Classes with ≥3000 training positives get `pos_weight=1.0`. The comment explains the specific failure: Reentrancy has ~3500 train positives and its raw ratio would produce `pos_weight ≈ 2.82`. In v5.2, this 2.82× amplification combined with CodeBERT's natural tendency to flag any external call as Reentrancy caused a behavioral collapse — the model predicted Reentrancy for everything. The min_samples guard caps it at 1.0.

**Decision 3 — `pos_weight_cap=10.0`:**

Even after sqrt scaling, a truly data-starved class (DoS: 257 samples → sqrt(170) ≈ 13) can exceed a safe gradient ceiling. Hard cap at 10.0 prevents runaway gradient spikes. The cap was 20.0 before (M-1/H-4 fix).

> ⚠️ **CRITICAL** — `pos_weight` is computed from **training indices only** (`train_labels = label_matrix[train_indices]`). If you accidentally included validation indices, you'd leak val set label statistics into the training objective — data leakage. This is a subtle point: the function takes `train_indices` as an argument precisely to enforce this.

**Data flow:**
```
label_csv (CSV on disk)
    └── read all labels into matrix [N_total, C]
    └── slice to train_indices only
    └── sum per column → pos_counts [C]
    └── for each class:
            if pos >= min_samples → weight=1.0
            else → weight=min(sqrt((N-pos)/pos), cap)
    └── return as [C] float32 tensor on device
```

---

## AUDIT

> **[AUDIT] A2 — `pos_weight` computed from CSV, but the dataset uses a cached `.pkl`**

`compute_pos_weight` reads `label_csv` directly from disk. The training dataset (`DualPathDataset`) may use `cache_path` (a `.pkl` file) that was built from the same CSV. There's an implicit assumption that both are consistent. If the label CSV is updated (e.g., re-labelled data) but the `.pkl` cache is not rebuilt, `compute_pos_weight` will use the new labels but the model will train on the old cached labels. The trainer has no check for this inconsistency.

> **[AUDIT] A3 — `pos == 0` zero-positive guard: uses `pos=1` as a sentinel**

```python
if pos == 0:
    logger.warning(...)
    pos = 1
```

Setting `pos=1` when there are zero positives gives `raw_ratio = (N-1)/1 ≈ N`. After sqrt and cap, this class gets `pos_weight=10.0`. The model will try to predict positives for a class with **no positive examples in the training set** — it will always be wrong, contributing a constant gradient push toward predicting positive. A better design would be to set `pos_weight=0.0` for zero-positive classes (or skip them entirely), signalling "no gradient for this class". The current behavior may cause training instability for completely absent classes.

> **[AUDIT] A4 — `compute_pos_weight` only used with `loss_fn="bce"`**

The `train()` function only passes `pos_weight` to `BCEWithLogitsLoss`. When `loss_fn="asl"`, `pos_weight` is computed but intentionally not passed to `AsymmetricLoss` (the comment says "ASL handles class imbalance via asymmetric gamma — adding pos_weight creates double-amplification"). But `compute_pos_weight` is always called unconditionally. This is a minor inefficiency: if `loss_fn="asl"` (the default), the function reads the CSV, does the computation, logs it, and then the result is discarded. The computation is not expensive, but the log output is misleading — it prints "pos_weight sqrt-scaled" every run even when those weights aren't used.

---

## Alternative: Config Approaches (P7)

### Hydra (Facebook Research)

```python
# Hydra config approach
@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    train(cfg)
```

Hydra uses YAML files + Python dataclasses and supports config composition, command-line overrides, and automatic experiment sweeps. Trade-off: more powerful but adds a dependency and a YAML config layer that some teams find harder to read than pure Python.

### Plain `argparse`

```python
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=2e-4)
args = parser.parse_args()
```

No validation, no serialization, no type checking. Works for scripts but doesn't scale to 30+ hyperparameters.

### Why SENTINEL uses `@dataclass`

Pure Python — no external dependency. `dataclasses.asdict()` serializes cleanly into checkpoints and MLflow. `__post_init__` handles validation. IDE autocomplete works. It's the right tool at this scale.

---

## Data Flow

```
caller (train.py CLI or notebook)
        │
        ▼
    TrainConfig(...)
        │ __post_init__ validates constraints
        │
        ├──► compute_pos_weight(label_csv, train_indices, ...)
        │           │
        │           ▼
        │    [C] pos_weight tensor (device)
        │
        └──► passed to train(config)
```

---

## 3 Things to Lock In (P10-C)

1. **Four separate learning rates — not one** — GNN, LoRA, fusion, and prefix projection each have a multiplier because they learn at fundamentally different rates and interact differently with the frozen CodeBERT backbone. Single-LR training caused GNN collapse and LoRA catastrophic forgetting in v5.x runs.

2. **`pos_weight` uses sqrt scaling + two caps** — raw inverse-frequency weights produce gradient spikes. `sqrt((N-pos)/pos)` tames the scale; `pos_weight_min_samples=3000` prevents amplifying well-represented classes (fixed Reentrancy collapse in v5.2); `pos_weight_cap=10.0` hard-limits the maximum.

3. **`__post_init__` validates at construction time** — failing fast at `TrainConfig(gnn_layers=2)` vs failing 30 minutes into training when the architecture breaks is the difference between a 2-second error and a wasted GPU hour.

---

## Challenge Questions

**Q1.** `TrainConfig` stores `lora_target_modules` as `list[str]` with a default factory. Why must mutable defaults in dataclasses use `field(default_factory=...)` instead of a direct list literal? What goes wrong without it?

**Q2.** `_parse_version("v5.2")` returns `(5, 2)`. How does Python compare tuples, and what comparison makes `(5, 2) < (8, 0)` evaluate to `True`?

**Q3.** `compute_pos_weight` uses `sqrt((N-pos)/pos)` instead of `(N-pos)/pos`. For a class with 257 positives out of 44,000 total:
- Compute the raw ratio
- Compute the sqrt-scaled weight
- Apply `pos_weight_cap=10.0`
- What does the final weight communicate to `BCEWithLogitsLoss`?

**Q4.** The config has both `eval_threshold=0.35` and `threshold=0.5`. Why are these different values, and what problem does the lower eval threshold solve during training? (The comment explains the specific failure mode — describe it.)

**Q5.** `torch.cuda.memory_reserved()` is used for VRAM monitoring instead of `memory_allocated()`. Explain the two-layer VRAM model and why reserved is more meaningful for OOM prediction.
