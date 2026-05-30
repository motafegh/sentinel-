# Training — Chunk 5: `train()` Setup Phase
**File:** `ml/src/training/trainer.py` (lines 752–1270)
**Covers:** Environment config, datasets, model init, checkpoint resume, loss construction, parameter groups, `torch.compile`, `OneCycleLR`

---

## Warm-Up Recall (from Chunk 4 — evaluate & train_one_epoch)

Answer from memory. One sentence each.

1. What does the DoS gradient scaling formula `w * logits + (1-w) * logits.detach()` do, and why doesn't it affect predictions?
2. Why does gradient accumulation divide by `_actual_window` instead of `accum_steps`?
3. Fix #28: what was the bug, and what is the correct order of operations after the last micro-batch in an accumulation window?

---

## Spaced Review (from Module 4 — Models)

One question from further back:

4. In `SentinelModel.forward()`, what is the "prefix warmup guard" and what does the model do differently before and after `gnn_prefix_warmup_epochs`?

---

## P5 — Big Picture: `train()` as an Orchestrator

`train()` is a 900-line function that assembles everything built so far into a working training run. It has two phases:

**Setup phase (this chunk, lines 752–1270):**
```
train(config)
  ├── Environment: TF32, offline mode, logging
  ├── Datasets: DualPathDataset × 2, shared cache
  ├── DataLoaders: pin_memory, fork workers, WeightedSampler
  ├── pos_weight computation
  ├── Model: SentinelModel(...).to(device) + C-1 dtype check
  ├── Checkpoint resume: strict=False, version gate, optimizer restore
  ├── Loss functions: ASL/focal/BCE + aux BCE
  ├── Parameter groups: 5 groups at different LRs
  ├── AdamW optimizer
  ├── torch.compile (submodule-level)
  └── OneCycleLR scheduler + Fix #32 resume
```

**Epoch loop phase (Chunk 6, lines 1270–1645):**
```
  └── mlflow.start_run():
        for epoch in range(start_epoch, config.epochs + 1):
            train_one_epoch(...)
            evaluate(...)
            log metrics, save checkpoint, check early stopping
```

---

## Section 1 — Environment Setup (lines 756–784)

```python
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
```

> **Learning mode: Understand the pattern** — why environment mutation belongs inside `train()`, not at module level.

Setting `os.environ` at module level mutates the process environment when the file is *imported*, not just when training runs. Any test, notebook, or other module that imports `trainer.py` would have its environment mutated as a side effect. Doing it inside `train()` means: "only affect the environment when the user explicitly starts a training run."

**TF32 (TensorFloat-32):** An NVIDIA Ampere GPU (RTX 3080+, A100) hardware feature. Matmul operations are computed with 10-bit mantissa precision internally (vs 23-bit for float32) but the inputs and outputs remain float32. This gives ~2× speedup on matrix multiplication with minimal accuracy impact for deep learning.

```python
torch.backends.cuda.matmul.allow_tf32 = True   # linear layers, attention
torch.backends.cudnn.allow_tf32       = True   # convolutions (not used in SENTINEL, but correct practice)
torch.backends.cudnn.benchmark        = True   # let cuDNN find the fastest conv algorithm for your input shape
```

**`TRANSFORMERS_OFFLINE=1`:** Prevents HuggingFace from making network requests to download model files. Forces it to use the local cache. Without this, training fails on an air-gapped GPU server with a confusing SSL timeout error instead of a clear "file not found" error.

---

## Section 2 — Dataset Loading + Shared Cache (lines 797–870)

```python
train_indices = np.load(Path(config.splits_dir) / "train_indices.npy")
val_indices   = np.load(Path(config.splits_dir) / "val_indices.npy")

# Load cache once, share between train and val datasets
_shared_cache: dict | None = None
if cache_path is not None and cache_path.exists():
    import pickle
    with open(cache_path, "rb") as _f:
        _shared_cache = pickle.load(_f)

train_dataset = DualPathDataset(
    graphs_dir=config.graphs_dir,
    tokens_dir=config.tokens_dir,
    indices=train_indices.tolist(),
    label_csv=label_csv_path,
    cache_path=None,  # ← shared below
)
if _shared_cache is not None:
    train_dataset.cached_data = _shared_cache    # inject directly

val_dataset = DualPathDataset(...)
if _shared_cache is not None:
    val_dataset.cached_data = _shared_cache      # same dict object
```

> **Learning mode: Master the detail** — the shared cache pattern is an MLOps optimization with a correctness guarantee worth knowing.

**The problem:** `DualPathDataset` loads a `.pkl` cache file (2.28 GB) to avoid re-reading individual graph files on every access. If two datasets each loaded their own copy, the main process would hold 4.56 GB of cache RAM — before any model weights or batch buffers.

**The solution:** Load the cache once into `_shared_cache`. Inject the same Python dict object into both datasets. Both datasets cover *disjoint* index subsets of the same cache — `train_indices` and `val_indices` don't overlap (they came from a train/val split). There's no correctness risk: reading different keys from the same dict is safe.

**Why `cache_path=None` at construction?** `DualPathDataset.__init__` loads the cache file when `cache_path` is provided. By passing `None`, we skip that load. Then we inject the already-loaded dict directly via `train_dataset.cached_data = _shared_cache`. This relies on knowledge of the dataset's internal attribute name — not ideal (tight coupling), but it works.

### DataLoader Configuration

```python
_loader_kwargs = dict(
    batch_size=config.batch_size,
    collate_fn=dual_path_collate_fn,
    num_workers=config.num_workers,
)
if _use_workers:
    _loader_kwargs.update(
        pin_memory=True,
        persistent_workers=config.persistent_workers,
        prefetch_factor=4,
        multiprocessing_context="fork",
    )
```

> **Learning mode: Understand the pattern** — each DataLoader kwarg has a reason.

| Parameter | Value | Why |
|-----------|-------|-----|
| `pin_memory=True` | True when workers>0 | Allocates CPU tensors in pinned (page-locked) memory — enables async GPU transfer, faster `.to(device)` |
| `persistent_workers=True` | True | Workers stay alive between epochs — no re-fork overhead per epoch |
| `prefetch_factor=4` | 4 | Each worker pre-fetches 4 batches — GPU is never idle waiting for data |
| `multiprocessing_context="fork"` | "fork" | Workers inherit parent's shared cache via **copy-on-write** — no 2.28 GB copy per worker |

**`fork` vs `spawn` multiprocessing:**

- `spawn`: creates a fresh Python process, re-imports everything, re-loads cache — 2.28 GB × num_workers RAM overhead
- `fork`: copies the parent process's virtual memory space (copy-on-write). Workers see the same `_shared_cache` object at the same memory address — zero extra RAM unless the worker writes to it (it never does)

> ⚠️ **CRITICAL** — `fork` is safe only because workers never call CUDA. CUDA contexts cannot be forked — doing so causes silent corruption. The comment confirms: "workers never call CUDA so fork is safe."

---

## Section 3 — Model Init + C-1 Dtype Check (lines 889–919)

```python
model = SentinelModel(
    num_classes=config.num_classes,
    gnn_hidden_dim=config.gnn_hidden_dim,
    ...
    lora_r=config.lora_r,
    gnn_prefix_k=config.gnn_prefix_k,
    ...
).to(device)

# C-1: verify GNN parameters are float32
_gnn_dtype = next(model.gnn.conv1.parameters()).dtype
if _gnn_dtype != torch.float32:
    raise RuntimeError(
        f"C-1: GNN conv1 parameters are {_gnn_dtype} (expected float32). "
        "BF16 global dtype pollution likely — check transformer_encoder.py DTYPE FIX."
    )
```

> **Learning mode: Master the detail** — C-1 is a production guard against a real failure mode.

**The BF16 global dtype pollution problem:**

`SentinelModel.__init__` loads GraphCodeBERT via HuggingFace. During that load, the model briefly sets the global PyTorch default dtype to BF16 (for memory efficiency during BERT weight loading). If this global change isn't reverted, any parameter created *after* the BERT load — including the GNN's conv layers — will be BF16 instead of float32.

The GNN was architecturally designed for float32: its edge embeddings, LayerNorm, and skip connections assume float32 precision. BF16 GNN parameters produce visually normal-looking loss curves but converge worse — the bug is invisible without this check.

`transformer_encoder.py` has a `DTYPE FIX` (a `try/finally` block) that restores `torch.get_default_dtype()` after the BERT load. C-1 verifies that fix worked. If the GNN is BF16, training fails immediately with a clear error instead of 10 hours later with mysteriously bad results.

---

## Section 4 — Checkpoint Resume (lines 921–1002)

Resume logic has three layers, each with its own failure mode:

### Layer 1: Load model weights (lines 928–940)

```python
ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
lora_skipped  = [k for k in missing if "lora_" in k]
other_missing = [k for k in missing if "lora_" not in k]
if lora_skipped:
    logger.warning(f"Resume: {len(lora_skipped)} LoRA keys not loaded (lora_r mismatch).")
if other_missing:
    raise RuntimeError(f"Resume: {len(other_missing)} non-LoRA keys missing: {other_missing[:5]}")
```

> **Learning mode: Master the detail** — `strict=False` is necessary but must be audited.

`strict=False` allows the checkpoint to have different keys than the current model. Required because:
- Older checkpoints may lack JK parameters (`gnn.jk.*`) added in v5.2+
- A checkpoint trained with `lora_r=8` has different LoRA weight shapes than the current `lora_r=16`

But `strict=False` is dangerous — missing keys get random initialization silently. The code audits the missing keys after loading:
- LoRA mismatches: tolerated (LoRA can restart from random init)
- Non-LoRA mismatches: hard error (missing GNN or fusion weights would break training)

**`weights_only=False`:** Normally `weights_only=True` is safer (prevents arbitrary code execution during unpickling). But LoRA PEFT objects stored in the checkpoint aren't in PyTorch's safe globals list, so `weights_only=True` would fail. This is a known HuggingFace/PEFT limitation.

### Layer 2: Resume state (patience, epoch, best_f1)

```python
if config.resume_model_only:
    # Fresh start — only weights are loaded
else:
    start_epoch      = ckpt.get("epoch", 0) + 1
    best_f1          = ckpt.get("best_f1", 0.0)
    patience_counter = ckpt.get("patience_counter", 0)

    # Also check the .state.json sidecar
    _resume_state_path = Path(config.resume_from).with_suffix(".state.json")
    if _resume_state_path.exists():
        _saved_state = json.loads(_resume_state_path.read_text())
        patience_counter = _saved_state.get("patience_counter", patience_counter)
```

**Why a `.state.json` sidecar alongside the `.pt` checkpoint?**

`torch.save()` is not atomic — if the process is killed mid-write, the `.pt` file is corrupt. The `.state.json` (written with `Path.write_text()`, which is closer to atomic) holds just three small values: `epoch`, `patience_counter`, `best_f1`. On resume, if the `.pt` loads successfully, the sidecar provides the most recent `patience_counter` (the `.pt` only updates the checkpoint when F1 improves — if the last 5 epochs didn't improve, the `.pt` has stale patience).

### Layer 3: Version gate (lines 971–994)

```python
ckpt_version_str = ckpt.get("model_version", "v0.0")
ckpt_ver  = _parse_version(ckpt_version_str)
model_ver = _parse_version(MODEL_VERSION)
if ckpt_ver < model_ver:
    logger.warning(f"Checkpoint v'{ckpt_version_str}' older than current '{MODEL_VERSION}'...")

ckpt_num_classes = ckpt_cfg.get("num_classes")
if ckpt_num_classes != config.num_classes:
    raise ValueError(...)
ckpt_arch = ckpt_cfg.get("architecture")
if ckpt_arch != ARCHITECTURE:
    raise ValueError(...)
```

Versioning checks: num_classes and architecture must match exactly (hard errors). Model version mismatch is a warning — new parameters start from random init, which is acceptable for a fresh training run on a new architecture.

---

## Section 5 — Loss Construction (lines 1003–1037)

```python
aux_loss_fn: nn.Module = nn.BCEWithLogitsLoss()

if config.loss_fn == "focal":
    _focal = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    class _FocalFromLogits(nn.Module):
        def forward(self, logits, targets):
            return _focal(torch.sigmoid(logits.float()), targets)
    loss_fn = _FocalFromLogits()

elif config.loss_fn == "asl":
    loss_fn = AsymmetricLoss(
        gamma_neg=config.asl_gamma_neg,
        gamma_pos=config.asl_gamma_pos,
        clip=config.asl_clip,
    )
    # pos_weight intentionally NOT passed to ASL

else:  # "bce"
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

> **Learning mode: Master the detail** — three design decisions here that connect back to Chunks 1 and 2.

**`_FocalFromLogits` wrapper:** `FocalLoss` (from Chunk 1) expects post-sigmoid probabilities. The training loop passes raw logits. Rather than modifying `FocalLoss`, a thin wrapper applies `sigmoid` before forwarding. This is the **adapter pattern** — adapts an existing interface without changing it.

**Why ASL doesn't get `pos_weight`:**

The comment explains: with `pos_weight` (e.g., DoS = 10×) *plus* ASL's `gamma_neg=2` asymmetric scaling, DoS positives received ~20,000× gradient signal compared to easy negatives in Run 3. This caused the GNN gradient share to collapse to 24% by epoch 16. ASL's asymmetric gamma *is* the class balancing mechanism — layering `pos_weight` on top is double-amplification.

**`aux_loss_fn = BCEWithLogitsLoss()` without pos_weight:**

The three auxiliary heads (GNN eye, transformer eye, fused eye) use plain BCE. Adding pos_weight to auxiliary heads would amplify rare-class gradients through the auxiliary pathway — through already-struggling intermediate representations — exacerbating instability. The main loss carries the class-balance signal; auxiliary heads provide pathway supervision.

---

## Section 6 — Parameter Groups (lines 1039–1122)

```python
_gnn_params        = []
_lora_params       = []
_fusion_params     = []
_prefix_proj_params = []
_other_params      = []
_seen_param_ids: set = set()

for _pname, _p in model.named_parameters():
    if not _p.requires_grad or id(_p) in _seen_param_ids:
        continue
    _seen_param_ids.add(id(_p))
    if _pname.startswith("gnn.") or _pname.startswith("gnn_eye_proj."):
        _gnn_params.append(_p)
    elif "lora_" in _pname:
        _lora_params.append(_p)
    elif (_pname.startswith("fusion.") or _pname.startswith("transformer_eye_proj.")
          or _pname.startswith("classifier.") or _pname.startswith("aux_")):
        _fusion_params.append(_p)
    elif (_pname.startswith("gnn_to_bert_proj.") or _pname.startswith("prefix_type_embedding.")):
        _prefix_proj_params.append(_p)
    else:
        _other_params.append(_p)

optimizer = AdamW(_param_groups, weight_decay=config.weight_decay, fused=True)
```

> **Learning mode: Master the detail** — the five-group split, why `_seen_param_ids` is needed, and `fused=True`.

### The Five Groups and Their Effective LRs

```
Group           LR multiplier    Effective LR   Reason
────────────────────────────────────────────────────────────────────────
GNN             × 2.5            5e-4           GNN collapsed to <10% grad share at base LR
LoRA            × 0.3            6e-5           Must not catastrophically forget CodeBERT
Fusion+         × 0.5            1e-4           4-5× higher grad norm than GNN at full LR
Prefix proj     × 5.0            1e-3           Cold-started after warmup, needs fast catch-up
Other           × 1.0            2e-4           Embeddings, classifier projections
```

> ⚠️ **CRITICAL** — These multipliers were tuned through empirical failure. "Phase 2-B1 (2026-05-14): GNN collapsed to ~10% gradient share by epoch 8 in v5.1-fix28" is the specific observation that motivated the 2.5× GNN boost. Each multiplier solves a documented problem, not aesthetic preference.

**`_seen_param_ids` deduplication:**

`model.named_parameters()` can yield the same parameter multiple times if it's reachable via multiple paths (e.g., a shared weight). `id(_p)` is the Python memory address — unique per object. Without this guard, a parameter in both `gnn.*` and another path would appear in multiple groups, double-counting its gradient in `optimizer.step()`.

**LoRA `weight_decay=0.0`:**

```python
_param_groups.append({"params": _lora_params, "lr": _lora_lr, "weight_decay": 0.0})
```

Weight decay adds L2 regularization: `θ ← θ - lr*(grad + λ*θ)`. LoRA adapters (A and B matrices in `W = W₀ + BA`) are the *incremental* adaptation signal. L2 decay pushes these toward zero — directly fighting the adaptation signal. Standard PEFT practice: weight decay=0 for LoRA, keep decay on the backbone.

**`fused=True` in AdamW:**

Fused optimizer uses a CUDA kernel that performs the optimizer update in one pass over the parameters, reducing memory bandwidth. Non-fused loops over parameters in Python. At ~50M trainable parameters, fused gives ~10–15% faster optimizer steps.

---

## Section 7 — `torch.compile` (lines 1124–1152)

```python
if config.use_compile:
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.cache_size_limit = 256

    for name in ("gnn", "fusion", "classifier",
                 "gnn_eye_proj", "transformer_eye_proj", "window_pooler",
                 "aux_gnn", "aux_transformer", "aux_fused"):
        sub = getattr(model, name, None)
        if sub is not None:
            setattr(model, name, torch.compile(sub, dynamic=True))
    # transformer NOT compiled
```

> **Learning mode: Understand the pattern** — why submodule-level compilation instead of `torch.compile(model)`.

`torch.compile` traces the model's computation graph using **Dynamo** and compiles it to optimized CUDA kernels via **Triton**. It can give 20–40% speedup on compatible models.

**Why not compile the whole model?**

CodeBERT + LoRA (in `model.transformer`) has HuggingFace-style Python control flow: `if output_attentions:`, `if use_cache:`, `for layer in self.layers:` loops. These create **graph breaks** — points where Dynamo cannot trace the computation as a single graph and falls back to eager mode. Graph breaks inside the transformer contaminate the GNN and fusion compilation contexts when the whole model is compiled together, degrading their optimization.

Compiling submodules *separately* isolates graph breaks to the transformer. The GNN, fusion, classifier, and auxiliary heads compile cleanly.

**`dynamic=True`:** Tells Dynamo to handle variable-shape inputs without recompiling for each new shape. The GNN receives batches with different numbers of nodes and edges per batch — without `dynamic=True`, Dynamo would recompile on every unique `(n_nodes, n_edges)` shape combination (hundreds of compilations before hitting the cache limit).

**`cache_size_limit=256`:** Default is 8. The GNN sees many unique `(n_nodes, n_edges)` shapes. Raising the limit prevents Dynamo from falling back to eager after 8 compilations.

---

## Section 8 — OneCycleLR + Fix #32 (lines 1154–1265)

```python
steps_per_epoch = (len(train_loader) + accum_steps - 1) // accum_steps

scheduler = OneCycleLR(
    optimizer,
    max_lr=_max_lrs,            # per-group max LR list
    epochs=config.epochs,       # ← Fix #32: FULL epoch count, not remaining
    steps_per_epoch=steps_per_epoch,
    pct_start=config.warmup_pct,
    anneal_strategy="cos",
)
```

> **Learning mode: Master the detail** — `OneCycleLR` shape, the Fix #32 bug, and `max_lr` as a list.

**`OneCycleLR` shape:**

```
LR
│    ╭─────╮
│   ╱       ╲
│  ╱         ╲_______________
│ ╱
└──────────────────────────── steps
  warmup      cosine anneal
  (10%)         (90%)
```

Phase 1 (warmup): LR increases from `max_lr/div_factor` to `max_lr` over `pct_start=10%` of total steps.
Phase 2 (anneal): LR decreases from `max_lr` to `max_lr/final_div_factor` over the remaining 90%.

`max_lr=_max_lrs` is a **list** (one per param group). OneCycleLR maintains a separate schedule for each group — the GNN group peaks at `5e-4` while LoRA peaks at `6e-5`. The list must match `param_groups` order exactly.

**Fix #32 — The Scheduler Resume Bug:**

Before the fix, this was:
```python
# WRONG — old code
scheduler = OneCycleLR(..., epochs=remaining_epochs, ...)
```

On resume from epoch 50 (out of 100), `remaining_epochs=51`. The scheduler thought it had `51 × steps_per_epoch` total steps. But the checkpoint's scheduler state was created assuming `100 × steps_per_epoch` total steps. When `scheduler.load_state_dict(ckpt["scheduler"])` was called, the total steps didn't match — the scheduler was silently discarded.

After the fix:
```python
# CORRECT
scheduler = OneCycleLR(..., epochs=config.epochs, ...)  # always the FULL count
```

The scheduler is always created with the same total steps as the original run. The checkpoint's state dict (which stores `last_epoch` = the step counter reached at resume point) loads correctly, and the scheduler picks up exactly where it left off.

**Resume guard: scheduler only restores if optimizer also restored:**

```python
if "scheduler" in ckpt and _optimizer_restored:
    if ckpt["scheduler"]["total_steps"] == full_total_steps:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        logger.warning("Scheduler total_steps mismatch — starts fresh from step 0")
```

Loading a scheduler state onto a mismatched optimizer causes `base_lrs` list-length errors (the scheduler stores one `base_lr` per param group — if the group count changed, the list lengths don't match). The `_optimizer_restored` guard prevents this.

---

## AUDIT

> **[AUDIT] A5 — `cache_path=None` + attribute injection is fragile coupling**

The shared cache pattern works by passing `cache_path=None` to `DualPathDataset` (skipping internal loading) then assigning `dataset.cached_data = _shared_cache` directly. This is tight coupling to an internal implementation detail of `DualPathDataset`. If `DualPathDataset` ever renames `cached_data` or changes how it checks for a loaded cache, `train()` will silently fail to use the cache (no error, just slow training). A better design would be a `DualPathDataset.attach_cache(cache_dict)` method or a constructor parameter like `preloaded_cache=_shared_cache`.

> **[AUDIT] A6 — `_FocalFromLogits` is a local class defined inside `train()`**

```python
class _FocalFromLogits(nn.Module):
    def forward(self, logits, targets):
        return _focal(torch.sigmoid(logits.float()), targets)
loss_fn = _FocalFromLogits()
```

Defining a class inside a function creates a new class object on every call to `train()`. This class can't be pickled cleanly (its `__qualname__` includes the enclosing function name: `train.<locals>._FocalFromLogits`). If you try to checkpoint `loss_fn` with `torch.save`, pickle will fail in some contexts. In this code `loss_fn` is never checkpointed directly (only `model.state_dict()` is saved), so it's harmless — but it's a footgun for anyone who extends the checkpointing.

> **[AUDIT] A7 — `weights_only=False` in `torch.load` is a security risk in shared environments**

`torch.load(..., weights_only=False)` allows arbitrary Python code execution during checkpoint deserialization (via pickle). On a personal workstation training with your own checkpoints this is fine. On a multi-user cluster or CI system where someone else could craft a malicious checkpoint file, this is a security vulnerability. The right fix is to use `weights_only=True` and add LoRA objects to PyTorch's safe globals: `torch.serialization.add_safe_globals([LoraConfig, PeftModel])`. Not done here.

---

## Data Flow

```
config
  │
  ├── Environment setup (TF32, offline)
  ├── Load train/val indices (npy)
  ├── Load shared cache (pkl, once)
  │
  ├── DualPathDataset × 2 (inject cache)
  │     └── DataLoader × 2 (fork workers, pin_memory, prefetch=4)
  │
  ├── compute_pos_weight → [C] tensor
  │
  ├── SentinelModel(...).to(device)
  │     └── C-1 dtype check (float32 guard)
  │
  ├── [optional] load checkpoint
  │     ├── model weights (strict=False, audit missing keys)
  │     ├── training state (epoch, best_f1, patience)
  │     └── version + architecture checks
  │
  ├── loss_fn (ASL / focal+wrapper / BCE+pos_weight)
  ├── aux_loss_fn (plain BCE, no pos_weight)
  │
  ├── 5 param groups → AdamW(fused=True)
  │
  ├── [optional] torch.compile(submodules, dynamic=True)
  │
  └── OneCycleLR(epochs=config.epochs)  ← Fix #32
        [optional] load scheduler state
```

---

## 3 Things to Lock In (P10-C)

1. **Shared cache = load once, inject into both datasets** — the same Python dict object serves train and val. Zero extra RAM because Python shares object references. The `cache_path=None` + attribute injection is the implementation mechanism, but the principle is: never load the same large object twice if the data is disjoint.

2. **Five separate learning-rate groups solve five documented failures** — GNN collapse (2.5×), LoRA forgetting (0.3×, weight_decay=0), fusion dominance (0.5×), cold-start prefix (5.0×). These aren't aesthetic choices — each has a specific epoch and run number in the git history.

3. **Fix #32: the scheduler must be created with `epochs=config.epochs`, not `remaining_epochs`** — OneCycleLR encodes `total_steps` in its state. Creating it with the wrong horizon makes the checkpoint's `last_epoch` offset invalid, silently resetting the learning rate schedule on every resume.

---

## Challenge Questions

**Q1.** The shared cache uses `fork` workers. Why is `fork` safe here but dangerous in general for CUDA code? What would happen if a DataLoader worker tried to call `tensor.cuda()` after forking?

**Q2.** `strict=False` in `load_state_dict` allows missing keys. Why is this necessary for LoRA checkpoints when changing `lora_r`? What specific property of LoRA weight matrices changes when `r` changes? And what happens to the missing LoRA keys at runtime?

**Q3.** The LoRA param group has `weight_decay=0.0`. Write out the AdamW update equation for a parameter `θ` and explain exactly which term weight decay affects — and why that term damages LoRA adaptation specifically.

**Q4.** `torch.compile(sub, dynamic=True)` is used for the GNN but not the transformer. Name two specific properties of HuggingFace BERT code that cause Dynamo graph breaks, and explain what a graph break means for compilation performance.

**Q5.** Fix #32 changed `epochs=remaining_epochs` to `epochs=config.epochs`. Walk through the failure: if you resume at epoch 50 of 100, what `total_steps` does the old code create vs the new code? Why does the mismatch silently discard the scheduler state?
