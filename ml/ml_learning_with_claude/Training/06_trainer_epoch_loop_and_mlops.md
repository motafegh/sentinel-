# Training — Chunk 6: Epoch Loop & MLOps
**File:** `ml/src/training/trainer.py` (lines 1270–1645)
**Covers:** MLflow setup, per-class label smoothing, aux loss warmup, epoch loop, JK monitoring, guardrails, atomic checkpoint, early stopping, `.state.json` sidecar

---

## Warm-Up Recall (from Chunk 5 — train() Setup)

Answer from memory. One sentence each.

1. What is the shared cache pattern and why does it save ~2.28 GB of RAM compared to loading independently?
2. Why does the LoRA parameter group have `weight_decay=0.0` while all other groups use the global `weight_decay`?
3. Fix #32: why must `OneCycleLR` be created with `epochs=config.epochs` (not `remaining_epochs`) on resume?

---

## P5 — Big Picture: The Epoch Loop

This chunk is the outer layer of the training loop — the part visible to the operator. Everything in Chunks 3–5 was setup. Here the model actually trains.

```
mlflow.start_run():
    log params
    for epoch in 1..config.epochs:
        set model._current_epoch
        compute aux_loss_weight (warmup ramp)
        [NC-1] reset Adam state for prefix proj at warmup transition
        train_one_epoch(...)          ← Chunk 4
        evaluate(...)                 ← Chunk 4
        gc.collect() + empty_cache()
        log metrics to MLflow
        [Phase 2-C1] log JK attention weights
        [IMP-M2] log prefix attention diagnostic
        log VRAM
        [BUG-M10] run training guardrails
        if f1_macro > best_f1:
            save checkpoint (atomic)
        else:
            patience_counter += 1
            if patience_counter >= patience: break
        write .state.json sidecar
    log checkpoint artifact to MLflow
```

---

## Section 1 — MLflow Setup + Parameter Logging (lines 1276–1326)

```python
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment(config.experiment_name)

with mlflow.start_run(run_name=config.run_name):
    params = {
        "num_classes":                 config.num_classes,
        "epochs":                      config.epochs,
        "effective_batch_size":        config.batch_size * accum_steps,
        "lr":                          config.lr,
        "loss_fn":                     config.loss_fn,
        "architecture":                ARCHITECTURE,
        ...
    }
    if pos_weight is not None:
        for name, pw in zip(CLASS_NAMES, pos_weight.cpu().tolist()):
            params[f"pos_weight_{name}"] = round(pw, 3)
    mlflow.log_params(params)
```

> **Learning mode: Understand the pattern** — MLflow as the experiment tracking layer.

**MLflow** is an open-source platform for tracking ML experiments. `sqlite:///mlruns.db` uses a local SQLite file as the backend — no server required. Alternatives: PostgreSQL (production), MLflow Tracking Server, or cloud backends (AWS, GCP). The local SQLite is appropriate here for solo research.

**`mlflow.start_run()`** is a context manager — everything inside it is associated with one training run. Parameters are logged once; metrics are logged per-step.

**`mlflow.log_params(params)` vs `mlflow.log_metrics(...)`:**
- `log_params`: hyperparameters that don't change during the run — logged once
- `log_metric`: values that change per epoch — logged with a `step=epoch` argument

**Why log `effective_batch_size = batch_size × accum_steps`?**

`batch_size=8` with `gradient_accumulation_steps=8` is functionally equivalent to a `batch_size=64` optimizer step. Logging only `batch_size` would make runs with different accumulation look incomparable. Logging `effective_batch_size` makes the actual training scale visible.

**Per-class pos_weights:** Logged as `pos_weight_Reentrancy=1.0`, `pos_weight_DenialOfService=9.8`, etc. This creates a searchable record of what class balancing was applied — essential for understanding why training improved or regressed across runs.

> **[AUDIT] A1 — `mlflow.set_tracking_uri("sqlite:///mlruns.db")` is hardcoded inside `train()`**

The SQLite path is not in `TrainConfig` — it can't be overridden without modifying the source. This is fine for a research project but would need to be configurable in production (different environments use different backends). A better design would be `config.mlflow_tracking_uri: str = "sqlite:///mlruns.db"`.

---

## Section 2 — Pre-Loop Setup (lines 1331–1341)

```python
_class_eps = torch.tensor(
    [config.class_label_smoothing.get(c, 0.05) for c in CLASS_NAMES[:config.num_classes]],
    dtype=torch.float32, device=device,
)

_consecutive_allzeros  = 0
_consecutive_gnn_coll  = 0
_class_death_counter   = [0] * config.num_classes
```

**`_class_eps` is built once before the loop**, not per-epoch. The per-class smoothing values don't change during training — computing this tensor once and passing it to `train_one_epoch` every epoch avoids redundant list lookups and tensor construction.

**Three guardrail counters (BUG-M10):**

These count consecutive epochs of each failure mode:
- `_consecutive_allzeros`: all-zeros prediction collapse (Hamming >0.85)
- `_consecutive_gnn_coll`: GNN gradient share <10%
- `_class_death_counter[c]`: per-class F1=0.0 streak

They're initialized to zero at the start of each run. On resume, they're reset — the guardrails don't accumulate across restarts.

---

## Section 3 — Per-Epoch Prefix Warmup Status (lines 1351–1363)

```python
if config.gnn_prefix_k > 0:
    _prefix_active = (epoch >= config.gnn_prefix_warmup_epochs)
    logger.info(f"GNN prefix K={config.gnn_prefix_k}: {'ACTIVE' if _prefix_active else f'WARMUP (starts ep{config.gnn_prefix_warmup_epochs})'}")
    mlflow.log_metric("prefix_active", int(_prefix_active), step=epoch)

    _proj = getattr(model, "gnn_to_bert_proj", None)
    if _proj is not None:
        _proj_norm = _proj.weight.data.norm().item()
        mlflow.log_metric("prefix_proj_weight_norm", _proj_norm, step=epoch)
```

> **Learning mode: Awareness only** — this is diagnostic monitoring for the prefix injection feature.

The weight norm of `gnn_to_bert_proj` (the linear layer that maps GNN node embeddings into the transformer's embedding space) should be:
- Constant during warmup (the projection isn't called, receives no gradient)
- Drifting from random init starting at `gnn_prefix_warmup_epochs`

If the norm stays flat after warmup, the projection isn't learning — a signal to increase `gnn_prefix_proj_lr_mult`.

---

## Section 4 — Aux Loss Warmup Ramp (Fix #33, lines 1371–1382)

```python
if config.aux_loss_warmup_epochs > 0 and epoch <= config.aux_loss_warmup_epochs:
    warmup_frac = (epoch - 1) / config.aux_loss_warmup_epochs
    effective_aux_weight = config.aux_loss_weight * warmup_frac
else:
    effective_aux_weight = config.aux_loss_weight
```

> **Learning mode: Master the detail** — the `(epoch - 1)` formula and why it starts at 0, not at `1/warmup_epochs`.

**What this ramp does:**

```
epoch=1:   warmup_frac = 0/8  = 0.000  → aux_weight = 0.000  (zero aux loss)
epoch=2:   warmup_frac = 1/8  = 0.125  → aux_weight = 0.0375
epoch=4:   warmup_frac = 3/8  = 0.375  → aux_weight = 0.1125
epoch=8:   warmup_frac = 7/8  = 0.875  → aux_weight = 0.2625
epoch=9:   warmup_frac = —   = 1.0    → aux_weight = 0.3    (full weight)
```

> ⚠️ **CRITICAL** — `(epoch - 1)` not `epoch` so that epoch 1 starts at **zero** auxiliary loss. Before Fix #33, the formula was `epoch / warmup_epochs`, which started at `1/8 = 12.5%` on epoch 1. Observed effect: "aux loss 2–4× main loss at epoch 1, causing the main classifier to learn slowly and rare classes to stay at F1=0." Starting at zero gives the main classifier a clean signal for the first epoch, then gradually introduces pathway supervision.

**Why aux loss warmup is needed at all:**

The three auxiliary heads (GNN eye, transformer eye, fused eye) share the same backbone but have their own classification projections initialized randomly. On epoch 1, their random projections produce high-confidence wrong predictions → high aux loss → large gradients that dominate over the main loss signal. The ramp suppresses this early instability.

---

## Section 5 — NC-1: Adam State Reset at Warmup Transition (lines 1384–1409)

```python
if (
    epoch == config.gnn_prefix_warmup_epochs
    and config.gnn_prefix_k > 0
    and config.gnn_prefix_proj_reset_on_warmup
):
    for _pg in optimizer.param_groups:
        if _pg.get("name") == "prefix_proj":
            for _p in _pg["params"]:
                optimizer.state[_p] = {}   # clear Adam state
            mlflow.log_metric("prefix_proj_adam_reset", 1, step=epoch)
            break
```

> **Learning mode: Master the detail** — why clearing Adam's state for one param group is the right operation here.

**The problem:** `gnn_to_bert_proj` received zero gradient during the warmup phase (the prefix code path was suppressed — `model._current_epoch < warmup_epochs`). AdamW's moment estimates (`m₁` and `m₂`) for these parameters were never updated — or were initialized to near-zero from the first gradient that randomly fires. When the prefix activates at epoch `warmup_epochs`, the stale moments give a distorted initial update.

**AdamW moment estimates:**
```
m₁ ← β₁ * m₁ + (1-β₁) * grad         ← first moment (EMA of gradients)
m₂ ← β₂ * m₂ + (1-β₂) * grad²        ← second moment (EMA of squared gradients)
θ  ← θ - lr * m₁_hat / (sqrt(m₂_hat) + ε)
```

If `m₁ ≈ 0` and `m₂ ≈ 0` (never updated), the denominator `sqrt(m₂_hat) + ε ≈ ε` — very small — making the update `lr * m₁_hat / ε` potentially very large on the first real gradient.

Setting `optimizer.state[_p] = {}` erases the accumulated moment history. When the first real gradient arrives at epoch `warmup_epochs`, Adam starts fresh with zero moments — the correct initial state for a cold-start parameter.

**Why only `prefix_proj`, not all params?** Every other parameter group has been updating normally during warmup. Resetting their moments would discard valuable gradient history.

---

## Section 6 — Per-Epoch Calls + Memory Management (lines 1411–1442)

```python
_epoch_t0 = time.perf_counter()
train_loss, nan_batch_count, last_gnn_share = train_one_epoch(...)

val_metrics = evaluate(...)

# Fix #27: release CUDA caching allocator free-blocks between epochs
gc.collect()
torch.cuda.empty_cache()
```

**`gc.collect()`** triggers Python's garbage collector to find and free reference cycles. Under normal conditions Python's reference counting handles most deallocation, but circular references (common in complex objects like model graphs) accumulate. Between epochs is the right time — doing it inside the epoch loop would cause a CUDA sync on every batch.

**`torch.cuda.empty_cache()`** releases PyTorch's CUDA memory cache back to the OS. This doesn't free actively-used tensors — it only releases the "reserved but currently unused" pool. After an epoch ends, many intermediate activation tensors from the final batch are freed but still held in the allocator's pool. Releasing them reduces reserved VRAM, making headroom for the next epoch.

> **[AUDIT] A2 — `gc.collect()` between epochs may not be needed**

Python's cyclic garbage collector runs automatically at threshold intervals (after ~700 new objects by default). An explicit `gc.collect()` call between epochs is conservative — it guarantees cycles from the epoch are freed before the next starts. On typical hardware this takes <1ms. The comment says it was added as Fix #27 alongside `empty_cache()` — the two are paired for completeness, but `gc.collect()` alone rarely makes a measurable difference.

---

## Section 7 — MLflow Metric Logging (lines 1444–1525)

```python
mlflow.log_metric("train_loss",       train_loss,       step=epoch)
mlflow.log_metric("nan_batch_count",  nan_batch_count,  step=epoch)
mlflow.log_metric("gnn_grad_share",   last_gnn_share,   step=epoch)
mlflow.log_metric("val_f1_macro",     val_metrics["f1_macro"],  step=epoch)
mlflow.log_metric("val_f1_micro",     val_metrics["f1_micro"],  step=epoch)
mlflow.log_metric("val_hamming",      val_metrics["hamming"],   step=epoch)
for name in CLASS_NAMES[:config.num_classes]:
    mlflow.log_metric(f"val_f1_{name}", val_metrics[f"f1_{name}"], step=epoch)
```

> **Learning mode: Understand the pattern** — the difference between params (logged once) and metrics (logged per step).

**Per-class F1 metrics:** 10 individual `val_f1_Reentrancy`, `val_f1_DenialOfService`, etc. metrics per epoch. This gives full visibility into class-level progress. In the MLflow UI, you can plot `val_f1_DenialOfService` over epochs and see exactly when DoS starts learning.

**`nan_batch_count`:** How many batches had NaN or Inf loss this epoch. Zero is expected. >5% suggests a gradient explosion or data issue. Logged every epoch so trends are visible.

**`gnn_grad_share`:** The GNN's fraction of the total gradient norm (from `train_one_epoch`). Should stay above 10%. A declining trend over epochs signals GNN collapse.

### JK Attention Monitoring (Phase 2-C1, lines 1448–1483)

```python
if config.gnn_use_jk and hasattr(model.gnn, "jk"):
    _jk_cache = getattr(model.gnn.jk, "last_weights", None)
    _jk_std_cache = getattr(model.gnn.jk, "last_weight_stds", None)
    if _jk_cache is not None:
        _jk_w = _jk_cache.cpu().tolist()   # [3] — mean phase weights
        _jk_s = _jk_std_cache.cpu().tolist() # [3] — std of per-node weights
        for _pi, (_w, _s) in enumerate(zip(_jk_w, _jk_s), start=1):
            mlflow.log_metric(f"jk_phase{_pi}_weight", _w, step=epoch)
            mlflow.log_metric(f"jk_phase{_pi}_std",    _s, step=epoch)
```

Recall from Models Chunk 3 (JK attention): `last_weights` and `last_weight_stds` are buffers on the JK module that cache the mean and std of attention weights across the batch. The trainer reads these after each epoch and logs them.

**Two collapse alerts:**

```python
if max(_jk_s) < 0.015 and epoch >= 3:
    logger.warning("⚠ JK STD COLLAPSE: per-node routing has collapsed...")
```
If all phase weight stds are <0.015, every node gets the same JK weights — the per-node routing has collapsed to a global average.

```python
if _max_w > 0.80:
    logger.warning("⚠ JK phase dominance: Phase N has 80% attention weight...")
```
If one phase gets 80%+ weight, the other two phases are contributing almost nothing.

### Prefix Attention Diagnostic (IMP-M2, lines 1485–1515)

```python
if config.gnn_prefix_k > 0 and epoch >= config.gnn_prefix_warmup_epochs:
    _diag_graphs, _diag_tokens, _ = next(iter(val_loader))
    model.eval()
    _prefix_attn = model.compute_prefix_attention_mean(
        _diag_graphs.to(device),
        _diag_tokens["input_ids"].to(device),
        _diag_tokens["attention_mask"].to(device),
    )
    model.train()
    if _prefix_attn < 0.002:
        logger.warning("⚠ prefix_attention_mean < 0.002 — transformer may be ignoring GNN prefix")
```

`compute_prefix_attention_mean` (Models Chunk 8) runs one validation batch in eval mode and returns the average attention weight the transformer's last layer placed on the prefix tokens. Near-zero (<0.002) for 5+ post-warmup epochs = the transformer is ignoring the GNN prefix — the prefix injection isn't working.

> ⚠️ **CRITICAL** — `model.eval()` is called before the diagnostic, then `model.train()` is called after. This must not be forgotten. If `model.train()` is accidentally removed, the rest of the epoch runs with Dropout disabled and BatchNorm using running stats — training on eval-mode produces artificially low loss and misleadingly good F1 metrics.

---

## Section 8 — Training Guardrails (BUG-M10, lines 1545–1585)

Three independent monitors run every epoch:

### Guard 1: All-Zeros Collapse

```python
if val_metrics["hamming"] > 0.85:
    _consecutive_allzeros += 1
    if _consecutive_allzeros >= 3:
        logger.critical(
            "ALL-ZEROS COLLAPSE DETECTED: Hamming > 0.85 for 3 consecutive epochs. "
            "Consider reducing gamma_neg, increasing dos_loss_weight, or checking weighted sampler."
        )
else:
    _consecutive_allzeros = 0
```

**Hamming > 0.85 = 85% of predictions wrong.** Since there are 10 classes and most contracts have 0–2 vulnerabilities, predicting all-zeros gives Hamming ≈ `(proportion of non-zero labels) ≈ 40%` — so Hamming=0.85 is far worse than always predicting zero. Wait — actually: Hamming loss = fraction of wrong predictions. If the model predicts all zeros and 15% of cells are positive (label=1), Hamming = 0.15 (not 0.85). Hamming=0.85 means 85% of all cells are predicted wrong — the model is predicting 1 for cells that are 0.

Actually the all-zeros collapse is Hamming ≈ `fraction of positive cells ≈ 0.15`. But if `gamma_neg` is too aggressive, all probabilities collapse to near zero → the model predicts zero for everything → Hamming = fraction of positive cells ≈ 0.15, NOT 0.85.

The guard threshold of 0.85 is for the opposite failure: the model predicts 1 everywhere → Hamming = fraction of negative cells ≈ 0.85. The comment "ALL-ZEROS COLLAPSE" is somewhat misleading — it fires when predictions are *wrong* for 85% of cells (more likely an all-ones collapse or random predictions).

> **[AUDIT] A3 — The "all-zeros collapse" label is inverted**
> 
> Predicting all-zeros gives Hamming ≈ 0.15 (15% of cells are positive). The threshold of 0.85 is actually closer to detecting an **all-ones collapse** or random predictions. The real all-zeros collapse signature is `f1_macro ≈ 0.0` with `hamming ≈ 0.15`, not `hamming > 0.85`. The guard is monitoring the wrong metric for the stated failure mode — though it does catch *a* collapse, just not the one it's named for.

### Guard 2: Class Death

```python
for _ci, _cname in enumerate(CLASS_NAMES):
    _cf1 = val_metrics.get(f"f1_{_cname}", 0.0)
    if _cf1 == 0.0:
        _class_death_counter[_ci] += 1
        if _class_death_counter[_ci] >= 5:
            logger.warning(f"CLASS DEATH: {_cname} F1=0.0 for {_class_death_counter[_ci]} consecutive epochs.")
    else:
        _class_death_counter[_ci] = 0
```

A class is "dead" when the model never predicts it correctly — F1=0.0 means either always predicts 0 (precision=undefined, recall=0) or predicts 1 on wrong samples (precision=0). After 5 consecutive epochs at F1=0.0, this fires as a warning. No automatic action is taken — it's a signal to adjust that class's loss weight or oversampling.

### Guard 3: GNN Collapse

```python
if last_gnn_share < 0.10:
    _consecutive_gnn_coll += 1
    if _consecutive_gnn_coll >= 5:
        logger.critical("GNN COLLAPSE: gnn_grad_share < 10% for 5 consecutive epochs.")
else:
    _consecutive_gnn_coll = 0
```

Mirrors the intra-epoch check in `train_one_epoch` but at epoch granularity. A sustained GNN grad share below 10% for 5 epochs is a harder failure than transient spikes — the GNN has effectively stopped contributing. The fix: increase `gnn_lr_multiplier`.

---

## Section 9 — Checkpoint Save (lines 1587–1628)

### Atomic Write Pattern

```python
if val_metrics["f1_macro"] > best_f1:
    best_f1 = val_metrics["f1_macro"]
    patience_counter = 0

    _tmp_path = checkpoint_path.with_suffix(".tmp")
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
            "model_version":    MODEL_VERSION,
            "config":           {**dataclasses.asdict(config), ...},
        },
        _tmp_path,
    )
    _tmp_path.replace(checkpoint_path)   # atomic on POSIX
```

> **Learning mode: Master the detail** — the atomic write pattern and the `._orig_mod.` key stripping.

**Atomic write:** `torch.save(_tmp_path)` then `_tmp_path.replace(checkpoint_path)`. If the process is killed during `torch.save`, only the `.tmp` file is corrupt — the existing `.pt` checkpoint is untouched. `Path.replace()` is an atomic rename on POSIX filesystems (Linux/macOS): the OS either completes the rename or doesn't — there's no state where `checkpoint_path` is partially written.

**`._orig_mod.` key stripping:**

`torch.compile(sub)` wraps the module in an `OptimizedModule`. When you call `model.state_dict()`, compiled submodule keys appear as `gnn._orig_mod.conv1.weight` instead of `gnn.conv1.weight`. Loading this checkpoint with an uncompiled model (e.g., for inference) would fail — the key names don't match. The replacement strips `._orig_mod.` from all keys, making the checkpoint format identical whether the model was compiled or not.

**What's saved in the checkpoint:**

| Key | Type | Why |
|-----|------|-----|
| `model` | dict | Parameter weights — the core artifact |
| `optimizer` | dict | Adam moment estimates — needed for true resume |
| `scheduler` | dict | LR schedule state — needed for Fix #32 |
| `epoch` | int | Where to resume from |
| `best_f1` | float | Early stopping baseline |
| `patience_counter` | int | Early stopping streak (also in `.state.json`) |
| `model_version` | str | Version gate (Phase 1-A6) |
| `config` | dict | Hyperparameters — reconstructable on any machine |

### `.state.json` Sidecar (lines 1625–1628)

```python
_state_path = checkpoint_path.with_suffix(".state.json")
_state_path.write_text(
    json.dumps({"epoch": epoch, "patience_counter": patience_counter, "best_f1": best_f1})
)
```

Written **every epoch**, not just on improvement. This is the correct behavior: `patience_counter` increments every epoch where F1 doesn't improve. The `.pt` checkpoint only saves when F1 improves — so after 5 non-improving epochs, the `.pt` has `patience_counter=0` but the real value is 5. The `.state.json` always holds the true current state.

---

## Section 10 — Early Stopping (lines 1618–1623)

```python
else:
    patience_counter += 1
    logger.info(f"No improvement for {patience_counter}/{config.early_stop_patience} epochs")
    if patience_counter >= config.early_stop_patience:
        logger.info(f"Early stopping after {epoch} epochs (best F1={best_f1:.4f})")
        break
```

`early_stop_patience=30` (v6 default). If validation F1 hasn't improved for 30 consecutive epochs, training stops. The outer `for` loop has no `else` clause — the `break` exits the loop, then execution continues after the loop.

**Why `eval_threshold=0.35` matters here:**

As noted in `TrainConfig` (Chunk 3): with `eval_threshold=0.5`, minority class predictions clustering at 0.35–0.50 flip between F1=0 and F1=0.15 on a ±0.03 probability shift. This creates ±0.04 macro-F1 noise per epoch — the patience counter fires on noise, not real failure. `eval_threshold=0.35` moves those predictions away from the decision boundary, giving a stable F1 signal for early stopping.

---

## Section 11 — MLflow Artifact Logging + Return (lines 1630–1640)

```python
    if checkpoint_path.exists():
        mlflow.log_artifact(str(checkpoint_path))

    logger.info(f"✅ Training complete. Best val F1-macro: {best_f1:.4f}")

return {
    "best_f1_macro":   best_f1,
    "final_epoch":     final_epoch,
    "early_stopped":   patience_counter >= config.early_stop_patience,
    "checkpoint_path": str(checkpoint_path),
}
```

`mlflow.log_artifact` uploads the checkpoint file to the MLflow artifact store (a local directory `./mlartifacts/` by default). This makes the best checkpoint retrievable from the MLflow UI — you can download it or reference it in downstream scripts by run ID.

**`mlflow.start_run()` context manager:**

The `with mlflow.start_run():` block that wraps the entire epoch loop means: if training crashes with an exception inside the loop, `mlflow.end_run(status="FAILED")` is called automatically. The run is marked as failed in the tracking UI, not left as "RUNNING" forever.

---

## Full `train()` Data Flow (both chunks combined)

```
train(config)
    │
    ├── Setup (Chunk 5)
    │   ├── env, datasets, dataloaders, model, checkpoint, losses, optimizer, scheduler
    │
    └── mlflow.start_run():
          │
          ├── log_params(...)
          │
          └── for epoch in 1..100:
                │
                ├── model._current_epoch = epoch
                ├── compute effective_aux_weight (ramp)
                ├── [NC-1] reset Adam state if epoch == warmup transition
                │
                ├── train_one_epoch(...)  → train_loss, nan_count, gnn_share
                ├── evaluate(...)         → val_metrics (F1, Hamming, per-class)
                │
                ├── gc.collect() + empty_cache()
                │
                ├── log_metric(train_loss, gnn_share, val_f1_*, ...)
                ├── [JK monitoring] log phase weights + std alerts
                ├── [prefix diagnostic] log attention_mean
                │
                ├── [guardrails] all-zeros, class death, GNN collapse
                │
                ├── if f1_macro > best_f1:
                │       save checkpoint (atomic .tmp → .pt)
                │       patience_counter = 0
                │   else:
                │       patience_counter += 1
                │       if >= patience: break
                │
                └── write .state.json sidecar
          │
          └── log_artifact(checkpoint)
```

---

## 3 Things to Lock In (P10-C)

1. **Aux loss warmup: `(epoch-1)/warmup_epochs` starts at 0, not 1/N** — the `-1` is load-bearing. Without it, the first epoch has 12.5% aux weight and the raw auxiliary gradient (from randomly-initialized head projections) dominates, stopping the main classifier from learning. Starting at zero gives one clean epoch.

2. **Atomic checkpoint write: save to `.tmp`, then `Path.replace()`** — if the process is killed mid-save, the real checkpoint is intact. Never write directly to the final path. Combine with `._orig_mod.` key stripping to make compiled and uncompiled checkpoints load identically.

3. **`.state.json` written every epoch; `.pt` written only on improvement** — the checkpoint's `patience_counter` goes stale after the first non-improving epoch. The sidecar always has the true current state. Resume uses the sidecar if it exists and is from the right epoch.

---

## Challenge Questions

**Q1.** The aux loss warmup formula is `warmup_frac = (epoch - 1) / config.aux_loss_warmup_epochs`. What would the warmup schedule look like if it were `epoch / config.aux_loss_warmup_epochs` instead? What specific failure does the `- 1` prevent?

**Q2.** NC-1 resets `optimizer.state[_p] = {}` for the `prefix_proj` param group at the warmup transition epoch. Write out what the AdamW update looks like for one parameter in the first gradient step *before* the reset vs *after* the reset. Why does a near-zero `m₂` make the update potentially large?

**Q3.** Checkpoint saves use `_tmp_path.replace(checkpoint_path)`. On a Linux filesystem, `os.rename()` (which `Path.replace()` calls internally) is atomic. What does "atomic" mean in this context — what failure modes does it protect against? What failure mode does it *not* protect against?

**Q4.** The `._orig_mod.` key stripping uses: `{k.replace("._orig_mod.", "."): v for k, v in _sd.items()}`. Under what condition is `any("._orig_mod." in k for k in _sd)` True, and what causes those keys to appear? Why would inference fail without stripping?

**Q5.** `mlflow.log_artifact(checkpoint_path)` is called after `mlflow.start_run()` exits (it's inside the context manager but after the loop). What does MLflow do if training crashes mid-epoch before reaching this line? Is the checkpoint still available for use outside MLflow?
