# Training — Chunk 4: Evaluation & Inner Training Loop
**File:** `ml/src/training/trainer.py` (lines 421–746)
**Covers:** `evaluate()`, `train_one_epoch()`, `_grad_norm()`, `_build_weighted_sampler()`

---

## Warm-Up Recall (from Chunk 3 — TrainConfig & Setup)

Answer from memory. One sentence each.

1. Why does `compute_pos_weight` use `sqrt((N-pos)/pos)` instead of the raw ratio `(N-pos)/pos`?
2. What does `pos_weight_min_samples=3000` do, and which specific training failure motivated it?
3. `TrainConfig` has both `eval_threshold=0.35` and `threshold=0.5`. What problem does the lower eval threshold solve during training?

---

## P5 — Big Picture: Three Functions, One Concern

This chunk covers the two innermost functions of the training loop:

```
train()                          ← outer orchestrator (Chunks 5–6)
  └── for each epoch:
        train_one_epoch()        ← THIS CHUNK: inner training loop
        evaluate()               ← THIS CHUNK: validation loop
```

Plus two supporting helpers:
- `_grad_norm()` — measures gradient norms by parameter group
- `_build_weighted_sampler()` — controls which samples the DataLoader sees

These four functions contain the core ML logic: how SENTINEL actually learns.

---

## Section 1 — `evaluate()` (lines 421–493)

### What it computes

Multi-label evaluation for SENTINEL requires more than a single accuracy number. Three metrics are computed:

**F1-macro:** Average F1 score across all 10 classes, each weighted equally.
- A class with 5 true positives and a class with 5,000 get equal weight.
- Captures whether the model works for *rare* classes (good for vulnerability detection — a 99% Reentrancy detector that misses all DoS is useless).

**F1-micro:** Aggregate F1 over all class predictions jointly.
- Weighted by class frequency — dominated by common classes.
- Complementary to macro: if macro >> micro, the model is better on common classes.

**Hamming loss:** Fraction of (sample, class) pairs incorrectly predicted.
- `hamming = wrong_predictions / (B × C)` where B=batch size, C=10 classes
- A Hamming of 0.85 means 85% of predictions are wrong — signals all-zeros collapse.

**F1 per class:** `f1_{ClassName}` for each of the 10 vulnerability types. Logged to MLflow individually. Exposes "class death" — a class stuck at F1=0.0 for many epochs.

### The evaluation loop

```python
def evaluate(model, loader, device, threshold=0.5, use_amp=True, tune_thresholds=False):
    model.eval()
    all_probs, all_true = [], []

    with torch.no_grad():
        for batch in tqdm(loader, ...):
            graphs, tokens, labels = batch
            graphs    = graphs.to(device)
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            labels    = labels.to(device).float()

            with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(graphs, input_ids, attention_mask)

            probs = torch.sigmoid(logits.float())    # ← note: .float() after autocast
            all_probs.append(probs.cpu().numpy())
            all_true.append(labels.long().cpu().numpy())
```

> **Learning mode: Understand the pattern** — `torch.no_grad()` + `model.eval()` are both required and serve different purposes.

**`model.eval()` vs `torch.no_grad()`:**

| What it does | `model.eval()` | `torch.no_grad()` |
|---|---|---|
| Disables Dropout | ✅ | ❌ |
| Freezes BatchNorm | ✅ | ❌ |
| Stops gradient computation | ❌ | ✅ |
| Saves memory (no grad graph) | ❌ | ✅ |

Both are needed. `model.eval()` changes the model's behavior (Dropout passes all activations through at full probability, BatchNorm uses running stats). `torch.no_grad()` stops PyTorch from building the computation graph, saving memory and time.

**`logits.float()` after autocast:**

`logits` exits the autocast block as BF16. `torch.sigmoid` on BF16 has insufficient precision for threshold comparisons near 0.5 (BF16 precision ≈ 0.015 — larger than the `eval_threshold=0.35` distance from common predictions). `.float()` here is the correct place — outside the autocast block, after the forward pass.

### BUG-M8: Threshold Sweep (lines 470–492)

```python
if tune_thresholds:
    _candidates = np.linspace(0.1, 0.9, 19)
    tuned = []
    for c in range(num_classes):
        best_t, best_f1 = threshold, 0.0
        for t in _candidates:
            preds_c = (y_probs[:, c] >= t).astype(int)
            f1_c = f1_score(y_true[:, c], preds_c, zero_division=0)
            if f1_c > best_f1:
                best_f1, best_t = f1_c, t
        tuned.append(float(best_t))

    y_pred_tuned = np.stack(
        [(y_probs[:, c] >= tuned[c]).astype(int) for c in range(num_classes)],
        axis=1,
    )
    f1_tuned = f1_score(y_true, y_pred_tuned, average="macro", zero_division=0)
    metrics["f1_macro_tuned"]   = float(f1_tuned)
    metrics["tuned_thresholds"] = tuned
```

> **Learning mode: Master the detail** — this is the per-class threshold optimization pattern you'll be asked about.

**The problem it solves:** A global threshold of 0.5 is suboptimal for multi-label problems with class imbalance. A rare class (DoS: 257 positives) may have its predictions concentrated in `[0.3, 0.6]` — a threshold of 0.5 will miss many. A common class (Reentrancy: 3500 positives) may have predictions in `[0.6, 0.95]` where 0.5 is fine.

**What it does:** For each class independently, sweep 19 candidate thresholds from 0.1 to 0.9 in steps of ~0.0444. Pick the threshold that maximizes that class's F1. Apply all 10 per-class thresholds simultaneously to get `f1_macro_tuned`.

**Why this is called during training (not just post-training):** It gives an upper bound on F1 that the model *could* achieve with optimal thresholds. If `f1_macro_tuned >> f1_macro`, the model has learned good probability distributions but the threshold is misaligned — tunable. If `f1_macro_tuned ≈ f1_macro`, the problem is model quality, not threshold.

> **[AUDIT] A1 — Threshold sweep on the validation set during training is data snooping**

The tuned thresholds in `metrics["tuned_thresholds"]` are computed on the validation set. If those thresholds are then used for early stopping (they're not directly — `patience` uses `f1_macro`, not `f1_macro_tuned`) the evaluation would be snooped. As implemented, `f1_macro_tuned` is logged to MLflow but early stopping uses `val_metrics["f1_macro"]`. This is correct — the sweep here is diagnostic. The *production* threshold optimization lives in `tune_threshold.py` (a separate script that uses a held-out test set). But it's worth knowing the risk: any time you optimize a threshold on a set and report that set's performance, you've overfit.

---

## Section 2 — `train_one_epoch()` (lines 499–698)

This is the heart of the training loop. Let's go through it in logical phases.

### Phase A: Setup (lines 519–538)

```python
model.train()
trainable_params = [p for p in model.parameters() if p.requires_grad]
accum_steps = max(1, gradient_accumulation_steps)
optimizer_step = 0
nan_loss_count = 0
last_gnn_share = 0.0
_run_main = _run_gnn_a = _run_tf_a = _run_fus_a = 0.0
_run_n = 0
_interval_t0 = time.perf_counter()
```

`trainable_params` is a pre-computed list of parameters with `requires_grad=True`. Used by `clip_grad_norm_` — passing the full model would include frozen CodeBERT parameters, wasting time computing norms of zero-gradient tensors.

### Phase B: Label Smoothing (lines 547–551)

```python
if class_eps is not None:
    labels = labels * (1.0 - class_eps) + 0.5 * class_eps
elif label_smoothing > 0.0:
    labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
```

> **Learning mode: Master the detail** — the label smoothing formula and why `0.5` not `0.0`.

**Label smoothing** prevents the model from becoming overconfident. A hard label `y=1.0` with `ε=0.10` becomes `y=0.95`:
```
soft_label = y * (1 - ε) + 0.5 * ε
           = 1.0 * 0.90 + 0.5 * 0.10
           = 0.90 + 0.05
           = 0.95
```

And a hard label `y=0.0` becomes `y=0.05`:
```
soft_label = 0.0 * 0.90 + 0.5 * 0.10 = 0.05
```

**Why `0.5` and not `0.0`?** Using `0.5 * ε` as the smoothing target instead of `ε` itself means both classes are smoothed symmetrically toward the midpoint. This is the standard formulation from the original label smoothing paper (Szegedy et al., 2016). Using `0.0` or `1.0` as the smoothing anchor would bias the model in one direction.

**`class_eps` is a `[C]` tensor:** Different classes have different noise rates. Reentrancy has 14% confirmed noise (many contracts with external calls mislabelled as Reentrancy). Timestamp has only 5% noise (structural checks exist). Per-class smoothing applies the right ε to each column independently.

### Phase C: Forward Pass with AMP (lines 553–570)

```python
with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
    logits, aux = model(graphs, input_ids, attention_mask, return_aux=True)

    # DoS gradient scaling
    if dos_loss_weight < 1.0:
        _dos_idx = CLASS_NAMES.index("DenialOfService")
        _logits_for_loss = logits.clone()
        _logits_for_loss[:, _dos_idx] = (
            dos_loss_weight * logits[:, _dos_idx]
            + (1.0 - dos_loss_weight) * logits[:, _dos_idx].detach()
        )
    else:
        _logits_for_loss = logits
```

> **Learning mode: Master the detail** — the DoS gradient scaling trick is interview-worthy.

**`return_aux=True`** tells `SentinelModel` to return the auxiliary head outputs alongside the main logits: `aux = {"gnn": ..., "transformer": ..., "fused": ..., "jk_entropy": ...}`.

**DoS gradient scaling (BUG-H6):**

DoS (DenialOfService) had only 3 training positives in early runs — essentially no signal. Even at 243 positives (current), it's the rarest class and produced unstable gradients when trained at full weight.

The trick: blend between the real logit (which has a gradient) and a `.detach()`'d copy (which has no gradient):

```
_logits_for_loss[:, dos_idx] = w * logits_dos + (1-w) * logits_dos.detach()
```

- `w=0.0` → `0 * logits + 1 * detach(logits)` = fully detached, zero gradient for DoS
- `w=0.5` → 50% of the normal gradient for DoS
- `w=1.0` → full gradient (normal, `_logits_for_loss = logits`)

> ⚠️ **CRITICAL** — `.detach()` removes a tensor from the computation graph. When you call `.backward()`, gradients flow through the attached part only. The formula `w * x + (1-w) * x.detach()` is a precise gradient scaler: it doesn't change the *predictions* (both terms evaluate to `x` numerically) but controls what fraction of `x`'s gradient is passed back. This is a common pattern for soft gradient masking.

### Phase D: Loss Combination (lines 571–610)

```python
main_loss  = loss_fn(_logits_for_loss, labels)

loss_gnn_a  = aux_loss_fn(aux["gnn"],         labels)
loss_tf_a   = aux_loss_fn(aux["transformer"], labels)
loss_fus_a  = aux_loss_fn(aux["fused"],       labels)
aux_loss    = loss_gnn_a + loss_tf_a + loss_fus_a

_window_start  = (batch_idx // accum_steps) * accum_steps
_actual_window = min(accum_steps, len(loader) - _window_start)
loss = (main_loss + aux_loss_weight * aux_loss) / _actual_window

# JK entropy regularizer
if jk_entropy_reg_lambda > 0.0:
    _jk_ent = aux.get("jk_entropy")
    if _jk_ent is not None:
        _H_max = math.log(3)
        _jk_reg = jk_entropy_reg_lambda * (_H_max - _jk_ent.clamp(max=_H_max))
        loss = loss + _jk_reg / _actual_window
```

> **Learning mode: Master the detail** — four loss components, the `_actual_window` normalization, and the JK regularizer are all interview-relevant.

**Four loss components:**

```
loss = (main_loss + aux_weight * (loss_gnn + loss_tf + loss_fused)) / window
```

- `main_loss`: ASL (or focal/BCE) on the fused three-eye output
- `loss_gnn_a`: BCE on the GNN-eye auxiliary head output
- `loss_tf_a`: BCE on the transformer-eye auxiliary head output
- `loss_fus_a`: BCE on the fused-eye auxiliary head output

The auxiliary heads receive supervision on their intermediate representations — they're forced to classify vulnerabilities using only their own path's representation, before cross-attention fusion. This prevents one path from free-riding on the other.

**`_actual_window` — the correct gradient accumulation denominator:**

Gradient accumulation divides the loss by the accumulation window size so that the accumulated gradient equals the gradient you'd get from one large batch. Standard approach: divide by `accum_steps`.

Problem: the last window of an epoch may have fewer batches. If `len(loader)=10` and `accum_steps=4`:
- Window 0: batches 0–3 → 4 batches
- Window 1: batches 4–7 → 4 batches
- Window 2: batches 8–9 → **2 batches**

If batch 8 and 9 divide by `accum_steps=4`, they're scaled as if they'll be followed by 2 more batches that never come. Their accumulated gradient would be 2× too small. `_actual_window = min(accum_steps, len(loader) - _window_start)` computes the actual window size — for the tail window it's 2, not 4.

**JK entropy regularizer (C-3):**

From Chunk 3 of the Models module: JK attention assigns weights to Phase 1, 2, and 3 representations. The entropy of those weights is:
- `H ≈ 0`: one phase gets all weight (collapsed — the other phases are ignored)
- `H ≈ log(3)`: uniform weights across 3 phases (JK isn't routing, just averaging)
- `H` in between: healthy

The regularizer penalizes low entropy:
```
jk_reg = λ × (log(3) - H)
```
When `H = log(3)` (uniform): penalty = 0. When `H = 0` (collapsed): penalty = `λ × log(3) ≈ 0.011`. This nudges JK away from collapsing to a single phase without forcing uniform weights.

### Phase E: Backward + Accumulation Step (lines 619–678)

```python
loss.backward()

is_last_batch = (batch_idx + 1 == len(loader))
is_accum_step = ((batch_idx + 1) % accum_steps == 0) or is_last_batch

if is_accum_step:
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)

    optimizer_step += 1
    should_log = (optimizer_step % log_interval == 0)
    if should_log:
        gnn_norm     = _grad_norm(model.gnn_eye_proj)
        gnn_enc_norm = _grad_norm(model.gnn)
        tf_norm      = _grad_norm(model.transformer_eye_proj)
        fused_norm   = _grad_norm(model.fusion)
        ...

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
```

> **Learning mode: Master the detail** — gradient accumulation, clip timing, and `set_to_none=True` are all interview-level topics.

**Gradient accumulation:**

```
Micro-batch 0: loss.backward()  ← gradients accumulate in .grad
Micro-batch 1: loss.backward()  ← gradients ADD to .grad (not replace)
...
Micro-batch N-1: loss.backward()
  → clip_grad_norm_()
  → optimizer.step()         ← update using sum of N micro-batch gradients
  → optimizer.zero_grad()    ← reset .grad to None
```

`loss.backward()` **adds** to existing `.grad` tensors (it doesn't zero them). This is the mechanic that makes accumulation work: gradients from N micro-batches sum before the optimizer step, producing the same update as one large batch of N × batch_size examples.

**Grad clipping before `optimizer.step()`:**

`clip_grad_norm_` rescales all gradients so the total L2 norm ≤ `grad_clip=1.0`. It's called *after* accumulation (all micro-batch gradients are in `.grad`) but *before* `optimizer.step()`. Order matters:

```
loss.backward() × N  → .grad = sum of N gradients
clip_grad_norm_()    → .grad rescaled if norm > 1.0
_grad_norm()         → READ the post-clip gradient norms (Fix #28)
optimizer.step()     → apply .grad to parameters
optimizer.zero_grad(set_to_none=True)  → clear .grad
```

> ⚠️ **CRITICAL** — Fix #28 documents a real bug: `_grad_norm()` was previously called *after* `zero_grad(set_to_none=True)`. With `set_to_none=True`, zeroing sets `.grad = None` (not `.grad = 0`). Checking a `None` grad returns 0 — every gradient norm was logged as 0.000 for the entire training history of earlier runs.

**`zero_grad(set_to_none=True)` vs `zero_grad()`:**

`zero_grad()` fills gradients with zeros. `zero_grad(set_to_none=True)` sets `.grad = None`. The `None` version is faster because PyTorch skips initialization when allocating gradient buffers for the next backward pass (instead of filling zeros, it allocates fresh). In practice: ~5–10% faster per optimizer step.

**GNN collapse detection (lines 657–669):**

```python
if _gnn_share < 0.10:
    _gnn_collapse_streak += 1
    if _gnn_collapse_streak >= 3:
        logger.warning("⚠ GNN collapse: share=... for 3 consecutive intervals")
```

`_gnn_share = gnn_norm / total_norm` — the fraction of the total gradient norm contributed by the GNN. If the GNN contributes <10% of the gradient for 3 consecutive log intervals, something is wrong: the GNN is not learning, only the transformer and fusion are driving updates. The fix is to increase `gnn_lr_multiplier`.

---

## Section 3 — `_grad_norm()` (lines 701–706)

```python
def _grad_norm(module: nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().norm(2).item() ** 2
    return total_sq ** 0.5
```

> **Learning mode: Understand the pattern** — L2 norm across all parameter gradients in a module.

This computes the **global L2 gradient norm** for one module:

```
||g||₂ = sqrt(Σ ||g_i||₂²)
```

where `g_i` is the gradient tensor for parameter `i`. Two details:

1. `.detach()` — we're reading gradient values, not computing anything differentiable. `.detach()` prevents accidental computation graph creation.
2. `.float()` — gradients may be BF16 (if the parameter is BF16). Converting to float32 gives accurate norms. (The GNN is float32, but LoRA adapters train in BF16 under autocast.)

`if p.grad is not None` — after `zero_grad(set_to_none=True)`, gradients are `None`, not zero. The guard prevents a `NoneType` attribute error.

---

## Section 4 — `_build_weighted_sampler()` (lines 712–746)

```python
def _build_weighted_sampler(dataset, label_csv_path, mode):
    df = pd.read_csv(label_csv_path).set_index("md5_stem")
    weights = []
    for md5 in dataset.paired_hashes:
        row = df.loc[md5]
        if mode == "positive":
            has_vuln = any(float(row.get(cls, 0)) == 1.0 for cls in CLASS_NAMES)
            w = 3.0 if has_vuln else 1.0
        elif mode == "DoS-only":
            w = 39.0 if float(row.get("DenialOfService", 0)) == 1.0 else 1.0
        elif mode == "all-rare":
            n_pos = sum(float(row.get(cls, 0)) for cls in CLASS_NAMES)
            w = float(n_pos) if n_pos > 0 else 1.0
        else:
            w = 1.0
        weights.append(w)

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
```

> **Learning mode: Understand the pattern** — `WeightedRandomSampler` is the PyTorch mechanism for oversampling rare examples.

**Why sampling, not loss weighting?**

`pos_weight` (loss weighting) amplifies the gradient contribution of rare positives. `WeightedRandomSampler` increases how *often* rare samples appear in batches. These are complementary interventions:
- Loss weighting: each sample appears once per epoch, but its loss counts more
- Weighted sampling: rare samples appear more often, each time contributing a normal-weight loss

**The three modes:**

| Mode | What it does | When to use |
|------|-------------|------------|
| `"positive"` | 3× weight for any vulnerable contract | BUG-H10: 60% of contracts have no vulnerability — shifts baseline ratio |
| `"DoS-only"` | 39× weight for DoS contracts | DoS had <3 positives in early runs — needed extreme oversampling |
| `"all-rare"` | Weight = number of positive labels in that contract | Multi-label rarity correction |
| `"none"` | Uniform (natural frequency) | Ablation / debugging |

**`replacement=True`:** Sampling with replacement means the same contract may appear multiple times in an epoch. This is expected for oversampled rare classes — it's by design.

**`num_samples=len(weights)`:** Each epoch still processes the same number of samples as the original dataset. The distribution has shifted, but the epoch length hasn't.

---

## AUDIT

> **[AUDIT] A2 — `evaluate` calls `model(graphs, input_ids, attention_mask)` without `return_aux`**

`train_one_epoch` calls `model(..., return_aux=True)` to get auxiliary head outputs. `evaluate` calls `model(graphs, input_ids, attention_mask)` without `return_aux`. This means evaluation only scores the main fused output — the three-eye loss breakdown is only visible during training. If one of the auxiliary heads has failed (e.g., GNN head stuck at F1=0), `evaluate` won't report it. A more complete evaluation would also score auxiliary head predictions.

> **[AUDIT] A3 — BUG-M8 threshold sweep sweeps all 10 classes every epoch on the full validation set**

The inner loop is `O(num_classes × num_thresholds × N)` where N = validation set size. With 10 classes, 19 thresholds, and ~8,000 val samples: `10 × 19 × 8000 = 1,520,000` comparisons per epoch. This is CPU numpy and runs fast (~50ms), but it happens on every single epoch, regardless of whether the model is still improving. The sweep could be run every 5–10 epochs without meaningful information loss.

> **[AUDIT] A4 — Per-class label smoothing is applied AFTER the batch is moved to device, inside the training loop**

```python
labels = labels.to(device).float()
if class_eps is not None:
    labels = labels * (1.0 - class_eps) + 0.5 * class_eps
```

`class_eps` is a `[C]` tensor on `device`. Every micro-batch recomputes `labels * (1 - class_eps)` in-place — that's fine. But this creates a new tensor on every forward pass rather than caching smoothed labels. For AMP+BF16 training, the cost is negligible, but if `class_eps` had different values per batch (e.g., for curriculum smoothing), this pattern would be the right one to use.

---

## Data Flow

```
DataLoader batch
    │
    ▼ (per micro-batch)
labels → per-class label smoothing → soft_labels
logits, aux ← model(graphs, tokens)  [inside autocast BF16]
    │
    ├── DoS gradient scaling (if dos_loss_weight < 1.0)
    │
    ├── main_loss   = loss_fn(logits_for_loss, soft_labels)
    ├── loss_gnn_a  = bce(aux["gnn"], soft_labels)
    ├── loss_tf_a   = bce(aux["transformer"], soft_labels)
    ├── loss_fus_a  = bce(aux["fused"], soft_labels)
    │
    └── loss = (main + aux_weight × (gnn_a + tf_a + fus_a)) / actual_window
              + jk_entropy_reg (if enabled)
                    │
                    ▼
              loss.backward()   ← accumulate into .grad
                    │
              (if is_accum_step):
                    │
              clip_grad_norm_() ← clamp total grad norm ≤ 1.0
              _grad_norm()      ← read norms (Fix #28: before zero_grad)
              optimizer.step()  ← apply .grad to params
              scheduler.step()  ← advance LR schedule
              zero_grad(set_to_none=True)
```

---

## 3 Things to Lock In (P10-C)

1. **The DoS gradient scaling trick** — `w * x + (1-w) * x.detach()` doesn't change predictions (both terms evaluate to `x`) but scales gradients. At `w=0.5`, DoS gets half the normal gradient. This is the correct way to reduce a class's gradient contribution without removing it entirely or changing inference.

2. **Gradient accumulation divides by `_actual_window`, not `accum_steps`** — the tail window of an epoch has fewer micro-batches. Dividing by the fixed `accum_steps` would under-scale the tail gradient. `_actual_window = min(accum_steps, remaining_batches)` is the correct normalization.

3. **Fix #28: read gradient norms after `clip_grad_norm_` but before `zero_grad(set_to_none=True)`** — `set_to_none=True` sets `.grad = None`. Reading a `None` gradient returns 0. Every gradient norm was logged as 0.000 in every earlier run because this order was wrong. The fix is: clip → read → step → zero.

---

## Challenge Questions

**Q1.** `train_one_epoch` calls `model(graphs, input_ids, attention_mask, return_aux=True)` and gets back `logits, aux`. What is in `aux`, and why do the auxiliary heads use plain `BCEWithLogitsLoss` instead of `AsymmetricLoss` with `pos_weight`?

**Q2.** The label smoothing formula is `labels * (1 - eps) + 0.5 * eps`. Why is the smoothing target `0.5` (not `0.0`)? Compute the smoothed label for `y=0.0` with `eps=0.14` (Reentrancy's noise rate).

**Q3.** Gradient accumulation with `accum_steps=8` and a loader of 10 batches: compute `_actual_window` for batch indices 0, 7, 8, and 9. Explain why batches 8 and 9 need a different value than batches 0–7.

**Q4.** `evaluate()` uses both `model.eval()` and `torch.no_grad()`. Explain what each one does — are they redundant? Give one concrete example of what breaks if you use only `torch.no_grad()` without `model.eval()`.

**Q5.** `WeightedRandomSampler` mode `"positive"` gives 3× weight to any contract with at least one vulnerability. With `num_samples=len(weights)` and `replacement=True`, roughly what fraction of sampled batches will contain vulnerable contracts if the original dataset is 40% vulnerable? Show the reasoning.
