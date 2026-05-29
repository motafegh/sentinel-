# Training — Chunk 2: Asymmetric Loss (ASL)
**File:** `ml/src/training/losses.py` (126 lines)
**Class:** `AsymmetricLoss`

---

## Warm-Up Recall (from Chunk 1 — Focal Loss)

Answer from memory. One sentence each.

1. What is `pt` in Focal Loss, and how is it computed from `predictions` and `targets`?
2. Why does `MultiLabelFocalLoss` use `register_buffer` for `alpha`, but `FocalLoss` stores `gamma` and `alpha` as plain Python floats?
3. What is the BF16 underflow failure mode, and what single line prevents it?

---

## Spaced Review (from Module 4 — Models)

One question from further back to fight the forgetting curve:

4. In `CrossAttentionFusion`, what does `_scatter_to_dense` do and why does it cause a `torch.compile` graph break?

---

## P5 — Big Picture: Why Focal Loss Isn't Enough

Focal Loss (Chunk 1) solves one half of the multi-label imbalance problem: it down-weights **easy examples** by multiplying BCE with `(1-pt)^γ`. Easy negatives (high confidence, correct) get suppressed.

But there's a deeper issue in **multi-label** classification that scalar focal loss doesn't fix:

### The Asymmetry Problem

In multi-label classification, negatives and positives are fundamentally different in nature:

- **Negatives (y=0):** Numerous, often easy (85%+ of cells). The challenge is to suppress them aggressively without losing signal from *hard* negatives.
- **Positives (y=1):** Rare, always hard. Even after focal weighting, they need a different, gentler treatment — over-suppressing positives defeats the purpose.

Focal Loss uses a **single gamma** for both. Setting `gamma=4` (aggressively suppress easy examples) works well for negatives — but it also crushes gradients from hard positives (high-loss, low-confidence positive cells where `pt` is low but `(1-pt)^4` still reduces the gradient more than needed).

**The insight of ASL (Asymmetric Loss, Ridnik et al. ICCV 2021):** give positives and negatives **independent gamma values**:

```
gamma_neg = 4  → aggressively down-weight easy negatives
gamma_pos = 1  → mildly focus positives (preserve gradient for rare, hard positives)
```

Plus: hard-threshold very confident negative predictions via a **probability clip** (probability margin), completely removing their contribution to the gradient.

### File Role in the System

`losses.py` provides `AsymmetricLoss` — the **primary production loss** used in `trainer.py` for SENTINEL's vulnerability classification task. `focalloss.py`'s classes are secondary utilities (used for auxiliary head training and experiments).

```
trainer.py
   └── AsymmetricLoss (primary)    ← losses.py
   └── _FocalFromLogits (auxiliary) ← focalloss.py
```

---

## Section 1 — The Clip Mechanism: Killing Easy Negatives Entirely

Focal Loss suppresses easy negatives by multiplying their loss by `(1-pt)^γ`. Even with `γ=4`, an easy negative with `pt=0.99` still contributes `(0.01)^4 = 1e-8` — tiny, but present, and summed over hundreds of thousands of cells.

ASL goes further: **shift the negative probability down before computing the focal weight**.

```python
prob_neg = (prob - self.clip).clamp(min=0.0)    # [B, C]
```

Where `clip = 0.05` by default. This means:

| prob (original) | prob_neg (after clip) | Effect |
|---|---|---|
| 0.03 (very confident negative) | 0.0 → clamped | **Zero gradient contribution** |
| 0.05 (confident negative) | 0.0 → clamped | **Zero gradient contribution** |
| 0.20 (uncertain negative) | 0.15 | Reduced, still present |
| 0.50 (hard negative) | 0.45 | Mostly preserved |

Any negative prediction below `clip=0.05` is **completely removed from the gradient**. The model is already saying "this contract is definitely not vulnerable to reentrancy" — there's nothing to learn there.

> ⚠️ **CRITICAL** — The clip doesn't modify `prob` (used for positive terms) — only `prob_neg` (used for negative terms). This is asymmetric by design: positive examples always use the original probability.

The clipped probability then feeds the **negative log term** and the **negative focal weight**:

```python
log_neg  = torch.log((1.0 - prob_neg).clamp(min=1e-8))    # [B, C]
focal_neg = prob_neg ** self.gamma_neg                     # [B, C]
```

When `prob_neg = 0`: `log_neg = log(1.0) = 0` and `focal_neg = 0^4 = 0`. Both terms zero out. Zero contribution from that cell, by construction.

---

## Section 2 — Full `AsymmetricLoss` Code Walkthrough (lines 49–126)

### `__init__` (lines 63–79)

```python
def __init__(
    self,
    gamma_neg:  Union[float, torch.Tensor] = 4.0,
    gamma_pos:  Union[float, torch.Tensor] = 1.0,
    clip:       Union[float, torch.Tensor] = 0.05,
    pos_weight: "torch.Tensor | None"      = None,
    reduction:  str                        = "mean",
) -> None:
    super().__init__()
    self.reduction = reduction

    self.register_buffer("gamma_neg", torch.as_tensor(gamma_neg, dtype=torch.float32))
    self.register_buffer("gamma_pos", torch.as_tensor(gamma_pos, dtype=torch.float32))
    self.register_buffer("clip",      torch.as_tensor(clip,      dtype=torch.float32))
    self.register_buffer("pos_weight", ...)
```

> **Learning mode: Master the detail** — `torch.as_tensor` + `register_buffer` pattern. This is BUG-M3's implementation.

**`Union[float, torch.Tensor]`:** Each hyperparameter can be either:
- A scalar float → broadcasts to all C classes
- A `[C]` tensor → per-class values

**`torch.as_tensor(gamma_neg, dtype=torch.float32)`:**
- If `gamma_neg` is a `float`, this creates a 0-dimensional (scalar) tensor
- If `gamma_neg` is already a `[C]` tensor, it passes through (or casts to float32 if needed)
- Either way, the result is a tensor that can be `register_buffer`'d

> ⚠️ **CRITICAL** — Why `register_buffer` and not a plain attribute? Because if you train on GPU, `gamma_neg` (a `[C]` tensor) must live on GPU to multiply with `[B, C]` tensors. If stored as a plain attribute, `.to(device)` would not move it — you'd get a device mismatch error at first use. A scalar float would convert fine (PyTorch auto-converts scalars), but a tensor would not.

### `forward` (lines 81–126)

> **Learning mode: Master the detail** — this is the production loss. Every line matters.

```python
def forward(
    self,
    logits: torch.Tensor,   # [B, C] raw logits (NO sigmoid)
    labels: torch.Tensor,   # [B, C] float {0.0, 1.0} or soft targets
) -> torch.Tensor:
    logits = logits.float()
    labels = labels.float()

    prob = torch.sigmoid(logits)                              # [B, C]

    prob_neg = (prob - self.clip).clamp(min=0.0)             # [B, C]

    log_pos  = torch.log(prob.clamp(min=1e-8))               # [B, C]
    log_neg  = torch.log((1.0 - prob_neg).clamp(min=1e-8))   # [B, C]

    focal_pos = (1.0 - prob) ** self.gamma_pos               # [B, C]
    focal_neg = prob_neg     ** self.gamma_neg                # [B, C]

    loss_pos = -labels         * focal_pos * log_pos         # [B, C]
    if self.pos_weight is not None:
        loss_pos = loss_pos * self.pos_weight  # [C] broadcasts over [B, C]
    loss_neg = -(1.0 - labels) * focal_neg * log_neg         # [B, C]
    loss     = loss_pos + loss_neg                           # [B, C]

    if self.reduction == "mean":
        return loss.mean()
    if self.reduction == "sum":
        return loss.sum()
    return loss
```

**Step by step:**

**Step 1 — BF16 guard:**
```python
logits = logits.float()
labels = labels.float()
```
Same pattern as `focalloss.py`. Mandatory for AMP safety. (Audit fix #6 applies here too, pre-emptively.)

**Step 2 — Sigmoid → probabilities:**
```python
prob = torch.sigmoid(logits)    # [B, C] values in (0, 1)
```
ASL operates on probabilities, not logits. Unlike `MultiLabelFocalLoss`, which used `binary_cross_entropy_with_logits` for numerical stability, ASL computes its own log terms manually — which is why the `.clamp(min=1e-8)` guards are needed separately.

**Step 3 — Clip the negative probability:**
```python
prob_neg = (prob - self.clip).clamp(min=0.0)    # [B, C]
```
`self.clip` is shape `()` (scalar) or `[C]`. Broadcasting: `[B, C] - () = [B, C]` or `[B, C] - [C] = [B, C]`. The `.clamp(min=0.0)` floors at zero so we never get negative probabilities.

**Step 4 — Log terms:**
```python
log_pos = torch.log(prob.clamp(min=1e-8))              # used for positive loss term
log_neg = torch.log((1.0 - prob_neg).clamp(min=1e-8))  # used for negative loss term
```
- `log_pos`: standard `-log(p)` for positive BCE. Uses original `prob`, not clipped.
- `log_neg`: `-log(1 - p_neg)`. Uses clipped `prob_neg`. When `prob_neg=0`, `log(1.0)=0`.

The `.clamp(min=1e-8)` prevents `log(0) = -inf`. This is the manual AMP safety that `binary_cross_entropy_with_logits` handles internally.

**Step 5 — Focal weights:**
```python
focal_pos = (1.0 - prob) ** self.gamma_pos    # [B, C]
focal_neg = prob_neg     ** self.gamma_neg    # [B, C]
```
Focal weights are **different** for positives and negatives:
- Positive focal weight: `(1-p)^γ_pos` — standard focal weight on confidence for positive class
- Negative focal weight: `p_neg^γ_neg` — **`p_neg`**, not `(1-p_neg)`. Why?

> ⚠️ **CRITICAL** — `focal_neg = prob_neg^γ_neg` not `(1-prob_neg)^γ_neg`. This is deliberate.
> 
> For negatives, the "easy" case is when `prob` is near 0 (model correctly predicts "not vulnerable"). After clipping: `prob_neg ≈ 0`. The focal weight `0^4 = 0` → loss crushed. Correct.
> 
> For negatives, the "hard" case is when `prob` is near 0.5 (model is uncertain). After clipping: `prob_neg ≈ 0.45`. The focal weight `0.45^4 ≈ 0.04` → loss somewhat preserved. Correct.
> 
> Compare with the positive focal weight `(1-prob)^γ_pos`: for positives, the easy case is `prob≈1`, so `(1-1)^1 = 0`. Same concept, different formula, because the "easy direction" is different.

**Step 6 — Combine:**
```python
loss_pos = -labels * focal_pos * log_pos           # [B, C]
loss_neg = -(1.0 - labels) * focal_neg * log_neg   # [B, C]
loss     = loss_pos + loss_neg                     # [B, C]
```
- `labels` acts as a selector: where `label=1`, only `loss_pos` contributes; where `label=0`, only `loss_neg` contributes.
- These terms are mutually exclusive for hard labels (0 or 1), but work correctly for soft targets too (e.g., `label=0.8` splits the contribution proportionally).

**`pos_weight` (optional):**
```python
if self.pos_weight is not None:
    loss_pos = loss_pos * self.pos_weight  # [C] broadcasts over [B, C]
```
An additional per-class scalar multiplier applied to positive losses. Stacks on top of the focal weight for even more precise control over rare classes.

---

## Section 3 — BUG-M3: Per-Class Tensor Hyperparameters

BUG-M3 (from the module docstring) refers to the per-class capability of `gamma_neg`, `gamma_pos`, and `clip`:

```python
# Example from docstring:
gamma_neg = torch.tensor([4.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
```

This allows applying **softer negative focus** to rare classes. For class 1 (DenialOfService — only 377 positive training examples), setting `gamma_neg=2` instead of 4 means easy negatives for DoS are suppressed less aggressively, preserving more gradient signal for that class's decision boundary.

**Why is this "BUG-M3"?** The docstring labels it because it was a feature added to fix a training bug: with scalar `gamma_neg=4`, the model was learning to predict zero for several rare classes (the easy negatives dominated even after focal suppression). Per-class gammas were the targeted fix.

**Broadcasting mechanics with `[C]` buffers:**

```python
# gamma_neg shape: [C]
# prob_neg shape:  [B, C]
focal_neg = prob_neg ** self.gamma_neg   # [B, C] ** [C] → [B, C]
```

PyTorch broadcasting rules: two tensors are compatible if, for each dimension (aligned from the right), the sizes are equal or one of them is 1. `[B, C] ** [C]`: the `[C]` aligns with the last dim of `[B, C]`. Each column c uses `gamma_neg[c]`. This is automatic — no explicit reshape needed.

---

## Section 4 — AMP Safety Pattern: The Full Picture

Both files follow the same AMP (Automatic Mixed Precision) safety pattern. Let's consolidate it:

```
torch.amp.autocast("cuda")
  │
  ├── Eligible ops → BF16 (faster, less memory)
  └── Ineligible ops → float32 (safer)

BF16 precision: ~3 decimal digits
   → p = 0.001 silently becomes 0.0 in BF16
   → log(0.0) = -inf
   → loss = nan
   → training destroyed, silently

Fix:
   logits = logits.float()   ← promotes to float32 BEFORE any computation
   labels = labels.float()
   
   Then: .clamp(min=1e-8) on all arguments to log()
         → prevents log(0) even if float32 rounding gets close to zero
```

> **Learning mode: Understand the pattern** — you'll see this in every numerically sensitive PyTorch module. It's not defensive — it's required for AMP correctness. The standard pattern is: cast inputs at the top of forward(), guard log() with clamp.

---

## Section 5 — The Full Evolution: BCE → Focal → ASL (P7)

| Loss | What it does | Limitation |
|------|-------------|-----------|
| **BCE** | Equal weight to all cells | Easy negatives dominate gradient |
| **Weighted BCE** | Fixed multiplier per class | Doesn't distinguish easy vs hard within a class |
| **Focal Loss** | `(1-pt)^γ` suppresses easy examples | Single gamma — can't treat positives and negatives differently |
| **ASL** | `gamma_neg ≠ gamma_pos` + clip | Full asymmetric control: aggressive negative suppression, gentle positive handling |

**When to use each:**

- **BCE**: Balanced datasets, regression-style outputs, or as a baseline
- **Weighted BCE**: Quick fix for mild imbalance, no tuning overhead
- **Focal Loss**: Object detection-style binary classification, moderate imbalance
- **ASL**: Multi-label classification with severe positive/negative asymmetry (SENTINEL's case)

**The interview answer:** "SENTINEL uses ASL because multi-label vulnerability detection has two compounding imbalances: (1) 85%+ negative cells across 10 classes, and (2) the rarest class has 100× fewer positives than negatives. A single gamma can't handle both simultaneously. ASL's independent gamma_neg and clip mechanism kill easy negatives completely, while gamma_pos preserves gradient signal for the rare, hard positives."

---

## Data Flow

```
Input: logits [B, C], labels [B, C]
           │
           ▼
       .float() cast  ← BF16 guard
           │
     sigmoid → prob [B, C]
           │
    ┌──────┴──────────────────────┐
    │ Positive branch             │ Negative branch
    │                             │
    │ prob                        │ prob_neg = (prob - clip).clamp(0)
    │                             │           ← clip=0.05 removes
    │                             │             very-confident negatives
    │ log_pos = log(prob+ε)       │ log_neg = log(1 - prob_neg + ε)
    │ focal_pos = (1-prob)^γ_pos  │ focal_neg = prob_neg^γ_neg
    │                             │
    │ loss_pos = -y*focal*log_pos │ loss_neg = -(1-y)*focal*log_neg
    │ (× pos_weight if set)       │
    └──────────────┬──────────────┘
                   │
             loss = loss_pos + loss_neg  [B, C]
                   │
            .mean() / .sum() / none → scalar
```

---

## 3 Things to Lock In (P10-C)

1. **ASL uses two gammas, not one** — `gamma_neg` (typically 4) aggressively suppresses easy negatives; `gamma_pos` (typically 1) gently focuses positives. This asymmetry is the entire point of the paper: positives and negatives have fundamentally different roles in multi-label learning.

2. **The clip mechanism zeros out confident negatives entirely** — any negative prediction below `clip=0.05` contributes literally zero gradient. This is harder than focal loss's suppression and handles the long tail of easy negatives that focal alone misses.

3. **`torch.as_tensor` + `register_buffer` = device-portable hyperparameter tensor** — when a hyperparameter can be per-class (a `[C]` tensor), it must be registered as a buffer so `.to(device)` moves it. `torch.as_tensor` converts a float to a 0-dim tensor first so the same `register_buffer` call works for both scalar and tensor inputs.

---

## Challenge Questions

**Q1.** ASL uses `focal_neg = prob_neg ** self.gamma_neg` while Focal Loss uses `(1-pt) ** gamma`. Both suppress easy negatives. For a negative example with `prob=0.04` (clip=0.05):
- What is `prob_neg`?
- What is `focal_neg` with `gamma_neg=4`?
- What is the loss contribution from this cell?

**Q2.** The clip mechanism shifts `prob_neg = (prob - clip).clamp(min=0)` before computing `log_neg = log(1 - prob_neg + ε)`. When `prob_neg = 0`, what does `log_neg` equal? Why does this mean the cell contributes zero gradient?

**Q3.** `gamma_neg` and `gamma_pos` are registered as buffers via `torch.as_tensor(gamma_neg, dtype=torch.float32)`. What two shapes can they be, and how does broadcasting make the `[C]` case work in `prob_neg ** self.gamma_neg`?

**Q4.** Write the complete formula for `loss_pos` and `loss_neg` using the variable names in the code. Which terms act as "selectors" that zero out the wrong branch for hard binary labels?

**Q5.** Compare ASL and `MultiLabelFocalLoss` on three dimensions: (a) numerical stability approach, (b) how per-class weighting is implemented, (c) whether the positive/negative focal treatment is symmetric or asymmetric.
