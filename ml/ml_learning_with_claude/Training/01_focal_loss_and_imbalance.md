# Training — Chunk 1: Focal Loss & The Imbalance Problem
**File:** `ml/src/training/focalloss.py` (143 lines)
**Classes:** `FocalLoss`, `MultiLabelFocalLoss`

---

## Warm-Up Recall (from Module 4 — SentinelModel)

Answer these from memory before reading on. One sentence each.

1. What is the "three-eye classifier" in `SentinelModel`, and why use three eyes instead of one?
2. What does `select_prefix_nodes` do, and what is the priority order it uses to pick nodes?
3. What is the "prefix warmup guard" and why does it exist?

---

## P5 — Big Picture: Why These Files Exist

The SENTINEL model outputs `[B, 10]` logits — one logit per vulnerability class per contract. Training requires a loss function that converts those logits into a scalar the optimizer can minimize.

The naive choice is **BCE (Binary Cross-Entropy)** applied independently to each of the 10 classes. But SENTINEL's dataset makes BCE a poor fit, and understanding why is the foundation of everything in these two files.

### The Training Data Reality

SENTINEL trains on ~44,000 Solidity contracts. Each contract is labelled with a subset of 10 vulnerability classes. Key numbers:

| Metric | Value |
|--------|-------|
| Training samples | ~44,000 contracts |
| Classes | 10 vulnerability types |
| Total (sample, class) cells | ~440,000 |
| Fraction that are **negative** (label=0) | **>85%** |
| Rarest class (e.g., DoS) | ~377 positive contracts |

This means for every one contract labelled `DoS=1`, there are roughly **116** labelled `DoS=0`. The model can achieve 99% accuracy on DoS by predicting zero for every single input — and BCE would reward it for doing so.

> ⚠️ **CRITICAL** — This is the core problem of **class imbalance in multi-label classification**. The optimizer's gradient budget gets dominated by easy negatives (contracts that obviously don't have DoS), leaving almost no gradient signal for the hard cases (contracts that do). The result: the model learns to predict "no vulnerability" for everything and reaches low loss with zero utility.

### The File Roles

```
focalloss.py ──── FocalLoss            ← binary/multi-label, post-sigmoid input
             └─── MultiLabelFocalLoss  ← multi-label, raw logit input, per-class alpha

losses.py ──────── AsymmetricLoss      ← production loss for SENTINEL (next chunk)
```

`focalloss.py` provides two variants of **Focal Loss** (Lin et al., 2017). Neither is the primary production loss — that's `AsymmetricLoss` in `losses.py`. But Focal Loss is the conceptual foundation that ASL (Asymmetric Loss) extends, so it must be understood first.

**Cross-file relationship (not yet taught):** `trainer.py` imports these classes and wraps `FocalLoss` in a `_FocalFromLogits` adapter (since `FocalLoss` expects post-sigmoid probabilities, not logits). You'll see this in Module 5.

---

## Section 1 — Standard BCE and Why It Fails

**BCE (Binary Cross-Entropy)** for a single (sample, class) cell:

```
loss = -[ y * log(p) + (1-y) * log(1-p) ]
```

where:
- `y` ∈ {0, 1} is the ground truth label
- `p` = sigmoid(logit) is the model's predicted probability

> **Learning mode: Understand the pattern** — you don't need to memorize this formula verbatim, but you must understand what each term represents.

The two terms are mutually exclusive — when `y=1`, the second term vanishes; when `y=0`, the first vanishes. So effectively:

- **When label=1 (positive):** loss = `-log(p)` — high loss if model predicted low probability
- **When label=0 (negative):** loss = `-log(1-p)` — high loss if model predicted high probability

The fatal flaw: **BCE treats all cells equally**. An easy negative (a contract that obviously has no reentrancy, model predicts p=0.001) contributes nearly zero loss per cell — but there are 116× more negatives than positives. The sum of those near-zero terms dominates the gradient signal from the rare positives.

### The Gradient Perspective

If p=0.001 and y=0:
- BCE loss ≈ -log(0.999) ≈ 0.001 per cell
- Across 116 such negatives: cumulative negative loss ≈ 0.116

If p=0.3 and y=1:
- BCE loss ≈ -log(0.3) ≈ 1.2 per positive cell

The one positive contributes 1.2; the 116 easy negatives contribute 0.116 total. Sounds okay? But the gradient *per parameter* is shaped by the total, and with 10 classes × 44K samples, the easy negatives win. The optimizer is trained to push everything toward zero.

---

## Section 2 — Focal Loss: The Core Idea (Lin et al., 2017)

**Focal Loss** was introduced in the RetinaNet paper for object detection — a problem with the same positive/negative imbalance (most image regions contain no object).

The fix: **multiply BCE by a modulating factor that suppresses easy examples**.

```
FL(p, y) = -(1 - pt)^γ * log(pt)
```

Where:
- `pt` = model's confidence on the **correct** class
  - if y=1: `pt = p` (model should predict high p — easy if p is already high)
  - if y=0: `pt = 1-p` (model should predict low p — easy if 1-p is already high)
- `γ` (gamma) = focusing exponent, typically 2.0. Controls how aggressively easy examples are suppressed.
- `(1 - pt)^γ` is the **focal weight** — approaches 0 for well-classified examples, stays near 1 for hard ones

> ⚠️ **CRITICAL** — The `pt` construction is the heart of focal loss. `pt` always represents the model's confidence on the *right* answer, regardless of whether the right answer is 0 or 1. When `pt` is high (easy example), `(1-pt)^γ` is near zero — the loss is crushed. When `pt` is low (hard example), `(1-pt)^γ` ≈ 1 — the loss is preserved.

**Concrete example with γ=2:**

| Example type | p | pt | (1-pt)^2 | BCE loss | FL loss |
|---|---|---|---|---|---|
| Easy negative (y=0) | 0.05 | 0.95 | 0.0025 | 0.051 | 0.00013 |
| Hard negative (y=0) | 0.5 | 0.5 | 0.25 | 0.693 | 0.173 |
| Easy positive (y=1) | 0.95 | 0.95 | 0.0025 | 0.051 | 0.00013 |
| Hard positive (y=1) | 0.3 | 0.3 | 0.49 | 1.204 | 0.590 |

Easy negatives get their loss crushed by 400×. Hard examples are barely reduced. This is exactly what we want.

### Alpha: Class Frequency Balancing

Focal loss also includes an `alpha` parameter for explicit class balancing:

```
FL(p, y) = -alpha_t * (1 - pt)^γ * log(pt)
```

Where `alpha_t = alpha` for positives, `1-alpha` for negatives. Typically `alpha=0.25` — positives get 0.25 weight, negatives get 0.75 weight. This supplements the focal mechanism with a fixed frequency-based correction.

---

## Section 3 — `FocalLoss` Class (lines 8–74)

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
```

> **Learning mode: Understand the pattern** — simple init, no buffers. `gamma` and `alpha` are plain floats stored as instance attributes. This is fine because they are scalars shared across all devices; there's no device placement issue.

**The forward pass:**

```python
def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # ── Audit fix #6: BF16 underflow guard ────────────────────────────
    predictions = predictions.float()
    targets = targets.float()
    # ──────────────────────────────────────────────────────────────────

    bce = F.binary_cross_entropy(predictions, targets, reduction="none")

    pt = torch.where(targets == 1, predictions, 1 - predictions)

    alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

    focal_loss = alpha_t * (1 - pt) ** self.gamma * bce

    return focal_loss.mean()
```

> **Learning mode: Master the detail** — each line has a specific purpose that will appear in challenges.

**Step by step:**

**Step 1 — BF16 guard (Audit fix #6):**
```python
predictions = predictions.float()
targets = targets.float()
```
`BF16` (Brain Float 16) is a 16-bit floating point format used by the training loop under `torch.amp.autocast`. It has only ~3 decimal digits of precision. A probability of 0.001 silently becomes `0.0` in BF16. Then `log(0.0) = -inf`, `loss = nan`. Casting to `float32` here prevents this regardless of autocast context.

> ⚠️ **CRITICAL** — This is a **silent failure mode**. Without this guard, training loss becomes NaN after a few batches and the training run is destroyed. The fix is a single `.float()` call, but forgetting it is catastrophically expensive.

**Step 2 — BCE base:**
```python
bce = F.binary_cross_entropy(predictions, targets, reduction="none")
```
Note: `F.binary_cross_entropy` expects **post-sigmoid probabilities**, not raw logits. This is why `FocalLoss`'s docstring says "expects POST-SIGMOID probabilities" and why `trainer.py` has a `_FocalFromLogits` wrapper. `reduction="none"` gives `[B, C]` tensor — we need per-element values to multiply by focal weights before reducing.

**Step 3 — pt (confidence on correct class):**
```python
pt = torch.where(targets == 1, predictions, 1 - predictions)
```
`torch.where(condition, x, y)` returns `x` where `condition` is True, `y` elsewhere. So:
- Where `target=1` → `pt = predictions` (model's predicted probability for positive class)
- Where `target=0` → `pt = 1 - predictions` (model's predicted probability for negative class = "not positive")

> **Learning mode: Master the detail** — `pt` is always the probability on the *right* answer. High `pt` → easy example → focal weight crushes loss. Low `pt` → hard example → loss preserved.

**Step 4 — alpha_t:**
```python
alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
```
Same pattern: positives get `alpha=0.25`, negatives get `1-alpha=0.75`. This is the standard focal loss alpha formulation.

**Step 5 — focal loss:**
```python
focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
return focal_loss.mean()
```
Multiply the base BCE loss by the focal weight and alpha, then average over all elements.

**Important:** `FocalLoss` is for binary or multi-label with scalar alpha — every class shares the same gamma and alpha. If class 3 (reentrancy, rare) and class 1 (DoS, also rare) have very different prevalence, scalar alpha can't capture both. That's what `MultiLabelFocalLoss` solves.

---

## Section 4 — `MultiLabelFocalLoss` Class (lines 77–143)

```python
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha: List[float], gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
```

> **Learning mode: Master the detail** — two key differences from `FocalLoss`:
> 1. `alpha` is a `List[float]` of length C (one per class) → stored as a buffer
> 2. Accepts **raw logits**, not post-sigmoid probabilities

**Why `register_buffer` for alpha?**

```python
self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
```

`register_buffer` was covered in Module 4 (JK attention chunk). Recall: a buffer is a tensor that:
- Lives on the module's device (moves with `.to(device)`)
- Is included in `state_dict()` (saved/loaded with checkpoints)
- Does NOT receive gradients (not a parameter)

Here `alpha` is a `[C]` tensor of class weights. If training on GPU, it must live on GPU for element-wise operations with `[B, C]` tensors. A plain Python list or CPU tensor would fail or silently move to CPU in matmuls.

**The forward pass:**

```python
def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits  = logits.float()
    targets = targets.float()

    p   = torch.sigmoid(logits)                          # [B, C]
    pt  = torch.where(targets == 1, p, 1 - p)           # [B, C]

    alpha = self.alpha.unsqueeze(0)                      # [1, C]
    alpha_t = torch.where(targets == 1, alpha, 1.0 - alpha)  # [B, C]

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
    return focal_loss.mean()
```

> **Learning mode: Master the detail** — the `alpha` broadcasting and `binary_cross_entropy_with_logits` are both interview-relevant.

**`alpha.unsqueeze(0)`:**
`self.alpha` is shape `[C]`. Tensors of shape `[C]` and `[B, C]` don't broadcast in PyTorch — the dimensions don't align from the right. Adding a leading dimension: `[C]` → `[1, C]` → broadcasts correctly to `[B, C]`.

**`F.binary_cross_entropy_with_logits` vs `F.binary_cross_entropy`:**

`binary_cross_entropy` requires probabilities (after sigmoid). `binary_cross_entropy_with_logits` takes raw logits and computes `sigmoid + BCE` in a single numerically stable operation using the **log-sum-exp trick**:

```
BCE(logit, y) = max(logit, 0) - logit*y + log(1 + exp(-|logit|))
```

This avoids computing `sigmoid(logit)` explicitly when `logit` is large positive (which would give `p ≈ 1.0`, `log(1-p) = -inf`). Using logits directly is more numerically stable.

---

## AUDIT

> **[AUDIT] A1 — The `MultiLabelFocalLoss` alpha_t bug (and its fix)**

The docstring says:
> *"alpha_t was previously `self.alpha.unsqueeze(0)` — the same weight applied to both positive AND negative examples per class."*

What this means: the old code was:
```python
# WRONG — old code
alpha_t = self.alpha.unsqueeze(0)  # [1, C], same for positives and negatives
```

The current (correct) code is:
```python
# CORRECT — current code
alpha_t = torch.where(targets == 1, alpha, 1.0 - alpha)
```

**Why the old code was wrong:** For a rare class with `alpha=0.9` (high weight for positives):
- Old code: negatives also got `alpha=0.9` weight → the loss on negatives (which dominate) was inflated 9× relative to default
- This made the model over-penalise **correct** negatives (cases where the model correctly predicts "no vulnerability") and under-penalise **missed** positives
- The fix inverts alpha for negatives (`1 - alpha = 0.1`), matching the original focal loss design where a high alpha for positives means low alpha for negatives

> ⚠️ **CRITICAL** — This is an easy bug to introduce and a hard one to diagnose. The loss value wouldn't crash or go NaN — the model would just silently train in the wrong direction for rare classes. Alpha bugs in focal loss implementations are common and often go unnoticed.

> **[AUDIT] A2 — `FocalLoss` uses plain float attributes, `MultiLabelFocalLoss` uses a buffer**

`FocalLoss.gamma` and `FocalLoss.alpha` are plain Python floats — not tensors, not buffers. This is fine: scalar floats compute correctly with GPU tensors (PyTorch converts them). But it means if you check `model.state_dict()`, you won't find `gamma` or `alpha` there. For reproducibility, if you wanted to save/restore the exact focal loss hyperparameters with a checkpoint, you'd need to track them separately. `MultiLabelFocalLoss` is better here — its `alpha` buffer is included in `state_dict()` automatically.

> **[AUDIT] A3 — `FocalLoss` has no numerically stable BCE path for logits**

`FocalLoss` uses `F.binary_cross_entropy` (requires post-sigmoid input). This forces the `_FocalFromLogits` wrapper in `trainer.py` to call `sigmoid` before passing to `FocalLoss`. Two sigmoid calls effectively happen: once in the wrapper, once inside `pt = torch.where(targets==1, predictions, 1-predictions)`. There's no issue with extra computation at inference scale, but `MultiLabelFocalLoss`'s design (accepting logits directly and using `binary_cross_entropy_with_logits`) is the better pattern — one class, one responsibility, no wrapper needed.

---

## Section 5 — Alternative Approaches (P7)

### Option 1: Class-weighted BCE

```python
weight = torch.tensor([n_total / (n_classes * n_pos_c) for c in classes])
F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
```

Simple inverse-frequency weighting. Doesn't require any paper-specific hyperparameters. Limitation: applies a fixed multiplier to all positive examples equally — doesn't distinguish between easy and hard positives.

### Option 2: Label Smoothing

Replace hard 0/1 labels with soft targets (0.05, 0.95). Reduces overconfidence but doesn't address the imbalance problem at all.

### Option 3: Focal Loss (this chunk)

Dynamically down-weights easy examples via `(1-pt)^gamma`. Self-adaptive: the model's own confidence determines how much each cell is suppressed.

### Why Focal Loss wins for SENTINEL

The key insight: the number of "easy negatives" (contracts obviously clean of a particular vulnerability) is large AND variable. A fixed weight can't match the dynamic suppression that `(1-pt)^gamma` provides. The model's confidence IS the information about whether an example is easy or hard — using it is strictly better than ignoring it.

---

## Data Flow

```
Input: logits [B, C], targets [B, C]  ← raw model output and ground truth labels
          │
          ▼
      .float() cast  ← BF16 guard (Audit fix #6)
          │
          ▼ (MultiLabelFocalLoss only)
      sigmoid(logits) → p [B, C]
          │
    ┌─────┴──────┐
    │            │
    pt [B,C]   alpha_t [B,C]
    (confidence  (class weight,
    on correct   conditioned on
    class)       target value)
    │            │
    └─────┬──────┘
          │
    focal_weight = (1-pt)^gamma [B,C]
          │
    loss = alpha_t * focal_weight * bce_loss [B,C]
          │
          ▼
       .mean() → scalar loss
```

---

## 3 Things to Lock In (P10-C)

1. **`pt` is the confidence on the correct answer** — `p` when `y=1`, `1-p` when `y=0`. This single construct is all that's needed to make focal loss work: high confidence → easy → `(1-pt)^γ ≈ 0` → loss suppressed.

2. **`alpha_t` must be conditioned on the target** — `alpha` for positives, `1-alpha` for negatives. Applying the same alpha to both (the bug) inverts class balancing for negatives.

3. **BF16 underflow kills log-based losses silently** — the `.float()` cast at the top of every forward is not defensive paranoia — it's mandatory for AMP correctness. Without it, `p ≈ 0` in BF16 → `log(0) = -inf` → `nan` loss → destroyed training run.

---

## Challenge Questions

Answer from memory. Do not re-read the material.

**Q1.** BCE treats all (sample, class) cells equally. In SENTINEL's training set, there are ~44,000 contracts and 10 classes. Roughly what fraction of the 440,000 cells are negative (label=0)? What does that mean for where BCE spends its gradient budget?

**Q2.** What is `pt` in Focal Loss? Write the formula for computing it from `predictions` and `targets` using `torch.where` — what value does it take when `target=1` vs `target=0`?

**Q3.** In `MultiLabelFocalLoss`, `self.alpha` has shape `[C]`. In the forward pass, it becomes `self.alpha.unsqueeze(0)`. What shape does this produce? Why is this reshape necessary before the `torch.where` call with `targets` of shape `[B, C]`?

**Q4.** The Audit fix in `MultiLabelFocalLoss` corrected how `alpha_t` was computed. Describe what the old (buggy) code did, and explain in one sentence why it inverted class balancing for rare classes.

**Q5.** `FocalLoss` uses `F.binary_cross_entropy` (requires probabilities), while `MultiLabelFocalLoss` uses `F.binary_cross_entropy_with_logits` (takes raw logits). What numerical stability advantage does the latter have? Why does the former require a wrapper in `trainer.py`?
