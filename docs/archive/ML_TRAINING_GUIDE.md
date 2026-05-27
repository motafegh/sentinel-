# SENTINEL ML — Training, Model & Configuration Guide

A technical learning reference grounded in this project's actual code, numbers, and decisions.
All values, shapes, and metrics are from real runs — not hypothetical examples.

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem)
2. [Model Architecture](#2-model-architecture)
3. [Training Concepts](#3-training-concepts)
4. [Configuration Reference](#4-configuration-reference)
5. [The v3 Story — What We Learned](#5-the-v3-story)
6. [Fine-Tuning vs Training From Scratch](#6-fine-tuning-vs-training-from-scratch)
7. [Practical Reference](#7-practical-reference)

---

## 1. The Problem

We have ~68,000 Solidity smart contracts. Each contract may have zero or more of 10 vulnerability types simultaneously (multi-label). We want a model that reads a contract and outputs a probability for each of the 10 classes.

This is harder than binary classification because:
- A contract can have 3 vulnerabilities at once
- Classes are highly imbalanced (IntegerUO: 5,343 positive contracts vs DenialOfService: 137)
- The model must be confident enough to threshold each class independently

**Output:** 10 sigmoid probabilities, one per class. Each is thresholded independently (not softmax — classes are not mutually exclusive).

---

## 2. Model Architecture

The model has three components that process the contract in parallel, then fuse their representations.

```
Contract Source Code
        │
        ├──────────────────┬──────────────────
        │                  │
   GNNEncoder         TransformerEncoder
   (graph path)        (text path)
        │                  │
   [N, 64]           [B, 512, 768]
        │                  │
        └──────────────────┘
                  │
         CrossAttentionFusion
                  │
              [B, 128]
                  │
            Classifier
                  │
             [B, 10] logits
```

### 2.1 GNNEncoder — the graph path

The contract is parsed into an **Abstract Syntax Tree (AST)**, then converted to a graph where:
- **Nodes** = functions, state variables, modifiers, events, constructors (one node per entity)
- **Edges** = relationships: CALLS, READS, WRITES, EMITS, INHERITS

Each node has an 8-dimensional feature vector (LOCKED — changing this means rebuilding all 68K graphs):

| Index | Feature | Values |
|-------|---------|--------|
| 0 | node type | STATE_VAR=0, FUNCTION=1, MODIFIER=2, EVENT=3, FALLBACK=4, RECEIVE=5, CONSTRUCTOR=6, CONTRACT=7 |
| 1 | visibility | public/external=0, internal=1, private=2 |
| 2 | is_pure | 0 or 1 |
| 3 | is_view | 0 or 1 |
| 4 | is_payable | 0 or 1 |
| 5 | is_reentrant | 0 or 1 |
| 6 | complexity | float (CFG node count) |
| 7 | loc | float (lines of code) |

**What the GNN learns:** structural patterns. A reentrancy vulnerability has a specific call-graph shape (external call before state update). The GNN learns to recognise these structural signatures.

**Architecture:** 3-layer GAT (Graph Attention Network):
- Layer 1: [N, 8] → [N, 64]
- Layer 2: [N, 64] → [N, 64]
- Layer 3: [N, 64] → [N, 64], heads=1

Returns: `(node_embeddings [N, 64], batch_assignment [N])` — NOT pooled yet. The CrossAttentionFusion does the pooling.

Edge types are embedded: `nn.Embedding(5, 16)` maps each edge type (0–4) to a 16-dim vector. This lets the model treat CALLS differently from READS.

### 2.2 TransformerEncoder — the text path

The raw Solidity source code is tokenised (up to 512 tokens, sliding window for longer contracts) and fed to **CodeBERT** — a 124M parameter transformer pretrained on code from GitHub.

**LoRA (Low-Rank Adaptation):** instead of fine-tuning all 124M parameters, we add small trainable matrices to the Query and Value projections in all 12 attention layers.

For each attention layer, instead of updating `W` (768×768), we add:
```
W_effective = W_frozen + B × A
```
Where:
- `A` shape: `(r, 768)` — projects down to rank r
- `B` shape: `(768, r)` — projects back up
- `r` = LoRA rank (8 in v3)

With r=8: each Q and V gets `(8×768) + (768×8) = 12,288` new parameters.
Total: 12 layers × 2 matrices (Q,V) × 12,288 = **294,912 trainable parameters**.
Frozen CodeBERT: **124,645,632 parameters**.

**Why LoRA?** Fine-tuning 124M params needs ~2GB just for weights, plus gradients and optimizer state (Adam stores 2 moment estimates per param = 6× the model size). That's 12GB+ for CodeBERT alone. LoRA reduces trainable params by 99.8%, making it fit in 8GB VRAM.

**Why no `torch.no_grad()` wrapper?** CodeBERT's frozen weights don't need gradients, but the LoRA adapters do. Wrapping in `no_grad()` would cut the gradient flow through the LoRA path. The frozen weights are frozen via `requires_grad=False`, which is different from `no_grad()`.

Returns: `last_hidden_state [B, 512, 768]` — one embedding per token.

### 2.3 CrossAttentionFusion

This is where the two paths meet. It runs cross-attention in both directions:

```
node_proj:   [N, 64]    → [N, 256]     (project GNN to common dim)
token_proj:  [B, 512, 768] → [B, 512, 256]  (project BERT to common dim)

Node → Token attention: each node attends to all tokens (what text corresponds to this node?)
Token → Node attention: each token attends to all nodes (what structure corresponds to this text?)

Masked mean pool → concat [B, 512] → Linear → ReLU → Dropout → [B, 128]
```

The output dimension (128) is **LOCKED** — the ZKML proxy model expects input_dim=128.

### 2.4 Classifier

```python
Linear(128, 10)  →  logits [B, 10]
```

**No sigmoid inside the model.** Raw logits are output. Sigmoid is applied:
- During training: inside `BCEWithLogitsLoss` (numerically stable)
- During inference: `sigmoid(logits)` in `Predictor._score()`
- During threshold tuning: `sigmoid(logits)` before comparing to thresholds

This design avoids double-sigmoid which would compress probabilities into a tiny range.

---

## 3. Training Concepts

### 3.1 Loss Function — BCEWithLogitsLoss + pos_weight

For multi-label classification, each class is treated as an independent binary problem. The loss is:

```
BCE(logit, target) = -[target × log(σ(logit)) + (1-target) × log(1-σ(logit))]
```

Where `σ` is sigmoid. This is computed per class per sample, then averaged.

**The imbalance problem:** DenialOfService has 137 positive contracts in 47,966 training samples. The model learns quickly that "always predict negative" gives 99.7% accuracy on DoS. Without correction, it ignores the class entirely.

**pos_weight fix:** multiply the positive term by `N_negative / N_positive`:

```
For DoS: pos_weight = (47,966 - 137) / 137 ≈ 68
Loss_DoS = -[68 × target × log(σ(logit)) + (1-target) × log(1-σ(logit))]
```

Now missing a DoS positive costs 68× more than missing a DoS negative. The model is forced to pay attention to DoS. This is why the model learns to detect DoS at all despite having so few samples.

**The side effect:** high pos_weight pushes recall up and precision down. The model learns to predict "yes" aggressively to avoid the heavy penalty for false negatives. This is why v3 had 6/10 classes over-predicting (predicted positives 1.6–2.3× the actual count).

**v3 pos_weights (from actual training log):**

| Class | pos_weight | Effect |
|-------|-----------|--------|
| DoS | 68.02 | Very aggressive — 137 samples |
| CallToUnknown | 7.59 | Moderately aggressive |
| Timestamp | 8.40 | Moderately aggressive |
| ExternalBug | 5.22 | Moderate |
| UnusedReturn | 5.09 | Moderate |
| TOD | 4.85 | Moderate |
| MishandledException | 3.53 | Mild |
| Reentrancy | 3.12 | Mild |
| GasException | 2.95 | Mild |
| IntegerUO | 0.92 | Near-balanced |

### 3.2 Optimizer — AdamW

AdamW (Adam with decoupled weight decay) is the standard for transformer fine-tuning.

Adam maintains **two moment estimates** per parameter:
- `m` = exponential moving average of gradients (first moment / "momentum")
- `v` = exponential moving average of squared gradients (second moment / "adaptive LR")

The update rule: `param -= lr × m / (√v + ε)`

This means Adam is self-calibrating — parameters that get large gradients get smaller effective LR, and vice versa. It converges faster than SGD on transformers.

**Why AdamW instead of Adam?** Standard Adam applies weight decay incorrectly (multiplied by the adaptive factor). AdamW decouples it: `param -= lr × m/(√v+ε) + weight_decay × param`. This regularises weights independently of gradient scale.

**The stale moments problem:** Adam's `m` and `v` are accumulated over all past gradients. If you resume training with a different batch size, the scale of gradients changes (larger batch = smaller gradient variance). The saved `m` and `v` were calibrated to the old scale — they're now stale. This is why changing batch_size requires either model-only resume (fresh Adam) or `--resume-reset-optimizer`.

### 3.3 Learning Rate Schedule — OneCycleLR

OneCycleLR is a one-cycle policy with three phases:

```
LR
↑
│        ╱╲
│       ╱  ╲
│      ╱    ╲___________
│_____╱
└──────────────────────► Steps
  warmup  peak  anneal
```

- **Warmup** (5% of steps): LR rises from `lr/25` to `lr`. Prevents early instability.
- **Peak** (middle): LR at maximum. Largest parameter updates.
- **Anneal** (remaining): LR decays to `lr/10000`. Very small updates.

**Why this matters for v3:** v3 ran 60 epochs. OneCycleLR completed its full cycle. By epoch 54, LR was near zero — the model physically could not make meaningful updates. The loss was still decreasing (gradient direction was correct) but the step size was too small to measure. This is why val F1 was flat from epoch 54 onward while train loss was still falling.

**The fix:** fine-tuning from v3 weights with a fresh OneCycleLR at lr=1e-4. The model starts at F1=0.47 territory and gets a new cycle of useful step sizes.

**Why lr=1e-4 and not 3e-4?** v3 used 3e-4 starting from random weights — high LR needed to escape random initialisation. We start from a good solution, so we need a smaller LR to refine it without overshooting. Too high → we destroy v3's learned representations. Too low → no improvement.

### 3.4 Automatic Mixed Precision (AMP)

Training in float32 uses 4 bytes per value. BF16 uses 2 bytes. AMP runs most operations in BF16 but keeps a float32 master copy for weight updates.

**Why BF16 and not FP16?** BF16 has the same exponent range as float32 (8 bits) but fewer mantissa bits. FP16 has a much smaller range and requires loss scaling to prevent underflow. BF16 "just works" for most operations without loss scaling.

**The NaN problem:** BF16 has ~3 decimal digits of precision. Probabilities near 0.001 become exactly 0.0 in BF16. Then `log(0) = -inf`. Then `inf - inf = NaN`. This is why FocalLoss has an explicit `.float()` cast — it forces FP32 before the `log` operation inside BCE.

**GradScaler:** PyTorch's GradScaler maintains a scale factor for gradients. If gradients overflow to `inf` (common in FP16), it reduces the scale. SENTINEL uses AMP in BF16 mode where the scaler is less critical but still present.

### 3.5 Early Stopping

After each validation epoch, if `val_f1_macro` does not improve, `patience_counter` increments. When it hits `early_stop_patience` (7 in v4 experiments), training stops and the best checkpoint is retained.

**Fix #11 and Fix #23 (learned the hard way):** The patience counter must be saved to disk. Without this, if training is interrupted and resumed, the counter resets to 0 — the model could train indefinitely past its plateau.

**Fix #25 (found in this session):** model-only resume (fine-tuning) must reset patience_counter to 0. Otherwise the counter from v3 (which was 6 at epoch 54, close to the limit of 10) would carry over, potentially stopping fine-tuning after only 1 epoch with no improvement.

### 3.6 Threshold Tuning

The model outputs probabilities in [0,1] per class. By default, 0.5 is used as the threshold: `pred = prob > 0.5`. But this is rarely optimal, especially with imbalanced data.

**Threshold tuning:** sweep thresholds from 0.05 to 0.95 in steps of 0.05 (19 values) for each class independently. Pick the threshold that maximises F1 on the validation set.

**Why thresholds differ so much between classes:**

| Class | Threshold | Meaning |
|-------|-----------|---------|
| IntegerUO | 0.50 | Model is well-calibrated — default works |
| DenialOfService | 0.95 | Model is underconfident about DoS. Only when it's 95% sure is it actually right. Below 0.95, too many false positives. |
| GasException | 0.55 | Slightly above default — minor calibration issue |
| ExternalBug | 0.65 | Model over-predicts, needs higher threshold to filter false positives |

A high threshold (0.95 for DoS) compensates for low model confidence. But it also means we're missing any DoS contracts where the model is only 80% confident. This is why DoS tuned F1 (0.40) is much higher than raw F1 at threshold=0.5 (0.23).

---

## 4. Configuration Reference

All configuration lives in `TrainConfig` dataclass in `ml/src/training/trainer.py`.

### 4.1 Model Architecture Fields (LOCKED for v4 sprint)

| Field | Value | Why Locked |
|-------|-------|-----------|
| `num_classes` | 10 | Changing reshapes classifier head — incompatible with v3 checkpoint |
| `fusion_output_dim` | 128 | ZKML proxy model has input_dim=128 hardcoded |
| `gnn_hidden_dim` | 64 | Changing reshapes all GNN layers |
| `lora_target_modules` | ['query', 'value'] | Changing adds/removes LoRA keys from checkpoint |

### 4.2 Training Hyperparameters

| Field | v3 Value | v4 Exp1 Value | What It Controls |
|-------|---------|---------------|-----------------|
| `epochs` | 60 | 30 | Max training epochs before stop |
| `batch_size` | 32 | 16 | Samples per gradient step. Larger = more stable gradient, more VRAM |
| `lr` | 3e-4 | 1e-4 | Peak learning rate for OneCycleLR |
| `weight_decay` | 1e-2 | 1e-2 | L2 regularisation in AdamW (penalises large weights) |
| `warmup_pct` | 0.05 | 0.05 | Fraction of steps for LR warmup (5% = 3 epochs out of 60) |
| `grad_clip` | 1.0 | 1.0 | Max gradient norm. Prevents exploding gradients |
| `early_stop_patience` | 10 | 7 | Epochs without improvement before stopping |

### 4.3 LoRA Fields (searchable in v4)

| Field | v3 | Meaning |
|-------|-----|---------|
| `lora_r` | 8 | Rank of LoRA matrices. Higher = more capacity, more VRAM |
| `lora_alpha` | 16 | Scaling factor. Effective LR for LoRA = `lr × lora_alpha / lora_r`. With r=8, alpha=16: scale=2.0 |
| `lora_dropout` | 0.1 | Dropout inside LoRA adapters |

**lora_alpha / lora_r scaling:** the convention `lora_alpha = 2 × lora_r` keeps the effective LoRA contribution constant as you change r. If you try r=16, use alpha=32.

### 4.4 Resume Fields

| Field | Default | Meaning |
|-------|---------|---------|
| `resume_from` | None | Path to checkpoint. If set, load model weights from this file |
| `resume_model_only` | True | **True** = fine-tune (fresh optimizer, epoch counter resets to 1). **False** = full continuation (restore optimizer + epoch counter) |
| `force_optimizer_reset` | False | With full resume, discard optimizer state anyway (for batch_size change) |

**When to use what:**
- Continuing an interrupted run with identical config → `--no-resume-model-only`
- Fine-tuning from a checkpoint with different hyperparameters → default (model-only)
- Batch size changed, want to keep exact epoch counter → `--no-resume-model-only --resume-reset-optimizer`

### 4.5 Loss Function Fields

| Field | Options | Notes |
|-------|---------|-------|
| `loss_fn` | "bce", "focal" | bce = BCEWithLogitsLoss + pos_weight. focal = FocalLoss |
| `focal_gamma` | 2.0 | Focuses learning on hard examples. Higher = more focus |
| `focal_alpha` | 0.25 | Weight for positive class. **Caution:** 0.25 reduces positive weight vs BCE pos_weight. Needs tuning for multi-label |

---

## 5. The v3 Story — What We Learned

v3 was trained from scratch: 60 epochs, batch=32, lr=3e-4, bce loss, lora_r=8. It was the first successful multi-label run and reached tuned F1-macro=**0.5069**.

### 5.1 What the training curves actually showed

From MLflow run `d2ee23a` (sqlite:///mlruns.db, experiment `sentinel-retrain-v3`):

| Epoch | train_loss | val_f1_macro | Key observation |
|-------|-----------|-------------|----------------|
| 1 | 1.117 | 0.260 | Random-ish start |
| 10 | 0.977 | 0.409 | Fast early learning |
| 30 | 0.816 | 0.441 | Slowing down |
| 54 | 0.690 | **0.4715** | Best epoch saved |
| 60 | 0.687 | 0.470 | Still improving, but barely |

**The train loss was still decreasing at epoch 60.** The model had not converged. Val F1 was flat because **OneCycleLR had decayed to near-zero step size** — not because the model hit a capacity ceiling.

### 5.2 The DenialOfService problem

DoS val F1 at every epoch fluctuated between 0.11 and 0.27 with no stable trend after epoch 17. This is not model instability — it's measurement noise:
- 137 val samples for DoS
- Changing 5 true positives (getting them right vs wrong) changes F1 by ±0.04
- At 4 batches/epoch exposure, the model never builds a stable DoS signal

No hyperparameter change fundamentally fixes this. The ceiling is ~0.40 tuned F1 with 137 samples.

### 5.3 The over-prediction pattern

6 of 10 classes had high recall but low precision (model predicted 1.6–2.3× more positives than actually existed). **Root cause: pos_weight BCE.**

High pos_weight pushes the model to predict positive aggressively. Threshold tuning compensates partially (raising thresholds above 0.5) but doesn't fix the calibration. The model is poorly calibrated — its probability scores don't reflect true confidence.

### 5.4 What the tuned thresholds reveal

The fact that DoS needs threshold=0.95 tells you the model only correctly identifies DoS when it's 95%+ confident. At 80% confidence, it produces too many false positives. This means the model's internal DoS representation is noisy — it sometimes fires for similar-looking patterns that aren't DoS.

The fact that IntegerUO works perfectly at threshold=0.50 tells you the model is well-calibrated for IntegerUO — when it says "70% probability", it's actually right 70% of the time.

### 5.5 The corrected diagnosis

Previous documentation said: *"The model reached its capacity ceiling."*

Actual finding: **The LR schedule ran out.** The model was still learning — just had no step size to act on the gradients. The right fix is a fresh LR cycle from v3 weights, not a bigger model.

---

## 6. Fine-Tuning vs Training From Scratch

### Training from scratch
- Starts with random weights
- Needs high LR to explore the loss landscape
- Needs many epochs to learn basic patterns
- Takes 60+ epochs to converge on this dataset
- Every epoch sees the full learning curve

### Fine-tuning from a checkpoint
- Starts with learned weights (already at F1=0.47 territory)
- Needs lower LR to refine without destroying good representations
- Fresh optimizer (no stale Adam moments)
- Takes fewer epochs (refinement, not discovery)
- Loss starts low (~0.7 vs ~1.1 from scratch)

**When fine-tuning is appropriate:**
- You have a good checkpoint and want to push it further
- You're changing one hyperparameter and want to isolate its effect
- You're adapting to new data or a related task

**When to train from scratch:**
- Architecture changes (lora_r, layer sizes) that change the weight shapes
- You suspect the checkpoint is in a bad local minimum
- The hyperparameter change is large enough that the old weights are no longer a good starting point

### The start_epoch bug (Fix #25)

When loading a checkpoint for fine-tuning, the trainer reads `epoch=54` from v3. Without Fix #25:
```
start_epoch = 54 + 1 = 55
remaining_epochs = config.epochs - start_epoch + 1 = 30 - 55 + 1 = -24
→ return immediately, no training
```

Fix: when `resume_model_only=True`, reset `start_epoch=1`. The checkpoint's epoch counter belongs to its training run, not yours.

### What fine-tuning looks like vs from-scratch

| | From scratch (v3 ep1) | Fine-tune from v3 (v4 ep1) |
|--|--|--|
| Initial loss | ~1.12 | ~0.70 |
| Epoch 1 F1 | ~0.26 | ~0.47+ (starting point) |
| LR behaviour | Fresh high LR, aggressive | Fresh LR at 1/3 of v3's peak |
| Adam moments | Zero | Zero (fresh) |
| What's learning | Everything from scratch | Small refinements to near-converged model |

---

## 7. Practical Reference

### 7.1 Running training

```bash
# Always set TRANSFORMERS_OFFLINE=1 at shell level (not inside Python)
# Always use ml/.venv/bin/python (not poetry, not system python)

# Training from scratch
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/train.py \
    --run-name my-run \
    --experiment-name sentinel-retrain-v4 \
    --epochs 60 \
    --batch-size 16 \        # 32 OOMs on RTX 3070 Laptop with LoRA
    --lr 3e-4 \
    --loss-fn bce

# Fine-tuning from v3 (model-only, fresh optimizer — the default)
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/train.py \
    --run-name my-finetune \
    --experiment-name sentinel-retrain-v4 \
    --resume ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
    --epochs 30 \
    --lr 1e-4 \              # lower than v3's 3e-4
    --loss-fn bce

# Full resume (continuing an interrupted run — exact same config)
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/train.py \
    --run-name my-run \
    --resume ml/checkpoints/my-run_best.pt \
    --no-resume-model-only \  # restore optimizer + epoch counter
    --epochs 60
```

### 7.2 Threshold tuning (always run after training)

```bash
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/my-run_best.pt
# Outputs: ml/checkpoints/my-run_best_thresholds.json
```

### 7.3 Reading MLflow results

```bash
# Must use SQLite backend, not file backend
TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python - << 'EOF'
import mlflow
mlflow.set_tracking_uri("sqlite:///mlruns.db")  # NOT file:///mlruns
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(["3"])  # 3 = sentinel-retrain-v3
for r in runs:
    print(r.data.metrics)
EOF
```

### 7.4 Checkpoint format

```python
# What's inside a checkpoint .pt file:
{
    "model":           state_dict,      # all model weights
    "optimizer":       state_dict,      # Adam m/v moments (only in full checkpoints)
    "scheduler":       state_dict,      # OneCycleLR state
    "epoch":           54,              # epoch when this checkpoint was saved
    "best_f1":         0.4715,          # raw val F1-macro at save time
    "patience_counter": 6,              # early stopping counter at save time
    "config":          {...},           # full TrainConfig as dict
}

# Loading rules:
# - Checkpoint .pt files → weights_only=False (LoRA peft objects)
# - Graph .pt files → weights_only=True (add_safe_globals for PyG types)
```

### 7.5 VRAM constraints (RTX 3070 Laptop 8 GB)

| Setting | Max safe |
|---------|---------|
| batch_size | 16 (8 safer with lora_r=16) |
| lora_r | 16 (32 → OOM) |
| batch_size=32 | OOM with LoRA r≥8 |
| use_amp | Must be True (cannot train in float32 on 8 GB) |
| Display takes | ~1.6 GB always reserved |
| Practical free VRAM | ~6.4 GB for training |

### 7.6 v3 Baseline (the target to beat)

| Metric | Value |
|--------|-------|
| Raw F1-macro (best epoch) | 0.4715 (epoch 54) |
| Tuned F1-macro | **0.5069** |
| v4 gate | tuned > 0.5069 on same val_indices.npy |
| MLflow experiment | sentinel-retrain-v3, run d2ee23a |
| Checkpoint | ml/checkpoints/multilabel-v3-fresh-60ep_best.pt |
| Thresholds | ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json |

### 7.7 Per-class v3 tuned results

| Class | Threshold | F1 | P | R | Support | v4 Floor |
|-------|-----------|-----|---|---|---------|----------|
| CallToUnknown | 0.70 | 0.394 | 0.322 | 0.507 | 1,266 | 0.344 |
| DenialOfService | 0.95 | 0.400 | 0.318 | 0.540 | 137 | 0.350 |
| ExternalBug | 0.65 | 0.435 | 0.312 | 0.715 | 1,622 | 0.385 |
| GasException | 0.55 | 0.550 | 0.403 | 0.867 | 2,589 | 0.500 |
| IntegerUO | 0.50 | 0.821 | 0.758 | 0.896 | 5,343 | 0.771 |
| MishandledException | 0.60 | 0.492 | 0.365 | 0.754 | 2,207 | 0.442 |
| Reentrancy | 0.65 | 0.536 | 0.449 | 0.665 | 2,501 | 0.486 |
| Timestamp | 0.75 | 0.479 | 0.403 | 0.591 | 1,077 | 0.429 |
| TOD | 0.60 | 0.477 | 0.342 | 0.787 | 1,800 | 0.427 |
| UnusedReturn | 0.70 | 0.486 | 0.395 | 0.631 | 1,716 | 0.436 |

P = Precision, R = Recall, TOD = TransactionOrderDependence.
v4 Floor = no class can drop more than 0.05 F1 from v3 tuned values.
