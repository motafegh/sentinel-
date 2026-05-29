# Models — TransformerEncoder Chunk 1: LoRA Setup & Flash Attention

> **File:** `ml/src/models/transformer_encoder.py` — **lines 1–165**
> **Classes covered:** `TransformerEncoder.__init__` + module-level import guard
> **Time:** ~30 minutes
> **Interview relevance:** AI (LoRA, BERT internals, attention variants), ML (fine-tuning strategies, parameter-efficient training), MLOps (VRAM management, BF16)

---

## Warm-Up Recall (from GNN chunks)

Answer these before reading — one sentence each:

1. What does `register_buffer` do that a plain Python attribute on `nn.Module` does not?
2. Why does Phase 2 of the GNN use `add_self_loops=False`?
3. What does JK entropy measure, and what does H ≈ 0 indicate?

*(If any of these feel uncertain, re-read `03_jk_attention_internals.md` section 2 or 5 before continuing.)*

---

## 1. Big Picture — What This File Does

**File role:** `transformer_encoder.py` is the text-reading half of SENTINEL's dual-path architecture. It takes raw Solidity source code — already tokenized into integer token IDs — and produces a rich 768-dimensional embedding vector for every token position.

**Its place in the system:**

```
DualPathDataset
     │
     ├── graph (.pt)  ──► GNNEncoder          ──► node_embs [N, 256]
     │                    (already taught)
     │
     └── tokens (.pt) ──► TransformerEncoder  ──► token_embs [B, W*L, 768]
                          (this file)
                               │
                          CrossAttentionFusion ──► fused_eye [B, 128]
                          (chunk 06)
```

**Two classes in this file:**
- `TransformerEncoder` (lines 75–306): wraps GraphCodeBERT (abbreviation: a BERT variant pretrained on code) with LoRA (Low-Rank Adaptation) adapters. Takes token IDs → returns all token embeddings.
- `WindowAttentionPooler` (lines 309–351): extracts and combines the CLS (Classification) token from each of the W sliding windows. Covered in the next chunk.

**Why this file exists (the problem it solves):**

GraphCodeBERT has 125 million parameters trained on 6 programming languages' worth of GitHub code. It understands code structure deeply. But it has never seen Solidity vulnerability patterns. Two naive options:

| Option | Params updated | Problem |
|--------|---------------|---------|
| Full fine-tune | 125M | OOM (out of memory) on 8GB VRAM; catastrophic forgetting |
| Frozen | 0 | CodeBERT never adapts to vulnerability semantics |
| **LoRA** | **~590K** | **Adapts without forgetting; fits in 8GB** |

**Catastrophic forgetting** (P11 — domain term): when you update all weights of a large pretrained model on a small dataset (~68K contracts), the model "overwrites" its general language understanding with narrow domain-specific patterns and loses generalization. LoRA avoids this by keeping all 125M weights frozen and only training small adapter matrices alongside them.

---

## 2. The Import Guard — Fail Loud, Fail Early (lines 55–72)

```python
# Lines 55-59
try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

# Lines 62-72
if not _PEFT_AVAILABLE:
    raise RuntimeError(
        "peft library is required for TransformerEncoder but is not installed.\n"
        "Install it with:  pip install peft\n"
        "LoRA is not optional — without it CodeBERT has 0 trainable parameters "
        "and cannot adapt to vulnerability semantics."
    )
```

> **[LEARNING MODE: Understand the pattern]** — Know *why* this is structured this way. Don't memorize the exact error message.

**Why `RuntimeError` at import time, not a warning?**

A warning-then-fallback would mean: training runs silently with 0 trainable Transformer parameters. The GNN trains, the fusion layer trains, but CodeBERT never adapts. After 50 epochs you evaluate, notice the model is barely better than GNN-only, then spend hours debugging before realising `peft` was never installed. 

> ⚠️ **CRITICAL** — Import-time `RuntimeError` vs runtime warning is a design decision with real consequences. Silent failures in ML training are among the hardest bugs to catch — the model trains, the loss decreases, but you're training something different from what you intended. Hard failures at startup are always preferable for required dependencies.

**Why `try/except ImportError` instead of checking `importlib.util.find_spec`?**

`try/except` is the idiomatic Python approach — it handles both "package not installed" and "package installed but broken" in one shot. `find_spec` would only catch the "not installed" case; a broken install would still raise at runtime. 

> **[AUDIT A1]** — The module-level `raise` triggers whenever `transformer_encoder` is *imported*, not just when `TransformerEncoder` is instantiated. Any code that does `from ml.src.models.transformer_encoder import TransformerEncoder` without needing to instantiate it (e.g., type-checking tools, scripts that just inspect the class) will also crash if `peft` is missing. A cleaner design would defer the check to `__init__` where it's actually needed — but the current approach ensures the problem surfaces immediately in a training context.

---

## 3. `LoraConfig` — The Four Parameters That Matter (lines 118–125)

```python
# Lines 112-125
if lora_target_modules is None:
    lora_target_modules = ["query", "value"]
elif isinstance(lora_target_modules, str):
    # Guard: MLflow may deserialise list as comma-joined string
    lora_target_modules = [s.strip() for s in lora_target_modules.split(",")]

lora_config = LoraConfig(
    r=lora_r,                          # rank: bottleneck dimension
    lora_alpha=lora_alpha,             # scale: controls update magnitude
    target_modules=lora_target_modules, # which BERT layers to inject into
    lora_dropout=lora_dropout,         # dropout on LoRA paths
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
```

> **[LEARNING MODE: Master the detail]** — The four LoRA parameters below underpin a very common AI interview question. Know them well.

**`r=16` — the rank:**

LoRA decomposes the weight update ΔW into two small matrices:
```
ΔW = B @ A   where B ∈ ℝ^{d×r},  A ∈ ℝ^{r×d}
```
`r=16` means the update lives in a 16-dimensional subspace of the full 768×768 space. The intuition: fine-tuning a large pretrained model for a specific task typically only requires changes in a low-dimensional subspace — you don't need to move in all 589,824 directions at once.

**`lora_alpha=32` — the scale:**

The LoRA output is scaled by `alpha/r = 32/16 = 2.0`. This controls how strongly the LoRA update influences the output relative to the frozen weights:
```
h = W_frozen @ x + (B @ A) @ x × (alpha/r)
```
Higher alpha/r → LoRA update has more influence. The default 2.0 is a common starting point.

**`target_modules=["query", "value"]`:**

BERT (Bidirectional Encoder Representations from Transformers) has 12 attention layers, each with four linear projections: Q (query), K (key), V (value), and output. LoRA is injected only into Q and V — not K and output — because:
- Q and V together control what information is attended to and what is returned
- Adding LoRA to K or the output projection adds parameters with diminishing returns
- This is the standard LoRA configuration from the original LoRA paper

**`bias="none"`:**

Don't train bias terms in the LoRA layers. Biases add constant offsets that can easily be captured by the existing frozen weights. Keeping them frozen reduces trainable parameters further.

> **[AUDIT A2]** — The MLflow string guard (`elif isinstance(lora_target_modules, str): split(",")`). MLflow logs hyperparameters as strings. When a run is resumed or parameters are loaded from an MLflow run, a Python list `["query", "value"]` gets serialized as the string `"query, value"`. Without this guard, `LoraConfig(target_modules="query, value")` would try to find a module literally named `"query, value"` in BERT and fail silently (injecting LoRA nowhere). This is a good defensive guard — but the fact that it's needed reveals a fragility in how training configs are serialized and restored.

---

## 4. LoRA Mathematics — The Low-Rank Update

> **[LEARNING MODE: Master the detail]** — This is one of the most-asked AI interview topics. Understand the math, not just the label.

**Standard fine-tuning:**
```
W_new = W_0 + ΔW      ← updates all d×d = 768×768 = 589,824 values
h     = W_new @ x
```

**LoRA:**
```
W_eff = W_0 + B @ A × (α/r)    ← only A and B are trained
h     = W_0 @ x  +  (B @ A) @ x × (α/r)
      = h_frozen  +  h_lora
```

Where:
- `W_0` is frozen — never updated, gradient never computed for it
- `A ∈ ℝ^{r×d}` — initialized with random Gaussian (provides diversity)
- `B ∈ ℝ^{d×r}` — initialized to **zero** — critical initialization detail

**Why B=0 at initialization?**

At the start of training, `BA = 0`, so `h_lora = 0`. The model output is identical to the frozen pretrained model. This means:
- Training starts from the exact pretrained baseline
- Gradients are well-behaved from the first step
- No "shock" to the pretrained representations

If A were initialized to zero instead, the gradient through A would be zero (since `∂L/∂A = B^T ∂L/∂(BA)`), and A would never update.

**Parameter count:**
```
12 BERT layers × 2 projections (Q, V) × 2 matrices (A, B)
= 48 LoRA matrices

Each matrix pair: A[768, 16] + B[16, 768] = 12,288 + 12,288 = 24,576 params
Total: 48 × 24,576 = 1,179,648 / 2 = 589,824 ≈ 590K trainable
```
(The /2 is because A and B each contribute 12,288, so 48 pairs × 12,288 × 2 = 1,179,648, but that is indeed ~590K pairs.)

> ⚠️ **CRITICAL** — At inference time, LoRA matrices can be **merged** into the frozen weights: `W_merged = W_0 + B @ A × (α/r)`. This produces a model with zero inference overhead — the LoRA adapters disappear into the base weights. SENTINEL does not do this (it uses the peft library's forward-pass approach), but this is a common interview question.

---

## 5. Flash Attention 2 vs SDPA — The Try/Finally Pattern (lines 134–149)

```python
# Lines 134-149
_prev_default_dtype = torch.get_default_dtype()
try:
    self.bert = AutoModel.from_pretrained(
        "microsoft/graphcodebert-base",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    logger.info("TransformerEncoder — Flash Attention 2 active")
except (ImportError, ValueError):
    self.bert = AutoModel.from_pretrained(
        "microsoft/graphcodebert-base",
        attn_implementation="sdpa",
    )
    logger.info("TransformerEncoder — SDPA active (flash-attn unavailable)")
finally:
    torch.set_default_dtype(_prev_default_dtype)
```

> **[LEARNING MODE: Master the detail]** — Two separate concepts here. Both are interview-relevant.

### 5a. Flash Attention 2 vs SDPA

**Standard attention (naive):**
```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```
The problem: materializing `Q K^T` for L=512 tokens produces a 512×512 matrix — per head, per layer, per sequence in the batch. For `B*W=32` sequences through 12 layers with 12 heads:
```
32 × 12 × 12 × 512 × 512 × 2 bytes (BF16) ≈ 2.4 GB just for attention matrices
```

**Flash Attention 2** (FA2): computes attention in CUDA tiles, never materializing the full L×L matrix. Mathematically identical output, O(L) memory instead of O(L²), 2–4× faster. Requires: Ampere+ GPU (RTX 3070 ✓) + `flash-attn` package.

**SDPA (Scaled Dot-Product Attention):** PyTorch 2.0+ built-in fused kernel. Slower than FA2 but requires no extra packages. Falls back to standard attention on older GPUs.

The `try/except` tries FA2 first; if `flash-attn` is not installed (`ImportError`) or the GPU doesn't support it (`ValueError`), it falls back to SDPA.

### 5b. The dtype pollution guard — the `try/finally`

**The problem:** `AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)` calls `torch.set_default_dtype(torch.bfloat16)` as a side effect. This global change persists after the call returns. Any `nn.Linear(...)` created after this point — the GNN's eye projection, the fusion layer, the classifier — would default to BF16 weights, causing:
- Dimension mismatches with float32 tensors from the dataset
- Subtle numerical bugs that only appear at certain batch sizes

**The fix — `try/finally`:**
```python
_prev = torch.get_default_dtype()     # save current (float32)
try:
    self.bert = AutoModel.from_pretrained(..., torch_dtype=bfloat16)
    # ← set_default_dtype(bfloat16) happens inside here
finally:
    torch.set_default_dtype(_prev)    # always restore, even if from_pretrained throws
```

`finally` runs regardless of whether the `try` block succeeds or raises — it's the correct pattern for restoring global state.

> ⚠️ **CRITICAL** — `torch.set_default_dtype` is a **global** side effect. Side effects that outlive the function call that caused them are among the most dangerous bugs in Python ML code — they're non-local, order-dependent, and appear only in specific call sequences. Always guard global state changes with save/restore.

**Why BF16 (brain float 16) instead of FP16?**

| Format | Exponent bits | Mantissa bits | Overflow risk | GradScaler needed |
|--------|---------------|---------------|--------------|------------------|
| FP32   | 8             | 23            | None         | No               |
| BF16   | 8             | 7             | None         | **No** ← key     |
| FP16   | 5             | 10            | High         | Yes              |

BF16 keeps the same exponent range as FP32 — no overflow or underflow during training. FP16's smaller exponent range causes gradient underflow (gradients round to 0), requiring a `GradScaler` to rescale losses before backward. BF16 is strictly more convenient for training.

---

## 6. `get_peft_model()` — Three Things It Does (lines 157–165)

```python
# Lines 157-165
self.bert = get_peft_model(self.bert, lora_config)

trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in self.bert.parameters() if not p.requires_grad)
```

> **[LEARNING MODE: Master the detail]** — "What does get_peft_model do?" is a direct interview question for any LoRA/PEFT role.

`get_peft_model(model, config)` modifies `model` in-place and returns it. It does exactly three things:

**1. Freezes all original weights:**
```python
for param in bert.parameters():
    param.requires_grad = False
```
Every one of the 125M CodeBERT parameters gets `requires_grad=False`. PyTorch's autograd engine will not build backward graph nodes for operations whose all inputs have `requires_grad=False` — so gradient computation is skipped for the entire frozen path, saving both memory and compute.

**2. Injects A and B matrices:**
For each module named in `target_modules` (here: `"query"` and `"value"` in all 12 BERT layers), `get_peft_model` replaces the original `nn.Linear` with a `peft.LoraLinear` wrapper that:
- Holds the original frozen weight `W_0`
- Adds trainable `A` and `B` matrices with `requires_grad=True`

**3. Wires the forward pass:**
The `LoraLinear` wrapper's `forward` computes:
```python
h = F.linear(x, W_0)                           # frozen path
h = h + (x @ A.T) @ B.T × (alpha / r)          # LoRA path
```
Both paths run in the same forward call. Gradients flow only through A and B.

> **[AUDIT A3]** — Note that `get_peft_model` returns the modified model but also modifies it in-place. The assignment `self.bert = get_peft_model(self.bert, lora_config)` is correct but could mislead a reader into thinking a new object is created. In fact, `self.bert` before and after point to the same object (just modified). This is a subtle aliasing issue — harmless here, but important to understand if you ever hold a reference to the original bert object.

**The no_grad() trap:**

```python
# WRONG — do not do this:
with torch.no_grad():
    outputs = self.bert(input_ids=input_ids, ...)
```

This would disable gradient computation for *everything* inside the context — including the LoRA A and B matrices that have `requires_grad=True`. The frozen weights don't need gradients, but `no_grad()` is not selective. PEFT already handles the split: frozen weights simply don't generate gradient nodes because their `requires_grad=False`. No manual `no_grad()` scope is needed or safe.

---

## 7. Alternative Approaches — P7

> **[LEARNING MODE: Understand the pattern]** — Know these trade-offs for system design interviews.

**Full fine-tuning:**
Update all 125M weights. Pros: maximum expressiveness. Cons: requires ~2GB for weights alone in FP16, plus ~6GB for optimizer states (Adam stores 2 moment tensors per parameter) — total ~8GB, right at the GPU limit. Catastrophic forgetting on 68K contracts vs 6M GitHub files.

**Frozen BERT + linear probe:**
Freeze everything, add a single linear classifier on top of CLS. Pros: fastest, no forgetting. Cons: can't adapt internal representations — BERT's attention never learns to focus on `msg.sender`, `call.value`, or reentrancy-related patterns.

**Adapter layers (Houlsby et al. 2019):**
Insert small bottleneck modules (down-project → nonlinearity → up-project) after each BERT layer. Similar parameter count to LoRA but modifies the residual stream directly. LoRA is generally preferred now because it has zero inference overhead when merged.

**Prefix tuning (Li & Liang 2021):**
Learn a set of "soft prompt" tokens prepended to the input. SENTINEL actually uses a form of this (GNN prefix injection), but for a different purpose — injecting structural context, not task-specific instructions.

**LoRA (Hu et al. 2021) — SENTINEL's choice:**
~590K trainable parameters (0.47% of total). Adapts internal attention patterns. No inference overhead if merged. No catastrophic forgetting because frozen weights are unchanged.

---

## 8. Cross-File Relationships

**Already taught — recall these connections:**
- The GNN eye output `node_embs [N, 256]` (GNN chunks 1–3) is what eventually reaches `CrossAttentionFusion` alongside the `token_embs` this class produces.
- `register_buffer` (GNN chunk 3) — same pattern applies inside peft's `LoraLinear` for storing A and B.

**Not yet taught — preview:**
- `CrossAttentionFusion` (chunk 06) consumes `token_embs [B, W*L, 768]` from this class. It projects the 768-dim tokens down to 256 before cross-attention with GNN nodes.
- `SentinelModel.__init__` (chunk 07) instantiates `TransformerEncoder` and passes it `lora_r`, `lora_alpha` etc from `TrainConfig`. That's where `gnn_prefix_k` and `gnn_prefix_warmup_epochs` live — they control whether `gnn_prefix_nodes` is passed to `forward()`.

---

## 3 Things to Lock In

1. **LoRA = low-rank decomposition of the weight update.** `ΔW = B @ A × (α/r)`. A is Gaussian-init, B is zero-init so `ΔW=0` at start. Only A and B (590K params) train; 125M frozen weights never get gradients.

2. **`torch.set_default_dtype` is a global side effect.** `from_pretrained(..., torch_dtype=bfloat16)` triggers it. Always wrap in `try/finally` with save/restore to prevent BF16 pollution of later `nn.Linear` layers.

3. **`get_peft_model` does 3 things:** freezes all base weights (`requires_grad=False`), injects A+B matrices per targeted layer, wires the `LoraLinear` forward pass to compute `h_frozen + h_lora`.

---

## Challenge Questions

Answer these from memory before checking the chunk:

1. Why is B initialized to zero in LoRA, and what would break if A were initialized to zero instead?

2. Why does SENTINEL use BF16 rather than FP16 for the BERT backbone, and what would need to change if FP16 were used?

3. `get_peft_model(self.bert, lora_config)` is called after `AutoModel.from_pretrained(...)`. Why must the Flash Attention implementation be set *before* `get_peft_model`, not after?

4. The module-level `RuntimeError` for missing `peft` fires at import time. Name one scenario where this causes a problem that deferring the check to `__init__` would avoid.

---

**Next:** `05_transformer_forward_and_window_pooler.md` — the `forward()` method: standard path, prefix injection (single and multi-window), IMP-M3 masking, position IDs, and `WindowAttentionPooler`.
