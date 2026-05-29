# Models — Chunk 2: TransformerEncoder, LoRA, Flash Attention & GNN Prefix Injection

> **File:** `ml/src/models/transformer_encoder.py`
> **What you'll learn:** How BERT is adapted with LoRA for code understanding, Flash Attention vs SDPA, multi-window processing, and the GNN prefix injection mechanism.
> **Time:** ~30 minutes
> **Interview relevance:** ML (fine-tuning strategies), AI (BERT, attention, LoRA — heavily asked), MLOps (VRAM optimization)

---

## 1. Why Not Fine-Tune All of BERT?

GraphCodeBERT has **125 million parameters**. Why not just train all of them?

| Strategy | Params | Problem |
|----------|--------|---------|
| Full fine-tune | 125M | OOM on 8GB VRAM; catastrophic forgetting on 68K contracts |
| Frozen | 0 trainable | CodeBERT never adapts to vulnerability semantics |
| **LoRA** | **~590K** | **Adapts Q+V attention; no forgetting; fits in 8GB** |

**Catastrophic forgetting**: when you fine-tune all weights of a large pretrained model on a small domain-specific dataset, the model tends to "forget" its general language understanding and overfit to the specific training set. The pretrained weights encode 6 years of GitHub code — you don't want to destroy that.

LoRA's key insight: **the weight update needed for fine-tuning has low intrinsic rank.** Instead of updating the full 768×768 weight matrix, you can learn two small matrices: A [768, r=16] and B [r=16, 768].

---

## 2. LoRA — The Mathematics

For a frozen weight matrix W₀ ∈ ℝ^{d×k}:

```
Standard fine-tune: W = W₀ + ΔW          (d×k parameters to update)
LoRA:               W = W₀ + BA           (B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k})
```

Output of the layer:
```
Standard: h = W x = (W₀ + ΔW) x
LoRA:     h = W₀ x + BA x × (α/r)
```

Where:
- `r` = rank (default 16) — the "bottleneck" dimension
- `α/r` = scale factor (32/16=2.0) — controls the magnitude of the LoRA update
- A is initialized with random Gaussian noise
- B is initialized to zero → at init, `BA=0`, so `h = W₀ x` (identical to frozen model)

**In SENTINEL:**
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],  # Q and V projections in all 12 BERT layers
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
self.bert = get_peft_model(self.bert, lora_config)
# Result: 125M frozen + 590K trainable
```

`get_peft_model()` does three things:
1. Marks all original BERT weights as `requires_grad=False`
2. Injects trainable A and B matrices into each targeted layer
3. Forward pass computes `W₀ x + BA x × (α/r)` transparently

**Parameter count:**
- 12 BERT layers × 2 projections (Q, V) × 2 matrices (A, B) = 48 LoRA matrices
- Each: 768×16 + 16×768 = 24,576 params
- Total: 48 × 24,576 ≈ **590K trainable parameters** vs 125M frozen

> 🎯 **INTERVIEW FOCUS:** "How does LoRA work?" — Low-rank decomposition of the weight update: instead of updating W, learn two small matrices A [d,r] and B [r,k]. At inference, fold them back: W_adapted = W_frozen + BA × (α/r). Only ~0.5% of model parameters are trained.

---

## 3. Why No `torch.no_grad()` Around `self.bert()`?

The docstring explicitly warns about this:

```
"NOTE — why there is no torch.no_grad() around self.bert():
peft's get_peft_model() marks every original CodeBERT weight with
requires_grad=False. Wrapping the ENTIRE self.bert() call in no_grad()
would also cut gradient flow to the LoRA A/B matrices..."
```

If you wrote:
```python
with torch.no_grad():
    outputs = self.bert(...)  # WRONG! Kills LoRA gradients
```

This would disable gradient computation for everything inside the context, including the LoRA A and B matrices. `requires_grad=False` on the frozen weights already tells PyTorch's autograd to skip building backward graph nodes for those operations. No manual `no_grad()` scope is needed — and adding one would silently kill LoRA training.

---

## 4. Flash Attention 2 vs SDPA

```python
try:
    self.bert = AutoModel.from_pretrained(
        "microsoft/graphcodebert-base",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
except (ImportError, ValueError):
    self.bert = AutoModel.from_pretrained(
        "microsoft/graphcodebert-base",
        attn_implementation="sdpa",   # fallback
    )
```

**Standard attention (naive):**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The problem: materializing `QK^T` for sequence length L=512 requires an L×L=262,144 element matrix **per layer per head**. For batch size 8×4 windows = 32 sequences: 32 × 12 layers × 12 heads × 262K = ~1.2 billion floats → **~4.8 GB VRAM just for attention matrices**.

**Flash Attention 2** uses **tiled CUDA kernels** that compute attention in blocks, never materializing the full L×L matrix. The result is identical, but:
- Memory: O(L) instead of O(L²) → saves ~3-4 GB VRAM
- Speed: 2-4× faster due to better GPU memory access patterns

**SDPA (Scaled Dot-Product Attention)**: PyTorch 2.0+ built-in fused attention kernel. Slower than Flash Attention 2 but no extra dependencies.

**BF16 precision:**
```python
torch_dtype=torch.bfloat16
```
BF16 (brain float 16) uses the same exponent range as float32 but fewer mantissa bits. It:
- Halves memory usage vs float32
- Requires compatible GPU (Ampere or newer — RTX 3070 ✓)
- With no GradScaler needed (unlike FP16 which requires loss scaling for stability)

**The dtype pollution guard:**
```python
_prev_default_dtype = torch.get_default_dtype()
try:
    self.bert = AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)
finally:
    torch.set_default_dtype(_prev_default_dtype)
```

`from_pretrained(..., torch_dtype=bfloat16)` calls `torch.set_default_dtype(bfloat16)` as a side effect. This would make all subsequently created `nn.Linear` layers (in the GNN, fusion layer, etc.) default to BF16 weights — causing dimension mismatches and subtle numerical bugs. The try/finally restores the original dtype regardless of whether the load succeeds.

> 🎯 **INTERVIEW FOCUS:** "What is Flash Attention and why is it useful?" — Tiled CUDA kernels that avoid materializing the O(L²) attention matrix. Same mathematical result, O(L) memory instead of O(L²), 2-4× faster.

---

## 5. Multi-Window Forward Pass

```python
def forward(self, input_ids, attention_mask, ...):
    if input_ids.dim() == 2:
        # Single window: [B, L]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # [B, L, 768]
    
    # Multi-window: [B, W, L] → [B*W, L] → BERT → [B, W*L, 768]
    B, W, L = input_ids.shape
    flat_ids  = input_ids.view(B * W, L)   # flatten batch × windows
    flat_mask = attention_mask.view(B * W, L)
    outputs   = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
    return outputs.last_hidden_state.view(B, W * L, 768)  # reassemble
```

For multi-window input `[B=8, W=4, L=512]`:
1. Reshape to `[32, 512]` — treat each window as an independent sequence
2. Run all 32 sequences through GraphCodeBERT in one batch
3. Reshape output `[32, 512, 768]` → `[8, 2048, 768]`

Running all windows in one forward pass (rather than 4 separate passes) maximizes GPU utilization and benefits from Flash Attention's batched tiling.

---

## 6. GNN Prefix Injection — The Key Innovation

**The idea:** Take the top-K GNN node embeddings (declaration-level nodes like functions), project them into BERT's embedding space, and prepend them as "soft prefix tokens" before the code tokens. The Transformer then attends to both the structural context (GNN embeddings) and the raw source text simultaneously.

```
Without prefix:
  [CLS][token_1][token_2]...[token_510][SEP]   ← 512 positions, 510 code tokens

With K=48 prefix:
  [gnn_0][gnn_1]...[gnn_47][CLS][token_1]...[token_462][SEP]
  ←── 48 GNN nodes ──→←─────────── 464 code tokens ──────────→
  ←────────────────── 512 total positions ───────────────────→
```

```python
# In forward() with prefix path:
K = gnn_prefix_nodes.shape[1]   # K=48
code_budget = L - K              # 512 - 48 = 464

code_ids  = input_ids[:, :code_budget]    # [B, 464]
word_embs = self._word_embeddings(code_ids)  # [B, 464, 768]

# Concatenate: [B, 48, 768] + [B, 464, 768] = [B, 512, 768]
inputs_embeds = torch.cat([gnn_prefix_nodes, word_embs], dim=1)
```

**Why `inputs_embeds` instead of `input_ids`?**
`input_ids` are integer token IDs — you can't pass GNN embeddings (continuous 768-dim vectors) as integer IDs. `inputs_embeds` bypasses the token embedding lookup and directly provides embedding vectors. The Transformer processes them identically — it doesn't know the "prefix tokens" came from a GNN.

**Position IDs:**
```python
prefix_pos   = torch.ones(B, K, dtype=long) * 1     # all prefix at position 1
code_pos     = torch.arange(3, 3 + code_budget)      # code at positions 3..466
position_ids = torch.cat([prefix_pos, code_pos])
```

RoBERTa (which GraphCodeBERT is based on) uses:
- Position 0: BOS (`<s>`)
- Position 1: padding
- Position 2: EOS (`</s>`)

By using position 1 for all prefix tokens, they have identical positional bias — the transformer can only distinguish them by their content (GNN embedding), not by position. Code tokens start at position 3 (after the special token range).

> 🎯 **INTERVIEW FOCUS:** "How do you inject graph information into a Transformer?" — Project GNN node embeddings into the transformer's embedding dimension, then prepend them as continuous prefix tokens using `inputs_embeds`. The transformer attends to both the structural (GNN) tokens and the text tokens.

---

## 7. Prefix Warmup — Why Defer Prefix Activation

```python
# From trainer.py:
if current_epoch < gnn_prefix_warmup_epochs:
    gnn_prefix_nodes = None   # prefix is OFF
else:
    gnn_prefix_nodes = select_prefix_nodes(...)  # prefix is ON
```

Training starts with the prefix **disabled** for 15 epochs. After epoch 15, the prefix fires.

**Why?**
At epoch 0, the GNN projection (`gnn_to_bert_proj: Linear(256, 768)`) is randomly initialized. Prepending random noise vectors as prefix tokens would confuse the Transformer and slow its learning of the basic code representations.

By training the Transformer for 15 epochs without prefix, it first learns to understand Solidity code from text alone. At epoch 15, the GNN embeddings are already meaningful (the GNN has been training for 15 epochs too), and the projection can be learned on top of both trained representations.

**At inference:**
```python
# predictor.py:
model._current_epoch = 9999   # prefix is always active at inference
```

The prefix is never disabled in production — inference always uses the GNN structural context.

---

## 8. IMP-M3: Prefix Count Masking

```python
# 95.5% of contracts fill all K=48 slots — this is a near-no-op
if gnn_prefix_counts is not None:
    prefix_mask = torch.zeros(B, K)
    for b in range(B):
        prefix_mask[b, :gnn_prefix_counts[b]] = 1
    full_mask = torch.cat([prefix_mask, code_mask], dim=1)
```

Some contracts have fewer than 48 eligible declaration nodes (small contracts). For those, the remaining prefix slots are zero-padded. Without masking, the transformer would attend to zero vectors — not harmful but wasteful.

`gnn_prefix_counts` tracks the actual number of real (non-padded) prefix nodes per contract in the batch. Zero-padded positions get `attention_mask=0`, making them invisible to the transformer.

---

## 9. `WindowAttentionPooler` — Getting One Vector Per Contract

After the forward pass, the output is `[B, W*L, 768]` — embeddings for every token across all windows. But the classifier needs one vector per contract (or one per class). The `WindowAttentionPooler` extracts the CLS token from each window and combines them:

```python
class WindowAttentionPooler(nn.Module):
    def forward(self, token_embs):
        # token_embs: [B, W*L, 768]
        W = WL // window_size
        
        # CLS position in window i: i*512 + prefix_k
        cls_indices = torch.arange(W) * window_size + prefix_k
        window_cls = token_embs[:, cls_indices, :]   # [B, W, 768]
        
        # Learned attention over W CLS vectors
        scores  = self.attn(window_cls)              # [B, W, 1]
        weights = torch.softmax(scores, dim=1)        # [B, W, 1]
        return (weights * window_cls).sum(dim=1)      # [B, 768]
```

**Why attention pooling over CLS tokens instead of mean?**
Not all windows are equally informative. Window 0 (beginning of contract) might contain only boilerplate license/import statements. Window 2 might contain the vulnerable function. Learned attention allows the model to weight windows based on their content.

---

## 10. Summary — The Transformer Path

```
Input: tokens [B, W=4, L=512], attention_mask [B, 4, 512]
Optional: gnn_prefix_nodes [B, K=48, 768]
   ↓
If prefix active:
  code_budget = 512 - 48 = 464
  inputs_embeds = concat([gnn_prefix, word_embeddings(code[:464])], dim=1) [B, 512, 768]
  Flatten: [B*4, 512, 768]
   ↓
GraphCodeBERT + LoRA (125M frozen + 590K trainable)
12 BERT layers × Flash Attention 2 (BF16)
   ↓
Output: [B*4, 512, 768] → reshape → [B, 4*512, 768] = [B, 2048, 768]
   ↓
WindowAttentionPooler:
  Extract CLS at positions [0, 512, 1024, 1536] (or +prefix_k)
  Weighted sum [B, 4, 768] → [B, 768]
   ↓
Output: token_embeddings [B, 2048, 768]  (for CrossAttentionFusion)
         pooled_tokens   [B, 768]          (for TF eye classifier)
```

---

## Interview Questions

1. **"What is LoRA and when would you use it?"**
   → LoRA (Low-Rank Adaptation) factorizes the weight update as W' = W + BA where B [d,r] and A [r,k] have r << min(d,k). Used when: (1) full fine-tuning OOMs or risks catastrophic forgetting, (2) you want multiple task-specific adapters on the same frozen base model, (3) fast fine-tuning iterations on limited GPU budget.

2. **"How would you inject graph-structured information into a Transformer?"**
   → Project graph node embeddings into the Transformer's embedding dimension, then prepend them as `inputs_embeds` prefix tokens before the main sequence. Use `attention_mask=0` for padding prefix positions. Use warmup epochs to avoid injecting random noise before the GNN is trained.

3. **"What's the difference between BF16 and FP16 for training?"**
   → Both use 16 bits. BF16 has the same exponent range as FP32 (prevents overflow/underflow) but 7 mantissa bits (vs FP32's 23). FP16 has more mantissa precision but smaller exponent range — requires GradScaler to prevent loss underflow. BF16 is more stable for training (no GradScaler needed) but requires Ampere+ GPUs.

4. **"How do multi-window transformers work? Why not just use a longer BERT?"**
   → Sliding windows split long sequences into overlapping 512-token chunks. Each window is processed independently by BERT, then pooled across windows. Alternative (longformer-style attention) uses sparse attention patterns on the full sequence. SENTINEL uses windowed because: (1) reuses pretrained 512-token CodeBERT exactly, (2) each window independently processes one coherent code section.

---

**Next:** `03_fusion_layer_and_sentinel_model.md` — CrossAttentionFusion, three-eye classifier, and the complete SentinelModel assembly.
