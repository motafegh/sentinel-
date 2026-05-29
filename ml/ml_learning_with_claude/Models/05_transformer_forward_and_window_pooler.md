# Models — TransformerEncoder Chunk 2: `forward()` & WindowAttentionPooler

> **File:** `ml/src/models/transformer_encoder.py` — **lines 167–351**
> **Classes covered:** `TransformerEncoder.forward()` + `WindowAttentionPooler`
> **Time:** ~35 minutes
> **Interview relevance:** AI (BERT internals, prefix tuning, position embeddings), ML (sliding window inference), MLOps (vectorization, diagnostic hooks)

---

## Warm-Up Recall (from chunk 04)

Answer in one sentence each — no looking back:

1. What are the three things `get_peft_model()` does to the BERT model?
2. Why is the `try/finally` needed around `AutoModel.from_pretrained(..., torch_dtype=bfloat16)`?
3. What is `lora_alpha` and what does the ratio `alpha/r` control?

---

## 1. Big Picture — What `forward()` Must Handle

The `TransformerEncoder.forward()` has four distinct paths depending on two binary flags:

```
gnn_prefix_nodes is None?
        │
   YES ─┤                        NO
        │                         │
  single-window?          single-window?
  [B, L]                  [B, L]
     │                       │
  lines 213-215          lines 227-264
  (simplest path)        (prefix, single)
        │                         │
  multi-window?           multi-window?
  [B, W, L]               [B, W, L]
     │                       │
  lines 217-222          lines 266-306
  (flatten/unflatten)    (prefix + expand)
```

All four paths produce the same output shape: `[B, W*L, 768]` (or `[B, L, 768]` for single-window). The transformer itself is identical in all paths — what changes is how the input is constructed and assembled.

---

## 2. The `_word_embeddings` Property (lines 167–170)

```python
@property
def _word_embeddings(self) -> nn.Embedding:
    """Word embedding layer of the underlying GraphCodeBERT model."""
    return self.bert.base_model.model.embeddings.word_embeddings
```

> **[LEARNING MODE: Understand the pattern]** — Know what this accesses and why it's needed. Don't memorize the attribute chain.

**What it is:** GraphCodeBERT (like all BERT variants) has a token embedding lookup table — a matrix of shape `[vocab_size, 768]` that maps each integer token ID to a 768-dim vector. `_word_embeddings` is a Python `@property` that navigates the peft/HuggingFace model hierarchy to get to it.

**Why the long attribute chain?**

After `get_peft_model()` wraps the model:
```
self.bert                          ← PeftModel wrapper
  .base_model                      ← LoraModel wrapper  
    .model                         ← the actual AutoModel (RoBERTa)
      .embeddings
        .word_embeddings            ← nn.Embedding [50265, 768]
```

Each wrapping layer adds one level of indirection. The `@property` hides this complexity behind a clean interface.

**Why is this needed at all?**

The prefix injection path needs to bypass the token embedding lookup and provide continuous vectors (GNN embeddings projected to 768-dim) directly. To do this, you call `self.bert(inputs_embeds=...)` instead of `self.bert(input_ids=...)`. But the code tokens also need to be embedded — and you can't pass both `input_ids` and `inputs_embeds` at the same time (BERT's API raises an error). So the code manually calls `_word_embeddings(code_ids)` to embed only the code tokens, then concatenates with the prefix embeddings.

---

## 3. Path 1 — Standard Single-Window (lines 211–215)

```python
if gnn_prefix_nodes is None:
    if input_ids.dim() == 2:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state    # [B, L, 768]
```

> **[LEARNING MODE: Awareness only]** — Simplest path. Know it exists and what it returns.

Four tokens of code → four embeddings, each 768-dimensional. `last_hidden_state` is BERT's output after all 12 transformer layers: every token's final contextual representation. Not just the CLS — every token.

---

## 4. Path 2 — Standard Multi-Window (lines 217–222)

```python
B, W, L = input_ids.shape            # e.g. [8, 4, 512]
flat_ids  = input_ids.view(B * W, L) # [32, 512]
flat_mask = attention_mask.view(B * W, L)
outputs   = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
return outputs.last_hidden_state.view(B, W * L, 768)  # [8, 2048, 768]
```

> **[LEARNING MODE: Master the detail]** — The flatten/unflatten pattern is the key insight. Understand why it works.

**Why this works:**

BERT processes sequences independently — it has no concept of "this is window 2 of 4 from contract X." Each sequence `[32, 512]` is treated as an independent sample. Running all 32 windows in one batch call is identical to running 4 separate batches of 8. The difference is GPU utilization: one large batch is much more efficient than four smaller ones.

**The reshape sequence:**
```
Input:  [B=8, W=4, L=512]
  ↓ .view(B*W, L)
Flat:   [32, 512]             ← BERT sees 32 independent sequences
  ↓ BERT forward
Output: [32, 512, 768]
  ↓ .view(B, W*L, 768)
Final:  [8, 2048, 768]        ← windows stitched back together along seq dim
```

The windows are stitched in order: positions `[0:512]` = window 0, `[512:1024]` = window 1, etc. This is the format `WindowAttentionPooler` and `CrossAttentionFusion` expect.

> ⚠️ **CRITICAL** — `.view()` requires the tensor to be contiguous in memory. If `input_ids` were the result of a non-contiguous operation (e.g., a transpose), `.view()` would throw. In practice, DataLoader output is always contiguous, but this is a potential failure mode if inputs are manually constructed.

---

## 5. Path 3 — Prefix Injection, Single-Window (lines 224–264)

This is the most mechanically complex path. Walk through it step by step.

```python
K = gnn_prefix_nodes.shape[1]    # K=48 GNN prefix tokens
B, L = input_ids.shape
code_budget = L - K               # 512 - 48 = 464 code tokens
```

**Step 1 — Truncate code tokens to make room for prefix:**
```python
code_ids  = input_ids[:, :code_budget]      # [B, 464] — keep CLS at position 0
code_mask = attention_mask[:, :code_budget] # [B, 464]
```

The first 464 tokens of each sequence are kept. Tokens 464–511 (the tail of the contract) are discarded. This is the cost of prefix injection — K positions of code coverage are traded for K positions of structural context.

**Step 2 — Build `inputs_embeds` by concatenating prefix and code embeddings:**
```python
word_embs = self._word_embeddings(code_ids).to(dtype=gnn_prefix_nodes.dtype)
#           [B, 464, 768]                                                    ↑ dtype match
inputs_embeds = torch.cat([gnn_prefix_nodes, word_embs], dim=1)  # [B, 512, 768]
```

The concatenation places 48 GNN vectors before 464 code token vectors. The BERT model receives this as `inputs_embeds` — it bypasses the embedding lookup entirely and treats these 512 continuous vectors as its input sequence.

> **[AUDIT A4]** — `.to(dtype=gnn_prefix_nodes.dtype)`: the word embeddings are in the model's default dtype (float32 or BF16 depending on the model load). `gnn_prefix_nodes` comes from the GNN encoder which operates in float32. If there's a dtype mismatch, the `cat` would fail. This explicit cast is correct — but it reveals that dtype consistency between GNN and Transformer paths is not enforced at a higher level (in `SentinelModel.forward`). A more robust design would normalize dtypes at the `SentinelModel` level rather than patching it here.

**Step 3 — IMP-M3: Build prefix attention mask (lines 239–244):**
```python
if gnn_prefix_counts is not None:
    prefix_mask = torch.zeros(B, K, dtype=attention_mask.dtype, device=attention_mask.device)
    for b in range(B):
        prefix_mask[b, :gnn_prefix_counts[b]] = 1   # real nodes → attend; padded → don't
else:
    prefix_mask = torch.ones(B, K, ...)   # all K slots are real
full_mask = torch.cat([prefix_mask, code_mask], dim=1)   # [B, 512]
```

> **[LEARNING MODE: Master the detail]** — IMP-M3 (Implementation improvement M3) handles small contracts.

Some Solidity contracts have fewer than K=48 declaration-level nodes (FUNCTION, MODIFIER, etc.). `select_prefix_nodes` in `SentinelModel` zero-pads the prefix tensor to always produce `[B, K, 768]`, but tracks how many nodes are real via `gnn_prefix_counts [B]`.

Without IMP-M3: the BERT transformer would attend to zero vectors in the padded prefix slots — not harmful (zero vectors produce low attention logits) but wastes attention capacity.

With IMP-M3: padded prefix slots get `attention_mask=0`, making them invisible to BERT's self-attention (masked positions receive -∞ before softmax → 0 attention weight).

> **[AUDIT A5]** — The `for b in range(B)` loop at line 241. This is a Python-level loop over batch size B. At B=64, this is 64 Python iterations — slow. A vectorized alternative:
> ```python
> # Vectorized version — no Python loop:
> idx  = torch.arange(K, device=prefix_mask.device).unsqueeze(0)  # [1, K]
> counts = gnn_prefix_counts.unsqueeze(1)                           # [B, 1]
> prefix_mask = (idx < counts).to(attention_mask.dtype)             # [B, K]
> ```
> The comment in the original code says "95.5% of contracts fill all K=48 slots," which means `gnn_prefix_counts` is often None and this branch is skipped entirely. The performance impact is minimal in practice — but it's a pattern that doesn't scale cleanly to larger batches or K values.

**Step 4 — Position IDs (lines 247–250):**
```python
prefix_pos   = input_ids.new_ones(B, K)      # all prefix tokens → position ID 1
code_pos     = torch.arange(3, 3 + code_budget, ...).unsqueeze(0).expand(B, -1)
position_ids = torch.cat([prefix_pos, code_pos], dim=1)   # [B, 512]
```

> **[LEARNING MODE: Master the detail]** — The position ID choices are not arbitrary. Know why these specific values.

GraphCodeBERT is based on RoBERTa (Robustly Optimized BERT). RoBERTa's position embedding table has a reserved structure:
- Position 0: BOS — beginning of sequence (`<s>`)
- Position 1: padding position
- Position 2: EOS — end of sequence (`</s>`)
- Positions 3+: actual content positions

By assigning **position 1** to all 48 prefix tokens, they all share the same positional embedding — the padding slot. This means:
- BERT cannot distinguish prefix tokens from each other by position (they all have the same position encoding)
- The transformer must rely purely on the content (the 768-dim GNN embedding) to distinguish between prefix tokens
- No positional bias toward "earlier prefix tokens are more important"

Code tokens start at **position 3** — skipping the special token slots (0, 1, 2). The highest position used is `3 + 464 - 1 = 466`, well within RoBERTa's 514-position limit.

```
Position ID layout:
  [1, 1, 1, ..., 1,   3,  4,  5, ..., 466]
   ←── 48 prefix ──→  ←── 464 code ──────→
```

**Step 5 — BERT forward call:**
```python
outputs = self.bert(
    inputs_embeds=inputs_embeds,    # [B, 512, 768] — NOT input_ids
    attention_mask=full_mask,       # [B, 512] — prefix + code mask combined
    position_ids=position_ids,      # [B, 512] — prefix at 1, code at 3-466
    output_attentions=output_attentions,   # IMP-M2 diagnostic
)
```

`inputs_embeds` and `input_ids` are mutually exclusive in BERT's API — passing both raises an error. By using `inputs_embeds`, we fully control what each of the 512 positions "sees" at the embedding layer.

**Step 6 — `output_attentions` diagnostic (lines 258–263):**
```python
if output_attentions and outputs.attentions is not None:
    attn = torch.stack(list(outputs.attentions), dim=0)  # [12, B, heads, L, L]
    prefix_attn_mean = attn[:, :, :, K:, :K].mean().item()
    return outputs.last_hidden_state, prefix_attn_mean
```

> **[LEARNING MODE: Understand the pattern]** — Know what this measures and when it's used.

`outputs.attentions` is a tuple of 12 tensors (one per BERT layer), each `[B, num_heads, L, L]`. The `[i, j]` entry is the attention weight from position i to position j.

The slice `[:, :, :, K:, :K]`:
- `K:` → rows = code token positions (the ones doing the attending)
- `:K` → columns = prefix token positions (what they're attending to)

This extracts: "how much do code tokens attend to prefix tokens?" averaged over all layers, heads, and sequences. Called once per validation epoch (not per training step — it adds ~15% overhead). If `prefix_attn_mean < 0.002` after 5+ warmup epochs, the transformer is ignoring the prefix — an IMP-M2 diagnostic gate signal.

---

## 6. Path 4 — Prefix Injection, Multi-Window (lines 266–306)

Most of the logic is identical to Path 3. The key difference: the prefix `[B, K, 768]` must be replicated across all W windows.

```python
B, W, L = input_ids.shape
flat_ids = input_ids[:, :, :code_budget].reshape(B * W, code_budget)   # [B*W, 464]

# Expand prefix: [B, K, 768] → [B*W, K, 768]
prefix_expanded = (
    gnn_prefix_nodes.unsqueeze(1)      # [B, 1, K, 768]
                    .expand(-1, W, -1, -1)  # [B, W, K, 768]
                    .reshape(B * W, K, 768) # [B*W, K, 768]
)
```

> **[LEARNING MODE: Master the detail]** — The `unsqueeze/expand/reshape` chain. Step through it mentally.

`unsqueeze(1)` inserts a new dimension at position 1: `[B, K, 768]` → `[B, 1, K, 768]`.

`expand(-1, W, -1, -1)` expands the size-1 dimension to W without copying memory: `[B, 1, K, 768]` → `[B, W, K, 768]`. `-1` means "keep this dimension as-is."

`reshape(B*W, K, 768)` flattens B and W into one dimension: `[B, W, K, 768]` → `[B*W, K, 768]`.

**Why `expand` not `repeat`?**

`expand` returns a view with a stride of 0 in the expanded dimension — the data is not copied. `repeat` allocates new memory. For W=4 windows and K=48 nodes with 768 dims, `expand` saves `4 × 48 × 768 × 4 bytes = ~600KB` per batch.

The prefix mask per-graph is also expanded across windows:
```python
prefix_mask = ...               # [B, K]
prefix_mask = prefix_mask.unsqueeze(1).expand(-1, W, -1).reshape(B * W, K)
# [B, K] → [B, 1, K] → [B, W, K] → [B*W, K]
```

Same prefix mask for all windows of the same contract — the same nodes are real/padded regardless of which window of the code we're looking at.

---

## 7. `WindowAttentionPooler` (lines 309–351)

```python
class WindowAttentionPooler(nn.Module):
    def __init__(self, hidden_dim=768, window_size=512, prefix_k=0):
        self.window_size = window_size
        self.prefix_k    = prefix_k
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, token_embs):    # [B, W*L, 768] or [B, L, 768]
        B, WL, D = token_embs.shape
        if WL <= self.window_size:
            return token_embs[:, self.prefix_k, :]   # single-window fast path
        W = WL // self.window_size
        cls_indices = torch.arange(W, device=...) * self.window_size + self.prefix_k
        window_cls = token_embs[:, cls_indices, :]   # [B, W, 768]
        scores  = self.attn(window_cls)              # [B, W, 1]
        weights = torch.softmax(scores, dim=1)       # [B, W, 1]
        return (weights * window_cls).sum(dim=1)     # [B, 768]
```

> **[LEARNING MODE: Master the detail]** — Know why CLS and learned attention pooling, not mean.

**Why CLS tokens specifically?**

CLS (Classification) is BERT's convention for the "summary token" — it attends to all other tokens in a window via bidirectional self-attention and ends up as a compressed representation of the entire window. It's the natural aggregation point for a window's content.

**CLS position with prefix:**

Without prefix: CLS is at position 0 of each window → `i * 512 + 0`.

With prefix (K=48): the 48 GNN prefix tokens occupy positions 0–47. CLS is now at position 48 → `i * 512 + prefix_k`.

```python
cls_indices = torch.arange(W) * self.window_size + self.prefix_k
# W=4, window_size=512, prefix_k=48:
# [0*512+48, 1*512+48, 2*512+48, 3*512+48] = [48, 560, 1072, 1584]
```

**Why learned attention, not mean?**

Not all windows are equally informative. A 4-window contract:
- Window 0: license header, pragma, imports — mostly boilerplate
- Window 1: state variable declarations
- Window 2: the `withdraw()` function with the reentrancy vulnerability
- Window 3: view functions and events

Mean pooling would weight all four equally. The learned attention can assign high weight to window 2 and low weight to window 0, focusing the transformer eye on the informative content.

**`bias=False`:** Same rationale as `_JKAttention.attn` — prevents content-independent window preference.

**Single-window fast path:**
```python
if WL <= self.window_size:
    return token_embs[:, self.prefix_k, :]
```
If the contract fit in one window (W=1, WL=512), return the CLS directly at `prefix_k`. No learned attention layer, no `softmax` — zero overhead.

---

## 8. Data Flow — Complete `forward()` Picture

```
input_ids [B, W, L]     attention_mask [B, W, L]     gnn_prefix_nodes [B, K, 768]?
        │                        │                              │
        └────────────────────────┼──────────────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │  gnn_prefix_nodes?     │
                    │  No → standard path    │
                    │  Yes → prefix path     │
                    └────────────────────────┘
                                 │
          Standard path          │         Prefix path
     ─────────────────           │    ─────────────────────────────
     view(B*W, L)                │    code_budget = L - K
     self.bert(input_ids)        │    _word_embeddings(code_ids[:code_budget])
     view(B, W*L, 768)           │    cat([prefix_nodes, word_embs])  → inputs_embeds
                                 │    build full_mask + position_ids
                                 │    self.bert(inputs_embeds=...)
                                 │    optionally: slice attn[:,:,:,K:,:K] for IMP-M2
                                 ↓
                    token_embs [B, W*L, 768]
                                 │
                         WindowAttentionPooler
                    cls_indices = i*512 + prefix_k
                    window_cls [B, W, 768]
                    attn(window_cls) → softmax → weighted sum
                                 │
                         pooled [B, 768]  ← used by transformer eye in SentinelModel
```

---

## 9. Alternative Approach — P7

**Sliding windows vs Longformer-style sparse attention:**

SENTINEL's sliding-window approach runs W independent BERT passes and pools the CLS tokens. An alternative is **Longformer** (Beltagy et al. 2020), which extends BERT to handle sequences up to 4096 tokens with:
- Local windowed attention: each token attends to a fixed window of neighbors
- Global attention for CLS and a few task-specific tokens

| | SENTINEL sliding window | Longformer |
|--|------------------------|-----------|
| Token coverage | 62% of median contract | 100% possible |
| VRAM | O(W × L²) per layer | O(L × window) per layer |
| Pretrained weights | Reuse 512-token CodeBERT directly | Requires Longformer checkpoint |
| Cross-window attention | None (windows independent) | Full (via global attention) |
| Prefix injection | Straightforward (per-window) | Complex (global attention tokens) |

SENTINEL uses sliding windows because: (1) CodeBERT's pretrained 512-token weights are reused exactly, (2) the GNN path already captures cross-function structural patterns that Longformer's cross-window attention would provide.

---

## 3 Things to Lock In

1. **Prefix injection forces a trade-off:** K prefix slots reduce code coverage by K tokens (`code_budget = L - K`). The 48 GNN embeddings get positions all assigned to 1 (the RoBERTa padding slot) — same positional encoding, content-only distinction.

2. **`expand` vs `repeat`:** `expand` is a zero-copy view with stride 0. Always prefer it over `repeat` when you only need broadcasting, not independent copies. Saves memory proportional to the expansion factor.

3. **`WindowAttentionPooler` CLS index formula:** `i * window_size + prefix_k`. Without prefix, CLS is at position 0 per window. With prefix, it shifts by K. Get this wrong and the "CLS" you extract is actually a code token, not the summary token.

---

## Challenge Questions

1. Why does prefix injection use `inputs_embeds` instead of `input_ids`, and why can't both be passed simultaneously?

2. In the multi-window prefix path, why is `expand` used instead of `repeat` when replicating the prefix across W windows? What would break if you used `repeat` instead?

3. The position ID for all 48 prefix tokens is set to 1 (RoBERTa's padding slot). What would happen if they were set to positions 0, 1, 2, ..., 47 instead — what information would be added or lost?

4. Look at the `output_attentions` diagnostic slice: `attn[:, :, :, K:, :K]`. What do the rows (`K:`) and columns (`:K`) represent, and what does a value near 0 for this mean about the model's behavior?

---

**Next:** `06_cross_attention_fusion.md` — `CrossAttentionFusion` and `_scatter_to_dense`: bidirectional cross-attention, `torch.compile` compatibility, and all the Fix # patches.
