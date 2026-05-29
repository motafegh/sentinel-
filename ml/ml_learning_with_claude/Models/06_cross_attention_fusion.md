# Models — Chunk 6: CrossAttentionFusion & `_scatter_to_dense`

> **File:** `ml/src/models/fusion_layer.py` — **lines 1–281**
> **Classes covered:** `_scatter_to_dense()` helper + `CrossAttentionFusion`
> **Time:** ~35 minutes
> **Interview relevance:** AI (cross-attention, multi-modal fusion), ML (masked pooling), MLOps (torch.compile, VRAM optimization, defensive invariants)

---

## Warm-Up Recall (from chunks 04 & 05)

One sentence each — no looking back:

1. What does `expand(-1, W, -1, -1)` do differently from `repeat(1, W, 1, 1)`, and why does it matter?
2. Why are all 48 GNN prefix tokens assigned position ID 1 in the TransformerEncoder?
3. The `output_attentions` diagnostic slices `attn[:, :, :, K:, :K]`. What does "rows K:" and "cols :K" represent?

---

## 1. Big Picture — What This File Does

**File role:** `fusion_layer.py` is the bridge between the two paths. After the GNN produces node embeddings and the Transformer produces token embeddings, this module makes them *talk to each other* before compression — then delivers a single `[B, 128]` vector per contract.

**Its place in the system:**

```
GNNEncoder
  node_embs [N, 256]   ← all nodes from all B graphs, flat
  batch     [N]        ← which graph each node belongs to
                 ↘
                  CrossAttentionFusion ──► fused_eye [B, 128]
                 ↗
TransformerEncoder
  token_embs [B, W*L, 768]
  attention_mask [B, W*L]
```

**What replaced what, and why:**

The original fusion was:
```python
# BEFORE — simple concat + MLP
fused = MLP(cat([pooled_gnn [B,64], pooled_tf [B,768]]))  →  [B, 64]
```

Both paths were individually pooled into single summary vectors *before* they ever interacted. The problem: when `withdraw()` as a GNN node and `"call.value"` as a token are both compressed into their respective summaries, the specific pairing information is lost.

```
AFTER — interact at fine granularity BEFORE pooling:
  Every GNN node   queries all 512 code tokens  → enriched nodes
  Every code token queries all GNN nodes         → enriched tokens
  Pool AFTER enrichment
```

Now `withdraw()` finds `"call.value"` before averaging. The reentrancy signal survives into the fused representation.

**Two components in this file:**
- `_scatter_to_dense()` (lines 68–117): a utility function that pads ragged node lists into a uniform dense tensor — required for `torch.compile` compatibility
- `CrossAttentionFusion` (lines 120–281): the bidirectional cross-attention module itself

---

## 2. `_scatter_to_dense` — Why It Exists (lines 68–117)

> **[LEARNING MODE: Master the detail]** — The `torch.compile` graph break is a concrete MLOps problem. The fix teaches a general pattern.

**The problem it solves:**

Graphs in a batch have different numbers of nodes. To run batched attention, you need a uniform tensor shape. PyG's (PyTorch Geometric's) built-in `to_dense_batch()` would be the natural choice:

```python
padded_nodes, mask = to_dense_batch(nodes_proj, batch)  # [B, max_n, 256]
```

But `to_dense_batch()` computes `max_n = counts.max()` — a value that depends on the actual data in the current batch. When `torch.compile` traces this:

```
torch.compile sees: "The output tensor shape depends on a runtime value."
Result: GRAPH BREAK — fusion's forward falls back to eager (Python) mode.
```

A graph break (GBreak) means `torch.compile` cannot fuse the operations on either side of the break into a single optimized CUDA kernel. The entire `CrossAttentionFusion` forward runs interpreted, losing all compile optimization.

**The fix — static `max_nodes=1024`:**

```python
def _scatter_to_dense(x, batch, num_graphs, max_nodes):  # max_nodes=1024 always
    N, D = x.shape
    # Count nodes per graph
    ones   = torch.ones(N, dtype=torch.long, device=x.device)
    counts = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
    counts.scatter_add_(0, batch, ones)                  # [num_graphs]

    # Per-graph start offsets in the flat node list
    offsets = torch.cat([
        x.new_zeros(1, dtype=torch.long),
        counts[:-1].cumsum(0),
    ])                                                   # [num_graphs]

    # Local index of each node within its graph
    local_idx = torch.arange(N, device=x.device) - offsets[batch]

    # BUG-C2: compute valid BEFORE clamping
    valid     = local_idx < max_nodes                    # [N] bool
    local_idx = local_idx.clamp(max=max_nodes - 1)       # safe index for scatter

    out  = x.new_zeros(num_graphs, max_nodes, D)
    mask = torch.zeros(num_graphs, max_nodes, dtype=torch.bool, device=x.device)
    out[batch[valid], local_idx[valid]]  = x[valid]
    mask[batch[valid], local_idx[valid]] = True
    return out, mask                                     # [B, 1024, D], [B, 1024]
```

> **[LEARNING MODE: Master the detail]** — Walk through the `scatter_add_` + `cumsum` + local index computation. This is a general pattern for batched ragged sequences.

**Step-by-step mechanism:**

```
Suppose batch = [0, 0, 0, 1, 1, 2, 2, 2, 2]  (9 nodes across 3 graphs)

counts = [3, 2, 4]       (nodes per graph via scatter_add_)

offsets = [0, 3, 5]      (cumsum of counts[:-1] = cumsum([3, 2]))
                          prepend 0 → [0, 3, 5]

local_idx = arange(9) - offsets[batch]
          = [0,1,2,3,4,5,6,7,8] - [0,0,0,3,3,5,5,5,5]
          = [0,1,2,0,1,0,1,2,3]   ← position of each node within its graph
```

Each node now knows its local position within its graph. These local indices are used to scatter into a `[B, 1024, D]` tensor.

**The BUG-C2 subtlety — why valid must be computed before clamping:**

```python
# WRONG ORDER (original bug):
local_idx = local_idx.clamp(max=max_nodes - 1)   # clamp first
valid     = local_idx < max_nodes                 # now ALL are < max_nodes → valid=True for excess!

# CORRECT ORDER (after fix):
valid     = local_idx < max_nodes                 # identify excess BEFORE clamp
local_idx = local_idx.clamp(max=max_nodes - 1)   # now safe to clamp for indexing
```

Without the fix: a graph with 1025 nodes has one node with `local_idx=1024`. After clamping, it maps to position 1023. Multiple excess nodes all clamp to 1023 — last-write-wins, producing a random embedding at position 1023. The mask at 1023 gets set to True (marking it as real). That random embedding then participates in attention as if it were a legitimate node.

With the fix: the check `local_idx < max_nodes` runs first, marking position 1024 as invalid. The clamped index is used only for safe tensor indexing — the actual scatter only happens for `valid` nodes. Excess nodes are silently dropped.

> **[AUDIT A6]** — "Affects <1% of the corpus at max_nodes=1024" — this claim is in the docstring but there's no assertion or metric tracking to verify it over time. As the training corpus evolves or new contracts are added, graphs could become larger. A production system should track the truncation rate (e.g., log when `_c4_truncation_warned` fires) via MLflow so that a growing truncation rate triggers a `max_nodes` increase.

**The `_c4_truncation_warned` global:**
```python
_c4_truncation_warned: bool = False

if _max_n > max_nodes:
    if not _c4_truncation_warned:
        _c4_truncation_warned = True
        logger.warning("C-4: graphs exceeding max_nodes detected. Fires once.")
```

`log once` pattern: the first truncation triggers a warning; subsequent ones are silently suppressed. This prevents log flooding when 1% of 44K contracts are large. The global `bool` is module-level state — it persists across training runs in the same Python process but resets on process restart.

---

## 3. `CrossAttentionFusion.__init__` (lines 120–195)

> **[LEARNING MODE: Understand the pattern]** — The module structure. Know what each sub-module does and why it exists.

```python
def __init__(self, node_dim=64, token_dim=768, attn_dim=256,
             num_heads=8, output_dim=128, dropout=0.1, max_nodes=1024):

    # Validation
    if attn_dim % num_heads != 0:
        raise ValueError(...)    # 256 % 8 = 0 ✓

    self.node_proj  = nn.Linear(node_dim,  attn_dim)   # 256→256
    self.token_proj = nn.Linear(token_dim, attn_dim)   # 768→256

    # BUG-C2: normalize tokens BEFORE projection
    self.token_norm = nn.LayerNorm(token_dim)           # [768]

    # Direction 1: nodes query tokens
    self.node_to_token = nn.MultiheadAttention(
        embed_dim=attn_dim, num_heads=num_heads,
        dropout=dropout, batch_first=True,
    )
    # Direction 2: tokens query nodes
    self.token_to_node = nn.MultiheadAttention(
        embed_dim=attn_dim, num_heads=num_heads,
        dropout=dropout, batch_first=True,
    )
    # Final compression
    self.output_proj = nn.Sequential(
        nn.Linear(attn_dim * 2, output_dim),   # 512→128
        nn.ReLU(),
        nn.Dropout(dropout),
    )
```

**Why `attn_dim=256` for both modalities?**

The GNN outputs 256-dim node embeddings. The Transformer outputs 768-dim token embeddings. For the two MHA modules to operate in the same space (Q, K, V must all have the same dimension), both are projected to a common `attn_dim=256`.

`attn_dim % num_heads == 0` is required because each of the 8 attention heads gets `256/8 = 32` dimensions. If `attn_dim` is not divisible by `num_heads`, PyTorch's MHA raises at runtime. The validation at init makes this fail immediately with a clear message.

**BUG-C2: `token_norm` — why LayerNorm before `token_proj`:**

CodeBERT hidden states have L2 norm ~10–15. The GNN's per-phase LayerNorm means node embeddings have norm ~1. When cross-attention computes dot products `Q · K^T`:
```
node_query [B, n, 256]  ·  token_key [B, 512, 256]^T
```
If token keys have 10–15× larger magnitude than node queries, the dot products are dominated by token norm rather than semantic relevance. Every node ends up attending to the highest-norm tokens regardless of content.

`token_norm = LayerNorm(768)` normalizes each token embedding to mean=0, std=1 *before* `token_proj` maps it to the 256-dim attention space. After normalization, both modalities are on the same scale.

> ⚠️ **CRITICAL** — This bug (BUG-C2) would be completely invisible in training: loss decreases, metrics look reasonable, but cross-attention is routing by magnitude not meaning. The only way to detect it is to inspect what tokens each node attends to — which requires `output_attentions=True` and manual inspection. This class of bug — subtle misalignment that doesn't cause crashes — is among the hardest to catch in ML systems.

**`batch_first=True`:**

PyTorch's `MultiheadAttention` defaults to sequence-first tensors (`[L, B, D]`). `batch_first=True` makes it accept `[B, L, D]` — matching the rest of SENTINEL's convention. Always set this in modern PyTorch code.

---

## 4. `CrossAttentionFusion.forward` — Step by Step (lines 197–281)

### Step 0 — Device check (Fix #4)

```python
if node_embs.device != token_embs.device:
    raise RuntimeError(
        f"Device mismatch: node_embs on {node_embs.device} "
        f"but token_embs on {token_embs.device}."
    )
```

> **[LEARNING MODE: Understand the pattern]**

In multi-GPU training or when debugging on CPU while the model is on CUDA, device mismatches produce cryptic PyTorch errors deep inside the attention computation. This check surfaces the problem immediately at the boundary where it can be diagnosed.

### Step 1 — Project both modalities to common space

```python
nodes_proj  = self.node_proj(node_embs)                    # [N, 256] → [N, 256]
tokens_proj = self.token_proj(self.token_norm(token_embs)) # [B,W*L,768] → [B,W*L,256]
```

`token_norm` runs first (normalizes 768-dim vectors), then `token_proj` maps 768→256. The result is two tensors in the same 256-dim space, with comparable magnitudes.

### Step 2 — Pad node embeddings to uniform dense batch

```python
B = token_embs.shape[0]
padded_nodes, node_real_mask = _scatter_to_dense(
    nodes_proj, batch, num_graphs=B, max_nodes=self.max_nodes
)
# padded_nodes:   [B, 1024, 256]  — zero-padded at trailing positions
# node_real_mask: [B, 1024]       — True = real node, False = padding
```

`nodes_proj` is `[N, 256]` — a flat list of all node embeddings from all B graphs. `batch` maps each node to its graph index. `_scatter_to_dense` reorganizes this into a padded `[B, 1024, 256]` tensor so attention can operate on the full batch at once.

**Mask convention inversion:**

```python
node_padding_mask  = ~node_real_mask           # [B, 1024]: True=IGNORE (padding)
token_padding_mask = (attention_mask == 0)     # [B, W*L]:  True=IGNORE (PAD token)
```

PyTorch's `MultiheadAttention` uses `key_padding_mask` where `True` means *ignore this position* — the opposite of the attention mask convention (1=real, 0=pad). Both masks are inverted to match PyTorch's convention.

### Step 3 — Node → Token cross-attention (Fix #26, Fix #8)

```python
# Q=padded_nodes [B,1024,256]  K=V=tokens_proj [B,W*L,256]
enriched_nodes, _ = self.node_to_token(
    query=padded_nodes,
    key=tokens_proj,
    value=tokens_proj,
    key_padding_mask=token_padding_mask,   # mask PAD tokens
    need_weights=False,                    # Fix #26
)
# enriched_nodes: [B, 1024, 256]

# Fix #8: zero-out padded node positions
enriched_nodes = enriched_nodes * node_real_mask.float().unsqueeze(-1)
```

> **[LEARNING MODE: Master the detail]** — The query/key/value assignments. Know which direction this is and what it produces.

**Q=nodes, K=V=tokens:** each of the 1024 node positions (real or padded) *queries* the full token sequence. The output at each node position is a weighted sum of token embeddings — enriched by whichever tokens the node "found most relevant."

After this: `withdraw()` node embedding now contains a mixture of `"call.value"`, `"transfer"`, `"balances"` token embeddings, weighted by relevance.

**Fix #26 — `need_weights=False`:**

By default, PyTorch MHA materializes the full attention weight matrix:
- `node_to_token`: `[B, 1024, W*L]` = `[8, 1024, 2048]` ≈ 128M values ≈ **512 MB** in float32

This is allocated and computed every forward pass but never read (the `_` discard). `need_weights=False` tells PyTorch:
1. Skip materializing the weight matrix
2. Use the fused efficient-attention CUDA kernel instead

Result: ~512 MB VRAM saved per forward, faster execution. The output `enriched_nodes` is mathematically identical.

> ⚠️ **CRITICAL** — `need_weights=False` is not just an optimization — it enables the flash-attention-style fused kernel inside MHA, which is faster and more memory-efficient. This matters for training throughput. Always set `need_weights=False` when you don't need the weight matrix.

**Fix #8 — zero-out padded positions:**

```python
enriched_nodes = enriched_nodes * node_real_mask.float().unsqueeze(-1)
# node_real_mask: [B, 1024]  → unsqueeze(-1) → [B, 1024, 1]
# broadcast over D=256: zeros out padded slots
```

Padded node positions (initially zero) went through attention. The softmax over token keys assigned nonzero weights to them — padded positions received a nonzero mixture of token embeddings.

The downstream pooling in Step 5 already uses `node_real_mask` to exclude padded positions. So the pooled output would be correct even without Fix #8. But:

> **[AUDIT A7]** — Fix #8 is a defensive structural invariant. Without it, `enriched_nodes` contains nonzero values at padded positions — a tensor state that could silently corrupt any future refactor that skips the mask. A reader might reasonably assume "padded positions in `enriched_nodes` are zero" and skip the mask. With Fix #8, that assumption is always true in the actual tensor. This is good defensive coding — make invariants structural, not documented.

### Step 4 — Token → Node cross-attention (Fix #26)

```python
# Q=tokens_proj [B,W*L,256]  K=V=padded_nodes [B,1024,256]
enriched_tokens, _ = self.token_to_node(
    query=tokens_proj,
    key=padded_nodes,
    value=padded_nodes,
    key_padding_mask=node_padding_mask,    # mask padded node positions
    need_weights=False,                    # Fix #26
)
# enriched_tokens: [B, W*L, 256]
```

**Q=tokens, K=V=nodes:** each of the W*L token positions queries the full node set. The output at each token position is a mixture of node embeddings — enriched by whichever nodes "explain" that token.

After this: the `"call.value"` token embedding contains a mixture of `withdraw()`, `balances`, and other node embeddings — whatever the model learned are structurally relevant to that token.

**`key_padding_mask=node_padding_mask`:** prevents tokens from attending to zero-padded node positions. Without this, tokens would spread attention weight across real nodes *and* meaningless zero padding, diluting the structural signal.

### Step 5 — Masked mean pooling (Fix #6)

```python
# Pool enriched nodes — real nodes only
node_weight  = node_real_mask.float().unsqueeze(-1)      # [B, 1024, 1]
node_sum     = (enriched_nodes * node_weight).sum(dim=1) # [B, 256]
node_count   = node_weight.sum(dim=1).clamp(min=1.0)     # [B, 1]
pooled_nodes = node_sum / node_count                     # [B, 256]

# Fix #6: pool enriched tokens — real tokens only (mask PAD positions)
token_weight  = attention_mask.float().unsqueeze(-1)      # [B, W*L, 1]
token_sum     = (enriched_tokens * token_weight).sum(dim=1)
token_count   = token_weight.sum(dim=1).clamp(min=1.0)
pooled_tokens = token_sum / token_count                   # [B, 256]
```

> **[LEARNING MODE: Master the detail]** — Masked mean pooling vs plain mean. Fix #6 is a correctness fix.

**Fix #6 — why token pooling must be masked:**

Before Fix #6, token pooling used plain mean: `enriched_tokens.mean(dim=1)`. This included PAD token positions (zeros in `attention_mask`) in the average. PAD tokens carry no code content — they were zero-padded to reach sequence length 512 (or W*512). Including them dilutes the mean with empty content.

For a 100-token contract in a 512-length window: 412 PAD tokens would contribute to the mean, diluting the 100 real token embeddings by 80%.

`attention_mask.float()` weights real tokens with 1.0 and PAD tokens with 0.0. The masked sum divided by the count of real tokens gives a true mean over real content only.

**`.clamp(min=1.0)` on count:** prevents division by zero for contracts where all tokens are padding (edge case: an empty contract graph). Returns a zero vector rather than NaN.

### Step 6 — Concatenate and project

```python
fused  = torch.cat([pooled_nodes, pooled_tokens], dim=1)  # [B, 512]
output = self.output_proj(fused)                           # [B, 128]
return output
```

`output_proj = Linear(512→128) → ReLU → Dropout`

The final `[B, 128]` is the fused eye — one of three 128-dim opinions that `SentinelModel` concatenates for the main classifier.

> ⚠️ **CRITICAL** — `output_dim=128` is the LOCKED dimension. `SentinelModel`'s classifier assumes `Linear(3 * 128, 192)` — i.e., all three eyes are 128-dim. Changing `output_dim` here without updating `SentinelModel` will silently produce wrong classifier input sizes, likely caught only when the `Linear` layer tries to multiply mismatched matrices at runtime.

---

## 5. Complete Data Flow

```
Inputs:
  node_embs [N, 256]      (GNNEncoder output — flat, all graphs)
  batch     [N]           (node-to-graph index)
  token_embs [B, W*L, 768] (TransformerEncoder output)
  attention_mask [B, W*L]  (1=real token, 0=PAD)

Step 1: Project
  nodes_proj  = node_proj(node_embs)                     [N, 256]
  tokens_proj = token_proj(token_norm(token_embs))       [B, W*L, 256]

Step 2: Pad nodes
  padded_nodes, node_real_mask = _scatter_to_dense(...)  [B, 1024, 256]
  node_padding_mask  = ~node_real_mask                   [B, 1024]  True=ignore
  token_padding_mask = (attention_mask == 0)             [B, W*L]   True=ignore

Step 3: Node → Token attention
  enriched_nodes = MHA(Q=padded_nodes, K=V=tokens_proj,
                       key_padding_mask=token_padding_mask,
                       need_weights=False)                [B, 1024, 256]
  enriched_nodes *= node_real_mask (Fix #8)

Step 4: Token → Node attention
  enriched_tokens = MHA(Q=tokens_proj, K=V=padded_nodes,
                        key_padding_mask=node_padding_mask,
                        need_weights=False)               [B, W*L, 256]

Step 5: Masked mean pool
  pooled_nodes  = masked_mean(enriched_nodes, node_real_mask) [B, 256]
  pooled_tokens = masked_mean(enriched_tokens, attention_mask) [B, 256]

Step 6: Concatenate + project
  fused  = cat([pooled_nodes, pooled_tokens])            [B, 512]
  output = output_proj(fused)                            [B, 128]  ← LOCKED
```

---

## 6. Cross-File Relationships

**Already taught — recall these connections:**

- `node_embs [N, 256]` comes from `GNNEncoder.forward()` (chunks 01–03). The GNN produces unnormalized node embeddings; `CrossAttentionFusion` receives them after `SentinelModel` routes them here.
- `token_embs [B, W*L, 768]` comes from `TransformerEncoder.forward()` (chunks 04–05). The attention mask from the sliding window tokenizer accompanies it.
- `register_buffer` (chunk 03) — same pattern for `_c4_truncation_warned` acting as persistent module-level state.

**Not yet taught — preview:**

- `SentinelModel.forward()` (chunk 08) calls `self.fusion(node_embs, batch, token_embs, flat_mask)`. It passes `flat_mask` — the multi-window mask flattened from `[B, W, L]` to `[B, W*L]` — so that CrossAttentionFusion's token pooling correctly masks padding windows, not just padding tokens within windows.

---

## 7. Alternative Approach — P7

> **[LEARNING MODE: Understand the pattern]** — This comparison is directly interview-relevant for "how do you fuse two modalities?"

**Option A — Early pooling + MLP (what was replaced):**
```
Pool GNN → [B, 64]
Pool TF  → [B, 768]
cat → MLP → [B, 64]
```
Pros: simple, fast. Cons: interaction only at the summary level — fine-grained node-token relationships lost.

**Option B — CrossAttentionFusion (SENTINEL's approach):**
```
All nodes query all tokens → enriched nodes
All tokens query all nodes → enriched tokens
Pool → compress
```
Pros: captures fine-grained structural-semantic interactions. Cons: O(N × W*L) attention — quadratic in number of nodes and tokens.

**Option C — Concatenate all node and token embeddings as one long sequence and apply self-attention:**
Pros: fully bidirectional in one pass. Cons: N nodes + W*L tokens = potentially 1024+2048 = 3072 positions — large; also treats node embeddings and token embeddings uniformly, losing the modality distinction.

**Option D — Gated fusion:**
```
gate = sigmoid(Linear(cat([pooled_gnn, pooled_tf])))
fused = gate * pooled_gnn + (1-gate) * pooled_tf
```
Pros: adaptive weighting between modalities; very fast. Cons: still early pooling — same fine-grained loss as Option A.

SENTINEL chose Option B because the vulnerability patterns (reentrancy's call-before-write) require `withdraw()` to find `"call.value"` specifically, not just know that there are calls somewhere.

---

## 3 Things to Lock In

1. **`_scatter_to_dense` uses static `max_nodes=1024` to avoid a `torch.compile` graph break.** The BUG-C2 fix: compute `valid = local_idx < max_nodes` *before* clamping — otherwise excess nodes clamp to position 1023 and overwrite each other with a random embedding that gets marked as real.

2. **`need_weights=False` on both MHA calls saves ~512 MB VRAM per forward pass** by skipping materialization of the `[B, 1024, W*L]` attention weight matrix and enabling the fused efficient-attention kernel.

3. **Fix #6: token pooling must be masked mean, not plain mean.** PAD tokens contribute nothing but dilute the average. A contract with 100 real tokens in a 512 window would have its embeddings diluted by 80% with plain mean.

---

## Challenge Questions

1. `_scatter_to_dense` computes `local_idx = arange(N) - offsets[batch]`. Walk through what `offsets` contains for a batch with 3 graphs of sizes [5, 3, 7], and what `local_idx` produces for the first node of each graph.

2. Why does `CrossAttentionFusion` use `key_padding_mask` (not `attention_mask`) in PyTorch's MHA, and why does it need to invert the mask (`~node_real_mask`)?

3. Without Fix #8, `enriched_nodes` has nonzero values at padded positions after node→token attention. The downstream masked mean pooling already excludes those positions via `node_real_mask`. So is Fix #8 strictly necessary for correctness? Justify your answer.

4. `output_dim=128` is described as "LOCKED". What would break at runtime if `CrossAttentionFusion` was instantiated with `output_dim=256` without any other changes?

---

**Next:** `07_sentinel_model_architecture.md` — module-level constants, `SentinelModel.__init__`, the three-eye structure, and auxiliary heads.
