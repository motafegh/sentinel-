# Models — Chunk 3: CrossAttentionFusion & the SentinelModel Three-Eye Architecture

> **Files:** `ml/src/models/fusion_layer.py`, `ml/src/models/sentinel_model.py`
> **What you'll learn:** Bidirectional cross-attention fusion, why the old concat+MLP was replaced, `_scatter_to_dense` for `torch.compile`, the three-eye architecture, auxiliary heads, and how the whole model assembles.
> **Time:** ~35 minutes
> **Interview relevance:** AI (cross-attention, multi-modal fusion), ML (multi-label classification, auxiliary losses), MLOps (VRAM optimization, torch.compile)

---

## 1. Why Replace Concat+MLP with Cross-Attention?

The original FusionLayer was simple:
```python
# BEFORE:
concat([pooled_gnn: B,64], [pooled_tf: B,768]) → MLP → [B, 64]
```

**The problem:** both paths were already pooled into single vectors before fusion. The structural information (which function called which) and the semantic information (what the tokens say) were averaged away before they could interact.

**The insight:** `withdraw()` as a GNN node should be able to directly attend to `"call.value"` and `"transfer"` tokens **before** pooling. Only then does the reentrancy pattern emerge at fine granularity.

**CrossAttentionFusion:**
```
BEFORE: concat([B,64], [B,768]) → MLP → [B,64]
        Two summaries already pooled — detail gone.

AFTER:  Node embeddings [N,256] attend to token embeddings [B,512,768]
        Token embeddings [B,512,768] attend to node embeddings [N,256]
        Pool AFTER enrichment → [B,128]
```

Both directions of enrichment matter:
- **Node→Token**: `withdraw()` finds `"call.value"` in the token sequence
- **Token→Node**: `"call.value"` finds the `withdraw()` node that emits it
- Then pool the enriched representations

---

## 2. The Architecture in Full

```
Input:
  node_embs [N, 256]           — all GNN node embeddings across the batch
  batch     [N]                — node-to-graph index
  token_embs [B, W*L, 768]     — all token embeddings
  attention_mask [B, W*L]      — 1=real token, 0=PAD

Step 1: Project both to common attention space (attn_dim=256)
  nodes_proj  = node_proj(node_embs)                         [N, 256]
  tokens_proj = token_proj(token_norm(token_embs))           [B, W*L, 256]
    ↑ BUG-C2: LayerNorm before projection (details below)

Step 2: Pad nodes to uniform dense batch
  padded_nodes, node_real_mask = _scatter_to_dense(...)      [B, max_nodes, 256]

Step 3: Node → Token cross-attention
  enriched_nodes = node_to_token(Q=padded_nodes, K=V=tokens) [B, max_nodes, 256]
  enriched_nodes *= node_real_mask  (zero padded slots)

Step 4: Token → Node cross-attention
  enriched_tokens = token_to_node(Q=tokens, K=V=padded_nodes)[B, W*L, 256]

Step 5: Masked mean pooling
  pooled_nodes  = mean(enriched_nodes,  mask=node_real_mask) [B, 256]
  pooled_tokens = mean(enriched_tokens, mask=attention_mask)  [B, 256]

Step 6: Concatenate and project
  fused = cat([pooled_nodes, pooled_tokens])                  [B, 512]
  output = output_proj(fused)                                 [B, 128]  ← LOCKED DIM
```

---

## 3. BUG-C2: Why `token_norm` Exists

```python
self.token_norm = nn.LayerNorm(token_dim)   # token_dim=768

# In forward:
tokens_proj = self.token_proj(self.token_norm(token_embs))
```

CodeBERT hidden states have L2 norm ~10–15. The GNN output after its own LayerNorm has norm ~1. Without normalization:
- In the node→token attention, dot products `Q·K^T` are computed as `[N,256] × [512,256]^T`
- Token keys are 10–15× larger in magnitude than node queries
- **Result**: attention weights are determined by token L2 norm, not semantic relevance
- Every node attends to the highest-norm tokens regardless of content

`token_norm` normalizes token embeddings to mean=0, std=1 before projection. Now the attention dot products measure semantic similarity, not magnitude.

> 🎯 **INTERVIEW FOCUS:** "Why would you add LayerNorm before a cross-attention projection?" — To equalize the magnitude of embeddings from different models/sources. When one encoder (BERT) produces embeddings with norm ~12 and another (GNN) produces norm ~1, the larger-norm embeddings dominate attention dot products, defeating the purpose of learned attention.

---

## 4. `_scatter_to_dense` — `torch.compile` Compatibility

PyG's `to_dense_batch()` uses a data-dependent `repeat()` operation whose size depends on the actual number of nodes per graph at runtime. This causes a **graph break** in `torch.compile`:

```python
# to_dense_batch does something like:
# max_n = counts.max()   ← data-dependent value
# out = x.new_zeros(B, max_n, D)  ← tensor size depends on runtime data
# torch.compile sees: "I can't trace this shape statically" → GRAPH BREAK
```

A graph break forces the entire `CrossAttentionFusion` forward to run in eager (Python) mode, losing all compile optimization.

`_scatter_to_dense` replaces this with a **static** `max_nodes=1024` constant:

```python
def _scatter_to_dense(x, batch, num_graphs, max_nodes):
    # max_nodes is always 1024 — compile sees a constant, no graph break
    out  = x.new_zeros(num_graphs, max_nodes, D)   # static shape!
    mask = torch.zeros(num_graphs, max_nodes, ...)
    
    # Compute per-node local index within its graph
    local_idx = arange(N) - offsets[batch]
    
    # BUG-C2 fix: compute valid BEFORE clamping
    valid     = local_idx < max_nodes
    local_idx = local_idx.clamp(max=max_nodes - 1)
    
    out[batch[valid], local_idx[valid]] = x[valid]
    mask[batch[valid], local_idx[valid]] = True
    return out, mask
```

**The BUG-C2 subtlety in `_scatter_to_dense`:** excess nodes (those with `local_idx >= max_nodes`) must be identified BEFORE clamping. Without this:
1. Nodes with `local_idx=1025` get clamped to `1024`
2. Multiple excess nodes all write to position 1024 (last-write-wins = random embedding)
3. The mask at position 1024 would be set to True, passing a random embedding to attention

The fix: mark `valid = local_idx < max_nodes` first, then clamp only for the scatter index. Only `valid` nodes are scattered; excess nodes are silently dropped (<1% of corpus).

---

## 5. Fix #26: `need_weights=False`

```python
enriched_nodes, _ = self.node_to_token(
    query=padded_nodes,
    key=tokens_proj,
    value=tokens_proj,
    key_padding_mask=token_padding_mask,
    need_weights=False,   # ← Fix #26
)
```

PyTorch's `MultiheadAttention` by default computes **and materializes** the full attention weight matrix:
- `node_to_token` weight matrix: `[B, max_nodes, W*L]` = `[8, 1024, 512]` ≈ 4M floats ≈ 8 MB
- `token_to_node` weight matrix: `[B, W*L, max_nodes]` = same size ≈ 8 MB
- **Total per forward pass: ~16 MB VRAM just for attention weights nobody reads**

`need_weights=False` tells PyTorch:
1. Skip materializing the weight matrix
2. Use the fused efficient-attention CUDA kernel (Flash Attention internally)
3. Result: same enriched output, ~16 MB VRAM saved, faster execution

This is a pure optimization — the attention computation itself (enriched outputs) is identical.

---

## 6. Fix #8: Zero Out Padded Node Positions

```python
enriched_nodes = enriched_nodes * node_real_mask.float().unsqueeze(-1)
```

After `node_to_token` attention, even padded node positions (initially zero) have received enrichment values from attending to tokens. The padded positions got nonzero values from the softmax-weighted token aggregation.

The mask pooling in Step 5 already excludes padded positions:
```python
node_weight = node_real_mask.float().unsqueeze(-1)  # 0.0 for padded
pooled_nodes = (enriched_nodes * node_weight).sum(1) / node_weight.sum(1)
```

So pooling is correct with or without Fix #8. But without Fix #8, the padded positions in `enriched_nodes` contain nonzero values — any future refactor that naively iterates over all positions would pick them up. Fix #8 makes the invariant structural: **padded positions are always zero in `enriched_nodes`**.

> 🎯 **INTERVIEW FOCUS:** This is a great example of defensive programming: make invariants explicit in the tensor state rather than relying on downstream code to always apply the correct mask.

---

## 7. The Three-Eye Architecture

```
                    ┌─────────────┐
                    │  GNN Path   │
                    │ node_embs   │
                    │   [N, 256]  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────────────┐
              │            │                    │
         GNN Eye      Transformer Eye       Fused Eye
              │            │                    │
    func_nodes only    WindowPooler       CrossAttentionFusion
    max+mean pool      CLS tokens         node↔token attention
         ↓                 ↓                    ↓
    gnn_eye_proj    transformer_eye_proj    (direct output)
    [B, 128]          [B, 128]              [B, 128]
              │            │                    │
              └────────────┴────────────────────┘
                           │
                    cat [B, 384]
                           │
                Linear(384, 192) → ReLU → Dropout
                           │
                  Linear(192, 10)
                           │
                  logits [B, 10]  (no Sigmoid!)
```

Each eye produces an independent `[B, 128]` opinion about the contract:
- **GNN eye**: "What do the structural patterns say?"
- **Transformer eye**: "What does the source text say?"
- **Fused eye**: "What emerges when both views interact at fine granularity?"

The main classifier learns how to weight and combine these three opinions.

---

## 8. GNN Eye — Why Function Nodes Only?

```python
_FUNC_TYPE_IDS = frozenset({
    NODE_TYPES["FUNCTION"],
    NODE_TYPES["MODIFIER"],
    NODE_TYPES["FALLBACK"],
    NODE_TYPES["RECEIVE"],
    NODE_TYPES["CONSTRUCTOR"],
})

func_mask = torch.isin(node_type_ids, _FUNC_IDS_CPU.to(device))
pool_embs  = node_embs[func_mask]
pool_batch = batch[func_mask]

gnn_max  = global_max_pool(pool_embs, pool_batch, size=num_graphs)
gnn_mean = global_mean_pool(pool_embs, pool_batch, size=num_graphs)
gnn_eye  = self.gnn_eye_proj(torch.cat([gnn_max, gnn_mean], dim=1))
```

**Why not pool all nodes?**

CFG_RETURN nodes represent statement endings and function returns. In a typical Solidity contract graph:
- CFG_RETURN: ~77% of all CFG nodes (median 93%)

If you pool all nodes, CFG_RETURN embeddings dominate the mean. The CFG_CALL and CFG_WRITE nodes that encode the critical reentrancy pattern (external call before state update) get drowned out.

**Why function-level nodes specifically?**

After Phase 3 (reverse-CONTAINS propagation upward), FUNCTION/MODIFIER nodes carry **aggregated CFG signal** — their embeddings already contain information about all the CFG nodes within them. Pooling only function-level nodes gives you:
- The summary of each function's internal control flow (from Phase 3)
- Without the noise of individual CFG nodes

**Max + Mean pooling (not just mean):**
```python
gnn_eye = gnn_eye_proj(cat([global_max_pool, global_mean_pool], dim=1))
```
- `global_mean_pool`: average behavior across all functions
- `global_max_pool`: captures the most extreme signal (the riskiest function)

For vulnerability detection, the most vulnerable function's signal should dominate. Max pooling finds it; mean pooling gives context.

---

## 9. Auxiliary Heads — Preventing Eye Dominance

```python
# Training only:
aux_gnn   = self.aux_gnn(gnn_eye)          # [B, 10]
aux_tf    = self.aux_transformer(tf_eye)   # [B, 10]
aux_fused = self.aux_fused(fused_eye)      # [B, 10]

# Trainer loss:
total_loss = main_loss + λ * (loss_gnn + loss_transformer + loss_fused)
# λ = 0.3
```

**The problem they solve:** The main classifier (`Linear(384→192→10)`) can learn to ignore one eye entirely — set its input weights to near-zero for `gnn_eye`, for instance, and route all gradient through `transformer_eye`. After epoch 43, the GNN eye's gradient share had collapsed to ~7% without auxiliary heads.

**How they fix it:** Each auxiliary head creates an independent loss gradient that flows directly into its eye's projection — bypassing the main classifier entirely. Even if the main classifier ignores the GNN eye, `loss_gnn` ensures GNN eye gradients stay alive.

`λ=0.3` means each auxiliary loss contributes 30% of the main loss. This is enough to keep each eye "alive" without overpowering the main task signal.

**At inference:** `return_aux=False` (default) — the three auxiliary heads never run. Zero inference overhead. The `if not return_aux: return logits` branch exits before any auxiliary computation.

> 🎯 **INTERVIEW FOCUS:** "How do you prevent gradient imbalance in multi-branch models?" — Auxiliary heads with per-branch independent losses. Each branch produces its own logits and loss, ensuring gradient flows to all branches even if the main classifier learns to favor one.

---

## 10. `select_prefix_nodes` — Priority + IMP-M1

```python
_PREFIX_NODE_PRIORITY = {
    NODE_TYPES["CONSTRUCTOR"]: 0,   # always include first
    NODE_TYPES["FALLBACK"]:    1,   # reentrancy-critical
    NODE_TYPES["RECEIVE"]:     2,   # reentrancy-critical
    NODE_TYPES["MODIFIER"]:    3,   # access control
    NODE_TYPES["FUNCTION"]:    4,   # general (selected last)
}
```

When K=48 prefix slots need to be filled from potentially many eligible nodes:

**Primary sort**: type priority (CONSTRUCTOR first, FUNCTION last)
**Secondary sort (IMP-M1)**: for FUNCTION nodes, sort by `−external_call_count` — functions with more external calls are selected first (more likely to be vulnerability entry points)

```python
sort_keys = [(priority, -ext_call_count if FUNCTION else 0.0, local_idx), ...]
sort_keys.sort()  # Python stable tuple sort
selected_local = [sk[2] for sk in sort_keys[:K]]
```

After projection:
```python
proj = self.gnn_to_bert_proj(g_embs[selected])     # [n_sel, 768]
# Add type-specific embedding so transformer knows node roles
proj += self.prefix_type_embedding(type_indices)    # [n_sel, 768]
```

The `prefix_type_embedding` is a 5-row lookup table (one row per declaration node type). It adds a learned bias vector to each projected prefix embedding — the Transformer can use this signal to distinguish "this prefix token came from a FALLBACK node" vs "this came from a FUNCTION node."

---

## 11. The Complete Forward Pass

```python
def forward(self, graphs, input_ids, attention_mask, return_aux=False):
    # 1. Flatten multi-window mask
    if input_ids.dim() == 3:
        flat_mask = attention_mask.view(B, W*L)   # [B, W*L]
    
    # 2. GNN path
    node_embs, batch, jk_entropy = self.gnn(graphs.x, graphs.edge_index, graphs.batch, edge_attr)
    # node_embs: [N, 256]

    # 3. GNN eye (function nodes only)
    node_type_ids = (graphs.x[:,0].float() * _MAX_TYPE_ID).round().long()
    func_mask = torch.isin(node_type_ids, _FUNC_IDS_CPU.to(device))
    gnn_eye = gnn_eye_proj(cat([max_pool(node_embs[func_mask]), mean_pool(...)]))
    # gnn_eye: [B, 128]

    # 4. GNN prefix selection (after warmup)
    if gnn_prefix_k > 0 and current_epoch >= warmup:
        gnn_prefix, gnn_prefix_counts = select_prefix_nodes(...)
    
    # 5. Transformer path
    token_embs = self.transformer(input_ids, attention_mask, gnn_prefix, ...)
    # token_embs: [B, W*L, 768]

    # 6. Transformer eye
    transformer_eye = transformer_eye_proj(window_pooler(token_embs))
    # transformer_eye: [B, 128]

    # 7. Fused eye
    fused_eye = self.fusion(node_embs, batch, token_embs, flat_mask)
    # fused_eye: [B, 128]

    # 8. Main classifier
    combined = cat([gnn_eye, transformer_eye, fused_eye])   # [B, 384]
    logits = self.classifier(combined)                       # [B, 10]

    if not return_aux:
        return logits
    
    # 9. Auxiliary heads (training only)
    aux = {"gnn": aux_gnn(gnn_eye), "transformer": ..., "fused": ..., "jk_entropy": ...}
    return logits, aux
```

**Two important details:**

**No Sigmoid in the model:**
```python
return logits  # raw logits, NOT sigmoid(logits)
```
Sigmoid is applied externally — in the loss function (`BCEWithLogitsLoss` is numerically more stable than `BCE(sigmoid(x))`) and in the predictor (where thresholds are applied).

**`_MAX_TYPE_ID` recovery:**
```python
node_type_ids = (graphs.x[:, 0].float() * _MAX_TYPE_ID).round().long()
```
Node feature[0] was stored as `type_id / 12.0` (normalized to [0,1]) by the graph extractor. This line recovers the integer type_id. The `.float()` guard prevents BF16 rounding errors, and `.round()` handles floating-point imprecision.

---

## 12. Parameter Count Summary

```
Sub-module               Trainable      Frozen
─────────────────────────────────────────────
GNNEncoder               ~2.4M          0
TransformerEncoder        590K          125M  (BERT frozen, LoRA trainable)
CrossAttentionFusion      ~600K          0
gnn_eye_proj               131K          0
transformer_eye_proj        98K          0
Classifier (3×128→10)      74K           0
Aux heads (3×Linear)       3.9K          0
─────────────────────────────────────────────
Total                    ~3.9M          125M
```

Only 3% of total parameters are trainable. The 125M frozen BERT parameters are the dominant mass but contribute zero gradient computation (beyond LoRA's 590K).

---

## 13. Summary — The Full SENTINEL Forward

```
Input batch:
  graphs: Batch (N nodes, E edges across B contracts)
  input_ids: [B, 4, 512]
  attention_mask: [B, 4, 512]
    ↓
GNNEncoder (3-phase, 8-layer GAT)
  → node_embs [N, 256]
    ↓ ┌──────────────────────────────────────────────┐
    │  GNN Eye           Transformer Eye   Fused Eye  │
    │  func nodes only   WindowPooler CLS  CrossAttn  │
    │  max+mean pool     learned weights   node↔tok   │
    │  [B,512]→[B,128]   [B,768]→[B,128]  [B,128]    │
    └──────────────────────────────────────────────┘
         ↓                    ↓                ↓
    cat([gnn_eye, transformer_eye, fused_eye]) → [B, 384]
         ↓
    Linear(384,192) → ReLU → Dropout → Linear(192,10)
         ↓
    logits [B, 10]  ← raw, no Sigmoid
```

---

## Interview Questions

1. **"What is cross-attention and how is it different from self-attention?"**
   → Self-attention: Q, K, V all come from the same sequence — each position attends to all others in the same sequence. Cross-attention: Q comes from one modality, K and V come from another — each position in sequence A attends to all positions in sequence B. Used here to let GNN nodes attend to code tokens and vice versa.

2. **"How do you fuse information from two different modalities (graph + text) in a neural network?"**
   → Option 1 (simple): pool each modality separately, concatenate, pass through MLP. Option 2 (SENTINEL): bidirectional cross-attention before pooling — each modality's elements directly attend to the other modality's elements. The second approach preserves fine-grained interactions lost by early pooling.

3. **"What is `torch.compile` and what causes a graph break?"**
   → `torch.compile` traces the computation graph and compiles it to optimized CUDA code. A "graph break" happens when the compiler encounters a data-dependent operation (tensor size determined by runtime values, Python control flow that depends on tensor contents). Fix: replace data-dependent shapes with static constants (e.g., `max_nodes=1024` instead of `counts.max()`).

4. **"Why would you use auxiliary losses in a multi-branch model?"**
   → Without auxiliary losses, one branch can "free-ride" — the main classifier learns to route gradient only through the most useful branch, starving the others. Auxiliary losses attach independent per-branch losses, ensuring all branches receive gradient regardless of main-classifier routing.

5. **"Why no Sigmoid inside the model?"**
   → `BCEWithLogitsLoss` combines sigmoid + BCE in a numerically stable way using the log-sum-exp trick. Applying sigmoid before the loss can cause underflow for very large or very small logits. Keeping raw logits also allows threshold tuning at inference without reprocessing.

---

**Next:** `04_training_loop_and_losses.md` — Trainer, gradient accumulation, BF16 AMP, AsymmetricLoss, and MLflow logging.
