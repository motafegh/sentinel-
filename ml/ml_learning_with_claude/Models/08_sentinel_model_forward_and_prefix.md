# Models — Chunk 8: SentinelModel `forward()` & Prefix Selection

> **File:** `ml/src/models/sentinel_model.py` — **lines 260–562**
> **Methods covered:** `select_prefix_nodes()`, `forward()`, `compute_prefix_attention_mean()`
> **Time:** ~35 minutes
> **Interview relevance:** ML (multi-label forward pass, auxiliary losses), AI (soft prompt injection, prefix tuning), MLOps (ghost graph handling, diagnostic hooks, warmup scheduling)

---

## Warm-Up Recall (from chunk 07)

One sentence each — no looking back:

1. Why are CONSTRUCTOR, FALLBACK, and RECEIVE ranked highest in `_PREFIX_NODE_PRIORITY`?
2. What is gradient starvation, and how do auxiliary heads prevent it?
3. `_MAX_TYPE_ID` is derived from `NODE_TYPES.values()` at import time — not hardcoded. What problem does this solve?

---

## 1. `select_prefix_nodes` — Picking the Right Nodes for BERT (lines 260–332)

> **[LEARNING MODE: Master the detail]** — This method embodies the bridge between the GNN world (integer node types, graph structure) and the Transformer world (continuous 768-dim vectors). Know the selection logic and why.

**Purpose:** After the GNN runs and produces `node_embs [N, 256]`, this method picks up to K=48 "representative" nodes per graph, projects them to 768-dim, and packages them as `[B, K, 768]` for injection into the Transformer's input.

```python
def select_prefix_nodes(self, node_embs, batch, node_type_ids, num_graphs):
    K = self.gnn_prefix_k                                   # 48
    prefix      = torch.zeros(num_graphs, K, 768, ...)      # [B, 48, 768]
    node_counts = torch.zeros(num_graphs, dtype=torch.long) # [B] — IMP-M3

    _EXT_CALL_DIM = 10   # feature[10] = external_call_count (log1p-normalised)
    _FUNCTION_ID  = NODE_TYPES["FUNCTION"]

    for g in range(num_graphs):
        g_mask  = batch == g
        g_types = node_type_ids[g_mask]
        g_embs  = node_embs[g_mask]

        eligible_local = [i for i, t in enumerate(g_types.tolist())
                          if t in _PREFIX_NODE_PRIORITY]
        if not eligible_local:
            continue     # no declaration nodes; prefix stays zero for this graph

        # Two-key sort: (type_priority, -ext_call_count_if_function, local_idx)
        sort_keys = []
        for local_idx in eligible_local:
            t    = g_types[local_idx].item()
            prio = _PREFIX_NODE_PRIORITY[t]
            sec  = -g_embs[local_idx, _EXT_CALL_DIM].item() if t == _FUNCTION_ID else 0.0
            sort_keys.append((prio, sec, local_idx))
        sort_keys.sort()
        selected_local = [sk[2] for sk in sort_keys[:K]]
        selected = torch.tensor(selected_local, ...)

        proj = self.gnn_to_bert_proj(g_embs[selected])     # [n_sel, 768]

        type_indices = torch.tensor(
            [_PREFIX_TYPE_IDX[g_types[i].item()] for i in selected.tolist()], ...
        )
        proj = proj + self.prefix_type_embedding(type_indices)   # [n_sel, 768]

        n_sel = proj.shape[0]
        prefix[g, :n_sel] = proj
        node_counts[g]    = n_sel

    return prefix, node_counts    # [B, K, 768], [B]
```

**The two-key sort — IMP-M1:**

When a contract has more than K=48 eligible nodes (many functions), some must be dropped. The sort determines which are kept:

1. **Primary key: `_PREFIX_NODE_PRIORITY[t]`** — CONSTRUCTOR(0) always comes before FALLBACK(1), RECEIVE(2), MODIFIER(3), FUNCTION(4). All CONSTRUCTOR nodes are selected before any FUNCTION nodes.

2. **Secondary key: `-ext_call_count` (FUNCTION nodes only)** — among FUNCTION nodes, those with more external calls are selected first. `external_call_count` is feature[10], stored log1p-normalized. More external calls → higher reentrancy risk → more important to include in the prefix.

```
Example: K=3, candidates are [FUNCTION(calls=5), CONSTRUCTOR, FUNCTION(calls=2)]
sort_keys: [(0, 0.0, idx_CTOR), (4, -5.0, idx_F5), (4, -2.0, idx_F2)]
After sort: [CONSTRUCTOR, FUNCTION(5 calls), FUNCTION(2 calls)]
Selected first 3: all of them
```

**`gnn_to_bert_proj` + `prefix_type_embedding`:**

```python
proj = self.gnn_to_bert_proj(g_embs[selected])   # Linear(256, 768)
proj = proj + self.prefix_type_embedding(type_indices)
```

The 256-dim GNN embedding is projected into 768-dim BERT space via a learned linear layer. Then a type-specific bias (`prefix_type_embedding`) is added. This bias is a 5-row embedding table — one row per declaration node type. Adding it lets the Transformer distinguish "this prefix position is a FALLBACK node" from "this prefix position is a regular FUNCTION node" via content, even though all prefix positions have the same position ID (1).

> **[AUDIT A10]** — This method uses a Python `for g in range(num_graphs)` loop. For a batch of B=8 graphs, this is 8 Python iterations, each doing Python list comprehensions over node lists. For large graphs (hundreds of functions), this could be slow. The method is called once per training step and once per inference — in practice the overhead is small (<5ms at B=8), but it does not scale to very large batches cleanly. A vectorized implementation using scatter operations would be more performant.

---

## 2. `forward()` — The Complete Three-Eye Pass (lines 334–488)

> **[LEARNING MODE: Master the detail]** — Walk through this as a sequence of transformations. Every shape annotation matters.

### Step 0 — Flatten the multi-window mask

```python
if input_ids.dim() == 3:
    B_tok, W, L = input_ids.shape
    flat_mask = attention_mask.view(B_tok, W * L)   # [B, W*L]
else:
    flat_mask = attention_mask                       # [B, L]
```

`CrossAttentionFusion` needs a flat token mask `[B, W*L]` to exclude PAD tokens from pooling. This line does the flattening before any forward pass runs — both the GNN and Transformer paths need to have completed before the fusion call, so the flat mask is computed upfront.

### Step 1 — GNN path

```python
edge_attr = getattr(graphs, "edge_attr", None) if self.use_edge_attr else None
node_embs, batch, _jk_entropy = self.gnn(
    graphs.x, graphs.edge_index, graphs.batch, edge_attr
)
# node_embs: [N, 256]   batch: [N]   _jk_entropy: scalar
```

`getattr(graphs, "edge_attr", None)` is defensive — some graph objects might not have `edge_attr`. In practice all v8 graphs have it, but this prevents an `AttributeError` crash on older checkpoints.

`_jk_entropy` is returned to the trainer (via `aux["jk_entropy"]` later) for the JK entropy regularizer.

### Step 2 — Recover node type IDs

```python
node_type_ids = (graphs.x[:, 0].float() * _MAX_TYPE_ID).round().long()
func_mask = torch.isin(node_type_ids, _FUNC_IDS_CPU.to(node_embs.device))
```

Feature column 0 (`graphs.x[:, 0]`) stores the normalized type ID. `* _MAX_TYPE_ID` and `.round()` recover the integer. `torch.isin` checks membership in `_FUNC_IDS_CPU` — returns `True` for FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes.

> ⚠️ **CRITICAL** — The `.float()` before multiplication is a guard against BF16 precision loss. If `graphs.x` arrives in BF16 (possible when BERT pollutes the default dtype), `x[:, 0] * 12.0` in BF16 might round `0.0833... × 12.0 = 0.999...` to `1.0` or `0.0` depending on the rounding mode — recovering the wrong type ID. `.float()` forces float32 for this computation only.

### Step 3 — Empty batch guard

```python
if batch.numel() == 0:
    B   = input_ids.size(0)
    dev = node_embs.device
    zeros = torch.zeros(B, self.num_classes, device=dev)
    if not return_aux:
        return zeros
    aux_zeros = {"gnn": ..., "transformer": ..., "fused": ..., }
    return zeros, aux_zeros
```

> **[LEARNING MODE: Understand the pattern]** — Why this guard exists and what it prevents.

An empty batch (`batch.numel() == 0`) means the DataLoader produced a batch with zero valid graphs — possible if all graphs in a batch failed schema validation or were filtered out. Without this guard:
- `batch.max()` raises `RuntimeError: max on empty tensor`
- The entire training step crashes, requiring a checkpoint reload

The guard returns correctly-shaped zero tensors instead. The trainer logs a warning and continues. Returning the correct type (`zeros` or `(zeros, aux_zeros)`) ensures the caller can unpack without a `ValueError`.

### Step 4 — GNN eye (function nodes only)

```python
num_graphs = int(batch.max().item()) + 1

pool_mask  = func_mask
pool_embs  = node_embs[pool_mask]      # [n_func, 256]
pool_batch = batch[pool_mask]          # [n_func]

gnn_max  = global_max_pool(pool_embs, pool_batch, size=num_graphs)   # [B, 256]
gnn_mean = global_mean_pool(pool_embs, pool_batch, size=num_graphs)  # [B, 256]
gnn_eye  = self.gnn_eye_proj(
    torch.cat([gnn_max, gnn_mean], dim=1)    # [B, 512]
)                                            # [B, 128]
```

**Why function nodes only — the key design decision:**

CFG_RETURN nodes make up ~77% of CFG nodes in a typical contract (median 93%). Pooling over all nodes gives you a global_max_pool that's dominated by CFG_RETURN embeddings — which encode "function exit" semantics, not vulnerability patterns. The CFG_CALL and CFG_WRITE nodes that encode the reentrancy pattern (call before write) get diluted.

After Phase 3's reverse-CONTAINS propagation, FUNCTION/MODIFIER nodes already carry their children's CFG signals aggregated upward. Pooling over them gives a function-level summary without the noise from individual CFG nodes.

**`size=num_graphs` parameter:**

`global_max_pool(pool_embs, pool_batch, size=num_graphs)` ensures the output always has B rows, even for graphs with no function nodes (ghost graphs — BUG-H2 fix). Without `size`, a ghost graph's batch index would be missing from `pool_batch`, and the pool output would have fewer than B rows — causing shape mismatches downstream.

For ghost graphs (no function nodes in `func_mask`), the `size` parameter ensures their row is filled with zeros. This is the correct degenerate behavior: an interface-only contract or empty graph produces a zero GNN eye, which the classifier treats as "no structural signal available."

### Step 5 — Prefix selection (warmup guard)

```python
gnn_prefix:       Optional[torch.Tensor] = None
gnn_prefix_counts: Optional[torch.Tensor] = None
if self.gnn_prefix_k > 0 and self._current_epoch >= self.gnn_prefix_warmup_epochs:
    gnn_prefix, gnn_prefix_counts = self.select_prefix_nodes(
        node_embs, batch, node_type_ids, num_graphs
    )
```

**The warmup guard:** During the first `gnn_prefix_warmup_epochs=15` epochs, `gnn_prefix` stays `None`. The Transformer runs on text alone. After epoch 15:
- The GNN has trained for 15 epochs → `node_embs` carry meaningful structural signals
- `gnn_to_bert_proj` (randomly initialized) can now learn to map these meaningful embeddings into BERT space, because the signals being projected are no longer random

If prefix were enabled from epoch 0, the random GNN embeddings injected as prefix tokens would add noise to every BERT forward pass, confusing the Transformer's text-based learning.

**At inference:**
```python
# In predictor.py:
model._current_epoch = 9999   # always above any warmup threshold
```
The prefix is always active at inference — the warmup condition is satisfied by setting the epoch counter to a large number.

### Step 6 — Transformer path

```python
token_embs = self.transformer(
    input_ids, attention_mask,
    gnn_prefix_nodes=gnn_prefix,
    gnn_prefix_counts=gnn_prefix_counts,
)
# token_embs: [B, W*L, 768] or [B, L, 768]
```

If `gnn_prefix` is `None` (warmup), this routes through the standard path in `TransformerEncoder.forward()`. If not None, it routes through the prefix injection path. The caller doesn't need to branch — `TransformerEncoder` handles both transparently.

### Step 7 — Transformer eye

```python
transformer_eye = self.transformer_eye_proj(
    self.window_pooler(token_embs)   # [B, 768]
)                                     # [B, 128]
```

`window_pooler` extracts CLS from each window and attention-weights them → `[B, 768]`. `transformer_eye_proj` compresses to `[B, 128]`.

### Step 8 — Fused eye

```python
fused_eye = self.fusion(node_embs, batch, token_embs, flat_mask)
# fused_eye: [B, 128]
```

`flat_mask` (computed in Step 0) is `[B, W*L]` — the flattened multi-window attention mask. This is passed as `CrossAttentionFusion`'s `attention_mask` so that PAD tokens in padding windows are excluded from token pooling.

### Step 9 — Main classifier

```python
combined = torch.cat([gnn_eye, transformer_eye, fused_eye], dim=1)  # [B, 384]
logits   = self.classifier(combined)                                  # [B, 10]

if self.num_classes == 1:
    logits = logits.squeeze(1)   # [B, 1] → [B] for binary BCEWithLogitsLoss

if not return_aux:
    return logits
```

`num_classes == 1` squeeze handles binary classification mode (single vulnerability class). `BCEWithLogitsLoss` expects `[B]` not `[B, 1]` for the binary case.

### Step 10 — Auxiliary heads (training only)

```python
aux_gnn   = self.aux_gnn(gnn_eye)               # [B, 10]
aux_tf    = self.aux_transformer(transformer_eye) # [B, 10]
aux_fused = self.aux_fused(fused_eye)            # [B, 10]

aux = {
    "gnn":         aux_gnn,
    "transformer": aux_tf,
    "fused":       aux_fused,
    "jk_entropy":  _jk_entropy,    # from GNN step — scalar
}
return logits, aux
```

Each auxiliary head is a single `Linear(128, 10)` applied directly to its eye's output — no shared layers with the main classifier. Trainer uses these to compute three additional BCE losses, each with weight λ=0.3.

`jk_entropy` is included in the aux dict so the trainer can add the JK entropy regularizer to the total loss without a separate model call.

---

## 3. Complete Forward Pass Data Flow

```
graphs (PyG Batch):  x[N,11]  edge_index[2,E]  edge_attr[E]  batch[N]
input_ids:           [B, W, L]
attention_mask:      [B, W, L]
return_aux:          bool
                         │
     ┌───────────────────┼────────────────────────┐
     │                   │                        │
  flat_mask          GNN path                     │
  [B, W*L]       node_embs [N,256]               │
                 batch [N]                        │
                 _jk_entropy                      │
                     │                            │
             recover node_type_ids                │
             func_mask (isin)                     │
                     │                            │
          ┌──────────┼─────────────┐              │
          │          │             │              │
    empty guard   GNN eye     prefix select       │
    → zero out    func nodes  (if warmup done)    │
                  max+mean                        │
                  gnn_eye_proj                    │
                  [B, 128]                        │
                               │                  │
                          gnn_prefix              │
                          [B, 48, 768]?           │
                               │                  │
                        Transformer path ─────────┘
                        token_embs [B, W*L, 768]
                               │
                    window_pooler → [B, 768]
                    transformer_eye_proj
                    [B, 128]
                               │
                    CrossAttentionFusion
                    (node_embs, batch, token_embs, flat_mask)
                    fused_eye [B, 128]
                               │
              cat([gnn_eye, tf_eye, fused_eye]) [B, 384]
                               │
                     classifier [B, 10] logits
                               │
                 return_aux? ──┴── True → also aux_gnn, aux_tf, aux_fused, jk_entropy
```

---

## 4. `compute_prefix_attention_mean` — IMP-M2 Diagnostic (lines 490–530)

```python
@torch.no_grad()
def compute_prefix_attention_mean(self, graphs, input_ids, attention_mask):
    if self.gnn_prefix_k == 0 or self._current_epoch < self.gnn_prefix_warmup_epochs:
        return None

    # ... run GNN, select prefix ...
    result = self.transformer(
        input_ids, attention_mask,
        gnn_prefix_nodes=gnn_prefix,
        output_attentions=True,     # ← expensive: materializes all attention matrices
    )
    if isinstance(result, tuple):
        _, prefix_attn_mean = result
        return prefix_attn_mean
    return None
```

> **[LEARNING MODE: Understand the pattern]** — A diagnostic hook that answers: "Is the Transformer actually using the GNN prefix, or ignoring it?"

**What it measures:** Mean attention weight from code token positions → prefix positions, averaged over all 12 layers, all 12 heads, and all sequences in the batch.

**Why `@torch.no_grad()`:** This is a diagnostic — no gradients needed. `no_grad()` here is correct (unlike the LoRA case) because this method never interacts with LoRA's gradient flow — it's called separately from training, not inside the training forward pass.

**When to call it:** Once per validation epoch (not per training step). It adds ~15% overhead because `output_attentions=True` forces materialization of all 12 × `[B, heads, L, L]` attention matrices.

**IMP-M2 gate signal:** If `prefix_attn_mean < 0.002` after 5+ epochs post-warmup, the Transformer is ignoring the prefix. Possible causes: `gnn_to_bert_proj` is not trained yet, prefix embeddings are too different from token embeddings in distribution, or the warmup was too short.

> **[AUDIT A11]** — The `isinstance(result, tuple)` check at line 519 is defensive coding for a subtle contract change. `TransformerEncoder.forward()` returns either `last_hidden_state` (a tensor) or `(last_hidden_state, prefix_attn_mean)` (a tuple) depending on `output_attentions`. The `isinstance` check handles both cases — but it also masks a potential API change: if `TransformerEncoder.forward()` ever returns a tuple for a *different* reason, this code would silently misinterpret the first element. A more robust design would use a `NamedTuple` or `dataclass` return type to make the tuple structure explicit.

---

## 5. Cross-File Relationships

**Already taught — recall these connections:**
- `_jk_entropy` (GNN chunk 03, section 5): returned by `GNNEncoder.forward()`, carried through to `aux["jk_entropy"]`, used by the trainer for entropy regularization.
- `CrossAttentionFusion.forward()` (chunk 06): `flat_mask` computed in Step 0 of this forward pass is what flows in as `attention_mask`. The flattening from `[B, W, L]` → `[B, W*L]` happens here, not in the fusion layer.
- `TransformerEncoder.forward()` (chunk 05): the `output_attentions=True` path and the `(last_hidden_state, prefix_attn_mean)` return — used here in `compute_prefix_attention_mean`.

**Not yet taught — preview:**
- `trainer.py` (Training module): sets `model._current_epoch = epoch` before each epoch's forward pass, calls `model(graphs, input_ids, attention_mask, return_aux=True)`, and receives `(logits, aux)`. Uses `aux["gnn"]`, `aux["transformer"]`, `aux["fused"]` for auxiliary losses, and `aux["jk_entropy"]` for the entropy regularizer.
- `predictor.py` (Inference module): sets `model._current_epoch = 9999` to force prefix active, calls `model(...)` with `return_aux=False` (inference default).

---

## 6. Alternative Approach — P7

**Multi-branch auxiliary losses vs. gradient clipping:**

Auxiliary heads are one solution to gradient imbalance. Alternatives:

**Option A — Gradient clipping per branch:** Clip gradients flowing into each branch separately, ensuring no branch receives disproportionately large or small updates. Requires custom backward hooks. More complex, less common.

**Option B — Gradient reversal layers (GRL):** Used in domain adaptation — flip the gradient sign to *prevent* a branch from learning something. Not applicable here (we want all branches to learn, not prevent learning).

**Option C — Deep supervision (cascaded auxiliary heads):** Add auxiliary predictions at multiple intermediate layers, not just the final representations. Common in image segmentation (U-Net). Would mean adding losses after each GNN phase, for example. More regularization but much more parameter overhead.

**Option D — Stop-gradient on dominant branch:** When one branch dominates, temporarily zero its gradient and let the others catch up. Requires dynamic gradient monitoring — complex.

SENTINEL's auxiliary head approach (Option A equivalent, simplified) is clean, simple, and adds negligible parameters. The key insight: the gradient pathways are independent from the very first layer.

---

## 3 Things to Lock In

1. **`select_prefix_nodes` uses a two-key Python sort:** primary = node type priority (CONSTRUCTOR first), secondary = `-external_call_count` for FUNCTION nodes. This encodes domain knowledge (more external calls = higher reentrancy risk) into the prefix selection.

2. **The warmup guard `_current_epoch >= gnn_prefix_warmup_epochs`** keeps the prefix off for 15 epochs, letting both GNN and Transformer stabilize before the cross-modal bridge is active. At inference, `_current_epoch = 9999` always passes this check.

3. **`global_max_pool(..., size=num_graphs)`** ensures ghost graphs (no function nodes) produce zero rows rather than dropping those batch indices. Without `size`, shape mismatches crash the classifier for any batch containing a ghost graph.

---

## Challenge Questions

1. In `select_prefix_nodes`, the secondary sort key for FUNCTION nodes is `-g_embs[local_idx, _EXT_CALL_DIM].item()`. Why negative? And why is this secondary key applied only to FUNCTION nodes, not MODIFIER or FALLBACK?

2. The forward pass computes `flat_mask = attention_mask.view(B, W*L)` before the GNN runs. Why must this be computed before the GNN, and why can't it be computed just before calling `self.fusion(...)`?

3. `global_max_pool(pool_embs, pool_batch, size=num_graphs)` — what does the `size` parameter do for ghost graphs (graphs with no function-level nodes)? What would happen without it?

4. `_current_epoch` is a plain Python integer attribute on `SentinelModel`, not a `register_buffer`. During DDP (Distributed Data Parallel) training, why does this not cause a problem, whereas a buffer tracking gradient statistics would need special handling?

---

**Models module complete.** ✅

The full Models folder now contains 8 chunks covering all four source files completely:

| Chunk | File | Lines |
|-------|------|-------|
| 01 | `gnn_encoder.py` — fundamentals | concepts |
| 02 | `gnn_encoder.py` — forward pass | 338–581 |
| 03 | `gnn_encoder.py` — JK attention internals | 76–131 |
| 04 | `transformer_encoder.py` — `__init__`, LoRA | 1–165 |
| 05 | `transformer_encoder.py` — `forward()`, pooler | 167–351 |
| 06 | `fusion_layer.py` — CrossAttentionFusion | 1–281 |
| 07 | `sentinel_model.py` — architecture & init | 1–259 |
| 08 | `sentinel_model.py` — forward & prefix | 260–562 |

**Before moving to Training**, confirm your understanding by answering (out loud):
- Can you trace a batch from raw graph + tokens to `[B, 10]` logits, naming the shape at each step?
- Can you explain LoRA, JK entropy, and prefix injection to a non-ML interviewer in plain terms?
- Can you name 3 AUDIT flags from this module and what better design they point toward?

**Next module:** `Training/01_trainer_overview_and_config.md` — `trainer.py` (1,633 lines), gradient accumulation, BF16 AMP, WeightedRandomSampler, and MLflow.
