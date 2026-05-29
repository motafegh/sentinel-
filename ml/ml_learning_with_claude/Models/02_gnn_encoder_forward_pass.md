# Models — GNN Encoder Chunk 2: The Three-Phase Forward Pass

> **File:** `ml/src/models/gnn_encoder.py` — **lines 338–581**
> **What you'll learn:** How the 8 GATConv layers actually execute — guards, edge mask construction, IMP-G1 layer-specific subsets, REVERSE_CONTAINS synthesis at runtime, the `_live` vs `_intermediates` distinction, and how all three phases chain together.
> **Time:** ~35 minutes
> **Interview relevance:** ML (GNN execution, directed vs undirected graphs), AI (attention on graphs), MLOps (defensive coding, diagnostic modes)

---

## 1. Before Any Computation: The Guards (lines 364–393)

```python
def forward(self, x, edge_index, batch, edge_attr=None, return_intermediates=False):
    # Guard 1: schema version
    if x.shape[1] != NODE_FEATURE_DIM:
        raise ValueError(
            f"GNNEncoder expects {NODE_FEATURE_DIM}-dim node features (schema v8) "
            f"but got {x.shape[1]}. Likely a stale v6 .pt file — re-run reextract_graphs.py."
        )
    # Guard 2: edge_attr required when use_edge_attr=True
    if self.use_edge_attr and edge_attr is None:
        raise ValueError(
            "GNNEncoder was built with use_edge_attr=True but edge_attr=None was passed."
        )
    # Guard 3: OOB node index
    if edge_index.numel() > 0 and edge_index.max() >= x.shape[0]:
        raise ValueError(
            f"edge_index contains node index {edge_index.max().item()} "
            f"but x has only {x.shape[0]} nodes. Graph .pt file is corrupted."
        )
    # Dtype normalization
    _param_dtype = next(self.parameters()).dtype
    if x.dtype != _param_dtype:
        x = x.to(_param_dtype)
```

**Why each guard exists:**

- **Guard 1** — Schema version mismatch. `NODE_FEATURE_DIM` is imported from `graph_schema.py`. If you trained on v8 graphs (11 features) and accidentally load a v6 graph (fewer features), the shapes mismatch silently in the linear layers, producing garbage. The error message tells you exactly what happened and what to run.

- **Guard 2** — This one is subtle. Without edge attributes, `cfg_mask` becomes all-zeros (no edges match CONTROL_FLOW type IDs), so Phase 2 runs over an empty edge set and produces no updates. The GNN silently trains with Phase 2 disabled — no error, no warning if you don't guard. Fail loud.

- **Guard 3** — An OOB node index in `edge_index` causes two different failure modes: on CPU, PyTorch silently reads garbage memory; on CUDA, you get an illegal-memory-access error with a useless traceback pointing into CUDA internals. This guard catches corruption early with a clear message.

- **Dtype normalization** — BERT loads in BF16 and can change PyTorch's global default dtype via `torch.set_default_dtype()`. Tests or other code that creates tensors with `torch.randn(...)` after BERT loads would produce BF16 tensors. This guard normalizes `x` to match the model's parameter dtype (float32) before any computation.

---

## 2. Edge Embeddings + OOB Clamping — Fix C1/H9 (lines 394–414)

```python
e = None
if self.edge_embedding is not None and edge_attr is not None:
    if edge_attr.numel() > 0:
        _max_valid = self.edge_embedding.num_embeddings - 1  # 10
        _oob_mask = (edge_attr < 0) | (edge_attr > _max_valid)
        if _oob_mask.any():
            logger.warning(
                f"GNNEncoder: {_oob_mask.sum().item()} OOB edge_attr value(s) "
                f"clamped to [0, {_max_valid}] ..."
            )
            edge_attr = edge_attr.clamp(0, _max_valid)
    e = self.edge_embedding(edge_attr)   # [E, edge_emb_dim=64]
```

`nn.Embedding` is just a lookup table indexed by integer. If `edge_attr` contains a type ID outside `[0, 10]`, `nn.Embedding` raises an index-out-of-range error. On CUDA, this manifests as an illegal memory access — a CUDA error with a traceback pointing inside CUDA kernels, giving you no indication of which contract caused it or even that the issue was an edge type ID.

**The fix**: clamp before the lookup, warn once, continue. The contract gets a slightly wrong edge embedding (type 0 or 10 instead of the real type) but doesn't crash the entire training run. One bad contract in 44K is acceptable; one CUDA crash that stops training for hours is not.

> 🎯 **INTERVIEW FOCUS:** "How do you handle bad data in a production ML training pipeline?" — Clamp and warn at system boundaries. Fatal errors (architecture mismatches) raise immediately. Data quality issues (OOB values in one graph) log a warning and continue — they affect <1% of the corpus and shouldn't abort a multi-hour run.

---

## 3. Edge Mask Construction (lines 416–498)

This is where the three phases get their different "views" of the graph. The same `edge_index` and `edge_attr` are filtered into different subsets:

```python
_CONTAINS         = EDGE_TYPES["CONTAINS"]        # 5
_CONTROL_FLOW     = EDGE_TYPES["CONTROL_FLOW"]     # 6
_REVERSE_CONTAINS = EDGE_TYPES["REVERSE_CONTAINS"] # 7 (runtime-only — not on disk)
_CALL_ENTRY       = EDGE_TYPES["CALL_ENTRY"]       # 8
_RETURN_TO        = EDGE_TYPES["RETURN_TO"]        # 9
_DEF_USE          = EDGE_TYPES["DEF_USE"]          # 10

struct_mask   = edge_attr <= _CONTAINS          # types 0,1,2,3,4,5
cfg_mask      = (edge_attr == _CONTROL_FLOW) |
                (edge_attr == _CALL_ENTRY)   |
                (edge_attr == _RETURN_TO)    |
                (edge_attr == _DEF_USE)             # types 6,8,9,10
contains_mask = edge_attr == _CONTAINS             # type 5 only
```

**`struct_mask` (types 0–5):** CALLS, READS, WRITES, EMITS, INHERITS, CONTAINS. All the "static structure" edges — who calls whom, what variables are read/written, inheritance hierarchy. Used by Phase 1.

**`cfg_mask` (types 6,8,9,10):** The dynamic execution edges — CONTROL_FLOW, CALL_ENTRY, RETURN_TO, DEF_USE. These encode runtime behavior, not just structure. Used by Phase 2.

**`contains_mask` (type 5 only):** CONTAINS edges specifically — FUNCTION → CFG_NODE. Used to build Phase 3's reverse edges.

### 3a. `phase2_edge_types` Ablation Parameter (lines 428–431)

```python
if self.phase2_edge_types is not None:
    cfg_mask = torch.zeros(edge_attr.shape[0], dtype=torch.bool, ...)
    for _t in self.phase2_edge_types:
        cfg_mask |= (edge_attr == _t)
else:
    cfg_mask = (CF | CALL_ENTRY | RETURN_TO | DEF_USE)  # default
```

When `phase2_edge_types` is passed at construction time, the Phase 2 edge mask is overridden. This enables ablation experiments:
- `phase2_edge_types=[6, 8, 9]` → remove DEF_USE, test if data-flow edges help
- `phase2_edge_types=[6, 10]` → ICFG-only ablation, test cross-function edges
- `phase2_edge_types=[]` → disable Phase 2 entirely

The default (`None`) uses all four types. This parameter lets researchers test which edge types contribute to performance without changing model code.

---

## 4. IMP-G1: Layer-Specific Phase 2 Subsets (lines 461–476)

Before IMP-G1, all three Phase 2 layers used the same `cfg_mask`. The problem: if all layers see identical edges, they build nearly-identical representations. The JK mechanism gets three very similar phase outputs to aggregate — not three genuinely different views.

**After IMP-G1:**
```python
_cf_mask   = (edge_attr == _CONTROL_FLOW)              # type 6 only
_icfg_mask = (edge_attr == _CALL_ENTRY) | (edge_attr == _RETURN_TO)  # types 8,9

cf_only_ei   = edge_index[:, _cf_mask]    # intra-function edges only
icfg_only_ei = edge_index[:, _icfg_mask]  # cross-function edges only
# phase2_ei (joint) stays for Layer 5
```

| Layer | Edge set | What it encodes |
|-------|----------|----------------|
| Layer 3 (`conv3`) | CF only (type 6) | Intra-function execution order: A→B→C within one function |
| Layer 4 (`conv3b`) | ICFG only (types 8,9) | Cross-function call structure: callee entry + return paths |
| Layer 5 (`conv3c`) | CF + ICFG joint | Integration: nodes enriched by layers 3+4 now aggregate across both |

**Why this produces genuinely different representations:**

After Layer 3, CFG nodes carry information about their within-function neighbors (1-hop CF). After Layer 4, nodes carry information about their cross-function neighbors (1-hop ICFG). Layer 5 then propagates signals that cross both — a CFG_CALL node now knows what's 1 hop away via both intra- and cross-function paths.

Each layer's output represents a different scope of context. The JK attention then learns which scope is most relevant for each node type.

---

## 5. REVERSE_CONTAINS — Runtime Synthesis (lines 481–497)

```python
fwd_contains_ei = edge_index[:, contains_mask]      # [2, E_contains]  FUNCTION→CFG
rev_contains_ei = fwd_contains_ei.flip(0)            # [2, E_contains]  CFG→FUNCTION
```

**`.flip(0)` flips the row dimension**: swaps source and destination nodes. If `fwd_contains_ei` has rows `[src, dst]`, `flip(0)` produces `[dst, src]`. Every FUNCTION→CFG_NODE edge becomes a CFG_NODE→FUNCTION edge.

**These reverse edges are NOT stored on disk.** The graph `.pt` files only contain the forward CONTAINS edges (FUNCTION→CFG). The reverse edges are synthesized in the forward pass. This is intentional:
1. No disk overhead — storing both directions would double the edge count
2. The reverse direction is always derivable from the forward direction
3. It was added later (Phase 3 design) without requiring re-extraction of all graphs

**The type-7 embeddings for the reverse direction:**
```python
if n_rev > 0:
    rev_type_ids = torch.full((n_rev,), _REVERSE_CONTAINS, ...)  # all = type 7
    rev_contains_ea = self.edge_embedding(rev_type_ids)           # [E_contains, 64]
```

REVERSE_CONTAINS (type 7) is a distinct entry in the edge embedding table — the GNN uses a different 64-dim vector for upward CFG→FUNCTION edges vs downward FUNCTION→CFG edges. This lets the GNN learn different attention weights for the two directions. The forward CONTAINS edges (type 5) use `fwd_contains_ea` directly from the stored embeddings.

> 🎯 **INTERVIEW FOCUS:** "How do you add bidirectional edges to a GNN without storing them twice?" — Synthesize them at runtime with `.flip(0)`. Assign a distinct edge type ID (and thus distinct embedding) to the reverse direction so the model can distinguish upward from downward message passing.

---

## 6. `_live` vs `_intermediates` — Gradient Flow (lines 499–504)

```python
_live: list[torch.Tensor] = []     # gradient-attached — fed into JK
_intermediates: dict       = {}    # detached clones — diagnostic only

# After Phase 1:
_live.append(x)                              # x has requires_grad=True
_intermediates["after_phase1"] = x.detach().clone()   # detached copy
```

**`_live`** holds the actual computation graph nodes. When `self.jk(_live)` runs at the end, it computes attention weights and a weighted sum over these tensors. During backward pass, gradients flow through the JK attention parameters and back into all three phase outputs simultaneously. This is critical — JK weights train by comparing which phase representation leads to better predictions.

**`_intermediates`** holds `.detach().clone()` copies for diagnostic use. `.detach()` severs the autograd connection — these tensors have no `grad_fn`, so no gradients flow through them. They're only returned when `return_intermediates=True` (a diagnostic/analysis mode, never used in normal training). Keeping them detached means they don't affect the training computation at all.

**Why not just use `x.clone()` without detach?**

Without `.detach()`, the intermediates would hold additional references to the computation graph nodes — increasing memory usage and potentially causing confusion if someone accidentally tried to call `.backward()` through them.

---

## 7. Phase 1 Forward: IMP-G2 in Code (lines 506–526)

```python
x_init = x                                               # [N, 11] — save raw input

# Layer 1: 11→256 with IMP-G2 input skip
x_skip = self.input_proj(x_init.to(_proj_dtype)).to(x.dtype)  # [N, 11]→[N, 256]
x  = self.conv1(x_init, struct_ei, struct_ea)                  # [N, 11]→[N, 256]
x  = self.relu(x + x_skip)                                     # skip BEFORE relu
x  = self.dropout(x)

# Layer 2: 256→256 with residual
x2 = self.conv2(x, struct_ei, struct_ea)               # [N, 256]→[N, 256]
x2 = self.relu(x2)
x  = x + self.dropout(x2)                              # residual: add, not replace

# Phase 1-A2: LayerNorm before collecting for JK
x = self.phase_norm[0](x)
_live.append(x)
```

**The IMP-G2 skip in action:**

At initialization, `conv1`'s GAT attention weights are random. With 8 heads and `add_self_loops=True`, the attention over N neighbors tends to be approximately uniform (1/N per neighbor). For a node with 20 structural neighbors, Layer 1 output is roughly the mean of all 20 neighbors' features — very noisy.

The skip connection `x + x_skip` adds the linear projection of the raw 11-dim features directly. Even when attention is near-uniform and conv output is noisy, the skip ensures the raw features (type ID, visibility, complexity, etc.) are always present in the 256-dim space. The network can immediately use these for gradient updates, not having to wait until attention weights converge.

**Residual order matters:** `x = x + dropout(x2)`, not `x = relu(x + x2)`. Dropout is applied to the convolutional branch only — the identity path (`x`) is never dropped. This ensures gradient always flows through the identity shortcut regardless of dropout rate.

---

## 8. Phase 2 Forward: IMP-G1 in Code (lines 528–547)

```python
# Layer 3: CONTROL_FLOW only — intra-function execution order
x2 = self.conv3(x, cf_only_ei, cf_only_ea)
x2 = self.relu(x2);  x = x + self.dropout(x2)

# Layer 4: CALL_ENTRY + RETURN_TO only — cross-function structure
x2 = self.conv3b(x, icfg_only_ei, icfg_only_ea)
x2 = self.relu(x2);  x = x + self.dropout(x2)

# Layer 5: CF + ICFG joint — integration layer
x2 = self.conv3c(x, phase2_ei, phase2_ea)
x2 = self.relu(x2);  x = x + self.dropout(x2)

x = self.phase_norm[1](x)    # LayerNorm after all three Phase 2 layers
_live.append(x)
```

**What happens to a CFG_CALL node through Phase 2:**

After Layer 3 (CF only): the CFG_CALL node aggregates messages from its intra-function predecessors and successors. It now knows the local execution context — what statement runs before and after it.

After Layer 4 (ICFG only): the CFG_CALL node also receives messages via CALL_ENTRY edges from the function it calls. It now knows something about the callee's entry context.

After Layer 5 (joint): the node aggregates across both types simultaneously — using its by-now-enriched neighbors. The integration layer sees nodes that already contain 2-hop intra-function context plus 1-hop cross-function context.

**Why 1 head for Phase 2?**

Phase 2 edges are directed (CONTROL_FLOW goes A→B). Multiple attention heads on directed edges tend to converge to similar patterns — there's usually one dominant direction of information flow. A single head with `out_channels=hidden_dim=256` (full capacity) learns that direction more efficiently than 8 heads each with 32 dims.

---

## 9. Phase 3 Forward: Bidirectional CONTAINS + IMP-G3 (lines 549–568)

```python
# Layers 6+7: CFG→FUNCTION (upward — reverse CONTAINS)
x2 = self.conv4(x, rev_contains_ei, rev_contains_ea)    # Layer 6: up
x2 = self.relu(x2);  x = x + self.dropout(x2)
x2 = self.conv4b(x, rev_contains_ei, rev_contains_ea)   # Layer 7: up again
x2 = self.relu(x2);  x = x + self.dropout(x2)

# Layer 8: FUNCTION→CFG (downward — IMP-G3)
x2 = self.conv4c(x, fwd_contains_ei, fwd_contains_ea)
x2 = self.relu(x2);  x = x + self.dropout(x2)

x = self.phase_norm[2](x)
_live.append(x)
```

**The signal flow:**

After Phase 2, CFG nodes carry execution-order context. But FUNCTION nodes don't yet know about their children's execution patterns — they've only been updated by Phase 1 (structural) edges.

**Layer 6 (up)**: CFG nodes send their Phase-2-enriched embeddings upward via rev_contains to their parent FUNCTION node. After Layer 6, each FUNCTION node carries a 1-hop aggregation of all its CFG children's execution contexts.

**Layer 7 (up again)**: A second upward hop. A FUNCTION that calls another FUNCTION now receives messages through both: its own CFG children (Layer 6) AND the called function's node (via CALLS edge in Phase 1, then upward propagation). Multi-function vulnerability patterns can now emerge.

**Layer 8 (down — IMP-G3)**: FUNCTION nodes distribute their enriched embeddings back DOWN to all their CFG children. After this layer, every CFG node carries:
- Its own execution-order context (Phase 2)
- Its parent function's aggregated context (Phase 3 up)
- The parent's enriched context redistributed (Phase 3 down)

**Why IMP-G3 (downward) matters for CrossAttentionFusion:**

Without the downward pass, CFG nodes and FUNCTION nodes have very different "depths" after Phase 3 — FUNCTION nodes are at depth 3 (all phases touched them) while CFG nodes might be effectively at depth 2. When CrossAttentionFusion attends node→token and token→node, it sees CFG nodes and FUNCTION nodes with different representational richness. After IMP-G3, all nodes carry Phase 3 depth equally.

**Zero-message behavior:**

```python
# From the module docstring:
# "FUNCTION nodes with no CFG children receive no upward Phase 3 messages.
#  conv returns zero; residual x = x + dropout(0) is a no-op."
```

A FUNCTION node with no CONTAINS edges (interface-only, empty function) receives a zero message from `conv4`. The residual `x = x + dropout(0) = x + 0 = x` is a no-op — the node keeps its Phase 1+2 embedding unchanged. This is correct behavior, not a bug.

---

## 10. JK Aggregation and Final Return (lines 570–581)

```python
if self.use_jk and self.jk is not None:
    x, _jk_entropy = self.jk(_live)   # _live = [phase1_out, phase2_out, phase3_out]
else:
    _jk_entropy = x.new_zeros(1).squeeze()   # scalar 0.0

if return_intermediates:
    return x, batch, _jk_entropy, _intermediates
return x, batch, _jk_entropy   # ← default
```

**`_live` contains 3 tensors, each `[N, 256]`** — the LayerNorm outputs after Phase 1, 2, and 3. All three are gradient-attached. `self.jk(_live)` is `_JKAttention.forward(_live)` — it computes per-node attention weights over the 3 phases and returns the weighted sum (details in the next chunk).

**`use_jk=False` fallback:**

When JK is disabled, `x` at this point is the Phase 3 output (the last assignment). `_jk_entropy` is set to scalar 0.0. The model falls back to v5 behavior — only the final phase output is used. This exists for checkpoint compatibility with older saved models.

**`return_intermediates=True`:**

Returns the `_intermediates` dict with detached per-phase outputs (`"after_phase1"`, `"after_phase2"`, `"after_phase3"`). Used by diagnostic scripts (e.g., `jk_weight_hist.py`) to visualize how node embeddings evolve across phases. Never used during training — adds a small memory overhead for the `.clone()` operations.

---

## 11. The Full Forward Pass — Data Flow Summary

```
x: [N, 11]  edge_index: [2, E]  edge_attr: [E]  batch: [N]
  │
  ├── Guards: schema, edge_attr, OOB index, dtype
  ├── Edge embeddings: edge_attr → e [E, 64]  (+ OOB clamp)
  │
  ├── Edge masks:
  │     struct_mask  → struct_ei, struct_ea       (types 0–5)
  │     cf_only_mask → cf_only_ei, cf_only_ea     (type 6)
  │     icfg_mask    → icfg_only_ei, icfg_only_ea (types 8,9)
  │     cfg_mask     → phase2_ei, phase2_ea        (types 6,8,9,10)
  │     contains_mask→ fwd_contains_ei/ea          (type 5)
  │                  → rev_contains_ei (flip) + type-7 ea
  │
  ├── Phase 1 (struct_ei):
  │     Layer 1: conv1(x_init) + input_proj(x_init) skip → [N,256]
  │     Layer 2: conv2(x) + residual → [N,256]
  │     LayerNorm → phase1_out → _live[0]
  │
  ├── Phase 2 (directed CF/ICFG):
  │     Layer 3: conv3(x,  cf_only)  → residual
  │     Layer 4: conv3b(x, icfg_only)→ residual
  │     Layer 5: conv3c(x, phase2)   → residual
  │     LayerNorm → phase2_out → _live[1]
  │
  ├── Phase 3 (bidirectional CONTAINS):
  │     Layer 6: conv4(x,  rev_contains) up → residual
  │     Layer 7: conv4b(x, rev_contains) up → residual
  │     Layer 8: conv4c(x, fwd_contains) down → residual  (IMP-G3)
  │     LayerNorm → phase3_out → _live[2]
  │
  └── JK aggregation: _JKAttention(_live) → x [N,256], jk_entropy
        │
        └── return x [N,256], batch [N], jk_entropy (scalar)
```

---

## Interview Questions

1. **"How do you make a GNN process different edge types differently?"**
   → Edge type embeddings (a lookup table mapping type ID → vector) appended to GATConv's attention computation. Different edge types get different learned embedding vectors. You can also filter `edge_index` to separate subsets per phase/layer and run different GATConv layers on different subsets (IMP-G1).

2. **"How do you add reverse edges to a GNN without storing them on disk?"**
   → Synthesize at runtime with `.flip(0)` on the edge index. Assign a distinct edge type ID to the reverse direction so the learned embedding distinguishes up vs down traversal. The reverse edges are derived deterministically so there's no need to persist them.

3. **"When a GNN layer has no edges to process, what happens?"**
   → GATConv on an empty edge set returns zero for every node (no messages to aggregate). With residual connections (`x = x + dropout(zero) = x`), this is a no-op — the node keeps its current embedding. This is correct behavior for nodes with no edges in a particular phase.

4. **"What's the difference between diagnostic data and training data in a forward pass?"**
   → Diagnostic data: `.detach().clone()` — severs autograd, stores a snapshot, zero gradient overhead. Training data: gradient-attached tensors in `_live` — gradients flow through these during backward. Never mix them: accidentally including detached tensors in a loss computation silently kills gradients.

---

**Next:** `03_jk_attention_internals.md` — `_JKAttention` class, `register_buffer`, per-node weights, entropy regularizer, and diagnostic infrastructure.
