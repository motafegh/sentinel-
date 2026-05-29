# Models — Chunk 1: GNN Fundamentals & Graph Attention Networks

> **File:** `ml/src/models/gnn_encoder.py`
> **What you'll learn:** Message passing from scratch, Graph Attention Networks (GAT), multi-head attention on graphs, edge embeddings, residual connections, LayerNorm, and the three-phase design rationale.
> **Time:** ~30 minutes
> **Interview relevance:** ML (GNNs — most asked AI/ML topic), AI (attention mechanisms)

---

## 1. Message Passing — The Core GNN Idea

All GNNs share one fundamental idea: **each node updates its representation by aggregating messages from its neighbors**.

After k rounds of message passing, each node's embedding contains information from all nodes within k hops.

```
Initial state:
  Node A: [feature_A]
  Node B: [feature_B]   ← neighbor of A
  Node C: [feature_C]   ← neighbor of A

After 1 round:
  Node A: aggregate([feature_A, feature_B, feature_C])
  
After 2 rounds:
  Node A: aggregate([embed_A_1hop, embed_B_1hop, embed_C_1hop])
  # A now knows about neighbors of B and C (2-hop neighborhood)
```

**In code (simplified):**
```python
# One message-passing step
for each node v:
    messages = [W * node_features[u] for u in neighbors(v)]
    new_features[v] = aggregate(messages) + W_self * node_features[v]
```

**For vulnerability detection:** After 8 rounds, a FUNCTION node's embedding contains information about all the CFG nodes it contains, all the state variables it reads/writes, and (via ICFG edges) the functions it calls. The model learns that certain patterns of neighbor combinations = vulnerability.

---

## 2. Why Attention? The Problem with Mean Aggregation

Simple mean aggregation treats all neighbors equally. But:

```
Function "withdraw":
  Neighbors: [STATE_VAR "balances", FUNCTION "transfer", CFG_NODE_CALL, CFG_NODE_WRITE]
```

Not all neighbors are equally relevant for reentrancy detection. The CFG_NODE_CALL that happens **before** the CFG_NODE_WRITE is the crucial signal. Mean pooling dilutes this.

**Graph Attention Networks (GAT)** learn to weight neighbors differently:
```
withdraw.embedding = α_balances * embed_balances
                   + α_transfer * embed_transfer
                   + α_cfg_call * embed_cfg_call     ← should be high weight
                   + α_cfg_write * embed_cfg_write    ← should be high weight
```

The attention weights `α` are **learned** — the model discovers which neighbor types matter.

---

## 3. How GATConv Computes Attention Weights

```python
self.conv1 = GATConv(
    in_channels=NODE_FEATURE_DIM,  # 11
    out_channels=_head_dim,         # 32 per head
    heads=8,
    concat=True,
    add_self_loops=True,
    edge_dim=_edge_dim,             # 64 — edge type embedding
)
```

For each edge `(u → v)`, GAT computes an attention coefficient:

```
e_uv = LeakyReLU( a^T · [W h_u || W h_v || W_e edge_type_uv] )
α_uv = softmax over all edges incoming to v: exp(e_uv) / Σ_k exp(e_uk)
```

Where:
- `h_u`, `h_v` are node features
- `W` is a learned linear transformation
- `W_e edge_type_uv` is the edge type embedding (64-dim)
- `a` is a learnable attention vector
- `||` means concatenation

The edge type embedding `W_e edge_type_uv` means the attention weight is different for a CALLS edge vs a CONTROL_FLOW edge. The GNN learns "a CONTROL_FLOW edge from a CFG_NODE_CALL carries a different kind of information than a READS edge."

**Output:** Each node v aggregates weighted neighbor messages:
```
h_v_new = ||_{k=1}^{heads} σ(Σ_u α^k_uv W^k h_u)
```

Where `||` is concatenation across attention heads.

---

## 4. Multi-Head Attention — Why 8 Heads in Phase 1?

```python
heads = 8
_head_dim = hidden_dim // heads  # 256 // 8 = 32
```

With 8 heads, each head learns a **different** attention pattern on the same graph:
- Head 1 might learn to attend strongly to CALLS edges
- Head 2 might attend to WRITES edges
- Head 3 might attend to control-flow predecessors

The 8 head outputs are concatenated: `8 × 32 = 256 dims`. This gives the model 8 different "perspectives" on each node's neighborhood.

**Phase 1 uses 8 heads; Phases 2+3 use 1 head:**
```python
# Phase 1: multi-head for rich structural encoding
self.conv1 = GATConv(..., heads=8, concat=True)   # output: hidden_dim=256

# Phase 2+3: single head for directed signal
self.conv3 = GATConv(..., heads=1, concat=False)  # output: hidden_dim=256
```

Why single head for CFG phases? Phase 2 processes **directed** CONTROL_FLOW edges where the direction itself carries the vulnerability signal. Multiple heads on directed edges tend to converge to the same pattern. Single head with full capacity (`out_channels=hidden_dim=256`) is more parameter-efficient.

---

## 5. `add_self_loops` — Why Phase 2 MUST Use `False`

```python
# Phase 1: add_self_loops=True  ✓
# Phase 2: add_self_loops=False ← CRITICAL comment in source

self.conv3 = GATConv(..., add_self_loops=False)
```

A self-loop adds an edge from each node to itself. In Phase 1, self-loops help stabilize training by giving each node access to its own features after aggregation (prevents features from being "washed out").

**Why NOT in Phase 2?**

CONTROL_FLOW edges encode execution order: `A → B` means "A executes before B." If you add a self-loop `A → A`, you're saying "A executes before itself" — meaningless for execution order.

Worse: in GAT, adding a self-loop means the node "attends to itself" in addition to its actual successors. For a CFG_NODE_CALL, its own embedding should flow forward to its successor (the CFG_NODE_WRITE in reentrancy). Self-loops dilute this directional signal by averaging the forward message with the node's own current state.

> 🎯 **INTERVIEW FOCUS:** "When would you disable self-loops in a GNN?" — When the edge direction carries the primary signal (execution order, data flow). Self-loops are helpful for isotropic aggregation but harmful for directional patterns.

---

## 6. The Residual Connection Pattern

```python
# Phase 1, Layer 2:
x2 = self.conv2(x, struct_ei, struct_ea)    # conv output
x2 = self.relu(x2)
x  = x + self.dropout(x2)                   # residual: x = x + conv(x)
```

**Residual connections** (from ResNet, 2015) add the input to the output: `output = x + f(x)`. This allows:

1. **Gradient flow**: gradients can flow directly through the identity path (the `+`), bypassing the convolution layers. This prevents vanishing gradients in deep networks.

2. **Preserving information**: the original node features are always preserved. The convolution learns only what to **add** to the existing representation, not to replace it entirely.

3. **Training stability**: the residual path initializes as a near-identity function. The network can start training immediately from the original features.

**IMP-G2 — The Input Skip:**
```python
# IMP-G2: save raw features before any conv
x_skip = self.input_proj(x_init)           # Linear(11, 256, bias=False)
x      = self.conv1(x_init, ...)
x      = self.relu(x + x_skip)             # skip BEFORE relu
```

This is a special residual for the very first layer where dimensions change (11 → 256). When GAT attention weights are near-uniform at initialization (before training), `conv1` output is just the mean of all neighbors' features — extremely noisy. The skip connection ensures the raw 11-dim features (normalized, informative) are always preserved in the 256-dim space. `input_proj` is just a `Linear(11, 256, bias=False)` — 2,816 parameters, negligible overhead.

---

## 7. LayerNorm After Each Phase

```python
self.phase_norm = nn.ModuleList([
    nn.LayerNorm(hidden_dim),  # after Phase 1
    nn.LayerNorm(hidden_dim),  # after Phase 2
    nn.LayerNorm(hidden_dim),  # after Phase 3
])

# Usage:
x = self.phase_norm[0](x)  # after Phase 1
```

**LayerNorm** normalizes each node's embedding vector to have mean 0 and std 1 (across the 256 feature dimensions). This is different from BatchNorm (which normalizes across the batch dimension).

**Why LayerNorm after each phase (not after each layer)?**
The JK mechanism (covered next) takes the outputs of all 3 phases and combines them. If Phase 1 (2 layers, higher norm) and Phase 2 (3 layers, lower norm) have very different L2 norms, Phase 1 would dominate the JK attention softmax. Per-phase LayerNorm equalizes the scales before JK aggregation.

---

## 8. Jumping Knowledge (JK) Connections — The Aggregation Strategy

```python
class _JKAttention(nn.Module):
    def __init__(self, channels, num_phases=3):
        self.attn = nn.Linear(channels, 1, bias=False)
    
    def forward(self, xs):  # xs = [phase1_out, phase2_out, phase3_out]
        stacked = torch.stack(xs, dim=1)        # [N, 3, 256]
        scores  = self.attn(stacked)             # [N, 3, 1]
        weights = torch.softmax(scores, dim=1)   # [N, 3, 1]
        return (weights * stacked).sum(dim=1)    # [N, 256]
```

**The problem JK solves:** In deep GNNs, later layers "over-smooth" node representations — after many rounds of aggregation, all nodes start looking similar. A FUNCTION node and a CFG_NODE_CHECK node might converge to nearly the same embedding.

**JK (Jumping Knowledge) networks** keep intermediate representations and combine them. The key insight: **different nodes might benefit from different "depths" of aggregation.**

- A CONTRACT node benefits most from Phase 1 (structural, 2 hops): who does it call, what does it inherit?
- A CFG_NODE_CALL benefits most from Phase 2 (execution order, 3 hops): what happened before and after this call?
- A FUNCTION node benefits from Phase 3 (bidirectional, 3 hops): what do its CFG children look like?

The JK attention mechanism learns this automatically: it assigns high weight to the phase most informative for each node type.

**Why `bias=False` in the attention linear?**
```python
self.attn = nn.Linear(channels, 1, bias=False)
```
A bias would add a constant to every node's score for each phase, potentially creating a phase preference independent of the node content. `bias=False` forces the attention to be purely content-dependent: "does this phase's embedding have the right structure?" not "is this generally the preferred phase?"

---

## 9. Edge Type Embeddings

```python
self.edge_embedding = nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)
# NUM_EDGE_TYPES=11, edge_emb_dim=64
```

This is a lookup table: each of the 11 edge types gets a 64-dimensional learnable vector. During the forward pass:

```python
e = self.edge_embedding(edge_attr)   # [E] int64 → [E, 64]
```

This embedding vector is appended to the attention computation in GATConv, making the attention weight aware of the edge type.

**Why not one-hot encode edge types?** One-hot would be an 11-dimensional vector, fixed. A learnable 64-dim embedding:
1. Captures semantic relationships between edge types (CALLS and CONTROL_FLOW are related — both represent "function uses another function")
2. Can be updated during training to encode whatever the model finds most useful

**Phase masks — only use relevant edge types per phase:**
```python
struct_mask = edge_attr <= EDGE_TYPES["CONTAINS"]       # types 0-5
cfg_mask    = (edge_attr == CONTROL_FLOW) | 
              (edge_attr == CALL_ENTRY) | 
              (edge_attr == RETURN_TO)                    # types 6,8,9

struct_ei = edge_index[:, struct_mask]
phase2_ei = edge_index[:, cfg_mask]
```

Boolean masks filter `edge_index` and `edge_attr` to only the relevant edges for each phase. Phase 1 only sees structural edges (0–5); Phase 2 only sees CFG/ICFG edges.

---

## 10. Input Guards — Defensive Production Code

```python
def forward(self, x, edge_index, batch, edge_attr=None, ...):
    # Guard 1: feature dimension
    if x.shape[1] != NODE_FEATURE_DIM:
        raise ValueError(f"expects {NODE_FEATURE_DIM}-dim features, got {x.shape[1]}")
    
    # Guard 2: edge_attr required when use_edge_attr=True
    if self.use_edge_attr and edge_attr is None:
        raise ValueError("use_edge_attr=True but edge_attr=None")
    
    # Guard 3: valid node indices
    if edge_index.max() >= x.shape[0]:
        raise ValueError(f"edge_index node index {edge_index.max()} out of range")
    
    # Guard 4: OOB edge type IDs → clamp not crash
    _oob_mask = (edge_attr < 0) | (edge_attr > _max_valid)
    if _oob_mask.any():
        logger.warning("OOB edge_attr clamped")
        edge_attr = edge_attr.clamp(0, _max_valid)
    
    # Guard 5: dtype normalization
    if x.dtype != _param_dtype:
        x = x.to(_param_dtype)
```

**Why these guards?**
- Guard 1: Schema version mismatch → clear error message
- Guard 2: Missing edge_attr silently disables Phase 2 → fail loud
- Guard 3: Out-of-bounds node index causes silent wrong results on CPU, illegal memory access on CUDA (hard to debug)
- Guard 4: OOB edge type causes `nn.Embedding` CUDA crash with useless traceback → clamp and warn instead
- Guard 5: BERT loads in BF16 and can poison the default dtype, causing GNN tensors to arrive as BF16 → normalize to float32

Each guard prevents a specific production failure mode.

---

## 11. Summary of the GNN Architecture

```
Input: x [N, 11], edge_index [2, E], edge_attr [E], batch [N]
   ↓
Edge embeddings: edge_attr → [E, 64]
   ↓
Phase 1 (Layers 1+2): struct_edges (0-5)
  Layer 1: GATConv(11→256, heads=8) + input_proj skip + ReLU + Dropout
  Layer 2: GATConv(256→256, heads=8) + residual
  LayerNorm → phase1_out [N, 256]
   ↓
Phase 2 (Layers 3+4+5): directed CFG/ICFG (6,8,9)
  Layer 3: GATConv(CF only, no self-loops) + residual
  Layer 4: GATConv(ICFG only, no self-loops) + residual
  Layer 5: GATConv(CF+ICFG joint, no self-loops) + residual
  LayerNorm → phase2_out [N, 256]
   ↓
Phase 3 (Layers 6+7+8): bidirectional CONTAINS
  Layer 6: GATConv(rev_CONTAINS up) + residual
  Layer 7: GATConv(rev_CONTAINS up) + residual
  Layer 8: GATConv(fwd_CONTAINS down) + residual
  LayerNorm → phase3_out [N, 256]
   ↓
JK Attention: [phase1_out, phase2_out, phase3_out] → weighted sum [N, 256]
   ↓
Output: node_embeddings [N, 256], batch [N], jk_entropy scalar
```

Total GNN parameters: ~2.4M (was 91K with hidden=128, heads-wise scaling)

---

## Interview Questions

1. **"Explain message passing in GNNs."**
   → Each node iteratively updates its embedding by aggregating information from its graph neighbors. After k layers, a node's embedding contains information from all nodes within k hops. The aggregation function (mean, sum, attention) determines how neighbor information is combined.

2. **"What is Graph Attention Network (GAT) and how does it differ from GCN?"**
   → GCN uses fixed aggregation weights (degree-normalized mean). GAT learns attention weights per edge based on the features of the source and destination nodes, allowing the model to focus on the most informative neighbors. GAT is more expressive but has more parameters.

3. **"Why is Jumping Knowledge (JK) useful in deep GNNs?"**
   → Deep GNNs over-smooth: after many layers, all nodes converge to similar embeddings. JK retains each layer's (or phase's) output and combines them, allowing different nodes to benefit from different aggregation depths. A node near many relevant neighbors benefits from deep aggregation; an isolated node benefits from shallow.

4. **"What is the purpose of residual connections in a GNN?"**
   → Allow gradients to flow directly through the network (preventing vanishing gradients), preserve the original node features through each layer, and initialize the network as a near-identity function that learns incrementally what to add.

---

**Next:** `02_gnn_encoder_forward_pass.md` — The actual forward pass: guards, edge mask construction, IMP-G1 layer-specific subsets, REVERSE_CONTAINS synthesis, and all three phases in code.
