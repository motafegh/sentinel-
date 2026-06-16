 # models — SENTINEL Model Architecture (v8.1 — Four-Eye, Run 7+)

Four modules, one forward pass, four independent "eyes" on the contract. Each eye produces a 128-dim embedding; all four are concatenated into a 512-dim vector that the classifier maps to 10 vulnerability classes.

**Current architecture:** 8-layer GNN (2+3+3 phases), GraphCodeBERT + LoRA (r=16), bidirectional cross-attention fusion, CFG Eye (IMP-R7-2), type embedding (BUG-R7-2), Phase 2 multi-head (IMP-R7-1).

Shapes and layer counts are **locked** — changes require full retrain.

---

## Module Map

| File | Class | Role |
|------|-------|------|
| `gnn_encoder.py` | `GNNEncoder` | Contract graph → node embeddings `[N, 256]` (three-phase 8-layer GAT) |
| `transformer_encoder.py` | `TransformerEncoder` | Source code → token embeddings `[B, L, 768]` (GraphCodeBERT + LoRA) |
| `transformer_encoder.py` | `WindowAttentionPooler` | Multi-window CLS pooling `[B, 768]` |
| `fusion_layer.py` | `CrossAttentionFusion` | Bidirectional cross-attention → `[B, 128]` |
| `sentinel_model.py` | `SentinelModel` | Four-eye orchestrator + classification head → `[B, 10]` |

---

## Architecture Overview

```
Input: (PyG graph batch, input_ids [B, W, L], attention_mask [B, W, L])
  │
  ├──→ GNNEncoder(return_phase2_embs=True)
  │     │                │
  │     │  node_embs     │  _phase2_x (Phase 2 output, gradient-attached)
  │     │  [N, 256]      │  [N, 256]
  │     │                │
  │     ▼                ▼
  │   GNN Eye:           CFG Eye (IMP-R7-2):
  │   Pool over          Pool over CFG_NODE
  │   FUNCTION/MODIFIER/ types (8-12)
  │   FALLBACK/          from _phase2_x
  │   RECEIVE/CONSTRUCTOR
  │   max+mean → [B,512] max+mean → [B,512]
  │   → gnn_eye_proj     → cfg_eye_proj
  │   → [B, 128]         → [B, 128]
  │     │                │
  │     └──→ select_prefix_nodes → gnn_to_bert_proj
  │          → gnn_prefix [B, K, 768]  (None during warmup)
  │                    │
  └──→ TransformerEncoder(with prefix)
        │  token_embs [B, W*L, 768]
        │
        ├──→ Transformer Eye:
        │    WindowAttentionPooler → [B, 768]
        │    → transformer_eye_proj → [B, 128]
        │
        └──→ Fused Eye:
             CrossAttentionFusion
             (node_embs + token_embs)
             → [B, 128]

  cat([gnn_eye, tf_eye, fused_eye, cfg_eye]) → [B, 512]
  → Linear(512,256) → ReLU → Dropout → Linear(256,10) → logits [B, 10]
```

**No sigmoid inside the model.** Raw logits output; `BCEWithLogitsLoss` during training, `sigmoid()` in predictor.

---

## GNNEncoder (`gnn_encoder.py`)

8-layer Graph Attention Network, three phases (2+3+3), Jumping Knowledge (JK) attention aggregation over all three phase outputs.

### Phase 1 — Structural + CONTAINS (Layers 1+2)

```
Edges: types 0–5 (CALLS, READS, WRITES, EMITS, INHERITS, CONTAINS)
add_self_loops=True
heads=8, concat=True → output = hidden_dim (256)

Layer 1: _GNN_IN_DIM (28) → hidden_dim
  BUG-R7-2: type_embedding nn.Embedding(14, 16) prepended to node features.
    type_id was stored as float(id)/13.0 in feat[0] — a continuous scalar.
    GATConv cannot learn categorical structure from a single float. The 16-dim
    learned embedding gives each of the 14 node types its own representation
    vector. _GNN_IN_DIM = NODE_FEATURE_DIM (12) + _TYPE_EMB_DIM (16) = 28.
    No graph re-extraction needed — the embedding is model-internal.
  IMP-G2: input_proj skip connection (Linear(28, 256, bias=False)) added
    before ReLU in Layer 1. Prevents raw feature loss when GAT attention
    weights start near-uniform at initialization.

Layer 2: hidden_dim → hidden_dim
  Residual connection from Layer 1 output.

LayerNorm after phase.
```

### Phase 2 — CFG + ICFG Directed (Layers 3+4+5)

```
add_self_loops=False  ← CRITICAL — self-loops cancel directional CF signal
IMP-R7-1: heads=4 (was 1), concat=True, out=64/head → output = hidden_dim (256)

IMP-G1: each layer processes a DISTINCT edge subset:
  Layer 3 (conv3):  CONTROL_FLOW(6) only — intra-function execution ordering
  Layer 4 (conv3b): CALL_ENTRY(8) + RETURN_TO(9) only — cross-function call structure
  Layer 5 (conv3c): CF(6) + CALL_ENTRY(8) + RETURN_TO(9) joint — integration layer

All three layers use residual connections from the previous output.
LayerNorm after complete Phase 2.
```

**Why Phase 2 heads changed from 1→4 (IMP-R7-1):** With only 1 head, Phase 2 compressed all directional information — call flow, data flow, control flow — into a single attention pattern. 4 heads allow the model to attend to multiple types of directional flow simultaneously (e.g., one head for state-write flow, another for external-call chains, etc.). This change is only effective because BUG-R7-1 fixed the gradient starvation — before that fix, increasing heads would have diversified a dead signal.

**Why `add_self_loops=False` is critical:** Self-loops add an edge from each node to itself, which means every node's representation includes its own previous state regardless of directional edges. In Phase 2, the whole point is to learn directional patterns (A→B vs B→A). Self-loops create a shortcut that bypasses the directional structure, making forward and reverse flows indistinguishable.

### Phase 3 — Bidirectional CONTAINS (Layers 6+7+8)

```
heads=1, concat=False. Upward and downward CONTAINS passes.

Layer 6 (conv4):  REVERSE_CONTAINS up — CFG→FUNCTION
  Phase 2-enriched CFG signal rises to function-level nodes.

Layer 7 (conv4b): REVERSE_CONTAINS up — second hop
  Multi-function patterns propagate across the contract.

Layer 8 (conv4c): CONTAINS down — FUNCTION→CFG (IMP-G3)
  Distributes enriched FUNCTION context back to CFG children.
  After this pass, ALL nodes carry Phase 3 depth so
  CrossAttentionFusion sees uniformly-enriched embeddings.

LayerNorm after complete Phase 3.
```

### JK Attention Aggregation

```python
class _JKAttention(nn.Module):
    """Learned attention aggregation over 3 phase outputs."""
```

For each node, computes a scalar score per phase embedding, softmax-normalises across phases, and returns the weighted sum. This lets the model learn which phase is most informative for each node type (e.g., structural for CONTRACT nodes, directed for CFG_NODE_CALL nodes).

**Diagnostic outputs** (stored in buffers, survive `.to(device)` and DDP):
- `last_weights [K]` — mean per-phase attention weights across all nodes
- `last_weight_stds [K]` — per-phase std (high std = genuinely routing different nodes to different phases)
- `last_node_weights [N, K]` — full per-node weights (eval mode only, for `jk_weight_hist.py` diagnostic)

**JK entropy regularizer:** `jk_entropy = -(w * log(w)).sum(dim=1).mean()` — low entropy means one phase dominates; the trainer can use this as a regularization signal.

### Edge Masking per Phase

| Phase | Edge Type IDs | Layers |
|-------|--------------|--------|
| Phase 1 | 0–5 (structural + CONTAINS forward) | conv1, conv2 |
| Phase 2 (Layer 3) | 6 (CONTROL_FLOW only) | conv3 |
| Phase 2 (Layer 4) | 8+9 (CALL_ENTRY + RETURN_TO only) | conv3b |
| Phase 2 (Layer 5) | 6+8+9 (joint CF + ICFG) | conv3c |
| Phase 3 (Layers 6+7) | 7 (REVERSE_CONTAINS, runtime) | conv4, conv4b |
| Phase 3 (Layer 8) | 5 (CONTAINS forward, for downward pass) | conv4c |

Phase 2 also includes DEF_USE(10) edges when `phase2_edge_types` is None (default). The `phase2_edge_types` constructor argument allows ablation (e.g., ICFG-only `[6, 8, 9]`, DFG-only `[6, 10]`).

### Forward Signature

```python
GNNEncoder.forward(
    x:                    Tensor[N, 11],          # Node features
    edge_index:           Tensor[2, E],            # Graph connectivity
    batch:                Tensor[N],               # Node→graph mapping
    edge_attr:            Tensor[E] | None,        # Edge type IDs
    return_intermediates: bool = False,            # Diagnostic detached outputs
    return_phase2_embs:   bool = False,            # Phase 2 WITH gradients (for aux/CFG eye)
) -> (node_embs[N, 256], batch[N], jk_entropy, ...)
```

**Return modes:**

| Mode | Returns |
|------|---------|
| Default | `(node_embs, batch, jk_entropy)` |
| `return_intermediates=True` | adds `{"after_phase1", "after_phase2", "after_phase3"}` (detached) |
| `return_phase2_embs=True` | adds `phase2_x [N, 256]` **with gradients** for auxiliary loss and CFG eye |

### Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `_TYPE_EMB_DIM` | 16 | Learned node-type embedding dim (BUG-R7-2) |
| `_NUM_NODE_TYPES` | 14 | Number of distinct node types (IDs 0–13) |
| `_GNN_IN_DIM` | 28 | 12 (features) + 16 (type embedding) — model-internal input dim |
| `SENTINEL_GNN_NUM_LAYERS` | 8 | Fixed architecture: 2+3+3 phases |

---

## TransformerEncoder (`transformer_encoder.py`)

`microsoft/graphcodebert-base` (124M params) + LoRA adapters (~590K trainable at r=16).

### LoRA Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lora_r` | 16 | LoRA rank — A/B matrix inner dimension |
| `lora_alpha` | 32 | LoRA scale factor (effective scale = alpha/r = 2.0) |
| `lora_dropout` | 0.1 | Dropout on LoRA paths |
| `lora_target_modules` | `["query", "value"]` | Which attention projections to adapt |

**Why LoRA:** Full fine-tune = 125M params → OOM on 8GB VRAM + catastrophic forgetting. Frozen = 0 trainable → never adapts to vulnerability semantics. LoRA = ~590K trainable params that adapt query+value attention to security patterns while all 125M backbone weights remain frozen.

**peft is a hard requirement.** If `peft` is not installed, `TransformerEncoder.__init__` raises `RuntimeError` immediately — a silent fallback would allow training with 0 trainable transformer parameters, discovered only at evaluation time.

### Attention Implementation

Tries Flash Attention 2 first (tiled CUDA kernels, avoids materialising the full `[B*W, 512, 512]` attention matrix). Falls back to SDPA if FA2 is unavailable. Must be set before `get_peft_model()` so LoRA sees the correct implementation.

### Standard Path (no prefix, or warmup epochs 0–14)

```
Input: input_ids [B, W, L], attention_mask [B, W, L]
  W windows of L=512 tokens each

Each window processed independently:
  GraphCodeBERT → last_hidden_state [B, 512, 768]

Multi-window: flatten to [B*W, 512] → CodeBERT → reshape to [B, W*L, 768]
```

### Prefix Injection Path (epoch ≥ gnn_prefix_warmup_epochs)

```
Input: input_ids [B, W, L], attention_mask [B, W, L],
       gnn_prefix_nodes [B, K, 768], gnn_prefix_counts [B]

Uses inputs_embeds instead of input_ids:
  prefix_embeds [B, K, 768] + code_embeds [B, code_budget, 768]
  = total sequence length K + code_budget = 512

Position IDs:
  Prefix: position_id = 1  (RoBERTa padding slot — no positional bias)
  Code:   position_ids = 3..3+code_budget-1

IMP-M3: actual node count masking
  gnn_prefix_counts [B] tracks real (non-padded) nodes per graph.
  Zero-padded prefix positions are masked in attention.
  95.5% of contracts fill all K=48 slots — masking is a no-op for them.

Multi-window: prefix is shared across all W windows (same K nodes per contract).
```

### WindowAttentionPooler

```python
class WindowAttentionPooler(nn.Module):
    """Pool W window-CLS embeddings into a single vector via learned attention."""
```

In multi-window mode, CLS of window i is at position `i*512 + prefix_k`. This module extracts those W CLS vectors and produces a learned attention-weighted sum → `[B, 768]`.

Single-window fallback: if `W*L <= 512`, returns CLS at position `prefix_k` directly — zero overhead, no learned weights invoked.

### Forward Signature

```python
TransformerEncoder.forward(
    input_ids:          Tensor[B, L] or [B, W, L],
    attention_mask:     Tensor[B, L] or [B, W, L],
    gnn_prefix_nodes:   Tensor[B, K, 768] | None,
    gnn_prefix_counts:  Tensor[B] | None,           # IMP-M3
    output_attentions:  bool = False,                # IMP-M2 diagnostic
) -> Tensor[B, L, 768] or [B, W*L, 768]
   # With output_attentions=True and prefix: (last_hidden_state, prefix_attn_mean: float)
```

### _word_embeddings Property

Accesses the `nn.Embedding` layer of the underlying GraphCodeBERT model through PEFT's internal layout. Tries multiple known paths in precedence order for robustness across PEFT versions. Validated at `__init__` time — failures surface at construction, not at the first forward pass.

---

## CrossAttentionFusion (`fusion_layer.py`)

Bidirectional cross-attention between graph nodes and token sequence. Both directions reinforce the structural-semantic bridge before pooling dilutes fine-grained signal.

### Architecture

```
1. Project nodes [N, 256] → [N, 256]                    (node_proj)
2. LayerNorm + project tokens [B, 512, 768] → [B, 512, 256]  (token_norm + token_proj, BUG-C2)
3. _scatter_to_dense → [B, max_nodes, 256]  (static max_nodes=1024; compile-safe)
4. Node→Token cross-attention (Q=nodes, K=V=tokens)
     → enriched_nodes [B, max_nodes, 256]
4b. Zero-out padded node positions (Fix #8)
5. Token→Node cross-attention (Q=tokens, K=V=nodes)
     → enriched_tokens [B, 512, 256]
6. Masked mean pooling of real nodes  → [B, 256]
7. Masked mean pooling of real tokens → [B, 256]
8. Concat [B, 512] → Linear(512, 128) + ReLU + Dropout → [B, 128]
```

### Key Design Decisions

**BUG-C2 — Token Normalization:** CodeBERT hidden states have L2 norm ~10-15; GNN output after LayerNorm has norm ~1. Without `token_norm`, token keys dominate cross-attention dot products by 10-15×, making node→token attention attend to highest-norm tokens rather than semantically relevant ones.

**Fix #8 — Zero-out padded enriched nodes:** After node→token attention, padded node positions receive nonzero values from token content. While masked pooling excludes them, explicitly zeroing makes the invariant structural.

**Fix #26 — `need_weights=False`:** Attention weight matrices (`[B, max_nodes, 512]` + `[B, 512, max_nodes]` ≈ 12.6 MB) were computed but never used. `need_weights=False` lets PyTorch use fused efficient-attention kernels, saving VRAM and reducing allocator fragmentation.

**`_scatter_to_dense`** replaces PyG's `to_dense_batch` to eliminate `GuardOnDataDependentSymNode` compile graph breaks. Uses static `max_nodes` (a config constant) instead of a data-dependent `repeat(size)`. Contracts with >1024 nodes have excess nodes truncated (affects <1% of corpus).

### Forward Signature

```python
CrossAttentionFusion.forward(
    node_embs:      Tensor[N, 256],     # All nodes across the batch
    batch:          Tensor[N],          # Node→graph mapping
    token_embs:     Tensor[B, WL, 768], # All token embeddings (single or multi-window)
    attention_mask: Tensor[B, WL],      # 1=real, 0=PAD
) -> Tensor[B, 128]
```

**`output_dim=128` is LOCKED** — downstream components (eye projections, classifier) depend on this exact shape.

---

## SentinelModel (`sentinel_model.py`)

Four-eye classifier orchestrating all sub-modules into a NUM_CLASSES-class multi-label vulnerability detector (10 in Run 12, 9 in Run 13).

### Four Eyes

| Eye | Input | Pooling | Projection | Output |
|-----|-------|---------|------------|--------|
| **GNN Eye** | `node_embs` (post-JK) | max+mean over FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes | `Linear(512, 128) + ReLU + Dropout` | `[B, 128]` |
| **Transformer Eye** | `token_embs` (post-CodeBERT) | `WindowAttentionPooler` → CLS | `Linear(768, 128) + ReLU + Dropout` | `[B, 128]` |
| **Fused Eye** | `node_embs` + `token_embs` | `CrossAttentionFusion` | (inside fusion) | `[B, 128]` |
| **CFG Eye** (IMP-R7-2) | `_phase2_x` (Phase 2 output, gradient-attached) | max+mean over CFG_NODE types (8–12) | `Linear(512, 128) + ReLU + Dropout` | `[B, 128]` |

**Why the CFG Eye pools over CFG nodes (BUG-R7-1 fix):** Phase 2 GATConv layers (conv3/conv3b/conv3c) operate on edges that connect ONLY CFG_NODE types (IDs 8–12). Function nodes have no incoming Phase 2 edges (self_loops=False), so they receive zero Phase 2 messages. Pooling over FUNCTION nodes for the Phase 2 auxiliary head sent gradient only through the residual path, never reaching conv3 — causing gradient starvation (Ph2/Ph1 ratio = 0.18–0.36). Pooling over CFG nodes creates a direct gradient path from classifier loss to conv3/conv3b/conv3c, fixing the starvation (Ph2/Ph1 ratio = 0.74–0.85).

### Main Classifier

```
cat([gnn_eye, tf_eye, fused_eye, cfg_eye]) → [B, 512]
→ Linear(512, 256) → ReLU → Dropout → Linear(256, 10) → logits [B, 10]
```

### Auxiliary Heads (Training Only)

Each auxiliary head produces independent logits for its eye, keeping that eye's gradient alive even if the main classifier learns to downweight it. Discarded at inference via `return_aux=False`.

| Head | Architecture | Input |
|------|-------------|-------|
| `aux_gnn` | `Linear(128, 10)` | GNN eye embedding |
| `aux_transformer` | `Linear(128, 10)` | Transformer eye embedding |
| `aux_fused` | `Linear(128, 10)` | Fused eye embedding |
| `aux_phase2` | `Linear(256,128) → GELU → Dropout → Linear(128,10)` | Phase 2 pooled over CFG nodes (BUG-R7-1) |

**Note:** There is no `aux_cfg` head. The CFG Eye gets its gradient from the main classifier loss directly through the concatenated 512-dim input — it does not need a separate auxiliary supervision branch.

### GNN Prefix Injection

Bridges GNN and Transformer by injecting structural context as soft prefix tokens:

```python
# Components on SentinelModel:
gnn_to_bert_proj      = Linear(256, 768)      # projects GNN hidden → BERT embedding dim
prefix_type_embedding = Embedding(5, 768)     # type-specific bias per STRUCTURAL_PREFIX_TYPES
```

**`select_prefix_nodes()`:**
- Priority: CONSTRUCTOR(0) > FALLBACK(1) > RECEIVE(2) > MODIFIER(3) > FUNCTION(4)
- Secondary sort: FUNCTION nodes by `external_call_count` (feat[10]) descending (IMP-M1)
- Returns top-K=48 declaration nodes per graph → `[B, K, 768]` projected prefix embeddings + `[B]` node counts (IMP-M3)

**Warmup suppression:**
- Epochs 0..(warmup-1): `gnn_prefix_nodes=None` — TransformerEncoder uses standard path
- `gnn_to_bert_proj` receives zero gradient during warmup
- Epoch `gnn_prefix_warmup_epochs` (default 15): projection fires from random init with a well-trained GNN

**Inference:** `predictor.py` sets `model._current_epoch = 9999` so prefix is always active.

### Forward Signature

```python
SentinelModel.forward(
    graphs:         Batch,                    # PyG Batch — batched contract graphs
    input_ids:      Tensor[B, L] or [B, W, L],
    attention_mask: Tensor[B, L] or [B, W, L],
    return_aux:     bool = False,             # True during training only
) -> Tensor[B, 10]  |  (Tensor[B, 10], dict)

# return_aux=False (default / inference):
#   logits [B, 10] — raw logits, NO Sigmoid

# return_aux=True (training):
#   (logits [B, 10], {"gnn": [B,10], "transformer": [B,10], "fused": [B,10],
#                      "phase2": [B,10], "jk_entropy": scalar})
```

### Schema-Derived Constants

| Constant | Value | Source | Purpose |
|----------|-------|--------|---------|
| `_MAX_TYPE_ID` | 13.0 | `max(NODE_TYPES.values())` | Recover integer type_id from normalised feat[0] |
| `_FUNC_TYPE_IDS` | {1,2,4,5,6} | `NODE_TYPES` | GNN Eye pooling target: function-level nodes |
| `_CFG_TYPE_IDS` | {8,9,10,11,12} | `NODE_TYPES` | CFG Eye + aux_phase2 pooling target |
| `_FUNC_IDS_CPU` | Tensor[5] | Pre-built from `_FUNC_TYPE_IDS` | Avoids allocation per forward pass |
| `_CFG_IDS_CPU` | Tensor[5] | Pre-built from `_CFG_TYPE_IDS` | Avoids allocation per forward pass |
| `_PREFIX_NODE_PRIORITY` | dict | `NODE_TYPES` | Prefix selection priority |
| `_PREFIX_TYPE_IDX` | dict | `NODE_TYPES` | Stable embedding index for prefix |

Import-time assertion: `_MAX_TYPE_ID == 13.0` — fires immediately if a new node type is added to `NODE_TYPES`, preventing silent misalignment.

### Diagnostic Methods

```python
model.compute_prefix_attention_mean(graphs, input_ids, attention_mask) -> float | None
# Returns mean attention weight code→prefix positions, averaged over all layers/heads.
# ~15% overhead — call once per validation epoch, not per training step.
# Returns None when prefix disabled or in warmup.

model.parameter_summary() -> None
# Logs trainable vs frozen parameter counts per sub-module.
```

### Instantiation

```python
model = SentinelModel(
    fusion_output_dim=128,
    dropout=0.3,
    num_classes=10,
    # GNN
    gnn_hidden_dim=256,
    gnn_num_layers=8,                 # 2+3+3 phases (fixed)
    gnn_heads=8,                      # Phase 1 heads
    gnn_dropout=0.2,
    use_edge_attr=True,
    gnn_edge_emb_dim=64,
    gnn_use_jk=True,
    gnn_jk_mode='attention',
    gnn_phase2_edge_types=None,       # None = all v8 Phase 2 types [6,8,9,10]
    # LoRA
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=None,         # default: ["query", "value"]
    # GNN prefix injection
    gnn_prefix_k=48,                  # 0 = disabled
    gnn_prefix_warmup_epochs=15,
    # Fusion
    fusion_max_nodes=1024,
)
model = model.to("cuda")
model._current_epoch = 0              # set before each epoch in trainer
```

### Loading a Checkpoint

```python
ckpt = torch.load("ml/checkpoints/gcb_best.pt", map_location=device, weights_only=False)
# Strip torch.compile prefix from keys:
state = {k.replace("._orig_mod.", "."): v for k, v in ckpt["model_state_dict"].items()}
model.load_state_dict(state)
model.eval()
model._current_epoch = 9999           # ensures prefix always active at inference
```

---

## Parameter Budget

| Sub-module | Trainable | Frozen |
|------------|-----------|--------|
| GNNEncoder (8-layer GAT + type emb + edge emb) | ~2.5M | 0 |
| GraphCodeBERT base | 0 | ~124M |
| LoRA adapters (Q+V × 12 layers) | ~590K | 0 |
| gnn_to_bert_proj + prefix_type_embedding | ~200K | 0 |
| CrossAttentionFusion | ~1.5M | 0 |
| Eye projections (gnn + tf + cfg) | ~200K | 0 |
| Classifier (512→256→10) | ~130K | 0 |
| Auxiliary heads (4×) | ~5K | 0 |
| **Total** | **~5.0M** | **~124M** |

---

## Vulnerability Classes (NUM_CLASSES-class multi-label; 10 in Run 12, 9 in Run 13)

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | CallToUnknown | Call to untrusted or unknown address |
| 1 | DenialOfService | Unbounded loops, gas-intensive operations |
| 2 | ExternalBug | External contract interaction bugs |
| 3 | GasException | Gas-related exceptions and limits |
| 4 | IntegerUO | Integer overflow/underflow |
| 5 | MishandledException | Unchecked return values, missing error handling |
| 6 | Reentrancy | Reentrancy via external callbacks |
| 7 | Timestamp | Timestamp dependency for logic |
| 8 | TransactionOrderDependence | Front-running / transaction ordering |
| 9 | UnusedReturn | Discarded return values from external calls |

---

## Run 7 Fix Summary (v8.1)

| ID | Type | Change | Impact |
|----|------|--------|--------|
| BUG-R7-1 | Bug fix | aux_phase2 pools CFG nodes (not FUNCTION) | Phase 2 gradient starvation fixed; Ph2/Ph1 ratio 0.18→0.74 |
| BUG-R7-2 | Bug fix | type_id scalar → nn.Embedding(14, 16) | Categorical node type representation; cleaner gradients |
| IMP-R7-1 | Improvement | Phase 2 heads 1→4 | Multi-pattern directional flow; only effective after BUG-R7-1 |
| IMP-R7-2 | Improvement | CFG Eye added (4th eye, classifier 384→512) | Direct gradient to conv3; execution-flow perspective |
| IMP-R7-3 | Improvement | Phase 2 aux weight 0.10→0.20 | Stronger Phase 2 supervision; amplifies BUG-R7-1 fix |

---

## Cross-Module Dependency Map

```
graph_schema.py ────────────────────────────────────────────────────┐
   │ NODE_FEATURE_DIM, NODE_TYPES, NUM_EDGE_TYPES, EDGE_TYPES      │
   │                                                                │
   ├──→ gnn_encoder.py                                             │
   │      imports: NODE_FEATURE_DIM, NODE_TYPES, NUM_EDGE_TYPES,   │
   │               EDGE_TYPES                                      │
   │      exports: GNNEncoder, _JKAttention, SENTINEL_GNN_NUM_LAYERS│
   │                                                                │
   ├──→ transformer_encoder.py                                     │
   │      imports: (none from graph_schema)                        │
   │      exports: TransformerEncoder, WindowAttentionPooler        │
   │                                                                │
   ├──→ fusion_layer.py                                            │
   │      imports: (none from graph_schema)                        │
   │      exports: CrossAttentionFusion, _scatter_to_dense         │
   │                                                                │
   ├──→ sentinel_model.py  ←───────────────────────────────────────┘
   │      imports: NODE_TYPES (from graph_schema)
   │      imports: GNNEncoder (from gnn_encoder)
   │      imports: TransformerEncoder, WindowAttentionPooler
   │      imports: CrossAttentionFusion (from fusion_layer)
   │      exports: SentinelModel, _FUNC_TYPE_IDS, _MAX_TYPE_ID
   │
   └──→ predictor.py (in inference/)
          imports: SentinelModel, _FUNC_TYPE_IDS, _MAX_TYPE_ID
          imports: NODE_FEATURE_DIM, NODE_TYPES
```

No circular dependencies. All model files import from `graph_schema.py` (single source of truth) and from each other in a strict DAG: schema → gnn_encoder → sentinel_model ← transformer_encoder ← fusion_layer.
