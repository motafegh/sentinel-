# SENTINEL ML Model Architecture — Technical Reference

**Version:** v5.2  
**Architecture ID:** `three_eye_v5`  
**Document status:** Authoritative specification  
**Source of truth:** `ml/src/models/`, `ml/src/preprocessing/graph_schema.py`

---

## What This Document Covers

This document is a complete technical specification of the SENTINEL v5.2 machine learning model for smart contract vulnerability detection. It describes every component — GNNEncoder, TransformerEncoder, CrossAttentionFusion, and the Three-Eye Classifier — with exact parameter counts, tensor shapes, constructor signatures, forward-pass semantics, and all locked constants. A new ML engineer reading this document has enough information to re-implement the model, load a checkpoint, run inference, and understand every design decision without consulting the source code. Numbers in this document are verified directly from the implementation.

---

## Table of Contents

1. [Locked Constants](#1-locked-constants)
2. [Graph Schema](#2-graph-schema)
   - 2.1 [Node Types](#21-node-types)
   - 2.2 [Edge Types](#22-edge-types)
   - 2.3 [Node Feature Layout](#23-node-feature-layout)
3. [GNNEncoder](#3-gnnencoder)
   - 3.1 [Constructor Parameters](#31-constructor-parameters)
   - 3.2 [Sub-module Parameter Breakdown](#32-sub-module-parameter-breakdown)
   - 3.3 [Three-Phase Forward Pass](#33-three-phase-forward-pass)
   - 3.4 [JK Attention Aggregation](#34-jk-attention-aggregation)
   - 3.5 [Per-Phase LayerNorm](#35-per-phase-layernorm)
   - 3.6 [Input/Output Contract](#36-inputoutput-contract)
4. [TransformerEncoder](#4-transformerencoder)
   - 4.1 [LoRA Configuration](#41-lora-configuration)
   - 4.2 [Parameter Counts](#42-parameter-counts)
   - 4.3 [Input/Output Contract](#43-inputoutput-contract)
5. [CrossAttentionFusion](#5-crossattentionfusion)
   - 5.1 [Sub-module Parameter Breakdown](#51-sub-module-parameter-breakdown)
   - 5.2 [Forward Pass](#52-forward-pass)
6. [Three-Eye Classifier (SentinelModel)](#6-three-eye-classifier-sentinelmodel)
   - 6.1 [GNN Eye](#61-gnn-eye)
   - 6.2 [Transformer Eye](#62-transformer-eye)
   - 6.3 [Fused Eye](#63-fused-eye)
   - 6.4 [Main Classifier](#64-main-classifier)
   - 6.5 [Auxiliary Heads](#65-auxiliary-heads)
   - 6.6 [Full Forward Pass](#66-full-forward-pass)
7. [Total Parameter Summary](#7-total-parameter-summary)
8. [Checkpoint Format](#8-checkpoint-format)
9. [Inference Notes](#9-inference-notes)
10. [Locked Constants Reference Table](#10-locked-constants-reference-table)

---

## 1. Locked Constants

These values are architectural invariants. Changing any of them requires rebuilding training data and/or retraining from scratch. They must never be hardcoded in modules other than `graph_schema.py`.

```python
MODEL_VERSION           = "v5.2"
ARCHITECTURE            = "three_eye_v5"
FEATURE_SCHEMA_VERSION  = "v3"       # cache-key suffix; bump to invalidate inference cache
NODE_FEATURE_DIM        = 12         # in_channels for GNNEncoder
NUM_CLASSES             = 10         # append-only; no reordering
NUM_EDGE_TYPES          = 8          # 7→8 in v5.2; row 7 = REVERSE_CONTAINS (runtime-only)
NUM_NODE_TYPES          = 13         # type_id range 0–12
fusion_output_dim       = 128        # LOCKED — ZKML proxy MLP depends on this output width
MAX_TOKEN_LENGTH        = 512        # CodeBERT sequence length
```

**`fusion_output_dim = 128` is hard-locked.** The downstream ZK proof circuit (EZKL/Groth16 proxy MLP: `Linear(128→64→32→10)`) takes a 128-dimensional contract embedding as input. Changing this value breaks the ZK pipeline without a full circuit re-export.

---

## 2. Graph Schema

Source: `ml/src/preprocessing/graph_schema.py`

### 2.1 Node Types

The `NODE_TYPES` dict maps declaration kind to integer ID. The ID is stored as `float(id) / 12.0` in `graph.x[:, 0]` (normalised to `[0, 1]`; see Section 2.3). IDs are stable — insertions are append-only; re-ordering invalidates all training data.

| ID | Name | Category |
|----|------|----------|
| 0 | `STATE_VAR` | Declaration |
| 1 | `FUNCTION` | Declaration |
| 2 | `MODIFIER` | Declaration |
| 3 | `EVENT` | Declaration |
| 4 | `FALLBACK` | Declaration |
| 5 | `RECEIVE` | Declaration |
| 6 | `CONSTRUCTOR` | Declaration |
| 7 | `CONTRACT` | Declaration |
| 8 | `CFG_NODE_CALL` | CFG — external call statement |
| 9 | `CFG_NODE_WRITE` | CFG — state variable write |
| 10 | `CFG_NODE_READ` | CFG — state variable read |
| 11 | `CFG_NODE_CHECK` | CFG — require/assert/if condition |
| 12 | `CFG_NODE_OTHER` | CFG — all other statement types |

CFG node type priority (when a single IR node spans multiple ops): CALL > WRITE > READ > CHECK > OTHER.

### 2.2 Edge Types

| ID | Name | Direction | Storage |
|----|------|-----------|---------|
| 0 | `CALLS` | function → called function | on disk |
| 1 | `READS` | function → state variable it reads | on disk |
| 2 | `WRITES` | function → state variable it writes | on disk |
| 3 | `EMITS` | function → event it emits | on disk |
| 4 | `INHERITS` | contract → parent contract (linearised MRO) | on disk |
| 5 | `CONTAINS` | function node → its CFG_NODE children | on disk |
| 6 | `CONTROL_FLOW` | CFG_NODE → successor CFG_NODE (directed) | on disk |
| 7 | `REVERSE_CONTAINS` | CFG_NODE → parent function | **runtime-only** |

`REVERSE_CONTAINS` (ID 7) is **never written to `.pt` files on disk**. It is generated inside `GNNEncoder.forward()` Phase 3 by flipping `CONTAINS(5)` edges: `edge_index[:, contains_mask].flip(0)`. The embedding table has 8 rows so index-7 lookups do not crash.

`graph.edge_attr` shape: `[E]` 1-D int64. Shape `[E, 1]` will crash `nn.Embedding`. Validate with `validate_graph_dataset.py` before training.

### 2.3 Node Feature Layout

12 scalar features per node (`NODE_FEATURE_DIM = 12`). Column index corresponds to position in `graph.x[:, i]`.

| Index | Name | Encoding | Notes |
|-------|------|----------|-------|
| 0 | `type_id` | `float(NODE_TYPES[kind]) / 12.0` | Normalised to `[0,1]`; raw 0–12 dominates dot product |
| 1 | `visibility` | `VISIBILITY_MAP` ordinal: public/external=0, internal=1, private=2 | Ordinal, not one-hot |
| 2 | `pure` | bool (0.0 / 1.0) | No state I/O |
| 3 | `view` | bool (0.0 / 1.0) | Read-only state |
| 4 | `payable` | bool (0.0 / 1.0) | Ether entry point |
| 5 | `complexity` | `float(len(func.nodes))` | CFG block count |
| 6 | `loc` | `float(len(source_mapping.lines))` | Lines of code |
| 7 | `return_ignored` | 0.0=captured / 1.0=discarded / −1.0=IR unavailable | Sentinel −1.0 = not safe |
| 8 | `call_target_typed` | 0.0=raw addr / 1.0=typed / −1.0=source unavailable | Sentinel −1.0 = not safe |
| 9 | `in_unchecked` | bool (0.0 / 1.0) | Function contains `unchecked{}` block |
| 10 | `has_loop` | bool (0.0 / 1.0) | Function contains a loop |
| 11 | `external_call_count` | `log1p(n) / log1p(20)`, clamped `[0,1]` | Log-normalised |

Non-Function nodes receive `0.0` for features `[2:]` except: `call_target_typed[8]` defaults to `1.0` (not applicable, safe default). CFG_NODE `in_unchecked[9]` is always `0.0` — not inherited from parent function.

To recover integer `type_id` from a stored feature vector:
```python
type_id = (x[:, 0].float() * 12.0).round().long()
```

---

## 3. GNNEncoder

Source: `ml/src/models/gnn_encoder.py`

Three-phase, four-layer Graph Attention Network. Returns node-level embeddings (NOT pooled). Pooling is performed in `SentinelModel` separately for each eye.

### 3.1 Constructor Parameters

```python
GNNEncoder(
    hidden_dim    = 128,
    heads         = 8,          # Phase 1 only; Phases 2+3 use heads=1
    dropout       = 0.2,
    use_edge_attr = True,
    edge_emb_dim  = 32,
    num_layers    = 4,          # stored for serialisation; validated in TrainConfig
    use_jk        = True,       # JK attention aggregation over all 3 phase outputs
    jk_mode       = 'attention' # only supported mode
)
```

Constraint: `hidden_dim` must be divisible by `heads`. With defaults: `128 / 8 = 16` dims per head.

### 3.2 Sub-module Parameter Breakdown

Total trainable: **70,272** (all parameters are trainable; no frozen weights in GNNEncoder).

| Sub-module | Shape / Config | Params |
|------------|---------------|--------|
| `edge_embedding` | `nn.Embedding(8, 32)` | 256 |
| `conv1` | `GATConv(in=12, out=16/head, heads=8, concat=True, add_self_loops=True)` | 6,144 |
| `conv2` | `GATConv(in=128, out=16/head, heads=8, concat=True, add_self_loops=True)` | 20,992 |
| `conv3` | `GATConv(in=128, out=128, heads=1, concat=False, add_self_loops=False)` | 20,992 |
| `conv4` | `GATConv(in=128, out=128, heads=1, concat=False, add_self_loops=False)` | 20,992 |
| `phase_norm` | `ModuleList([LayerNorm(128)] × 3)` | 768 |
| `jk` | `_JKAttention: Linear(128, 1, bias=False)` | 128 |
| **Total** | | **70,272** |

Note: `conv1` uses `out_channels=16` (per-head dim), not 128. With `heads=8, concat=True` the total output is `8 × 16 = 128`. Passing `out_channels=128` with `heads=8, concat=True` would produce 1024-dim output — a common mistake.

### 3.3 Three-Phase Forward Pass

```
Input:  x [N, 12],  edge_index [2, E],  batch [N],  edge_attr [E] int64
```

**Edge masking (applied before each phase):**
```python
struct_mask   = edge_attr <= 5   # types 0–5: CALLS, READS, WRITES, EMITS, INHERITS, CONTAINS
cfg_mask      = edge_attr == 6   # CONTROL_FLOW only
contains_mask = edge_attr == 5   # CONTAINS only (used to build Phase 3 reverse edges)
```

**Phase 1 — Structural aggregation (conv1 + conv2):**
- Edges: `struct_mask` (types 0–5)
- `add_self_loops=True`
- Layer 1: `[N, 12] → [N, 128]` (no residual; dimensions differ)
- Layer 2: `[N, 128] → [N, 128]` + residual from Layer 1
- `phase_norm[0](x)` applied after residual
- Output: `[N, 128]`, appended to `_live` for JK

**Phase 2 — CFG-directed aggregation (conv3):**
- Edges: `cfg_mask` (CONTROL_FLOW only)
- `add_self_loops=False` — **CRITICAL**: self-loops cancel the directional execution-order signal that CONTROL_FLOW encodes. This flag must never be changed to `True`.
- `heads=1, concat=False` — single relationship type; full hidden_dim capacity
- `[N, 128] → [N, 128]` + residual from Phase 1
- Non-CFG_NODE nodes have no CONTROL_FLOW edges; GATConv returns zero for them (they carry their Phase 1 embedding unchanged through the residual)
- `phase_norm[1](x)` applied after residual
- Output: `[N, 128]`, appended to `_live` for JK

**Phase 3 — Reverse-CONTAINS aggregation (conv4):**
- Edges: `CONTAINS(5)` edges with src↔dst flipped; type-7 embeddings generated at runtime
  ```python
  rev_contains_ei = edge_index[:, contains_mask].flip(0)   # [2, E_contains]
  rev_type_ids    = torch.full((n_rev,), 7, dtype=torch.long, device=...)
  rev_contains_ea = edge_embedding(rev_type_ids)            # [E_contains, 32]
  ```
- Direction: CFG_NODE → parent FUNCTION (propagates Phase-2-enriched CFG signal up to function nodes)
- `add_self_loops=False` — only CFG→function aggregation wanted
- `[N, 128] → [N, 128]` + residual from Phase 2
- Zero-message behaviour: FUNCTION nodes with no CFG children receive no Phase 3 messages; residual is a no-op (`x = x + dropout(0)`). This is correct — do not add self-loops to "fix" this.
- `phase_norm[2](x)` applied after residual
- Output: `[N, 128]`, appended to `_live` for JK

### 3.4 JK Attention Aggregation

`_JKAttention` module (custom implementation, not PyG's `JumpingKnowledge`).

**Why custom:** explicit gradient-flow analysis; no LSTM overhead; output dimension is exactly `hidden_dim`; per-phase attention weights are inspectable for monitoring.

**Algorithm:**
```python
stacked = stack([phase1_out, phase2_out, phase3_out], dim=1)  # [N, 3, 128]
scores  = Linear(128, 1, bias=False)(stacked)                  # [N, 3, 1]
weights = softmax(scores, dim=1)                               # [N, 3, 1]
output  = (weights * stacked).sum(dim=1)                       # [N, 128]
```

Side effect: `self.last_weights = weights.squeeze(-1).mean(0).detach()` — cached as `[3]` tensor for per-epoch monitoring without an extra forward pass.

**Critical — live intermediates:** `_live` is collected **without `.detach()`**. JK attention weights receive gradients through these tensors. The `_intermediates` diagnostic dict uses `.detach().clone()` for backward compatibility with tests — do not confuse the two.

Non-negotiable gate: `test_jk_gradient_flow` verifies that `jk.attn` receives non-zero gradients after a backward pass.

### 3.5 Per-Phase LayerNorm

```python
self.phase_norm = nn.ModuleList([LayerNorm(128), LayerNorm(128), LayerNorm(128)])
```

Applied after each phase's residual connection, before appending to `_live`. Without LayerNorm, Phase 1 (two conv layers + residual) produces higher norms than Phases 2+3 (one layer each), causing JK softmax to ignore later phases. Always present regardless of `use_jk` setting.

### 3.6 Input/Output Contract

```
Input:
  x:          [N, 12]   float32   node features
  edge_index: [2, E]    int64     graph connectivity
  batch:      [N]       int64     node → graph index
  edge_attr:  [E]       int64     edge type IDs in [0, 7] (7 = runtime REVERSE_CONTAINS)
              Required when use_edge_attr=True; raises ValueError if None

Output (return_intermediates=False, default):
  node_embeddings: [N, 128]
  batch:           [N]

Output (return_intermediates=True):
  node_embeddings: [N, 128]
  batch:           [N]
  intermediates:   {"after_phase1": [N,128], "after_phase2": [N,128], "after_phase3": [N,128]}
                   (detached — diagnostic only, not used for gradients)
```

Runtime guards:
- `use_edge_attr=True` with `edge_attr=None` → raises `ValueError` immediately (silent failure would zero out Phase 2)
- `edge_index.max() >= x.shape[0]` → raises `ValueError` (OOB node index produces silent wrong attention or CUDA illegal-memory-access)

---

## 4. TransformerEncoder

Source: `ml/src/models/transformer_encoder.py`

CodeBERT (`microsoft/codebert-base`) with LoRA adapters injected into targeted attention projections. All original CodeBERT weights are frozen; only LoRA matrices are trainable.

### 4.1 LoRA Configuration

```python
LoraConfig(
    r              = 16,
    lora_alpha     = 32,          # effective scale = alpha/r = 2.0
    target_modules = ["query", "value"],   # all 12 transformer layers
    lora_dropout   = 0.1,
    bias           = "none",
    task_type      = "FEATURE_EXTRACTION",
)
```

Applied to all 12 attention layers of CodeBERT. Each targeted projection receives two injected matrices A `[768, r]` and B `[r, 768]`. Forward pass: `W_frozen @ x + (B @ A) @ x × (alpha/r)`. Gradients flow only through A and B.

**Why no `torch.no_grad()` wrapping `self.bert()`:** peft's `get_peft_model()` marks all original weights with `requires_grad=False`. PyTorch does not build backward nodes for ops with all-frozen inputs. Wrapping the entire call in `no_grad()` would also cut gradients to LoRA A/B matrices — silently killing LoRA training. The gradient split is handled by peft internally.

`peft` is a **hard requirement**. Missing it raises `RuntimeError` at import time (not a silent fallback with 0 trainable params).

`TRANSFORMERS_OFFLINE=1` must be set at the shell level before importing transformers. CodeBERT weights are loaded from the local model cache; the environment variable prevents network access.

### 4.2 Parameter Counts

| Component | Trainable | Frozen |
|-----------|-----------|--------|
| LoRA A+B matrices (12 layers × Q+V) | 589,824 | — |
| CodeBERT backbone | — | 124,645,632 |

### 4.3 Input/Output Contract

```
Input:
  input_ids:      [B, 512]   int64    CodeBERT token IDs
  attention_mask: [B, 512]   int64    1=real token, 0=PAD

Output:
  last_hidden_state: [B, 512, 768]   float32   ALL token embeddings
```

The full `[B, 512, 768]` tensor (not just the CLS token) is returned so CrossAttentionFusion can compute per-node attention over all 512 token positions before pooling.

---

## 5. CrossAttentionFusion

Source: `ml/src/models/fusion_layer.py`

Bidirectional cross-attention between GNN node embeddings and CodeBERT token embeddings. Both modalities are projected to a common attention dimension, mutually enriched via two MHA passes, then pooled and concatenated. Output dimension is **128 — LOCKED**.

### 5.1 Sub-module Parameter Breakdown

Total trainable: **821,888** (all parameters are trainable).

| Sub-module | Config | Params |
|------------|--------|--------|
| `node_proj` | `Linear(128, 256)` | 33,024 |
| `token_proj` | `Linear(768, 256)` | 196,864 |
| `node_to_token` | `MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)` | 263,168 |
| `token_to_node` | `MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)` | 263,168 |
| `output_proj` | `Linear(256 × 2=512, 128)` + ReLU + Dropout | 65,664 |
| **Total** | | **821,888** |

Constructor signature:
```python
CrossAttentionFusion(
    node_dim   = 128,    # must match GNNEncoder.hidden_dim
    token_dim  = 768,    # CodeBERT output dimension per token
    attn_dim   = 256,    # common projection dim; must be divisible by num_heads
    num_heads  = 8,      # attn_dim / num_heads = 32 dims per head
    output_dim = 128,    # LOCKED
    dropout    = 0.1,
)
```

### 5.2 Forward Pass

```
Input:
  node_embs:      [N, 128]       all nodes across the batch (not padded)
  batch:          [N]            node → graph index
  token_embs:     [B, 512, 768]  all token embeddings
  attention_mask: [B, 512]       1=real token, 0=PAD
```

Steps:

1. **Project to common attention space**
   ```
   nodes_proj  = node_proj(node_embs)    → [N, 256]
   tokens_proj = token_proj(token_embs)  → [B, 512, 256]
   ```

2. **Pad nodes to uniform length per batch**
   ```
   padded_nodes, node_real_mask = to_dense_batch(nodes_proj, batch)
   → padded_nodes [B, max_nodes, 256],  node_real_mask [B, max_nodes] (True=real)
   node_padding_mask  = ~node_real_mask        # True=pad  (MHA convention)
   token_padding_mask = (attention_mask == 0)  # True=PAD
   ```

3. **Node → Token cross-attention** (every node queries all 512 tokens)
   ```
   Q=padded_nodes [B, max_nodes, 256]   K=V=tokens_proj [B, 512, 256]
   key_padding_mask=token_padding_mask
   need_weights=False   # skips weight materialisation; uses fused efficient-attn kernel
   → enriched_nodes [B, max_nodes, 256]
   enriched_nodes *= node_real_mask.float().unsqueeze(-1)   # zero out pad positions
   ```

4. **Token → Node cross-attention** (every token queries all graph nodes)
   ```
   Q=tokens_proj [B, 512, 256]   K=V=padded_nodes [B, max_nodes, 256]
   key_padding_mask=node_padding_mask
   need_weights=False
   → enriched_tokens [B, 512, 256]
   ```

5. **Masked mean pooling**
   ```
   pooled_nodes  = masked_mean(enriched_nodes,  node_real_mask)   → [B, 256]
   pooled_tokens = masked_mean(enriched_tokens, attention_mask)   → [B, 256]
   ```

6. **Concatenate and project**
   ```
   fused  = cat([pooled_nodes, pooled_tokens], dim=1)  → [B, 512]
   output = output_proj(fused)                          → [B, 128]
   ```

```
Output: [B, 128]   fused contract embedding   (LOCKED output_dim)
```

Device assertion: raises `RuntimeError` at forward time if `node_embs` and `token_embs` are on different devices.

---

## 6. Three-Eye Classifier (SentinelModel)

Source: `ml/src/models/sentinel_model.py`

The top-level model. Three parallel "eyes" each produce a 128-dim opinion vector; they are concatenated and passed through a linear classifier. No Sigmoid is applied inside the model.

### 6.1 GNN Eye

Pool over function-declaration nodes only. After Phase 3 reverse-CONTAINS aggregation, these nodes carry aggregated CFG execution-order signal. Pooling over all nodes was dominated by `CFG_NODE_RETURN` (77% of CFG node mass in the dataset), drowning the CALL/WRITE/COND signal.

**Pooling target type IDs:** `{FUNCTION=1, MODIFIER=2, FALLBACK=4, RECEIVE=5, CONSTRUCTOR=6}`

```python
node_type_ids = (graphs.x[:, 0].float() * 12.0).round().long()
func_mask = sum(node_type_ids == tid for tid in {1,2,4,5,6})  # bool [N]
```

**Ghost graph fallback:** if a graph has no function-level nodes (interface-only contract, ghost graph), include ALL its nodes in the pool. This prevents `global_max/mean_pool` from silently dropping that graph and returning `B-k` outputs instead of `B`.

```python
# Detect graphs with no function nodes; add their nodes to pool_mask
graph_has_func[batch[func_mask]] = True
fallback_mask = ~graph_has_func[batch]
pool_mask = func_mask | fallback_mask
```

**Projection:**
```
global_max_pool(pool_embs, pool_batch)   → [B, 128]
global_mean_pool(pool_embs, pool_batch)  → [B, 128]
cat([max, mean])                         → [B, 256]
gnn_eye_proj: Linear(256,128) + ReLU + Dropout(0.3)  → [B, 128]   (32,896 params)
```

### 6.2 Transformer Eye

```
token_embs[:, 0, :]                                    → [B, 768]   (CLS token)
transformer_eye_proj: Linear(768,128) + ReLU + Dropout(0.3)  → [B, 128]   (98,432 params)
```

CLS is CodeBERT's full-sequence summary via 12-layer bidirectional attention over all 512 positions. It is order-aware and distinct from the masked-mean pool used inside fusion.

### 6.3 Fused Eye

```
CrossAttentionFusion(node_embs, batch, token_embs, attention_mask)  → [B, 128]
```

Encodes joint evidence that neither modality holds alone. A `withdraw()` function node can directly attend to `call.value` and `transfer` tokens before pooling loses that granularity.

### 6.4 Main Classifier

```
cat([gnn_eye, transformer_eye, fused_eye])  → [B, 384]
Linear(384, 10)                             → [B, 10]   raw logits   (3,850 params)
```

**No Sigmoid inside the model.** Applied externally:
- During training: `BCEWithLogitsLoss` (numerically stable, fuses sigmoid + BCE)
- During inference: `torch.sigmoid(logits)` → per-class probabilities; apply per-class thresholds from `_thresholds.json`

For `num_classes=1`, logits are squeezed: `[B, 1] → [B]`.

### 6.5 Auxiliary Heads

Training-only mechanism to prevent one eye from dominating and causing the others' gradients to vanish.

```python
aux_gnn         = Linear(128, 10)   # applied to gnn_eye         (1,290 params)
aux_transformer = Linear(128, 10)   # applied to transformer_eye (1,290 params)
aux_fused       = Linear(128, 10)   # applied to fused_eye       (1,290 params)
# Total aux: 3,870 params
```

Loss formula:
```
loss = main_loss + 0.3 × (loss_gnn + loss_transformer + loss_fused)
```

`λ = 0.3` keeps each eye's gradient signal alive even if the main classifier assigns low weight to that eye.

Activation: `forward(return_aux=True)` returns `(logits, {"gnn": ..., "transformer": ..., "fused": ...})`. Default is `return_aux=False` — zero inference overhead.

### 6.6 Full Forward Pass

```python
def forward(
    graphs:         Batch,           # PyG Batch — batched contract graphs
    input_ids:      Tensor,          # [B, 512]
    attention_mask: Tensor,          # [B, 512]
    return_aux:     bool = False,
) -> Tensor | tuple[Tensor, dict]:
```

Data flow summary:
```
graphs  → GNNEncoder  → node_embs [N, 128]
                      ↓                  ↓
             GNN eye pool          CrossAttentionFusion ←── token_embs [B, 512, 768]
             + projection                                        ↑
                ↓ [B, 128]              ↓ [B, 128]     TransformerEncoder
                                                          + CLS extraction
                                                              ↓ [B, 128]
                         cat([gnn_eye, fused_eye, tf_eye]) → [B, 384]
                                       ↓
                              Linear(384, 10) → logits [B, 10]
```

---

## 7. Total Parameter Summary

| Component | Trainable | Frozen |
|-----------|-----------|--------|
| GNNEncoder | 70,272 | 0 |
| TransformerEncoder (LoRA only) | 589,824 | 124,645,632 |
| CrossAttentionFusion | 821,888 | 0 |
| `gnn_eye_proj` (Linear 256→128 + ReLU + Dropout) | 32,896 | 0 |
| `transformer_eye_proj` (Linear 768→128 + ReLU + Dropout) | 98,432 | 0 |
| `classifier` (Linear 384→10) | 3,850 | 0 |
| `aux_heads` (3 × Linear 128→10) | 3,870 | 0 |
| **Total** | **1,621,032** | **124,645,632** |

Effective trainable fraction: 1.3% of total weights. The model achieves strong generalisation through the frozen 124M CodeBERT representation base and targeted LoRA adaptation.

---

## 8. Checkpoint Format

Saved with `torch.save(checkpoint_dict, path)`.

**Top-level keys:**

| Key | Type | Description |
|-----|------|-------------|
| `model` | `state_dict` | Full model state (frozen CodeBERT weights + LoRA matrices + all other params) |
| `optimizer` | `state_dict` | Optimizer state (AdamW with per-group LRs) |
| `scheduler` | `state_dict` | LR scheduler state |
| `epoch` | `int` | Last completed epoch |
| `best_f1` | `float` | Best macro-F1 (threshold-tuned) seen so far |
| `patience_counter` | `int` | Early stopping counter |
| `model_version` | `str` | `"v5.2"` |
| `config` | `dict` | Full `TrainConfig` serialised as dict |

**Loading policy:**

| File type | `weights_only` | `safe_globals` required |
|-----------|---------------|------------------------|
| Graph `.pt` files | `True` | `Data, DataEdgeAttr, DataTensorAttr, GlobalStorage` |
| Checkpoint `.pt` files | `False` | N/A (LoRA state dict contains peft objects) |

Attempting `weights_only=True` on a checkpoint will raise `UnpicklingError` because LoRA state dicts contain peft class instances.

---

## 9. Inference Notes

**Sigmoid:** `torch.sigmoid(logits)` must be applied externally. Per-class thresholds are stored in a companion `_thresholds.json` file.

**Threshold application:**
```python
probs = torch.sigmoid(logits)           # [B, 10]
preds = probs > thresholds[class_idx]   # [B, 10] bool
```

**Auxiliary heads at inference:** pass `return_aux=False` (default). Auxiliary heads add no overhead and their weights do not affect the main logits.

**Feature schema version:** inference cache keys include `FEATURE_SCHEMA_VERSION = "v3"` as a suffix. If the schema changes, bump this constant to invalidate stale cached graph/token tensors.

**TRANSFORMERS_OFFLINE:** must be set to `1` at the shell level before importing `transformers`. CodeBERT weights are loaded from the local model cache at `microsoft/codebert-base`.

**AMP / BF16:** when using automatic mixed precision, recover `type_id` with `.float()` before the `* 12.0` multiplication:
```python
node_type_ids = (graphs.x[:, 0].float() * 12.0).round().long()
```
BF16 round-trip precision loss on the `/ 12.0` normalised value can shift type IDs by ±1 without the explicit `.float()` cast.

---

## 10. Locked Constants Reference Table

All constants below are authoritative. Changing any of them has the consequence listed in the "Breaks" column.

| Constant | Value | Source | Breaks if changed |
|----------|-------|--------|-------------------|
| `MODEL_VERSION` | `"v5.2"` | `sentinel_model.py` | Checkpoint compatibility |
| `ARCHITECTURE` | `"three_eye_v5"` | `sentinel_model.py` | Checkpoint compatibility |
| `FEATURE_SCHEMA_VERSION` | `"v3"` | `graph_schema.py` | Cache invalidation (must bump) |
| `NODE_FEATURE_DIM` | `12` | `graph_schema.py` | Requires full re-extraction + retrain |
| `NUM_CLASSES` | `10` | `graph_schema.py` | Requires retrain; append-only |
| `NUM_EDGE_TYPES` | `8` | `graph_schema.py` | Requires GNN embedding table resize |
| `NUM_NODE_TYPES` | `13` | `graph_schema.py` | Requires re-extraction + retrain |
| `fusion_output_dim` | `128` | `fusion_layer.py` | Breaks ZKML proxy circuit |
| `MAX_TOKEN_LENGTH` | `512` | `graph_schema.py` / tokenizer | Breaks token cache + OOM risk |
| `GNNEncoder.hidden_dim` | `128` | `gnn_encoder.py` | Breaks fusion input + all eye projections |
| `GNNEncoder.heads` (Phase 1) | `8` | `gnn_encoder.py` | Changes conv1/conv2 param count |
| `GNNEncoder.heads` (Phase 2+3) | `1` | `gnn_encoder.py` | Breaks directional signal; see Phase 2 note |
| `GATConv Phase 2 add_self_loops` | `False` | `gnn_encoder.py` | Cancels CONTROL_FLOW directional signal |
| `GATConv Phase 3 add_self_loops` | `False` | `gnn_encoder.py` | Dilutes reverse-CONTAINS aggregation |
| `CrossAttentionFusion.attn_dim` | `256` | `fusion_layer.py` | Changes all MHA param counts |
| `CrossAttentionFusion.num_heads` | `8` | `fusion_layer.py` | `attn_dim` must remain divisible |
| `LoRA r` | `16` | `transformer_encoder.py` | Changes trainable param count |
| `LoRA alpha` | `32` | `transformer_encoder.py` | Changes effective scale (alpha/r=2.0) |
| `LoRA target_modules` | `["query","value"]` | `transformer_encoder.py` | Changes what CodeBERT adapts |
| `edge_attr shape` | `[E]` 1-D int64 | `graph_schema.py` | `nn.Embedding` crashes on `[E,1]` |
| `REVERSE_CONTAINS ID` | `7` | `graph_schema.py` | Runtime edge generation in Phase 3 |
| `type_id normalisation` | `float(id) / 12.0` | `graph_extractor.py` | Raw 0–12 dominates dot product |
| `aux loss λ` | `0.3` | `sentinel_model.py` | Eye gradient balance |
| `classifier dropout` | `0.3` | `sentinel_model.py` | Regularisation |
| `GNN dropout` | `0.2` | `gnn_encoder.py` | Regularisation |
