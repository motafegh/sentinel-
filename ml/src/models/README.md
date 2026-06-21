# ml/src/models — SENTINEL Model Architecture

Four neural network modules that implement the v8.1 four-eye smart contract vulnerability classifier.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `sentinel_model.py` | 670 | Top-level `SentinelModel` — assembles four eyes + classifier + auxiliary heads |
| `gnn_encoder.py` | 667 | `GNNEncoder` — 8-layer three-phase GAT with JK attention |
| `transformer_encoder.py` | 388 | `TransformerEncoder` — GraphCodeBERT + LoRA + prefix injection + `WindowAttentionPooler` |
| `fusion_layer.py` | 282 | `CrossAttentionFusion` — bidirectional cross-attention between nodes and tokens |
| `__init__.py` | 0 | Empty |

---

## Module Dependency Graph

```
sentinel_model.py
  |-- gnn_encoder.py         (GNNEncoder)
  |-- transformer_encoder.py (TransformerEncoder, WindowAttentionPooler)
  |-- fusion_layer.py        (CrossAttentionFusion)
  |-- graph_schema.py        (NODE_TYPES for type masks)
```

---

## sentinel_model.py

### SentinelModel

The top-level `nn.Module` that assembles the full forward pass.

**Constructor args:**
- `num_classes=10`, `fusion_output_dim=128`, `dropout=0.3`
- GNN: `gnn_hidden_dim=256`, `gnn_num_layers=8`, `gnn_heads=8`, `gnn_dropout=0.2`, `use_edge_attr=True`, `gnn_edge_emb_dim=64`, `gnn_use_jk=True`, `gnn_jk_mode='attention'`
- LoRA: `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.1`, `lora_target_modules=["query","value"]`
- Prefix: `gnn_prefix_k=0`, `gnn_prefix_warmup_epochs=15`
- Fusion: `fusion_max_nodes=2048`
- Ablation: `drop_complexity_feature=False`, `appnp_alpha=0.0`

**Forward pass:**
```python
def forward(self, graphs, input_ids, attention_mask, return_aux=False):
    # return_aux=False (inference): logits [B, num_classes]
    # return_aux=True  (training):  (logits, {"gnn", "transformer", "fused", "phase2", "jk_entropy"})
```

**Key methods:**
- `select_prefix_nodes()` — selects top-K declaration nodes by type priority + external_call_count
- `compute_prefix_attention_mean()` — diagnostic: mean code->prefix attention weight
- `parameter_summary()` — logs trainable vs frozen params per sub-module

**Module-level constants:**
- `_MAX_TYPE_ID = 13.0` — asserted at import to catch schema drift
- `_FUNC_TYPE_IDS` — frozenset of FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR type IDs
- `_CFG_TYPE_IDS` — frozenset of 5 CFG_NODE type IDs (8-12)
- `_CEI_TYPE_IDS` — frozenset of CALL+WRITE+CHECK (3 types, for aux_phase2)
- `_PREFIX_NODE_PRIORITY` — priority dict for prefix selection
- `_PREFIX_TYPE_IDX` — embedding index mapping (0-4)

---

## gnn_encoder.py

### GNNEncoder

Three-phase, 8-layer GAT encoder. Fixed at `SENTINEL_GNN_NUM_LAYERS = 8`.

**Architecture:**
- **Phase 1 (layers 1+2):** Structural edges (types 0-5), 8 heads, add_self_loops=True
  - Layer 1: `_GNN_IN_DIM(28)` -> `hidden_dim(256)` with IMP-G2 input_proj skip
  - Layer 2: `hidden_dim` -> `hidden_dim` with residual
- **Phase 2 (layers 3+4+5):** CFG+ICFG directed edges, 4 heads, add_self_loops=False
  - Layer 3 (conv3): CONTROL_FLOW only
  - Layer 4 (conv3b): CALL_ENTRY + RETURN_TO (+ EXTERNAL_CALL in v9)
  - Layer 5 (conv3c): CF + ICFG joint (integration layer)
- **Phase 3 (layers 6+7+8):** Bidirectional CONTAINS, 1 head
  - Layer 6 (conv4): REVERSE_CONTAINS up
  - Layer 7 (conv4b): REVERSE_CONTAINS up (second hop)
  - Layer 8 (conv4c): CONTAINS down (IMP-G3)

**Type embedding:** `Embedding(14, 16)` prepended to node features at runtime -> `_GNN_IN_DIM=28`.

**Edge embedding:** `Embedding(12, 64)` concatenated to edge features per message.

**JK Attention:** `_JKAttention` — learned softmax-weighted sum over 3 phase outputs with entropy tracking.

**Key design decisions:**
- Phase 2 self-loops MUST be False — they cancel directional control-flow signal
- Phase 3 uses REVERSE_CONTAINS (type 7) for upward and CONTAINS (type 5) for downward
- JK collects live (non-detached) tensors for gradient flow
- Per-phase LayerNorm prevents Phase 1 norm from dominating JK softmax
- `refresh_dtype_cache()` must be called after runtime dtype casts

---

## transformer_encoder.py

### TransformerEncoder

GraphCodeBERT with LoRA fine-tuning.

**Key facts:**
- Base model: `microsoft/graphcodebert-base` (124M params, frozen)
- LoRA: r=16, alpha=32 on Q+V of all 12 layers (~590K trainable)
- Flash Attention 2 preferred, SDPA fallback
- peft library is a hard requirement

**Forward:**
- Single-window `[B, L]` -> `[B, L, 768]`
- Multi-window `[B, W, L]` -> `[B, W*L, 768]` (flattened along seq dim)
- GNN prefix path: uses `inputs_embeds` with prefix at positions 0..K-1

**Word embedding access:** `_word_embeddings` property tries 3 known PEFT paths for version compatibility.

### WindowAttentionPooler

Extracts CLS of each window and combines via learned attention weights.
- Single-window fallback: returns CLS directly
- Multi-window: `cls_indices = arange(W) * window_size + prefix_k`

---

## fusion_layer.py

### CrossAttentionFusion

Bidirectional cross-attention between GNN nodes and CodeBERT tokens.

**Architecture:**
1. Project nodes `[N,256]` -> `[N,256]`
2. LayerNorm + project tokens `[B,512,768]` -> `[B,512,256]`
3. `_scatter_to_dense` -> static `[B,max_nodes,256]` (compile-safe, max_nodes=2048)
4. Node->Token cross-attention -> enriched_nodes
5. Token->Node cross-attention -> enriched_tokens
6. Masked mean pooling of real nodes -> `[B,256]`
7. Masked mean pooling of real tokens -> `[B,256]`
8. Concat `[B,512]` -> Linear + ReLU -> `[B,128]`

**Key fixes:**
- BUG-C2: `token_norm` LayerNorm before token projection
- Fix #8: Zero-out padded node positions after cross-attention
- Fix #26: `need_weights=False` saves ~12.6 MB VRAM
- Fix #6: Masked mean pooling (was plain mean)
- `_scatter_to_dense` replaces PyG's `to_dense_batch` for torch.compile compatibility

**output_dim=128 is LOCKED** -- ZKML proxy MLP depends on this.
