# models — SENTINEL Model Architecture (v8 + GraphCodeBERT + GNN Prefix)

Four modules, one forward pass. Shapes and layer counts are **locked** — changes require full retrain.

---

## Module Map

| File | Class | Role |
|------|-------|------|
| `gnn_encoder.py` | `GNNEncoder` | Contract graph → structural embedding `[B, 256]` |
| `transformer_encoder.py` | `TransformerEncoder` | Source code → semantic embedding `[B, 768]` with optional GNN prefix |
| `fusion_layer.py` | `CrossAttentionFusion` | Bidirectional cross-attention → `[B, 128]` |
| `sentinel_model.py` | `SentinelModel` | Three-eye orchestrator + classification head → `[B, 10]` |

---

## GNNEncoder (`gnn_encoder.py`)

7-layer Graph Attention Network, three phases, Jumping Knowledge (JK) attention aggregation.

```
Phase 1 — Structural + CONTAINS (layers 1+2)
  GAT over edge types 0–5, 8 heads, add_self_loops=True
  LayerNorm after phase

Phase 2 — CF + CALL_ENTRY + RETURN_TO directed (layers 3+4+5, 3 hops)
  conv3:  CF(6), 1 head
  conv3b: CALL_ENTRY(8)
  conv3c: RETURN_TO(9)
  add_self_loops=False  ← directional signal preserved
  LayerNorm after phase

Phase 3 — REVERSE_CONTAINS type-7 (layers 6+7)
  1 head; REVERSE_CONTAINS generated at runtime from CONTAINS edges
  LayerNorm after phase

JK attention aggregation over all 7 layer outputs → hidden_dim=256
Edge embedding: Embedding(11, 64) concatenated to each message
```

**Forward signature:**
```python
GNNEncoder.forward(x, edge_index, edge_attr, batch) -> Tensor[B, 256]
#   x:          [N, 11]  float32 — node feature matrix
#   edge_index: [2, E]   int64   — directed edges (COO format)
#   edge_attr:  [E]      int64   — edge type indices (1-D)
#   batch:      [N]      int64   — maps each node to its graph index
```

Also exports `select_prefix_nodes(x, batch, k) -> Tensor[B, K, 256]` for GNN prefix injection.

**NodeType IntEnum** (`graph_schema.py`): 13 types — CONTRACT, FUNCTION, PARAMETER, VARIABLE, EVENT, MODIFIER, STRUCT, ENUM, STATE_VARIABLE, BLOCK, UNCHECKED_BLOCK, TMP_VARIABLE, ERROR. Always use `NodeType.FUNCTION` etc., never raw integers.

`STRUCTURAL_PREFIX_TYPES = frozenset({FUNCTION, MODIFIER, CONSTRUCTOR, FALLBACK, RECEIVE})` — declaration nodes eligible for prefix selection.

---

## TransformerEncoder (`transformer_encoder.py`)

`microsoft/graphcodebert-base` (124M params) + LoRA adapters.

**LoRA configuration:**
- Base model frozen; LoRA r=16, α=32 on Q+V of all 12 layers
- Only LoRA A/B matrices (trainable ~3.1M params) receive gradients

**Standard path** (warmup epochs 0–14, or when `gnn_prefix_nodes=None`):
```
Input: input_ids [B, 4, 512], attention_mask [B, 4, 512]

Each of 4 windows processed independently:
  GraphCodeBERT → last_hidden_state [B, 512, 768]
  WindowAttentionPooler → CLS at position prefix_k (default 0 without prefix)
  → [B, 768] per window

Mean over 4 windows → [B, 768]
```

**Prefix path** (epoch ≥ gnn_prefix_warmup_epochs):
```
Input: input_ids [B, 4, 512], attention_mask [B, 4, 512],
       gnn_prefix_nodes [B, K, 768]

Uses inputs_embeds instead of input_ids:
  prefix_embeds [B, K, 768] + code_embeds [B, code_budget, 768]
  = total sequence length K + code_budget = 512

Position IDs:
  Prefix: position_id = 1  (RoBERTa padding slot)
  Code:   position_ids = 3..3+code_budget-1

WindowAttentionPooler: CLS at i*window_size + prefix_k
  (offset shifts CLS extraction by prefix length)
```

**Forward signature:**
```python
TransformerEncoder.forward(input_ids, attention_mask, gnn_prefix_nodes=None) -> Tensor[B, 768]
#   input_ids:        [B, 4, 512]    long
#   attention_mask:   [B, 4, 512]    long
#   gnn_prefix_nodes: [B, K, 768] | None  — projected GNN nodes
```

---

## GNN Prefix Injection (`sentinel_model.py`)

Bridges GNN and Transformer by injecting structural context as soft prefix tokens.

```python
# Components on SentinelModel:
gnn_to_bert_proj      = Linear(256, 768)      # projects GNN hidden → BERT embedding dim
prefix_type_embedding = Embedding(5, 768)     # type-specific bias per STRUCTURAL_PREFIX_TYPES
```

**select_prefix_nodes():**
Priority: CONSTRUCTOR > FALLBACK > RECEIVE > MODIFIER > FUNCTION
Returns top-K=48 declaration nodes per graph in the batch.

**Warmup suppression:**
- Epochs 0..(warmup-1): `gnn_prefix_nodes=None` — TransformerEncoder uses standard path
- `gnn_to_bert_proj` receives zero gradient during warmup
- Epoch `gnn_prefix_warmup_epochs` (default 15): projection fires from random init

**Inference:** `predictor.py` sets `model._current_epoch = 9999` so prefix is always active regardless of checkpoint's warmup setting.

---

## CrossAttentionFusion (`fusion_layer.py`)

Bidirectional cross-attention between graph nodes and token sequence.

```
1. Project nodes [N, 256] → [N, 256]
2. LayerNorm(768) + project tokens [B, 512, 768] → [B, 512, 256]
3. _scatter_to_dense → [B, 1024, 256]  (max_nodes=1024; compile-safe, zero graph breaks)
4. Node→Token cross-attention → enriched_nodes [B, 1024, 256]
5. Token→Node cross-attention → enriched_tokens [B, 512, 256]
6. Masked mean pooling of real nodes  → [B, 256]
7. Masked mean pooling of real tokens → [B, 256]
8. Concat [B, 512] → Linear(512, 128) + ReLU → [B, 128]
```

**output_dim=128 is LOCKED** — the ZKML proxy MLP (M2) input depends on this exact shape.

`_scatter_to_dense` replaces PyG's `to_dense_batch` to eliminate `GuardOnDataDependentSymNode` compile graph breaks. Zero graph breaks confirmed in production.

**Forward signature:**
```python
CrossAttentionFusion.forward(node_emb, edge_index, batch, token_emb) -> Tensor[B, 128]
```

---

## SentinelModel (`sentinel_model.py`)

Three-eye classifier orchestrating all sub-modules.

**Full forward pass:**
```
graphs (PyG Batch), input_ids [B, 4, 512], attention_mask [B, 4, 512]
        │                        │
        ▼                        ▼
   GNNEncoder              select_prefix_nodes → gnn_to_bert_proj
   [B, 256]                [B, K, 768]  (None during warmup)
        │                        │
        │           TransformerEncoder (with prefix)
        │                  [B, 768]
        │                        │
        └── CrossAttentionFusion ┘
                [B, 128]

GNN eye:    max_pool + mean_pool over FUNCTION nodes → [B, 512] → Linear → [B, 128]
TF eye:     pooled token emb → Linear(768, 128) → [B, 128]
Fused eye:  CrossAttentionFusion → [B, 128]

Concat [B, 384] → Linear(384, 192) → GELU → Linear(192, 10) → [B, 10] logits
```

**Auxiliary heads** (training only): one `Linear(128, 10)` per eye for auxiliary loss. Disabled in `model.eval()` mode.

**Forward signature:**
```python
SentinelModel.forward(graphs, input_ids, attention_mask) -> Tensor[B, 10]
#   graphs:         PyG Batch  — from dual_path_collate_fn
#   input_ids:      [B, 4, 512]  long
#   attention_mask: [B, 4, 512]  long
# Returns: raw logits [B, 10] — apply sigmoid for probabilities
```

Output is **raw logits** — apply `sigmoid()` for probabilities, use `BCEWithLogitsLoss` during training.

**Instantiation:**
```python
model = SentinelModel(
    gnn_hidden_dim=256,
    gnn_layers=7,
    lora_r=16,
    lora_alpha=32,
    gnn_prefix_k=48,                  # 0 = disabled
    gnn_prefix_warmup_epochs=15,
)
model = model.to("cuda")
model._current_epoch = 0             # set before each epoch in trainer
```

**Loading a checkpoint:**
```python
ckpt = torch.load("ml/checkpoints/gcb_best.pt", map_location=device, weights_only=False)
# Strip torch.compile prefix from keys:
state = {k.replace("._orig_mod.", "."): v for k, v in ckpt["model_state_dict"].items()}
model.load_state_dict(state)
model.eval()
model._current_epoch = 9999          # ensures prefix always active at inference
```

**Parameter summary (approximate):**

| Sub-module | Trainable | Frozen |
|------------|-----------|--------|
| GNNEncoder (7-layer GAT) | ~2.5M | 0 |
| GraphCodeBERT base | 0 | 124M |
| LoRA adapters (Q+V × 12) | ~3.1M | 0 |
| gnn_to_bert_proj + prefix_type_embedding | ~200K | 0 |
| CrossAttentionFusion | ~1.5M | 0 |
| Classifier + eye projectors | ~300K | 0 |
| **Total** | **~7.6M** | **~124M** |
