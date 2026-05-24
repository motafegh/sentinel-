# models — SENTINEL Model Architecture (v8 + GraphCodeBERT + GNN Prefix)

Four modules, one forward pass. Shapes and layer counts are **locked** — changes require full retrain.

**Current architecture:** 8-layer GNN (2+3+3 phases), Flash Attention 2, IMP improvements (G1, G2, G3, M1, M3, C2, #26)

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

8-layer Graph Attention Network, three phases (2+3+3), Jumping Knowledge (JK) attention aggregation.

```
Phase 1 — Structural + CONTAINS (layers 1+2)
  GAT over edge types 0–5, 8 heads, add_self_loops=True
  IMP-G2: input_proj skip connection (Linear(11,256)) added before relu in Layer 1
         Prevents raw feature loss when GAT attention weights start near-uniform
  LayerNorm after phase

Phase 2 — CFG + ICFG directed (layers 3+4+5)
  IMP-G1: each layer processes a DISTINCT edge subset (vs same cfg_mask before)
  conv3:  CONTROL_FLOW(6) only — intra-function execution ordering
  conv3b: CALL_ENTRY(8) + RETURN_TO(9) only — cross-function call structure
  conv3c: CF(6)+CALL_ENTRY(8)+RETURN_TO(9) joint — integration layer
  add_self_loops=False (CRITICAL — self-loops cancel directional signal)
  heads=1, concat=False
  LayerNorm after phase

Phase 3 — Bidirectional CONTAINS (layers 6+7+8)
  conv4:  REVERSE_CONTAINS up — CFG→FUNCTION (Phase 2 signal rises)
  conv4b: REVERSE_CONTAINS up — second hop (multi-function patterns)
  conv4c: CONTAINS down (IMP-G3) — FUNCTION→CFG, distributes enriched
          FUNCTION context back to CFG children. All nodes carry Phase 3 depth after this.
  1 head
  LayerNorm after phase

JK attention aggregation over all 8 layer outputs → hidden_dim=256
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
- Only LoRA A/B matrices (trainable ~590K params) receive gradients
- Flash Attention 2 support (falls back to SDPA if unavailable)

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
       gnn_prefix_nodes [B, K, 768], gnn_prefix_counts [B]

Uses inputs_embeds instead of input_ids:
  prefix_embeds [B, K, 768] + code_embeds [B, code_budget, 768]
  = total sequence length K + code_budget = 512

Position IDs:
  Prefix: position_id = 1  (RoBERTa padding slot)
  Code:   position_ids = 3..3+code_budget-1

IMP-M3: actual node count masking
  gnn_prefix_counts [B] tracks real (non-padded) nodes per graph
  Zero-padded prefix positions are masked in attention (95.5% of contracts fill all K slots)

WindowAttentionPooler: CLS at i*window_size + prefix_k
  (offset shifts CLS extraction by prefix length)
```

**Forward signature:**
```python
TransformerEncoder.forward(input_ids, attention_mask, gnn_prefix_nodes=None, gnn_prefix_counts=None, output_attentions=False) -> Tensor[B, 768]
#   input_ids:        [B, 4, 512]    long
#   attention_mask:   [B, 4, 512]    long
#   gnn_prefix_nodes: [B, K, 768] | None  — projected GNN nodes
#   gnn_prefix_counts: [B] | None  — real node counts (IMP-M3)
#   output_attentions: bool  — returns prefix_attn_mean when True (IMP-M2)
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
Secondary sort: FUNCTION nodes by external_call_count descending (IMP-M1)
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
2. token_norm LayerNorm(768) + project tokens [B, 512, 768] → [B, 512, 256] (BUG-C2 fix)
3. _scatter_to_dense → [B, 1024, 256]  (max_nodes=1024; compile-safe, zero graph breaks)
4. Node→Token cross-attention → enriched_nodes [B, 1024, 256]
5. Token→Node cross-attention → enriched_tokens [B, 512, 256]
6. Masked mean pooling of real nodes  → [B, 256]
7. Masked mean pooling of real tokens → [B, 256]
8. Concat [B, 512] → Linear(512, 128) + ReLU → [B, 128]
```

**output_dim=128 is LOCKED** — the ZKML proxy MLP (M2) input depends on this exact shape.

`_scatter_to_dense` replaces PyG's `to_dense_batch` to eliminate `GuardOnDataDependentSymNode` compile graph breaks. Zero graph breaks confirmed in production.

**Key improvements:**
- BUG-C2: `token_norm` LayerNorm before token projection prevents CodeBERT embeddings (L2 norm ~10-15) from dominating cross-attention dot products
- Fix #26: `need_weights=False` on both MHA calls saves ~12.6 MB VRAM per forward pass by skipping attention weight matrix materialization

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
    gnn_num_layers=8,                 # 2+3+3 phases (IMP-G3 added conv4c)
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
| GNNEncoder (8-layer GAT) | ~2.4M | 0 |
| GraphCodeBERT base | 0 | 124M |
| LoRA adapters (Q+V × 12) | ~590K | 0 |
| gnn_to_bert_proj + prefix_type_embedding | ~200K | 0 |
| CrossAttentionFusion | ~1.5M | 0 |
| Classifier + eye projectors | ~300K | 0 |
| **Total** | **~5.0M** | **~124M** |
