# models — SENTINEL Model Architecture

Four modules, one forward pass. Architecture is **locked** — do not change shapes or layer counts.

---

## Module Map

| File | Class | Role |
|---|---|---|
| `gnn_encoder.py` | `GNNEncoder` | Contract graph → structural embedding `[B, 64]` |
| `transformer_encoder.py` | `TransformerEncoder` | Solidity source → semantic embedding `[B, 768]` |
| `fusion_layer.py` | `FusionLayer` | Concat + MLP: `[B, 832]` → `[B, 64]` |
| `sentinel_model.py` | `SentinelModel` | Orchestrates all three + classification head |

---

## GNNEncoder (`gnn_encoder.py`)

Graph Attention Network that reads contract AST/CFG structure.

**Architecture:**

```
Input: x [N, 8]  (N = total nodes across batch, 8 node features)

conv1: GATConv(8  → 8,  heads=8, concat=True)  →  [N, 64]   ReLU  Dropout(0.2)
conv2: GATConv(64 → 8,  heads=8, concat=True)  →  [N, 64]   ReLU  Dropout(0.2)
conv3: GATConv(64 → 64, heads=1, concat=False) →  [N, 64]

global_mean_pool(x, batch)  →  [B, 64]   (one embedding per contract)
```

**Forward signature:**
```python
GNNEncoder.forward(x, edge_index, batch) -> Tensor[B, 64]
#   x:          [N, 8]  — node feature matrix (all graphs in batch, concatenated)
#   edge_index: [2, E]  — directed edges (COO format)
#   batch:      [N]     — maps each node to its graph index, e.g. [0,0,0,1,1,2,...]
```

**Receptive field:** 3 hops — each node integrates information from nodes up to 3 edges away.

**Trainable parameters:** ~33K

---

## TransformerEncoder (`transformer_encoder.py`)

Wraps `microsoft/codebert-base` as a **frozen** feature extractor. No weights are updated during training.

**Architecture:**

```
Input: input_ids [B, 512], attention_mask [B, 512]

CodeBERT (125M params, all frozen)
   └── BERT encoder × 12 layers
   └── last_hidden_state: [B, 512, 768]

CLS token extraction: [:, 0, :]  →  [B, 768]
```

**Forward signature:**
```python
TransformerEncoder.forward(input_ids, attention_mask) -> Tensor[B, 768]
#   input_ids:      [B, 512]  long  — CodeBERT token IDs
#   attention_mask: [B, 512]  long  — 1=real token, 0=padding
```

**Why frozen:** CodeBERT already understands code syntax and semantics from pre-training on GitHub.
Fine-tuning it would require ~3× more GPU memory and risk catastrophic forgetting.
The `torch.no_grad()` block in `forward()` prevents computation graph construction — saves memory at runtime.

**Trainable parameters:** 0 (CodeBERT's 124,645,632 params have `requires_grad=False`)

---

## FusionLayer (`fusion_layer.py`)

Concatenates GNN and Transformer embeddings, then compresses through a two-layer MLP.

**Architecture:**

```
gnn_out:         [B, 64]
transformer_out: [B, 768]

torch.cat(dim=1)   →  [B, 832]

Linear(832 → 256)
ReLU
Dropout(0.3)
Linear(256 → 64)
ReLU               →  [B, 64]
```

**Forward signature:**
```python
FusionLayer.forward(gnn_out, transformer_out) -> Tensor[B, 64]
#   gnn_out:         [B, 64]
#   transformer_out: [B, 768]
# Raises ValueError if batch dimensions don't match.
```

**Trainable parameters:** ~213K (both Linear layers)

---

## SentinelModel (`sentinel_model.py`)

Top-level model — composes all three encoders and adds the classification head.

**Full forward pass:**

```
graphs (PyG Batch), input_ids [B, 512], attention_mask [B, 512]
   │                    │                    │
   ▼                    └────────────────────┘
GNNEncoder                  TransformerEncoder
   │  [B, 64]                    │  [B, 768]
   └──────────── FusionLayer ────┘
                    │  [B, 64]
              Linear(64 → 1)
                 Sigmoid        →  [B, 1]
               squeeze(1)       →  [B]   values in [0, 1]
```

**Forward signature:**
```python
SentinelModel.forward(graphs, input_ids, attention_mask) -> Tensor[B]
#   graphs:         PyG Batch  — batched contract graphs from dual_path_collate_fn
#   input_ids:      [B, 512]   long
#   attention_mask: [B, 512]   long
# Returns: vulnerability scores [B], dtype float32, values in [0, 1]
```

**Critical:** output is **already sigmoid-activated**. Use `BCELoss`, not `BCEWithLogitsLoss`.

**Instantiation:**
```python
model = SentinelModel()   # all defaults match training config
model = model.to("cuda")
```

Default constructor arguments match the locked architecture:
```python
SentinelModel(
    gnn_dim=64,
    transformer_dim=768,
    fusion_output_dim=64,
    dropout=0.3,
)
```

**Loading a checkpoint:**
```python
checkpoint = torch.load("ml/checkpoints/sentinel_best.pt",
                        map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()
```

**Parameter summary** (call `model.parameter_summary()` to log):

| Sub-module | Trainable | Frozen |
|---|---|---|
| GNNEncoder | ~33K | 0 |
| TransformerEncoder | 0 | 124,645,632 |
| FusionLayer | ~213K | 0 |
| Classifier | ~65 | 0 |
| **Total** | **239,041** | **124,645,632** |
