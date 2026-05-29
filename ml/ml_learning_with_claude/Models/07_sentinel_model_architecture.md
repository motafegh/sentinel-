# Models — Chunk 7: SentinelModel Architecture & `__init__`

> **File:** `ml/src/models/sentinel_model.py` — **lines 1–259**
> **What you'll learn:** Why three eyes, module-level constants as single source of truth, the complete `__init__` assembly, auxiliary heads and why they exist, and `parameter_summary`.
> **Time:** ~30 minutes
> **Interview relevance:** ML (multi-branch models, auxiliary losses, parameter counting), AI (model assembly, multi-label classification), MLOps (schema-derived constants, parameter auditing)

---

## Warm-Up Recall (from chunk 06)

One sentence each — no looking back:

1. What is the BUG-C2 `valid`-before-clamp fix in `_scatter_to_dense`, and what goes wrong without it?
2. Why does Fix #6 use masked mean pooling for tokens rather than plain mean?
3. `need_weights=False` on MHA — what two things does it skip, and approximately how much VRAM does it save per forward pass?

---

## 1. Big Picture — What `sentinel_model.py` Does

**File role:** `sentinel_model.py` is the top-level model. It owns all three sub-encoders, orchestrates their forward passes, selects GNN prefix nodes, assembles the three eyes, and produces logits. It is the only file `trainer.py` interacts with directly.

**The three-eye concept:**

SENTINEL makes three independent "opinions" about each contract, then lets a learned classifier combine them:

```
Contract
   │
   ├─► GNNEncoder    ──► "What do structural patterns say?"
   │   (execution order, call graphs, data flow)
   │        │
   │        ▼                         GNN eye [B, 128]
   │   function nodes                      │
   │   max+mean pool                       │
   │   gnn_eye_proj                        │
   │                                       │
   ├─► TransformerEncoder ──► "What does the source text say?"
   │   (raw Solidity + LoRA)               │
   │        │                              │
   │        ▼                  Transformer eye [B, 128]
   │   WindowAttentionPooler               │
   │   transformer_eye_proj                │
   │                                       │
   └─► CrossAttentionFusion ──► "What emerges when both interact?"
       (node ↔ token cross-attn)           │
                              Fused eye [B, 128]
                                           │
              ┌────────────────────────────┘
              │
         cat([gnn_eye, tf_eye, fused_eye])  →  [B, 384]
              │
         Linear(384, 192) → ReLU → Dropout
              │
         Linear(192, 10)  →  logits [B, 10]
              │
         (no Sigmoid — applied externally)
```

**Why three eyes instead of one pooled output?**

A single pooled representation forces one compromised view — either structural or semantic, never both at full fidelity. Three independent 128-dim opinions let the main classifier learn which combination matters for each vulnerability class:

- `Reentrancy` might be 70% fused eye (requires structural + semantic interaction)
- `IntegerUO` (integer underflow/overflow) might be 80% transformer eye (syntax pattern in the source text)
- `DenialOfService` (DoS) might be 60% GNN eye (call graph structure)

The classifier learns these weightings from data. No hand-coded feature engineering needed.

---

## 2. Module-Level Constants — Single Source of Truth (lines 70–113)

> **[LEARNING MODE: Master the detail]** — Know why these are module-level, not hardcoded literals.

```python
# Line 75
_MAX_TYPE_ID: float = float(max(NODE_TYPES.values()))   # 12.0 for v8 schema
```

`NODE_TYPES` is imported from `graph_schema.py` (already taught — Module 1). The graph extractor stores node type as `float(type_id / _MAX_TYPE_ID)` — normalized to [0, 1]. To recover the integer type ID in the model:
```python
node_type_ids = (graphs.x[:, 0].float() * _MAX_TYPE_ID).round().long()
```

By deriving `_MAX_TYPE_ID` from `NODE_TYPES.values()` at import time, adding a new node type to `graph_schema.py` automatically updates the denormalization constant here. No manual edit needed — a classic Single Source of Truth pattern.

```python
# Lines 79-89
_FUNC_TYPE_IDS: frozenset[int] = frozenset({
    NODE_TYPES["FUNCTION"],     # 1
    NODE_TYPES["MODIFIER"],     # 2
    NODE_TYPES["FALLBACK"],     # 4
    NODE_TYPES["RECEIVE"],      # 5
    NODE_TYPES["CONSTRUCTOR"],  # 6
})
_FUNC_IDS_CPU: torch.Tensor = torch.tensor(sorted(_FUNC_TYPE_IDS), dtype=torch.long)
```

`_FUNC_TYPE_IDS` is the set of node types the GNN eye pools over (function-declaration nodes only, not CFG nodes). `_FUNC_IDS_CPU` is a pre-built CPU tensor of those IDs, used every forward pass in `torch.isin(node_type_ids, _FUNC_IDS_CPU.to(device))`.

> **[AUDIT A8]** — `_FUNC_IDS_CPU` is created at module load time and moved to device in `forward()` with `.to(device)`. This `.to(device)` call is a no-op if the tensor is already on the right device (PyTorch checks and skips). However, if multiple devices are used (e.g., DataParallel), the first `.to(cuda:0)` leaves it on `cuda:0` permanently, and subsequent calls from `cuda:1` would silently succeed (`.to(cuda:1)` would create a new tensor on `cuda:1` — but the module-level `_FUNC_IDS_CPU` stays on `cuda:0`). This is a latent DataParallel bug that wouldn't surface in single-GPU training.

The module-level assertion immediately below:
```python
assert _FUNC_IDS_CPU.numel() == len(_FUNC_TYPE_IDS) and _FUNC_IDS_CPU.min() >= 0, (
    f"NC-2: _FUNC_IDS_CPU has unexpected shape or values: {_FUNC_IDS_CPU.tolist()}"
)
```
Fires at import time — catches any accidental modification to `_FUNC_TYPE_IDS` that makes it inconsistent with `_FUNC_IDS_CPU`. (P5 recall: module-level assertions as import-time invariant checks — same pattern as seen in `graph_schema.py`.)

```python
# Lines 97-113
_PREFIX_NODE_PRIORITY: dict[int, int] = {
    NODE_TYPES["CONSTRUCTOR"]: 0,   # always first
    NODE_TYPES["FALLBACK"]:    1,   # reentrancy-critical
    NODE_TYPES["RECEIVE"]:     2,   # reentrancy-critical
    NODE_TYPES["MODIFIER"]:    3,   # access control
    NODE_TYPES["FUNCTION"]:    4,   # general (last if K forces truncation)
}

_PREFIX_TYPE_IDX: dict[int, int] = {
    NODE_TYPES["FUNCTION"]:    0,
    NODE_TYPES["MODIFIER"]:    1,
    NODE_TYPES["FALLBACK"]:    2,
    NODE_TYPES["RECEIVE"]:     3,
    NODE_TYPES["CONSTRUCTOR"]: 4,
}
_NUM_PREFIX_TYPES: int = 5
```

`_PREFIX_NODE_PRIORITY` controls which nodes fill the K=48 prefix slots when there are more eligible nodes than slots. Lower number = selected first.

`_PREFIX_TYPE_IDX` maps raw `NODE_TYPES` integer IDs to stable indices 0–4 for the `prefix_type_embedding` lookup table. These two dicts use different orderings intentionally — priority is by vulnerability importance; embedding index is alphabetical/arbitrary but stable.

> ⚠️ **CRITICAL** — CONSTRUCTOR, FALLBACK, and RECEIVE are Solidity-specific entry points (P11 — domain terms):
> - **CONSTRUCTOR**: runs once at deployment. Controls initial state.
> - **FALLBACK**: called when no function signature matches or when Ether is sent with data. Classic reentrancy entry point.
> - **RECEIVE**: called when Ether is sent with no data (pure Ether transfers). Another reentrancy entry. Both FALLBACK and RECEIVE being high-priority reflects the system's domain knowledge: these are the nodes most likely to be exploited in reentrancy attacks.

---

## 3. `SentinelModel.__init__` — Full Assembly (lines 116–258)

> **[LEARNING MODE: Understand the pattern]** — The constructor as a wiring diagram. Know what each block creates and why.

### 3a. Sub-module instantiation (lines 178–202)

```python
self.gnn = GNNEncoder(
    hidden_dim=gnn_hidden_dim,      # 256
    heads=gnn_heads,                # 8
    dropout=gnn_dropout,
    use_edge_attr=use_edge_attr,
    edge_emb_dim=gnn_edge_emb_dim,  # 64
    num_layers=gnn_num_layers,      # 8 (2+3+3 phases)
    use_jk=gnn_use_jk,
    jk_mode=gnn_jk_mode,
    phase2_edge_types=gnn_phase2_edge_types,
)
self.transformer = TransformerEncoder(
    lora_r=lora_r, lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    lora_target_modules=lora_target_modules,
)
self.fusion = CrossAttentionFusion(
    node_dim=gnn_hidden_dim,        # must match GNNEncoder.hidden_dim
    token_dim=768,
    attn_dim=256,
    num_heads=8,
    output_dim=fusion_output_dim,   # 128 — the LOCKED eye width
    dropout=dropout,
)
```

**Coupling between `gnn_hidden_dim` and `CrossAttentionFusion.node_dim`:**

`CrossAttentionFusion` receives `node_dim=gnn_hidden_dim`. If you change `gnn_hidden_dim` (e.g., to 512 for a larger model), you must also update `CrossAttentionFusion(node_dim=...)`. Both are driven by the same parameter here, so this is safe — but it's a hardcoded dependency worth knowing about.

### 3b. GNN prefix injection modules (lines 204–212)

```python
if gnn_prefix_k > 0:
    self.gnn_to_bert_proj = nn.Linear(gnn_hidden_dim, 768)
    self.prefix_type_embedding = nn.Embedding(_NUM_PREFIX_TYPES, 768)
```

These modules only exist when `gnn_prefix_k > 0` (default: 0, disabled; active: 48). The `if` at construction time means:
- When disabled: `self.gnn_to_bert_proj` doesn't exist — accessing it raises `AttributeError`
- When enabled: a 256→768 linear projection maps GNN embeddings into BERT's input space

`prefix_type_embedding` is a 5-row table (one per declaration node type). It adds a type-specific bias vector to each projected prefix — letting the Transformer distinguish "this came from a FALLBACK node" from "this came from a FUNCTION node."

### 3c. Eye projections (lines 214–228)

```python
# GNN eye: max+mean pool → [B, 2*gnn_hidden_dim] → [B, eye_dim=128]
self.gnn_eye_proj = nn.Sequential(
    nn.Linear(2 * gnn_hidden_dim, eye_dim),   # Linear(512, 128)
    nn.ReLU(),
    nn.Dropout(dropout),
)

# Transformer eye: window-pooled CLS → [B, 768] → [B, eye_dim=128]
self.window_pooler = WindowAttentionPooler(
    hidden_dim=768, window_size=512, prefix_k=gnn_prefix_k
)
self.transformer_eye_proj = nn.Sequential(
    nn.Linear(768, eye_dim),                   # Linear(768, 128)
    nn.ReLU(),
    nn.Dropout(dropout),
)
```

**Why `2 * gnn_hidden_dim` input to `gnn_eye_proj`?**

The GNN eye uses both `global_max_pool` and `global_mean_pool` over function-level nodes, then concatenates them: `cat([max_pool, mean_pool])` → `[B, 2×256] = [B, 512]`. Max pooling captures the most extreme signal (riskiest function); mean pooling captures average behavior. Together they give a richer summary than either alone.

`prefix_k=gnn_prefix_k` is passed to `WindowAttentionPooler` so it knows where CLS sits within each window (at position 0 without prefix, at position K with prefix).

### 3d. Main classifier (lines 231–240)

```python
_cls_hidden = 192
self.classifier = nn.Sequential(
    nn.Linear(3 * eye_dim, _cls_hidden),   # Linear(384, 192)
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(_cls_hidden, num_classes),    # Linear(192, 10)
)
```

`3 * eye_dim = 3 × 128 = 384` — the three eyes concatenated. A hidden layer at 192 (halfway between input and output) adds capacity for learning cross-eye interactions without overfitting on 44K contracts. No `Sigmoid` — applied externally in `BCEWithLogitsLoss` and the predictor.

**Why 192 specifically?**

192 = 384/2. This is a common heuristic: halve the input dimension for the hidden layer. It's not mathematically optimal — a hyperparameter search might find 128 or 256 performs better — but 192 is a reasonable default that provides adequate capacity without risk of overfitting.

### 3e. Auxiliary heads (lines 243–248)

```python
self.aux_gnn         = nn.Linear(eye_dim, num_classes)   # Linear(128, 10)
self.aux_transformer = nn.Linear(eye_dim, num_classes)   # Linear(128, 10)
self.aux_fused       = nn.Linear(eye_dim, num_classes)   # Linear(128, 10)
```

> **[LEARNING MODE: Master the detail]** — The "why" of auxiliary heads. Common interview topic.

**The gradient starvation problem:**

The main classifier learns to route gradient through whichever eye is most useful. If the GNN eye is initially noisy (early training), the classifier might learn to assign near-zero weight to `gnn_eye` and route all signal through `transformer_eye`. After this, `gnn_eye_proj` and `GNNEncoder` receive near-zero gradients — they stop improving. By epoch 43 in one experiment, the GNN eye's gradient contribution had collapsed to ~7%.

**How auxiliary heads fix it:**

Each auxiliary head creates an independent loss:
```python
# In trainer.py:
loss_gnn  = criterion(aux_gnn,  labels)
loss_tf   = criterion(aux_tf,   labels)
loss_fuse = criterion(aux_fused, labels)
total_loss = main_loss + 0.3 * (loss_gnn + loss_tf + loss_fuse)
```

`loss_gnn` gradient flows directly into `gnn_eye` → `gnn_eye_proj` → `GNNEncoder`, bypassing the main classifier entirely. Even if the main classifier routes zero gradient to the GNN eye, `loss_gnn` keeps it training.

**At inference — zero overhead:**

```python
if not return_aux:
    return logits   # ← exits here in default/inference mode
```

The auxiliary heads never execute during inference. `return_aux=False` is the default. The three `nn.Linear` layers exist in memory but receive no forward call.

> **[AUDIT A9]** — Auxiliary heads add 3 × `Linear(128, 10)` = 3 × 1,280 = 3,840 parameters. Negligible. But they also add 3 loss computations per training step — each requiring a separate `criterion(aux_*, labels)` call with its own backward graph. At 44K contracts × 50 epochs, this is a real compute cost. A more efficient implementation might share the label tensor and batch the criterion calls. In practice the overhead is small (<5% of step time) but worth knowing.

---

## 4. Parameter Count and `parameter_summary` (lines 532–562)

```python
def parameter_summary(self) -> None:
    components = {
        "GNNEncoder":            self.gnn,
        "TransformerEncoder":    self.transformer,
        "CrossAttentionFusion":  self.fusion,
        "gnn_eye_proj":          self.gnn_eye_proj,
        "transformer_eye_proj":  self.transformer_eye_proj,
        "Classifier (3×eye→C)":  self.classifier,
        "aux_gnn":               self.aux_gnn,
        "aux_transformer":       self.aux_transformer,
        "aux_fused":             self.aux_fused,
    }
    for name, module in components.items():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        logger.info(f"  {name}: {trainable:,} trainable | {frozen:,} frozen")
```

> **[LEARNING MODE: Understand the pattern]** — Know what `p.requires_grad` means and how to use it for auditing.

`p.requires_grad` is `True` for trainable parameters and `False` for frozen ones. Summing `p.numel()` (number of elements) for each gives the parameter count. This is the standard pattern for auditing model parameter counts:

```python
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
```

**Approximate counts for default config:**

| Sub-module | Trainable | Frozen |
|------------|-----------|--------|
| GNNEncoder | ~2.4M | 0 |
| TransformerEncoder (CodeBERT + LoRA) | ~590K | ~125M |
| CrossAttentionFusion | ~600K | 0 |
| Eye projections (×2) | ~230K | 0 |
| Classifier | ~74K | 0 |
| Aux heads (×3) | ~3.9K | 0 |
| **Total** | **~3.9M** | **~125M** |

Only ~3% of all parameters are trainable. The 125M frozen CodeBERT weights dominate the memory footprint but contribute zero gradient computation (aside from the 590K LoRA adapters within them).

---

## 5. Cross-File Relationships

**Already taught — recall these connections:**
- `GNNEncoder` (chunks 01–03): produces `node_embs [N, 256]` and `batch [N]`. `_FUNC_TYPE_IDS` here selects which of these N nodes to pool for the GNN eye.
- `TransformerEncoder` (chunks 04–05): produces `token_embs [B, W*L, 768]`. `WindowAttentionPooler` lives inside it but is instantiated here in `SentinelModel.__init__`.
- `CrossAttentionFusion` (chunk 06): takes both, produces `fused_eye [B, 128]`. `output_dim=128` is the LOCKED dimension that determines `3 * eye_dim = 384` for the classifier input.
- `graph_schema.NODE_TYPES` (Module 1): `_MAX_TYPE_ID`, `_FUNC_TYPE_IDS`, `_PREFIX_NODE_PRIORITY` all derive from it — the single-source-of-truth chain continues into the model.

**Not yet taught — preview:**
- `forward()` (chunk 08): where `_FUNC_TYPE_IDS`, `_PREFIX_NODE_PRIORITY`, and all sub-modules get called in sequence. Also covers `select_prefix_nodes()` — the per-graph prefix selection loop.

---

## 3 Things to Lock In

1. **`_MAX_TYPE_ID` is derived from `NODE_TYPES.values()` at import time** — not hardcoded as 12. Adding a node type to `graph_schema.py` automatically propagates to the model's denormalization. This is the single-source-of-truth pattern applied across file boundaries.

2. **Auxiliary heads prevent gradient starvation.** Without them, the main classifier can learn to route near-zero gradient to any eye it finds less useful. With them, each eye has a direct independent loss gradient regardless of main classifier routing. Zero inference overhead — `return_aux=False` exits before they run.

3. **Three eyes, each 128-dim, concatenated to 384.** The LOCKED `output_dim=128` flows from `CrossAttentionFusion` → `gnn_eye_proj` → `transformer_eye_proj` → `Linear(384, 192)`. Changing any one without the others causes a shape mismatch at the classifier.

---

## Challenge Questions

1. `_FUNC_IDS_CPU` is a module-level tensor created at import. What is the risk in DataParallel (multi-GPU) training, and how would you fix it?

2. The GNN eye projection takes `Linear(2 * gnn_hidden_dim, eye_dim)` = `Linear(512, 128)`. Why `2 * gnn_hidden_dim` and not just `gnn_hidden_dim`? What information would be lost by using only one pooling operation?

3. Auxiliary head `aux_gnn = Linear(128, 10)` directly takes `gnn_eye` as input. If the main classifier learns to zero out all weights for `gnn_eye`, but `aux_gnn` is still trained, what happens to the GNN encoder's gradient during backprop?

4. `gnn_to_bert_proj` and `prefix_type_embedding` only exist when `gnn_prefix_k > 0`. What Python error would you get at runtime if `gnn_prefix_k=0` at construction but `select_prefix_nodes()` is called anyway?

---

**Next:** `08_sentinel_model_forward_and_prefix.md` — `select_prefix_nodes()`, the full three-eye forward pass, ghost graph handling, prefix warmup guard, and the `compute_prefix_attention_mean` diagnostic.
