# SENTINEL v5 — Complete ML Module Overhaul
# Comprehensive Technical Plan

| Field        | Value                                   |
|--------------|-----------------------------------------|
| Date         | 2026-05-10                              |
| Revision     | 1.3 — auxiliary loss + num_layers wiring + edge-type numbering fixes |
| Status       | ACTIVE — Implementation Guide           |
| Supersedes   | v4 exp1 (`multilabel-v4-finetune-lr1e4_best.pt`) |
| Author       | Post-audit synthesis (code + results)   |
| Priority     | Critical — production model is broken   |

### Revision 1.1 Changes (external review)

Seven issues raised by external review; all assessed, six applied:

| # | Issue | Decision |
|---|---|---|
| 1 | Missing `CONTAINS` edge — function → CFG_NODE link absent from final schema | **Applied** — added `CONTAINS:5`, shifted `CONTROL_FLOW` to `6`, `NUM_EDGE_TYPES=7` |
| 2 | No fallbacks for `return_ignored` / `call_target_typed` | **Applied** — text-based fallbacks added to §2.2B; Slither version pin added to §9 |
| 3 | Augmentation sources underspecified; no automated generation strategy | **Applied** — mutation-based generation strategy added to §3.2 |
| 4 | GNN depth of 4 potentially insufficient for diameter-6 CFGs | **Accepted as-is** — 4 is correct starting point; layer count is a constructor arg and trivial to change if needed |
| 5 | Inline threshold sweep every epoch doubles training time | **Applied** — replaced with lightweight calibration log; full sweep post-training only |
| 6d | Validation split strategy ambiguous re: augmented data | **Applied** — clarified in §3.3 and §5.3: original v4 split for metric tracking; augmented data training-only |
| 7 | Per-class F1 floor too tight at 0.05 given `reentrant` removal | **Applied** — relaxed from 0.05 to 0.10 in §8.1 |

### Revision 1.2 Changes (three-eye classifier architecture)

The principal author proposed adding independent per-modality classifier inputs alongside
the existing fused representation. After technical review, this is adopted as a core v5
architectural decision, not a future experiment. See §4.5 and §7.5 for full rationale.

| Component | Change |
|---|---|
| `sentinel_model.py` | Add `gnn_eye_proj Linear(256,128)`, `transformer_eye_proj Linear(768,128)`; classifier becomes `Linear(384, 10)` |
| Forward pass | GNN eye (max+mean pool → proj), Transformer eye (CLS token → proj), Fused eye (CrossAttentionFusion) → concatenate → classify |
| `trainer.py` | Add per-eye gradient norm logging to catch eye dominance early |
| §4.5 | New decision record: three-eye architecture rationale |

### Revision 1.3 Changes (auxiliary loss, num_layers wiring, edge-type numbering)

Three corrections applied after final author review:

| # | Correction |
|---|---|
| 1 | **Stale edge-type bullets in §2.1 / §2.3** — earlier bullets incorrectly listed `CONTROL_FLOW:5` and `NUM_EDGE_TYPES 5→6`. Fixed to `CONTAINS:5`, `CONTROL_FLOW:6`, `NUM_EDGE_TYPES 5→7` throughout. |
| 2 | **`gnn_layers` was a dead config field** — `TrainConfig.gnn_layers` was added in Rev 1.2 but GNNEncoder had no matching `num_layers` constructor argument, so the value was silently ignored. Added `num_layers: int = 4` to GNNEncoder with a `NotImplementedError` guard for values other than 4 (see §2.3D). |
| 3 | **Auxiliary loss for eye-independence enforcement** — added three per-eye auxiliary classifier heads (`aux_gnn`, `aux_transformer`, `aux_fused`, each `Linear(128,10)`) to `sentinel_model.py`; `forward()` gains `return_aux: bool = False`; trainer loss becomes `main + λ*(loss_gnn + loss_tf + loss_fused)` with `λ=0.1` (see §2.5D–E, §2.6B, §4.5). |

---

## 0. The Problem, Stated Plainly

v4 cleared a validation gate of F1-macro=0.5422. Then we tested it on 20
hand-crafted contracts and it collapsed:

- **Detection rate: 15%** — 3 of 19 expected vulnerabilities found.
- **Specificity: 33%** — 2 of 3 safe contracts were flagged as vulnerable.
- The model fires `Reentrancy` and `CallToUnknown` on nearly every contract that
  contains an external call, regardless of whether protections exist.
- `DenialOfService`, `IntegerUO` on Solidity 0.8+, `TransactionOrderDependence`,
  and `GasException` are essentially invisible.

Validation metrics were lying. The model learned shallow shortcuts that score well
on a held-out slice of the same dataset distribution but generalize to nothing.

**Nothing from the current trained model is being preserved. This is a clean rebuild.**

---

## 1. Why the Current Model Is Broken — Full Root Cause Audit

All claims below are verified against the actual source code as of 2026-05-10.

### 1.1 The Graph Has No Sense of Time or Order

**File:** `ml/src/preprocessing/graph_extractor.py`, lines 484–511

The edge extraction loop uses exactly five relation types:
```
CALLS    func → internal call
READS    func → state variable read
WRITES   func → state variable written
EMITS    func → event fired
INHERITS contract → parent
```

Slither's per-function CFG — `func.nodes` and each node's `.sons` (successors) —
is **never accessed**. The graph cannot encode execution order.

**Consequence:** A contract that calls an external address *before* zeroing a balance
(vulnerable to reentrancy) produces an **identical graph** to one that zeros the balance
*before* calling (safe CEI pattern). Same nodes, same edges, same topology. The GNN
outputs identical embeddings for both. The cross-attention layer cannot compensate
because its query vectors come from those embeddings. The model is structurally incapable
of learning the difference. It defaults to: "contract contains a low-level call → fire
Reentrancy/CallToUnknown." That is exactly the behavior we observe in manual tests.

### 1.2 Node Features Carry No Semantic Vulnerability Signal

**File:** `ml/src/preprocessing/graph_schema.py`, lines 56, 148–157

Current 8-dimensional feature vector:
```
[type_id, visibility, pure, view, payable, reentrant, complexity, loc]
```

Six critical semantic signals are absent:

| Missing Feature | Why It Matters |
|---|---|
| `return_ignored` | The defining difference between `MishandledException`/`UnusedReturn` and safe exception handling. Without this, the model cannot distinguish `(bool ok,) = call(...)` from bare `call(...)`. |
| `call_target_typed` | Whether the callee is a typed interface or a raw `address`. CallToUnknown should only fire for raw address calls; currently fires on all external calls. |
| `in_unchecked` | Solidity 0.8+ arithmetic overflow requires an `unchecked {}` block. This boolean is the primary signal for `IntegerUO` on modern contracts. Currently invisible. |
| `has_loop` | Unbounded loops are the primary DoS pattern. Without this flag, DoS is nearly undetectable. |
| `gas_intensity` | Expensive operations inside loops, large storage reads, etc. Needed for `GasException`. |
| `external_call_count` | Normalized count of external calls distinguishes an incidental call from a call-heavy function that is structurally suspect. |

**Additional note on the `reentrant` feature (feature[5]):**
This is Slither's own `is_reentrant` flag. Slither runs its own reentrancy detector
internally. If this flag is already set by Slither, the model is receiving a pre-computed
vulnerability signal as a training feature — essentially learning to echo Slither's answer.
This causes two problems: (1) it makes the model dependent on Slither's accuracy (which is
imperfect), and (2) it means the model is not learning to detect reentrancy from first
principles. In v5, we will **remove** this feature and replace it with the structural
features above. The model should detect reentrancy from call order, not from Slither's
label.

### 1.3 GNN Architecture Deficiencies

**File:** `ml/src/models/gnn_encoder.py`, line 127

```python
self.conv1 = GATConv(in_channels=8, ...)  # hardcoded — must be NODE_FEATURE_DIM
```

`in_channels=8` is hardcoded rather than imported from `graph_schema.NODE_FEATURE_DIM`.
Changing the feature dimension without catching this breaks the model silently at training
time. This is a code quality bug that must be fixed.

Additionally, the current 3-layer GAT may be insufficient depth for propagating
control-flow information through complex functions. Control-flow subgraphs within a
single function may have diameter > 3, meaning information from the entry block cannot
reach the exit block in 3 hops.

### 1.4 Cross-Attention Query Vectors Are Structurally Blind

**File:** `ml/src/models/fusion_layer.py`, lines 196–204

The `node_to_token` cross-attention uses GNN node embeddings as queries. Because the GNN
embeddings encode no execution order and no semantic flags, the queries can only
express: "I am a payable function that reads/writes a balance state variable." They cannot
ask: "Does a call precede the write?" or "Is the return value ignored here?"

The bidirectional cross-attention is correctly implemented and theoretically sound.
Its failure is entirely upstream: garbage-in, garbage-out from the GNN. Fix the GNN
features and edges, and the cross-attention machinery works as designed.

### 1.5 Data Distribution Is Severely Imbalanced

**Source:** `ml/src/training/trainer.py`, line 500 (self-documenting comment in code)

```
DenialOfService  —   137 training samples
IntegerUO        — 5,343 training samples  (39× ratio)
```

At `batch_size=16`, DoS appears in ~4 batches per epoch. The model's gradient signal
for DoS is dominated by noise (137 val samples → 5 TPs = ±0.04 F1 swing per batch).
No hyperparameter change fixes a 39× class imbalance. **Data is the only fix for DoS.**

v3 per-class precision/recall at tuned thresholds shows **6 of 10 classes are
over-predicting 1.6–2.3×** (high recall, low precision). This is the validation-side
evidence of the same problem seen in manual tests (false positives on safe contracts).

### 1.6 Label Co-occurrence Teaches the Wrong Correlations

The training dataset contains many contracts with multiple simultaneous labels
(e.g., a contract vulnerable to both Reentrancy and MishandledException). With only
137 DoS samples, the model cannot learn "what makes DoS unique" because the positive
examples are too few to isolate the signal. Instead it learns "multi-vulnerable contracts
tend to have external calls" → fires every call-adjacent label on any contract with a
call. This explains the "everything fires" behavior on timestamp contracts in manual tests.

---

## 2. What Changes — Complete List, No Locks

This section lists every file that changes and exactly what changes in it.
Nothing is exempt. Existing docstring constraints about "locked" or "do not change
without retraining" are acknowledged but do not block changes — the full retrain is
planned anyway.

### 2.1 `ml/src/preprocessing/graph_schema.py`

**Changes:**
1. `FEATURE_SCHEMA_VERSION`: `"v1"` → `"v2"`
2. `NODE_FEATURE_DIM`: `8` → `13`
   (removing `reentrant`, adding 6 new features — net +5)
3. `FEATURE_NAMES`: remove `reentrant`, add 6 new entries
4. `EDGE_TYPES`: add `"CONTAINS": 5` (function → CFG_NODE), add `"CONTROL_FLOW": 6` (CFG_NODE → successor)
5. `NUM_EDGE_TYPES`: `5` → `7`

**New `FEATURE_NAMES` (13 dims):**
```python
FEATURE_NAMES = (
    "type_id",              # 0  — NODE_TYPES int, as before
    "visibility",           # 1  — VISIBILITY_MAP ordinal 0-2, as before
    "pure",                 # 2  — bool, as before
    "view",                 # 3  — bool, as before
    "payable",              # 4  — bool, as before
    # "reentrant" REMOVED — was Slither's own detection, teaching a shortcut
    "complexity",           # 5  — CFG block count (was 6), as before
    "loc",                  # 6  — lines of code (was 7), as before
    "return_ignored",       # 7  — NEW: bool, low-level call return not captured
    "call_target_typed",    # 8  — NEW: bool, all calls to typed interfaces (not raw address)
    "in_unchecked",         # 9  — NEW: bool, body contains unchecked arithmetic block
    "has_loop",             # 10 — NEW: bool, function contains a loop
    "gas_intensity",        # 11 — NEW: float [0,1], heuristic for expensive gas patterns
    "external_call_count",  # 12 — NEW: float, log-normalized count of external calls
)
```

**New `EDGE_TYPES`:**
```python
EDGE_TYPES = {
    "CALLS":        0,   # function → internally-called function
    "READS":        1,   # function → state variable read
    "WRITES":       2,   # function → state variable written
    "EMITS":        3,   # function → event fired
    "INHERITS":     4,   # contract → parent contract
    "CONTAINS":     5,   # function → its CFG_NODE children (NEW)
    "CONTROL_FLOW": 6,   # CFG node → successor CFG node (execution order, NEW)
}
NUM_EDGE_TYPES = 7
```

**Why `CONTAINS` is required (reviewer gap #1):**
Without a `CONTAINS` edge from each function node to its child CFG_NODE blocks, the
statement-level subgraph is a disconnected island. The GNN cannot propagate function-level
properties (payable, visibility, external_call_count) down to statement nodes, and
statement-level patterns (call-before-write) cannot flow back up to the function node.
Reusing `CALLS` for this purpose is ambiguous and confusing — `CALLS` already means
"this function invokes another function." A separate `CONTAINS` type is the clean fix.

### 2.2 `ml/src/preprocessing/graph_extractor.py`

**Changes:**

**A. Remove `reentrant` from `_build_node_features()`**
The `reentrant = 1.0 if getattr(obj, "is_reentrant", False) else 0.0` line is deleted.

**B. Add 6 new features to `_build_node_features()`**

Computation logic for each:

```
return_ignored:
    PRIMARY (Slither IR): iterate func.slithir_operations.
    For any LowLevelCall or HighLevelCall, check the .lvalue attribute.
    If lvalue is None, the return value is discarded → set 1.0.
    FALLBACK (text-based, used if Slither IR raises AttributeError or
    returns no operations): scan func.source_mapping.content for patterns
    matching bare `.call{` or `.call(` not preceded by `=` or `,` on the
    same line. Regex: r'(?<![=,(])\s*\.call[\({]' — imperfect but catches
    the most common pattern. Log a WARNING when fallback is used.
    For non-Function nodes: 0.0

call_target_typed:
    PRIMARY (Slither type analysis): iterate func.high_level_calls and
    func.low_level_calls. For each, check receiver type via
    call.destination.type (or call.called.type). If ContractType → typed.
    If AddressType → raw address → set 0.0. If all are ContractType or
    there are no external calls → set 1.0.
    FALLBACK (text-based, used if type resolution raises or returns None):
    scan func.source_mapping.content. If any pattern matching
    r'address\([^)]+\)\.call' or r'\baddress\b.*\.call' is found → 0.0.
    If only interface-style calls are found (e.g., IToken(addr).transfer) → 1.0.
    Log a WARNING when fallback is used.
    For non-Function nodes: 1.0 (not applicable, default safe)

in_unchecked:
    PRIMARY (Slither node types): iterate func.nodes and check for
    nodes with type == NodeType.ASSEMBLY or containing UNCHECKED_BEGIN
    (Slither ≥0.9.3).
    FALLBACK (regex): scan func.source_mapping.content for the literal
    string "unchecked" — reliable for Solidity 0.8+.
    For non-Function nodes: 0.0

has_loop:
    For Function nodes: Slither exposes func.is_loop_present (some
    versions) or we check func.nodes for nodes with type in
    {NodeType.IFLOOP, NodeType.STARTLOOP, NodeType.ENDLOOP}.
    For non-Function nodes: 0.0

gas_intensity:
    For Function nodes: heuristic combining:
        - complexity (CFG block count) / 50.0, clamped [0,1]
        - 0.3 bonus if has_loop=True (loops are expensive)
        - 0.2 bonus if external_call_count_raw > 2
        Final: min(1.0, base + bonuses)
    For non-Function nodes: 0.0

external_call_count:
    For Function nodes: count len(func.high_level_calls) +
    len(func.low_level_calls). Apply log1p normalization:
    log1p(count) / log1p(20), clamped [0,1].
    (log1p(20)≈3.0, so 20 calls → 1.0, 1 call → 0.23, 5 calls → 0.60)
    For non-Function nodes: 0.0
```

**C. Add `_build_control_flow_edges()` function**

Slither exposes intra-function CFG via `func.nodes` (a list of `SlitherNode` objects)
and each node's `.sons` (successor nodes in the CFG). We map each SlitherNode to
a graph-level node index and add CONTROL_FLOW edges.

Implementation approach — "statement-node expansion":
Rather than just adding CFG edges between function nodes (which wouldn't help because
all statements in a function share the same function node), we introduce a new node
type `CFG_NODE` and a new statement-level node for each Slither CFG node within a
function. These statement-level nodes are connected by CONTROL_FLOW edges.

This is a significant but necessary change. The function-level node remains as before,
but we also insert child nodes for each CFG block, connected to the function node via
a `CONTAINS` edge (type 5) and to each other via CONTROL_FLOW edges (type 6).

**Statement-node feature vector** (same 13-dim format, mostly zeroed):
```
type_id       = NODE_TYPES["CFG_NODE"]   (new type, value=8)
visibility    = 0.0
pure, view, payable = inherited from parent function
complexity    = 1.0 (each CFG node is one basic block)
loc           = float(len(node.source_mapping.lines)) if available
return_ignored = 1.0 if node contains a call whose lvalue is None
call_target_typed = 1.0 if all calls in this node are to typed interfaces
in_unchecked  = 1.0 if node is inside an unchecked scope
has_loop      = 1.0 if node type is IFLOOP/STARTLOOP
gas_intensity = heuristic for this block
external_call_count = log-normalized call count in this block
```

**Graph construction order** (must be deterministic for reproducibility):
```
CONTRACT node (index 0)
STATE_VAR nodes (indices 1..V)
FUNCTION nodes (indices V+1..V+F)
  For each function: CFG_NODE children (in func.nodes order)
MODIFIER nodes
EVENT nodes
```

**D. Add `NODE_TYPES["CFG_NODE"] = 8` to graph_schema.py**

### 2.3 `ml/src/models/gnn_encoder.py`

**Changes:**

**A. Import and use `NODE_FEATURE_DIM` instead of hardcoding 8:**
```python
from ml.src.preprocessing.graph_schema import NUM_EDGE_TYPES, NODE_FEATURE_DIM
# ...
self.conv1 = GATConv(in_channels=NODE_FEATURE_DIM, ...)  # was hardcoded 8
```

**B. Increase default layers from 3 to 4:**
Control-flow subgraphs within a single function can have diameter >3 (entry→branch→loop
body→exit is already 3 hops). A 4th layer ensures information can propagate through
reasonably complex intra-function CFGs.
```python
self.conv4 = GATConv(
    in_channels=hidden_dim, out_channels=hidden_dim,
    heads=1, concat=False, dropout=dropout,
    edge_dim=_edge_dim,
)
```
Add residual connections between same-dimension layers (conv2→conv3, conv3→conv4)
to prevent gradient vanishing in the deeper network:
```python
x2 = self.conv2(x, edge_index, edge_attr=e)
x2 = self.relu(x2)
x2 = self.dropout(x2 + x)  # residual: x from layer 1 output
```

**C. Update `NUM_EDGE_TYPES` import (automatically picks up 7 from schema).**
The `nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)` line already imports the constant —
no change needed in that line, but the value will automatically be 7.

**D. Add `num_layers: int = 4` constructor parameter:**
`TrainConfig` gains a `gnn_layers` field (see §2.6A) that must map to an actual GNNEncoder
constructor argument, otherwise the config value is silently ignored.

```python
def __init__(
    self,
    hidden_dim:    int   = 128,
    heads:         int   = 8,
    dropout:       float = 0.2,
    use_edge_attr: bool  = True,
    edge_emb_dim:  int   = 32,
    num_layers:    int   = 4,    # NEW — must be ≥3; only 4 is fully tested in v5
) -> None:
```

For v5.0, the implementation keeps the explicit 4-layer structure (conv1/conv2/conv3/conv4)
and validates that `num_layers == 4` at init time with a clear error message:
```python
if num_layers != 4:
    raise NotImplementedError(
        f"GNNEncoder num_layers={num_layers} is not supported in v5.0. "
        "Only num_layers=4 is implemented. Generalised layer-count support "
        "is planned for v5.1."
    )
```
This way the field is wired end-to-end (TrainConfig → GNNEncoder) and won't silently
break if someone sets `gnn_layers=5` in an experiment. The NotImplementedError is a
clear signal, not a silent no-op.

**E. Update default `hidden_dim` from 64 to 128:**
With 13 input features, more CFG structure, and deeper architecture, the GNN needs
more capacity. 128-dim nodes with 8 heads (16 dims/head) is a reasonable expansion.
```python
def __init__(
    self,
    hidden_dim:    int   = 128,   # was 64
    heads:         int   = 8,
    ...
```
`CrossAttentionFusion` must be updated to receive `node_dim=128` (it already takes
this as a constructor argument, so `SentinelModel` just needs to pass `gnn_hidden_dim=128`).

### 2.4 `ml/src/models/fusion_layer.py`

**No structural changes required.** The bidirectional cross-attention is correctly
implemented. The `node_dim` constructor argument already accommodates any GNN output
dimension. The attention masks and pooling are correct.

**One optional improvement** (consider for v5.1): add a residual projection from the
mean-pooled input token embedding to the output, to preserve surface-level token
information that might be discarded by the attention pooling. This is a small change
but not blocking.

### 2.5 `ml/src/models/sentinel_model.py`

**Changes:**

**A. Add two new projection layers (the "eyes"):**
```python
from torch_geometric.nn import global_max_pool, global_mean_pool

# GNN eye: max+mean pool over node embeddings → project to output_dim
self.gnn_eye_proj = nn.Sequential(
    nn.Linear(gnn_hidden_dim * 2, fusion_output_dim),  # 256 → 128
    nn.ReLU(),
    nn.Dropout(dropout),
)

# Transformer eye: CLS token (position 0) → project to output_dim
self.transformer_eye_proj = nn.Sequential(
    nn.Linear(768, fusion_output_dim),  # 768 → 128
    nn.ReLU(),
    nn.Dropout(dropout),
)
```

**B. Update classifier input dimension from `fusion_output_dim` to `fusion_output_dim * 3`:**
```python
# Three eyes concatenated: gnn_eye [B,128] + transformer_eye [B,128] + fused_eye [B,128]
self.classifier = nn.Linear(fusion_output_dim * 3, num_classes)  # 384 → 10
```

**C. Updated forward pass:**
```python
# ── GNN path ──────────────────────────────────────────────────────────────
node_embs, batch = self.gnn(graphs.x, graphs.edge_index, graphs.batch, edge_attr)
# node_embs: [N, 128]

# GNN eye: max+mean pool — "is any node dangerous?" + "what's the typical node?"
gnn_max  = global_max_pool(node_embs, batch)                  # [B, 128]
gnn_mean = global_mean_pool(node_embs, batch)                 # [B, 128]
gnn_eye  = self.gnn_eye_proj(torch.cat([gnn_max, gnn_mean], dim=1))  # [B, 128]

# ── Transformer path ───────────────────────────────────────────────────────
token_embs = self.transformer(input_ids, attention_mask)
# token_embs: [B, 512, 768]

# Transformer eye: CLS token — CodeBERT's hierarchical sequence summary
# Position 0 is the CLS token: 12 layers of self-attention over all 512 positions.
# Distinct from the masked-mean pool used inside CrossAttentionFusion.
transformer_eye = self.transformer_eye_proj(token_embs[:, 0, :])  # [B, 128]

# ── Fused eye: bidirectional cross-attention ───────────────────────────────
fused_eye = self.fusion(node_embs, batch, token_embs, attention_mask)  # [B, 128]

# ── Classifier: each eye votes independently ───────────────────────────────
combined = torch.cat([gnn_eye, transformer_eye, fused_eye], dim=1)  # [B, 384]
logits   = self.classifier(combined)                                  # [B, 10]
```

**D. Add three auxiliary classifier heads (one per eye):**
```python
# Auxiliary heads — used only during training to enforce per-eye signal.
# Each independently predicts all 10 labels from one eye alone.
# Discarded at inference (return_aux=False in forward()).
self.aux_gnn         = nn.Linear(fusion_output_dim, num_classes)  # 128 → 10
self.aux_transformer = nn.Linear(fusion_output_dim, num_classes)  # 128 → 10
self.aux_fused       = nn.Linear(fusion_output_dim, num_classes)  # 128 → 10
```
Total auxiliary parameters: 3 × (128×10 + 10) = ~3.9K. Negligible.

**E. Add `return_aux: bool = False` to `forward()` signature:**
```python
def forward(
    self,
    graphs:         Batch,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    return_aux:     bool = False,   # NEW: True during training, False at inference
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    # (compute gnn_eye, transformer_eye, fused_eye as in §2.5C above)
    combined = torch.cat([gnn_eye, transformer_eye, fused_eye], dim=1)
    logits   = self.classifier(combined)

    if return_aux:
        aux = {
            "gnn":         self.aux_gnn(gnn_eye),          # [B, 10] logits
            "transformer": self.aux_transformer(transformer_eye),  # [B, 10]
            "fused":       self.aux_fused(fused_eye),       # [B, 10]
        }
        return logits, aux   # trainer unpacks this tuple

    return logits  # inference path: plain tensor, unchanged interface
```

The `return_aux=False` default means `predictor.py` and all inference callers need
zero changes — they never see the auxiliary outputs.

**F. Update `gnn_hidden_dim` default from 64 to 128 and pass to CrossAttentionFusion.**

**G. Update docstring** to describe the three-eye architecture and the rationale (see §4.5).

### 2.6 `ml/src/training/trainer.py`

**Changes:**

**A. Update `TrainConfig` defaults for v5:**
```python
gnn_hidden_dim:   int   = 128     # was 64
gnn_layers:       int   = 4       # new — wired to GNNEncoder(num_layers=4)
checkpoint_name:  str   = "multilabel-v5-fresh_best.pt"
epochs:           int   = 60      # longer run for fresh-from-scratch training
lr:               float = 2e-4    # slightly lower than default 3e-4 for stability
aux_loss_weight:  float = 0.1     # new — λ for auxiliary eye losses (0 = disabled)
```

**B. Update training loop for auxiliary loss:**
```python
# In train_one_epoch — replace the simple forward call:
if config.aux_loss_weight > 0:
    logits, aux_logits = model(graphs, input_ids, attention_mask, return_aux=True)
    main_loss = loss_fn(logits, labels)
    aux_loss  = sum(loss_fn(v, labels) for v in aux_logits.values())
    loss      = main_loss + config.aux_loss_weight * aux_loss
    # Log individual loss components for diagnostics
    mlflow.log_metric("loss_main", main_loss.item(), step=global_step)
    mlflow.log_metric("loss_aux",  aux_loss.item(),  step=global_step)
else:
    logits = model(graphs, input_ids, attention_mask)
    loss   = loss_fn(logits, labels)
```

The `loss_main` vs `loss_aux` ratio in MLflow immediately shows whether any auxiliary
eye is failing independently (aux_loss >> main_loss → one eye is confused without help
from the others; this is expected early but should close by epoch 20).

**C. Add per-eye gradient norm logging to detect eye dominance:**
```python
# Log after scaler.unscale_(optimizer), before clip_grad_norm_
# Run every log_interval batches (not every step — too noisy)
if (batch_idx + 1) % log_interval == 0:
    for eye_name, module in [
        ("gnn_eye",         model.gnn_eye_proj),
        ("transformer_eye", model.transformer_eye_proj),
        ("fused",           model.fusion),
    ]:
        gnorm = sum(p.grad.norm().item() ** 2
                    for p in module.parameters() if p.grad is not None) ** 0.5
        mlflow.log_metric(f"grad_norm_{eye_name}", gnorm, step=global_step)
```
If any eye's gradient norm collapses to near-zero while others remain healthy, that eye
is being suppressed — the classifier has stopped using it. Catch this early, not after
60 epochs. Acceptable range: all three eyes should have grad norms within an order of
magnitude of each other throughout training.

**D. Add per-class calibration logging (not a full threshold sweep):**
Running a full per-class threshold sweep (13 thresholds × 10 classes = 130 forward
passes of the val set) on every improving epoch would approximately double epoch time.
Instead, log a lightweight calibration signal every epoch:
```python
# After evaluate(), log per-class predicted-positive counts alongside true counts.
# This catches over-prediction early without a full sweep.
for i, name in enumerate(CLASS_NAMES):
    pred_pos  = int(y_pred[:, i].sum())
    true_pos  = int(y_true[:, i].sum())
    mlflow.log_metric(f"pred_pos_{name}", pred_pos, step=epoch)
    mlflow.log_metric(f"true_pos_{name}", true_pos, step=epoch)
    mlflow.log_metric(f"pred_true_ratio_{name}", pred_pos / max(1, true_pos), step=epoch)
```
A `pred_true_ratio` >> 1 flags persistent over-prediction at a glance.
The full threshold sweep remains in the separate `tune_threshold.py` script and
is run once on the final best checkpoint (Phase 6), not during training.

**E. Loss function for v5:**
Start with `BCEWithLogitsLoss(pos_weight=...)` as before. After the first evaluation
checkpoint (epoch ~10), assess the precision/recall profile per class. If over-prediction
persists despite better graph features, switch to FocalLoss with per-class alpha tuned
to match the desired precision-recall trade-off (α > 0.5 for rare classes, α ~0.25 for
well-represented ones).

Do **not** use the global `α=0.25` FocalLoss default from the binary-mode preset —
that was documented in `2026-05-09-v3-analysis-and-v4-direction.md` to severely hurt
rare class recall.

### 2.7 `ml/src/inference/preprocess.py`

After the schema change, the inference preprocessor must use the new graph extractor.
Since both are driven by the single `graph_extractor.py` canonical implementation,
**no manual change is needed** — the inference path automatically picks up the new
features and edge types. However, the inference predictor (`predictor.py`) must be
updated to load a checkpoint with `num_classes=10` and the new architecture dimensions.

### 2.8 `ml/scripts/validate_graph_dataset.py`

Update validation assertions:
```python
assert graph.x.shape[1] == NODE_FEATURE_DIM   # will now check for 13
assert graph.edge_attr.max() < NUM_EDGE_TYPES  # will now check for 7
```
Also add assertions that:
- At least some graphs contain `CONTAINS` edges (type 5) — function → CFG_NODE links exist.
- At least some graphs contain `CONTROL_FLOW` edges (type 6) — CFG ordering was extracted.
Both checks catch silent extraction failures where new edge logic ran but produced no output.

---

## 3. Data Pipeline — Complete Plan

### 3.1 Current Dataset Inventory

From analysis of training runs and code comments:

| Class | Training Samples | Problem Level |
|---|---|---|
| IntegerUO | ~5,343 | None — strongest class |
| GasException | ~2,589 | Moderate — over-predicts |
| MishandledException | ~2,207 | Moderate — over-predicts |
| Reentrancy | ~2,501 | Moderate — false positives on safe CEI |
| TransactionOrderDependence | ~1,800 | High — over-predicts 2.3× |
| ExternalBug | ~1,622 | High — over-predicts 2.3× |
| UnusedReturn | ~1,716 | Moderate — over-predicts |
| Timestamp | ~1,077 | Moderate |
| CallToUnknown | ~1,266 | High — fires on all calls |
| DenialOfService | ~137 | Critical — data starvation |
| **Safe contracts** | **Unknown — likely very few** | **Critical — false positive source** |

### 3.2 Required Data Augmentation

Priority order (highest impact first):

**Priority 1 — Safe Contracts (most impactful for false positive reduction)**

The model over-predicts because it has never learned "what a safe contract looks like
when it contains an external call." We need safe contracts that contain all the same
surface patterns that trigger false positives, but with correct protections.

Target: **500+ safe contracts** including:
- Contracts using CEI (checks-effects-interactions) with low-level `call()`
- Contracts using typed interface calls (no raw `address`)
- Contracts using `transfer()` or `send()` for ETH movements (safe alternatives)
- Contracts with pull-payment patterns
- Contracts with ReentrancyGuard modifier
- Contracts with proper return-value checking on `call()`
- Contracts with `unchecked {}` used for gas optimization (not overflow risk)
- Contracts with bounded loops (gas-safe)

**Generation strategy (automated, highest yield):**
Rather than sourcing safe contracts from external libraries (which may already overlap
with the training corpus), generate them by *mutating existing vulnerable contracts* into
their safe equivalents. This directly teaches the model the structural distinctions:

```
Reentrancy → CEI-safe version:
  Take every reentrancy training contract. Swap the order of the external call and
  the state-write (balance[msg.sender] = 0 before call, not after). Label as safe.
  Result: model sees structurally near-identical graphs where only CFG edge order differs.

MishandledException / UnusedReturn → safe version:
  Take every bare `call()` and wrap it: `(bool ok,) = addr.call(...); require(ok);`
  Label as safe (return value now captured and checked).

CallToUnknown → typed-interface version:
  Take every raw-address `call()` and replace with a typed interface call
  `ITarget(addr).method()`. Label as safe (target is typed).

unchecked DoS-safe → bounded loop:
  Take every unbounded-loop DoS contract and add a max-iteration guard
  (`require(arr.length <= 100)`). Label as safe (loop is bounded).
```

This mutation approach ensures maximum structural overlap between vulnerable and safe
examples, which forces the model to learn the *discriminating structural feature*
rather than any incidental syntactic difference. Write as a script
`ml/scripts/generate_safe_variants.py` that processes the existing corpus.

External sources (supplement, not primary): OpenZeppelin library contracts, Solmate,
forge-std examples. These provide style diversity but may already be in the corpus.

**Priority 2 — DenialOfService (critical data starvation)**

Target: **300+ DoS contracts** including:
- Unbounded loops over unbounded arrays
- Loops with external calls inside
- Gas-intensive operations (large storage reads, nested mappings)
- Contracts that can be bricked by a single large-gas transaction
- `SELFDESTRUCT` abuse patterns (gas refund DoS)

Sources: SmartBugs dataset, SWC-registry issue #128, manually generated.

**Priority 3 — IntegerUO with `unchecked` (Solidity 0.8+ gap)**

Target: **200+ contracts** using `unchecked {}` with actual overflow risk (0.8+):
- Unchecked counter increments that can wrap
- Unchecked balance arithmetic
- Unchecked loop indices

Sources: Generate from templates. The pattern is simple enough to automate.

**Priority 4 — CallToUnknown disambiguation**

Target: **200+ contracts** that call typed interfaces (NOT vulnerable to CallToUnknown)
to teach the model that `interface.method()` is different from `address(x).call{...}("")`.

**Priority 5 — TransactionOrderDependence and MishandledException**

Target: **100+ each** of additional examples to improve precision.

### 3.3 Data Pipeline Steps

1. **Source/generate new contracts** per priority list above.
2. **Compile each with the appropriate solc version** (matching pragma).
3. **Run graph extractor** with new v2 schema to produce `.pt` graph files.
4. **Run tokenizer** to produce `.pt` token files.
5. **Label each contract** in `multilabel_index.csv`:
   - Safe contracts: all labels = 0.
   - Vulnerable contracts: use Slither detectors + manual review per label.
6. **Re-extract ALL existing ~68K contracts** with new v2 schema
   (`python ml/src/data_extraction/ast_extractor.py --force`).
7. **Regenerate splits with explicit separation of augmented data:**
   - **Original v4 validation set** (the BCCC-derived hold-out): kept intact, used
     exclusively for F1-macro metric tracking and comparability to v4 results.
     Augmented contracts are **never added to this set**.
   - **Augmented contracts**: added to the training set only. A small held-out slice
     (10%) of augmented contracts forms a secondary "behavioral val" set used to
     monitor whether safe-contract specificity and feature-specific recall are
     improving, but this is **not** used for the F1-macro gate.
   - The original test set is also kept clean (no augmented data).
   This design means: validation F1-macro is directly comparable to v4, and behavioral
   generalization is measured through the manual-test suite (§8.2), not through an
   in-distribution eval on augmented data.
8. **Validate** with updated `validate_graph_dataset.py` (checks dim=13, edge types 0-6).

### 3.4 Labeling Protocol for Augmented Contracts

Every new contract must be labeled using at least two sources:

1. **Slither detectors** (`slither <contract.sol> --json -`) — automated first pass.
2. **Manual review** by a human — especially for safe contracts where Slither
   false positives must be caught.
3. For safe contracts: verify Slither produces zero findings AND manually confirm
   the protective patterns are correctly implemented.

Label noise is the main risk for augmented data. A mislabeled "safe" contract that
actually has a vulnerability teaches the model the wrong thing.

---

## 4. Model Architecture Choices — Decision Log

This section records decisions and the reasoning, so they don't get relitigated.

### 4.1 Why Remove the `reentrant` Slither Flag

The `reentrant = Slither.is_reentrant` feature gives the model pre-computed
vulnerability information. During training, whenever a contract is vulnerable to
reentrancy, the GNN receives `reentrant=1.0`. The model can achieve high Reentrancy
F1 simply by learning "when feature[5]=1.0, predict Reentrancy=1." This is not
reentrancy detection — it is Slither wrapper.

In inference on real contracts, Slither's `is_reentrant` is noisy (false positives
and negatives). The model inherits that noise directly.

We want the model to detect reentrancy from structural patterns (call order vs. state
write order), not from Slither's label. Removing the feature forces the model to use
the new CFG edges and semantic features to make its own determination.

**Risk:** Reentrancy F1 may drop initially if the model was relying heavily on this
shortcut. Mitigation: control-flow edges + `return_ignored` + `external_call_count`
should more than compensate.

### 4.2 Why Add Statement-Level CFG Nodes (Not Just Function-Level Edges)

The alternative (adding CONTROL_FLOW edges *between function nodes*) does not help.
All statements in a function share the same function-level graph node. Adding an edge
from function A to function A (self-loop) or between two functions in execution order
does not encode *intra*-function statement ordering.

The only meaningful encoding of "call before write" is to have separate graph nodes for
the call statement and the write statement, connected by a directed CONTROL_FLOW edge.
This requires statement-level nodes.

The cost is increased graph size. A function with 10 CFG nodes (basic blocks) produces
10 child graph nodes plus 1 function node = 11 nodes total (was 1). Graphs will be
larger. This is a cost we must pay to fix the root cause.

**Memory/compute impact:** Average Slither CFG node count per function varies but is
typically 5–20. For a contract with 10 functions averaging 10 nodes each, total nodes
increase from ~20 to ~120. GNN message passing scales O(N+E) — this is a 6× increase
in N. DataLoader batch sizes may need reducing from 32 to 16. Accept this.

### 4.3 Why 4 GNN Layers With Residuals

A simple function like:
```solidity
function withdraw(uint amount) external {
    require(balance[msg.sender] >= amount);    // node 0
    (bool ok,) = msg.sender.call{value:amount}(""); // node 1 — CALL
    balance[msg.sender] -= amount;             // node 2 — WRITE
}
```
The CFG path is node0 → node1 → node2 (diameter=2, reachable in 2 hops). A 3-layer
GNN reaches this. But with exception handling, modifiers, and branch structures, real
function CFGs have diameter 4–6. We use 4 layers as a practical minimum.

Residual connections prevent vanishing gradients in the deeper network. Adding `x2 = f(x) + x`
(where shapes match) is a standard technique; it adds negligible parameters.

### 4.4 GATv2 vs GAT

GATv2 (`GATv2Conv` in PyG) corrects a known expressiveness limitation of GAT: in GAT,
the attention score is computed from a *fixed* linear combination of the source and
destination node features, which is equivalent to a rank-1 approximation. GATv2 allows
the attention score to depend jointly on both nodes in a more expressive way.

For v5.0: keep GAT (GATConv) for stability and to isolate the impact of the new features
and CFG edges. GATv2 is a v5.1 consideration if v5.0 results are limited.

### 4.5 Three-Eye Classifier Architecture

**Decision:** Add two independent classifier inputs alongside the existing fused
representation — a GNN-only "structural eye" and a Transformer-only "semantic eye" —
so the classifier receives direct, unmediated signals from each modality.

**The problem with a single fused bottleneck:**
The current v4 architecture (and the initial v5 plan before this revision) routes all
information through a single 128-dim fused vector before the classifier. This creates
two failure modes:

1. If cross-attention learns a poor mapping early in training, gradients cannot drive
   GNN or Transformer improvements independently — both are blamed equally for the
   fusion's error, even if one modality is already producing useful embeddings.
2. Different vulnerability classes benefit from different signal types. DoS is best
   detected from graph structure (`has_loop`, node counts). MishandledException is best
   detected from token sequence (`(bool ok,) = call(...)`). Forcing both through the
   same 128-dim compression discards the natural specialization.

**Why three eyes, not two:**
The fused eye is kept because the bidirectional cross-attention genuinely produces
information that neither modality holds alone — it encodes "which structural patterns
co-occur with which token sequences." The individual eyes are additive, not replacements.
Each eye answers a different question:

| Eye | What it answers | Inductive bias |
|---|---|---|
| GNN eye | "What structural patterns exist in the graph?" | Max+mean pool: captures both presence (is any node dangerous?) and distribution (what's typical?) |
| Transformer eye | "What does the raw source code say?" | CLS token: CodeBERT's 12-layer hierarchical summary, fully order-aware |
| Fused eye | "How do structural patterns and tokens co-locate?" | Cross-attention: encodes joint evidence that neither modality holds alone |

**Why max+mean pool for the GNN eye (not plain mean):**
Vulnerability detection is an existential claim: a contract with ONE reentrancy-vulnerable
function is vulnerable, regardless of how many safe functions it also contains. Global
max pool captures "is there at least one node exhibiting feature X?" — the right inductive
bias. Global mean pool captures the "average node" — also useful for overall contract
character. Concatenating both (256-dim) and projecting to 128-dim lets the model choose
its own weighting between the two pooling strategies.

**Why the existing CLS token for the Transformer eye (not a new learnable one):**
CodeBERT's CLS token at position 0 of `last_hidden_state` is already a hierarchical
sequence summary produced by 12 layers of bidirectional self-attention over all 512
positions. It is order-aware by construction and distinct from the masked-mean pool
used inside CrossAttentionFusion. Adding a *new* learnable CLS token would require
prepending it before CodeBERT runs, shifting all positional embeddings — a structural
change to the Transformer that is not warranted when an equivalent representation already
exists at zero cost.

**Why this is v5, not v5.1:**
Deferring this to v5.1 would mean running a full 60-epoch training run, then changing
the classifier head, then running another full run. The parameter addition is ~130K
(negligible), the code change is ~20 lines in sentinel_model.py, and the architectural
benefit is highest at the start of training when gradients are most uncertain. Integrating
it now costs nothing in compute and removes a known architectural weakness from the start.

**Risk: eye dominance:**
If one eye quickly learns a shortcut, the classifier may weight it heavily and suppress
the others.

Primary mitigation — **auxiliary loss (§2.6B, §2.5D–E):** Three per-eye auxiliary
classifier heads (`aux_gnn`, `aux_transformer`, `aux_fused`, each `Linear(128,10)`)
are added to `sentinel_model.py`. During training, the total loss is:

```
loss = main_loss + λ * (loss_gnn_eye + loss_transformer_eye + loss_fused_eye)
```

where λ=0.1 (`aux_loss_weight` in TrainConfig). This forces each eye to produce
independently useful logits, regardless of how the main classifier weights the
concatenated output. The auxiliary heads are discarded at inference (`return_aux=False`
is the default in `forward()`). They add ~3.9K parameters — negligible.

This is **structural prevention**: even if the classifier learns to ignore one eye's
contribution to the concatenated vector, the auxiliary loss keeps that eye's parameters
receiving gradient signal from the training labels. Eye dominance suppression cannot
happen silently.

Diagnostic mitigation — **per-eye gradient norm logging (§2.6C):** Gradient norms
are logged every `log_interval` batches to detect if suppression somehow persists despite
the auxiliary loss. Acceptable criterion: all three eyes maintain gradient norms within
one order of magnitude of each other. If any eye's norm collapses, it is a signal to
increase λ or investigate the auxiliary loss computation.

### 4.6 Frozen vs Trained GNN/Fusion/Classifier

The v4 exp1 doc (`2026-05-10-v4-exp1-complete.md`) describes model-only resume from v3.
**Importantly, the GNN, fusion, and classifier were NOT frozen in v4** — the AdamW
optimizer trained all `requires_grad=True` parameters (confirmed in `trainer.py:818`).
Only CodeBERT was frozen via LoRA's `requires_grad=False`.

For v5: **train everything from scratch.** No pretrained checkpoint to resume from
(the schema change makes v4 weights incompatible anyway). GNN, fusion, classifier, and
LoRA adapters all start from random initialization. CodeBERT backbone stays frozen (125M
params) with LoRA adapters trained on top.

---

## 5. Training Strategy

### 5.1 Hyperparameters

```
epochs:              60 (with early stopping patience=10)
batch_size:          16 (reduced from 32 due to larger graphs)
lr:                  2e-4
weight_decay:        1e-2
warmup_pct:          0.10 (10% warmup — longer than default 5% for stability from scratch)
grad_clip:           1.0
loss_fn:             bce (start here; evaluate at epoch 10 for over-prediction)
early_stop_patience: 10
lora_r:              16 (increased from 8 for more LoRA capacity)
lora_alpha:          32 (maintain alpha/r = 2.0 scale factor)
gnn_hidden_dim:      128
gnn_heads:           8 (each head gets 16 dims)
gnn_dropout:         0.2
gnn_edge_emb_dim:    32 (increased from 16 for 6 edge types)
fusion_attn_dim:     256
fusion_num_heads:    8
fusion_output_dim:   128
```

### 5.2 Weighted Sampler

Enable `use_weighted_sampler="all-rare"` from the start (not just for DoS).
This ensures that the few DoS examples and newly augmented rare-class contracts
appear more frequently per epoch. The `_build_weighted_sampler` implementation
in `trainer.py` is already functional for this mode.

### 5.3 Training Phases

**Phase A — Smoke run (1-2 epochs, 10% data subsample)**
Use `smoke_subsample_fraction=0.10`. Verify:
- Graph loading succeeds (new schema, 13-dim features, 7 edge types)
- Forward pass completes without shape errors
- Loss decreases in the first few batches
- No CUDA OOM at batch_size=16 with larger graphs
- If OOM at batch_size=16: try batch_size=8 first. If still OOM: add gradient
  accumulation (`accumulation_steps=4` with batch_size=4 → effective batch=16).
  Gradient accumulation is already compatible with GradScaler/AMP.

**Phase B — Short run (15 epochs, full data)**
Evaluate per-class metrics at epoch 15 to check:
- Is every class showing positive F1 improvement from epoch 1?
- Is over-prediction pattern (recall >> precision) improving vs v4?
- Is DoS F1 improving with new graph features + weighted sampler?

**Phase C — Full run (60 epochs)**
Early stopping patience=10 gives the model room to recover from temporary plateaus.

### 5.4 Loss Function Escalation

At epoch 15 evaluation, check the precision/recall ratio per class:
- If ratio is still >2× (predicting 2× more positives than true), switch to FocalLoss
  with per-class alpha tuning.
- FocalLoss alpha for rare classes (DoS, CTU): α=0.75 (preserve recall signal).
- FocalLoss alpha for over-predicted classes (ExternalBug, TOD): α=0.25 (focus on
  hard examples, reduce easy positive gradient).

This requires implementing a `per_class_alpha` version of FocalLoss, which the current
`focalloss.py` does not support (it uses a scalar alpha). Add this as a new class:
`MultiLabelFocalLoss(alpha: List[float], gamma: float)`.

### 5.5 Threshold Tuning

After training completes, run `tune_threshold.py` on the full validation set.
Target thresholds per class:
- For classes with recall >> precision: raise threshold until precision ≥ 0.5.
- For classes with low recall (DoS): accept lower threshold to improve detection.

Additionally, add a **behavioral test gate**: all 20 manual-test contracts must
be passed through the model and the results compared to expected outputs.
A training run is only "accepted" if:
- F1-macro (validation) > 0.56 (v4 baseline + 2 points)
- Manual detection rate > 70% (was 15%)
- False positive rate on safe contracts < 20% (was 67%)

---

## 6. Implementation Order — Phase-by-Phase

### Phase 0 — Quick Code Quality Fixes (½ day, no re-extraction needed)

These are correctness bugs independent of the v5 schema change:

1. **Fix `in_channels=8` hardcode in `gnn_encoder.py`**
   - Change `in_channels=8` → `in_channels=NODE_FEATURE_DIM`
   - Add import: `from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM`

2. **Confirm GNN/fusion/classifier training behavior is documented**
   - Add a clear comment in `trainer.py` that GNN, fusion, and classifier ARE trained
     (not frozen) — previous documentation implied otherwise.

3. **Verify inference path uses `graph_extractor.py` (not a stale copy)**
   - Read `ml/src/inference/preprocess.py` and confirm it imports from graph_extractor.
   - If it has any hardcoded feature logic, remove it and replace with the import.

4. **Add NODE_FEATURE_DIM import to validate_graph_dataset.py**
   - Replace any hardcoded `8` with `NODE_FEATURE_DIM` so validation automatically
     adapts to schema changes.

5. **Add missing CLI flags to `ml/scripts/train.py`** *(implementation gap — Phase 5 commands fail without these)*
   - `--lora-r` (type=int, default=8) → wires to `TrainConfig.lora_r`
   - `--lora-alpha` (type=int, default=16) → wires to `TrainConfig.lora_alpha`
   - `--smoke-subsample-fraction` (type=float, default=1.0) → wires to `TrainConfig.smoke_subsample_fraction`
   - All three TrainConfig fields already exist; only the CLI argument declarations were missing.

6. **Add `--force` flag to `ml/src/data_extraction/ast_extractor.py`** *(Phase 4 command fails without it)*
   - When set, deletes `checkpoint.json` from the output directory before extraction begins.
   - This forces reprocessing of all contracts — the correct behavior for a schema change.
   - Mutually exclusive with `--resume`.

7. **Add `--freeze-val-test` flag to `ml/scripts/create_splits.py`** *(Phase 4 command fails without it)*
   - When set, keeps existing `val_indices.npy` and `test_indices.npy` unchanged.
   - All indices not already in val/test (including any new augmented rows) go to train only.
   - Prevents augmented contracts from contaminating the original v4 validation set.

**Commit these fixes before Phase 1 begins.**

### Phase 1 — Schema & Extractor Overhaul (1–2 days)

1. Update `graph_schema.py`:
   - New `FEATURE_SCHEMA_VERSION = "v2"`
   - New `NODE_FEATURE_DIM = 13`
   - New `FEATURE_NAMES` (13 entries, removing `reentrant`, adding 6 new)
   - New `NODE_TYPES["CFG_NODE"] = 8`
   - New `EDGE_TYPES["CONTAINS"] = 5`    (function → CFG_NODE child)
   - New `EDGE_TYPES["CONTROL_FLOW"] = 6` (CFG_NODE → successor)
   - New `NUM_EDGE_TYPES = 7`
   - Update invariant assertion at bottom of file

2. Update `graph_extractor.py`:
   - Remove `reentrant` from `_build_node_features()`
   - Implement 6 new feature computations (see §2.2B for pseudocode)
   - Implement `_build_control_flow_edges()` — CFG node traversal
   - Update `extract_contract_graph()` to insert CFG_NODE child nodes and
     CONTROL_FLOW edges after the function nodes
   - Update shape comment at top: `graph.x [N, 13]` not `[N, 8]`
   - Update dimension guard assertion

3. Write unit tests for new features in `ml/tests/test_preprocessing.py`:
   - Test a known-vulnerable contract (e.g., classic reentrancy) and assert:
     - CFG_NODE nodes exist
     - CONTROL_FLOW edges exist between them
     - The function node with a bare `call()` has `return_ignored=1.0` if no lvalue
     - `in_unchecked=1.0` for a function containing `unchecked {}`
     - `has_loop=1.0` for a function containing a for loop
   - Test a known-safe CEI contract and assert:
     - CFG node order encodes write-before-call (or call-after-write in CEI)

4. Smoke-test extraction on 10 contracts manually to verify features look sensible.

### Phase 2 — Model Architecture Update (½ day)

1. Update `gnn_encoder.py`:
   - `in_channels=NODE_FEATURE_DIM` (import from schema)
   - `hidden_dim=128` default
   - Add 4th GATConv layer
   - Add residual connections (conv2→3, conv3→4)
   - Update module docstring

2. Update `sentinel_model.py`:
   - `gnn_hidden_dim=128` default
   - Pass `node_dim=128` to CrossAttentionFusion
   - Update docstring

3. Update `trainer.py` TrainConfig defaults (§5.1 hyperparameters).

4. Run model instantiation unit test (`test_model.py`) to confirm no shape errors
   with the new architecture. Update test fixtures to use 13-dim features.

### Phase 3 — Data Augmentation (parallel with Phase 1 & 2, 3–5 days)

This phase can run in parallel with Phase 1 and 2 since it does not depend on them.

1. Write/source safe contract templates (priority 1 from §3.2).
2. Write/source DoS contract templates (priority 2).
3. Write `unchecked`-overflow contract templates (priority 3).
4. Write typed-interface contract templates (priority 4).
5. Label all new contracts using the protocol in §3.4.
6. Test that `extract_contract_graph()` can process all new contracts without error.

### Phase 4 — Full Re-Extraction (1 day + compute)

```bash
# Rebuild all ~68K + new graphs with v2 schema.
# --force deletes the existing checkpoint so all contracts are reprocessed.
python ml/src/data_extraction/ast_extractor.py --force

# Validate the rebuild
python ml/scripts/validate_graph_dataset.py \
  --graphs-dir ml/data/graphs \
  --check-dim 13 \
  --check-edge-types 7 \
  --check-contains-edges \
  --check-control-flow
```

Expected output: every `.pt` graph file has `x.shape[1]=13` and
`edge_attr.max() <= 6`. Any file that fails is logged and skipped.

After extraction, update splits — **preserving the original v4 val/test sets**:
```bash
# --freeze-val-test keeps val_indices.npy and test_indices.npy unchanged.
# Augmented contracts (new rows in multilabel_index.csv) go to train only.
# This preserves F1-macro comparability with v4 experiments.
python ml/scripts/create_splits.py \
  --freeze-val-test \
  --multilabel-csv ml/data/processed/multilabel_index.csv \
  --splits-dir ml/data/splits
```

**Do NOT run `create_splits.py` without `--freeze-val-test` after augmented data
has been added to `multilabel_index.csv`. A full re-split would assign augmented
contracts to val/test, violating the §3.3 guarantee.**

### Phase 5 — Smoke Run Then Full Training (3–5 days)

```bash
# Phase A: smoke run (--smoke-subsample-fraction requires train.py ≥ Phase 0 fixes)
python ml/scripts/train.py \
  --run-name v5-smoke \
  --smoke-subsample-fraction 0.10 \
  --epochs 2 \
  --batch-size 16

# Phase B: 15-epoch check (--lora-r / --lora-alpha require train.py ≥ Phase 0 fixes)
python ml/scripts/train.py \
  --run-name v5-check-15ep \
  --epochs 15 \
  --batch-size 16 \
  --lr 2e-4 \
  --weighted-sampler all-rare \
  --lora-r 16 \
  --lora-alpha 32

# Phase C: full run
python ml/scripts/train.py \
  --run-name v5-full \
  --epochs 60 \
  --batch-size 16 \
  --lr 2e-4 \
  --weighted-sampler all-rare \
  --lora-r 16 \
  --lora-alpha 32 \
  --early-stop-patience 10
```

### Phase 6 — Evaluation & Threshold Tuning (1 day)

1. Run `tune_threshold.py` on validation set.
2. Run all 20 manual-test contracts through the model.
3. Compare per-class results to v4 baseline.
4. Document results in `docs/changes/2026-05-XX-v5-results.md`.
5. If acceptance criteria met → promote to production.
6. If not → identify remaining gaps and plan v5.1.

---

## 7. Technical Specifications Reference

### 7.1 New Node Feature Vector (13 dims)

```
Index  Name                Type     For Function Nodes        For Non-Function Nodes
─────  ─────────────────── ──────── ──────────────────────── ─────────────────────────
0      type_id             float    NODE_TYPES[kind]          NODE_TYPES[kind]
1      visibility          float    VISIBILITY_MAP ordinal     VISIBILITY_MAP ordinal
2      pure                float    1.0 if func.pure          0.0
3      view                float    1.0 if func.view          0.0
4      payable             float    1.0 if func.payable       0.0
5      complexity          float    len(func.nodes)           0.0
6      loc                 float    len(source_mapping.lines) len(source_mapping.lines)
7      return_ignored      float    see §2.2B                 0.0
8      call_target_typed   float    see §2.2B                 1.0 (not applicable)
9      in_unchecked        float    see §2.2B                 0.0
10     has_loop            float    see §2.2B                 0.0
11     gas_intensity       float    see §2.2B                 0.0
12     external_call_count float    see §2.2B                 0.0
```

### 7.2 New Edge Type Vocabulary (7 types)

```
ID  Name           Direction              Semantics
──  ─────────────  ─────────────────────  ─────────────────────────────────────
0   CALLS          function → function    internal function call
1   READS          function → state_var   state variable read
2   WRITES         function → state_var   state variable write
3   EMITS          function → event       event emission
4   INHERITS       contract → contract    inheritance (MRO order)
5   CONTAINS       function → cfg_node    function owns this basic block (NEW)
6   CONTROL_FLOW   cfg_node → cfg_node    sequential execution order (NEW)
```

`CONTAINS` links each function node to every one of its CFG_NODE children. This is
what makes the CFG subgraph connected to the rest of the contract graph. Without it,
the GNN cannot pass function-level properties (payable, visibility) down to statement
nodes, nor aggregate statement-level patterns back up to the function node.

### 7.3 New Node Type Vocabulary (9 types)

```
ID  Name          What it represents
──  ────────────  ──────────────────────────────────────────
0   STATE_VAR     Contract state variable
1   FUNCTION      Regular function
2   MODIFIER      Solidity modifier
3   EVENT         Solidity event
4   FALLBACK      Fallback function
5   RECEIVE       Receive function
6   CONSTRUCTOR   Constructor
7   CONTRACT      Contract node (root)
8   CFG_NODE      Intra-function control-flow basic block (NEW)
```

### 7.4 GNN Architecture (v5)

```
Input:    x [N, 13], edge_index [2, E], edge_attr [E] (int64, 0-6)
          batch [N] (node-to-graph mapping)

Layer 1 (GATConv): in=13, out=128 (8 heads × 16), concat=True, edge_dim=32
  ReLU + Dropout(0.2)

Layer 2 (GATConv): in=128, out=128 (8 heads × 16), concat=True, edge_dim=32
  ReLU + Dropout(0.2)
  + residual from Layer 1 output

Layer 3 (GATConv): in=128, out=128 (8 heads × 16), concat=True, edge_dim=32
  ReLU + Dropout(0.2)
  + residual from Layer 2 output

Layer 4 (GATConv): in=128, out=128, heads=1, concat=False, edge_dim=32
  (no activation — passed directly to CrossAttentionFusion)

Edge embedding: nn.Embedding(7, 32) → 224 trainable parameters
  (covers CALLS, READS, WRITES, EMITS, INHERITS, CONTAINS, CONTROL_FLOW)
  Number of layers is a constructor argument; 4 is the default, adjustable via TrainConfig.

Output: node_embeddings [N, 128], batch [N]
  — pooling for the GNN eye happens in SentinelModel, NOT here:
    global_max_pool  → [B, 128]
    global_mean_pool → [B, 128]
    cat → [B, 256] → gnn_eye_proj → gnn_eye [B, 128]
```

### 7.5 Full Model Architecture (v5) — Three-Eye Classifier

```
Input: (PyG graph batch, input_ids [B, 512], attention_mask [B, 512])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GNN PATH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GNNEncoder(in=13, hidden=128, layers=4, heads=8, edge_types=7)
  → node_embs [N, 128]

  ┌─────────────────────────────────────────────────────┐
  │  GNN Eye (structural opinion)                       │
  │  global_max_pool(node_embs, batch)  → [B, 128]      │
  │  global_mean_pool(node_embs, batch) → [B, 128]      │
  │  cat → [B, 256]                                     │
  │  gnn_eye_proj: Linear(256,128) + ReLU + Dropout     │
  │  → gnn_eye [B, 128]                                 │
  └─────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSFORMER PATH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CodeBERT(frozen, 125M) + LoRA(r=16, alpha=32, q+v)
  → token_embs [B, 512, 768]   (all positions)

  ┌─────────────────────────────────────────────────────┐
  │  Transformer Eye (semantic opinion)                 │
  │  token_embs[:, 0, :]  → CLS token [B, 768]         │
  │  (12-layer BERT hierarchical sequence summary;      │
  │   order-aware, distinct from masked-mean pool)      │
  │  transformer_eye_proj: Linear(768,128)+ReLU+Dropout │
  │  → transformer_eye [B, 128]                         │
  └─────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CROSS-ATTENTION FUSION (uses both paths)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CrossAttentionFusion(node_dim=128, token_dim=768,
                     attn_dim=256, num_heads=8, output_dim=128)
  node_embs attend to token_embs (→ structurally-enriched nodes)
  token_embs attend to node_embs (→ semantically-enriched tokens)
  masked mean pool both → concat → project

  ┌─────────────────────────────────────────────────────┐
  │  Fused Eye (joint structural+semantic opinion)      │
  │  → fused_eye [B, 128]                               │
  └─────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THREE-EYE CLASSIFIER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cat([gnn_eye, transformer_eye, fused_eye]) → [B, 384]
Linear(384, 10) → logits [B, 10]   (no sigmoid — applied externally)

Each class learns its own weighting across the three eyes.
No information is discarded before the final decision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUXILIARY HEADS (training only — discarded at inference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
forward(..., return_aux=True)  ← called by trainer only

  aux_gnn         = Linear(128, 10)(gnn_eye)          → [B, 10]
  aux_transformer = Linear(128, 10)(transformer_eye)   → [B, 10]
  aux_fused       = Linear(128, 10)(fused_eye)         → [B, 10]

  returns (logits, {"gnn": aux_gnn, "transformer": aux_transformer, "fused": aux_fused})

  Training loss = main_loss + 0.1 × (loss_aux_gnn + loss_aux_tf + loss_aux_fused)

forward(..., return_aux=False)  ← inference default
  returns logits [B, 10]  — unchanged interface; callers need zero changes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINABLE PARAMETER ESTIMATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GNNEncoder (4 layers, edge emb 7×32):         ~88K
LoRA adapters (r=16, 12L × Q+V):              ~590K
CrossAttentionFusion:                         ~600K
gnn_eye_proj (Linear 256→128 + act):          ~33K
transformer_eye_proj (Linear 768→128):        ~98K
Classifier (Linear 384→10):                   ~3.9K
Auxiliary heads (3 × Linear 128→10):          ~3.9K
─────────────────────────────────────────────────────
Total trainable:                              ~1.42M
Frozen (CodeBERT backbone):                   ~125M
```

---

## 8. Acceptance Criteria

A v5 model is accepted for promotion only when ALL of the following hold:

### 8.1 Validation Set Metrics
- F1-macro (tuned thresholds) > **0.5622** (v4 + 2 points)
- No per-class F1 more than **0.10** below its v4 value (floor rule)
  (Relaxed from 0.05 — removing the `reentrant` Slither shortcut may cause a temporary
  Reentrancy F1 drop; classes already near 0.40 in v4 should not fail on a 0.05 move.
  0.10 is realistic for a from-scratch retrain that fundamentally changes the feature
  space. Tighten back to 0.05 in v5.1 once the architecture is settled.)
- DenialOfService tuned F1 > **0.40** (same as v4, but with more data, this should be easy)

### 8.2 Behavioral Test Suite (20 manual contracts)
- Detection rate (true positives / expected positives): > **70%** (was 15%)
- Safe-contract specificity (true negatives / safe contracts): > **66%** (was 33%)
- False positive rate on safe contracts with CEI/typed calls/pull-payment: **< 20%**

### 8.3 Per-Vulnerability Checks
| Vulnerability | Minimum Behavioral Test Pass Rate |
|---|---|
| Reentrancy (classic, cross-function) | 2/3 detected |
| Safe CEI contracts classified clean | 2/3 correct |
| `unchecked` IntegerUO (Solidity 0.8+) | 2/3 detected |
| DenialOfService (unbounded loop) | 1/2 detected |
| MishandledException (ignored return) | 2/3 detected |
| CallToUnknown (raw address call) | 2/3 detected, typed interface calls NOT flagged |

---

## 9. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Slither `func.nodes` API or IR differs across Slither versions | Medium | High | **Pin Slither version** in `ml/pyproject.toml` (e.g., `slither-analyzer==0.10.x`). Document required version in README. Test on a sample of 100 contracts before full extraction. Add try/except with fallback for `return_ignored`, `call_target_typed`, and `in_unchecked` (see §2.2B). |
| `return_ignored` / `call_target_typed` IR fallback is imprecise | Low | Medium | Text-based fallbacks are conservative (may under-count). Log all fallback uses to a separate log file during extraction. After extraction, audit contracts where fallback was triggered to estimate false-negative rate. |
| Larger graphs (CFG nodes) cause CUDA OOM at batch_size=16 | Medium | High | Reduce to batch_size=8 as first step. If still OOM, add gradient accumulation: `accumulation_steps=4, batch_size=4` → effective batch=16. GradScaler already compatible. Profile during smoke run Phase A before committing to a batch size. |
| New features computed incorrectly (bugs in Slither IR iteration) | High | High | Unit test with hand-crafted contracts where expected feature values are known exactly. Do not proceed to full extraction until unit tests pass. |
| Augmented data introduces label noise | Medium | Medium | Manual review of 10% random sample of new contracts before adding to dataset. Any contract with uncertain labeling goes to a separate review set. |
| v5 over-fits to augmented data, regresses on original validation set | Low | High | Keep original v4 validation split intact as a secondary evaluation. Do not include original val set in training even after augmentation. |
| DoS F1 still poor despite 300 new samples | Medium | Medium | Accept DoS as a known-hard class; acceptance criterion is 0.40 (same as v4). More data helps but DoS is architecturally hard to distinguish without execution traces. |
| Full re-extraction takes >24 hours | Medium | Low | Parallelise across multiple CPU workers. The `ast_extractor.py` script already supports `--num-workers`. Accept longer compute time. |

---

## 10. Files Changed Summary

| File | Change Type | Phase |
|---|---|---|
| `ml/src/preprocessing/graph_schema.py` | Schema update (dim, types, version) | 1 |
| `ml/src/preprocessing/graph_extractor.py` | Feature removal, 6 new features, CFG edges, CFG nodes | 1 |
| `ml/src/models/gnn_encoder.py` | Import fix, 4th layer, residuals, hidden_dim=128 | 2 |
| `ml/src/models/sentinel_model.py` | Three-eye architecture: `gnn_eye_proj`, `transformer_eye_proj`, classifier `Linear(384,10)`, max+mean GNN pool, CLS transformer eye; auxiliary heads `aux_gnn/aux_transformer/aux_fused`; `return_aux` interface | 2 |
| `ml/src/training/trainer.py` | New defaults, auxiliary loss training loop (`return_aux=True`, λ=0.1), per-eye gradient norm logging, per-class calibration log (pred/true ratio), per-class FocalLoss option | 2 |
| `ml/src/training/focalloss.py` | Add `MultiLabelFocalLoss(alpha: List[float])` | 2 |
| `ml/scripts/validate_graph_dataset.py` | Replace hardcoded dims with NODE_FEATURE_DIM | 0 |
| `ml/src/inference/preprocess.py` | Verify it imports from graph_extractor (no stale copy) | 0 |
| `ml/scripts/train.py` | Add `--lora-r`, `--lora-alpha`, `--smoke-subsample-fraction` CLI flags | 0 |
| `ml/src/data_extraction/ast_extractor.py` | Add `--force` flag (delete checkpoint → full re-extraction) | 0 |
| `ml/scripts/create_splits.py` | Add `--freeze-val-test` flag (preserve val/test when adding augmented data) | 0 |
| `ml/scripts/generate_safe_variants.py` | NEW: mutation-based safe contract generator (CEI, typed-interface, bounded-loop variants) | 3 |
| `ml/tests/test_preprocessing.py` | New unit tests for CFG edges and new features | 1 |
| `ml/tests/test_model.py` | Update fixtures for 13-dim inputs | 2 |
| `multilabel_index.csv` | Add augmented contracts | 3 |
| `ml/data/splits/` | Update with `--freeze-val-test` after augmentation | 4 |
| `ml/data/graphs/*.pt` | Full re-extraction with v2 schema | 4 |
| `ml/data/tokens/*.pt` | Re-pair with new graph files | 4 |

---

## 11. What This Does NOT Change

- `ml/src/models/fusion_layer.py` — the CrossAttentionFusion implementation is correct.
- `ml/src/data_extraction/tokenizer.py` — token handling is independent of graph schema.
- `ml/src/datasets/dual_path_dataset.py` — dataset loading is schema-agnostic (reads
  whatever `x.shape` the graph has).
- `agents/` module — no changes.
- `zkml/` module — proxy model will need to be retrained after v5 is complete, but
  the architecture there is independent.
- `contracts/` — Solidity contracts unchanged.

---

## 12. Immediately After This Document — What to Do First

1. **Checkout the dev branch** (`claude/debug-model-overprediction-Q5aAC`).
2. **Phase 0** first: fix `in_channels=8` hardcode, verify inference path, update
   validate script. These are safe changes that improve code quality immediately.
3. **Then Phase 1**: Update schema and extractor. Unit test before any extraction.
4. **Data augmentation (Phase 3)** can start in parallel on a different branch.
5. **Do not begin full re-extraction (Phase 4)** until Phase 1 unit tests pass.
6. **Do not begin training (Phase 5)** until Phase 4 extraction completes and
   `validate_graph_dataset.py` reports zero errors.

The single biggest risk in this plan is moving to full re-extraction before the
feature extraction logic is verified on hand-crafted test cases. Resist the urge to
skip unit testing. A wrong `return_ignored` implementation on 68K graphs wastes a
full extraction run.
