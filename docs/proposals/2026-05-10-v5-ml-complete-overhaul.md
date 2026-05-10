# SENTINEL v5 — Complete ML Module Overhaul
# Comprehensive Technical Plan

| Field        | Value                                   |
|--------------|-----------------------------------------|
| Date         | 2026-05-10                              |
| Status       | ACTIVE — Implementation Guide           |
| Supersedes   | v4 exp1 (`multilabel-v4-finetune-lr1e4_best.pt`) |
| Author       | Post-audit synthesis (code + results)   |
| Priority     | Critical — production model is broken   |

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
4. `EDGE_TYPES`: add `"CONTROL_FLOW": 5`
5. `NUM_EDGE_TYPES`: `5` → `6`

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
    "CONTROL_FLOW": 5,   # CFG node → successor CFG node (execution order)
}
NUM_EDGE_TYPES = 6
```

### 2.2 `ml/src/preprocessing/graph_extractor.py`

**Changes:**

**A. Remove `reentrant` from `_build_node_features()`**
The `reentrant = 1.0 if getattr(obj, "is_reentrant", False) else 0.0` line is deleted.

**B. Add 6 new features to `_build_node_features()`**

Computation logic for each:

```
return_ignored:
    For Function nodes: iterate func.slithir_operations (Slither IR).
    Check for any LowLevelCall or HighLevelCall whose return value is
    not subsequently assigned. Slither's IR Call objects have a .lvalue
    attribute — if None, the return is discarded.
    For non-Function nodes: 0.0

call_target_typed:
    For Function nodes: iterate func.high_level_calls and func.low_level_calls.
    For each, check the receiver type. Slither's Variable.type gives
    ContractType or AddressType. If any call is to AddressType (raw address),
    set to 0.0 (not fully typed). If all are ContractType or there are
    no external calls, set to 1.0.
    For non-Function nodes: 1.0 (not applicable, default safe)

in_unchecked:
    For Function nodes: check func.contains_assembly or iterate
    func.nodes for node_type == NodeType.TRY. For Slither ≥0.9.3,
    use func.nodes and check for UNCHECKED_BEGIN type.
    Simpler fallback: regex on func.source_mapping.content for
    the literal string "unchecked" — reliable for Solidity 0.8+.
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
a new `CONTAINS` edge (or simply reuse CALLS) and to each other via CONTROL_FLOW.

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

**C. Update `NUM_EDGE_TYPES` import (automatically picks up 6 from schema).**
The `nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)` line already imports the constant —
no change needed in that line, but the value will automatically be 6.

**D. Update default `hidden_dim` from 64 to 128:**
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
- Update `gnn_hidden_dim` default from 64 to 128.
- Pass `gnn_hidden_dim=128` to `CrossAttentionFusion(node_dim=128, ...)`.
- Update docstring to reflect new architecture.

No structural changes to the forward pass — it already unpacks `(node_embs, batch)`
from GNN and passes everything correctly to fusion.

### 2.6 `ml/src/training/trainer.py`

**Changes:**

**A. Update `TrainConfig` defaults for v5:**
```python
gnn_hidden_dim:   int   = 128     # was 64
gnn_layers:       int   = 4       # new field (requires GNNEncoder change)
checkpoint_name:  str   = "multilabel-v5-fresh_best.pt"
epochs:           int   = 60      # longer run for fresh-from-scratch training
lr:               float = 2e-4    # slightly lower than default 3e-4 for stability
```

**B. Add per-class threshold persistence to evaluation:**
After each epoch, if we improved, also save the optimal per-class thresholds by running
a quick sweep over [0.3, 0.35, 0.40, ... 0.95] per class on the validation set.
Currently this only happens in the separate `tune_threshold.py` script.
Inline it into the training loop so the best checkpoint always ships with calibrated
thresholds.

**C. Loss function for v5:**
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
assert graph.edge_attr.max() < NUM_EDGE_TYPES  # will now check for 6
```
Also add an assertion that at least some graphs contain `CONTROL_FLOW` edges (type 5),
to catch silent extraction failures.

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

Sources: OpenZeppelin library contracts, Solmate, forge-std examples, manually written.

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
7. **Regenerate stratified splits** ensuring:
   - Validation split preserved from v4 (for comparability) OR
   - New stratified split if the augmented data changes distribution enough
     to require it. Given the scale of augmentation, regenerate from scratch
     with stratified k-fold ensuring all classes appear in val and test.
8. **Validate** with updated `validate_graph_dataset.py` (checks dim=13, edge types 0-5).

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

### 4.5 Frozen vs Trained GNN/Fusion/Classifier

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
- Graph loading succeeds (new schema, 13-dim features, 6 edge types)
- Forward pass completes without shape errors
- Loss decreases in the first few batches
- No CUDA OOM at batch_size=16 with larger graphs

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

**Commit these fixes before Phase 1 begins.**

### Phase 1 — Schema & Extractor Overhaul (1–2 days)

1. Update `graph_schema.py`:
   - New `FEATURE_SCHEMA_VERSION = "v2"`
   - New `NODE_FEATURE_DIM = 13`
   - New `FEATURE_NAMES` (13 entries, removing `reentrant`, adding 6 new)
   - New `NODE_TYPES["CFG_NODE"] = 8`
   - New `EDGE_TYPES["CONTROL_FLOW"] = 5`
   - New `NUM_EDGE_TYPES = 6`
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
# Rebuild all ~68K + new graphs with v2 schema
python ml/src/data_extraction/ast_extractor.py --force

# Validate the rebuild
python ml/scripts/validate_graph_dataset.py \
  --graphs-dir ml/data/graphs \
  --check-dim 13 \
  --check-edge-types 6 \
  --check-control-flow
```

Expected output: every `.pt` graph file has `x.shape[1]=13` and
`edge_attr.max() <= 5`. Any file that fails is logged and skipped.

After extraction, regenerate splits:
```bash
python ml/scripts/create_splits.py --stratified --seed 42
python ml/scripts/build_multilabel_index.py
```

### Phase 5 — Smoke Run Then Full Training (3–5 days)

```bash
# Phase A: smoke run
python ml/scripts/train.py \
  --run-name v5-smoke \
  --smoke-subsample-fraction 0.10 \
  --epochs 2 \
  --batch-size 16

# Phase B: 15-epoch check
python ml/scripts/train.py \
  --run-name v5-check-15ep \
  --epochs 15 \
  --batch-size 16 \
  --lr 2e-4 \
  --use-weighted-sampler all-rare \
  --lora-r 16 \
  --lora-alpha 32

# Phase C: full run
python ml/scripts/train.py \
  --run-name v5-full \
  --epochs 60 \
  --batch-size 16 \
  --lr 2e-4 \
  --use-weighted-sampler all-rare \
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

### 7.2 New Edge Type Vocabulary (6 types)

```
ID  Name           Direction              Semantics
──  ─────────────  ─────────────────────  ─────────────────────────────────────
0   CALLS          function → function    internal function call
1   READS          function → state_var   state variable read
2   WRITES         function → state_var   state variable write
3   EMITS          function → event       event emission
4   INHERITS       contract → contract    inheritance (MRO order)
5   CONTROL_FLOW   cfg_node → cfg_node    sequential execution order
```

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
Input:    x [N, 13], edge_index [2, E], edge_attr [E] (int64, 0-5)
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

Edge embedding: nn.Embedding(6, 32) → 192 trainable parameters

Output: node_embeddings [N, 128], batch [N]
```

### 7.5 Full Model Architecture (v5)

```
Input: (PyG graph batch, input_ids [B, 512], attention_mask [B, 512])

GNN path:
  GNNEncoder(in=13, hidden=128, layers=4, heads=8) → [N, 128]

Transformer path:
  CodeBERT(frozen, 125M) + LoRA(r=16, alpha=32, q+v) → [B, 512, 768]

Fusion:
  CrossAttentionFusion(node_dim=128, token_dim=768, attn_dim=256,
                       num_heads=8, output_dim=128) → [B, 128]

Classifier:
  Linear(128, 10) → [B, 10] logits (no sigmoid)

Trainable parameters (estimated):
  GNNEncoder:           ~85K   (4 layers + edge embedding, 13→128)
  LoRA adapters:        ~590K  (r=16, 12 layers × Q+V)
  CrossAttentionFusion: ~600K  (projections + attention + output)
  Classifier:           ~1.3K
  Total trainable:      ~1.28M
  Frozen (CodeBERT):    ~125M
```

---

## 8. Acceptance Criteria

A v5 model is accepted for promotion only when ALL of the following hold:

### 8.1 Validation Set Metrics
- F1-macro (tuned thresholds) > **0.5622** (v4 + 2 points)
- No per-class F1 more than 0.05 below its v4 value (floor rule)
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
| Slither `func.nodes` API differs across Slither versions | Medium | High | Pin Slither version in `ml/pyproject.toml`. Test on a sample of contracts before full extraction. Add try/except with fallback to no-CFG extraction (emit warning). |
| `in_unchecked` detection via `func.nodes` node types unreliable | Medium | Medium | Fall back to regex on `func.source_mapping.content` for `"unchecked"`. Both approaches implemented; use API first, regex as backup. |
| Larger graphs (CFG nodes) cause CUDA OOM at batch_size=16 | Medium | High | Reduce to batch_size=8 if needed. Profile peak GPU memory during smoke run Phase A. |
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
| `ml/src/models/sentinel_model.py` | gnn_hidden_dim=128, updated defaults | 2 |
| `ml/src/training/trainer.py` | New defaults, inline threshold tuning, per-class FocalLoss option | 2 |
| `ml/src/training/focalloss.py` | Add `MultiLabelFocalLoss(alpha: List[float])` | 2 |
| `ml/scripts/validate_graph_dataset.py` | Replace hardcoded dims with NODE_FEATURE_DIM | 0 |
| `ml/src/inference/preprocess.py` | Verify it imports from graph_extractor (no stale copy) | 0 |
| `ml/tests/test_preprocessing.py` | New unit tests for CFG edges and new features | 1 |
| `ml/tests/test_model.py` | Update fixtures for 13-dim inputs | 2 |
| `multilabel_index.csv` | Add augmented contracts | 3 |
| `ml/data/splits/` | Regenerate after augmentation | 4 |
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
