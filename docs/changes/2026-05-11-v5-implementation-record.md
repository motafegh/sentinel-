# SENTINEL v5 Implementation Record
**Date:** 2026-05-11  
**Proposal:** `docs/proposals/2026-05-10-v5-ml-complete-overhaul-FINAL-v1.6.md`  
**Status:** Phases 0–3 complete. Phases 4–6 pending (re-extraction, training, evaluation).

---

## 0. Context

v4 cleared validation gate (tuned F1-macro 0.5422) but collapsed on 20 hand-crafted contracts
(detection rate 15%, specificity 33%). Root cause: the graph had no sense of execution order.
v4's `GNNEncoder` could not distinguish "call before write" (reentrancy-vulnerable) from
"write before call" (CEI-safe) because `func.sons` was never accessed and no CFG subgraph
was built. v5 is a clean rebuild of every layer that touches the graph.

The proposal went through three external critic rounds (Rev 1.4–1.6), fixing ten issues before
implementation began. Two were critical blockers that would have made the code silently wrong:
`node_metadata` never specced and `graph_idx = len(...)` left as a Python ellipsis placeholder.

---

## 1. Schema Changes — `ml/src/preprocessing/graph_schema.py`

### What Changed

| Constant | v4 | v5 |
|---|---|---|
| `FEATURE_SCHEMA_VERSION` | `"v1"` | `"v2"` |
| `NODE_FEATURE_DIM` | 8 | 12 |
| `NUM_EDGE_TYPES` | 5 | 7 |
| `NUM_NODE_TYPES` | 8 | 13 |

### New Node Feature Vector (12 dims, order LOCKED)

| Idx | Name | Function Nodes | Non-Function |
|---|---|---|---|
| 0 | `type_id` | `NODE_TYPES[kind]` (0–12) | `NODE_TYPES[kind]` |
| 1 | `visibility` | `VISIBILITY_MAP` ordinal 0–2 | same |
| 2 | `pure` | 1.0 if pure | 0.0 |
| 3 | `view` | 1.0 if view | 0.0 |
| 4 | `payable` | 1.0 if payable | 0.0 |
| 5 | `complexity` | `len(func.nodes)` | 0.0 |
| 6 | `loc` | `len(source_mapping.lines)` | same |
| 7 | `return_ignored` | 0.0 / 1.0 / **-1.0** (sentinel) | 0.0 |
| 8 | `call_target_typed` | 0.0 / 1.0 / **-1.0** (sentinel) | 1.0 (N/A) |
| 9 | `in_unchecked` | 1.0 if inside `unchecked {}` | **0.0** (never inherited) |
| 10 | `has_loop` | 1.0 if loop present | 0.0 |
| 11 | `external_call_count` | log1p(count)/log1p(20) | 0.0 |

**Removed from v4:** `reentrant` (Slither shortcut — labels the answer), `gas_intensity` (circular).

**Sentinel values (-1.0):** `return_ignored` and `call_target_typed` use -1.0 when IR or source
is unavailable. Defaulting to 0.0 (safe) on failure caused systematic false negatives in v4.

**`in_unchecked` for CFG nodes is always 0.0.** Never inherited from the parent function.
A function with any `unchecked {}` block would otherwise mark all its CFG children as
`in_unchecked=1.0`, including statements outside the unchecked scope.

### New Edge Type Vocabulary (7 types)

| ID | Name | Direction | Semantics |
|---|---|---|---|
| 0 | `CALLS` | function → function | internal function call |
| 1 | `READS` | function → state_var | state variable read |
| 2 | `WRITES` | function → state_var | state variable write |
| 3 | `EMITS` | function → event | event emission |
| 4 | `INHERITS` | contract → contract | inheritance (MRO order) |
| 5 | `CONTAINS` | function → cfg_node | function owns this CFG node |
| 6 | `CONTROL_FLOW` | cfg_node → cfg_node | directed execution order |

### New Node Types (13 total)

```
0  STATE_VAR    1  FUNCTION      2  MODIFIER      3  EVENT
4  FALLBACK     5  RECEIVE       6  CONSTRUCTOR   7  CONTRACT
8  CFG_NODE_CALL   9  CFG_NODE_WRITE  10  CFG_NODE_READ
11 CFG_NODE_CHECK  12 CFG_NODE_OTHER
```

CFG subtypes matter because a single `CFG_NODE=8` would give a CALL statement and a
WRITE statement identical initial embeddings before any message passing.

### Slither Version Guard

`graph_schema.py` asserts `slither-analyzer >= 0.9.3` at import time using
`importlib.metadata.version()` (not `slither.__version__`, which does not exist on the
top-level module). This is a hard failure — the pipeline must not run on older Slither
versions that lack `is_loop_present` or the correct `NodeType` API.

---

## 2. Graph Extractor — `ml/src/preprocessing/graph_extractor.py`

### Five New Node Features

**`return_ignored` (idx 7):**  
Walks Slither IR operations. Any `Return` operation that discards a `HighLevelCall`
or `LowLevelCall` result is counted. Returns 1.0 if any ignored return, 0.0 if none,
-1.0 sentinel if IR is unavailable.

**`call_target_typed` (idx 8):**  
Detects whether external calls use a typed interface (`ITarget(addr).method()`) vs a
raw address call (`addr.call(data)`). Uses `ElementaryType` with `.name == "address"`
check (not `AddressType` which does not exist in the installed Slither version).
Non-function nodes default to 1.0 (not applicable). Returns -1.0 when source mapping
is unavailable.

**`in_unchecked` (idx 9):**  
Two-strategy detection: Slither IR (`CheckedAdd`, `CheckedSub` etc. absent → in unchecked)
with regex fallback (`re.search(r'\bunchecked\s*\{', source_content)`) for cases where
IR is incomplete. Regex handles whitespace variants including `unchecked\n{`.

**`has_loop` (idx 10):**  
`getattr(func, "is_loop_present", None) is True` — explicit `is True` required because
`getattr` on MagicMock returns a truthy MagicMock, not False.

**`external_call_count` (idx 11):**  
`math.log1p(count) / math.log1p(20)` — log-normalized to [0,1] assuming 20 calls is
the practical maximum.

### CFG Subgraph Construction

Three new functions handle CFG node extraction:

**`_cfg_node_type(slither_node)`:**  
Maps Slither `NodeType` to our CFG_NODE_* integers using priority order:
`CALL > WRITE > READ > CHECK > OTHER` (documented in proposal §2.2C).
Catches ImportError for `slither.slithir.operations` gracefully.

**`_build_cfg_node_features(slither_node, func, cfg_type)`:**  
Builds the 12-dim feature vector for CFG nodes. Key invariants:
- `in_unchecked` = **always 0.0** for CFG nodes (never inherited from parent function)
- `loc` from `source_mapping.lines` or 0.0 if unavailable
- Synthetic Slither nodes (ENTRY_POINT, EXPRESSION) have no source_mapping and get loc=0.0

**`_build_control_flow_edges(func, x_list, node_index_map)`:**  
Iterates `func.nodes` (Slither CFG nodes), appends feature vectors to `x_list` in-place,
populates `node_index_map[slither_node] → graph_idx` where `graph_idx = len(x_list) - 1`
(explicit assignment, not a placeholder). Adds `CONTAINS` edges (function → CFG_NODE)
and `CONTROL_FLOW` edges (CFG_NODE → CFG_NODE via `.sons`).

Also stores `node_metadata` — a list of dicts `{name, type, source_lines}` — on the
`Data` object so the pre-flight test and validation can verify graph structure.

### Duck-Typing for Function Detection

`isinstance(obj, Function)` returns False for test mocks, causing constructor override
to never run in v4. v5 uses:
```python
_is_function = hasattr(obj, "nodes") and hasattr(obj, "pure")
```

---

## 3. GNN Encoder — `ml/src/models/gnn_encoder.py`

### Architecture: Three-Phase, Four-Layer GAT

The fundamental flaw with v4's two-phase design was signal propagation:
```
v4 (broken):
  Phase 1: CONTAINS (function → CFG_NODE) — runs on pre-CFG embeddings
  Phase 2: CONTROL_FLOW — enriches CFG_NODE embeddings
  ← Function node NEVER receives Phase-2-enriched CFG embeddings.
     Execution order information is trapped in the CFG subgraph.

v5 (correct):
  Phase 1: structural + forward CONTAINS — function context flows DOWN
  Phase 2: CONTROL_FLOW directed — order encoded in CFG nodes
  Phase 3: reverse-CONTAINS (CFG→function) — order signal flows UP
```

**Phase 1 (Layers 1+2):** `GATConv(12→128, heads=8, add_self_loops=True)`  
Edge mask includes types 0–5 (all structural edges + CONTAINS).
Both layers run; output is BN + ReLU + Dropout.

**Phase 2 (Layer 3):** `GATConv(128→128, heads=1, add_self_loops=False)`  
Edge mask: type 6 only (CONTROL_FLOW). `heads=1` required — multi-head would
concatenate to 128×8=1024 and break the residual connection.
Graphs without any CONTROL_FLOW edges pass unchanged (zero messages).

**Phase 3 (Layer 4):** `GATConv(128→128, heads=1, add_self_loops=False)`  
Edge mask: type 5 (CONTAINS), but with `edge_index.flip(0)` — edges reversed so
CFG_NODE messages flow UP to their parent FUNCTION nodes.
`add_self_loops=False` is critical: self-loops would corrupt Phase 3 by injecting
a node's own Phase 2 embedding back into itself via the reversed edge.

**Known Limitation — Phase 3 edge embedding symmetry:**  
Reversed CONTAINS edges (Phase 3) use the same type-5 embedding as forward CONTAINS
edges (Phase 1). The GNN cannot fully encode the directional asymmetry ("context flowing
down" vs "order signal flowing up"). GATConv's positional asymmetry provides partial
compensation. Deferred to v5.1: `REVERSE_CONTAINS = 7` with a dedicated embedding.

### Parameters

```
in_channels:    12   (NODE_FEATURE_DIM, LOCKED)
hidden_dim:     128  (default)
heads:          8    (Phase 1 only; Phases 2+3 use heads=1)
dropout:        0.2
use_edge_attr:  True
edge_emb_dim:   32   (Embedding(7, 32))
num_layers:     4    (validated in TrainConfig.__post_init__)
```

Approximate trainable parameters: ~90K.

### Interface

```python
gnn(x, edge_index, batch, edge_attr, return_intermediates=False)
# → (node_embs [N,128], batch [N])
# or, with return_intermediates=True:
# → (node_embs [N,128], batch [N], {
#       "after_phase1": [N,128],
#       "after_phase2": [N,128],
#       "after_phase3": [N,128]
#    })
```

`return_intermediates` is used by the pre-flight test (`test_cfg_embedding_separation.py`)
to verify that Phase 3 actually changes the function node embedding.

---

## 4. Sentinel Model — `ml/src/models/sentinel_model.py`

### Three-Eye Architecture

Each eye answers a different question and is constrained to produce a 128-dim embedding:

| Eye | Input | What it answers | Pooling |
|---|---|---|---|
| GNN eye | Phase 3 node embeddings [N,128] | "What structural patterns exist?" | max-pool + mean-pool → [B,256] → Linear(256,128) |
| Transformer eye | CLS token [B,768] | "What does source code say?" | CLS only → Linear(768,128) |
| Fused eye | CrossAttentionFusion output [B,128] | "How do structure and tokens co-locate?" | Fusion output (unchanged) |

Classifier: `cat([gnn_eye, transformer_eye, fused_eye])` → [B, 384] → `Linear(384, 10)` → logits.

**Eye dominance prevention:** Auxiliary classification heads (three separate `Linear(128, 10)`)
allow each eye to receive its own gradient signal. Aux loss weight λ=0.1 prevents one eye from
dominating.

### Auxiliary Heads

```python
model.aux_gnn         # Linear(128, 10) — GNN eye prediction head
model.aux_transformer # Linear(128, 10) — Transformer eye prediction head
model.aux_fused       # Linear(128, 10) — Fused eye prediction head
```

Active only during training. `return_aux=False` (default) returns only main logits.

### `gnn_num_layers` Wiring

`SentinelModel` now accepts `gnn_num_layers` and passes it through to `GNNEncoder`.
Previously hardcoded to 3 layers (never wired); `TrainConfig.gnn_layers` had no effect.

### Interface

```python
model(graphs, input_ids, attention_mask, return_aux=False)
# → logits [B, C]  (return_aux=False, default)
# → (logits [B, C], {"gnn": [B,C], "transformer": [B,C], "fused": [B,C]})  (return_aux=True)
```

No sigmoid inside the model. Inference: `torch.sigmoid(logits)`.

---

## 5. Trainer — `ml/src/training/trainer.py`

### `TrainConfig.__post_init__()` Guard

```python
if self.gnn_layers != 4:
    raise NotImplementedError(
        f"gnn_layers={self.gnn_layers} is not supported in v5.0. "
        "Only gnn_layers=4 (Phase 1+2+3+3) is implemented..."
    )
```

Fails fast at config construction, before any data loading or GPU allocation.

### `pos_weight` — sqrt Scaling

v4 used uncapped positive-class weights, which caused over-prediction in 6/10 classes.
v5 applies sqrt to compress extreme imbalances:

```python
# Approximate post-augmentation reference values:
# CallToUnknown: sqrt((N-n)/n) ≈ 6.5
# DenialOfService: sqrt((N-n)/n) ≈ 12.1  (after augmentation to ~440 samples)
# IntegerUO: sqrt((N-n)/n) ≈ 2.7
pos_weight[i] = sqrt((N - n_pos[i]) / n_pos[i])
```

No hard cap. Computed fresh from `multilabel_index.csv` after augmentation.

### Auxiliary Loss Training Loop

```python
with torch.amp.autocast("cuda", enabled=use_amp):
    logits, aux = model(graphs, input_ids, attention_mask, return_aux=True)
    main_loss = loss_fn(logits, labels)
    aux_loss = (
        loss_fn(aux["gnn"],         labels) +
        loss_fn(aux["transformer"], labels) +
        loss_fn(aux["fused"],       labels)
    )
    loss = main_loss + aux_loss_weight * aux_loss
```

Effective aux weight ≈ 23% of total loss (3 aux heads × 0.1 weight / (1 + 3×0.1)).

### Per-Eye Gradient Norm Logging

Every `log_interval` batches, logs gradient norms for each eye's projection layer:

```python
gnn_norm   = _grad_norm(model.gnn_eye_proj)
tf_norm    = _grad_norm(model.transformer_eye_proj)
fused_norm = _grad_norm(model.fusion)
```

`_grad_norm(module)` computes `sqrt(sum(p.grad.float().norm(2)**2 for p in module.parameters()))`.
Used to detect eye dominance during training without waiting for end-of-epoch evaluation.

### New `TrainConfig` Fields (v5)

```python
gnn_hidden_dim:    int   = 128   # GNN node embedding width
gnn_layers:        int   = 4     # validated; only 4 supported
gnn_heads:         int   = 8     # Phase 1 GAT attention heads
gnn_dropout:       float = 0.2
use_edge_attr:     bool  = True
gnn_edge_emb_dim:  int   = 32
aux_loss_weight:   float = 0.1
lora_r:            int   = 16    # was 8 in v4
lora_alpha:        int   = 32    # was 16 in v4
lora_dropout:      float = 0.1
```

### v5 Training Defaults (vs v4)

| Parameter | v4 | v5 |
|---|---|---|
| `epochs` | 30–40 | 60 |
| `batch_size` | 32 | 16 (larger graphs with CFG nodes) |
| `lr` | 1e-4 | 2e-4 |
| `early_stop_patience` | 7 | 10 |
| `lora_r` | 8 | 16 |
| `lora_alpha` | 16 | 32 |
| `warmup_pct` | varies | 0.10 |
| `run_name` default | `multilabel-v4-*` | `multilabel-v5-fresh` |

---

## 6. Loss Functions — `ml/src/training/focalloss.py`

### `FocalLoss` (unchanged)

Scalar alpha, post-sigmoid probabilities. Audit fix #6 (BF16 underflow guard) applied
in a prior session — `.float()` cast at top of `forward()`.

### `MultiLabelFocalLoss` (NEW)

Per-class alpha list for multi-label binary cross-entropy on **raw logits**:

```python
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha: List[float], gamma: float = 2.0) -> None:
        # alpha registered as buffer — moves to device with .to(device)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # BF16 guard: logits.float(), targets.float()
        # sigmoid applied internally
        # binary_cross_entropy_with_logits for numerical stability (no log(0))
```

Key differences from `FocalLoss`:
- Accepts **raw logits** (sigmoid applied internally) — no external wrapper needed
- Alpha is `List[float]` not scalar — enables per-class imbalance correction
- Multi-label `[B,C]` only

---

## 7. Predictor — `ml/src/inference/predictor.py`

Updated to load v5 checkpoints (`ARCHITECTURE = "three_eye_v5"`):

### Architecture Registry

```python
_ARCH_TO_FUSION_DIM: dict[str, int] = {
    "three_eye_v5":         128,
    "cross_attention_lora": 128,  # v4
    "legacy":               64,
    "legacy_binary":        64,
}

_ARCH_TO_NODE_DIM: dict[str, int] = {
    "three_eye_v5":         12,   # v5 schema
    "cross_attention_lora": 8,    # v4 schema
    "legacy":               8,
    "legacy_binary":        8,
}
```

### Architecture-Aware Model Construction

```python
_is_v5 = (architecture == "three_eye_v5")
self.model = SentinelModel(
    ...
    gnn_hidden_dim   = saved_cfg.get("gnn_hidden_dim",  128 if _is_v5 else 64),
    gnn_num_layers   = saved_cfg.get("gnn_layers", 4),
    gnn_edge_emb_dim = saved_cfg.get("gnn_edge_emb_dim", 32 if _is_v5 else 16),
    lora_r           = saved_cfg.get("lora_r",  16 if _is_v5 else 8),
    lora_alpha       = saved_cfg.get("lora_alpha", 32 if _is_v5 else 16),
)
```

### Warmup Update

Warmup dummy graph uses `_ARCH_TO_NODE_DIM.get(self.architecture, 12)` instead of
the hardcoded `8`. The dummy graph's `edge_attr` is also provided (required by GNNEncoder
for phase mask computation).

---

## 8. Test Suite — `ml/tests/`

### `test_preprocessing.py` — Full Rewrite for v5

All MagicMock-based property exception injection was replaced with real dataclasses
and `patch.dict("sys.modules", ...)` for import failures. Key test classes:

- **`TestComputeReturnIgnored`** — verifies 0.0/1.0/-1.0 semantics
- **`TestComputeCallTargetTyped`** — uses `ElementaryType` (not `AddressType`), sentinel on missing source
- **`TestComputeInUnchecked`** — regex fallback with whitespace variants
- **`TestComputeHasLoop`** — `is True` guard against truthy MagicMock
- **`TestBuildNodeFeatures`** — duck-typing for Function detection, CFG subtype assignment

### `test_model.py` — Rewrite for v5 Three-Eye Architecture

```python
NODE_FEATURE_DIM = 12    # v2 schema fixture
edge_attr = torch.zeros(n-1, dtype=torch.long)  # required; CALLS type 0

# Stub transformer avoids loading 500MB CodeBERT in tests
class _StubTransformer(nn.Module):
    def forward(self, input_ids, attention_mask):
        return torch.zeros(B, 512, 768)
```

Key new tests:

| Test | Assertion |
|---|---|
| `test_forward_return_aux_shapes` | `(logits [B,10], {"gnn","transformer","fused"} each [B,10])` |
| `test_classifier_input_dim_is_384` | `model.classifier.in_features == 384` |
| `test_aux_heads_exist_and_output_dim` | all three aux heads: `in=128, out=10` |
| `test_gnn_return_intermediates_keys` | `(x, batch, {"after_phase1","after_phase2","after_phase3"})` |
| `test_gnn_phase3_changes_function_node` | Phase 3 must change node 0 embedding when CONTAINS+CONTROL_FLOW present |
| `test_forward_returns_logits_not_probabilities` | `(out < 0).any()` — negative values prove no sigmoid applied |
| `test_mostly_pad_attention_mask_does_not_crash` | 1 real + 511 PAD tokens, no NaN |
| `test_forward_with_cfg_edges` | CONTAINS + CONTROL_FLOW graph, shape `(1,10)`, no NaN |

**All-PAD mask caveat:** CrossAttentionFusion softmax over all-masked keys produces NaN.
This is documented as a known limitation — not a bug. All-PAD cannot occur for valid
contract inputs (every contract has ≥1 real token). Test uses 1 real + 511 PAD.

### `test_cfg_embedding_separation.py` (NEW — Pre-flight test)

Validates that the v5 GNN produces **different** embeddings for reentrancy-vulnerable
vs CEI-safe contracts before any training:

```
Contract A (vulnerable): require → external_call → state_write
Contract B (safe):       require → state_write → external_call
```

With `torch.manual_seed(42)` for reproducibility (tightened from threshold=0.95 to
**0.85** after Rev 1.5 critique on single-seed reliability).

`_find_function_node()` helper uses pre-extracted `node_metadata` list (not f-string
list comprehension, which had a syntax error in the proposal's earlier draft).

**This test must pass before Phase 4 (full re-extraction).** If it fails, the GNN
architecture is incorrect and training will learn nothing useful.

### Test Results (62 passed, 13 deselected/integration, 4 warnings)

All unit tests pass. Integration tests (marked with `@pytest.mark.integration`) are
deselected by default because they require the full graph dataset.

---

## 9. Training CLI — `ml/scripts/train.py`

All v5 hyperparameters exposed as CLI flags. New flags added for v5:

```bash
--gnn-hidden-dim   128    # GNN embedding width
--gnn-layers       4      # must be 4; raises NotImplementedError otherwise
--gnn-heads        8      # Phase 1 attention heads
--gnn-dropout      0.2
--gnn-edge-emb-dim 32     # edge type embedding dim
--no-edge-attr            # disable edge-type embeddings (degrades perf)
--aux-loss-weight  0.1    # λ for three-eye auxiliary loss
--lora-r           16     # LoRA rank (was 8 in v4)
--lora-alpha       32     # LoRA scale (keep alpha/r = 2.0)
--lora-dropout     0.1
```

Run name default changed from `multilabel-v4-*` to `multilabel-v5-fresh`.

---

## 10. Data Augmentation Scripts — `ml/scripts/`

### `generate_safe_variants.py` (NEW)

Mutation-based safe contract generator implementing the §3.5 two-step verification protocol:

1. **`reentrancy-cei`** — `_swap_call_and_write()`: finds state writes after external calls
   and moves them before the call. Skips `require`/`emit` lines (checks call success);
   stops at `}`, `return`, `revert`. Handles multi-statement patterns via reverse-order
   call processing.

2. **`mishandled-exception`** — `_wrap_bare_call()`: wraps bare `addr.call(...)` with
   `(bool _ok,) = ...; require(_ok, "call failed");`.

3. **`call-to-unknown`** — `_annotate_typed_interface()`: annotates raw address call
   sites with a typed-interface comment for manual review.

4. **`dos-bounded`** — `_add_dos_loop_guard()`: inserts `require(arr.length <= 100)`
   before unbounded `for`/`while` loops.

**Verification gate (§3.5):**

```python
def generate_cei_safe(path, out_dir, solc_override, strategy, dry_run):
    # Step 1: Compilation check (MUST precede Slither)
    compile_result = _compile_solidity(safe_path, solc_override)
    if compile_result.returncode != 0:
        safe_path.unlink()
        return None   # syntactically invalid swap

    # Step 2: Slither reentrancy check
    findings = _run_slither(safe_path, solc_override, detectors=check_detectors)
    if bad_findings:
        safe_path.unlink()
        return None   # semantically still vulnerable
```

Slither is invoked via CLI subprocess (`--json`) rather than Python API, to avoid
version-dependent API surface.

**Target yields (from BCCC-SCsVul-2024):**

| Source | Strategy | Target |
|---|---|---|
| `SourceCodes/Reentrancy/` (17,698 files) | `reentrancy-cei` | 500+ safe variants |
| `SourceCodes/MishandledException/` (5,154 files) | `mishandled-exception` | 100+ safe variants |
| `SourceCodes/DenialOfService/` (12,394 files) | `dos-bounded` | 300+ safe variants |

### `extract_augmented.py` (NEW)

Bridges augmented `.sol` files into the training pipeline:

1. Extracts PyG graph (v2 schema, 12-dim nodes, 7 edge types) via `extract_contract_graph()`
2. Tokenizes with CodeBERT (max_length=512, MD5-keyed `.pt` files)
3. Appends rows to `multilabel_index.csv` (idempotent — skips known MD5 stems)

**Label semantics:**
- Safe variants (CEI-swapped, wrap-called, loop-guarded): all-zeros (NonVulnerable)
- `--label ClassName` sets one class to 1
- `--label-json '{"Reentrancy": 1}'` for multi-label

**Requires `TRANSFORMERS_OFFLINE=1`** before running (CodeBERT loads from `.cache/huggingface`).

### `run_augmentation.sh` (NEW)

Full pipeline orchestrator for Phase 3:

```bash
# Dry-run smoke test (5 contracts, no writes):
DRY_RUN=1 MAX_CONTRACTS=5 bash ml/scripts/run_augmentation.sh

# Full run:
export TRANSFORMERS_OFFLINE=1
bash ml/scripts/run_augmentation.sh
```

Runs all three generation strategies, three extraction passes, split update
(`create_splits.py --freeze-val-test`), and validation.

### `validate_graph_dataset.py` — Updated

Added `--check-cfg-subtypes` flag: verifies `x[:,0]` contains at least one type_id in 8–12
(CFG subtype nodes). Without this, a graph may pass all other checks yet have no CFG content
because `_build_control_flow_edges()` was silently skipped (e.g., Slither returned no nodes
for the function body). Also fixed wrong default hint in `--check-dim` docstring (13→12).

**Full v5 validation command (run after Phase 4 re-extraction):**
```bash
poetry run python ml/scripts/validate_graph_dataset.py \
    --check-dim 12 \
    --check-edge-types 7 \
    --check-contains-edges \
    --check-control-flow \
    --check-cfg-subtypes
```

---

## 11. Deviations from Proposal

| Proposal Item | Actual Implementation | Reason |
|---|---|---|
| `Slither` Python API for `_run_slither()` | CLI subprocess (`slither --json`) | Version-stable; avoids API surface changes across Slither minor versions |
| `graph_idx = len(node_index_map) + ...` (ellipsis placeholder in Rev 1.5) | `graph_idx = len(x_list)` (append-then-index) | Bug fix applied in Rev 1.6; pre-existing placeholder was a critical blocker |
| `AddressType` from slither.core.solidity_types | `ElementaryType` with `.name == "address"` check | `AddressType` does not exist in installed Slither version |
| `slither.__version__` for version check | `importlib.metadata.version("slither-analyzer")` | `slither.__version__` does not exist on the top-level module |
| `isinstance(obj, Function)` for duck-typing | `hasattr(obj, "nodes") and hasattr(obj, "pure")` | `isinstance` returns False for MagicMock objects in tests |
| `type(mock).attr = property(...)` in tests | Real dataclasses with `@property` | MagicMock property assignment is unreliable for exception injection |
| `test_all_pad_attention_mask_does_not_crash` | `test_mostly_pad_attention_mask_does_not_crash` (1 real + 511 PAD) | All-PAD produces legitimate NaN in cross-attention softmax; not a bug |
| SmartBugs + SolidiFI augmentation (Priority 2+) | BCCC only | Only BCCC data is used in this project |

---

## 12. Pending — Phases 4–6

### Phase 4 — Full Re-Extraction (~1 day + compute)

```bash
# Re-extract all 68K graphs with v5 schema (12-dim, 7 edge types)
poetry run python ml/src/data_extraction/ast_extractor.py --force

# Validate
poetry run python ml/scripts/validate_graph_dataset.py \
    --check-dim 12 \
    --check-edge-types 7 \
    --check-contains-edges \
    --check-control-flow \
    --check-cfg-subtypes

# Update splits (freeze val/test — mandatory)
poetry run python ml/scripts/create_splits.py \
    --multilabel-index ml/data/processed/multilabel_index.csv \
    --splits-dir ml/data/splits \
    --freeze-val-test
```

**`pos_weight` must be recomputed after augmentation.** The distribution changes when
500+ safe contracts and 300+ DoS contracts are added. Pre-augmentation values from v4
will underweight the newly balanced classes.

### Phase 5 — Training Sequence

```bash
# Phase A: Smoke run (shape + VRAM check)
poetry run python ml/scripts/train.py \
    --run-name v5-smoke \
    --smoke-subsample-fraction 0.10 --epochs 2 --batch-size 16

# Phase B: 15-epoch check (over-prediction gate)
poetry run python ml/scripts/train.py \
    --run-name v5-check-15ep \
    --epochs 15 --batch-size 16 --lr 2e-4 \
    --weighted-sampler all-rare --lora-r 16 --lora-alpha 32

# Phase C: Full training
poetry run python ml/scripts/train.py \
    --run-name v5-full \
    --epochs 60 --batch-size 16 --lr 2e-4 \
    --weighted-sampler all-rare --lora-r 16 --lora-alpha 32 \
    --early-stop-patience 10
```

After Phase A, check `p95(nodes_per_graph)` in MLflow logs. If `p95 × 16 > VRAM`,
reduce to `--batch-size 8`.

### Phase 6 — Evaluation

1. `tune_threshold.py` on validation set.
2. Run all 20 behavioral-test contracts from `ml/scripts/test_contracts/` + additional suite.
3. Compare per-class F1 against v4 floors (each class must exceed v4 tuned F1 − 0.05).
4. Gate: F1-macro > 0.58, DoS tuned F1 > 0.55, behavioral 70% detection, 66% specificity.
5. If gate cleared: `promote_model.py` to update active checkpoint.

---

## 13. Acceptance Criteria (from §10)

| Gate | Target | v4 Baseline |
|---|---|---|
| F1-macro (tuned, validation) | **> 0.58** | 0.5422 |
| DenialOfService tuned F1 | **> 0.55** | ~0.43 |
| Behavioral detection rate | **> 70%** | 15% |
| Behavioral specificity | **> 66%** | 33% |
| No class below v4 floor | all 10 classes | v4 per-class F1 − 0.05 |
| Contract A/B pair | **mandatory** | both misclassified |

---

## 14. Files Changed / Created

### Modified
| File | Change |
|---|---|
| `ml/src/preprocessing/graph_schema.py` | v2 schema: NODE_FEATURE_DIM=12, NUM_EDGE_TYPES=7, CFG subtypes, Slither version guard |
| `ml/src/preprocessing/graph_extractor.py` | 5 new features, CFG subgraph, duck-typing, ElementaryType, is True guard |
| `ml/src/models/gnn_encoder.py` | Three-phase 4-layer GAT; edge type masking; return_intermediates |
| `ml/src/models/sentinel_model.py` | Three-eye classifier; aux heads; gnn_num_layers wired; return_aux |
| `ml/src/training/trainer.py` | gnn_layers guard; sqrt pos_weight; aux loss loop; per-eye grad norm; v5 defaults |
| `ml/src/training/focalloss.py` | Added MultiLabelFocalLoss |
| `ml/src/inference/predictor.py` | v5 checkpoint loading; _ARCH_TO_NODE_DIM; architecture-aware defaults |
| `ml/scripts/train.py` | All v5 CLI flags; v5 defaults; TrainConfig wired |
| `ml/scripts/validate_graph_dataset.py` | --check-cfg-subtypes; dim hint fixed (13→12) |
| `ml/tests/test_preprocessing.py` | Full rewrite for v5 (MagicMock fixes, duck-typing, ElementaryType) |
| `ml/tests/test_model.py` | Full rewrite for v5 (12-dim fixtures, edge_attr, return_aux, return_intermediates) |

### Created
| File | Purpose |
|---|---|
| `ml/tests/test_cfg_embedding_separation.py` | Pre-flight: Contract A (vuln) vs B (safe) must embed differently |
| `ml/scripts/generate_safe_variants.py` | Mutation-based safe contract generator + two-step verification |
| `ml/scripts/extract_augmented.py` | Graph+token extraction + label indexing for augmented contracts |
| `ml/scripts/run_augmentation.sh` | Full Phase 3 pipeline orchestrator |
