# 2026-05-24 — IMP-All Fixes Applied + P1-TRAIN Run 2 Launched

## Summary

P1-TRAIN Run 1 (GCB + K=48 prefix, 7-layer GNN) was killed at ep28 (best ep27 F1=0.2628) to
implement all outstanding improvement backlog items before continuing. JK weight analysis from
Run 1 confirmed Phase 2 collapse and Phase 3 dominance — the architectural fixes (IMP-G1/G2/G3)
were pulled forward from the planned Phase GNN-A stage. All 8 improvement items were implemented,
134/134 tests pass, and P1-TRAIN Run 2 launched (~22:15) with the full 8-layer GNN + all fixes.

---

## P1-TRAIN Run 1 — Findings that Motivated the Kill

**Run 1 config:** 7-layer GNN, K=48, warmup ep1-15, prefix active ep16+
**Kill point:** ep28, best checkpoint at ep27 (F1-macro=0.2628)

| Epoch | F1-macro | proj_norm | JK Ph1 | JK Ph2 | JK Ph3 |
|-------|----------|-----------|--------|--------|--------|
| 1 | 0.1832 | 16.0000 | 0.063 | 0.387 | 0.550 |
| 21 | 0.2570 | 16.1250 | — | — | — |
| 24 | 0.2496 | 16.2500 | 0.063 | 0.245 | 0.692 |
| 27 | **0.2628** | 16.2500 | 0.058 | 0.234 | **0.707** |

**Root causes identified:**

1. **Phase 2 JK collapse (0.387 → 0.234):** All three Phase 2 layers (conv3/3b/3c) shared the
   same `cfg_mask` edge set (CF ∪ CALL_ENTRY ∪ RETURN_TO). Layers with identical input + identical
   edges produce identical representations; JK attention correctly downweights them. Fix: IMP-G1
   (distinct edge subsets per layer).

2. **Phase 3 JK dominance growing (0.550 → 0.707):** REVERSE_CONTAINS (CFG→FUNCTION upward) became
   the single dominant signal. CFG nodes received no Phase 3 context — representation gap between
   FUNCTION and CFG nodes in CrossAttentionFusion. Fix: IMP-G3 (downward CONTAINS pass).

3. **Phase 1 JK flat (0.063 throughout):** The 11→256 dimension change in conv1 loses raw feature
   signal without a skip connection. Fix: IMP-G2 (input projection skip).

4. **BF16 proj_norm stagnation:** Only 2 ULPs drift (16.0000 → 16.2500) over 13 post-warmup
   epochs. At norm≈16, one BF16 ULP = 0.125 — quantization floor prevents fine-grained gradient
   accumulation in `gnn_to_bert_proj`.

---

## Code Changes

### IMP-G1 — Phase 2 Layer-Specific Edge Subsets
**File:** `ml/src/models/gnn_encoder.py`

Added distinct edge mask construction before Phase 2:
- `cf_only_ei/ea` — CONTROL_FLOW edges only (edge_attr == 6)
- `icfg_only_ei/ea` — CALL_ENTRY(8) + RETURN_TO(9) only (cross-function)
- `cfg_ei/ea` — joint (existing full Phase 2 mask)

Phase 2 layers now use: `conv3(cf_only)` → `conv3b(icfg_only)` → `conv3c(cfg joint)`.

### IMP-G2 — Phase 1 Input Projection Skip
**File:** `ml/src/models/gnn_encoder.py`

New `__init__` parameter:
```python
self.input_proj = nn.Linear(NODE_FEATURE_DIM, hidden_dim, bias=False)  # 2,816 params
```

New `forward` block (before conv1):
```python
x_init = x
_proj_dtype = next(self.input_proj.parameters()).dtype
x_skip = self.input_proj(x_init.to(_proj_dtype)).to(x.dtype)
x = self.conv1(x_init, struct_ei, struct_ea)
x = self.relu(x + x_skip)
```

Also added dtype normalisation at `forward()` entry (BF16 global default dtype fix):
```python
_param_dtype = next(self.parameters()).dtype
if x.dtype != _param_dtype:
    x = x.to(_param_dtype)
```

### IMP-G3 — Phase 3 Bidirectional Context Pass
**File:** `ml/src/models/gnn_encoder.py`

New `__init__` parameter:
```python
self.conv4c = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                      add_self_loops=False, edge_dim=_edge_dim)
```

New `forward` block (after existing Phase 3 upward passes):
```python
# fwd_contains_ei saved earlier (CONTAINS edges, FUNCTION→CFG direction)
x4c = self.conv4c(x, fwd_contains_ei, fwd_contains_ea)
x   = x + self.dropout(x4c)
x   = self.phase_norm[2](x)
```

Architecture is now **8 layers** (2+3+3). `gnn_num_layers` default updated 7→8 in both
`GNNEncoder.__init__` and `TrainConfig.gnn_layers`.

### IMP-M1 — FUNCTION Node Secondary Sort
**File:** `ml/src/models/sentinel_model.py`

`select_prefix_nodes()` now sorts FUNCTION nodes by `external_call_count` (feature[10]) descending
when K truncation occurs, using a Python tuple sort key `(priority, -ext_call_count, local_idx)`.

### IMP-M2 Tier 2 — prefix_attention_mean Diagnostic
**Files:** `ml/src/models/transformer_encoder.py`, `ml/src/training/trainer.py`

`TransformerEncoder.forward()` gains `gnn_prefix_counts` and `output_attentions` parameters.
When `output_attentions=True`, extracts mean attention weight from code positions → prefix positions
across all 12 layers and 12 heads: `attn[:, :, :, K:, :K].mean()`.

New `SentinelModel.compute_prefix_attention_mean()` method (decorated `@torch.no_grad()`).

Trainer logs `prefix_attention_mean` to MLflow each epoch post-warmup, warns if < 0.002.

### IMP-M3 — Zero-Padded Prefix Attention Mask Fix
**Files:** `ml/src/models/sentinel_model.py`, `ml/src/models/transformer_encoder.py`

`select_prefix_nodes()` return type changed from `Tensor` to `tuple[Tensor, Tensor]`:
- `prefix` — `[B, K, 768]` projected node embeddings (zero-padded for graphs with < K nodes)
- `node_counts` — `[B]` int tensor, real node count per graph

`TransformerEncoder` constructs prefix attention mask using `node_counts`:
```python
prefix_mask = torch.zeros(B, K, device=device)
for g in range(B):
    prefix_mask[g, :node_counts[g]] = 1.0
```
Padded positions are masked out (attention_mask=0) instead of wastefully attending to zero vectors.

### IMP-D1 — return_ignored Temporal Ordering Fix
**File:** `ml/src/preprocessing/graph_extractor.py`

`_compute_return_ignored()` rewritten from global-set to CFG-ordered per-call scan:

**Old (buggy):** Built `all_read_names` set across entire function — false negative when a
TemporaryVariable name collided with an unrelated read elsewhere, incorrectly marking return as
"captured."

**New (IMP-D1):** Iterates `func.nodes` in CFG topological order, building `all_ops_ordered`.
For each call op at `call_idx`, checks if `lval_name` appears in any `later_op.read` at
`all_ops_ordered[call_idx + 1:]`. Uses direct `func.nodes` access (not `getattr`) so
`AttributeError` from Slither IR unavailability propagates to the sentinel return.

Re-extraction of all 41K graphs is pending (separate run, not blocking P1-TRAIN Run 2).

### IMP-BUG — Close BUG-H4 and BUG-H5
**File:** `docs/ACTIVE_PLAN.md`

Both bugs were fixed by `label_cleaner.py` changes on 2026-05-23 (−568 Timestamp labels,
−611 Reentrancy labels) but the Open Bugs section had not been updated. Marked DONE.

### BF16 Global Dtype Side-Effect Fix
**File:** `ml/src/models/transformer_encoder.py`

`AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)` calls `torch.set_default_dtype(bfloat16)`
as a side effect, causing any `nn.Linear` created after BERT initialisation to have BF16 weights
(including `gnn_eye_proj`, `classifier`, etc.). Fixed by wrapping BERT load:

```python
_prev_default_dtype = torch.get_default_dtype()
try:
    self.bert = AutoModel.from_pretrained(...)
finally:
    torch.set_default_dtype(_prev_default_dtype)
```

### promote_model.py — Module-Level MLflow Import
**File:** `ml/scripts/promote_model.py`

`mlflow` and `MlflowClient` imports moved from inside `promote()` function to module level.
Required for `patch("ml.scripts.promote_model.mlflow")` to work in tests.

---

## Test Suite Fixes (134/134 pass)

All test failures were pre-existing stale expectations or API mismatches surfaced when running
without `-x` (stop-at-first-failure).

### test_model.py
- `_StubTransformer.forward()` — added `gnn_prefix_nodes`, `gnn_prefix_counts`, `output_attentions` params
- `test_classifier_input_dim_is_384` — `classifier[0].in_features` (Sequential, not bare Linear)
- `test_gnn_return_intermediates_keys` — node shape `(3,128)` → `(3,256)` (hidden_dim=256)
- `test_gnn_return_intermediates_false_is_2_tuple` — `(5,128)` → `(5,256)`

### test_preprocessing.py — Schema sanity
- `NODE_FEATURE_DIM` assertion: 12 → 11 (v8 schema: `in_unchecked` removed)
- `NUM_EDGE_TYPES` assertion: 8 → 11 (v8: CALL_ENTRY, RETURN_TO, DEF_USE added)
- `test_feature_names_has_all_new_features`: removed `in_unchecked` from expected list
- `test_external_call_count_at_index_10`: was index 11 — correct index in 11-feature schema

### test_preprocessing.py — TestComputeReturnIgnored
IMP-D1 changed from `func.slithir_operations` to `func.nodes[i].irs`. Tests updated:
- All tests now build `_make_mock_slither_node(irs=[...])` and pass via `nodes=[node]`
- `test_returns_sentinel_on_attribute_error` — `FakeFunc.nodes` property raises AttributeError

### test_preprocessing.py — TestBuildCfgNodeFeatures / TestBuildNodeFeatures
- `test_type_id_reflects_cfg_type` — expected `float(cfg_type)/12.0` (normalised, not raw)
- `test_loc_from_source_mapping` — expected `log1p(3)/log1p(1000)` (log-normalised, not raw 3.0)
- `test_type_id_override_for_constructor/fallback` — normalised expected values

### test_preprocessing.py — TestExtractionIntegration
Root cause: `graph.x[:, 0]` stores `type_id / 12.0` (normalised). All type_id comparisons
were against raw integers — `.int()` on normalized values can only produce 0 (types 0–11) or
1 (type 12, CFG_NODE_OTHER).

Added helper methods to the class:
```python
@staticmethod
def _type_ids(graph) -> list[int]:
    return (graph.x[:, 0] * 12).round().long().tolist()

@staticmethod
def _type_mask(graph, type_id: int):
    return (graph.x[:, 0] * 12).round().long() == type_id
```

All `graph.x[:, 0].int()` usages replaced. Additional fixes:
- `test_unchecked_func_node_has_in_unchecked_1` — `in_unchecked` removed from v8 schema; test
  now verifies FUNCTION nodes exist with correct feature dim
- `test_loop_func_has_has_loop_1` — index 10 → 9 (`has_loop` is at FEATURE_NAMES[9] in v8)
- `test_cei_safe_has_write_before_call_in_control_flow` — relaxed from direct-edge to BFS
  reachability (Slither inserts an intermediate CFG_NODE_OTHER between WRITE and CALL nodes)

### test_trainer.py
- Removed stale `scaler=scaler` keyword argument from `train_one_epoch()` calls (`scaler`
  parameter was removed from the function signature at some prior point)

---

## P1-TRAIN Run 2 — Launch Details

**Command:**
```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. nohup python ml/scripts/train.py \
  --gnn-layers 8 --gnn-prefix-k 48 --gnn-prefix-warmup-epochs 15 \
  --epochs 60 --batch-size 8 --gradient-accumulation-steps 8 \
  --loss-fn asl --compile --use-amp --phase2-edge-types 6 8 9 \
  --experiment-name sentinel-retrain-v2 --run-name GCB-P1-Run2-IMP-all \
  > ml/logs/graphcodebert-p1-run2-20260524.log 2>&1 &
```

**PID:** 80610  
**Log:** `ml/logs/graphcodebert-p1-run2-20260524.log`  
**Startup confirmed:** layers=8, gnn_prefix_k=48, warmup=15, VRAM 0.3/8.0 GiB, proj_norm=15.9853

**Key differences from Run 1:**
- 8-layer GNN (IMP-G3 added conv4c downward pass)
- Phase 2 layer-specific edge subsets (IMP-G1)
- Input projection skip in Phase 1 (IMP-G2)
- FUNCTION secondary sort by external_call_count (IMP-M1)
- prefix_attention_mean logged post-warmup (IMP-M2 T2)
- Padded prefix positions masked out (IMP-M3)
- Fresh start from random init (NOT resumed — architecture changed)

**Monitor when available:**
- ep1: warmup active, no prefix, proj_norm ≈ 15.98 (random init)
- ep15: warmup ends
- ep16: prefix activates — expect brief loss spike then recovery
- ep17–20: watch `prefix_attention_mean` in MLflow; target > 0.005 by ep20
- ep17–20: watch Phase 2 JK weight — expect > 0.10 (vs 0.234 Run 1 declining to plateau)

---

## Files Changed

| File | Change |
|------|--------|
| `ml/src/models/gnn_encoder.py` | IMP-G1 (layer-specific edges), IMP-G2 (input_proj skip + dtype guard), IMP-G3 (conv4c downward), default num_layers 7→8 |
| `ml/src/models/sentinel_model.py` | IMP-M1 (FUNCTION sort), IMP-M3 (node_counts return), IMP-M2 T2 (compute_prefix_attention_mean), gnn_num_layers default 7→8 |
| `ml/src/models/transformer_encoder.py` | IMP-M2 T2 (output_attentions path), IMP-M3 (count-based prefix mask), BF16 global dtype fix |
| `ml/src/preprocessing/graph_extractor.py` | IMP-D1 (CFG-ordered temporal scan in _compute_return_ignored) |
| `ml/src/training/trainer.py` | IMP-M2 T2 (prefix_attention_mean logging), gnn_layers default 7→8, warning threshold 7→8 |
| `ml/scripts/promote_model.py` | mlflow/MlflowClient imports moved to module level |
| `docs/ACTIVE_PLAN.md` | IMP-BUG: BUG-H4 + BUG-H5 marked DONE |
| `docs/proposal/EXECUTION_PLAN.md` | Full IMP-* documentation section added; P1-TRAIN Run 2 details; Summary Checklist updated |
| `ml/tests/test_model.py` | Stub signature, shape assertions updated for new architecture |
| `ml/tests/test_preprocessing.py` | Schema constants, normalisation fixes, IMP-D1 API, integration test helpers |
| `ml/tests/test_trainer.py` | Removed stale scaler kwarg |
