

# 🔴 GROUP 3 — ADVERSARIAL AUDIT: GNN Path End-to-End

**Files:** `gnn_encoder.py` (316 lines), `fusion_layer.py` (262 lines), `sentinel_model.py` (312 lines)  
**Path traced:** `graph.x [N,12]` → GNNEncoder → `node_embs [N,128]` → function-level pool → `gnn_eye [B,128]` → classifier  
**Cross-path:** `node_embs [N,128]` + `token_embs [B,512,768]` → CrossAttentionFusion → `fused_eye [B,128]` → classifier

---

## FINDING 3.1 — Phase 3 Reverse-CONTAINS Edge Embedding Symmetry: The Signal Leak

**Severity: HIGH (the documented v5.0 known limitation, now audited in detail)**

Lines 274-278 of `gnn_encoder.py`:
```python
rev_contains_ei = edge_index[:, contains_mask].flip(0)   # [2, E_contains]
rev_contains_ea = e[contains_mask] if e is not None else None
```

The reversed CONTAINS edges (Phase 3: CFG_NODE → FUNCTION) use the **same edge embedding** as forward CONTAINS edges (Phase 1: FUNCTION → CFG_NODE). This means the GAT attention mechanism receives the same edge feature vector for both directions. The only asymmetry comes from the GATConv positional distinction (source vs target node), which provides partial but incomplete directional information.

**The hostile read in detail:** Consider a function `withdraw()` with two CFG children: `CFG_NODE_CALL` (external call) and `CFG_NODE_WRITE` (state write). In Phase 1, the FUNCTION node sends messages TO both CFG children via forward CONTAINS (edge type 5). In Phase 3, the CFG children send messages BACK to the FUNCTION node via reversed CONTAINS (also edge type 5, same embedding).

The attention weight that `withdraw()` allocates to receiving from `CFG_NODE_CALL` vs `CFG_NODE_WRITE` is computed as:
```
α(CALL→withdraw) = softmax(Q_withdraw · K_CALL + e_5)
α(WRITE→withdraw) = softmax(Q_withdraw · K_WRITE + e_5)
```

The edge bias `e_5` is identical for both, so the model must distinguish CALL from WRITE entirely from the node embeddings (K_CALL vs K_WRITE). This works — BUT it means the model cannot learn "information flowing UP from children is fundamentally different from information flowing DOWN from parents." In a correct design, the UP direction should carry an aggregation signal ("here's what I found in the CFG"), while the DOWN direction carries a broadcasting signal ("here's my function-level context"). The same `e_5` embedding cannot encode this semantic difference.

**Impact on gradient collapse:** The documentation states that GNN eye gradient share collapsed to ~7% by epoch 43. The Phase 3 symmetry means the GNN cannot learn a strong "CFG summary flows up" signal — the UP and DOWN messages look the same to the attention mechanism. The classifier then learns to weight the GNN eye lower (since its signal is noisier), causing the gradient to collapse.

**Recommendation:** Implement `REVERSE_CONTAINS = 7` as planned in v5.1. This requires:
1. Adding `"REVERSE_CONTAINS": 7` to `EDGE_TYPES` in `graph_schema.py`
2. Bumping `NUM_EDGE_TYPES` to 8
3. Constructing Phase 3 edge embeddings from `e[contains_mask]` but with `edge_attr = 7` instead of reusing type-5 embeddings
4. This is a **schema change** requiring `FEATURE_SCHEMA_VERSION` bump and full retrain

---

## FINDING 3.2 — Ghost Graph Fallback Silently Degrades Predictions

**Severity: HIGH**

`sentinel_model.py` lines 227-234:
```python
if pool_mask.any():
    pool_embs  = node_embs[pool_mask]
    pool_batch = batch[pool_mask]
else:
    # Ghost graph (interface-only extraction) — fall back to all nodes
    pool_embs  = node_embs
    pool_batch = batch
```

When no function-level nodes exist (ghost graph from interface-only extraction), the code falls back to pooling over ALL nodes — including CONTRACT, STATE_VAR, and any stray CFG nodes. This was intended to prevent crashes, but:

1. **The fallback produces a valid tensor** — no error, no warning, no logging. The model proceeds to make a prediction on a meaningless graph.
2. **The prediction is junk** — the GNN eye is computed from structural nodes that carry no vulnerability signal, but the classifier treats it as equally valid as a real GNN eye.
3. **At inference time**, the API returns a confident prediction on a contract that shouldn't be analyzed. The user has no way to know the prediction is based on a ghost graph.
4. **During training**, ghost graphs in the dataset produce GNN eye embeddings from non-function nodes. The model learns that these patterns are associated with whatever label the ghost graph has (likely 0/safe), reinforcing the "predict safe" bias.

**The hostile read:** This is the exact scenario described in Finding 2.12 (interface fallback in `_select_contract`). The graph extraction layer creates a ghost graph, and the model layer accepts it without complaint. Two layers of fallback compound into a silent accuracy degradation.

**Recommendation:** Instead of falling back, return a **zero GNN eye** with a `graph_quality` flag:
```python
if pool_mask.any():
    pool_embs = node_embs[pool_mask]
    pool_batch = batch[pool_mask]
else:
    gnn_eye = torch.zeros(B, self.gnn_eye_proj[0].out_features, device=node_embs.device)
    # Signal to classifier that GNN eye is uninformative
```
At minimum, log a WARNING when this happens during inference.

---

## FINDING 3.3 — `global_max_pool` on Single-Node Batch: Silent Wrong Results

**Severity: MEDIUM**

`sentinel_model.py` line 236:
```python
gnn_max = global_max_pool(pool_embs, pool_batch)  # [B, gnn_hidden_dim]
```

`global_max_pool` groups nodes by `pool_batch` and takes the element-wise maximum. For a graph where only one function-level node passes the pool mask, `gnn_max == gnn_mean` (max of a single element = mean of a single element). The `gnn_eye_proj` then receives `[gnn_max, gnn_mean]` = `[v, v]` — a vector where the first half equals the second half.

This isn't wrong per se, but it means the projection layer receives a degenerate input pattern that it never learned to handle well (during training, most graphs have 5-20 function nodes, giving distinct max and mean). The projection's learned weights expect diverse max-vs-mean patterns; receiving identical halves wastes half the input capacity.

**The hostile read:** A tiny contract with one function (common in exploit-focused test contracts) gets a degenerate GNN eye. The classifier may produce a less confident prediction, or worse, a consistently biased prediction for single-function contracts.

**Recommendation:** Low priority, but consider adding a `num_func_nodes` feature to the GNN eye input so the classifier can condition on graph size.

---

## FINDING 3.4 — `aux_loss_weight` Inconsistency: CLI Default 0.3 vs TrainConfig Default 0.1

**Severity: HIGH**

Two different defaults exist for the auxiliary loss weight:

| Location | Default | When Used |
|---|---|---|
| `trainer.py:288` `TrainConfig.aux_loss_weight` | **0.1** | When TrainConfig is constructed programmatically |
| `scripts/train.py:194` `--aux-loss-weight` | **0.3** | When training via CLI |

The v5 documentation says λ=0.1 (line 44 of `sentinel_model.py` docstring). But the CLI overrides it to 0.3.

**The hostile read:** The documentation and model docstring say 0.1, but CLI training uses 0.3. A 3× increase in auxiliary loss weight means:
- Each eye's gradient is 3× stronger relative to the main classifier
- This can prevent eye dominance but can also cause the model to optimize for per-eye accuracy at the expense of joint accuracy
- If someone tunes the model with the CLI (0.3) and then creates a TrainConfig programmatically (0.1), they get different training dynamics with no explanation

**Recommendation:** Unify the default. The v5.1 proposal recommends increasing to 0.3, so update the TrainConfig default to match the CLI, and update the model docstring.

---

## FINDING 3.5 — No Gradient Collapse Detection, Only Logging

**Severity: HIGH**

`trainer.py:494-503` logs per-eye gradient norms:
```python
_gn_gnn = _grad_norm(model.gnn_eye_proj)
_gn_tf  = _grad_norm(model.transformer_eye_proj)
_gn_fus = _grad_norm(model.fusion)
```

But this is purely logging — no programmatic action is taken. If the GNN eye gradient norm drops to near-zero (the v5.0 collapse pattern), training continues with a dead eye. The loss may still decrease (the transformer and fused eyes compensate), but the model is effectively 2-eyed, not 3-eyed.

**The hostile read:** The v5.0 model trained to epoch 43 before anyone noticed the GNN eye had collapsed. The gradient norms were probably logged but nobody was watching. The model appeared to converge (loss decreased) while losing its structural analysis capability.

**Recommendation:** Add a gradient collapse detection callback:
```python
if _gn_gnn / max(_gn_tf, 1e-8) < 0.05:  # GNN gradient < 5% of transformer gradient
    logger.warning(f"Epoch {epoch}: GNN eye gradient collapse detected! "
                   f"gnn_grad={_gn_gnn:.6f} tf_grad={_gn_tf:.6f}")
    # Optionally: increase aux_loss_weight, decrease learning rate for other eyes
```

---

## FINDING 3.6 — `CrossAttentionFusion` Default `node_dim=64` Doesn't Match `GNNEncoder` Output

**Severity: MEDIUM**

`fusion_layer.py` line 79:
```python
node_dim: int = 64,
```

But `GNNEncoder` outputs `hidden_dim=128` by default. And `SentinelModel` correctly passes `node_dim=gnn_hidden_dim` (line 136). So the runtime is correct — but if anyone instantiates `CrossAttentionFusion()` with defaults (as the test files do), they get `node_dim=64`, which doesn't match the production model's `node_dim=128`.

**The hostile read:** The test `test_forward_output_shape` (line 49) creates `CrossAttentionFusion(node_dim=64, ...)` and feeds it 64-dim node embeddings. This test passes but doesn't test the production configuration (128-dim). If a bug exists only at 128 dims (e.g., attention pattern changes with wider inputs), the test won't catch it.

**Recommendation:** Update `CrossAttentionFusion.__init__` default to `node_dim=128` to match the production configuration. Update test defaults accordingly.

---

## FINDING 3.7 — `node_type_ids` Denormalisation Precision: Floating-Point Round-Trip

**Severity: MEDIUM**

`sentinel_model.py` line 221:
```python
node_type_ids = (graphs.x[:, 0] * 12.0).round().long()
```

This denormalises `type_id` by multiplying by 12.0 and rounding. The original normalisation was `float(type_id) / 12.0` (in `graph_extractor.py:563`). Let's check the round-trip for each type_id:

| Raw type_id | Normalised (÷12) | Denormalised (×12) | After round() | Correct? |
|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | ✅ |
| 1 | 0.0833... | 0.999... | 1 | ✅ |
| 5 | 0.4166... | 4.999... | 5 | ✅ |
| 6 | 0.5 | 6.0 | 6 | ✅ |
| 8 | 0.6666... | 7.999... | 8 | ✅ |
| 12 | 1.0 | 12.0 | 12 | ✅ |

The round-trip works for all current type_ids because `1/12 * 12 = 1` in IEEE 754 double precision (the 12 cancels). However, this is fragile:

1. If `graphs.x` is stored as `float16` (half precision) instead of `float32`, the round-trip breaks: `float16(1.0/12.0) * 12.0 ≈ 0.9995... * 12.0 ≈ 11.99...` which rounds to 12, not 1.
2. If the model is ever quantised to INT8 or FP16 for inference, the pool mask silently breaks.

**The hostile read:** PyTorch's default dtype for `torch.tensor()` is `float32` (line 823 of `graph_extractor.py`), so this is safe in the current pipeline. But if someone adds mixed-precision training (AMP) and the graph features get cast to FP16, the pool mask would assign every node to the wrong type_id, causing the GNN eye to pool over the wrong nodes.

**Recommendation:** Add a dtype guard at the top of `SentinelModel.forward()`:
```python
if graphs.x.dtype != torch.float32:
    logger.warning(f"graphs.x dtype is {graphs.x.dtype}, expected float32 — "
                   "pool mask denormalisation may be incorrect")
```

---

## FINDING 3.8 — `struct_mask` Includes CONTAINS Edges in Phase 1, But Phase 3 Also Uses CONTAINS

**Severity: MEDIUM (architectural concern)**

`gnn_encoder.py` lines 258-260:
```python
struct_mask   = edge_attr <= 5    # types 0-5 (includes CONTAINS)
cfg_mask      = edge_attr == 6
contains_mask = edge_attr == 5
```

Phase 1 uses `struct_mask` which includes CONTAINS (type 5). Phase 3 uses `contains_mask` which is also CONTAINS (type 5). This means:

- In Phase 1, FUNCTION nodes send messages DOWN to CFG_NODE children via CONTAINS edges.
- In Phase 3, CFG_NODE nodes send messages UP to FUNCTION parents via the same CONTAINS edges (reversed).

But there's a subtle issue: `struct_mask` includes ALL edge types 0-5, including CONTAINS. So in Phase 1, `conv1` and `conv2` process CONTAINS edges in both directions (because `add_self_loops=True` and GATConv adds the transpose by default? Actually no — PyG GATConv does NOT add reverse edges. The `struct_ei` contains edges in their original direction only. So CONTAINS edges in Phase 1 flow FUNCTION→CFG_NODE only.)

Wait — I need to verify this. Let me check: `edge_index[:, struct_mask]` selects edges where `edge_attr <= 5`. CONTAINS edges (type 5) flow from FUNCTION to CFG_NODE (as built in `graph_extractor.py:784`). So Phase 1 processes them in the forward direction: FUNCTION → CFG_NODE. Phase 3 reverses them with `.flip(0)`. This is correct.

**But**: Phase 1 also includes CONTAINS edges in the `struct_ei` passed to BOTH `conv1` AND `conv2`. After Phase 1, CFG_NODE nodes have already received messages from their parent FUNCTION via CONTAINS. Then Phase 3 reverses these same edges. The CFG_NODE's Phase-3 message to the FUNCTION includes information that originally came FROM the FUNCTION via Phase 1 CONTAINS. This creates a **circular information flow**:

```
Phase 1: FUNCTION →(CONTAINS)→ CFG_NODE
Phase 3: CFG_NODE →(reversed CONTAINS)→ FUNCTION
```

The FUNCTION node's Phase 3 embedding includes a "bounce-back" of its own Phase 1 output, filtered through the CFG_NODE's transformation. This isn't inherently wrong (it's how GNNs work), but it means the FUNCTION node's Phase 3 embedding is partially a function of itself, not purely of its CFG children's independent information.

**Recommendation:** This is an architectural observation, not a bug. The current design is defensible (residual connections make circular flow acceptable). But for v5.1, consider whether Phase 1 should EXCLUDE CONTAINS edges (use only types 0-4 for pure structural aggregation), leaving CONTAINS to Phase 3 only. This would make the information flow acyclic: Phase 1 aggregates inter-function context, Phase 2 encodes CFG order, Phase 3 propagates CFG signal UP to functions.

---

## FINDING 3.9 — `CrossAttentionFusion` Doesn't Handle Empty Node Lists

**Severity: HIGH**

`fusion_layer.py` line 181:
```python
padded_nodes, node_real_mask = to_dense_batch(nodes_proj, batch)
```

If `nodes_proj` is empty (N=0 nodes), `to_dense_batch` returns empty tensors. The subsequent attention operations would fail or produce degenerate results.

When can N=0 happen? When a graph is loaded that has zero nodes after some filtering step. This shouldn't happen if `EmptyGraphError` is raised properly (Finding 2.12), but the ghost graph fallback means interface-only graphs DO enter the model with very few nodes (maybe 1-2, not 0).

However, there's a more realistic scenario: if `batch` has size B=4 but one of the 4 graphs has 0 nodes (possible if the graph filtering is wrong), `to_dense_batch` would produce a `padded_nodes` tensor where one graph has `max_nodes=0` for that batch element. The `node_real_mask` would be all-False for that graph, and `node_padding_mask` would be all-True.

Then in Step 3 (node→token attention), the Q tensor for that graph would be empty. PyTorch's `MultiheadAttention` would receive a query of shape `[1, 0, 256]` which may crash or produce undefined behavior.

**Recommendation:** Add a guard in `CrossAttentionFusion.forward()`:
```python
if node_embs.size(0) == 0:
    return torch.zeros(B, self.output_dim, device=token_embs.device)
```

---

## FINDING 3.10 — GNN Eye Uses Both `global_max_pool` and `global_mean_pool` but No Normalisation

**Severity: MEDIUM**

`sentinel_model.py` lines 236-240:
```python
gnn_max  = global_max_pool(pool_embs, pool_batch)   # [B, gnn_hidden_dim]
gnn_mean = global_mean_pool(pool_embs, pool_batch)  # [B, gnn_hidden_dim]
gnn_eye  = self.gnn_eye_proj(
    torch.cat([gnn_max, gnn_mean], dim=1)           # [B, 2*gnn_hidden_dim]
)
```

`global_max_pool` returns the element-wise maximum across function nodes, while `global_mean_pool` returns the mean. The max values are typically larger than the mean values (max ≥ mean for any distribution). The `gnn_eye_proj` receives a concatenated vector where the first 128 dims are systematically larger than the second 128 dims.

The `gnn_eye_proj` is a `Linear(256, 128)` that can learn to compensate for this scale difference. But the systematic asymmetry means the first 128 input dimensions have higher variance and may dominate the gradient, similar to the feature scale issue in Finding 2.9.

**Recommendation:** Consider normalising `gnn_max` and `gnn_mean` independently before concatenation (e.g., LayerNorm), or using only one pooling method. The max+mean combo provides complementary information (max captures outliers, mean captures average), so keeping both is reasonable but normalisation would help.

---

## FINDING 3.11 — `num_layers=4` Stored But Not Validated in GNNEncoder

**Severity: LOW**

`gnn_encoder.py` line 120-125:
```python
num_layers: int = 4,
...
self.num_layers = num_layers
```

The docstring says "validation fires in TrainConfig.__post_init__()". But GNNEncoder itself never checks that `num_layers >= 4`. If someone passes `num_layers=2`, GNNEncoder creates conv1+conv2 but conv3 and conv4 are still created (they're unconditional). The `num_layers` attribute is stored but never used to gate layer creation.

Actually, looking at the code more carefully, ALL four conv layers are always created regardless of `num_layers`. The `num_layers` is only stored for serialisation. This means:
- `num_layers=2` would still create conv3 and conv4, but they'd never be called
- `num_layers=5` would be expected to have 5 layers but only 4 exist

**The hostile read:** If v5.1 implements `num_layers=5` for 2 CONTROL_FLOW hops, the existing GNNEncoder doesn't support it — it always creates exactly 4 conv layers. The `num_layers` parameter is decorative, not functional.

**Recommendation:** Either (a) make `num_layers` actually control the number of layers (dynamic layer creation), or (b) remove the parameter and hardcode 4, documenting that changing the number of layers requires code changes.

---

## FINDING 3.12 — Phase 2 CONTROL_FLOW Mask Assumes Edge Type 6, But No Validation

**Severity: MEDIUM**

`gnn_encoder.py` line 259:
```python
cfg_mask = edge_attr == 6
```

This hardcodes `6` as the CONTROL_FLOW edge type ID. If the edge type vocabulary ever changes (e.g., adding REVERSE_CONTAINS as type 7, or inserting a new type that shifts CONTROL_FLOW to 7), this comparison silently breaks. Phase 2 would process zero edges (all False mask), and the GNN would lose all CFG ordering signal.

The constant `NUM_EDGE_TYPES=7` is imported from `graph_schema`, but the specific edge type IDs (0-6) are NOT imported — they're only available through the `EDGE_TYPES` dict.

**Recommendation:** Import `EDGE_TYPES` and use:
```python
cfg_mask = edge_attr == EDGE_TYPES["CONTROL_FLOW"]
```
This makes the code self-documenting and resilient to schema changes.

---

## Summary Table

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 3.1 | Phase 3 reverse-CONTAINS edge embedding symmetry | HIGH | Architecture |
| 3.2 | Ghost graph fallback silently degrades predictions | HIGH | Robustness |
| 3.3 | `global_max_pool` on single-node graphs: degenerate input | MEDIUM | Edge case |
| 3.4 | `aux_loss_weight` inconsistency: CLI 0.3 vs TrainConfig 0.1 | HIGH | Config drift |
| 3.5 | No gradient collapse detection, only logging | HIGH | Observability |
| 3.6 | `CrossAttentionFusion` default node_dim=64 ≠ GNNEncoder output 128 | MEDIUM | Test coverage |
| 3.7 | `node_type_ids` denormalisation fragile with FP16/quantisation | MEDIUM | Precision |
| 3.8 | Phase 1 includes CONTAINS, creating circular info flow with Phase 3 | MEDIUM | Architecture |
| 3.9 | `CrossAttentionFusion` doesn't handle empty node lists | HIGH | Robustness |
| 3.10 | GNN eye max+mean pool scale asymmetry | MEDIUM | Feature engineering |
| 3.11 | `num_layers` parameter is decorative, not functional | LOW | Code quality |
| 3.12 | Phase 2 hardcodes edge type 6 instead of using EDGE_TYPES dict | MEDIUM | Maintainability |

**4 HIGH, 6 MEDIUM, 2 LOW**

The four HIGH findings (3.1, 3.2, 3.4, 3.9) are the priority. Finding 3.1 (Phase 3 symmetry) is the root cause of the gradient collapse that killed v5.0. Finding 3.2 (ghost graph fallback) perpetuates the interface-extraction problem from Group 2. Finding 3.4 (aux_loss_weight drift) can silently change training dynamics. Finding 3.9 (empty nodes in fusion) is a crash risk.

---

Ready for **Group 4: Transformer Path End-to-End** (`transformer_encoder.py` + `fusion_layer.py` + `sentinel_model.py`) whenever you want to continue.