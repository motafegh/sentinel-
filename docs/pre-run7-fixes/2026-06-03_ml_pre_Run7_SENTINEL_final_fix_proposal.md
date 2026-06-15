# SENTINEL Run 7 — Final Fix Proposal

**Status:** Final — replaces all previous draft proposals  
**Date:** 2026-06-03  
**Baseline:** Run 6 (`GCB-P1-Run6-v10-20260602`), best ep24 F1=0.2971, in progress  
**Target:** Run 7 — same v10 data, same splits, architecture + training signal corrections only  
**No re-extraction required for any fix in this document.**

---

## Executive Summary

This proposal consolidates findings from three independent sources: (1) the adversarial code audit (13 CRITICAL, 51 WARNING issues across 27 files), (2) the interpretability experiment suite (21 experiments, EXP-L1 through EXP-S4), and (3) source-code-level gradient tracing that confirmed or corrected the original weakness analysis (W1–W12). Five changes survive the validation filter — two are bugs (broken gradient paths), three are architecture/training improvements (suboptimal but not broken).

**Expected macro F1 gain over converged Run 6 peak: +0.06–0.10**  
**Estimated Run 7 peak: 0.40–0.45** (data ceiling — label co-occurrence, EMITS extraction bug — then becomes the binding constraint)

| ID | Type | File | Impact | Risk |
|----|------|------|--------|------|
| BUG-R7-1 | Bug | sentinel_model.py | HIGH | LOW |
| BUG-R7-2 | Bug | gnn_encoder.py | MEDIUM | MEDIUM |
| IMP-R7-1 | Architecture | gnn_encoder.py | MEDIUM | LOW |
| IMP-R7-2 | Architecture | sentinel_model.py | HIGH | MEDIUM |
| IMP-R7-3 | Training | trainer.py / train.py | LOW (amplifies others) | LOW |

### What This Proposal Does NOT Include (and Why)

These items are explicitly deferred beyond Run 7:

| Item | Reason for Deferral |
|------|-------------------|
| Label-correlation penalty (W9) | 99% DoS↔Reentrancy co-occurrence cannot be solved by a loss term; requires label correction. A penalty risks suppressing genuinely different contracts. |
| EMITS edge extraction fix | Only 12 EMITS edges across 41K contracts (confirmed: graph_extractor.py bug). Fixing it requires re-extraction → separate Run 8 after data pipeline work. |
| `max_nodes` increase (1024→2048) | Requires re-extraction and VRAM recalibration. Not worth the risk when Run 7 is purely architecture-focused. |
| Temperature scaling calibration | Post-hoc; applies to any checkpoint. Can be done after Run 7 without code changes. |
| Timestamp size normalisation | Requires re-extraction. Deferred to data pipeline sprint. |
| solc-select upgrade | Unblocks counterfactual validation (EXP-L6) but does not affect training. Can be done in parallel. |

---

## BUG-R7-1 — aux_phase2 supervises FUNCTION nodes that Phase 2 never updated

### Root Cause (Source-Code Verified)

In `sentinel_model.py:521`:
```python
# CURRENT — BUG:
phase2_pooled = global_mean_pool(_phase2_x[pool_mask], pool_batch, size=num_graphs)
# pool_mask = func_mask  →  FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes
```

`_phase2_x` is the output of Phase 2 layers (conv3, conv3b, conv3c) with `add_self_loops=False`. FUNCTION nodes have zero incoming CONTROL_FLOW, CALL_ENTRY, RETURN_TO, or DEF_USE edges. GATConv with `add_self_loops=False` returns **zero vectors** for nodes with no incoming edges.

For FUNCTION nodes in Phase 2:
```
x2 = conv3(x, cf_only_ei)    # cf_only_ei has no FUNCTION node targets → x2[func] = 0
x2 = relu(x2)                 # relu(0) = 0, gradient = 0
x  = x + dropout(x2)          # x[func] ≈ x_phase1[func] unchanged
```

The gradient chain from `aux_phase2_loss` through `_phase2_x[func_mask]`:
```
loss → phase2_pooled → _phase2_x[func_mask] → phase_norm[1] → x_prev + dropout(x2)
  identity branch (x_prev):  gradient → Phase 1 parameters        ← WRONG TARGET
  phase2 branch  (x2):       gradient → relu'(0) = 0 → DIES       ← Phase 2 gets nothing
```

**Result:** `aux_phase2` has been training Phase 1 parameters through the residual pass-through at FUNCTION nodes. Phase 2 conv layers (conv3, conv3b, conv3c) receive almost zero gradient from this loss. The intended purpose is exactly inverted.

### Additional Context: Why This Was Hard to Spot

Phase 3's REVERSE_CONTAINS edges (conv4/conv4b, CFG→FUNCTION upward pass) do propagate Phase 2 signal to FUNCTION nodes indirectly. This means the GNN eye's final pooled output is not completely blind to Phase 2 — it receives attenuated signal through the Phase 3 bridge. However, the aux_phase2 loss, which is supposed to provide a **direct** training signal to Phase 2 conv parameters, is structurally disconnected because it pools the wrong node types. The fused eye (CrossAttentionFusion over ALL nodes) provides yet another alternative gradient path, but this is diffuse and not targeted.

The net effect is **gradient starvation** of Phase 2 — it learns slowly through indirect paths rather than being directly supervised. This explains the interpretability findings: EXP-L2 measured CFG edge ablation effect at 1.08×10⁻⁶, five orders of magnitude below the 0.03 significance threshold.

### Fix

`sentinel_model.py:521` — pool over CFG_NODE_* types (8–12) instead:

```python
# FIXED — pool over CFG nodes, which Phase 2 actually updated:
_CFG_NODE_TYPES = torch.tensor([8, 9, 10, 11, 12], device=node_embs.device)
cfg_node_mask   = torch.isin(node_type_ids, _CFG_NODE_TYPES)
# Ghost-graph guard: if no CFG nodes, zero vector (same as existing ghost handling)
if cfg_node_mask.any():
    phase2_pooled = global_mean_pool(
        _phase2_x[cfg_node_mask], batch[cfg_node_mask], size=num_graphs
    )
else:
    phase2_pooled = torch.zeros(
        num_graphs, self.gnn.hidden_dim,
        device=node_embs.device, dtype=_phase2_x.dtype
    )
aux_phase2_logits = self.aux_phase2(phase2_pooled)
```

Note: `node_type_ids` is already computed at line 425 and available in scope. The `_CFG_NODE_TYPES` tensor allocation is inside `return_aux=True` branch only — training path, no inference overhead.

Also update the stale comment above line 520:
```python
# Phase 2 CEI aux head: pool phase2 embeddings over CFG_NODE_* nodes (types 8–12).
# FUNCTION nodes are excluded — they receive zero Phase 2 messages and pooling them
# would route gradient to Phase 1 parameters instead of Phase 2 conv layers.
```

### Risk Assessment

**LOW.** One-line logic change in a training-only branch. No shape changes. No new parameters. The only failure mode is a graph with no CFG nodes — the ghost-graph guard above handles this.

### Verification Gate

After implementation, run a 2-epoch smoke test and extract the gradient of `conv3.lin_l.weight` with respect to `aux_phase2_loss` specifically. Should be non-zero after this fix. If gradient is still zero, the fix was not applied correctly.

---

## BUG-R7-2 — type_id encoded as continuous float; GNN treats categorical as numeric

### Root Cause (Source-Code Verified)

In `graph_extractor.py:708`, node type is stored as:
```python
float(cfg_type) / _MAX_TYPE_ID    # e.g. FUNCTION = 1/12 = 0.0833
```

In `gnn_encoder.py:554`, this float enters GATConv directly:
```python
x = self.conv1(x_init, struct_ei, struct_ea)   # GATConv(in_channels=11, ...)
```

GATConv computes attention as a linear function of node features. With type_id as a scalar on [0.0, 0.083, 0.167, ... 1.0], the attention kernel treats FUNCTION (0.083) and STATE_VAR (0.0) as differing by 0.083 in one of 11 dimensions — geometrically trivial. There is no representation space for the model to learn "FUNCTION and CFG_NODE_CALL require fundamentally different message aggregation rules."

A learnable `nn.Embedding(13, 16)` gives each of the 13 node types an independent 16-dim vector with no imposed ordering. The GAT then learns arbitrary similarity structures between node categories, which is essential for routing CONTAINS, READS, WRITES, and CF edges correctly by source and target type.

**No re-extraction needed.** The integer type_id is recoverable inside `GNNEncoder.forward()` from `x[:, 0]` via `round(x[:, 0] * 12)`. The `.pt` files stay unchanged.

### Fix

**gnn_encoder.py — `__init__`:**

Add after the existing imports and before class definition:
```python
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES, EDGE_TYPES, NODE_TYPES

_NODE_TYPE_EMB_DIM: int = 16          # embedding dim for node type (13 categories)
_GNN_MAX_TYPE_ID:   float = float(max(NODE_TYPES.values()))   # 12.0 for v8 schema
assert _GNN_MAX_TYPE_ID == 12.0, (
    f"_GNN_MAX_TYPE_ID={_GNN_MAX_TYPE_ID} expected 12.0. "
    "A node type was added — update gnn_encoder.py node_type_embedding size."
)
# Effective GNN input dim after replacing float type_id with embedding:
# NODE_FEATURE_DIM dims - 1 (type_id float) + _NODE_TYPE_EMB_DIM = 11 - 1 + 16 = 26
_GNN_EFFECTIVE_INPUT_DIM: int = NODE_FEATURE_DIM - 1 + _NODE_TYPE_EMB_DIM  # 26
```

**GNNEncoder.__init__:** add embedding, update dependent layers:
```python
# Node type embedding: replaces scalar type_id/12.0 with learnable 16-dim vector.
# 13 types × 16 dims = 208 parameters — negligible.
self.node_type_embedding = nn.Embedding(13, _NODE_TYPE_EMB_DIM)

# Update conv1 and input_proj to accept the enriched input dim (26 instead of 11)
self.input_proj = nn.Linear(_GNN_EFFECTIVE_INPUT_DIM, hidden_dim, bias=False)
self.conv1 = GATConv(
    in_channels=_GNN_EFFECTIVE_INPUT_DIM,   # 26 (was 11)
    out_channels=_head_dim,
    heads=heads,
    concat=True,
    add_self_loops=True,
    edge_dim=_edge_dim,
)
```

**GNNEncoder.forward():** enrich `x` AFTER the shape validation check, BEFORE any conv:
```python
# Validate disk format (still 11 dims on disk)
if x.shape[1] != NODE_FEATURE_DIM:
    raise ValueError(...)   # ← existing check stays unchanged

# BUG-R7-2: recover integer type_id, embed it, replace scalar dim 0.
# x[:, 0] = float(type_id) / 12.0 → multiply back, round, clamp for safety.
if x.dtype != self._param_dtype:
    x = x.to(self._param_dtype)
_type_id_int = (x[:, 0].float() * _GNN_MAX_TYPE_ID).round().clamp(0, 12).long()
_type_emb    = self.node_type_embedding(_type_id_int).to(self._param_dtype)
x = torch.cat([x[:, 1:], _type_emb], dim=-1)                   # [N, 10+16=26]
# x is now _GNN_EFFECTIVE_INPUT_DIM=26 dims; conv1 and input_proj expect 26.
```

This goes between the shape check and the `x_init = x` line (IMP-G2 skip capture).

### Risk Assessment

**MEDIUM.** Changes GNNEncoder's input interface internally. The disk format (11-dim graph.x) is unchanged. Checkpoint incompatibility is expected and acceptable — Run 7 trains fresh.

**Cautions:**

1. **validate_graph_integrity check** (`edge_index.max() >= x.shape[0]`) uses `x.shape[0]` which is the node count — this is unaffected by the enrichment (shape[1] changes, not shape[0]). No issue.

2. **dtype ordering matters.** The dtype cast `x = x.to(self._param_dtype)` must be applied BEFORE the embedding lookup. After enrichment, `x` is float32 (from cat with embedding output). If AMP changes dtype mid-forward, the embedding output must match. The code above applies the dtype cast to the original `x` first, then casts `_type_emb` to match.

3. **input_proj weight shape changes** from [256, 11] to [256, 26]. Any pre-trained checkpoint for input_proj will be incompatible. This is expected — Run 7 trains from scratch.

---

## IMP-R7-1 — Phase 2 uses heads=1 (8× capacity gap vs Phase 1)

### Current State (Source-Code Verified)

Phase 1 convs (`conv1`, `conv2`): `heads=8, concat=True` → 8 independent attention distributions  
Phase 2 convs (`conv3`, `conv3b`, `conv3c`): `heads=1, concat=False` → 1 attention distribution

Phase 2 is asked to simultaneously encode: intra-function execution order (CF edges), cross-function call structure (CALL_ENTRY/RETURN_TO), and data flow (DEF_USE). With one attention head, these three structurally different relationships compete for the same attention distribution. Multi-head attention exists precisely for this case.

### Why heads=1 Was Originally Chosen

The original rationale (preserved in gnn_encoder.py comments) was that heads=1 with full hidden_dim capacity gives Phase 2 the maximum per-head representation power for encoding execution order. This makes sense for a single-relationship encoding task, but Phase 2 now handles three distinct edge subsets (CF-only in layer 3, ICFG-only in layer 4, joint in layer 5). The IMP-G1 layer-specific edge design increases Phase 2's structural diversity but the single attention head cannot specialise across these different relationship types.

### Fix

**gnn_encoder.py — `__init__`:** Change all three Phase 2 convs.

The key identity that keeps output shape and parameter count identical:
```
heads=1, concat=False, out_channels=hidden_dim=256:
    output: [N, 256],  params: 1 × (256×256) = 65,536

heads=4, concat=True,  out_channels=hidden_dim//4=64:
    output: [N, 4×64=256],  params: 4 × (256×64) = 65,536
```

Same output shape [N, 256] → residual connections `x = x + dropout(x2)` are unchanged.  
Same parameter count → no overfitting concern from added capacity.

```python
_ph2_out_per_head: int = hidden_dim // 4   # 64; 4 heads × 64 = 256 = hidden_dim

self.conv3 = GATConv(
    in_channels=hidden_dim,
    out_channels=_ph2_out_per_head,   # 64 (was 256)
    heads=4,                           # (was 1)
    concat=True,                       # (was False) — 4×64=256 = hidden_dim ✓
    add_self_loops=False,
    edge_dim=_edge_dim,
)
# conv3b and conv3c: identical change
```

Update docstring comment:
```
Phase 2 (Layers 3+4+5): heads=4, concat=True, out=64 per head → 256 total
  4 heads allow simultaneous specialisation: execution order / call structure
  / data-flow / interaction patterns. Same parameter count as heads=1.
```

### Risk Assessment

**LOW.** Output shape is identical to current. Parameter count is identical. Residual connections are unaffected. The only behavioral change is richer attention capacity — four independent attention patterns instead of one. This is strictly a capacity improvement, not a structural change.

**Cautions:**

1. With `concat=True`, GATConv internally computes 4 separate `[N, 64]` outputs and concatenates them to `[N, 256]`. This is numerically different from the `concat=False` mean-averaging. The residual `x + x2` still works. Phase 3 still receives `[N, 256]`. No issue.

2. If BUG-R7-1 is not applied, increasing heads=4 will not help much — the gradient still dies at the pooling stage. **IMP-R7-1 must be applied after BUG-R7-1.**

---

## IMP-R7-2 — GNN eye is blind to Phase 2 sequential signal; add CFG eye

### Root Cause (Source-Code Verified)

The GNN eye pools exclusively over FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes (`pool_mask = func_mask`). Phase 2 (CFG edges) does not update FUNCTION nodes — their Phase 2 representation is near-identical to their Phase 1 representation (residual pass-through of zeros). The GNN eye therefore carries no execution-sequence information.

JK aggregation partially helps for the **fused eye** path (which processes all node types). But the GNN eye — responsible for 27–40% of gradient share — sees only Phase 1 + Phase 3 aggregated signal, which destroys CEI sequence ordering.

The fix: add a fourth eye that pools Phase 2 output directly over CFG nodes, giving the classifier a clean gradient path through conv3/conv3b/conv3c. This is complementary to BUG-R7-1 (which gives Phase 2 a training gradient) — BUG-R7-1 trains Phase 2 better, IMP-R7-2 makes Phase 2's learned representations visible at inference time.

### Why Not Just Fix the GNN Eye Pooling?

One might ask: why add a 4th eye instead of simply including CFG nodes in the GNN eye pool? The answer is separation of concerns. The GNN eye's current function-level pooling gives a clean "contract-level structural summary" signal. Mixing CFG nodes into that pool would change the GNN eye's semantics (it would become a mix of function-level and statement-level features) and could destabilise the existing well-calibrated gradient share (27–40%). A dedicated CFG eye preserves the GNN eye's identity while adding a new, orthogonal information channel.

### Fix

**sentinel_model.py — `__init__`:**
```python
# CFG eye: direct pool of Phase 2 output over CFG_NODE_* nodes.
# Provides classifier with unattenuated CEI-sequence signal.
# Input: max+mean of _phase2_x[cfg_nodes] → [B, 2×gnn_hidden_dim]
self.cfg_eye_proj = nn.Sequential(
    nn.Linear(2 * gnn_hidden_dim, eye_dim),   # 512 → 128
    nn.ReLU(),
    nn.Dropout(dropout),
)

# Classifier now takes 4 eyes: gnn + transformer + fused + cfg
_cls_input  = 4 * eye_dim    # 512 (was 384)
_cls_hidden = 256             # increased from 192 to handle wider input
self.classifier = nn.Sequential(
    nn.Linear(_cls_input,  _cls_hidden),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(_cls_hidden, num_classes),
)
```

**sentinel_model.py — `forward()`:**

1. Always return `_phase2_x` from GNN (remove the `return_aux` condition):
```python
# Always need _phase2_x for cfg_eye (both training and inference)
_gnn_out = self.gnn(
    graphs.x, graphs.edge_index, graphs.batch, edge_attr,
    return_phase2_embs=True,   # always (was: return_phase2_embs=return_aux)
)
node_embs, batch, _jk_entropy, _phase2_x = _gnn_out
```

2. Build the CFG eye (after `node_type_ids` and `func_mask` are computed):
```python
# CFG eye: pool Phase 2 output over CFG_NODE_* types (8–12)
_CFG_NODE_TYPES = torch.tensor([8, 9, 10, 11, 12], device=node_embs.device)
cfg_node_mask   = torch.isin(node_type_ids, _CFG_NODE_TYPES)
if cfg_node_mask.any():
    p2_max  = global_max_pool(
        _phase2_x[cfg_node_mask], batch[cfg_node_mask], size=num_graphs
    )
    p2_mean = global_mean_pool(
        _phase2_x[cfg_node_mask], batch[cfg_node_mask], size=num_graphs
    )
else:
    # Ghost/interface-only graph: no CFG nodes → zero embedding
    p2_max = p2_mean = torch.zeros(
        num_graphs, self.gnn.hidden_dim,
        device=node_embs.device, dtype=_phase2_x.dtype
    )
cfg_eye = self.cfg_eye_proj(torch.cat([p2_max, p2_mean], dim=1))   # [B, 128]
```

3. Update the main classifier combination:
```python
combined = torch.cat([gnn_eye, transformer_eye, fused_eye, cfg_eye], dim=1)  # [B, 512]
logits   = self.classifier(combined)
```

4. Update the empty-batch guard (around line 445) to zero `cfg_eye`:
```python
zeros    = torch.zeros(B, self.num_classes, device=dev)
if not return_aux:
    return zeros
aux_zeros = {
    "gnn":         torch.zeros(B, self.num_classes, device=dev),
    "transformer": torch.zeros(B, self.num_classes, device=dev),
    "fused":       torch.zeros(B, self.num_classes, device=dev),
    "phase2":      torch.zeros(B, self.num_classes, device=dev),
    "jk_entropy":  torch.tensor(0.0, device=dev),
}
return zeros, aux_zeros
```

5. Update `parameter_summary()` to include `cfg_eye_proj`.

### Risk Assessment

**MEDIUM.** Classifier input shape changes 384→512. This is a clean architectural change with no interaction effects on existing eyes. The main risks:

1. **VRAM:** `_phase2_x` is now retained across the full forward pass (was only during training). This adds `N_total × hidden_dim × sizeof(bf16)` = approximately `mean_nodes × 256 × 2 bytes`. For a batch of 8 contracts at ~150 nodes each: 8×150×256×2 = 614 KB — negligible. At worst case (large contracts): still well under 100 MB.

2. **Backward speed:** One additional pooling operation + one Linear(512, 256) backward. Estimated overhead: <5% per step.

3. **Early training instability:** The CFG eye starts from random init while the GNN eye has a head start. This is the same situation as the existing three eyes. The aux loss warmup (Fix #33 from previous runs) already handles early-epoch gradient imbalance.

4. **Inference cost:** `_phase2_x` is now always computed (was conditional). The extra computation is the Phase 2 output itself, which was already computed internally — we just retain the reference. The only new overhead is the `cfg_eye_proj` forward pass at inference time (~65K parameters, negligible).

---

## IMP-R7-3 — aux_phase2 weight 0.10 → 0.20

### Rationale

After BUG-R7-1 fix, `aux_phase2` will actually provide gradient to Phase 2 conv layers for the first time. The current weight of 0.10 was calibrated for a broken loss that reached Phase 1 instead. Now that the gradient path is correct, doubling the weight is appropriate to give Phase 2 meaningful training signal.

Do not exceed 0.20. The total auxiliary loss at 0.30 base already represents a significant fraction of the training signal. Adding 0.20 for Phase 2 on top brings total auxiliary weight to 0.50 (0.30 + 0.20) — beyond this, auxiliary heads can dominate main loss early in training. The existing warmup over 8 epochs handles this.

### Fix

**trainer.py — `TrainConfig`:**
```python
aux_phase2_loss_weight: float = 0.20   # was 0.10; now meaningful (BUG-R7-1 fixed)
```

**train.py — CLI default:**
```python
p.add_argument("--aux-phase2-loss-weight", type=float, default=0.20, ...)
```

### Risk Assessment

**LOW.** The warmup mechanism prevents early instability. The gradient path is now correct (BUG-R7-1), so this weight actually reaches Phase 2 parameters.

**Caution:** If BUG-R7-1 is not correctly applied, increasing this weight will amplify the wrong gradient (to Phase 1 instead of Phase 2). **IMP-R7-3 must be applied after BUG-R7-1 is verified.**

---

## What NOT to Change for Run 7

The following are explicitly preserved. Do not modify these without a separate proposal.

| Parameter | Value | Reason to preserve |
|-----------|-------|-------------------|
| `jk_entropy_reg_lambda` | 0.005 | Phases balanced (0.315–0.374) — working correctly |
| `gnn_lr_multiplier` | 2.5 | GNN gradient share 27–40% — well-calibrated |
| `lora_lr_multiplier` | 0.3 | Prevents CodeBERT catastrophic forgetting |
| `fusion_lr_multiplier` | 0.5 | RC1 fix — prevents fusion gradient dominance |
| `asl_gamma_neg` | 2.0 | BUG-C4 confirmed — 4.0 caused all-zeros collapse |
| `asl_clip` | 0.01 | BUG-M2 confirmed — 0.05 caused oscillation |
| `gnn_hidden_dim` | 256 | Well-dimensioned; increasing risks VRAM |
| `lora_r` / `lora_alpha` | 16 / 32 | Working; changing invalidates Run 6 comparison |
| `batch_size` × `grad_accum` | 8 × 8 = 64 effective | GPU-calibrated |
| `early_stop_patience` | 30 | Appropriate for 100-epoch run |
| `dos_loss_weight` | 0.5 | Existing mitigation; data problem not model |
| `class_label_smoothing` | (existing per-class) | Calibrated to known noise rates |
| `eval_threshold` | 0.35 | Prevents threshold-boundary F1 noise in patience |
| Data paths | v10_deduped / cached_dataset_v10.pkl | Stable, validated |

**Specifically: do not add a label-correlation penalty (W9) for DoS/Reentrancy in Run 7.** The data co-occurrence problem (99% DoS↔Reentrancy) cannot be solved by a loss term — it requires label correction. A correlation penalty risks suppressing predictions for contracts that genuinely differ. This is deferred to after data re-labeling work.

---

## Dependency Graph

The five fixes have strict ordering dependencies. Violating this order risks compounding errors or applying fixes to a codebase where they cannot function correctly.

```
BUG-R7-1 (aux_phase2 pooling fix)
    ↓  verified
IMP-R7-1 (Phase 2 heads 1→4)       ← depends on BUG-R7-1 for gradient to reach Phase 2
    ↓
BUG-R7-2 (type_id embedding)        ← independent of above but same file as IMP-R7-1
    ↓
IMP-R7-2 (CFG eye)                  ← requires _phase2_x always available; benefits from all above
    ↓  smoke test
IMP-R7-3 (aux weight 0.10→0.20)     ← meaningless without BUG-R7-1
```

BUG-R7-2 and IMP-R7-1 both modify `gnn_encoder.py` and can be implemented in parallel (same edit session), but IMP-R7-1's effectiveness depends on BUG-R7-1 being correct.

---

## Smoke Test Gates (Before Full Run 7)

Run 2 epochs at 10% data subsample:
```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
  --run-name GCB-P1-Run7-smoke-YYYYMMDD \
  --experiment-name sentinel-multilabel \
  --epochs 2 \
  --smoke-subsample-fraction 0.1
```

All gates must pass before launching full Run 7:

| Gate | Criterion | Failure Action |
|------|-----------|----------------|
| G1 — No NaN | Zero NaN losses in first 50 steps | Debug loss path before proceeding |
| G2 — GNN share | ≥ 15% by step 100 of epoch 1 | Increase `gnn_lr_multiplier` |
| G3 — Ph2/Ph1 ratio | > 0.25 by step 200 | BUG-R7-1 or IMP-R7-1 not applied correctly |
| G4 — CFG eye gradient | `cfg_eye_proj` grad norm > 0 at step 100 | IMP-R7-2 forward path broken |
| G5 — JK balance | No phase weight < 0.10 | Entropy reg not functioning |
| G6 — VRAM | < 7.0/8.0 GiB peak during epoch 1 | Reduce batch_size before full run |
| G7 — Shape sanity | `combined` tensor shape `[B, 512]`; logits `[B, 10]` | Classifier not rebuilt correctly |

**Additional verification (one-off, not during smoke):**  
After smoke epoch 1, extract gradient of `conv3.lin_l.weight` with respect to `aux_phase2_loss` specifically. Should be non-zero after BUG-R7-1 fix. This confirms the bug fix actually reached Phase 2 parameters.

**Phase gate:** Run smoke test after step 4 (IMP-R7-2, before IMP-R7-3) to verify no shape errors. Run smoke test again after step 5 for final G1 verification.

---

## Implementation Order

Implement in this sequence to avoid compounding errors:

1. **BUG-R7-1** — `sentinel_model.py:521` — 5 lines, lowest risk, verify first
2. **BUG-R7-2** — `gnn_encoder.py` — add embedding, update conv1/input_proj
3. **IMP-R7-1** — `gnn_encoder.py` — change Phase 2 heads (parallel with BUG-R7-2 in same edit session)
4. **IMP-R7-2** — `sentinel_model.py` — add CFG eye, update classifier
5. **IMP-R7-3** — `trainer.py` + `train.py` — weight change, last

Run smoke test after step 4 (before step 5) to verify no shape errors.  
Run smoke test again after step 5 for final G1 verification.

---

## Parameter Count Changes

| Component | Run 6 | Run 7 | Delta |
|-----------|-------|-------|-------|
| node_type_embedding | 0 | 13×16 = 208 | +208 |
| input_proj | 11×256 = 2,816 | 26×256 = 6,656 | +3,840 |
| conv1 | GATConv(11,32,h=8) ≈ 9,984 | GATConv(26,32,h=8) ≈ 23,552 | +13,568 |
| cfg_eye_proj | 0 | 512×128+128 = 65,664 | +65,664 |
| classifier | 384×192+192 = 73,920 | 512×256+256 = 131,328 | +57,408 |
| Phase 2 convs × 3 | unchanged (same param count) | unchanged | 0 |
| **Total delta** | | | **~+140K params** |

Run 6 total trainable: ~2.4M GNN + ~1.2M LoRA + ~500K fusion/classifier ≈ 4.1M  
Run 7 total trainable: ~4.24M — a 3.4% increase. No overfitting concern on 29K contracts.

---

## Run 7 Launch Command

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
  --run-name GCB-P1-Run7-v10-YYYYMMDD \
  --experiment-name sentinel-multilabel \
  --epochs 100 \
  --aux-phase2-loss-weight 0.20
```

All other parameters use train.py defaults (already updated in steps above).  
Splits: `ml/data/splits/v10_deduped/`  
Cache: `ml/data/cached_dataset_v10.pkl`

---

## Expected Training Dynamics vs Run 6

| Signal | Run 6 observed | Run 7 expected |
|--------|---------------|----------------|
| Ph2/Ph1 grad ratio | 0.15–0.43 | 0.40–0.80 (BUG-R7-1 + IMP-R7-1) |
| GNN eye gradient share | 27–40% | 25–38% (CFG eye adds 4th path, dilutes slightly) |
| JK phase weights | ~0.33/0.33/0.33 | ~0.30/0.35/0.35 (Phase 2 richer, JK may weight higher) |
| Reentrancy F1 | ~0.27 est. | 0.40–0.50 (direct CEI gradient path via CFG eye) |
| IntegerUO F1 | 0.68 | 0.70–0.73 (marginal, near ceiling) |
| GasException F1 | 0.36 | 0.45–0.50 |
| Macro F1 peak | ~0.36–0.38 | ~0.40–0.45 |

If Phase 2 now receives real gradient (BUG-R7-1 verified), expect noticeable improvement in Ph2/Ph1 ratio within the first 5 epochs. If ratio stays at Run 6 levels (0.15–0.20), re-examine the BUG-R7-1 implementation before continuing.

---

## Post-Run 7 Roadmap (Beyond This Proposal)

These items are not part of Run 7 but are tracked for the next sprint:

1. **EMITS edge extraction fix** — Currently only 12 EMITS edges across 41K contracts. Fixing requires graph_extractor.py patch + full re-extraction. The 15.46× UnusedReturn enrichment ratio suggests EMITS edges carry strong signal once extracted correctly. Target: Run 8.

2. **Label correction sprint** — DoS↔Reentrancy 99% co-occurrence is a data problem, not a model problem. Requires manual audit or automated label cleaning (label_cleaner.py exists but needs validation). Target: Run 8 data pipeline.

3. **Temperature scaling calibration** — Post-hoc, zero training cost. Can be applied to any Run 7 checkpoint after training. Expected ECE reduction from ~0.25 to <0.05. Target: immediately after Run 7.

4. **max_nodes increase (1024→2048)** — Requires VRAM recalibration and re-extraction. Large-contract Timestamp F1 (0.364 for >150 nodes) suggests truncation is active. Target: Run 8 data pipeline.

5. **Timestamp size normalisation** — Cohen's d = 1.657 (positive vs negative). Size shortcut is confirmed active. Requires re-extraction with normalised cfg_count_norm feature. Target: Run 8 data pipeline.

6. **FUNCTION→FUNCTION call edges** — Currently CALL_ENTRY goes from caller CFG node to callee ENTRYPOINT CFG node, with no direct FUNCTION→FUNCTION edge. Adding this would give Phase 2 a direct gradient path to FUNCTION nodes. Requires graph_extractor.py change + re-extraction. Target: Run 8.
