# SENTINEL Run 7 — Actionable Implementation Plan

**Companion to:** SENTINEL-Run7-Final-Fix-Proposal.md  
**Date:** 2026-06-03  
**Purpose:** Step-by-step checklist for implementing all 5 Run 7 fixes, with exact file locations, line references, code snippets, verification steps, and rollback procedures.

---

## Quick Reference: The 5 Fixes

| # | ID | File(s) | What | Est. Time |
|---|-----|---------|------|-----------|
| 1 | BUG-R7-1 | sentinel_model.py:521 | Pool CFG nodes in aux_phase2 instead of FUNCTION nodes | 10 min |
| 2 | BUG-R7-2 | gnn_encoder.py | Replace scalar type_id with nn.Embedding(13, 16) | 30 min |
| 3 | IMP-R7-1 | gnn_encoder.py | Phase 2 heads=1→4, concat=True, out=64/head | 15 min |
| 4 | IMP-R7-2 | sentinel_model.py | Add CFG eye (4th eye) + widen classifier | 45 min |
| 5 | IMP-R7-3 | trainer.py, train.py | aux_phase2_loss_weight 0.10→0.20 | 5 min |

**Total estimated implementation time: ~2 hours**  
**Total estimated smoke-test + verification time: ~1 hour**

---

## Pre-Flight Checklist

Before touching any code:

- [ ] **Backup current source.** Copy `upload/src_ext/src/` to `upload/src_ext/src_run6_backup/`
- [ ] **Verify Run 6 is still training** (or has completed). Do not interfere with an active run.
- [ ] **Confirm data paths are stable:** `ml/data/splits/v10_deduped/` and `ml/data/cached_dataset_v10.pkl` exist
- [ ] **Python environment is active:** `source ml/.venv/bin/activate`
- [ ] **GPU is free:** `nvidia-smi` shows < 1 GiB used
- [ ] **Git state is clean:** `git status` shows no uncommitted changes (or stash them)

---

## Step 1: BUG-R7-1 — Fix aux_phase2 Pooling

**File:** `upload/src_ext/src/models/sentinel_model.py`  
**Target line:** ~521  
**Risk:** LOW  
**Depends on:** Nothing (first fix)

### 1a. Replace the pooling logic

**Find** (around line 520–522):
```python
# Phase 2 CEI aux head: pool phase2 embeddings over function nodes
phase2_pooled    = global_mean_pool(_phase2_x[pool_mask], pool_batch, size=num_graphs)
aux_phase2_logits = self.aux_phase2(phase2_pooled)   # [B, num_classes]
```

**Replace with:**
```python
# Phase 2 CEI aux head: pool phase2 embeddings over CFG_NODE_* nodes (types 8–12).
# FUNCTION nodes are excluded — they receive zero Phase 2 messages and pooling them
# would route gradient to Phase 1 parameters instead of Phase 2 conv layers.
_CFG_NODE_TYPES = torch.tensor([8, 9, 10, 11, 12], device=node_embs.device)
cfg_node_mask   = torch.isin(node_type_ids, _CFG_NODE_TYPES)
if cfg_node_mask.any():
    phase2_pooled = global_mean_pool(
        _phase2_x[cfg_node_mask], batch[cfg_node_mask], size=num_graphs
    )
else:
    phase2_pooled = torch.zeros(
        num_graphs, self.gnn.hidden_dim,
        device=node_embs.device, dtype=_phase2_x.dtype
    )
aux_phase2_logits = self.aux_phase2(phase2_pooled)   # [B, num_classes]
```

### 1b. Verify

- [ ] `node_type_ids` is already computed at line ~425 — confirm it is in scope
- [ ] `_phase2_x` has shape `[N, 256]` where N includes CFG nodes
- [ ] No import changes needed — `torch` and `global_mean_pool` are already imported
- [ ] The `_CFG_NODE_TYPES` tensor is only allocated inside the `return_aux=True` branch (training only, no inference cost)

### 1c. Unit test (quick)

```python
# In a Python shell with the model loaded:
# 1. Create a mini-batch with known node types
# 2. Forward with return_aux=True
# 3. Check that aux_phase2_logits is non-zero
# 4. Check that grad of conv3.lin_l.weight is non-zero when backprop through aux_phase2 loss
```

### 1d. Rollback

If something goes wrong, revert the change — restore the original 3 lines. The old behavior (broken gradient) is what Run 6 had, so you can always fall back.

---

## Step 2: BUG-R7-2 — Add Node Type Embedding

**File:** `upload/src_ext/src/models/gnn_encoder.py`  
**Risk:** MEDIUM  
**Depends on:** Nothing (can be parallel with Step 1)

### 2a. Add constants after imports (top of file, before class definition)

```python
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES, EDGE_TYPES, NODE_TYPES

_NODE_TYPE_EMB_DIM: int = 16
_GNN_MAX_TYPE_ID:   float = float(max(NODE_TYPES.values()))
assert _GNN_MAX_TYPE_ID == 12.0, (
    f"_GNN_MAX_TYPE_ID={_GNN_MAX_TYPE_ID} expected 12.0. "
    "A node type was added — update gnn_encoder.py node_type_embedding size."
)
_GNN_EFFECTIVE_INPUT_DIM: int = NODE_FEATURE_DIM - 1 + _NODE_TYPE_EMB_DIM  # 26
```

### 2b. Update GNNEncoder.__init__

**Add the embedding layer** (after `self.edge_embedding` block):
```python
# Node type embedding: replaces scalar type_id/12.0 with learnable 16-dim vector.
self.node_type_embedding = nn.Embedding(13, _NODE_TYPE_EMB_DIM)
```

**Update input_proj** — find:
```python
self.input_proj = nn.Linear(NODE_FEATURE_DIM, hidden_dim, bias=False)
```
Replace with:
```python
self.input_proj = nn.Linear(_GNN_EFFECTIVE_INPUT_DIM, hidden_dim, bias=False)
```

**Update conv1** — find:
```python
self.conv1 = GATConv(
    in_channels=NODE_FEATURE_DIM,
    out_channels=_head_dim,
    ...
)
```
Replace `in_channels=NODE_FEATURE_DIM` with:
```python
in_channels=_GNN_EFFECTIVE_INPUT_DIM,   # 26 (was 11)
```

### 2c. Update GNNEncoder.forward()

Find the shape validation check and the dtype cast. **Between the dtype cast and `x_init = x`**, insert:

```python
# BUG-R7-2: recover integer type_id, embed it, replace scalar dim 0.
_type_id_int = (x[:, 0].float() * _GNN_MAX_TYPE_ID).round().clamp(0, 12).long()
_type_emb    = self.node_type_embedding(_type_id_int).to(self._param_dtype)
x = torch.cat([x[:, 1:], _type_emb], dim=-1)   # [N, 10+16=26]
```

**Important:** The dtype cast must happen BEFORE the embedding. The correct order is:
1. Shape validation check (unchanged)
2. dtype cast: `if x.dtype != self._param_dtype: x = x.to(self._param_dtype)`
3. Type embedding enrichment (new code above)
4. `x_init = x` (existing IMP-G2 skip capture)

### 2d. Update parameter_summary() if it references input dimensions

Find any string that says `in=11` or `NODE_FEATURE_DIM` in parameter_summary() output and update to reflect `_GNN_EFFECTIVE_INPUT_DIM=26`.

### 2e. Verify

- [ ] `_GNN_EFFECTIVE_INPUT_DIM = 26` (11 - 1 + 16)
- [ ] `conv1` accepts 26-dim input
- [ ] `input_proj` accepts 26-dim input
- [ ] After enrichment, `x.shape[1] == 26` before `x_init = x`
- [ ] `conv2` input is still `hidden_dim=256` (unchanged — it receives Phase 1 output, not raw features)
- [ ] Phase 2 convs receive `hidden_dim=256` (unchanged)
- [ ] `validate_graph_integrity` check uses `x.shape[0]` (node count), not `x.shape[1]` — unaffected

### 2f. Rollback

Revert: remove the embedding, restore `NODE_FEATURE_DIM` in conv1/input_proj, remove the enrichment block in forward(). The disk format (.pt files) is unchanged.

---

## Step 3: IMP-R7-1 — Phase 2 Heads 1→4

**File:** `upload/src_ext/src/models/gnn_encoder.py` (same file as Step 2)  
**Risk:** LOW  
**Depends on:** BUG-R7-1 should be applied first (otherwise the extra heads don't get gradient)

### 3a. Update conv3, conv3b, conv3c definitions

For each of the three Phase 2 convs (`conv3`, `conv3b`, `conv3c`), change:

```python
# FROM:
self.conv3 = GATConv(
    in_channels=hidden_dim,
    out_channels=hidden_dim,
    heads=1,
    concat=False,
    add_self_loops=False,
    edge_dim=_edge_dim,
)

# TO:
_ph2_out_per_head: int = hidden_dim // 4   # 64
self.conv3 = GATConv(
    in_channels=hidden_dim,
    out_channels=_ph2_out_per_head,   # 64 (was 256)
    heads=4,                           # (was 1)
    concat=True,                       # (was False) — 4×64=256 = hidden_dim
    add_self_loops=False,
    edge_dim=_edge_dim,
)
```

Apply the same change to `conv3b` and `conv3c`.

### 3b. Verify

- [ ] Output shape: `4 × 64 = 256 = hidden_dim` — residual connections `x = x + dropout(x2)` still work
- [ ] Parameter count: `4 × (256 × 64) = 65,536` per conv — identical to `1 × (256 × 256) = 65,536`
- [ ] `add_self_loops=False` is preserved (CRITICAL — self-loops cancel directional CF signal)
- [ ] No other convs are affected (conv4/conv4b/conv4c in Phase 3 remain heads=1)

### 3c. Rollback

Revert: change back to `heads=1, concat=False, out_channels=hidden_dim` for all three convs.

---

## Step 4: IMP-R7-2 — Add CFG Eye (4th Eye)

**File:** `upload/src_ext/src/models/sentinel_model.py`  
**Risk:** MEDIUM  
**Depends on:** Steps 1–3 should be applied first

### 4a. Update SentinelModel.__init__

**Add cfg_eye_proj** (after `transformer_eye_proj`):
```python
# CFG eye: direct pool of Phase 2 output over CFG_NODE_* nodes.
self.cfg_eye_proj = nn.Sequential(
    nn.Linear(2 * gnn_hidden_dim, eye_dim),   # 512 → 128
    nn.ReLU(),
    nn.Dropout(dropout),
)
```

**Update classifier** — find:
```python
_cls_hidden = 192
self.classifier = nn.Sequential(
    nn.Linear(3 * eye_dim, _cls_hidden),
    ...
)
```
Replace with:
```python
_cls_input  = 4 * eye_dim    # 512 (was 384)
_cls_hidden = 256             # increased from 192
self.classifier = nn.Sequential(
    nn.Linear(_cls_input, _cls_hidden),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(_cls_hidden, num_classes),
)
```

### 4b. Update SentinelModel.forward()

**Change GNN call to always return _phase2_x** — find:
```python
_gnn_out = self.gnn(
    graphs.x, graphs.edge_index, graphs.batch, edge_attr,
    return_phase2_embs=return_aux,
)
if return_aux:
    node_embs, batch, _jk_entropy, _phase2_x = _gnn_out
else:
    node_embs, batch, _jk_entropy = _gnn_out
```
Replace with:
```python
_gnn_out = self.gnn(
    graphs.x, graphs.edge_index, graphs.batch, edge_attr,
    return_phase2_embs=True,   # always needed for cfg_eye
)
node_embs, batch, _jk_entropy, _phase2_x = _gnn_out
```

**Add CFG eye computation** (after `func_mask` and `node_type_ids` are computed, after the GNN eye, before the main classifier):
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
    p2_max = p2_mean = torch.zeros(
        num_graphs, self.gnn.hidden_dim,
        device=node_embs.device, dtype=_phase2_x.dtype
    )
cfg_eye = self.cfg_eye_proj(torch.cat([p2_max, p2_mean], dim=1))   # [B, 128]
```

**Update classifier input** — find:
```python
combined = torch.cat([gnn_eye, transformer_eye, fused_eye], dim=1)  # [B, 3*eye_dim]
```
Replace with:
```python
combined = torch.cat([gnn_eye, transformer_eye, fused_eye, cfg_eye], dim=1)  # [B, 4*eye_dim]
```

**Update empty-batch guard** — find the `if batch.numel() == 0:` block and ensure `cfg_eye` would be zeroed. No change needed if the guard returns before cfg_eye is computed — but verify the logic still short-circuits correctly.

### 4c. Update parameter_summary()

Add `cfg_eye_proj` to the summary output if the method lists eye projections.

### 4d. Update class docstring

Update the top-of-file docstring to reflect the 4-eye architecture:
- Change "Three-eye" → "Four-eye"
- Change `[B, 384]` → `[B, 512]`
- Add CFG eye description
- Update classifier description: `3*eye_dim` → `4*eye_dim`, `_cls_hidden=192` → `_cls_hidden=256`

### 4e. Verify

- [ ] `combined.shape == [B, 512]` (4 × 128)
- [ ] `logits.shape == [B, num_classes]` (10)
- [ ] `cfg_eye_proj` appears in `model.parameters()` count
- [ ] `_phase2_x` is available even when `return_aux=False` (inference path)
- [ ] Ghost-graph guard: contracts with no CFG nodes produce zero cfg_eye
- [ ] No import errors — `global_max_pool` is already imported

### 4f. Rollback

Revert: remove `cfg_eye_proj`, restore `3 * eye_dim` in classifier, restore `return_phase2_embs=return_aux`, remove cfg_eye computation and concat. This is the most complex rollback — take care.

---

## Step 5: IMP-R7-3 — Increase aux_phase2_loss_weight

**File 1:** `upload/src_ext/src/training/trainer.py`  
**File 2:** `upload/fullfiles_extracted/scripts/train.py`  
**Risk:** LOW  
**Depends on:** BUG-R7-1 must be verified first

### 5a. Update trainer.py

Find the TrainConfig or equivalent dataclass:
```python
aux_phase2_loss_weight: float = 0.10
```
Change to:
```python
aux_phase2_loss_weight: float = 0.20   # was 0.10; now meaningful (BUG-R7-1 fixed)
```

### 5b. Update train.py CLI default

Find:
```python
p.add_argument("--aux-phase2-loss-weight", type=float, default=0.10, ...)
```
Change to:
```python
p.add_argument("--aux-phase2-loss-weight", type=float, default=0.20, ...)
```

### 5c. Verify

- [ ] `--aux-phase2-loss-weight 0.20` appears in training logs on first epoch
- [ ] The old default (0.10) is no longer used

### 5d. Rollback

Revert: change both values back to 0.10.

---

## Step 6: Smoke Test

After all 5 fixes are applied, run the smoke test:

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
  --run-name GCB-P1-Run7-smoke-$(date +%Y%m%d) \
  --experiment-name sentinel-multilabel \
  --epochs 2 \
  --smoke-subsample-fraction 0.1
```

### Gate Checklist

Run through each gate during the smoke test:

| Gate | How to Check | Pass Criterion |
|------|-------------|----------------|
| G1 — No NaN | Check training log for `loss=nan` | Zero NaN losses in first 50 steps |
| G2 — GNN share | Check `gnn_grad_share` metric | ≥ 15% by step 100 of epoch 1 |
| G3 — Ph2/Ph1 ratio | Check `phase2_phase1_grad_ratio` metric | > 0.25 by step 200 |
| G4 — CFG eye gradient | Check `cfg_eye_proj` grad norm in log | > 0 at step 100 |
| G5 — JK balance | Check `jk_phase_weights` metric | No phase weight < 0.10 |
| G6 — VRAM | Check `nvidia-smi` during epoch 1 | < 7.0/8.0 GiB peak |
| G7 — Shape sanity | Check model summary or add print | `combined` shape `[B, 512]`, logits `[B, 10]` |

### If a Gate Fails

| Gate | Failure | Diagnostic |
|------|---------|-----------|
| G1 | NaN losses | Check BUG-R7-2 dtype ordering; add `torch.autograd.detect_anomaly()` |
| G2 | GNN share < 15% | Check `gnn_lr_multiplier` is still 2.5; check Phase 2 heads change |
| G3 | Ph2/Ph1 ratio ≤ 0.25 | BUG-R7-1 not applied correctly — verify aux_phase2 pools CFG nodes |
| G4 | CFG eye grad = 0 | IMP-R7-2 cfg_eye computation is broken — check cfg_node_mask.any() |
| G5 | JK phase weight < 0.10 | `jk_entropy_reg_lambda` was accidentally changed — check it's 0.005 |
| G6 | VRAM > 7.0 GiB | Reduce `batch_size` from 8 to 6 or 4 |
| G7 | Wrong shapes | Classifier not rebuilt — check `_cls_input = 4 * eye_dim` |

### One-Off Gradient Verification

After smoke epoch 1, add a temporary hook to verify the BUG-R7-1 fix:

```python
# In the training loop, after aux_phase2_loss.backward():
# Check that conv3.lin_l.weight.grad is non-zero from aux_phase2 specifically.
# This requires a separate backward pass with retain_graph=True on just the aux loss.
# If grad is zero, BUG-R7-1 is not working.
```

Alternatively, check the training log for `phase2_phase1_grad_ratio` — if it increases from the Run 6 baseline (0.15–0.20) to > 0.40 by epoch 2, BUG-R7-1 is working.

---

## Step 7: Launch Full Run 7

Only after all smoke test gates pass:

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
  --run-name GCB-P1-Run7-v10-$(date +%Y%m%d) \
  --experiment-name sentinel-multilabel \
  --epochs 100 \
  --aux-phase2-loss-weight 0.20
```

### During Full Run — Monitor These Metrics

| Metric | Expected Trend | Alert Threshold |
|--------|---------------|-----------------|
| `macro_f1` (val) | Rising steadily to 0.40+ by epoch 30 | Stalls below 0.32 at epoch 20 |
| `phase2_phase1_grad_ratio` | 0.40–0.80 | Stays below 0.25 (BUG-R7-1 failed) |
| `jk_phase_weights[1]` (Phase 2) | Increasing from 0.32 toward 0.35+ | Drops below 0.20 |
| `aux_phase2_loss` | Decreasing after warmup | Increasing or NaN |
| `cfg_eye_proj` grad norm | Non-zero throughout | Zero for > 5 consecutive steps |
| VRAM | Stable at ~6–7 GiB | Spikes above 7.5 GiB |

### Early Stopping Criteria

If any of these occur in the first 10 epochs, **stop the run and investigate**:

1. `macro_f1` is consistently below 0.25 by epoch 10 (worse than Run 6 pace)
2. `phase2_phase1_grad_ratio` stays below 0.20 (BUG-R7-1 not effective)
3. NaN losses appear (BUG-R7-2 dtype issue)
4. VRAM exceeds 7.5 GiB consistently (need to reduce batch_size)

---

## File Change Summary

| File | Changes | Lines Modified (est.) |
|------|---------|----------------------|
| `src/models/sentinel_model.py` | BUG-R7-1 (pool fix), IMP-R7-2 (CFG eye + classifier), docstring update | ~40 |
| `src/models/gnn_encoder.py` | BUG-R7-2 (type embedding), IMP-R7-1 (heads), constants | ~30 |
| `src/training/trainer.py` | IMP-R7-3 (weight default) | 1 |
| `scripts/train.py` | IMP-R7-3 (CLI default) | 1 |

**Total lines changed: ~72 across 4 files**

---

## Caution Notes

1. **Do NOT change the disk format.** All .pt graph files must remain at `NODE_FEATURE_DIM=11`. The type_id embedding enrichment happens inside `GNNEncoder.forward()` — the disk representation is unchanged.

2. **Do NOT modify Phase 3 convs.** `conv4`, `conv4b`, `conv4c` remain at `heads=1, concat=False`. Phase 3 processes REVERSE_CONTAINS/CONTAINS edges which have a simpler structure (up/down hierarchy) that doesn't benefit from multi-head attention in the same way.

3. **Do NOT add new edge types.** The CALL_ENTRY→FUNCTION direct edge (proposed as future work) requires graph_extractor.py changes and re-extraction. Not part of Run 7.

4. **Do NOT change the warmup schedule.** The existing 8-epoch prefix warmup and aux loss warmup are calibrated for the current architecture. Changing them alongside the 5 fixes would create confounding variables.

5. **Checkpoint incompatibility is expected.** Run 7 trains from scratch. Do not attempt to load a Run 6 checkpoint — the model dimensions have changed (input_proj, conv1, classifier, cfg_eye_proj are all different sizes).

6. **The `_CFG_NODE_TYPES` tensor in BUG-R7-1 and IMP-R7-2 uses the same node type IDs [8, 9, 10, 11, 12].** These correspond to CFG_NODE_CALL, CFG_NODE_BRANCH, CFG_NODE_RETURN, CFG_NODE_OTHER, ENTRYPOINT per graph_schema.py. If the schema changes, both locations must be updated.

---

## Rollback Procedure (Full)

If the entire Run 7 changeset needs to be reverted:

```bash
# Restore from backup
cp -r upload/src_ext/src_run6_backup/* upload/src_ext/src/

# Verify
python -c "from ml.src.models.sentinel_model import SentinelModel; print('Import OK')"
```

This restores the exact Run 6 source state. No .pt files were modified during Run 7 implementation.

---

## Completion Criteria

Run 7 is considered successful if:

- [ ] All 5 smoke test gates pass
- [ ] Training completes at least 30 epochs without early-stop trigger
- [ ] `macro_f1` (val) reaches ≥ 0.40 at any epoch
- [ ] `phase2_phase1_grad_ratio` stabilizes above 0.40
- [ ] Reentrancy F1 improves by at least +0.05 over Run 6 best
- [ ] No NaN losses, no VRAM crashes, no shape errors

If `macro_f1` peaks below 0.40 but above 0.36, the run is still informative — it confirms the architecture changes help but the data ceiling (label co-occurrence, EMITS bug) is the next bottleneck. In that case, proceed to the Post-Run 7 Roadmap items.

If `macro_f1` peaks below 0.36 (worse than Run 6), investigate whether a regression was introduced — check the smoke test gate logs and compare gradient flow metrics against Run 6 baseline.
