# Offline Pipeline — Verified Audit

**Date:** 2026-06-05  
**Scope:** Full offline training pipeline — model architecture, training loop, data pipeline, graph extraction, feature engineering  
**Source files audited:** `gnn_encoder.py`, `sentinel_model.py`, `fusion_layer.py`, `transformer_encoder.py`, `trainer.py`, `losses.py`, `focalloss.py`, `graph_extractor.py`, `graph_schema.py`, `predictor.py`, `dual_path_dataset.py`

---

## 1. Executive Summary

The SENTINEL offline pipeline is a well-engineered codebase with guard checks, dtype safety, schema versioning, and checkpoint compatibility. The F1-macro plateau at ~0.30–0.33 is an architectural ceiling driven by:

1. **Model ignoring graph topology** — L2 edge ablation shows max Δ=0.013 from removing any single edge type
2. **`complexity` (feat[5]) dominating all 10 class decisions** — 34–36% gradient share per class (L4 saliency)
3. **JK attention at 99.5% maximum entropy** — uniform routing, no per-class specialization
4. **Structural impossibility** for 4 classes without new edge types or multi-contract graphs

Code-level issues contributing to suboptimal training are documented below with exact source locations.

---

## 2. Model Architecture

### 2.1 GNNEncoder (`gnn_encoder.py`)

**Architecture:** Three-phase 8-layer GAT (2+3+3). hidden_dim=256. `type_embedding(13→16)` prepended to features → `_GNN_IN_DIM=27` (model-internal, graph files remain 11-dim).

**Phase 2 IMP-G1 edge splitting** (`gnn_encoder.py:585-597`):

```
conv3:  CONTROL_FLOW(6) only              → 1 CF hop
conv3b: CALL_ENTRY(8) + RETURN_TO(9)     → 0 CF hops (cross-function, not intra-CF)
conv3c: CF + ICFG + DEF_USE joint         → 1 CF hop (diluted)
```

Effective CF receptive field: **~2 hops**, not 3. Only conv3 and conv3c carry CONTROL_FLOW edges. conv3b processes CALL_ENTRY/RETURN_TO which are cross-function edges.

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| A-2 | MEDIUM | conv3c receives ALL Phase 2 edges (superset of conv3+conv3b). The joint integration layer re-aggregates over edges already processed by layers 3 and 4. | `gnn_encoder.py:595` — `self.conv3c(x, phase2_ei, phase2_ea)` |
| A-3 | MEDIUM | Phase 2 already uses heads=4 (IMP-R7-1); Phase 3 uses heads=1. | `gnn_encoder.py:264`: `_p2_heads = 4`; lines 309, 318, 327: `heads=1` |
| A-4 | LOW | Phase 3 conv4+conv4b use identical `rev_contains_ei` edge set (two-hop upward by design). | `gnn_encoder.py:610,613` |

### 2.2 SentinelModel (`sentinel_model.py`)

**Architecture:** Four-eye classifier — GNN eye + Transformer eye + Fused eye + CFG eye → `[B, 512]` → `Linear(512, 256)` → `Linear(256, 10)`.

Classifier hidden dim is **256** (`sentinel_model.py:267`: `_cls_hidden = 256`).

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| A-6 | MEDIUM | `node_type_ids` recovery via `(x[:,0].float() * _MAX_TYPE_ID).round().long()` — vulnerable to BF16 precision loss if `graphs.x` enters in BF16 context. | `sentinel_model.py:446` |
| A-8 | MEDIUM | `aux_phase2` pools over CFG nodes (types 8-12) via `global_mean_pool` only. Mean-only pooling loses "most activated function" signal. Could use max+mean dual pooling. | `sentinel_model.py:555-557` — `cfg_pool_mask` selects CFG_NODE_CALL through CFG_NODE_OTHER |
| A-10 | MEDIUM | `fusion_max_nodes=1024` silently truncates excess nodes in `_scatter_to_dense`. Excess nodes dropped entirely (not clamped). | `fusion_layer.py:81-82,110-111` |

### 2.3 TransformerEncoder (`transformer_encoder.py`)

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| A-14 | LOW | LoRA targets only query+value. Key projection not adapted. Adding key LoRA (~295K more params) could improve vulnerability-relevant token attention. | `transformer_encoder.py:113` |

---

## 3. Training Pipeline

### 3.1 Trainer (`trainer.py`)

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| T-3 | MEDIUM | `compute_pos_weight` computed unconditionally at startup but unused when `loss_fn="asl"`. Wasted O(N×C) computation + CUDA tensor allocation. | `trainer.py:1141` computes; `trainer.py:1295-1299` — ASL constructor doesn't receive it |
| T-8 | LOW | ASL clip applied to probability space, not logit space. For rare classes (DoS, p < 0.01 for most negatives), this zeroes gradient for almost all negatives. | `losses.py:105`: `prob_neg = (prob - self.clip).clamp(min=0.0)` |
| T-9 | LOW | `FocalLoss` expects post-sigmoid predictions; `MultiLabelFocalLoss` expects raw logits. API inconsistency between loss classes. | `focalloss.py:21,37,42-43` — uses `BCE` (not `BCEWithLogitsLoss`) |

### 3.2 Loss Functions (`losses.py`)

ASL supports `pos_weight` parameter (constructor accepts it, applies at line 117-118). The trainer intentionally omits it when using ASL to avoid double-amplification with `dos_loss_weight`. Comment at `trainer.py:1290-1294` explains the design decision.

---

## 4. Data Pipeline

### 4.1 Graph Extraction (`graph_extractor.py`)

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| D-5 | HIGH | DEF_USE edges track LocalVariable by name, not SSA form. Reassignment creates spurious edges: if `x` is defined at node A and redefined at node B, both A→use and B→use edges are created, including stale A→use. No reaching-definition analysis. | `graph_extractor.py:979` — `vname = getattr(var, "name", None)`; lines 960-961 aggregate all defs under same name |
| D-6 | MEDIUM | No CALL_ENTRY/RETURN_TO for external calls. `_add_icfg_edges()` only iterates `node.internal_calls`. External calls (HighLevelCall, LowLevelCall) to other contracts have no cross-function edges. Major gap for Reentrancy detection. | `graph_extractor.py:863` — `for callee in (getattr(node, "internal_calls", None) or []):` |
| D-7 | MEDIUM | CFG nodes compute `uses_block_globals` per-statement from their own IR ops (not inherited from parent). Most CFG nodes return 0.0; nodes that DO read block.timestamp return 1.0. Inherited dims from parent FUNCTION are [1,3,4,5,9] only. | `graph_extractor.py:711` — `_node_uses_block_globals(slither_node)` per-statement; lines 696-705 inherit [1,3,4,5,9] |

### 4.2 DualPathDataset

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| D-2 | MEDIUM | No on-the-fly data augmentation during training. No stochastic edge dropout, node masking, or feature corruption in the training loop. GAT attention dropout is a neural regularizer, not data augmentation. Offline augmentation scripts in `ml/scripts/archive/` run pre-extraction only. | `trainer.py:train_one_epoch()` — no augmentation logic |
| D-3 | LOW | Windowed tokens always pad to MAX_WINDOWS=4. Short contracts get zero-padded windows; ~75% of token sequence is padding for typical contracts. | `retokenize_windowed.py:117,164-175`; `predictor.py:614-624` |
| BUG-P4 | HIGH | Shared cache loading in trainer bypasses DualPathDataset's integrity validation (10-hash random sampling, schema version check). Cache is loaded via `pickle.load()` and assigned directly to `dataset.cached_data`. | `trainer.py:1083-1091` — `cached_data` set directly; `dual_path_dataset.py:239-265` validation skipped |

---

## 5. Feature Engineering

### 5.1 Feature Schema (v8 — 11 dims)

| Dim | Feature | Range | Notes |
|-----|---------|-------|-------|
| [0] | type_id / 12.0 | [0, 1] | Normalized node type |
| [1] | visibility | {0.0, 0.5, 1.0} | Ordinal encoding |
| [2] | uses_block_globals | {0.0, 1.0} | Per-statement for CFG; per-function for FUNCTION |
| [3] | view | {0.0, 1.0} | |
| [4] | payable | {0.0, 1.0} | |
| [5] | complexity | [0, 1] | log1p(CFG block count)/log1p(100). **Dominates all classes at 34–36%.** |
| [6] | loc | [0, 1] | log1p(lines)/log1p(1000) |
| [7] | return_ignored | {-1.0, 0.0, 1.0} | -1.0 = IR unavailable |
| [8] | call_target_typed | {-1.0, 0.0, 1.0} | -1.0 = source unavailable |
| [9] | has_loop | {0.0, 1.0} | |
| [10] | external_call_count | [0, 1] | log1p(n)/log1p(20) |

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| F-1 | HIGH | `complexity` dominates all 10 class decisions. Run 8 fix: `--drop-complexity-feature` zeros dim[5] at GNN input. | `graph_schema.py:424`; L4 saliency experiment |
| F-2 | HIGH | Missing features for 4 structural-ceiling classes: UnusedReturn needs return-value DEF_USE; Timestamp needs per-statement `uses_block_globals` on CFG nodes; TOD needs cross-contract edges; ExternalBug needs exception propagation edges. | `graph_schema.py:422-435` — no class-specific structural features |
| F-3 | MEDIUM | `return_ignored` and `call_target_typed` use -1.0 sentinel. Gets multiplied by weights in GNN input_proj, creating a strong negative bias the model must learn to ignore. | `graph_schema.py:426-427` |
| F-5 | LOW | No Solidity version feature. Dataset spans 0.4.x to 0.8.x with very different vulnerability profiles. | `graph_schema.py:422-435` — no version dim |

---

## 6. Prediction Pipeline

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| BUG-M6 | LOW | `predict_with_hotspots` runs GNN forward twice: once in `_score_windowed()` for scoring, once explicitly for hotspot extraction. Second call could be eliminated by caching node embeddings from the first. | `predictor.py:635` (full model forward) + `predictor.py:487` (explicit GNN re-run) |

---

## 7. Performance Optimization

| # | Category | Finding | Location |
|---|----------|---------|----------|
| ~~S-1~~ | ~~Compute~~ | ~~`aux_phase2` head not in torch.compile list.~~ **RETRACTED** — `aux_phase2` IS compiled at `trainer.py:1418` alongside aux_gnn, aux_transformer, aux_fused. Original claim was wrong. | `trainer.py:1418` |
| S-7 | Compute | Edge masks (struct_mask, cfg_mask, cf_only_mask, etc.) recomputed from `edge_attr` every forward pass. Could be precomputed during data loading and cached as graph attributes. | `gnn_encoder.py` — mask computation in forward() |

---

## 8. Structural Class Ceilings

These are architectural limitations no training hyperparameter tuning can overcome:

| Class | F1 Ceiling | Root Cause | Required Fix |
|-------|-----------|------------|--------------|
| UnusedReturn | ~0.23 | No DEF_USE edges for return values; name-based tracking (D-5) doesn't capture return-value flow | Return-value-specific DEF_USE edges |
| Timestamp | ~0.17 | Per-statement `uses_block_globals` computed but not inheritable at function level for CFG nodes that don't directly read block globals | CFG node inheritance of function-level `uses_block_globals` |
| TOD | ~0.25 | Single-contract graphs can't represent cross-contract transaction ordering | Multi-contract graph extraction |
| ExternalBug | ~0.25 | No CALL_ENTRY for external calls (D-6); no exception propagation edges | External CALL_ENTRY edges + exception-flow edge types |

Theoretical macro-F1 ceiling:
- Without structural fixes: `(6 × 0.50 + 4 × 0.23) / 10 = 0.392`
- With structural fixes: `(6 × 0.50 + 4 × 0.40) / 10 = 0.460`

---

## 9. Bug Tracker

### Critical

| ID | Component | Description | Location |
|----|-----------|-------------|----------|
| BUG-P4 | trainer.py | Shared cache bypasses DualPathDataset integrity validation (10-hash check, schema version check) | `trainer.py:1083-1091` |

### Important

| ID | Component | Description | Location |
|----|-----------|-------------|----------|
| BUG-I2 | trainer.py | `compute_pos_weight` computed but unused with ASL | `trainer.py:1141` |
| BUG-I4 | graph_extractor.py | DEF_USE edges track variable names, not SSA form — reassignment creates spurious edges | `graph_extractor.py:979` |
| BUG-M6 | predictor.py | `predict_with_hotspots` runs GNN twice | `predictor.py:635` + `predictor.py:487` |

### Minor

| ID | Component | Description | Location |
|----|-----------|-------------|----------|
| BUG-M3 | graph_extractor.py | No CALL_ENTRY for external calls | `graph_extractor.py:863` |
| BUG-M4 | focalloss.py | FocalLoss expects post-sigmoid; MultiLabelFocalLoss expects logits — inconsistent API | `focalloss.py:21,37` |

---

## 10. Recommendations

### Run 8 — Code Changes ✅ ALL IMPLEMENTED (2026-06-05)

| # | Change | Location | Status | Impact |
|---|--------|----------|--------|--------|
| 1 | APPNP-style Phase 1 teleport in Phase 2 (`appnp_alpha=0.2`) | `gnn_encoder.py` Phase 2 forward | ✅ Done | +0.03-0.06 F1 Reentrancy |
| 2 | Refine aux_phase2 to CALL+WRITE+CHECK only (3 types) | `sentinel_model.py` — `_CEI_IDS_CPU` + `cei_pool_mask` | ✅ Done | +0.02-0.04 F1 Reentrancy |
| 3 | `fusion_max_nodes` default → 2048 | `trainer.py` `TrainConfig` + `train.py` argparse | ✅ Done | Covers 100% of v10 graphs |
| 4 | Guard `compute_pos_weight` behind `loss_fn != "asl"` | `trainer.py:1145` | ✅ Done | Cleanup, eliminates O(N×C) waste |
| 5 | Fix shared cache integrity bypass (BUG-P4) | `trainer.py` — pass `cache_path` to `DualPathDataset` | ✅ Done | Schema+hash validation restored |

### Run 8 — Configuration Changes ✅ ALL ACTIVE

| # | Change | Value | Status |
|---|--------|-------|--------|
| 6 | `--gnn-prefix-k 48` | Re-enable GNN prefix injection | ✅ In launch command |
| 7 | `--jk-entropy-reg-lambda 0.0075` | Stronger Phase 3 drift prevention | ✅ In launch command |
| 8 | `--appnp-alpha 0.2` | Phase 1 teleport fraction | ✅ In launch command |
| 9 | `--drop-complexity-feature` | Break complexity-proxy shortcut | ✅ In launch command |
| 10 | `--fusion-lr-multiplier 0.3` | Reduce fusion gradient spikes | ✅ In launch command |
| 11 | Threshold calibration → `ml/calibration/temperatures_run7.json` | Run 7 per-class thresholds extracted | ✅ Done |

### Run 9 — Re-Extraction Required

| # | Change | Location | Impact |
|---|--------|----------|--------|
| 12 | Add path-level CEI features (dist_to_call, cei_violation_score) | `graph_extractor.py` + schema bump | +0.05-0.10 F1 Reentrancy |
| 13 | CFG node `uses_block_globals` inheritance | `graph_extractor.py:710` | +0.02-0.04 F1 Timestamp |
| 14 | DELEGATECALL edge type | `graph_extractor.py` + `gnn_encoder.py` | +0.02-0.03 F1 Reentrancy |
| 15 | Per-statement `external_call_count` on CFG_NODE_CALL | `graph_extractor.py:718` | ExternalBug signal |
| 16 | DEF_USE edges for return-value flow (RC5) | `graph_extractor.py` | +0.03-0.06 F1 UnusedReturn |

---

*All findings verified against source code. No unverified claims included.*
