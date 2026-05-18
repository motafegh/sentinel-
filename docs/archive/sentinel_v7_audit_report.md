# Sentinel v7.0 Pre-Training Hostile Audit Report

**Codebase:** 28ordi-11pm (commit 46a8f9d)  
**Date:** 2026-05-18  
**Scope:** Full ML source code audit before v7.0 training launch

---

## Executive Summary

The codebase has undergone significant improvements since the v6.0 collapse (F1=0.1717). The schema has been upgraded from v6 (12-dim) to **v7 (11-dim)** by removing the `in_unchecked` feature (87.9% of Solidity 0.4.x contracts predate `unchecked{}`). DoS augmentation infrastructure is in place. ASL parameters have been tuned (gamma_neg: 4→2, clip: 0.05→0.01) to prevent collapse.

**However, this audit found 6 CRITICAL issues that must be fixed before v7.0 training, or the training run will either fail or produce unreliable results.**

---

## Severity Classification

| Severity | Definition | Count |
|----------|-----------|-------|
| CRITICAL | Will cause training failure, data corruption, or silently wrong results | 6 |
| FIND | Bug or inconsistency that produces wrong behavior under specific conditions | 7 |
| WARN | Maintenance trap, stale documentation, or suboptimal behavior | 15 |

---

## CRITICAL Findings (Must Fix Before Training)

### C1: create_cache.py defaults to wrong token directory

**File:** `scripts/create_cache.py` line 144  
**Issue:** Default `--tokens-dir` is `ml/data/tokens` (old single-window format), but the entire pipeline now uses `ml/data/tokens_windowed` (sliding window [4,512] format). Running `create_cache.py` with defaults will cache tokens with shape `[512]` instead of `[4, 512]`, causing a shape mismatch crash at training time.

**Fix:** Change default from `"ml/data/tokens"` to `"ml/data/tokens_windowed"`.

---

### C2: DoS gradient leaks through auxiliary loss heads

**File:** `src/training/trainer.py` lines 534-536  
**Issue:** The main loss correctly detaches the DoS column when `dos_loss_weight < 1.0`, but all three auxiliary classification heads (`gnn`, `transformer`, `fused`) receive the full `labels` tensor including DoS. With `aux_loss_weight=0.3` and three heads, approximately **47% of effective DoS gradient** bypasses the mask. If DoS is "structurally unlearnable" (BUG-H6 rationale), this leaked gradient is harmful and may teach the aux heads spurious DoS↔Reentrancy correlations that propagate to the main classifier.

**Fix:** Apply the same DoS masking to auxiliary logits:
```python
if dos_loss_weight < 1.0:
    for key in aux:
        _aux = aux[key].clone()
        _aux[:, _dos_idx] = aux[key][:, _dos_idx].detach()
        aux[key] = _aux
```

---

### C3: train.py CLI `--gnn-layers` default disagrees with TrainConfig

**File:** `scripts/train.py` line 154 vs `src/training/trainer.py` line 171  
**Issue:** CLI defaults `--gnn-layers` to **6**, but `TrainConfig` defaults it to **7** (updated for BUG-H1 Phase 2 having 3 conv layers: 2+3+2=7). Running `python scripts/train.py` produces a **6-layer GNN** (missing one Phase 2 layer), while constructing `TrainConfig()` directly produces a 7-layer GNN.

**Fix:** Change `scripts/train.py` line 154 from `default=6` to `default=7`.

---

### C4: `dos_loss_weight` not exposed via CLI

**File:** `scripts/train.py`  
**Issue:** `TrainConfig.dos_loss_weight=0.0` controls DoS gradient masking (BUG-H6). This is critical for enabling DoS training after augmentation. **No CLI flag exists.** Users cannot re-enable DoS gradients without editing source code. After the DoS augmentation pipeline is run, the user will need to set `dos_loss_weight=1.0` but has no way to do so from the command line.

**Fix:** Add to `train.py`:
```python
p.add_argument("--dos-loss-weight", type=float, default=0.0,
               help="DoS gradient weight (0.0=masked, 1.0=normal)")
```

---

### C5: Train-inference tokenization skew

**File:** `src/data_extraction/tokenizer.py` vs `src/inference/preprocess.py`  
**Issue:** The offline tokenizer (`tokenizer.py`) produces **single-window** tokens (truncated at 512). The inference pipeline (`preprocess.py`) uses **sliding windows** (stride=256, up to 8 windows). The model never trains on tokens beyond position ~510 of long contracts, but at inference time those tokens are visible. Functions defined late in long contracts (withdrawal patterns, access control, DoS loops in large contracts) are invisible during training but visible during inference.

**Status:** If `retokenize_windowed.py` has been run to produce `ml/data/tokens_windowed/`, this is partially mitigated — training will use windowed tokens. But the offline `tokenizer.py` still produces single-window output for any future extractions.

**Fix:** After running `retokenize_windowed.py`, verify that `create_cache.py` uses `--tokens-dir ml/data/tokens_windowed` (see C1). Long-term: make windowed tokenization the default in `tokenizer.py`.

---

### C6: No runtime feature-dimension validation in dataset or model

**File:** `src/datasets/dual_path_dataset.py`, `src/models/gnn_encoder.py`  
**Issue:** If any stale `.pt` file still has 12-dim features (from v6 schema), `torch.load` succeeds but `GATConv` crashes with a cryptic matmul shape error. Neither `DualPathDataset.__getitem__()` nor `GNNEncoder.forward()` validates `graph.x.shape[1] == NODE_FEATURE_DIM`. The `validate_graph_dataset.py` script exists but is offline-only.

**Fix:** Add a shape guard in `GNNEncoder.forward()`:
```python
if x.shape[1] != NODE_FEATURE_DIM:
    raise ValueError(f"Expected x with {NODE_FEATURE_DIM} features, got {x.shape[1]}")
```

---

## FIND-Level Issues

### F1: `gnn_layers` actual count is 7, docstrings say 6
**File:** `src/models/gnn_encoder.py`  
Phase 2 has 3 conv layers (conv3, conv3b, conv3c from BUG-H1 fix), not 2. Total = 2+3+2 = 7. Multiple docstrings and comments say "6-layer". `num_layers` default=6 in the encoder is wrong. Code functions correctly at runtime (all 7 layers execute), but any code relying on `num_layers` for checkpoint validation will be incorrect.

### F2: 6 docstrings in graph_extractor.py say 12 dims, code returns 11
**File:** `src/preprocessing/graph_extractor.py`  
The v6→v7 change (removing `in_unchecked`) reduced `NODE_FEATURE_DIM` from 12 to 11, but docstrings at lines 36, 490, 492, 632, 635, 878 still reference "12". Runtime assertions catch actual shape mismatches, but this is a serious maintenance trap.

### F3: `_compute_in_unchecked` is dead code
**File:** `src/preprocessing/graph_extractor.py` lines 318-344, 681, 699  
The function is invoked on every Function node but its return value is never used (not included in the 11-dim feature vector). Wastes CPU and misleads maintainers.

### F4: `weighted-sampler` CLI missing "positive" mode
**File:** `scripts/train.py` line 177  
CLI offers `["none", "DoS-only", "all-rare"]`, but `TrainConfig` default is `"positive"` (3× weight on any-vuln rows, BUG-H10 fix). The most common sampling mode is unreachable from CLI.

### F5: Hash format split between offline and online pipelines
**File:** `src/utils/hash_utils.py`, `src/inference/preprocess.py`  
Offline pipeline: `abc123` (32-char MD5). Online/inference: `abc123_v7` (35-char with schema suffix). `validate_hash()` rejects the 35-char format. Cross-referencing between training and inference hashes silently fails.

### F6: `weights_only=False` security regression
**File:** `scripts/dedup_multilabel_index.py`, `scripts/label_cleaner.py`  
Both use `torch.load(pt_path, weights_only=False)` allowing arbitrary code execution. `build_multilabel_index.py` correctly uses `weights_only=True` with safe globals.

### F7: `_label_row()` docstring says `dos_*.sol` implies DoS=1
**File:** `scripts/inject_augmented.py` line 13  
Docstring says `dos_*.sol` but code correctly uses `dos_vuln_*` prefix (safe variants are `dos_safe_*`). Docstring is misleading.

---

## WARN-Level Issues

| ID | File | Issue |
|----|------|-------|
| W1 | `sentinel_model.py:292-294` | `graph_has_func` tensor computed but never read (dead code from old fallback_mask) |
| W2 | `sentinel_model.py:209` | Logs "SentinelModel v6" but trainer ARCHITECTURE = "three_eye_v5" |
| W3 | `transformer_encoder.py:31-32` | Stale param count: says 295K trainable at r=8, but default is r=16 (~590K) |
| W4 | `fusion_layer.py:84,143` | Default `node_dim=64` in constructor; runtime always receives 256 |
| W5 | `graph_schema.py:31` | CHANGE POLICY says next version is v3; actual is v7, next should be v8 |
| W6 | `graph_schema.py:344` | FEATURE_NAMES comment says visibility "ordinal 0-2"; actual is 0.0/0.5/1.0 |
| W7 | `graph_extractor.py:498` | BUG-C3 comment says has_loop at [10]; actually at [9] in v7 |
| W8 | `trainer.py:525` | `CLASS_NAMES.index("DenialOfService")` called per batch; should be precomputed |
| W9 | `trainer.py:668,670` | Sampler weights (3×/39×) hardcoded, not in TrainConfig |
| W10 | `label_cleaner.py` | No DoS precondition exists (missed opportunity for FP reduction) |
| W11 | `inject_augmented.py:195` | CSV write is non-atomic (no tmp+rename); crash → corrupted CSV |
| W12 | `create_cache.py` | No tokenizer config version in cache pickle; stale tokens undetected |
| W13 | `ast_extractor.py:143` | `get_solc_binary()` uses `Path.cwd()` instead of `get_project_root()` |
| W14 | `ast_extractor.py` | No timeout on Slither invocation; one hanging contract stalls entire pool |
| W15 | `tokenizer.py:170` | Truncation detection flags 510-token contracts as "truncated" (false positive) |

---

## What PASSED (No Issues Found)

| Component | Check | Verdict |
|-----------|-------|---------|
| `graph_schema.py` | NODE_FEATURE_DIM=11, all 11 features correctly defined | PASS |
| `graph_schema.py` | 8 edge types defined, consistent with extractor | PASS |
| `graph_schema.py` | 13 NODE_TYPE entries (0-12) | PASS |
| `graph_schema.py` | Import-time assertions catch dim mismatches | PASS |
| `graph_extractor.py` | All normalizations bounded [0,1] (log1p for loc/complexity/ext_call_count) | PASS |
| `graph_extractor.py` | DoS features: has_loop (dim 9) + ext_call_count (dim 10) with Transfer/Send | PASS |
| `graph_extractor.py` | Empty graph, zero edges, missing attributes handled | PASS |
| `trainer.py` | CLASS_NAMES: 10 classes, correct alphabetical order | PASS |
| `trainer.py` | ASL loss: correct Ridnik et al. implementation, gamma_neg=2 prevents collapse | PASS |
| `trainer.py` | Gradient clipping at 1.0 after unscale | PASS |
| `trainer.py` | Per-group learning rates correctly applied | PASS |
| `trainer.py` | Label smoothing: per-class epsilon, correct formula | PASS |
| `losses.py` | ASL: log clamping at 1e-8 prevents NaN | PASS |
| `focalloss.py` | Both focal loss variants correct; alpha_t bug already fixed | PASS |
| `build_multilabel_index.py` | CLASS_COLS→CLASS_NAMES mapping correct for all 10 BCCC columns | PASS |
| `dedup_multilabel_index.py` | OR-merge mathematically correct; no label loss | PASS |
| `inject_augmented.py` | `dos_vuln_*` vs `dos_safe_*` labeling bug is FIXED | PASS |
| `retokenize_windowed.py` | Sliding window + linspace sub-sampling correct | PASS |
| `sentinel_model.py` | Classifier 384→192→10, all dimensions chain correctly | PASS |
| `sentinel_model.py` | LoRA: frozen base + r=16, alpha=32, target=[query,value] | PASS |
| `gnn_encoder.py` | JK attention, residual connections, phase structure correct | PASS |
| `fusion_layer.py` | Cross-attention dims compatible, masked pooling, device guard | PASS |
| `dual_path_dataset.py` | Label loading, safe globals, cache validation | PASS |
| `hash_utils.py` | Path-based MD5 consistent within offline pipeline | PASS |
| All files | CLASS_NAMES identical across all 4 defining files | PASS |

---

## Pre-Training Checklist

Before running `train.py` for v7.0, these items must be resolved:

- [ ] **C1:** Fix `create_cache.py` default `--tokens-dir` to `ml/data/tokens_windowed`
- [ ] **C2:** Apply DoS masking to auxiliary loss heads in `trainer.py`
- [ ] **C3:** Fix `train.py --gnn-layers` default from 6 to 7
- [ ] **C4:** Add `--dos-loss-weight` CLI flag to `train.py`
- [ ] **C5:** Verify `retokenize_windowed.py` completed successfully; rebuild cache with correct dir
- [ ] **C6:** Add shape guard in `GNNEncoder.forward()` for stale 12-dim .pt files
- [ ] **F6:** Replace `weights_only=False` with safe globals in dedup + label_cleaner
- [ ] **F4:** Add "positive" to `--weighted-sampler` CLI choices
