# Full Data Validation Report — v6 Training Dataset
**Date:** 2026-05-17  
**Scope:** Fresh end-to-end validation of all 44,470 graphs, tokens, labels, cache, and splits  
**Context:** v6.0 training running (epoch ~16/100, F1-macro stalled at 0.1717 since epoch 9)  
**Method:** All checks run fresh on actual current files — not based on prior audit reports

---

## What Was Tested

Six validation sections were executed:

1. **Graph feature integrity** — all 44,470 graphs, 5,585,695 total nodes
2. **Token integrity** — 1,000 randomly sampled token files
3. **CSV label integrity** — all 44,470 rows, all 10 classes
4. **File alignment** — 3-way check: CSV ↔ graphs ↔ tokens
5. **Split integrity** — train/val/test indices, stratification, coverage
6. **Cache integrity** — pkl load, structure, sample items

---

## Section 1: Graph Feature Integrity (ALL 44,470 graphs)

5,585,695 total nodes examined across all graphs.

### Feature range check (12-dim feature vector)

| Dim | Feature | Expected range | Actual max | OOR nodes | OOR% | Status |
|-----|---------|---------------|-----------|-----------|------|--------|
| 0 | type_id/12.0 | [0, 1] | 1.0 | 0 | 0.00% | ✅ PASS |
| 1 | visibility | {0, 1} | **2.0** | **35,624** | **0.64%** | ❌ FAIL |
| 2 | uses_block_globals | {0, 1} | 1.0 | 0 | 0.00% | ✅ PASS |
| 3 | view | {0, 1} | 1.0 | 0 | 0.00% | ✅ PASS |
| 4 | payable | {0, 1} | 1.0 | 0 | 0.00% | ✅ PASS |
| 5 | complexity | [0, 1] | **48.0** | **764** | **0.01%** | ❌ FAIL |
| 6 | loc | [0, 1] | **2167.0** | **75,328** | **1.35%** | ❌ FAIL |
| 7 | return_ignored | {-1, 0, 1} | 1.0 | 0 | 0.00% | ✅ PASS |
| 8 | call_target_typed | {-1, 0, 1} | 1.0 | 0 | 0.00% | ✅ PASS |
| 9 | in_unchecked | {0, 1} | 0.0 | 0 | 0.00% | ✅ PASS (dead) |
| 10 | has_loop | {0, 1} | 1.0 | 0 | 0.00% | ✅ PASS |
| 11 | ext_call_count | [0, 1] | 1.0 | 0 | 0.00% | ✅ PASS |

### BUG-1: loc stored as raw line count (dim[6])

- **Graphs affected:** 2,856 / 44,470 — **6.42%**
- **Nodes affected:** 75,328 across those graphs
- **Severity:** CRITICAL — raw values up to 2,167 vs normalized range [0, 1]. A value of 2,167 is ~2,167× larger than the max normalized value. This dominates GNN dot products for those batches.
- **Node types affected:** FUNCTION (58,195 nodes), VARIABLE (6,329), RETURN_PARAM (4,810), EVENT (2,030), MODIFIER (1,006), CFG_NODE_BLOCK (2,856), CONTRACT (102)
- **Distribution of worst offenders:** 875 graphs with max_loc 10–50 / 1,200 graphs with 100–500 / 30 graphs with 1,000–3,000
- **Root cause:** The v7 re-extraction (2026-05-17) fixed the extractor code (`log1p(N)/log1p(1000)`) but 2,856 graphs were **not re-extracted** — they were patched in-place by a separate script that did not normalize all node types correctly. Those graphs still carry pre-fix raw values.
- **Fix:** In-place patch — read each .pt file, apply `min(log1p(raw)/log1p(1000), 1.0)` to dim[6] where value > 1.0, save back.

### BUG-2: complexity stored as raw CFG block count (dim[5])

- **Graphs affected:** 37 / 44,470 — **0.08%**
- **Nodes affected:** 764 — max observed value: 48
- **Severity:** LOW — negligible scope (37 graphs), but same root cause as BUG-1
- **Root cause:** Same as BUG-1 — those 37 graphs were not properly re-extracted or patched
- **Fix:** In-place patch — apply `min(log1p(raw)/log1p(100), 1.0)` to dim[5] where value > 1.0

### BUG-3: visibility=2 for private functions (dim[1])

- **Graphs affected:** 7,854 / 44,470 — **17.66%**
- **Nodes affected:** 35,624 — node types: FUNCTION (18,149), CONTRACT (17,451), MODIFIER (24)
- **Severity:** HIGH — value of 2 is outside the [0,1] range declared in the schema. Every affected graph sends an out-of-range feature into the GNN on every forward pass.
- **Root cause:** `VISIBILITY_MAP` in `graph_schema.py` line 273: `"private": 2`. This is intentional ordinal encoding ("private > internal > public") but the schema declares [0,1] range.
- **MEMORY.md claim:** "v5: private+internal both map to 0" — **this is wrong**. The actual code still has `private=2, internal=1`. The memory was never updated to reflect the actual schema.
- **Options:**
  - **Option A:** Normalize: `public=0.0, internal=0.5, private=1.0` — stays in [0,1], preserves ordinal ordering
  - **Option B:** Binary: `public/external=0, internal/private=1` — simpler, loses private/internal distinction
  - **Option C:** Keep as-is — the GNN weight matrix can learn to handle value=2, but it breaks normalization assumptions
- **Fix:** Patch in-place, update VISIBILITY_MAP, rebuild cache

### Other graph checks

| Check | Result | Status |
|-------|--------|--------|
| NaN in any feature | 0 nodes | ✅ PASS |
| Inf in any feature | 0 nodes | ✅ PASS |
| Graphs with ANY OOR feature (BUG-1 OR BUG-3) | **9,973 / 44,470 (22.4%)** | ❌ FAIL |
| Graphs with 0 nodes | 0 | ✅ PASS |
| Graphs with 0 edges | 319 (0.72%) | ⚠️ WARN |
| edge_attr shape != 1D [E] | 0 | ✅ PASS |
| edge_attr values outside [0, 7] | 0 | ✅ PASS |
| Negative edge indices | 0 | ✅ PASS |

**319 zero-edge graphs:** These are valid isolated contracts (single-node graphs). The GNN produces embeddings from self-loops only — weak signal but not corrupt.

**Graph size distribution:**
| Metric | Nodes | Edges |
|--------|-------|-------|
| Min | 1 | 0 |
| Max | 1,734 | 3,492 |
| Mean | 125.6 | 211.8 |
| p50 | 88 | 134 |
| p95 | 323 | 588 |

**Key finding:** 22.4% of graphs (9,973) have at least one out-of-range feature being fed to the model on every training batch. The GNN cannot distinguish "this contract has a very long function" from "this is a corrupt feature" — it sees a raw loc value of 2,167 and the weight matrix must absorb that.

---

## Section 2: Token Integrity (1,000 random files)

| Check | Result | Status |
|-------|--------|--------|
| Shape [4, 512] | 0 / 1,000 wrong | ✅ PASS |
| dtype int64 | 0 / 1,000 wrong | ✅ PASS |
| Vocab range [0, 50,265) | 0 / 1,000 OOR | ✅ PASS |
| CLS token at each window start | 3,756 / 4,000 windows | ✅ PASS |
| Fully-padded windows | 244 / 4,000 (6.1%) | ✅ PASS |
| NaN or Inf | 0 | ✅ PASS |

**Window distribution across the 1,000 sampled contracts:**
- 1 real window (short contracts): 34 (3.4%)
- 2 windows: 38 (3.8%)
- 3 windows: 66 (6.6%)
- 4 windows (full): 862 (86.2%)

86.2% of contracts use all 4 windows — the windowed tokenizer is working correctly. 244 empty windows are correct padding for contracts shorter than 4 × 512 tokens.

**Schema version tag:** Token files are stamped `feature_schema_version=v4`. This is cosmetic — the tokenization logic (CodeBERT, stride, windowing) did not change between schema v4 and v5. Token content is correct for v6 training.

**Token section: no actionable bugs.**

---

## Section 3: CSV Label Integrity

| Check | Result | Status |
|-------|--------|--------|
| Total rows | 44,470 | — |
| Duplicate md5_stem | 0 | ✅ PASS |
| All 10 label columns present | Yes | ✅ PASS |
| Label values outside {0, 1} | 0 | ✅ PASS |

### Per-class counts

| Class | Count | % of rows | Train positives | Status |
|-------|-------|-----------|----------------|--------|
| IntegerUO | 13,797 | 31.0% | 9,697 | ✅ |
| GasException | 4,957 | 11.1% | 3,448 | ✅ |
| Reentrancy | 4,498 | 10.1% | 3,126 | ✅ |
| MishandledException | 4,186 | 9.4% | 2,957 | ✅ |
| CallToUnknown | 3,256 | 7.3% | 2,261 | ✅ |
| TransactionOrderDependence | 3,028 | 6.8% | 2,112 | ✅ |
| ExternalBug | 3,009 | 6.8% | 2,123 | ✅ |
| UnusedReturn | 2,716 | 6.1% | 1,899 | ✅ |
| Timestamp | 961 | 2.2% | 679 | ⚠️ low |
| **DenialOfService** | **346** | **0.8%** | **215** | ❌ critical |

### Label density distribution

| Labels per row | Count | % |
|---------------|-------|---|
| **0 (no vulnerability)** | **26,710** | **60.1%** |
| 1 | 5,274 | 11.9% |
| 2 | 6,396 | 14.4% |
| 3 | 3,436 | 7.7% |
| 4 | 1,386 | 3.1% |
| 5+ | 1,268 | 2.9% |

**60.1% of all rows are safe/clean contracts** with no vulnerability label. These are genuine negatives from the BCCC dataset — not a labeling error. However combined with ASL γ⁻=4 (aggressive negative down-weighting), the model is incentivized to predict 0 for everything, which maximizes Hamming accuracy (~90%) while producing near-zero F1. The rising Hamming (0.82→0.90 over epochs 9–15) alongside stagnant F1 (0.17) is the signature of this collapse.

### DenialOfService co-occurrence (CRITICAL)

| Co-label | DoS rows with this label | % |
|---------|--------------------------|---|
| Reentrancy | 341 / 346 | **98.6%** |
| GasException | 217 / 346 | 62.7% |
| CallToUnknown | 0 / 346 | 0.0% |
| Others | < 5 / 346 | < 1.5% |

**Pure single-label DoS in training: 3 samples.** The model cannot learn to distinguish DoS from Reentrancy — they are effectively the same class from the model's perspective. This explains F1=0.022 for DoS at epoch 15.

### Pure-label rate per class

| Class | Pure (single-label) | Total | Pure% |
|-------|--------------------|----|-------|
| Timestamp | 374 | 961 | 38.9% |
| IntegerUO | 3,820 | 13,797 | 27.7% |
| GasException | 640 | 4,957 | 12.9% |
| TOD | 148 | 3,028 | 4.9% |
| Reentrancy | 118 | 4,498 | 2.6% |
| MishandledException | 77 | 4,186 | 1.8% |
| UnusedReturn | 49 | 2,716 | 1.8% |
| ExternalBug | 34 | 3,009 | 1.1% |
| CallToUnknown | 9 | 3,256 | 0.3% |
| **DenialOfService** | **5** | **346** | **1.4%** |

Most classes appear almost exclusively in multi-label combinations — the model must disentangle correlated vulnerabilities from shared graph patterns.

---

## Section 4: File Alignment

| Check | Result | Status |
|-------|--------|--------|
| CSV rows | 44,470 | — |
| Graph .pt files | 44,470 | — |
| Token .pt files | 44,470 | — |
| CSV md5 missing graph | 0 | ✅ PASS |
| CSV md5 missing token | 0 | ✅ PASS |
| Graph not in CSV | 0 | ✅ PASS |
| Token not in CSV | 0 | ✅ PASS |

**Perfect 3-way alignment. No orphaned files, no missing files.**

---

## Section 5: Split Integrity

| Check | Result | Status |
|-------|--------|--------|
| train + val + test = 44,470 | 31,128 + 6,669 + 6,673 = 44,470 | ✅ PASS |
| train ∩ val | 0 overlap | ✅ PASS |
| train ∩ test | 0 overlap | ✅ PASS |
| val ∩ test | 0 overlap | ✅ PASS |
| OOB indices | 0 | ✅ PASS |
| All 44,470 indices covered | Yes | ✅ PASS |

### Per-class positive rate across splits (stratification quality)

| Class | Train% | Val% | Test% | Max diff | Status |
|-------|--------|------|-------|----------|--------|
| CallToUnknown | 7.26 | 7.48 | 7.43 | 0.22pp | ✅ |
| DenialOfService | 0.69 | 1.02 | 0.94 | 0.33pp | ✅ |
| ExternalBug | 6.82 | 6.40 | 6.88 | 0.48pp | ✅ |
| GasException | 11.08 | 11.59 | 11.03 | 0.56pp | ✅ |
| IntegerUO | 31.15 | 30.69 | 30.77 | 0.46pp | ✅ |
| MishandledException | 9.50 | 9.39 | 9.04 | 0.46pp | ✅ |
| Reentrancy | 10.04 | 10.36 | 10.21 | 0.32pp | ✅ |
| Timestamp | 2.18 | 2.32 | 1.90 | 0.42pp | ✅ |
| TOD | 6.78 | 6.67 | 7.06 | 0.39pp | ✅ |
| UnusedReturn | 6.10 | 6.13 | 6.11 | 0.03pp | ✅ |

Label density: Train 0.916 / Val 0.921 / Test 0.914 — nearly identical.

**Splits are clean. No leakage, no distribution shift.**

---

## Section 6: Cache Integrity

| Check | Result | Status |
|-------|--------|--------|
| File size | 2.47 GB | — |
| Total entries | 44,470 + 1 metadata key | ✅ PASS |
| `__schema_version__` key | `v5` | — |
| Sample 10 items: graph x shape [N,12] | 10/10 correct | ✅ PASS |
| Sample 10 items: token shape [4,512] | 10/10 correct | ✅ PASS |
| edge_attr shape 1D | 10/10 correct | ✅ PASS |
| Per-item `feature_schema_version` | `v4` | ⚠️ cosmetic |

The `v4` per-item tag is a cosmetic mismatch — the cache was built from token files that were stamped v4. The token content is correct. The graphs in the cache carry whatever features are in `ml/data/graphs/` — meaning **BUG-1, BUG-2, BUG-3 corrupt features are also present in the cache** (the cache is a snapshot of the graph .pt files at build time).

---

## Consolidated Findings

### Active bugs in the data being fed to training right now

| ID | Bug | Graphs | Impact on training | Fix |
|----|-----|--------|--------------------|-----|
| BUG-1 | `loc` raw line count instead of log-normalized | 2,856 (6.4%) | GNN sees values up to 2,167 on a feature that should be [0,1]; corrupts attention weights for those batches | In-place patch + cache rebuild |
| BUG-2 | `complexity` raw CFG block count | 37 (0.08%) | Same as BUG-1, negligible scope | In-place patch + cache rebuild |
| BUG-3 | `visibility=2` for private — out of [0,1] | 7,854 (17.7%) | OOR value on 17.7% of all graphs every forward pass | Schema decision + patch + cache rebuild |
| — | **Any OOR feature (BUG-1 or BUG-3)** | **9,973 (22.4%)** | — | — |

### Label/dataset issues (not bugs, but training impact)

| Issue | Scope | Training impact |
|-------|-------|----------------|
| 60.1% of rows have 0 labels | 26,710 rows | Combined with ASL γ⁻=4, incentivizes model to predict 0 for everything → Hamming rises, F1 collapses |
| DoS: 3 pure training samples, 98.6% Reentrancy co-occurrence | 346 DoS total | Model cannot learn DoS independently — predicts Reentrancy as proxy |
| `in_unchecked` (dim[9]) is all-zero | all 44,470 | Dead feature — wastes one of 12 dimensions |
| EMITS/INHERITS edges: 0 in all graphs | all 44,470 | Two of 8 edge type slots unused |
| 319 graphs have 0 edges | 0.72% | Very weak GNN signal for those contracts |

### What is clean

| Component | Status |
|-----------|--------|
| File alignment (CSV ↔ graphs ↔ tokens) | ✅ Perfect 3-way match |
| Token files (shape, dtype, vocab, CLS/SEP/PAD) | ✅ All correct |
| Split integrity (no leakage, no shift) | ✅ Clean |
| NaN / Inf in features | ✅ Zero |
| edge_attr shape and value range | ✅ Correct |
| Schema version (graph_schema.py) | ✅ v5 |
| Contract selection heuristic | ✅ Fixed (most_derived) in v7 extraction |

---

## Training Context

**Current state (as of writing):**
- Epoch ~16/100, F1-macro = 0.1717 best (achieved at epoch 9), stalled since
- Hamming rising (0.82→0.90) while F1 stagnant = model converging toward predicting 0 for everything
- v5.2 baseline: F1-macro = 0.3422 (the buggy version performs better than v6 at epoch 16)
- All per-class F1 values below v5.2 baseline: IntegerUO 0.494 vs 0.732, GasException 0.223 vs 0.407

**Hypothesized root causes of stalled F1 (in order of likely impact):**
1. **22.4% of graphs have corrupt features** (BUG-1/BUG-3) — GNN learning corrupted batches
2. **ASL γ⁻=4 too aggressive** — with 60% zero-label rows, model learns "predict 0" is safe; Hamming≈90% at near-zero F1
3. **DoS/Timestamp data starvation** — drags macro average down; DoS F1=0.02, Timestamp F1=0.05
4. **Dead in_unchecked feature** — wastes 1/12 of feature capacity

---

## Decisions Required

The following decisions need to be made before proceeding:

**D1 — Fix corrupted graphs and resume training?**
- Fix BUG-1 (2,856 graphs) + BUG-2 (37 graphs) in-place
- Decision on BUG-3 visibility: normalize to [0,1] or keep ordinal?
- Rebuild cache (~20 min)
- Kill training → resume from epoch 9 checkpoint (F1=0.1717)

**D2 — Reduce ASL γ⁻?**
- Current: γ⁻=4, γ⁺=1
- Proposed: γ⁻=2, γ⁺=1 — less aggressive negative suppression
- Rationale: 60% zero-label rows + γ⁻=4 may be causing the Hamming-rises/F1-stalls collapse
- Only applicable if restarting training (requires changing TrainConfig)

**D3 — Handle DoS class?**
- Option A: Drop DoS from training entirely (too few pure samples to learn)
- Option B: Keep but accept ~0 F1 on DoS
- Option C: Data augmentation (Phase 4 — find more pure DoS contracts)

**D4 — What to do with `in_unchecked` dead feature?**
- Drop dim[9] and rebuild with NODE_FEATURE_DIM=11
- Or keep as-is (wastes capacity but doesn't corrupt)
- Dropping requires schema bump and full re-extraction

---

*Document written 2026-05-17. Training PID 6530. Best checkpoint: `ml/checkpoints/v6.0-20260517_best.pt` (epoch 9, F1=0.1717).*
