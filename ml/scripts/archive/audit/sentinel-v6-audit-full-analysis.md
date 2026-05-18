# SENTINEL v6 Audit — Full Analysis Report

**Date:** 2026-05-17  
**Tasks completed:** 20/20  
**Total runtime:** 5.1 minutes  

---

## Executive Summary

The audit confirms **9 previously suspected bugs** and reveals **6 critical new findings** that change our understanding of the dataset's quality and the model's likely behavior.

### Most Critical New Finding

**BUG-6 (wrong contract selection) is far worse than estimated.** The "Most Functions" heuristic is only **52.6% accurate** — nearly a coin flip. The "Last Contract" heuristic achieves **87.4% accuracy**, making it a strictly better default.

### Severity-Ranked Findings

| Priority | Finding | Impact |
|----------|---------|--------|
| **P0** | BUG-6: 47.4% wrong contract selection | Almost half the multi-contract graphs extract the wrong contract |
| **P0** | Timestamp 48.2% mislabelled | Nearly half of Timestamp=1 labels have no supporting evidence |
| **P1** | BUG-1: CFG loc raw (max=129) | CFG node features out of [0,1] range |
| **P1** | BUG-2: complexity raw (max=67) | Declaration node features out of [0,1] range |
| **P1** | BUG-5/7/8: 3 dead features + 2 dead edge types | in_unchecked, EMITS, INHERITS never fire |
| **P2** | BUG-3: visibility=2 for private | Exceeds [0,1] normalization |
| **P2** | DoS has only 7 pure-label contracts | Class is effectively untrainable |
| **P2** | CFG nodes have zero semantic features | GNN gets no direct signal from 72% of nodes |
| **P3** | BUG-9: .send() return_ignored misses 3 cases | Minor extraction gap |
| **P3** | Graph size borderline confound | AUC 0.55-0.64 from size alone |

---

## Detailed Findings by Category

---

### 1. DATA INTEGRITY (Tasks 09, 11, 12, 13, 26)

#### Task 11: File Alignment — ✅ CLEAN

| Set | Count |
|-----|-------|
| CSV rows | 44,470 |
| Graph .pt files | 44,470 |
| Token_windowed .pt files | 44,470 |
| **CSV ∩ Graphs ∩ Tokens** | **44,470 (100%)** |

All three data sources are perfectly aligned. No missing files, no orphans. Retokenization checkpoint confirmed completed (timestamp: 2026-05-16T23:12:22).

#### Task 12: Token Integrity — ✅ CLEAN

All 9 checks pass on all 100 sampled files:
- input_ids shape, attention_mask shape, padding windows, real windows, vocab range, num_tokens match, no NaN, no negative, schema v4

Token stats: mean 1,737 tokens/contract, 82% use all 4 windows, max 2,044 tokens.

#### Task 13: Graph Structural Integrity — ⚠️ MINOR

| Check | Pass Rate |
|-------|-----------|
| x_shape_12 | 100% |
| edge_index valid | 100% |
| edge_attr range | 100% |
| no NaN/Inf | 100% |
| CONTAINS→CFG targets | 100% |
| CF nodes are CFG | 100% |
| **at_least_1_edge** | **99.2% (4 failures)** |

4 out of 44,470 graphs have zero edges (disconnected). These are: `7319be6b...`, `12a9a38c...`, `5d686fe6...`, `1fd0a54c...`.

**Edge type distribution confirms BUG-7 and BUG-8:**

| Edge Type | Count | Notes |
|-----------|-------|-------|
| CALLS (0) | 5,974 | |
| READS (1) | 8,495 | |
| WRITES (2) | 8,622 | |
| **EMITS (3)** | **0** | **BUG-7: completely absent** |
| **INHERITS (4)** | **0** | **BUG-8: completely absent** |
| CONTAINS (5) | 48,175 | |
| CONTROL_FLOW (6) | 41,233 | |

#### Task 26: Stale v5 Contamination — ✅ CLEAN

All 44,470 graphs use the correct v4 (12-dim) schema. Zero stale 8-dim graphs found.

---

### 2. FEATURE QUALITY (Tasks 09, 01, 21)

#### Task 09: Feature Range Audit

**Confirmed bugs (previously known):**

| Bug | Feature | Problem | Evidence |
|-----|---------|---------|----------|
| BUG-1 | loc [6] | CFG nodes: raw values, not log-normalized | CFG max=129.0 (should be ≤1.0) |
| BUG-2 | complexity [5] | Declaration nodes: raw values, not normalized | DECL max=67.0 (should be ≤1.0) |
| BUG-3 | visibility [1] | Private functions get value=2 | DECL max=2.0 (exceeds [0,1]) |

**Feature value ranges (500 graphs, 66,288 nodes):**

| Feature | DECL min | DECL max | CFG min | CFG max | Problem? |
|---------|----------|----------|---------|---------|----------|
| type_id | 0.000 | 0.583 | 0.667 | 1.000 | ✅ by design |
| visibility | 0.000 | **2.000** | 0.000 | 0.000 | ⚠️ BUG-3 |
| uses_block_globals | 0.000 | 1.000 | 0.000 | 0.000 | ✅ |
| view | 0.000 | 1.000 | 0.000 | 0.000 | ✅ |
| payable | 0.000 | 1.000 | 0.000 | 0.000 | ✅ |
| complexity | 0.000 | **67.000** | 0.000 | 0.000 | ⚠️ BUG-2 |
| loc | 0.100 | **946.000** | 1.000 | **129.000** | ⚠️ BUG-1 |
| return_ignored | 0.000 | 1.000 | 0.000 | 0.000 | ✅ |
| call_target_typed | 0.000 | 1.000 | 1.000 | 1.000 | ✅ (CFG always 1.0) |
| in_unchecked | 0.000 | **0.000** | 0.000 | **0.000** | ⚠️ BUG-5: dead |
| has_loop | 0.000 | 1.000 | 0.000 | 0.000 | ✅ |
| ext_call_count | 0.000 | 0.949 | 0.000 | 0.000 | ✅ |

**Critical observation:** CFG nodes have **zero values** for ALL semantic features (visibility, uses_block_globals, view, payable, complexity, return_ignored, in_unchecked, has_loop, ext_call_count). Only `loc`, `type_id`, and `call_target_typed` are set on CFG nodes. This means 72% of all nodes (48,175 CFG out of 66,288 total) carry almost no feature signal.

#### Task 01: Activation Split (Declaration vs CFG)

This confirms and extends the Task 09 finding:

**Dead features (zero activation everywhere):**
- `in_unchecked` — never fires in ANY class, on ANY node type

**Declaration-only features (always 0 on CFG nodes):**
- `external_call_count`, `has_loop`, `return_ignored`, `uses_block_globals`, `view`, `payable`, `visibility`, `complexity`

**Activation rates by class (DECL nodes only, key features):**

| Class | uses_bg | ret_ign | has_loop | ext_call |
|-------|---------|---------|----------|----------|
| CallToUnknown | 0.0% | 0.65% | 0.65% | **11.3%** |
| Reentrancy | 2.4% | 3.4% | 1.6% | **11.3%** |
| Timestamp | **3.3%** | 1.0% | 4.0% | 14.3% |
| IntegerUO | 1.6% | 1.4% | 2.4% | 9.5% |
| DoS | 0.0% | 0.0% | **14.0%** | 2.3% |

**Key insight:** `uses_block_globals` only fires at 3.3% even for Timestamp contracts. Combined with BUG-6 (wrong contract), this means the feature misses most timestamp signals.

#### Task 21: Feature Correlation & Redundancy

**Highly correlated pairs (|r| > 0.5):**

| Pair | Pearson | Spearman |
|------|---------|----------|
| type_id ↔ loc | 0.094 | **0.741** |
| complexity ↔ ext_call_count | **0.540** | 0.467 |

**PCA variance explained:**
- 90% variance needs 9 components (out of 12)
- PC12 has 0.0000 variance → `in_unchecked` is completely dead, contributing nothing

**Drop recommendations (highest redundancy + lowest unique info):**
1. `complexity` — 50.7% unique info, correlated with ext_call_count (r=0.54)
2. `type_id` — 71.3% unique info, correlated with loc (Spearman r=0.74)
3. `visibility` — 79.5% unique info, correlated with type_id (r=-0.41)

**Note:** The correlation between type_id and loc makes sense — CFG nodes have higher type_id AND higher loc (because BUG-1 makes CFG loc raw). After fixing BUG-1, this correlation should decrease significantly.

---

### 3. ARCHITECTURE ISSUES (Tasks 16, 17, 18)

#### Task 16: Wrong Contract Selection — 🔴 CRITICAL

This is the most impactful finding. Previous estimates put the wrong-selection rate at 7-28%. The actual rate is **47.4%**.

| Heuristic | Accuracy | Wrong Rate | 95% CI |
|-----------|----------|------------|--------|
| Most Functions (current) | 52.6% | **47.4%** | [39.2%, 55.8%] |
| Last Contract | **87.4%** | 12.6% | [8.0%, 19.2%] |

**Per-class wrong rate (Most Functions):**
- CallToUnknown: 56.2%
- IntegerUO: 48.0%
- Reentrancy: 47.4%
- ExternalBug: 50.0%
- GasException: 41.2%

**Common wrong-selection pattern:** The extractor picks `StandardToken` (a library with many functions) instead of the actual vulnerable contract (e.g., `ERC20Token`, `BTPCoin`, `FANBASE`). Of 64 wrong selections by "Most Functions", many choose `StandardToken` — a base class that is never the vulnerable contract.

**By contract count:**
- 2 contracts per file: 17.6% wrong (MF) vs 0% (LC)
- 6+ contracts per file: 50-100% wrong (MF) vs 10-50% (LC)

**Recommendation:** Switch to "Last Contract" heuristic immediately for a 3.8× improvement (47.4% → 12.6% wrong). Or better, extract all contracts into a merged graph.

#### Task 17: SafeMath Viability

SafeMath absence is a **weak signal** for IntegerUO:

| | IntegerUO=1 | NonVulnerable |
|---|------------|---------------|
| SafeMath present | 110 (55%) | 90 (45%) |
| SafeMath absent | 90 (45%) | 110 (55%) |

Only 45% of IntegerUO contracts lack SafeMath — essentially coin-flip discriminative power. SafeMath is NOT a viable replacement for `in_unchecked` as a feature.

Graph-based SafeMath detection has only 43.6% recall — the graph only captures SafeMath in 17/39 source-verified cases.

**Solidity version:** 100% of analyzed contracts are <0.8.0. Zero 0.8.x contracts found in the IntegerUO/NonVulnerable samples.

#### Task 18: Solidity Version Distribution

| Version | Count | Percentage |
|---------|-------|------------|
| **0.4.x** | 1,758 | **87.9%** |
| 0.5.x | 160 | 8.0% |
| 0.8.x | 1 | 0.1% |
| no_pragma | 81 | 4.0% |

This confirms BUG-5: `in_unchecked` (which only fires on Solidity ≥0.8.0) is dead because 99.9% of the dataset is pre-0.8.0. The feature was designed for a Solidity version that barely exists in the data.

**0.5.x anomaly:** IntegerUO rate jumps to 70% for 0.5.x (vs 34.7% for 0.4.x), suggesting version-specific vulnerability patterns.

---

### 4. LABEL QUALITY (Tasks 19, 20)

#### Task 19: Timestamp Label Quality — 🔴 CRITICAL

**48.2% of Timestamp=1 contracts appear mislabelled.**

| Category | Description | Count | Rate |
|----------|-------------|-------|------|
| (a) | Signal in source AND feature fires | 623 | 28.4% |
| (b) | Signal in source but feature doesn't fire | 491 | 22.4% |
| **(c)** | **No signal AND feature doesn't fire** | **1,056** | **48.2%** |
| (d) | No signal but feature fires | 21 | 1.0% |

**Category (b) breakdown:** 202 of 491 cases are due to wrong contract selection (BUG-6). The remaining 289 are due to Slither IR omissions, inline assembly, or indirect access.

**Category (c):** 1,056 contracts labelled Timestamp=1 but with NO block globals in source AND no feature activation. These are likely mislabelled in the BCCC dataset.

**Implication:** The Timestamp class has ~48% noise. A model trained on these labels will learn to predict "not Timestamp" for truly timestamp-dependent contracts (because they're outnumbered by mislabelled ones), and may learn spurious correlations from the mislabelled majority.

#### Task 20: DoS ↔ Reentrancy Separability

| Metric | Value |
|--------|-------|
| Total DoS=1 | 377 |
| DoS + Reentrancy | 370 (98.1%) |
| **DoS only** | **7 (1.9%)** |

Only 7 pure-DoS contracts exist in the entire dataset. The classes are effectively the same label.

**DoS-only vs DoS+Reentrancy features:**

| Feature | DoS-only (n=7) | DoS+Ree (n=370) |
|---------|-----------------|------------------|
| mean_nodes | 20.4 | 102.3 |
| mean_edges | 33.3 | 151.9 |
| mean_has_loop | 0.36 | 0.02 |
| mean_complexity | 5.56 | 3.64 |

DoS-only contracts are **much smaller** and have **higher loop rates**, which is the correct DoS signal (unbounded loops). But with only 7 examples, the model cannot learn this pattern.

**DoS-only contracts across splits:** train=3, val=1, test=3. With 3 training examples, DoS-specific learning is impossible.

---

### 5. CONFOUNDS & SHIFTS (Tasks 22, 23, 25)

#### Task 22: Graph Size Confound

**Logistic Regression AUC using only [num_nodes, num_edges, num_functions]:**

| Class | AUC | Status |
|-------|-----|--------|
| Timestamp | **0.637** | Borderline |
| IntegerUO | 0.618 | Borderline |
| GasException | 0.609 | Borderline |
| All others | 0.55-0.59 | Low |

No class exceeds AUC 0.65, so graph size is not a strong confound. However, **Mann-Whitney U tests are significant (p<0.01) for 8 out of 10 classes**, meaning vulnerable contracts ARE significantly larger — the model could partially learn "bigger = more vulnerable."

**Size vs label (top-25% vs bottom-25%):**

| Class | Bottom-25% | Top-25% | Difference |
|-------|-----------|---------|------------|
| IntegerUO | 21.1% | **46.4%** | +25.3% |
| GasException | 7.2% | 17.5% | +10.3% |
| MishandledException | 7.3% | 15.2% | +7.9% |

IntegerUO is the most size-biased: 46.4% of large graphs are IntegerUO vs only 21.1% of small graphs.

#### Task 23: .send() Unchecked Prevalence

| Metric | Value |
|--------|-------|
| ME contracts with .send() | 32/500 (6.4%) |
| Of those, unchecked .send() | 4 (12.5%) |
| Unchecked .send() missed by graph | **3** |

BUG-9 confirmed: 3 contracts have unchecked `.send()` in source but `return_ignored=0` in the graph. The extractor doesn't detect `.send()` as a call whose return value should be checked.

#### Task 25: Split Distribution Shift — ✅ CLEAN

| Split | Samples | 
|-------|---------|
| train | 31,142 |
| val | 6,661 |
| test | 6,667 |

Per-class positive rates are consistent across splits (within ±1%). KS tests on feature distributions show no significant shift. Solidity version distribution is consistent. The splits are well-balanced.

---

### 6. ALIGNMENT (Tasks 10, 24)

#### Task 10/24: Graph-Token Alignment — ⚠️ PATH FORMAT MISMATCH

| Metric | Result |
|--------|--------|
| Hash match | 1/1 (100% of those with hash in both) |
| Hash missing (graph side) | 99/100 |
| Path match | 0/95 (0%) |
| Path mismatch | 95/95 (100%) |
| **Stem↔hash verification** | **100/100 (100%)** |
| **Decode match** | **10/10 (100%)** |

**Finding:** Graph files store `contract_path` as a relative path (e.g., `BCCC-SCsVul-2024/SourceCodes/IntegerUO/abc.sol`), while token files store it as an absolute path (e.g., `/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/abc.sol`). This causes 100% "path mismatch" but is NOT a data integrity issue — they point to the same file.

The `contract_hash` attribute is missing from 99% of graph files (only 1 of 100 sampled graphs had it), but present in all token files. This is also not a bug — the graph extractor simply didn't save this attribute.

**The important verification passes:** stem↔hash verification is 100% (every .pt filename matches the MD5 of its contract path), and token decode verification is 100% (all 10 decoded tokens match their source files).

---

### 7. EXTENDED CHECKS (Tasks 14, 15)

#### Task 14: Sub-Sampling Coverage — ✅ SAFE

10 contracts with >2000 tokens were analyzed. All 10 show 100% survival rate for vulnerability-relevant windows under linspace sub-sampling.

The `linspace(0, W-1, 4)` strategy covers start, ~1/3, ~2/3, and end of the contract, which happens to capture the vulnerability-relevant code in all tested cases. However, this is a small sample (n=10), and the survival rate could be lower for contracts where vulnerable code is in the middle 1/3 not captured by a window.

#### Task 15: in_unchecked Regex — ⚠️ FALSE POSITIVES

The regex `\bunchecked\s*\{` produces false positives on:
- Comments: `// unchecked { this is a comment }` — MATCHES (false positive)
- Strings: `string memory s = "unchecked {"` — MATCHES (false positive)

However, since the feature is completely dead (0 activation), this doesn't affect the current dataset. If the feature were to be fixed and the regex used as a fallback, it would need comment/string stripping first.

---

## Consolidated Bug List (Updated)

| Bug | Description | Severity | Confirmed? | Fix Priority |
|-----|-------------|----------|-----------|-------------|
| BUG-1 | CFG loc not log-normalized (max=129) | High | ✅ | P1 |
| BUG-2 | complexity not normalized (max=67) | High | ✅ | P1 |
| BUG-3 | visibility=2 for private (exceeds [0,1]) | Medium | ✅ | P2 |
| BUG-5 | in_unchecked dead (0.8.x <0.1% of data) | High | ✅ | P1 (replace feature) |
| BUG-6 | Wrong contract selection (47.4% wrong!) | **Critical** | ✅ (worse than estimated) | **P0** |
| BUG-7 | EMITS edges never created | Medium | ✅ | P2 |
| BUG-8 | INHERITS edges never created | Medium | ✅ | P2 |
| BUG-9 | .send() return_ignored missed | Low | ✅ (3 cases) | P3 |
| NEW-1 | Timestamp 48.2% mislabelled | **Critical** | ✅ | **P0** |
| NEW-2 | CFG nodes carry zero semantic features | High | ✅ | P2 (design issue) |
| NEW-3 | DoS has only 7 pure-label samples | High | ✅ | P2 (merge or augment) |
| NEW-4 | contract_path: relative vs absolute mismatch | Low | ✅ (cosmetic) | P3 |
| NEW-5 | 4 disconnected graphs | Low | ✅ | P3 |

---

## Priority Action Plan

### P0 — Fix Immediately (before any training)

1. **Switch contract selection to "Last Contract" heuristic** in `graph_extractor.py` → reduces wrong-selection from 47.4% to 12.6%. Or better: extract ALL contracts into a merged graph.

2. **Re-label or exclude Timestamp class** — 48.2% mislabelling makes this class actively harmful. Options:
   - Remove Timestamp from training labels entirely
   - Re-label using the (a) category contracts only (623 verified)
   - Add a "timestamp_verified" column

### P1 — Fix Before Next Training Run

3. **Fix BUG-1**: Apply `log1p(loc) / log1p(1000)` normalization to CFG loc, matching the declaration path.

4. **Fix BUG-2**: Apply `log1p(complexity) / log1p(50)` normalization (p99=12, so log1p(50) gives good spread).

5. **Replace in_unchecked (BUG-5)**: Since SafeMath is not viable (Task 17), replace with `pragma_version` (0.4.x=0, 0.5.x=0.5, 0.8.x=1.0) which is a meaningful signal for IntegerUO detection.

### P2 — Fix in Next Iteration

6. **Fix BUG-3**: Change visibility encoding from ordinal (0/1/2) to one-hot or binary (public/external=1, private/internal=0).

7. **Merge DoS into Reentrancy or augment** (BUG-20): With only 7 pure-DoS samples, the class is untrainable. Options:
   - Merge DoS label into Reentrancy
   - Generate synthetic DoS-only examples
   - Apply extreme class weighting

8. **Investigate EMITS/INHERITS edge absence** (BUG-7/8): The code creates these edges but dependency filtering removes them. Consider adding a `is_inherited` flag to declaration nodes instead.

9. **Address CFG node feature emptiness**: Consider propagating parent function features to CFG nodes during extraction, or adding CFG-specific features (branch_count, loop_depth, etc.).

### P3 — Nice to Have

10. Fix BUG-9 (.send() detection)
11. Normalize contract_path storage (relative vs absolute)
12. Investigate 4 disconnected graphs
13. Consider increasing MAX_WINDOWS from 4 to 6-8

---

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| Total contracts | 44,470 |
| File alignment | 100% (CSV ∩ Graphs ∩ Tokens) |
| Stale v5 graphs | 0 |
| Token integrity | 100% pass rate |
| Wrong contract rate (current) | **47.4%** |
| Wrong contract rate (Last Contract) | 12.6% |
| Timestamp mislabel rate | **48.2%** |
| Dead features | in_unchecked (always 0) |
| Dead edge types | EMITS (0), INHERITS (0) |
| DoS pure-label count | 7 |
| DoS↔Reentrancy co-occurrence | 98.1% |
| Solidity 0.4.x dominance | 87.9% |
| Solidity 0.8.x presence | 0.1% |
| Graph size confound (max AUC) | 0.637 (Timestamp) |
| Split distribution shift | None detected |
| Sub-sampling vuln coverage | 100% (n=10) |
