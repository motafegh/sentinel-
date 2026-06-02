# SENTINEL Interpretability Master Report

**Model:** GCB-P1-Run4-no-asl-pw_best.pt — epoch 32 — macro-F1 = 0.3362
**Val split:** 6,236 contracts (of 41,577 total)
**Cache:** ml/data/cached_dataset_v8.pkl — schema v8, 11-dim node features
**Report date:** 2026-06-01 (updated — audit fixes: B1 gradient method, L2 structural ablation, L3 reclassified ARCH-N/A, L4 rerun)
**Checkpoint:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Cache and Checkpoint Validation](#2-cache-and-checkpoint-validation)
3. [Experiment Results Table](#3-experiment-results-table)
4. [Layer 1 — Structural Analysis (S1–S4, A1–A2)](#4-layer-1--structural-analysis)
5. [Layer 2 — Expressivity Analysis (E1–E4)](#5-layer-2--expressivity-analysis)
6. [Layer 3 — Learning Analysis (L1–L10, A3–A4)](#6-layer-3--learning-analysis)
7. [Root Cause Analysis: Why Is Phase 2 Underused?](#7-root-cause-analysis)
8. [Capacity Ceiling Analysis](#8-capacity-ceiling-analysis)
9. [Actionable Recommendations for Run 5](#9-actionable-recommendations)
10. [Appendix: Per-Class Analysis](#10-appendix-per-class-analysis)

---

## 1. Executive Summary

This report documents a structured interpretability study of the SENTINEL dual-path GNN+GraphCodeBERT model trained during Run 4 (GCB-P1-Run4-no-asl-pw). The study comprised 21 experiments across three analysis layers: structural data quality, architectural expressivity, and learned behaviour.

### What Was Tested

A total of 21 experiments were run, spanning:
- Data integrity and structural graph properties (S1–S4, A1–A2)
- Theoretical GNN expressivity bounds (E1–E4)
- What the trained model actually learned (L1–L10, A3–A4)

Experiments requiring the Solidity compiler (`solc`) were blocked by a toolchain gap. Experiments requiring the trained checkpoint were executed using the Run 4 ep32 checkpoint.

### Central Finding

**The GNN's Phase 2 (control-flow graph / ICFG / DFG) is structurally present in the data but contributes less signal than Phase 3 in the trained model.**

> ⚠️ **Validation note (2026-05-30):** Several claims in this section were corrected after independent validation. See `VALIDATION_SUMMARY.md` and the per-experiment docs for details. The core conclusion stands but magnitude claims have been updated.

The evidence is threefold and mutually reinforcing:

1. **JK phase weights (EXP-L1):** Phase 2 weight = 0.322, the lowest of the three phases across all 10 vulnerability classes. Phase 3 (REVERSE_CONTAINS) = 0.346 is marginally dominant. **However:** JK entropy = 1.0984 / max 1.0986 = 99.98% of theoretical maximum. The 24pp gap is statistically real (936 samples) but practically small — all three phases contribute nearly equally. The claim "Phase 2 is meaningfully underused" is an overstatement; more precise: "Phase 2 is marginally least weighted."

2. **Edge ablation (EXP-L2):** The original ablation (zeroing edge embeddings) was methodologically flawed — it removed edge TYPE signal but not edge STRUCTURE (edges remained in edge_index). **Corrected structural ablation** (removing edges from edge_index entirely) gives a 10,944× larger effect: embedding combined CFG drop = 1.11 × 10⁻⁶ vs structural combined drop = 0.0121. More importantly, structural ablation reveals **Phase 2 edges SUPPRESS Reentrancy predictions** (all CF edge types produce positive deltas on removal). The model is not merely ignoring Phase 2 CFG edges — it is using them as downward signal for Reentrancy. Even the structural result is below the 0.03 threshold in absolute magnitude, but the sign reversal is the key diagnostic.

3. **Aux eye contribution (EXP-A4):** The GNN eye alone beats a trivial baseline for only 3 of 10 vulnerability classes (CallToUnknown, IntegerUO, Timestamp). For Reentrancy — the class most reliant on CFG structure — the GNN eye achieves F1 = 0.182, barely above the random baseline of 0.170. **Additional nuance (EXP-L5):** Phase 2 AUROC = 0.618 > Phase 1 AUROC = 0.612 for Reentrancy — Phase 2 is not completely useless, but its signal is not linearly separable at the decision boundary.

### What This Means

The model has reached a local minimum where:
- Structural hierarchy (CONTRACT → FUNCTION → CFG node via CONTAINS/REVERSE_CONTAINS) is the primary GNN signal.
- Token semantics from GraphCodeBERT LoRA carry the bulk of the predictive capacity.
- The control-flow, interprocedural call, and data-flow graph structure encoded in Phase 2 edges (CONTROL_FLOW, CALL_ENTRY, RETURN_TO, DEF_USE) is geometrically present but has not been exploited.

This explains the F1 = 0.3362 ceiling: the model cannot improve further on call-graph-dependent vulnerabilities (Reentrancy, ExternalBug, TransactionOrderDependence) until Phase 2 carries meaningful signal.

### What Was Not Found

- No phase collapse (JK entropy 1.0935–1.0986, near theoretical max of log(3) = 1.099).
- No pooling failure (100% of graphs have ≥1 FUNCTION-type node; fallback never triggered).
- No fundamental expressivity barrier (WL-distinguishability test passed for all 4 tested classes).
- No evidence that CFG nodes are absent from the data (99.5% of contracts have CFG nodes — a previous agent's "0 CFG nodes" claim was incorrect).

---

## 2. Cache and Checkpoint Validation

### 2.1 Cache Integrity

The cache `ml/data/cached_dataset_v8.pkl` was audited on 2026-05-30.

| Property | Value | Verdict |
|---|---|---|
| Total entries | 41,577 | Matches train (29,103) + val (6,236) + test (6,237) + 1 |
| Graph node feature dim | 11 | v8 schema — correct |
| Token schema label | v7 | Tokenisation unchanged v7→v8; cosmetic label only |
| CFG nodes present | 99.5% of contracts | Correct |
| CALL_ENTRY (type 8) edges | 64.2% of contracts | Correct |
| RETURN_TO (type 9) edges | 55.6% of contracts | Correct |
| DEF_USE (type 10) edges | 79.9% of contracts | Correct |
| CONTAINS (type 5) edges | 99.6% of contracts | Correct |
| CONTROL_FLOW (type 6) edges | 99.6% of contracts | Correct |
| REVERSE_CONTAINS (type 7) | 0 edges in cache | Expected: runtime-computed at gnn_encoder.py:483–485 |
| EMITS (type 3) | 12 edges total across 41K contracts | Near-zero; not a useful signal |

**The cache is correct and complete.** The absence of REVERSE_CONTAINS edges in the pkl file is by architectural design: `gnn_encoder.py` lines 483–485 flip CONTAINS edges at runtime to construct REVERSE_CONTAINS. This is not a data defect.

The earlier agent report stating "CFG nodes absent from cached_dataset_v8.pkl" was incorrect. CFG nodes are present in 99.5% of contracts. The EXP-A2 test that returned 0 results used a query for node type `"CFG_NODE"` that did not match the actual stored node type string; the data is correct.

### 2.2 Checkpoint Integrity

The Run 4 best checkpoint is at epoch 32, F1 = 0.3362.

| Property | Value |
|---|---|
| Checkpoint file | `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` |
| Best epoch | 32 |
| Macro-F1 (val) | 0.3362 |
| Kill epoch | 44 (F1 locked 0.31–0.34 for 12 epochs) |
| Key caveat | Checkpoint dtype is BF16; call `.float()` before diagnostic inference |
| Key caveat | torch.compile adds `._orig_mod.` infix to state dict keys — strip before loading |

Training history confirms the run plateaued after ep32:
- ep21: F1 = 0.3224 (new best, prefix plateau resolved)
- ep25: F1 = 0.3272 (new best)
- ep32: F1 = 0.3362 (final best)
- ep33–ep44: F1 oscillated 0.31–0.34 — capacity ceiling confirmed

---

## 3. Experiment Results Table

All 21 experiments with status, key numeric result, and interpretation.

| ID | Name | Layer | Priority | Status | Key Number | Interpretation |
|----|------|-------|----------|--------|------------|----------------|
| S1 | Structural Trace Audit | Structural | P0 | **BLOCKED** | solc missing | Test contract extraction failed; val-split IntegerUO pattern 92.6% (pass), Reentrancy 14.3% (fail) |
| S2 | Edge Enrichment Ratio | Structural | P0 | **FAIL** | CONTROL_FLOW ratio = 1.004× | CFG edges are ubiquitous (99.6%); no discriminative enrichment per class |
| S3 | Feature Distribution | Structural | P1 | **PASS** ⚠️ | Timestamp cfg_call_count Cohen_d=1.592 (SHORTCUT); return_ignored d=0.716 for UnusedReturn | Timestamp shortcut confirmed (train split); "dead feature" finding retracted — was artifact of wrong node type |
| S4 | ICFG-Lite Path Audit | Structural | P0 | **PASS** | 76% have CALL_ENTRY | ICFG connectivity is present in data; 69% have full CALL_ENTRY+RETURN_TO chain |
| A1 | Pooling Node-Type Audit | Structural | P0 | **PASS** | 100% have FUNCTION node | No fallback pooling; RECEIVE nodes absent from corpus |
| A2 | CFG Feature Inheritance | Structural | P1 | **FAIL** | 0 graphs with INHERITS parents | Node type query mismatch — not a real data defect; to retry after node-type string audit |
| E1 | K-Hop Receptive Field | Expressivity | P0 | **FAIL** | CEI reachable = 38.2% at k=8 (DEF_USE fix) | CEI chain not reachable in 62% of Reentrancy-positive contracts; A2 PASS (85.5% CONTAINS coverage) |
| E2 | WL Distinguishability | Expressivity | P0 | **PASS** | Timestamp 0% collision | All 4 tested classes WL-distinguishable; no fundamental expressivity barrier |
| E3 | Message Propagation Sim | Expressivity | P1 | **FAIL** | Random weights show no signal | Expected: needs trained checkpoint; untrained GNN cannot differentiate Phase 2 activation |
| E4 | Direction Sensitivity | Expressivity | P1 | **FAIL** | All 4 edge types: Directed = Undirected = 89.1%, diff = 0.0% | Direction adds zero WL discriminative power for CONTROL_FLOW, DEF_USE, CALL_ENTRY, RETURN_TO |
| A3 | JK Entropy Logging | Learning | P1 | **PASS** | Entropy 1.0935–1.0986 | Near-maximum entropy (log(3)=1.099); no phase collapse across 47 epochs |
| A4 | Aux Eye Contribution | Learning | P1 | **FAIL** | GNN useful 3/10 classes | GNN eye alone useful only for CallToUnknown, IntegerUO, Timestamp |
| L1 | JK Weight Analysis | Learning | P1 | **FAIL** | Phase2 = 0.322 (lowest) | Phase 3 dominates all 10 classes; Phase 2 hypothesis wrong for all 4 tested classes |
| L2 | Edge Ablation (inference) | Learning | P1 | **FAIL** | CFG embedding drop=1.11e-6; structural drop=0.0121 (10,944× larger) | Structural ablation added 2026-06-01; Reentrancy structural deltas POSITIVE (Phase 2 edges suppress Reentrancy); CONTROL_FLOW structural Δ Timestamp=+0.163 |
| L3 | Attention Visualization | Learning | P2 | **ARCHITECTURAL N/A** *(audit 2026-06-01)* | All GAT weights = 1.0 (uniform) | 100% CF fraction was architecturally guaranteed (conv3 wired CF-only); real finding: no selective attention within CFG; PASS retracted |
| L4 | Gradient Saliency | Learning | P1 | **COMPLETE** *(rerun 2026-06-01)* | external_call_count rank 1 all classes (21–24%) | Global sensitivity artifact confirmed with correct feature names; Timestamp uses_block_globals=10.0% FAIL; Reentrancy CFG_NODE_CALL+WRITE=8.9% FAIL |
| L5 | Probing Classifiers | Learning | P2 | **FAIL** | Reentrancy Δ=-0.0069 (Phase2 vs Phase1); IntegerUO Phase1=0.419 | Max+mean [512] pooling fix reveals IntegerUO signal (was 0.114 with mean-only); Reentrancy still 0% Phase2 gain |
| L6 | Counterfactual Contracts | Learning | P2 | **FAIL (1/4)** | UnusedReturn PASS (+0.122); Reentrancy/IntegerUO/Timestamp FAIL | Model cannot detect CEI violation, unchecked overflow, or timestamp branching on novel minimal contracts |
| L7 | Calibration & Size Analysis | Learning | P2 | **PENDING** | requires checkpoint | — |
| L8 | Permutation Importance | Learning | P2 | **PENDING** | requires checkpoint | — |
| L9 | Attention Rollout | Learning | P2 | **FAIL** | delta = −0.00654 (safe > vulnerable CW attribution) | Relative-rank criterion: safe contract has higher mean CALL/WRITE attribution than vulnerable; original PASS retracted |
| L10 | Training Ablation Commands | Learning | P2 | **PASS** | 12 commands generated | Ablation scaffold created; CONTAINS/CONTROL_FLOW predicted highest impact |

**PASS: 5, FAIL: 14, COMPLETE: 10 (incl. L3 ARCH-N/A, L4 rerun), PENDING: 0**  
*(B1–B4 complete — see §10)*

---

## 4. Layer 1 — Structural Analysis

This layer validates that the data reaching the model is correct and that known vulnerability patterns are geometrically representable in the graph.

### 4.1 EXP-S1: Structural Trace Audit

**Status: BLOCKED** (test contract portion) / **PARTIAL** (val-split portion)

The test contract portion failed because `solc` is not installed in the Python venv. The val-split pattern detection ran independently.

Val-split pattern results:
| Class | Positive contracts | Pattern found | Pattern rate | Pass |
|-------|-------------------|---------------|--------------|------|
| IntegerUO | 27 | 25 | 92.6% | Yes |
| Timestamp | 1 | 1 | 100.0% | Yes |
| MishandledException | 13 | 7 | 53.9% | Yes |
| Reentrancy | 7 | 1 | 14.3% | No |
| UnusedReturn | 2 | 0 | 0.0% | No |

The Reentrancy pattern rate of 14.3% is unexpectedly low and likely reflects the CEI-reachability limitation found in EXP-E1: only 37.7% of Reentrancy-positive contracts have a direct CEI path via CALL_ENTRY + WRITES within k=8 hops.

**Fix required:** `pip install solc-select && solc-select upgrade && solc-select use 0.8.25`

### 4.2 EXP-S2: Edge Enrichment Ratio

**Status: FAIL**

This experiment tested whether specific edge types were statistically enriched in vulnerability-positive contracts vs. the overall baseline. All 8 named checks failed the 1.3× enrichment threshold.

The primary reason is that the two most structurally important edge types (CONTAINS, CONTROL_FLOW) are near-ubiquitous:

| Edge Type | Baseline % | Description |
|-----------|-----------|-------------|
| CONTAINS (5) | 99.6% | Present in essentially all contracts |
| CONTROL_FLOW (6) | 99.6% | Present in essentially all contracts |
| READS (1) | 95.1% | Near-universal |
| WRITES (2) | 95.6% | Near-universal |

When baseline prevalence is 99.6%, a 1.3× enrichment ratio would require 100%+ occurrence in positives — mathematically impossible. This is a threshold calibration failure, not a data failure.

Edges with actual discriminative potential (selected by high enrichment ratio *and* lower baseline):

| Edge | Class | Enrichment ratio |
|------|-------|-----------------|
| EMITS (3) | UnusedReturn | 15.46× |
| EMITS (3) | Reentrancy | 3.46× |
| CALL_ENTRY (8) | Timestamp | 1.41× |
| RETURN_TO (9) | Timestamp | 1.58× |
| RETURN_TO (9) | UnusedReturn | 1.53× |
| CALL_ENTRY (8) | UnusedReturn | 1.40× |

**EMITS edge note:** EMITS has only 12 total edges across 41,577 contracts (baseline: 0.051%). Despite high ratios, this provides zero practical signal due to near-total absence in the corpus.

**CONTROL_FLOW:** Enrichment ratio = 1.004× for Reentrancy and 0.9995× for IntegerUO. The control-flow graph is universally present and provides no class-discriminative structural signal at the edge-presence level. Discriminative power must come from CFG *topology*, not CFG presence.

### 4.3 EXP-S3: Feature Distribution Per Class

**Status: PASS with WARNING** *(re-run 2026-05-31, train split, n=5000)*

Node feature statistics computed across 5,000 train-split contracts for 10 vulnerability classes. Shortcut threshold: d ≥ 1.0. The metric `mean_return_ignored_func` is now correctly computed over FUNCTION nodes only.

| Class | Metric | Cohen's d | Shortcut? |
|-------|--------|-----------|-----------|
| Timestamp | cfg_call_count | **1.592** | **YES** |
| Timestamp | total_cfg_nodes | 1.241 | YES |
| Timestamp | total_nodes | 1.201 | YES |
| UnusedReturn | mean_return_ignored_func | 0.716 | No |
| UnusedReturn | function_count | 0.531 | No |
| Reentrancy | total_nodes | 0.309 | No |
| IntegerUO | function_count | 0.301 | No |

**Timestamp size shortcut (CONFIRMED — train split):** All three size metrics exceed d=1.0. Timestamp-positive contracts are systematically larger in training data. The model's F1=0.329 may be partly driven by size rather than `block.timestamp` detection.

**mean_return_ignored_func (CORRECTED — dead feature finding RETRACTED):** The previous report claimed this feature was "dead — all zeros." This was an artifact of computing it over CFG_NODE_* nodes. `return_ignored` (dim 7) is a function-level feature, intentionally zero on CFG nodes. Re-running over FUNCTION nodes shows real variance: UnusedReturn has the highest signal (d=0.716), consistent with `return_ignored` capturing unchecked return values. No other class exceeds d=0.25 for this metric.

### 4.4 EXP-S4: ICFG-Lite Path Audit

**Status: PASS**

100 val-split contracts were audited for interprocedural call graph connectivity.

| Metric | Result |
|--------|--------|
| Contracts with CALL_ENTRY (type 8) edge | 76/100 (76%) |
| Contracts with full CALL_ENTRY + RETURN_TO chain | 69/100 (69%) |
| Contracts with write-after-return pattern | 69/100 (69%) |

This confirms the graph extractor is correctly encoding interprocedural structure. The CALL_ENTRY and RETURN_TO edges represent the re-entry path in CEI (Checks-Effects-Interactions) patterns. Their presence in 76% of audited contracts confirms the data is structurally rich enough for Reentrancy detection.

The discrepancy with EXP-L2 (edge ablation showing near-zero effect) is therefore not a data gap — the ICFG edges exist but the model has not learned to use them.

### 4.5 EXP-A1: Pooling Node-Type Audit

**Status: PASS**

| Metric | Result |
|--------|--------|
| Contracts audited | 1,859 |
| Contracts with ≥1 FUNCTION-like node | 100% (pct_has_funclike_node = 100.0) |
| Fallback pooling triggered | 0% (pct_fallback = 0.0) |
| Mean FUNCTION-like nodes per contract | 20.20 |
| Mean FUNCTION-like fraction of all nodes | 17.73% |

Node type breakdown (absolute counts across 1,859 contracts):
- FUNCTION: 31,243
- MODIFIER: 2,778
- CONSTRUCTOR: 2,690
- FALLBACK: 844
- RECEIVE: 0 (absent from corpus)

The pooling step in the GNN (which aggregates over FUNCTION-type nodes) never falls back to all-node mean pooling. The absence of RECEIVE nodes is consistent with older Solidity contracts in the corpus predating EIP-2315.

### 4.6 EXP-A2: CFG Feature Inheritance

**Status: FAIL (code bug fixed 2026-05-31, data gap remains)**

The experiment searched for CONTAINS-linked (function→CFG_NODE) parent/child node pairs and checked feature consistency. 0 graphs with CFG parents were found across 470 audited contracts.

**Root cause (identified 2026-05-31):** `_CONTAINS_EDGE = 0` was hardcoded, matching CALLS (type 0) instead of CONTAINS (type 5). The code was searching for CALLS edges between function nodes, not CONTAINS edges. This bug has been fixed: `_CONTAINS_EDGE = EDGE_TYPES["CONTAINS"]` (=5).

After the fix, the underlying data question remains: do stored graphs have CONTAINS (type 5) edges between FUNCTION nodes and CFG_NODE children? EXP-A2 should be rerun to get a valid inheritance coverage rate.

---

## 5. Layer 2 — Expressivity Analysis

This layer tests whether the GNN architecture is theoretically capable of solving the classification task, independent of what it has learned.

### 5.1 EXP-E1: K-Hop Receptive Field

**Status: FAIL**

Three analyses were run:

**Analysis 1: CEI Reachability** — Can the GNN propagate information from a CFG_CALL node to a CFG_WRITE node (the CEI check-effect pair) via Phase 2 edges within k=8 hops?

| Group | Reachability rate at k=8 |
|-------|--------------------------|
| Reentrancy-positive (n=199) | **37.7%** |
| Reentrancy-negative (n=187) | **26.7%** |
| Pass criterion | ≥50% positive reach |

The 37.7% rate is below the 50% pass threshold. This means for the majority of Reentrancy-positive contracts, the GNN's k=8 hop depth is insufficient to connect the call site to the state write through Phase 2 edges alone. The vulnerability pattern involves paths that are either longer than 8 hops or require traversing CONTAINS/REVERSE_CONTAINS edges (Phase 1/3) to reach the relevant nodes.

**Analysis 2: Function Aggregation** — **(redesigned 2026-05-31)**

Original question: "Can FUNCTION nodes reach ≥50% of ALL CFG nodes via Phase 3 edges?" used `REVERSE_CONTAINS` (type 7), which is **runtime-only** and never stored in .pt graph files. BFS found no REVERSE_CONTAINS edges → 0.1% was expected and uninformative.

Redesigned question: "What fraction of FUNCTION nodes have ≥1 CFG_NODE child via CONTAINS (type 5) within 1 hop?" This directly tests CFG extraction completeness. Rerun needed with redesigned script.

**Analysis 3: Contract Coverage** — **(redesigned 2026-05-31)**

Original question: "Can CONTRACT nodes reach FUNCTION nodes within 2 Phase 1 hops?" No CONTRACT→FUNCTION edge type exists in v8 schema (CONTAINS=5 goes FUNCTION→CFG_NODE, not CONTRACT→FUNCTION). Zero result was guaranteed by design, not a data gap.

Redesigned question: "What fraction of FUNCTION nodes can reach ≥1 other FUNCTION node via CALLS within 2 hops?" Tests intra-contract call graph connectivity. Rerun needed with redesigned script.

**Interpretation (updated):** Analysis 1 (CEI reachability 37.7%) is valid and meaningful. Analyses 2 and 3 had structural design bugs — redesigned (2026-05-31), rerun needed for meaningful results. The CEI reachability finding explains why the model struggles with Reentrancy: the relevant structural evidence does not fit within k=8 Phase 2 hops for 62% of contracts.

### 5.2 EXP-E2: WL Distinguishability

**Status: PASS**

50 val-split contracts per class were sampled. All pairwise cross-class comparisons were tested for WL-distinguishability at rounds 1–8.

| Class | Pairs tested | Final collision rate | Pass |
|-------|-------------|---------------------|------|
| Reentrancy | 45 | 11.11% | Yes |
| IntegerUO | 44 | 4.55% | Yes |
| Timestamp | 44 | **0.00%** | Yes |
| CallToUnknown | 45 | 6.67% | Yes |

All collision rates are stable across rounds 1–8 (no further graph equivalences emerge after round 1). This means the graph structures are WL-distinguishable from round 1 onward — the model sees distinct input structures per class.

**Key implication:** The model's failure to learn Phase 2 features is not because the graphs are indistinguishable. The architecture has the expressive power to distinguish them; the learning dynamics simply haven't exploited this capacity.

Timestamp's 0% collision rate is consistent with the size shortcut finding in EXP-S3: Timestamp-positive contracts are structurally distinct from all other contracts by virtue of their size alone, making them trivially WL-distinguishable.

### 5.3 EXP-E3: Message Propagation Simulation

**Status: FAIL (expected)**

100 reentrancy-positive and 100 reentrancy-negative contracts were run through the GNN with random weights. Cosine similarity of Phase 2 activations was measured.

| Measurement | Positive | Negative |
|-------------|----------|---------|
| Phase 1 CONTROL_FLOW cosine | 0.9937 | 0.9941 |
| Phase 2 CONTROL_FLOW cosine | 0.9119 | 0.9082 |
| Phase 1 CALL_ENTRY cosine | 0.8040 | 0.7894 |
| Phase 2 CALL_ENTRY cosine | 0.6565 | 0.6536 |

CALL_ENTRY Phase 2 minus Phase 1 differential (positive): −0.1475 (threshold for pass: ≥+0.02).

This result is the expected baseline: random weights produce no differential activation. The experiment was designed to be re-run with the trained checkpoint, which would show whether Phase 2 CALL_ENTRY activations diverge between positive and negative contracts. That re-run is still pending (EXP-L3 and EXP-L4).

### 5.4 EXP-E4: Direction Sensitivity

**Status: FAIL** *(re-run 2026-05-31 — extended to 4 edge types)*

92 Reentrancy pos/neg pairs tested under directed vs. undirected WL for each of 4 Phase 2 edge types independently.

| Edge Type | ID | Directed | Undirected | Diff | Status |
|-----------|----|----------|------------|------|--------|
| CONTROL_FLOW | 6 | 89.1% | 89.1% | 0.0% | FAIL |
| DEF_USE | 10 | 89.1% | 89.1% | 0.0% | FAIL |
| CALL_ENTRY | 8 | 89.1% | 89.1% | 0.0% | FAIL |
| RETURN_TO | 9 | 89.1% | 89.1% | 0.0% | FAIL |

The original run (2026-05-30) tested CONTROL_FLOW only and found 0% difference. This re-run confirms the same result for all four Phase 2 edge types. "Edge direction adds no discriminative power" now applies to the complete Phase 2 edge set, not just CONTROL_FLOW.

**Implication:** The 89.1% base WL distinguishability is unchanged across all tests — structural content is high regardless of direction. The key discriminative information is in node feature distributions (type, count), not edge direction. This does not rule out directional attention learning in GAT (which can occur even when WL hashes are identical), but at the structural topology level, direction is not the primary separator.

---

## 6. Layer 3 — Learning Analysis

This layer examines what the trained model has actually learned, using the Run 4 ep32 checkpoint.

### 6.1 EXP-A3: JK Entropy Logging

**Status: PASS**

JK attention weights are 3-way softmax scores (Phase 1, Phase 2, Phase 3). A perfectly uncertain routing has entropy = log(3) = 1.0986 nats. The λ=0.005 entropy regulariser was added in Run 4 to prevent phase collapse.

| Metric | Value |
|--------|-------|
| Entropy range (47 epochs) | 1.0935 – 1.0986 |
| Mean entropy | 1.0973 |
| Max possible (log(3)) | 1.0986 |
| Epochs parsed | 47 (ep1–ep48) |

Selected epoch progression of phase weights (mean across contracts):

| Epoch | Phase 1 | Phase 2 | Phase 3 | Entropy |
|-------|---------|---------|---------|---------|
| 1 | 0.332 | 0.328 | 0.340 | 1.0985 |
| 14 | 0.356 | 0.335 | 0.309 | 1.0969 |
| 26 | 0.303 | 0.331 | 0.366 | 1.0956 |
| 32 | 0.309 | 0.322 | 0.369 | 1.0957 |
| 37 | 0.305 | 0.313 | 0.381 | 1.0935 |
| 47 | 0.312 | 0.320 | 0.367 | 1.0960 |

**Trend observation:** Across Run 4, Phase 3 weight progressively increases (0.340 at ep1 → 0.381 at ep37 → 0.367 at ep47) while Phase 2 progressively decreases (0.328 → 0.313). The regulariser prevents outright collapse but does not prevent the model from gradually shifting weight toward Phase 3.

The ep37 epoch shows the sharpest Phase 3 peak (0.381) and lowest Phase 2 trough (0.313), with entropy at 1.0935 — the global minimum observed. The entropy regulariser kept Phase 2 alive but did not force the model to *learn* from Phase 2 edges.

**Conclusion:** JK entropy is healthy (no collapse), but the entropy regulariser does not ensure that Phase 2 carries discriminative signal. It only ensures Phase 2 contributes to the mixture. Routing weight ≠ learned utility.

### 6.2 EXP-L1: JK Weight Analysis

**Status: FAIL**

936 val-split samples were processed (64 skipped). Hypotheses tested:
- Reentrancy → Phase 2 dominant (CEI requires CFG)
- IntegerUO → Phase 2 dominant (overflow requires DFG)
- Timestamp → Phase 1 dominant (purely structural)
- UnusedReturn → Phase 1 dominant (purely structural)

**Result: 0/4 hypotheses confirmed.**

Per-class JK weight breakdown (at ep32 best checkpoint):

| Class | Phase 1 | Phase 2 | Phase 3 | Dominant | n positives |
|-------|---------|---------|---------|----------|-------------|
| CallToUnknown | 0.3331 | 0.3213 | **0.3457** | Phase 3 | 58 |
| DenialOfService | 0.3368 | 0.3232 | **0.3400** | Phase 3 | 4 |
| ExternalBug | 0.3279 | 0.3205 | **0.3516** | Phase 3 | 64 |
| GasException | 0.3357 | 0.3237 | **0.3406** | Phase 3 | 108 |
| IntegerUO | 0.3343 | 0.3237 | **0.3420** | Phase 3 | 314 |
| MishandledException | 0.3318 | 0.3213 | **0.3469** | Phase 3 | 90 |
| Reentrancy | 0.3325 | 0.3226 | **0.3449** | Phase 3 | 91 |
| Timestamp | 0.3111 | 0.3178 | **0.3711** | Phase 3 | 10 |
| TOD | 0.3338 | 0.3231 | **0.3431** | Phase 3 | 73 |
| UnusedReturn | 0.3338 | 0.3235 | **0.3427** | Phase 3 | 24 |

Every single vulnerability class has Phase 3 as the dominant phase. Phase 2 is the lowest-weighted phase for 9 of 10 classes (exception: Timestamp where Phase 1 = 0.311 is lowest, but Phase 2 = 0.318 is still below Phase 3 = 0.371).

**The Phase 2 deficit is consistent and global.** It is not class-specific: Reentrancy (CFG-dependent) and GasException (structural) both show Phase 2 suppression to essentially the same degree (Reentrancy: 0.3226, GasException: 0.3237).

**Entropy note:** Mean per-class entropy is 0.3333... = uniform weight, and the JK entropy from the log field is 1.095–1.099 (near max). The phase routing is nearly uniform, with Phase 3 having a ~0.015–0.030 unit advantage over Phase 2. This small but consistent advantage compounds: over 8 transformer-style attention aggregations, small weight differences mean Phase 3 information is consistently selected.

### 6.3 EXP-L2: Edge Ablation (Inference)

**Status: FAIL** *(structural ablation added 2026-06-01)*

190 val-split samples were processed. Two ablation methods were measured: (1) embedding-zero (original, soft) and (2) structural removal (new, hard — physically removes edges from `edge_index`).

**Embedding ablation results (soft — zeroes edge type embedding, edges remain in graph):**

| Edge Ablated | Class | Δ logit | Threshold | Result |
|-------------|-------|---------|-----------|--------|
| CONTROL_FLOW (6) | Reentrancy | −1.05 × 10⁻⁵ | 0.03 | FAIL |
| CALL_ENTRY (8) | Reentrancy | −5.18 × 10⁻⁷ | 0.03 | FAIL |
| DEF_USE (10) | IntegerUO | −4.32 × 10⁻¹⁰ | 0.02 | FAIL |
| EMITS (3) | — | +2.06 × 10⁻⁸ | — | INFO |

**Combined CFG drop (embedding, CONTROL_FLOW + CALL_ENTRY):** 1.11 × 10⁻⁶ — five orders of magnitude below threshold.

**Structural ablation results (hard — removes edges from edge_index):**

| Edge Removed | Reentrancy Δ | Timestamp Δ | IntegerUO Δ |
|-------------|-------------|-------------|-------------|
| CONTROL_FLOW (6) | **+0.0203** | **+0.1629** | −0.0031 |
| CALL_ENTRY (8) | **+0.0101** | +0.0195 | +0.0041 |
| RETURN_TO (9) | **+0.0182** | +0.0001 | −0.0037 |
| DEF_USE (10) | ≈ 0 | ≈ 0 | ≈ 0 |

**Combined Phase 2 structural drop (Reentrancy):** +0.0121 (POSITIVE)

**Embedding vs structural ratio:** 10,944×

**Critical finding — Phase 2 edges SUPPRESS Reentrancy:** All three active Phase 2 edge types (CONTROL_FLOW, CALL_ENTRY, RETURN_TO) show positive deltas for Reentrancy when removed structurally. Removing CFG edges *increases* the Reentrancy prediction. This means Phase 2 edges are negatively contributing to Reentrancy scores — the model has learned to treat Phase 2 CFG connectivity as evidence AGAINST Reentrancy, likely because smaller contracts (with less CFG structure) happen to be over-represented as Reentrancy-positive in the training distribution.

**Critical finding — Timestamp strongly suppressed by CONTROL_FLOW:** Removing CONTROL_FLOW edges structurally increases Timestamp prediction by +0.1629 — the largest structural ablation effect observed. Phase 2 CFG edges are actively suppressing Timestamp predictions, consistent with the size-shortcut hypothesis: the model has learned that Timestamp-positive contracts are large AND complex (high CFG), and uses CF structure as a downward signal.

**DEF_USE has zero structural effect** for all classes — even when edges are physically removed, no prediction changes. DEF_USE is architecturally present but computationally disconnected from the final prediction.

**Practical consequence:** An attacker who strips all CFG edges would see the Reentrancy prediction *increase* — the model would become *more* likely to flag a Reentrancy vulnerability. Phase 2 is working against Reentrancy detection, not for it.

### 6.4 EXP-A4: Aux Eye Contribution

**Status: FAIL**

470 val-split samples were used. Each of the three sub-classifiers (GNN eye, Transformer eye, Fused eye) and the main ensemble head were evaluated independently.

**F1 by classifier head:**

| Class | Baseline F1 | GNN F1 | TF F1 | Fused F1 | Main F1 | GNN useful? |
|-------|------------|--------|-------|----------|---------|-------------|
| CallToUnknown | 0.142 | **0.255** | 0.264 | 0.295 | 0.286 | Yes |
| DenialOfService | 0.025 | 0.000 | 0.800 | 1.000 | 1.000 | No |
| ExternalBug | 0.117 | 0.000 | 0.000 | 0.000 | 0.200 | No |
| GasException | 0.190 | 0.000 | 0.036 | 0.131 | 0.231 | No |
| IntegerUO | 0.401 | **0.548** | 0.728 | 0.767 | 0.755 | Yes |
| MishandledException | 0.158 | 0.000 | 0.000 | 0.000 | 0.000 | No |
| Reentrancy | 0.170 | 0.182 | 0.389 | 0.444 | 0.438 | No (marginal) |
| Timestamp | 0.017 | **0.286** | 0.286 | 0.333 | 0.571 | Yes |
| TOD | 0.123 | 0.000 | 0.000 | 0.000 | 0.059 | No |
| UnusedReturn | 0.052 | 0.000 | 0.000 | 0.000 | 0.300 | No |

**AUC-ROC by classifier head:**

| Class | Main | GNN | TF | Fused |
|-------|------|-----|-----|-------|
| CallToUnknown | 0.846 | 0.757 | 0.815 | 0.845 |
| DenialOfService | 1.000 | 0.648 | 1.000 | 1.000 |
| ExternalBug | 0.870 | 0.768 | 0.835 | 0.866 |
| GasException | 0.832 | 0.676 | 0.766 | 0.839 |
| IntegerUO | 0.909 | 0.727 | 0.872 | 0.909 |
| MishandledException | 0.834 | 0.735 | 0.760 | 0.858 |
| Reentrancy | 0.884 | 0.788 | 0.831 | 0.880 |
| Timestamp | 0.992 | 0.989 | 0.990 | 0.993 |
| TOD | 0.859 | 0.748 | 0.798 | 0.863 |
| UnusedReturn | 0.965 | 0.929 | 0.962 | 0.970 |

**Key observations:**

1. **GNN useful for only 3 classes:** The GNN eye alone beats the random baseline by ≥5 F1 percentage points for CallToUnknown (0.255 vs 0.142), IntegerUO (0.548 vs 0.401), and Timestamp (0.286 vs 0.017).

2. **GNN useless for 7 classes:** For DenialOfService, ExternalBug, GasException, MishandledException, Reentrancy, TOD, and UnusedReturn, the GNN eye produces F1 = 0 on held-out val data (the threshold was optimised on a separate set). The GNN alone has no discriminative power for these classes.

3. **Reentrancy is the most concerning case:** Reentrancy is theoretically most dependent on CFG structure (CEI pattern), yet the GNN eye achieves F1 = 0.182 — barely above baseline. The Transformer eye (F1 = 0.389) and Fused eye (F1 = 0.444) substantially outperform the GNN, suggesting GraphCodeBERT's token attention is detecting Reentrancy patterns more effectively than the structural GNN.

4. **Main head benefits from ensemble:** The main head (concatenating all three eyes into [B, 384] → Linear → 10 classes) outperforms all individual eyes for every class except DenialOfService (where Fused = 1.000 = Main). The ensemble is working — the complementarity of GNN and TF eyes adds value even when the GNN eye alone is weak.

5. **Timestamp AUC-ROC = 0.989 for GNN eye:** The high AUC-ROC for Timestamp (GNN: 0.989, Main: 0.992) confirms the size-shortcut hypothesis from EXP-S3. The GNN can rank Timestamp contracts almost perfectly by AUC-ROC using purely structural features, consistent with Timestamp-positive contracts being structurally distinctive (~1.70× larger in val by node count).

### 6.5 EXP-L3: Attention Visualization

**Status: ARCHITECTURAL N/A** *(audit 2026-06-01 — original PASS retracted)*

> **Audit note:** The original PASS status reported "100% CF fraction in top-20 attention edges" as evidence of learned selective attention. This is a false positive. conv3 is wired to a CF-only edge_index — it physically cannot attend to any non-CF edge. The 100% result is a mathematical certainty, not a learned property. The PASS was retracted 2026-06-01.

**Real headline finding:** All GAT attention weights in conv3 = **1.0 (uniform)**. The model has learned no selective attention within the control-flow graph. Every CF edge receives identical attention regardless of semantic importance.

11 hand-crafted test contracts were processed. Results:

| Metric | Value |
|--------|-------|
| CF fraction in top-20 edges | 100% (all 11 contracts) |
| CFG_NODE_CALL → CFG_NODE_WRITE in top edges | 0/11 contracts |
| Attention weight range | 1.000 – 1.000 (uniform) |

The CEI pattern (CFG_CALL → WRITE transition that defines Reentrancy) is absent from top-attention edges for all contracts, including the reentrancy-vulnerable ones. The model applies mean aggregation over all CF neighbors, not selective aggregation.

**Implication:** Combined with EXP-L1 (Phase 2 JK weight lowest), EXP-L2 (CFG ablation ≈ 0 for embedding, POSITIVE for structural), and EXP-L4 (external_call_count dominates over CFG_NODE_CALL for Reentrancy), this confirms the Phase 2 CFG layers have not learned discriminative attention within the control-flow graph.

See `docs/interpretability/exp_l3_attention_visualization.md` for full details.

---

### 6.6 EXP-L4: Gradient Saliency

**Status: COMPLETE** *(rerun 2026-06-01 with correct feature names from graph_schema)*

> **Fix note:** The original run (2026-05-30) used hardcoded stale feature names (pre-v8 labels) for dims 3–9. The script was fixed to import FEATURE_NAMES from `graph_schema.py`. The rerun confirms the same qualitative conclusions but with correct feature labels.

492–500 val-split contracts per class (48 for Timestamp — all positives used). Gradient saliency computed via backward pass from class logit through BCEWithLogitsLoss.

**Per-class top-3 feature dimensions (correct v8 feature names):**

| Class | Rank 1 | Rank 2 | Rank 3 |
|-------|--------|--------|--------|
| CallToUnknown | external_call_count (21.8%) | complexity (11.0%) | call_target_typed (8.9%) |
| DenialOfService | external_call_count (23.9%) | complexity (10.1%) | uses_block_globals (9.3%) |
| ExternalBug | external_call_count (22.7%) | complexity (11.4%) | visibility (9.3%) |
| GasException | external_call_count (23.4%) | complexity (10.7%) | visibility (8.9%) |
| IntegerUO | external_call_count (22.2%) | complexity (10.2%) | uses_block_globals (9.2%) |
| MishandledException | external_call_count (23.9%) | complexity (10.6%) | visibility (8.8%) |
| Reentrancy | external_call_count (23.0%) | complexity (10.9%) | visibility (8.8%) |
| Timestamp | external_call_count (21.4%) | complexity (10.9%) | uses_block_globals (10.0%) |
| TOD | external_call_count (22.5%) | complexity (11.2%) | visibility (8.9%) |
| UnusedReturn | external_call_count (21.3%) | complexity (10.8%) | visibility (10.0%) |

**Per-class top-3 node types:**

| Class | Rank 1 | Rank 2 | Rank 3 |
|-------|--------|--------|--------|
| CallToUnknown | FUNCTION (49.9%) | CFG_NODE_OTHER (22.2%) | CFG_NODE_WRITE (4.6%) |
| Reentrancy | FUNCTION (46.5%) | CFG_NODE_OTHER (25.4%) | CFG_NODE_CALL (4.5%) |
| Timestamp | FUNCTION (48.6%) | CFG_NODE_OTHER (20.4%) | CFG_NODE_READ (6.3%) |
| TOD | FUNCTION (46.5%) | CFG_NODE_OTHER (24.1%) | CFG_NODE_WRITE (4.7%) |
| UnusedReturn | FUNCTION (46.8%) | CFG_NODE_OTHER (27.8%) | CFG_NODE_CALL (4.7%) |

**Specific pass/fail checks:**

| Check | Result |
|-------|--------|
| Timestamp: uses_block_globals (dim 2) ≥ 20% | **FAIL** — 10.0% |
| Reentrancy: CFG_NODE_CALL + CFG_NODE_WRITE ≥ 20% | **FAIL** — 8.9% combined |

**Key conclusion — global sensitivity artifact:** `external_call_count` dominates for ALL 10 vulnerability classes with nearly identical attribution (21.3%–23.9%). `complexity` is universally second (10.1%–11.4%). Neither is class-discriminative — they measure contract size/complexity which correlates with all vulnerability labels. The saliency signal is driven by global contract properties, not class-specific structural patterns.

This means the gradient does not discriminate between classes at the feature level: the GNN's backward signal for Reentrancy looks essentially the same as for Timestamp or GasException. The class-specific information is encoded elsewhere (likely in the three-eye classifier head, or in the Transformer path).

See `ml/interpretability_results/l4_feature_saliency_heatmap.png` for the full heatmap.

---

### 6.7 EXP-L6: Counterfactual Contracts

**Status: FAIL (1/4 pairs pass)** *(run 2026-05-31 after solc-select fix)*

All 4 test contract pairs compiled successfully after fixing the solc broken symlink and running `solc-select use 0.8.20`.

| Pair | Vuln score | Safe score | Delta | Result |
|------|-----------|-----------|-------|--------|
| CEI_reentrancy | 0.2962 | 0.3032 | −0.0071 | **FAIL** |
| integer_uo | 0.4146 | 0.4718 | −0.0642 | **FAIL** |
| timestamp | 0.0399 | 0.0399 | +0.0000 | **FAIL** |
| unused_return | 0.1695 | 0.0481 | **+0.1214** | PASS |

**Critical finding:** The model does not correctly distinguish vulnerable from safe contracts for 3 of 4 pairs:
- **Reentrancy:** Safe contract scores HIGHER (model does not detect CEI violation)
- **IntegerUO:** Safe contract scores HIGHER (0.4718 vs 0.4146 — model responds to corpus-level complexity, not `unchecked {}` syntax)
- **Timestamp:** Both contracts score identically (0.0399) — pure size shortcut, no temporal branching detection
- **UnusedReturn:** Only PASS — 12.1pp gap in correct direction, consistent with EXP-B4 finding that `return_ignored` has some (but not dominant) saliency

This is the strongest evidence that the model's predictions for most vulnerability classes do not trace to structurally correct reasons. The Run 5 CEI auxiliary loss (Interp-2) directly targets the Reentrancy failure.

### 6.8 EXP-L10: Training Ablation Commands

**Status: PASS (scaffold only)**

12 training ablation commands were generated covering all 11 edge types. The `--ablate-edge-type` flag is not yet implemented in `train.py`. The experiment recommends using EXP-L2 (inference ablation) as a faster proxy.

Predicted impact ranking from L10 scaffold (based on architectural analysis, not measured):

| Edge type | Predicted F1 impact |
|-----------|-------------------|
| CONTAINS (phase 1/3) | −0.03 to −0.06 |
| REVERSE_CONTAINS (phase 3) | −0.04 to −0.08 |
| CONTROL_FLOW (phase 2) | −0.05 to −0.10 (predicted) |
| CALL_ENTRY (phase 2) | −0.03 to −0.06 (predicted) |

Note: the predicted CONTROL_FLOW impact of −0.05 to −0.10 is **directly contradicted** by EXP-L2's measured impact of <0.00001. The training ablation experiment (full retraining without an edge type) would give different results from inference ablation, but the inference result is already very damning for CFG utility.

---

## 7. Root Cause Analysis

### 7.1 The Central Question

Why does Phase 2 (control-flow / interprocedural / data-flow) have the lowest JK weight and near-zero ablation effect, despite CONTROL_FLOW and CALL_ENTRY edges being present in 99.6% and 64% of contracts respectively?

### 7.2 Evidence Chain

```
[DATA] CFG nodes present in 99.5% of contracts              (Cache audit)
[DATA] CALL_ENTRY edges in 64.2% of contracts               (Cache audit)
[DATA] Full CALL_ENTRY+RETURN_TO chain in 69% of val-100   (EXP-S4)
[MODEL] JK Phase 2 weight = 0.322 (lowest, consistently)   (EXP-L1)
[MODEL] CFG embedding ablation = 1.11e-6 (near zero)        (EXP-L2)
[MODEL] CFG structural ablation → POSITIVE Reentrancy Δ     (EXP-L2)
[MODEL] GNN eye useful for only 3/10 classes                (EXP-A4)
       ↓
CONCLUSION: CFG structure is present but not exploited
```

### 7.3 Candidate Causes

**Cause A: Gradient vanishing through Phase 2 layers**

Phase 2 layers (L3, L4, L5) process CFG_CALL → CFG_WRITE paths that can require many hops. EXP-E1 showed that only 37.7% of Reentrancy-positive contracts have a CEI path reachable within k=8 hops using Phase 2 edges. For the 62.3% of contracts where the path is longer, Phase 2 contributes nothing to the gradient signal for Reentrancy. The gradient from the Reentrancy label flows backward through the classifier and attention layers, but the Phase 2 GNN layers receive sparse, fragmented gradient because the vulnerability-relevant topology is geometrically inaccessible.

**Cause B: Structural shortcut exploitation**

The JK weight trend in EXP-A3 shows Phase 3 weight increasing monotonically from ep1 (0.340) to ep37 (0.381). This is consistent with the model discovering early in training that CONTAINS/REVERSE_CONTAINS hierarchy provides a reliable, dense, low-noise signal that generalises well. Once this shortcut is established, Phase 2 is a higher-difficulty, lower-reward pathway that the optimizer deprioritises.

CONTAINS and REVERSE_CONTAINS edges are present in 99.6% of contracts. Every contract has a CONTRACT → FUNCTION → CFG-node hierarchy. This hierarchy is a consistent structural signal that correlates with contract complexity, which in turn correlates weakly with many vulnerability classes (larger contracts have more code and thus more possible vulnerabilities). The model may be exploiting contract complexity encoded in structural depth rather than specific vulnerability patterns.

**Cause C: Insufficient Phase 2-specific supervision**

The training loss (AsymmetricLoss on the final 10-class output) provides no signal specifically for whether Phase 2 edges are being used. The JK entropy regulariser (λ=0.005) ensures Phase 2 is not collapsed to zero weight, but it does not reward Phase 2 for carrying discriminative signal. The model can satisfy the entropy regulariser by routing Phase 2 edges through in a way that contributes noise (or near-zero signal) to the JK aggregation, and the regulariser is satisfied.

An explicit auxiliary loss — for example, a binary Reentrancy prediction on Phase 2 features alone, with a separate loss weight — would force Phase 2 to become discriminative for Reentrancy rather than just present.

**Cause D: Training convergence ordering**

Phase 1 and Phase 3 (structural edges: CONTAINS, INHERITS, CALLS) provide large, consistent, locally-dense signal. A FUNCTION node in Phase 1 can immediately aggregate features from all its child CFG nodes via CONTAINS edges in 1 hop. Phase 2 requires multi-hop traversal across function boundaries via CALL_ENTRY/RETURN_TO. In early epochs, Phase 1/3 converge first and dominate the gradient signal, locking in a parameter regime where Phase 2 improvement offers diminishing returns.

### 7.4 Most Likely Combined Cause

The evidence is most consistent with a combination of B (structural shortcut) and C (no Phase 2 supervision), with A (gradient vanishing) as a contributing factor:

1. Phase 1/3 structural hierarchy provides a sufficient baseline signal (cause B).
2. Without explicit auxiliary supervision, Phase 2 never gets a direct training signal to become discriminative (cause C).
3. For classes where CFG paths are geometrically long (>8 hops), gradient vanishing ensures Phase 2 remains uninformative (cause A).
4. The entropy regulariser keeps Phase 2 "alive" but not useful (prevents cause but doesn't fix B or C).

### 7.5 Why This Matters for Specific Vulnerabilities

| Vulnerability | Why CFG matters | Current GNN status |
|--------------|-----------------|-------------------|
| Reentrancy | Requires CEI path: call site → external call → state write | GNN eye F1=0.182, TF eye F1=0.389 |
| ExternalBug | Requires cross-function call graph | GNN eye F1=0.000 |
| TOD | Requires state read/write ordering | GNN eye F1=0.000 |
| IntegerUO | Requires data-flow from arithmetic to reachable state | GNN eye F1=0.548 |

Reentrancy, ExternalBug, and TransactionOrderDependence are the three classes most dependent on CFG/ICFG structure. They are also among the classes with the lowest ep32 F1 values: Reentrancy 0.182 (GNN eye), ExternalBug 0.000 (GNN eye), TOD 0.000 (GNN eye). The correlation is direct.

---

## 8. Capacity Ceiling Analysis

### 8.1 Why F1 = 0.3362 Is a Ceiling

Run 4 was killed at ep44 after F1 locked at 0.31–0.34 for 12 consecutive epochs. The best checkpoint is ep32 at F1 = 0.3362. This section explains why F1 could not improve further.

**Evidence of ceiling:**

| Metric | Value |
|--------|-------|
| F1 ep21 (new best) | 0.3224 |
| F1 ep25 (new best) | 0.3272 |
| F1 ep32 (final best) | 0.3362 |
| F1 ep33–ep44 (locked) | 0.31–0.34 |
| Patience exhausted at ep44 | 12/30 |

The F1 improvement from ep21 to ep32 (+0.014) came from the GNN prefix warming up (prefix active from ep15, NC-1 Adam reset at warmup end). After ep32, no further structural improvements were available within the current parameterisation.

### 8.2 What Limits the Ceiling

**Factor 1: Phase 2 contributes no discriminative signal (primary limiter)**

For the 7 classes where GNN eye F1 = 0 (ExternalBug, GasException, MishandledException, Reentrancy, TOD, UnusedReturn, DenialOfService), the GNN's contribution to the main head is additive noise rather than signal. The cross-attention fusion and the three-eye linear classifier must learn to suppress GNN contributions for these classes. This is a wasted capacity: 768-dim GNN prefix projections and 256-dim GNN embeddings that add no classification signal.

**Factor 2: Transformer encoder carries the weight of hard classes**

The Transformer eye alone achieves F1 = 0.389 for Reentrancy — more than double the GNN eye (0.182). This means GraphCodeBERT is detecting Reentrancy from token patterns (likely learned to associate `call.value()` or `.call{}` with Reentrancy annotations in the training data), not from graph structure. Token-level detection has inherent limits because it cannot reason about cross-function execution order.

**Factor 3: Class imbalance and small positive sets**

Several classes have very few positive examples in the val split used for EXP-A4 (n=470 samples):
- DenialOfService: 6 positives (1.3%)
- Timestamp: 4 positives (0.9%)
- UnusedReturn: 13 positives (2.8%)

F1 for these classes is highly unstable. The macro-F1 average is dominated by the behaviour on minority classes, making it sensitive to threshold choices and random variation.

**Factor 4: Timestamp size shortcut**

The model's F1 = 0.329 for Timestamp at ep32 relies partially on contract size. If the evaluation set has the same size distribution bias as the training set, the model appears effective. A calibrated evaluation on Timestamp contracts of average size would likely show lower F1.

### 8.3 What Would Break the Ceiling

For F1 to improve beyond 0.34, the model needs to gain discriminative signal from Phase 2 CFG structure. The specific improvements required:

1. Phase 2 must learn to detect CEI patterns for Reentrancy → requires auxiliary loss or path supervision.
2. Phase 2 must detect data-flow chains for IntegerUO/UnusedReturn → requires DEF_USE edges to carry gradient.
3. Timestamp shortcut must be addressed → requires explicit regularisation against contract size or size-stratified evaluation.

Without these changes, adding more training epochs to Run 4 would not improve F1 — the model has exhausted what it can learn from the current training signal.

---

## 9. Actionable Recommendations

Prioritised recommendations for Run 5, in order of expected impact.

### P0: Phase 2 Auxiliary Loss (Highest Expected Impact)

Add an explicit per-phase auxiliary loss that trains Phase 2 representations to be discriminative for CFG-dependent classes:

```python
# Proposed addition to trainer.py
# After GNN forward pass, before CrossAttentionFusion:
gnn_phase2_logits = aux_head_phase2(phase2_embeddings)  # Linear(256, num_classes)
aux_loss_phase2 = criterion(gnn_phase2_logits, labels)
total_loss += phase2_aux_weight * aux_loss_phase2  # weight ~ 0.1
```

This directly forces Phase 2 to carry discriminative signal for all classes, not just when the JK aggregation routes through it. Expected F1 improvement: +0.04 to +0.08 based on the gap between current GNN eye performance and Transformer eye performance for Reentrancy/ExternalBug.

### P0: CEI Path Supervision for Reentrancy

Add a graph-level binary auxiliary task specifically for CEI-path detection. This requires labelling whether a contract has a complete CFG_CALL → external_call → CFG_WRITE chain (EXP-S4 showed 69% of Reentrancy-positive val contracts have this chain). The auxiliary task would penalise Phase 2 layers for failing to distinguish contracts with and without CEI chains.

### P1: Fix Timestamp Size Confound

Add contract-size normalisation to the node features or add an explicit adversarial regulariser against contract size for Timestamp. Alternatively, rebalance the Timestamp training set to match the size distribution of Timestamp-negative contracts (Sol-3 data gating).

### P1: Raise max_nodes Limit

The current max_nodes = 1024 truncates contracts with >1024 nodes. Timestamp-positive contracts have mean 344 nodes but std = 294 — a substantial fraction exceed 1024. Setting max_nodes = 2048 before Run 5 would prevent truncation from silently removing the vulnerability-relevant subgraph for large contracts.

### P1: IMP-D1 Re-extraction

Run `reextract_graphs.py` to rebuild the 41K graphs. This is needed to:
- Correct any max_nodes boundary effects
- Enable EXP-A2 (CFG feature inheritance) to run correctly
- Validate that node type strings match what the experiments expect

### P1: Fix solc-select

```bash
solc-select upgrade && solc-select use 0.8.25
```

This unblocks EXP-L6 (counterfactual contracts) and EXP-S1 (structural trace on test contracts), both of which provide the most interpretable validation of whether the model correctly responds to vulnerability-specific structural changes.

### P2: Add EMITS Edge Extraction

The EMITS edge (event emission) has only 12 occurrences across 41,577 contracts. The graph extractor is either missing event emission nodes or failing to extract EMITS edges from the AST. EMITS has a 15.46× enrichment ratio for UnusedReturn in EXP-S2. Correct extraction could provide a clean structural signal for UnusedReturn detection.

### P2: Gradient Saliency Study (EXP-L4)

Run gradient saliency analysis on the ep32 checkpoint to identify which node features and edge types receive the largest gradients for each vulnerability class. This would confirm or refute the four candidate causes from Section 7.3 by measuring actual gradient flow into Phase 2 layers.

### P2: Run L3–L5 Pending Experiments

EXP-L3 (attention visualisation), EXP-L4 (gradient saliency), and EXP-L5 (probing classifiers) are all pending checkpoint loading. These three experiments together would provide a complete picture of what the trained model has learned. They should be run as a batch against the ep32 checkpoint.

---

## 10. Appendix: Per-Class Analysis

Detailed breakdown for each of the 10 vulnerability classes. For each class: F1 at ep32 (from training logs), GNN eye F1 and AUC-ROC (EXP-A4), JK phase weights (EXP-L1), and structural characteristics (EXP-S2, EXP-S3).

---

### 10.1 CallToUnknown

**F1 at ep32:** ~0.247 (bottom quartile from training log notes)
**Val positives in EXP-A4:** 39/470 (8.3%)
**GNN eye F1:** 0.255 | **TF eye F1:** 0.264 | **Main F1:** 0.286
**GNN AUC-ROC:** 0.757 | **Main AUC-ROC:** 0.846
**JK weights:** Ph1=0.333, Ph2=0.321, Ph3=**0.346**

**Structural profile (EXP-S2):**
- CALLS edge enrichment: 0.972× (no enrichment — CallToUnknown contracts are *less* likely to have CALLS edges than baseline)
- CALL_ENTRY enrichment: 1.047×
- INHERITS enrichment: 1.098×
- Mean total nodes positive: 158.8 vs negative: 124.0 (Cohen's d = 0.260)

**Interpretation:** The GNN eye is marginally useful for CallToUnknown (F1 = 0.255 vs baseline 0.142). The class is characterised by having external call interactions but paradoxically *fewer* CALLS edges than baseline — likely because CallToUnknown involves `delegatecall` or `call()` patterns that may be represented differently in the graph. The model uses Phase 3 structure (hierarchy) as the primary signal; Phase 2 interprocedural edges provide some signal (CALL_ENTRY enrichment 1.047×) but below significance.

**Run 5 priority:** Medium. CEI path supervision would help but is not the primary limiter.

---

### 10.2 DenialOfService

**F1 at ep32:** Volatile (small class)
**Val positives in EXP-A4:** 6/470 (1.3%)
**GNN eye F1:** 0.000 | **TF eye F1:** 0.800 | **Main F1:** 1.000
**GNN AUC-ROC:** 0.648 | **Main AUC-ROC:** 1.000
**JK weights:** Ph1=0.337, Ph2=0.323, Ph3=**0.340**

**Structural profile (EXP-S2):**
- CALLS enrichment: 1.174× (moderate — DoS often involves unbounded loops over caller arrays)
- CALL_ENTRY enrichment: 1.154×
- INHERITS enrichment: 1.141×

**Interpretation:** With only 6 positives in the eval set, F1 = 1.000 for the main head and F1 = 0 for the GNN eye are both unreliable. The perfect main-head score likely reflects the main head memorising a few discriminative features that happen to be present in all 6 val positives. The GNN eye's AUC-ROC = 0.648 (barely above random) confirms the GNN has learned nothing useful for DoS detection.

DenialOfService detection fundamentally requires detecting gas-expensive loops or state-iteration patterns that exhaust the gas limit. This requires DFG-level data flow analysis that the model is not performing.

**Run 5 priority:** Low (too few samples to train reliably — Sol-2 data enrichment first).

---

### 10.3 ExternalBug

**F1 at ep32:** ~0.246 (noted as bottom in training log)
**Val positives in EXP-A4:** 31/470 (6.6%)
**GNN eye F1:** 0.000 | **TF eye F1:** 0.000 | **Main F1:** 0.200
**GNN AUC-ROC:** 0.768 | **Main AUC-ROC:** 0.870
**JK weights:** Ph1=0.328, Ph2=0.320, Ph3=**0.352**

**Structural profile (EXP-S2):**
- CALLS enrichment: 1.146×
- CALL_ENTRY enrichment: 1.229× (highest of all non-Timestamp/UnusedReturn classes for CALL_ENTRY)
- RETURN_TO enrichment: 1.231×
- function_count Cohen's d: 0.651 (ExternalBug contracts have significantly more functions)

**Interpretation:** ExternalBug has the highest CALL_ENTRY enrichment of any standard class (1.229×), and the contracts are significantly larger (more functions, Cohen's d = 0.651). Despite this structural signal, both the GNN eye and TF eye individually produce F1 = 0 — the main head achieves F1 = 0.200 purely through the ensemble fusion. The GNN has AUC-ROC = 0.768 which is above-chance, suggesting some discriminative signal in the ranking, but threshold selection is failing.

ExternalBug requires detecting patterns where a state variable is modified based on the return value of an external contract call that can fail silently. This is exactly the kind of interprocedural analysis that Phase 2 should enable. The 1.229× CALL_ENTRY enrichment is the strongest structural signal available. Phase 2 auxiliary loss is the primary lever for this class.

**Run 5 priority:** High. Phase 2 auxiliary loss should directly benefit ExternalBug.

---

### 10.4 GasException

**F1 at ep32:** ~0.317
**Val positives in EXP-A4:** 55/470 (11.7%)
**GNN eye F1:** 0.000 | **TF eye F1:** 0.036 | **Main F1:** 0.231
**GNN AUC-ROC:** 0.676 | **Main AUC-ROC:** 0.832
**JK weights:** Ph1=0.336, Ph2=0.324, Ph3=**0.341**

**Structural profile (EXP-S2):**
- CALLS enrichment: 1.063×
- CALL_ENTRY enrichment: 1.080×
- function_count Cohen's d: 0.238

**Interpretation:** GasException is one of the most common vulnerability classes (1,960 positives in the full dataset, 55/470 in the eval subset). Despite this, the GNN eye achieves F1 = 0. The AUC-ROC = 0.676 indicates some discriminative signal in the rankings, but below the threshold for useful classification.

GasException involves unhandled failure of `send()` or `transfer()` calls, or gas-expensive operations that cause transactions to revert. The structural signal is weak (enrichment ratios near 1.0× for all edge types), and the main head relies heavily on the Transformer's token-level features (detecting `send()` / `transfer()` keyword patterns).

**Run 5 priority:** Medium. Token-level patterns are the most accessible signal.

---

### 10.5 IntegerUO

**F1 at ep32:** 0.647 (best-performing class per training log)
**Val positives in EXP-A4:** 157/470 (33.4%)
**GNN eye F1:** 0.548 | **TF eye F1:** 0.728 | **Main F1:** 0.755
**GNN AUC-ROC:** 0.727 | **Main AUC-ROC:** 0.909
**JK weights:** Ph1=0.334, Ph2=0.324, Ph3=**0.342**

**Structural profile (EXP-S2):**
- DEF_USE enrichment: 1.009× (essentially zero enrichment)
- function_count Cohen's d: 0.298
- total_cfg_nodes Cohen's d: 0.162

**Interpretation:** IntegerUO is the model's strongest class. The GNN eye F1 = 0.548 (vs baseline 0.401) is the only class where GNN alone is substantially useful. However, the structural analysis reveals a paradox: DEF_USE edges — which should carry the data-flow signal for integer overflow (value → arithmetic → overflow) — show essentially zero class enrichment (1.009×) and near-zero ablation effect (+1.5e-8 for DEF_USE removal on IntegerUO).

The GNN eye's success for IntegerUO is therefore likely driven by structural complexity (larger contracts with more arithmetic operations tend to have IntegerUO) rather than DEF_USE graph traversal. The class's high prevalence (1,960/41K ≈ 4.7% of contracts are IntegerUO-positive) provides dense enough training signal for the GNN to learn some structural correlation.

**Run 5 priority:** Low (already performing well). Monitoring for size confound.

---

### 10.6 MishandledException

**F1 at ep32:** ~0.158
**Val positives in EXP-A4:** 44/470 (9.4%)
**GNN eye F1:** 0.000 | **TF eye F1:** 0.000 | **Main F1:** 0.000
**GNN AUC-ROC:** 0.735 | **Main AUC-ROC:** 0.834
**JK weights:** Ph1=0.332, Ph2=0.321, Ph3=**0.347**

**Structural profile (EXP-S2):**
- CALLS enrichment: 1.110×
- CALL_ENTRY enrichment: 1.146×
- function_count Cohen's d: 0.296

**Interpretation:** MishandledException is an anomalous case: the main head achieves F1 = 0 on the eval set despite having AUC-ROC = 0.834. This indicates the model has moderate discriminative capability (it can rank contracts by MishandledException probability reasonably well) but threshold calibration fails — the optimal threshold from the training distribution does not generalise to this eval subset.

EXP-S1 val-split pattern detection found 53.8% of MishandledException-positive contracts have the relevant structural pattern (call without revert check), suggesting the graph does encode the relevant information. The failure is in converting ranked predictions into binary classifications, not in encoding.

**Run 5 priority:** Medium. Threshold calibration tuning + phase 2 auxiliary loss.

---

### 10.7 Reentrancy

**F1 at ep32:** ~0.169 (noted as near-bottom in training log)
**Val positives in EXP-A4:** 48/470 (10.2%)
**GNN eye F1:** 0.182 | **TF eye F1:** 0.389 | **Main F1:** 0.438
**GNN AUC-ROC:** 0.788 | **Main AUC-ROC:** 0.884
**JK weights:** Ph1=0.333, Ph2=0.323, Ph3=**0.345**

**Structural profile (EXP-S2):**
- CALL_ENTRY enrichment: 1.065×
- RETURN_TO enrichment: 1.076×
- WRITES enrichment: 1.046× (Reentrancy has 100% WRITES edge presence)
- CALLS enrichment: 1.037×
- total_cfg_nodes Cohen's d: 0.334
- function_count Cohen's d: 0.413

**Interpretation:** Reentrancy is the class with the largest gap between theoretical importance (most cited vulnerability, requires CEI analysis) and actual model performance (GNN eye F1 = 0.182, only marginally above baseline 0.170).

All structural indicators suggest the data is appropriate:
- EXP-S4: 76% of Reentrancy-positive val contracts have CALL_ENTRY edges, 69% have full CEI chain
- WRITES edge: 100% presence in Reentrancy-positive contracts (vs 95.6% baseline)
- Larger, more complex contracts (function_count d=0.413)

Despite this, EXP-L2 shows CALL_ENTRY ablation effect = −5.3e-7 on Reentrancy logit. The model has completely failed to exploit the CEI structural signal.

The Transformer eye's F1 = 0.389 (vs GNN eye 0.182) suggests GraphCodeBERT is detecting Reentrancy from textual patterns (likely the co-occurrence of `.call{value:`, state variable assignments, and the CEI violation pattern in source code tokens). This is a weaker signal than CEI graph traversal but is apparently the dominant information pathway in the current model.

**Run 5 priority:** Highest. Reentrancy is the canonical smart contract vulnerability. Phase 2 auxiliary loss + CEI path supervision are the primary interventions.

---

### 10.8 Timestamp

**F1 at ep32:** 0.329 (mid-tier)
**Val positives in EXP-A4:** 4/470 (0.9%)
**GNN eye F1:** 0.286 | **TF eye F1:** 0.286 | **Main F1:** 0.571
**GNN AUC-ROC:** 0.989 | **Main AUC-ROC:** 0.992
**JK weights:** Ph1=0.311, Ph2=0.318, Ph3=**0.371** (strongest Phase 3 of all classes)

**Structural profile (EXP-S2, EXP-S3):**
- total_cfg_nodes Cohen's d: **0.643** (val), 2.34× ratio in training (shortcut flag TRUE — confirmed)
- total_nodes Cohen's d: **0.643** (val) — corrected from originally reported 1.657
- function_count Cohen's d: 1.277
- cfg_call_count Cohen's d: 1.110
- CALL_ENTRY enrichment: **1.414×** (strongest of all standard classes)
- RETURN_TO enrichment: **1.578×** (strongest of all standard classes)

**Interpretation:** Timestamp is the most structurally distinctive class. Timestamp-positive contracts are larger (~1.70× in val, Cohen's d = 0.643; 2.34× in training split) and more interconnected (1.578× enrichment in RETURN_TO edges) than the baseline. The size difference is a confirmed shortcut signal the model exploits. EXP-L7 shows F1=1.0 for small contracts collapsing to 0.364 for large ones — direct evidence of size-dependent prediction.

The GNN AUC-ROC of 0.989 — near-perfect by ranking alone — confirms the GNN has learned to exploit contract size as a proxy for Timestamp vulnerability. The high Phase 3 JK weight (0.371) suggests the GNN is using the CONTAINS/REVERSE_CONTAINS hierarchy most strongly for this class, consistent with size being the primary structural signal (larger contracts have deeper hierarchies).

This is a potential shortcut: `block.timestamp` misuse does not fundamentally require large contracts. The high AUC-ROC and F1 for this class may be inflated by distribution bias in the training set.

**Run 5 priority:** Medium. Sol-3 data gating (Timestamp filter) should be implemented to verify whether F1 holds when controlling for contract size.

---

### 10.9 TransactionOrderDependence (TOD)

**F1 at ep32:** ~0.235 (near-bottom per training log)
**Val positives in EXP-A4:** 33/470 (7.0%)
**GNN eye F1:** 0.000 | **TF eye F1:** 0.000 | **Main F1:** 0.059
**GNN AUC-ROC:** 0.748 | **Main AUC-ROC:** 0.859
**JK weights:** Ph1=0.334, Ph2=0.323, Ph3=**0.343**

**Structural profile (EXP-S2):**
- READS enrichment: 0.986× (below baseline — TOD contracts slightly less likely to have READS edges)
- CALL_ENTRY enrichment: 1.101×
- function_count Cohen's d: 0.324

**Interpretation:** TOD is among the weakest-performing classes (F1 = 0.059 for main head, F1 = 0 for all individual eyes). AUC-ROC = 0.859 indicates reasonable ranking ability but complete threshold failure.

TOD vulnerabilities involve a transaction being front-run: the attacker observes a pending transaction and inserts a competing transaction that changes state before the victim's transaction executes. Detecting this requires understanding the contract's state dependency graph — which state variables are read and written in which functions, and whether a reordering of transactions can change outcomes.

This is precisely the kind of cross-function data-flow analysis that Phase 2 DEF_USE edges should enable. The READS enrichment of 0.986× (slightly *below* baseline) suggests the structural signature of TOD-vulnerable contracts is not concentrated in read-edge density but in the specific pattern of reads and writes across function boundaries.

**Run 5 priority:** High. TOD is fundamental to DeFi security. Phase 2 DEF_USE auxiliary loss directly targets this class.

---

### 10.10 UnusedReturn

**F1 at ep32:** ~0.052 (training log notes this as a hard class)
**Val positives in EXP-A4:** 13/470 (2.8%)
**GNN eye F1:** 0.000 | **TF eye F1:** 0.000 | **Main F1:** 0.300
**GNN AUC-ROC:** 0.929 | **Main AUC-ROC:** 0.965
**JK weights:** Ph1=0.334, Ph2=0.324, Ph3=**0.343**

**Structural profile (EXP-S2):**
- CALLS enrichment: **1.306×** (borderline threshold of 1.3×)
- CALL_ENTRY enrichment: **1.403×**
- RETURN_TO enrichment: **1.532×**
- DEF_USE enrichment: 1.175×
- function_count Cohen's d: 0.786 (large contracts)
- EMITS enrichment: 15.46× (4.65× for CallToUnknown, but only 1 EMITS edge in the UnusedReturn-positive subset)

**Interpretation:** UnusedReturn has exceptionally high AUC-ROC = 0.929 for the GNN eye, yet F1 = 0. This is the most extreme AUC-ROC vs F1 gap in the dataset. The model can rank UnusedReturn contracts almost perfectly but cannot select a threshold that converts this to accurate binary predictions.

The high CALL_ENTRY (1.403×) and RETURN_TO (1.532×) enrichment indicate that UnusedReturn-positive contracts have unusually rich interprocedural call graphs. This is structurally sensible: `UnusedReturn` occurs when the return value of an external call (e.g., `ERC20.transfer()`) is not checked. Detecting this requires tracing the call return value through the RETURN_TO edge and verifying that it is not read afterward.

The EMITS enrichment is high (15.46×) but based on only 1 contract with an EMITS edge in the positive subset — statistically meaningless with n=1.

The model has learned the right structural priors (GNN AUC-ROC = 0.929) but cannot convert them to useful predictions. Two interventions are needed: (1) phase 2 auxiliary loss on RETURN_TO traversal, and (2) improved threshold calibration (potentially through Platt scaling or temperature calibration on the GNN eye output).

**Run 5 priority:** High. GNN has strong discriminative signal but poor calibration. Threshold tuning + Phase 2 supervision.

---

## Summary Table: Per-Class Status and Run 5 Priority

| Class | ep32 F1 | GNN eye F1 | GNN AUC | Main AUC | Primary limiter | Run 5 priority |
|-------|---------|-----------|---------|----------|----------------|----------------|
| CallToUnknown | ~0.247 | 0.255 | 0.757 | 0.846 | Threshold + Phase 2 | Medium |
| DenialOfService | unstable | 0.000 | 0.648 | 1.000 | Too few samples | Low |
| ExternalBug | ~0.246 | 0.000 | 0.768 | 0.870 | Phase 2 not exploited | High |
| GasException | ~0.317 | 0.000 | 0.676 | 0.832 | Token patterns sufficient | Medium |
| IntegerUO | 0.647 | 0.548 | 0.727 | 0.909 | Already working | Low |
| MishandledException | ~0.158 | 0.000 | 0.735 | 0.834 | Threshold calibration | Medium |
| Reentrancy | ~0.169 | 0.182 | 0.788 | 0.884 | Phase 2 CEI not learned | Highest |
| Timestamp | 0.329 | 0.286 | 0.989 | 0.992 | Size shortcut risk | Medium |
| TOD | ~0.235 | 0.000 | 0.748 | 0.859 | Phase 2 DEF_USE | High |
| UnusedReturn | ~0.052 | 0.000 | 0.929 | 0.965 | Threshold calibration | High |

---

## 10. Phase B — New Measurements (run 2026-05-31)

Phase B scripts closed the measurement gaps identified in the interpretability audit. All ran with the Run 4 ep32 checkpoint.

| ID | Script | Key Finding | Status |
|----|--------|-------------|--------|
| B1 | `exp_b1_phase2_gradient_norm.py` | Phase 1 > Phase 2 > Phase 3 gradient norm for all 10 classes. Phase 2 = 72–91% of Phase 1 (corrected BCEWithLogitsLoss method; was 75–86% with raw logit). Timestamp highest absolute norms and highest P2/P1 ratio (91.3%). | **COMPLETE** *(method corrected 2026-06-01)* |
| B2 | `exp_b2_per_eye_ece.py` | Individual eyes well-calibrated (ECE 0.057–0.065). Main head severely miscalibrated (ECE 0.249). Temperature scaling must target main head. | **COMPLETE** |
| B3 | `exp_b3_jk_weight_distribution.py` | Universal Phase 3 > Phase 1 > Phase 2 ordering across all classes. No class selectively upweights Phase 2. Std per class: 0.01–0.03. | **COMPLETE** |
| B4 | `exp_b4_unusedreturn_saliency.py` | `external_call_count` and `complexity` dominate both high- and low-scoring UnusedReturn contracts. `return_ignored` ranks 4th (high-scoring) with only 2.3% relative difference vs low-scoring — size shortcut confirmed for UnusedReturn. | **COMPLETE** |

**B1 gradient norm detail (mean per phase per class — corrected BCEWithLogitsLoss run, 2026-06-01):**

| Class | Phase1 | Phase2 | Phase3 | P2/P1 ratio |
|-------|--------|--------|--------|-------------|
| CallToUnknown | 0.051893 | 0.041190 | 0.035657 | 79.4% |
| DenialOfService | 0.034882 | 0.025193 | 0.020473 | 72.2% |
| ExternalBug | 0.044038 | 0.033292 | 0.028976 | 75.6% |
| GasException | 0.041313 | 0.032616 | 0.025441 | 78.9% |
| IntegerUO | 0.039092 | 0.028423 | 0.022379 | 72.7% |
| MishandledException | 0.045841 | 0.036230 | 0.031026 | 79.0% |
| Reentrancy | 0.050614 | 0.037378 | 0.035040 | 73.8% |
| Timestamp | 0.094683 | 0.086430 | 0.073734 | **91.3%** |
| TOD | 0.033661 | 0.025190 | 0.020505 | 74.8% |
| UnusedReturn | 0.080223 | 0.061586 | 0.052428 | 76.8% |

> **Method correction (2026-06-01):** Original run backpropagated through raw logit `logits[0, class_idx]`. Corrected run uses `F.binary_cross_entropy_with_logits(logits[0, class_idx].unsqueeze(0), target=1)` to match training-time gradient. Absolute magnitudes are ~3.5× smaller after correction (sigmoid derivative dampens gradient at high logit values), but Phase 1 > Phase 2 > Phase 3 ordering is unchanged for all classes. P2/P1 ratio range: 72.2% (DenialOfService) – 91.3% (Timestamp).

**B2 ECE summary:**

| Eye | Mean ECE | Range |
|-----|---------|-------|
| Fused | 0.057 | 0.022–0.078 |
| Transformer | 0.059 | 0.022–0.091 |
| GNN | 0.065 | 0.023–0.129 |
| **Main head** | **0.249** | **0.183–0.310** |

Temperature calibration (`ml/calibration/temperatures_run4.json`) reduces main head ECE from 0.249 → 0.028. Individual eyes do not need calibration.

Individual experiment docs: `docs/interpretability/exp_b1_phase2_gradient_norm.md` through `exp_b4_unusedreturn_saliency.md`.

---

## 11. Addendum — Script Fixes (2026-05-31 + 2026-06-01)

**2026-05-31 fixes (four bugs):**

| Script | Bug | Fix Applied |
|--------|-----|-------------|
| `exp_l4_gradient_saliency.py` | Stale hardcoded FEATURE_NAMES (same as L8) — dims 3–9 mislabelled | Import from graph_schema |
| `exp_a2_cfg_inheritance.py` | `_CONTAINS_EDGE = 0` matched CALLS (type 0) not CONTAINS (type 5) | `EDGE_TYPES["CONTAINS"]` |
| `exp_e1_receptive_field.py` | Analysis 2: used REVERSE_CONTAINS (runtime-only) → always 0%; Analysis 3: no CONTRACT→FUNCTION edge in schema | Redesigned both analyses |
| `exp_l3_attention_visualization.py` | Only hooked conv3 (CF-only) — missed conv3b (CALL_ENTRY+RETURN_TO) | Now hooks both in one forward pass |

**2026-06-01 fixes (four more bugs, from audit):**

| Script / Doc | Bug | Fix Applied |
|-------------|-----|-------------|
| `exp_b1_phase2_gradient_norm.py` | `logits[0, class_idx].backward()` backpropagated through raw logit — not representative of training (BCEWithLogitsLoss). Absolute magnitudes non-comparable to training gradients. | Changed to `F.binary_cross_entropy_with_logits(logit.unsqueeze(0), target=1).backward()`. Absolute values ~3.5× smaller; ordering unchanged. |
| `exp_l2_edge_ablation.py` | EDGE_TYPE_NAMES had CALLS/CONTAINS swapped at indices 0 and 5 (cosmetic, logic unaffected). More critically: no structural ablation implemented — reported "0.0048/0.014" were estimates or came from separate hard-coded logic, not the main ablation loop. | Fixed name order. Added `_structural_ablate_graph()` helper and second ablation loop for Phase 2 edge types (6, 8, 9, 10). |
| `exp_e1_receptive_field.py` | Analysis 2 dict key mismatch: `results["analysis2_function_aggregation"]` → KeyError on JSON output | Changed to `results["analysis2_function_cfg_coverage"]` |
| `docs/interpretability/exp_l3_attention_visualization.md` | PASS status recorded for a result that is architecturally guaranteed (100% CF fraction when conv3 receives only CF edges). False positive. | Status changed to ARCHITECTURAL N/A; uniform attention (all weights = 1.0) moved to headline finding. |

---

*End of SENTINEL Interpretability Master Report*
*Generated: 2026-05-30  Updated: 2026-06-01 (audit fixes: B1 gradient method, L2 structural ablation, L3 ARCH-N/A, L4 rerun; all 25 experiments complete)*
*Based on: GCB-P1-Run4-no-asl-pw_best.pt (ep32, F1=0.3362)*
*Experiments: exp_a1–exp_l10, exp_s1–exp_s4, exp_e1–exp_e4, exp_a3, exp_a4, exp_b1–exp_b4 (all complete, no pending)*
