# SENTINEL Run 9 × SolidiFI — Live Analysis Document

**Status:** IN PROGRESS  
**Started:** 2026-06-10  
**Checkpoint:** `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt` (epoch 52, val F1=0.2965)  
**Dataset:** SolidiFI — 350 standalone `.sol` files, 50 per category × 7 categories

---

## Analysis Plan & Progress Tracker

This section is the roadmap. Steps are checked off as they complete; findings are linked to the relevant section below.

### Phase A — Infrastructure & Baseline  *(done before this session)*
- [x] **A1** Fix `_detect_solc_version()` to detect upper-bound of range pragmas (`>=0.4.22 <0.6.0` → `0.5.17`). Applied to both `ml/src/inference/preprocess.py` and `ml/scripts/reextract_graphs.py`.
- [x] **A2** Run full benchmark on 341 clean SolidiFI contracts (0 errors). Save results to `docs/training/benchmark_run9_solidifi_2026-06-10.md`.
- [x] **A3** Contamination check: 341/350 clean (9 Unchecked-Send near-dups excluded). Tiers 1–3 (exact hash, normalized hash, token Jaccard).
- [x] **A4** Expose that Tier/Tuned recall = 1.00 is **trivially achieved** (model fires ≥0.25 on every class on every contract). → See §5.
- [x] **A5** Run rank-based (Top-K) analysis — the honest metric. → See §6.
- [x] **A6** Pipeline consistency check: re-extract 8 BCCC training graphs via inference pipeline → all 8 IDENTICAL to stored `.pt` files. → See §14.

### Phase B — Per-Eye Instrumentation  *(next)*
- [ ] **B1** Write `ml/scripts/diag_per_eye_solidifi.py`: run each SolidiFI contract through the model with `return_aux=True`, capture per-eye logits (GNN eye, Transformer eye, Fused eye, CFG eye + aux_phase2), save results to a JSON.
- [ ] **B2** Analyse per-eye probabilities: for each category, which eye is responsible for correct predictions? Which eye is confusing the classifier? → Fills §13.
- [ ] **B3** Identify the dominant eye per class and document it with probability tables.

### Phase C — Graph-Level Inspection  *(next)*
- [ ] **C1** For each category, extract a representative graph (e.g. buggy_1.sol) and print actual node feature vectors. Verify feat[2] (uses_block_globals), feat[7] (return_ignored), feat[10] (external_call_count), feat[11] (in_unchecked_block).
- [ ] **C2** For Timestamp-Dependency: compare a Top-1 HIT contract vs a Top-1 MISS contract — what does feat[2] look like? Is the timestamp signal actually in the graph?
- [ ] **C3** For Re-entrancy: compare a HIT vs MISS — does feat[7] (return_ignored) and feat[10] (external_call_count) differ between them?
- [ ] **C4** For Unhandled-Exceptions: why does the model score MishandledException near-randomly when BCCC labels were VERIFIED 100% clean? Check feat[7] (return_ignored).
- [ ] **C5** For TOD/CallToUnknown: confirm random-level scores are consistent with near-zero label quality in BCCC. Document.

### Phase D — Manual Source Inspection  *(in progress)*
- [x] **D1** Read `Overflow-Underflow/buggy_1.sol` — injected `bug_intou*` functions with `uint8` underflow/overflow + `require(balances[msg.sender] - _value >= 0)` pattern. Documented in §7.
- [ ] **D2** Read 2–3 Timestamp-Dependency contracts: one Top-1 HIT and one Top-1 MISS. What `block.timestamp` / `now` patterns are present?
- [ ] **D3** Read 2–3 Re-entrancy contracts: one HIT (Top-1) and one MISS (rank ≥ 3). Is the re-entrancy pattern clearly visible in source?
- [ ] **D4** Read 1–2 Unhandled-Exceptions contracts. Is `.call()` return value clearly unchecked? Why does the model score MishandledException at ~0.41 (near-random)?
- [ ] **D5** Read 1–2 TOD contracts. What does the injected TOD pattern look like in source?
- [ ] **D6** Read 1–2 Unchecked-Send contracts. What does `send()` without `.value` check look like?

### Phase E — Deep Dive Sections (§7–§12)  *(to fill)*
- [x] **E1** §14 Pipeline consistency — DONE.
- [x] **E2** §5 Benchmark numbers — DONE.
- [x] **E3** §6 Top-K rank analysis — DONE.
- [ ] **E4** §7 Overflow-Underflow deep dive — partially started (D1 done, need graph features + per-eye).
- [ ] **E5** §8 Timestamp-Dependency deep dive — need D2 + C2 + B2 results.
- [ ] **E6** §9 Re-entrancy deep dive — need D3 + C3 + B2 results.
- [ ] **E7** §10 Unchecked-Send deep dive — need D6 + C5 + B2 results.
- [ ] **E8** §11 TOD deep dive — need D5 + C5 + B2 results.
- [ ] **E9** §12 Unhandled-Exceptions deep dive — need D4 + C4 + B2 results.
- [ ] **E10** §13 Model internals: per-eye predictions — needs Phase B complete.
- [ ] **E11** §15 Root causes summary — final synthesis, written last.

### Phase F — Synthesis  *(last)*
- [ ] **F1** Write §15 Root causes summary: connect label quality → Top-K score for each class.
- [ ] **F2** Write actionable recommendations for Run 10 / v2 data pipeline based on findings.

---

**Current step:** B1 — writing the per-eye diagnostic script.

---

## Table of Contents

1. [What is SolidiFI?](#1-what-is-solidifi)
2. [What is SENTINEL Run 9?](#2-what-is-sentinel-run-9)
3. [Pipeline: How a contract flows through the model](#3-pipeline)
4. [Per-category class mapping](#4-category-mapping)
5. [Overall benchmark numbers](#5-benchmark-numbers)
6. [Top-K rank analysis — the real metric](#6-top-k-rank-analysis)
7. [Deep dive: Overflow-Underflow (100% Top-1)](#7-overflow-underflow)
8. [Deep dive: Timestamp-Dependency (48% Top-1)](#8-timestamp)
9. [Deep dive: Re-entrancy (36% Top-1)](#9-reentrancy)
10. [Deep dive: Unchecked-Send — CallToUnknown (0% Top-1)](#10-unchecked-send)
11. [Deep dive: TOD (0% Top-1)](#11-tod)
12. [Deep dive: Unhandled-Exceptions (0% Top-1)](#12-unhandled-exceptions)
13. [Model internals: per-eye predictions](#13-model-internals)
14. [Pipeline consistency: training vs inference](#14-pipeline-consistency)
15. [Root causes summary](#15-root-causes)

---

## 1. What is SolidiFI?

SolidiFI is a synthetic benchmark of 350 Solidity contracts, 50 per vulnerability category:

| SolidiFI category | What the vulnerability is |
|---|---|
| Overflow-Underflow | Integer arithmetic that wraps around (before Solidity 0.8 safe-math) |
| Re-entrancy | External call before state update — attacker re-enters and drains |
| Timestamp-Dependency | Decision logic controlled by `block.timestamp` / `now` |
| TOD | Transaction Order Dependence — miner reorders txs to front-run reward |
| Unchecked-Send | `send()`/`transfer()` return value not checked |
| Unhandled-Exceptions | Low-level `.call()` return value not checked |
| tx.origin | `require(tx.origin == msg.sender)` — phishing vulnerability |

**How contracts are built:** SolidiFI takes a real ERC-20 token contract from Etherscan (2019, called "HotDollarsToken / EIP20Interface") and **injects** snippets of vulnerable code at random positions. The base contract's `pragma solidity >=0.4.22 <0.6.0`.

**Contamination check:** We checked all 350 contracts against the BCCC training set (Tier 1 exact hash, Tier 2 normalized hash, Tier 3 token Jaccard). Result: **341/350 clean** (9 Unchecked-Send contracts are near-duplicates of training data — excluded from analysis below).

**Solc version fix:** The pragma `>=0.4.22 <0.6.0` + injected code uses `address payable` syntax (Solidity 0.5.0+). The inference pipeline's `_detect_solc_version()` was picking `0.4.26` (from the lower bound `0.4.22`) which failed to compile. **Fixed** to detect the upper bound `<0.6.0` and select `0.5.17` instead.

---

## 2. What is SENTINEL Run 9?

Run 9 is the current best model checkpoint, trained on the BCCC-SCsVul-2024 dataset (~41,576 contracts).

**Architecture (four_eye_v8):**
```
Input: one .sol contract

  ┌─────────────────────────────────┐
  │  Slither + solc                 │  → AST/CFG graph  (Graph G)
  │  GraphCodeBERT tokenizer        │  → token sequence (Tokens T)
  └─────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  GNNEncoder (8-layer GAT, 3 phases)                         │
  │    Phase 1 (L1+L2): structural edges (CALLS, READS, WRITES) │
  │    Phase 2 (L3+L4+L5): control-flow (CONTROL_FLOW, ICFG)   │
  │    Phase 3 (L6+L7+L8): REVERSE_CONTAINS up + CONTAINS down │
  │    → JK attention aggregation → gnn_embedding [B, 256]      │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │  TransformerEncoder (GraphCodeBERT frozen + LoRA r=16)      │
  │    → tf_embedding [B, 768]                                  │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │  CrossAttentionFusion                                        │
  │    GNN nodes attend to BERT token sequence                  │
  │    → fused_embedding [B, 128]                               │
  └──────────────────────────────────────────────────────────────┘

  Four "eyes" each project to 128-dim, concatenated → [B, 512]:
    eye1: GNN embedding
    eye2: Transformer embedding
    eye3: Fused (cross-attention) embedding
    eye4: CFG-specific embedding (from Phase 2 GNN layer)

  Classifier: Linear(512→256) → GELU → Linear(256→10) → sigmoid
```

**Training data quality (BCCC Phase 5 label audit):**

| Class | Labels retained | Gate | What this means |
|---|---|---|---|
| IntegerUO | 100% (16,740) | VERIFIED ✓ | Clean labels, high confidence |
| UnusedReturn | 100% (3,229) | VERIFIED ✓ | Clean |
| MishandledException | 100% (5,154) | VERIFIED ✓ | Clean |
| Reentrancy | 9.6% (1,699/17,698) | VERIFIED ✓ | 89.4% FP in BCCC — training was mostly noise |
| CallToUnknown | 2.1% (239/11,131) | PROVISIONAL | 86.9% FP — near-random labels |
| Timestamp | 40.2% (1,075/2,674) | BEST-EFFORT | Partial signal |
| DenialOfService | 10.1% (1,252/12,394) | BEST-EFFORT | Partial |

---

## 3. Pipeline

For each `.sol` contract, the pipeline does:

```
1. _detect_solc_version(source)
   → reads pragma statement
   → picks the latest patch of the declared minor version
   → for ">=0.4.22 <0.6.0": detects upper bound <0.6.0 → picks 0.5.17

2. Slither + solc-0.5.17
   → parses AST, builds CFG, computes data-flow
   → 12 node features extracted per node (see below)
   → 12 edge types

3. GraphCodeBERT tokenizer
   → splits source into 512-token windows (up to 4 windows)

4. model.forward(graph, tokens)
   → returns logits for 10 classes
   → sigmoid → probabilities [0, 1] per class

5. Thresholding
   → Tier mode: ≥0.55 = confirmed, ≥0.25 = suspicious
   → Tuned mode: per-class threshold from calibration JSON
```

**12 node features (v9 schema):**

| Index | Name | What it captures |
|---|---|---|
| 0 | type_id | Node type (STATE_VAR=0, FUNCTION=1, ..., CFG_NODE_ARITH=13), normalized /13.0 |
| 1 | visibility | public/external/internal/private |
| 2 | uses_block_globals | 1.0 if node reads block.timestamp / block.number / now |
| 3 | view | 1.0 if function is view/pure |
| 4 | payable | 1.0 if function is payable |
| 5 | complexity | log1p(#CFG nodes in function) / log1p(100) |
| 6 | loc | log1p(lines) / log1p(1000) |
| 7 | return_ignored | 1.0 if return value discarded |
| 8 | call_target_typed | 1.0 if call target is typed (interface call), 0 if raw address |
| 9 | has_loop | 1.0 if function contains a loop |
| 10 | external_call_count | log1p(#external calls) / log1p(20) |
| 11 | in_unchecked_block | 1.0 if node is inside `unchecked{}` (0.8+); **for pre-0.8: always 1.0** |

**12 edge types:**

| ID | Name | What it is |
|---|---|---|
| 0 | CALLS | function → function it calls |
| 1 | READS | function → state variable it reads |
| 2 | WRITES | function → state variable it writes |
| 3 | EMITS | function → event it emits |
| 4 | INHERITS | contract → parent contract |
| 5 | CONTAINS | function → its CFG statement nodes |
| 6 | CONTROL_FLOW | CFG statement → next statement |
| 7 | REVERSE_CONTAINS | CFG node → parent function (Phase 3 only) |
| 8 | CALL_ENTRY | calling node → callee entry point (internal calls) |
| 9 | RETURN_TO | callee terminal → call-site successor |
| 10 | DEF_USE | CFG node defining variable → node reading it |
| 11 | EXTERNAL_CALL | self-loop on CFG node that makes an external call |

**14 node types:**

| ID | Name |
|---|---|
| 0 | STATE_VAR |
| 1 | FUNCTION |
| 2 | MODIFIER |
| 3 | EVENT |
| 4 | FALLBACK |
| 5 | RECEIVE |
| 6 | CONSTRUCTOR |
| 7 | CONTRACT |
| 8 | CFG_NODE_CALL |
| 9 | CFG_NODE_WRITE |
| 10 | CFG_NODE_READ |
| 11 | CFG_NODE_CHECK |
| 12 | CFG_NODE_OTHER |
| 13 | CFG_NODE_ARITH |

---

## 4. Category Mapping

| SolidiFI category | SENTINEL class | Training label quality |
|---|---|---|
| Overflow-Underflow | IntegerUO | VERIFIED (100% clean) |
| Re-entrancy | Reentrancy | VERIFIED but 89.4% FP in raw BCCC |
| Timestamp-Dependency | Timestamp | BEST-EFFORT (40.2% retained) |
| TOD | TransactionOrderDependence | Very sparse / noisy |
| Unchecked-Send | CallToUnknown | PROVISIONAL (86.9% FP in raw BCCC) |
| Unhandled-Exceptions | MishandledException | VERIFIED (100% clean) |
| tx.origin | (no SENTINEL class — FP probe only) | — |

---

## 5. Benchmark Numbers

341 contracts processed. 0 errors.

**Graph size distribution:**
- Median: 250 nodes (training median: 90 nodes — SolidiFI is 2.8× larger than training average)
- Mean: 272 nodes
- Min: 15 nodes, Max: 936 nodes
- Contracts with <30 nodes: 1.5%

**Tier recall (≥0.25 threshold):**

| Category | SENTINEL class | N | Tier Recall |
|---|---|---|---|
| Overflow-Underflow | IntegerUO | 50 | 1.00 |
| Re-entrancy | Reentrancy | 50 | 1.00 |
| TOD | TransactionOrderDependence | 50 | 1.00 |
| Timestamp-Dependency | Timestamp | 50 | 1.00 |
| Unchecked-Send | CallToUnknown | 41 | 1.00 |
| Unhandled-Exceptions | MishandledException | 50 | 1.00 |

⚠️ **These numbers are misleading.** The model predicts `suspicious` (≥0.25) for almost every class on almost every contract. The FP probe (tx.origin, 50 contracts with no SENTINEL class) shows 100% pass the suspicious threshold and 86% pass the confirmed (≥0.55) threshold. This means the model fires on everything — the recall=1.0 is trivially achieved.

---

## 6. Top-K Rank Analysis — The Real Metric

For each contract, all 10 class probabilities are ranked. **Does the correct class get rank #1?**

| Category | SENTINEL class | N | Top-1% | Top-2% | Top-3% |
|---|---|---|---|---|---|
| Overflow-Underflow | IntegerUO | 50 | **100%** | 100% | 100% |
| Timestamp-Dependency | Timestamp | 50 | 48% | 74% | 82% |
| Re-entrancy | Reentrancy | 50 | 36% | 90% | 98% |
| TOD | TransactionOrderDependence | 50 | 0% | 0% | 2% |
| Unchecked-Send | CallToUnknown | 41 | 0% | 0% | 5% |
| Unhandled-Exceptions | MishandledException | 50 | 0% | 6% | 20% |

**Reading guide:** "Top-1%" = fraction of contracts where the correct class has the single highest probability. "Top-2%" = correct class is either #1 or #2.

This is the real picture. Only IntegerUO is genuinely learned. The rest ranges from partial (Timestamp, Reentrancy) to nothing (TOD, CallToUnknown).

---

## 7. Overflow-Underflow (IntegerUO) — 100% Top-1

*(Section being filled — see below)*

---

## 8. Timestamp-Dependency — 48% Top-1

*(Section being filled — see below)*

---

## 9. Re-entrancy — 36% Top-1

*(Section being filled — see below)*

---

## 10. Unchecked-Send (CallToUnknown) — 0% Top-1

*(Section being filled — see below)*

---

## 11. TOD — 0% Top-1

*(Section being filled — see below)*

---

## 12. Unhandled-Exceptions (MishandledException) — 0% Top-1

*(Section being filled — see below)*

---

## 13. Model Internals: Per-Eye Predictions

*(Section being filled)*

---

## 14. Pipeline Consistency: Training vs Inference

**Test:** Take 8 training contracts from the BCCC training set, re-extract their graphs using the current inference pipeline, and compare byte-by-byte to the stored `.pt` graph files.

**Result:** 8/8 IDENTICAL.

```
[    0] 605c1b61...  solc=0.4.26  IDENTICAL  pragma=^0.4.12
[  100] c0296bc2...  solc=0.4.26  IDENTICAL  pragma=^0.4.24
[ 5000] 51ca164c...  solc=0.4.26  IDENTICAL  pragma=^0.4.25
[10000] dad4ccd5...  solc=0.5.17  IDENTICAL  pragma=^0.5.0
[20000] 8fa2dad6...  solc=0.4.26  IDENTICAL  pragma=^0.4.21
[30000] b85266f4...  solc=0.4.26  IDENTICAL  pragma=^0.4.16
[40000] 3e109525...  solc=0.4.26  IDENTICAL  pragma=^0.4.21
[41000] ac0b15bb...  solc=0.4.26  IDENTICAL  pragma=^0.4.18
```

**Conclusion:** The inference pipeline produces byte-identical graphs to what was used during training, for the BCCC training distribution. The `_detect_solc_version` fix (range pragma → upper-bound detection) only affects contracts with `>=X.Y.Z <A.B.C` pragmas, which do not appear in BCCC training data.

---

## 15. Root Causes Summary

*(To be filled as analysis completes)*

---

*This document is updated live as analysis proceeds.*
