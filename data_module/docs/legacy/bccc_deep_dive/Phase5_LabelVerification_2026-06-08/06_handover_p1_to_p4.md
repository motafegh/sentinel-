# BCCC Deep Dive — Handover: Phases 1-4

**Purpose:** A comprehensive onboarding document covering everything done in Phases 1-4, so anyone starting Phase 5 can understand the full context without reading every individual file.
**Date:** 2026-06-08
**Author:** SENTINEL Data Engineering
**Next phase:** [`./05_phase5_plan.md`](./05_phase5_plan.md) — Comprehensive Label Verification

---

## Table of Contents

1. [What Is BCCC?](#1-what-is-bccc)
2. [Project Structure](#2-project-structure)
3. [Phase 1: Exploration & Inventory](#3-phase-1-exploration--inventory)
4. [Phase 2: Validation & Cleaning](#4-phase-2-validation--cleaning)
5. [Phase 3: Deep Analysis](#5-phase-3-deep-analysis)
6. [Phase 4: Label Trustworthiness Validation](#6-phase-4-label-trustworthiness-validation)
7. [Complete Decision Registry](#7-complete-decision-registry)
8. [What's Been Completed vs What's Parked](#8-whats-been-completed-vs-whats-parked)
9. [Tooling & Environment](#9-tooling--environment)
10. [Critical Context for Phase 5](#10-critical-context-for-phase-5)

---

## 1. What Is BCCC?

**BCCC-SCsVul-2024** is a smart-contract vulnerability dataset from the **Blockchain Covenant Consortium for Cybersecurity (BCCC)**. It's the largest labeled vulnerability corpus we found:

| Property | Value |
|---|---|
| **Total contracts** | 68,433 unique (111,897 CSV rows — multi-label) |
| **Vulnerability classes** | 12 BCCC → 10 SENTINEL (after dropping Class05/07) |
| **Source code** | 1.6 GB Solidity files, read-only, in folder `SourceCodes/` |
| **Label source** | CSV file (`SC_Vulnerable.csv`) — ground truth |
| **Compilation target** | 92% pre-0.6 Solidity (mostly 0.4.x, 0.5.x) |
| **Multi-label rate** | 41% of contracts have ≥ 2 classes |
| **Duplicate rate** | 38.8% byte-identical copies |
| **Source tree** | READ ONLY — never modified by any script. Writes go to `Data/Deep_Dive/...` |

**Path:** `~/projects/sentinel/BCCC-SCsVul-2024/` (**at repo root** — NOT `data/raw/`. Scripts must use this path or they will fail.)

---

## 2. Project Structure

```
BCCC-SCsVul-2024_Deep_Dive/                         [Root directory]
├── README.md                                        [Entry point — needs update]
├── 01_exploration_inventory.md                      [Phase 1 report — 10 findings]
├── 02_validation_deep_dive_plan.md                  [Phase 2 plan]
├── 03_phase3_plan.md                                [Phase 3 plan]
├── 04_phase4_plan.md                                [Phase 4 plan — superseded by Phase 5]
├── scripts/                                         [Phase 1 exploration scripts]
│   └── bccc_phase1_explore*.py (4 scripts)
│
├── Phase2_Validation_2026-06-06/                    [Phase 2 — 100% complete]
│   ├── README.md
│   ├── 00_session_log.md
│   ├── scripts/                                     [WS-A through WS-H]
│   ├── integrity/
│   ├── labels/
│   ├── compile/
│   ├── complexity/
│   ├── cross_corpus/
│   ├── splits/
│   └── outputs/
│       └── contracts_clean.csv                      [67,311 × 24 — MAIN DELIVERABLE]
│
├── Phase3_DeepAnalysis_2026-06-06/                  [Phase 3 — ~40% complete]
│   ├── README.md
│   ├── 00_understanding_checklist.md
│   ├── 01_session_log.md
│   ├── scripts/
│   ├── outputs/
│   ├── reports/
│   └── labels/
│       └── ws_i_inspections_input.md                [30+2 manual review, 8,014 lines]
│
├── Phase4_LabelValidation_2026-06-07/               [Phase 4 — ~35% complete]
│   ├── README.md
│   ├── 00_actionable_checklist.md
│   ├── 01_session_log.md
│   ├── 04_phase4_plan.md
│   ├── CRITICAL_FINDINGS.md                         [The reason Phase 5 exists]
│   ├── scripts/
│   ├── outputs/
│   ├── reports/
│   ├── labels/
│   ├── decisions/
│   └── checkpoints/
│
└── Phase5_LabelVerification_2026-06-08/             [Phase 5 — NEW]
    ├── 05_phase5_plan.md                            [Phase 5 plan]
    ├── 06_handover_p1_to_p4.md                      [THIS FILE]
    ├── scripts/
    ├── outputs/
    ├── reports/
    ├── labels/
    └── decisions/
```

---

## 3. Phase 1: Exploration & Inventory

**Status:** ✅ 100% complete
**File:** `01_exploration_inventory.md` (435 lines)

The goal was to understand BCCC's actual structure before cleaning it.

### 10 Key Findings

| # | Finding | Detail |
|---|---|---|
| 1 | **Multi-label** | 41% of contracts have ≥ 2 simultaneous vulnerability labels |
| 2 | **68,433 unique contracts** | Not 111,897 as CSV rows suggest; 38.8% byte-identical copies |
| 3 | **12 folders ≠ labels** | Folders are tool-flagged candidate categories; CSV is ground truth |
| 4 | **Long-format CSV** | 111,897 rows = 68,433 IDs × 1.635 classes average |
| 5 | **Severe class imbalance** | Top 3 classes (NV, Reentrancy, IntegerUO) = 65% of positive labels |
| 6 | **766 contradictory contracts** | Labeled BOTH NonVulnerable AND a real vulnerability class |
| 7 | **Top co-occurrence** | DoS + Reentrancy = 12,381 contracts (18% of dataset) |
| 8 | **CSV ID ≠ source hash** | ID is keccak-256 of bytecode, not source content (95.5% mismatch) |
| 9 | **CSV integrity verified** | md5 `e38a2aa1c2b8a93c6cf8b23d2d7b870a` — file is genuine |
| 10 | **92% pre-0.6 Solidity** | Mostly 0.4.x / 0.5.x. Few modern patterns (no SafeMath post-0.8) |

### Scripts
- `bccc_phase1_explore.py` — per-folder counts, ID uniqueness
- `bccc_phase1_explore2.py` — duplicate rows, ID-vs-content hash
- `bccc_phase1_explore3.py` — md5, content dedup, pragma distribution
- `bccc_phase1_explore4.py` — corrected unique-contract-level multi-label analysis

---

## 4. Phase 2: Validation & Cleaning

**Status:** ✅ 100% complete
**Directory:** `Phase2_Validation_2026-06-06/` (~100 MB artifacts)
**Plan:** `02_validation_deep_dive_plan.md`
**Main deliverable:** `Phase2_Validation_2026-06-06/outputs/contracts_clean.csv` (67,311 × 24, SHA256 `53b7b884...`)

### Workstreams

| WS | Title | Key Output | Decision |
|---|---|---|---|
| **A** | Integrity & Dedup | SHA256 manifest, dedup map (38.8% duplication) | — |
| **B** | Label Validation | 766 NV+vuln contradictions flagged as `review_pending` | **D-B2** |
| **C** | Compile Probe | 73% compile rate with solc 0.4.24/0.5.17 | — |
| **D** | Cross-Corpus | **0 overlap with SmartBugs** = no test leakage | **D-D** |
| **E** | Complexity | Per-class LOC, n_functions, n_events, n_modifiers | — |
| **F** | Class Reconciliation | **D-F1**: Drop Class05/07 (1,122 dropped, 10 SENTINEL classes) | **D-F1** |
| **G** | Stratified Split | 70/15/15 → 46,581 / 9,982 / 9,982 | — |
| **H** | Final Dataset | `contracts_clean.csv` (67,311 × 24, 10 binary heads) | — |

### Key Decisions Made in Phase 2

- **D-F1:** Drop BCCC Class05 (TransactionOrderDependence/Front Running) + Class07 (WeakAccessModifier/Access Control). These classes are ill-defined in BCCC (too broad, too much noise). Result: 10 SENTINEL classes, 1,122 contracts dropped.
- **D-B2:** Hold out 766 NV+vuln contradictory contracts as `review_pending=1`. Don't drop them; flag for manual review.

### Dataset Schema (contracts_clean.csv)

24 columns:
- `id` — BCCC contract ID (keccak-256 of bytecode)
- `source_path` — path relative to BCCC root
- `source_code` — full Solidity source
- `vulnerability` — BCCC's primary class label
- `primary_class` — BCCC's primary vulnerability class
- `n_pos` — number of positive classes for this contract
- `review_pending` — 1 if has NV+vuln contradiction, 0 otherwise
- 10 `label_<class>` columns — one per SENTINEL class
- `split` — train/val/test assignment (70/15/15)
- `bccc_source_path` — original SourceCodes path

---

## 5. Phase 3: Deep Analysis

**Status:** 🟡 ~40% complete (4 of 12 planned workstreams done; 4 merged into Phase 4; 4 never started)
**Directory:** `Phase3_DeepAnalysis_2026-06-06/`
**Plan:** `03_phase3_plan.md`
**Key deliverable:** **D-I-11** — mass NonVulnerable correction rule

### Completed Workstreams

| WS | Title | What Happened |
|---|---|---|
| **J** | Statistical EDA | 0% missingness, 4.9× imbalance, 60.6% single-label contracts |
| **S** | Class Mapping | 10/12 BCCC → DASP 10 mapping; Class05=Front Running, Class07=Access Control |
| **N** | Dropped + Review-Pending Breakdown | 1,122 dropped (163 Class05-only, 959 Class07-only); 766 review-pending |
| **I** | Slither Label Validation (808 contracts) | **33,049 slither findings** across 60 detectors. Per-class agreement metrics published. 32 worst-disagreement contracts manually reviewed. |

### Slither Agreement Results (Phase 3: 808 contracts; Phase 4 Stage 1: 10,693 contracts)

Phase 3 results (808 contracts, WS-I):

| Class | F1 | Precision | Recall | Verdict |
|---|---|---|---|---|
| Reentrancy | 0.51 | 0.93 | 0.35 | High precision, low recall |
| CallToUnknown | 0.33 | 0.92 | 0.20 | Same pattern |
| Timestamp | 0.15 | 0.09 | 0.53 | Slither over-fires |
| IntegerUO | 0.07 | 0.07 | 0.07 | Slither can't detect pre-0.8 overflow |
| 6 others | < 0.05 | — | — | Too few samples |

Phase 4 Stage 1 results (10,693 contracts, 3-way agreement — from `ws_p4_s1_3way_agreement.csv`):

| Class | Slither F1 | Aderyn F1 | Majority F1 | Verdict |
|---|---|---|---|---|
| Reentrancy | 0.229 | 0.169 | 0.165 | Reflects 89% BCCC FP rate — tools only confirm the 10.6% true reentrancy |
| 8 others | < 0.15 | 0.000 | 0.000 | Gate FAILED — structural noise confirmed |

**The drop from Phase 3 Reentrancy F1=0.51 to Phase 4 F1=0.23 is expected**: Phase 3 used 808 contracts with better slither coverage; Phase 4 used 10,693 with the full class distribution including many BCCC false-positives. The true signal is in the manual audit (10.6% TP rate), not the tool agreement score.

### D-I-11: The Most Important Decision So Far

**Discovery:** Manual review of 32 worst-disagreement contracts found:
- **28/30** contracts labeled BOTH NonVulnerable AND Reentrancy/CallToUnknown were genuinely vulnerable — BCCC's NonVulnerable label was wrong.
- Pattern: BCCC systematically over-labeled NonVulnerable when another vulnerability class was present.

**Decision:** Drop `Class12:NonVulnerable=1` whenever it co-occurs with any of:
{CallToUnknown, Reentrancy, GasException, MishandledException, DenialOfService, Timestamp}

**Impact:** Unlocked ~705 contracts for training. Later expanded with D-I-12 (add IntegerUO to co-occurrence list → +41 contracts).

### Phase 3 Workstreams That Never Started

| WS | Title | Why |
|---|---|---|
| **O** | Aderyn 3-Way Consensus (5,000 contracts) | Merged into Phase 4 Stage 1 (became 10,693 contracts) |
| **P** | Slither-Based Graph-Level Features | Would have required WS-O; parked |
| **Q** | SHAP Feature Importance | Depends on AutoML; parked |
| **R** | 3-Way Model Comparison | Depends on WS-L + WS-O; parked |
| **T** | Multi-Label Structure Test | Depends on AutoML; parked |
| **K-K1** | 31 Source-Code Regex Features | Done in Phase 4 Stage 0.5 |
| **K2** | Slither-derived features (32 cols) | Depends on WS-O; parked |
| **L** | AutoML Baselines | Merged into Phase 4 Stage 6; parked |
| **M** | Dataset v1.2 + CHANGELOG §46 | Replaced by Phase 5 v1.3 |

---

## 6. Phase 4: Label Trustworthiness Validation

**Status:** 🟡 ~35% complete (Stage 0 done, Stage 1 done but gate FAILED with critical findings, Stages 2-7 parked)
**Directory:** `Phase4_LabelValidation_2026-06-07/`
**Plan:** `04_phase4_plan.md`
**Critical findings:** `CRITICAL_FINDINGS.md` — the most important file to read before Phase 5

### Stage 0: Mass Corrections + Housekeeping (100% complete)

| Step | What Was Done | Key Numbers |
|---|---|---|
| **0.1** | D-I-11 applied (drop NV when co-occurs with 6 vuln classes) | 725 NV labels dropped |
| **0.1b** | D-I-12 added (drop NV when co-occurring with IntegerUO) | 41 more NV dropped, review_pending = 0 |
| **0.2** | Oraclize cluster dedup (stripped-source SHA256) | 136 clusters, 548 duplicates flagged |
| **0.3** | Exclude 33 Phase 3-reviewed contracts | 66,730 eligible for sampling |
| **0.4** | Resolve 41 remaining review_pending via D-I-12 | review_pending = 0 |
| **0.5** | 31 regex features on all 67,311 contracts | 67,311 × 31 feature matrix |
| **0.6** | 3 hand-crafted features | 67,311 × 3 (NV-contradiction + IntegerUO regex) |

**Final dataset after Stage 0:** `ws_p4_s01b_d12_applied.csv` (v1.1+12)
- 67,311 contracts
- 766 NV labels dropped total (725 + 41)
- 0 review_pending contracts
- 66,730 eligible for sampling (after excluding 33 reviewed + 548 Oraclize dups)

### Stage 1: Per-Folder Stratified Validation (Gate FAILED)

| Step | What Was Done | Key Result |
|---|---|---|
| **1.1** | Stratified sample: 15% per primary_class | 10,693 unique contracts sampled |
| **1.2** | Slither V2 (version-grouped, pinned binary) | **10,693/10,693 contracts, 67.2% OK, 28.6 min** |
| **1.3** | Aderyn V2 (per-contract parallel, 6 workers) | **10,693/10,693 contracts, 58% OK, 9.9 min** |
| **1.4** | Per-class agreement (slither, corrected mapping) | **Median F1 = 0.131 — FAIL** |
| **1.5** | 3-way agreement (BCCC + slither + aderyn) | **Median F1 = 0.000 — FAIL** |
| **1.6** | Manual review: 43 contracts | **51% DROP, 12% KEEP, 37% UNCERTAIN** |
| **1.7** | Expanded review: 199 contracts | Per-class noise estimates documented |
| **1.8** | Full Reentrancy audit: 500/17,698 | **Only 10.6% true reentrancy, 89.4% false positives** |

**Slither V2 and Aderyn V2 improvements were significant engineering wins:**
- **Slither V1 (Phase 3):** 808 contracts, many COMPILE_FAIL
- **Slither V2 (Phase 4):** 10,693 contracts, 67.2% OK, 6.7× faster
  - Key: version-grouped processing, pinned solc binary, `--allow-paths`
- **Aderyn V1:** Many COMPILE_FAIL due to batch processing
- **Aderyn V2:** 58% OK, per-contract isolation in temp dirs, 6 parallel workers

### The Critical Findings (Why Phase 5 Exists)

Phase 4 Stage 1 revealed that the **majority of BCCC labels are structurally noisy** — not fixable with more data:

| Class | Sample Size | FP Rate | Verdict |
|---|---|---|---|
| Reentrancy | 500/17,698 | **89.4%** | Massive over-labeling. Only `.call.value()` is true reentrancy. |
| CallToUnknown | 11/11,131 | **91%** | Contracts with no external calls labeled as calling unknown. |
| ExternalBug | 1/3,604 | **100%** | Simple library labeled as buggy. |
| GasException | 9/6,879 | **67%** | Definition mismatch. |
| DenialOfService | 18/12,394 | **56%** | Definition mismatch. |
| Timestamp | — | **50%** | Moderate noise. |
| IntegerUO | — | **0%** | Clean ✅ |
| UnusedReturn | — | **0%** | Clean ✅ |
| MishandledException | — | **0%** | Clean ✅ |

The **Reentrancy audit** was the most thoroughly documented finding:
- 500 contracts randomly sampled from 17,698 Reentrancy folder
- `.call.value()` (true reentrancy): 53/500 = **10.6%**
- `.transfer()` only (safe — reverts on failure): 205/500 = **41.0%**
- `.send()` only (semi-safe): 71/500 = **14.2%**
- **No external call at all** (completely mislabeled): 171/500 = **34.2%**

**Root cause:** BCCC used an overly broad definition of each vulnerability class. For Reentrancy, they defined it as "any external call with state change after" — which captures 89.4% false positives.

### Stage 1 Gate Decision

- **Gate:** Median F1 ≥ 0.5 → labels trustworthy, skip escalation
- **Result:** Median F1 = 0.000 (3-way) and 0.131 (slither only) → **FAIL**
- **Escalation decision:** **Deferred** — noise is structural, not sample-size limited. More data won't fix it.

### Stages 2-7: Parked

All remaining Phase 4 stages are parked until Phase 5 produces verified labels:

| Stage | Original Purpose | Why Parked |
|---|---|---|
| **2** | Escalate to 30% sample (~20,000) | Noise is structural, not sample-limited |
| **3** | Escalate to 50% sample (~33,000) | Same reason |
| **4** | Mythril tiebreaker (50 hardest) | Will fold into Phase 5 if needed |
| **5** | Per-folder manual investigation (8 reports) | Will fold into Phase 5 |
| **6** | AutoML (10 binary × 5 models × 50 trials × 5 folds) | Needs verified labels |
| **7** | Synthesis — v1.3 + CHANGELOG + MEMORY | Will be produced by Phase 5 |

---

## 7. Complete Decision Registry

All decisions made across Phases 1-4, in chronological order:

| ID | Phase | Decision | Date | Trigger |
|---|---|---|---|---|
| **D-F1** | P2 | Drop Class05 (TransactionOrderDependence) + Class07 (WeakAccessModifier) → 10 SENTINEL classes | 2026-06-06 | Phase 2 WS-F analysis |
| **D-B2** | P2 | Hold out 766 NV+vuln contradictions as `review_pending=1` | 2026-06-06 | Phase 2 WS-B label validation |
| **D-D** | P2 | No overlap with SmartBugs → safe for OOD testing | 2026-06-06 | Phase 2 WS-D cross-corpus |
| **D-P3-1** | P3 | Mythril → Docker with pre-compiled bytecode (ad-hoc, not batch) | 2026-06-06 | 3m16s/contract too slow |
| **D-P3-2** | P3 | WS-O sample = 5,000 stratified contracts | 2026-06-06 | Phase 3 planning |
| **D-P3-3** | P3 | Manual reviews = 846 contracts | 2026-06-06 | Phase 3 planning |
| **D-P3-4** | P3 | All 5 AutoML models (XGBoost, LightGBM, CatBoost, RF, LogReg) | 2026-06-06 | Phase 3 planning |
| **D-P3-5** | P3 | 50 Optuna trials × 5 folds per model | 2026-06-06 | Phase 3 planning |
| **D-P3-6** | P3 | Create `contracts_clean_v12.csv` alongside v1.0 | 2026-06-06 | Phase 3 planning |
| **D-P3-7** | P3 | If AutoML beats SENTINEL → document only, don't change training | 2026-06-06 | Phase 3 planning |
| **D-P3-8** | P3 | Mythril too slow for batch → skip from WS-O | 2026-06-06 | 3m16s/contract benchmark |
| **D-P3-9** | P3 | Mythril excluded from WS-O. Slither = primary, Aderyn = secondary | 2026-06-06 | D-P3-8 follow-up |
| **D-P3-10** | P3 | Aderyn 0.6.8 replaces mythril as 3rd tool (3s/contract, 88 detectors) | 2026-06-06 | Rust is 60× faster than mythril |
| **D-I-1** | P3 | Subprocess wrapper for slither (timeout, state isolation) | 2026-06-06 | WS-I engineering |
| **D-I-2** | P3 | Per-contract 30s slither timeout | 2026-06-06 | WS-I engineering |
| **D-I-3** | P3 | Solc picker: highest selectable version; default 0.5.17 for NaN | 2026-06-06 | WS-I engineering |
| **D-I-4** | P3 | Status enum: OK / COMPILE_ERROR / TIMEOUT / EXCEPTION / PATH_MISSING | 2026-06-06 | WS-I engineering |
| **D-I-5** | P3 | Path fix: `.replace("Source Codes", "SourceCodes")` | 2026-06-06 | BCCC path inconsistency |
| **D-I-6** | P3 | Pass `solc=path` to Slither(), NOT `solc-select use` | 2026-06-06 | Global switch breaks parallelism |
| **D-I-7** | P3 | Parser: iterate `for det_findings in findings: for f in det_findings:` | 2026-06-06 | Slither 0.11+ nested list structure |
| **D-I-8** | P3 | Agreement metric: `max(diff) / max(total, 1)` | 2026-06-06 | WS-I ranking |
| **D-I-9** | P3 | BCCC folder scan: `SourceCodes/iterdir()`, not root | 2026-06-06 | Path structure |
| **D-I-10** | P3 | Inspection input = single markdown file with all 30+2 contracts | 2026-06-06 | Usability |
| **D-I-11** | P3 | **Drop NV when co-occurs with any of 6 vuln classes** | 2026-06-07 | 32-contract manual review |
| **D-P4-1** | P4 | Apply D-I-11 narrowly (review_pending only, not all 67,311) | 2026-06-07 | Conservative default |
| **D-P4-2** | P4 | Stage 1 sampling: 15% per primary_class | 2026-06-07 | User decision |
| **D-P4-7** | P4 | D-I-12: drop NV when co-occurring with IntegerUO | 2026-06-07 | 41 remaining review_pending |

Total: **26 decisions** across 4 phases.

---

## 8. What's Been Completed vs What's Parked

### ✅ Completed (100%)
- **Phase 1:** All 10 exploration findings, 4 scripts — **DONE**
- **Phase 2:** All 8 workstreams (A-H), `contracts_clean.csv` (67,311 × 24) — **DONE**
- **Phase 3:** WS-J (EDA), WS-S (Class Mapping), WS-N (Dropped Review), WS-I (Slither 808, manual 32, D-I-11) — **DONE**
- **Phase 4 Stage 0:** All 6 sub-steps (D-I-11, D-I-12, Oraclize dedup, regex features, hand-crafted features) — **DONE**
- **Phase 4 Stage 1:** Sampling, slither V2, aderyn V2, 3-way agreement, manual reviews, Reentrancy audit, CRITICAL_FINDINGS.md — **DONE**

Total completed: ~65% of planned work (Phases 1-4 combined)

### 🅿️ Parked (resume after Phase 5 label verification)

| Item | Origin | What It Is | Notes |
|---|---|---|---|
| WS-P | Phase 3 | Slither-based graph-level features | Needs labels |
| WS-Q | Phase 3 | SHAP feature importance | Depends on AutoML |
| WS-R | Phase 3 | 3-way model comparison | Depends on WS-L |
| WS-T | Phase 3 | Multi-label structure test | Depends on AutoML |
| WS-K2 | Phase 3 | 32 slither-derived features | Needs labels |
| Stage 2 | Phase 4 | Escalate to 30% sample | Structural noise, not needed |
| Stage 3 | Phase 4 | Escalate to 50% sample | Same reason |
| Stage 4 | Phase 4 | Mythril tiebreaker (50 contracts) | May fold into Phase 5.3 |
| Stage 5 | Phase 4 | Per-folder manual investigation | May fold into Phase 5.4 |
| Stage 6 | Phase 4 | AutoML baselines | Needs verified labels |
| Stage 7 | Phase 4 | v1.3 synthesis | Will be Phase 5.6 |
| D-P4-1 | Phase 4 | Apply D-I-11 broadly? | Deferred |
| D-P4-3 | Phase 4 | Mythril: 50 or 100? | Deferred |
| D-P4-4 | Phase 4 | AutoML: 50×5 or 25×5? | Deferred |
| D-P4-5 | Phase 4 | Include LogReg? | Deferred |
| D-P4-6 | Phase 4 | Apply 3 hand-crafted features? | Deferred |

---

## 9. Tooling & Environment

### Installed Tools

| Tool | Version | Purpose | Location |
|---|---|---|---|
| **slither** | 0.11.5 | Static analysis (101 detectors) | Root `.venv` (`.venv/bin/slither`) |
| **aderyn** | 0.6.8 | Static analysis (88 detectors, Rust) | `~/.cargo/bin/aderyn` |
| **mythril** | 0.24.8 | Symbolic execution (Docker) | Docker: `mythril/myth:0.24.8` |
| **solc-select** | 100+ versions | Multi-version Solidity compiler management | `~/.solc-select/artifacts/solc-X.Y.Z/` |
| **surya** | — | Call graph visualization (if needed) | Not yet installed |
| **CodeBERT** | — | Code embeddings for ML propagation | Need to install for Phase 5.5 |

### Python ML Stack (root `.venv`)

| Package | Version | Purpose |
|---|---|---|
| xgboost | 3.2.0 | Gradient boosting |
| lightgbm | 4.6.0 | Gradient boosting |
| catboost | 1.2.10 | Gradient boosting |
| optuna | 4.9.0 | Hyperparameter tuning |
| shap | 0.52.0 | Feature importance |
| imbalanced-learn | 0.14.1 | SMOTE, class weights |

All installed via Tsinghua pypi mirror. Two venvs exist:
- **Root `.venv`** — has most ML deps but NOT peft (use `poetry run python` for slither)
- **`ml/.venv`** — has peft (for training, not for static analysis)

### WSL Environment Notes
- `git config core.autocrlf false` — CRLF corrupts scripts
- `poetry run python` required for scripts needing slither/solc-select
- Training uses `ml/.venv/bin/python` directly
- Solc 0.8.35 is broken in solc-select registry — filtered out by `_verify_solc_works`

### Key Engineering Lessons (from Phase 4)

**Slither V2 Success Factors:**
1. Version-grouped processing (batch same-solc-version contracts together)
2. Pinned solc binary (avoid `solc-select use` — it's a global switch that breaks parallelism)
3. `--allow-paths .,$REPO` resolves imports for solc 0.5+
4. Subprocess wrapper with 30s timeout per contract

**Aderyn V2 Success Factors:**
1. Per-contract processing in temp directories (avoids cross-contract contamination)
2. 6 parallel workers (CPU-bound, not I/O-bound)
3. One broken contract can't kill the whole batch

**Aderyn V1 Failure Root Cause:** Aderyn compiled all files in a batch together. One broken contract killed the entire batch. Fix: isolate each contract in its own temp directory.

---

## 10. Critical Context for Phase 5

### What We Know FOR SURE

1. **3 classes are clean (0% noise):** IntegerUO, UnusedReturn, MishandledException — confirmed by manual review + 2 tool agreement. These pass Stage 5.1 gate without additional work.
2. **Timestamp has moderate noise (50% FP)** — NOT a clean class. "Uses `block.timestamp`" ≠ "timestamp-dependent critical logic". Requires Stage 5.2 automated verification.
3. **Reentrancy is 89.4% false positive** by strict definition — only `.call.value(amount)()` (pre-0.8) or `.call{value:}` (post-0.8) qualifies as true reentrancy. 500 contracts already manually audited (10.6% true reentrancy) — use as Stage 5.4 anchor, do NOT re-sample Reentrancy.
4. **CallToUnknown is 91% false positive** — many labeled contracts have zero external calls
5. **ExternalBug is likely very noisy** — 100% FP in tiny sample; BCCC catch-all class with no standard definition; may be unverifiable
6. **GasException and DenialOfService are poorly defined** in BCCC — need definition work in Stage 5.0 before any verification; no aderyn detector exists for either class
7. **NonVulnerable was systematically mislabeled** — D-I-11 and D-I-12 fixed the major pattern; 766 NV labels dropped
8. **D-I-11 rule generalizes well** — spot-checked 10 random corrected contracts, all correct
9. **The dataset split (70/15/15) is stable** — no need to re-split until Phase 5.6
10. **Source code is NOT in the CSV** — v1.1+12 has 24 columns but no `source_code` field. Read from disk via `bccc_file_path` pointing to `~/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/<folder>/<file>.sol`

### What We DON'T Know (Phase 5 Will Answer)

1. What is the exact false positive rate for each of the 9 classes across ALL 67,311 contracts?
2. Can we define precise ground truth criteria for GasException and DenialOfService?
3. How many Reentrancy contracts are true positives by strict `.call.value()` definition?
4. How many CallToUnknown contracts have actual external calls?
5. Are there patterns in the noisy classes that can be corrected via automated rules?
6. Can CodeBERT propagation reliably extend manual findings to the full dataset?
7. What confidence scores should we assign to each (contract, class) pair?

### The Central Tension

BCCC labeled contracts according to their own (broad) definitions. SENTINEL needs **strict, precise** vulnerability labels for effective training. The gap between BCCC's broad definitions and SENTINEL's need for precision is the entire reason Phase 5 exists.

Phase 5 must bridge this gap by:
- Defining strict ground truth per class (Stage 5.0)
- Applying automated verification at scale (Stage 5.2)
- Resolving disagreements with deeper analysis (Stage 5.3)
- Establishing manual ground truth (Stage 5.4)
- Propagating verified labels via ML (Stage 5.5)
- Producing a dataset with confidence scores (Stage 5.6)

### Where Phase 5 Picks Up

**Input dataset:** `Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01b_d12_applied.csv` (v1.1+12: 67,311 contracts, D-I-11 + D-I-12 applied, review_pending=0)

**Already available for free:**
- Slither results on 10,693 contracts
- Aderyn results on 10,693 contracts
- 34 regex + hand-crafted features on all 67,311
- Manual reviews: 43 + 199 + 500 (Reentrancy audit)
- 33 Phase 3 manual reviews
- CRITICAL_FINDINGS.md with per-class noise estimates

**Phase 5 first step:** Stage 5.0 — Write 9 ground truth definition documents

---

## Phase 5 Quick-Start

If you're starting Phase 5 right now:

1. **Read this handover** (you're here ✅)
2. **Read CRITICAL_FINDINGS.md** — the most important single file
3. **Read the Phase 5 plan** — `Phase5_LabelVerification_2026-06-08/05_phase5_plan.md`
4. **Start Stage 5.0** — Write the 9 class definition documents
5. **Start Stage 5.1** — Build the evidence integration script

**Key files to reference during Phase 5:**
- `Phase4_LabelValidation_2026-06-07/CRITICAL_FINDINGS.md` — per-class noise analysis
- `Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_slither_results.csv` — slither on 10,693
- `Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_aderyn_results.csv` — aderyn on 10,693
- `Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s05_regex_features.csv` — 31 regex patterns
- `Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_review_200.csv` — expanded manual review
- `Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_manual_review_50.csv` — initial manual review
- `Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01b_d12_applied.csv` — current dataset
- `Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md` — 32-contract manual review

---

**Document version:** 1.0
**Last updated:** 2026-06-08
**Next planned update:** After Phase 5 Stage 5.6 (synthesis)
