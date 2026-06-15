# BCCC-SCsVul-2024 Deep Dive

**Date:** 2026-06-08
**Project:** SENTINEL (decentralised smart contract security oracle)
**Purpose:** Validate, clean, and prepare the BCCC-SCsVul-2024 dataset for SENTINEL training.
**Source of truth:** `BCCC-SCsVul-2024/` (1.6 GB, **read-only**)
**Total deliverables:** 120+ files across 25 directories
**Current phase:** Phase 5 — Comprehensive Label Verification (see below)

---

## TL;DR — The 30-Second Version

1. **BCCC is the biggest smart-contract vulnerability corpus we could find** (111,897 label rows = 68,433 unique contracts, 12 vulnerability classes, multi-label).
2. **Phase 1** explored it and found 10 structural issues (multi-label, 38.8% duplication, 766 logical contradictions, etc.).
3. **Phase 2** (8 workstreams, ✅ complete) produced SENTINEL-ready `contracts_clean.csv` (67,311 contracts × 24 cols).
4. **Phase 3** validated labels via slither (808 contracts) + 32-contract manual review → **D-I-11**: drop NonVulnerable when co-occurring with real vuln classes.
5. **Phase 4 Stage 1** scaled to 10,693 contracts (slither V2 + aderyn V2) and performed 3-way agreement + 199-contract manual review + 500-contract Reentrancy audit → **CRITICAL FINDING: 89% of Reentrancy labels are false positives. Label noise is structural, not sample-size limited.**
6. **Phase 5** (started 2026-06-08) abandons the old approach. Its sole mission: systematically verify ALL 67,311 BCCC labels using every available method (static analysis, regex, symbolic execution, manual review, CodeBERT propagation) before any AutoML or model training.

---

## How to Read This Folder

| If you are… | Read this first |
|---|---|---|
| A reviewer/collaborator with 5 minutes | This README (you're here) |
| **Starting Phase 5 (label verification)** | **`Phase5_LabelVerification_2026-06-08/06_handover_p1_to_p4.md`** (comprehensive handover of all phases) |
| A reviewer with 30 minutes | This README → Handover doc → `Phase2_Validation_2026-06-06/README.md` → `Phase4_LabelValidation_2026-06-07/CRITICAL_FINDINGS.md` |
| Re-running the work yourself | This README → `02_validation_deep_dive_plan.md` (Phase 2) → `03_phase3_plan.md` (Phase 3) → `04_phase4_plan.md` (Phase 4) → `05_phase5_plan.md` (Phase 5) → the relevant `scripts/` |
| Looking for the final dataset for SENTINEL training | **Current version:** `Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01b_d12_applied.csv` (v1.1+12) **or wait for** `Phase5_LabelVerification_2026-06-08/outputs/contracts_clean_v13.csv` |
| Understanding why we can't trust labels yet | [`Phase4_LabelValidation_2026-06-07/CRITICAL_FINDINGS.md`](Phase4_LabelValidation_2026-06-07/CRITICAL_FINDINGS.md) — Reentrancy 89% FP, CallToUnknown 91% FP |

---

## Directory Layout

```
BCCC-SCsVul-2024_Deep_Dive/                                 120+ files, 25 dirs
├── README.md                                              [this file - the entry point]
├── 01_exploration_inventory.md                            [Phase 1: 10 findings]
├── 02_validation_deep_dive_plan.md                        [Phase 2 plan: 8 workstreams]
├── 03_phase3_plan.md                                      [Phase 3 plan: 6 workstreams, 4 sessions]
├── 04_phase4_plan.md                                      [Phase 4 plan: 7 stages — now superseded by Phase 5]

├── scripts/                                               [Phase 1: 4 exploration scripts]
│   ├── README.md
│   ├── bccc_phase1_explore.py
│   ├── bccc_phase1_explore2.py
│   ├── bccc_phase1_explore3.py
│   └── bccc_phase1_explore4.py
│
├── Phase2_Validation_2026-06-06/                          [Phase 2: 8 workstreams, ~100MB]
│   ├── README.md                                          [Phase 2 entry point]
│   ├── 00_session_log.md                                  [Phase 2 timeline]
│   ├── scripts/                                           [WS-A through WS-H, 9 scripts]
│   ├── integrity/                                         [WS-A: SHA256 + dedup map]
│   │   ├── manifest.md
│   │   └── dedup_map.csv
│   ├── labels/                                            [WS-B, F: label consistency + reconciliation]
│   │   ├── label_validation_report.md
│   │   ├── class_reconciliation_decision.md               [D-F1: drop Class05/07]
│   │   ├── contracts_filtered.csv
│   │   ├── dropped_contracts.csv
│   │   ├── review_pending_ids.csv                         [766 NV+vuln contradictions]
│   │   ├── label_consistency.csv
│   │   ├── folder_csv_consistency.csv
│   │   ├── class_cooccurrence.csv
│   │   ├── samples_nv_vuln.csv
│   │   └── samples_multi_pos.csv
│   ├── compile/                                           [WS-C: solc compilation probe]
│   │   ├── compilation_report.md                          [73% compile rate]
│   │   └── compile_results.csv
│   ├── complexity/                                        [WS-E: per-class complexity stats]
│   │   ├── complexity_report.md
│   │   ├── per_class_stats.csv
│   │   └── per_contract_stats.csv
│   ├── cross_corpus/                                      [WS-D: BCCC vs SmartBugs overlap]
│   │   ├── overlap_report.md                              [0 overlap = no test leakage]
│   │   └── bccc_vs_smartbugs_overlap.csv
│   ├── splits/                                            [WS-G: 70/15/15 stratified split]
│   │   ├── split_summary.md
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── outputs/                                           [WS-H: FINAL SENTINEL DATASET]
│       ├── README.md                                      [The v1.0 schema + usage]
│       ├── contracts_clean.csv                            [67,311 × 24 — MAIN DELIVERABLE]
│       └── split_assignments.csv
│
├── Phase3_DeepAnalysis_2026-06-06/                        [Phase 3: ~40% complete, 3.4MB]
│   ├── README.md                                          [Phase 3 entry point]
│   ├── 00_understanding_checklist.md                      [Teaching doc: Problem/Solution/Context per stage]
│   ├── 01_session_log.md                                  [Session 1+2 timeline]
│   ├── scripts/                                           [WS-I + WS-J + WS-N, 11 scripts]
│   ├── outputs/                                           [WS-J + WS-I + WS-N artifacts]
│   ├── reports/                                           [WS-J, WS-S, WS-N reports]
│   └── labels/                                            [WS-I: 32-contract manual review]
│       └── ws_i_inspections_input.md                      [8,014 lines, input to D-I-11]
│
├── Phase4_LabelValidation_2026-06-07/                    [Phase 4: ~35% complete, critical findings]
│   ├── README.md                                          [Phase 4 entry point]
│   ├── 00_actionable_checklist.md                        [Session tracker]
│   ├── 01_session_log.md                                  [Session 1 timeline]
│   ├── 04_phase4_plan.md                                  [Original Phase 4 plan — superseded]
│   ├── CRITICAL_FINDINGS.md                              [**READ FIRST: Why Phase 5 exists**]
│   ├── scripts/                                           [Stage 0 + Stage 1 scripts]
│   │   ├── ws_p4_s01_apply_d11.py
│   │   ├── ws_p4_s01b_apply_d12.py
│   │   ├── ws_p4_s02_oraclize_dedup.py
│   │   ├── ws_p4_s03_exclude_reviewed.py
│   │   ├── ws_p4_s05_regex_features.py
│   │   ├── ws_p4_s06_handcrafted_features.py
│   │   ├── ws_p4_s1_sampling.py
│   │   ├── ws_p4_s1_slither.py                           [V2: 10,693 contracts, 67.2% OK]
│   │   ├── ws_p4_s1_aderyn.py                            [V2: 10,693 contracts, 58% OK]
│   │   ├── ws_p4_s1_agreement.py
│   │   └── ws_p4_s1_3way_agreement.py
│   ├── outputs/
│   │   ├── ws_p4_s01b_d12_applied.csv                    [v1.1+12 dataset — current best]
│   │   ├── ws_p4_s1_slither_results.csv                  [Slither on 10,693]
│   │   ├── ws_p4_s1_aderyn_results.csv                   [Aderyn on 10,693]
│   │   ├── ws_p4_s1_3way_agreement_report.md             [3-way: median F1=0.000]
│   │   ├── ws_p4_s1_review_200.csv                       [199-contract expanded review]
│   │   ├── ws_p4_s1_manual_review_50.csv                 [43-contract initial review]
│   │   └── ws_p4_s05_regex_features.csv                  [31 regex on all 67,311]
│   ├── reports/
│   ├── labels/
│   ├── decisions/                                        [D-I-12 + D-P4-7]
│   └── checkpoints/
│
└── Phase5_LabelVerification_2026-06-08/                   [Phase 5: Comprehensive Label Verification — NEW]
    ├── 05_phase5_plan.md                                  [Phase 5 plan — 6 stages, gated]
    ├── 06_handover_p1_to_p4.md                           [Full handover: Phases 1-4 findings]
    ├── scripts/
    ├── outputs/
    ├── reports/
    ├── labels/
    └── decisions/
```

---

## The Three Phases at a Glance

### Phase 1: Exploration & Inventory ✅ (435 lines)
**Goal:** Understand BCCC's actual structure before cleaning it.
**File:** [`01_exploration_inventory.md`](01_exploration_inventory.md)

**10 key findings:**
1. **Multi-label:** 41% of contracts have ≥2 simultaneous vulnerability labels
2. **68,433 unique contracts** (not 111,897); 38.8% are exact byte-identical copies
3. **12 folders = "candidate categories"** (tool-flagged); **CSV = ground truth** (verified labels)
4. **Long-format CSV:** 111,897 rows = 68,433 IDs × 1.635 classes
5. **Severe class imbalance:** top-3 (NV, Reentrancy, IntegerUO) = 65% of positive labels
6. **766 contradictory contracts** labeled BOTH NV AND a real vulnerability
7. **Top co-occurrence:** DoS+Reentrancy = 12,381 contracts (18% of dataset)
8. **CSV "ID" is keccak-256 of bytecode** (not sha256 of source content; 95.5% mismatch)
9. **CSV integrity verified** (md5 `e38a2aa1c2b8a93c6cf8b23d2d7b870a`); per-file content NOT independently verifiable
10. **92% pre-0.6 Solidity** (mostly 0.4.x/0.5.x)

**Scripts:** [`scripts/`](scripts/) (4 explore scripts, all read-only on BCCC)

### Phase 2: Validation & Cleaning ✅ (100 MB artifacts, 8 workstreams)
**Goal:** Produce a SENTINEL-ready dataset.
**File:** [`Phase2_Validation_2026-06-06/README.md`](Phase2_Validation_2026-06-06/README.md)
**Plan:** [`02_validation_deep_dive_plan.md`](02_validation_deep_dive_plan.md)

| WS | Title | Headline output |
|---|---|---|
| **A** | Integrity & Dedup | SHA256 manifest, dedup map (38.8% duplication) |
| **B** | Label Validation | 766 NV+vuln contradictions, 100% folder↔class agreement |
| **C** | Compile Probe | 73% compile rate with solc 0.4.24/0.5.17 |
| **D** | Cross-Corpus | **0 overlap with SmartBugs** (no test leakage) |
| **E** | Complexity | Per-class LOC, n_functions, n_events, n_modifiers |
| **F** | Class Reconciliation | **D-F1: drop Class05/07** (1,122 contracts dropped, 10 SENTINEL classes) |
| **G** | Stratified Split | 70/15/15 → 46,581 / 9,982 / 9,982 (review_pending held out) |
| **H** | Final Dataset | `contracts_clean.csv` (67,311 × 24, SHA256 `53b7b884c3ae...`) |

**Main deliverable:** [`Phase2_Validation_2026-06-06/outputs/contracts_clean.csv`](Phase2_Validation_2026-06-06/outputs/contracts_clean.csv) — 67,311 contracts × 24 cols in SENTINEL v9 schema (10 classes, 10 binary heads).

### Phase 3: Deep Analysis ✅ (WS-J, WS-S, WS-N, WS-I complete; 4 workstreams merged to Phase 4; 4 never started)
**Goal:** Validate the *labels* with an independent static analyzer (slither).
**File:** [`Phase3_DeepAnalysis_2026-06-06/README.md`](Phase3_DeepAnalysis_2026-06-06/README.md)
**Plan:** [`03_phase3_plan.md`](03_phase3_plan.md)

| WS | Title | Status | Output |
|---|---|---|---|---|
| **J** | Statistical EDA | ✅ Done (Session 1) | 0% missingness, 4.9× imbalance, 60.6% single-label |
| **S** | Class Mapping | ✅ Done (Session 1) | 10/12 BCCC → DASP 10; Class05→Front Running, Class07→Access Control (both dropped) |
| **N** | Dropped + Review-Pending Breakdown | ✅ Done (Session 1) | 1,122 dropped (163 Class05-only, 959 Class07-only); 766 review-pending (705 n_pos=3) |
| **I** | Slither Label Validation (808 contracts) | ✅ Done (Sessions 2+3) | Per-class agreement metrics + 32-contract manual review → **D-I-11** |
| **O** | Aderyn 3-Way Consensus (5,000 contracts) | 🅿️ Merged into Phase 4 | Done in Phase 4 Stage 1 (10,693 contracts) |
| **K-K1** | 31 Source-Code Regex Features | 🅿️ Merged into Phase 4 | Done in Phase 4 Stage 0.5 |
| **L** | AutoML 5-Model × 50 Optuna Trials × 5 Folds | 🅿️ Parked — needs verified labels | Will resume after Phase 5 |
| **M** | Dataset v1.2 + CHANGELOG §46 | 🅿️ Replaced by Phase 5 v1.3 | Will be Phase 5.6 |
| **P** | Slither Graph-Level Features | ❌ Never started — parked | After Phase 5 |
| **Q** | SHAP Feature Importance | ❌ Never started — parked | After Phase 5 |
| **R** | 3-Way Model Comparison | ❌ Never started — parked | After Phase 5 |
| **T** | Multi-Label Structure Test | ❌ Never started — parked | After Phase 5 |
| **K2** | 32 Slither-derived features | ❌ Never started — parked | After Phase 5 |

**WS-I Stage 2 headline results (808 contracts, slither 0.11.5):**

| Class | F1 | Precision | Recall | Verdict |
|---|---|---|---|---|
| **Reentrancy** | **0.51** | **0.93** | 0.35 | ✅ High precision — labels trustworthy when positive |
| **CallToUnknown** | 0.33 | 0.92 | 0.20 | ✅ Same — high precision |
| Timestamp | 0.15 | 0.09 | 0.53 | ⚠️ Slither over-fires (lots of false positives) |
| IntegerUO | 0.07 | 0.07 | 0.07 | ❌ Slither can't detect pre-0.8 overflow — needs Aderyn |
| 6 other classes | <0.05 | — | — | Too few samples (n=7-19) for statistical power |

**Manual review (32 contracts):** 0 KEEPs, 28 MODIFY drop-NV, 2 MODIFY reclassify, 6 template clusters identified → **D-I-11** (drop NV when co-occurs with vuln classes). Unlocked ~705 contracts for training.

See also: [`Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md`](Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md) (425 lines)

### Phase 4: Label Trustworthiness Validation 🟡 (Stage 0 ✅, Stage 1 ✅ with critical findings, Stages 2-7 parked)
**Goal:** Validate labels per-folder with stratified sampling, apply D-I-11 + D-I-12 mass corrections, scale up slither/aderyn, measure 3-way agreement.
**File:** [`Phase4_LabelValidation_2026-06-07/README.md`](Phase4_LabelValidation_2026-06-07/README.md)
**Critical findings:** [`Phase4_LabelValidation_2026-06-07/CRITICAL_FINDINGS.md`](Phase4_LabelValidation_2026-06-07/CRITICAL_FINDINGS.md) — **read this before Phase 5**

**Stage 0 results (2026-06-07, ✅ complete):**
- v1.1+12 dataset: 67,311 contracts, 766 NV labels dropped (725 by D-I-11 + 41 by D-I-12), `review_pending=0`
- 136 Oraclize clusters, 548 duplicates flagged
- 33 reviewed contracts excluded → 66,730 eligible for sampling
- 31 regex features + 3 hand-crafted features computed on all 67,311

**Stage 1 results (2026-06-07/08, ✅ complete — gate FAILED):**
- Slither V2 (10,693 contracts, 67.2% OK, 28.6 min)
- Aderyn V2 (10,693 contracts, 58% OK, 9.9 min)
- **3-way agreement: median F1 = 0.000 — FAIL**
- Manual review 43 contracts: **51% DROP, 12% KEEP**
- Expanded review 199 contracts: per-class noise estimates
- **Reentrancy audit 500/17,698: 89.4% false positives**
- **CRITICAL_FINDINGS.md** published documenting structural label noise across 5/9 classes

**Key finding:** Label noise is **structural** (BCCC definitions are too broad), not sample-size limited. Escalation deferred. Phase 5 created to address this.

### Phase 5: Comprehensive Label Verification 🆕 (Design — execution pending)
**Goal:** Systematically verify ALL 67,311 BCCC labels using every available method before any AutoML or model training. Produce verified dataset with per-contract confidence scores.
**File:** [`Phase5_LabelVerification_2026-06-08/05_phase5_plan.md`](Phase5_LabelVerification_2026-06-08/05_phase5_plan.md)
**Handover (Phases 1-4):** [`Phase5_LabelVerification_2026-06-08/06_handover_p1_to_p4.md`](Phase5_LabelVerification_2026-06-08/06_handover_p1_to_p4.md)

**6 stages, incrementally gated:**

| Stage | Purpose | Methods | Gate |
|---|---|---|---|
| **5.0** | Ground truth definitions | 9 class definition docs | All definitions reviewed |
| **5.1** | Existing evidence integration | Merge slither/aderyn/manual/regex data | Per-class evidence sufficiency |
| **5.2** | Bulk automated verification | Regex, static analysis on all 67,311 | Per-class agreement ≥ 80% |
| **5.3** | Discrepancy resolution | Mythril, structural analysis, manual | ≥ 90% disputes resolved |
| **5.4** | Manual ground truth | 200-contract systematic review | ≥ 20 per noisy class |
| **5.5** | ML-assisted propagation | CodeBERT embedding + clustering | Propagation ≥ 85% accuracy |
| **5.6** | Synthesis | `contracts_clean_v13.csv` with confidence scores | Complete dataset + report |

**Status:** ⏳ Stage 5.0 (definitions) — first step

---

## Key Decisions

| ID | Decision | Date | Doc |
|---|---|---|---|---|
| **D-F1** | Drop BCCC Class05 (TransactionOrderDependence) + Class07 (WeakAccessMod) — 1,122 contracts dropped, 10 SENTINEL classes | 2026-06-06 | [`Phase2_Validation_2026-06-06/labels/class_reconciliation_decision.md`](Phase2_Validation_2026-06-06/labels/class_reconciliation_decision.md) |
| **D-B2** | Hold out 766 NV+vuln contradictions as `review_pending=1` for manual review | 2026-06-06 | [`Phase2_Validation_2026-06-06/labels/label_validation_report.md`](Phase2_Validation_2026-06-06/labels/label_validation_report.md) |
| **D-D** | No byte-identical overlap with SmartBugs → safe to use SmartBugs as OOD test set | 2026-06-06 | [`Phase2_Validation_2026-06-06/cross_corpus/overlap_report.md`](Phase2_Validation_2026-06-06/cross_corpus/overlap_report.md) |
| **D-P3-10** | **Aderyn 0.6.8 (Cyfrin, Rust) replaces mythril as 3rd tool.** 3s/contract, 88 detectors, JSON/MD/SARIF, no Docker. 3-way consensus: BCCC vs slither vs aderyn. | 2026-06-06 | [`03_phase3_plan.md`](03_phase3_plan.md) |
| **D-I-11** | **Drop `Class12:NonVulnerable` when co-occurring with any of {CallToUnknown, Reentrancy, GasException, MishandledException, DenialOfService, Timestamp}.** Unlocks ~705 contracts. Based on 32-contract manual review: 28/30 had NV + confirmed vulnerability. | 2026-06-07 | [`Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md`](Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md) |
| **D-I-12** | Drop NV when co-occurring with IntegerUO. Resolves 41 remaining `review_pending` contracts. | 2026-06-07 | [`Phase4_LabelValidation_2026-06-07/decisions/D-I-12_drop_nv_with_integeruo.md`](Phase4_LabelValidation_2026-06-07/decisions/D-I-12_drop_nv_with_integeruo.md) |
| **D-P4-1** | Apply D-I-11 narrowly (review_pending only, not all 67,311) — conservative default | 2026-06-07 | [`Phase4_LabelValidation_2026-06-07/00_actionable_checklist.md`](Phase4_LabelValidation_2026-06-07/00_actionable_checklist.md) |
| **D-P4-2** | Stage 1 sampling: 15% per primary_class | 2026-06-07 | same |
| **D-P4-7** | Add D-I-12: drop NV when co-occurring with IntegerUO | 2026-06-07 | [`Phase4_LabelValidation_2026-06-07/decisions/D-I-12_drop_nv_with_integeruo.md`](Phase4_LabelValidation_2026-06-07/decisions/D-I-12_drop_nv_with_integeruo.md) |

---

## Critical Context for Future Sessions

### What "validation" means here
We did **not** manually read all 67,311 contracts. We used a combination of:
1. **Static analysis** (slither + aderyn) on 10,693 sampled contracts
2. **Manual review** of 43 + 199 + 500 contracts across multiple phases
3. **Regex analysis** of all 67,311 contracts (34 patterns)
4. **D-I-11/D-I-12** rule-based corrections from manual findings

The 30 worst-disagreement contracts from Phase 3 are in [`Phase3_DeepAnalysis_2026-06-06/labels/ws_i_inspections_input.md`](Phase3_DeepAnalysis_2026-06-06/labels/ws_i_inspections_input.md). The expanded 199-contract review and 500-contract Reentrancy audit are in [`Phase4_LabelValidation_2026-06-07/outputs/`](Phase4_LabelValidation_2026-06-07/outputs/).

### Label quality verdict (Phase 5 starting point)
- ✅ **Clean (0% noise):** IntegerUO, UnusedReturn, MishandledException, Timestamp
- ❌ **Heavy noise (56-100% FP):** Reentrancy (89%), CallToUnknown (91%), ExternalBug (100%), GasException (67%), DenialOfService (56%)
- ✅ **NV fixed:** D-I-11 + D-I-12 applied (766 corrections, `review_pending=0`)
- ⏳ **AutoML deferred** until Phase 5 completes label verification

### The central problem
BCCC's vulnerability definitions are **overly broad**. Their Reentrancy class includes any contract with an external call and state change after — including `.transfer()` (which reverts on failure and is safe). Their CallToUnknown class includes contracts with zero external calls. Their GasException/DenialOfService classes lack precise definitions. This means training on BCCC labels as-is would teach SENTINEL incorrect patterns rather than true vulnerabilities.

Phase 5 exists to bridge this gap: define strict ground truth, verify labels systematically, and produce a dataset SENTINEL can actually learn from.

### Reproducibility
All Phase 2 scripts are in [`Phase2_Validation_2026-06-06/scripts/`](Phase2_Validation_2026-06-06/scripts/) and are idempotent. All Phase 3 scripts are in [`Phase3_DeepAnalysis_2026-06-06/scripts/`](Phase3_DeepAnalysis_2026-06-06/scripts/). The BCCC source tree is **never modified** (read-only).

### Tools installed
- **slither 0.11.5** (root `.venv`, eth-abi 5.2.0, eth-utils 6.0.0)
- **aderyn 0.6.8** at `~/.cargo/bin/aderyn` (Cyfrin, Rust, 88 detectors, 3s/contract)
- **AutoML stack** (root `.venv`, via Tsinghua pypi mirror): xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10, optuna 4.9.0, shap 0.52.0, imbalanced-learn 0.14.1
- **mythril 0.24.8 Docker** (kept for ad-hoc deep analysis, not batch — too slow)
- **solc-select** with 100+ solc versions at `~/.solc-select/artifacts/solc-X.Y.Z/solc-X.Y.Z`

### Memory & cross-references
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` (190 lines) — has a Phase 3 in-flight entry
- `docs/CHANGELOG.md` — needs §46 entry after Stage 4 of Phase 3
- `docs/ml/adr/INDEX.md` — ADR-0005 (BCCC dataset choice) is the architectural *why*

---

## Current Phase: Phase 5 — Comprehensive Label Verification 🆕

**Status:** ⏳ Design complete, execution pending
**Plan:** [`Phase5_LabelVerification_2026-06-08/05_phase5_plan.md`](Phase5_LabelVerification_2026-06-08/05_phase5_plan.md)
**Handover:** [`Phase5_LabelVerification_2026-06-08/06_handover_p1_to_p4.md`](Phase5_LabelVerification_2026-06-08/06_handover_p1_to_p4.md)

Phase 5 verifies all 67,311 BCCC labels across 9 classes using 6 incremental stages (each with gates). After Phase 5, verified labels enable resuming the parked workstreams (AutoML, graph features, SHAP, etc.).

**Immediate next steps:**
1. Stage 5.0: Write 9 ground truth definition documents (one per class)
2. Stage 5.1: Build evidence integration from existing Phase 1-4 data
3. Stage 5.2: Run bulk automated verification on all 67,311 contracts

---

## Related

- `BCCC-SCsVul-2024/` — source dataset (read-only, 1.6 GB)
- `docs/ml/adr/INDEX.md` — ADR-0005 (BCCC dataset choice) — architectural *why*
- `docs/CHANGELOG.md` — version history (needs entries after Phase 5)
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` — SENTINEL project memory
- `ml/data/processed/multilabel_index.csv` — SENTINEL's v10 label file (uses this BCCC v1.0)
- `ml/data/splits/v10_deduped/` — SENTINEL's v10 splits (uses this BCCC v1.0's `split_assignments.csv`)

---
**Last updated:** 2026-06-08 (Phase 4 Stage 1 critical findings, Phase 5 created, all pending work parked)
