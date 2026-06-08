# BCCC-SCsVul-2024 Deep Dive — Phase 3: Deep Analysis

**Title:** Phase 3 Deep Analysis — Label Validation via Independent Static Analysis (Slither + Aderyn)
**Date started:** 2026-06-06 (Session 1)
**Date in flight:** 2026-06-06 (Session 2)
**Author:** SENTINEL Data Engineering
**Source dataset:** `BCCC-SCsVul-2024/` (1.6 GB, read-only)
**Phase 1 reference:** [`../01_exploration_inventory.md`](../01_exploration_inventory.md) (435 lines, 10 findings)
**Phase 2 reference:** [`../Phase2_Validation_2026-06-06/README.md`](../Phase2_Validation_2026-06-06/README.md) (8 workstreams, complete)
**Phase 2 deliverable:** [`../Phase2_Validation_2026-06-06/outputs/contracts_clean.csv`](../Phase2_Validation_2026-06-06/outputs/contracts_clean.csv) (67,311 × 24, SENTINEL v1.0)
**Phase 3 plan reference:** [`../03_phase3_plan.md`](../03_phase3_plan.md) (~1030 lines, 6 workstreams, 4-5 sessions)
**Root README:** [`../README.md`](../README.md) — table of contents for the whole deep dive
**Status:** 🟡 **In flight** — Session 1 (WS-J/S/N) done; Session 2 (WS-I) Stage 2 done; manual review of 30 contracts pending

---

## 1. Purpose

Phase 2 cleaned the *file* integrity. **Phase 3 cleans the *label* integrity** — it validates BCCC's vulnerability labels using an independent static analyzer (slither, plus aderyn in Session 3) so that SENTINEL can train on trustworthy ground truth.

**Why this matters:** SENTINEL's Run 7 F1=0.3074 and Run 9 v11 F1=0.2586 could be capped by label noise. If 30% of "Reentrancy" labels are actually clean contracts, SENTINEL wastes capacity learning to detect patterns that aren't there. Cleaning labels could raise the F1 ceiling by 5-15 points.

**No source files in `BCCC-SCsVul-2024/` are modified.** All writes are to this Phase 3 directory and downstream `outputs/`, `labels/`, `reports/`.

---

## 2. Workstream Index

| WS | Title | Status | Output directory | Top-level deliverable |
|---|---|---|---|---|
| **J** | Statistical EDA on 67,311 clean contracts | ✅ Done (Session 1) | `outputs/`, `reports/` | `ws_j_eda_report.md` + 4 CSVs |
| **S** | BCCC class → DASP mapping | ✅ Done (Session 1) | `reports/` | `ws_s_class_mapping.md` |
| **N** | Dropped + review-pending breakdown | ✅ Done (Session 1) | `outputs/`, `reports/` | `ws_n_dropped_review_report.md` + 2 CSVs |
| **I** | Slither label validation (808 contracts) | 🟡 Stage 2 done; manual review pending (Session 2) | `outputs/`, `labels/` | `ws_i_agreement_report.md` + 30-contract inspection input |
| **O** | Aderyn 3-way consensus (5,000 contracts) | ⏳ Pending (Session 3) | `outputs/`, `labels/` | `ws_o_aderyn_consensus.md` |
| **K-K1** | 31 source-code regex features on 67,311 contracts | ⏳ Pending (Session 3-4) | `outputs/` | `ws_k_regex_features.csv` |
| **L** | AutoML 5-model × 50 Optuna trials × 5 folds | ⏳ Pending (Session 4-5) | `outputs/`, `reports/` | `ws_l_automl_report.md` |
| **M** | Dataset v1.2 + CHANGELOG §46 + MEMORY update | ⏳ Pending (Session 5) | `outputs/`, `../docs/` | `contracts_clean_v12.csv` + CHANGELOG entry |

**Total estimated:** 35-50 h, 5-6 sessions

---

## 3. Headline Results So Far

### WS-J (Statistical EDA) ✅
- 0% missingness
- 4.9× class imbalance
- 60.6% single-label, 39.4% multi-label
- 92% of review-pending = NV + Reentrancy + CallToUnknown (3 labels)

### WS-S (BCCC class → DASP mapping) ✅
- 10/12 BCCC classes map to DASP categories
- Class05 (TransactionOrderDependence) → DASP "Front Running" (dropped — D-F1)
- Class07 (WeakAccessMod) → DASP "Access Control" (dropped — D-F1)

### WS-N (Dropped + review-pending breakdown) ✅
- 1,122 dropped (163 Class05-only, 959 Class07-only)
- 766 review-pending (705 n_pos=3, 703 = NV + Reentrancy + CallToUnknown)

### WS-I (Slither label validation, 808 contracts) 🟡

**Run results:**
- 808 contracts → 757 OK + 51 EXCEPTION (6.3% compile fail rate, vs 27% expected)
- **33,049 slither findings** across 60 unique detectors
- 100% of contracts have at least one finding (mostly `naming-convention` quality noise)

**Per-class agreement (BCCC label vs slither hit on mapped detectors):**

| Class | F1 | Precision | Recall | Verdict |
|---|---:|---:|---:|---|
| **Reentrancy** | **0.51** | **0.93** | 0.35 | ✅ BCCC labels trustworthy when positive |
| **CallToUnknown** | 0.33 | 0.92 | 0.20 | ✅ Same — high precision |
| Timestamp | 0.15 | 0.09 | 0.53 | ⚠️ Slither over-fires |
| IntegerUO | 0.07 | 0.07 | 0.07 | ❌ Slither can't detect pre-0.8 overflow — needs Aderyn (D-P3-10) |
| 6 other classes | <0.05 | — | — | Too few samples (n=7-19) for statistical power |

**Overall:** macro-F1=0.13, micro-F1=0.24, micro-P=0.37, micro-R=0.18

**Top slither detectors that fired:**
1. `naming-convention` × 15,088 (quality noise)
2. `deprecated-standards` × 2,901 (pragma noise)
3. `dead-code` × 2,345 (quality)
4. `solc-version` × 1,533 (pragma)
5. `external-function` × 1,311 (quality)

**Manual review input:** 30 worst-disagreement + 2 maxing contracts dumped to [`labels/ws_i_inspections_input.md`](labels/ws_i_inspections_input.md) (8,014 lines, with BCCC folders + source code + BCCC labels + slither findings + Decision checkboxes).

---

## 4. Directory Layout

```
Phase3_DeepAnalysis_2026-06-06/
├── README.md                                       [this file - entry point]
├── 00_understanding_checklist.md                   [Teaching doc: Problem/Solution/Context per stage]
├── 01_session_log.md                               [Session 1+2 timeline]
├── scripts/                                        [10 Python scripts, all read-only on BCCC]
│   ├── ws_j_statistical_eda.py                     [WS-J: EDA]
│   ├── ws_n_dropped_review.py                      [WS-N: dropped/review-pending breakdown]
│   ├── ws_i_stage1_sample_and_harness.py           [WS-I Stage 1: sample + slither harness + solc picker]
│   ├── ws_i_stage2_run_and_agreement.py            [WS-I Stage 2: full 808-contract run + agreement]
│   ├── ws_i_stage2_resume.py                       [WS-I Stage 2: resume script for empty rows]
│   ├── ws_i_stage2_rerun.py                        [WS-I Stage 2: full re-run with fixed parser]
│   ├── ws_i_stage2_final_resume.py                 [WS-I Stage 2: only-fill-empty final resume]
│   ├── ws_i_stage2_final_report.py                 [WS-I Stage 2: final report generator]
│   ├── ws_i_stage2_build_inspections.py            [WS-I Stage 2: build 30-contract inspection input]
│   └── _test_slither_5.py                          [WS-I Stage 1: 5-contract harness test]
├── outputs/                                        [All numeric / CSV outputs]
│   ├── ws_j_numeric_feature_summary.csv
│   ├── ws_j_feature_missingness.csv
│   ├── ws_j_cooccurrence_bccc.csv
│   ├── ws_j_cooccurrence_bccc_pct.csv
│   ├── ws_n_dropped_breakdown.csv
│   ├── ws_n_review_pending_breakdown.csv
│   ├── ws_n_review_pending_samples/                [Stratified samples of 766 review-pending]
│   ├── ws_n_dropped_samples/                       [Stratified samples of 1,122 dropped]
│   ├── ws_i_sample_818.csv                         [808 contracts stratified (6 buckets)]
│   ├── ws_i_slither_results.csv                    [808 rows × 10 cols, 33,049 findings]
│   ├── ws_i_agreement_report.md                    [Per-class agreement metrics + interpretation]
│   ├── ws_i_worst_30_for_review.csv                [30 worst-disagreement contracts (machine-readable)]
│   └── ws_i_harness_test.json                      [5-contract harness test result]
├── reports/                                        [Markdown reports per workstream]
│   ├── ws_j_eda_report.md
│   ├── ws_s_class_mapping.md
│   └── ws_n_dropped_review_report.md
└── labels/                                         [Manual review material]
    └── ws_i_inspections_input.md                   [30+2 contracts, 8,014 lines, ready for review]
```

---

## 5. How to Use This Folder

1. **Read this README** for orientation.
2. **Read the understanding checklist** ([`00_understanding_checklist.md`](00_understanding_checklist.md)) — it's a *teaching doc* with Problem/Solution/Context per stage.
3. **Read the session log** ([`01_session_log.md`](01_session_log.md)) for what was actually done and when.
4. **Read the headline results** above, then dive into per-workstream reports.
5. **Manual review** of 30 contracts: open [`labels/ws_i_inspections_input.md`](labels/ws_i_inspections_input.md), fill in Decision checkboxes (KEEP / MODIFY / REVIEW-NEEDED / FALSE-POSITIVE-CONTRACT), save as `ws_i_disagreement_inspections.md`.

---

## 6. Key Decisions (D-P3-* and D-I-*)

See [`../README.md`](../README.md) §"Key Decisions" for the full table. Highlights:

- **D-P3-10** (biggest, Session 2): Aderyn 0.6.8 (Cyfrin, Rust) replaces mythril for Session 3 3-way consensus on 5,000 contracts
- **D-I-1 through D-I-10** (Session 2): slither harness design + Stage 2 fixes (findings parser, solc picker, agreement metric, folder scan, etc.)
- **D-I-6**: Pass `solc=path` to `Slither()`, NOT `solc-select use` (global switch breaks parallelism)
- **D-I-7**: Slither 0.11+ findings parser must iterate `for det_findings in findings: for f in det_findings:` (list-of-lists structure)

---

## 7. What's Pending

| Item | Owner | Est. |
|---|---|---|
| Manual review of 30 contracts in `labels/ws_i_inspections_input.md` | User | 1-2 hours |
| Stage 3 (WS-K-K1): 31 source-code regex features on 67,311 contracts | Session 3 | 2-3 hours |
| Stage 4: Synthesis — `contracts_clean_v12.csv` + CHANGELOG §46 + MEMORY update | Session 5 | 1 hour |
| Session 3 (WS-O): Aderyn 3-way consensus on 5,000 contracts | Session 3 | 4-6 hours |
| Session 4-5 (WS-L): AutoML 5-model × 50 Optuna trials × 5 folds | Session 4-5 | 8-12 hours |

---

## 8. Related

- [`../README.md`](../README.md) — **root table of contents** for the whole BCCC deep dive
- [`../01_exploration_inventory.md`](../01_exploration_inventory.md) — Phase 1 (exploration)
- [`../02_validation_deep_dive_plan.md`](../02_validation_deep_dive_plan.md) — Phase 2 plan
- [`../03_phase3_plan.md`](../03_phase3_plan.md) — Phase 3 plan
- [`../04_phase4_plan.md`](../04_phase4_plan.md) — **Phase 4 plan (NEW 2026-06-07)** — applies D-I-11 + per-folder validation + AutoML
- [`../Phase2_Validation_2026-06-06/README.md`](../Phase2_Validation_2026-06-06/README.md) — Phase 2 entry point
- [`../Phase2_Validation_2026-06-06/outputs/contracts_clean.csv`](../Phase2_Validation_2026-06-06/outputs/contracts_clean.csv) — Phase 2 deliverable (67,311 × 24, SENTINEL v1.0)
- `BCCC-SCsVul-2024/` — source dataset (read-only)
- `docs/ml/adr/INDEX.md` — ADR-0005 (BCCC dataset choice)
- `docs/CHANGELOG.md` — needs §47-48 entries after Phase 4 (was §46 for Phase 3; superseded)
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` — SENTINEL project memory

---

**Last updated:** 2026-06-07 (Session 3 manual review complete; D-I-11 formalized; Phase 4 plan written)
