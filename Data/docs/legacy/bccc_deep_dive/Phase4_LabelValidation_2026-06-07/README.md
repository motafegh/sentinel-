# BCCC-SCsVul-2024 Deep Dive — Phase 4: Label Trustworthiness Validation

**Title:** Phase 4 — Per-Folder Label Trustworthiness Validation, Mass NonVulnerable Correction, and AutoML Baselines
**Date started:** 2026-06-07
**Status:** 🟡 **Stage 0 ✅ complete (all 6 sub-steps), ready for Stage 1**

---

## Quick Links

| If you are… | Open this |
|---|---|
| Starting Session 1 right now | [`00_actionable_checklist.md`](00_actionable_checklist.md) — the working todo list |
| Wanting the design rationale | [`04_phase4_plan.md`](04_phase4_plan.md) — the full plan with stages + decisions |
| Needing context from prior phases | [`../README.md`](../README.md) — root table of contents |
| Needing D-I-11 justification | [`../Phase3_DeepAnalysis_2026-06-06/decisions/D-I-11_drop_nv_with_vuln.md`](../Phase3_DeepAnalysis_2026-06-06/decisions/D-I-11_drop_nv_with_vuln.md) |
| Needing your 32-contract manual review | [`../Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md`](../Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md) |

---

## The Plan in 30 Seconds

**Goal:** Validate BCCC labels per-folder with stratified sampling (10,000 unique contracts), apply the D-I-11 mass correction, scale up mythril to a 50-contract tiebreaker, run AutoML, and produce `contracts_clean_v13.csv`.

**7 stages, 4 sessions, 40-70h total:**

1. **Stage 0** (Session 1, 1-2h): Apply D-I-11 + Oraclize dedup + exclude 32 reviewed + spot-review 61 + 31 regex features + 3 hand-crafted features → `contracts_clean_v11.csv` + sampling frame + 34 features
2. **Stage 1** (Session 1, ~9h): Slither + aderyn on 15% per primary_class (~10,000 unique contracts) → median F1 across 8 classes
3. **Stage 2** (Session 2, ~9h, only if Stage 1 median F1 < 0.5): Escalate to 30% sampling
4. **Stage 3** (Session 3, ~12h, only if Stage 2 median F1 < 0.5): Escalate to 50% sampling
5. **Stage 4** (Session 2, 2.5h, downstream of Stage 1): Mythmil on 50 hardest disagreements
6. **Stage 5** (Sessions 2-3, 2-4 days): Per-folder manual investigation (3-5 contracts × 8 folders)
7. **Stage 6** (Session 3, 10-12h): AutoML on v1.3 (10 binary × 5 models × 50 trials × 5 folds)
8. **Stage 7** (Session 4, 2-3h): Synthesis → `contracts_clean_v13.csv` + CHANGELOG §47-49 + MEMORY

**Decisions to make at session start:** 7 (D-P4-1 through D-P4-7) — see [`00_actionable_checklist.md`](00_actionable_checklist.md) §"Decisions pending"

---

## Directory Layout

```
Phase4_LabelValidation_2026-06-07/
├── README.md                                    [this file]
├── 00_actionable_checklist.md                   [Working todo list — start here]
├── 04_phase4_plan.md                            [Design doc with all 7 stages]
├── 01_session_log.md                            [Session-by-session timeline, to be created]
├── scripts/                                     [15 Python scripts, to be created]
│   ├── ws_p4_s01_apply_d11.py                   [Stage 0.1]
│   ├── ws_p4_s02_oraclize_dedup.py              [Stage 0.2]
│   ├── ws_p4_s03_exclude_reviewed.py            [Stage 0.3]
│   ├── ws_p4_s04_resolve_remaining_review.py    [Stage 0.4]
│   ├── ws_p4_s05_regex_features.py              [Stage 0.5]
│   ├── ws_p4_s06_handcrafted_features.py        [Stage 0.6]
│   ├── ws_p4_s1_sampling.py                     [Stage 1]
│   ├── ws_p4_s1_slither.py                      [Stage 1]
│   ├── ws_p4_s1_aderyn.py                       [Stage 1]
│   ├── ws_p4_s1_agreement.py                    [Stage 1]
│   ├── ws_p4_s2_escalate_30pct.py               [Stage 2]
│   ├── ws_p4_s3_escalate_50pct.py               [Stage 3]
│   ├── ws_p4_s4_mythril_tiebreaker.py           [Stage 4]
│   ├── ws_p4_s5_per_folder_manual.py            [Stage 5]
│   ├── ws_p4_s6_automl_v13.py                   [Stage 6]
│   └── ws_p4_s7_synthesis_v13.py                [Stage 7]
├── outputs/                                     [Numeric/CSV outputs, to be created]
│   ├── ws_p4_s01_d11_applied.csv                [v1.1 dataset]
│   ├── ws_p4_s01_d11_report.md                  [D-I-11 impact report]
│   ├── ws_p4_s02_dedup_clusters.csv             [Oraclize cluster identification]
│   ├── ws_p4_s02_sampling_frame.csv             [v1.1 − Oraclize dups − 32 reviewed]
│   ├── ws_p4_s04_remaining_review_pending.md    [~61 review_pending decision]
│   ├── ws_p4_s05_regex_features.csv             [67,311 × 31]
│   ├── ws_p4_s06_handcrafted_features.csv       [67,311 × 3]
│   ├── ws_p4_s1_sample_15pct.csv                [~10,000 contracts]
│   ├── ws_p4_s1_slither_results.csv
│   ├── ws_p4_s1_aderyn_results.csv
│   ├── ws_p4_s1_per_folder_agreement.md
│   ├── ws_p4_s2_contracts_clean_v12.csv         [v1.2 dataset, if Stage 1-2 corrections]
│   ├── ws_p4_s2_per_folder_agreement.md
│   ├── ws_p4_s3_per_folder_agreement.md
│   ├── ws_p4_s4_mythril_3way.md
│   ├── ws_p4_s6_automl_report.md
│   ├── ws_p4_s6_shap_top10.png
│   ├── ws_p4_s7_contracts_clean_v13.csv         [v1.3 FINAL DELIVERABLE]
│   └── ws_p4_s7_v13_split_assignments.csv
├── reports/                                     [8 per-folder manual investigations]
│   ├── ws_p4_s5_call_to_unknown_investigation.md
│   ├── ws_p4_s5_reentrancy_investigation.md
│   ├── ws_p4_s5_integer_uo_investigation.md
│   ├── ws_p4_s5_denial_of_service_investigation.md
│   ├── ws_p4_s5_gas_exception_investigation.md
│   ├── ws_p4_s5_mishandled_exception_investigation.md
│   ├── ws_p4_s5_external_bug_investigation.md
│   └── ws_p4_s5_timestamp_investigation.md
├── labels/                                      [Empty in Phase 4 — manual review happened in Phase 3]
├── decisions/                                   [Empty in Phase 4 — D-I-11 is in Phase 3 dir]
└── checkpoints/                                 [Intermediate state saves for crash recovery]
```

---

## Status by Stage

| Stage | Description | Status |
|---|---|---|
| **0.1** | Apply D-I-11 → v1.1 (725 NV dropped) | ✅ Done |
| **0.1b** | Apply D-I-12 → v1.1+12 (41 NV dropped, review_pending=0) | ✅ Done |
| **0.2** | Oraclize dedup (136 clusters, 548 dups) | ✅ Done |
| **0.3** | Exclude 33 reviewed (66,730 eligible) | ✅ Done |
| **0.4** | Resolve 41 review_pending via D-I-12 | ✅ Done |
| **0.5** | 31 regex features (67,311 × 31) | ✅ Done |
| **0.6** | 3 hand-crafted features (67,311 × 3) | ✅ Done |
| **1** | Per-folder stratified validation (15% × 10k) | ⏳ Pending (next) |
| **2** | Escalate to 30% (if Stage 1 median F1 < 0.5) | ⏳ Pending or skipped |
| **3** | Escalate to 50% (if Stage 2 median F1 < 0.5) | ⏳ Pending or skipped |
| **4** | Mythmil tiebreaker on 50 hardest | ⏳ Pending |
| **5** | Per-folder manual investigation (8 folders) | ⏳ Pending |
| **6** | AutoML on v1.3 (10 binary × 5 models × 50×5) | ⏳ Pending |
| **7** | Synthesis → v1.3 + CHANGELOG §47-49 | ⏳ Pending |

---

## Key Files to Read Before Starting

1. **[`00_actionable_checklist.md`](00_actionable_checklist.md)** (this dir) — the working todo list with checkboxes
2. **[`04_phase4_plan.md`](04_phase4_plan.md)** (this dir) — the full plan, 517 lines
3. **[`../Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md`](../Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md)** (425 lines) — your manual review, source of D-I-11
4. **[`../Phase3_DeepAnalysis_2026-06-06/decisions/D-I-11_drop_nv_with_vuln.md`](../Phase3_DeepAnalysis_2026-06-06/decisions/D-I-11_drop_nv_with_vuln.md)** (115 lines) — D-I-11 formal writeup
5. **[`../Phase2_Validation_2026-06-06/outputs/contracts_clean.csv`](../Phase2_Validation_2026-06-06/outputs/contracts_clean.csv)** — v1.0 dataset, the input

---

## Related

- [`../README.md`](../README.md) — root table of contents for the whole BCCC deep dive
- [`../01_exploration_inventory.md`](../01_exploration_inventory.md) — Phase 1
- [`../02_validation_deep_dive_plan.md`](../02_validation_deep_dive_plan.md) — Phase 2 plan
- [`../03_phase3_plan.md`](../03_phase3_plan.md) — Phase 3 plan
- [`../04_phase4_plan.md`](../04_phase4_plan.md) — Phase 4 plan (canonical design doc)
- [`../Phase2_Validation_2026-06-06/README.md`](../Phase2_Validation_2026-06-06/README.md) — Phase 2 entry point
- [`../Phase3_DeepAnalysis_2026-06-06/README.md`](../Phase3_DeepAnalysis_2026-06-06/README.md) — Phase 3 entry point
- `BCCC-SCsVul-2024/` — source dataset (read-only)
- `docs/ml/adr/INDEX.md` — ADR-0005 (BCCC dataset choice)
- `docs/CHANGELOG.md` — needs §47-49 entries after Phase 4
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` — SENTINEL project memory

---

**Last updated:** 2026-06-07 (folder created, actionable checklist written, plan copied, ready to start Session 1)
