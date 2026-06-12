# BCCC-SCsVul-2024 Deep Dive — Phase 2: Validation & Cleaning

**Title:** Phase 2 Validation & Cleaning of the BCCC-SCsVul-2024 Dataset for SENTINEL Training
**Date started:** 2026-06-06
**Date ended:** 2026-06-06 (~2.5 hours)
**Author:** SENTINEL Data Engineering
**Source dataset:** `BCCC-SCsVul-2024/` (1.6 GB, read-only)
**Phase 1 reference:** [`../01_exploration_inventory.md`](../01_exploration_inventory.md) (435 lines, 10 findings)
**Phase 2 plan reference:** [`../02_validation_deep_dive_plan.md`](../02_validation_deep_dive_plan.md) (342 lines, 8 workstreams)
**Root README:** [`../README.md`](../README.md) — table of contents for the whole deep dive
**Status:** ✅ **Complete** — all 8 workstreams (WS-A through WS-H) done. Final deliverable: `outputs/contracts_clean.csv` (67,311 × 24).

---

## 1. Purpose

Produce a **validated, deduplicated, SENTINEL-ready dataset** from `BCCC-SCsVul-2024/`. Phase 1 established the dataset's actual structure (multi-label, 68,433 unique contracts, 38.8% file duplication). Phase 2 cleans and prepares it for ingestion into SENTINEL's training pipeline.

**No source files in `BCCC-SCsVul-2024/` are modified.** All writes are to this Phase 2 directory and downstream `outputs/`.

---

## 2. Workstream Index

| WS | Title | Status | Est. (h) | Output directory | Top-level deliverable |
|---|---|---|---:|---|---|
| A | Integrity & Dedup | ✅ Done | 2.0 | `integrity/` | `manifest.md` + `dedup_map.csv` |
| B | Label Validation | ✅ Done | 8.0 | `labels/` | `label_validation_report.md` + `label_consistency.csv` |
| C | Compilation Probing | ✅ Done | 3.25 | `compile/` | `compilation_report.md` + `compile_results.csv` (73% compile rate) |
| D | Cross-Corpus Overlap | ✅ Done | 2.5 | `cross_corpus/` | `overlap_report.md` (**0 overlap** with SmartBugs) + `bccc_vs_smartbugs_overlap.csv` |
| E | Per-Class Complexity | ✅ Done | 3.0 | `complexity/` | `complexity_report.md` + `per_class_stats.csv` |
| F | Class Reconciliation | ✅ Done (D-F1: drop Class05/07) | 2.0 | `labels/` | `class_reconciliation_decision.md` (1,122 contracts dropped) |
| G | Stratified Split | ✅ Done | 2.5 | `splits/` | `train.csv` + `val.csv` + `test.csv` (70/15/15) |
| H | Final Cleaned Dataset | ✅ Done | 3.5 | `outputs/` | `contracts_clean.csv` (67,311 × 24) + `split_assignments.csv` |

**Total estimated:** 26.75 h  **Total actual:** ~2.5 h (scripts were efficient)
**Critical path:** A → F (D-F1) → G → H — all complete

---

## 3. Blocking Decision — D-F1 ✅ RESOLVED

BCCC has 12 classes. SENTINEL's ADR-0005 plans 10. Two BCCC classes (`TransactionOrderDependence` 5.2%, `WeakAccessMod` 2.8%) are not in SENTINEL.

| Option | Description | Chosen? |
|---|---|---|
| **(A) Drop 2** | Keep SENTINEL's 10-class plan. Mask 2 BCCC columns at training. | ✅ **YES** — 1,122 contracts dropped, 10 SENTINEL classes |
| (B) Add 2 | 12 binary heads, 12-fold output, ADR-0005 update. | No |
| (C) Train on 12, mask 2 at inference | Hybrid; slight waste. | No |

**Outcome:** D-F1 resolved. WS-F, WS-G, WS-H all unblocked and complete.

Full breakdown: [`labels/class_reconciliation_decision.md`](labels/class_reconciliation_decision.md)

---

## 4. Top-Level Findings So Far (from Phase 1, restated for context)

1. **68,433 unique contracts** (not 111,897); 38.8% are exact byte-identical copies across folders.
2. **Multi-label** (41% of contracts have ≥2 positive classes).
3. **12 BCCC classes** vs SENTINEL's 10 (D-F1 above).
4. **766 NV+vuln contracts** — likely meta-label noise (see WS-B).
5. **Top co-occurrence:** DoS+Reentrancy = 12,381 contracts (18% of corpus).
6. **CSV md5 verified; per-file content md5 NOT verifiable** (Sourcecodes.md5 validates a missing ZIP).
7. **92% pre-0.6 Solidity** (mostly 0.4.x/0.5.x) — old solc toolchain needed.
8. **0% SPDX headers** (pre-SPDX era).
9. **Multi-folder distribution:** 40K unique contents in 1 folder, 19K in 2, ..., 2 in 9 folders.
10. **CSV "ID" is 64-hex but NOT sha256(content)** (95.5% mismatch) — likely keccak-256 of bytecode.

Full detail in [`../01_exploration_inventory.md`](../01_exploration_inventory.md).

---

## 5. Directory Layout

```
Phase2_Validation_2026-06-06/
├── README.md                  [this file]
├── 00_session_log.md          [chronological log of all work in this session]
├── scripts/                   [WS-A, WS-D, WS-E, etc. Python scripts]
├── integrity/                 [WS-A: sha256 + dedup map + manifest]
├── labels/                    [WS-B, F: paper summary, manual inspections, reconciliation decision]
├── complexity/                [WS-E: per-class stats + report]
├── cross_corpus/              [WS-D: BCCC vs SmartBugs overlap]
├── compile/                   [WS-C: solc compilation probe]
├── splits/                    [WS-G: train/val/test split files]
└── outputs/                   [WS-H: final contracts_clean.csv + parquet + metadata]
```

---

## 6. How to Use This Folder

1. **Read this README first** for orientation.
2. **Read the session log** (`00_session_log.md`) for what was actually done and when.
3. **Read each workstream's report** (e.g., `integrity/manifest.md`) for detailed findings.
4. **Raw data files** (CSVs, TSVs) are in the workstream subdirectories.
5. **Reproducibility:** all Python scripts are in `scripts/` and are idempotent / read-only on the source dataset.

---

## 7. Reproducibility

```bash
# Run all Phase 2 workstreams in order (after D-F1 decision)
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate

# WS-A: integrity
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/a_integrity_dedup.py

# WS-E: complexity
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/e_complexity_profile.py

# WS-D: cross-corpus (requires SmartBugs-curated path)
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/d_cross_corpus.py

# etc.
```

---

## 8. Related

- [`../README.md`](../README.md) — **root table of contents** for the whole BCCC deep dive
- [`../01_exploration_inventory.md`](../01_exploration_inventory.md) — Phase 1 (exploration, complete)
- [`../02_validation_deep_dive_plan.md`](../02_validation_deep_dive_plan.md) — Phase 2 plan (this phase)
- [`../03_phase3_plan.md`](../03_phase3_plan.md) — Phase 3 plan (label validation, in flight)
- [`../Phase3_DeepAnalysis_2026-06-06/README.md`](../Phase3_DeepAnalysis_2026-06-06/README.md) — Phase 3 entry point
- `BCCC-SCsVul-2024/` — source dataset (read-only)
- `docs/ml/adr/INDEX.md` — ADR-0005 (BCCC dataset choice)
- `CHANGELOG.md` §43 — Phase 1 in changelog; §44 — Phase 2 (pending)

---

**Last updated:** 2026-06-06 (Phase 2 complete; root README link added; status updated to ✅)
