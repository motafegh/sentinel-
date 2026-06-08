# Phase 2 Session Log — BCCC-SCsVul-2024 Validation & Cleaning

**Date:** 2026-06-06
**Duration:** ~2.5 hours
**Status:** All 8 workstreams (WS-A through WS-H) complete

---

## Timeline

### 16:50 — Phase 2 start
- Folder created: `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/`
- 8 subdirs (scripts/, integrity/, labels/, complexity/, cross_corpus/, compile/, splits/, outputs/) + README
- Run 9 v11 second training still in flight (PID 3362523, awaiting ep1 val ~17:23)

### 16:55 — WS-A: Integrity & Dedup (19s)
- Computed sha256 of all 111,897 BCCC files
- Built `dedup_map.csv` (68,433 unique contracts, 38.8% duplicate rate)
- Wrote `integrity/manifest.md` confirming CSV md5 match
- Output: `integrity/sha256_all_files.tsv` (16 MB), `integrity/dedup_map.csv` (68,433 rows)

### 17:10 — WS-E: Per-Class Complexity Profile
- Initial run failed: `agg` variable shadowing `aggregate()` builtin
- Fix: renamed `agg` → `agg_v`; re-ran successfully
- Output: `complexity/per_contract_stats.csv` (68,433), `complexity/per_class_stats.csv` (12), `complexity/per_primary_class_stats.csv` (12), `complexity/complexity_report.md`
- Finding: NonVulnerable and WeakAccessMod contracts are simplest (193/275 mean LOC)

### 17:25 — D-F1 user decision
- User chose **A: Drop 2 BCCC classes** (WeakAccessMod + TransactionOrderDependence)
- Net: 5,480 contracts affected, 1,122 fully dropped (only had those classes)

### 17:30 — WS-D: Cross-Corpus Overlap
- Result: **0 byte-identical overlap** between BCCC and SmartBugs-curated
- Implication: SmartBugs is a clean OOD test set per ADR-0005 (no leakage)

### 17:35 — WS-B: Label Validation (long-format discovery)
- Major finding: CSV is **LONG format** (111,897 rows = 68,433 contracts × avg 1.635 classes)
- Each row has exactly 1 positive class
- Same ID appears multiple times with different single classes → true multi-label
- 100% MATCH between folder membership and CSV positive classes (dataset internally consistent)
- 766 NV+vuln contradictions at contract level (not 766 per-row as Phase 1 miscounted)

### 17:50 — WS-F: Class Reconciliation
- Applied D-F1: stripped Class05/Class07 from kept contracts, dropped 1,122 that had only those classes
- Applied D-B2: flagged 766 NV+vuln contradictions as `review_pending=1`
- Output: 67,311 surviving contracts, `labels/contracts_filtered.csv`, `labels/dropped_contracts.csv`, `labels/review_pending_ids.csv`, `labels/class_reconciliation_decision.md`

### 17:55 — D-B2 user decision
- User chose **D: Manual review 766** (held out from initial training)
- Implementation: review_pending=1 flag, excluded from splits

### 18:00 — WS-C: Compilation Probing
- Initial run failed: solc-select not in subprocess PATH
- Fix: hardcoded `ml/.venv/bin/solc-select` and `ml/.venv/bin/solc` paths
- 100 stratified contracts across 5 solc versions
- Result: **73% compilation success**, 17 PRAGMA, 7 SYNTAX, 1 IMPORT, 2 OTHER
- 0.4.24 best: 50/65 success (77%)
- Bytecode median 1.8 KB, max 12.5 KB

### 18:20 — WS-G: Stratified Split Design
- Iterative-stratification not installable (pip hung); used sklearn train_test_split with derived `(has_vuln, primary_vuln_class)` key
- Result: 46,581 train / 9,982 val / 9,982 test / 766 review_pending
- Distribution: ~70/15/15 across all 10 classes

### 19:00 — WS-H: Final Cleaned Dataset
- Initial run failed: complexity column names didn't match (was `loc_total`, not `loc`)
- Fix: renamed columns; merged complexity stats
- Output: `outputs/contracts_clean.csv` (17.3 MB, 67,311 × 24), `outputs/contracts_clean.parquet` (9.2 MB), `outputs/split_assignments.csv` (4.8 MB), `outputs/metadata.json`, `outputs/README.md`

### 19:10 — Phase 2 complete
- 100 MB total in Phase 2 directory
- 8 reproducible scripts
- 5 deliverable output files in `outputs/`

---

## Decisions Log

| Decision | Choice | Result |
|---|---|---|
| D-F1 | Drop 2 BCCC classes (WeakAccessMod + TransactionOrderDependence) | 1,122 contracts dropped; 67,311 kept |
| D-B2 | Manual review 766 NV+vuln contradictions | 766 flagged `review_pending=1`; excluded from splits |
| D-D (auto) | No overlap with SmartBugs-curated → clean OOD | Use SmartBugs as OOD test set per ADR-0005 |

## File Hashes

| File | sha256 |
|---|---|
| contracts_clean.csv | `53b7b884c3ae38446bd3f1f0460c916d01e5a2b5ef96ee972eed7d8628f59e7a` |
| contracts_clean.parquet | `a60b43087d30f855c19864263c5d59978e2259920c40f8c9389d818f36630af6` |

## Next Steps

1. Update MEMORY.md and CHANGELOG.md with Phase 2 findings
2. (Optional) Run 500-contract compilation probe for tighter error rate estimate
3. (Optional) Resolve review-pending 766 via manual inspection
4. (Future) Integrate contracts_clean.csv into SENTINEL training pipeline (replaces cached_dataset_v9.pkl)
