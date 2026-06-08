# Phase 3 Session Log — BCCC Deep Dive, Session 2 (2026-06-06)

**Date:** 2026-06-06 (Session 2)
**Focus:** WS-I (Slither label validation) Stage 1
**Status:** Stage 1 complete; Stage 2 (full 808-contract run) pending

---

## Timeline

### Stage 0 (preparation)
- Updated Phase 3 plan (`03_phase3_plan.md`) with Aderyn (D-P3-10), confirming slither + Aderyn as 2-tool cross-validation
- Confirmed D-P3-1 (mythril Docker) retained for ad-hoc use; D-P3-9 (mythril excluded from batch) still active
- Created `Phase3_DeepAnalysis_2026-06-06/00_understanding_checklist.md` — running teaching doc

### Stage 1: Sample + Slither Harness
- **Script:** `scripts/ws_i_stage1_sample_and_harness.py`
- **Sample output:** `outputs/ws_i_sample_818.csv` (808 contracts, 6 buckets, with placeholder columns for slither results)
- **Harness test output:** `outputs/ws_i_harness_test.json` (5 contracts, 3 OK, 2 EXCEPTION)

**Sub-problems fixed:**
1. **Path mismatch** — Phase 2 stored "Source Codes" (with space), actual dir is "SourceCodes" (no space). `fix_path()` does `.replace("Source Codes", "SourceCodes")` for all 67,311 paths. Zero missing files after fix.
2. **Slither 0.11+ detector auto-registration** — Newer slither API requires explicit `slither.register_detector(cls)` for each detector class. The CLI does this internally. Fix: driver script imports `slither.detectors.all_detectors` and registers all 101 detector classes. Confirmed 101 detectors registered per run.
3. **Solc version selection** — Implemented `pick_solc_version(pragma)` that handles `^X.Y.Z`, `>=X.Y <Z.W`, exact `X.Y.Z`, NaN. Also verifies each candidate is selectable in solc-select (0.8.35 is on disk but broken in registry). Default = 0.5.17 for missing pragmas (most common BCCC version).
4. **NaN pragma handling** — `pragma="nan"` (from `pd.read_csv` coercion) now correctly falls back to default. Empty/`False`/None also handled.

**Test results (5 contracts):**
- 00333aa3... (review_pending, n_pos=3, ^0.4.11) → OK, 0 hits, 101 detectors, 1.0s
- 007ccde4... (review_pending, n_pos=3, ^0.4.19) → OK, 0 hits, 101 detectors, 1.1s
- 59fc8354... (multi_positive, n_pos=2, ^0.6.12) → EXCEPTION (real compile fail)
- d416cec8... (multi_positive, n_pos=2, NaN pragma) → EXCEPTION (real compile fail)
- 147725c1... (maxing, n_pos=8, 0.4.24) → OK, 0 hits, 101 detectors, 0.7s

**Key empirical finding:** All 3 OK contracts had BCCC labels for Reentrancy/CallToUnknown/IntegerUO/ExternalBug but **slither found 0 issues**. This is consistent with **BCCC label noise**: either the patterns are non-exploitable, or the contracts are too simple for detectors to fire. WS-I's job is to quantify this.

---

## Decisions Log

| ID | Decision | Choice | Rationale |
|---|---|---|---|
| D-P3-10 (new) | Replace mythril with Aderyn as 2nd static analyzer | Aderyn 0.6.8 (Cyfrin, Rust) | ~65x faster than mythril, 88 detectors, JSON/MD/SARIF output, no Docker needed |
| D-I-1 | Subprocess wrapper for slither | Yes | Timeout works reliably; isolates per-contract state |
| D-I-2 | Per-contract timeout | 30s | Bounded total runtime; matches BCCC 27% compile-fail rate |
| D-I-3 | Solc version picker | Highest selectable version satisfying pragma | 92% of BCCC is pre-0.6; default 0.5.17 for NaN |
| D-I-4 | Status enum | OK / COMPILE_ERROR / TIMEOUT / EXCEPTION / PATH_MISSING | Each maps to a specific handling decision (include vs exclude from agreement metrics) |
| D-I-5 | Path fix | `.replace("Source Codes", "SourceCodes")` | Phase 2 used pre-rename path; dir was renamed to SourceCodes |

---

## Outputs Created

| File | Size | Contents |
|---|---:|---|
| `00_understanding_checklist.md` | ~200 lines | Running teaching doc (Stage 0-1 done, Stage 2-4 pending) |
| `scripts/ws_i_stage1_sample_and_harness.py` | ~400 lines | Sample construction + slither harness + picker |
| `outputs/ws_i_sample_818.csv` | 808 rows | Stratified sample: 766 review_pending + 2 maxing + 40 multi_pos + 0 disagreement |
| `outputs/ws_i_harness_test.json` | 5 records | Test results from 5-contract harness validation |
| `03_phase3_plan.md` (updated) | ~1030 lines | Added Aderyn (D-P3-10), WS-O now 3-way, mythril reserved for ad-hoc |

---

## Next Steps (Stage 2)

1. Run slither on the full 808-contract sample
2. Compute per-class agreement metrics (BCCC label vs slither hit set)
3. Identify 30 worst-disagreement contracts for manual inspection
4. Update `ws_i_sample_818.csv` with the disagreement sample
5. Write 30 manual reviews of disagreements → `labels/ws_i_disagreement_inspections.md`
6. Write 2 maxing-contract inspections → `labels/ws_i_maxing_inspections.md`
7. Final decision matrix: do any labels change? → update `contracts_clean_v11.csv` if so

**Estimated time:** 3-5 hours (808 × 1-5s avg = 13-67 min for the slither run itself; rest is analysis + manual review).
