# Phase 4 Session 1 — Log

**Date:** 2026-06-07
**Focus:** Stage 0 (1-2h estimate, all 6 sub-steps completed)
**Status:** ✅ Complete — ready for Stage 1

---

## Timeline

| Time | Action | Output |
|---|---|---|
| 0:00 | Plan + folder created; 4 files written (README, checklist, plan copy) | Phase 4 dir structure |
| 0:05 | Stage 0.1 — D-I-11 applied | `ws_p4_s01_d11_applied.csv` (725 NV dropped, 41 RP remain) |
| 0:10 | Stage 0.2 — Oraclize dedup | 136 clusters, 548 dups flagged |
| 0:15 | Stage 0.3 — Exclude 33 reviewed | 66,730 eligible for sampling |
| 0:20 | Stage 0.4 — Surface finding: all 41 RP are IntegerUO+NV | D-P4-7 question raised |
| 0:25 | D-P4-7: user chose D-I-12 | D-I-12 decision doc written |
| 0:30 | Stage 0.1b — D-I-12 applied | `ws_p4_s01b_d12_applied.csv` (41 NV dropped, RP=0) |
| 0:35 | Re-ran s02 + s03 with v1.1+12 input | final sampling frame |
| 0:40 | Stage 0.5 — 31 regex features | `ws_p4_s05_regex_features.csv` (67,311 × 31) |
| 0:50 | Stage 0.6 — 3 hand-crafted features | `ws_p4_s06_handcrafted_features.csv` (67,311 × 3) |
| 1:00 | Updated README, checklist, deliverables tracker | All `[x]` marked |

---

## Decisions made in Session 1

| ID | Decision | Outcome |
|---|---|---|
| **D-P4-1** | Apply D-I-11 narrowly (review_pending only) | Default kept — no Stage 1.5 check needed yet |
| **D-P4-2** | 15% per primary_class sampling for Stage 1 | **TBD at Stage 1 start (next session)** |
| **D-P4-3** | 50 mythmil contracts for Stage 4 | **TBD at Stage 4 start (Session 2)** |
| **D-P4-4** | 50 Optuna trials × 5 folds for Stage 6 | **TBD at Stage 6 start (Session 3)** |
| **D-P4-5** | Include LogReg in AutoML | **TBD at Stage 6 start (Session 3)** |
| **D-P4-6** | Apply 3 hand-crafted features to v1.3 | **TBD at Stage 6 start (Session 3)** |
| **D-P4-7** | What to do with 41 review_pending | ✅ **Add D-I-12 rule** (user chose) |

---

## Numbers from Session 1

| Metric | Value |
|---|---:|
| v1.0 contracts | 67,311 |
| D-I-11 NV labels dropped | 725 |
| D-I-12 NV labels dropped | 41 |
| **Total NV labels dropped (D-I-11 + D-I-12)** | **766** |
| review_pending before | 766 |
| review_pending after | **0** |
| Oraclize clusters identified | 136 (548 total members) |
| Reviewed contracts excluded | 33 |
| **Eligible for Stage 1 sampling** | **66,730** |
| Regex features computed | 31 (67,311 × 31) |
| Hand-crafted features computed | 3 (67,311 × 3) |
| Stage 1 expected sample size (15%) | ~10,000 unique contracts |
| h01_nv_but_has_reentrancy_call fires | 15,144 contracts (all in NV=1) |
| h02_nv_but_has_external_call fires | 17,249 contracts (all in NV=1) |
| h03_unsafe_arith_no_safemath fires | 63,035 contracts (~94% of all) |

---

## Outputs produced in Session 1

| File | Description |
|---|---|
| `outputs/ws_p4_s01_d11_applied.csv` | v1.1 dataset (D-I-11 applied) |
| `outputs/ws_p4_s01_d11_report.md` | D-I-11 application report |
| `outputs/ws_p4_s01b_d12_applied.csv` | v1.1+12 dataset (D-I-11 + D-I-12 applied) — **final v1.1+12** |
| `outputs/ws_p4_s02_dedup_clusters.csv` | 136 Oraclize-style clusters |
| `outputs/ws_p4_s02_sampling_frame.csv` | v1.1+12 + is_oraclize_dup flag |
| `outputs/ws_p4_s03_exclude_reviewed.csv` | v1.1+12 + reviewed_in_phase3 flag — **final sampling frame** |
| `outputs/ws_p4_s05_regex_features.csv` | 31 regex features for 67,311 contracts |
| `outputs/ws_p4_s06_handcrafted_features.csv` | 3 hand-crafted features |
| `decisions/D-I-12_drop_nv_with_integeruo.md` | D-I-12 decision doc |
| `scripts/ws_p4_s0[1,1b,2,3,5,6]_*.py` | 6 stage-0 scripts (all runnable) |

---

## Verification done

- **D-I-11 spot-check:** 10 random corrected contracts, all in review_pending, all triggered by ≥1 of the 6 classes ✅
- **D-I-12 spot-check:** all 41 corrections are IntegerUO+NV with n_pos=2, primary_class=IntegerUO ✅
- **Oraclize dedup sanity:** 122 all-zero regex features (0.18%) and 0 all-one — distribution looks healthy ✅
- **No source missing:** 0 missing sources across all 67,311 contracts in regex + handcrafted feature scripts ✅
- **Sampling frame composition:** 66,730 eligible = 67,311 − 33 reviewed − 548 dups = 66,730 ✅

---

## Next session: Session 2 (Stage 1)

**Estimated time:** 9-12h (slither ~5h, aderyn ~2.5h, agreement analysis ~1h, parallel = ~6h wall clock)

**To do at session start:**
1. Decide D-P4-2: 10% or 15% sampling (default 15%)
2. Run `ws_p4_s1_sampling.py` (stratified 15% per primary_class)
3. Run `ws_p4_s1_slither.py` (subprocess wrapper, 30s timeout per contract, ~5h)
4. Run `ws_p4_s1_aderyn.py` (parallel with slither, ~2.5h)
5. Run `ws_p4_s1_agreement.py` (per-class F1, median across 8 classes)
6. **DECISION GATE:** median F1 ≥ 0.5? If yes, skip to Stage 4. If no, escalate to Stage 2 (30%).
7. Sub-step 1.5: D-P4-1 generalization check (apply D-I-11 broadly?)

**Expected sample size:** ~10,000 contracts

---

**Last updated:** 2026-06-07 (end of Session 1)
