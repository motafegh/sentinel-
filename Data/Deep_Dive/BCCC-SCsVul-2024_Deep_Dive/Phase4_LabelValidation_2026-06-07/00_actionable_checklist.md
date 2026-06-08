# Phase 4 Actionable Checklist

**Use this file as your working session log.** Mark each item `[x]` when done. The `04_phase4_plan.md` is the design doc; this is the execution tracker.

**Sessions:**
- [Session 1 (10-12h, this is happening now)](#session-1--1012h)
- [Session 2 (12-24h)](#session-2--1224h)
- [Session 3 (16-30h)](#session-3--1630h)
- [Session 4 (2-3h)](#session-4--23h)

---

## Session 1 (10-12h)

### Stage 0.1: Apply D-I-11 (15 min) — FAST WIN ✅
- [x] 0.1.1 Read `Phase3_DeepAnalysis_2026-06-06/decisions/D-I-11_drop_nv_with_vuln.md` (115 lines)
- [x] 0.1.2 Write `scripts/ws_p4_s01_apply_d11.py` (rule: drop NV when co-occurs with any of 6 vuln classes)
- [x] 0.1.3 Apply to `contracts_clean.csv` → produce `outputs/ws_p4_s01_d11_applied.csv` (v1.1)
- [x] 0.1.4 Count: **725 contracts had NV dropped, 41 review_pending remain** (predicted ~705 / ~61)
- [x] 0.1.5 Spot-check 10 random corrected contracts (all in review_pending, all triggered by 1+ class) — confirmed
- [x] 0.1.6 Write `outputs/ws_p4_s01_d11_report.md` (numbers + sample changes)
- [x] 0.1.7 Mark `[x]` in checklist

### Stage 0.2: Oraclize cluster dedup (15 min) — FAST WIN ✅
- [x] 0.2.1 Compute `source_stripped_sha256` for every contract in v1.1 (67,311 contracts, 0 missing)
- [x] 0.2.2 Group by hash → identify clusters of ≥3 near-identical contracts
- [x] 0.2.3 **Found 136 clusters (predicted ~100-200), 548 total members**
- [x] 0.2.4 Mark non-representative duplicates with `is_oraclize_dup=1`
- [x] 0.2.5 Write `outputs/ws_p4_s02_dedup_clusters.csv` + `outputs/ws_p4_s02_sampling_frame.csv`
- [x] 0.2.6 Mark `[x]` in checklist
- **Top cluster: 36 contracts, smallest 92 clusters of size 3 — real Oraclize-style templates**

### Stage 0.3: Exclude 32 reviewed contracts (1 min) — FAST WIN ✅
- [x] 0.3.1 Load 32 IDs from `Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md`
- [x] 0.3.2 Add `reviewed_in_phase3=1` flag to those contracts in v1.1 (**33 unique, not 32** — worst_30 already includes 2 maxing)
- [x] 0.3.3 Exclude from `ws_p4_s02_sampling_frame.csv` → `ws_p4_s03_exclude_reviewed.csv` (66,730 eligible)
- [x] 0.3.4 Mark `[x]` in checklist

### Stage 0.4: Resolve ~41 remaining review_pending (30 min) — REQUIRES DECISION ✅
- [x] 0.4.1 Count: **41 review_pending remain** (better than predicted ~61)
- [x] 0.4.2 Surface finding: all 41 are `IntegerUO + NonVulnerable` only (n_pos=2, just those 2 classes)
- [x] 0.4.3 Check: all 41 have primary_class=IntegerUO, no other classes → clean uniform pattern
- [x] 0.4.4 D-P4-7 decision: **Add D-I-12: drop NV when co-occurring with IntegerUO**
- [x] 0.4.5 Write `decisions/D-I-12_drop_nv_with_integeruo.md` (decision doc) + run `ws_p4_s01b_apply_d12.py`
- [x] 0.4.6 Output: `ws_p4_s01b_d12_applied.csv` (v1.1+12, **review_pending=0**, 41 NV dropped)
- [x] 0.4.7 Re-ran s02 (dedup, default input → v1.1+12) + s03 (exclude 33, 66,730 eligible)
- [x] 0.4.8 Mark `[x]` in checklist

### Stage 0.5: Compute 31 regex features (30-60 min) — FAST WIN ✅
- [x] 0.5.1 Define 31 regex patterns (5 pragma, 2 timestamp, 1 tx.origin, 6 calls, 2 crypto, 4 control, 3 safety, 3 types, 3 funcs, 2 mods)
- [x] 0.5.2 Write `scripts/ws_p4_s05_regex_features.py` (one-pass scan, 1-2s per contract)
- [x] 0.5.3 Apply to all 67,311 contracts in v1.1
- [x] 0.5.4 Output: `outputs/ws_p4_s05_regex_features.csv` (67,311 × 31)
- [x] 0.5.5 Sanity: 122 all-zero (0.18%), 0 all-one, top features realistic
- [x] 0.5.6 Mark `[x]` in checklist

### Stage 0.6: Compute 3 hand-crafted features (10 min) — FAST WIN ✅
- [x] 0.6.1 `nv_but_has_reentrancy_call` = NV=1 AND has `.call{value:}` or `.transfer()` or `.send()` (15,144 fires)
- [x] 0.6.2 `nv_but_has_external_call` = NV=1 AND has any external call (17,249 fires)
- [x] 0.6.3 `unsafe_arith_no_safemath` = (arithmetic op exists) AND (not inside SafeMath) (63,035 fires, ~94% of contracts)
- [x] 0.6.4 Write `scripts/ws_p4_s06_handcrafted_features.py`
- [x] 0.6.5 Apply to 67,311 contracts → `outputs/ws_p4_s06_handcrafted_features.csv` (67,311 × 3)
- [x] 0.6.6 Sanity: h01/h02 perfectly correlated with NV=1 (by construction), h03 is the broad baseline
- [x] 0.6.7 Mark `[x]` in checklist

### Stage 1: Per-folder stratified validation (~9h) — REQUIRES DECISION
- [x] 1.0 **DECISION D-P4-2**: 15% sampling chosen (user: "few get all, many get 15%")
- [x] 1.1 Stratified sample: 15% per primary_class (100% for Reentrancy/UnusedReturn)
- [x] 1.2 Sample size: **10,693** unique contracts (8 classes × 15% + 2 classes × 100%)
- [x] 1.3 Write `scripts/ws_p4_s1_sampling.py`
- [x] 1.4 Output: `outputs/ws_p4_s1_sample.csv` (with `in_stage1_sample` flag)
- [x] 1.5 Write `scripts/ws_p4_s1_slither.py` (V2: version-grouped, pinned binary, --allow-paths)
- [x] 1.6 Output: `outputs/ws_p4_s1_slither_results.csv` (**DONE: 10,693/10,693, 67.2% OK**, 28.6 min)
- [x] 1.7 Write `scripts/ws_p4_s1_aderyn.py` (V2: per-contract parallel, 6 workers)
- [x] 1.8 Output: `outputs/ws_p4_s1_aderyn_results.csv` (**DONE: 10,693/10,693, 58% OK**, 9.9 min)
- [x] 1.9 Write `scripts/ws_p4_s1_agreement.py` (per-class F1, median F1 gate, corrected BCCC class mapping)
- [x] 1.10 Output: `outputs/ws_p4_s1_agreement_report.md` — **FAIL: median F1=0.131** (corrected mapping)
- [x] 1.11 Write `scripts/ws_p4_s1_3way_agreement.py` (3-way consensus)
- [x] 1.12 Output: `outputs/ws_p4_s1_3way_agreement_report.md` — **FAIL: median F1=0.000**
- [x] 1.13 **DECISION GATE**: **FAIL** (median F1=0.131 < 0.5, not IntegerUO sole outlier) → Escalation considered
- [x] 1.14 Manual review: 43 high-uncertainty contracts → 51% DROP, 12% KEEP
- [x] 1.15 Expanded review: 199 contracts (TP + FN) → **CRITICAL FINDING: Reentrancy 89% false positives**
- [x] 1.16 Full Reentrancy audit: 500/17,698 → **Only 10.6% true reentrancy, 89.4% false positives**
- [x] 1.17 Documented in `CRITICAL_FINDINGS.md` — for further investigation before proceeding
- [x] 1.18 **ESCALATION DECISION DEFERRED** — noise is structural, not sample-size limited
- [x] 1.19 Mark `[x]` in checklist

### Session 1 wrap-up
- [x] Update root README with Phase 4 Session 1 results
- [x] Update MEMORY with v1.1 + 41 review_pending (D-I-12 applied) + 33 reviewed + 548 dups
- [x] Update `01_session_log.md` with timeline + decisions
- [x] Write `outputs/ws_p4_s01_d11_report.md` ✅
- [x] Surface 41 review_pending → resolved via D-I-12
- [ ] Update `01_session_log.md` with Stage 1 progress (slither running, 484/10693)

---

## Session 2 (12-24h)

### Stage 4: Mythril tiebreaker (2.5h) — DOWNSTREAM of Stage 1
- [ ] 4.1 From Stage 1 results, rank contracts by `(slither_disagreement + aderyn_disagreement) / 2`
- [ ] 4.2 Pick top 50, ensuring 2-3 per folder (cover all 8 folders)
- [ ] 4.3 Write `scripts/ws_p4_s4_mythril_tiebreaker.py` (Docker `mythril/myth:0.24.8`, 3min/contract, 5min timeout)
- [ ] 4.4 Run mythril on 50 contracts (~2.5h)
- [ ] 4.5 Output: `outputs/ws_p4_s4_mythril_3way.md` (per-folder 3-way consensus table)
- [ ] 4.6 Mark `[x]` in checklist

### Stage 2: Escalation to 30% per unique contract (only if Stage 1 median F1 < 0.5)
- [ ] 2.1 Add another 15% to sample (total ~20,000 contracts)
- [ ] 2.2 Run slither + aderyn on new contracts (~9h)
- [ ] 2.3 Re-compute per-class F1, decision gate
- [ ] 2.4 Output: `outputs/ws_p4_s2_contracts_clean_v12.csv` (if applying per-folder corrections)
- [ ] 2.5 Output: `outputs/ws_p4_s2_per_folder_agreement.md`
- [ ] 2.6 Mark `[x]` in checklist (or `[skipped]` if Stage 1 gate passed)

### Stage 5 part 1: Per-folder manual investigation, 4 folders (~10-20h)
- [ ] 5.1 Pick 3-5 contracts per folder (1 high-agreement, 1 disagreement, 1 maxing, 1 typical, 1 Oraclize if relevant)
- [ ] 5.2 Read source for each, document actual vulnerability profile
- [ ] 5.3 Write `reports/ws_p4_s5_call_to_unknown_investigation.md`
- [ ] 5.4 Write `reports/ws_p4_s5_reentrancy_investigation.md`
- [ ] 5.5 Write `reports/ws_p4_s5_integer_uo_investigation.md`
- [ ] 5.6 Write `reports/ws_p4_s5_denial_of_service_investigation.md`
- [ ] 5.7 Note any new D-P4-* decisions needed
- [ ] 5.8 Mark `[x]` in checklist

### Session 2 wrap-up
- [ ] Update root README + MEMORY with Session 2 results
- [ ] Update `01_session_log.md` with timeline + decisions

---

## Session 3 (16-30h)

### Stage 3: Escalation to 50% per unique contract (only if Stage 2 median F1 < 0.5)
- [ ] 3.1 Add another 20% to sample (total ~33,000 contracts)
- [ ] 3.2 Run slither + aderyn on new contracts (~12h)
- [ ] 3.3 Re-compute per-class F1, final per-class documentation
- [ ] 3.4 Output: `outputs/ws_p4_s3_per_folder_agreement.md`
- [ ] 3.5 Mark `[x]` in checklist (or `[skipped]`)

### Stage 5 part 2: Per-folder manual investigation, 4 folders (~10-20h)
- [ ] 5.9 Write `reports/ws_p4_s5_gas_exception_investigation.md`
- [ ] 5.10 Write `reports/ws_p4_s5_mishandled_exception_investigation.md`
- [ ] 5.11 Write `reports/ws_p4_s5_external_bug_investigation.md`
- [ ] 5.12 Write `reports/ws_p4_s5_timestamp_investigation.md`
- [ ] 5.13 Mark `[x]` in checklist

### Stage 6: AutoML on v1.3 (10-12h, can run overnight)
- [ ] 6.1 **DECISION D-P4-4**: 50 trials × 5 folds (12,500 fits) or 25×5 (6,250 fits)?
- [ ] 6.2 Build v1.3: combine D-F1 + D-B2 + D-I-11 + per-folder corrections from Stage 5
- [ ] 6.3 Write `scripts/ws_p4_s6_automl_v13.py` (10 binary × 5 models × N trials × 5 folds)
- [ ] 6.4 Per-class imbalance strategy (SMOTE for n<1000, scale_pos_weight for 1k-10k, none for n≥10k)
- [ ] 6.5 Run Optuna search for all 50 (model × class) combinations
- [ ] 6.6 Output: `outputs/ws_p4_s6_automl_report.md` + `ml/calibration/automl_v13_*.json` × 5
- [ ] 6.7 Output: `outputs/ws_p4_s6_shap_top10.png` (top features per class)
- [ ] 6.8 **Comparison framing:** AutoML (34 features) vs SENTINEL (graph + CodeBERT)
- [ ] 6.9 Mark `[x]` in checklist

### Session 3 wrap-up
- [ ] Update root README + MEMORY with Session 3 results
- [ ] Update `01_session_log.md` with timeline + decisions

---

## Session 4 (2-3h)

### Stage 7: Synthesis
- [ ] 7.1 Build v1.3: v1.2 + Stage 5 manual + Stage 6 AutoML-informed threshold tuning
- [ ] 7.2 Write `scripts/ws_p4_s7_synthesis_v13.py`
- [ ] 7.3 Output: `outputs/ws_p4_s7_contracts_clean_v13.csv` (FINAL DELIVERABLE)
- [ ] 7.4 Output: `outputs/ws_p4_s7_v13_split_assignments.csv` (70/15/15 split)
- [ ] 7.5 Write `docs/CHANGELOG.md` §47 (v1.1 + D-I-11)
- [ ] 7.6 Write `docs/CHANGELOG.md` §48 (v1.2 + per-folder)
- [ ] 7.7 Write `docs/CHANGELOG.md` §49 (v1.3 + Stage 5/6)
- [ ] 7.8 Update `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` with Phase 4 final state
- [ ] 7.9 Update root README + Phase 4 README with v1.3 results
- [ ] 7.10 Mark `[x]` in checklist

---

## Decisions pending across all sessions

| ID | Decision | Default | Where decided |
|---|---|---|---|
| **D-P4-1** | Apply D-I-11 to review_pending only or all 67,311? | review_pending only | Stage 1.5 |
| **D-P4-2** | Stage 1 sampling: 10% or 15%? | 15% | Session 1 start |
| **D-P4-3** | Mythmil: 50 or 100 contracts? | 50 | Session 2 start |
| **D-P4-4** | AutoML: 50×5 or 25×5 fits? | 50×5 | Session 3 start |
| **D-P4-5** | Include LogReg? | Yes | Session 3 start |
| **D-P4-6** | Apply 3 hand-crafted features to v1.3? | Yes | Session 3 start |
| **D-P4-7** | What to do with ~61 remaining review_pending? | Spot-review 5-10 | Stage 0.4 |

---

## Cumulative deliverables tracker

| Deliverable | Path | Status |
|---|---|---|
| v1.1 dataset (D-I-11) | `outputs/ws_p4_s01_d11_applied.csv` | ✅ |
| v1.1+12 dataset (D-I-11 + D-I-12) | `outputs/ws_p4_s01b_d12_applied.csv` | ✅ |
| Oraclize dedup map | `outputs/ws_p4_s02_dedup_clusters.csv` | ✅ |
| Sampling frame | `outputs/ws_p4_s02_sampling_frame.csv` | ✅ |
| Excluded reviewed frame | `outputs/ws_p4_s03_exclude_reviewed.csv` | ✅ |
| 41 remaining review_pending decision | `decisions/D-I-12_drop_nv_with_integeruo.md` | ✅ |
| 31 regex features | `outputs/ws_p4_s05_regex_features.csv` | ✅ |
| 3 hand-crafted features | `outputs/ws_p4_s06_handcrafted_features.csv` | ✅ |
| Session 1 log | `01_session_log.md` | ✅ |
| Stage 1 sample | `outputs/ws_p4_s1_sample.csv` | ✅ |
| Slither V2 results | `outputs/ws_p4_s1_slither_results.csv` | ✅ |
| Aderyn V2 results | `outputs/ws_p4_s1_aderyn_results.csv` | ✅ |
| Stage 1 agreement (slither) | `outputs/ws_p4_s1_agreement_report.md` | ✅ |
| Stage 1 3-way agreement | `outputs/ws_p4_s1_3way_agreement_report.md` | ✅ |
| Manual review 43 | `outputs/ws_p4_s1_manual_review_50.csv` | ✅ |
| Manual review 199 | `outputs/ws_p4_s1_review_200.csv` | ✅ |
| **CRITICAL_FINDINGS.md** | `CRITICAL_FINDINGS.md` | ✅ |
| v1.2 dataset (if Stage 1-2 corrections) | `outputs/ws_p4_s2_contracts_clean_v12.csv` | ⏳ |
| Mythril 3-way | `outputs/ws_p4_s4_mythril_3way.md` | ⏳ |
| 8 per-folder manual reports | `reports/ws_p4_s5_*.md` × 8 | ⏳ |
| AutoML report | `outputs/ws_p4_s6_automl_report.md` | ⏳ |
| AutoML calibrations | `ml/calibration/automl_v13_*.json` × 5 | ⏳ |
| SHAP plot | `outputs/ws_p4_s6_shap_top10.png` | ⏳ |
| **v1.3 dataset (FINAL)** | `outputs/ws_p4_s7_contracts_clean_v13.csv` | ⏳ |
| v1.3 split | `outputs/ws_p4_s7_v13_split_assignments.csv` | ⏳ |
| CHANGELOG entries | `docs/CHANGELOG.md` §47-49 | ⏳ |
| MEMORY update | `~/.claude/.../memory/MEMORY.md` | ⏳ |

---

## Phase 4 quick-stats

- **Stages:** 7 (0.1-0.6, 1, 2, 3, 4, 5, 6, 7)
- **Decisions:** 7 (D-P4-1 through D-P4-7)
- **Sessions:** 4
- **Total time:** 40-70h
- **Total deliverables:** 20+ files
- **Main output:** `contracts_clean_v13.csv` (D-F1 + D-B2 + D-I-11 + per-folder corrections + Stage 5 manual + Stage 6 AutoML tuning)
