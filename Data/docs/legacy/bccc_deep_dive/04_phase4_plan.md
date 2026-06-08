# BCCC-SCsVul-2024 Deep Dive — Phase 4 Plan

**Title:** Phase 4 — Per-Folder Label Trustworthiness Validation, Mass NonVulnerable Correction, and AutoML Baselines
**Date:** 2026-06-07
**Author:** SENTINEL Data Engineering
**Source dataset:** `BCCC-SCsVul-2024/` (1.6 GB, read-only)
**Phase 3 deliverable (input):** [`../Phase2_Validation_2026-06-06/outputs/contracts_clean.csv`](../Phase2_Validation_2026-06-06/outputs/contracts_clean.csv) (67,311 × 24, SENTINEL v1.0)
**Phase 3 manual review (input):** [`../Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md`](../Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md) (32 contracts, 425 lines, **0 KEEPs, 28 MODIFY drop-NV, 2 MODIFY reclassify, 6 template clusters identified**)
**Phase 3 plan reference:** [`../03_phase3_plan.md`](../03_phase3_plan.md)
**Root README:** [`../README.md`](../README.md) — table of contents for the whole deep dive
**Status:** ⏳ **Planning complete, execution pending**

---

## 1. Why Phase 4 Exists (the gap analysis)

Phase 3 confirmed label quality on a **biased 808-contract sample** (95% review_pending). It proved:
- ✅ Reentrancy labels: 93% precision (trustworthy)
- ✅ CallToUnknown labels: 92% precision (trustworthy)
- ❌ IntegerUO labels: F1=0.07 with slither alone (need aderyn)
- ⏳ 7 other classes: too few samples to verify
- 🔬 **30+2 manual inspections revealed a systematic pattern:** `Class12:NonVulnerable` is wrong in 28/30 contracts where it co-occurs with `Reentrancy` and/or `CallToUnknown`

**Phase 4 closes the remaining gaps:**

| Gap from Phase 3 | Phase 4 answer |
|---|---|
| Sample biased toward review_pending (95%) | Per-folder stratified sampling (10-15% per folder, 8 folders) |
| IntegerUO F1=0.07 with slither alone | Add aderyn (D-P3-10) for cross-validation |
| Only 2 maxing contracts manually reviewed | Extend manual investigation to **3-5 contracts per folder** (theme-level) |
| `Class12:NonVulnerable` systemic error | Apply **D-I-11** mass correction before sampling |
| No quantitative label-quality signal for 7 other classes | Per-folder agreement analysis (8 separate confusion matrices) |
| No AutoML baseline | Run all 5 models × 50 Optuna trials × 5 folds on v1.3 |
| Mythril too slow for batch (3m/contract) | Use as 3rd-opinion tiebreaker on 50 hardest cases |

---

## 2. The 11 Design Additions (from the planning session + friend's review)

| ID | Addition | Source | Why |
|---|---|---|---|
| **α** | **Apply D-I-11 FIRST** (Stage 0) | Mine | Your manual review found a systemic pattern. Apply it as a 0-cost mass correction before any sampling. |
| **β** | **Sample size calculator + dedup** (15% per folder of **unique contracts** after Oraclize dedup, after excluding 32 reviewed) | Mine + friend | 15% is feasible in 1 session (~3h slither + ~1.5h aderyn). 30% = ~6h, 60% = ~12h. |
| **γ** | **Sequential with stop gates** (A → evaluate → B only if **median F1** < 0.5 → C only if < 0.5) | Yours + mine + friend | Saves 4-8 hours of compute if A is sufficient. **Use median (not mean) to be robust to IntegerUO outlier (F1=0.07).** |
| **δ** | **Mythmil 50-contract tiebreaker** (2-3 per folder, selected by highest disagreement **from Stage 1-2**) | Mine + friend | 3min × 50 = 2.5h budget. **Mythmil is a strict downstream of Stage 1-2 (cannot run in parallel).** |
| **ε** | **Use 30 manual review contracts as labeled anchors** | Mine | Their decisions (28 MODIFY drop-NV, 2 MODIFY reclassify) are a training set for a rule-based NV-contradiction detector. |
| **ζ** | **3 hand-crafted features** (2 NV-contradiction + 1 IntegerUO pre-0.8 regex) | Mine + friend | `nv_but_has_reentrancy_call`, `nv_but_has_external_call`, `unsafe_arith_no_safemath`. |
| **η** | **Oraclize dedup** (stripped-source SHA256, exclude near-duplicates from sampling frame) | Friend | 17 of 32 reviewed are Oraclize boilerplate; ~100-200 likely in full dataset. Sampling without dedup inflates agreement metrics. |
| **θ** | **Deliverable: `contracts_clean_v13.csv`** (v1.0 → v1.1 D-I-11 → **v1.2 per-folder corrections from Stage 1-2** → v1.3 = v1.2 + Stage 5 manual + Stage 6 AutoML tuning) | Mine + friend | v1.2 was missing in original plan. |
| **ι** | **Exclude 32 reviewed contracts from sampling frame** (`reviewed_in_phase3=1` flag) | Friend | Re-running slither/aderyn on them double-counts. |
| **κ** | **AutoML spec** (10 binary classifiers, one per class, **micro-F1** for primary comparison, per-class SMOTE, same 70/15/15 split) | Friend | Original plan was under-specified. |
| **λ** | **F1 gate uses median (not mean)** to be robust to IntegerUO outlier | Friend | Macro/mean F1 will fail the 0.5 gate on IntegerUO alone (F1=0.07). |

---

### Why these were added (friend's review identified 11 issues)

| # | Issue | Fix |
|---|---|---|
| 1 | Sampling unit inconsistent (1,400 vs 11,442) | β — unique contracts after dedup, explicit in Stage 1 |
| 2 | Mythmil "parallel" claim wrong (depends on Stage 1-2 disagreement) | δ — removed parallel claim, now sequential |
| 3 | WS-K-K1 prerequisite missing for Stage 6 | New Stage 0.5 = 31 regex features scan (1-pass on 67,311 contracts) |
| 4 | F1 gate will fail on IntegerUO alone | γ, λ — use median F1, exclude IntegerUO from gate calc |
| 5 | 61 remaining review_pending unaddressed | New Stage 0.6 — small targeted review of those 61 |
| 6 | Stage 5 time estimate (16-24h) too optimistic | New estimate: 2-4 days for 8 folders (24-40 contracts) |
| 7 | AutoML under-specified (binary vs multi-label, F1 variant, imbalance, split) | κ — full spec in Stage 6 |
| 8 | Oraclize cluster dedup missing | η — added to Stage 0 |
| 9 | 32 reviewed contracts should be excluded from sampling | ι — added to Stage 0 |
| 10 | IntegerUO pre-0.8 regex feature needed | ζ — 3rd hand-crafted feature |
| 11 | v1.1 → v1.3 jump unclear (no v1.2) | θ — defined v1.2 (per-folder corrections from Stage 1-2 only) |

---

## 3. The 7 Stages (sequential with stop gates)

### Stage 0: Apply D-I-11 + Dedup + Exclude-reviewed + Build regex features [Additions α, ζ, η, ι]
**Output:** `Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s0_*` (4 files)
**Decision:** New — **D-I-11** (formalized in `Phase3_DeepAnalysis_2026-06-06/decisions/D-I-11_drop_nv_with_vuln.md`)
**Time:** ~1-2 hours total for all 4 sub-steps

**Sub-step 0.1: Apply D-I-11 (15 min) [Addition α]**
1. Read your `ws_i_disagreement_inspections.md` (32 contracts done)
2. Apply rule: drop `Class12:NonVulnerable=1` when it co-occurs with any of {CallToUnknown, Reentrancy, GasException, MishandledException, DenialOfService, Timestamp}
3. Apply to `contracts_clean.csv` → produce `contracts_clean_v11.csv`
4. Set `review_pending=0` for affected contracts (now included in training)
5. Spot-check 10 random corrected contracts (all should still have ≥1 positive vulnerability class)
6. Output: `outputs/ws_p4_s0_d11_applied.csv` + `outputs/ws_p4_s0_d11_report.md`

**Sub-step 0.2: Oraclize cluster dedup (15 min) [Addition η]**
1. Compute `source_stripped_sha256` for every contract in v1.1 (strip whitespace + comments)
2. Group contracts by hash → identify clusters of ≥3 near-identical contracts
3. From each cluster, keep 1 representative + mark the rest `is_oraclize_dup=1`
4. Expected: ~100-200 Oraclize duplicates identified (per friend's extrapolation from 808-contract sample)
5. **All `is_oraclize_dup=1` contracts are excluded from Stage 1-3 sampling frame** (would inflate agreement metrics)
6. Output: `outputs/ws_p4_s0_dedup_clusters.csv` + `outputs/ws_p4_s0_sampling_frame.csv` (v1.1 minus Oraclize dups)

**Sub-step 0.3: Exclude 32 reviewed contracts (1 min) [Addition ι]**
1. Add `reviewed_in_phase3=1` flag to the 32 contract IDs from `ws_i_disagreement_inspections.md`
2. Exclude from sampling frame
3. Output: `outputs/ws_p4_s0_sampling_frame.csv` (final: v1.1 minus Oraclize dups minus 32 reviewed)

**Sub-step 0.4: Resolve remaining ~61 review_pending (30 min) [Fixes friend's #5]**
1. After D-I-11, ~61 contracts still have `review_pending=1` (they had NV but co-occurrence didn't trigger D-I-11, e.g., NV alone or NV with only UnusedReturn)
2. **Decision D-P4-7 (TBD):** What to do with them?
   - **Default (chosen):** Spot-review 5-10 of the 61 (~10% sample) to see if they're systematic or edge cases
   - If they look like another systemic pattern → another D-I-* mass correction
   - If they're truly edge cases → keep as `review_pending=1`, exclude from training, document in v1.1 metadata
3. Output: `outputs/ws_p4_s0_remaining_review_pending.md` (decision + 5-10 spot review notes)

**Sub-step 0.5: Compute 31 regex features on 67,311 contracts (30-60 min) [Fixes friend's #3, replaces "WS-K-K1 pending"]**
1. Apply all 31 regex features (defined in `03_phase3_plan.md` §WS-K-K1) to every contract in v1.1
2. One-pass scan, ~1-2s per contract → ~30-60 min total
3. Output: `outputs/ws_p4_s0_5_regex_features.csv` (67,311 × 31)
4. This is the prerequisite for Stage 6 (AutoML) — must complete before Stage 6

**Sub-step 0.6: Compute 3 hand-crafted features (~10 min) [Addition ζ]**
1. `nv_but_has_reentrancy_call` = (Class12:NonVulnerable=1) AND (`.call{value:}` or low-level `.call()` exists)
2. `nv_but_has_external_call` = (Class12:NonVulnerable=1) AND (any external contract call exists)
3. `unsafe_arith_no_safemath` = (arithmetic op `+`, `-`, `*` exists) AND (not inside `SafeMath.add/sub/mul` call) [Friend's #10 — addresses the known IntegerUO F1=0.07 gap directly]
4. All 3 = 100% recall on the 30 reviewed contracts (the 2 NV features) and high signal for IntegerUO
5. Output: `outputs/ws_p4_s0_6_handcrafted_features.csv` (67,311 × 3) — joined to regex features for Stage 6
6. **Total feature count for Stage 6: 31 regex + 3 hand-crafted = 34 features**

**Caveat for sub-step 0.1:** Rule may also apply to the **non-review-pending** 67,311 contracts (where BCCC didn't flag for review but the same NonVulnerable error could exist). Stage 1 will measure if the rule generalizes. **D-P4-1 (TBD in Phase 4):** Apply D-I-11 broadly (all contracts) or narrowly (only review_pending)?
- **Default:** Apply narrowly (review_pending only) to be conservative
- **Override:** If Stage 1.5 generalization check shows the rule applies to non-review_pending (e.g., 10%+ of non-review_pending contracts have the same NV+Reentrancy co-occurrence), apply broadly

---

### Stage 1: Per-folder stratified validation — Step A (15% per folder of **unique contracts**) [Additions β, γ, λ]
**Output:** `outputs/ws_p4_s1_per_folder_agreement.md` + `outputs/ws_p4_s1_results.csv`
**Tools:** slither 0.11.5 + aderyn 0.6.8 in parallel
**Time:** ~3h (slither) + ~1.5h (aderyn) = ~4.5h total wall time
**Sampling unit:** **Unique contracts** (not (contract, folder) pairs) — every contract in v1.1 is in 0-8 folders, so the unique count is the deduped total

**Sampling math (corrected from friend's #1):**
- Pre-v1.1: 67,311 contracts in 8 SENTINEL folders
- After Stage 0.2 (Oraclize dedup): exclude ~100-200 Oraclize dups → ~67,100 unique contracts
- After Stage 0.3 (exclude 32 reviewed): ~67,070 unique contracts
- **Sample 15% of unique contracts, stratified by primary_class** = ~10,000 unique contracts (not 1,400, not 11,442)
- 15% of unique contracts is much smaller than 15% per folder because folder memberships overlap
- **Wait — let me recalculate:** if every contract is in ~1.5 folders on average, then 15% per folder is 15% × 8 / 1.5 = 80% of unique contracts. That's too much.

**Corrected sampling approach (friend's #1 fix):**
- **Sample 15% of unique contracts in the sampling frame** (~10,000 contracts)
- **Stratify by primary_class** so all 10 classes are represented proportionally
- This is **~10× larger** than the original 1,400 estimate
- Compute time scales: 10,000 contracts × 2 tools × ~5s/contract / 6 workers = ~4.6h wall time per tool = ~9h total (sequential) or ~4.5h (parallel)

**Steps:**
1. Start from `outputs/ws_p4_s0_sampling_frame.csv` (67,070 unique contracts after dedup + exclude-reviewed)
2. Stratified sample: 15% per primary_class, then deduplicate (a contract is sampled once even if in multiple classes)
3. Expected sample size: ~10,000 unique contracts (15% of 67,070)
4. For sampled contracts, run both slither and aderyn (~4.5h parallel, ~9h sequential)
5. Compute per-class agreement: for each of 10 classes, TP/FP/FN/TN, precision, recall, F1
6. Output 10 confusion matrices + 10 agreement reports (one per class)
7. **Decision gate:** compute **MEDIAN F1 across the 8 SENTINEL classes** (exclude NonVulnerable since slither has no NV detector). Friend's #4 fix: median is robust to IntegerUO outlier.
   - **If median F1 ≥ 0.5** → labels trustworthy, skip Stages 2-3
   - **If median F1 < 0.5** → escalate to Stage 2
   - **Special case:** if IntegerUO alone is the only class with F1 < 0.5 (and the other 7 classes have F1 ≥ 0.5) → do NOT escalate. IntegerUO is a known limitation of static analysis for pre-0.8 code. Document the gap, rely on AutoML + the `unsafe_arith_no_safemath` feature for this class.

**Sub-step 1.5: D-I-11 generalization check (for D-P4-1)**
- For the ~10,000 sampled contracts, measure how many have the NV+vuln co-occurrence pattern
- If >5% of non-review_pending contracts have the same pattern → D-I-11 should be applied broadly (D-P4-1 → yes)
- If <1% → D-I-11 is review_pending-specific (D-P4-1 → no, keep narrow)
- This will inform whether D-I-11 is review-pending-specific or dataset-wide

---

### Stage 2: Escalation to Step B (30% per unique contract) [Only if Stage 1 median F1 < 0.5]
**Output:** `outputs/ws_p4_s2_per_folder_agreement.md`
**Time:** ~9h slither + ~4.5h aderyn = ~13.5h additional (parallel) or ~13.5h (sequential)

**Steps:**
1. Add another 15% of unique contracts to the sample (total 30% of sampling frame, ~20,000 contracts)
2. Re-run agreement analysis on the combined Stage 1+2 = 30% sample
3. **Decision gate:** if **median F1 across 8 classes** still < 0.5 → escalate to Stage 3

**Trigger condition:** Stage 1 median F1 < 0.5 (and IntegerUO is not the only outlier). Otherwise STOP and proceed to Stage 4 with Stage 1 results.

---

### Stage 3: Step C (50% per unique contract) [Only if Stage 2 median F1 < 0.5]
**Output:** `outputs/ws_p4_s3_per_folder_agreement.md`
**Time:** ~18h slither + ~9h aderyn = ~27h additional

**Steps:**
1. Add another 20% of unique contracts (total 50% of sampling frame, ~33,000 contracts)
2. Re-run agreement on combined sample
3. Document per-class final F1, precision, recall, FPR

**Trigger condition:** Stage 2 median F1 < 0.5. Otherwise STOP after Stage 2.

---

### Stage 4: Mythril tiebreaker on 50 hardest cases [Addition δ]
**Output:** `outputs/ws_p4_s4_mythril_3way.md`
**Time:** 2.5h (50 contracts × 3min)
**Strict downstream of Stage 1 (or Stage 2 if escalated) — CANNOT run in parallel** (friend's #2 fix)

**Steps:**
1. From the Stage 1 (or Stage 2) sample, rank contracts by `(slither_disagreement + aderyn_disagreement) / 2`
2. Pick top 50, ensuring 2-3 per folder (so we cover all 8 folders)
3. Run mythril on each (Docker `mythril/myth:0.24.8` — D-P3-1)
4. Compare: BCCC label vs slither vs aderyn vs mythril
5. Document per-folder "3-way consensus" — for each folder, what % of BCCC's labels are confirmed by 2 of 3 tools?

**Deliverable:** Per-folder table:
| Folder | n_sampled | BCCC ∩ Slither | BCCC ∩ Aderyn | BCCC ∩ Mythril | 2-of-3 consensus |
|---|---:|---:|---:|---:|---:|
| CallToUnknown | 1,670 | 92% | 88% | 85% | 90% |
| Reentrancy | 2,655 | 93% | 89% | 87% | 91% |
| ... | ... | ... | ... | ... | ... |

---

### Stage 5: Per-folder manual investigation (theme-level) [Your idea #4 + friend's #6]
**Output:** `reports/ws_p4_s5_<folder>_investigation.md` × 8 folders
**Time:** **2-4 days for 8 folders** (friend's #6 fix — original 16-24h was optimistic)
- Per folder: 3-5 contracts × ~1-2h per contract = 3-10h
- × 8 folders = 24-80h of focused manual work
- **Realistic:** 2-4 days across 2-3 dedicated sessions (not 2-3h per folder)

**Steps:**
1. After Stage 1 (or Stage 2/3), for each of 8 SENTINEL folders:
   - Pick 3-5 representative contracts (1 high-agreement, 1 disagreement, 1 maxing, 1 typical, 1 from Oraclize cluster if relevant)
   - Read source, document the contract's actual vulnerability profile
   - Note BCCC's label, slither/aderyn/mythril findings, and your manual verdict
2. Output: 8 markdown reports, one per folder
3. Pattern: "for folder X, BCCC labels are [trustworthy / partially trustworthy / not trustworthy] because [reason]"
4. Update D-F1, D-I-11, or create new D-P4-* decisions as needed

**Why this complements your 30-contract review:** Your review was on contracts with the *worst* slither disagreement. Per-folder investigation looks at *all* contracts in a folder, including the ones that agree. If 95% of folder X contracts are correctly labeled but 5% aren't, the per-folder view captures that.

**Session plan:** Spread Stage 5 across Sessions 2-3 (4 folders per session) instead of cramming into one.

---

### Stage 6: AutoML on v1.3 [Your idea #5 + Additions ζ, κ]
**Output:** `outputs/ws_p4_s6_automl_report.md` + `ml/calibration/automl_v13_<model>.json` × 5
**Time:** 8-12h total (parallel: 50 trials × 5 folds × 5 models × 10 classes = 2,500 fits)
**Prerequisite:** Stage 0.5 (31 regex) + Stage 0.6 (3 hand-crafted) = 34 features total

**Specification (friend's #7 fix):**

1. **Task formulation:** **10 binary classifiers, one per SENTINEL class** (not a single multi-label estimator)
   - Each binary classifier is independently trained and tuned
   - Output: 10 sets of predictions, one per class
   - Easier to debug per-class issues; mirrors SENTINEL's 10 binary heads (ADR-0002)

2. **Primary metric:** **Micro-F1 across 10 classes** (per-class predictions pooled)
   - Secondary metrics: per-class F1, precision, recall, macro-F1, weighted-F1
   - Calibration: per-class thresholds tuned on val set

3. **Class imbalance handling (per class):**
   - Class size n < 1,000 (rare: ExternalBug 3,604, UnusedReturn 3,229, Timestamp 2,674): **SMOTE oversampling** + scale_pos_weight
   - Class size 1,000 ≤ n < 10,000 (medium: GasException 6,879, MishandledException 5,154): **scale_pos_weight only**
   - Class size n ≥ 10,000 (common: Reentrancy 17,698, IntegerUO 16,740, DoS 12,394, CallToUnknown 11,131, NonVulnerable 26,914): **no oversampling** (already balanced enough)
   - **SMOTE caveat:** works very differently for rare classes (Timestamp n=2,674 → synthetic samples) vs common (NonVulnerable n=26,914 → no synthetics). Document per-class strategy.

4. **Train/val/test split:** **Same 70/15/15 as SENTINEL v1.3** (uses `split_assignments.csv` from Stage 0)
   - No re-splitting. AutoML and SENTINEL train on the exact same contracts.
   - This is what makes the comparison fair.

5. **5 models:** XGBoost, LightGBM, CatBoost, RandomForest, LogReg
   - 4 gradient-boosted + 1 bagged + 1 linear (sanity baseline)

6. **Hyperparameter tuning:** Optuna 50 trials × 5-fold stratified CV per (model × class) = 50 × 5 × 5 × 10 = 12,500 fits
   - Per-class optuna search spaces (e.g., XGBoost's max_depth differs for rare vs common classes)

7. **Output:** Per-class F1, per-fold variance, top 5 features via SHAP
   - **Framing of comparison to SENTINEL** (friend's #7):
     - AutoML gets **34 static regex + count features** (loss of source structure)
     - SENTINEL gets **full graph embeddings + CodeBERT subword representations** (preserves source structure)
     - **If AutoML ≥ SENTINEL** → graph structure adds nothing (unlikely but meaningful — would justify simpler baselines)
     - **If AutoML << SENTINEL** → graph structure is doing the heavy lifting (validates SENTINEL's complexity)
     - **If AutoML ≈ SENTINEL** → SENTINEL's complexity is not justified, simpler model preferred
   - Comparison on **micro-F1** as the primary metric
   - Document the comparison explicitly in `ws_p4_s6_automl_report.md` (not just numbers)

---

### Stage 7: Synthesis — v1.3 + CHANGELOG §47-49 + MEMORY update [Friend's #11 fix]
**Output:** `Phase4_LabelValidation_2026-06-07/outputs/contracts_clean_v13.csv` + `docs/CHANGELOG.md` §47-49 + MEMORY update
**Versioning (friend's #11):**
- **v1.0** (Phase 2): D-F1 + D-B2, 67,311 contracts
- **v1.1** (Phase 4 Stage 0): + D-I-11 (drop NonVulnerable), ~705 contracts unlocked
- **v1.2** (Phase 4 Stage 1-2 only, **intermediate**): + per-folder label corrections from slither+aderyn agreement (e.g., if Stage 1 finds 50 CallToUnknown contracts in folder X are wrong, fix them)
- **v1.3** (Phase 4 final): + Stage 5 manual investigation + Stage 6 AutoML tuning adjustments

**Why v1.2 exists as intermediate:** friend pointed out the v1.1 → v1.3 jump was unclear. v1.2 captures the algorithmic corrections (from slither/aderyn agreement) before the human-judgment corrections (Stage 5). This makes the dataset evolution traceable.

**Steps:**
1. Build v1.1 from D-I-11 (Stage 0)
2. Build v1.2 from Stage 1-2 per-folder agreement (algorithmic corrections)
3. Build v1.3 from Stage 5 manual corrections + Stage 6 AutoML-informed threshold tuning
4. Re-run stratified split (70/15/15) on v1.3
5. CHANGELOG §47 (v1.1 + D-I-11), §48 (v1.2 + per-folder), §49 (v1.3 + Stage 5/6)
6. MEMORY update with Phase 4 final state
7. Update root README + Phase 4 README

---

## 4. Decision Gates (the "stop early" criteria) [Friend's #4 fix: median, not mean]

| Stage | Continue if... | Stop and move on if... |
|---|---|---|
| **Stage 1** | **Median F1** < 0.5 across 8 classes (and IntegerUO is not the sole outlier) | **Median F1 ≥ 0.5** (labels trustworthy, no need to escalate) |
| **Stage 2** | Stage 1 median F1 < 0.5 | Stage 1 median F1 ≥ 0.5 (Stage 2 was skipped) |
| **Stage 3** | Stage 2 median F1 < 0.5 | Stage 2 median F1 ≥ 0.5 (Stage 3 was skipped) |
| **Stage 4** | Always run (downstream of Stage 1) | — |
| **Stage 5** | Per-folder investigation always runs | — |
| **Stage 6** | Always run | — |

**Special case (friend's #4):** If **IntegerUO alone** is the only class with F1 < 0.5 (and the other 7 classes have F1 ≥ 0.5):
- Do NOT escalate to Stage 2/3
- Document the gap as "known static-analysis limitation for pre-0.8 code"
- Rely on AutoML (Stage 6) + the `unsafe_arith_no_safemath` feature (Stage 0.6) for this class

**Best case:** Stage 1 median F1 ≥ 0.5 → only Stages 0, 1, 4, 5, 6, 7 run = ~25h
**Worst case:** Stage 3 needed (median F1 < 0.5 in both Stage 1 and Stage 2) → all 7 stages run = ~75h (revised up due to friend's #6)
**Expected:** Stage 1 median F1 in 0.4-0.6 range → Stages 0, 1, 2, 4, 5, 6, 7 = ~55h

---

## 5. Estimated Timeline [Revised per friend's #1, #3, #6]

**Compute time estimates (revised, larger sampling):**
- Stage 1 sample size: ~10,000 unique contracts (15% of 67,070 sampling frame), not 1,400
- Stage 1 compute: 10,000 × 2 tools × ~5s/contract / 6 workers = ~4.6h wall time per tool
- Stage 2 compute: 10,000 more contracts = ~4.6h additional per tool
- Stage 3 compute: 13,000 more contracts = ~6h additional per tool
- Stage 4: 50 contracts × 3min = 2.5h (sequential, no parallelism)
- Stage 6: 12,500 fits = ~10-12h (can run overnight)

| Session | Stages | Est. time |
|---|---|---|
| **Session 1** | Stage 0 (4 sub-steps, ~1.5h) + Stage 1 (10k contracts, ~9h compute parallel) | 10-12h |
| **Session 2** | Stage 4 (mythril 50, 2.5h) + Stage 2 if needed (~9h) + Stage 5 first 4 folders (~10-20h) | 12-24h |
| **Session 3** | Stage 3 if needed (~12h) + Stage 5 remaining 4 folders (~10-20h) + Stage 6 (AutoML, 10-12h) | 16-30h |
| **Session 4** | Stage 7 (synthesis) + CHANGELOG + MEMORY + review | 2-3h |
| **Total** | — | **40-70h** (revised up from 30-50h) |

**Why larger:** friend's #1 forced us to clarify sampling unit (unique contracts, not pairs). 10,000 unique contracts is the right size for statistical power on per-class F1. **Friend's #6 forced realistic Stage 5 time estimate (24-40h of focused work, not 16-24h).**

---

## 6. Tools & Compute

| Tool | Purpose | Status |
|---|---|---|
| **slither 0.11.5** | Static analysis (101 detectors, Python) | ✅ Installed in root `.venv` |
| **aderyn 0.6.8** | Static analysis (88 detectors, Rust) | ✅ Installed at `~/.cargo/bin/aderyn` |
| **mythril 0.24.8** | Symbolic execution (17 detectors, Docker) | ✅ Docker image kept (D-P3-1) — for ad-hoc use |
| **xgboost 3.2.0** | AutoML — gradient boosting | ✅ Installed (Tsinghua mirror) |
| **lightgbm 4.6.0** | AutoML — gradient boosting | ✅ Installed |
| **catboost 1.2.10** | AutoML — gradient boosting | ✅ Installed |
| **optuna 4.9.0** | Hyperparameter tuning | ✅ Installed |
| **shap 0.52.0** | Feature importance | ✅ Installed |
| **imbalanced-learn 0.14.1** | SMOTE, class weights | ✅ Installed |
| **solc-select 100+ versions** | Multi-version Solidity compilation | ✅ `~/.solc-select/artifacts/solc-X.Y.Z/` |

**Compute budget:**
- Stage 1: 1,400 contracts × 2 tools × ~5s/contract = ~4h wall time (with 6 workers)
- Stage 2: 1,400 more contracts = ~4h additional
- Stage 3: 1,400 more = ~4h additional
- Stage 4: 50 contracts × 3min = 2.5h (sequential, no parallelism)
- Stage 6: 1,250 AutoML fits = ~8-12h (can run overnight)
- **Total compute: ~25-35h wall time across 3-4 sessions**

---

## 7. Directory Layout (Phase 4) [Updated with all Stage 0 sub-steps]

```
Phase4_LabelValidation_2026-06-07/                 [to be created when Session 1 starts]
├── README.md                                      [Phase 4 entry point]
├── 00_session_log.md                              [Session-by-session timeline]
├── 00_understanding_checklist.md                  [Teaching doc: Problem/Solution/Context per stage]
├── 04_phase4_plan.md                              [This file]
├── scripts/
│   ├── ws_p4_s01_apply_d11.py                     [Stage 0.1: D-I-11 mass correction → v1.1]
│   ├── ws_p4_s02_oraclize_dedup.py                [Stage 0.2: Oraclize cluster dedup by stripped-source SHA256]
│   ├── ws_p4_s03_exclude_reviewed.py              [Stage 0.3: add reviewed_in_phase3=1 flag, exclude]
│   ├── ws_p4_s04_resolve_remaining_review.py      [Stage 0.4: spot-review ~61 remaining review_pending]
│   ├── ws_p4_s05_regex_features.py                [Stage 0.5: 31 regex features on 67,311 contracts]
│   ├── ws_p4_s06_handcrafted_features.py          [Stage 0.6: 2 NV + 1 IntegerUO regex features]
│   ├── ws_p4_s1_sampling.py                       [Stage 1: 15% per primary_class → ~10,000 contracts]
│   ├── ws_p4_s1_slither.py                        [Stage 1: slither batch on ~10,000 contracts]
│   ├── ws_p4_s1_aderyn.py                         [Stage 1: aderyn batch on ~10,000 contracts]
│   ├── ws_p4_s1_agreement.py                      [Stage 1: median F1 across 8 classes, decision gate]
│   ├── ws_p4_s2_escalate_30pct.py                 [Stage 2: add 15% to sample]
│   ├── ws_p4_s3_escalate_50pct.py                 [Stage 3: add 20% to sample]
│   ├── ws_p4_s4_mythril_tiebreaker.py             [Stage 4: 50 contracts × 3 tools (downstream of S1-2)]
│   ├── ws_p4_s5_per_folder_manual.py              [Stage 5: 3-5 contracts per folder (theme-level)]
│   ├── ws_p4_s6_automl_v13.py                     [Stage 6: 10 binary × 5 models × 50 trials × 5 folds]
│   └── ws_p4_s7_synthesis_v13.py                  [Stage 7: build v1.3 + CHANGELOG §47-49]
├── outputs/
│   ├── ws_p4_s01_d11_applied.csv                  [v1.1: D-I-11 mass correction]
│   ├── ws_p4_s01_d11_report.md                    [How many contracts affected, sample changes]
│   ├── ws_p4_s02_dedup_clusters.csv               [Oraclize cluster identification]
│   ├── ws_p4_s02_sampling_frame.csv               [v1.1 − Oraclize dups − 32 reviewed = ~67,070]
│   ├── ws_p4_s04_remaining_review_pending.md      [Decision on ~61 review_pending]
│   ├── ws_p4_s05_regex_features.csv               [67,311 × 31 regex features]
│   ├── ws_p4_s06_handcrafted_features.csv         [67,311 × 3 hand-crafted features]
│   ├── ws_p4_s1_sample_15pct.csv                  [~10,000 contracts × primary_class stratification]
│   ├── ws_p4_s1_slither_results.csv               [Slither findings on ~10,000 contracts]
│   ├── ws_p4_s1_aderyn_results.csv                [Aderyn findings on ~10,000 contracts]
│   ├── ws_p4_s1_per_folder_agreement.md           [10 confusion matrices + median F1 decision gate]
│   ├── ws_p4_s2_contracts_clean_v12.csv           [v1.2: per-folder algorithmic corrections]
│   ├── ws_p4_s2_per_folder_agreement.md           [Stage 2 results if escalated]
│   ├── ws_p4_s3_per_folder_agreement.md           [Stage 3 results if escalated]
│   ├── ws_p4_s4_mythril_3way.md                   [50-contract 3-way consensus table]
│   ├── ws_p4_s6_automl_report.md                  [10 binary × 5 models × 50 trials × 5 folds]
│   ├── ws_p4_s6_shap_top10.png                    [Feature importance visualization]
│   ├── ws_p4_s7_contracts_clean_v13.csv           [v1.3 MAIN DELIVERABLE]
│   └── ws_p4_s7_v13_split_assignments.csv         [70/15/15 split for v1.3]
├── reports/
│   ├── ws_p4_s5_call_to_unknown_investigation.md  [Stage 5: 3-5 CallToUnknown contracts]
│   ├── ws_p4_s5_reentrancy_investigation.md       [Stage 5: 3-5 Reentrancy contracts]
│   ├── ws_p4_s5_integer_uo_investigation.md       [Stage 5: 3-5 IntegerUO contracts]
│   ├── ws_p4_s5_denial_of_service_investigation.md
│   ├── ws_p4_s5_gas_exception_investigation.md
│   ├── ws_p4_s5_mishandled_exception_investigation.md
│   ├── ws_p4_s5_external_bug_investigation.md
│   └── ws_p4_s5_timestamp_investigation.md
├── labels/                                        [Empty in Phase 4 — manual review happened in Phase 3]
│   └── (Stage 5 will write 8 investigation reports here)
└── decisions/
    └── D-I-11_drop_nv_with_vuln.md                [Formal D-I-11 writeup, in Phase 3 dir]
```

---

## 8. Decisions to Make Before Session 1 [Updated with new D-P4-7]

| ID | Decision | Default | Override |
|---|---|---|---|
| **D-P4-1** | Apply D-I-11 to review_pending only (conservative) or to all 67,311 contracts (broad)? | review_pending only | Apply broadly if Stage 1.5 shows the rule generalizes |
| **D-P4-2** | Stage 1 sampling: 10% (~6,700 contracts, faster) or 15% (~10,000, more statistical power)? | 15% | 10% if we want to fit Session 1 in 8h |
| **D-P4-3** | Mythril: 50 contracts (2.5h, budget) or 100 (5h, more confidence)? | 50 | 100 if Session 2 has time |
| **D-P4-4** | AutoML: 50 trials × 5 folds (12,500 fits, ~12h) or 25 trials × 5 folds (6,250 fits, ~6h)? | 50×5 | 25×5 if Session 3 has time pressure |
| **D-P4-5** | Include LogReg in AutoML? (Slow + linear, may not help) | Yes | No if it adds > 1h with no F1 gain |
| **D-P4-6** | Apply 3 hand-crafted features (2 NV + 1 IntegerUO regex) to v1.3? | Yes, add them | No, defer to v1.4 |
| **D-P4-7** (NEW) | What to do with the ~61 remaining review_pending after D-I-11? | Spot-review 5-10 of them in Stage 0.4 | (a) drop all, (b) keep held out forever, (c) Stage 5 covers them |

---

## 9. Key Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Stage 1 F1 < 0.5 → all 3 stages run → 50h total | Medium (50%) | Sequential gates mean we stop early if possible |
| D-I-11 mass correction breaks some contracts that are correctly labeled NV | Low | Stage 0.5 spot-checks 10 random corrected contracts |
| Mythril Docker too slow / hangs on some contracts | Medium | Add 5min per-contract timeout; if mythril fails, document as "no mythril verdict" |
| AutoML beats SENTINEL by 10+ points | Low | Document as D-P3-7; don't change SENTINEL training |
| Oraclize cluster (17 contracts) skews stratified split | Medium | Document in v1.3 metadata; consider removing the 16 near-duplicates |
| Tsinghua pypi mirror blocks new packages needed for AutoML | Low | All deps already installed; if new ones needed, use Tsinghua or copy from ml/.venv |

---

## 10. Key Files to Read Before Phase 4 Starts

1. **[`../Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md`](../Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md)** (425 lines) — your manual review. This is the source of D-I-11 and the 6 cluster templates.
2. **[`../Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_agreement_report.md`](../Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_agreement_report.md)** (141 lines) — the per-class agreement that motivated per-folder investigation.
3. **[`../Phase3_DeepAnalysis_2026-06-00_understanding_checklist.md`](../Phase3_DeepAnalysis_2026-06-06/00_understanding_checklist.md)** (242 lines) — teaching doc with Problem/Solution/Context per stage. Will be extended in Phase 4.
4. **[`../Phase2_Validation_2026-06-06/outputs/contracts_clean.csv`](../Phase2_Validation_2026-06-06/outputs/contracts_clean.csv)** — the v1.0 dataset that D-I-11 will modify.
5. **[`../03_phase3_plan.md`](../03_phase3_plan.md)** — the Phase 3 plan (for context on D-P3-1 through D-P3-10).

---

## 11. What Phase 4 Does NOT Do

- ❌ Re-train SENTINEL on v1.3 (that's Run 10+ in the main training pipeline, not Phase 4)
- ❌ Run mythril on more than 50 contracts (too slow)
- ❌ Manually review more contracts (your 30-contract review was the manual cap; Phase 4 scales it programmatically via the 6 cluster templates)
- ❌ Add new SENTINEL architecture (ADR-0002 stays at 10 classes; D-F1 stays as drop Class05/07)
- ❌ Re-run SmartBugs OOD benchmark (use existing OOD data from Run 7/9)

---

## 12. Deliverables Summary [Updated with v1.1, v1.2, v1.3 + 34 features]

| Deliverable | Path | Purpose |
|---|---|---|
| `contracts_clean_v11.csv` | `outputs/ws_p4_s0_d11_applied.csv` | v1.1: D-I-11 mass correction only (after Stage 0.1) |
| `contracts_clean_v12.csv` | `outputs/ws_p4_s2_contracts_clean_v12.csv` | v1.2: + per-folder algorithmic corrections (after Stage 1-2) |
| `contracts_clean_v13.csv` | `outputs/ws_p4_s7_contracts_clean_v13.csv` | **Final v1.3** (after Stages 0-6, including Stage 5 manual + Stage 6 AutoML-informed) |
| `ws_p4_s0_dedup_clusters.csv` | `outputs/ws_p4_s0_dedup_clusters.csv` | Oraclize cluster identification (Stage 0.2) |
| `ws_p4_s0_sampling_frame.csv` | `outputs/ws_p4_s0_sampling_frame.csv` | Final sampling frame (v1.1 − Oraclize dups − 32 reviewed) |
| `ws_p4_s0_5_regex_features.csv` | `outputs/ws_p4_s0_5_regex_features.csv` | 31 regex features on 67,311 contracts (Stage 0.5) |
| `ws_p4_s0_6_handcrafted_features.csv` | `outputs/ws_p4_s0_6_handcrafted_features.csv` | 3 hand-crafted features (2 NV + 1 IntegerUO regex) (Stage 0.6) |
| 8 per-folder investigation reports | `reports/ws_p4_s5_*.md` | Theme-level label quality (Stage 5) |
| Per-folder agreement metrics | `outputs/ws_p4_s1_per_folder_agreement.md` (or s2/s3) | Quantitative label quality (Stage 1-3) |
| Mythril 3-way consensus | `outputs/ws_p4_s4_mythril_3way.md` | 3rd-opinion tiebreaker (Stage 4) |
| AutoML report | `outputs/ws_p4_s6_automl_report.md` | 10 binary × 5 models × 50 trials × 5 folds (Stage 6) |
| D-I-11 formal writeup | `Phase3_DeepAnalysis_2026-06-06/decisions/D-I-11_drop_nv_with_vuln.md` | Decision documentation (115 lines) |

---

## 13. Related

- [`../README.md`](../README.md) — root table of contents (will be updated with Phase 4 link)
- [`../01_exploration_inventory.md`](../01_exploration_inventory.md) — Phase 1
- [`../02_validation_deep_dive_plan.md`](../02_validation_deep_dive_plan.md) — Phase 2 plan
- [`../03_phase3_plan.md`](../03_phase3_plan.md) — Phase 3 plan (with D-P3-10 aderyn integration)
- [`../Phase2_Validation_2026-06-06/README.md`](../Phase2_Validation_2026-06-06/README.md) — Phase 2 entry point
- [`../Phase3_DeepAnalysis_2026-06-06/README.md`](../Phase3_DeepAnalysis_2026-06-06/README.md) — Phase 3 entry point
- [`../Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md`](../Phase3_DeepAnalysis_2026-06-06/labels/ws_i_disagreement_inspections.md) — your manual review (input to D-I-11)
- `BCCC-SCsVul-2024/` — source dataset (read-only)
- `docs/ml/adr/INDEX.md` — ADR-0005 (BCCC dataset choice)
- `docs/CHANGELOG.md` — needs §47-48 entries after Phase 4
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` — SENTINEL project memory (updated 2026-06-07 with Phase 3 final state + Phase 4 plan preview)

---

**Last updated:** 2026-06-07 (Plan drafted, 8 additions integrated, 6 stages defined, decision gates established, D-I-11 formalized)
