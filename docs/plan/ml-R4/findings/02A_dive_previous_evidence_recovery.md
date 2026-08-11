# 02A — DIVE Previous Evidence Recovery

- **Run ID:** R4-P1-DIVE-20260716
- **Phase:** 1
- **Date:** 2026-07-16
- **Status:** COMPLETE

## Evidence Sources Recovered

| Evidence set ID | Artifacts | Method | Status |
|---|---|---|---|
| R4-PREV-DIVE-EBRE | R4-P0-EVD-001 (150-contract crosswalk review md) | Manual source review, seed=42, 75 EB + 75 RE | RECOVERED_VERIFIED |
| R4-PREV-DIVE-SLITHER | R4-P0-EVD-002 (175-contract Slither-agreed review md) | Manual source review, seed=7, 100 EB + 75 RE | RECOVERED_VERIFIED |
| R4-PREV-DIVE-CORROB | R4-P0-EVD-003 (Slither corroboration JSON, 2MB) | Automated Slither agreement, 15,920 EB + 11,018 RE checked | RECOVERED_VERIFIED |
| R4-PREV-DIVE-ADERYN1 | data_module/audit/2026-06-18_dive_aderyn_per_contract_v1.json (116KB) | Aderyn 0.6.8, 175 Slither-agreed contracts | RECOVERED_VERIFIED |
| R4-PREV-DIVE-ADERYN2 | data_module/audit/2026-06-18_dive_aderyn_on_slither_disagreed_v1.json (190KB) | Aderyn 0.6.8, 400 Slither-disagreed contracts | RECOVERED_VERIFIED |
| R4-PREV-DIVE-SLITHER-CACHE | data_module/data/slither_cache/dive/ (17,287 files) | Slither per-contract, all DIVE classes | RECOVERED_PARTIAL (no individual hashes) |
| R4-PREV-DIVE-ADERYN-CACHE | data_module/data/aderyn_cache/dive/ (573 files) | Aderyn per-contract cache | RECOVERED_PARTIAL |
| R4-PREV-DIVE-LABELS | R4-P0-LBL-001 (DIVE_Labels.csv, 22,330 rows) | Source-native automated labels (8 DASP classes) | RECOVERED_VERIFIED |
| R4-PREV-CROSSWALKS | R4-P0-XWK-001 (dive.yaml crosswalk) | Pipeline config mapping 8 DASP -> 10 SENTINEL classes | RECOVERED_VERIFIED |
| R4-PREV-DATA-AUDIT | R4-P0-EVD-007 (data module audit) | 8-file audit suite + v2_full_audit/ with 7 files + 6 plans | RECOVERED_VERIFIED |

## Historical Claims Recovered

### Claim 1: DIVE folder labels are drastically over-labeled
- **Evidence:** Crosswalk sample validation (150 contracts, seed=42)
- **ExternalBug folder TP rate:** 5.3% (4/75 per header TALLY; 3/75 per CONTRACT TABLE — count discrepancy of 1)
- **Reentrancy folder TP rate:** 4.0% (3/75) / 4.2% (3/72 excluding UNCLEAR)
- **Verdict:** Both folders over-labeled. Option B (drop DIVE folder labels) recommended.

### Claim 2: DIVE+Slither agreement does NOT improve precision
- **Evidence:** Slither agreed subset validation (175 contracts, seed=7)
- **EB agreed TP rate:** 4.0% (4/100) — comparable to raw folder 5.3%
- **RE agreed TP rate:** 2.7% (2/75) — worse than raw folder 4.2%
- **Claimed mechanism:** Both tools fire on same superficial patterns (OZ libraries, standard ERC20, `approveAndCall`, `sendValue`)

### Claim 3: Triple-agreement (DIVE+Slither+Aderyn) is even worse
- **EB triple TP rate:** 3.0% (2/66)
- **RE triple TP rate:** 1.7% (1/59)
- **Aderyn-only on Slither-disagreed:** 0.0% (0/30) — pure noise

### Claim 4: Recommended v3.1 label counts
- **ExternalBug:** 39 solidifi + 17 smartbugs_curated + 4 manual seeds = 60 positives
- **Reentrancy:** 39 solidifi + 30 smartbugs_curated + 2 manual seeds = 71 positives

## Contradictions and Ambiguities

1. **EB TP count discrepancy:** Crosswalk validation md header tally and conclusion say 4 TP / 71 FP, but per-contract table lists exactly 3 TP / 72 FP. Off by exactly 1. The per-contract table is the raw data; the tally is likely a transcription error.

2. **Sampling frame distinction:** The 150-contract crosswalk sample (seed=42, raw DIVE folder) and the 175-contract Slither-agreed sample (seed=7, agreed_shas) are DISJOINT sampling frames. They are independent, not sequential/superset. However, the Slither-agreed sample conclusions (4.0% EB, 2.7% RE) are drawn from the same DIVE distribution, not a held-out population.

3. **Option B vs current state:** The DEEP-recommended option B (drop DIVE folder labels) has NOT been implemented. The current active export still includes all 22,493 contracts with DIVE folder-mapped labels. The actual v3 export has 16,638 ExternalBug positives (from DIVE Access Control/ mapping), vs recommended 60.

## Duplicate/Overlap Detection

| Evidence set pair | Overlap detected | Notes |
|---|---|---|
| R4-PREV-DIVE-EBRE vs R4-PREV-DIVE-SLITHER | NONE (disjoint frames) | Different seeds (42 vs 7), different sampling frames (raw folder vs Slither-agreed) |
| R4-PREV-DIVE-SLITHER vs R4-PREV-DIVE-ADERYN1 | FULL (175 same contracts) | Aderyn ran on the SAME 175 contracts as the Slither-agreed review |
| R4-PREV-DIVE-CORROB vs R4-PREV-DIVE-SLITHER | PARTIAL (100/6804 EB + 75/8258 RE sampled) | The 175-contract review is a MANUAL sample drawn from the automated Slither-agreed set |
| R4-PREV-DIVE-CORROB vs R4-PREV-DIVE-LABELS | FULL (all 22,073 labels) | Slither corroboration checks DIVE labels against Slither |
| R4-PREV-DIVE-ADERYN1 vs R4-PREV-DIVE-ADERYN2 | NONE (different frames) | Per-contract vs disagreed-subset |
| R4-PREV-DIVE-EBRE vs R4-PREV-DATA-AUDIT | PARTIAL | Data audit references DIVE in labeling/verification stages |

## Dependency Groups

- **Group D-A:** R4-PREV-DIVE-EBRE, R4-PREV-DIVE-SLITHER, R4-PREV-DIVE-CORROB, R4-PREV-DIVE-ADERYN1, R4-PREV-DIVE-ADERYN2 — all investigate DIVE EB/RE label quality. They are NOT independent: each builds on or samples from the prior.
- **Group D-B:** R4-PREV-DIVE-LABELS, R4-PREV-CROSSWALKS — source labels + mapping. Independent from D-A.
- **Group D-C:** R4-PREV-DIVE-SLITHER-CACHE, R4-PREV-DIVE-ADERYN-CACHE — tool caches. Independent evidence, tool-correlated.

## Unavailable Artifacts

| Item | Impact |
|---|---|
| DIVE non-EB/RE class reviews (Arithmetic, DoS, Time Manip, etc.) | No per-class precision evidence for 6 of 8 DIVE classes |
| DIVE second independent reviewer sign-off | Both review mds single-author; no inter-rater reliability |
| DIVE /tmp sample lists (temp files, never committed) | Sample membership fully recoverable from md tables + seeds 42/7 |

## Recommended Phase 2 Actions

- Register the EB TP count discrepancy as a gap if precise count matters for DROP/KEEP decisions
- Import DIVE crosswalk review TP/FP verdicts into evidence ledger for ExternalBug and Reentrancy
- Note: Option B (drop DIVE folder labels) would reduce 16,638 EB → 60 and 11,399 RE → 71, a >99% reduction
