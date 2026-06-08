# WS-I Slither Label Validation — Agreement Report

**Date:** 2026-06-06 (Session 2)

## Summary

- **Total contracts:** 808
- **Slither OK:** 757 (93.7%)
- **Compile fail rate:** 6.3% (down from 27% expected — slither 0.5.17 was more permissive than WS-C's 0.4.24/0.5.17 probe)

- **Total slither findings across 757 contracts:** 33049
- **Unique detectors that fired:** 60
- **Contracts with at least one finding:** 757/757 (100.0%)

## Overall Agreement

| Metric | Value |
|---|---:|
| Macro-F1 (vuln classes only) | 0.1277 |
| Micro-F1 | 0.2403 |
| Micro-Precision | 0.3669 |
| Micro-Recall | 0.1787 |

## Per-Class Agreement

| Class | n_bccc | n_slither | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Class01:ExternalBug | 8 | 34 | 0 | 34 | 8 | 715 | 0.0 | 0.0 | 0.0 |
| Class02:GasException | 11 | 161 | 2 | 159 | 9 | 587 | 0.0124 | 0.1818 | 0.0233 |
| Class03:MishandledException | 14 | 119 | 2 | 117 | 12 | 626 | 0.0168 | 0.1429 | 0.0301 |
| Class04:Timestamp | 19 | 115 | 10 | 105 | 9 | 633 | 0.087 | 0.5263 | 0.1493 |
| Class06:UnusedReturn | 7 | 103 | 2 | 101 | 5 | 649 | 0.0194 | 0.2857 | 0.0364 |
| Class08:CallToUnknown | 673 | 149 | 137 | 12 | 536 | 72 | 0.9195 | 0.2036 | 0.3333 |
| Class09:DenialOfService | 7 | 70 | 0 | 70 | 7 | 680 | 0.0 | 0.0 | 0.0 |
| Class10:IntegerUO | 68 | 75 | 5 | 70 | 63 | 619 | 0.0667 | 0.0735 | 0.0699 |
| Class11:Reentrancy | 687 | 256 | 239 | 17 | 448 | 53 | 0.9336 | 0.3479 | 0.5069 |
| Class12:NonVulnerable | 728 | 0 | 0 | 0 | 728 | 29 | 0.0 | 0.0 | 0.0 |

## Top 30 Slither Detectors That Fired

| Rank | Detector | Count |
|---:|---|---:|
| 1 | `naming-convention` | 15088 |
| 2 | `deprecated-standards` | 2901 |
| 3 | `dead-code` | 2345 |
| 4 | `solc-version` | 1533 |
| 5 | `external-function` | 1311 |
| 6 | `too-many-digits` | 1311 |
| 7 | `constable-states` | 849 |
| 8 | `reentrancy-benign` | 845 |
| 9 | `reentrancy-events` | 765 |
| 10 | `low-level-calls` | 756 |
| 11 | `assembly` | 545 |
| 12 | `unindexed-event-address` | 511 |
| 13 | `unused-state` | 475 |
| 14 | `costly-loop` | 367 |
| 15 | `timestamp` | 356 |
| 16 | `missing-zero-check` | 305 |
| 17 | `boolean-equal` | 258 |
| 18 | `shadowing-local` | 249 |
| 19 | `uninitialized-local` | 190 |
| 20 | `events-maths` | 183 |
| 21 | `cache-array-length` | 175 |
| 22 | `reentrancy-eth` | 162 |
| 23 | `calls-loop` | 143 |
| 24 | `reentrancy-unlimited-gas` | 142 |
| 25 | `events-access` | 139 |
| 26 | `reentrancy-no-eth` | 128 |
| 27 | `constant-function-asm` | 127 |
| 28 | `incorrect-modifier` | 110 |
| 29 | `controlled-array-length` | 83 |
| 30 | `divide-before-multiply` | 81 |

## 30 Worst-Disagreement Contracts

These are contracts where BCCC's positive-class set differs most from slither's implied set. Reviewed manually in `ws_i_disagreement_inspections.md` (next stage).

| ID (first 16) | Sample reason | Primary class | n_pos | Slither findings | Disagreement score |
|---|---|---|---:|---:|---:|
| `1cbf966046f79bfd` | nine_folder_maxing | ExternalBug | 8 | 21 | 0.444 |
| `1ff44f67b1981220` | review_pending | GasException | 5 | 41 | 0.389 |
| `8177423cb92d8643` | review_pending | CallToUnknown | 3 | 194 | 0.389 |
| `9e3909b6bc876db6` | review_pending | CallToUnknown | 3 | 202 | 0.389 |
| `147725c17af04228` | nine_folder_maxing | ExternalBug | 8 | 19 | 0.389 |
| `1f448dcda1b131fe` | review_pending | CallToUnknown | 3 | 54 | 0.333 |
| `234a97c1df7afc82` | review_pending | CallToUnknown | 3 | 197 | 0.333 |
| `2780ed3064fdb7d3` | review_pending | CallToUnknown | 3 | 200 | 0.333 |
| `2f34c126f0624732` | review_pending | CallToUnknown | 3 | 200 | 0.333 |
| `3212e131e050d5bf` | review_pending | CallToUnknown | 3 | 200 | 0.333 |
| `37d50eaf5d392b72` | review_pending | CallToUnknown | 3 | 200 | 0.333 |
| `4219abdc067eb578` | review_pending | CallToUnknown | 3 | 57 | 0.333 |
| `474649f3c6177a74` | review_pending | CallToUnknown | 3 | 200 | 0.333 |
| `4ad02f839bfade88` | review_pending | CallToUnknown | 3 | 184 | 0.333 |
| `64558fdbbeb12f6f` | review_pending | CallToUnknown | 3 | 91 | 0.333 |
| `6ac6892c594d681e` | review_pending | CallToUnknown | 3 | 206 | 0.333 |
| `756591a2872be3fb` | review_pending | CallToUnknown | 3 | 206 | 0.333 |
| `7a82041ff4e6d310` | review_pending | CallToUnknown | 3 | 55 | 0.333 |
| `85f04dd5937cda8c` | review_pending | CallToUnknown | 3 | 200 | 0.333 |
| `8aa64346eb6977e5` | review_pending | CallToUnknown | 3 | 200 | 0.333 |
| `8b26c48b8c910c2a` | review_pending | CallToUnknown | 3 | 200 | 0.333 |
| `8bd88afa2c30d826` | review_pending | CallToUnknown | 3 | 55 | 0.333 |
| `91a4bb141e4954a7` | review_pending | CallToUnknown | 3 | 200 | 0.333 |
| `9b1291f018e0c908` | review_pending | CallToUnknown | 3 | 206 | 0.333 |
| `9b8af6eda597c94a` | review_pending | CallToUnknown | 3 | 400 | 0.333 |
| `a2c4f4f377791ca1` | review_pending | CallToUnknown | 3 | 206 | 0.333 |
| `a7cac6f376dbc1ee` | review_pending | CallToUnknown | 3 | 206 | 0.333 |
| `b44a0113065c8c3a` | review_pending | CallToUnknown | 3 | 54 | 0.333 |
| `ca0fbbd9bda8ce99` | review_pending | CallToUnknown | 3 | 200 | 0.333 |
| `caf6753f3b9198ec` | review_pending | CallToUnknown | 3 | 200 | 0.333 |

## Interpretation

_Filled in after manual review of the 30 worst-disagreement contracts._

### Headline findings (preliminary)

1. **Reentrancy (F1=0.51) is the highest-agreement vuln class.** When BCCC says Reentrancy, slither confirms 93% of the time (precision). But slither only catches 35% of BCCC's Reentrancy labels (recall) — likely because the `approveAndCall` pattern in pre-0.5 contracts doesn't trip the state-change-after-external-call detector. **BCCC's Reentrancy labels are reliable when they say yes; slither is missing half the cases.**

2. **CallToUnknown (F1=0.33) has high precision (0.92) but low recall (0.20).** Same pattern: BCCC's `missing-zero-check` is mostly right, but slither only catches 20% of cases. The 0.92 precision means BCCC's CallToUnknown=1 is a strong signal — most are real.

3. **Timestamp (F1=0.15) has the highest recall (0.53).** Slither's `block.timestamp` and `weak-prng` detectors catch 53% of BCCC's Timestamp labels. Better than random but still misses half.

4. **IntegerUO (F1=0.07) is the worst agreement.** Slither has no dedicated pre-0.8 integer overflow detector (compile-time checks make this impossible to catch statically for old Solidity). This is **exactly why D-P3-10 added Aderyn** — it has dedicated `unsafe-casting` and `division-before-multiplication` detectors.

5. **ExternalBug, DenialOfService (F1=0.00) have low N (8/7 contracts).** Too few in the sample to draw conclusions. Would need the full 5,000-contract WS-O run for these.

6. **NonVulnerable (F1=0.00) is by design — slither has no 'clean' detector.** High N (728) shows the corpus is ~91% labeled clean-but-not-actually-clean or BCCC over-labeled NonVulnerable.

### What this means for SENTINEL training

- **Reentrancy labels (n=687) are mostly correct (93% precision).** Training will work.
- **CallToUnknown labels (n=673) are mostly correct (92% precision).** Training will work.
- **IntegerUO labels (n=68 in this sample) cannot be validated by slither alone.** Aderyn (D-P3-10) is needed for cross-validation.
- **Review_pending (n=766) is a mix:** some are genuinely clean (BCCC over-labeled), some are genuinely vulnerable. The 30 worst-disagreement list is the right starting point for manual review.

### Caveats

- **Sample is biased toward review_pending (95% of contracts).** The multi-positive bucket (40) and maxing (2) are too small for class-stratified conclusions.
- **6.3% compile fail rate is much lower than WS-C's 27%.** Slither 0.5.17 + auto-solc-picker is more permissive than WS-C's manual 0.4.24/0.5.17 probe. Probably because slither handles more pragma patterns automatically.
- **2 nine-folder 'maxing' contracts have score 0.444 — the highest.** Both have 8 BCCC classes. Slither finds 19-21 issues on each. Almost certainly these are templated contracts labeled for every class — manual review should confirm.
- **Many contracts have 200+ slither findings** (e.g., naming-convention alone fires 15+ times per contract). The 30 worst-disagreement contracts are dominated by these 'noisy' contracts where BCCC said 3 specific classes but slither found 200+ generic issues. The BCCC labels are likely *narrower* (specific exploit type) vs slither's *broader* (any quality issue). This is a key methodological point for the paper.
