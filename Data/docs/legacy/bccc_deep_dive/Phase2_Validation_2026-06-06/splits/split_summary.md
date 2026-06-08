# WS-G: Stratified Split — Summary

**Date:** 2026-06-06
**Version:** v1
**Method:** 2-stage stratified (has_vuln, primary_vuln_class)

## Split Sizes

| Split | n | % |
|---|---:|---:|
| Train | 46,581 | 70.0% |
| Val | 9,982 | 15.0% |
| Test | 9,982 | 15.0% |
| **Training pool total** | **66,545** | **100.0%** |
| Held out (review_pending) | 766 | — |

## Per-Class Prevalence (post-split, training pool only)

| Class | Train | Val | Test | Total | Distribution |
|---|---:|---:|---:|---:|---|
| `Class01:ExternalBug` | 2,542 | 527 | 534 | 3,603 | 71/15/15 |
| `Class02:GasException` | 4,822 | 1,029 | 1,023 | 6,874 | 70/15/15 |
| `Class03:MishandledException` | 3,625 | 748 | 776 | 5,149 | 70/15/15 |
| `Class04:Timestamp` | 1,862 | 399 | 399 | 2,660 | 70/15/15 |
| `Class06:UnusedReturn` | 2,253 | 478 | 498 | 3,229 | 70/15/15 |
| `Class08:CallToUnknown` | 7,300 | 1,546 | 1,582 | 10,428 | 70/15/15 |
| `Class09:DenialOfService` | 8,683 | 1,845 | 1,859 | 12,387 | 70/15/15 |
| `Class10:IntegerUO` | 11,693 | 2,524 | 2,481 | 16,698 | 70/15/15 |
| `Class11:Reentrancy` | 11,913 | 2,501 | 2,573 | 16,987 | 70/15/15 |
| `Class12:NonVulnerable` | 18,303 | 3,923 | 3,922 | 26,148 | 70/15/15 |


## Stratification Method

For each contract, derive a stratify key:
- `V_<primary_class>` if has ≥1 vuln class
- `N_none` if pure NV
- `RARE` if the derived key has <2 samples (bucket them together)

`primary_class` = rarest positive vuln class (e.g., Timestamp at 3.97% is rarer than Reentrancy at 26.29%).

This is a **simple approximation** of iterative stratification (Sechidis et al. 2011).
For best results, install `iterative-stratification` (network currently slow/blocked) and re-run with `MultilabelStratifiedKFold`.

## Decisions Applied

- **D-F1:** Dropped 1,122 contracts (only had Class05/Class07)
- **D-B2:** Held out 766 NV+vuln contradictions for manual review (NOT in any split)
- **NV class:** Treated as Class12 (binary); pure-NV contracts are stratified by `N_none`

## Files

- `train.csv` (46,581 contracts)
- `val.csv` (9,982 contracts)
- `test.csv` (9,982 contracts)
- `bccc_splits_v1.json` (metadata)
- `split_summary.md` (this file)

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/g_stratified_split.py
```
