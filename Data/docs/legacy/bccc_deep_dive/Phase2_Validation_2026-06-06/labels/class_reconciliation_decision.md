# WS-F: Class Reconciliation — Decision Log

**Date:** 2026-06-06
**Status:** Complete (D-F1 applied; D-B2 marked for review)

## Decisions Applied

### D-F1: Drop 2 BCCC classes (WeakAccessMod + TransactionOrderDependence)

**Chosen option:** A — Drop 2 BCCC classes
- WeakAccessMod (Class07): 1,918 contracts (2.80%)
- TransactionOrderDependence (Class05): 3,562 contracts (5.21%)

**Result:**
- **1,122 contracts dropped** (had ONLY Class05/Class07, no other vulns, and not pure-NV).
- Contracts with Class05/Class07 AND any other non-dropped vuln class: 4,358 (kept, with Class05/Class07 stripped).
- Surviving contracts: 67,311

### D-B2: NV+vuln contradictions → manual review

**Chosen option:** D — Manual review 766
- 766 contracts have Class12=1 AND ≥1 vuln class.
- These are flagged `review_pending=1` in `contracts_filtered.csv`.
- They will be **excluded from initial training** (WS-G) but can be re-added after review.
- See `review_pending_ids.csv` for the full list.

## Final 10-Class Schema (SENTINEL v9-aligned)

| Order | Class |
|---:|---|
| 1 | `Class01:ExternalBug` |
| 2 | `Class02:GasException` |
| 3 | `Class03:MishandledException` |
| 4 | `Class04:Timestamp` |
| 5 | `Class06:UnusedReturn` |
| 6 | `Class08:CallToUnknown` |
| 7 | `Class09:DenialOfService` |
| 8 | `Class10:IntegerUO` |
| 9 | `Class11:Reentrancy` |
| 10 | `Class12:NonVulnerable` |


This matches SENTINEL's locked v9 schema (per ADR-0001).

## Final Per-Class Prevalence (n=67,311)

| Class | n | % |
|---|---:|---:|
| `Class01:ExternalBug` | 3,604 | 5.35% |
| `Class02:GasException` | 6,879 | 10.22% |
| `Class03:MishandledException` | 5,154 | 7.66% |
| `Class04:Timestamp` | 2,674 | 3.97% |
| `Class06:UnusedReturn` | 3,229 | 4.80% |
| `Class08:CallToUnknown` | 11,131 | 16.54% |
| `Class09:DenialOfService` | 12,394 | 18.41% |
| `Class10:IntegerUO` | 16,740 | 24.87% |
| `Class11:Reentrancy` | 17,698 | 26.29% |
| `Class12:NonVulnerable` | 26,914 | 39.98% |


## Per-Contract n_pos Distribution (filtered)

| n_pos | contracts | % |
|---:|---:|---:|
| 1 | 40,787 | 60.59% |
| 2 | 18,499 | 27.48% |
| 3 | 5,073 | 7.54% |
| 4 | 1,740 | 2.59% |
| 5 | 879 | 1.31% |
| 6 | 278 | 0.41% |
| 7 | 50 | 0.07% |
| 8 | 5 | 0.01% |


## Summary Stats

- **Surviving contracts:** 67,311
- **Pure NV contracts (no vuln):** 26,148 (38.8%)
- **Contracts with ≥1 vuln:** 41,163 (61.2%)
- **Review-pending (NV+vuln contradictions):** 766 (1.14%)

## What This Means for Training

- **Vuln training set:** 40,397 contracts (excluding review-pending)
- **Clean NV test set:** 26,148 contracts
- **Review-pending set:** 766 contracts (held out, manual review needed)

## Files

- `contracts_filtered.csv` — 67,311 contracts × 12 cols (10 class labels + id + review_pending)
- `dropped_contracts.csv` — 1,122 contracts dropped per D-F1
- `review_pending_ids.csv` — 766 contracts flagged for manual review
- `class_reconciliation_decision.md` — this file

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/f_class_reconciliation.py
```
