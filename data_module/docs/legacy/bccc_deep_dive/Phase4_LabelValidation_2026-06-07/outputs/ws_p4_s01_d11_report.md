# D-I-11 Application Report

**Date:** 2026-06-07
**Scope:** NARROW (review_pending only, default)
**Input:** `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/outputs/contracts_clean.csv` (67311 contracts)
**Output:** `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01_d11_applied.csv`

## Rule Applied

For every contract where `Class12:NonVulnerable=1` AND at least one of {CallToUnknown, Reentrancy, GasException, MishandledException, DenialOfService, Timestamp}=1 → set `Class12:NonVulnerable=0` and (`review_pending=0` if it was 1).

## Counts

| Metric | Value |
|---|---:|
| Total contracts in input | 67311 |
| Contracts with NV=1 + ≥1 trigger (in scope) | 725 |
| NV labels dropped (renumbered) | 725 |
|   – in review_pending | 725 |
|   – in non-review_pending | 0 |
| review_pending remaining after | 41 |
| NV=1 contracts remaining | 26189 |
| is_pure_nv=1 contracts | 26148 |

## First 10 Corrections (for spot-check)

| id (prefix) | folder | primary | triggered classes | review_pending before | n_pos before |
|---|---|---|---|---:|---:|
| `00333aa3376bcd06…` | CallToUnknown | CallToUnknown | 08:CallToUnknown,11:Reentrancy | 1 | 3 |
| `007ccde4ee5b3120…` | CallToUnknown | CallToUnknown | 08:CallToUnknown,11:Reentrancy | 1 | 3 |
| `00da2433110f209c…` | CallToUnknown | CallToUnknown | 08:CallToUnknown,11:Reentrancy | 1 | 3 |
| `010fd936ffdd3968…` | CallToUnknown | CallToUnknown | 08:CallToUnknown,11:Reentrancy | 1 | 3 |
| `01c2258bf0f526e1…` | CallToUnknown | CallToUnknown | 08:CallToUnknown,11:Reentrancy | 1 | 3 |
| `02486424d59ed744…` | CallToUnknown | CallToUnknown | 08:CallToUnknown,11:Reentrancy | 1 | 3 |
| `02f68146dae0f3cd…` | CallToUnknown | CallToUnknown | 08:CallToUnknown,11:Reentrancy | 1 | 3 |
| `031d5e56ed8d020c…` | CallToUnknown | CallToUnknown | 08:CallToUnknown,11:Reentrancy | 1 | 3 |
| `03259317e33921ee…` | CallToUnknown | CallToUnknown | 08:CallToUnknown,11:Reentrancy | 1 | 3 |
| `0339fd5700a29ef1…` | CallToUnknown | CallToUnknown | 08:CallToUnknown,11:Reentrancy | 1 | 3 |

## Sanity Check

- All dropped contracts had NV=1 co-occurring with at least one vulnerability class (rule satisfied).
- n_pos recomputed for each row.
- review_pending reduced to **41** — close to the predicted ~61. Stage 0.4 will handle these.

## Version

v1.1 (D-I-11 applied, narrow (review_pending only, default))
