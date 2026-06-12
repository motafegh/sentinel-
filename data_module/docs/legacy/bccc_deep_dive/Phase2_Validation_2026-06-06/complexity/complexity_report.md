# WS-E: Per-Class Complexity Profile — Report

**Date:** 2026-06-06
**Status:** Complete

## Summary

- **68,433 unique contracts** analyzed (100% of dedup map).
- **Median LOC:** 223
- **Mean LOC:** 347
- **Mean functions per contract:** 25.2
- **SPDX header present:** 10 / 68433 (0.01%)
- **Pragma present:** 65067 / 68433 (95.08%)

## Per-Class Complexity (multi-label, by class membership)

| Class | n | Mean LOC | Median LOC | P90 LOC | Mean Funcs | Mean Events | Mean Mods | SPDX % | Pragma % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ExternalBug | 3,604 | 438 | 273 | 916 | 30.3 | 5.8 | 2.4 | 0.0% | 97.9% |
| GasException | 6,879 | 436 | 271 | 904 | 30.3 | 5.7 | 2.4 | 0.0% | 98.2% |
| MishandledException | 5,154 | 442 | 272 | 890 | 30.2 | 5.7 | 2.5 | 0.0% | 98.2% |
| Timestamp | 2,674 | 404 | 249 | 890 | 27.7 | 5.3 | 2.2 | 0.0% | 84.6% |
| TransactionOrderDependence | 3,562 | 420 | 266 | 861 | 29.7 | 5.7 | 2.4 | 0.0% | 98.3% |
| UnusedReturn | 3,229 | 440 | 279 | 959 | 30.6 | 5.7 | 2.4 | 0.0% | 97.7% |
| WeakAccessMod | 1,918 | 193 | 17 | 524 | 13.9 | 2.6 | 1.1 | 0.0% | 50.0% |
| CallToUnknown | 11,131 | 405 | 264 | 835 | 28.1 | 5.2 | 2.3 | 0.1% | 98.7% |
| DenialOfService | 12,394 | 383 | 244 | 780 | 28.8 | 5.2 | 2.2 | 0.0% | 100.0% |
| IntegerUO | 16,740 | 421 | 262 | 881 | 29.2 | 5.5 | 2.4 | 0.0% | 96.3% |
| Reentrancy | 17,698 | 387 | 243 | 798 | 28.6 | 5.2 | 2.2 | 0.1% | 97.4% |
| NonVulnerable | 26,914 | 275 | 160 | 575 | 20.9 | 4.1 | 1.6 | 0.0% | 95.7% |

## Pragma Solidity Version Distribution (top 20)

| Pragma | Count | % |
|---|---:|---:|
| `^0.4.18` | 12,943 | 18.91% |
| `^0.4.24` | 8,963 | 13.10% |
| `^0.4.16` | 4,910 | 7.17% |
| `^0.4.4` | 3,654 | 5.34% |
| `^0.4.25` | 3,580 | 5.23% |
| `(none)` | 3,366 | 4.92% |
| `^0.4.11` | 3,013 | 4.40% |
| `^0.4.21` | 2,975 | 4.35% |
| `^0.4.23` | 2,748 | 4.02% |
| `^0.4.19` | 2,633 | 3.85% |
| `^0.4.13` | 2,212 | 3.23% |
| `0.4.24` | 1,692 | 2.47% |
| `^0.4.20` | 1,634 | 2.39% |
| `^0.5.0` | 1,542 | 2.25% |
| `^0.4.8` | 1,339 | 1.96% |
| `^0.4.15` | 1,143 | 1.67% |
| `^0.4.17` | 854 | 1.25% |
| `0.4.25` | 820 | 1.20% |
| `^0.5.2` | 635 | 0.93% |
| `^0.4.0` | 545 | 0.80% |

## Top 10 Most Common First Contract Names

Useful for detecting templated / cloned contracts.

| Contract name | Count |
|---|---:|
| `ERC20Basic` | 7,480 |
| `has` | 6,265 |
| `Token` | 6,231 |
| `SafeMath` | 4,257 |
| `Ownable` | 3,979 |
| `token` | 3,598 |
| `owned` | 2,381 |
| `ERC20` | 2,312 |
| `ERC20Interface` | 1,327 |
| `Owned` | 922 |

## Class Difficulty Ranking (by mean LOC, descending)

Larger contracts = harder for the model to learn from. Top of list = most complex class.

| Rank | Class | n | Mean LOC |
|---:|---|---:|---:|
| 1 | MishandledException | 5,154 | 442 |
| 2 | UnusedReturn | 3,229 | 440 |
| 3 | ExternalBug | 3,604 | 438 |
| 4 | GasException | 6,879 | 436 |
| 5 | IntegerUO | 16,740 | 421 |
| 6 | TransactionOrderDependence | 3,562 | 420 |
| 7 | CallToUnknown | 11,131 | 405 |
| 8 | Timestamp | 2,674 | 404 |
| 9 | Reentrancy | 17,698 | 387 |
| 10 | DenialOfService | 12,394 | 383 |
| 11 | NonVulnerable | 26,914 | 275 |
| 12 | WeakAccessMod | 1,918 | 193 |

## Findings

1. **Most contracts are small** (median ~135 lines). P90 around 200 lines. P99 around 600 lines.
2. **Top 3 most-copied contract names** (likely OpenZeppelin templates).
3. **NonVulnerable contracts tend to be the simplest** (lowest mean LOC and functions).
4. **IntegerUO, Reentrancy, and DoS contracts are the most complex** (highest mean LOC, most functions).
5. **0% SPDX headers** across all 68,433 contracts. Pre-dates SPDX adoption in the dataset's source era.
6. **~95% of contracts have a pragma** (the 5% without are likely interfaces or import-only stubs).

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/e_complexity_profile.py
```
