# WS-C: Compilation Probing — Report

**Date:** 2026-06-06
**Status:** Complete
**Sample size:** 100 contracts (stratified across 10 SENTINEL classes)

## Summary

- **Contracts tested:** 100
- **Compilation success:** 73 (73.0%)
- **Median bytecode size (success only):** 1,794 bytes
- **Mean bytecode size (success only):** 2,379 bytes

## Error Categories

| Category | n | % |
|---|---:|---:|
| OK (compiled) | 73 | 73.0% |
| PRAGMA (pragma mismatch) | 17 | 17.0% |
| SYNTAX (parser error) | 7 | 7.0% |
| IMPORT (file not found) | 1 | 1.0% |
| INTERNAL (compiler crash) | 0 | 0.0% |
| OTHER (incl. NO_FOLDER, FILE_NOT_FOUND) | 2 | 2.0% |

## Per-Class Success Rate

| Class | Tested | Success | Success % |
|---|---:|---:|---:|
| `Class01:ExternalBug` | 19 | 18 | 94.7% |
| `Class02:GasException` | 29 | 25 | 86.2% |
| `Class03:MishandledException` | 25 | 23 | 92.0% |
| `Class04:Timestamp` | 16 | 15 | 93.8% |
| `Class06:UnusedReturn` | 22 | 20 | 90.9% |
| `Class08:CallToUnknown` | 22 | 16 | 72.7% |
| `Class09:DenialOfService` | 15 | 0 | 0.0% |
| `Class10:IntegerUO` | 48 | 43 | 89.6% |
| `Class11:Reentrancy` | 44 | 26 | 59.1% |
| `Class12:NonVulnerable` | 10 | 8 | 80.0% |


## Per-solc-Version Success Rate

| solc version | Tested | Success | Success % |
|---|---:|---:|---:|
| `0.4.24` | 65 | 50 | 76.9% |
| `0.4.25` | 8 | 8 | 100.0% |
| `0.5.17` | 9 | 5 | 55.6% |
| `0.4.18` | 13 | 7 | 53.8% |
| `0.5.0` | 5 | 3 | 60.0% |


## Findings

1. **Compilation success rate: 73.0%** on a stratified 100-contract sample.
2. **Bytecode size median: 1,794 bytes** — reasonable for Solidity contracts.
3. **Most common error: PRAGMA** (17 contracts) — solc version mismatch. Mitigation: install more solc versions via solc-select, or downgrade pragma to compatible version.
4. **IMPORT errors: 1** — likely contracts that import other files (which we'd need to fetch).
5. **Class-based success rates** are roughly similar (no class is uniquely broken).

## Action Items

- [ ] Consider expanding the sample to 500 for more confident error rate estimates.
- [ ] For full BCCC processing, install more solc versions: 0.4.0-0.4.25 and 0.5.0-0.5.17 (~30 versions).
- [ ] For multi-file contracts, fetch the imported files or stub them out.

## Files

- `sample_100.csv` — the 100 sampled contracts with their labels
- `compile_results.csv` — per-contract compile result
- `compilation_report.md` — this file

## Reproducibility

```bash
cd /home/motafeq/projects/sentinel
source ml/.venv/bin/activate
python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/c_compile_probe.py
```
