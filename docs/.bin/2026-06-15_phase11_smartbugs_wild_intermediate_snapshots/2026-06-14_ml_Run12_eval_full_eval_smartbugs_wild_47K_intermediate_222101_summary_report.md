# Run 12 SmartBugs Wild Full Evaluation — 2026-06-14 22:21

## Overview

- **Total contracts processed:** 10
- **Successful:** 7 (70.0%)
- **Errors:** 3 (30.0%)
- **Total elapsed:** 0m04s
- **Throughput:** 2.80 predictions/sec

## Speed (excluding warmup)

| Percentile | Time (ms) |
|---|---|
| mean | 357.7 |
| p50 | 213.6 |
| p95 | 776.4 |
| p99 | 820.7 |

## Per-Class Distribution (top class)

| Class | Count | Pct | Mean conf |
|---|---|---|---|
| Reentrancy | 3 | 42.9% | 0.814 |
| ExternalBug | 2 | 28.6% | 0.801 |
| DenialOfService | 1 | 14.3% | 0.997 |
| UnusedReturn | 1 | 14.3% | 0.796 |


## Trigger Stats (>= 0.5 tuned threshold)

- Contracts with >=1 trigger: 7 / 7 (100.0%)
- Mean triggers per contract: 1.57
- p50: 2, p95: 2, max: 2

## Error Samples (first 10)

- `0x00000000000fe8503db73c68f1a1874eb9d86883`: Graph extraction infrastructure failure for 'sentinel_prep_58yd9uyc.sol': Slither failed to parse 's...
- `0x00000000002b13cccec913420a21e4d11b2dcd3c`: Graph extraction infrastructure failure for 'sentinel_prep_czgu9f68.sol': Slither failed to parse 's...
- `0x000000000063b99b8036c31e91c64fc89bff9ca7`: Graph extraction infrastructure failure for 'sentinel_prep_4oeo5ema.sol': Slither failed to parse 's...


## Artifacts

- State: `/home/motafeq/projects/sentinel/ml/data/smartbugs_wild_eval_state.json`
- Log: `/home/motafeq/projects/sentinel/ml/logs/smartbugs_wild_eval_20260614_222057.log`
- Per-contract: `ml/reports/Run12_smartbugs_wild_FULL_<date>_per_contract.json`
- This summary: `ml/reports/Run12_smartbugs_wild_FULL_<date>_<time>_summary.md`

Generated: 2026-06-14T22:21:01.938518
