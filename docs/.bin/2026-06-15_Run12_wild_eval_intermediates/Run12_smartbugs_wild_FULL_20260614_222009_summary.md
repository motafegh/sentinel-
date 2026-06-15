# Run 12 SmartBugs Wild Full Evaluation — 2026-06-14 22:20

## Overview

- **Total contracts processed:** 5
- **Successful:** 3 (60.0%)
- **Errors:** 2 (40.0%)
- **Total elapsed:** 0m02s
- **Throughput:** 7.41 predictions/sec

## Speed (excluding warmup)

| Percentile | Time (ms) |
|---|---|
| mean | 134.9 |
| p50 | 139.2 |
| p95 | 150.0 |
| p99 | 150.9 |

## Per-Class Distribution (top class)

| Class | Count | Pct | Mean conf |
|---|---|---|---|
| ExternalBug | 1 | 33.3% | 0.734 |
| DenialOfService | 1 | 33.3% | 0.997 |
| Reentrancy | 1 | 33.3% | 0.893 |


## Trigger Stats (>= 0.5 tuned threshold)

- Contracts with >=1 trigger: 3 / 3 (100.0%)
- Mean triggers per contract: 1.33
- p50: 1, p95: 2, max: 2

## Error Samples (first 10)

- `0x00000000000fe8503db73c68f1a1874eb9d86883`: Graph extraction infrastructure failure for 'sentinel_prep_58yd9uyc.sol': Slither failed to parse 's...
- `0x00000000002b13cccec913420a21e4d11b2dcd3c`: Graph extraction infrastructure failure for 'sentinel_prep_czgu9f68.sol': Slither failed to parse 's...


## Artifacts

- State: `/home/motafeq/projects/sentinel/ml/data/smartbugs_wild_eval_state.json`
- Log: `/home/motafeq/projects/sentinel/ml/logs/smartbugs_wild_eval_20260614_222006.log`
- Per-contract: `ml/reports/Run12_smartbugs_wild_FULL_<date>_per_contract.json`
- This summary: `ml/reports/Run12_smartbugs_wild_FULL_<date>_<time>_summary.md`

Generated: 2026-06-14T22:20:09.152652
