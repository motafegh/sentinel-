# Run 12 SmartBugs Wild Full Evaluation — 2026-06-14 22:22

## Overview

- **Total contracts processed:** 200
- **Successful:** 163 (81.5%)
- **Errors:** 37 (18.5%)
- **Total elapsed:** 1m07s
- **Throughput:** 2.53 predictions/sec

## Speed (excluding warmup)

| Percentile | Time (ms) |
|---|---|
| mean | 395.8 |
| p50 | 233.3 |
| p95 | 1061.7 |
| p99 | 2509.1 |

## Per-Class Distribution (top class)

| Class | Count | Pct | Mean conf |
|---|---|---|---|
| ExternalBug | 59 | 36.2% | 0.821 |
| Reentrancy | 33 | 20.2% | 0.794 |
| Timestamp | 33 | 20.2% | 0.974 |
| UnusedReturn | 16 | 9.8% | 0.955 |
| DenialOfService | 11 | 6.7% | 0.926 |
| IntegerUO | 9 | 5.5% | 0.692 |
| TransactionOrderDependence | 2 | 1.2% | 0.988 |


## Trigger Stats (>= 0.5 tuned threshold)

- Contracts with >=1 trigger: 159 / 163 (97.5%)
- Mean triggers per contract: 2.30
- p50: 2, p95: 5, max: 5

## Error Samples (first 10)

- `0x00000000000fe8503db73c68f1a1874eb9d86883`: Graph extraction infrastructure failure for 'sentinel_prep_2ae32vgj.sol': Slither failed to parse 's...
- `0x00000000002b13cccec913420a21e4d11b2dcd3c`: Graph extraction infrastructure failure for 'sentinel_prep_182xbr8g.sol': Slither failed to parse 's...
- `0x000000000063b99b8036c31e91c64fc89bff9ca7`: Graph extraction infrastructure failure for 'sentinel_prep_8pld0c1c.sol': Slither failed to parse 's...
- `0x00000000e82eb0431756271f0d00cfb143685e7b`: Graph extraction infrastructure failure for 'sentinel_prep_y2qegpi1.sol': Slither failed to parse 's...
- `0x00000007b0390fc9ca72f534366f5c02d5af5334`: Graph extraction infrastructure failure for 'sentinel_prep_xq7cw4vg.sol': Slither failed to parse 's...
- `0x000000085824f23a070c2474442ed014c0e46b58`: Graph extraction infrastructure failure for 'sentinel_prep_g360_xav.sol': Slither failed to parse 's...
- `0x0000000a9e27410f13dd4818488bf1e706c9a2fe`: Graph extraction infrastructure failure for 'sentinel_prep_fsal3b24.sol': Slither failed to parse 's...
- `0x000003ed2eb44cded8ade31c01dda60da466b2d1`: Graph extraction infrastructure failure for 'sentinel_prep_npxdybjq.sol': Slither failed to parse 's...
- `0x000983ba1a675327f0940b56c2d49cd9c042dfbf`: Graph extraction infrastructure failure for 'sentinel_prep_ilcq1u6h.sol': Slither failed to parse 's...
- `0x001a589dda0d6be37632925eaf1256986b2c6ad0`: Graph extraction infrastructure failure for 'sentinel_prep_q14gpeye.sol': Slither failed to parse 's...


## Artifacts

- State: `/home/motafeq/projects/sentinel/ml/data/smartbugs_wild_eval_state.json`
- Log: `/home/motafeq/projects/sentinel/ml/logs/smartbugs_wild_eval_20260614_222124.log`
- Per-contract: `ml/reports/Run12_smartbugs_wild_FULL_<date>_per_contract.json`
- This summary: `ml/reports/Run12_smartbugs_wild_FULL_<date>_<time>_summary.md`

Generated: 2026-06-14T22:22:31.908904
