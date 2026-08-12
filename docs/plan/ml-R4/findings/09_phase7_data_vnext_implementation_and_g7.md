# 09 — Phase 7 DATA vNext Implementation and G7 Result

- **Phase:** R4 Phase 7 — DATA vNext Implementation
- **Gate:** G7 PASS
- **Dataset:** `sentinel-r4-vnext-v1`
- **Export schema:** `v2`
- **Graph schema:** `v9`
- **Implementation merge:** `81d9c547d3610e2cfb12a5927a7a78b5693430c2`
- **Local G7 evidence commit:** `5bd9c19eb46cd804b34ac0c2cd598767f10c7fad`

## Result

DATA vNext v2 is now implemented as an additive semantic overlay over the existing representation lineage. Historical v1 artifacts and the graph/token representation bytes were not rewritten.

Final semantic population:

| Measure | Result |
|---|---:|
| contracts | 22,493 |
| contract×class rows | 224,930 |
| represented / physically bound contracts | 21,657 |
| excluded incomplete-representation contracts | 836 |
| positive targets | 1,007 |
| negative targets | 0 |
| STRONG rows | 403 |
| WEAK rows | 604 |
| effective loss cells | 852 |
| outcome-metric cells | 118 |

## Local representation binding

The protected/local representation tree was verified without DVC fetching and without recording the physical local filesystem path.

- required contracts: 21,657
- checked contracts: 21,657
- expected files: 64,971
- checked files: 64,971
- missing files: 0
- mismatches: 0
- extractor: `v2.1-windowed-gcb`
- graph schema: `v9`
- physical path recorded: `false`
- binding digest: `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`

## Frozen limitations carried forward

G7 does **not** solve evidence that does not exist:

- no confirmed-negative training population exists in policy v1;
- GasException and UnusedReturn remain supervision-disabled;
- MODEL_SELECTION remains positive-only limited;
- THRESHOLD_FIT remains unsupported/empty;
- CALIBRATION_FIT remains unsupported/empty;
- UNTOUCHED_ACCEPTANCE remains unsupported/empty/frozen.

These limitations are inputs to Phase 8/9, not implementation defects to patch away.

## G7 assessment

**G7 PASS.** The versioned v2 bundle reproduces from frozen semantic inputs, validates independently, physically binds all required representations, preserves historical v1, and is suitable for the approved Phase-6 training roles.

Phase 8 may now adapt the existing training consumer to the exact v2 target/strength/mask/role contract and retrain the unchanged four-eye architecture.
