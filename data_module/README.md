# SENTINEL DATA Module

`data_module/` owns Solidity acquisition, preprocessing, graph/token representation, historical label/export compatibility, and the repaired DATA vNext implementation path.

> **Current authority:** for new DATA/ML work, use the R4 evidence/policy/role artifacts under [`docs/plan/ml-R4`](../docs/plan/ml-R4) and the canonical [DATA handbook](../docs/handbook/03_data_pipeline.md). Historical parser/crosswalk/merge/export code remains reproducibility/compatibility material; it is not authoritative new label truth.

## Current state

Stable `main` has passed R4 **G6**:

- historical population: **22,493 contracts**;
- evidence ledger: **224,930 contract×class rows**;
- physical representation coverage: **21,657 contracts**;
- incomplete-representation population: **836 contracts**, frozen `EXCLUDED` at leakage-group level;
- accepted semantic policy: `data-vnext-policy-v1`;
- frozen role policy: `r4-vnext-roles-v1`;
- historical zero / absence / unsupported / dropped state is **not** a confirmed negative;
- GasException and UnusedReturn supervision are disabled pending evidence;
- DIVE Front Running→TransactionOrderDependence is weak-positive only;
- threshold-fit, calibration-fit, and untouched-acceptance roles are intentionally unsupported/empty.

Phase 7 DATA vNext implementation is active on `r4/phase7-data-vnext-implementation`. Remote semantic generation/validation is green, but G7 still requires local binding to the existing 21,657 physical representation triplets before merge.

## Two DATA paths

### Historical v1 compatibility

The existing ingestion/labeling/merger/split/export stack and binary `class_0..class_9` artifacts are preserved so Run12 can be reproduced and historical findings can be audited.

Do **not** infer current truth from legacy behavior such as:

- source non-target cells becoming `0`;
- folder absence becoming `0`;
- unsupported classes becoming `0`;
- historical `NonVulnerable` synthesis;
- positive-precedence merging over collapsed zeros;
- old train/val/test role names.

### DATA vNext

The repaired semantic unit is `contract_id × class_index` and carries explicit:

- source-native claim/provenance;
- canonical outcome state;
- nullable target;
- training signal/strength (`STRONG`, `WEAK`, `NONE`);
- loss/metric eligibility;
- evidence/limitations;
- frozen leakage-safe dataset role.

A numeric target `0` requires class-specific `CONFIRMED_NEGATIVE` evidence. Policy v1 currently authorizes no blanket negative source.

## Physical representation contract

R4 does **not** change the current graph/token representation schema:

- graph schema: `v9`;
- node feature dimension: 12;
- node types: 14;
- edge types: 12;
- class order: locked ten-class order;
- token windows: up to `[4,512]`.

DATA vNext is therefore an additive semantic overlay over the existing representation population, not a synthetic replacement contract corpus.

## Historical lifecycle code

The historical package still contains the acquisition/representation lifecycle:

```text
ingest → preprocess → represent → label → verify → split → register → analyze → export
freshness is a separate lifecycle check
```

`sentinel_data.cli::_run_label` remains an incomplete historical CLI seam. Do not describe `sentinel-data run` as a complete DATA-vNext builder.

## Important current files

```text
data_module/sentinel_data/
  ingestion/                 historical/current acquisition mechanics
  preprocessing/             normalization/compiler/dedup mechanics
  representation/            v9 graph/token representation
  labeling/                  historical label parsers/crosswalk/merger
  verification/              historical/source verification utilities
  splitting/                 historical split utilities
  export/                    historical v1 export compatibility
  vnext/                     DATA vNext v2 implementation (Phase 7 branch until G7)

docs/plan/ml-R4/
  ledger/                    evidence ledger
  specs/                     accepted DATA vNext policy
  schemas/                   contract×class semantic schema
  manifests/                 frozen role/support/acceptance manifests
  adrs/                      semantic decisions
```

## Verification

On canonical `main` through G6:

```bash
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
python3 docs/handbook/tools/verify_handbook.py static
```

Phase-7 build/local commands belong to the dedicated Phase-7 branch until G7 is complete.

## Do not silently weaken these invariants

- Historical zero is not negative truth.
- Source absence/tool silence is not negative truth.
- Disabled classes do not receive zero targets.
- Weak evidence does not become strong or metric-grade evidence.
- Leakage groups do not cross incompatible roles.
- Historical v1 artifacts are immutable.
- DATA vNext v2 readers must never silently fall back to v1 binary semantics.

For the full current explanation, see [DATA pipeline](../docs/handbook/03_data_pipeline.md), [DATA artifacts / ML seam](../docs/handbook/04_data_artifacts.md), and [current status](../docs/handbook/16_current_status.md).
