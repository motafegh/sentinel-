# SENTINEL DATA Module

`data_module/` owns Solidity acquisition, preprocessing, graph/token representation, historical label/export compatibility, and the repaired DATA/R4 implementation path.

> **Current authority:** for new DATA/ML work, use the current R4 evidence/policy/role/representation decisions under [`docs/plan/ml-R4`](../docs/plan/ml-R4) and the canonical [DATA handbook](../docs/handbook/03_data_pipeline.md). Historical parser/crosswalk/merge/export code and historical representation lineages remain reproducibility/compatibility material; they are not authoritative new label or training truth.

## Current state

Historical R4 **G0–G7 remain PASSED and immutable**. **Phase 8 is IN_PROGRESS.** The current DATA authority is layered:

- R4-D-008 accepts repaired-v2 physical DATA: **22,540 contracts**, **225,400 contract×class rows**, and **67,620 graph/token/sidecar files** with binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- R4-D-009 accepts logical V3 grouping/roles: **22,394 groups**, maximum group size **7**, **146 normalized-code edges**, and zero address-authority edges;
- accepted semantic policy remains `data-vnext-policy-v1`;
- historical zero / absence / unsupported / dropped state is **not** a confirmed negative;
- confirmed negatives remain **zero**; candidate #2 has primary-review support only and still requires genuinely independent agreement;
- GasException and UnusedReturn supervision remain disabled pending evidence;
- threshold-fit, calibration-fit, and untouched-acceptance roles remain intentionally unsupported/empty;
- R4-D-010 preserves graph schema v9 for historical reproduction but makes v9 ineligible for a new full training run;
- R4-D-011 accepts the exact **V10 V2.6** 22,540-identity physical representation lineage under extractor `v2.6-r4-call-semantics-deterministic-cfg-mutators`, digest `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`;
- R4-D-012 promotes `target_aware_guarded_v1` only for a **fresh versioned token/representation candidate**. That successor lineage is not yet separately physically accepted;
- full training remains unauthorized.

## Historical compatibility vs current R4 authority

### Historical v1 / G7 compatibility

The existing ingestion/labeling/merger/split/export stack and binary `class_0..class_9` artifacts are preserved so Run12 and historical findings can be reproduced/audited. Historical G7 role authority was `r4-vnext-roles-v1`; it remains an immutable historical identifier, not the current logical V3 role authority.

Do **not** infer current truth from legacy behavior such as:

- source non-target cells becoming `0`;
- folder absence becoming `0`;
- unsupported classes becoming `0`;
- historical `NonVulnerable` synthesis;
- positive-precedence merging over collapsed zeros;
- old train/val/test role names.

The historical G7 publication and graph-schema-v9 lineage remain immutable evidence, not the current new-full-training physical authority.

### Repaired semantic layer

The repaired semantic unit is `contract_id × class_index` and carries explicit:

- source-native claim/provenance;
- canonical outcome state;
- nullable target;
- training signal/strength (`STRONG`, `WEAK`, `NONE`);
- loss/metric eligibility;
- evidence/limitations;
- leakage-safe dataset role.

A numeric target `0` requires class-specific `CONFIRMED_NEGATIVE` evidence. Policy v1 currently authorizes no blanket negative source.

### Current physical representation boundary

Keep these lineages separate:

1. **v9** — immutable historical/reproducibility evidence. R4-D-010 makes it ineligible for a new full training run.
2. **V10 V2.6 / R4-D-011** — accepted exact 22,540-identity physical graph/token/sidecar root. It preserves accepted-v9 token bytes and corrects graph semantics; physical acceptance alone does not authorize training.
3. **R4-D-012 successor** — a new versioned candidate using `target_aware_guarded_v1`. It must preserve population/graph/runtime authority while versioning expected token-selector/token-payload changes, then receive a separate binding and physical acceptance decision.

The model-facing token tensor contract remains `[4,512]`; model architecture remains frozen while these evidence and evaluation gates are resolved.

## Historical lifecycle code and DVC boundary

The package still contains the acquisition/representation lifecycle:

```text
ingest → preprocess → represent → label → verify → split → register → analyze → export
freshness is a separate lifecycle check
```

`data_module/dvc.yaml` belongs to this module-local historical lifecycle, and `data_module/.dvc/` is the DVC root for those operations. Run DVC from `data_module/` when intentionally using that pipeline.

This DVC lifecycle must not be confused with current R4 physical authority. A fresh clone does **not** automatically contain or reconstruct the accepted R4-D-011 physical V10 V2.6 root, the historical Run12 checkpoint, or every proving/runtime artifact. Current R4 artifact identity and acceptance come from the tracked R4 evidence/decision chain, not from the existence of `dvc.yaml`.

The repository root also has a separate local/repository-level DVC context. Its public config intentionally contains no machine-specific remote; private/local remote paths belong in ignored `config.local` files rather than tracked configuration.

`sentinel_data.cli::_run_label` remains an incomplete historical CLI seam. Do not describe `sentinel-data run` or `dvc repro` as a complete current-R4 builder.

## Important current areas

```text
data_module/sentinel_data/
  ingestion/                 acquisition mechanics
  preprocessing/             normalization/compiler/dedup mechanics
  representation/            graph/token representation implementation
  labeling/                  historical label parsers/crosswalk/merger
  verification/              historical/source verification utilities
  splitting/                 historical split utilities
  export/                    historical v1 export compatibility
  vnext/                     DATA vNext/R4 semantic implementation

data_module/dvc.yaml        historical module-local lifecycle orchestration

docs/plan/ml-R4/
  ledger/                    evidence ledger
  specs/                     accepted DATA vNext policy
  schemas/                   contract×class semantic schema
  manifests/                 role/support/acceptance manifests
  adrs/                      semantic/physical decisions including R4-D-008 through R4-D-012
  evidence/                  durable bounded acceptance/research evidence
```

## Verification

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

Those checks validate only their declared scope. Current physical/training authority must be read from the R4 status matrix and accepted decision/evidence chain; do not infer authorization from the existence of a representation, DVC stage, or passing historical gate.

## Do not silently weaken these invariants

- Historical zero is not negative truth.
- Source absence/tool silence is not negative truth.
- Disabled classes do not receive zero targets.
- Weak evidence does not become strong or metric-grade evidence.
- Leakage groups do not cross incompatible roles.
- Historical v1/v9 artifacts are immutable.
- Do not patch the R4-D-011 root in place.
- R4-D-012 requires a fresh versioned candidate and separate physical acceptance.
- DATA/R4 readers must never silently fall back to historical binary semantics.
- DVC availability does not equal current R4 artifact authority.
- Physical validity does not equal training authorization or model quality.

For the full current explanation, see [DATA pipeline](../docs/handbook/03_data_pipeline.md), [DATA artifacts / ML seam](../docs/handbook/04_data_artifacts.md), [operations](../docs/handbook/14_operations.md), and [current status](../docs/handbook/16_current_status.md).
