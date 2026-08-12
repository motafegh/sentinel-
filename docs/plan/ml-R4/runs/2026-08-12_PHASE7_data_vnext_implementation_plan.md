# R4 Phase 7 — DATA vNext Implementation Plan

**Phase:** 7 — DATA vNext Implementation  
**Gate:** G7  
**Branch:** `r4/phase7-data-vnext-implementation`  
**Entry condition:** G6 PASS / `data-vnext-policy-v1` + `r4-vnext-roles-v1` frozen  
**Execution mode:** additive v2 implementation; historical v1 remains immutable

## 1. Objective

Materialize the accepted semantic policy and role freeze into deterministic versioned DATA vNext artifacts that later ML code can consume without guessing label semantics.

## 2. Architectural decision for this phase

R4 changes label/role semantics but does not change graph/token feature extraction. Re-copying tens of thousands of `.pt` representations into a new export would add storage, integrity, and synchronization risk without adding semantic value.

Therefore DATA vNext v2 is a **semantic overlay** that:

- publishes new label-state and ML target/mask artifacts;
- binds immutable representation requirements by contract/source;
- leaves graph/token `.pt` files in the existing representation tree;
- requires a final local physical-binding check before G7 can pass.

The overlay does not reuse historical `labels.parquet` or old train/val/test roles.

## 3. Frozen inputs

- Phase-3 ledger: `docs/plan/ml-R4/ledger/evidence_ledger_v1.parquet`
  - SHA-256 `3983cc2b3317515d546c784449b583ac9a7c23ac8da267ee10f5640857cd0ac7`
- Phase-5 policy: `docs/plan/ml-R4/specs/data_vnext_policy_v1.json`
  - status `ACCEPTED_G5`
- Phase-5 contract-class schema: `docs/plan/ml-R4/schemas/data_vnext_label_state_v1.schema.json`
- Phase-6 partition root: `docs/plan/ml-R4/manifests/p6_partition_manifest.json`
  - status `FROZEN_G6`
- Phase-6 contract roles: `docs/plan/ml-R4/manifests/p6_contract_role_manifest.jsonl`
- Phase-6 support/unsupported/acceptance manifests.

## 4. Versioned implementation surfaces

Create production package:

`data_module/sentinel_data/vnext/`

with:

- `policy.py` — class/source policy interpretation, no IO side effects;
- `builder.py` — frozen-input → v2 semantic overlay;
- `validator.py` — semantic/hash/population/role validation;
- `loader.py` — explicit v2 read-only interface; fail closed on v1/missing fields;
- `representations.py` — local physical representation binding verification;
- `__init__.py` — stable public surface.

Create export-format contract:

`data_module/sentinel_data/export/format_schema/v2.yaml`

Historical `v1.yaml` remains untouched.

## 5. v2 output layout

Target directory:

`data_module/data/exports/sentinel-r4-vnext-v1/`

Committed semantic outputs:

```text
manifest.json
source_registry.json
crosswalk_registry.json
evidence_snapshot.json
representation_requirements.json
label_states.parquet
ml_targets.parquet
validation_report.json
```

Local-only/final-binding output:

```text
representation_binding_report.json
```

The final binding report may be committed after local execution because it contains hashes/counts/paths only, not protected raw representation bytes.

## 6. Canonical label-state derivation

For each Phase-3 contract×class row:

### Strong positives

`historical_target == 1` and:

- source `solidifi`, enabled class → `CONFIRMED_POSITIVE`, target `1`, `STRONG`;
- source `smartbugs_curated`, approved enabled class except Timestamp → `CONFIRMED_POSITIVE`, target `1`, `STRONG`.

SmartBugs Timestamp is withheld because committed evidence cannot distinguish direct `time_manipulation` from superseded `bad_randomness→Timestamp`.

### Weak positive

- source `dive` + historical positive + TransactionOrderDependence → target `1`, `WEAK`, canonical outcome remains unresolved (`NOT_REVIEWED`/`UNKNOWN`), outcome metrics masked.

### Masked/unlabeled

Everything else, including:

- every historical zero;
- every unsupported/disabled class;
- all other DIVE historical positives;
- ambiguous SmartBugs Timestamp positives;
- dropped/out-of-taxonomy/mapped-NonVulnerable states;

receives nullable target, `training_strength=NONE`, no source-policy loss eligibility, and no outcome metric eligibility.

No v2 path emits target `0` under policy v1 because no confirmed-negative evidence exists.

## 7. Final effective role masks

The canonical semantic row preserves source-policy training eligibility.

The derived `ml_targets.parquet` additionally applies frozen contract role:

- strong target effective loss = true only in `TRAIN_STRONG`;
- weak target effective loss = true only in `TRAIN_WEAK`;
- `MODEL_SELECTION`/`INTERNAL_AUDIT` strong targets are not training-loss eligible but remain positive outcome-evaluation candidates within their stated positive-only limitations;
- `TRAIN_UNLABELED` and `EXCLUDED` have no supervised loss cells;
- threshold/calibration/untouched-acceptance remain absent.

## 8. Representation binding

Remote build can verify Phase-3/6 claims that all non-excluded groups have `representation_available=true`.

Final local check requires, for every non-excluded contract/source:

```text
data_module/data/representations/<source>/<contract_id>.pt
data_module/data/representations/<source>/<contract_id>.tokens.pt
data_module/data/representations/<source>/<contract_id>.rep.json
```

The sidecar must bind:

- `sha256 == contract_id`
- `source == expected source`
- `schema_version == v9`
- graph/token files must be non-empty.

G7 remains blocked if any required physical artifact is missing/mismatched.

## 9. Validation invariants

- exact ten-class order;
- exactly 224,930 canonical rows / 22,493 contracts;
- exact Phase-6 contract role coverage;
- exactly 836 excluded contracts;
- no target `0`;
- GasException/UnusedReturn no supervised targets;
- DIVE weak target only on TOD;
- SmartBugs Timestamp no strong target;
- effective training masks compatible with frozen role;
- unsupported threshold/calibration/acceptance roles remain absent;
- output hashes match manifest;
- repeated build from same inputs produces byte-stable Parquet/JSON outputs (except manifest generation metadata, which must also be deterministic or explicitly excluded from artifact identity);
- no historical artifact overwritten.

## 10. CI / local gates

### Remote CI

- unit tests over synthetic fixtures;
- build semantic overlay from committed ledger/policy/roles;
- validate output identities/counts;
- publish deterministic semantic outputs to the Phase-7 branch;
- regression gates G3–G6 remain green.

### Local final gate

One command will:

1. rebuild or verify the committed semantic overlay;
2. scan physical local representations;
3. emit `representation_binding_report.json`;
4. run final G7 validator.

No DVC command is required.

## 11. Non-goals

Do not in Phase 7:

- alter graph/token extractor behavior;
- regenerate representations automatically;
- change label/source policy or role allocation;
- modify ML dataset/collate/loss/trainer yet;
- choose weak-label numeric loss weight;
- invent threshold/calibration/acceptance sets;
- overwrite historical export directories.
