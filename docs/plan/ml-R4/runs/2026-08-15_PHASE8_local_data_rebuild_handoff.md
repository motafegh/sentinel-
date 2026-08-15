# R4 Phase 8 — Local Repaired-DATA Rebuild and Acceptance Handoff

**Date:** 2026-08-15
**Canonical branch:** `main`
**Phase:** 8 — Existing Architecture Retraining
**Gate:** G8 remains OPEN
**Repository-repair base:** `a10fae041cc5f436b5607b6fd54fcabf63386059`
**State:** EXECUTED / SUPERSEDED FOR CURRENT AUTHORITY. The physical repaired-v2 rebuild/acceptance and bounded GPU smoke completed locally. Full 100-epoch training remains unauthorized. Current authority: `2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md`.

## Authority and purpose

This file supersedes the old Phase-8 pretraining launch instructions for the current execution boundary. It is the restart contract for work that cannot be completed from repository-only CI because the required raw Solidity corpora, historical solc binaries, generated representations/parquets, and GPU are local/Git-ignored.

Higher authority remains:

1. executable source/config/tests;
2. committed machine-readable R4 policy/artifacts;
3. `PLAN_STATUS_MATRIX.md`;
4. this handoff for the exact local execution sequence.

Do not use this file to reinterpret historical G7. `sentinel-r4-vnext-v1` and `r4-vnext-roles-v1` remain immutable historical evidence, but the 2026-08-14 live audit makes them unsuitable as the input to the full Phase-8 retrain.

## Repository repair that is already implemented

The repaired lineage is versioned instead of overwriting history:

- preprocessing artifact: `sentinel-preprocessed-r4-v2`;
- preprocessing metadata schema: `2`;
- provenance schema: `r4-provenance-v1`;
- evidence ledger: `evidence-ledger-r4-v2`;
- leakage grouping: `r4-leakage-groups-v2`;
- role partition: `r4-vnext-roles-v2`;
- DATA publication: `sentinel-r4-vnext-v2`;
- representation extractor: `v2.2-r4-repaired`;
- token coverage schema: `r4-token-coverage-v1`;
- graph schema remains `v9`;
- token tensor shape remains `[4, 512]`;
- model architecture remains `four_eye_v8` / `v8.1`.

Repository implementation includes:

1. lexical Solidity-safe comment normalization with repaired line-preserving mode;
2. exact/normalized-code duplicate identity separated from address-family evidence;
3. deterministic source-record provenance aggregation after worker staging;
4. version-aware solc flags, including no `--allow-paths` for unsupported 0.4.x compilers;
5. compilation of the exact normalized bytes that are promoted;
6. evidence-preserving file graph selection: unique inheritance leaves where possible and a disconnected union of all unrelated application leaves (or executable libraries for library-only files), with requested/actual post-extraction assertion;
7. frozen `[4,512]` token output with explicit pre-subsampling token/window coverage evidence;
8. deterministic leakage grouping after exact-content identity, normalized-code identity, explicit family provenance, and conservative same-source address-family evidence;
9. source-native repaired claim reconstruction preserving SmartBugs `time_manipulation` versus `bad_randomness`, SolidiFI injected-class authority, and DIVE weak-TOD semantics;
10. a separate dynamic repaired evidence-ledger/role/publication/binding lineage with no target zero and no threshold/calibration/acceptance fabrication;
11. a repaired-v2 ML dataset/run-binding seam for bounded GPU smoke without weakening historical v1 guards;
12. repository CI and synthetic regression tests covering repaired invariants while revalidating the frozen historical G6 artifacts;
13. full-source completeness manifests enforced before claims, grouping, or representations;
14. publication bound to the materialized evidence ledger and GPU acceptance bound to the exact publication/binding hashes.

## Permanent semantic constraints

These do not change during the local rebuild:

- unknown / unsupported / source absence is not negative;
- target `0` is forbidden without class-specific `CONFIRMED_NEGATIVE` evidence; current policy v1 has none;
- `GasException` and `UnusedReturn` remain supervision-disabled;
- DIVE Front Running→TransactionOrderDependence remains WEAK training-only;
- other current DIVE positive categories remain masked unless a later policy version changes them;
- SmartBugs direct `time_manipulation→Timestamp` is strong-positive source evidence;
- SmartBugs historical `bad_randomness→Timestamp` remains superseded/no-target;
- threshold fitting, calibration fitting, and untouched acceptance remain unsupported/empty;
- Run12 learned weights, optimizer/scheduler state, thresholds, calibration, and historical labels are not repaired-v2 truth;
- no full-training launch is allowed by this handoff.

## Local prerequisites

Run from the canonical repository root. The tracked worktree must be clean. Untracked personal audit/plan files do not alter source binding, but do not place repaired generated artifacts under tracked paths.

Required local material:

- `data_module/data/raw/dive/` + ingestion manifest;
- `data_module/data/raw/smartbugs_curated/` + ingestion manifest;
- `data_module/data/raw/solidifi/` + ingestion manifest;
- DIVE labels CSV at the configured/default location;
- historical solc-select artifacts needed by the raw pragma population;
- the existing ML environment with PyArrow, PyTorch, PyTorch Geometric, Transformers, PEFT and the accepted GraphCodeBERT cache;
- local GPU only for the final bounded smoke, not for the rebuild itself.

Portable acquisition status is recorded in `docs/plan/ml-R4/specs/p8_repaired_source_acquisition_v1.json`. Existing ingestion-manifest bytes remain authoritative where portable reacquisition metadata is incomplete.

## Exact execution order

### 0. Synchronize and freeze repository source

```bash
cd ~/projects/sentinel

git pull --ff-only origin main

git branch --show-current
git rev-parse HEAD
git status --short --untracked-files=no
```

Required state:

- branch is `main`;
- tracked status is empty;
- record the exact HEAD SHA in the local evidence notes;
- after generated-data execution begins, do not pull/switch/edit tracked source until the local rebuild evidence has been reviewed.

### 1. Verify the existing protected raw bytes

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_verify_repaired_raw_inputs.py
```

Expected historical manifest-record counts are evidence checks, not repaired output counts:

- DIVE: 22,330;
- SmartBugs Curated: 143;
- SolidiFI: 350;
- total: 22,823.

This command must pass SHA-256/size/path checks for every manifest record before repaired preprocessing starts. It proves agreement with the existing local ingestion manifests; it does **not** prove portable reacquisition.

### 2. Run environment/source prerequisites

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py prerequisites
```

Resolve any missing raw source, DIVE label CSV, historical solc binary, or Python runtime dependency before continuing.

### 3. Rebuild repaired preprocessing into fresh versioned roots

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py \
  preprocess --source dive --workers 8

PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py \
  preprocess --source smartbugs_curated --workers 4

PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py \
  preprocess --source solidifi --workers 4
```

Default output root:

`data_module/data/sentinel-preprocessed-r4-v2/`

The root must be fresh. Do not overwrite historical `data/preprocessed` or reuse a partially populated repaired directory.

For each source preserve:

- `repaired_preprocessing_manifest.json`;
- all `.meta.json` source/provenance companions;
- `dropped.csv`, if any.

A non-zero explicit drop count is **not automatically a failure**. Every drop must have an explicit reason and is adjudicated later by the repaired-lineage acceptance profiler. Address equality must never appear as a deletion reason.

### 4. Reconstruct source-native repaired claims

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py claims
```

Generated local artifact:

`data_module/data/r4-v2-build/source_claims.jsonl`

Must contain zero target-0 claims. Preserve source-record provenance and all cross-source exact-content claims.

### 5. Build final leakage-family grouping before role assignment

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py grouping
```

Generated local artifact:

`data_module/data/r4-v2-build/grouping.json`

Review at least:

- unique artifact identities;
- cross-source exact identities;
- normalized-code identity edges;
- explicit family edges;
- same-source address-family edges;
- final group count.

Grouping changes role/split atomicity only; it does not delete contracts or change label truth.

### 6. Rebuild strict repaired representations

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py \
  represent --source dive --workers 8

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py \
  represent --source smartbugs_curated --workers 4

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py \
  represent --source solidifi --workers 4
```

Default output root:

`data_module/data/representations-r4-v2/`

Required properties:

- every successful artifact has graph + token + repaired sidecar;
- graph schema `v9`;
- extractor `v2.2-r4-repaired`;
- requested graph contract == actual graph contract;
- multi-contract file labels retain every unrelated inheritance leaf rather than selecting one heuristically;
- token tensor remains `[4,512]`;
- coverage telemetry is present;
- any failure is recorded explicitly in `representation_failures.jsonl`.

The full local acceptance gate currently requires zero representation failures for required contracts; do not hide failures by deleting rows.

If a complete representation attempt ends with explicit failures, preserve that
source directory unchanged and use the failed-tail recovery command only after
the compatibility correction is committed and validated:

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py \
  recover-representations --source dive --workers 4 \
  --failed-attempt-dir /path/to/immutable/failed/dive
```

The normal destination must again be fresh. Recovery is accepted only when the
failed attempt was a full build bound to the same preprocessing manifest, its
structured failure identities reconcile exactly, all successful triples are
present for byte reuse, every failed identity is retried, the final manifest
reports zero failures, and physical completeness passes. Parse-only or
graph-source-compatibility artifacts remain visible in sidecars and the final
binding report; they are not silently equivalent to full-analysis graphs.

### 7. Materialize the separate repaired evidence ledger

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_build_repaired_evidence_ledger.py
```

Generated local artifacts:

- `data_module/data/r4-v2-build/evidence_ledger_v2.parquet`;
- `data_module/data/r4-v2-build/evidence_ledger_v2_manifest.json`.

This ledger is role-independent and must contain zero confirmed-negative/target-zero rows.

### 8. Freeze repaired roles and publish local DATA v2 candidate

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py publish
```

Default publication root:

`data_module/data/exports/sentinel-r4-vnext-v2/`

This creates dynamic population counts from the repaired physical lineage. Do **not** expect the old 22,493 / 13,509 / role counts to remain identical.

Before physical binding the publication status must remain:

`REPAIRED_CANDIDATE_LOCAL_ACCEPTANCE_REQUIRED`

### 9. Bind the repaired publication to physical representation bytes

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_repaired_data.py bind
```

A passing report must deserialize and validate every required graph/token/sidecar triple, requested/actual graph target-set equality, graph tensor/schema/edge bounds, token tensor/mask/coverage parity, extractor identity, and frozen token shape. It produces a deterministic content binding without recording the machine-specific representation root and reports graph component/node/edge distributions.

A passing binding changes the generated publication status only to:

`REPAIRED_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED`

This still does **not** authorize full training.

### 10. Run and persist the repaired lineage acceptance profiler

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_audit_repaired_lineage.py \
  --output data_module/data/r4-v2-build/repaired_lineage_audit.json
```

Required result before any GPU action:

`repository_data_acceptance_passed: true`

and always:

`training_authorized: false`

Review the full JSON rather than relying only on exit code.

### 11. Run the read-only bounded-window coverage experiment

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_compare_bounded_window_strategies.py \
  --output data_module/data/r4-v2-build/bounded_window_experiment.json
```

This compares the current four-window linspace control against a target-contract-aware bounded four-window candidate. It does not rewrite representations and cannot promote a new selector. Review:

- contracts/windows over cap;
- control global retained-token ratio;
- target-aware global retained-token ratio;
- control target-contract coverage;
- target-aware target-contract coverage;
- improved/regressed record counts;
- any failures.

The governing decision is `decisions/2026-08-15_PHASE8_LONG_CONTRACT_TOKEN_STRATEGY.md`.

### 12. Only after steps 1–11 pass/review: bounded repaired-data GPU smoke

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_run_repaired_micro_smoke.py \
  --output data_module/data/r4-v2-build/repaired_gpu_smoke.json
```

This smoke:

- uses dynamic repaired `TRAIN_STRONG` / `TRAIN_WEAK` / `MODEL_SELECTION` counts;
- loads no Run12 learned weights;
- writes no checkpoint;
- runs only a bounded number of train/selection batches;
- binds source commit, repaired DATA hashes/binding, runtime and accepted GraphCodeBERT snapshot;
- must report finite losses and at least one optimizer step;
- must report `full_training_authorized: false` even when it passes.

Stop after this result and review the repaired counts, attrition, binding, coverage experiment and smoke before changing governance again.

## Historical audit expectations — hypotheses, not pass criteria

The 2026-08-14 audit identified recoverable effects that motivate the rebuild:

- 65 compile-valid content-distinct positive records were historically removed solely by address equality;
- one valid solc-0.4.9 SmartBugs record was removed by the old unconditional compiler flag;
- five direct SmartBugs `time_manipulation→Timestamp` positives were physically identifiable;
- the identified maximum strong-cell increase was up to 71 **before** representation success and role re-freeze.

Do not assert `+66 contracts` or `+71 strong cells` as the rebuilt truth. Actual counts can differ because repaired normalization changes content identity, exact cross-source identities are aggregated, compile outcomes are re-evaluated on the promoted bytes, grouping is recomputed, representation target validation can expose additional failures, and roles are re-frozen atomically.

The acceptance profiler reports actual deltas against the historical audit. Those observed local values become evidence only after the full local stages have completed.

## What constitutes success before GPU smoke

Physical repaired-DATA acceptance requires all of the following:

- raw manifests byte-verify;
- no address-based deletion;
- no target-zero source claims or semantic rows;
- SmartBugs `bad_randomness` remains no-target;
- repaired grouping is present and final;
- repaired evidence ledger exists and agrees with publication target/strength counts;
- required representations have zero binding failures;
- requested graph target equals actual graph target for every required representation;
- graph schema and extractor versions are exact;
- token tensors stay `[4,512]`;
- publication and representation binding agree;
- generated outputs contain no machine-specific physical root;
- token coverage is reported honestly without an invented adequacy threshold.

## Rollback and rerun guidance

Historical artifacts are the rollback root and must not be altered:

- `sentinel-r4-vnext-v1`;
- `r4-vnext-roles-v1`;
- historical `data/preprocessed`;
- historical `data/representations`;
- Run12 artifacts.

Repaired local outputs are new versioned/generated roots only:

- `data_module/data/sentinel-preprocessed-r4-v2/`;
- `data_module/data/representations-r4-v2/`;
- `data_module/data/r4-v2-build/`;
- `data_module/data/exports/sentinel-r4-vnext-v2/`.

If a repaired stage fails:

1. preserve/copy the failing manifest/log/report first;
2. identify whether the failure is code, source, compiler, representation, or evidence-policy related;
3. do not patch generated parquet/JSON by hand;
4. if a repository source fix is required, commit/validate it first;
5. archive or remove **only the repaired-v2 generated roots** for the failed attempt;
6. rerun from a fresh repaired root on one exact clean source commit.

Never “repair” a failed build by editing historical v1 artifacts.

## Source acquisition limitation

SolidiFI has an exact repository pin in the acquisition descriptor. SmartBugs currently has an audited commit prefix and still needs the full commit captured before portable recovery is claimed. DIVE still needs a stable external archive locator/hash. These portability gaps do not invalidate byte-verified local raw manifests, but they remain reproducibility risks to close separately.

## Stop line

This handoff authorizes repository-local DATA rebuild, validation, read-only coverage analysis, and the bounded repaired-data GPU smoke only.

It does **not** authorize:

- `p8_run_training.sh`;
- 100 epochs;
- checkpoint promotion;
- threshold fitting;
- calibration fitting;
- acceptance inspection;
- invented negatives;
- architecture/input-shape changes.

The bounded smoke evidence was preserved and reviewed. The resulting governance
decision does not authorize the full Phase-8 horizon: every effective loss cell
is positive-only and the current token selector remains under review.

**Repository repair and physical DATA acceptance complete locally; bounded GPU
smoke passed; 100-epoch training not authorized.**
