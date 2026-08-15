# SENTINEL ML Working Instructions

This file extends root `CLAUDE.md` for `ml/`. Root project authority and committed R4 machine-readable policy/manifests remain higher authority.

## Current Phase-8 ML state

The historical G7 DATA/ML lineage remains immutable and reproducible:

- DATA publication: `sentinel-r4-vnext-v1`;
- role partition: `r4-vnext-roles-v1`;
- graph schema: `v9`;
- historical representation extractor: `v2.1-windowed-gcb`;
- historical G7 representation binding digest: `7637461f6643d398c7a0446412fedd8877914c7b9ed41309dab45f18ed96f420`.

The 2026-08-14 real-data readiness audit found physical preprocessing/representation defects that prevent this historical v1 lineage from serving as the full Phase-8 retrain input. G7 remains valid evidence for v1; it is not erased or rewritten.

Repository repair defines a separate repaired lineage:

- preprocessing: `sentinel-preprocessed-r4-v2`;
- evidence ledger: `evidence-ledger-r4-v2`;
- grouping: `r4-leakage-groups-v2`;
- role partition: `r4-vnext-roles-v2`;
- DATA publication: `sentinel-r4-vnext-v2`;
- representation extractor: `v2.2-r4-repaired`;
- graph schema remains `v9`;
- model token tensor remains `[4,512]`;
- architecture remains `four_eye_v8` / `v8.1`.

Repository-safe implementation/tests and the protected local physical rebuild are complete. At evidence source commit `fb31326da4420c2289822c2a6db8a022ac25876a`, repaired-v2 binds 22,540 contracts / 67,620 files with digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`. The generated artifacts are Git-ignored and absent from a fresh clone. The bounded CUDA smoke passed without Run12 weights or a checkpoint; no model-quality result is claimed.

Repaired-v2 physical authority and the launch hold are governed by:

- R4-D-008 in `docs/plan/ml-R4/DECISION_REGISTER.md`;
- `docs/plan/ml-R4/adrs/ADR-R4-008-repaired-v2-data-acceptance-and-phase8-no-launch.md`;
- `docs/plan/ml-R4/runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md`.

## Historical-v1 compatibility boundary

Do not loosen the existing v1 guards merely to make repaired data load:

- `ml/src/datasets/vnext_dataset.py` remains bound to historical G7/v1 semantics;
- `ml/src/training/vnext_binding.py` remains the historical Phase-8 v1 run-binding contract;
- old fixed v1 population assertions remain reproducibility evidence;
- old Run12 artifacts remain historical only.

The repaired path is separate:

- `ml/src/datasets/vnext_repaired_dataset.py` consumes only physically bound `sentinel-r4-vnext-v2` and derives repaired populations dynamically;
- `ml/src/training/vnext_repaired_binding.py` currently binds bounded repaired-data smoke evidence and explicitly records `full_training_authorized=false`;
- the repaired CUDA micro-smoke is complete evidence, not permission for the 100-epoch runner;
- additional repaired-v2 GPU work is authorized only when it is a **bounded comparative diagnostic** explicitly tied to R4-D-008 / the long-contract decision or a later approved objective experiment. It must not write/promote a training checkpoint or silently become a full-run substitute.

Do not route `p8_run_training.sh` to repaired v2 until governance explicitly re-authorizes the full run after objective/evaluation and selector decisions.

## Permanent R4 constraints

### Unknown is not negative

The historical consumer path treated every binary `0` as a supervised negative. That behavior is **not valid for DATA vNext**.

Current/repaired training must carry:

- authorized nullable target;
- training strength (`STRONG`, `WEAK`, `NONE`);
- effective loss mask;
- outcome/metric mask;
- frozen dataset role;
- DATA/policy/config lineage.

Never fill an unknown/masked/disabled vNext cell with zero just to satisfy an old loss API.

### Current class support

The ten-output order remains locked. `GasException` and `UnusedReturn` are supervision-disabled under `data-vnext-policy-v1` until later evidence-backed policy changes.

DIVE Front Running→TransactionOrderDependence is weak-positive only. Weak does not become strong or metric-grade evidence automatically.

### Evaluation roles

Current first-baseline limitations remain:

- training strong: supported;
- training weak: supported for DIVE TOD only;
- training unlabeled: supported;
- model selection: positive-only limited;
- threshold fit: `UNSUPPORTED_EMPTY`;
- calibration fit: `UNSUPPORTED_EMPTY`;
- untouched acceptance: `UNSUPPORTED_EMPTY_FROZEN`.

Do not run a historical utility and then infer that the corresponding evidence role exists. A threshold/calibration script executing successfully is not authorization to fit policy on unknown/exposed data.

### Run12 compatibility

Preserve Run12/checkpoint companions for reproducibility and comparison. Do not overwrite them with repaired artifacts.

A future repaired checkpoint that keeps the same architecture is still semantically new and must bind:

- exact repaired DATA artifact/policy/roles;
- exact repaired representation binding;
- training config and seed/initialization;
- strong/weak numeric optimization handling;
- checkpoint hash;
- checkpoint-selection evidence/limitations;
- any later authorized threshold/calibration artifacts.

Do not automatically reuse Run12 weights, optimizer/scheduler state, thresholds, calibration, drift, or proxy-agreement evidence.

## Architecture and long-contract boundary

The repository repair deliberately does **not** change architecture or frozen input shape.

Current contract:

- architecture: `four_eye_v8`;
- model version: `v8.1`;
- class count/order: existing locked ten-class order;
- graph schema: `v9`;
- token tensor: `[4,512]`;
- historical four-window linspace selection remains the repaired control;
- token coverage is explicit diagnostic evidence using `r4-token-coverage-v1`.

Do not infer that `[4,512]` shape validity means long-contract adequacy. The governing decision is:

`docs/plan/ml-R4/decisions/2026-08-15_PHASE8_LONG_CONTRACT_TOKEN_STRATEGY.md`

A target-contract-aware bounded-window comparison is implemented for evidence collection. The first corpus-wide coverage experiment is complete and promising but not sufficient for promotion. Production selector promotion requires a versioned extractor lineage plus identical-initialization bounded GPU comparison, regression-case review, worst-case large-graph diagnostics, and repeated physical binding/smoke evidence. Encoding more than four windows is an architecture/input-capacity change and requires a separate architecture decision.

## Repaired training-horizon rule

Do not copy historical v1 dataloader/scheduler constants into a future repaired-v2 run.

Current measured active optimizer population is 899 contracts over 831 active leakage groups. With the existing one-member-per-group sampler, batch size 8, and gradient accumulation 8, **planning-only arithmetic** would be 104 micro-batches and 13 optimizer/scheduler steps per epoch, or 1,300 steps across 100 epochs. The historical v1 values 88 / 11 / 1,100 are not repaired-v2 authority.

No repaired-v2 full-run horizon is currently authorized. Any later objective/selector/population change must recompute the actual dataloader length and scheduler horizon from the final bound configuration, then record it in the run binding/checkpoint lineage.

## ML runtime provenance

Any repaired GPU diagnostic or later authorized full run must bind:

- exact source commit;
- repaired DATA manifest/content hashes;
- repaired representation binding digest;
- policy/grouping/claims/role identities;
- frozen architecture/model/class order;
- seed and optimization configuration;
- actual runtime package versions;
- accepted GraphCodeBERT snapshot `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`.

Do not change packages during an active evidence-generating run.

## Before modifying source

1. Determine whether the task touches historical Run12/v1 compatibility or the repaired-v2 path.
2. Read the exact source/tests.
3. If DATA semantics/evidence roles are involved, read current R4 policy/manifest/decision records before coding.
4. Preserve the architecture freeze unless an explicit later R4 decision unfreezes it.
5. Add failure tests for missing masks/strength/roles/bindings where relevant.
6. Do not weaken DATA semantics or historical replay guards to minimize ML changes; use the separate repaired seam.
7. If changing objective, selector, grouping, representation semantics, or evaluation roles, create a new versioned identity/binding rather than mutating accepted repaired-v2 evidence in place.

## Validation discipline

For repository-safe repair/current Phase-8 compatibility work, use the dedicated workflow:

`.github/workflows/r4-phase8-data-repair.yml`

It compiles repaired DATA/ML/local scripts, runs repaired regressions, revalidates frozen historical G6 artifacts, and performs a diff whitespace/error check from the real-data repair base.

Repository CI cannot substitute for protected local physical evidence. The accepted repaired-v2 local gates have already been executed, but **any changed physical lineage** must repeat the applicable local gates:

- raw/input byte and provenance verification where inputs change;
- actual historical solc execution where preprocessing/compiler policy changes;
- regenerated repaired representations where selector/graph semantics change;
- regenerated repaired parquets/roles/publication where DATA semantics/grouping change;
- physical representation binding;
- repaired-lineage acceptance audit;
- relevant token/graph coverage experiment;
- bounded repaired-data CUDA diagnostic.

Historical/supplementary ML testing specs remain useful only when they do not conflict with current R4 semantics.

## Coding conventions

- type hints on ML source interfaces;
- import canonical schema constants; do not duplicate class/graph constants;
- explicit versioned config for training/evidence semantics;
- structured metrics/log fields;
- focused tests near the relevant source area;
- decision numbers require measured evidence and explicit config;
- no silent failures/skips/defaults that can contaminate evaluation.

## Training stop line

Until the positive-only objective/evaluation limitation and token-selector decision are resolved and governance is updated again:

- do not run the 100-epoch job;
- do not create/promote a repaired full-training checkpoint;
- do not reuse Run12 learned weights;
- do not tune thresholds/calibration;
- do not inspect/manufacture acceptance data;
- do not weaken v1/v2 provenance or role guards just to make execution convenient;
- bounded comparative diagnostics are permitted only under an explicit versioned experimental contract and must remain non-promotional.

**Current ML status:** repaired-v2 physical DATA accepted locally for bounded research; bounded CUDA smoke passed; all 899 effective loss cells are positive-only; target-aware selector promising but not promoted; grouping/compatibility/file-union representation sensitivity remains open evidence; G8 open; 100-epoch training not authorized.
