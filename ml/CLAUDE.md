# SENTINEL ML Working Instructions

This file extends root `CLAUDE.md` for `ml/`. Root project authority and committed R4 machine-readable policy/manifests remain higher authority.

## Current Phase-8 ML state

Historical G7/v1 and Run12 remain immutable reproducibility/comparison roots. They are not current repaired training truth.

R4-D-008 physically accepted the repaired-v2 DATA source/representation layer:

- 22,540 contracts / 67,620 graph-token-sidecar files;
- physical representation binding digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- representation extractor `v2.2-r4-repaired`;
- graph schema `v9`;
- token tensor `[4,512]`;
- architecture `four_eye_v8` / `v8.1` remains frozen.

That physical acceptance remains valid.

R4-D-009 / ADR-R4-009 supersedes **V2 grouping/roles for future ML research** because the full-population audit found a 10,327-contract DIVE component produced by arbitrary shared-address grouping. The active logical candidate is therefore:

- `r4-leakage-groups-v3`;
- `r4-vnext-roles-v3`;
- `sentinel-r4-vnext-v3`;
- ML adapter `ml/src/datasets/vnext_logical_v3_dataset.py`.

V3 reuses accepted repaired-v2 graph/token/sidecar bytes. Local V3 generation and acceptance are pending; until they pass, no repaired logical partition is authorized for full training.

## Compatibility boundaries

Do not loosen historical guards:

- `ml/src/datasets/vnext_dataset.py` remains historical G7/v1;
- `ml/src/training/vnext_binding.py` remains historical Phase-8 v1 run binding;
- Run12 learned weights/optimizer/scheduler/threshold/calibration remain historical only.

Repaired/v2 remains a physical/historical research seam:

- `ml/src/datasets/vnext_repaired_dataset.py` binds `sentinel-r4-vnext-v2`;
- V2 role-dependent research outputs are historical after R4-D-009.

Current V3 research seam:

- `ml/src/datasets/vnext_logical_v3_dataset.py` requires V3 dataset/grouping/partition identity and a passing V3 physical binding report;
- it reuses the accepted repaired-v2 representation root;
- it fails closed if address grouping authority is re-enabled or the V3 binding is absent/mismatched.

Do not route the full 100-epoch runner to V2 or V3 until governance explicitly re-authorizes training.

## Permanent R4 semantics

### Unknown is not negative

No unknown/masked/disabled cell may be filled with zero to satisfy an old loss API. Current model inputs must preserve nullable targets, strength, effective-loss masks, outcome/metric masks, roles, and lineage.

Policy v1 still has zero confirmed-negative source authority. `GasException` and `UnusedReturn` remain supervision-disabled. DIVE Front Running→TransactionOrderDependence remains weak-positive only.

### Evaluation roles

Current limitations remain:

- training strong: supported;
- training weak: supported for authorized weak evidence;
- training unlabeled: supported as a role, not as negative truth;
- model selection: positive-only limited;
- threshold fit: `UNSUPPORTED_EMPTY`;
- calibration fit: `UNSUPPORTED_EMPTY`;
- untouched acceptance: `UNSUPPORTED_EMPTY_FROZEN`.

The old V2 confirmed-negative pilot queue is obsolete because its group reservations came from the superseded V2 partition. Generate/review the V3 queue only after V3 acceptance.

## Architecture and selector boundary

Keep architecture/input shape frozen for this tranche:

- architecture `four_eye_v8`;
- model `v8.1`;
- ten locked outputs;
- graph schema `v9`;
- token tensor `[4,512]`.

`target_aware_guarded_v1` remains a research candidate, not a promoted extractor. V2 evidence showed strong requested-target coverage improvement, but those statistics were conditioned on superseded V2 roles.

After V3 acceptance, regenerate selector coverage and CUDA evidence against V3 roles. The V3 CUDA launcher must execute its requested worst-case graph probes; unlike the earlier V2 run, missing sensitivity evidence cannot silently produce an empty probe set.

Encoding more than four windows is an architecture/input-capacity change and needs a separate architecture decision.

## Training-horizon rule

Do not copy historical v1 or V2 scheduler counts into a V3 run.

The earlier V2 active population of 831 groups and planning arithmetic of 104 micro-batches / 13 optimizer steps per epoch are now historical V2 partition facts. The V3 acceptance audit derives the corrected active optimizer contracts/groups and reports fresh batch-8/accum-8 planning arithmetic.

Even those V3 counts remain planning-only until the final objective, selector, roles, and training configuration are explicitly authorized and bound.

## Local V3 execution

Follow:

`docs/plan/ml-R4/runs/2026-08-15_PHASE8_logical_v3_grouping_repair_handoff.md`

Order:

1. build V3 grouping;
2. freeze V3 roles/publication;
3. prove unchanged physical binding;
4. pass V2→V3 logical acceptance;
5. regenerate V3 representation-sensitivity and selector population evidence;
6. generate the V3 negative-review queue;
7. rerun identical-initialization selector CUDA comparison with mandatory worst-case probes;
8. review evidence before any selector/objective/training decision.

## Runtime provenance

Any GPU diagnostic or future authorized run must bind exact source commit, V3 DATA/group/partition manifest hashes, the unchanged physical representation digest, architecture/class order, seed/config, package versions, and accepted GraphCodeBERT snapshot `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`.

Do not change packages during an evidence-generating run.

## Validation discipline

Repository-safe Phase-8 validation is:

`.github/workflows/r4-phase8-data-repair.yml`

It compiles repaired/V3 DATA+ML entry points, runs regressions, revalidates frozen historical G6, and runs the repair-tranche diff gate.

Repository CI cannot prove the Git-ignored V3 publication or local GPU evidence. Physical/local acceptance must execute the handoff commands against the existing accepted repaired-v2 trees.

## Training stop line

Until V3 is accepted locally and evaluation/selector decisions are made:

- do not run the 100-epoch job;
- do not create/promote a repaired teacher checkpoint;
- do not reuse Run12 learned weights;
- do not tune thresholds/calibration;
- do not manufacture negative or acceptance evidence;
- do not promote `target_aware_guarded_v1`;
- do not approve PU learning merely to bypass the lack of negative evaluation evidence.

**Current ML status:** physical repaired-v2 representations remain accepted; V2 grouping/roles are superseded for future research; logical V3 is implemented and awaits protected local acceptance; confirmed-negative evidence and selector promotion remain unresolved; G8 open; full training unauthorized.
