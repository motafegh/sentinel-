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

V3 reuses accepted repaired-v2 graph/token/sidecar bytes. Protected-local V3
acceptance and the hardened evidence snapshot are complete.

R4-D-010 preserves repaired-v2/v9 for reproduction but makes v9 ineligible for
the new full run. R4-D-011 physically accepts only the exact protected-local
graph-schema-v10 root `representations-r4-v3-candidate`, extractor
`v2.6-r4-call-semantics-deterministic-cfg-mutators`, binding digest
`d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`.
Repository consumers support the exact 17-kind v10 edge vocabulary without OOB
clamping or checkpoint resizing. All 22,540 identities pass binding and the
complete 355-identity structural evidence has zero unexplained drift. This
physical acceptance does not authorize selector promotion or training.

## Compatibility boundaries

Do not loosen historical guards:

- `ml/src/datasets/vnext_dataset.py` remains historical G7/v1;
- `ml/src/training/vnext_binding.py` remains historical Phase-8 v1 run binding;
- Run12 learned weights/optimizer/scheduler/threshold/calibration remain historical only.

Repaired/v2 remains a physical/historical research seam:

- `ml/src/datasets/vnext_repaired_dataset.py` binds `sentinel-r4-vnext-v2`;
- V2 role-dependent research outputs are historical after R4-D-009.

Current accepted V3 logical seam:

- `ml/src/datasets/vnext_logical_v3_dataset.py` requires V3 dataset/grouping/partition identity and a passing V3 physical binding report;
- the historical adapter reuses the accepted repaired-v2 representation root;
- it fails closed if address grouping authority is re-enabled or the V3 binding is absent/mismatched.

Future v10 training seam:

- `ml/src/datasets/vnext_logical_v3_v10_dataset.py` requires exact V3/v10/
  extractor/root identities;
- it rejects a diagnostic candidate and requires both a separate physical
  acceptance report and an explicit binding-specific training decision;
- `build_v10_run_binding` carries that decision and forbids historical
  checkpoint reuse;
- current candidate reports cannot satisfy these guards.

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
- graph schema `v10` for any future new run; v9 only for historical reproduction;
- token tensor `[4,512]`.

`target_aware_guarded_v1` remains a research candidate, not a promoted extractor. V2 evidence showed strong requested-target coverage improvement, but those statistics were conditioned on superseded V2 roles.

The hardened V3 selector/CUDA evidence is complete but does not promote the
selector. Initial v10 comparison reuses accepted v9 token bytes exactly.

Encoding more than four windows is an architecture/input-capacity change and needs a separate architecture decision.

## Training-horizon rule

Do not copy historical v1 or V2 scheduler counts into a V3 run.

The earlier V2 active population of 831 groups and planning arithmetic of 104 micro-batches / 13 optimizer steps per epoch are now historical V2 partition facts. The V3 acceptance audit derives the corrected active optimizer contracts/groups and reports fresh batch-8/accum-8 planning arithmetic.

Even those V3 counts remain planning-only until the final objective, selector, roles, and training configuration are explicitly authorized and bound.

## Local v10 preservation boundary

Follow:

`docs/plan/ml-R4/runs/2026-09-02_PHASE8_v10_v26_physical_acceptance_and_no_launch.md`

Order:

1. preserve the exact R4-D-011 root and digest; do not regenerate or overwrite it;
2. keep candidate #2 independent review separate from representation acceptance;
3. resolve objective/evaluation evidence and selector promotion independently;
4. keep training closed pending explicit later run-control and training authorization.

## Runtime provenance

Any GPU diagnostic or future authorized run must bind exact source commit, V3 DATA/group/partition manifest hashes, the unchanged physical representation digest, architecture/class order, seed/config, package versions, and accepted GraphCodeBERT snapshot `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`.

Do not change packages during an evidence-generating run.

## Validation discipline

Repository-safe Phase-8 validation is:

`.github/workflows/r4-phase8-data-repair.yml`

It compiles repaired/V3 DATA+ML entry points, runs regressions, revalidates frozen historical G6, and runs the repair-tranche diff gate.

Repository CI cannot prove the Git-ignored V3 publication or local GPU evidence. Physical/local acceptance must execute the handoff commands against the existing accepted repaired-v2 trees.

## Training stop line

Until v10 is physically accepted and later evaluation/selector/training decisions are made:

- do not run the 100-epoch job;
- do not create/promote a repaired teacher checkpoint;
- do not reuse Run12 learned weights;
- do not tune thresholds/calibration;
- do not manufacture negative or acceptance evidence;
- do not promote `target_aware_guarded_v1`;
- do not approve PU learning merely to bypass the lack of negative evaluation evidence.

**Current ML status:** repaired-v2/v9 remains accepted historical physical
evidence; logical V3 is accepted; v10 repository consumers, bounded regressions,
and full diagnostic mechanics pass, but physical v10 acceptance is blocked by
26 parse-only contracts. Confirmed-negative evidence and selector promotion
remain unresolved; G8 is open and full training is unauthorized.
