# Phase-8 V3 hardened evidence snapshot closeout

**Date:** 2026-08-16
**Canonical branch:** `main`
**Logical authority:** R4-D-009 / accepted logical V3
**Evidence gap:** R4-GAP-007
**State:** HARDENED V3 RESEARCH EVIDENCE SNAPSHOT COMMITTED
**Training:** NOT AUTHORIZED
**G8:** OPEN

## Purpose

This record closes the post-acceptance V3 evidence-hardening/regeneration tranche that began in `2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`.

The earlier hardening handoff remains historical execution evidence. Its instruction to regenerate acceptance/sensitivity/selector/queue/GPU reports and then create a final snapshot is now complete and must not be treated as pending work.

## Canonical evidence boundary

Repository hardening source commit used by all regenerated protected-local reports:

`83bd566b9c4f4f653e530c2c0f5c990858dd759d`

Final Git-safe evidence snapshot commit on `main`:

`44fbb9c1d2033be8002fe404d650cf09f08b0f29`

Durable snapshot root:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

The snapshot helper completed with:

- `SNAPSHOT_EXIT=0`;
- `coherence=PASS`;
- every JSON listed in `SHA256SUMS.txt` verified `OK` before commit;
- no generated DATA root was force-added;
- the large per-contract selector payload remained local and is represented by a bounded summary plus source SHA-256.

## Accepted physical and logical state

The closeout does not change R4-D-008 or R4-D-009.

Physical repaired-v2 remains accepted:

- contracts: 22,540;
- contract×class rows: 225,400;
- graph/token/sidecar files: 67,620;
- positive / unknown / confirmed-negative targets: 1,080 / 224,320 / 0;
- STRONG / WEAK semantic cells: 474 / 606;
- binding digest: `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

Accepted logical V3 remains:

- dataset: `sentinel-r4-vnext-v3`;
- grouping: `r4-leakage-groups-v3`;
- partition: `r4-vnext-roles-v3`;
- groups: 22,394;
- maximum group size: 7;
- normalized-code edges: 146;
- address-authority edges: 0;
- physical binding unchanged from repaired-v2.

## Corrected outcome-population reporting

The hardened acceptance rerun passed with no failed checks and reproduced the audited role split:

- optimizer-active contracts/groups: 932 / 932;
- combined outcome-metric population: 143 contracts / 142 unique groups;
- active `MODEL_SELECTION`: 71 contracts / 71 groups;
- active `INTERNAL_AUDIT`: 72 contracts / 71 groups;
- `training_authorized=false`.

`143/142` must never again be described as the MODEL_SELECTION population. The ML adapter permits `MODEL_SELECTION` only for model-selection loading; `INTERNAL_AUDIT` is not loaded into that trainer/evaluation path.

## Hardened representation sensitivity

The regenerated sensitivity report is bound to:

- dataset/grouping/partition versions;
- publication manifest SHA-256;
- repaired-v2 physical binding digest;
- source commit `83bd566b9c4f4f653e530c2c0f5c990858dd759d`.

`MODEL_SELECTION` and `INTERNAL_AUDIT` are separated in the report. Worst-case selector probes are selected only from optimizer-active or actual MODEL_SELECTION records.

Full training remains unauthorized.

## Hardened CPU selector evidence

The lineage-bound CPU selector rerun completed successfully and reproduced the earlier coverage result:

- records analyzed: 1,018;
- over four windows: 737;
- guarded improved: 476;
- equal: 261;
- control fallback: 261;
- regressed: 0;
- failures: 0;
- `promotion_authorized=false`;
- `changes_bound_representations=false`.

This is durable research evidence, not selector promotion authority.

## Hardened confirmed-negative review queue

The regenerated V3 queue now enforces the invariant in code rather than relying on the observed artifact:

- queued cells: 200;
- 25 candidates for each of eight enabled classes;
- reserved leakage groups: 200;
- group uniqueness scope: `GLOBAL_ACROSS_ENABLED_CLASSES`;
- all candidates: `PENDING_REVIEW`;
- all current targets: `None`;
- all roles at queue creation: `TRAIN_UNLABELED`;
- `negative_truth_claim=false`;
- queue bound to the same V3 publication manifest and source commit.

This regenerated durable queue is now the only V3 queue eligible for R4-GAP-007 pilot adjudication. Queue membership is still not negative truth.

## Hardened CUDA selector comparison

The regenerated CUDA comparison completed successfully with:

- status: `LOGICAL_V3_BOUNDED_RESEARCH_COMPLETE`;
- identical initialization: true;
- required/completed worst-case probes: 4 / 4;
- source commit: `83bd566b9c4f4f653e530c2c0f5c990858dd759d`;
- sensitivity report SHA-256: `fe4cf9aae0d5bbf2737501cab86f00b445ee6b0901203d4f67bfdf24def45322`;
- no Run12 weights;
- no checkpoint;
- `selector_promotion_authorized=false`;
- `full_training_authorized=false`.

Positive-only CUDA evidence still cannot establish false-positive discrimination or production model quality.

## Snapshot coherence gate

The final helper validated the decision-critical reports against one V3 lineage before writing durable evidence. The committed `snapshot_coherence_v1.json` binds:

- V3 publication manifest/version/status;
- partition/version/role counts;
- representation binding report and digest;
- logical acceptance PASS and source commit;
- grouping audit address policy;
- representation sensitivity lineage;
- globally unique pending negative-review queue;
- CPU selector lineage and zero regressions;
- CUDA report lineage, exact sensitivity SHA, identical initialization, complete worst-case probes, and no training/promotion authority.

`coherence=PASS` means these evidence files belong to one coherent V3 publication/binding/source state. It does not create negative truth, promote a selector, or authorize training.

## Current Phase-8 blockers

The post-acceptance V3 evidence-regeneration blocker is closed.

The remaining scientific/governance blockers are now:

1. **R4-GAP-007 confirmed-negative pilot** — zero confirmed negatives still exist. The regenerated durable queue is ready for class-specific dual review.
2. **Selector promotion decision** — `target_aware_guarded_v1` remains unpromoted. Before a promotion ADR, prove full-population equivalence between the historical control selector and the currently bound representation token tensors for the relevant population.
3. **Training/evaluation design** — no threshold/calibration/untouched-acceptance evidence exists, and the 100-epoch Phase-8 run remains unauthorized.

## Next controlled work

Primary next track:

- begin R4-GAP-007 pilot adjudication using only the committed hardened V3 queue;
- preserve class-specific primary review plus independent agreeing verification;
- accepted negatives remain `EVALUATION_ONLY_NOT_TRAINING_AUTHORITY` unless a later versioned policy grants optimizer authority;
- use observed pilot yield before expanding review volume.

Secondary independent track:

- design/execute the full-population bound-token control-equivalence check required before any guarded-selector promotion ADR.

Do not start the 100-epoch run, infer target `0`, reuse Run12 state, fit thresholds/calibration, or silently promote the selector.

## Restart authority

For current DATA/ML Phase-8 work, read in order:

1. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
2. this closeout;
3. `docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md`;
4. `docs/plan/ml-R4/DECISION_REGISTER.md` and `adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md`;
5. `2026-08-16_PHASE8_v3_evidence_hardening_handoff.md` only as historical execution detail.

The current restart point is **after** coherent V3 snapshot commit `44fbb9c1d2033be8002fe404d650cf09f08b0f29`, not before regeneration.
