# Phase-8 V3 hardened evidence snapshot closeout

**Date:** 2026-08-16
**Canonical branch:** `main`
**Logical authority:** R4-D-009 / accepted logical V3
**Evidence gap:** R4-GAP-007
**State:** HARDENED V3 RESEARCH EVIDENCE SNAPSHOT COMMITTED AND FRESH-CLONE VERIFIED
**Training:** NOT AUTHORIZED
**G8:** OPEN

## Purpose

This record closes the post-acceptance V3 evidence-hardening/regeneration tranche that began in `2026-08-16_PHASE8_v3_evidence_hardening_handoff.md`.

The earlier hardening handoff remains historical execution evidence. Its instruction to regenerate acceptance/sensitivity/selector/queue/GPU reports and then create a final snapshot is now complete and must not be treated as pending work.

A second protected-local audit after the snapshot commit identified three **future-safety / verification** gaps: the snapshot queue predicate could fail open for an empty candidate list or an invalid outcome state, CI did not re-verify the actual committed snapshot from a fresh clone, and one build-stage summary inside the snapshot could be mistaken for current instructions. These findings did not show current-data corruption, model-selection leakage, fabricated negatives, representation corruption, or unauthorized training. The follow-through below closes those verification gaps without rebuilding physical DATA or rerunning CUDA.

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

## Post-closeout verification follow-through

The second audit reproduced two fail-open cases in the original queue coherence predicate: an empty candidate list could satisfy `all([])`, and a candidate whose `current_outcome_state` was changed to `CONFIRMED_POSITIVE` was not rejected. The committed real queue itself remained valid.

The snapshot verifier is now hardened to require the exact pilot contract:

- exactly 200 candidate cells;
- exactly eight enabled classes and 25 candidates per class;
- canonical 10-class indices for the enabled classes: `0,1,2,4,5,6,7,8`;
- queue ordinals `1..25` within every enabled class;
- deterministic candidate IDs matching the production queue-ID function;
- 200 globally unique candidate groups;
- the reserved-group set exactly matching the candidate-group set;
- `PENDING_REVIEW`, target `None`, `TRAIN_UNLABELED`, and `negative_truth_claim=false` for every candidate;
- `current_outcome_state` restricted to `UNKNOWN` or `NOT_REVIEWED`.

Focused regressions now prove that an empty queue, `CONFIRMED_POSITIVE` outcome, invalid candidate ID, and class imbalance all fail closed. The Phase-8 repository regression suite increased from 141 to **145 passing tests**.

Fresh-clone CI now runs:

`docs/plan/ml-R4/scripts/p8_verify_committed_logical_v3_snapshot.py`

against the **actual committed snapshot**, not only synthetic fixtures. The verifier:

1. requires the exact 11-JSON snapshot inventory;
2. verifies every SHA-256 listed in the original `SHA256SUMS.txt`;
3. recomputes the strengthened cross-report coherence contract over committed files;
4. semantically validates the full 200-row queue;
5. requires the snapshot index/addendum that classifies the early summary correctly;
6. preserves training and selector-promotion stop lines.

Fresh-clone verification on the committed evidence produced:

- `committed_snapshot=PASS`;
- `json_checksums_verified=11`;
- `coherence_checks_recomputed=60`;
- `queue_cells=200`;
- `queue_groups=200`;
- `queue_outcomes={"NOT_REVIEWED": 48, "UNKNOWN": 152}`;
- evidence source commit `83bd566b9c4f4f653e530c2c0f5c990858dd759d`;
- `training_authorized=false`;
- `selector_promotion_authorized=false`.

The same CI validation also passed the 145-test regression suite and historical frozen G6 validation. An intermediate run was red only because the newly added `SNAPSHOT_INDEX.md` used Markdown hard-break trailing spaces; those formatting-only spaces were removed without changing evidence semantics.

### Historical build-stage summary addendum

`logical_v3_summary.json` is intentionally **not rewritten**. It remains a historical build-stage artifact whose recorded status says research regeneration was pending at that earlier point.

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/SNAPSHOT_INDEX.md` now explicitly classifies it as:

`HISTORICAL_BUILD_STAGE_SUMMARY`

and directs current readers to the downstream acceptance/sensitivity/selector/GPU/queue/coherence evidence instead. The index is non-destructive contextual metadata added after the original snapshot commit. It is intentionally not retroactively inserted into `SHA256SUMS.txt`; the original checksum ledger continues to bind the exact 11 JSON files from snapshot commit `44fbb9c1d2033be8002fe404d650cf09f08b0f29`.

No physical DATA artifact, snapshot JSON, representation, queue content, selector report, or CUDA result was regenerated for this verification follow-through.

## Current Phase-8 blockers

The post-acceptance V3 evidence-regeneration and committed-snapshot verification blockers are closed.

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
5. `docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/SNAPSHOT_INDEX.md` for snapshot-internal chronology;
6. `2026-08-16_PHASE8_v3_evidence_hardening_handoff.md` only as historical execution detail.

The current restart point is **after** coherent V3 snapshot commit `44fbb9c1d2033be8002fe404d650cf09f08b0f29` and after the fresh-clone verification follow-through. R4-GAP-007 is the primary next track; regeneration is not pending.
