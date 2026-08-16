# Phase-8 V3 evidence hardening handoff

**Date:** 2026-08-16  
**Canonical branch:** `main`  
**Logical authority:** R4-D-009 / accepted logical V3  
**Evidence gap:** R4-GAP-007  
**State:** repository hardening implemented; protected local evidence regeneration required before final snapshot  
**Training:** NOT AUTHORIZED  
**G8:** OPEN

## Why this handoff exists

A protected-local audit after the initial V3 research checkpoint found five repository/reporting defects. None invalidates the accepted repaired-v2 physical bytes. None establishes model leakage into the trainer. Two defects affect the interpretation/lineage quality of the current research reports, and three are future-safety/invariant-enforcement gaps that must be fixed before a durable final V3 snapshot is created.

Therefore the earlier instruction to run the final snapshot immediately is superseded. **Do not create the final V3 snapshot until the affected reports are regenerated with the hardened code.**

## Audit findings and disposition

### 1. MODEL_SELECTION and INTERNAL_AUDIT were conflated in reporting

`p8_audit_logical_v3_acceptance.py` previously treated every row with any `outcome_metric_mask_*` as a “model selection” row. V3 intentionally enables outcome metrics for both `MODEL_SELECTION` and `INTERNAL_AUDIT`, so the reported `143 contracts / 142 groups` was the **combined outcome-metric/audit population**, not the active model-selection population.

Protected-local audit observed:

- active `MODEL_SELECTION`: **71 contracts / 71 groups**;
- active `INTERNAL_AUDIT`: **72 contracts / 71 groups**;
- combined outcome-metric/audit population: **143 contracts / 142 unique groups**.

The frozen role assignment itself remains:

- `MODEL_SELECTION`: 73 contracts / 71 groups;
- `INTERNAL_AUDIT`: 73 contracts / 71 groups.

The difference between frozen role count and active metric-bearing count is expected when a reserved role contract has no active metric cell.

The ML dataset adapter was audited separately: `MODEL_SELECTION_ROLES = {"MODEL_SELECTION"}`, training and model-selection datasets must be separate, and `INTERNAL_AUDIT` is not an allowed trainer/model-selection role. This was a reporting/research-population defect, **not trainer leakage**.

Repository fix:

- acceptance reporting now separates `outcome_metric_*`, `model_selection_*`, and `internal_audit_*` counts;
- it fails if outcome metric masks appear on a role other than `MODEL_SELECTION` or `INTERNAL_AUDIT`;
- acceptance reports now bind themselves to publication manifest hash, physical binding digest, parent manifest hash, and source commit.

### 2. Final snapshot helper did not prove cross-report coherence

The old helper copied/sanitized files and hashed the resulting snapshot but did not first prove that all source reports described the same V3 manifest/binding.

Repository fix:

`p8_snapshot_logical_v3_evidence.py` now fails closed before creating the durable output directory unless it proves coherence across:

- V3 publication manifest/version/status;
- partition version and role counts;
- representation binding report hash/digest and pass state;
- logical acceptance pass state and lineage;
- grouping-audit address policy;
- representation-sensitivity lineage;
- confirmed-negative queue manifest binding, unknown state, and global group uniqueness;
- selector CPU lineage, zero experiment failures, and zero guarded coverage regressions;
- GPU report lineage, sensitivity-report SHA binding, identical initialization, complete worst-case probes, no Run12 weights/checkpoint, and no training/selector authorization.

A passing snapshot now writes `snapshot_coherence_v1.json` in addition to `SHA256SUMS.txt`.

### 3. Representation sensitivity was insufficiently self-identifying

The old report recorded paths/statistics but not enough immutable lineage to reject stale-report mixing.

Repository fix:

`p8_profile_representation_sensitivity.py` now records:

- dataset version;
- grouping version;
- partition version;
- publication `manifest.json` SHA-256;
- representation binding digest;
- source commit.

It also validates the publication `ml_targets.parquet` SHA before profiling.

A related reporting defect was corrected at the same time: `model_selection_active` now means **role == MODEL_SELECTION and metric-active**. `INTERNAL_AUDIT` is tracked separately. Worst-case selector GPU probes are selected only from optimizer-active or actual MODEL_SELECTION rows.

### 4. Queue group uniqueness existed in the current artifact but was not guaranteed

The existing real V3 queue happened to contain 200 cells from 200 distinct groups. However, the generic builder only enforced one representative per group **within each class**, so one leakage group could have been reserved again for another class.

Repository fix:

- selected groups are now excluded globally as subsequent classes are filled;
- the builder fails closed if the requested per-class balance cannot be satisfied with globally distinct groups;
- `group_uniqueness_scope = GLOBAL_ACROSS_ENABLED_CLASSES` is recorded;
- the V3 queue script asserts `len(reserved_group_ids) == queued_cells`;
- the V3 queue records source commit and validates the bound `ml_targets.parquet` hash.

The current 200-cell artifact was not shown to contain duplicates, but it must be regenerated so the durable evidence is produced by the enforcing implementation and includes the new lineage metadata.

### 5. Explicit family identifiers were not source-namespaced

V3 allowed explicit source-provided identifiers such as `project_id` and `family_id` to create grouping edges. The evidence key previously omitted the source namespace, so two independent corpora using the same local identifier could have merged.

Current accepted V3 is unaffected because the protected V3 population has **zero explicit-family edges**. This is nevertheless a correctness defect for future populations.

Repository fix:

Explicit family evidence keys are now:

```text
<source>:<field>:<value>
```

Normalized-code identity remains global authority. Exact artifact identity remains global by artifact hash. Arbitrary address literals remain diagnostic-only.

Regression coverage proves:

- same explicit ID within one source still groups;
- same explicit ID across two unrelated sources does not group;
- normalized-code identity still groups globally.

## Additional lineage hardening

The CPU selector experiment now records the same publication/binding/source lineage fields as sensitivity evidence and validates the publication `ml_targets.parquet` hash before running.

The V3 CUDA selector comparison now refuses a sensitivity report unless its dataset/grouping/partition versions, publication manifest SHA, physical binding digest, and source commit match the current run. The GPU report also records the exact sensitivity-report SHA used.

This means the affected local reports must be regenerated as one coherent tranche after synchronizing to the hardened `main`.

## What remains accepted

The hardening does **not** invalidate these accepted facts:

- repaired-v2 physical population: 22,540 contracts / 225,400 contract×class rows;
- physical representation files: 67,620;
- physical binding digest: `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`;
- positive / unknown / confirmed-negative targets: 1,080 / 224,320 / 0;
- STRONG / WEAK semantic cells: 474 / 606;
- accepted logical identifiers: `r4-leakage-groups-v3`, `r4-vnext-roles-v3`, `sentinel-r4-vnext-v3`;
- accepted V3 grouping population: 22,394 groups, max group size 7, 146 normalized-code edges, zero address-authority edges;
- current dataset had zero explicit-family edges, so source-namespacing hardening does not require rebuilding the accepted grouping artifact;
- full training remains unauthorized.

## Local regeneration required before final snapshot

First synchronize local `main` to the current remote head. Do not remove or overwrite the accepted physical repaired-v2 roots.

Then regenerate **only the affected reports** in this order.

### 1. Re-run corrected V3 acceptance reporting

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_audit_logical_v3_acceptance.py \
  > /tmp/p8_v3_acceptance_hardened.log
```

Expected decision boundary:

- status `PASS`;
- combined outcome-metric population remains 143 contracts / 142 groups;
- MODEL_SELECTION and INTERNAL_AUDIT are reported separately;
- protected-local audit expectation is 71/71 active MODEL_SELECTION and 72/71 active INTERNAL_AUDIT;
- training remains unauthorized.

If those active counts differ, stop and inspect rather than editing documentation to match an unexplained result.

### 2. Re-run lineage-bound representation sensitivity

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_profile_representation_sensitivity.py \
  --overlay data_module/data/exports/sentinel-r4-vnext-v3 \
  --representations-root data_module/data/representations-r4-v2 \
  --output data_module/data/r4-v3-logical-build/representation_sensitivity_v1.json \
  > /tmp/p8_v3_sensitivity_hardened.log
```

This may change the worst-case selector probe list if an INTERNAL_AUDIT contract previously entered the combined metric-active rankings. Review the new list rather than assuming it is byte-identical to the earlier report.

### 3. Re-run lineage-bound CPU selector comparison

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_compare_bounded_window_selector_v1.py \
  --publication-root data_module/data/exports/sentinel-r4-vnext-v3 \
  --preprocessed-root data_module/data/sentinel-preprocessed-r4-v2 \
  --representations-root data_module/data/representations-r4-v2 \
  --output data_module/data/r4-v3-logical-build/bounded_window_selector_v1.json \
  > /tmp/p8_v3_selector_hardened.log
```

The earlier 476 improved / 261 equal / 0 regressed result is historical until this lineage-bound rerun reproduces it.

### 4. Rebuild the globally unique V3 negative-review queue

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_build_confirmed_negative_review_queue_v3.py \
  > /tmp/p8_v3_negative_queue_hardened.log
```

Required:

- 200 cells if the current population still satisfies 25 × 8;
- 200 distinct reserved groups;
- all `PENDING_REVIEW`;
- all target `None`;
- all `TRAIN_UNLABELED`;
- `negative_truth_claim=false`;
- `group_uniqueness_scope=GLOBAL_ACROSS_ENABLED_CLASSES`.

Do not adjudicate before this regenerated queue is inspected.

### 5. Re-run V3 CUDA selector comparison

Only after the regenerated sensitivity report exists:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_run_selector_gpu_compare_v3.py \
  > /tmp/p8_v3_selector_gpu_hardened.log
```

The script now verifies sensitivity lineage before using its worst-case IDs.

Required:

- identical initialization true;
- all requested worst-case probes completed;
- finite bounded execution;
- no Run12 weights;
- no checkpoint;
- no selector promotion;
- no full-training authorization.

### 6. Only then create the final snapshot

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_snapshot_logical_v3_evidence.py
```

The helper must print:

```text
coherence=PASS
```

and create:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

including `snapshot_coherence_v1.json` and `SHA256SUMS.txt`.

If coherence validation fails, **do not hand-copy or edit reports to make the snapshot pass**. Fix the upstream stale/mismatched report and rerun it.

## Selector-promotion boundary

Even after a coherent final snapshot, `target_aware_guarded_v1` remains unpromoted. Before a promotion ADR, add/execute a full-population equivalence check proving that the historical control selector reproduces the currently bound representation token tensors exactly for the relevant population. Coverage/CUDA evidence alone is not sufficient to silently replace the bound extractor policy.

## Confirmed-negative boundary

R4-GAP-007 remains the principal scientific blocker. Once the coherent V3 snapshot is committed, prioritize the pilot adjudication before a full training objective decision.

Queue membership is not negative truth. Any accepted class-specific negative remains evaluation-only unless a later versioned policy grants optimizer authority.

## Stop conditions

Stop if any of the following occurs:

- corrected acceptance is not `PASS`;
- active MODEL_SELECTION / INTERNAL_AUDIT counts do not reconcile and the difference is unexplained;
- sensitivity/selector/queue/GPU report lineage does not match the current V3 publication manifest hash and physical binding digest;
- globally distinct queue groups cannot satisfy the requested class balance;
- a queue candidate has target `0` before adjudication;
- GPU sensitivity hash/lineage validation fails;
- worst-case probes are incomplete;
- the final snapshot coherence gate fails;
- any change attempts to rebuild physical repaired-v2 representations, reuse Run12 state, promote selector silently, or authorize the 100-epoch run.

## Current restart rule

This file supersedes the earlier “snapshot next” instruction in:

`runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md`.

That earlier checkpoint remains historical evidence for the pre-hardening observations. Use this hardening handoff as the active restart boundary until the regenerated coherent V3 snapshot is committed.
