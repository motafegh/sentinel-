# Phase-8 logical V3 grouping repair — local execution handoff

**Date:** 2026-08-15  
**Canonical branch:** `main`  
**Decision:** R4-D-009 / ADR-R4-009  
**State:** repository implementation complete; local logical V3 generation/acceptance pending  
**Training:** NOT AUTHORIZED

## Purpose

The repaired-v2 physical DATA remains accepted, but full-population evidence showed `r4-leakage-groups-v2` over-connects unrelated contracts through arbitrary same-source Ethereum address literals. One DIVE component contains 10,327 contracts and is driven by ubiquitous addresses such as the Uniswap V2 router, dead/zero addresses, WETH, and other common constants.

This handoff rebuilds **only the logical lineage**:

- grouping;
- role partition;
- DATA publication rows/manifests;
- physical binding report for the new logical publication;
- negative-review reservations;
- selector/sensitivity research populations.

It does **not** rebuild raw sources, preprocessing, Slither graphs, GraphCodeBERT token tensors, or sidecars.

## Immutable physical inputs reused

- preprocessing: `sentinel-preprocessed-r4-v2`;
- source/provenance claims: repaired-v2 `source_claims.jsonl`;
- role-independent semantic evidence: `evidence-ledger-r4-v2`;
- representations: `representations-r4-v2`;
- extractor: `v2.2-r4-repaired`;
- graph schema: `v9`;
- token tensor shape: `[4,512]`;
- parent physical representation binding digest: `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`.

Do not remove or overwrite these inputs.

## New logical lineage

- grouping: `r4-leakage-groups-v3`;
- partition: `r4-vnext-roles-v3`;
- publication: `sentinel-r4-vnext-v3`;
- logical build: `r4-logical-lineage-v3`.

Generated local roots:

- `data_module/data/r4-v3-logical-build/`;
- `data_module/data/exports/sentinel-r4-vnext-v3/`.

Both are generated/local evidence. Historical V1/V2 roots remain immutable.

## Mandatory grouping policy

V3 grouping authority:

- normalized-code identity;
- explicit source family/project identifiers;
- exact artifact identity.

Ethereum address literals are **diagnostic only**. V3 must contain zero `same_source_shared_address_candidate` union edges.

## Local execution order

Use the existing accepted R4 ML virtual environment and a synchronized clean tracked worktree. The five pre-existing untracked audit/plan files do not fail the tracked-clean gate.

### 1. Prerequisites

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py prerequisites
```

Must pass before generated V3 output is written.

### 2. Corrected grouping

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py grouping
```

Expected invariant: `address_edges = 0`.

### 3. Role freeze and V3 publication

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py publish
```

This must recompute role-independent semantic cells and prove they are identical to the accepted V2 evidence ledger before writing V3 roles/publication.

### 4. Rebind the unchanged physical representation population

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py bind
```

Hard requirement: V3 binding digest must equal the accepted repaired-v2 physical digest because no graph/token/sidecar bytes changed.

### 5. Grouping audit

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py audit
```

Review the new group-count and largest-group distribution. The V2 10,327-member address-connected component must no longer exist.

### 6. V2→V3 logical acceptance gate

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_audit_logical_v3_acceptance.py
```

This gate must prove:

- contract/cell population unchanged;
- target/strength semantics unchanged;
- zero confirmed negatives;
- address grouping authority disabled;
- zero address union edges;
- giant V2 group removed;
- all physical files still validate;
- physical representation binding digest unchanged.

It also reports the new active optimizer groups and planning-only batch-8/accum-8 step arithmetic. Those numbers are **not** a training authorization.

### 7. Summary

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py summarize
```

At this point the logical V3 correction may be reviewed. Do not start training.

## Research regeneration after V3 acceptance

The V2 negative queue and V2 role-dependent research outputs become historical population-specific evidence. Generate fresh V3 evidence in this order.

### 8. Representation sensitivity under V3 roles

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_profile_representation_sensitivity.py \
  --overlay data_module/data/exports/sentinel-r4-vnext-v3 \
  --representations-root data_module/data/representations-r4-v2 \
  --output data_module/data/r4-v3-logical-build/representation_sensitivity_v1.json
```

### 9. Selector population comparison under V3 roles

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_compare_bounded_window_selector_v1.py \
  --publication-root data_module/data/exports/sentinel-r4-vnext-v3 \
  --preprocessed-root data_module/data/sentinel-preprocessed-r4-v2 \
  --representations-root data_module/data/representations-r4-v2 \
  --output data_module/data/r4-v3-logical-build/bounded_window_selector_v1.json
```

Selector promotion remains prohibited even if coverage improves.

### 10. New confirmed-negative pilot queue

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_build_confirmed_negative_review_queue_v3.py
```

Do not manually adjudicate the obsolete V2 queue. V3 candidates remain UNKNOWN until class-specific evidence and independent review confirm otherwise.

### 11. Identical-initialization CUDA selector comparison + mandatory worst-case probes

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_run_selector_gpu_compare_v3.py
```

Unlike the earlier V2 run, this V3 launcher fails if worst-case probes were requested but the sensitivity report is absent/empty or the expected probes are not completed.

It must not load Run12 weights or write a checkpoint.

### 12. Git-safe evidence snapshot

Only after stages 1–11 complete successfully:

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_snapshot_logical_v3_evidence.py
```

The helper sanitizes repository-local paths, snapshots small decisive reports, summarizes the large selector report instead of committing its per-contract 10s-of-MB payload, and writes SHA-256 bindings.

Commit only:

`docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/`

Do not force-add generated DATA roots.

## Stop conditions

Stop and preserve evidence if any of these occurs:

- V3 semantic rows differ from the accepted role-independent V2 evidence ledger;
- any address-authority grouping edge appears;
- the V3 physical binding digest differs from repaired-v2;
- any representation triple fails validation;
- a V3 queue contains target `0` before explicit adjudication;
- the CUDA comparison loses identical initialization;
- requested worst-case probes do not all complete;
- any script attempts to change graph schema, token shape, model architecture, Run12 weights, threshold/calibration roles, or untouched acceptance.

## Rollback

An unsuccessful V3 attempt is rolled back by archiving/removing **only** the generated V3 roots and rerunning from a clean commit. Never modify accepted V2 physical artifacts or historical V1/V2 evidence.

## Current decision boundary

Repository tooling implements the corrected grouping/partition lineage. Physical repaired-v2 remains the reusable source/representation root. Full Phase-8 training remains unauthorized until V3 is accepted locally and the regenerated evaluation/selector evidence is reviewed.
