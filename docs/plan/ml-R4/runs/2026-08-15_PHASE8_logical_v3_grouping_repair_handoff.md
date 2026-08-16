# Phase-8 logical V3 grouping repair — local execution handoff

**Date:** 2026-08-15
**Canonical branch:** `main`
**Decision:** R4-D-009 / ADR-R4-009
**State:** COMPLETED LOCALLY on 2026-08-16; retained as execution procedure/history
**Training:** NOT AUTHORIZED

> Completion authority: `runs/2026-08-16_PHASE8_logical_v3_acceptance_and_research_checkpoint.md`.
> All stages 1–11 below completed successfully. R4-D-009 is now ACCEPTED. Stage 12 final Git-safe evidence packaging remains the next repository evidence-snapshot step; it is not a missing logical-acceptance gate.

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

**Completed:** PASS.

### 2. Corrected grouping

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py grouping
```

Expected invariant: `address_edges = 0`.

**Completed:** PASS — 22,540 artifacts, 22,394 groups, max group size 7, 146 normalized-code edges, 14,851 address literals observed, zero address-authority edges.

### 3. Role freeze and V3 publication

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py publish
```

This must recompute role-independent semantic cells and prove they are identical to the accepted V2 evidence ledger before writing V3 roles/publication.

**Completed:** PASS — population/target/strength semantics unchanged; V3 role counts recorded in the completion checkpoint.

### 4. Rebind the unchanged physical representation population

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py bind
```

Hard requirement: V3 binding digest must equal the accepted repaired-v2 physical digest because no graph/token/sidecar bytes changed.

**Completed:** PASS — 22,540 contracts / 67,620 files, digest `16dd4a3f98c34e52e5c411b39268361881efede07e8f3f52d0c060dd1c5bb6dd`, exact parent match.

### 5. Grouping audit

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py audit
```

Review the new group-count and largest-group distribution. The V2 10,327-member address-connected component must no longer exist.

**Completed:** PASS — largest V3 group = 7; no address-based grouping authority or large address-connected group remains.

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

**Completed:** PASS — all acceptance checks true; 932 effective loss cells, 143 model-selection outcome cells, confirmed negatives still zero, full training false.

### 7. Summary

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_rebuild_logical_v3.py summarize
```

**Completed:** `LOGICAL_V3_REBUILD_COMPLETE_RESEARCH_REGENERATION_PENDING` at the time of generation; subsequent stages below then completed.

## Research regeneration after V3 acceptance

The V2 negative queue and V2 role-dependent research outputs are historical population-specific evidence. Fresh V3 evidence was generated in the required order.

### 8. Representation sensitivity under V3 roles

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_profile_representation_sensitivity.py \
  --overlay data_module/data/exports/sentinel-r4-vnext-v3 \
  --representations-root data_module/data/representations-r4-v2 \
  --output data_module/data/r4-v3-logical-build/representation_sensitivity_v1.json
```

**Completed:** PASS. Physical telemetry remained consistent. Seven optimizer-active compatibility cases remain, all `TRAIN_WEAK`; MODEL_SELECTION contains zero compatibility-mode contracts. Interim sanitized evidence committed at `5e19fdf3a134ef2eb5b72df166a157c421fa811b`.

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

**Completed:** PASS — 1,018 analyzed, 737 over four windows, guarded improved 476 / equal-fallback 261 / regressed 0; median target coverage ~63.01% historical vs ~87.94% guarded. Interim summary committed at `a51f28e0684f63cec69af2e76efcfc518035a21a`.

Selector promotion remains prohibited until a separate explicit promotion ADR/versioned extractor decision.

### 10. New confirmed-negative pilot queue

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_build_confirmed_negative_review_queue_v3.py
```

**Completed:** PASS — 200 cells, 25 per enabled class, 200 reserved groups, all `PENDING_REVIEW`, all target `None`, all `TRAIN_UNLABELED`, `negative_truth_claim=false`. R4-GAP-007 now governs any future adjudication. Do not manually adjudicate the obsolete V2 queue.

### 11. Identical-initialization CUDA selector comparison + mandatory worst-case probes

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=.:data_module \
./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_run_selector_gpu_compare_v3.py
```

Unlike the earlier V2 run, this V3 launcher fails if worst-case probes were requested but the sensitivity report is absent/empty or the expected probes are not completed.

**Completed:** PASS — RTX 3070 Laptop GPU, BF16, identical initialization verified, 4 train + 4 selection batches per strategy, 4/4 mandatory worst-case probes completed, no Run12 weights, no checkpoint. Guarded positive-only model-selection NLL was 0.66014 vs 0.68474 control; peak allocated memory 956.68 MB vs 967.36 MB control. These are bounded positive-only research results, not discrimination evidence or training authorization.

### 12. Git-safe evidence snapshot

This is the remaining packaging step before moving on:

```bash
PYTHONPATH=.:data_module ./ml/.venv/bin/python \
  docs/plan/ml-R4/scripts/p8_snapshot_logical_v3_evidence.py
```

The helper sanitizes repository-local paths, snapshots small decisive reports, summarizes the large selector report instead of committing its per-contract payload, and writes SHA-256 bindings.

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

Stages 1–11 encountered none of these stop conditions.

## Rollback

An unsuccessful later V3-derived experiment is rolled back by selecting/removing only that new versioned/generated artifact. Never modify accepted repaired-v2 physical artifacts, accepted V3 logical authority, or historical V1/V2 evidence in place.

## Current decision boundary

Logical V3 is accepted under R4-D-009. Physical repaired-v2 remains the reusable source/representation root. V3 selector/representation/negative-queue/CUDA research has been regenerated and reviewed. The guarded selector is evidence-ready for a separate promotion decision but not yet promoted. R4-GAP-007 authorizes confirmed-negative pilot review but adjudication has not started. Full Phase-8 training remains unauthorized and G8 remains open.
