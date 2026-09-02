# ADR-R4-011 — V10 V2.6 physical representation acceptance and training hold

Date: 2026-09-02
Status: ACCEPTED
Decision ID: R4-D-011
Scope: R4 physical graph representation lineage and Phase-8 launch boundary

## Context

R4-D-010 withdrew graph schema v9 from eligibility for the new full training
run and required a separately versioned V10 lineage with corrected call-kind
semantics, complete physical binding, source-to-graph reconciliation, and an
explicit acceptance record.

The V2.3 lineage became the frozen structural reference. V2.4 closed the
26-contract parse-only tail. V2.5 added deterministic persistent-storage WRITE
classification but its full audit exposed a larger population than the original
20-case evidence. V2.6 narrowly added persistent-storage collection
`push`/`pop` recognition while preserving call-node priority and excluding
memory receivers and arbitrary member calls.

Fresh V2.6 construction and evidence now satisfy R4-D-010's physical
requirements. The exact candidate contains 22,540 identities and 67,620 files,
preserves every accepted-V9 token byte, has no missing/extra/invalid artifacts,
uses the required 22,539+1 runtime split, and binds to digest
`d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`.
Three fresh generations and three stable semantic-evidence passes cover the
actual 355-identity structural-drift population. V4 independently re-proves 349
persistent-storage WRITE corrections and 6 index-equivalent graphs, leaving
zero unexplained drift.

## Decision

Accept the exact protected-local root
`data_module/data/v10-v26-full-candidate-attempt-2026-09-01-a/representations-r4-v3-candidate`
as the immutable V10 physical representation lineage under extractor
`v2.6-r4-call-semantics-deterministic-cfg-mutators` and graph schema `v10`.

This acceptance grants only physical representation authority for controlled
research and possible later training eligibility. It does not change logical
V3 grouping/roles, label truth, token-selector authority, objective semantics,
threshold/calibration support, model architecture, or production status.

The fixed 100-epoch Phase-8 run remains not authorized and G8 remains open.

## Consequences

- R4-B008 is closed for this exact root and digest.
- R4-GAP-008 is resolved; its historical V2.5 blocker and all intermediate
  diagnostic roots remain preserved.
- The accepted V2.6 root is immutable. Any later graph semantic or token
  selector change requires a new versioned root, binding, and acceptance.
- V9 remains eligible only for historical reproduction, not a new full run.
- Full training still requires separate closure of negative-evaluation,
  selector, objective, threshold/calibration, and launch-authority gates.
- Diagnostic reports remain evidence records and retain their original false
  acceptance/training flags; this ADR plus the machine-readable R4-D-011 record
  supplies the governance decision.

## Rollback

Rollback is artifact selection among immutable prior lineages. Never overwrite
the accepted V2.6 root or reinterpret a prior diagnostic lineage as current
training authority.

## Evidence

- `runs/2026-09-02_PHASE8_v10_v26_physical_acceptance_and_no_launch.md`;
- `evidence/2026-09-02_v10_v26_physical_acceptance/acceptance.json`;
- `runs/2026-08-30_PHASE8_v10_v25_full_population_structural_analysis.md`;
- protected-local refreshed binding report SHA-256
  `93a4d15e0793d7b144fc5cc98dbd29627f0d7372cb56e2431a79f8d02c761311`;
- protected-local refreshed V4 audit SHA-256
  `1e037c5d22dbe03a7b5f303ea3c3fa11facb17fcf751c9f7432197e194a2a994`;
- accepted source commit
  `012a3359449865e3a8ab2a3d4a46ac4859bb6cb6`.
