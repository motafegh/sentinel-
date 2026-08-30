# Phase-8 V10 V2.5 current restart checkpoint

Date: 2026-08-27; execution result updated 2026-08-30
Status: STAGES A-D PASS; STAGE E FULL-POPULATION STRUCTURAL EVIDENCE BLOCKER
Scope: current R4-B008 / V10 physical-candidate restart authority only; no physical acceptance or training authority

## Current authoritative state

Historical R4 G0-G7 remain passed and immutable. Phase 8 remains `IN_PROGRESS` and G8 is not passed.

R4-D-008 accepted repaired-v2 physical DATA as immutable reproducibility evidence. R4-D-009 accepted logical V3 grouping/roles/publication. R4-D-010 withdrew graph schema v9 from eligibility for the new full training lineage and requires a separately versioned V10 physical candidate before any later training decision.

The original V10 `v2.3-r4-call-semantics` diagnostic lineage is historical structural reference evidence, not the current candidate extractor. The 26-contract parse-only remediation is complete under the later compatibility lineage; the protected V2.4 candidate has 22,540 identities, exact accepted-V9 token bytes, zero parse-only artifacts, zero unclassified call IR, and the required 22,539 primary + 1 identity-bound runtime split. That V2.4 candidate remains diagnostic history and is not physically accepted.

The later 20-contract unexpected structural-drift investigation is also complete. Under extractor `v2.5-r4-call-semantics-deterministic-cfg` and exact primary Slither 0.10.0, three fresh bounded generations resolved all 20 identities with zero unexplained drift:

- 8 identities: exact node-index-invariant labelled directed-multigraph equivalence through unchanged edge type 10;
- 12 identities: deterministic persistent-storage `CFG_NODE_WRITE` corrections backed by exact expression-level storage-write evidence;
- final bounded verifier: `bounded_v25_reproducibility_passed = true`;
- `zero_unexplained_drift = true`;
- blockers: none.

The bounded evidence chain was regenerated into persistent protected-local
roots after the restart audit found its original merged semantic input had been
lost with `/tmp` and exposed nondeterministic informational ordering. The
semantic probe and Stage-B deferral validation are now hardened, and the
replacement chain is cryptographically linked and validated:

- source transition audit SHA-256: `5793b059e7e5149424e10a5361a5b0e420b1f86f3630920e36344c5737fd4f9b`;
- bounded V2.5 report SHA-256: `67192b2a81383af74f70ed3ed6e1c0dfbd50d6b9525a9a939a250653e2a53adc`;
- merged semantic evidence SHA-256: `16e264fbed941ab16ead47dacd4e19c7a02511539e0950664e2cdc28373bfa8e`;
- evidence-chain preflight SHA-256: `1d28f9b2f4a597ff04f62052cad95713dafd6169f5d0f97de100fde452e542cb`;
- persistent evidence root: `data_module/data/r4-v10-v25-evidence-deterministic-v2`;
- persistent repeat roots: `data_module/data/r4-v10-v25-bounded-repeat-{1,2,3}`;
- focused hardening/evidence tests and evidence-chain preflight passed.

## Current physical-candidate construction boundary

The next candidate must use:

- graph schema: `v10`;
- extractor: `v2.5-r4-call-semantics-deterministic-cfg`;
- final candidate basename: `representations-r4-v3-candidate`;
- primary runtime: Slither 0.10.0 / crytic-compile 0.3.11;
- exact identity-bound exception: `dive/caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9` under Slither 0.11.5;
- accepted token bytes copied exactly from `representations-r4-v2`.

A single full-generation process cannot produce the required heterogeneous runtime split safely. The approved staged build is therefore:

1. **Stage A — primary attempt:** `p8_generate_v10_v25_primary_attempt.py` generates exactly 22,539 ordinary identities under Slither 0.10.0 and records the one runtime exception as `IdentityBoundRuntimeDeferred` without invoking extraction for it.
2. **Stage B — primary staging:** `p8_stage_v10_v25_primary_attempt.py` fail-closed validates and transfers only the 22,539 proven primary triples into a fresh final-lineage root.
3. **Stage C — exception fill:** generate exactly the declared exception under Slither 0.11.5 into the staged root.
4. **Stage D — bind:** ordinary V10 binder must prove 22,540 identities, token-byte equality, V10/V2.5 graph+sidecar identity, zero degraded analysis, call-count reconciliation, and exact 22,539 + 1 runtime distribution.
5. **Stage E — full V3 transition audit:** `p8_audit_v10_transition_v3.py` must preserve all V2 mechanical checks and re-prove the exact 8+12 bounded structural evidence classes against the actual full V2.5 candidate while rejecting any new non-parse-only drift.
6. Only after explicit review of the complete V3 report may a separate physical-acceptance decision record be considered.

## Protected-local readiness already proven

On 2026-08-27, before Stage A:

- staging/full-gate scripts compile;
- semantic evidence regeneration is byte-stable across independent randomized processes;
- the persistent replacement bounded report passes 8+12 with zero blockers;
- the replacement SHA-bound evidence-chain preflight passes;
- full-gate focused tests pass 12/12;
- Stage-A + staging focused tests pass 9/9;
- accepted V9 population = 22,540;
- repaired-preprocessed population = 22,540;
- ordinary primary population = 22,539;
- runtime exceptions = 1;
- primary environment = Slither 0.10.0;
- exception environment = Slither 0.11.5;
- protected V2.4 candidate and frozen V2.3 structural-reference roots are present.

This was the pre-execution boundary. The protected-local execution result is recorded below and supersedes this sentence for current status.

### 2026-08-30 interrupted Stage-A continuation

A four-worker Stage-A attempt was cleanly stopped for host shutdown before its
final report.  The protected root
`data_module/data/v10-v25-primary-attempt-2026-08-28-214ca542d-retry3-w4/representations-r4-v3-candidate`
contains 13,974 apparent complete file triples and no incomplete set by suffix
inventory.  This is not a passed attempt.  Continue only through the
fail-closed opt-in resume protocol in the full-candidate staging plan: validate
every existing payload and binding, quarantine incomplete bytes if found,
generate only the remaining ordinary identities, and recompute the complete
Stage-A evidence.  Stage B remained blocked until the resumed report passed.

### 2026-08-30 protected-local execution result

The fail-closed resume and subsequent stages completed without mutating accepted
or historical roots:

- Stage A passed with 22,539/22,539 ordinary identities, 15,211 reused only
  after full validation, 7,328 generated in the successful resume, zero
  incomplete quarantines, one declared exception deferred, and zero unexpected
  failures; report SHA-256
  `a227a3a6d2340c7f3ab3bb15687fad4002f66c29ebd05d64108f7dce13deeb76`;
- Stage B passed with 22,539 staged identities and 67,617 hardlinks; records
  digest `a1b3ff1d00a66532076f14e72661e21e3989984f6a5e6cf0ad79c8cb8613fb8a`;
  report SHA-256
  `3f24b4b294340580cf4579cc5b7c230b6c803cabe7234aacc31f6a3d0bf4fdf0`;
- Stage C filled only
  `dive/caa35c1a5906269bbe5e70de780d105c2968ece4fc038d7f7208efee681aeec9`
  under Slither 0.11.5; report SHA-256
  `0994b8921905b1db82f01f5f16868a85252cc432cac615c85f73692107a301d8`;
- Stage D passed 22,540/22,540 identities, zero missing/extra/invalid, exact
  accepted-V9 token bytes, and the 22,539 Slither-0.10 + one Slither-0.11.5
  split; binding digest
  `17c5f334c75015fdaf89b1a9f77522af5185f2485c24df4e1e64917dc944f021`;
  report SHA-256
  `3cab4b19d7708b8d706699577dbfcaebf504b6ceb918c60a21956441fa238774`.

Stage E then correctly failed closed. Base mechanics passed, but the full audit
found 311 raw non-parse-only structural drifts and only the original 8 index +
12 storage-WRITE bounded cases were eligible for exact re-proof. The audit
leaves 298 identities unapproved and records
`PASS_BASE_MECHANICS_WITH_STRUCTURAL_EVIDENCE_BLOCKER`,
`physical_acceptance=false`, and `training_authorized=false`; report SHA-256
`b469e63e91e22b75eea1f66432e7cbddf4461289b32c4f77ddf7bba39f82031f`.

A full-population diagnostic probe classified the 311 raw identities as 298
feature/metadata classification drifts, 12 proven node-order/index cases, and
one semantic-structure drift
(`dive/bfa512a7a831999fa8140cd667e84524d3e01b09fb3cb258955f09b680863d62`).
It found 895 uniquely identifiable semantic node diffs across 183 contracts;
128 contracts require duplicate-safe node matching. Probe SHA-256:
`d9c512015d180c67fee6dc8848952992914abed07f373ebdc7845a3398b1b3b4`.

The current continuation authority is
`2026-08-30_PHASE8_v10_v25_full_population_structural_evidence_plan.md`.
Stages A-D must not be repeated merely to bypass the false Stage-E gate.

## Current restart order

Read in this order:

1. `PLAN_STATUS_MATRIX.md`;
2. this checkpoint;
3. `runs/2026-08-30_PHASE8_v10_v25_full_population_structural_evidence_plan.md`;
4. `reviews/R4-GAP-008/2026-08-26_v10_v25_bounded_structural_closure.md`;
5. `runs/2026-08-26_PHASE8_v10_v25_full_candidate_staging.md`;
6. `adrs/ADR-R4-010-versioned-external-call-representation-correction.md` for the original accepted semantic decision;
7. `runs/2026-08-16_PHASE8_v3_hardened_evidence_snapshot_closeout.md` for the accepted pre-pilot V3 baseline;
8. `runs/2026-08-21_PHASE8_gap007_candidate2_primary_review.md` for the separate negative-evidence track.

The August 21 V10 implementation handoffs, the parse-only working plan, and the August 23 structural-drift handoff are historical execution records. They must not be used as the current restart boundary when they describe intermediate blockers already closed.

## Separate open gates

R4-GAP-007 remains open: candidate #1 is `NOT_CONFIRMED`; candidate #2 primary review supports a negative but independent agreement is still required. Confirmed negatives remain zero.

Selector promotion remains unauthorized and still requires its separate bound-token control-equivalence evidence and decision.

Threshold fitting, calibration fitting, and untouched acceptance remain unsupported/empty.

## Stop lines

- Do not overwrite accepted V9/repaired-v2 artifacts, the frozen V2.3 structural reference, or the protected V2.4 diagnostic candidate.
- Do not run population-wide Slither 0.11.5.
- Do not relabel or manually edit graph/sidecar runtime or extractor fields.
- Do not treat bounded V2.5 success as physical acceptance.
- Do not infer target `0` from unlabeled/source/tool absence.
- Do not promote the selector as part of V10 construction.
- Do not launch Phase-8 full training. Training remains unauthorized until separate later governance explicitly permits it.
