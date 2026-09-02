# Phase-8 V10 V2.6 physical acceptance and no-launch decision

Date: 2026-09-02
Decision ID: `R4-D-011`
Physical decision: `ACCEPTED_IMMUTABLE_LOCAL_PHYSICAL_REPRESENTATION`
Training decision: `NOT_AUTHORIZED`
Gate: G8 remains open

## Outcome

Accept the exact protected-local graph-schema-V10 representation root
`data_module/data/v10-v26-full-candidate-attempt-2026-09-01-a/representations-r4-v3-candidate`
as the immutable physical representation lineage for controlled R4 research and
possible future-training eligibility under later gates. The accepted extractor
is `v2.6-r4-call-semantics-deterministic-cfg-mutators`; the binding digest is
`d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`.

This decision does not authorize training. It does not promote the target-aware
selector, create confirmed-negative truth, approve a positive-unlabeled
objective, fit thresholds/calibration, populate untouched acceptance, or reuse
Run12 learned state.

## Source-first acceptance review

The acceptance review ran against published source commit
`012a3359449865e3a8ab2a3d4a46ac4859bb6cb6`, which was equal on local `main`,
`origin/main`, and the remote `main` branch before review. Git had zero tracked
changes. Pre-existing unrelated untracked user files remained outside the
review and were not used as evidence.

The refreshed binder independently checked every candidate artifact and
reproduced:

- 22,540 accepted-V9 identities and 22,540 candidate identities;
- 67,620 graph/token/sidecar files;
- zero missing, extra, invalid, or failure-ledger artifacts;
- 22,540 token files byte-identical to accepted V9;
- exact runtime split: 22,539 Slither 0.10.0 primary identities and one declared
  Slither 0.11.5 identity-bound exception;
- binding digest
  `d9f925588913e66476cfbc097bace7daa7e673295fe2a243760313d0bef5ebdd`.

The refreshed V4 audit then independently rechecked all 22,540 transitions and
reproduced:

- 355 raw non-parse-only structural-drift identities;
- 349 proven persistent-storage WRITE corrections;
- 6 exact node-index-invariant graph equivalences;
- three stable exact-runtime graph generations and three byte-identical
  semantic-evidence reports;
- zero unapproved, missing, failed, or unexplained structural identities;
- status `PASS_TRANSITION_EVIDENCE_RECONCILED_PENDING_PHYSICAL_DECISION`.

All 13 implementation hashes embedded in the refreshed audit matched the files
at the published commit. Additional decisive source hashes were recorded in the
local review output. The audit report's `repository_worktree_dirty=true` is
explained solely by preserved untracked user files; the tracked worktree count
was zero.

## Evidence bindings

| Evidence | SHA-256 |
|---|---|
| Stage-A primary report | `46f63d24ed614a6dfd427c3c6c19512578e9c5092e05300a3fb1445002e753cf` |
| Stage-B primary-stage report | `85286706b189fa09d06e4113aeeb4168bb283a9bd0ce6bdd8e64b862ae4cb41f` |
| Stage-C exception-fill report | `3df8583b0929086b5ef9a7d4135499fa47f2989e1115f7a8d3e0dab2ef1f15bc` |
| Refreshed candidate binding report | `93a4d15e0793d7b144fc5cc98dbd29627f0d7372cb56e2431a79f8d02c761311` |
| Semantic evidence, each of three reports | `92afb95f5335226ee28c99969779af0dd5f69da4296cf400736ff4c4e75bce42` |
| Full-population probe | `9a1cf96465613b61fae2d10ccaa81def0548663a4c4711ca745841f6354e7a55` |
| Refreshed current-commit V4 audit | `1e037c5d22dbe03a7b5f303ea3c3fa11facb17fcf751c9f7432197e194a2a994` |

The machine-readable decision is
`evidence/2026-09-02_v10_v26_physical_acceptance/acceptance.json`.

## Acceptance meaning and immutability

Acceptance means this exact root and digest are physically complete,
internally bound, semantically reconciled against the frozen V2.3 structural
reference, and eligible to serve as the V10 physical parent for later bounded
research. The accepted root must not be regenerated, patched, renamed in place,
or silently rebound. A future semantic or selector change requires a new
versioned lineage and new acceptance evidence.

The diagnostic binder and V4 reports intentionally retain
`physical_acceptance=false`: those tools provide evidence and cannot grant
governance authority. R4-D-011 and its machine-readable acceptance record are
the authority that changes the physical state to accepted.

## Why G8 and training remain open

Physical representation integrity is only one Phase-8 prerequisite. The
remaining blockers are independent:

1. confirmed-negative evaluation support remains zero; candidate #2 still
   requires genuinely independent agreement;
2. the target-aware four-window selector remains unpromoted;
3. no evidence-honest supervised-negative or positive-unlabeled objective has
   been accepted;
4. threshold-fit, calibration-fit, and untouched-acceptance populations remain
   unsupported or empty;
5. no explicit full-training authorization exists.

Consequently, no checkpoint, optimizer/scheduler horizon, 100-epoch run, model
quality claim, or production acceptance follows from this decision.

## Rollback

Rollback means selecting a prior immutable hash-bound lineage for historical
reproduction. It does not mean modifying this accepted V2.6 root, restoring v9
eligibility for a new full run, or authorizing training from a historical
artifact.
