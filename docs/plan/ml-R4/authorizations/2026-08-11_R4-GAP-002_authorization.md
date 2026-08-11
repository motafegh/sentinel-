# R4-GAP-002 Authorization Record

- **Gap ID:** `R4-GAP-002`
- **Phase:** R4 Phase 4 — Targeted Evidence-Gap Adjudication
- **Date:** 2026-08-11
- **Status:** APPROVED
- **Gap reason:** `UNREVIEWED_SOURCE_CLASS_STRATUM`
- **Approval mode:** delegated technical/governance approval

## Owner delegation

The human owner explicitly delegated routine technical and project-governance approvals to the AI assistant and instructed it to choose and execute the technically strongest option based on project evidence.

This authorization therefore records an owner-delegated technical decision, not an assumption of silent consent. The delegation does not permit fabricated evidence, irreversible external actions outside the approved project workflow, or value/policy decisions that cannot be resolved from engineering evidence.

## Approved decision problem

Determine the first DATA vNext role of five active DIVE-positive source strata whose source-specific semantic precision is not established:

| DIVE native stratum | Canonical class | Locked v9 index |
|---|---|---:|
| `DoS` | `DenialOfService` | 1 |
| `Arithmetic` | `IntegerUO` | 4 |
| `Time manipulation` | `Timestamp` | 7 |
| `Front Running` | `TransactionOrderDependence` | 8 |
| `Unchecked Return Values` | `UnusedReturn` | 9 |

Allowed role outcomes are: retain as higher-authority evidence only if supported; retain as weak signal if justified; otherwise mask or exclude. Failure of a DIVE-positive assertion does **not** create a confirmed negative.

## Explicitly outside this authorization

- DIVE `Bad Randomness` remapping;
- Web3Bugs acquisition;
- BCCC import or Stage 5.5 propagation;
- general Slither/Aderyn consensus benchmarking;
- Echidna/fuzzing program;
- exploit-PoC program;
- model architecture changes;
- threshold or calibration changes;
- historical DATA artifact mutation.

## Frozen semantic rubric

The review rubric is the class-name/current-index semantic reconciliation in:

`docs/plan/ml-R4/findings/05_phase4_gap_entry_and_class_definition_reconciliation.md`

Recovered BCCC class-definition prose may inform semantics, but its stale historical numeric class IDs are not authoritative. The locked v9 schema is authoritative.

## Initial review design

The initial blind review is a deterministic **screening batch**, not a fixed final sample size:

- 20 DIVE-positive contracts per approved stratum;
- 100 total initial contracts;
- historical TRAIN groups only;
- any group touching historical val/test is excluded from initial review;
- review-group precedence: `project_group_id` → `dedup_group_id` → `contract_id`;
- no review group reused across the five strata;
- deterministic SHA-256 ranking bound to `R4-GAP-002` and the committed Phase-3 ledger SHA;
- model predictions, tiers, tool votes, merger outcome, and non-target historical labels hidden from initial semantic review where practical.

Twenty per stratum is intentionally only a first screening batch. If evidence is clearly inadequate for a trusted role, the project may stop and mask/exclude rather than spend more review. If a stratum appears strong enough for a higher-authority role, the sample must expand adaptively and second review is required before promotion.

## Stop conditions

Stop rather than broaden scope if:

- the committed Phase-3 ledger identity changes;
- deterministic population/sample identity cannot be established;
- class semantics require a new taxonomy/policy decision;
- a proposed high-authority role lacks independent review support;
- work drifts into non-authorized sources, model changes, thresholds, or calibration.
