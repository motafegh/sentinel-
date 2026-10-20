# Phase 8 — current-state alignment and R4-GAP-007 definition blocker

**Date:** 2026-10-20  
**Baseline before reconciliation:** `02a720f9a10f816cc16d04e6d59d0242ef748b03`  
**State after audit:** ALIGNED — M4 complete; M5 held; R4-GAP-007 semantic-definition alignment active

## Why this record exists

After R4-D-015 closed the M4 bounded pilot, several current-navigation/governance files still described M4 as pending or routed directly to candidate #2 independent review. Three subsequent external-review attempts then exposed both procedural contamination and an under-specified `CallToUnknown` class boundary.

This audit reconciled the canonical current-state surfaces so future agents do not restart completed work or treat non-authoritative reviews as candidate truth.

## Accepted current state

- Phase 8 remains `IN_PROGRESS`; G8 is open.
- DATA/representation physical completion is closed for current ML input under R4-D-011 + R4-D-013.
- R4-D-014 remains the bounded positive-only objective/evaluation authority.
- ML M0-M4 are complete.
- R4-D-015 outcome is `PASS_BOUNDED_PILOT_FULL_TRAINING_STILL_HELD`.
- M5/full training is not authorized.
- confirmed-negative targets remain zero.
- candidate #1 remains `NOT_CONFIRMED`.
- candidate #2 remains `UNKNOWN / PENDING_REVIEW / target None`.
- threshold-fit, calibration-fit and untouched-acceptance roles remain unsupported/empty.
- Run12 remains the historical operational baseline.

## Candidate #2 review-attempt disposition

The three recent external attempts are retained only as diagnostics:

1. repository-aware substantive review — procedurally contaminated by prior candidate direction in governance;
2. strict bundle-only review — correctly stopped because the v1 bundle lacked sufficient neutral class/policy material and the session also had candidate-specific prior context;
3. ZIP + neutral-material review — procedurally contaminated and returned a different result under literal broad taxonomy wording.

None is admissible as the required independent agreeing verification. Do not vote-count them.

## New active blocker

Repository text is semantically inconsistent for `CallToUnknown`:

- locked `taxonomy.yaml` includes unchecked low-level calls **or calls to unverified external addresses**;
- DIVE crosswalk says `CallToUnknown` is reserved for raw low-level `call` / `send` / `delegatecall` where the callee is unverified;
- SolidiFI crosswalk likewise explains the class through unchecked low-level `send/call/delegatecall` behavior.

This can materially change whether an address-level `transfer` is considered potentially in-scope. A reviewer must not invent precedence between these definitions.

R4-B009 / R4-R027 therefore block further candidate #2 adjudication until the neutral class-definition audit completes.

## Reconciled current-state surfaces

The audit updated:

- root `CLAUDE.md`;
- `docs/plan/ml-R4/README.md`;
- `docs/plan/ml-R4/START_HERE_AGENT.md`;
- `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
- `docs/plan/ml-R4/EVIDENCE_GAP_REGISTER.md`;
- `docs/plan/ml-R4/RISK_AND_BLOCKER_REGISTER.md`;
- `docs/plan/ml-R4/CLAIM_STATUS_MATRIX.md`;
- `docs/plan/system-finalization/technical-completion/00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md`;
- `docs/plan/system-finalization/technical-completion/02_ML_TRAINING_EVALUATION_COMPLETION_PLAN.md`;
- new active working record `docs/plan/system-finalization/technical-completion/working/2026-10-20_ml_gap007_call_to_unknown_semantic_alignment.md`.

The dated `working/2026-10-03_ml_m4_bounded_pilot.md` record is retained as historical execution evidence. Its in-progress stop lines are explicitly superseded by R4-D-015 and the reconciled current-state documents; it is not a current restart authority.

## Current restart action

Perform the neutral `CallToUnknown` definition/provenance audit independently of candidate #2 outcome.

Then, only if the class boundary is resolved:

1. record any material definition change through required ADR/schema/versioning governance;
2. build a self-contained candidate #2 blind-review v2 package;
3. obtain genuinely isolated independent review with no candidate-specific prior context;
4. formally adjudicate candidate #2;
5. reassess objective/evaluation/M5 authority only after new evidence exists.

Do not repeat M4 and do not launch full training while R4-B009 remains open.
