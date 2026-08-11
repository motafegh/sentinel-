# SENTINEL V3 Runtime + Feedback Alignment — Execution Plan

**Date:** 2026-08-11  
**Branch:** `system/v3-runtime-feedback-alignment`  
**Base:** canonical `main` after ZKML/contracts V3 alignment (`2c7dff98f7d8cd66140c622e3f2789292f4c78a6`)  
**Status:** IN_PROGRESS

## Goal

Align the remaining AGENTS-side runtime truth and feedback interfaces with the newly canonical V3 registry protocol **without inventing model policy**.

The previous alignment established:

- V3 EIP-712 context-attested submission semantics;
- fail-closed/read-only audit MCP runtime boundary;
- V1/V2/V3 registry observability;
- canonical ZKML proof semantics and verifier integration.

Two adjacent AGENTS surfaces remain historically V1/R0-shaped:

1. `agents/src/contracts/submission.py` — R0 proof/finality compatibility truth schema;
2. `agents/src/ingestion/feedback_loop.py` — V1 scalar `AuditSubmitted` event → threshold → RAG bridge.

This track will make those boundaries version-aware and fail-closed while preserving the explicit R4 policy boundary.

## Non-goals / stop lines

This branch MUST NOT:

- create or store a private signing key;
- construct/sign/broadcast a production V3 transaction;
- claim a KMS/HSM signer exists;
- deploy contracts or rotate live roots;
- retrain ML/ZKML artifacts;
- change DATA labels, ML thresholds, calibration or model-selection policy;
- reuse the V1 scalar `5734` cutoff for V3;
- invent a per-class V3 feedback threshold;
- treat raw V3 class-score field elements as calibrated probabilities;
- mark R4 Phase 3/G3 passed;
- start R4 Phase 4.

## Workstream A — submission truth schema

### A1. Reconstruct current consumers

Trace every import/use of `agents/src/contracts/submission.py` and classify which fields are:

- historical R0 compatibility;
- live report/API contract;
- V3-incompatible;
- safe to extend without changing policy.

### A2. Add versioned V3 truth representation

If source evidence supports it, add a distinct V3/context-attested submission truth type rather than overloading old R0 proof-scope fields.

Minimum V3 truth should distinguish:

- proof verification state;
- context-attestation/policy state;
- signer state;
- transaction/broadcast/finality state;
- exact request digest;
- chain/registry identity;
- model/proxy/DATA/schema identities;
- explicit unavailable/ineligible reasons.

No state may silently collapse `not attempted`, `policy rejected`, `unavailable`, and `confirmed` into one boolean.

## Workstream B — V3 feedback event observation

### B1. Separate observation from promotion

The feedback loop should be able to **observe and decode** `AuditSubmittedV3` without implying the event is automatically eligible for RAG/training feedback.

### B2. Preserve historical V1 compatibility

Existing V1 event ingestion may remain for historical/replay purposes, but it must be explicitly versioned and must not be confused with V3 context-attested finality.

### B3. V3 event identity

A V3 observation should retain enough identity to correlate to the exact registry record/request:

- transaction hash / block / log index where available;
- target contract;
- request digest;
- round ID;
- teacher/proxy/DATA/schema hashes;
- policy signer/verifier identity if recoverable from registry record;
- class-score field elements.

## Workstream C — feedback eligibility boundary

### C1. No legacy threshold reuse

The branch will explicitly reject automatic V3 RAG promotion until a versioned policy is supplied by future R4/model-evaluation work.

### C2. Structured ineligibility

V3 observations may be returned/stored as structurally valid observations with an explicit state such as `POLICY_UNAVAILABLE` / `NOT_EVALUATED`, but never silently dropped and never promoted under the V1 threshold.

### C3. Future injection point

Define a narrow policy interface so later measured R4 acceptance/calibration logic can decide whether a V3 observation is eligible for RAG/training feedback without rewriting chain-event decoding.

## Workstream D — tests and CI

Add dependency-light tests for:

- V1 vs V3 event decoding;
- V3 request-digest identity retention;
- no V3 use of the V1 `5734` threshold;
- explicit not-evaluated/ineligible result rather than silent skip;
- versioned submission truth invariants;
- fail-closed malformed-event handling;
- no private-key / signing / broadcast capability in the analysis feedback path.

Use GitHub Actions for remotely reproducible checks where possible.

## Workstream E — documentation / closeout

Update only source-adjacent references that are demonstrably stale. Record:

- what the branch makes structurally V3-ready;
- what remains blocked on R4 model policy;
- what remains blocked on a real isolated signer/live network;
- exact local/live checks still required later.

## Acceptance boundary for this branch

This branch may be called complete when:

1. AGENTS submission truth can represent V3 context-attested states without lying about signing/finality;
2. `AuditSubmittedV3` can be decoded/observed with exact identity retained;
3. V3 observations cannot enter RAG through the old scalar threshold;
4. missing V3 feedback policy is explicit and fail-closed, not a silent skip;
5. tests prove the separation between historical V1 feedback and V3 observations;
6. no private-key/live transaction capability was introduced;
7. remote tests are green.

Completion means **V3 runtime/feedback source interfaces are structurally aligned**. It does not mean V3 feedback policy is measured, a production signer is deployed, or live end-to-end finality has been accepted.
