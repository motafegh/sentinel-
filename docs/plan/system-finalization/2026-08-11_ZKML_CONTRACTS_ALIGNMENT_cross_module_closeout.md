# SENTINEL ZKML + Contracts Alignment — Cross-Module Closeout

**Date:** 2026-08-11  
**Branch:** `system/zkml-contracts-alignment`  
**Purpose:** record the final cross-module compatibility pass after the core ZKML/contracts V3 work.

## V3 registry observability — CLOSED

A fail-closed write boundary is not sufficient if the live read path cannot see the records produced by the new protocol.

The pre-alignment audit MCP queried only the historical V1 mapping:

- `getLatestAudit`
- `getAuditHistory`
- `hasAudit` / `getAuditCount`

and decoded only the V1 scalar tuple.

That was functionally stale because `AuditRegistry` already has separate V1, V2 and V3 storage/query surfaces.

The live read-only audit MCP is now version-aware while preserving the same three protocol-neutral tool names:

### `get_latest_audit`

Returns the newest persisted record across V3, V2 and V1 **by on-chain timestamp**. It does not simply prefer the numerically highest protocol version.

The response exposes:

- `protocol_version`;
- aggregate `total_count`;
- `counts_by_protocol`;
- the protocol-specific persisted fields.

### `get_audit_history`

Reads V3/V2/V1 histories, decodes them without erasing protocol identity, merges them, sorts newest-first by timestamp, and applies the existing hard return cap.

### `check_audit_exists`

Returns aggregate existence/count and exact counts for V3/V2/V1.

## Persisted score interpretation

The read boundary does not invent new model semantics.

- V1 keeps its historical scalar score/label compatibility representation.
- V2/V3 return raw ten-class field elements and provenance identities.
- V2/V3 persistence records are **not** converted into a scalar SAFE/VULNERABLE label.
- The read layer does not blindly divide arbitrary field elements by 8192 and call the result a probability; signed/fixed-point interpretation belongs to the exact artifact/model policy layer.

This avoids turning a storage decoder into an unmeasured decision policy.

## Runtime write containment remains intact

Version-aware reads do not reopen submission capability.

The live MCP still exposes exactly three read tools. A call to `submit_audit` or another undeclared write-like name is rejected at dispatch with `attempted=false`, before the historical `_submit.py` compatibility module is imported.

## Historical upgrade behavior

The generic read semantics work naturally across a V1/V2→V3 registry upgrade:

1. historical V1/V2 storage remains readable;
2. before the first V3 record, `get_latest_audit` can still return the newest historical record;
3. after V3 activation, legacy writes are disabled, so new accepted records advance through V3 only;
4. timestamp ordering therefore preserves the real persisted chronology without a hardcoded protocol-priority rule.

## V3 feedback-loop compatibility — DEFERRED BY POLICY, NOT SILENTLY PORTED

`agents/src/ingestion/feedback_loop.py` is a legacy V1 feedback bridge. Executable source currently:

- listens to `AuditSubmitted` (V1 scalar event), not `AuditSubmittedV3`;
- uses the legacy scalar `score` field;
- uses a historical hard-coded score-felt threshold of `5734`;
- converts that scalar back to a human score with `/8192`;
- feeds selected observations into RAG.

That behavior must **not** be copied into V3 mechanically.

V3 stores ten class-score field elements and a context-attested provenance record. R4 is also actively repairing DATA labels, target semantics, calibration and future acceptance policy. There is currently no measured basis for converting the old V1 scalar cutoff into a V3 ten-class feedback-ingestion rule.

Therefore this alignment branch intentionally does **not**:

- reuse `5734` as a V3 threshold;
- invent a per-class V3 RAG-ingestion threshold;
- call raw V3 field elements calibrated probabilities without the promoted artifact policy;
- claim that the existing feedback loop closes the V3 production loop.

The current V1 listener is historical compatibility code. Because V3 activation disables new V1/V2 writes, it cannot silently ingest new V3 submissions under the old scalar policy.

## Future feedback integration prerequisites

A separate V3 feedback track should begin only after the relevant R4/model policy is available. It should define and test:

1. which V3 audit outcomes are eligible to enter RAG;
2. the exact promoted model/data/calibration identities required for eligibility;
3. per-class or report-level acceptance semantics derived from measurement rather than copied constants;
4. how a V3 event is correlated to the exact persisted V3 record/request digest;
5. how verified policy context is represented in the RAG document schema;
6. how unavailable/ineligible evidence is retained without being interpreted as a clean/negative finding;
7. replay/dedup semantics based on V3 request/transaction identity;
8. regression tests proving a legacy V1 observation cannot be mistaken for V3 verified finality.

## Submission-truth schema follow-up

`agents/src/contracts/submission.py` remains the R0 proof/finality compatibility schema. It does not yet model a completed V3 isolated-signer transaction lifecycle.

That is expected at this checkpoint because the repository currently defines the unsigned V3 policy-request contract but does not claim a production KMS/HSM signer, broadcaster, receipt monitor, or live V3 deployment.

The future signer/report/feedback integration should extend submission truth deliberately rather than overloading the old R0 fields.

## Alignment boundary

This closeout establishes:

> ZKML proof semantics, V3 contract context binding, runtime write containment, deployment/upgrade behavior, and V1/V2/V3 registry observability are aligned at the source/protocol level.

Remaining feedback/signer/live-network work is a distinct integration and policy stage. It is not evidence that the current ZKML/contracts source baseline is still stale, and it must not be solved by importing arbitrary legacy thresholds into V3.
