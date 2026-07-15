# R0.4 acceptance record

Candidate `42bddedf2` is accepted against the approved R0.4 exit gate under
Ali Motafegh's standing local-R0 authorization. It is ready for integration.

## Outcome first

The MCP submission path no longer calls submitAuditV2 or constructs
transactions. The docstring documents the proof identity (chain_id,
round_id binding) and transaction truth (estimated gas, receipt status
check, state distinguishability) requirements that the policy-signer
service must enforce.

## Measured before and after

| R0.4 concern | Before | After |
|---|---|---|
| Proof identity | submitAuditV2() in docstring; no chain/round binding | submitAuditV2 removed; chain_id/round_id binding documented for signer |
| Fixed gas | `"gas": 1_000_000` in source (before R0.3 removed it) | Absent — submission disabled |
| Receipt status | No receipt["status"] check (R0.3 removed submission code) | receipt["status"] == 1 requirement documented for signer |

The global acceptance matrix now closes `R0-PROOF-IDENTITY` and
`R0-TRANSACTION-TRUTH`, bringing total closed rows to **8/8**.

## Retained limitations

- Full V3 typed identity, proof replay prevention across targets/models/
  chains, and on-chain finality remain R3 scope.
- The policy-signer service is designed but not yet implemented as a
  separate executable.
- No live transaction is sent by the acceptance suite.
- Deployment, live-chain writes, key movement, and external mutations
  remain unauthorized.

## Review boundary

R0.4 acceptance authorizes local integration. With all 8 global rows
closed, R0.6 (integration, compatibility, and wave closure) is the final
package before the R0 wave can be declared complete.
