# R0 final closure record

- Executable candidate: `5e45fbed7e7a04edb568dbe1c5250aed90200f49`
- Approved D2 baseline: `1256d9aab45add9cf2d23fe33aaa944303259012`
- Probe bundle digest: `f8fe8042d60678a66302fed72170476d2945fe0c0f11465f771e9b139faac12c`
- Probe bundle commit: `5e45fbed7e7a04edb568dbe1c5250aed90200f49`
- Environment comparison fingerprint: `719974d3e8b1d34256c9eee252c471871c172a88176895a5cc9d3f3ea34c9f86`

## Measured result

The same committed probe bundle produced `invariant_passed=false` for all eight rows on the approved
baseline and `invariant_passed=true` for all eight rows on the executable candidate. All 16 records
are schema-valid and have exact matching comparison keys per row.

On 2026-07-16, Ali approved finalization without another review cycle. That owner decision is
recorded on all before and after records. The closure validator reports `complete=true`: 8/8 rows
closed, zero malformed records, zero invalid artifacts, and zero unknown matrix rows.

## Regression evidence

- AGENTS auth, gateway, audit, persistence, and submission group: 134 passed
- AGENTS transaction lifecycle group: 46 passed
- R0 evidence harness: 34 passed
- DATA export group: 33 passed
- Total non-overlapping tests: 247 passed, 0 failed

## Containment truth

- Legacy V2 proof scope remains `legacy_proxy_only_unbound` and is not verified/finality eligible.
- The R0 policy signer rejects every declared scope; no caller can self-declare V3 eligibility.
- The analysis/MCP process has no raw signing key or transaction-broadcast path.
- Submission truth is canonical across the gateway, final report, persisted report/CAS, and
  fail-closed feedback metadata.
- No merge, push, deployment, live-chain write, key movement, or model promotion occurred.

## How the R0 boundaries work

R0 separates an analysis result from a verified on-chain claim. The gateway authenticates public
mutation, while the analysis/MCP process has no signing key or broadcast path. Analysis produces a
canonical submission record containing proof scope, policy decision, transaction state, eligibility,
and any ineligibility reason. That same record is retained by the final report, gateway response,
content-addressed persisted report, and feedback metadata.

Legacy V2 artifacts may still be generated for compatibility, but their scope is always
`legacy_proxy_only_unbound`. Metadata such as chain, round, or contract address does not turn V2 into
an identity-bound proof. The containment policy therefore rejects every submission scope until R3
implements and validates the real V3 circuit, verifier, signer, and governance boundary.

DATA releases use a mandatory authenticated descriptor that commits the exact inventory and semantic
manifest before unsafe loading. R0 evidence uses one committed probe bundle, verified before and after
execution, against both the approved baseline and candidate. This makes “the test changed” and “the
artifact changed” detectable rather than silently comparable.

The core maintainer rule is simple: unavailable or unbound evidence may be useful diagnostically, but
it must never be upgraded into successful, verified, or final evidence by fallback behavior.

## Validation command

Run from the repository root:

```bash
python -m scripts.r0_evidence validate \
  --evidence-dir docs/changes/system-engineering/2026-07-14_SYSTEM_R0_EVIDENCE_containment/final_candidate_5e45/records \
  --expected-baseline 1256d9aab45add9cf2d23fe33aaa944303259012 \
  --expected-candidate 5e45fbed7e7a04edb568dbe1c5250aed90200f49 \
  --expected-probe-bundle-commit 5e45fbed7e7a04edb568dbe1c5250aed90200f49 \
  --expected-probe-bundle-sha256 f8fe8042d60678a66302fed72170476d2945fe0c0f11465f771e9b139faac12c
```

The expected exit code is zero and the report must contain `"complete": true`.

## Integration state

R0 is formally closed and integrated into `main` at merge commit `c11daaf247d295f41bebcad9896d6d3163d06c36`.
The merged branch includes the bounded integration corrections `da6923df9` and `ddaecbd6e`: explicit
process configuration takes precedence over local dotenv values, and audit mock tests no longer
depend on pytest collection order.

The complete post-merge R0 verification is green: 134 AGENTS boundary/report tests, 46 transaction
tests, 34 evidence-harness tests, and 33 DATA export tests (247 total, zero failures). The validator
remains `complete=true` with 8/8 rows closed and zero malformed, invalid, or unknown artifacts.

Unrelated ML/R4 work and local audit documents present during integration were preserved and excluded
from the R0 merge commits. No push, deployment, live-chain write, key movement, or model promotion was
performed as part of integration.
