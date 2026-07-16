# R0.1 acceptance record

Candidate `9e656ee4eb0a434eb97ba882eca1eee7c09a9c25` is accepted against the approved R0.1 exit gate under Ali Motafegh's standing local-R0 authorization. It is ready for integration.

## Outcome first

Sentinel now fails closed when ML inference or audit dependencies are unavailable, degraded, explicitly mocked, malformed, or provenance-mismatched. Such results cannot shape evidence, routing shortcuts, evaluation metrics, report finality, proof generation, or submission. Independently valid static evidence remains usable, and the report truthfully withholds finality and requests manual review.

## Measured before and after

| R0.1 concern | Before at `1256d9aab` | After at `9e656ee4e` |
|---|---|---|
| ML outage | Returned `safe`, ten probabilities, vulnerabilities, and a mock hash as plausible live evidence | Returns explicit terminal status only; no label, probabilities, vulnerabilities, or model identity |
| Comparable invariant | `invariant_passed=false`; 0/3 assertions pass | `invariant_passed=true`; 3/3 assertions pass; exit code 0 |
| Environment | Series-2 fingerprint `c21fac2037ea…` | Exact same fingerprint and comparison key `ac27a85060fc…` |
| Provenance | No canonical producer/input/output binding | Strict status plus input/output digest verification at producer and consumer boundaries |
| Downstream eligibility | Transport-shaped payloads could affect report/eval/proof/submission | Missing, mock, degraded, unavailable, malformed, or mutated ML results are rejected centrally |
| Readiness | Process/configuration success masqueraded as dependency health | Liveness and readiness are separate; gateway preserves live/degraded/mock/unavailable truth |
| Focused verification | R0.1 tests absent | 221/221 AGENTS focused tests and 2/2 ML wire-contract tests pass |
| AGENTS regression | R0.0: 638 pass, 9 environmental failures | 665 pass, the same 9 environmental failures; zero new failures |

The global acceptance matrix now closes exactly `R0-EVIDENCE-OUTAGE`, covering the R0 containment slices of `D2-AGT-001`, `D2-AGT-012`, and `D2-AGT-016`. The other seven global rows remain open for their owning packages.

## Retained limitations

- Six RAG tests still fail because the isolated checkout lacks the required seed corpora.
- Three static-analysis/smoke paths still fail because `solc` is unavailable on `PATH`.
- R0.1 establishes the pre-proof provenance gate. Full proof identity, public-signal binding, transaction construction, receipt truth, and V2/V3 claims remain R0.4 work.
- Deployment, live-chain writes, key movement, model promotion, artifact deletion, and contract administration remain unauthorized.

## Review boundary

R0.1 acceptance authorizes local integration and progression to R0.2. It does not claim complete R0 closure, accept the seven remaining global rows, or authorize any external mutation. LM Studio remained off throughout the outage measurement.
