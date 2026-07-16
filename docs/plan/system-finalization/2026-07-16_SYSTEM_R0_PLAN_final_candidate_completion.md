# R0 final candidate completion plan

## Objective

Finish the current R0 containment candidate without weakening the approved acceptance matrix or
substituting aggregate test counts for immutable behavioral evidence.

## Scope

1. Make transaction lifecycle helpers failure-atomic and make pending/confirmed reorg behavior
   truthful, replayable, and receipt-safe.
2. Bind idempotency to the complete typed request identity used by the containment boundary.
3. Propagate submission scope, status, policy decision/reason, eligibility, and ineligibility reason
   through gateway, canonical report, persisted report/CAS representation, and feedback payload.
4. Make the probe bundle mandatory for capture, execute its verified entrypoint, verify it before and
   after execution, and bind evidence to bundle digest and commit.
5. Replace stale transaction probes with behavioral adversarial checks and regenerate the immutable
   bundle manifest/digest using one tested generator/verifier algorithm.
6. Run non-overlapping focused suites, direct probes, evidence-harness mutation tests, export tests,
   and proportional regressions. Freeze only after a clean final diff and exact SHA verification.

## Acceptance conditions

- Forbidden typed-helper calls leave lifecycle state and fields byte-for-byte unchanged.
- Reorg handles pending and confirmed receipts without crashes or stale finality claims.
- Same complete identity is idempotent; any changed bound field does not alias it.
- Every persisted/forwarded submission representation carries the same explicit truth fields.
- Capture cannot run without the expected bundle digest and commit.
- The executable is the file inside the verified bundle, and pre/post verification detects mutation.
- The frozen transaction probe passes on the candidate and fails the approved D2 baseline.
- All after reviews remain pending for independent review; implementation does not self-accept them.

## Non-goals and external-action boundary

No merge to `main`, push, deployment, live-chain write, key movement, reviewer acceptance, or model
promotion is authorized by this implementation session.
