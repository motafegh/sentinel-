# Sentinel R0 evidence package

Status: `R0.1_ACCEPTED — 1/8 GLOBAL ROWS CLOSED`

This package is the immutable measurement boundary for the approved R0 containment wave. The captured D2 baseline proves that all eight invariants fail at commit `1256d9aab45add9cf2d23fe33aaa944303259012`. R0.1 has now closed exactly `R0-EVIDENCE-OUTAGE` through comparable after evidence and accepted review; the remaining seven rows stay open until their owning packages satisfy the same rule.

## Authoritative artifacts

- `2026-07-14_SYSTEM_R0_EVIDENCE_matrix_rows.json` owns the eight stable acceptance row identifiers and their D2 mappings.
- `2026-07-14_SYSTEM_R0_EVIDENCE_command_manifest_v2.json` owns the frozen, runtime-bound probe contracts.
- `2026-07-14_SYSTEM_R0_EVIDENCE_baseline_manifest_v2.json` binds the clean baseline, environment fingerprint, manifests, record digests, and accepted review state.
- `baseline_series_2/*_before_v2.json` contains the comparable expected-failing record for every matrix row.
- `acceptance/2026-07-14_SYSTEM_R0_ENVIRONMENT_r0-0_candidate.json` proves the R0.0 candidate was measured from a clean committed worktree.
- `acceptance/2026-07-14_SYSTEM_R0_EVIDENCE_r0-0_acceptance.json` and its handoff record the accepted evidence-harness package.
- `acceptance/2026-07-14_SYSTEM_R0_ENVIRONMENT_r0-1_candidate.json` binds R0.1 to clean implementation commit `9e656ee4e` and the baseline comparison fingerprint.
- `acceptance/2026-07-14_SYSTEM_R0-EVIDENCE-OUTAGE_after_r0-1.json` is the accepted comparable after record closing `R0-EVIDENCE-OUTAGE`.
- `acceptance/2026-07-14_SYSTEM_R0_EVIDENCE_r0-1_acceptance.json` and `acceptance/2026-07-14_SYSTEM_R0_REVIEW_r0-1_handoff.md` contain the measured R0.1 acceptance decision and limitations.
- `schemas/r0/evidence_record_v1.schema.json` and `schemas/r0/runtime_config_v1.schema.json` are the versioned interchange schemas.

## Closure rule

A matrix row closes only when the harness finds a before and after record with the same comparison key, a failing before invariant, a successful after invariant with exit code zero, test references, a candidate commit, and accepted reviewer decisions on both records. Pending, rejected, missing, malformed, incomparable, blocked, skipped, unavailable, or dirty-worktree evidence cannot close a row.

R0.0 and R0.1 are accepted. R0.2–R0.6 must continue to produce clean candidate identities, comparable after records, regression measurements, and explicit accepted review before their rows close.
