# Sentinel R0 evidence package

Status: `R0.6_ACCEPTED — 8/8 GLOBAL ROWS CLOSED — R0 WAVE COMPLETE`

This package is the immutable measurement boundary for the approved R0 containment wave. The captured D2 baseline proves that all eight invariants fail at commit `1256d9aab45add9cf2d23fe33aaa944303259012`. R0.1 closed `R0-EVIDENCE-OUTAGE`. R0.2 has now closed `R0-REPORT-CONTAINMENT` and `R0-ARCHIVE-CONTAINMENT` through comparable after evidence and accepted review; the remaining five rows stay open until their owning packages satisfy the same rule.

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
- `acceptance/2026-07-14_SYSTEM_R0-REPORT-CONTAINMENT_after_r0-2.json` is the accepted after record closing `R0-REPORT-CONTAINMENT`.
- `acceptance/2026-07-14_SYSTEM_R0-ARCHIVE-CONTAINMENT_after_r0-2.json` is the accepted after record closing `R0-ARCHIVE-CONTAINMENT`.
- `acceptance/2026-07-14_SYSTEM_R0_EVIDENCE_r0-2_acceptance.json` and `acceptance/2026-07-14_SYSTEM_R0_REVIEW_r0-2_handoff.md` contain the measured R0.2 acceptance decision and limitations.
- `schemas/r0/evidence_record_v1.schema.json` and `schemas/r0/runtime_config_v1.schema.json` are the versioned interchange schemas.

## Closure rule

A matrix row closes only when the harness finds a before and after record with the same comparison key, a failing before invariant, a successful after invariant with exit code zero, test references, a candidate commit, and accepted reviewer decisions on both records. Pending, rejected, missing, malformed, incomparable, blocked, skipped, unavailable, or dirty-worktree evidence cannot close a row.

R0.0, R0.1, and R0.2 are accepted. R0.3–R0.6 must continue to produce clean candidate identities, comparable after records, regression measurements, and explicit accepted review before their rows close.
