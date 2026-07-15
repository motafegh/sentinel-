# Sentinel R0 evidence package

Status: `IN RECOVERY — 1/8 GLOBAL ROWS CANONICALLY CLOSED`

This package is the immutable measurement boundary for the approved R0 containment wave. The captured D2 baseline proves that all eight invariants fail at commit `1256d9aab45add9cf2d23fe33aaa944303259012`. R0.1 closed `R0-EVIDENCE-OUTAGE`. R0.2 recovery work was committed at `beb8e5250` but the after evidence (V3 series) was semantically invalid — contradictory proof-identity record and source-string probes that do not test behavioral invariants. The V3 records have been quarantined. R0.2 rows `R0-REPORT-CONTAINMENT` and `R0-ARCHIVE-CONTAINMENT` remain open until valid behavioral after evidence is produced. The remaining five rows also stay open until their owning packages satisfy the same rule.

## Authoritative artifacts

- `2026-07-14_SYSTEM_R0_EVIDENCE_matrix_rows.json` owns the eight stable acceptance row identifiers and their D2 mappings.
- `2026-07-14_SYSTEM_R0_EVIDENCE_command_manifest_v2.json` owns the frozen, runtime-bound probe contracts.
- `2026-07-14_SYSTEM_R0_EVIDENCE_baseline_manifest_v2.json` binds the clean baseline, environment fingerprint, manifests, record digests, and accepted review state.
- `baseline_series_2/*_before_v2.json` contains the comparable expected-failing record for every matrix row.
- `acceptance/2026-07-14_SYSTEM_R0_ENVIRONMENT_r0-0_candidate.json` proves the R0.0 candidate was measured from a clean committed worktree.
- `acceptance/2026-07-14_SYSTEM_R0_ENVIRONMENT_r0-1_candidate.json` binds R0.1 to clean implementation commit `9e656ee4e` and the baseline comparison fingerprint.
- `acceptance/2026-07-14_SYSTEM_R0-EVIDENCE-OUTAGE_after_r0-1.json` is the canonical comparable after record closing `R0-EVIDENCE-OUTAGE` (1/8 rows closed).
- `acceptance/historical_summaries/` contains pre-corrective-commit artifacts (old ad hoc after summaries, handoff notes, historical acceptance ledgers) that are preserved for audit trail but do not constitute canonical evidence.
- `schemas/r0/evidence_record_v1.schema.json` and `schemas/r0/runtime_config_v1.schema.json` are the versioned interchange schemas.

## Closure rule

A matrix row closes only when the harness finds a before and after record with the same comparison key, a failing before invariant, a successful after invariant with exit code zero, test references, a candidate commit, and accepted reviewer decisions on both records. Pending, rejected, missing, malformed, incomparable, blocked, skipped, unavailable, or dirty-worktree evidence cannot close a row.

R0.1 is accepted (R0-EVIDENCE-OUTAGE — 1/8 rows canonically closed). The remaining seven rows remain open. R0.2–R0.6 must produce clean candidate identities, comparable after records with `kind: r0_evidence_record`, regression measurements, behavioral probes that execute the actual boundary (not source-string patterns), and explicit accepted review before their rows close.
