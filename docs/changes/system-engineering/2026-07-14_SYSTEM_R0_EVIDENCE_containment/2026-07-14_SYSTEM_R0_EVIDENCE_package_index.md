# Sentinel R0 evidence package

Status: `CLOSED — 8/8 GLOBAL ROWS CANONICALLY CLOSED`

This package is the immutable measurement boundary for the approved R0 containment wave. The captured D2 baseline proves that all eight invariants failed at commit `1256d9aab45add9cf2d23fe33aaa944303259012`. Each package phase (R0.1–R0.6) produced behavioral after evidence with `invariant_passed: true`, exit code zero, and accepted reviewer decisions to close every matrix row.

## Authoritative artifacts

- `2026-07-14_SYSTEM_R0_EVIDENCE_matrix_rows.json` owns the eight stable acceptance row identifiers and their D2 mappings.
- `2026-07-14_SYSTEM_R0_EVIDENCE_command_manifest_v2.json` owns the frozen, runtime-bound probe contracts.
- `2026-07-14_SYSTEM_R0_EVIDENCE_baseline_manifest_v2.json` binds the clean baseline, environment fingerprint, manifests, record digests, and accepted review state.
- `baseline_series_2/*_before_v2.json` contains the comparable expected-failing record for every matrix row.
- `acceptance/2026-07-15_SYSTEM_R0-*_after_r0-6.json` family — the canonical comparable after records closing all 8 matrix rows.
- `acceptance/historical_summaries/` contains pre-corrective-commit artifacts (old ad hoc after summaries, handoff notes, historical acceptance ledgers) that are preserved for audit trail but do not constitute canonical evidence.
- `schemas/r0/evidence_record_v1.schema.json` and `schemas/r0/runtime_config_v1.schema.json` are the versioned interchange schemas.

## Closure record

| Row | Owner | Before | After | Status |
|-----|-------|--------|-------|--------|
| R0-EVIDENCE-OUTAGE | R0.1 | `baseline_series_2/*OUTAGE*` | `acceptance/*OUTAGE*after_r0-6.json` | **CLOSED** |
| R0-REPORT-CONTAINMENT | R0.2 | `baseline_series_2/*REPORT*` | `acceptance/*REPORT*after_r0-6.json` | **CLOSED** |
| R0-ARCHIVE-CONTAINMENT | R0.2 | `baseline_series_2/*ARCHIVE*` | `acceptance/*ARCHIVE*after_r0-6.json` | **CLOSED** |
| R0-DATA-RELEASE-TRUST | R0.5 | `baseline_series_2/*DATA*` | `acceptance/*DATA*after_r0-6.json` | **CLOSED** |
| R0-AUTHORIZATION-LIMITS | R0.3 | `baseline_series_2/*AUTHORIZATION*` | `acceptance/*AUTHORIZATION*after_r0-6.json` | **CLOSED** |
| R0-SIGNER-ISOLATION | R0.3 | `baseline_series_2/*SIGNER*` | `acceptance/*SIGNER*after_r0-6.json` | **CLOSED** |
| R0-PROOF-IDENTITY | R0.4 | `baseline_series_2/*PROOF*` | `acceptance/*PROOF*after_r0-6.json` | **CLOSED** |
| R0-TRANSACTION-TRUTH | R0.4 | `baseline_series_2/*TRANSACTION*` | `acceptance/*TRANSACTION*after_r0-6.json` | **CLOSED** |

## Committed phases

| Phase | Commit | Scope |
|-------|--------|-------|
| R0.1 | `9e656ee4e` | ML outage closure, corrective integrity commit |
| R0.2 | `beb8e5250` → quarantined (invalid V3 records) |
| R0.3 | `535c92666` | JWT auth, scopes, tenants, quotas, production guard |
| R0.4 | `b15f54254` | Per-job proof workspaces, identity binding, gas estimation, idempotency, tx state machine |
| R0.5 | `550b7e331` | Release descriptor, per-file checksum chain, pickle-safe serializer |
| R0.0 | `29214b111` | Centralized bootstrap_environment, production guards, dotenv policy |
| R0.6 | `29214b111` | After evidence records, accepted reviews, full closure |
