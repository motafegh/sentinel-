# Sentinel R0 evidence package

Status: `R0.0_IMPLEMENTATION_IN_PROGRESS`

This package is the immutable measurement boundary for the approved R0 containment wave. It does not claim that any R0 security invariant is fixed. The captured D2 baseline proves that all eight invariants fail at commit `1256d9aab45add9cf2d23fe33aaa944303259012`; every row therefore remains open until a comparable after record and an explicit reviewer decision exist.

## Authoritative artifacts

- `2026-07-14_SYSTEM_R0_EVIDENCE_matrix_rows.json` owns the eight stable acceptance row identifiers and their D2 mappings.
- `2026-07-14_SYSTEM_R0_EVIDENCE_command_manifest.json` owns the stable probe contracts.
- `2026-07-14_SYSTEM_R0_EVIDENCE_baseline_manifest.json` binds the clean baseline commit, environment, manifests, record digests, and review state.
- `baseline/2026-07-14_SYSTEM_R0_ENVIRONMENT_before.json` records the redacted execution environment and dependency-lock digests.
- `baseline/*_before.json` contains one expected-failing record for each matrix row.
- `baseline/2026-07-14_SYSTEM_R0_COVERAGE_before.json` proves closure is currently false because after records do not yet exist.
- `schemas/r0/evidence_record_v1.schema.json` and `schemas/r0/runtime_config_v1.schema.json` are the versioned interchange schemas.

## Closure rule

A matrix row closes only when the harness finds a before and after record with the same comparison key, a failing before invariant, a successful after invariant with exit code zero, test references, a candidate commit, and accepted reviewer decisions on both records. Pending, rejected, missing, malformed, incomparable, blocked, skipped, unavailable, or dirty-worktree evidence cannot close a row.

R0.0 itself will be offered for review only after its focused and regression suites, baseline digest verification, runtime-profile checks, and package acceptance ledger are committed on the isolated branch.
