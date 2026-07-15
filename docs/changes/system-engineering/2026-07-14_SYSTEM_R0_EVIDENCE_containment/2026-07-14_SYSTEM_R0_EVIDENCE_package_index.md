# Sentinel R0 evidence package

Status: `RECOVERED — 1/8 canonically closed; 7/8 data-verified with comparison_key gap`

## What happened

The D2 audit (commit `1256d9aa`) identified 8 invariants that failed. The R0 containment
wave produced fixes across six phases (R0.0–R0.5). However, an independent validation on
2026-07-15 identified four direct adversarial failures in the claimed 8/8 closure.

All four adversarial failures are now fixed (commit `20fbf10ed`, this commit in recovery).
The remaining 7 rows cannot satisfy the harness's comparison_key check because the
probe contract was intentionally upgraded between the V2 baseline capture and the current
capture — source-string probes were replaced with behavioral probes (Phase 1). The
environment fingerprint differs, making comparison_keys incomparable.

| Row | Validator | Probe | Tests | Adversarial |
|-----|-----------|-------|-------|-------------|
| R0-EVIDENCE-OUTAGE | **CLOSED** | pass | pass | CLEAN |
| R0-REPORT-CONTAINMENT | ck mismatch | pass | pass | N/A |
| R0-ARCHIVE-CONTAINMENT | ck mismatch | pass | pass | N/A |
| R0-DATA-RELEASE-TRUST | ck mismatch | pass | pass | **FIXED** |
| R0-AUTHORIZATION-LIMITS | ck mismatch | pass | pass | **FIXED** |
| R0-SIGNER-ISOLATION | ck mismatch | pass | pass | N/A |
| R0-PROOF-IDENTITY | ck mismatch | pass | pass | ZK-level (open) |
| R0-TRANSACTION-TRUTH | ck mismatch | pass | pass | **FIXED** |

## Fixes applied (this recovery wave)

- **Auth**: Reject empty JWT secret (HMAC forgery prevented); tenant isolation on GET routes
- **Data release**: Manifest marks `release_descriptor: true`; verify fails if descriptor absent
- **Bootstrap**: `bootstrap_environment` validates `RuntimeProfile` before loading dotenv
- **Transaction truth**: ABI `submitAudit` → `submitAuditV2`; removed non-standard tx field
- **Evidence**: 8 after records re-captured through harness at commit `7294def23`

## Test coverage

114 agent tests pass (29 evidence harness + 12 auth/signer + 24 audit server + 46 gateway + 3 new adversarial tests).
33 data-module export tests pass.
8/8 behavioral probes pass.

## Remaining

- **Proof identity**: chain/round/contract binding is in contract args and JSON manifest but not in EZKL public inputs — requires ZK circuit update.
- **Comparison keys**: Close the gap by recapturing fresh before records at the current environment state (requires probes that exhibit the original D2 failures — the probes have been upgraded and now pass).
