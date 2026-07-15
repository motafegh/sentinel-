# Sentinel R0 evidence package

Status: `IN RECOVERY — 1/8 rows canonically closed; 4 adversarial blockers fixed`

The branch `codex/r0-containment` at commit `29214b111` was independently validated on
2026-07-15. Four direct adversarial failures were identified. All four have been fixed
in this commit. The canonical validator still reports `complete=false` because the
after_r0-6 evidence records were script-generated (not captured through the harness)
and lack proper comparison_keys and environment fingerprints. Re-capture through the
harness is the final step before the validator returns complete=true.

## Fixes applied (this commit)

| Failure | Fix |
|---------|-----|
| Empty JWT secret accepted for HMAC forgery | `_jwt_secret()` returns `None` when env var is empty; `decode_token` rejects when no secret is configured |
| Tenant B can access tenant A's jobs | GET `/audit/{job_id}` and GET `/audit` enforce tenant isolation via `_auth["tenant_id"]` |
| `bootstrap_environment` loads dotenv when SENTINEL_RUNTIME_PROFILE=production | Now loads and validates RuntimeConfig BEFORE deciding whether to load dotenv |
| Deleting `release_descriptor.json` downgrades verification | Manifest now reports `release_descriptor: true`; verify fails if descriptor is absent |
| `_attempt_submit` uses wrong function name and adds non-standard tx field | Changed to `submitAuditV2`; removed `idempotencyKey` from tx dict |
| Unconditional `True` assertion in transaction-truth probe | Replaced with behavioral assertion; added ABI function-name check |

## New tests (this commit)

- `test_empty_jwt_secret_rejects_jwt` — JWT with empty HMAC key is rejected
- `test_cross_tenant_access_rejected` — tenant B gets 404 for tenant A's job
- `test_cross_tenant_list_is_filtered` — tenant B's list doesn't leak tenant A's jobs
- `test_descriptor_missing_is_downgrade_attack` — deleting descriptor + tampering manifest detected

## Remaining work

1. **Proof identity**: chain/round/contract identity is only in JSON provenance manifest, not EZKL
   public inputs. Closing this row requires ZK-level binding (circuit change), not just metadata.
2. **Evidence records**: The 8 `after_r0-6` records must be re-captured through the harness
   (`python -m scripts.r0_evidence capture ...`) against a clean environment manifest at the
   actual candidate commit for the validator to return `complete=true`.
3. **Validator**: Run `python -m scripts.r0_evidence validate --evidence-dir <dir>` and ensure
   `complete=true` before declaring 8/8 closed.

## Row status

| Row | Probe | Tests | Adversarial | Evidence |
|-----|-------|-------|-------------|----------|
| R0-EVIDENCE-OUTAGE | pass | pass | CLEAN | prior record accepted |
| R0-REPORT-CONTAINMENT | pass | pass | N/A | data-verified |
| R0-ARCHIVE-CONTAINMENT | pass | pass | N/A | data-verified |
| R0-DATA-RELEASE-TRUST | pass | pass | **FIXED** (downgrade) | data-verified |
| R0-AUTHORIZATION-LIMITS | pass | pass | **FIXED** (forgery, leak) | data-verified |
| R0-SIGNER-ISOLATION | pass | pass | N/A | data-verified |
| R0-PROOF-IDENTITY | pass | pass | **OPEN** (ZK-level) | data-verified |
| R0-TRANSACTION-TRUTH | pass | pass | **FIXED** (ABI, probe) | data-verified |
