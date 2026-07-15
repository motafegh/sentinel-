# Sentinel R0 evidence package

Status: `IN RECOVERY — 0/8 canonically closed`

All reviews are pending. The canonical validator returns `complete=false` for all 8 rows
until before/after records with matching comparison keys are captured at the same probe
contract version.

## What was fixed (commit 70a3141c9)

| Issue | Fix | Probe assertion |
|-------|-----|-----------------|
| Data-release descriptor bypass | Code-enforced mandatory descriptor; stripping flag+file still detected | `descriptor_code_enforced`, `descriptor_stripped_still_detected` |
| Transaction ABI wrong arg count | `_attempt_submit` matches Solidity `submitAuditV2` (5 args: address, uint256[10], bytes, uint256[], bytes32) | `abi_fn_submitAuditV2`, `abi_no_chain_id_param` |
| Proof identity not cryptographically bound | Identity hash XORed into fusion embedding → proof output depends on chain/round/contract/model | (probe assertion passes) |
| Empty JWT secret forgery | `_jwt_secret()` returns None; `decode_token` rejects | `test_empty_jwt_secret_rejects_jwt` |
| Cross-tenant access | GET routes enforce `_auth["tenant_id"]` | `test_cross_tenant_access_rejected`, list filtered |
| Production guard bypass | `bootstrap_environment` validates `RuntimeProfile` before dotenv | (existing tests) |

## Unfixed

- **Proof identity EZKL-level**: Identity binding via XOR perturbation of features, not circuit-level public input. Minor score perturbation; full cryptographic binding requires circuit update.
- **Probe contract version mismatch**: Baseline records were captured at probe version V2; current probes are at the post-behavioral-upgrade version. Comparison keys don't match. Freezing the probe contract and capturing both baseline and candidate at the same freeze commit is the required next step.
- **Reviews**: All records are pending; no rows accepted.
- **Validator**: Returns `complete=false` for all 8 rows.

## Test coverage

114/114 agent tests pass, 33/33 export tests pass, 8/8 behavioral probes pass.
4 adversarial tests added (empty secret, tenant isolation x2, descriptor downgrade).
