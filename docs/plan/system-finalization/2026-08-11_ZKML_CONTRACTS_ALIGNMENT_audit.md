# SENTINEL ZKML + Contracts Alignment Audit

**Date:** 2026-08-11  
**Branch:** `system/zkml-contracts-alignment`  
**Status:** IN_PROGRESS  
**Method:** executable source first; docs used only after source verification.

## Current executable seam

1. `ml/src/inference/api.py` exposes `/fusion-embedding` with exactly 128 floats plus teacher checkpoint SHA-256 and structured execution status.
2. `agents/src/mcp/servers/audit/_submit.py` requires the live ML model hash to equal the requested model identity, then runs the local 128→64→32→10 proxy.
3. The existing EZKL V2 circuit has 128 public inputs + 10 public outputs (138 signals), with public input/output visibility.
4. `AuditRegistry.submitAuditV2` checks the 10 supplied class scores against public signals 128..137 and stores a caller-supplied `modelHash`.
5. R0 deliberately marks the proof `legacy_proxy_only_unbound`; the policy signer rejects it because the circuit does not bind contract/chain/round/model identity.

## Finding register

### ZC-P0-001 — legacy direct-send script bypasses signer isolation

**Severity:** P0 / containment violation  
**Source:** `zkml/src/ezkl/extract_calldata.py`

The script writes `submit_audit.sh` containing:

- `cast send`;
- `--private-key $DEPLOYER_PRIVATE_KEY`;
- direct invocation of `AuditRegistry.submitAuditV2(...)`.

This conflicts with the implemented R0 boundary in `agents/src/security/policy_signer.py` and `agents/src/mcp/servers/audit/_submit.py`, where the analysis process has no raw signing key and every current proof scope is rejected pending typed identity-bound V3 protocol work.

**Required disposition:** remove/disable direct transaction generation. The helper may remain as a decoder/verification-bundle tool, but it must not generate a signing path that bypasses policy.

### ZC-P0-002 — current on-chain V2 record accepts an unbound model hash

**Severity:** P0 / trust-model gap (currently contained by policy rejection)  
**Source:** `contracts/src/AuditRegistry.sol::submitAuditV2`

`modelHash` is a caller-supplied `bytes32`; the V2 proof verifies only proxy computation over 128 fusion inputs and does not commit the teacher checkpoint identity. The registry stores/emits `modelHash` without proving it.

**Current containment:** `_submit.py` rejects V2 proof finality and `policy_signer.py` rejects all proof scopes. Therefore this is not currently an accepted verified-audit path, but it blocks production completion.

**Required disposition:** V3 must commit an identity digest in public signals and Solidity must recompute/compare that digest before storing a verified record.

### ZC-P0-003 — current proof cannot establish audited-contract identity

**Severity:** P0 / semantic proof-scope gap (currently contained)  
**Source:** V2 EZKL public signals + `AuditRegistry.submitAuditV2`

The proof commits only fusion[128] and proxy outputs[10]. `contractAddress`, `chain_id`, `round_id`, teacher model identity, and target data/model protocol identity are outside the circuit. An off-chain provenance manifest cannot upgrade this proof scope.

**Required disposition:** define typed identity-bound V3 public-input protocol before enabling policy acceptance.

### ZC-P0-004 — proxy output is double-sigmoided outside the circuit

**Severity:** P0 / cross-module correctness  
**Sources:** `zkml/src/distillation/train_proxy.py`, `zkml/src/distillation/proxy_model.py`, `agents/src/mcp/servers/audit/_submit.py`, `zkml/src/ezkl/run_proof.py`, `zkml/src/distillation/export_onnx.py`

Executable training behavior is unambiguous:

```python
teacher_scores = torch.sigmoid(teacher_logits)       # [B,10] probabilities
loss = MSELoss(proxy(features), teacher_scores)      # raw proxy forward is fitted to probabilities
agreement = (proxy_scores >= 0.5) == (teacher_scores >= 0.5)
```

Therefore the current V2 proxy's `forward()` output is the student **score/probability approximation**, despite stale `ProxyModel` comments calling it a raw logit. `export_onnx.py` exports that exact forward pass, so the EZKL circuit proves those raw student scores.

But `_submit.py` does `scores = torch.sigmoid(proxy(features))`, and `run_proof.py` also sigmoids proxy output for comparisons. That second sigmoid changes the value. The off-chain `result["class_scores"]` therefore does not mean the same thing as the 10 EZKL public outputs that later replace `class_score_felts` from the proof.

**Required disposition:** for the existing V2 artifact semantics, treat `ProxyModel.forward()` as `proxy_scores` and remove the second sigmoid in consumers. Do not retrain remotely merely to make stale logit terminology true. Rename/comment behavior to match the trained artifact. Add tests that make a second sigmoid impossible to reintroduce silently.

### ZC-P1-001 — legacy binary contract path remains first-class

**Severity:** P1 / stale surface  
**Source:** `AuditRegistry.submitAudit`, `_audits`, and multiple Foundry tests.

The registry still exposes the old 65-signal, single-score path while the live ML/ZKML architecture is 128→10. Tests heavily exercise the legacy path and provide relatively little protection for the V2/current seam.

**Required disposition:** explicitly deprecate/quarantine legacy writes and move current tests toward the versioned multi-class/identity-bound protocol. Storage compatibility can remain for upgrade safety.

### ZC-P1-002 — ZK circuit settings are explicitly UNSAFE

**Severity:** P1 / production readiness blocker  
**Source:** `zkml/ezkl/settings.json`

Tracked EZKL settings use `check_mode: "UNSAFE"` (EZKL 23.0.5). No production claim should rely on this bundle until a supported production-safe settings/proof workflow is generated and independently verified.

**Required disposition:** preserve current artifacts as historical V2 evidence; generate a new versioned V3 proof bundle under an explicitly accepted safe-mode policy during local cryptographic validation.

### ZC-P1-003 — proxy training/calibration point at a stale v2 export

**Severity:** P1 / reproducibility and future-retraining blocker  
**Sources:** `zkml/src/distillation/train_proxy.py`, `zkml/src/distillation/generate_calibration.py`

Both files hardcode:

`data_module/data/exports/sentinel-v2-baseline-2026-06-12`

while the active/reconstructed Run12 DATA lineage is the later `sentinel-v3-smartbugs-2026-06-13` family and R4 has established that historical label semantics require repair before retraining. A future proxy retrain against the hardcoded v2 export would not be a faithful student of the intended post-R4 teacher/data contract.

**Required disposition:** do not retrain the proxy yet. Externalize/bind the teacher checkpoint + source export identity in a versioned ZKML artifact manifest. Future retraining must consume an explicitly promoted R4 DATA/ML bundle, not this stale implicit path.

### ZC-P2-001 — stale source commentary around proxy parameter count/output semantics

`ProxyModel` executable architecture is 128→64→32→10 = 10,666 parameters, while comments/docstrings still say roughly 8K/8,330 in places and call outputs raw logits even though training fits them directly to teacher probabilities. These comments are stale and actively obscure the cross-module score bug.

## Open reconstruction items

- exact verifier ABI/public-input expectations and generated verifier identity;
- artifact identity binding between `proxy_best.pt`, `proxy.onnx(.data)`, compiled circuit, settings, VK and Solidity verifier;
- current Foundry V2 coverage and legacy-write containment tests;
- public-signal field encoding/range assumptions;
- deployment scripts and stale Sepolia addresses;
- whether any other direct signing/write path survives R0;
- minimal V3 protocol shape compatible with future R4 DATA/ML evolution without hardcoding label-policy decisions.
