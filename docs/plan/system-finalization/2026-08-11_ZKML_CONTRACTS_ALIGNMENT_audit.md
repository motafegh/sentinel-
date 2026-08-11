# SENTINEL ZKML + Contracts Alignment Audit

**Date:** 2026-08-11  
**Branch:** `system/zkml-contracts-alignment`  
**Status:** IN_PROGRESS — remote-safe implementation + CI validation  
**Method:** executable source first; historical docs/commits are supporting evidence only.

## Reconstructed executable seam

1. `ml/src/inference/api.py` exposes `/fusion-embedding` with exactly 128 floats plus the active teacher checkpoint SHA-256 and structured execution status.
2. `agents/src/mcp/servers/audit/_submit.py` checks the live ML model identity, runs the 128→64→32→10 student, generates/verifies the legacy V2 EZKL proof, and then crosses the R0 policy-signer boundary.
3. The tracked EZKL V2 circuit exposes 128 public inputs + 10 public outputs = 138 public signals.
4. Historical `AuditRegistry.submitAuditV2` verifies those ten output felts but does not cryptographically bind target/model/context metadata.
5. R0 deliberately classifies the circuit proof scope as `legacy_proxy_only_unbound`; policy signing/finality is fail-closed for that scope.

## Critical distinction

The existing ZK proof establishes the **proxy computation statement** represented by the V2 circuit. It does not establish that:

- the 128 fusion values were honestly produced by the teacher from the named Solidity contract;
- the caller-supplied teacher model identity is part of the proof;
- the proof is intrinsically tied to a target address, chain, registry, agent, or round.

The V3 registry design on this branch therefore uses **dual attestation**, not an inflated proof claim:

1. EZKL verifies the exact proxy proof/public signals.
2. A dedicated policy signer supplies an EIP-712 signature over the audit context and the exact proof/signals/scores.

The accepted V3 *submission* is context-bound even though the neural proof itself remains proxy-only.

## Finding register

### ZC-P0-001 — legacy direct-send helper bypassed signer isolation

**Status:** FIXED REMOTELY  
**Former source:** `zkml/src/ezkl/extract_calldata.py`

Historical behavior generated `cast send --private-key ... submitAuditV2(...)`, bypassing the R0 policy-signer boundary.

**Fix:** the helper is now a read-only V2 proof decoder. It emits no transaction/signing script, reads no private key, validates the exact 138-signal layout, and reports `submission_eligible=false`. Regression tests scan for raw-key/direct-send capabilities.

### ZC-P0-002 — V2 model identity is caller metadata, not proof-bound

**Status:** CONTAINED BY V3 DESIGN; HISTORICAL V2 BEHAVIOR RETAINED FOR TEST EVIDENCE

`submitAuditV2(..., bytes32 modelHash)` historically stores a caller-supplied model hash. The proxy proof does not prove that value.

A tracked V2 Foundry test now demonstrates that the same proof/signals can be stored under two different model hashes before V3 activation.

**V3 disposition:** EIP-712 policy attestation binds `teacherModelHash`, `proxyBundleHash`, DATA identity, class-schema identity, exact proof hash, public-signal hash, score hash, target bytecode identity, agent and round. The contract stores the resulting signed request digest.

### ZC-P0-003 — V2 proof is replayable across audit context

**Status:** CONTAINED BY V3 DESIGN; HISTORICAL V2 BEHAVIOR RETAINED FOR TEST EVIDENCE

V2's 138 signals contain fusion inputs + proxy outputs only. A tracked V2 Foundry test demonstrates the same mock-verified proof can be submitted for different target addresses.

**V3 disposition:** the EIP-712 domain binds `block.chainid` + registry proxy address; the request binds target `codehash`, agent, round and artifact identities. `requestDigest` is one-time-use. Cross-target, cross-agent, cross-registry, model, proof, and signal substitution tests are included.

### ZC-P0-004 — proxy output was double-sigmoided outside the circuit

**Status:** PARTIALLY FIXED; AGENTS `_submit.py` MINIMAL PATCH STILL REQUIRED

Training source establishes:

```python
teacher_scores = torch.sigmoid(teacher_logits)
loss = MSELoss(proxy(features), teacher_scores)
agreement = (proxy_scores >= 0.5) == (teacher_scores >= 0.5)
```

Therefore the trained V2 student's direct `forward()` output is the probability-regression score that ONNX/EZKL proves. A second `sigmoid()` changes the statement.

**Fixed:**
- `ProxyModel` now declares `teacher_probability_regression_v1` output semantics;
- proxy tests lock the direct-score contract;
- `run_proof.py` no longer applies a second sigmoid and no longer cherry-picks the easiest contract.

**Still open:** `agents/src/mcp/servers/audit/_submit.py` still contains the historical `torch.sigmoid(proxy(...))` site. It is finality-contained by R0 but must be minimally patched without disturbing the hardened transaction engine.

### ZC-P0-005 — Solidity historically accepts direct unbound V1/V2 writes

**Status:** FIXED IN V3 ACTIVATION PATH; REQUIRES FOUNDRY/UPGRADE VALIDATION

R0 disabled signing/finality in the AGENTS policy layer, but the pre-V3 registry itself still allowed any sufficiently staked caller to invoke V1/V2 directly.

**V3 disposition:** `initializeV3()` is a one-way upgrade boundary. It configures the dedicated V3 verifier + policy signer and sets `legacySubmissionsDisabled=true`. Historical V1/V2 query/storage APIs remain readable; new V1/V2 writes revert after V3 activation.

### ZC-P1-001 — legacy binary path dominated tracked contract tests

**Status:** IMPROVED

The original tracked suite heavily tested the 65-signal V1 path while the live architecture had already moved to 128+10.

**Fix:** added tracked V2 and V3 suites. V2 tests preserve historical limitations; V3 tests cover context binding/replay/substitution semantics.

### ZC-P1-002 — EZKL `check_mode` semantics were previously over-interpreted

**Status:** CORRECTED

Tracked settings use `check_mode="UNSAFE"` on EZKL 23.0.5. The enum name alone is not evidence that the generated proof is unsound or automatically production-invalid; current official EZKL examples also use that mode in proving workflows.

**Disposition:** bundle validation reports `check_mode` and EZKL version as explicit **review metadata**. It is not classified as a security verdict. Production ineligibility is independently established by the unbound V2 proof scope until the dual-attestation V3 path is fully validated.

### ZC-P1-003 — proxy training/calibration silently targeted stale v2 DATA

**Status:** FIXED FOR FUTURE RETRAINS

Historical distillation/calibration hardcoded `sentinel-v2-baseline-2026-06-12` despite later Run12/R4 lineage.

**Fix:**
- `train_proxy.py` requires explicit `--export-dir`; no historical default;
- checkpoint metadata binds teacher SHA-256, export-manifest SHA-256, circuit/output semantics, seed and measured agreement;
- `generate_calibration.py` likewise requires an explicit export and writes a lineage manifest;
- proxy retraining is intentionally deferred until R4 promotes the appropriate DATA/ML bundle.

### ZC-P1-004 — ZKML artifact bundle had filename provenance, not explicit identity

**Status:** IMPROVED; LOCAL CRYPTOGRAPHIC VALIDATION STILL REQUIRED

The historical chain `proxy_best.pt → ONNX → settings/compiled circuit → VK → Solidity verifier` lacked one authoritative machine-readable identity chain.

**Fix:**
- ONNX export now writes a manifest binding checkpoint and ONNX hashes plus parity evidence;
- `validate_bundle.py` hashes/checks the tracked historical V2 bundle, protocol dimensions, visibility, scale and verifier ABI;
- structural validity and production eligibility are reported separately.

This validator is an integrity/protocol check, not a substitute for `ezkl.verify` or on-chain verifier execution.

### ZC-P1-005 — duplicate generated verifier sources created ambiguous authority

**Status:** FIXED REMOTELY

`contracts/src/ZKMLVerifier.sol` and `contracts/standalone/ZKMLVerifier.sol` had different generated constants/blob identities.

**Fix:** the stale standalone copy was removed. `contracts/src/ZKMLVerifier.sol` is the single tracked canonical generated verifier, and `.gitignore` now makes that intentional so regeneration produces a reviewable diff.

### ZC-P1-006 — verifier rotation was described but not implemented

**Status:** FIXED IN V3 PATH; REQUIRES UPGRADE TESTING

Historical comments described the verifier as swappable, but no owner setter existed.

**Fix:** V3 has a dedicated appended `zkmlVerifierV3` with owner-only rotation and explicit events. Historical verifier semantics remain untouched.

### ZC-P1-007 — contract dependency bootstrap was not reproducible from a fresh root clone

**Status:** FIXED REMOTELY

`contracts/.gitmodules` is not a root `.gitmodules`, so Git does not initialize those libraries as repository submodules.

**Fix:** `contracts/scripts/bootstrap_deps.sh` clones and verifies the exact revisions already recorded in `contracts/foundry.lock`. Branch CI uses that same bootstrap.

### ZC-P2-001 — stale proxy comments/parameter count/output terminology

**Status:** FIXED IN EXECUTABLE PROXY SOURCE; README CLEANUP PENDING

Actual architecture is 128→64→32→10 with 10,666 parameters. Source now reflects this and locks output semantics. Module READMEs still contain older ~8.3K/logit wording.

### ZC-P2-002 — `zkml/README.md` and `contracts/README.md` are materially stale

**Status:** OPEN CLEANUP

Examples include 64/65-signal descriptions, `submit_audit.sh`, old checkpoint paths, obsolete verifier-tracking claims, stale single-score limitations, and compiler statements that no longer match current config.

**Disposition:** rewrite after executable/CI state settles; documentation must describe V1/V2 as historical and V3 as pending local validation.

## V3 registry protocol implemented on branch

`AuditRegistry` now appends V3 storage rather than reordering historical storage.

V3 request context includes:

- target contract address + runtime bytecode hash;
- round ID;
- teacher checkpoint hash;
- proxy/circuit bundle hash;
- DATA version hash;
- class-schema hash;
- expiry deadline.

The EIP-712 request additionally binds:

- submitting agent;
- exact `keccak256(proof)`;
- exact `keccak256(abi.encode(publicSignals))`;
- exact `keccak256(abi.encode(classScoreFelts))`;
- EIP-712 domain: current chain ID + registry proxy address.

Acceptance requires:

1. V3 initialized and legacy submissions disabled;
2. target has deployed bytecode;
3. policy signature not expired;
4. minimum stake;
5. exact 138-signal historical proxy-proof layout;
6. all 10 score felts equal proof public outputs;
7. unused request digest;
8. signature recovered from configured policy signer;
9. V3 verifier accepts the proof.

The replay marker is set before the external verifier call; a failed/reverted verifier call reverts the marker with the transaction.

## Remote validation lane

Root workflow `.github/workflows/system-alignment.yml` now runs on this branch:

- dependency-light ZKML boundary tests;
- tracked V2 bundle integrity/protocol report;
- Foundry v1.7.1;
- deterministic dependency bootstrap from locked SHAs;
- registry/test build and Foundry tests with the generated Halo2 verifier isolated from the registry unit-compile lane.

## Still required before merge / production claims

1. Patch AGENTS `_submit.py` to remove the remaining second sigmoid without altering R0 transaction-state behavior.
2. Make the policy-signer service produce the exact V3 EIP-712 digest/signature only after its own identity/provenance checks pass.
3. Run/close remote Foundry and boundary CI findings.
4. Locally validate the canonical generated verifier and actual EZKL V2 bundle together (`witness → prove → verify`).
5. Run storage-layout/upgrade tests against a pre-V3 registry state, including V1/V2 history preservation after upgrade.
6. Run Anvil adversarial integration: valid V3, replay, target/model/data/schema/signal/proof substitution, signer rotation and invalid proof.
7. Confirm deployed bytecode/ABI and verifier identities before any network deployment.
8. Rewrite stale module READMEs only after the executable seam is accepted.

No branch merge or production-readiness claim is permitted until the required local cryptographic and upgrade validation is complete.
