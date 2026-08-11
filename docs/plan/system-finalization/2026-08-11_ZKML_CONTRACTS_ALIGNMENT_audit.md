# SENTINEL ZKML + Contracts Alignment Audit

**Date:** 2026-08-11  
**Branch:** `system/zkml-contracts-alignment`  
**Status:** IN_PROGRESS — V3 registry/policy protocol remotely validated; historical generated-verifier proof compatibility still under investigation  
**Method:** executable source and executable tests first; historical docs/commits are supporting evidence only.

## Reconstructed executable seam

1. `ml/src/inference/api.py` exposes `/fusion-embedding` with exactly 128 floats plus the active teacher checkpoint SHA-256 and structured execution status.
2. `agents/src/mcp/servers/audit/_submit.py` checks the live ML model identity, runs the 128→64→32→10 student, generates/verifies the legacy V2 EZKL proof, and then crosses the R0 policy-signer boundary.
3. The tracked EZKL V2 circuit exposes 128 public inputs + 10 public outputs = 138 public signals.
4. Historical `AuditRegistry.submitAuditV2` verifies those ten output felts but does not cryptographically bind target/model/context metadata.
5. R0 deliberately classifies the circuit proof scope as `legacy_proxy_only_unbound`; direct proof-only finality remains fail-closed.

## Critical trust distinction

The existing ZK proof establishes the **proxy computation statement** represented by the V2 circuit. It does not establish that:

- the 128 fusion values were honestly produced by the named teacher from the named Solidity bytecode;
- the caller-supplied teacher/DATA/schema identity is itself part of the ZK statement;
- the proof is intrinsically tied to a target address, chain, registry, agent, or audit round.

The V3 protocol therefore uses **dual attestation** rather than inflating the proof claim:

1. EZKL verifies the exact proxy proof/public signals.
2. A dedicated policy signer supplies an EIP-712 signature binding the exact proof/signals/scores to the audit context.

The accepted V3 **submission** is context-bound even though the neural proof itself remains proxy-only.

# Finding register

## ZC-P0-001 — legacy direct-send helper bypassed signer isolation

**Status:** FIXED REMOTELY

Historical `zkml/src/ezkl/extract_calldata.py` generated a raw-key `cast send` path around the R0 policy signer.

**Fix:** the helper is now a read-only legacy V2 decoder. It emits no transaction/signing script, reads no private key, validates the exact 138-signal layout and reports `submission_eligible=false`. Regression tests reject reintroduction of direct-write capability.

## ZC-P0-002 — V2 model identity is caller metadata, not proof-bound

**Status:** HISTORICAL LIMITATION; CONTAINED BY V3 ACCEPTANCE PROTOCOL

A tracked V2 Foundry test proves the same mock-verified proof/signals can be stored under different caller-supplied model hashes before V3 activation.

**V3 disposition:** policy attestation binds teacher model, proxy bundle, DATA version, class schema, exact proof/signals/scores, target bytecode, agent and round to a one-time EIP-712 request digest.

## ZC-P0-003 — V2 proof is replayable across audit context

**Status:** HISTORICAL LIMITATION; CONTAINED BY V3 ACCEPTANCE PROTOCOL

A tracked V2 Foundry test demonstrates the same proof can be submitted for different target addresses.

**V3 disposition:** the EIP-712 domain binds chain ID + registry proxy, while the request binds target `codehash`, agent, round and artifact identities. Exact replay, target/model/proof/signal/agent substitution and cross-registry replay are covered by V3 tests.

## ZC-P0-004 — proxy output was double-sigmoided outside the circuit

**Status:** PARTIALLY FIXED; ONE HARDENED AGENTS CALL SITE REMAINS

Training semantics are executable and unambiguous:

```python
teacher_scores = torch.sigmoid(teacher_logits)
loss = MSELoss(proxy(features), teacher_scores)
agreement = (proxy_scores >= 0.5) == (teacher_scores >= 0.5)
```

The trained student's direct `forward()` output is therefore the probability-regression score that ONNX/EZKL proves. Applying another sigmoid changes the value.

**Fixed remotely:**
- `ProxyModel` declares `teacher_probability_regression_v1` output semantics;
- architecture/score tests lock the direct-score contract;
- `run_proof.py` uses the direct student score and no longer cherry-picks an easy contract.

**Still open:** `agents/src/mcp/servers/audit/_submit.py` still contains the historical `torch.sigmoid(proxy(...))` site. It is finality-contained by R0. Patch must be minimal because the same file contains the accepted hardened transaction engine.

## ZC-P0-005 — Solidity historically accepts unbound V1/V2 writes directly

**Status:** FIXED IN V3 ACTIVATION PATH; REMOTELY VALIDATED

`initializeV3()` is a one-way UUPS activation boundary that configures the dedicated V3 verifier + policy signer and sets `legacySubmissionsDisabled=true`. Historical V1/V2 records remain readable; new legacy writes revert.

**Remote evidence:** ordinary Solidity build and full Foundry V1/V2/V3/upgrade test suite pass.

## ZC-P1-001 — legacy binary path dominated tracked contract tests

**Status:** FIXED/REBALANCED

Added tracked V2, V3, Python↔Solidity digest and UUPS upgrade suites. Historical V2 limitations remain executable regression evidence rather than prose only.

## ZC-P1-002 — EZKL `check_mode` semantics were previously over-interpreted

**Status:** CORRECTED

`check_mode` and EZKL version are reported as review metadata. The enum name alone is not treated as a cryptographic verdict. V2 production ineligibility is independently established by its unbound proof scope.

## ZC-P1-003 — proxy training/calibration silently targeted stale v2 DATA

**Status:** FIXED FOR FUTURE RETRAINS

- `train_proxy.py` requires explicit `--export-dir`;
- checkpoint metadata binds teacher SHA-256, export-manifest SHA-256, circuit/output semantics, seed and agreement;
- `generate_calibration.py` likewise requires an explicit export and writes lineage metadata;
- proxy retraining remains deferred until R4 promotes a DATA/ML bundle.

## ZC-P1-004 — ZKML artifact bundle had filename provenance, not explicit identity

**Status:** SUBSTANTIALLY IMPROVED; REAL HISTORICAL PROOF/VERIFIER PAIR STILL OPEN

Implemented:
- ONNX manifest with checkpoint + ONNX identities and PyTorch↔ONNX parity evidence;
- calibration manifest with teacher/DATA identity;
- setup-lineage validator and setup manifest contract;
- historical V2 bundle validator with explicit artifact SHA-256 identities;
- dependency-light setup-lineage tests, all green remotely.

**Remaining evidence gap:** the new real-verifier CI lane currently does not yet establish that tracked `zkml/ezkl/proof.json` verifies against canonical `contracts/src/ZKMLVerifier.sol`. This is isolated from V3 registry logic and must be resolved without regenerating either artifact blindly.

## ZC-P1-005 — duplicate generated verifier sources created ambiguous authority

**Status:** FIXED REMOTELY

The stale `contracts/standalone/ZKMLVerifier.sol` copy was removed. `contracts/src/ZKMLVerifier.sol` is the sole tracked canonical generated verifier source.

## ZC-P1-006 — verifier rotation was described but not implemented

**Status:** FIXED AND REMOTELY TESTED IN V3

V3 has a dedicated appended verifier with owner-only rotation and explicit events. Historical verifier storage/semantics remain intact.

## ZC-P1-007 — contract dependencies were not reproducible from a fresh root clone

**Status:** FIXED REMOTELY

`contracts/scripts/bootstrap_deps.sh` reconstructs exact dependency revisions from the existing lock identities. GitHub Actions uses the same deterministic bootstrap.

## ZC-P1-008 — staking is instantaneous eligibility, not persistent audit accountability

**Status:** OPEN PRODUCTION-POLICY LIMITATION; NOT A V3 CRYPTOGRAPHIC BLOCKER

`SentinelToken` checks `stakedBalance >= MIN_STAKE`, but an agent can unstake immediately after submission. Slashing is a separate owner action and there is no per-audit bond/challenge lock.

**Interpretation:** the current mechanism proves stake eligibility at transaction time. It must not be described as durable per-audit economic accountability.

**Disposition:** do not invent a lock period or tokenomics during ZKML alignment. Define the accountability/slashing policy separately before production claims require stronger economics.

## ZC-P2-001 — stale proxy comments/parameter count/output terminology

**Status:** EXECUTABLE SOURCE FIXED; README CLEANUP PENDING

Actual student architecture is 128→64→32→10 with 10,666 parameters and probability-regression output semantics.

## ZC-P2-002 — `zkml/README.md` and `contracts/README.md` are materially stale

**Status:** OPEN DERIVED-DOCUMENT CLEANUP

They still describe historical V1/65-signal behavior, obsolete direct-signing flows, incorrect artifact paths/parameter counts and outdated verifier/deployment assumptions. They will be rewritten from the accepted executable state rather than patched incrementally.

# V3 protocol now implemented

`AuditRegistry` appends V3 storage without reordering historical state.

The signed context binds:

- target address + runtime bytecode hash;
- round ID;
- teacher checkpoint identity;
- proxy/circuit bundle identity;
- DATA version identity;
- class-schema identity;
- expiry deadline;
- submitting agent;
- exact `keccak256(proof)`;
- exact `keccak256(abi.encode(publicSignals))`;
- exact `keccak256(abi.encode(classScoreFelts))`;
- EIP-712 domain: chain ID + registry proxy address.

Acceptance requires:

1. V3 initialized and legacy writes disabled;
2. target has deployed bytecode;
3. policy attestation unexpired;
4. minimum current stake;
5. exact 138-signal proxy-proof layout;
6. all ten score felts equal proof public outputs;
7. unused request digest;
8. signature from configured policy signer;
9. V3 verifier accepts the proof.

Replay state is marked before the external verifier call; a reverted/failed verifier call atomically rolls the marker back.

# Remote validation evidence

The system-alignment workflow currently exercises four independent concerns.

## Green: ZKML boundary/lifecycle

- read-only V2 proof decoder containment tests;
- V2 artifact-bundle structural validator;
- ONNX/calibration/setup lineage fail-closed tests.

## Green: V3 Python policy protocol

- deterministic EIP-712 request construction;
- all context fields change digest;
- invalid/expired/tampered requests rejected;
- legacy proof-only path remains rejected.

## Green: ordinary Solidity/Foundry registry suite

- ordinary compiler profile (no permanent `via-IR` requirement);
- V1/V2 historical tests;
- V3 adversarial substitution/replay tests;
- Python↔Solidity EIP-712 golden digest;
- pre-V3→V3 UUPS history/storage preservation;
- V3 legacy-write shutdown.

The earlier stack-too-deep problem was removed structurally by reducing V3 codegen pressure while preserving the golden digest contract.

## Open: canonical generated Halo2 verifier vs tracked historical proof

A dedicated CI lane now:

1. converts tracked `zkml/ezkl/proof.json` into a generated Solidity fixture;
2. deploys the actual canonical `Halo2Verifier`;
3. separately tests the tracked proof and a mutated public output.

This lane remains the outstanding remote cryptographic-artifact question. Its result is not allowed to alter the already-green V3 registry/policy evidence: it is an artifact-pair compatibility question.

# Hash naming contract

New V3 interfaces must name the algorithm explicitly:

- `teacher_checkpoint_sha256`
- `proxy_checkpoint_sha256`
- `proof_sha256` for off-chain artifact identity where retained
- `proof_keccak256` for EVM/V3 binding
- `public_signals_keccak256`
- `class_score_felts_keccak256`
- `request_digest`

A new generic `proof_hash` field is forbidden at module boundaries because historical code used both SHA-256 and Keccak under that name.

# Still required before merge / production claims

1. Resolve the canonical generated verifier ↔ tracked historical proof lane without blindly regenerating artifacts.
2. Apply the minimal direct-score patch to hardened AGENTS `_submit.py` and add focused regression coverage.
3. Integrate an actual isolated signer service that signs only validated V3 requests; no private key belongs in the analysis process.
4. Locally run EZKL witness → prove → off-chain verify for the canonical bundle/version selected for continuation.
5. Run Anvil service-level ML → proxy → proof → policy attestation → V3 registry submission and adversarial substitutions.
6. Verify deployed bytecode/ABI/verifier/artifact identities before any network deployment.
7. Decide separately whether production accountability requires per-audit stake locking/slashing semantics.
8. Rewrite stale `zkml/` and `contracts/` READMEs from the accepted executable state.

PR #62 remains draft and must not be merged as a finished production alignment until the required cryptographic/runtime checks are closed.
