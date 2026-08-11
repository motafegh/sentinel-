# SENTINEL ZKML + Contracts Alignment Audit

**Date:** 2026-08-11  
**Branch:** `system/zkml-contracts-alignment`  
**Status:** REMOTE SOURCE/PROTOCOL ALIGNMENT COMPLETE  
**Scope:** `zkml/`, `contracts/`, and their active ML/AGENTS security seams.  
**Method:** executable source first; historical documentation treated as supporting evidence only.

## Conclusion

The stale ZKML/contracts baseline has been reconstructed and aligned remotely without changing R4 DATA/ML label policy, teacher architecture, thresholds, or protected training artifacts.

The resulting baseline has two intentionally separate layers:

1. **Historical V2 neural proof** — proves the 128→10 proxy computation over the supplied public fusion inputs and proxy outputs. It remains `legacy_proxy_only_unbound` and is not a standalone finality protocol.
2. **V3 submission protocol** — combines that exact proxy-proof statement with an EIP-712 policy attestation that binds the audit context. This does not inflate the ZK claim; it supplies a separately authenticated context assertion enforced by `AuditRegistry`.

Remote CI validates the source/protocol baseline. A newly retrained proxy/circuit/verifier bundle is deliberately deferred until R4 promotes the intended DATA/ML lineage.

## Canonical executable seam

```text
Solidity source
    ↓
ML /fusion-embedding
    ↓
128-d fusion embedding + teacher checkpoint identity
    ↓
ProxyModel: 128 → 64 → 32 → 10
    ↓
10 direct student scores
    ↓
EZKL V2 proof: 128 public inputs + 10 public outputs = 138 signals
    ↓
V3 policy request
    ├─ proof/public-signal/score hashes
    ├─ target address + runtime codehash
    ├─ agent + round
    ├─ teacher/proxy/DATA/class-schema identities
    ├─ chain ID + registry domain
    └─ expiry
    ↓
EIP-712 policy attestation
    ↓
AuditRegistry.submitAuditV3
```

## Closed findings

### ZC-P0-001 — legacy direct-send helper bypassed signer isolation

**Status: CLOSED**

Historical `zkml/src/ezkl/extract_calldata.py` generated a raw-key `cast send` path to `submitAuditV2`.

The helper is now read-only. It may decode/inspect the historical proof but does not emit a transaction, accept a private key, or claim V2 finality eligibility.

### ZC-P0-002 — V2 model identity was caller metadata

**Status: CONTAINED / REPLACED BY V3 CONTEXT ATTESTATION**

V2 historically stored a caller-supplied model hash that was not committed by the neural proof. This limitation is preserved in V2 regression tests rather than hidden.

V3 binds the teacher-model identity, proxy-bundle identity, DATA identity, class-schema identity, exact proof, public signals, scores, target, agent, round, deadline, chain, and registry into the policy-signed request digest.

### ZC-P0-003 — V2 proof was replayable across audit context

**Status: CONTAINED / REPLACED BY V3 CONTEXT ATTESTATION**

The V2 proof itself contains only fusion inputs and proxy outputs. V3 adds replay-resistant context binding with an EIP-712 domain and one-time request digest.

Tracked tests cover replay and target/agent/registry/model/proof/signal substitution.

### ZC-P0-004 — proxy output was double-sigmoided outside the circuit

**Status: CLOSED FOR LIVE RUNTIME**

Training source proves the current student target is `sigmoid(teacher_logits)` and MSE is applied directly to `ProxyModel.forward()`. Therefore the direct student output is the canonical score vector; another sigmoid changes the statement.

The current ZKML proof helper uses the direct score. The historical `_submit.py` compatibility engine still contains the old transform, but it is no longer reachable through the live audit MCP:

- the live server uses `_readonly_handlers.py`;
- exactly three query tools are registered;
- `submit_audit` and unknown write-like calls are rejected at dispatch with `attempted=false`;
- rejection occurs before `_submit.py` is imported;
- the public `audit_server` shim exports the read-only server and no `_handle_submit_audit` symbol.

This avoids modifying the accepted R0 transaction-state engine merely to clean unreachable historical compatibility code.

### ZC-P0-005 — Solidity allowed direct unbound V1/V2 writes

**Status: CLOSED IN V3 ACTIVATION PATH**

`initializeV3()` is a one-way protocol boundary. It configures the V3 verifier/policy signer and permanently disables new V1/V2 writes while preserving historical storage and read APIs.

Fresh deployment activates V3 before the deployment script exits. Existing-registry upgrade uses `upgradeToAndCall(...initializeV3...)` atomically.

### ZC-P1-001 — test coverage was dominated by legacy V1

**Status: CLOSED**

Tracked suites now cover V2 limitations, V3 context/replay behavior, Python↔Solidity EIP-712 digest parity, pre-V3→V3 upgrade/storage preservation, and the real generated verifier with the tracked proof.

### ZC-P1-002 — EZKL `check_mode` was over-interpreted

**Status: CLOSED AS CLASSIFICATION ERROR**

`check_mode` and EZKL version are recorded as artifact metadata. The enum name is not used as a substitute for actual proof validation or as an automatic security verdict. V2 finality ineligibility follows from its unbound proof scope, independently of that setting name.

### ZC-P1-003 — proxy training/calibration silently targeted stale DATA

**Status: CLOSED FOR FUTURE RETRAINS**

Future proxy training and calibration require an explicit DATA export. No old export is silently selected. New checkpoint/calibration metadata binds the relevant lineage identities.

A production-candidate retrain remains deferred until R4 promotes the intended DATA/ML bundle.

### ZC-P1-004 — artifact lineage was implicit

**Status: CLOSED FOR BASELINE / REGENERATE AGAIN FOR FUTURE BUNDLE**

The branch adds explicit checkpoint→ONNX/calibration/circuit bundle identity validation and a machine-readable V2 bundle validator. The tracked proof is also exercised against the canonical generated Solidity verifier in CI.

Every future regenerated bundle must repeat the same identity chain with its own hashes.

### ZC-P1-005 — duplicate generated verifier authority

**Status: CLOSED**

The stale standalone generated verifier was removed. `contracts/src/ZKMLVerifier.sol` is the single tracked canonical generated verifier for the retained V2 evidence bundle.

### ZC-P1-006 — verifier rotation described but absent

**Status: CLOSED IN V3**

V3 has dedicated verifier/policy-signer roots and owner-controlled rotation with explicit events. Historical verifier storage semantics remain untouched for upgrade compatibility.

### ZC-P1-007 — Foundry dependencies not reproducible from root clone

**Status: CLOSED**

`contracts/scripts/bootstrap_deps.sh` checks out the exact revisions recorded in `contracts/foundry.lock`. CI uses the same bootstrap path.

### ZC-P2-001 / ZC-P2-002 — stale source commentary and module READMEs

**Status: CLOSED FOR CURRENT BASELINE**

`zkml/README.md` and `contracts/README.md` now describe the executable 128→10 / 138-signal V2 boundary, direct student-score semantics, V3 trust model, deterministic dependency bootstrap, read-only runtime audit MCP, and R4-dependent regeneration boundary.

## V3 trust model

V3 deliberately uses **dual attestation**.

### Neural proof

The configured ZK verifier verifies the proxy computation represented by the proof/public signals. It does not prove the full teacher model or source-to-fusion derivation.

### Policy attestation

`agents/src/security/policy_signer.py` constructs and validates an unsigned `AuditRequestV3` eligible for a separately isolated signer. The analysis process contains no private key, transaction construction, RPC broadcast, or receipt handling.

The EIP-712 request binds:

- submitting agent;
- target address and runtime bytecode hash;
- round ID;
- teacher checkpoint hash;
- proxy/circuit bundle hash;
- DATA version hash;
- class-schema hash;
- exact proof hash;
- exact public-signals hash;
- exact ten-score hash;
- deadline;
- chain ID and registry proxy address through the EIP-712 domain.

`AuditRegistry.submitAuditV3` verifies the signature, one-time request digest, stake, target bytecode, exact 138-signal layout, score/output equality, deadline, and ZK proof.

## Remote validation evidence

The dedicated `SENTINEL system alignment` GitHub Actions workflow is green on the aligned executable head across five jobs:

1. **zkml-boundary-tests — PASS**
   - dependency-light proof/protocol tests;
   - tracked V2 artifact-bundle validation.

2. **audit-mcp-containment-tests — PASS**
   - exact live tool set is read-only;
   - hidden V2 submit is not exported or dispatched;
   - rejected writes do not import `_submit.py`.

3. **v3-policy-protocol-tests — PASS**
   - EIP-712 request construction, validation, expiry, substitution and digest behavior.

4. **contracts-foundry — PASS**
   - deterministic dependency bootstrap;
   - registry build;
   - complete tracked Foundry suite including V1/V2/V3 and upgrade/storage tests.

5. **generated-verifier-proof — PASS**
   - Solidity fixture generated from tracked `zkml/ezkl/proof.json`;
   - canonical generated `Halo2Verifier` accepts the tracked proof/signals;
   - a mutated public output is rejected fail-closed. The generated verifier may reject by `false` or by revert; successful `true` is forbidden.

## Deliberately deferred work

These are **future production/integration tasks**, not unresolved source-alignment defects.

### Post-R4 proxy/circuit/verifier regeneration

After R4 promotes the intended DATA/ML bundle, regenerate and validate:

1. proxy checkpoint against that explicit export;
2. ONNX + parity/lineage manifest;
3. calibration bundle from the same lineage;
4. EZKL settings/compiled circuit/SRS/proving key/verification key;
5. canonical generated Solidity verifier;
6. fresh proof/public-signal fixture;
7. artifact-bundle identity/protocol report;
8. generated-verifier + Foundry integration;
9. V3 deployment roots for the candidate bundle.

### Isolated signer / live network operations

The repository defines the unsigned policy-request contract but does not claim a production KMS/HSM signing backend or live chain deployment. Those operational components require explicit key-management, deployment, monitoring, and chain-specific acceptance work.

### Full future-bundle service integration

The eventual promoted bundle should be exercised end-to-end through ML → fusion → proxy → proof → isolated policy signer → V3 registry, including Anvil/adversarial and intended-network validation.

## Merge meaning

Merging this alignment branch means:

> the **source/protocol baseline for ZKML + contracts is aligned, fail-closed, reproducible, and remotely validated** relative to the current Sentinel system.

It does **not** mean:

- R4 DATA/ML remediation is complete;
- a new proxy/circuit bundle has been trained/promoted;
- a production signer has been deployed;
- a live network deployment has been accepted.
