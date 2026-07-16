# SENTINEL D2 ZKML/contracts source-engineering audit appendix

**Audit date:** 2026-07-14
**Runtime baseline:** `4b5bd333c`
**Mode:** audit only; recovered track evidence normalized to stable D2 IDs
**Track status:** `track-reproduced`; D2-ZKC-001 is independently primary-reproduced on isolated local Anvil

Completed the bounded ZKML/contracts audit without editing tracked files. The D2 worktree remains clean.

## Baseline evidence

- Clean ZKML suite: **34 passed, 3 skipped**
  - Skips require the ignored teacher checkpoint, proving key/SRS, and live GPU/EZKL prerequisites.
- Clean tracked Foundry suites: **52 passed**
  - 15 registry, 3 invariant, 20 combined, 14 token.
  - A plain fresh-clone `forge test` fails because `contracts/lib` is absent.
  - Tests pass when using the separately installed local Forge/OpenZeppelin libraries.
- Ignored local `AuditRegistryV2.t.sol`: **14 passed**, but absent from a fresh clone due the root `test` ignore rule.
- Current local total of 66 therefore must not be described as fresh-clone coverage.
- Current proof:
  - 138 public signals: 128 fusion values + 10 outputs.
  - 3,072 proof bytes.
- Runtime bytecode:
  - Verifier: 13,425 bytes.
  - AuditRegistry: 7,961 bytes.
  - SentinelToken: 3,053 bytes.
- One warm local proof run:
  - witness 0.503 s
  - prove 2.143 s
  - verify 0.023 s
- Current PT, ONNX, compiled model, VK, generated verifier, and tracked proof are mutually compatible in the tested path.
- Storage inspection shows V1 `_audits` at slot 2 and V2 `_auditsV2` appended at slot 3. This is encouraging, but no historical-layout upgrade test exists.

## Findings

### D2-ZKC-001 — P0 defect/protocol exploit: proof replay and audit identity are unbound

**Symbols**

- `contracts/src/AuditRegistry.sol::AuditRegistry.submitAuditV2`
- `contracts/src/ZKMLVerifier.sol::Halo2Verifier.verifyProof`
- `agents/src/mcp/servers/audit/_submit.py::_run_submit`
- `zkml/ezkl/settings.json::run_args`

**Evidence**

The proof contains only the supplied fusion vector and proxy outputs. It contains no chain ID, contract address, runtime-code hash, reference block, source hash, audit round, teacher hash, manifest hash, deadline, or nonce.

`submitAuditV2` verifies the proof and output fields, then independently accepts caller-supplied `contractAddress` and `modelHash`. It does not consume proof hashes or round IDs.

On Anvil, the exact same valid proof/signals were successfully submitted:

1. For target A with model hash zero.
2. Again for target B with an unrelated all-`ff` model hash.

Both histories were accepted as verified.

**Violated invariant**

A verified record must be inseparably bound to one audit identity, execution manifest, result, and round.

**Impact**

One valid proof can create apparently verified histories for arbitrary contracts and model identities, including repeated submissions.

**Recommendation**

Introduce a V3 typed commitment binding identity, manifest, canonical scores, deterministic verdict, evidence, proof envelope, active-set snapshot, deadline, and nonce. Require matching EIP-712 attestations from quorum and consume the round/digest.

Define the proof envelope hash over verifier/circuit identity, proof bytes, and public signals—not proof bytes alone.

**Rejected alternatives**

- A used-proof mapping alone does not bind target, model, source, or round.
- Adding unconstrained identity values as circuit public inputs does not prove a relationship. Any passthrough must be constrained or bound by the quorum commitment.

**Migration/rollback**

Deploy a new V3 coordinator. Preserve V1/V2 reads and pause or clearly deprecate legacy writes after shadow verification. Rollback pauses new V3 finalizations without rewriting completed records.

**Required tests**

Cross-target, cross-model, cross-chain, stale-round, duplicate-proof, nonce, deadline, and conflicting-commitment adversarial tests using the real verifier.

---

### D2-ZKC-002 — P1 cryptographic/trust risk: proof semantics are narrower than the stored claim

**Symbols**

- `zkml/src/ezkl/run_proof.py::generate_proof`
- `zkml/src/ezkl/setup_circuit.py::run_pipeline`
- `agents/src/mcp/servers/audit/_submit.py::build_provenance_manifest`

**Evidence**

Settings use:

- public inputs
- public outputs
- fixed parameters
- `check_mode="UNSAFE"`

The proof verifies the fixed proxy computation over an operator-supplied fusion vector. It does not verify:

- raw Solidity source
- deployed runtime bytecode
- teacher execution
- preprocessing
- evidence fusion
- deterministic verdict
- provenance manifest

The official EZKL security guidance also says EZKL remains unaudited and provides no security guarantees.

**Impact**

A valid proof can be described more strongly than it actually warrants.

**Recommendation**

Treat the proof only as a proxy-computation proof. Require quorum to attest the source/code identity, pinned teacher execution, tools, deterministic fusion, and proof envelope belong to the same run. Review safe check mode and the complete compiled circuit before production use.

**Rejected alternative**

An unsigned or ordinary operator provenance document is not equivalent to a ZK proof of teacher execution.

**Compatibility**

No legacy record should be reinterpreted. V3 should explicitly version the proof semantics.

---

### D2-ZKC-003 — P1 correctness/availability defect: fixed transaction gas fails and reverted receipts are reported as submitted

**Symbol**

- `agents/src/mcp/servers/audit/_submit.py::_run_submit`

**Evidence**

Measured locally with the real generated verifier and tracked proof:

- `verifyProof`: approximately 655,315 gas.
- Full nonzero V2 submission estimate: 1,016,818 gas.
- Successful submission with 1.1 million limit: 1,002,498 gas.
- Current hardcoded 1,000,000 limit reverted with receipt status zero.

After waiting for the receipt, `_run_submit` unconditionally sets `status="submitted"` and never checks `receipt.status`.

`_SUBMIT_CONFIRM_BLOCKS` is also imported but unused.

**Impact**

The normal real-verifier path can fail while clients receive a false submitted result.

**Recommendation**

Estimate gas with a bounded policy and margin, use current fee mechanics, require `receipt.status == 1`, wait the configured confirmations, and verify the emitted commitment/round.

**Tests**

Real-verifier gas test, deliberate out-of-gas receipt, status-zero receipt, delayed confirmation, replacement transaction, and reorg tests.

---

### D2-ZKC-004 — P1 concurrency/operational risk: proof artifacts and transaction nonces are shared

**Symbols**

- `agents/src/mcp/servers/audit/_submit.py::_run_submit`
- `zkml/src/ezkl/run_proof.py::{PROOF_INPUT,WITNESS,PROOF}`

**Evidence**

All jobs share:

- `zkml/ezkl/proof_input.json`
- `zkml/ezkl/witness.json`
- `zkml/ezkl/proof.json`
- one operator account nonce read with `get_transaction_count`

The submission service deletes these files after each job, although the files are tracked in Git. Concurrent jobs can overwrite, delete, or submit one another’s artifacts and race the same nonce.

**Recommendation**

Use content-addressed per-job/per-operator workspaces, exclusive leases, atomic writes, immutable outputs, and an account nonce manager or serialized broadcaster.

**Rejected alternative**

A process-local lock is insufficient for multiple workers/operators.

**Tests**

Concurrent proof jobs, failure during atomic publish, worker restart, duplicate nonce, replacement transaction, and proof/workspace ownership tests.

---

### D2-ZKC-005 — P1 scientific/correctness defect: proxy outputs have contradictory probability/logit semantics

**Symbols**

- `zkml/src/distillation/proxy_model.py::ProxyModel.forward`
- `zkml/src/distillation/train_proxy.py::train`
- `zkml/src/distillation/corpus_distill.py::main`
- `agents/src/mcp/servers/audit/_submit.py::_run_submit`
- `zkml/src/ezkl/run_proof.py::extract_corpus_contract_features`

**Evidence**

Training minimizes:

`MSE(proxy_raw_output, sigmoid(teacher_logits))`

Therefore the raw proxy output is trained as a probability estimate. However:

- `ProxyModel` calls it a raw logit.
- The service and proof CLI apply another sigmoid.
- EZKL proves the raw ONNX output.
- The service later overwrites on-chain felts with the raw proof output.
- Provenance records the extra-sigmoid floating values.

On ten tracked corpus contracts:

- Raw proxy agreement with teacher at 0.5: 0.96.
- Extra-sigmoid agreement: 0.40.
- Raw agreement at production per-class thresholds: 0.94.
- Extra-sigmoid agreement: 0.51.

These are illustrative in-corpus measurements, not held-out scientific validation.

**Recommendation**

Choose one versioned semantic contract:

1. Preferably include the bounded probability activation inside the proxy/ONNX/circuit and retrain/regenerate everything.
2. Or explicitly define current raw outputs as probability estimates and remove every extra sigmoid.

Never transform proof outputs outside the circuit while claiming the transformed values were proved.

**Tests**

Teacher → raw proxy → ONNX → quantized circuit → Solidity decoding parity across held-out data and boundary cases.

---

### D2-ZKC-006 — P1 scientific risk: distillation promotion is not fail-closed or reproducible

**Symbols**

- `zkml/src/distillation/train_proxy.py::train`
- `zkml/src/distillation/corpus_distill.py::main`
- `zkml/src/ezkl/run_proof.py::generate_proof`
- `zkml/tests/test_distillation.py`

**Evidence**

- Only a state dict is saved; teacher hash, data hash, splits, thresholds, metrics, seed, environment, and circuit identity are absent.
- The 95% target is logged but not enforced as a promotion gate.
- Agreement is micro element-wise threshold accuracy, which can be inflated by dominant negatives.
- Distillation uses uniform 0.5, while the local teacher’s operational thresholds range from 0.05 to 0.5.
- Corpus mode uses a random split of a small hand-authored corpus without an independent test set.
- `generate_proof` documents disagreement as a failure but selects the best of ten and proceeds even with disagreements.
- Tests copy the agreement function inline instead of exercising production source.

**Recommendation**

Create an immutable promotion manifest with independent held-out evaluation, classwise agreement, calibration error, threshold agreement, exact-match rate, worst-class results, quantization error, teacher/data hashes, and enforced gates.

**Rejected alternative**

A single 95% micro-agreement number is not enough to authorize a verifier.

---

### D2-ZKC-007 — P1 provenance/artifact defect: fixed parameters invalidate the “keys survive retraining” assumption

**Symbols**

- `zkml/src/ezkl/setup_circuit.py::run_pipeline`
- `zkml/ezkl/settings.json::run_args.param_visibility`
- `zkml/src/distillation/export_onnx.py::export`

**Evidence**

Parameters are `Fixed`. Source comments claim proving/verifying keys survive weight retraining because only architecture matters.

I changed one weight in a temporary ONNX while preserving the architecture, compiled it with current settings, and attempted to use the current PK/VK. Verification rejected it with an unsatisfied constraint system.

Therefore weights are bound into the compiled circuit/key/verifier.

No runtime check confirms compatibility among:

- proxy checkpoint
- ONNX plus external data
- settings
- compiled circuit
- PK/VK
- Solidity verifier bytecode

**Impact**

A retrained local proxy can silently produce preview/provenance values different from the circuit actually proved on-chain.

**Recommendation**

Create a content-addressed artifact bundle manifest and fail before inference/proving when any hash differs. Retraining requires ONNX export, compile, setup, verifier generation/deployment, and verifier-registry activation.

**Tests**

Single-byte/weight mutation at every seam must fail before proof generation.

---

### D2-ZKC-008 — P1 provenance/integration defect: model and proof hashes diverge across boundaries

**Symbols**

- `agents/src/mcp/servers/audit/_submit.py::{_run_submit,build_provenance_manifest}`
- `contracts/src/AuditRegistry.sol::submitAuditV2`

**Evidence**

- ML’s actual model hash updates `result["model_hash"]`.
- The transaction still submits the original caller-provided `model_hash`.
- Validation checks only string length, not hexadecimal validity or equality with ML.
- Service returns SHA-256 of proof bytes.
- Contract stores Keccak-256 of proof bytes.
- Provenance receives an empty operator address due `_OPERATOR_KEY and ""`.
- Manifest records extra-sigmoid scores rather than proof output felts.
- It excludes proof hash, target, source/code identity, chain, verifier, circuit, round, nonce, and deadline.
- It may be unsigned and is not durably/content-addressably published.

**Recommendation**

Replace ambiguous `modelHash` with separate teacher, proxy, circuit, verifier-code, tool, schema, and configuration hashes inside one canonical execution manifest. Use one domain-separated proof-envelope hash and EIP-712 signatures.

---

### D2-ZKC-009 — P1 compatibility defect: one verifier cannot service both V1 and V2 write paths

**Symbols**

- `contracts/src/AuditRegistry.sol::{zkmlVerifier,submitAudit,submitAuditV2}`
- `contracts/src/ZKMLVerifier.sol::Halo2Verifier.verifyProof`
- `contracts/standalone/ZKMLVerifier.sol::Halo2Verifier.verifyProof`

**Evidence**

- Registry stores one verifier address.
- V1 expects 65 signals.
- V2 expects 138 signals.
- Current generated verifier enforces exactly 138.
- Standalone legacy verifier enforces 65.
- The real V2 verifier reverted on a 65-signal call.
- Mock tests claim V1/V2 coexistence because the mock ignores shape.

**Impact**

Only one real write path can function at a time.

**Recommendation**

Use an immutable versioned verifier registry keyed by circuit ID and signal-layout hash. Preserve legacy reads; freeze or route legacy writes explicitly.

---

### D2-ZKC-010 — P1 protocol gap: stake does not secure an audit round

**Symbols**

- `contracts/src/SentinelToken.sol::{stake,unstake,slash}`
- `contracts/src/AuditRegistry.sol::submitAuditV2`

**Evidence**

- Any account holding minimum stake can append.
- Stake can be withdrawn immediately after submission.
- There is no operator admission, active-set snapshot, round lock, challenge period, or unbonding cooldown.
- Owner can slash arbitrary amounts without objective evidence.
- Initial supply is entirely controlled by the deployer.

**Recommendation**

Separate operator registry/stake vault states: pending, active, exiting, withdrawn, slashed. Snapshot active operators per round, lock stake through finality/challenge, use cooldown, and permit slashing only for objectively proven faults such as double-signing conflicting commitments.

---

### D2-ZKC-011 — P1 architecture gap: no quorum, round, or finality exists

**Symbol**

- `contracts/src/AuditRegistry.sol::AuditRegistry`

Each submission is an independent append. Multiple contradictory “verified” audits are valid history. There is no canonical result.

**Recommendation**

V3 finalizes only one matching deterministic commitment after `ceil(2N/3)` unique active-set attestations. LLM/RAG output must remain outside that commitment.

---

### D2-ZKC-012 — P1 governance/upgrade risk: immediate deployer control

**Symbols**

- `contracts/src/AuditRegistry.sol::{pause,unpause,_authorizeUpgrade}`
- `contracts/src/SentinelToken.sol::slash`
- `contracts/script/Deploy.s.sol::Deploy.run`

**Evidence**

Deployment leaves token ownership, registry pause, upgrades, and slashing under the deployer. There is no multisig, timelock, verifier-change delay, upgrade proposal state, or storage-layout gate.

Initialization does not reject zero/non-contract verifier or token addresses.

**Recommendation**

Use a multisig-owned timelock for admission, verifier activation, upgrades, and parameter changes. Give a narrowly scoped guardian pause authority. Require codehash, interface, layout, and activation-delay checks.

**Migration**

Prefer a new V3 coordinator rather than mutating the V1/V2 proxy.

---

### D2-ZKC-013 — P1 fresh-clone/CI/artifact availability gap

**Evidence**

- Fresh-clone Foundry fails before collection due absent libraries.
- `contracts/.gitmodules` declares dependencies but there are no Git gitlinks.
- `contracts/.github/workflows/test.yml` is nested and not an active repository workflow.
- Root `.gitignore` rule `test` hides new contract tests, including V2 coverage.
- Teacher checkpoint, thresholds, 138 MB proving key, and 4 MB SRS are ignored/local-only with no tracked DVC pointer or acquisition manifest.
- Tracked proof/witness files are mutable runtime artifacts.

**Recommendation**

Add a root workflow, deterministic dependency acquisition from the lock, narrow ignore rules, track V2 tests, and publish a verified artifact acquisition manifest. Keep per-run proof/witness outputs out of canonical source control.

---

### D2-ZKC-014 — P1 secret-management defect

**Symbol**

- `zkml/src/ezkl/extract_calldata.py::RPC_URL`

A provider credential is embedded in tracked source.

**Recommendation**

Rotate/revoke it, load RPC only from environment/secret storage, and add repository secret scanning. Do not reproduce the value in audit documentation.

---

### D2-ZKC-015 — P1 verifier/deployment lifecycle gap

**Symbols**

- `zkml/src/ezkl/setup_circuit.py::run_pipeline`
- `contracts/script/Deploy.s.sol::Deploy.run`
- `zkml/src/ezkl/extract_calldata.py::main`

**Evidence**

- Setup does not generate the EVM verifier even though later steps assume it exists.
- Verifier deployment is a separate manual operation.
- Addresses are hardcoded/defaulted in multiple locations.
- No deployment manifest binds verifier address/codehash to VK/settings/circuit/proxy.
- `contracts/standalone/ZKMLVerifier.sol` is a stale 65-signal verifier alongside the current 138-signal source.
- Deployment validates address equality but not codehash or signal layout.

**Recommendation**

Automate immutable verifier generation, codehash verification, deployment receipt capture, registry activation, and artifact-manifest publication.

---

### D2-ZKC-016 — P2 testability debt: mocks hide the trust-boundary failures

**Symbols**

- `contracts/test/mocks/MockZKMLVerifier.sol::MockZKMLVerifier.verifyProof`
- `contracts/test/InvariantAuditRegistry.t.sol::invariant_audit_count_monotonic`
- `zkml/tests/test_provenance.py`
- `zkml/tests/test_distillation.py`

**Evidence**

- Contract tests do not use a real proof/verifier.
- The “monotonic” invariant only checks count ≤ total and fixed targets are not the handler’s random targets, making it largely vacuous.
- Provenance tests construct dictionaries rather than importing the production function.
- Distillation tests copy logic.
- Upgrade tests upgrade to the same implementation and do not compare storage layouts.

**Recommendation**

Add real verifier integration, adversarial replay tests, production-symbol tests, meaningful stateful invariants, and storage-layout golden files.

---

### D2-ZKC-017 — P2 gas/storage scalability debt

**Symbols**

- `contracts/src/AuditRegistry.sol::{AuditResultV2,getAuditHistoryV2}`

A V2 record occupies 448 bytes/14 storage slots, and full histories are copied to memory without pagination. Quorum signatures stored similarly would be prohibitively expensive.

**Recommendation**

V3 stores only compact finality state and roots. Put detailed evidence, proof, and signatures in content-addressed off-chain storage; expose paginated/indexed events and views.

---

### D2-ZKC-018 — P2 dependency/configuration debt

- Foundry lock uses OpenZeppelin `v5.6.0-rc.0`.
- Foundry actually compiles with 0.8.22 while source comments repeatedly state 0.8.20.
- The generated verifier successfully compiled with 0.8.22, contradicting the manual ≤0.8.17 lifecycle claim.

Before production, use stable pinned dependencies and make compiler policy executable rather than comment-based.

## V3 coordinator constraints

### Typed identity

```text
AuditIdentity {
  chainId
  target
  runtimeCodeHash
  referenceBlock
  optionalSourceHash
  roundId
}
```

A historical runtime-code hash must come from a pinned archive RPC and be attested by quorum; the coordinator cannot recover historical code using current `EXTCODEHASH`.

### Execution manifest

Separate hashes for:

- DATA/schema/export
- teacher checkpoint and thresholds
- proxy checkpoint
- ONNX and external data
- EZKL settings/compiled circuit/PK-VK identity
- verifier address/codehash
- tool images/versions
- deterministic configuration
- class ordering and score encoding

### Deterministic commitment

```text
DeterministicCommitment {
  identityHash
  manifestHash
  normalizedScoresHash
  deterministicVerdictRoot
  evidenceRoot
  proofEnvelopeHash
  activeSetId
  deadline
  nonce
}
```

`proofEnvelopeHash` should bind:

```text
circuitId
verifier
keccak256(proof)
keccak256(canonicalPublicSignals)
```

A versioned verifier adapter should enforce exact signal count, output positions, encoding, and valid score range before deriving the normalized score hash.

### Operator/quorum model

- Active set snapshot fixed per round.
- N restricted to the governed pilot range 5–9.
- Threshold `ceil(2N/3)`:
  - 5 → 4
  - 6 → 4
  - 7 → 5
  - 8 → 6
  - 9 → 6
- Unique signer bitmap is sufficient for the pilot.
- Same operator cannot attest two commitments for a round.
- Conflicting valid EIP-712 signatures are objective slash evidence.
- Support EOA recovery and optionally ERC-1271 operator identities.

### Round state machine

```text
OPEN → COLLECTING → FINALIZED
  └──────────────→ EXPIRED/CANCELLED
```

Finalized state is immutable. Membership, verifier, and manifest cannot change after round creation.

### Compact storage

Store only:

- final commitment digest
- identity/manifest/evidence roots
- active-set ID
- attester bitmap
- finalized block/time
- status

Proofs, public signals, signatures, reports, and detailed evidence remain content-addressed off-chain.

### Governance

- Multisig plus timelock controls admission, verifier activation, pause, and upgrades.
- Emergency guardian can pause but not upgrade or confiscate.
- Verifiers are added/versioned, not silently replaced.
- Existing rounds retain their verifier and active-set snapshot.
- Objective-fault slashing only.
- Unbonding exceeds maximum round plus challenge duration.

### Truth boundary

- EZKL proves only the fixed proxy computation.
- Quorum attests source/code identity, teacher execution, deterministic tools/fusion, and proof belong to one manifest.
- LLM/RAG narrative has a separate advisory root and cannot change the deterministic commitment.
- Any evidence channel that cannot reproduce byte-identical canonical results under a pinned manifest remains advisory.

### Migration

1. Stabilize artifact validation, proof workspaces, transaction handling, and CI.
2. Implement shared Python/Solidity commitment test vectors.
3. Deploy new operator vault, verifier registry, and V3 coordinator.
4. Run V3 in shadow mode against existing audits.
5. Validate 5–9 operator quorum, gas, recovery, and disagreement behavior.
6. Governed cutover for new rounds.
7. Preserve V1/V2 reads permanently; do not reinterpret legacy records.

### Essential V3 tests

- EIP-712 domain, chain, coordinator, nonce, and deadline replay.
- Same proof against another identity/model/manifest must fail.
- All quorum thresholds for N=5…9.
- Duplicate, inactive, exiting, and post-snapshot operators.
- Conflicting commitments and objective equivocation evidence.
- Stake lock, unbonding, and withdrawal across active rounds.
- Real verifier signal-layout and score-range checks.
- Artifact-hash mutation and verifier-codehash mismatch.
- Concurrent proof jobs and worker recovery.
- V1/V2 historical read compatibility.
- Storage-layout and governed upgrade checks.
- Worst-case real-verifier plus nine-attestation gas benchmark.
