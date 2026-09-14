# Contracts / On-Chain Completion Plan

**Status:** ACTIVE — audit may proceed; verifier-lineage migration waits for ZKML handoff  
**Parent:** `00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md`  

## 1. Objective

Keep SENTINEL’s on-chain trust boundary aligned with the selected ZKML/provenance lineage, validate upgrade/storage/signature/proof behavior, and decide explicitly whether autonomous signing/broadcast/finality belongs in the intended product scope.

The current contracts are already substantial. This plan is therefore an audit/reconciliation/completion plan, not a rewrite.

## 2. Current executable baseline

Primary sources include:

- `contracts/src/AuditRegistry.sol`
- `contracts/src/IZKMLVerifier.sol`
- `contracts/src/ZKMLVerifier.sol`
- `contracts/src/SentinelToken.sol`
- `contracts/test/AuditRegistry*.t.sol`
- `contracts/test/InvariantAuditRegistry.t.sol`
- `contracts/test/ZKMLVerifierRealProof.t.sol`
- deployment/verification scripts under `contracts/script/` and `contracts/scripts/`
- agents policy-request and V3 submission-state code.

Current V3 behavior already separates:

- proxy-proof verification;
- EIP-712 policy/provenance attestation;
- signer state;
- transaction state;
- finality state.

The analysis/MCP side intentionally has no production private key, transaction broadcast or receipt-finality authority.

## 3. Trust claims that must remain separate

Never collapse these into one “verified” boolean in planning, code, UI or reports:

1. proxy ZK proof verified;
2. public signals match supplied class-score felts;
3. V3 context/provenance signature is valid;
4. target bytecode/context matches the signed request;
5. caller has required stake/authorization;
6. transaction was signed;
7. transaction was broadcast;
8. transaction was confirmed and not reverted/replaced;
9. on-chain record exists;
10. the underlying teacher/audit conclusion is actually correct.

The contract can enforce several of these protocol statements. It cannot establish the final ML/security-truth claim merely from a valid proxy proof.

## 4. Work package C0 — contract and test audit

**State:** `AUDITING`

Trace and document:

- V1/V2 historical write paths;
- V3 initialization and legacy-disable transition;
- staking requirements;
- pause/unpause authority;
- verifier update authority;
- policy-signer update authority;
- request-digest construction;
- replay protection;
- proof/public-signal checks;
- code-hash binding;
- storage layout across upgrades;
- read/query functions and event semantics;
- token/staking assumptions;
- real-proof test fixture provenance.

Review unit, negative, upgrade, invariant and real-proof tests for missing protocol branches rather than relying on test count.

## 5. Work package C1 — new ZKML verifier compatibility

**State:** `BLOCKED` until ZKML produces an accepted bundle/verifier identity

When ZKML hands off a new verifier:

1. verify `IZKMLVerifier` interface compatibility;
2. verify exact public-signal count/order and field encoding;
3. verify proxy output offsets/classes;
4. compare generated verifier behavior with the existing real-proof fixture;
5. decide whether simple `setZkmlVerifierV3()` replacement is sufficient or whether protocol/storage changes are required;
6. bind verifier address/artifact hash to the release/deployment record;
7. preserve old verifier lineage for rollback/history.

Do not assume equal input/output dimensions imply verifier compatibility.

## 6. Work package C2 — upgrade/storage safety

Audit UUPS storage compatibility from historical/pre-V3 layout through current implementation and any proposed successor.

Required checks:

- no reordered/removed occupied storage slots;
- new storage append-only where required;
- initializer/reinitializer versions correct;
- implementation constructor disables initialization;
- upgrade authorization remains owner-controlled as intended;
- upgrade preserves historical audit records;
- upgrade preserves V3 replay-protection state;
- pause and trust-root settings remain coherent after upgrade.

Run storage-layout tooling where available in addition to Foundry behavioral upgrade tests.

## 7. Work package C3 — cross-language V3 digest/provenance parity

Maintain golden-vector agreement between:

- Solidity `computeAuditDigestV3` / internal digest logic;
- Python `agents/src/security/policy_signer.py`;
- any isolated signing-service implementation introduced later.

Test variations in:

- chain ID;
- registry address;
- agent;
- target address/code hash;
- round ID;
- teacher/proxy/data/schema hashes;
- proof/public-signal/class-score hashes;
- deadline.

Any one-field mutation should alter the digest as expected and invalidate a stale signature.

## 8. Work package C4 — V3 authorization and adversarial tests

Ensure tests cover at least:

- insufficient stake;
- expired policy signature;
- wrong signer;
- wrong chain/domain;
- wrong target code hash;
- proof mutation;
- public-signal mutation;
- class-score mismatch;
- reused request digest/replay;
- verifier rejection;
- paused registry;
- legacy writes after V3 disables them;
- trust-root rotation;
- unauthorized upgrade;
- target with no code;
- malformed/incorrect proof signal count.

Use a real generated verifier/proof fixture for at least one end-to-end positive path after ZKML rebuild.

## 9. Work package C5 — invariant/property testing

Review and extend invariants around:

- replay protection cannot be bypassed;
- only verified V3 submissions become verified V3 records;
- recorded identity fields equal the validated request context;
- legacy-disabled state never re-enables legacy submission implicitly;
- pause blocks write paths;
- owner-only trust-root changes remain protected;
- audit history remains append-only under normal protocol operations;
- staking checks cannot be bypassed through alternate write paths.

If fuzzing finds a protocol ambiguity, resolve semantics before adding a narrow regression test.

## 10. Work package C6 — deployment/upgrade rehearsal

Before any real network action:

- deploy/redeploy locally or on an ephemeral test environment;
- initialize V3;
- configure verifier/policy signer;
- stake as required;
- submit a known real proof/context request;
- query stored result;
- rotate verifier/signer in rehearsal;
- perform an upgrade rehearsal;
- confirm historical state persists;
- rehearse pause/emergency behavior;
- capture gas/runtime observations where useful.

No production/mainnet deployment, real funds, irreversible upgrade, or real secret-key action is authorized by this plan alone. Those remain explicit human-controlled external actions.

## 11. Work package C7 — isolated signer/broadcaster scope decision

**State:** `DEFERRED / PRODUCT-SCOPE DECISION`

Current code deliberately stops at unsigned policy-request preparation and state modeling. Decide whether Sentinel should actually own an isolated production submission service.

If **not** in scope:

- keep the boundary explicit;
- document V3 request preparation as the terminal automated capability;
- do not imply autonomous on-chain finality.

If **in** scope, design a separate security domain with at minimum:

- isolated key management/signing;
- allowlisted chain/registry;
- request-policy validation;
- nonce management;
- transaction construction;
- gas/fee policy;
- broadcast;
- receipt/reorg/replacement handling;
- idempotency;
- audit log;
- least privilege;
- secret redaction;
- explicit human/automated authorization model.

The analysis MCP must not become the private-key holder simply for convenience.

## 12. Work package C8 — finality truth integration

If signer/broadcaster is later implemented, map actual external state into the existing independent state dimensions:

```text
eligible
→ signed
→ prepared
→ broadcast
→ pending
→ confirmed / reverted / dropped / replaced / failed
```

Positive `verified_audit_finality` may be asserted only after contract confirmation and all independent proof/context/signer/transaction/finality states are mutually consistent.

## 13. Work package C9 — documentation/public-claim reconciliation

After final technical state is known, verify that:

- README/handbook does not imply ZK proves the whole audit;
- “verified” wording distinguishes proof, attestation and transaction finality;
- historical V1/V2 compatibility is not presented as current preferred protocol;
- signer/broadcaster absence or presence is explicit;
- deployed verifier/registry addresses are recorded only when real and current.

## 14. Stop/fail conditions

Stop and investigate if:

- new verifier/public-signal semantics do not match registry assumptions;
- storage layout is unsafe;
- Python/Solidity digest parity breaks;
- replay protection can be bypassed;
- stale signatures survive identity/domain changes;
- generated verifier provenance is unclear;
- proof/context/finality claims become conflated;
- implementation requires real keys/funds/irreversible external actions without explicit human authorization.

## 15. Completion criteria

Contracts/on-chain is `ACCEPTED` when the selected ZKML verifier lineage is protocol-compatible, V3 digest/provenance behavior is cross-language verified, upgrade/storage/invariant/adversarial tests pass, and the deployment authority boundary is explicit. Autonomous signer/broadcaster/finality may remain `DEFERRED` without blocking technical completion unless the project explicitly adopts it as a required product capability.
