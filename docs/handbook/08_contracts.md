# 08 — Contracts and on-chain registry

**Read this when:** you need staking, V1/V2/V3 audit storage, verifier calls, V3 policy binding, deployment, or UUPS upgrades.

**Skip this if:** you only consume off-chain reports and never operate/query chain state.

**Estimated reading time:** 13 minutes.

## 30-second summary

`AuditRegistry` is an upgradeable, staked audit-history registry with three versioned protocols. V1 is historical scalar storage. V2 is the historical ten-score proxy-proof path and does not bind audit context/model identity. **V3 is the current submission protocol:** after `initializeV3`, new V1/V2 writes are disabled, while old history remains readable. V3 verifies the same class of proxy proof plus a separate EIP-712 policy signature that binds target bytecode and audit/model/data/schema identities to the agent, chain, and registry.

## Just-enough mental model

```text
historical:
V1 scalar records ─┐
V2 ten-score records ├─→ remain readable
                    ┘

current V3 write protocol:
staked agent
  + target runtime code
  + proxy proof / 138 signals / 10 scores
  + model/proxy/DATA/schema identities
  + round/deadline
  + policy signature
        ↓
submitAuditV3
        ↓
verify context + signer + anti-replay + proof + output equality
        ↓
append V3 record
```

V3 context attestation and ZK proof are complementary but distinct trust claims.

## Actual runtime/source walkthrough

### Storage/versioning

[`AuditRegistry.sol`](../../contracts/src/AuditRegistry.sol) preserves V1 and V2 storage order, then appends V3 state:

- V3 verifier address;
- V3 policy-signer address;
- `legacySubmissionsDisabled`;
- V3 audit records;
- used-request-digest replay protection.

`initializeV3(verifier, policySigner)` is a reinitializer that sets the V3 trust roots and permanently disables new legacy submissions through the stored flag. Historical V1/V2 query functions remain available.

### Historical writes

`submitAudit` and `submitAuditV2` remain in the ABI for historical/storage compatibility, but both require `!legacySubmissionsDisabled`. Once V3 is initialized, they reject new writes.

### V3 submission

`submitAuditV3`:

1. requires V3 initialized, verifier/signer configured, target has code, deadline valid, and sufficient stake;
2. requires exactly 138 public signals;
3. requires signals 128–137 equal the supplied ten class-score field elements;
4. hashes proof, public signals, scores, and target runtime bytecode;
5. computes the exact EIP-712 request digest including agent, target, round, teacher/proxy/DATA/schema identities, deadline, chain ID, and registry address;
6. rejects reused request digest;
7. verifies the signature recovers the configured policy signer;
8. verifies the proxy proof;
9. stores the complete V3 provenance/context record and emits `AuditSubmittedV3`.

### Queries

V1, V2, and V3 each retain explicit `has/getLatest/getHistory/getCount` methods. The live audit MCP wraps these as protocol-neutral version-aware reads.

### Trust-root controls

Owner-only controls include pause/unpause, V3 verifier rotation, V3 policy-signer rotation, and UUPS upgrade authorization. Those are centralized governance/security boundaries and must be treated accordingly.

## Interfaces, data shapes, and configuration

V3 context includes:

```text
contractAddress
roundId
teacherModelHash
proxyBundleHash
dataVersionHash
classSchemaHash
deadline
```

The signed digest additionally binds:

```text
agent
contractCodeHash
chainId
registryAddress
proofHash
publicSignalsHash
classScoreFeltsHash
```

Fixed proof layout remains:

- `NUM_CLASSES = 10`
- `INPUT_OFFSET = 128`
- total V2/V3 public signals = 138.

The contract stores `keccak256(proof)` as `proofHash`.

## Failure modes and current limitations

- V3 proof verification still proves only the retained proxy circuit.
- Policy signature authenticates context/provenance but does not prove teacher/source/AGENTS execution.
- A production policy-signing/broadcast service is not claimed by the current analysis runtime.
- Owner compromise can affect pause, signer/verifier rotation, and upgrades.
- UUPS storage compatibility remains a permanent constraint.
- `check_mode="UNSAFE"` in the retained EZKL settings is an external proof-assurance limitation despite contract verification passing.
- V1/V2 records remain readable and must not be mistaken for V3-bound provenance.

## Common change recipe

For V3 verifier/signer/implementation changes:

1. preserve storage layout and historical reads;
2. test fresh deployment and V2→V3 upgrade path;
3. verify `initializeV3` disables new legacy writes;
4. test digest parity with off-chain `policy_signer.py`;
5. test replay, expiry, code-hash, score/signal, signature, proof, stake, pause, and rotation failures;
6. bind deployment/verifier/signer identities without committing secrets;
7. verify read-only audit MCP returns V3 provenance correctly;
8. update security/status docs before operational claims change.

## Verification commands

```bash
cd contracts
forge build
forge test
```

Focused V3 suites include registry V3 behavior, golden digest parity, upgrade/storage compatibility, and real-proof verifier tests. Live broadcast/signing is a separate integration concern.

## Optional deep references

- [`AuditRegistry.sol`](../../contracts/src/AuditRegistry.sol)
- [`policy_signer.py`](../../agents/src/security/policy_signer.py)
- [ZKML](07_zkml.md)
- [Runtime flows](02_runtime_flows.md)
- [Security and trust](12_security_and_trust.md)

## Technical mastery layer

### Prerequisite knowledge

Know ERC-20 staking, UUPS storage, EIP-712, ECDSA recovery, replay protection, bytecode hashes, ZK verifier interfaces, and event/query versioning.

### Source map and reading order

Read V1/V2 storage first for compatibility, then V3 appended storage, `initializeV3`, digest construction, `submitAuditV3`, V3 queries, rotation controls, upgrade tests, and off-chain digest builder in `policy_signer.py`.

### Execution trace and worked example

A V3 request for a deployed target computes the runtime `codehash`, binds it with proof/signal/score hashes and model/data identities, receives an authorized policy signature, then reaches `submitAuditV3`. The registry rejects replay/expiry/mismatch before appending a V3 result. Later the read-only audit MCP can return that record together with V1/V2 history.

### Implementation practice

Any V3 field change is a protocol migration affecting EIP-712 typehash/digest, off-chain request builder, tests, contract ABI/storage/event interpretation, and all submitting/querying components. Do not mutate it as a local helper change.

### Review and ownership check

Can you explain which legacy operations are historical, why V3 disables new legacy writes, every field protected by the V3 digest, and the separate responsibilities of policy signer versus ZK verifier?
