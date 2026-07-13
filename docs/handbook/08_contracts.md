# 08 — Contracts and on-chain registry

**Read this when:** you need staking, V1/V2 audit storage, verifier calls, deployment, upgrades, or query invariants.

**Skip this if:** you only consume off-chain reports and do not operate chain state.

**Estimated reading time:** 13 minutes.

## 30-second summary

`SentinelToken` supplies ERC-20 stake and owner-controlled slashing. Upgradeable `AuditRegistry` requires minimum stake and a valid verifier call before storing proof hashes and scores. V1 stores one score; V2 stores ten scores plus a model hash. The registry verifies score/public-signal consistency, not the full teacher or AGENTS verdict semantics.

## Just-enough mental model

```text
agent stakes SNTL ─┐
proof + 138 signals ├→ AuditRegistry proxy → verifier → append immutable audit record
model hash ─────────┘
owner: pause/unpause, authorize UUPS implementation upgrade, slash stake
```

The proxy address is stable while implementations can change. Storage layout compatibility is therefore a permanent upgrade constraint.

## Actual runtime/source walkthrough

- [`SentinelToken.sol`](../../contracts/src/SentinelToken.sol) — `SentinelToken::MIN_STAKE`, `::stake`, `::unstake`, `::slash`, and `::stakedBalance` implement the economic gate.
- [`AuditRegistry.sol`](../../contracts/src/AuditRegistry.sol) — `AuditRegistry::initialize` binds verifier/token. The constructor disables direct implementation initialization.
- `::submitAudit` is the V1 single-score path and binds its score to a public signal.
- `::submitAuditV2` requires at least 138 signals, verifies the proof, compares all ten output slots beginning at 128, then appends `AuditResultV2`.
- V1 queries are `hasAudit`, `getLatestAudit`, `getAuditHistory`, `getAuditCount`; V2 equivalents carry the `V2` suffix.
- `::pause` and `::unpause` stop/start submissions. `::_authorizeUpgrade` is owner-only and emits `ImplementationUpgraded`.
- [`IZKMLVerifier.sol`](../../contracts/src/IZKMLVerifier.sol) defines `verifyProof`; [`ZKMLVerifier.sol`](../../contracts/src/ZKMLVerifier.sol) is generated from current EZKL artifacts.
- [`Deploy.s.sol`](../../contracts/script/Deploy.s.sol) deploys token, verifier, implementation, and ERC1967/UUPS proxy wiring.

## Interfaces, data shapes, and configuration

`AuditRegistry` constants are `NUM_CLASSES=10` and `INPUT_OFFSET=128`. V2 accepts:

```solidity
submitAuditV2(
    address contractAddress,
    uint256[10] classScores,
    bytes proof,
    uint256[] publicSignals,
    bytes32 modelHash
)
```

Stored `proofHash` is `keccak256(proof)`. Scores are fixed-point field elements; human interpretation divides by 8192 when the score semantics match scale 13. `modelHash` is bytes32 supplied by the caller; the contract does not recompute a teacher checkpoint hash.

Network/RPC addresses and private keys belong in deployment environment/configuration, never in this handbook. Foundry settings live in [`foundry.toml`](../../contracts/foundry.toml).

## Failure modes and current limitations

- Insufficient stake, pause state, verifier rejection, insufficient V2 signals, or score mismatch reverts atomically.
- The verifier validates the proxy circuit only; see [ZKML](07_zkml.md) for the exact claim.
- UUPS upgrades can corrupt state if variable order/types are changed incompatibly.
- Owner control is centralized for pause, upgrade authorization, and slashing in the current MVP.
- V1’s hard-coded public-signal assumption belongs to the historical single-score layout; V2 is the current ten-class path.
- `contracts/test/AuditRegistryV2.t.sol` exists locally but is ignored by the root `test` pattern and is not fresh-clone coverage. The tracked suite status is in [current status](16_current_status.md).

## Common change recipe

For a verifier upgrade:

1. Regenerate and independently verify EZKL artifacts and the Solidity verifier.
2. Add tracked tests for valid proof, tampering, signal length/order, and score mismatch.
3. Deploy the new verifier and decide whether a registry implementation change is needed.
4. If upgrading the registry, run storage-layout comparison and proxy upgrade tests.
5. Update deployment manifests/addresses without documenting secrets.
6. Run local Anvil end-to-end submission before Sepolia.

For any class/order change, update DATA, ML, ZKML, contract constants/methods, tests, and every query consumer as one versioned migration.

## Verification commands

```bash
cd contracts
forge build                                      # smoke
forge test                                       # module
anvil --port 8545                                 # live chain, separate terminal
forge script script/Deploy.s.sol --rpc-url http://127.0.0.1:8545 --broadcast  # live
```

Never paste a private key into a committed command or document. Current counts are only in [current status](16_current_status.md).

## Optional deep references

- [`AuditRegistry.sol`](../../contracts/src/AuditRegistry.sol) — public contract interface
- [`contracts/test`](../../contracts/test) — tracked tests only constitute clone coverage
- [ZKML](07_zkml.md)
- [Operations](14_operations.md)
- [Change playbooks](15_change_playbooks.md)

## Technical mastery layer

### Prerequisite knowledge

Know ERC-20 approval/staking, verifier interfaces, events/mappings, UUPS proxy storage, and Foundry.

### Source map and reading order

Read `SentinelToken`, `IZKMLVerifier`, registry storage/initialize/V1/V2 submit/query/upgrade methods, deployment scripts, then tracked tests. [T06](technical/06_contracts_storage_upgrades.md) provides the guard and storage trace.

### Execution trace and worked example

V2 requires stake, valid verifier result, and ten submitted class fields equal to public instances beginning at 128. It appends a V2 record and event while V1 history remains intact. Proof hash is `keccak256(proof)`; model hash remains asserted provenance.

### Implementation practice

[L06](labs/06_contract_registry_invariant.md) adds a test-only history/upgrade invariant. Storage variables are appended, never reordered; pre/post-upgrade state is asserted.

### Review and ownership check

Can you separate verifier, stake, score, pause, and upgrade authorization failures and state which are cryptographic?
