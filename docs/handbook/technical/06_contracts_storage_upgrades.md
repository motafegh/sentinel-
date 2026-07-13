# T06 — Contracts, verifier, storage, deployment, and upgrades

## Learning outcome

You can explain every registry guard and storage record, trace a V1/V2 submission, and add an invariant or upgrade safely without corrupting UUPS storage.

## Prerequisites

Read [Contracts](../08_contracts.md). Know Solidity mappings/events, ERC-20 approvals, proxy initialization, `delegatecall`, and Foundry tests.

## Source map and reading order

1. `contracts/src/SentinelToken.sol::SentinelToken` — staking accounting and owner slash.
2. `contracts/src/IZKMLVerifier.sol::IZKMLVerifier` and generated `ZKMLVerifier.sol`.
3. `contracts/src/AuditRegistry.sol::{initialize,submitAudit,submitAuditV2}`.
4. Registry query methods, pause controls, and `::_authorizeUpgrade`.
5. `contracts/script/` deployment/upgrade scripts.
6. Tracked `contracts/test/{SentinelToken,AuditRegistry,SentinelTest}.t.sol`; treat ignored `AuditRegistryV2.t.sol` as local-only.

## Entry point and complete call chain

Deployment creates token and verifier, deploys registry implementation behind an ERC1967 proxy, and calls `initialize` once with dependency addresses and owner. An agent acquires/approves/stakes token, then submits proof bytes and public instances. V1 checks stake, verifier result, and one score. V2 checks the same trust gates and all ten output positions beginning at offset 128, stores `AuditResultV2`, hashes proof bytes with `keccak256`, and emits `AuditSubmittedV2`. Query methods read append-only per-contract histories.

## Important symbols and configuration

- `MIN_STAKE` is enforced through token staking state.
- `NUM_CLASSES=10` and `INPUT_OFFSET=128` bind registry decoding to the circuit layout.
- V1 and V2 have separate mappings and query families; compatibility is intentional.
- UUPS implementation storage executes through proxy storage. Existing variable order/types must not move.
- Pause blocks submissions, not historical reads. Upgrade authorization is owner-only.

## Annotated source excerpt

Source: `contracts/src/AuditRegistry.sol::AuditResultV2`

```solidity
struct AuditResultV2 {
    uint256[10] classScores;
    bytes32 proofHash;
    bytes32 modelHash;
    uint256 timestamp;
    address agent;
    bool verified;
}
```

This record commits to submitted class fields, proof hash, asserted teacher model hash, time, and caller. It does not store proof bytes or authenticate how the model hash was produced.

## Worked example

For contract `C`, agent `A` stakes at least the minimum and calls V2 with ten felts and 138 instances. The verifier returns true. Registry compares `classScores[i]` with instance `128+i`, appends one record to `_auditsV2[C]`, and emits an event. `getAuditCountV2(C)` increments while V1 history remains unchanged.

## Success trace

Proxy is initialized once; dependency addresses are nonzero/expected; stake is sufficient; proof verifies; every class field matches; storage appends; event arguments and query result agree; upgrade tests preserve old state.

## Failure trace

Insufficient stake, invalid proof, one mismatched score, or pause reverts. Directly calling an uninitialized implementation is not a valid deployment. Reordering storage or changing struct meaning can corrupt proxy state. The root ignore rule means the local V2 test is not fresh-clone evidence.

## Design reasoning and rejected alternatives

Proof hashing avoids storing large proof bytes. Append-only histories preserve audit chronology. The verifier interface allows regenerated implementations without hard-coding generated contract type. V2 was added alongside V1 rather than mutating V1 semantics. UUPS reduces proxy overhead but concentrates upgrade responsibility in storage discipline and owner controls.

## Safe change walkthrough

For a new storage field, append it after existing fields, never reorder or narrow types, add a pre-upgrade state fixture and post-upgrade equality assertions, test unauthorized upgrade rejection, then run storage-layout tooling and deployment dry-run. For verifier replacement, verify circuit/version/public layout first and separate verifier deployment from registry implementation changes.

## Guided lab

Complete [L06 — registry invariant and upgrade test](../labs/06_contract_registry_invariant.md).

## Tests and expected results

```bash
cd contracts && forge test --match-path 'test/AuditRegistry.t.sol' -vv
```

Expected: tracked registry guard, query, pause, and upgrade tests pass. V2 local tests may add evidence in this checkout but must not be reported as fresh-clone coverage.

## Review questions

Which component checks cryptography versus stake versus output equality? Why store a proof hash? What is unsafe about inserting a variable? How do V1 and V2 histories coexist?

## Ownership checklist

- I can enumerate submission guards in order.
- I can map circuit outputs to registry fields.
- I can inspect initialization and proxy ownership.
- I never claim ignored tests as cloned coverage.
