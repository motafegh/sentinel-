# SENTINEL Contracts

`contracts/` is the on-chain trust and persistence layer for SENTINEL. It is a
Foundry project containing the staking token, upgradeable audit registry, ZK
verifier interface, and the canonical EZKL-generated Halo2 verifier.

The current design must be read in protocol versions:

- **V1** — historical single-score / 65-signal audit format;
- **V2** — historical 10-class / 138-signal proxy-proof format;
- **V3** — current context-attested submission protocol on the alignment branch.

V1/V2 storage and read APIs are preserved for upgrade compatibility. Once V3 is
initialized, new V1/V2 writes are disabled.

## Components

### `SentinelToken.sol`

ERC-20 plus staking collateral used by `AuditRegistry`.

The registry requires the submitting agent to satisfy `MIN_STAKE` before an
audit can be accepted. Slashing/staking policy remains separate from ZK proof
validity.

### `IZKMLVerifier.sol`

Minimal verifier boundary:

```solidity
verifyProof(bytes proof, uint256[] instances) returns (bool)
```

`AuditRegistry` depends on the interface rather than generated verifier source.

### `ZKMLVerifier.sol`

Canonical tracked EZKL/Halo2 verifier for the historical V2 bundle. The
repository intentionally keeps one generated verifier authority under
`contracts/src/`; stale duplicate generated copies are not accepted.

Remote system-alignment CI compiles this verifier and exercises the tracked
canonical EZKL proof against it.

### `AuditRegistry.sol`

UUPS-upgradeable audit registry.

Historical storage order is preserved. V3 state is appended rather than
reordering V1/V2 storage.

## Historical V2 proof statement

V2 uses exactly 138 public signals:

```text
0..127    proxy input: 128 fusion values
128..137  proxy output: 10 class-score field elements
```

`submitAuditV2` verifies the ZK proof and checks every supplied class score
against public signals 128..137.

That proof does **not** bind the target contract, chain, registry, audit round,
or teacher-model identity. Historical V2 also accepted caller-provided
`modelHash`. Those limitations are retained as explicit regression evidence;
V2 is not the current trusted finality protocol.

## V3 protocol

V3 uses dual attestation rather than overstating what the neural proof proves:

1. the configured V3 ZK verifier validates the proxy computation/proof;
2. a dedicated policy signer authenticates the exact audit context using EIP-712.

The V3 typed request binds:

- submitting agent;
- target contract address;
- target runtime `codehash`;
- round ID;
- teacher checkpoint hash;
- proxy/circuit bundle hash;
- DATA version hash;
- class-schema hash;
- `keccak256(proof)`;
- hash of all public signals;
- hash of all ten class-score field elements;
- expiry deadline;
- EIP-712 domain containing the current chain ID and registry proxy address.

`submitAuditV3` additionally requires:

- V3 initialization completed;
- legacy V1/V2 writes disabled;
- target address has deployed runtime bytecode;
- signature is not expired;
- submitting agent satisfies minimum stake;
- exactly 138 public signals;
- all ten output signals match the supplied class-score felts;
- request digest has not already been used;
- EIP-712 signature recovers the configured policy signer;
- configured V3 verifier accepts the proof.

The accepted V3 record stores the context identities and signed request digest.
The ZK proof remains a proxy-computation proof; the policy signature is the
separate authenticated context assertion.

## Replay and substitution resistance

The request digest is one-time-use and is domain-separated by chain + registry.
Changing the target, target bytecode, agent, round, model/data/schema identity,
proof, signals, scores, registry, chain, or deadline changes the signed request.

The V3 Foundry suite covers replay and substitution behavior as well as upgrade
and storage-preservation paths.

## Legacy-write containment

`initializeV3(verifier, policySigner)` is a one-way protocol activation boundary.
It appends/configures V3 roots and sets `legacySubmissionsDisabled = true`.
After activation, V1/V2 query/history functions remain readable, but their write
entry points revert.

Fresh deployments on the alignment branch activate V3 before the deployment
script exits. A fresh deployment is considered misconfigured if the V3 verifier
or policy signer is zero or if legacy writes remain enabled.

## Build and dependencies

The project uses the compiler configured in `contracts/foundry.toml` (currently
Solidity 0.8.22 for the validated branch build).

A root clone does not automatically populate the Foundry libraries from
`contracts/.gitmodules`. Use the deterministic bootstrap script, which checks
out the revisions recorded in `contracts/foundry.lock`:

```bash
cd contracts
./scripts/bootstrap_deps.sh
forge build
forge test
```

The system-alignment GitHub Actions workflow uses the same bootstrap path.

## Tests

Important tracked suites include:

```text
AuditRegistry.t.sol                 historical V1 behavior
AuditRegistryV2.t.sol               V2 multi-class + limitation evidence
AuditRegistryV3.t.sol               V3 context/replay/substitution behavior
AuditRegistryV3GoldenDigest.t.sol   Python ↔ Solidity EIP-712 digest parity
AuditRegistryV3Upgrade.t.sol        pre-V3 storage/history upgrade preservation
ZKMLVerifierRealProof.t.sol         actual generated verifier + tracked proof
SentinelToken.t.sol                 staking/token behavior
InvariantAuditRegistry.t.sol        registry invariants
```

The generated-verifier proof test treats either `false` or a revert as a valid
fail-closed rejection of a mutated proof/public-signal statement. A successful
`true` result for mutated data is the forbidden outcome.

## Fresh V3 deployment

`contracts/script/Deploy.s.sol` is a privileged deployment workflow. It is not a
runtime audit-submission path.

Required environment identities include:

```text
DEPLOYER_PRIVATE_KEY
ZKML_VERIFIER_V3
AUDIT_POLICY_SIGNER_V3
```

The script:

1. deploys `SentinelToken`;
2. deploys the `AuditRegistry` implementation + ERC1967 proxy;
3. initializes the base registry;
4. immediately calls `initializeV3`;
5. verifies V3 roots, owner, token, and legacy-write disablement before exiting.

Network/RPC broadcast flags are supplied through the normal Foundry invocation.
Do not copy deployment private-key handling into runtime audit processes.

## Upgrading an existing registry

`contracts/script/UpgradeRegistryV3.s.sol` performs a UUPS upgrade with
`initializeV3(...)` in the same `upgradeToAndCall` transaction. It verifies that
the broadcaster is the current owner and checks the configured V3 roots after
the upgrade.

Before a real network upgrade, validate the candidate implementation, generated
verifier identity, policy signer, storage layout, and deployment bytecode on the
intended chain.

## Runtime signing boundary

The analysis/MCP process does not own a raw transaction-signing key. Runtime V3
requests are prepared/validated at the policy boundary in
`agents/src/security/policy_signer.py`; isolated signing/broadcast is a separate
security domain.

Legacy helpers must not generate direct `cast send --private-key ...` audit
transactions. `zkml/src/ezkl/extract_calldata.py` is read-only by design.

## Important invariants

- UUPS historical storage order must not be reordered.
- V3 state is append-only relative to historical storage.
- V2 public-signal count is exactly 138.
- Ten class-score felts equal public outputs 128..137.
- V3 request digest is EIP-712 domain-separated by chain + registry.
- Target runtime bytecode identity is signed.
- Request digest replay is rejected.
- V1/V2 writes remain disabled after V3 activation.
- Generated verifier identity must match the exact ZK artifact bundle being deployed.
- Deployment-key handling is not a runtime audit-signing design.

## Current status

The alignment branch has remote green validation for:

- dependency-light ZKML boundary tests;
- V3 policy/EIP-712 protocol tests;
- deterministic Foundry dependency bootstrap;
- registry build and Foundry test suite;
- canonical generated Halo2 verifier exercised with the tracked EZKL proof.

This does not authorize deploying a newly retrained ZK bundle. R4 DATA/ML work
must first promote the intended model/data lineage; any new proxy/circuit/VK/
verifier bundle then requires fresh identity, proof, integration, and deployment
validation.
