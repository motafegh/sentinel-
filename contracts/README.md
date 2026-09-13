# SENTINEL Contracts Module

`contracts/` is the on-chain trust/persistence layer: staking token, upgradeable audit registry, verifier interface/generated verifier, V1/V2 historical storage compatibility, and the current V3 context-attested submission protocol.

> **Current protocol:** V3. V1/V2 history remains readable for compatibility. After V3 initialization, new V1/V2 writes are disabled.

## Protocol versions

- **V1** — historical scalar/single-score audit format.
- **V2** — historical ten-score / 138-public-signal proxy-proof format without full audit-context binding.
- **V3** — current protocol: retained proxy proof + EIP-712 policy/context attestation.

V3 does **not** make the neural circuit prove more than V2 did. It adds authenticated context around the retained proof.

## V3 trust model

A V3 request binds:

- submitting agent;
- target contract and runtime `codehash`;
- chain ID and registry proxy address;
- audit round;
- teacher-model hash;
- proxy-bundle hash;
- DATA-version hash;
- class-schema hash;
- proof hash;
- hash of all 138 public signals;
- hash of all ten score field elements;
- expiry deadline.

`submitAuditV3` verifies:

1. V3 initialized / legacy writes disabled;
2. target has runtime code;
3. deadline valid;
4. agent satisfies stake requirement;
5. exactly 138 public signals;
6. outputs 128–137 equal the supplied ten scores;
7. request digest has not been used;
8. EIP-712 signature recovers the configured policy signer;
9. configured V3 verifier accepts the proxy proof.

The stored V3 record preserves the context/provenance identities and signed digest.

## Upgrade / compatibility rules

`AuditRegistry.sol` preserves historical V1/V2 storage order and appends V3 state. `initializeV3(verifier, policySigner)` is the activation boundary and sets `legacySubmissionsDisabled = true`.

Permanent rules:

- never reorder historical UUPS storage;
- preserve V1/V2 read/history APIs;
- reject new legacy writes after V3 activation;
- keep the V3 EIP-712 request schema synchronized with `agents/src/security/policy_signer.py`;
- treat signer/verifier rotation and UUPS upgrades as privileged governance operations;
- do not place runtime private-key handling in the analysis MCP.

## Proof boundary

The retained verifier checks the 128→10 proxy circuit:

```text
public signals 0..127   = fusion inputs
public signals 128..137 = ten proxy output field elements
```

A valid proof establishes the proxy computation only. The V3 policy signature authenticates surrounding context/provenance separately. Neither proves the full teacher, Solidity source analysis, LangGraph execution, or final AGENTS verdict.

The retained EZKL bundle still uses `check_mode="UNSAFE"`, which remains a production-assurance limitation outside Solidity verification correctness.

## Main files

```text
contracts/src/SentinelToken.sol       staking/token
contracts/src/IZKMLVerifier.sol       verifier interface
contracts/src/ZKMLVerifier.sol        generated retained verifier
contracts/src/AuditRegistry.sol       V1/V2/V3 UUPS registry
contracts/script/Deploy.s.sol         fresh deployment
contracts/script/UpgradeRegistryV3.s.sol  V3 upgrade path
contracts/test/                       V1/V2/V3/storage/digest/proof tests
```

Important tracked tests include:

- `AuditRegistry.t.sol`
- `AuditRegistryV2.t.sol`
- `AuditRegistryV3.t.sol`
- `AuditRegistryV3GoldenDigest.t.sol`
- `AuditRegistryV3Upgrade.t.sol`
- `ZKMLVerifierRealProof.t.sol`
- `SentinelToken.t.sol`
- invariant coverage

## Build / test

```bash
cd contracts
./scripts/bootstrap_deps.sh
forge build
forge test
```

Deployment/broadcast requires explicit local/network credentials and trust-root configuration. Never commit private keys or copy deployment-key handling into runtime audit processes.

## Runtime boundary

The live audit MCP is read-only. `agents/src/security/policy_signer.py` builds/validates V3 request/digest semantics but intentionally contains no key, signing, transaction construction, broadcast, or receipt handling.

A production V3 signer/broadcaster would be a separate security domain with explicit KMS/HSM/key custody, transaction/retry/finality/reorg handling, and deployment evidence. It is **not** claimed today.

## R4 relationship

Current V3/proxy artifacts remain tied to the historical teacher/proxy lineage. Historical R4 G0–G7 remain PASSED; **Phase 8 is IN_PROGRESS and no repaired teacher has been trained/promoted**.

The current DATA/ML physical path has advanced through R4-D-011, which accepts the exact V10 V2.6 representation lineage, and R4-D-012, which authorizes guarded selection only for a fresh successor candidate that still requires separate physical acceptance. Full training remains unauthorized; confirmed negatives remain zero; threshold/calibration/untouched-acceptance support remains unavailable.

Therefore no new proxy/circuit/verifier bundle should be regenerated or bound into V3 merely because the DATA/representation lineage advanced. A repaired teacher must first be explicitly trained, evaluated, selected, and promoted under later authority. Only then should proxy redistillation, agreement measurement, circuit/verifier regeneration, V3 identity binding, and integration/deployment validation occur.

For current detail, see [contracts handbook](../docs/handbook/08_contracts.md), [ZKML](../docs/handbook/07_zkml.md), [runtime flows](../docs/handbook/02_runtime_flows.md), and [current status](../docs/handbook/16_current_status.md).
