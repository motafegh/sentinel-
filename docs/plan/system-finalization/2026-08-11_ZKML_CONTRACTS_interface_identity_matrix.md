# SENTINEL ZKML ↔ Contracts Interface and Identity Matrix

**Branch:** `system/zkml-contracts-alignment`  
**Date:** 2026-08-11  
**Status:** executable-contract companion to the alignment audit

## 1. Interface chain

| Producer | Output | Shape / encoding | Consumer | Current semantic contract |
|---|---|---|---|---|
| ML `/fusion-embedding` | `fusion_embedding` | 128 finite floats | ZKML student | teacher fusion representation for one contract request |
| ML `/fusion-embedding` | `model_hash` | `0x` + SHA-256(checkpoint bytes) | AGENTS policy/provenance | active teacher checkpoint file identity |
| `ProxyModel.forward()` | student scores | 10 floats | ONNX/EZKL + audit submission | `teacher_probability_regression_v1`; **no second sigmoid** |
| EZKL V2 proof | public inputs | 128 field elements | verifier / provenance | quantized student inputs; not proof of teacher/source execution |
| EZKL V2 proof | public outputs | 10 field elements | registry V2/V3 | quantized direct student scores |
| EZKL V2 proof | proof bytes | opaque bytes | verifier + V3 policy request | proxy-computation proof only (`legacy_proxy_only_unbound`) |
| V3 policy boundary | EIP-712 request digest | bytes32 Keccak | isolated signer + registry | context attestation over exact proof/signals/scores and audit identities |
| `AuditRegistry.submitAuditV3` | accepted record | append-only V3 storage | client/indexer | dual-attested submission: proof verified + policy context signature verified |

## 2. Hash algorithms — names must be explicit

The historical code used the generic term `proof_hash` for different algorithms. That is unsafe for cross-module identity reasoning.

| Identity | Algorithm | Purpose | Canonical name going forward |
|---|---|---|---|
| teacher checkpoint file | SHA-256 bytes | reproducible file identity | `teacher_checkpoint_sha256` |
| proxy checkpoint file | SHA-256 bytes | reproducible file identity | `proxy_checkpoint_sha256` |
| ONNX/settings/VK/etc. | SHA-256 bytes | artifact integrity/manifest identity | `<artifact>_sha256` |
| historical AGENTS proof evidence | SHA-256 proof bytes | off-chain file/evidence identity | `proof_sha256` |
| Solidity proof binding | Keccak-256 proof bytes | EVM/V3 request + stored proof identity | `proof_keccak256` |
| V3 public-signal binding | `keccak256(abi.encode(uint256[]))` | bind all 138 public values | `public_signals_keccak256` |
| V3 ten-score binding | `keccak256(abi.encode(uint256[10]))` | bind submitted score array | `class_score_felts_keccak256` |
| target executable identity | EVM `EXTCODEHASH` / address `.codehash` | bind deployed runtime bytecode | `contract_code_hash` |
| V3 accepted request | EIP-712 Keccak digest | replay/domain/context identity | `request_digest` |

### Rule

A field named only `proof_hash` is ambiguous at module boundaries and must not be introduced in new V3 APIs/manifests. Historical structures may retain it for compatibility, but adapters must label the algorithm when exposing it to new code.

## 3. Proof truth vs policy truth

### EZKL proof truth

The V2 circuit can support the statement:

> the published proxy outputs are consistent with the circuit computation over the published 128 inputs and the verifier's fixed circuit parameters/artifacts.

It does **not**, by itself, support:

- these 128 inputs came from the named teacher checkpoint;
- they came from this target Solidity bytecode;
- the named DATA version was used;
- the caller/model metadata is part of the ZK statement;
- the proof belongs to this chain, registry, agent, or audit round.

### V3 policy truth

The V3 EIP-712 request authenticates the policy service's statement that the exact:

- proof bytes;
- 138 public signals;
- ten score felts;
- target address + runtime bytecode hash;
- agent;
- round;
- teacher model identity;
- proxy bundle identity;
- DATA version identity;
- class-schema identity;
- chain + registry domain;
- expiry deadline

belong to one accepted audit request.

**V3 therefore binds the accepted submission, not the mathematical scope of the neural circuit.**

## 4. Versioned write surfaces

| Surface | Status after V3 activation | Trust meaning |
|---|---|---|
| `submitAudit` V1 | write disabled; history readable | historical scalar protocol |
| `submitAuditV2` | write disabled; history readable | historical 128+10 proxy proof with caller metadata |
| `submitAuditV3` | only current write path | proxy proof + policy-signed context |

`initializeV3()` is intentionally one-way for legacy write containment. Verifier and policy signer can rotate through explicit owner-only V3 setters; re-enabling V1/V2 writes is not exposed.

## 5. Artifact lineage

Future ZKML generation must be a chain of identities, not filenames:

```text
promoted DATA export manifest SHA-256
        +
active teacher checkpoint SHA-256
        ↓
proxy checkpoint metadata
        ↓
ONNX manifest + parity evidence
        +
calibration manifest
        ↓ matching teacher + DATA lineage
EZKL setup manifest
        ↓
settings / compiled circuit / SRS / PK / VK hashes
        ↓
canonical generated Solidity verifier identity
        ↓
V3 proxy_bundle_hash / deployment evidence
```

Current historical V2 artifacts may be inspected by `validate_bundle.py`, but anonymous historical artifacts must not silently become inputs to a newly generated circuit bundle.

## 6. Current remote validation status

As of this document update:

- dependency-light ZKML boundary tests: PASS on GitHub Actions;
- historical V2 artifact-bundle structural validator: PASS;
- V3 Python policy-request tests: PASS;
- Python golden V3 digest generation: PASS;
- Foundry V3 build/tests: in remote validation; final result not yet recorded here.

## 7. Still deliberately local/runtime-bound

Even after remote Foundry tests pass, the following remain separate evidence requirements:

1. canonical generated `ZKMLVerifier.sol` compilation under its intended compiler/config;
2. actual EZKL witness → prove → off-chain verify with the canonical historical/new bundle;
3. actual generated Solidity verifier validating that proof;
4. Anvil service-level ML → proxy → proof → policy signature → registry submission;
5. final `_submit.py` minimal direct-score patch and service integration without regressing R0 transaction-state behavior;
6. post-R4 proxy retraining only after a DATA/ML bundle is promoted.
