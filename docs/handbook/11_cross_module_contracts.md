# 11 — Cross-module contracts

**Read this when:** you need to know what must stay compatible across DATA, ML, ZKML, Contracts, and AGENTS.

**Skip this if:** you are reading a subsystem only for orientation; return here before changing it.

**Estimated reading time:** 14 minutes.

## 30-second summary

The most dangerous SENTINEL changes are semantic changes that cross directories while tensor lengths still look valid. Current critical contracts are: v9 graph/token representation, locked ten-class order, DATA vNext outcome/training/role semantics, current Run12 versus future repaired-checkpoint identity, fusion[128], retained 138-signal proxy-proof layout, V3 EIP-712 context binding, read-only audit-MCP behavior, and versioned report/feedback provenance.

## Just-enough mental model

| Producer | Current contract | Consumers |
|---|---|---|
| DATA/R4 | v9 physical representation + vNext semantic state + frozen role | future ML retrain/evaluation |
| historical DATA v1 | binary labels / old split/export identity | Run12 reproducibility only |
| ML | ten scores/eyes, fusion[128], checkpoint hash | AGENTS, proxy/ZKML |
| ZKML | proxy proof over 128 inputs + 10 outputs | V3 registry protocol |
| V3 policy layer | fully bound EIP-712 audit context | `AuditRegistry.submitAuditV3` |
| Contracts | V1/V2 history + V3 context-bound records | read-only audit MCP/operations |
| AGENTS | evidence/verdict/report + versioned observation | gateway/evaluation/feedback |

Compatibility means **meaning + ordering + version + provenance + failure semantics**, not merely matching array length.

## Actual runtime/source walkthrough

### DATA → ML

The physical representation contract remains v9 graph `x[N,12]` plus `[4,512]` token windows. The semantic contract for new training is no longer historical `y[10]` binary truth. `data-vnext-policy-v1` defines per contract×class outcome state, nullable target, training strength, mask eligibility, and provenance; `r4-vnext-roles-v1` assigns the leakage-safe role.

Historical v1 `SentinelDataset`/collate remain compatibility consumers for Run12 and must not silently consume vNext as old binary labels.

### ML → AGENTS / ZKML

Current inference still serves the historical Run12 teacher. API output and fusion width remain compatible:

- ten independent probabilities/tiers/eye outputs;
- checkpoint identity;
- fusion embedding exactly `[128]`.

A future DATA-vNext retrain can keep these shapes while changing checkpoint meaning. Shape compatibility therefore does not authorize reuse of Run12 thresholds, calibration, drift baseline, proxy agreement, or V3 model/data hashes.

### ZKML → V3 registry

The retained proxy produces ten outputs from 128 public inputs; public signals remain 138 total and outputs start at offset 128. That is only the proxy-computation statement.

V3 separately binds:

- target address/runtime bytecode hash;
- submitting agent;
- chain ID / registry address;
- round/deadline;
- teacher-model hash;
- proxy-bundle hash;
- DATA-version hash;
- class-schema hash;
- proof/public-signal/ten-score hashes.

The configured policy signer attests the EIP-712 digest; the contract independently verifies the proxy proof.

### Contracts → audit MCP

V1/V2/V3 storage remains readable. The live audit MCP provides protocol-neutral read operations and returns protocol/version provenance. It is not the transaction-signing or broadcast boundary.

### AGENTS → feedback

V3 observation and feedback policy are versioned. Current V3 promotion policy is intentionally unavailable, so V3 observations may be durable/pending but do not automatically mutate RAG/training data.

## Interfaces, data shapes, and configuration

### Compatibility registry

| Invariant | Current value/status | Primary authority |
|---|---|---|
| graph schema | v9 | DATA graph schema source |
| node/edge dimensions | 12 / 14 node types / 12 edge types | DATA graph schema |
| class order/count | locked list / 10 | DATA graph schema + R4 policy |
| token windows | 4 × 512 | ML tokenizer |
| repaired label semantics | `data-vnext-policy-v1` | R4 policy/spec/ADRs |
| leakage-safe roles | `r4-vnext-roles-v1` | R4 Phase-6 manifests |
| Run12 | historical operational teacher | checkpoint/config lineage |
| fusion width | 128 | teacher model/API |
| proxy | 128→64→32→10, 10,666 params | ZKML source/artifacts |
| proof signals | 128 inputs + 10 outputs = 138 | EZKL settings / registry |
| V3 protocol | context-attested EIP-712 + proxy proof | policy signer + AuditRegistry |
| audit MCP | three read-only version-aware tools | live audit server/handlers |
| V3 feedback policy | unavailable / no automatic promotion | V3 feedback policy/runtime |

## Failure modes and current limitations

- Reordered class semantics can corrupt training while all tensors still have length 10.
- Filling vNext unknowns with zero recreates the historical label corruption.
- Same model architecture does not mean same checkpoint semantics.
- Same fusion width does not mean old proxy agreement remains valid after retraining.
- A valid V3 signature does not prove the teacher; a valid proof does not authenticate the full audit context by itself.
- V1/V2 records do not carry V3 context and must remain version-distinguishable.
- Audit MCP read availability does not imply write authority.
- threshold/calibration/untouched-acceptance roles are unsupported for the first repaired baseline; downstream code must preserve that limitation.

## Common change recipe

Before changing any cross-module invariant:

1. identify semantic owner and all versioned artifacts;
2. classify old behavior as compatibility/history versus current path;
3. add mismatch tests before changing producer output;
4. regenerate dependent artifacts in order rather than patching consumers to accept ambiguity;
5. bind hashes/versions at every boundary;
6. update R4/ADR decisions where DATA/ML semantics change;
7. update V3 digest/protocol tests if any context identity changes;
8. update canonical handbook/current status together.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
cd contracts && forge test
```

## Optional deep references

- [DATA artifacts](04_data_artifacts.md)
- [ML inference](05_ml_model_inference.md)
- [ZKML](07_zkml.md)
- [Contracts](08_contracts.md)
- [Security and trust](12_security_and_trust.md)

## Technical mastery layer

### Prerequisite knowledge

Know producer/consumer compatibility, semantic versioning, tensor/JSON/ABI schemas, masks/roles, hashes, EIP-712, and artifact rollout.

### Source map and reading order

Follow the current seam in this order: R4 policy/role artifacts → physical DATA representation → historical ML consumer/current model API → proxy/settings → policy signer → `AuditRegistry.submitAuditV3` → read-only audit MCP → feedback observation.

### Execution trace and worked example

A contract’s graph/token bytes can remain v9-identical while its historical binary label is replaced by explicit vNext state/masks. A future retrained teacher can still output ten logits/fusion[128], yet needs new checkpoint/DATA/proxy identities. A V3 request then binds those identities with proof/context before registry storage.

### Implementation practice

For every cross-module change, write a compatibility record containing old/new meaning, producer, consumers, artifacts/hashes, migration, mixed-version behavior, rollout, rollback, and tests. Never use “same shape” as the compatibility argument.

### Review and ownership check

Can you trace one class from source evidence through vNext state/role, future training target, teacher output, proxy output, V3 score/hash binding, registry record, and read-only observation without losing its version/provenance meaning?
