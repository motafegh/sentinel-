# 11 — Cross-module contracts

**Read this when:** you need to know what meaning, identity, shape, role, or trust assumptions must stay compatible across DATA, ML, AGENTS, ZKML, and Contracts.

**Skip this if:** you are only orienting inside one subsystem; return here before changing a producer/consumer seam.

**Estimated reading time:** 14 minutes.

## 30-second summary

The most dangerous SENTINEL changes are cross-module semantic changes that still satisfy old shapes. Current compatibility must therefore distinguish **historical runtime/reproduction contracts** from the **current R4 future-training contracts**.

Today, the live ML runtime still serves historical Run12 against its historical v9-compatible inference seam. Separately, a possible repaired teacher must use `data-vnext-policy-v1`, accepted logical V3 grouping/roles, the exact D-011 V10 V2.6 physical graph lineage, and a separately accepted D-012 guarded-selector successor token lineage. Separately again, fusion `[128]`, the retained 138-signal proxy proof, V3 EIP-712 context binding, read-only registry observation, and versioned feedback each own distinct trust claims.

Compatibility means **meaning + ordering + version + provenance + role/authorization + failure semantics**, not merely array length.

## Just-enough mental model

| Producer | Current contract | Consumers |
|---|---|---|
| historical DATA / Run12 seam | historical v9-compatible graph/token inputs + old model/checkpoint semantics | current Run12 inference/runtime continuity |
| R4 semantic layer | `data-vnext-policy-v1` + accepted logical V3 grouping/roles | repaired dataset/training/evaluation code |
| R4 physical representation | exact D-011 V10 V2.6 graph/control-token identity | future repaired candidate construction |
| D-012 selector | fresh guarded-selector token successor, not yet separately accepted | future repaired candidate only |
| current ML runtime | ten Run12 scores/eyes, model identity, fusion `[128]` | AGENTS; retained proxy seam |
| ZKML | proxy proof over 128 inputs + 10 outputs | V3 registry protocol |
| V3 policy layer | fully bound EIP-712 audit context | `AuditRegistry.submitAuditV3` |
| Contracts | V1/V2 history + V3 context-bound records | read-only audit MCP / operations |
| AGENTS | evidence/verdict/report + versioned observation | gateway/evaluation/feedback |

Do not connect the R4 future-training seam directly to current Run12 merely because dimensions remain compatible.

## Actual runtime/source walkthrough

### Historical DATA → current Run12 runtime

The current inference service remains bound to the historical Run12 operational baseline. Its preprocessing/model/checkpoint companions preserve the historical representation/model interface needed for runtime continuity.

That contract is historical compatibility, not the physical/evidence authority for a new repaired full run.

### Current R4 DATA → future repaired ML

The future-training seam now has four distinct owners:

1. **Semantic policy:** `data-vnext-policy-v1` preserves explicit outcome/training state and forbids unknown→negative collapse.
2. **Logical authority:** D-009 accepts `r4-leakage-groups-v3` / `r4-vnext-roles-v3` and the V3 publication/logical lineage.
3. **Physical graph authority:** D-010 withdraws v9 from eligibility for the new full run; D-011 accepts the exact V10 V2.6 physical graph lineage.
4. **Token-selection successor:** D-012 permits `target_aware_guarded_v1` only in a fresh versioned candidate requiring separate physical acceptance.

A repaired ML consumer must therefore bind semantic state, role/group identity, graph lineage, token-selector lineage, class order, and artifact/run identities together. Same tensor dimensions do not permit substituting the historical seam.

### ML → AGENTS

Current inference still serves Run12 outputs:

- ten independent probabilities/tiers/eye signals;
- checkpoint/model identity;
- hotspots/representation evidence;
- fusion embedding exactly `[128]`.

AGENTS treats these as learned evidence, not ground truth. A future repaired checkpoint may preserve response shape while changing its semantic identity completely; all consumers must distinguish model/data/checkpoint lineage.

### ML → ZKML

The retained proxy consumes fusion `[128]` and maps `128→64→32→10`. Shape stability does **not** guarantee that historical proxy agreement transfers to a repaired teacher. If a new selected teacher changes fusion distribution/meaning, proxy distillation/evaluation and bundle identity must be reconsidered before promotion.

### ZKML → V3 registry

The retained proxy proof exposes 128 public inputs + ten public outputs = 138 public signals. That proves only the fixed proxy computation.

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

V1/V2/V3 storage remains readable. The live audit MCP exposes only version-aware read operations and carries protocol/version provenance. It is not the transaction-signing or broadcasting boundary.

### AGENTS → feedback

V3 observation and feedback policy are versioned. Current V3 promotion policy is intentionally unavailable, so observed V3 events may be durable/pending without automatically mutating RAG or DATA/ML truth.

## Interfaces, data shapes, and configuration

### Compatibility registry

| Invariant | Current value/status | Meaning / owner |
|---|---|---|
| class order/count | locked list / 10 | cross-module semantic ordering |
| historical runtime representation | v9-compatible historical seam | Run12 compatibility only |
| current R4 physical graph authority | exact V10 V2.6 / D-011 | accepted possible-future-training graph lineage |
| current guarded token selector | `target_aware_guarded_v1` | D-012; fresh successor still needs separate acceptance |
| token tensor shape | `[4,512]` | historical-control/guarded-window compatibility contract |
| repaired semantic policy | `data-vnext-policy-v1` | current outcome/training authority |
| historical G6 roles | `r4-vnext-roles-v1` | immutable G6/G7 reproduction evidence |
| current logical roles | `r4-vnext-roles-v3` | accepted D-009 logical authority |
| current runtime teacher | Run12 historical operational baseline | live ML API continuity |
| repaired teacher | not promoted | full repaired training unauthorized |
| fusion width | 128 | ML→ZKML seam |
| proxy | 128→64→32→10, 10,666 params | retained ZKML computation |
| proof signals | 128 inputs + 10 outputs = 138 | retained EZKL / registry layout |
| V3 protocol | context-attested EIP-712 + proxy proof | policy signer + AuditRegistry |
| audit MCP | three read-only version-aware tools | live audit server/handlers |
| V3 feedback policy | unavailable / no automatic promotion | feedback policy/runtime |
| threshold fit | `UNSUPPORTED_EMPTY` | no authorized fitting role |
| calibration fit | `UNSUPPORTED_EMPTY` | no authorized fitting role |
| untouched acceptance | `UNSUPPORTED_EMPTY_FROZEN` | no authorized untouched corpus |

### Compatibility principles

1. **Same shape ≠ same meaning.** A ten-output repaired teacher is not Run12 merely because both output ten values.
2. **Same physical population ≠ same representation eligibility.** D-008 repaired DATA survives while v9 becomes ineligible for the new full run under D-010.
3. **Accepted graph ≠ selected token successor.** D-011 and D-012 are separate identities/decisions.
4. **Proof ≠ provenance.** The proxy proof and V3 policy signature establish different claims.
5. **Read access ≠ write authority.** The audit MCP may observe records without signing/submitting them.
6. **Observed chain state ≠ learning truth.** Feedback requires its own policy/authority.

## Failure modes and current limitations

- Reordered class semantics can corrupt consumers while tensors still have length 10.
- Filling repaired unknowns with zero recreates the historical label corruption.
- Using `r4-vnext-roles-v1` as the current logical authority ignores D-009 V3 grouping/roles.
- Using v9 for a new full run violates D-010.
- Treating D-011 as though it already contains D-012 guarded-selector successor bytes collapses two versioned physical identities.
- Same four-eye architecture does not make a future checkpoint semantically interchangeable with Run12.
- Same fusion width does not make old proxy agreement valid for a repaired teacher.
- A valid V3 signature does not prove teacher/source/AGENTS execution; a valid proof does not authenticate the full audit context by itself.
- V1/V2 records lack V3 context and must remain version-distinguishable.
- Audit MCP read availability does not imply write authority.
- threshold/calibration/untouched-acceptance evidence remains unsupported for the repaired path.

## Common change recipe

Before changing any cross-module invariant:

1. identify semantic owner and all producer/consumer versions;
2. classify current behavior as historical runtime compatibility, accepted R4 evidence, pending candidate, or current protocol/runtime;
3. state old/new meaning before touching shapes;
4. add mismatch tests before changing producer output;
5. regenerate dependent artifacts in order rather than patching consumers to accept ambiguity;
6. bind hashes/versions/roles at every semantic boundary;
7. update R4/ADR decisions where DATA/ML semantics or physical acceptance change;
8. update V3 digest/protocol tests if context identity changes;
9. update [Architecture](01_architecture.md), [Security and trust](12_security_and_trust.md), and [Current status](16_current_status.md) together when the accepted seam changes.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
cd contracts && forge test
```

The G6 validator remains historical compatibility evidence; it does not override later D-009/D-011/D-012 authority.

## Optional deep references

- [Architecture](01_architecture.md)
- [DATA artifacts](04_data_artifacts.md)
- [ML model/inference](05_ml_model_inference.md)
- [ML training/quality](06_ml_training_quality.md)
- [ZKML](07_zkml.md)
- [Contracts](08_contracts.md)
- [Security and trust](12_security_and_trust.md)
- [Current status](16_current_status.md)

## Technical mastery layer

### Prerequisite knowledge

Know producer/consumer compatibility, semantic versioning, tensor/JSON/ABI schemas, partial-label masks/roles, artifact hashes, EIP-712, ZK statement scope, and versioned rollout.

### Source map and reading order

Follow the seam in this order: current R4 policy/status → D-009 logical V3 → D-011/D-012 physical decisions → repaired ML dataset/training consumer → current Run12 API seam → fusion/proxy/settings → policy signer → `AuditRegistry.submitAuditV3` → read-only audit MCP → feedback observation.

### Execution trace and worked example

A contract can preserve its historical v1/v9 representation for Run12 reproduction while current R4 semantics assign explicit nullable targets and a logical V3 role. A future repaired candidate can preserve ten output classes/fusion `[128]` while consuming D-011 graphs plus a separately accepted guarded token successor. That new checkpoint then needs a new model/data identity, and any ZKML/V3 integration must bind the new identity rather than inheriting Run12 meaning from shape compatibility.

### Implementation practice

For every cross-module change, record old/new meaning, owner, producer, consumers, artifact identities, mixed-version behavior, rollout/rollback, and mismatch tests. Never use “same shape” or “same file count” as the semantic compatibility argument.

### Review and ownership check

Can you trace one class from source evidence through repaired state, logical V3 role/group, D-011 graph + pending guarded token successor, repaired training target, teacher output, proxy output, V3 score/hash binding, registry record, and read-only observation—while separately identifying the historical Run12 path?