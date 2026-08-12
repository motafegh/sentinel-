# 17 — Reference registry

**Read this when:** you need a glossary, source-symbol map, configuration registry, artifact classification, or document-authority lookup.

**Skip this if:** you are following a learning path for the first time; use it as a lookup page.

**Estimated reading time:** 12 minutes to scan.

## 30-second summary

This is the handbook’s lookup layer. Current authority is executable source + committed machine-readable policies/manifests, the canonical handbook, and active R4 decisions. Historical v1/Run12/V1-V2 artifacts remain important for reproduction but are not current authority for new DATA/ML or chain work. Technical guides/labs remain useful supplementary teaching material; some examples were authored against pre-R4/pre-V3 behavior and are subordinate to canonical chapters/current status.

## Just-enough mental model

```text
behavioral truth: source + machine policy/manifests
        ↓
canonical explanation: docs/handbook
        ↓
active DATA/ML governance: docs/plan/ml-R4
        ↓
supplementary guides/labs
        ↓
historical plans/reports/learning context
```

When two layers disagree, move upward in this hierarchy and inspect the exact current source/artifact identity.

## Actual runtime/source walkthrough

### Glossary

| Term | Current meaning |
|---|---|
| DATA v1 | historical binary label/export semantics used by Run12 compatibility |
| DATA vNext / v2 | repaired semantic layer: outcome state, nullable target, strength, masks, provenance, frozen roles |
| evidence ledger | 224,930 contract×class rows reconstructing historical source/target state |
| outcome state | evidence conclusion such as confirmed positive/negative, unknown, conflicting, not reviewed |
| training strength | `STRONG`, `WEAK`, or `NONE`; separate from canonical truth |
| dataset role | leakage-group purpose such as train strong/weak/unlabeled, model selection, internal audit, excluded |
| Run12 | historical operational teacher/checkpoint baseline; not vNext-retrained |
| teacher | full four-eye `SentinelModel` |
| proxy/student | retained frozen 128→64→32→10 ZK-compatible model |
| fusion embedding | 128-value teacher representation consumed by proxy |
| proof scope | computation actually established by the ZK circuit; currently proxy-only |
| V3 | context-attested registry submission protocol with EIP-712 policy signature + proxy proof |
| policy signer | isolated authority for V3 context attestation; current analysis code builds/validates requests but does not hold signing keys |
| audit MCP | live read-only V1/V2/V3 registry observation service on 8012 |
| Rule 5C | failures/skips/unavailable evidence must be explicit, never empty-as-clean |
| provenance | origin/process/context claim; not automatically cryptographic ground truth |
| smoke/module/live | targeted implementation / subsystem / real external-operation evidence tiers |

### Source-symbol map

| Concern | Stable/current reference |
|---|---|
| historical DATA stage registry | `data_module/sentinel_data/cli.py::STAGES` |
| physical graph/class schema | `data_module/sentinel_data/representation/graph_schema.py::FEATURE_SCHEMA_VERSION`, `::CLASS_NAMES` |
| R4 DATA policy | `docs/plan/ml-R4/specs/data_vnext_policy_v1.json` |
| R4 frozen roles | `docs/plan/ml-R4/manifests/p6_partition_manifest.json` |
| historical ML dataset seam | `ml/src/datasets/sentinel_dataset.py::SentinelDataset` |
| teacher | `ml/src/models/sentinel_model.py::SentinelModel` |
| inference API | `ml/src/inference/api.py::app` |
| proxy/circuit | `zkml/src/distillation/proxy_model.py::ProxyModel`, `::CIRCUIT_VERSION` |
| circuit proof | `zkml/src/ezkl/run_proof.py::main` / proof helpers |
| V3 registry | `contracts/src/AuditRegistry.sol::submitAuditV3` |
| V3 digest/request | `agents/src/security/policy_signer.py::compute_v3_digest`, `::build_v3_request` |
| live audit MCP | `agents/src/mcp/servers/audit/_server.py::run_server`, `_readonly_handlers.py` |
| LangGraph | `agents/src/orchestration/graph.py::build_graph` |
| gateway/jobs | `agents/src/api/gateway.py::create_app`, `sqlite_job_store.py::SqliteJobStore` |
| V3 feedback | versioned submission/observation/policy/runtime modules under `agents/src/contracts` and `agents/src/ingestion` |

### Configuration registry

| Area | Location |
|---|---|
| historical DATA acquisition/config | [`data_module/config.yaml`](../../data_module/config.yaml) |
| R4 DATA semantics | [`data_vnext_policy_v1.json`](../plan/ml-R4/specs/data_vnext_policy_v1.json) |
| R4 roles | [`p6_partition_manifest.json`](../plan/ml-R4/manifests/p6_partition_manifest.json) |
| ML training/MLOps | [`ml/scripts`](../../ml/scripts) + [`mlops_config.json`](../../ml/mlops_config.json) |
| AGENTS verdict/routing | [`agents/configs`](../../agents/configs) |
| EZKL circuit | [`zkml/ezkl/settings.json`](../../zkml/ezkl/settings.json) |
| V3 contract trust roots | `AuditRegistry` V3 verifier/policy-signer storage + deployment/upgrade config |
| handbook facts | [`handbook.toml`](_meta/handbook.toml) |

### Environment-variable registry

Names only; inspect source for exact current defaults:

- gateway: port/database/health/audit-limit variables;
- ML: `SENTINEL_CHECKPOINT`, drift/determinism/resource variables;
- MCP: per-service ports/upstream/index/mock/timeouts;
- chain observation: RPC + registry address;
- future isolated V3 signer/broadcaster: must define its own external key/KMS/RPC boundary; no secret value belongs here;
- testing: `TMPDIR`, `TMP`, `TEMP`.

### Artifact matrix

| Classification | Examples | Current meaning |
|---|---|---|
| tracked current governance | R4 ledger, policy, schema, Phase-6 role/support/acceptance manifests | fresh-clone semantic authority through G6 |
| tracked source/tests | DATA/ML/AGENTS/ZK/contracts source, V3 tests | implementation/verification source |
| tracked historical/reproducibility | proxy/ONNX/settings/compiled/VK/generated verifier, historical reports | present but may require regeneration after repaired teacher |
| historical local/protected | Run12 checkpoint/companions, large physical representations depending on checkout | required for specific reproduction/local G7; not guaranteed clone assets |
| generated/local runtime | RAG indexes, DBs, caches, witness/proof workspaces | recreated by operation/build |
| private operational | secrets/signing keys/RPC credentials/proving material where applicable | never Git/documented values |

The old claim that `contracts/test/AuditRegistryV2.t.sol` is merely ignored/local is obsolete. Current tracked contract tests include V2 compatibility plus V3 behavior, golden-digest parity, V3 upgrade/storage, and real-proof verification.

## Interfaces, data shapes, and configuration

The machine-readable handbook registry owns canonical page names, required sections, ports/routes/tools, critical dimensions, selected source ownership, artifacts, and test tiers. It must mirror the **live** service surface: audit MCP has three read-only tools, while V3 submission exists at the contract/policy boundary rather than as an analysis-MCP tool.

R4 machine artifacts are stronger sources for DATA semantic state than prose summaries. Content hashes identify evidence/partition/data artifacts independently of mutable paths.

## Failure modes and current limitations

- A registered source path can exist while prose semantics are stale; validator truth checks therefore need semantic phrases as well as symbol existence.
- A tracked historical artifact can be reproducible without being production/current authority.
- Same tensor shape/version name can hide changed model/data meaning.
- Supplementary guides/labs can teach mechanics while containing old examples; they must not override canonical chapters.
- Local files can make one checkout appear more complete than a fresh clone.

## Common change recipe

When adding/changing a reference:

1. classify it as current canonical, active governance, supplementary, historical/reproducibility, generated/local, or private;
2. register stable source/policy/artifact identity;
3. update the owning canonical chapter/current status;
4. update validator discovery if the **live** entry point changed;
5. keep historical material but label its authority honestly;
6. run static/inventory checks and PR CI.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
git ls-files
```

## Optional deep references

### Learning-material classification

| Material | Classification | How to use it now |
|---|---|---|
| `docs/handbook/*.md` canonical chapters | current | primary explanatory/navigation authority |
| `docs/plan/ml-R4/` | active controlling DATA/ML governance/evidence | authoritative for R4 phase/semantic/role state |
| `docs/handbook/technical/` | **supplementary learning guides** | use for source-reading mechanics; verify pre-R4/pre-V3 examples against canonical chapters/source |
| `docs/handbook/labs/` | **supplementary practice labs** | safe exercises where prerequisites still match; not current-state authority |
| `docs/learning/` | historical/useful conceptual context | verify operational/current claims before reuse |
| prior plans/reports/experiments | historical evidence/context | preserve for auditability, not current implementation instructions |

### Canonical deep references

- R4 evidence/policy/role artifacts under [`docs/plan/ml-R4`](../plan/ml-R4)
- current ADRs/decision registers bound to current implementation
- module tests/evidence tied to explicit commits/artifact hashes

### Historical compatibility roots

- historical v1 DATA exports/splits/labels;
- Run12 checkpoint/threshold/calibration companions;
- V1/V2 registry history/ABI paths;
- retained proxy proof bundle until replaced after repaired teacher selection.

Historical compatibility is deliberate; it must not be silently presented as the current path.

## Technical mastery layer

### Prerequisite knowledge

Know repository navigation, semantic versioning, artifact provenance, partial-label semantics, EIP-712, source-symbol lookup, and current-vs-historical authority.

### Source map and reading order

Resolve a term here, jump to the owning canonical chapter, then executable source/R4 machine artifact. Use supplementary guides/labs only after current status is clear.

### Execution trace and worked example

For an on-chain audit today, the registry lookup leads to V3 context storage and read-only MCP observation; V2 remains historical. For a DATA target, lookup leads to vNext policy/role artifacts; historical `y[10]` remains Run12 compatibility. For a future retrain, those identities must propagate into the new checkpoint/proxy/V3 hashes.

### Implementation practice

When introducing a public symbol/config/artifact/error, add its authority/classification, owner, consumers, version/hash, failure behavior, verification path, and historical compatibility relationship.

### Review and ownership check

Can you locate the current DATA policy, frozen roles, current teacher status, live audit-MCP tool surface, V3 submission method, proof scope, and candidate Phase-7 status without relying on a historical guide?
