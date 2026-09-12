# 17 — Reference registry

**Read this when:** you need a glossary, source-symbol map, configuration registry, artifact classification, or document-authority lookup.

**Skip this if:** you are following a learning path for the first time; use it as a lookup page.

**Estimated reading time:** 12 minutes to scan.

## 30-second summary

This is the handbook’s lookup layer. Current authority is executable source/config/tests + committed machine-readable policies/evidence, followed by the canonical handbook and active R4 decisions. Historical v1/G6/G7/Run12/V1-V2 artifacts remain important for reproduction but are not current authority for a new DATA/ML run or V3 trust claims. Technical guides/labs remain useful **supplementary learning guides**; some examples predate R4/V3 and must be checked against current canonical chapters/status.

Current DATA/ML lookup must distinguish `data-vnext-policy-v1`, accepted logical V3 authority under D-009, accepted V10 V2.6 physical representation under D-011, and the still-pending D-012 guarded-token successor. Full repaired training remains unauthorized.

## Just-enough mental model

```text
behavioral truth: source / config / tests
        ↓
machine semantic + acceptance truth: current R4 evidence / ADRs / manifests
        ↓
canonical explanation: docs/handbook
        ↓
supplementary guides/labs
        ↓
historical plans/reports/learning context
```

When two layers disagree, move upward in this hierarchy and inspect the exact current artifact/decision identity.

## Actual runtime/source walkthrough

### Glossary

| Term | Current meaning |
|---|---|
| DATA v1 | historical binary label/export semantics used by Run12 compatibility |
| DATA vNext / v2 | repaired semantic layer introduced at G7: explicit outcome state, nullable target, strength, masks, provenance; still important compatibility/evidence |
| evidence ledger | historical Phase-3 224,930-row contract×class reconstruction; later repaired physical population is larger under D-008 |
| `data-vnext-policy-v1` | current repaired semantic supervision policy |
| logical V3 | accepted D-009 leakage grouping/role/publication authority; address literals diagnostic-only |
| D-011 | exact accepted V10 V2.6 physical representation identity for a possible future repaired run; not training authorization |
| D-012 | guarded-selector decision: `target_aware_guarded_v1` only for a fresh successor requiring separate physical acceptance |
| outcome state | evidence conclusion such as confirmed positive/negative, unknown, conflicting, not reviewed |
| training strength | `STRONG`, `WEAK`, or `NONE`; separate from canonical truth |
| dataset role | leakage-safe purpose such as train strong/weak/unlabeled, model selection, internal audit, excluded |
| Run12 | historical operational teacher/checkpoint baseline; not R4-retrained |
| teacher | full four-eye `SentinelModel` |
| proxy/student | retained frozen 128→64→32→10 ZK-compatible model |
| fusion embedding | 128-value teacher representation consumed by proxy |
| proof scope | computation actually established by the ZK circuit; currently proxy-only |
| V3 registry | context-attested submission protocol with EIP-712 policy signature + proxy proof |
| policy signer | isolated authority for V3 context attestation; current analysis code builds/validates requests but does not hold signing keys |
| audit MCP | live read-only V1/V2/V3 registry observation service on 8012 |
| Rule 5C | failures/skips/unavailable evidence must be explicit, never empty-as-clean |
| provenance | origin/process/context claim; not automatically cryptographic ground truth |
| smoke/module/live | targeted implementation / subsystem / real external-operation evidence tiers |

### Source-symbol / authority map

| Concern | Stable/current reference |
|---|---|
| historical DATA stage registry | `data_module/sentinel_data/cli.py::STAGES` |
| historical graph/class compatibility schema | `data_module/sentinel_data/representation/graph_schema.py::FEATURE_SCHEMA_VERSION`, `::CLASS_NAMES` |
| R4 DATA policy | `docs/plan/ml-R4/specs/data_vnext_policy_v1.json` |
| historical G6 role freeze | `docs/plan/ml-R4/manifests/p6_partition_manifest.json` |
| current logical V3 authority | `docs/plan/ml-R4/adrs/ADR-R4-009-logical-v3-leakage-grouping-correction.md` + accepted V3 evidence/publication records |
| current V10 physical authority | `docs/plan/ml-R4/adrs/ADR-R4-011-v10-v26-physical-representation-acceptance.md` |
| guarded-token successor decision | `docs/plan/ml-R4/adrs/ADR-R4-012-target-aware-guarded-selector-promotion.md` |
| current DATA/ML restart state | `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md` + September 2 run records |
| historical ML dataset seam | `ml/src/datasets/sentinel_dataset.py::SentinelDataset` |
| repaired ML dataset seam | `ml/src/datasets/vnext_dataset.py::VNextTrainingDataset` |
| repaired Phase-8 training mechanics | `ml/src/training/vnext_*` + group sampler / Phase-8 config |
| teacher | `ml/src/models/sentinel_model.py::SentinelModel` |
| inference API | `ml/src/inference/api.py::app` |
| proxy/circuit | `zkml/src/distillation/proxy_model.py::ProxyModel`, `::CIRCUIT_VERSION` |
| circuit proof | `zkml/src/ezkl/run_proof.py::main` / proof helpers |
| V3 registry | `contracts/src/AuditRegistry.sol::submitAuditV3` |
| V3 digest/request | `agents/src/security/policy_signer.py::compute_v3_digest`, `::build_v3_request` |
| live audit MCP | `agents/src/mcp/servers/audit/_server.py::run_server`, `_readonly_handlers.py` |
| LangGraph | `agents/src/orchestration/graph.py::build_graph` |
| gateway/jobs | `agents/src/api/gateway.py::create_app`, `sqlite_job_store.py::SqliteJobStore` |
| V3 feedback | versioned observation/policy/runtime modules under `agents/src/ingestion` |

### Configuration registry

| Area | Location / authority |
|---|---|
| historical DATA acquisition/config | [`data_module/config.yaml`](../../data_module/config.yaml) |
| R4 DATA semantics | [`data_vnext_policy_v1.json`](../plan/ml-R4/specs/data_vnext_policy_v1.json) |
| historical G6 roles | [`p6_partition_manifest.json`](../plan/ml-R4/manifests/p6_partition_manifest.json) |
| current logical/physical R4 state | [Current status](16_current_status.md) + D-009/D-011/D-012 ADR/evidence chain |
| ML historical training/MLOps | [`ml/scripts`](../../ml/scripts) + [`mlops_config.json`](../../ml/mlops_config.json) |
| ML repaired training mechanics | `ml/src/training/vnext_*` and Phase-8 settings/binding code |
| AGENTS verdict/routing | [`agents/configs`](../../agents/configs) + executable routing/verdict source |
| EZKL circuit | [`zkml/ezkl/settings.json`](../../zkml/ezkl/settings.json) |
| V3 contract trust roots | `AuditRegistry` V3 verifier/policy-signer storage + deployment/upgrade config |
| handbook structural/source-validator facts | [`handbook.toml`](_meta/handbook.toml) |

### Handbook metadata scope

`_meta/handbook.toml` currently contains two kinds of information that must not be confused:

1. **live structural/service facts** used by the validator (ports, LangGraph nodes, MCP tool surfaces, proof dimensions, tracked artifacts);
2. **historical G7/source-discoverable compatibility anchors** such as the v9 schema/G6 role manifest/verified G7 commit.

Those historical metadata fields remain intentionally validator-compatible until the P5 documentation-currentness work upgrades the validator to machine-check the later D-009/D-011/D-012 authority chain. They **do not override** [Current status](16_current_status.md), current R4 machine evidence, or the canonical architecture chapters.

This distinction is important because the current accepted future-training physical authority is V10 V2.6 even though the historical `graph_schema.py` / G7 validator metadata still describe v9 compatibility.

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
| tracked current governance | current R4 status, policy, ADRs, V3/D-011/D-012 evidence/manifests/review records | current semantic/acceptance authority at their stated scope |
| tracked historical/reproduction governance | Phase-3 ledger, G6 role/support/acceptance manifests, G7 export/binding records | immutable compatibility/history, not latest logical/physical authority |
| tracked source/tests | DATA/ML/AGENTS/ZK/contracts source, V3 tests | implementation/verification source |
| tracked retained proof artifacts | proxy/ONNX/settings/compiled/VK/generated verifier | present reproduction boundary; may require regeneration after repaired teacher selection |
| historical local/protected | Run12 checkpoint/companions; D-011 heavy physical representation root | required for specific reproduction/current physical review; not guaranteed fresh-clone assets |
| generated/local runtime | RAG indexes, DBs, caches, witness/proof workspaces | recreated by operation/build |
| private operational | secrets/signing keys/RPC credentials/proving material where applicable | never Git/documented values |

Current tracked contract tests include V2 compatibility plus V3 behavior, golden-digest parity, V3 upgrade/storage, and real-proof verification.

## Interfaces, data shapes, and configuration

The machine-readable handbook registry owns canonical page names, required sections, ports/routes/tools, critical runtime/source dimensions, selected source ownership, artifacts, and test tiers. It must mirror the **live service surface**: audit MCP has three read-only tools, while V3 submission exists at the contract/policy boundary rather than as an analysis-MCP tool.

R4 machine artifacts are stronger sources for current DATA semantic/logical/physical state than the G7-era validator baseline or prose summaries. Exact digests/versions identify evidence independently of mutable paths.

## Failure modes and current limitations

- A registered source path can exist while prose semantics are stale; source existence is not semantic currentness.
- The handbook validator still contains historical G6/G7 checks and phrase-based assertions; green CI does not by itself prove every current R4 statement.
- A tracked historical artifact can be reproducible without being current training/production authority.
- Same tensor shape/version label can hide changed model/data meaning.
- Supplementary guides/labs can teach mechanics while containing old examples; they must not override canonical chapters.
- Local files can make one checkout appear more complete than a fresh clone.

## Common change recipe

When adding/changing a reference:

1. classify it as current canonical, active governance, historical/reproduction, supplementary, generated/local, or private;
2. register stable source/policy/artifact identity;
3. distinguish live/source-compatible metadata from current semantic/acceptance authority;
4. update the owning canonical chapter/current status;
5. update validator discovery when the **live** entry point changes;
6. update semantic validator logic in P5 when a machine-checkable current authority changes;
7. keep historical material but label its authority honestly;
8. run static/inventory checks and PR CI.

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
| `docs/plan/ml-R4/` | active controlling DATA/ML governance/evidence | authoritative for current R4 semantic/logical/physical/gate state |
| `docs/handbook/technical/` | **supplementary learning guides** | use for source-reading mechanics; verify pre-R4/pre-V3 examples against canonical chapters/source |
| `docs/handbook/labs/` | **supplementary practice labs** | safe exercises where prerequisites still match; not current-state authority |
| `docs/learning/` | historical/useful conceptual context | verify operational/current claims before reuse |
| prior plans/reports/experiments | historical evidence/context | preserve for auditability, not current implementation instructions |

### Canonical deep references

- [Architecture](01_architecture.md)
- [Current status](16_current_status.md)
- R4 evidence/policy/role/representation artifacts under [`docs/plan/ml-R4`](../plan/ml-R4)
- current ADRs/decision registers bound to implementation/artifact identity
- module tests/evidence tied to explicit commits/artifact hashes

### Historical compatibility roots

- historical v1/G7 DATA exports/splits/labels;
- historical G6 role manifests;
- Run12 checkpoint/threshold/calibration companions;
- v9 representation lineage;
- V1/V2 registry history/ABI paths;
- retained proxy proof bundle until replaced after repaired teacher selection.

Historical compatibility is deliberate; it must not be silently presented as the current repaired path.

## Technical mastery layer

### Prerequisite knowledge

Know repository navigation, semantic versioning, artifact provenance, partial-label semantics, representation versioning, EIP-712, source-symbol lookup, and current-vs-historical authority.

### Source map and reading order

Resolve a term here, jump to the owning canonical chapter, then executable source/current R4 machine artifact. Use supplementary guides/labs only after current status is clear. For DATA/ML, do not stop at G6/G7 metadata when later D-009/D-011/D-012 authority exists.

### Execution trace and worked example

For an on-chain audit, V3 context storage and read-only MCP observation are current protocol surfaces while V2 remains historical. For current DATA/ML planning, lookup starts at `data-vnext-policy-v1`, logical V3, D-011, and D-012—not at the historical G6 partition alone. For the live ML service, Run12 remains the historical operational model until an explicitly accepted/promoted repaired checkpoint replaces it.

### Implementation practice

When introducing a public symbol/config/artifact/error, record its authority/classification, owner, consumers, version/hash, failure behavior, verification path, and historical compatibility relationship.

### Review and ownership check

Can you locate the current DATA policy, logical V3 authority, D-011 physical acceptance, D-012 pending successor decision, Run12 runtime status, live audit-MCP tool surface, V3 submission method, and proxy proof scope without treating historical G6/G7 metadata as current semantic authority?