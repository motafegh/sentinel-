# 17 — Reference registry

**Read this when:** you need a glossary, source-symbol map, configuration/environment registry, artifact classification, or historical-doc status.

**Skip this if:** you are following a learning path for the first time; use it as a lookup page.

**Estimated reading time:** 12 minutes to scan.

## 30-second summary

This is the handbook’s lookup layer. It maps stable concepts to source symbols and distinguishes canonical deep references from useful history and superseded plans. The machine-readable companion is [`_meta/handbook.toml`](_meta/handbook.toml).

## Just-enough mental model

Use `relative/path::symbol` to locate truth. Links open the file; symbols survive line movement. Source remains authoritative if a historical document disagrees.

## Actual runtime/source walkthrough

### Glossary

| Term | Meaning |
|---|---|
| DATA | ten-stage data lifecycle and artifacts |
| teacher | full four-eye `SentinelModel` |
| proxy/student | frozen 128→64→32→10 ZK-compatible model |
| fusion embedding | 128-value teacher representation consumed by proxy |
| deterministic evidence | reproducible tool/model evidence eligible for `verdict_provable` |
| full evidence | deterministic plus RAG/LLM evidence used by `verdict_full` |
| Rule 5C | failures/skips must be explicit, never empty-as-clean |
| artifact hash | integrity identity for an export/checkpoint/proof-related file |
| provenance | claim about origin/process; not automatically cryptographic proof |
| smoke/module/live | fast targeted / full subsystem / real external execution tiers |

### Source-symbol map

| Concern | Stable reference |
|---|---|
| DATA stage registry | `data_module/sentinel_data/cli.py::STAGES`, `::_STAGE_FN` |
| schema/classes | `data_module/sentinel_data/representation/graph_schema.py::FEATURE_SCHEMA_VERSION`, `::CLASS_NAMES` |
| dataset seam | `ml/src/datasets/sentinel_dataset.py::SentinelDataset` |
| teacher | `ml/src/models/sentinel_model.py::SentinelModel` |
| inference API | `ml/src/inference/api.py::app` |
| proxy/circuit | `zkml/src/distillation/proxy_model.py::ProxyModel`, `::CIRCUIT_VERSION` |
| circuit setup/proof | `zkml/src/ezkl/setup_circuit.py::setup_circuit`, `run_proof.py::main` |
| registry | `contracts/src/AuditRegistry.sol::AuditRegistry` |
| state/graph | `agents/src/orchestration/state.py::AuditState`, `graph.py::build_graph` |
| fusion verdict | `agents/src/orchestration/verdict/fuse.py::fuse` |
| gateway/jobs | `agents/src/api/gateway.py::create_app`, `sqlite_job_store.py::SqliteJobStore` |
| reliability | `agents/src/eval/reliability_fit.py::_fit_cell` |
| injection defense | `agents/src/security/prompt_sanitize.py::sanitize_for_prompt` |

### Configuration registry

| Area | Location |
|---|---|
| DATA sources/policies | [`data_module/config.yaml`](../../data_module/config.yaml) |
| ML training/MLOps | [`train.py`](../../ml/scripts/train.py) and [`mlops_config.json`](../../ml/mlops_config.json) |
| AGENTS verdict/routing policy | [`agents/configs/verdicts_default.yaml`](../../agents/configs/verdicts_default.yaml) |
| fitted reliability | [`agents/configs/reliability_v3.yaml`](../../agents/configs/reliability_v3.yaml) |
| EZKL circuit | [`zkml/ezkl/settings.json`](../../zkml/ezkl/settings.json) |
| Foundry/networks | [`contracts/foundry.toml`](../../contracts/foundry.toml) |
| handbook facts | [`handbook.toml`](_meta/handbook.toml) |

### Environment-variable registry

Names only; inspect source for current defaults and never paste values:

- gateway: `GATEWAY_PORT`, database/health interval and audit limit variables;
- ML: `SENTINEL_CHECKPOINT`, `SENTINEL_DRIFT_BASELINE`, `SENTINEL_DETERMINISTIC`;
- MCP: per-service port variables, ML API URL, mock flags, timeout/index variables;
- LLM/RAG: model/base URL and index/embedder configuration names;
- chain: RPC URL, registry address, operator key, confirmation configuration;
- testing: `TMPDIR`, `TMP`, `TEMP`.

Use `rg -n 'os.getenv|os.environ' agents/src ml/src zkml/src` to generate the exact current inventory.

### Artifact matrix

| Classification | Examples | Fresh clone behavior |
|---|---|---|
| tracked | source/config, proxy, ONNX, compiled circuit, settings, VK, generated verifier | present via Git |
| DVC-managed local | Run 12 checkpoint and companions in this checkout | pointer files are also ignored/untracked here; separate acquisition needed |
| regenerated | caches, RAG indexes, gateway/checkpoint DBs, witness/proof inputs | created by commands/services |
| ignored/private | proving key; secrets; operator state | must be securely supplied/regenerated |
| ignored local | DATA exports/splits; `AuditRegistryV2.t.sol` | not clone coverage or guaranteed availability |

## Interfaces, data shapes, and configuration

The machine-readable registry owns canonical page names, required template sections, ports/routes, critical dimensions, source ownership, artifacts, and test tiers. Update it whenever a referenced fact changes, then run both validator modes.

## Failure modes and current limitations

- This map can lag source; static validation catches declared critical facts, not every semantic change.
- Local files can make a checkout look more complete than a fresh clone.
- Historical docs can contain excellent reasoning and stale current-state claims simultaneously.
- Source comments/docstrings are not stronger than executable behavior.

## Common change recipe

When adding a document or reference:

1. Decide whether it is canonical handbook, canonical deep reference, useful history, or superseded plan.
2. Give canonical pages source ownership in metadata.
3. Link by relative path plus `path::symbol` where useful.
4. Avoid volatile counts outside status and avoid secrets everywhere.
5. Run static link/path/fact validation.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
git ls-files
git check-ignore -v <suspected-local-artifact>
```

## Optional deep references

### Learning-material classification

| Material | Classification | How to use it now |
|---|---|---|
| `docs/handbook/technical/` and `docs/handbook/labs/` | current and source-validated | primary technical study/implementation layer; checked by symbol/template/preflight validation |
| `docs/learning/01_*` through `10_*` | useful historical deep context | strong explanations, especially for AGENTS; verify every operational claim against the matching T07–T09 guide and source |
| `docs/learning/agents-ownership/` | useful precursor, partially superseded | exercises informed the ownership approach; use v3 labs for current commands, paths, gaps, and reset rules |
| `docs/learning/2026-06-23_agents_module_mastery_roadmap.md` and `LEARNING_DOCS_SPEC.md` | historical planning context | explains learning intent, not current runtime truth |
| `docs/learning/plans/` | superseded plans | retain for auditability; do not use as implementation instructions |

“Refreshed” in v3 means the topic was re-derived into the matching technical guide; it does not mean the older file was silently rewritten or fully revalidated.

### Canonical deep reference

- [`docs/learning`](../learning) — focused conceptual tutorials; verify source/current status before operational use.
- [`docs/ml/adr`](../ml/adr) and other ADR directories — design rationale; source defines current implementation.
- module testing specs and evaluation reports bound to explicit artifacts/commits.

### Useful historical context

- [`docs/proposal`](../proposal), [`docs/reports`](../reports), prior experiments, run notes, and archived analyses.
- Older module READMEs may explain intent but are not current authority.

### Superseded material

- [`docs/plan/system-finalization/handbook`](../plan/system-finalization/handbook) — D1 v1 per-page plans, superseded by the D1 v3 master plan and this implemented handbook.
- Historical plans remain for auditability; they are not required reading for the core learning path.

## Technical mastery layer

### Prerequisite knowledge

Know repository navigation, Python/Solidity symbol syntax, configuration precedence, environment variables, and artifact ownership.

### Source map and reading order

Use this page to resolve a term/config/artifact, then jump to the owning canonical chapter, registered guide, and lab. Machine-readable ownership lives in [`handbook.toml`](_meta/handbook.toml); symbol existence is validated by AST/constrained Solidity parsing.

### Execution trace and worked example

For `class_scores`, the map leads from proxy output/public positions through `AuditRegistry.submitAuditV2`, then report/query boundaries. For a configuration like `GATEWAY_PORT`, it leads to source default, service registry, operations command, and health check without copying an environment value.

### Implementation practice

When introducing a public symbol/config/artifact/error, add its owner, stable `path::symbol`, default or classification, consumers, and verification route. Historical documents are classified current/refreshed/context/superseded rather than silently trusted.

### Review and ownership check

Can you locate a symbol or error in source from this registry and decide whether its associated document/artifact is canonical, historical, local-only, or regenerated?
