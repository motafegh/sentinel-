# 09 — AGENTS orchestration

**Read this when:** you need `AuditState`, the 14-node LangGraph, routing, evidence fusion, dual verdicts, timing, or degraded behavior.

**Skip this if:** you only operate MCP processes; read [agent services](10_agents_services.md).

**Estimated reading time:** 15 minutes.

## 30-second summary

Every audit starts at `ml_assessment`, always runs `quick_screen` and `evidence_router`, then either jumps to synthesis or fans out into selected evidence tools. Deep paths converge through on-chain history, consensus, and cross-validation. Every path then continues through `synthesizer → reflection → explainer → visualizer`; the graph ends at `visualizer`, not `synthesizer`.

## Just-enough mental model

```text
ml_assessment → quick_screen → evidence_router
                                  ├─ fast ────────────────┐
                                  └─ selected fan-out     │
                                     RAG/static/graph/formal
                                              ↓           │
                         audit_check → consensus → debate │
                                              └───────────┤
                                                  synthesizer
                                                     ↓
                                      reflection → explainer → visualizer → END
```

Routing chooses effort. Evidence records what ran. Fusion chooses verdicts. The report explains both decisions and failures.

## Actual runtime/source walkthrough

[`graph.py`](../../agents/src/orchestration/graph.py) — `agents/src/orchestration/graph.py::build_graph` registers exactly 14 nodes:

1. `ml_assessment` calls the inference path and records model evidence/hash.
2. `quick_screen` runs low-cost Slither/Aderyn checks on every contract.
3. `evidence_router` records active tools using deterministic routing policy.
4. `rag_research`, `static_analysis`, `graph_explain`, and `formal_verification` are possible deep-path branches.
5. `audit_check` reads prior registry evidence.
6. `consensus_engine` creates per-class weighted consensus/confidence.
7. `cross_validator` runs rule/LLM prosecutor-defender-judge validation when configured.
8. `synthesizer` calls deterministic `fuse()`, builds dual verdicts, narrative, provenance block, security block, and an unsubmitted on-chain placeholder.
9. `reflection`, `explainer`, and `visualizer` enrich the report and close the graph.

Fast routing still reaches all four post-routing nodes beginning with `synthesizer`. Deep branches converge at `audit_check`. [`timing.py`](../../agents/src/orchestration/timing.py) — `::timed_node` wraps every registered node.

## Interfaces, data shapes, and configuration

[`state.py`](../../agents/src/orchestration/state.py) — `agents/src/orchestration/state.py::AuditState` is a `total=False` typed dictionary. Important field groups:

- immutable input: `contract_code`, `contract_address`;
- ML/routing: `ml_result`, `model_hash`, `quick_screen_hits`, `routing_decisions`;
- evidence: `evidence_list`, `static_findings`, `rag_results`, `audit_history`, graph/formal findings;
- decisions: `consensus_verdict`, `verdict_provable`, `verdict_full`, confirmations/contradictions/confidence;
- accountability: `tool_status`, `injection_matches`, `error`, reflection/timing/explanations;
- output: `final_report`, narrative, visualization.

Append reducers merge routing, evidence, and injection lists; `_merge_tool_status` merges per-tool status. A tool result must distinguish `ran=true, findings=[]` from `ran=false, reason=...`.

`verdict_provable` uses deterministic evidence only. `verdict_full` may include nondeterministic RAG/LLM evidence. Despite its name, the former is not currently proved by EZKL; “provable” here means designed as the deterministic tier that could be anchored by a future proof.

Routing/consensus/confidence policies load from versioned YAML in [`agents/configs`](../../agents/configs). Timeouts live in [`timeouts.py`](../../agents/src/orchestration/timeouts.py). Decision-number changes require evaluation evidence.

## Failure modes and current limitations

- LangGraph’s gateway path does not invoke the separate MCP `submit_audit` tool.
- `final_report["on_chain"]` initially reports `submitted=false` with null transaction/proof fields.
- A node error can produce a partial report; inspect `tool_status` and `error` before trusting empty evidence.
- The graph checkpointer falls back to memory if SQLite support is unavailable, losing restart persistence.
- LLM-disabled/deterministic mode changes RAG/debate/narrative behavior and must be visible in the report.
- Reflection and attribution are explanatory layers, not independent verification.
- Current suite failures are recorded in [current status](16_current_status.md).

## Common change recipe

To add a node:

1. Define its input/output subset and structured failure status.
2. Decide whether outputs are deterministic evidence, nondeterministic evidence, or explanation only.
3. Add fields/reducers to `AuditState` only when necessary.
4. Register the timed node and explicit edges/routing.
5. Update fan-in expectations, synthesis, eval fixtures, timeouts, and graph topology tests.
6. Update metadata and cross-module contracts; run static validation.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
cd agents
poetry run pytest tests/test_graph.py tests/test_state.py -q   # smoke/targeted
poetry run pytest -q                                           # module
cd ..
python3 docs/handbook/tools/verify_handbook.py static           # topology contract
```

Current counts are only in [current status](16_current_status.md).

## Optional deep references

- [`nodes`](../../agents/src/orchestration/nodes) — one source file per node
- [`verdict`](../../agents/src/orchestration/verdict) — evidence model, reliability, and fusion
- [`docs/learning/01_orchestration_pipeline.md`](../learning/01_orchestration_pipeline.md)
- [`docs/learning/02_evidence_model_fuse.md`](../learning/02_evidence_model_fuse.md)
- [Security and trust](12_security_and_trust.md)

## Technical mastery layer

### Prerequisite knowledge

Know async DAGs, partial typed state, reducers, fan-out/fan-in, evidence weights, and deterministic versus nondeterministic sources.

### Source map and reading order

Read `state.py::AuditState`, graph routing/build, all node modules in execution order, then evidence/reliability/fuse. [T07](technical/07_agents_orchestration_evidence.md) curates the current source; older learning pages are supporting context only.

### Execution trace and worked example

Input crosses all always-run nodes, conditionally fans into deep evidence, converges through audit/consensus/cross-validation/synthesis, and always ends `reflection → explainer → visualizer`. `fuse()` emits deterministic-only and all-evidence verdicts with source-family discount.

### Implementation practice

[L07](labs/07_agents_state_fusion.md) constructs reducer/evidence cases. A new node requires owned state fields, reducer choice, `tool_status`, failure semantics, routing edges, isolation tests, and report consumption.

### Review and ownership check

Can you trace both paths through all 14 registered nodes and prove a timeout never becomes safety evidence?
