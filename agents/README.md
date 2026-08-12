# SENTINEL AGENTS Module

`agents/` owns the 14-node LangGraph audit pipeline, gateway/job persistence, evidence fusion, RAG, five MCP services, security controls, evaluation utilities, V3 registry observation, and feedback boundaries.

> **Current authority:** the gateway/LangGraph path is **off-chain**. The live audit MCP on port 8012 is **read-only**. V3 signing/broadcast is outside the analysis MCP security domain, and current V3 feedback policy does not automatically promote observations into RAG or DATA/ML truth.

For the full current model, start with [runtime flows](../docs/handbook/02_runtime_flows.md), [AGENTS services](../docs/handbook/10_agents_services.md), and [security/trust](../docs/handbook/12_security_and_trust.md).

## Current topology

```text
client
  ↓
gateway :8000
  ↓
14-node LangGraph
  ├─ ML :8001
  ├─ inference MCP :8010
  ├─ RAG MCP :8011
  ├─ audit MCP :8012  READ ONLY
  ├─ graph inspector :8013
  └─ representation MCP :8014
  ↓
off-chain report
```

Graph order:

```text
ml_assessment → quick_screen → evidence_router
        ↓ selected/fast evidence paths
rag/static/graph/formal → audit_check → consensus_engine → cross_validator
        ↓
synthesizer → reflection → explainer → visualizer → END
```

The gateway does not sign or broadcast a registry transaction.

## Live MCP tools

| Port | Service | Current live tools |
|---:|---|---|
| 8010 | inference | `predict`, `batch_predict` |
| 8011 | RAG | `search` |
| 8012 | audit | `get_latest_audit`, `get_audit_history`, `check_audit_exists` |
| 8013 | graph inspector | `get_graph_hotspots` |
| 8014 | representation | `get_function_cfgs` |

`agents/src/mcp/servers/audit/_server.py` imports `_readonly_handlers.py`. Historical mutable audit-submission code remains for compatibility/history but is not the live analysis service write surface.

## V3 protocol relationship

V3 submission is a separate trust boundary:

```text
retained proxy proof
+ fully bound V3 request
+ isolated policy attestation
+ external transaction authority
→ AuditRegistry.submitAuditV3
```

`agents/src/security/policy_signer.py` builds/validates the V3 request/digest but intentionally contains no private key, signing, transaction construction, broadcast, or receipt handling.

The analysis MCP must not regain write authority by importing historical `_submit.py` unless a new approved security architecture explicitly changes that boundary.

## Feedback state

Current V3 feedback behavior separates:

1. registry/event observation;
2. versioned submission/finality truth;
3. feedback-policy decision;
4. mutation/promotion.

The current V3 promotion policy is intentionally unavailable. Therefore a V3 observation may be durably journaled as pending but does **not** automatically become RAG knowledge or DATA/ML ground truth.

Do not reuse the historical V1 scalar threshold as V3 policy.

## Evidence and verdict rules

- tool not run ≠ tool ran clean;
- `tool_status` must expose unavailable/degraded execution;
- deterministic and nondeterministic evidence remain distinguishable;
- `verdict_provable` is a historical naming convention for deterministic evidence fusion; it is **not** the statement proved by the retained EZKL circuit;
- Run12 ML output remains historical-model evidence until a repaired DATA-vNext teacher is trained;
- registry history is evidence/provenance, not automatic vulnerability ground truth.

## Main source areas

```text
agents/src/orchestration/   AuditState, graph, nodes, deterministic fusion
agents/src/api/             gateway + SQLite job persistence
agents/src/mcp/             five live MCP services
agents/src/rag/             retrieval/index lifecycle
agents/src/security/        prompt-injection + V3 policy-request boundary
agents/src/contracts/       versioned V3 submission truth models
agents/src/ingestion/       feedback observation/policy/runtime
agents/src/eval/            orchestration/evidence evaluation utilities
agents/configs/             externalized decision/reliability policy
```

## Running the off-chain runtime

```bash
cd agents
poetry install

poetry run python -m src.mcp.servers.inference_server
poetry run python -m src.mcp.servers.rag_server
poetry run python -m src.mcp.servers.audit_server
poetry run python -m src.mcp.servers.graph_inspector_server
poetry run python -m src.mcp.servers.representation_server
poetry run python -m src.api.gateway
```

External services/artifacts such as the Run12 ML checkpoint, LM Studio/RAG index, RPC registry reads, and analyzer binaries remain explicit prerequisites. Keep secrets in local environment configuration; never commit them.

## Verification

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
cd agents
poetry run pytest -q
cd ..
python3 docs/handbook/tools/verify_handbook.py static
```

Do not carry old volatile test counts in this README. Current suite totals belong in `docs/handbook/16_current_status.md` only after an intentional rerun against a named commit/environment.

## Permanent boundaries

- Gateway audit completion is off-chain only.
- Audit MCP is read-only.
- V3 signing/broadcast is outside the analysis MCP.
- A valid proxy proof is not proof of the AGENTS verdict.
- V3 policy attestation binds context; it does not expand the circuit statement.
- V3 feedback does not auto-promote while policy is unavailable.
- Missing tools/services must surface explicit status rather than empty-as-clean results.

For current detail, see [AGENTS orchestration](../docs/handbook/09_agents_orchestration.md), [services](../docs/handbook/10_agents_services.md), [evaluation](../docs/handbook/13_evaluation.md), and [current status](../docs/handbook/16_current_status.md).
