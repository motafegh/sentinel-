# SENTINEL AGENTS — Current Architecture Diagram

This is the current visual index for `agents/`. For exact status and limitations, see [`../docs/handbook/16_current_status.md`](../docs/handbook/16_current_status.md).

## 1. Off-chain analysis runtime

```mermaid
flowchart LR
    C[Client] --> G[Gateway :8000]
    G --> LG[14-node LangGraph]
    LG --> ML[ML API :8001]
    LG --> I[MCP inference :8010]
    LG --> R[MCP RAG :8011]
    LG --> A[MCP audit :8012 READ ONLY]
    LG --> GI[MCP graph inspector :8013]
    LG --> RP[MCP representation :8014]
    LG --> OUT[Off-chain report]
```

Gateway completion is not a V3 transaction. The report begins with unsubmitted on-chain state unless an independent external submission system acts later.

## 2. Fourteen-node LangGraph

```mermaid
flowchart TD
    M[1 ml_assessment] --> Q[2 quick_screen]
    Q --> E[3 evidence_router]
    E -->|deep| R1[4 rag_research]
    E -->|deep| S1[5 static_analysis]
    E -->|deep| G1[6 graph_explain]
    E -->|deep| F1[7 formal_verification]
    R1 --> AC[8 audit_check]
    S1 --> AC
    G1 --> AC
    F1 --> AC
    AC --> C1[9 consensus_engine]
    C1 --> CV[10 cross_validator]
    E -->|fast| SY[11 synthesizer]
    CV --> SY
    SY --> RF[12 reflection]
    RF --> EX[13 explainer]
    EX --> V[14 visualizer]
    V --> END[END]
```

`audit_check` observes registry history. It does not submit a transaction.

## 3. Live MCP surface

| Port | Service | Current tools |
|---:|---|---|
| 8010 | inference | `predict`, `batch_predict` |
| 8011 | RAG | `search` |
| 8012 | audit | `get_latest_audit`, `get_audit_history`, `check_audit_exists` |
| 8013 | graph inspector | `get_graph_hotspots` |
| 8014 | representation | `get_function_cfgs` |

Audit MCP :8012 is deliberately read-only. Its live server imports `_readonly_handlers.py`.

## 4. V3 protocol is a separate trust domain

```mermaid
flowchart LR
    F[ML fusion 128] --> P[Retained proxy/EZKL proof]
    P --> REQ[Fully bound V3 request]
    REQ --> PS[Isolated policy attestation]
    PS --> TX[External transaction authority]
    TX --> REG[AuditRegistry.submitAuditV3]
    REG --> RO[Audit MCP read-only observation]
```

The analysis MCP does not hold private keys, sign, construct, or broadcast the V3 transaction.

## 5. Feedback boundary

```mermaid
flowchart LR
    REG[V3 registry event] --> OBS[Versioned observation/finality truth]
    OBS --> POL[Feedback policy]
    POL -->|policy currently unavailable| J[Durable pending journal]
    J -. no automatic promotion .-> RAG[RAG / DATA truth]
```

Historical V1 scalar feedback behavior must not be reused as V3 promotion policy without measured evidence.

## 6. Evidence/trust reminders

- `tool_status.ran=false` is not a clean result.
- learned/tool/RAG/LLM outputs are evidence, not ground truth.
- `verdict_provable` is not the retained EZKL circuit statement.
- Run12 ML evidence remains historical-model evidence until R4 retraining.
- an on-chain V3 record proves protocol/proxy/context checks, not vulnerability ground truth.

## Source map

```text
agents/src/api/                     gateway/job store
agents/src/orchestration/           graph/state/nodes/verdict
agents/src/mcp/servers/             live five-service mesh
agents/src/mcp/servers/audit/       read-only registry observation
agents/src/security/policy_signer.py  V3 request/digest boundary
agents/src/contracts/               V3 submission truth models
agents/src/ingestion/               V3 feedback observation/policy/runtime
agents/src/rag/                     retrieval/index lifecycle
agents/src/eval/                    orchestration/evidence evaluation
```

Canonical detail: [`../docs/handbook/09_agents_orchestration.md`](../docs/handbook/09_agents_orchestration.md), [`../docs/handbook/10_agents_services.md`](../docs/handbook/10_agents_services.md), and [`../docs/handbook/12_security_and_trust.md`](../docs/handbook/12_security_and_trust.md).
