# Step 1 — Big Picture

## Topology

```
START → ml_assessment → quick_screen → evidence_router
              │                              │
              │                    ┌─── deep path ───┐
              │                    │                   │
              │              rag_research      static_analysis      graph_explain
              │                    │                   │                   │
              │                    └─────────┬─────────┴───────────────────┘
              │                          audit_check
              │                               │
              │                       consensus_engine
              │                               │
              │                       cross_validator  ← the debate
              │                               │
              └─── fast path ────────────→ synthesizer
                                               │
                                          reflection
                                               │
                                          explainer
                                               │
                                          visualizer
                                               │
                                              END
```

## Where each node actually runs

| Node | Runs as | Talks to |
|---|---|---|
| `ml_assessment` | network call | MCP `inference_server` (:8010) → ML FastAPI (:8001) |
| `quick_screen`, `static_analysis` | in-process | Slither (Python lib) + Aderyn (subprocess) directly |
| `rag_research` | network call | MCP `rag_server` (:8011) |
| `graph_explain` | network call (in-process Slither fallback) | MCP `graph_inspector_server` (:8013) |
| `audit_check` | network call | MCP `audit_server` (:8012) |
| `consensus_engine`, `explainer`, `visualizer` | in-process, pure logic | nothing external |
| `cross_validator`, `synthesizer`, `reflection` | in-process, but make LLM calls | LM Studio (external app) |

So a single audit run coordinates: LM Studio + ML FastAPI + 4 MCP servers + the
LangGraph process itself = 6+ OS processes.

## Why it's 13 nodes now, not 9

`consensus_engine`, `reflection`, `explainer`, `visualizer` are Phase A additions
(2026-06-21). The original 9-node graph stopped at `synthesizer` — no self-critique,
no attribution breakdown, no visual output. `cross_validator` also changed shape
(1 LLM call → 3-role debate) without changing node count.

## Fast path vs deep path

Fast path requires TWO signals to agree the contract is safe: ML (all classes below
`DEEP_THRESHOLDS`) AND `quick_screen` (zero High/Critical Slither/Aderyn hits). If
either disagrees, it goes deep. This two-signal gate is why `quick_screen` runs on
EVERY contract, not just the ones ML flags.

→ You now know: an audit is a multi-process distributed system, not a single
script — and 4 of the 13 nodes plus the debate upgrade are this week's additions.
