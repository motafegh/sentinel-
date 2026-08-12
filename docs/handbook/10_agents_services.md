# 10 — AGENTS services, RAG, gateway, and feedback

**Read this when:** you need the five MCP servers, gateway jobs, RAG, read-only chain observation, or feedback ingestion.

**Skip this if:** you only need in-process LangGraph logic; read [orchestration](09_agents_orchestration.md).

**Estimated reading time:** 15 minutes.

## 30-second summary

AGENTS is a process topology, not one server. Gateway :8000 runs asynchronous off-chain audits; ML :8001 serves model inference; five MCP services occupy :8010–8014. The important current change is that **audit MCP :8012 is read-only and version-aware across V1/V2/V3**. Runtime transaction signing/submission was removed from the analysis MCP security domain. V3 feedback observation is versioned and fail-closed: without an authorized V3 feedback policy, new V3 observations are journaled pending and do not automatically enter RAG or DATA/ML.

## Just-enough mental model

```text
client → gateway:8000 → in-process LangGraph
                         ├→ ML API:8001
                         └→ MCP clients as nodes need them

MCP:
8010 inference
8011 RAG
8012 audit registry READS ONLY
8013 graph inspector
8014 representation

V3 chain record → versioned observation → feedback policy
policy unavailable → durable pending journal → NO automatic promotion
```

MCP availability does not imply gateway wiring, transaction authority, or evidence trust.

## Actual runtime/source walkthrough

### Five MCP services

| Port | Service | Current live tools |
|---:|---|---|
| 8010 | inference | `predict`, `batch_predict` |
| 8011 | RAG | `search` |
| 8012 | audit | `get_latest_audit`, `get_audit_history`, `check_audit_exists` |
| 8013 | graph inspector | `get_graph_hotspots` |
| 8014 | representation | `get_function_cfgs` |

[`audit/_server.py`](../../agents/src/mcp/servers/audit/_server.py) imports [`_readonly_handlers.py`](../../agents/src/mcp/servers/audit/_readonly_handlers.py). Write names are rejected before historical mutable submission code is imported. The three query names are protocol-neutral and observe V1/V2/V3 history through versioned reads.

### Gateway and JobStore

[`gateway.py`](../../agents/src/api/gateway.py) validates/enqueues audits, invokes the LangGraph asynchronously, and persists job/report state through SQLite. Gateway completion means an off-chain report completed; it says nothing about V3 signing or chain submission.

### RAG lifecycle

RAG fetchers/chunker/embedder/index/retriever remain separate from model ground truth. Retrieval combines dense/sparse evidence with metadata/provenance. A RAG hit is supporting evidence, not a label.

### Feedback lifecycle

There are two important eras:

- historical feedback code contains older scalar/V1 compatibility behavior;
- current V3 observation/runtime code separates chain observation, policy decision, and mutation.

For V3, the promotion policy version is intentionally unavailable. A V3 event can be observed and durably journaled, but automatic V3→RAG promotion remains disabled. No old scalar threshold may be silently ported to V3.

## Interfaces, data shapes, and configuration

Gateway routes remain:

- `GET /`
- `GET /health`
- `POST /audit`
- `GET /audit/{job_id}`
- `GET /audit`

Audit MCP read results include protocol version/provenance so V1 scalar, V2 ten-score, and V3 context-bound records remain distinguishable.

V3 feedback runtime must preserve explicit states such as observed, not-evaluated/policy-unavailable, pending, and mutation-blocked. Absence of policy is not a default deny/allow threshold—it is an explicit product state.

## Failure modes and current limitations

- Gateway health can be degraded while jobs still exist; inspect dependency details.
- Gateway completion never implies a transaction.
- Audit MCP must stay read-only; re-importing/signing through historical `_submit.py` would violate the current trust boundary.
- A production V3 signer/broadcaster is not implemented in the analysis service.
- V3 automatic feedback/RAG promotion is intentionally disabled until measured policy exists.
- Graph-inspector fallback evidence is not equivalent to ML hotspots and must retain provenance.
- RAG indexes are generated/local and may be stale or unavailable.
- Feedback is untrusted until provenance/review/policy authorizes mutation.

## Common change recipe

For a new MCP tool:

1. decide whether it is read-only or mutating before implementation;
2. keep signing/broadcast outside the analysis MCP unless a new security architecture explicitly changes that boundary;
3. define narrow schema, resource limits, explicit degraded/error states, and provenance;
4. wire a graph node only intentionally;
5. update metadata/tool inventory and security docs.

For V3 feedback policy:

1. derive policy from measured R4/ML/evaluation evidence;
2. version it explicitly;
3. add replay/finality/reorg/idempotency tests;
4. preserve pending journal behavior for unqualified observations;
5. only then permit controlled mutation/RAG promotion.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
cd agents
poetry run pytest -q -k 'gateway or mcp or feedback or submission_v3'
cd ..
python3 docs/handbook/tools/verify_handbook.py static
curl -fsS http://127.0.0.1:8012/health
```

## Optional deep references

- [Runtime flows](02_runtime_flows.md)
- [Security and trust](12_security_and_trust.md)
- [Operations](14_operations.md)
- [`audit/_readonly_handlers.py`](../../agents/src/mcp/servers/audit/_readonly_handlers.py)

## Technical mastery layer

### Prerequisite knowledge

Know MCP schemas, async HTTP, SQLite state machines, read/write trust separation, event ingestion, finality, and RAG provenance.

### Source map and reading order

Read gateway store/runner, each MCP live server, audit `_server.py` + `_readonly_handlers.py` + `_versioned_reads.py`, then V3 submission truth/observation/policy/runtime modules under `agents/src/contracts` and `agents/src/ingestion`.

### Execution trace and worked example

A gateway job runs the graph and stores a report. Separately, if a V3 chain record exists, audit MCP can read it. Feedback observer can record that V3 event; because current V3 promotion policy is unavailable, the event becomes durable pending feedback rather than automatic RAG knowledge.

### Implementation practice

Treat read observation, policy decision, mutation, signing, and transaction broadcasting as separate capabilities with separate tests and permissions. Do not collapse them into one convenient MCP handler.

### Review and ownership check

Can you list the five MCP services and state which one is read-only, what a V3 feedback event does today, and why neither gateway completion nor audit-MCP availability implies chain-write authority?
