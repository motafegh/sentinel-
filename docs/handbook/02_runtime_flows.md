# 02 — Runtime flows

**Read this when:** you need to distinguish what happens during an off-chain audit, a direct ZK/on-chain submission, or feedback ingestion.

**Skip this if:** never when operating the system; confusing these flows creates false production claims.

**Estimated reading time:** 12 minutes.

## 30-second summary

SENTINEL currently has three separate flows. The gateway runs an asynchronous off-chain LangGraph audit and stores a report. The audit MCP’s `submit_audit` independently fetches a fusion embedding, runs the proxy, creates/verifies an EZKL proof, and submits V2 scores. Feedback ingestion separately turns completed reports into reviewable future knowledge/data. The first flow does not invoke the second. Its report begins with an unsubmitted placeholder rather than transaction evidence.

## Just-enough mental model

```text
Flow A — report: client → gateway → LangGraph → SQLite job/report
Flow B — chain:  MCP submit_audit → ML fusion → proxy → EZKL → AuditRegistryV2
Flow C — learn:  reports/feedback → ingest/dedup/review → RAG or future DATA/ML cycle
```

These flows can refer to the same contract, but today they do not share one transactional workflow or automatically bind their identities.

## Actual runtime/source walkthrough

### A. Off-chain gateway audit

1. `POST /audit` validates `source_code`, optional address, timeout, and metadata.
2. [`gateway.py`](../../agents/src/api/gateway.py) creates a queued SQLite job and starts a background task.
3. The task marks running and calls `graph.ainvoke` with source/address.
4. [`graph.py`](../../agents/src/orchestration/graph.py) runs the 14-node route described in [orchestration](09_agents_orchestration.md).
5. The synthesizer creates `final_report["on_chain"] = {submitted: false, tx_hash: null, proof_hash: null, ...}`.
6. Enrichment completes at `visualizer`; the gateway stores the report and marks the job completed.
7. The client polls `GET /audit/{job_id}`.

No step calls MCP `submit_audit`.

### B. Direct ZK/on-chain submission

1. A client explicitly invokes the audit MCP tool `submit_audit(source_code, contract_address, model_hash)` on port 8012.
2. [`_submit.py`](../../agents/src/mcp/servers/audit/_submit.py) calls ML `POST /fusion-embedding`.
3. It loads the proxy checkpoint, produces ten scores, and converts them to scale-13 field values.
4. It writes proof input, generates witness/proof, verifies off-chain, and uses the proof’s actual quantized outputs.
5. It builds an optional operator provenance assertion.
6. With RPC, registry, key, funding, and stake available, it calls `submitAuditV2`; otherwise it returns structured `partial` or `failed` status.

### C. Feedback ingestion

1. Completed reports or externally supplied findings enter [`ingestion/pipeline.py`](../../agents/src/ingestion/pipeline.py).
2. Deduplication and feedback handling normalize identities and records.
3. Reviewable outputs may feed a RAG rebuild or a future DATA labeling/training cycle.
4. No automatic self-training or trust escalation occurs merely because the system produced a report.

## Interfaces, data shapes, and configuration

| Flow | Entry | Durable output | Main prerequisites |
|---|---|---|---|
| off-chain | gateway `POST /audit` | SQLite job + report/checkpoint/runtime files | gateway, ML/tool services as selected |
| direct chain | MCP `submit_audit` | proof response + chain transaction/record | ML, proxy, EZKL artifacts, RPC, key, funding, stake, deployment |
| feedback | ingestion functions/schedulers | normalized/deduplicated feedback/RAG inputs | source/report provenance and review policy |

Flow A returns an agent verdict and evidence report. Flow B stores proxy scores/model hash/proof hash. Those are different shapes and claims.

## Failure modes and current limitations

- A completed Flow A job does not imply proof generation or chain submission.
- Flow A’s placeholder on-chain fields can be mistaken for an attempted transaction unless `submitted` and status are inspected.
- Flow B shares mutable proof paths, uses inconsistent proof-hash algorithms across response/contract, and can propagate a caller hash rather than the ML-returned hash.
- Flow B’s provenance may be unsigned and remains off-chain.
- Flow C can amplify model errors if generated reports are accepted as ground truth without independent review.
- Cross-flow correlation currently relies on contract address, hashes, timestamps, and operator discipline rather than one enforced audit ID.

## Common change recipe

To integrate Flow A and B safely:

1. Define one versioned submission request/result schema and audit identity.
2. Decide the explicit policy gate for submission; do not submit merely because an off-chain audit completed.
3. Use per-job temporary proof workspaces and consistent hash algorithms.
4. Validate/propagate the ML-returned model hash.
5. Persist submitted/partial/failed results back into the same job transactionally.
6. Add idempotency, retry, concurrency, and chain-reorg tests.
7. Update trust claims and evaluation before describing the flow as integrated.

## Verification commands

```bash
curl -fsS http://127.0.0.1:8000/health                                      # smoke/live
curl -fsS -X POST http://127.0.0.1:8000/audit -H 'content-type: application/json' \
  --data-binary @agents/tests/fixtures/gateway_audit_request.json             # off-chain, if fixture exists
python3 docs/handbook/tools/verify_handbook.py live --services                # service probes
python3 docs/handbook/tools/verify_handbook.py live --ezkl --anvil            # opt-in direct path prerequisites
```

Use a locally created non-secret request body if the named fixture is absent. Full commands and startup order are in [operations](14_operations.md).

## Optional deep references

- [AGENTS orchestration](09_agents_orchestration.md)
- [AGENTS services](10_agents_services.md)
- [ZKML](07_zkml.md)
- [Contracts](08_contracts.md)

## Technical mastery layer

### Prerequisite knowledge

Know async jobs, request/response schemas, evidence accumulation, proof/public inputs, and chain transactions.

### Source map and reading order

Off-chain: `gateway.py::create_app.submit_audit` → `::_run_job` → `graph.py::build_graph`. Direct chain: `audit/_submit.py::_run_submit` → `api.py::fusion_embedding` → EZKL → `AuditRegistry.submitAuditV2`. Feedback: `feedback_loop.py::OnChainListener` → `::FeedbackIngester`. See [T08](technical/08_services_rag_gateway.md) and [T10](technical/10_end_to_end_debugging.md).

### Execution trace and worked example

For source `Vault.sol`, the gateway returns a job ID and later a report whose `on_chain` field begins unsubmitted. A separate tool invocation can obtain the 128-value fusion embedding, produce ten proxy fields/138 public instances, and submit a transaction. The event may later enter feedback ingestion. None of these later steps is implied by gateway completion.

### Implementation practice

When changing a payload, update producer model, consumer, persistence, redaction, error mapping, focused tests, and the cross-module registry together. [L08](labs/08_services_rag_gateway_recovery.md) covers jobs; [L10](labs/10_end_to_end_capstone.md) covers both flows.

### Review and ownership check

Can you produce separate success and failure timelines for all three flows and identify the first durable record in each?
