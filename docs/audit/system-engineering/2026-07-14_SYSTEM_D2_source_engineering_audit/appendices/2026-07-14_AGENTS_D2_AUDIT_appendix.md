# SENTINEL D2 AGENTS and Services Source Engineering Audit

**Audit date:** 2026-07-14
**Locked executable baseline:** `4b5bd333c63ab7a7ec83810fbbae54f3ebf1b493`
**Audit worktree:** `/home/motafeq/projects/sentinel-d2-audit`
**Track:** AGENTS orchestration, gateway, jobs, MCP, RAG/feedback, LLM isolation, persistence, concurrency, and observability
**Disposition:** `APPROVED_FOR_R0_PLANNING`; runtime findings remain open
**Runtime changes:** none

> **Post-integration status:** This appendix preserves track-local `candidate` and `track-reproduced` labels as handoff evidence. The canonical registry has since adjudicated every accepted P1, merged `D2-AGT-013` into `D2-ZKC-004`, and records the final evidence state. Ali approved D2 for R0 planning; runtime code still requires a reviewed R0 plan.

## 1. Executive conclusion

The AGENTS module is a capable single-host research pipeline, but it is not yet a reliable production or decentralized-operator execution layer. The graph has explicit evidence records, dual verdicts, configurable routing, fail-soft nodes, persistent job records, and broad unit coverage. Those strengths are undermined at the interfaces between components: outage fallback can fabricate successful ML evidence; user-controlled report identifiers reach filesystem paths; production RAG scores and metadata do not match fusion assumptions; the gateway removes the fields required to audit degraded runs; graph and job recovery are disconnected; and several concurrent-state paths are unsafe or unbounded.

This appendix records **21 provisional findings**:

| Severity | Primary-reproduced | Track-reproduced | Candidate | Total |
|---|---:|---:|---:|---:|
| P0 | 2 | 0 | 0 | 2 |
| P1 | 2 | 4 | 10 | 16 |
| P2 | 0 | 1 | 2 | 3 |
| **Total** | **4** | **5** | **12** | **21** |

`primary-reproduced` means the primary D2 audit independently reproduced the behavior and recorded it in the D2 scratch history. `track-reproduced` means this appendix owner reproduced the correctness/recovery behavior with isolated, non-mutating probes. `candidate` means executable source establishes a credible defect or architectural gap, but primary reproduction or policy review is still required before registry acceptance.

The two P0s block a trustworthy public gateway. The P1 set blocks claims that a completed gateway job is complete, reproducible, recoverable, or safe to use as decentralized consensus input. Runtime remediation belongs in a later implementation branch; D2 should first deduplicate these records against DATA, ML, and ZK/contracts findings.

## 2. Audit scope and method

### 2.1 Baseline integrity

The audit branch had documentation-only recovery commits above the locked baseline. The command `git diff --name-only 4b5bd333c..HEAD -- agents` returned no paths, establishing that the inspected AGENTS executable source was unchanged from `4b5bd333c`.

Inventory at the baseline:

- 92 production Python files under `agents/src`.
- 41 `test_*.py` modules under `agents/tests`.
- 1,187 tracked paths under `agents`, including documentation and fixtures.
- Production Python ownership by package: API 5, config 3, eval 8, ingestion 6, LLM 1, MCP 14, orchestration 32, RAG 18, security 5.

### 2.2 Evidence discipline

- Existing scratch findings were treated as leads, then traced to executable `path::symbol` locations.
- Primary security reproductions are referenced without extending them; this track performed only defensive correctness, recovery, and test probes.
- Temporary SQLite databases and in-memory/minimal LangGraph instances were used for recovery and reducer checks.
- Canonical models, indexes, reports, proofs, databases, and configuration files were not intentionally modified.
- The full test run did create ignored address-named report files through existing test behavior. This side effect is itself documented in `D2-AGT-018`.

### 2.3 Commands and measured evidence

| Purpose | Command | Result |
|---|---|---|
| Baseline source equivalence | `git diff --name-only 4b5bd333c..HEAD -- agents` | Empty |
| Full AGENTS suite | `TMPDIR=/tmp TMP=/tmp TEMP=/tmp PYTHONPATH=/home/motafeq/projects/sentinel-d2-audit/agents /home/motafeq/projects/sentinel/agents/.venv/bin/python -m pytest agents/tests -q` | 622 passed, 11 failed, 1 skipped; 634 collected; 26.16 s |
| Gateway/store focus | Same environment, `pytest agents/tests/test_gateway.py agents/tests/test_p10_gateway.py -q` | 57 passed; 1.51 s |
| Job recovery | Temporary `SqliteJobStore`, one QUEUED and one RUNNING row | Recovery returned RUNNING only; QUEUED remained queued |
| Parallel error reducer | Minimal two-branch `StateGraph(AuditState)` with both branches writing `error` | `InvalidUpdateError` for concurrent update |
| Reliability launch directory | Load reliability from repository root and from `agents/` | Root: L1 `0.39/0.82`; `agents/`: L3 `0.127/0.343` for ML/Slither Reentrancy |

No GPU, live LM Studio, live ML service, live MCP transport, Sepolia, Anvil, EZKL proof-generation, transaction gas, or multi-process recovery benchmark was available or required for this bounded source track. Performance and live-service claims therefore remain unverified; they must not be inferred from unit timing.

## 3. Current architecture inventory

### 3.1 Execution and data flow

The public entry is FastAPI `POST /audit`. `gateway.create_app` validates the request, writes a `QUEUED` row, and launches one `asyncio` task per request. `_run_job` changes the row to `RUNNING`, constructs the graph with checkpointing disabled, invokes it under one wall-clock timeout, selects a reduced result envelope, and stores `COMPLETED` or `FAILED` in SQLite.

The graph is:

`ml_assessment → quick_screen → evidence_router → fast/deep branch`

- Fast: `synthesizer → reflection → explainer → visualizer`.
- Deep fan-out: selected RAG/static tools plus `graph_explain` and `formal_verification`; fan-in at `audit_check`; then `consensus_engine → cross_validator → synthesizer → reflection → explainer → visualizer`.

The shared `AuditState` is a `total=False` typed dictionary. `routing_decisions`, `evidence_list`, and `injection_matches` append; `tool_status` performs a one-level per-tool merge; most other keys replace. `consensus_engine` emits raw ML/static evidence plus a derived consensus record. `synthesizer` adds RAG evidence, calls `fuse`, emits `verdict_provable` and `verdict_full`, builds `final_report`, and writes JSON. Post-synthesis nodes enrich the in-memory report and write HTML.

External calls cross five local SSE MCP services, the ML API, LM Studio, static-analysis binaries, optional Halmos/Foundry, the RAG index, and Sepolia. The on-chain audit MCP also owns proof generation and a key-bearing submission path.

### 3.2 Interfaces and process ownership

| Process/interface | Default | Public operations | State/dependency |
|---|---|---|---|
| Audit gateway | `0.0.0.0:8000` | `POST /audit`, `GET /audit`, `GET /audit/{job_id}`, `GET /health`, `GET /` | SQLite jobs, graph tasks |
| Inference MCP | `0.0.0.0:8010` | `predict`, `batch_predict`; `/sse`, `/messages/`, `/health` | ML API `:8001`, shared HTTP client |
| RAG MCP | `0.0.0.0:8011` | `search`; `/sse`, `/messages/`, `/health` | FAISS, BM25, chunks, embedding model |
| Audit MCP | `0.0.0.0:8012` | `get_latest_audit`, `get_audit_history`, `check_audit_exists`, `submit_audit` | Sepolia RPC, registry ABI, operator key, EZKL files |
| Graph MCP | `0.0.0.0:8013` | `get_graph_hotspots`; `/sse`, `/messages/`, `/health` | ML `/hotspots`, Slither fallback |
| Representation MCP | `0.0.0.0:8014` | `get_function_cfgs`; `/sse`, `/messages/`, `/health` | DATA CFG builder, Slither/solc |
| LM Studio | environment URL, fallback WSL address | OpenAI-compatible chat and embedding calls | Models and GPU outside repository |
| Feedback loop | continuous Sepolia poller | Consumes `AuditSubmitted` | feedback cursor, reports, RAG index |

All MCP services use the same SSE shape. The gateway probes local health URLs, but probe success is not an execution manifest and is not bound to a job.

### 3.3 Persistent and mutable state

| State | Writer(s) | Reader(s) | Current coordination |
|---|---|---|---|
| `data/jobs.db` | Gateway/`SqliteJobStore` | Gateway API/lifespan | Per-process threading lock; CWD-relative default |
| `agents/data/checkpoints.db` | Default `build_graph()` | Graph resume callers | SQLite saver; explicitly disabled by gateway |
| `agents/data/reports/{address}.json` | Synthesizer | Feedback loop/humans | Direct non-atomic write; address-derived name |
| `agents/data/reports/{address}_hotspot.html` | Visualizer | Humans | Direct non-atomic write; address-derived name |
| FAISS/BM25/chunks | Build pipeline and feedback ingester | RAG server | File lock for writers; per-file atomic replacement; no atomic multi-file generation swap |
| RAG seen hashes/metadata | Ingestion paths | Ingestion/retriever | File-backed; separately updated from core index |
| Feedback cursor | On-chain listener | On-chain listener | Direct JSON write |
| EZKL proof input/witness/proof | Audit submission MCP | Same call and transaction builder | Shared canonical files; no per-job workspace |
| Operator nonce | Audit submission MCP | RPC/account | No local per-key allocator or lock |
| Module globals | LLM client, MCP clients/retrievers, config singleton, graph singleton | All calls in process | Import-time environment and process-global mutation |

The RAG index writer lock is a positive control. It does not make the three-file FAISS/chunks/BM25 update one atomic generation; a reader starting between replacements can fail the size check, which is safer than silently mis-associating chunks but still creates an availability window.

### 3.4 Configuration precedence and observability

- `gateway.py` loads `agents/.env` with `override=True` before its imports. Other modules also load dotenv at import time, not always with the same path or override semantics.
- Primary verdict config is package-anchored and cached in a mutable process singleton.
- L3 reliability config is CWD-relative, creating launch-directory-dependent fusion behavior.
- URLs, ports, timeouts, mock modes, LLM toggles, keys, checkpoint paths, and debate controls are spread across module-level environment reads and YAML.
- Uniform node START/DONE timing logs are useful, as are SQLite status counts and cached upstream health checks.
- There is no durable per-job execution manifest containing config hash, dependency versions, source hash, tool statuses, model/index/artifact hashes, route, timings, retry/lease history, or final evidence commitment.
- `error` is a single replace-valued string, not a structured multi-error collection.

## 4. Baseline suite interpretation

The full suite's 622 passes demonstrate substantial unit coverage of routing, evidence records, verdict fusion, gateway CRUD, prompt sanitation, static-tool adapters, and report generation. They do not constitute a green baseline.

The 11 failures divide into:

1. **Six fresh-clone data failures.** Five promised seed corpora are absent under `agents/data/knowledge`; the SWC fetcher consequently returns no canonical IDs.
2. **Three toolchain-resolution failures.** `solc` is not resolvable in the test process, breaking the deep smoke and two real-Slither checks even though other local tooling is available.
3. **Two reliability/configuration failures.** The CWD-relative L3 path falls back to L1 from the repository root. In the full suite, a prior test also leaves the mutable config singleton with `ml_weight_scale=1.0`, contaminating later expectations.

The one skipped benchmark must be kept separate from passing coverage. The deep smoke also wrote ignored report JSON and HTML into the worktree, proving that the suite is not fully side-effect isolated.

## 5. Findings summary

| ID | Severity | Status | Classification | Short title |
|---|---|---|---|---|
| D2-AGT-001 | P0 | primary-reproduced | correctness/trust | ML outage becomes successful mock evidence |
| D2-AGT-002 | P0 | primary-reproduced | security/integrity | Unvalidated report identifier reaches filesystem paths |
| D2-AGT-003 | P1 | primary-reproduced | correctness/integration | Production RAG evidence is inert at the fusion seam |
| D2-AGT-004 | P1 | primary-reproduced | observability/interface | Gateway strips dual-verdict and execution evidence |
| D2-AGT-005 | P1 | track-reproduced | recovery/architecture | Durable jobs are disconnected from graph recovery |
| D2-AGT-006 | P1 | candidate | interface/cost control | Per-request `no_llm` promises behavior it does not apply |
| D2-AGT-007 | P2 | candidate | observability | Execution path is inferred from finding cardinality |
| D2-AGT-008 | P1 | candidate | configuration/integration | Graph inspector defaults to the gateway, not ML service |
| D2-AGT-009 | P1 | candidate | scientific/correctness | Fusion counts derived consensus as an independent witness |
| D2-AGT-010 | P1 | candidate | trust model | “Provable” means emitter-asserted deterministic |
| D2-AGT-011 | P1 | candidate | security/architecture | Public service surfaces lack an authentication boundary |
| D2-AGT-012 | P1 | candidate | correctness/trust | Chain initialization failure becomes realistic mock history |
| D2-AGT-013 | P1 | candidate | concurrency/integration | Proof workspace and transaction nonce are shared |
| D2-AGT-014 | P1 | track-reproduced | configuration/reproducibility | Reliability weights depend on launch directory |
| D2-AGT-015 | P1 | track-reproduced | orchestration/recovery | Parallel tool errors crash the graph reducer |
| D2-AGT-016 | P1 | candidate | observability | Rule 5C tool status is incomplete and omitted from report |
| D2-AGT-017 | P1 | candidate | availability/concurrency | Gateway has no real capacity bound or backpressure |
| D2-AGT-018 | P1 | track-reproduced | packaging/testability | Fresh clone cannot satisfy advertised AGENTS baseline |
| D2-AGT-019 | P2 | candidate | persistence/concurrency | Report bridge is non-atomic and last-writer-wins |
| D2-AGT-020 | P1 | candidate | scientific/truth boundary | Feedback corpus overstates what the ZK proof guarantees |
| D2-AGT-021 | P2 | track-reproduced | test isolation | Test mutates the cached production config singleton |

## 6. Detailed findings

### D2-AGT-001 — ML outage becomes successful mock evidence

- **Severity / status:** P0 / `primary-reproduced`
- **Classification / owner:** correctness, evidence integrity / ML integration and AGENTS orchestration
- **Source:** `agents/src/mcp/servers/inference_server.py::_call_inference_api`, `::_mock_prediction`; `agents/src/orchestration/nodes/ml_assessment.py::ml_assessment`
- **Invariant:** An unavailable production ML service must produce explicit `ran=False` degraded state, never a plausible prediction.
- **Evidence:** With `_MOCK_MODE=False` and the HTTP client made to raise `httpx.ConnectError`, the primary audit observed `has_error=False`, label `safe`, probability data, and a `mock_model_hash_*` value without a mock/degraded marker. `ml_assessment` accepted the payload and wrote `tool_status.ml.ran=True`.
- **Impact / affected modules:** An outage can generate fabricated deterministic evidence, a safe-looking report, and an input to fusion. Affected: inference MCP, ML node, fusion, gateway, downstream proof/submission consumers.
- **Recommendation:** Restrict mock output to explicit startup/test configuration. Convert all request failures to a typed unavailable response; validate prediction provenance and model hash before accepting `ran=True`.
- **Rejected alternatives:** Logging the fallback is insufficient because consumers use the response body. Adding a `mock_model_hash` naming convention is insufficient because it is not a validated status contract.
- **Compatibility / migration / rollback:** Additive degraded fields are wire-compatible; temporarily fail closed on missing status. Rollback may restore the old adapter only in explicit non-production mock profiles.
- **Dependencies / required tests:** Shared ML response schema; outage, timeout, malformed response, explicit mock-mode, and gateway end-to-end tests.
- **Duplicate relationships / primary verification:** Related to ML service fallback findings, but AGENTS owns acceptance semantics. Primary verification complete.

### D2-AGT-002 — Unvalidated report identifier reaches filesystem paths

- **Severity / status:** P0 / `primary-reproduced`
- **Classification / owner:** security, artifact integrity / Gateway and AGENTS report persistence
- **Source:** `agents/src/api/models.py::AuditRequest.contract_address`; `agents/src/orchestration/nodes/synthesizer.py::synthesizer`; `agents/src/orchestration/nodes/visualizer.py::visualizer`
- **Invariant:** A public request identifier must not select a filesystem location outside a job-scoped report directory.
- **Evidence:** The primary audit used an isolated report root and confirmed that an address containing `../` wrote JSON outside that root. Source shows the same value also names HTML output. No canonical repository artifact was touched.
- **Impact / affected modules:** A request can overwrite service-account-accessible JSON/HTML paths, corrupt reports or artifacts, and collide across jobs. Affected: gateway validation, synthesizer, visualizer, feedback bridge.
- **Recommendation:** Validate canonical Ethereum addresses at the gateway; use server-generated job IDs as filenames; keep address only as data; resolve and enforce containment; write atomically in per-job directories.
- **Rejected alternatives:** Character replacement alone creates collisions. A suffix check does not enforce containment. Relying on downstream on-chain address validation is ineffective because persistence happens first.
- **Compatibility / migration / rollback:** Preserve address in API/report schemas. Migrate legacy address-named reports through an indexed lookup. Roll back by disabling disk report/HTML persistence, not by re-enabling address-derived paths.
- **Dependencies / required tests:** API address schema, feedback lookup migration; traversal, absolute path, separator, Unicode, collision, symlink, and concurrent same-address tests.
- **Duplicate relationships / primary verification:** Visualizer is the same root cause and should merge here. Primary verification complete.

### D2-AGT-003 — Production RAG evidence is inert at the fusion seam

- **Severity / status:** P1 / `primary-reproduced`
- **Classification / owner:** correctness, schema/score integration / RAG and verdict fusion
- **Source:** `agents/src/rag/retriever.py::HybridRetriever.search`; `agents/src/orchestration/verdict/emit.py::emit_rag_evidence`; `agents/src/orchestration/nodes/_helpers.py::_best_rag_score`; `agents/src/rag/chunker.py::Chunk.metadata`
- **Invariant:** Production retrieval output must use the same class key and normalized score domain consumed by fusion, confidence, and attribution.
- **Evidence:** Primary reproduction emitted zero evidence for production `metadata.vuln_type` and one for test-only `metadata.vulnerability_type`. Independently, source establishes an RRF maximum of `2/60 = 0.0333`, while fusion and attribution floors begin at 0.30 and legacy verdict thresholds at 0.50/0.80. Tests inject 0–1 “similarity” scores rather than real RRF scores.
- **Impact / affected modules:** RAG can enrich narrative text but contributes no production fusion evidence, confidence, or metric attribution. Reported multi-source corroboration and calibration are invalid.
- **Recommendation:** Define a versioned retrieval result schema. Keep raw RRF rank score separately; add a measured normalized relevance/calibration field; standardize `vuln_type`; make consumers reject unknown score domains.
- **Rejected alternatives:** Merely lowering the floor preserves an uncalibrated rank score. Merely renaming metadata leaves the score-domain defect. Scaling by a fixed constant without validation invents probability semantics.
- **Compatibility / migration / rollback:** Emit both legacy and canonical keys during one release and version the score fields. Roll back by excluding RAG from fusion while retaining narrative retrieval.
- **Dependencies / required tests:** RAG index fixture, production retriever serialization, calibration dataset; end-to-end retriever→MCP→node→evidence tests using actual RRF values.
- **Duplicate relationships / primary verification:** Combines the metadata and score-domain manifestations as one root interface finding. Metadata seam primary-reproduced; score-domain source-verified and awaits primary acceptance.

### D2-AGT-004 — Gateway strips dual-verdict and execution evidence

- **Severity / status:** P1 / `primary-reproduced`
- **Classification / owner:** observability, API contract / Gateway
- **Source:** `agents/src/api/gateway.py::_run_job`; `agents/src/orchestration/nodes/synthesizer.py::synthesizer`
- **Invariant:** A completed persisted job must retain the fields needed to distinguish clean, skipped, degraded, advisory, and deterministic execution.
- **Evidence:** Primary reproduction passed graph output containing failed `tool_status`, `evidence_list`, `verdict_provable`, and `verdict_full`; the persisted response omitted all four. `final_report` also omits tool status and evidence detail.
- **Impact / affected modules:** API clients cannot audit Rule 5C, dual-verdict separation, or evidence lineage. A degraded execution can look complete; feedback and evaluation consume an incomplete record.
- **Recommendation:** Define one versioned canonical result envelope and persist it losslessly or by content-addressed reference. Include tool statuses, manifest, dual verdicts, evidence commitment/details, timings, and degradation summary.
- **Rejected alternatives:** Adding counts only cannot prove which evidence drove a verdict. Keeping data solely in LangGraph state fails because gateway checkpointing is disabled.
- **Compatibility / migration / rollback:** Add fields without removing legacy report keys; backfill only when source artifacts exist. Rollback can suppress new fields from old clients while retaining storage.
- **Dependencies / required tests:** Canonical result schema and storage migration; round-trip field equality, degraded-run, schema-version, and backward-client tests.
- **Duplicate relationships / primary verification:** Related to `D2-AGT-016`, but this finding covers gateway loss after graph execution. Primary verification complete.

### D2-AGT-005 — Durable jobs are disconnected from graph recovery

- **Severity / status:** P1 / `track-reproduced`
- **Classification / owner:** recovery, job architecture / Gateway and orchestration
- **Source:** `agents/src/api/gateway.py::create_app`, `::lifespan`, `::_run_job`; `agents/src/api/sqlite_job_store.py::recover_pending`; `agents/src/orchestration/graph.py::build_graph`
- **Invariant:** Every accepted durable job must deterministically reach a terminal state or be safely reclaimed after process loss.
- **Evidence:** A temporary database with QUEUED and RUNNING rows returned only RUNNING from `recover_pending`; QUEUED remained queued. Lifespan marks recovered RUNNING jobs failed but never reclaims QUEUED. The default gateway graph factory calls `build_graph(use_checkpointer=False)`, so documented node-level checkpoint recovery is unavailable to gateway jobs.
- **Impact / affected modules:** Crash windows strand QUEUED work forever; RUNNING work is discarded rather than resumed; the two SQLite stores do not form one transaction or recovery protocol.
- **Recommendation:** Introduce durable claim/lease/heartbeat/attempt fields, idempotent workers, startup reclamation for both queued and expired-running jobs, and a job-to-graph checkpoint identity. Use one explicit recovery state machine.
- **Rejected alternatives:** Marking every startup row failed prevents hanging but does not deliver accepted work. Enabling the checkpointer alone does not atomically bind job and graph state.
- **Compatibility / migration / rollback:** Add columns and states while mapping legacy rows conservatively. Rollback should drain new claims and retain read compatibility with legacy four-state jobs.
- **Dependencies / required tests:** Job schema, worker ownership, idempotent report writes; crash-before-task, crash-after-claim, mid-node crash, lease expiry, retry exhaustion, duplicate worker, and restart tests.
- **Duplicate relationships / primary verification:** Cross-system distributed job architecture should reference this finding. Track reproduction complete; primary verification pending.

### D2-AGT-006 — Per-request `no_llm` promises behavior it does not apply

- **Severity / status:** P1 / `candidate`
- **Classification / owner:** interface correctness, cost/privacy control / Gateway and LLM integration
- **Source:** `agents/src/api/gateway.py::submit_audit`, `::_patch_no_llm`; `agents/src/orchestration/nodes/_helpers.py::_llm_enabled`
- **Invariant:** An accepted request-level “skip LLM” control must disable every advisory LLM call for that job.
- **Evidence:** When the app is not globally no-LLM, the query handler only writes `_effective_no_llm` into metadata. That metadata is not passed into graph state and `_llm_enabled` reads process environment only. Existing test asserts only that metadata is recorded.
- **Impact / affected modules:** A caller can incur unrequested LLM execution, latency, GPU use, and source disclosure despite the API promise.
- **Recommendation:** Add an immutable per-job execution policy to initial graph state and have every LLM node consult it. Keep global no-LLM as a startup ceiling, not a monkeypatch.
- **Rejected alternatives:** Process-global monkeypatching cannot safely support mixed concurrent requests. Renaming the parameter to “record intent” preserves a misleading control.
- **Compatibility / migration / rollback:** Preserve the query parameter and make behavior match it. Rollback is global no-LLM enforcement for all jobs.
- **Dependencies / required tests:** Execution policy schema; mixed concurrent jobs, all LLM nodes, fallback, and zero-call assertions against a fake client.
- **Duplicate relationships / primary verification:** None known. Primary reproduction pending.

### D2-AGT-007 — Execution path is inferred from finding cardinality

- **Severity / status:** P2 / `candidate`
- **Classification / owner:** observability / Orchestration
- **Source:** `agents/src/orchestration/nodes/synthesizer.py::synthesizer`
- **Invariant:** The reported route must reflect graph routing and attempted tools, independent of whether tools find issues.
- **Evidence:** `path_taken` is `deep` only when `rag_results` or `static_findings` is non-empty. A deep run with clean results, explicit failures, or only graph/formal output is labeled `fast`.
- **Impact / affected modules:** Route metrics, audit explanations, failure diagnosis, and evaluation stratification are wrong.
- **Recommendation:** Set an explicit route enum and selected/attempted node list at `evidence_router`; persist it unchanged.
- **Rejected alternatives:** Inferring from routing strings remains fragile. Inferring from tool status fails while status coverage is incomplete.
- **Compatibility / migration / rollback:** Add route fields and preserve legacy `path_taken` as a derived alias. Rollback affects observability only.
- **Dependencies / required tests:** Route schema; deep-clean, deep-failed, formal-only, screen-escalated, and fast tests.
- **Duplicate relationships / primary verification:** None known. Primary reproduction pending.

### D2-AGT-008 — Graph inspector defaults to the gateway, not ML service

- **Severity / status:** P1 / `candidate`
- **Classification / owner:** configuration, service integration / Graph inspector MCP
- **Source:** `agents/src/mcp/servers/graph_inspector_server.py::_ML_API_URL`, `::_analyze_hotspots_gnn`
- **Invariant:** Default internal service routing must target the documented owner of an endpoint.
- **Evidence:** `_ML_API_URL` defaults to `http://localhost:8000`, while the ML API and gateway health map place ML on 8001. The inspector requests `/hotspots`, which the gateway does not expose.
- **Impact / affected modules:** Default deployments silently miss GNN hotspots and fall back to weaker analysis/mock behavior; visual attribution is degraded.
- **Recommendation:** Use one typed service map, startup validation, and fail-visible backend status. Default to the ML service URL used elsewhere.
- **Rejected alternatives:** Requiring an undocumented environment override leaves default deployment broken. Silent fallback hides configuration drift.
- **Compatibility / migration / rollback:** Environment overrides remain valid. Rollback can disable the GNN backend explicitly rather than misroute it.
- **Dependencies / required tests:** Service configuration schema; default URL contract and startup endpoint capability tests.
- **Duplicate relationships / primary verification:** May overlap ML service interface inventory. Primary reproduction pending.

### D2-AGT-009 — Fusion counts derived consensus as an independent witness

- **Severity / status:** P1 / `candidate`
- **Classification / owner:** scientific correctness / Verdict fusion
- **Source:** `agents/src/orchestration/nodes/consensus_engine.py::consensus_engine`; `agents/src/orchestration/verdict/emit.py::emit_consensus_evidence`; `agents/src/orchestration/verdict/fuse.py::FAMILIES`, `::_fuse_for_evidence`
- **Invariant:** A derived aggregate must not add independent evidentiary mass on top of the observations from which it was computed.
- **Evidence:** The node emits ML and static evidence, then emits `source="consensus"` derived from the same ML/static signals with reliability 0.85. `FAMILIES` has no consensus mapping, so fusion treats it as a separate family.
- **Impact / affected modules:** Confidence and verdict bands double-count correlated observations; stated de-correlation guarantees do not hold.
- **Recommendation:** Make consensus a diagnostic view only, or replace its inputs rather than append to them. Represent provenance dependencies explicitly and calibrate on held-out data.
- **Rejected alternatives:** Adding consensus to the ML or static family arbitrarily still loses its mixed dependency graph. Reducing 0.85 is not principled de-correlation.
- **Compatibility / migration / rollback:** Keep legacy consensus display while excluding it from canonical fusion. Rollback is the current display-only diagnostic path.
- **Dependencies / required tests:** Evidence provenance graph and calibration; monotonic no-double-count, duplicate input, and held-out reliability tests.
- **Duplicate relationships / primary verification:** Related to any ML calibration finding. Primary reproduction pending.

### D2-AGT-010 — “Provable” means emitter-asserted deterministic

- **Severity / status:** P1 / `candidate`
- **Classification / owner:** trust model, naming correctness / Evidence and cross-system architecture
- **Source:** `agents/src/orchestration/verdict/evidence.py::Evidence`; `agents/src/orchestration/verdict/fuse.py::fuse`; evidence constructors
- **Invariant:** A field named `verdict_provable` must be bound to verifiable inputs, code/artifact identity, and a proof statement.
- **Evidence:** The tier filters only on an emitter-provided boolean. ML, static, RAG, consensus, and formal constructors can set `deterministic=True`; no execution manifest, input hash, tool version, artifact hash, or proof reference is required. RAG is marked deterministic even though deterministic mode skips it as non-deterministic.
- **Impact / affected modules:** Consumers can mistake local deterministic assertions for cryptographically proved evidence and commit an overbroad verdict on-chain.
- **Recommendation:** Rename the current tier to `deterministic_claim` until a canonical manifest binds source, configuration, artifacts, tool outputs, and proof scope. Only proved statements enter a proof tier.
- **Rejected alternatives:** Documentation caveats cannot repair a misleading machine field. Hashing only the final label omits execution identity.
- **Compatibility / migration / rollback:** Add explicit evidence assurance level and keep old name as deprecated alias. Rollback should remove “provable” claims, not widen them.
- **Dependencies / required tests:** V3 execution manifest and ZK truth boundary; cross-language canonical vectors and negative binding tests.
- **Duplicate relationships / primary verification:** Likely duplicate/parent-child with ZK/contracts proof-binding findings; deduplicate in the unified registry. Primary verification pending.

### D2-AGT-011 — Public service surfaces lack an authentication boundary

- **Severity / status:** P1 / `candidate`
- **Classification / owner:** security architecture / Gateway and MCP deployment
- **Source:** `agents/src/api/gateway.py::create_app`, `::run`; MCP Starlette application/`run_server` functions
- **Invariant:** Public or operator-capable services must authenticate and authorize callers before accepting work or key-bearing operations.
- **Evidence:** Gateway and all five MCP services bind `0.0.0.0`; no authentication/authorization middleware or route dependency is visible. Audit MCP exposes `submit_audit`, which can use the configured operator key. This is source review only; no network probing was performed.
- **Impact / affected modules:** Deployment without an external trusted proxy exposes audit workload, source persistence, resource use, chain reads, and potentially signed submissions.
- **Recommendation:** Bind internal MCP services to a private interface, require mutually authenticated service identity and per-tool authorization, and put rate/size policy at the public gateway. Separate read and submit capabilities.
- **Rejected alternatives:** Assuming firewall protection is not an application trust boundary unless deployment policy enforces and tests it. A shared static bearer token is insufficient for independent operators.
- **Compatibility / migration / rollback:** Stage auth in observe/enforce modes with explicit health exemptions. Rollback is private-loopback binding with submit disabled.
- **Dependencies / required tests:** Deployment manifests, identity/PKI, secret management; unauthenticated/unauthorized, capability, rotation, replay, and audit-log tests.
- **Duplicate relationships / primary verification:** Path-write impact is owned by `D2-AGT-002`; this finding covers the general service boundary. Primary review pending.

### D2-AGT-012 — Chain initialization failure becomes realistic mock history

- **Severity / status:** P1 / `candidate`
- **Classification / owner:** correctness, truth boundary / Audit MCP
- **Source:** `agents/src/mcp/servers/audit/_lifecycle.py::_on_startup`; `_handlers.py` read handlers; `_decode.py::_mock_audit_result`, `::_mock_history`; `agents/src/orchestration/nodes/audit_check.py::audit_check`
- **Invariant:** Failure to initialize a real chain client must remain unavailable, not become apparently verified chain history.
- **Evidence:** Startup catches every ABI/RPC initialization exception and sets `_MOCK_MODE=True`. Read handlers then return realistic records, including `verified=True`, without a mock/degraded marker. `audit_check` treats only an `error` key as failure and accepts the records.
- **Impact / affected modules:** Reports and narratives can cite fabricated prior audits after production chain failure, contaminating feedback and operator decisions.
- **Recommendation:** Separate explicit test mock configuration from production failure state. Return typed unavailable status and propagate chain ID, block, registry, and data-source assurance.
- **Rejected alternatives:** Logging “mock mode” is invisible to remote consumers. Changing fake addresses or timestamps does not make the response safe.
- **Compatibility / migration / rollback:** Add source/status fields; fail closed in non-test profiles. Rollback is to disable chain history, not auto-mock it.
- **Dependencies / required tests:** Audit result schema; ABI missing, wrong chain, RPC outage, explicit mock, and report propagation tests.
- **Duplicate relationships / primary verification:** Same failure-mode class as `D2-AGT-001`, but a distinct external source and owner. Primary reproduction pending.

### D2-AGT-013 — Proof workspace and transaction nonce are shared

- **Severity / status:** P1 / `candidate`
- **Classification / owner:** concurrency, transaction integrity / Audit submission MCP
- **Source:** `agents/src/mcp/servers/audit/_submit.py::_run_submit`
- **Invariant:** Concurrent submissions must not share mutable proof artifacts or allocate the same account nonce.
- **Evidence:** Every call writes canonical `zkml/ezkl/proof_input.json`, `witness.json`, and `proof.json` without a per-job directory or lock. Nonce uses `get_transaction_count(operator_address)` without explicit pending semantics or a per-key allocator. Source review only; no live proof or chain probe was performed.
- **Impact / affected modules:** Concurrent jobs can cross-read/overwrite proof data, delete another call's files, or replace transactions through nonce collision.
- **Recommendation:** Use content-addressed per-job proof workspaces with atomic publication and cleanup ownership. Serialize nonce allocation per key or use a durable transaction manager with pending reconciliation.
- **Rejected alternatives:** A single global lock prevents corruption but eliminates distributed throughput and does not solve crash recovery. Random filenames without manifest binding do not guarantee ownership.
- **Compatibility / migration / rollback:** Submission result schema can remain stable. Rollback is single-worker submit mode with explicit queueing.
- **Dependencies / required tests:** ZK artifact manager, job identity, transaction manager; concurrent proof, crash cleanup, nonce replacement, restart, and idempotent resubmit tests.
- **Duplicate relationships / primary verification:** Likely overlaps ZK/contracts shared-proof and nonce findings; merge under the cross-system owner. Primary reproduction pending.

### D2-AGT-014 — Reliability weights depend on launch directory

- **Severity / status:** P1 / `track-reproduced`
- **Classification / owner:** configuration, reproducibility / Verdict reliability
- **Source:** `agents/src/orchestration/verdict/reliability.py::L3_RELIABILITY_PATH`, `::_load_l3_table`, `::load_reliability`
- **Invariant:** Identical source, input, and declared configuration must produce identical reliability weights regardless of current working directory.
- **Evidence:** From `agents/`, the tracked L3 file loads and returns Reentrancy ML/Slither values `0.127/0.343`; from repository root, `configs/reliability_v1.yaml` is not found and L1 fallback returns `0.39/0.82`. Full suite observed the Slither mismatch.
- **Impact / affected modules:** Fusion verdicts and confidence change with launch directory. The fallback is not surfaced in report/tool status, defeating reproducibility.
- **Recommendation:** Resolve default paths relative to the package or require an absolute config path; validate table bounds/schema; record config hash and fallback source in every execution manifest.
- **Rejected alternatives:** Mandating a working directory in prose remains fragile across systemd, containers, tests, and operator deployments.
- **Compatibility / migration / rollback:** Keep environment override precedence; make default resolution deterministic. Rollback can require explicit absolute configuration and refuse implicit fallback.
- **Dependencies / required tests:** Config manifest; launch from root, agents, arbitrary directory, malformed/missing/versioned file, and value-bound tests.
- **Duplicate relationships / primary verification:** None known. Track reproduction complete; primary verification pending.

### D2-AGT-015 — Parallel tool errors crash the graph reducer

- **Severity / status:** P1 / `track-reproduced`
- **Classification / owner:** orchestration correctness, fail-soft behavior / AuditState
- **Source:** `agents/src/orchestration/state.py::AuditState.error`; `agents/src/orchestration/nodes/rag_research.py::rag_research`; `agents/src/orchestration/nodes/static_analysis.py::static_analysis`; graph deep fan-out
- **Invariant:** Concurrent fail-soft tool errors must be accumulated without turning into a graph-level reducer failure.
- **Evidence:** `error` has default replacement semantics. A minimal two-node fan-out over `AuditState`, with both branches returning an error, deterministically raised LangGraph `InvalidUpdateError` stating the key can receive only one value per step.
- **Impact / affected modules:** Simultaneous RAG/static failures can abort the whole audit instead of producing a degraded report, contradicting node-level fail-soft design.
- **Recommendation:** Replace the scalar with structured append-only error events keyed by node/attempt; derive a summary for legacy reports. Never use last-writer-wins for parallel diagnostics.
- **Rejected alternatives:** Catching the graph exception in gateway only marks the job failed and loses both causes. Serializing the fan-out sacrifices latency without solving future parallel writers.
- **Compatibility / migration / rollback:** Keep a derived `error` string for clients. Rollback can force deterministic serial execution as a temporary mitigation.
- **Dependencies / required tests:** Error-event schema; two/three simultaneous failures, duplicate retries, successful sibling, and gateway degraded-report tests.
- **Duplicate relationships / primary verification:** None known. Track reproduction complete; primary verification pending.

### D2-AGT-016 — Rule 5C tool status is incomplete and omitted from report

- **Severity / status:** P1 / `candidate`
- **Classification / owner:** observability, failure semantics / Orchestration nodes and report schema
- **Source:** `AuditState.tool_status`; `rag_research`, `audit_check`, `graph_explain`, `formal_verification`, `synthesizer`
- **Invariant:** Every selected external tool must emit attempted/ran/status/reason/identity, and the canonical report must preserve it.
- **Evidence:** ML, quick/static tools, and Halmos write status. RAG and audit history use result-or-error without tool status. `graph_explain` returns empty structures on exceptions without error/status. `synthesizer.final_report` excludes the state status map.
- **Impact / affected modules:** Clean-empty, route-skipped, dependency-missing, timed-out, and failed results remain indistinguishable for several tools; coverage and scientific metrics are unreliable.
- **Recommendation:** Define a mandatory tool-execution record and initialize it for every selected node. Validate completeness before completion and include it in the canonical result envelope.
- **Rejected alternatives:** Inferring status from empty lists recreates the ambiguity Rule 5C was designed to remove. Logs are not durable job evidence.
- **Compatibility / migration / rollback:** Additive report field; legacy clients can ignore it. Rollback should mark the whole report degraded when completeness cannot be established.
- **Dependencies / required tests:** Execution manifest and gateway persistence; clean-empty, skipped, unavailable, timeout, malformed, retry, and successful status tests for every tool.
- **Duplicate relationships / primary verification:** `D2-AGT-004` covers later gateway stripping; do not merge because either defect independently loses observability. Primary reproduction pending.

### D2-AGT-017 — Gateway has no real capacity bound or backpressure

- **Severity / status:** P1 / `candidate`
- **Classification / owner:** availability, concurrency / Gateway
- **Source:** `agents/src/api/gateway.py::submit_audit`, `::create_app`
- **Invariant:** A public gateway must bound accepted work to available worker, GPU, LLM, static-tool, and proof capacity.
- **Evidence:** OpenAPI advertises 503 “All graph slots busy,” but no semaphore, queue bound, worker pool, or rejection path exists. Every request immediately creates an `asyncio` task. Tests assert distinct concurrent IDs only.
- **Impact / affected modules:** Load can exhaust memory, saturate external services, increase timeouts, and amplify shared-file and nonce races. Accepted jobs have no scheduling fairness or priority.
- **Recommendation:** Persist first, then claim through a bounded worker pool with per-resource concurrency classes, queue limits, deadlines, cancellation, and observable admission decisions.
- **Rejected alternatives:** A single global semaphore lacks durable queue/recovery and resource-specific limits. Relying on the overall timeout does not bound accepted concurrency.
- **Compatibility / migration / rollback:** Continue returning 202 for accepted durable work; return documented 429/503 only before acceptance. Rollback is one-worker mode with a finite queue.
- **Dependencies / required tests:** Durable worker/lease architecture; burst, fairness, saturation, cancellation, restart, queue-full, and per-resource limit tests.
- **Duplicate relationships / primary verification:** Connects to `D2-AGT-005` and `D2-AGT-013`, but has a distinct admission-control root cause. Primary reproduction pending.

### D2-AGT-018 — Fresh clone cannot satisfy advertised AGENTS baseline

- **Severity / status:** P1 / `track-reproduced`
- **Classification / owner:** packaging, deployability, testability / AGENTS module
- **Source:** `agents/tests/test_rag_fetchers.py`, `test_smoke_e2e.py`, `test_static_analysis_real_slither.py`; seed corpus paths and tool resolution
- **Invariant:** Tracked dependencies and documented setup must reproduce the declared test baseline from a clean worktree.
- **Evidence:** Full suite failed six RAG tests because five seed corpora were absent and SWC IDs empty; three tests failed because `solc` was not on the test process PATH. 622 passed, 11 failed, 1 skipped. The skipped benchmark is not coverage.
- **Impact / affected modules:** Operators cannot reproduce RAG or real static-analysis behavior from the repository. Local ignored artifacts can mask missing deployment inputs.
- **Recommendation:** Publish versioned content-addressed artifact acquisition with license/provenance, add fail-fast preflight for binaries, and split hermetic unit, artifact integration, and live suites with explicit gates.
- **Rejected alternatives:** Committing opaque large local artifacts without provenance is not reproducible supply-chain management. Skipping absent prerequisites hides deployability failures.
- **Compatibility / migration / rollback:** No wire impact. Rollback is to declare the affected features unavailable and prevent production startup.
- **Dependencies / required tests:** Artifact manifest, checksum/signature verification, solc/toolchain installer; fresh clone, offline cache, checksum failure, missing binary, and supported-platform tests.
- **Duplicate relationships / primary verification:** May link to DATA/RAG artifact findings. Track reproduction complete; primary verification pending.

### D2-AGT-019 — Report bridge is non-atomic and last-writer-wins

- **Severity / status:** P2 / `candidate`
- **Classification / owner:** persistence, concurrency / Synthesizer, visualizer, feedback loop
- **Source:** `agents/src/orchestration/nodes/synthesizer.py::synthesizer`; `visualizer.py::visualizer`; `agents/src/ingestion/feedback_loop.py::_read_vuln_type_from_report`
- **Invariant:** A feedback consumer must read a complete report tied to the exact audit event/job.
- **Evidence:** JSON and HTML use direct `write_text` to address-derived shared names. Repeated/concurrent audits for one address overwrite each other. Feedback reads only by address and treats partial/missing/invalid JSON as `unknown`; no job/tx/proof identity binds the file to the event.
- **Impact / affected modules:** Feedback can ingest the wrong vulnerability class, lose report history, or observe a partial file. Last-writer-wins breaks provenance.
- **Recommendation:** Store immutable content-addressed report objects, atomically publish a versioned index, and bind feedback events to a report/proof commitment.
- **Rejected alternatives:** Atomic replacement alone prevents partial reads but not wrong-version association or history loss.
- **Compatibility / migration / rollback:** Preserve address lookup as a derived latest pointer. Rollback is disabling automated feedback ingestion when no committed report matches.
- **Dependencies / required tests:** Canonical report identity and event binding; concurrent same-address, crash mid-write, stale/latest, missing commitment, and replay tests.
- **Duplicate relationships / primary verification:** Filesystem containment is owned by `D2-AGT-002`; this finding covers legitimate concurrent writers. Primary reproduction pending.

### D2-AGT-020 — Feedback corpus overstates what the ZK proof guarantees

- **Severity / status:** P1 / `candidate`
- **Classification / owner:** scientific correctness, truth boundary / Feedback ingestion and ZK architecture
- **Source:** `agents/src/ingestion/feedback_loop.py::FeedbackIngester.process_event`
- **Invariant:** Generated knowledge must describe exactly the proved statement and no more.
- **Evidence:** Feedback text says “the model computation is cryptographically guaranteed to be honest.” The current proof covers proxy inference over supplied inputs, not teacher execution, source/address identity, retrieval, static tools, or full AGENTS fusion.
- **Impact / affected modules:** An overclaim is recursively embedded into RAG, where later audits may cite it as precedent. This compounds model/proof semantic error across runs.
- **Recommendation:** Generate text from a versioned proof-statement descriptor; explicitly name proxy circuit, input commitment, outputs, verifier, and unproved assumptions.
- **Rejected alternatives:** A generic disclaimer elsewhere does not remove false corpus content. Calling the proxy “the model” obscures the teacher/proxy distinction.
- **Compatibility / migration / rollback:** Rebuild or tombstone affected corpus chunks after correcting semantics. Rollback is to omit proof claims from feedback documents.
- **Dependencies / required tests:** V3 proof envelope/truth boundary; golden text for each proof version and negative tests for unbound teacher/source identity.
- **Duplicate relationships / primary verification:** Likely merge with the ZK/contracts proof-semantics finding; retain this record as the AGENTS downstream-consumer manifestation. Primary verification pending.

### D2-AGT-021 — Test mutates the cached production config singleton

- **Severity / status:** P2 / `track-reproduced`
- **Classification / owner:** test isolation / AGENTS tests and config
- **Source:** `agents/tests/test_consensus_voting.py::test_ml_weight_scaled_down`; `agents/src/config/loader.py::_CONFIG`, `::reload_config`
- **Invariant:** A test must restore process-global configuration before subsequent tests execute.
- **Evidence:** The test calls `reload_config()`, mutates `cfg.consensus.ml_weight_scale = 1.0`, and does not restore it. Later full-suite reliability test reads the mutated singleton while reading expected scale 0.5 from YAML, contributing to the observed 1.56 back-computed value.
- **Impact / affected modules:** Suite results depend on order, obscure the separate L3 path defect, and reduce confidence in regression evidence.
- **Recommendation:** Use immutable config models or a fixture that snapshots/resets the singleton and environment after every test. Avoid mutating objects returned by production singleton access.
- **Rejected alternatives:** Ordering the reliability test earlier hides contamination without fixing isolation.
- **Compatibility / migration / rollback:** Test-only behavior; production immutability is additive hardening. Rollback is process-isolated test execution.
- **Dependencies / required tests:** Config fixture; random-order, repeated-suite, parallel worker, and mutation-rejection tests.
- **Duplicate relationships / primary verification:** Distinct from production CWD defect `D2-AGT-014`. Track reproduction complete; primary verification pending.

## 7. Cross-system implications

The current system cannot safely use the AGENTS result as a decentralized quorum payload. An operator can reach a completed result through different launch directories, fallback modes, missing artifacts, RAG score semantics, and LLM policy. The gateway then removes evidence needed to detect those differences. A quorum over only the final labels would therefore hide execution divergence rather than solve it.

Before V3 quorum, each operator needs the same canonical job identity and execution manifest, including:

- Source bytes/hash, target address/runtime code hash, chain ID, block reference, and request schema version.
- DATA/ML/proxy/circuit/verifier/config/index/tool version and content hashes.
- Selected route, tool-execution records, normalized evidence schema, and error events.
- Deterministic commitment separated from LLM/RAG narrative and other advisory output.
- Job attempt, lease owner, per-job workspace, nonce allocation, report commitment, and timestamps.

The deterministic commitment must exclude LLM narrative and raw RAG ordering unless those inputs and algorithms are themselves canonicalized. EZKL proof scope must remain explicitly narrower than the AGENTS audit conclusion.

## 8. Required implementation sequence

1. **Trust/correctness blockers:** `D2-AGT-001`, `002`, `003`, `004`, `010`, `012`, `020` and their cross-track parents.
2. **Execution semantics:** Canonical result/tool-status/error/evidence schemas; fix parallel reducers, route identity, reliability config, and consensus provenance.
3. **Durable jobs:** Bounded admission, worker leases/heartbeats/retries, checkpoint identity, per-job report/proof workspaces, and transaction management.
4. **Service boundary:** Private/authenticated MCP transport, capability separation, startup preflight, and canonical service configuration.
5. **Reproducibility:** Fresh-clone artifacts/toolchains, immutable config, production-shape RAG tests, and execution manifests.
6. **Measured quality/performance:** RAG calibration, fusion reliability, LLM latency/cost, static-tool latency, job recovery, concurrency, and end-to-end operator equivalence.

## 9. Track acceptance matrix

| Requirement | State | Evidence/blocker |
|---|---|---|
| Orchestration state/reducers/routing | Audited | Architecture inventory; `D2-AGT-007`, `009`, `015` |
| Evidence/fusion and dual verdicts | Audited, blocked | `D2-AGT-003`, `004`, `009`, `010`, `014` |
| Gateway/jobs/recovery | Audited, blocked | 57 focused tests; `D2-AGT-004`, `005`, `006`, `017` |
| MCP transport/auth/config | Source-audited, live probes absent | Interface table; `D2-AGT-008`, `011`, `012` |
| RAG/feedback | Audited, blocked | Production schema/score trace; `D2-AGT-003`, `019`, `020` |
| LLM isolation | Audited, blocked | `D2-AGT-006`; no live LLM measurement |
| Persistence/concurrency | Audited, blocked | Persistent-state table; `D2-AGT-002`, `005`, `013`, `015`, `017`, `019` |
| Observability | Audited, blocked | `D2-AGT-004`, `007`, `016` |
| Full AGENTS suite | Executed, failing | 622 passed, 11 failed, 1 skipped |
| Performance/VRAM/live services | Not established | No suitable live-service/GPU/proof environment in this track |
| Runtime unchanged | Satisfied | Baseline diff empty for `agents`; appendix-only repository change |

## 10. Final disposition

The AGENTS/services source track is integrated but is not implementation-authorizing evidence by itself. The primary audit subsequently adjudicated the candidate P1s, merged exact duplicates, reconciled all 21 rows into the unified registry, and recorded missing live-service, performance, GPU, proof, and multi-worker evidence as explicit blockers rather than skips.

Ali approved D2 on 2026-07-14 with mandatory acceptance-matrix and before/after evidence. This appendix authorizes R0 planning only; no runtime fix is authorized without that reviewed plan.
