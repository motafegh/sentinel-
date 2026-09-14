# Agents / Orchestration Capability Completion Plan

**Status:** ACTIVE — source audit and bounded repairs may proceed in parallel with DATA/ML  
**Parent:** `00_MASTER_TECHNICAL_COMPLETION_ROADMAP.md`  

## 1. Objective

Bring the agentic audit runtime to an evidence-backed capability boundary: every consequential node/tool must either work on the classes of contracts it claims to support, fail/degrade explicitly, or be narrowed honestly. The goal is not to add more nodes; it is to make the existing orchestration technically dependable and semantically truthful.

## 2. Current source baseline

Substantive components already exist under `agents/src/`, including:

- API gateway and job stores;
- LangGraph orchestration and state reducers;
- ML assessment;
- quick screening;
- Slither/Aderyn static analysis;
- RAG research;
- graph explanation;
- Halmos formal verification;
- evidence routing, consensus, cross-validation, synthesis, reflection and explanation;
- verdict/evidence/reliability fusion;
- MCP servers for inference, representation, graph inspection and audit reads/submission preparation;
- evaluation/benchmark/gating utilities;
- ingestion/feedback loops;
- persistence/report writing;
- LLM and prompt/injection security controls;
- explicit execution-status/provenance contracts.

A wired node is not automatically a complete capability. Source-level behavior and real-contract evidence control the claim.

## 3. Primary source owners to audit

At minimum:

- `agents/src/orchestration/graph.py`
- `agents/src/orchestration/state.py`
- `agents/src/orchestration/routing.py`
- `agents/src/orchestration/nodes/`
- `agents/src/orchestration/verdict/`
- `agents/src/contracts/execution.py`
- `agents/src/contracts/submission_v3.py`
- `agents/src/eval/`
- `agents/src/mcp/servers/`
- `agents/src/rag/`
- `agents/src/ingestion/`
- `agents/src/security/`
- `agents/src/persistence/`
- gateway/job-store sources and relevant tests.

## 4. Work package A0 — topology and capability audit

**State:** `AUDITING`

For every graph node and external dependency, record:

- entry condition;
- input state fields;
- external dependency/tool;
- success output;
- clean-empty output;
- degraded/unavailable/failed behavior;
- provenance/evidence eligibility;
- routing consequence;
- timeout/retry behavior;
- tests;
- real-contract evidence;
- advertised versus actual capability.

Produce a compact capability matrix rather than relying on diagram labels.

**Exit:** no consequential node remains “assumed working” solely because it exists in the graph.

## 5. Work package A1 — Halmos/formal-verification repair

**State:** `INVESTIGATING`

Current source weakness to resolve:

- the node discovers a contract name but the generated harness assumes a `Target`-style interface with methods such as `deposit()` and `withdraw()`;
- arbitrary contracts therefore commonly fail compilation or exercise irrelevant properties;
- fail-soft behavior keeps the graph alive, but that is not equivalent to generic formal verification.

Investigation must decide between evidence-backed options such as:

1. **contract-aware property generation** from ABI/AST/Slither structure for a bounded set of invariant families;
2. **template dispatch** only when the contract exposes a compatible shape;
3. **explicitly scoped formal-verification capability** that runs only on supported patterns and reports `SKIPPED_POLICY`/`UNAVAILABLE` elsewhere;
4. a hybrid of the above.

Do not generate plausible-looking but semantically meaningless assertions merely to make Halmos run.

Required validation:

- supported vulnerable contracts where a counterexample should be found;
- supported safe/clean examples where the property should hold;
- unsupported interfaces produce explicit non-evidence status;
- compile errors/timeouts are distinguishable from proven/clean results;
- emitted formal evidence accurately reflects what was actually checked.

## 6. Work package A2 — static-analysis execution truth

Audit Slither and Aderyn paths for:

- detector registration and scoping correctness;
- full-scan fallback behavior;
- ExternalBug/inter-contract-call extraction;
- tool version/runtime assumptions;
- temp-file/solc behavior;
- timeout and missing-binary handling;
- mapping from tool findings to canonical vulnerability classes;
- whether a clean result is emitted only when the tool genuinely ran successfully.

Use real vulnerable and non-vulnerable contracts, not only mocked detector outputs.

## 7. Work package A3 — ML routing and provenance

Validate end to end:

```text
inference result
→ execution-status validation
→ eligible ML result
→ routing thresholds/classes
→ static/RAG/graph/formal branches
→ evidence fusion
```

Required checks:

- mock/degraded/unavailable ML cannot become evidence;
- changed repaired-model class support is respected after R4 promotion;
- routing does not hard-code stale Run12 assumptions;
- thresholds used for routing are versioned policy and not mistaken for calibrated outcome thresholds;
- ML absence triggers safe fallback behavior rather than “safe contract” semantics.

## 8. Work package A4 — RAG / LLM evidence boundary

Audit:

- corpus/index provenance and freshness;
- retrieval metadata;
- source attribution;
- prompt-injection/sanitization boundaries;
- whether retrieved text can override system/security policy;
- LLM failure/timeout behavior;
- distinction between retrieved precedent/advice and deterministic/static evidence;
- synthesis language when RAG is absent or low-quality.

Add adversarial fixtures for malicious Solidity comments, malicious retrieved text and conflicting evidence.

## 9. Work package A5 — evidence fusion, consensus and synthesis

Test the decision path using controlled scenarios:

- ML positive + static corroboration;
- ML positive + static disagreement;
- static finding with ML unavailable;
- RAG-only weak context;
- formal counterexample;
- formal tool unavailable;
- all deterministic tools clean;
- one or more tools fail;
- conflicting high-strength sources;
- insufficient evidence/abstention.

Verify:

- reliability weights have evidence/rationale;
- unavailable is never treated as clean;
- dependent tools are not falsely counted as independent corroboration;
- confidence/verdict language matches evidence strength;
- synthesis preserves uncertainty rather than forcing a vulnerability/safe binary.

## 10. Work package A6 — gateway, persistence and job lifecycle

Audit:

- request validation/auth boundaries;
- job-store persistence/recovery;
- duplicate/idempotent submissions;
- cancellation/timeouts;
- restart behavior;
- report-write atomicity;
- state/result versioning;
- SQLite/in-memory parity where intended;
- sensitive data/logging behavior.

The API should never report job completion if required result persistence failed.

## 11. Work package A7 — MCP boundaries

Validate that each MCP server’s actual authority matches its public/tool contract.

Especially:

- inference MCP returns bound live/mock/failure status correctly;
- representation/graph-inspector tools cannot imply stronger source/model truth than they possess;
- audit MCP remains read-only where intended;
- submission preparation cannot be confused with signing, broadcasting or finality;
- external tool failures are structured and consumed downstream.

## 12. Work package A8 — real-contract integration/evaluation corpus

Build or freeze a bounded integration corpus containing:

- simple canonical vulnerabilities;
- multi-contract/inheritance/library cases;
- long contracts;
- contracts that trigger unavailable/degraded tool paths;
- contracts with conflicting detector evidence;
- contracts suitable for formal properties;
- contracts intentionally unsupported by formal properties;
- adversarial prompt/RAG content.

Run full graph executions and inspect both final report and intermediate tool status/evidence.

Metrics should include more than vulnerability accuracy, for example:

- tool execution completeness;
- false-clean rate under dependency failure;
- abstention correctness;
- provenance completeness;
- routing correctness;
- report/evidence consistency;
- latency/resource budgets where relevant.

## 13. Work package A9 — repaired-model migration

After inference promotion:

- update any class/routing assumptions;
- verify model-hash/provenance handling;
- rerun the real-contract integration corpus;
- compare graph-path changes caused by the new model;
- confirm restricted/unsupported classes do not receive overstated agent verdicts.

## 14. Stop/fail conditions

Stop and investigate if:

- a tool failure can still appear as clean evidence;
- formal verification emits “proven” evidence for a property not meaningfully related to the contract;
- mock output enters production evidence fusion;
- routing depends on stale or unbound thresholds;
- final synthesis claims certainty stronger than its evidence;
- MCP submission preparation is represented as signing/broadcast/finality;
- real-contract tests materially contradict architectural claims.

## 15. Completion criteria

Agents/orchestration is `ACCEPTED` when consequential paths have source-level contracts plus real-contract evidence, unsupported paths are explicit, provenance survives through final synthesis, and repaired-model integration can occur without reintroducing stale assumptions. Adding more agents/nodes is not a completion criterion.
