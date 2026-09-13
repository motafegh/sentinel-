# Case Study 06 — Tool silence is not a clean result

## 1. Decision in one sentence

SENTINEL represents unavailable, failed, degraded, and successful-with-zero-findings tool executions as different states so missing evidence cannot silently become a benign security conclusion.

## 2. Problem

A multi-tool audit pipeline routinely encounters timeouts, missing analyzers, unavailable services, disabled LLM paths, and partial execution.

The dangerous implementation shortcut is to normalize all of these outcomes into an empty findings list. Downstream fusion can then mistake “the tool never produced evidence” for “the tool ran successfully and found nothing.”

For security analysis, those states have opposite epistemic meaning.

## 3. Evidence

The current 14-node LangGraph carries explicit accountability state, including `tool_status` and node errors, alongside the evidence itself.

The orchestration contract requires a tool result to distinguish:

- `ran=true, findings=[]`;
- `ran=false, reason=...`.

The current handbook also records that node failures may produce a partial report, that deterministic/LLM-disabled modes change available evidence, and that a reader must inspect tool execution status before interpreting empty output.

## 4. Shortcut rejected

The project rejects “empty means clean.”

It also rejects silent fallback behavior where a missing dependency, timeout, analyzer failure, or disabled evidence source is converted into a normal zero-finding result merely to simplify downstream types.

## 5. Decision

Execution status is part of the evidence model.

Each tool boundary must make unavailable or degraded execution visible to downstream routing, fusion, reporting, and review. Deterministic evidence and nondeterministic evidence remain distinguishable, and explanatory layers do not become independent verification simply because they produce text.

## 6. Implementation and validation

`AuditState` carries `tool_status`, evidence collections, errors, and the final report as separate fields. Per-tool status is merged independently from findings.

The graph keeps evidence acquisition, consensus/fusion, and explanation as separate responsibilities. A failure can therefore yield a bounded partial report rather than silently fabricating clean evidence.

The same rule applies across external services: missing ML, RAG, analyzer, registry, or persistence capability must surface as execution state rather than disappear.

## 7. Result

The final report can distinguish three materially different claims:

1. a tool ran and produced findings;
2. a tool ran successfully and produced no findings;
3. the tool did not establish evidence because execution was unavailable, failed, or degraded.

That distinction makes downstream confidence and human review more honest.

## 8. Remaining limitation

Explicit status does not make a failed tool succeed, and it does not prove that the remaining evidence is sufficient for a vulnerability verdict.

The pipeline can still produce partial results, and external runtime dependencies remain real operational prerequisites. The value of the design is that those limitations stay visible.

## 9. Evidence trail

Primary references:

- `agents/src/orchestration/state.py`;
- `agents/src/orchestration/graph.py` and orchestration nodes;
- `docs/handbook/09_agents_orchestration.md`;
- `agents/README.md`.

Source and tests remain the authority for exact runtime behavior.
