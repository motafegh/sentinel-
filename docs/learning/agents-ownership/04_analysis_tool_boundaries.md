# Agents Module Ownership — 04: Analysis-Tool Boundaries

## Ownership Target

Know the responsibility, input/output contract, and degraded behavior of every deep-path analysis channel without learning each tool's internal implementation.

## Source Reading Order

1. `agents/src/orchestration/nodes/static_analysis.py`
2. `agents/src/orchestration/nodes/_helpers.py` — Aderyn execution and result normalization
3. `agents/src/orchestration/nodes/rag_research.py`
4. `agents/src/orchestration/nodes/graph_explain.py`
5. `agents/src/orchestration/nodes/formal_verification.py`
6. `agents/src/orchestration/nodes/audit_check.py`
7. `agents/src/orchestration/verdict/emit.py`

## Items to Own

- Which channels run on every request versus deep-path requests only.
- The output state fields and evidence emitted by static analysis, RAG, graph explanation, formal verification, and audit history.
- Why static findings and quick-screen hits have different purposes.
- How `ExternalBug` gains external-call context.
- The meaning and limits of retrieved evidence, model hotspots, and formal evidence.
- How missing binaries, unavailable services, malformed results, and empty inputs are surfaced through `tool_status` or structured state.
- Why a tool failure is not a clean result under Rule 5C.

## Boundary Matrix Exercise

Create a matrix with one row per channel: trigger, external dependency, returned state, evidence source, and explicit degraded status.

## Verification

```bash
cd agents
TMP=/tmp TEMP=/tmp TMPDIR=/tmp poetry run pytest \
  tests/test_static_analysis_real_slither.py \
  tests/test_static_analysis_real_aderyn.py \
  tests/test_rag_query.py \
  tests/test_formal_verification.py -q
```

## Completion Check

- Which tool result can escalate an ML-safe contract into deep analysis?
- Which fields distinguish a failed tool from a clean tool run?
- Which source file converts raw tool results into `Evidence`?
- Why is RAG support not proof that a specific contract has a vulnerability?

## Intentionally Out of Scope

- Detector algorithms, vector-index internals, and symbolic-execution internals.
- Evidence-fusion mathematics.
- LLM prompts, gateway jobs, and ZK submission.
