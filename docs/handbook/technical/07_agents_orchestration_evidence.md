# T07 — AGENTS state, routing, evidence, fusion, and nodes

## Learning outcome

You can trace an `AuditState` through the 14-node graph, explain reducer behavior and fast/deep routing, and derive deterministic and full verdicts from evidence.

## Prerequisites

Read [AGENTS orchestration](../09_agents_orchestration.md). Know typed dictionaries, async calls, fan-out/fan-in, reducers, and weighted evidence.

## Source map and reading order

1. `agents/src/orchestration/state.py::AuditState` and `::_merge_tool_status`.
2. `graph.py::{_route_from_evidence_router,build_graph}`.
3. Nodes in runtime order, beginning `nodes/ml_assessment.py` and `quick_screen.py`.
4. `nodes/evidence_router.py`, deep evidence nodes, consensus/cross-validation, synthesizer.
5. `verdict/{evidence,emit,reliability,fuse}.py`.
6. Post-synthesis `reflection`, `explainer`, and `visualizer` nodes.

## Entry point and complete call chain

Caller supplies only contract code/address. `ml_assessment` records teacher output/evidence; `quick_screen` runs every time; `evidence_router` records decisions and selects fast synthesis or deep branches. RAG, static, graph explanation, and formal verification can fan out, then converge at `audit_check`. Consensus and cross-validation interpret accumulated evidence, `synthesizer` builds the report, and every path continues through `reflection → explainer → visualizer → END`.

## Important symbols and configuration

- `AuditState(total=False)` lets nodes return partial updates.
- Append reducers preserve routing/evidence history; `_merge_tool_status` merges per-tool dictionaries so sibling status is not lost.
- `tool_status` distinguishes ran/succeeded/failed/unavailable instead of treating absence as negative evidence.
- Evidence records include class, source, polarity, strength, reliability, and deterministic flag.
- `fuse()` de-correlates sources in the same family, emits `verdict_provable` from deterministic evidence, and `verdict_full` from all evidence.

## Annotated source excerpt

Source: `agents/src/orchestration/verdict/fuse.py::fuse`

```python
det_items = [e for e in items if e.deterministic]
verdict_provable, conf_provable, _ = _fuse_for_evidence(det_items)
verdict_full, conf_full, driving = _fuse_for_evidence(items)
```

“Provable” here means reproducible deterministic evidence fusion. It is not the computation inside the EZKL proxy circuit.

## Worked example

ML supports Reentrancy with strength `0.8`, reliability `0.7`; Slither supports with `0.9×0.8`; a nondeterministic RAG result supports with `0.6×0.5`. If ML and another ML head share a family, each is discounted by family count. Deterministic fusion excludes RAG, while full fusion includes it. A failed Aderyn run contributes status/error, not a REFUTES item.

## Success trace

Every node returns only owned fields; reducers preserve parallel updates; routing has a human-readable trace; unavailable tools are explicit; evidence provenance is retained; fusion is deterministic; synthesizer produces structured report; enrichment reaches visualizer.

## Failure trace

A node error is captured fail-soft when designed and may yield a partial report. A timeout is not evidence of safety. Wrong reducer semantics can overwrite another branch. Routing code must not inspect raw source or import LLM logic; isolation tests enforce that boundary. Checkpointer fallback loses restart persistence.

## Design reasoning and rejected alternatives

Typed shared state makes cross-node contracts visible. Routing is deterministic and isolated from source/LLM to resist injection-influenced control flow. Evidence-family discounting prevents correlated tools from counting as independent confirmations. Dual verdicts preserve a deterministic claim while allowing richer nondeterministic explanation.

## Safe change walkthrough

For a new node, declare state fields/reducers first, implement fail-soft output and `tool_status`, register node, add routing/edges, add source-isolation and success/failure tests, then update timing and report consumption. Verify both fast and deep paths still reach all three post-synthesis nodes.

## Guided lab

Complete [L07 — AuditState, routing, status, and fuse](../labs/07_agents_state_fusion.md).

## Tests and expected results

```bash
cd agents && TMPDIR=/tmp TMP=/tmp TEMP=/tmp poetry run pytest -q \
  tests/test_verdict_fuse.py tests/test_routing_isolation.py tests/test_p2_evidence_integration.py
```

Expected: family discount, dual-verdict, routing-isolation, and evidence integration assertions pass. See status for unrelated current suite failures.

## Review questions

Why is `tool_status` separate from evidence? Which nodes always run? What reaches `verdict_provable`? How can a parallel branch accidentally erase state?

## Ownership checklist

- I can draw both fast and deep paths through visualizer.
- I know each field’s writer and reducer.
- I distinguish unavailable, clean, and refuting evidence.
- I can calculate family-discounted fusion.
