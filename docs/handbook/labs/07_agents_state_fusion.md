# L07 — Trace AuditState, routing, status, and fuse

## Learning objective

Construct evidence/status updates and verify reducers, source-family discount, routing isolation, and dual verdicts.

## Prerequisites

Read [T07](../technical/07_agents_orchestration_evidence.md). Use the AGENTS Poetry environment.

## Source reading order

`state.py::{AuditState,_merge_tool_status}` → `graph.py::_route_from_evidence_router` → `verdict/evidence.py` → `reliability.py` → `fuse.py::fuse` → related tests.

## Setup and artifact requirements

Tier is smoke. No model, RAG index, RPC, or proof artifact is required.

## Initial observation

```bash
cd agents && TMPDIR=/tmp TMP=/tmp TEMP=/tmp poetry run pytest -q \
  tests/test_verdict_fuse.py tests/test_routing_isolation.py
```

## Controlled edit

Add a test with two deterministic same-family supports, one deterministic refute, and one nondeterministic support. Assert family discount applies, `verdict_provable` excludes the last item, and `verdict_full` can differ. Add a reducer assertion that updating Aderyn status preserves Slither status.

## Expected success output

Fusion confidence matches manual signed/reliability calculation; deterministic and full item sets differ; sibling tool status remains present.

## Expected failure output

Marking a failed tool as REFUTES should violate Rule 5C-oriented expectations. Removing the reducer causes branch updates to overwrite one another. Routing source/LLM access fails isolation tests.

## Verification

Run observation and `verify_handbook.py lab --check L07`.

## Reset and cleanup

Restore the edited tests; no external state exists.

## Completion rubric

Complete when you can calculate both verdict tiers and identify every state writer/reducer involved.

## Review questions

Why discount families? Why is a timeout not REFUTES? Which post-synthesis nodes always run?

## Classification

Smoke; safe preflight; controlled unit-test edit.
