# L08 — Exercise MCP, RAG, gateway, and SQLite recovery

## Learning objective

Validate a tool schema, durable job lifecycle/recovery, and RAG integrity failure without requiring a full live deployment.

## Prerequisites

Read [T08](../technical/08_services_rag_gateway.md). Use the AGENTS Poetry environment.

## Source reading order

`inference_server.py` tool list/handler → `api/models.py` → `sqlite_job_store.py` → `gateway.py::_run_job` → `rag/retriever.py::HybridRetriever`.

## Setup and artifact requirements

Tier is module. Unit tests create temporary SQLite. Live RAG requires a synchronized local index; live gateway flow requires ML/MCP dependencies.

## Initial observation

```bash
cd agents && TMPDIR=/tmp TMP=/tmp TEMP=/tmp poetry run pytest -q \
  tests/test_inference_server.py tests/test_gateway.py tests/test_p10_gateway.py
```

## Controlled edit

Add a SQLite test that creates queued and running records, reconstructs the store as if after restart, invokes recovery, and asserts neither record is presented as completed. Add a tool-schema negative case with a missing required source field. Edit tests only.

## Expected success output

Job persists across store instances; transitions remain conditional; recovery is explicit; invalid tool input returns structured error; public job serialization omits contract source.

## Expected failure output

Illegal completed-from-queued transition leaves state unchanged. A deliberately mismatched FAISS/chunk fixture raises corruption rather than returning wrong chunks.

## Verification

Run the selected tests and `verify_handbook.py lab --check L08`.

## Reset and cleanup

Restore tests and remove only temporary SQLite/index fixtures. Stop manually launched services.

## Completion rubric

Complete when you can explain every job state and distinguish unit/mocked service behavior from live health.

## Review questions

Why return 202? What makes recovery durable? Which operation proves RAG index synchronization?

## Classification

Module; safe preflight; controlled temporary persistence fixtures.
