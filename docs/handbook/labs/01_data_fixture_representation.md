# L01 — Trace a DATA fixture through representation

## Learning objective

Trace one fixture from preprocessed Solidity/metadata into graph, token, and representation artifacts; understand cache and force behavior.

## Prerequisites

Read [T01](../technical/01_data_pipeline_internals.md). Use the DATA environment. Work on a disposable branch/worktree for the edit.

## Source reading order

`data_module/tests/test_representation/test_orchestrator.py` → `data_module/sentinel_data/representation/orchestrator.py::represent_source` → `::_extract_one` → graph extractor/tokenizer → `graph_schema.py`.

## Setup and artifact requirements

Tier is smoke. Tracked test fixtures and a functioning DATA test environment are sufficient; no production export is needed.

## Initial observation

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp data_module/.venv/bin/python -m pytest \
  data_module/tests/test_representation/test_orchestrator.py -q
```

Inspect the temporary output assertions: each accepted contract has `.pt`, `.tokens.pt`, and `.rep.json` companions.

## Controlled edit

In a disposable worktree, add a test beside `test_represent_source_cache_hit` that runs once, records output modification times, runs again without `force`, and asserts cached counters increase while times remain unchanged. Then call with `force=True` and assert extraction counters—not cache counters—advance. Edit only the test.

## Expected success output

The new test passes; the second call is a cache hit bound to schema/extractor version; forced execution recomputes outputs.

## Expected failure output

Changing the test’s metadata schema/extractor marker should invalidate the cache assertion. Removing the preprocessed directory should raise the explicit “run preprocess first” `FileNotFoundError`.

## Verification

Run the observation command again, then `python3 docs/handbook/tools/verify_handbook.py lab --check L01`.

## Reset and cleanup

```bash
git restore data_module/tests/test_representation/test_orchestrator.py
```

Remove only temporary directories created by pytest; never delete repository DATA artifacts as cleanup.

## Completion rubric

Complete when you can identify the three outputs, explain cache keys, and show forced recomputation without a production-code edit.

## Review questions

Why is metadata paired by SHA-256? Which version changes invalidate cache? What failure counter prevents a fake representation?

## Classification

Smoke; safe preflight; controlled test-only edit.
