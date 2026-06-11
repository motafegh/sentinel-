# ADR-0007 — Representation Port Design (Stage 2)

**Date:** 2026-06-11  
**Status:** Accepted  
**Author:** SENTINEL data engineering  
**Stage:** 2 (Representation Extraction)  
**Refs:** `docs/proposal/Data_Module_Proposals/actionable_plans/03_stage_2_representation.md` · [ADR-0003](0003-dual-path-four-eye-architecture.md)

---

## Context

Stage 2 moves graph and token extraction from `ml/src/preprocessing/` and
`ml/src/data_extraction/` into `sentinel_data/representation/`.  The active
training pipeline (Run 9, F1=0.3081) uses the `ml/` code directly.  Stage 7
(export + seam) is the stage that actually switches the ML module over to the
new path.  Stage 2 must not break the active pipeline.

Three concrete risks drove the design:

1. **Extraction divergence** — a rewrite masquerading as a port could
   silently change node features or edge types, breaking Run 9 resume.
2. **Schema drift** — a wrong constant (e.g. `NODE_FEATURE_DIM=11` instead
   of `12`) would silently produce mismatched tensors that only fail at the
   DataLoader boundary.
3. **Version mixing** — v8 graphs from disk mixed with v9 graphs extracted
   live caused the Run 8 regression.  A new pipeline needs explicit
   invalidation.

---

## Decision

### D-2.1 — Thin adapter pattern (no extraction logic changes)

`sentinel_data/representation/` re-exports from `ml/src/preprocessing/`
and `ml/src/data_extraction/` via Python re-exports:

```python
# graph_extractor.py (thin adapter)
from ml.src.preprocessing.graph_extractor import (
    extract_contract_graph, GraphExtractionConfig, ...
)
```

Both import paths resolve to the **same Python function object** — `old is new`
is guaranteed by construction.  The byte-identical regression test
(`Data/tests/test_representation/test_byte_identical_regression.py`)
asserts this at the object level and verifies tensor equality for 10 real
SolidiFI contracts.

**Rationale:** the alternative (a code copy) would require maintaining two
copies of 1,800 lines of extraction logic and would immediately diverge on
the next bug fix.  The thin adapter makes Stage 7's seam swap a one-line
change (update the live `ml/` code to import from `sentinel_data`).

### D-2.2 — Schema frozen at v9

The active schema (`ml/src/preprocessing/graph_schema.py`) is v9:

| Constant | Value |
|----------|-------|
| `FEATURE_SCHEMA_VERSION` | `"v9"` |
| `NODE_FEATURE_DIM` | `12` |
| `NUM_NODE_TYPES` | `14` |
| `NUM_EDGE_TYPES` | `12` |
| `_MAX_TYPE_ID` | `13.0` |

The Stage 0 stub had these constants backwards (dict[int, str] instead of
dict[str, int]).  Task 2.1 fixed this as part of the port.  No new schema
features are added in v2; schema changes require a version bump and a new
ADR.

The **schema-dim gate test** (`x.shape[-1] == NODE_FEATURE_DIM`) catches any
accidental dimension mismatch at CI time.

### D-2.3 — Windowed tokenizer, not single-window

The v1 training data used `microsoft/graphcodebert-base` with `[4, 512]`
windowed output (stride=256, linspace sub-sampling).  The Stage 0 thin
adapter incorrectly pointed to `ml/src/data_extraction/tokenizer.py`
(codebert-base, single-window `(512,)` output).

**Fix:** created `ml/src/data_extraction/windowed_tokenizer.py` extracting
the per-file windowed tokenization from `ml/scripts/retokenize_windowed.py`.
The thin adapter in `sentinel_data/representation/tokenizer.py` now points
there.  Token files have shape `[4, 512]` and the key `"num_windows"`.

**A-1 fix bundled here:** `_strip_comments()` (removes `/* */` and `//`)
is applied before tokenization so the token budget is spent on code rather
than OpenZeppelin SafeMath docstrings.

### D-2.4 — Only the CFG builder ships; PDG/call-graph/opcode deferred to v3.1

The standalone CFG builder (`cfg_builder.py`) is opt-in via `--emit-cfg`
(default off).  It is consumed by Stage 6's complexity-proxy detector, not
by the Run 11 training graph.

PDG, call-graph, and opcode builders are placeholder files with deferral
docstrings.  Per AUDIT_PATCHES 2-P9: these are v3+ features whose payoff for
Run 11 is unclear and whose engineering cost is non-trivial.  The directory
structure is in place for a v3.1 drop-in.

### D-2.5 — Content-addressed cache keyed on (sha256, schema_version, extractor_version)

Cache key: `(sha256 of source, FEATURE_SCHEMA_VERSION, EXTRACTOR_VERSION)`.
The sidecar (`.rep.json`) is the canonical cache record.  A change to the
schema OR extractor version invalidates all files for that source.

`versioner.py` maintains a per-representations-root `_version_registry.json`.
On each run, `check_and_evict()` compares the live versions against the
registry and evicts stale entries before extraction begins.  This prevents
the Run 8 silent-mix-of-versions failure mode.

### D-2.6 — Sidecar provenance (.rep.json)

Every successfully extracted contract produces a `.rep.json` sidecar with:
`sha256`, `source`, `original_path`, `schema_version`, `extractor_version`,
`node_count`, `edge_count`, `window_count`, `compute_time_ms`, `pragma`,
`solc_version`.

The sidecar is the cache record AND the provenance record.  Stage 4
(verification) and Stage 6 (analysis) can re-read sidecars without
re-running extraction.

### D-2.7 — `ml/` on sys.path is a CLI responsibility

The thin adapters import from `ml.src.*`.  The `conftest.py` adds the
repo root + `ml/` to `sys.path` for pytest.  The CLI `cli.py` does the same
at startup via a path-setup block at module level.

This avoids making `ml/` a formal install dependency of `sentinel_data`
(which would create a circular dep: Data → ml → Data) while keeping the
thin-adapter pattern working for both CLI and test contexts.

### D-2.8 — `hash_utils` dropped; content-addressed via Stage 1 SHA-256

The v1 orchestrator (`ast_extractor.py`) named output files by MD5
(`ml/src/utils/hash_utils.get_contract_hash`).  The v2 orchestrator uses
SHA-256 from Stage 1's `meta.json`.  This decouples the output filename from
the tokenization path and aligns with the cache key contract.

---

## Consequences

**Positive:**
- Byte-identical extraction guaranteed by object identity (`old is new`).
- Schema-dim gate catches shape mismatches at CI time.
- Version registry prevents silent graph-version mixing.
- Stage 7 seam swap is a one-line change.
- A-1 (comment stripping), A-2 (RETURN_TO edges), A-3 (interface injection)
  fixes are gated by regression tests.

**Negative:**
- The thin adapter is temporary; Stage 7 makes `ml/` import from
  `sentinel_data` rather than the other way around.  Until then, the
  direction feels backwards (the new code depends on the old code).
- `ml/` on sys.path is implicit; a developer installing only `sentinel_data`
  (without the full repo) will get an import error on any extraction call.
  The error message in `graph_schema.py`'s `__getattr__` fallback is clear,
  but it's not a pip-installable dependency.

**Deferred to v3.1:** PDG, call-graph, opcode builders; DVC integration
(moved to Stage 7); multiprocessing pool for the orchestrator.

---

## Gate

The byte-identical regression test is the acceptance gate for this ADR.
All 10 SolidiFI fixtures must pass; `old is new` must hold; all 45 Stage 2
representation tests must pass.
