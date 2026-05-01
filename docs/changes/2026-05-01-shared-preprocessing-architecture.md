# 2026-05-01 — §5.10 Shared Preprocessing Architecture

## Summary

Implemented the §5.10 shared preprocessing refactor: extracted the duplicated Solidity-to-graph
feature engineering logic from two independent files into a canonical shared package.
Also delivered T1-C (sliding-window tokenization) as part of the preprocess.py thin wrapper.

Commit: `cfc08d8`  Branch: `claude/review-project-progress-X8YuG`

---

## Problem Being Solved

`ml/data_extraction/ast_extractor.py` (offline batch, ~68K training contracts) and
`ml/src/inference/preprocess.py` (online inference, one contract per API request) each
contained verbatim copies of:

- `NODE_TYPES`, `VISIBILITY_MAP`, `EDGE_TYPES` dicts (identical in both)
- `node_features()` / `_node_feat()` — 8-dim feature vector logic
- The full edge-building loop (CALLS, READS, WRITES, EMITS, INHERITS)
- `contracts[0]` multi-contract selection policy

Any change to feature engineering had to be applied in two places. A missed sync caused
**silent accuracy regression**: inference receives features encoded differently from training,
with no error signal. This is the worst class of ML bug because it has no stack trace.

---

## New Package: `ml/src/preprocessing/`

### `graph_schema.py` — single source of truth for feature encoding

| Symbol | Type | Purpose |
|--------|------|---------|
| `FEATURE_SCHEMA_VERSION` | `str = "v1"` | Appended to inference cache keys; bump on any schema change |
| `NODE_FEATURE_DIM` | `int = 8` | GNNEncoder in_channels; hardcoded in checkpoint |
| `NUM_EDGE_TYPES` | `int = 5` | Width of EDGE_TYPES vocabulary |
| `NODE_TYPES` | `dict[str, int]` | 8 node categories (STATE_VAR=0 … CONTRACT=7) |
| `VISIBILITY_MAP` | `dict[str, int]` | Ordinal encoding (public=0, internal=1, private=2) |
| `EDGE_TYPES` | `dict[str, int]` | 5 relation types (CALLS=0 … INHERITS=4) |
| `FEATURE_NAMES` | `tuple[str, ...]` | 8-element name registry for drift detection / explainability |

The module also contains a compile-time `assert len(FEATURE_NAMES) == NODE_FEATURE_DIM`
that catches length divergence at import time, not deep in a training loop.

### `graph_extractor.py` — canonical extraction (never returns None)

**Exception hierarchy:**
```
GraphExtractionError               — base
  SolcCompilationError             — bad Solidity  → HTTP 400 / skip
  SlitherParseError                — infra failure → HTTP 500 / skip
  EmptyGraphError                  — 0 AST nodes   → HTTP 400 / skip
```

**`GraphExtractionConfig` dataclass:**
```python
multi_contract_policy: str = "first"          # "first" | "by_name"
target_contract_name: str | None = None
include_edge_attr: bool = True
solc_binary: str | Path | None = None         # None → system PATH
solc_version: str | None = None               # for --allow-paths version check
allow_paths: str | None = None                # offline: project_root
```

**`extract_contract_graph(sol_path, config) → Data`:**
- Never returns None; always raises `GraphExtractionError` on failure
- Calls `_build_node_features()` (replication of training-time logic, with CONSTRAINT comment)
- Node insertion order fixed: CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs
- Edge attr shape: `[E]` 1-D int64 (PyG convention) — fixes offline `[E,1]` vs online `[E]` mismatch
- Feature dim validated at extraction boundary (not deep inside GATConv)
- 0-node graph raises `EmptyGraphError` with actionable message

### `__init__.py` — re-exports all public symbols

---

## Refactored: `ml/data_extraction/ast_extractor.py` (V4.2 → V4.3)

**Removed:** `NODE_TYPES`, `VISIBILITY_MAP`, `EDGE_TYPES` inline dicts (~25 lines).
**Removed:** `node_features()` standalone function (~40 lines).
**Removed:** `_get_slither_instance()` method (Slither setup now inside `graph_extractor.py`).

**Added imports:**
```python
from src.preprocessing.graph_extractor import (
    GraphExtractionConfig, GraphExtractionError, extract_contract_graph
)
from src.preprocessing.graph_schema import NODE_TYPES, VISIBILITY_MAP, EDGE_TYPES
```

**`contract_to_pyg()` is now a thin wrapper:**
1. Build `GraphExtractionConfig(solc_binary=..., solc_version=..., allow_paths=project_root)`
2. Call `extract_contract_graph(Path(contract_path), config)`
3. Catch `GraphExtractionError` → return `None` (batch skip-and-log policy preserved)
4. Attach offline-specific metadata: `contract_hash`, `contract_path`, `y`

**Kept (offline-only):** `parse_solc_version()`, `solc_supports_allow_paths()`, `get_solc_binary()`,
`get_project_root()`, `extract_batch_with_checkpoint()`.

---

## Refactored: `ml/src/inference/preprocess.py`

**Removed:** `_NODE_TYPES`, `_VISIBILITY_MAP`, `_EDGE_TYPES` inline dicts (~25 lines).
**Removed:** `_extract_graph()` body — 80 lines of extraction logic.

**Added imports:**
```python
from ..preprocessing.graph_extractor import (
    EmptyGraphError, GraphExtractionConfig, GraphExtractionError,
    SlitherParseError, SolcCompilationError, extract_contract_graph
)
from ..preprocessing.graph_schema import FEATURE_SCHEMA_VERSION
from ..utils.hash_utils import get_contract_hash, get_contract_hash_from_content
```

**`_extract_graph()` is now a thin exception translator:**
```python
# SolcCompilationError / EmptyGraphError → ValueError  (HTTP 400)
# SlitherParseError                      → RuntimeError (HTTP 500)
# RuntimeError (Slither not installed)   → propagates unchanged
```

**Cache key improvement:**
```python
# Before: contract_hash = hashlib.md5(source_code.encode()).hexdigest()
# After:  contract_hash = f"{content_md5}_{FEATURE_SCHEMA_VERSION}"
# Stale cached graphs are automatically rejected after any schema change.
```

**Added (T1-C — sliding-window tokenization):**
- `process_source_windowed(source_code, stride=256, max_windows=8)` → `(graph, list[dict])`
- `_tokenize_sliding_window(source_code, hash, stride, max_windows)` → `list[dict]`
- Short contracts (≤512 tokens) return a 1-element list — no overhead vs single-window path
- Each window dict includes `"window_index": int` for predictor aggregation

---

## Bugs Fixed

| Bug | Location | Fix |
|-----|----------|-----|
| `edge_attr` shape `[E,1]` offline vs `[E]` online | `ast_extractor.py` | Unified to `[E]` in shared extractor |
| 0-node graph: generic `ValueError("zero graph nodes")` | `preprocess.py` | `EmptyGraphError` with actionable message |
| Feature dim mismatch: crashes in GATConv | both files | Validated at extraction boundary in `graph_extractor.py` |
| Inline `hashlib.md5()` ignoring schema version | `preprocess.py` | `hash_utils` + `FEATURE_SCHEMA_VERSION` suffix |

---

## Verification Checklist

- `from ml.src.preprocessing import extract_contract_graph, GraphExtractionConfig` → imports cleanly
- `extract_contract_graph()` raises `SolcCompilationError` on bad Solidity (not generic ValueError)
- `extract_contract_graph()` raises `EmptyGraphError` on all-dependency file
- `graph.x.shape[1] == 8` enforced inside extractor; `SlitherParseError` on mismatch
- `graph.edge_attr.shape == (E,)` — 1-D tensor (not `(E,1)`)
- `contract_to_pyg()` returns `None` on any `GraphExtractionError` (batch policy intact)
- `process_source()` hash contains `_v1` suffix
- `process_source_windowed()` returns ≥2 dicts for a 600-token contract
