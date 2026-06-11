# `sentinel_data.representation` — Turning Source Code into Model-Consumable Artifacts

## What This Module Does

The representation module is Stage 3 of the SENTINEL data pipeline. It takes preprocessed `.sol` files and produces two types of model-consumable artifacts:

1. **Graph representations (`.pt`)** — PyG `Data` objects with node features, edge indices, and labels, ready for the GNN (Graph Neural Network)
2. **Token representations (`.pt`)** — Windowed token sequences for GraphCodeBERT, ready for the Transformer encoder

This module is the bridge between "source code on disk" and "tensors in GPU memory." Every contract that the model trains on passes through this module first.

## Why This Matters

The representation is the most architecturally significant part of the pipeline because:

1. **It determines what the model can learn** — the graph schema (14 node types, 12 edge types, 12 features) defines the model's vocabulary
2. **It must be byte-identical to the old path** — the thin-adapter pattern ensures the new import path produces exactly the same output as the old `ml/src/preprocessing/` path
3. **It must be cacheable and versionable** — re-extracting 22K contracts takes hours; the content-addressed cache makes re-runs instant

## Architecture Overview

```
Preprocessed .sol + meta.json
        │
        ▼
┌─────────────────────────────────────────────────┐
│              orchestrator.py (v2 manifest-driven) │
│  Reads meta.json → runs extraction → writes       │
│  graph .pt + token .pt + sidecar .rep.json        │
└──────────────┬──────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌──────────┐
│ graph_ │ │ tokenizer│ │ cache_   │
│ schema │ │ .py     │ │ manager  │
│ .py    │ │         │ │ .py      │
└────────┘ └────────┘ └──────────┘
    │
    ▼
┌─────────────────────┐
│ graph_extractor.py   │
│ (thin adapter →      │
│  ml/src/preprocessing)│
└─────────────────────┘
```

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `__init__.py` | Re-exports all public symbols from schema + extractor | 64 |
| `graph_schema.py` | Thin adapter — re-exports v9 schema constants from `ml/src/preprocessing/` | 148 |
| `graph_extractor.py` | Thin adapter — re-exports `extract_contract_graph` from `ml/src/preprocessing/` | 77 |
| `tokenizer.py` | Thin adapter — re-exports windowed tokenizer from `ml/src/data_extraction/` | 72 |
| `orchestrator.py` | v2 manifest-driven orchestrator — reads Stage 1 output, writes representations | 344 |
| `cache_manager.py` | Content-addressed representation cache | 119 |
| `versioner.py` | Schema/extractor version registry — prevents silent version mixing | 100 |
| `cfg_builder.py` | Standalone CFG builder for opt-in `--emit-cfg` | 274 |
| `call_graph.py` | Call-graph builder — **deferred to v3.1** | 30 |
| `opcode_extractor.py` | EVM opcode extractor — **deferred to v3.1** | 31 |
| `pdg_builder.py` | PDG builder — **deferred to v3.1** | 29 |

## The v9 Schema Constants

The active graph schema is **v9** (verified against `ml/src/preprocessing/graph_schema.py:161,175,218`):

| Constant | Value | Description |
|----------|-------|-------------|
| `FEATURE_SCHEMA_VERSION` | `"v9"` | Schema version string |
| `NODE_FEATURE_DIM` | `12` | Dimension of each node's feature vector |
| `NUM_NODE_TYPES` | `14` | Number of distinct node types (STATE_VAR, FUNCTION, etc.) |
| `NUM_EDGE_TYPES` | `12` | Number of distinct edge types (CONTAINS, CALLS, etc.) |
| `_MAX_TYPE_ID` | `13.0` | Maximum type ID (0-indexed) |
| `NUM_CLASSES` | `10` | Number of vulnerability classes (LOCKED) |

The 10-class taxonomy (LOCKED — class order matches all existing checkpoints):

| Index | Class | Description |
|-------|-------|-------------|
| 0 | CallToUnknown | Call to unknown address via `.call{}` / `.delegatecall{}` |
| 1 | DenialOfService | Gas griefing or unbounded iteration |
| 2 | ExternalBug | Cross-contract call to non-interface target |
| 3 | GasException | Unchecked `send()` / `transfer()` |
| 4 | IntegerUO | Integer overflow/underflow |
| 5 | MishandledException | Call with unused return value |
| 6 | Reentrancy | State change after external call |
| 7 | Timestamp | Dependence on `block.timestamp` / `now` |
| 8 | TransactionOrderDependence | `tx.origin` in permission check |
| 9 | UnusedReturn | Internal function call with unused return |

## The Thin-Adapter Pattern

This module uses a **thin-adapter pattern** for the core extraction functions. Instead of copying code from `ml/src/preprocessing/` into `sentinel_data/`, the adapters re-export the same objects:

```python
# sentinel_data/representation/graph_extractor.py
from ml.src.preprocessing.graph_extractor import (
    extract_contract_graph,
    GraphExtractionConfig,
    GraphExtractionError,
    SolcCompilationError,
    SlitherParseError,
    EmptyGraphError,
)
```

**Why thin-adapter?**
- **Single source of truth** — bug fixes in `ml/` automatically propagate to the new path
- **Byte-identical output** — same code, different import name
- **1-line seam swap in Stage 7** — delete the wrapper, rebind the import
- **No `sys.path` hacks** — clean package boundaries

The thin adapters will be deleted in Stage 7 when the seam swap moves the source of truth from `ml/` to `sentinel_data/`.

## The v2 Orchestrator

The `orchestrator.py` is a **new** module (not a port from v1). It reads the per-source manifest from Stage 1 and produces representations:

```python
def represent_source(source, cfg, data_dir, *, dry_run, force, limit, output_dir, emit_cfg):
    """Run the full representation pipeline for one source.
    
    Reads data/preprocessed/<source>/*.sol + *.meta.json from Stage 1,
    runs the graph_extractor on each, writes to
    data/representations/<source>/<sha256>.{pt,tokens.pt,rep.json}.
    """
```

**Key features:**
- **Content-addressed caching** — skips already-computed files unless `--force` is passed
- **Cache key** = `(sha256, schema_version, extractor_version)` — any change invalidates
- **Sidecar `.rep.json`** records provenance: schema version, extractor version, node/edge counts, compute time
- **Multiprocessing support** — parallel extraction across workers

## Content-Addressed Cache

The cache manager ensures that re-extraction is only done when necessary:

```python
# Check if already cached
if is_cached(output_dir, sha256, schema_version, extractor_version):
    # Skip — cache hit
    pass
else:
    # Extract and cache
    graph = extract_contract_graph(sol_path)
    torch.save(graph, f"{output_dir}/{sha256}.pt")
```

The cache is invalidated when:
- The schema version changes (e.g. v9 → v10)
- The extractor version changes (e.g. after a bug fix in `graph_extractor.py`)
- The Slither version changes (which can affect extraction output)

The `versioner.py` maintains a global `data/representations/_version_registry.json` that tracks the current `(schema_version, extractor_version)` pair.

## The CFG Builder

The CFG (Control Flow Graph) builder is an **opt-in** representation channel, gated by `--emit-cfg`:

```python
cfg_artifact = build_cfg(sol_path, config, sha256, source)
# cfg_artifact.functions → [CfgFunction(name, nodes, edges, num_loops, max_depth)]
```

It produces a normalized per-function CFG as a JSON-serializable artifact. The CFG is **not used for training in Run 11** — it's consumed by Stage 6's `complexity_proxy_risk` detector, which needs a flat CFG view to spot loops, dead code, etc.

**Deferred to v3.1:** PDG builder, call-graph builder, opcode extractor. These are placeholders with deferral docstrings.

## How to Use

```bash
# Represent a single source
sentinel-data represent --source solidifi

# Represent with multiprocessing
sentinel-data represent --source dive --workers 4

# Force re-extraction (ignore cache)
sentinel-data represent --source scabench --force

# Limit to N files
sentinel-data represent --source solidifi --limit 100

# Opt-in CFG generation
sentinel-data represent --source solidifi --emit-cfg

# Dry-run
sentinel-data represent --source solidifi --dry-run
```

## Output Format

For each contract, the orchestrator produces 3 files:

| File | Content | Format |
|------|---------|--------|
| `<sha256>.pt` | PyG `Data` object (graph) | torch pickle |
| `<sha256>.tokens.pt` | Tokenized input for GraphCodeBERT | torch tensor |
| `<sha256>.rep.json` | Provenance sidecar | JSON |

The `.rep.json` sidecar:
```json
{
  "schema_version": "v9",
  "extractor_version": "v2.1-windowed-gcb",
  "node_count": 42,
  "edge_count": 87,
  "window_count": 4,
  "compute_time_ms": 1250,
  "cache_hit": false
}
```

## Performance Budget

The budget is **100-file extraction in < 1 minute on 8 cores** (within ±10% of the old `ml/src/preprocessing/` path). The thin-adapter adds ~0 overhead since it re-exports the same code.

## Pipeline Position

```
Stage 2: Preprocessing (flatten, compile, dedup, normalize)
    ↓
Stage 3: Representation ← YOU ARE HERE (graph + token extraction)
    ↓
Stage 4: Labeling (assign vulnerability classes)
```

## Design Decisions

1. **Thin-adapter over copy-paste** — ensures byte-identical output and trivial seam swap
2. **Content-addressed cache** — re-extraction of 22K contracts takes hours; cache makes re-runs instant
3. **SHA-256 from Stage 1** — no re-hashing at Stage 2; `meta.sha256` is the cache key
4. **CFG builder as opt-in** — not needed for Run 11 training; used by Stage 6 analysis
5. **PDG/call-graph/opcode deferred to v3.1** — Run 11 doesn't need them; time is better spent on Stage 4 verification
