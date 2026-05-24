# ml/src/preprocessing — Graph Schema and Extraction

Core graph construction logic shared between training and inference pipelines.

## Purpose

Contains the graph schema definitions and extraction logic that enforce consistency between offline batch extraction and online inference. Both pipelines import from these modules to guarantee feature parity — any divergence here causes silent accuracy regressions.

## Components

### `graph_schema.py`
**Graph Schema Definitions (v8)**

Defines the complete graph structure used throughout SENTINEL.

**Key constants:**
```python
NODE_FEATURE_DIM        = 11
FEATURE_SCHEMA_VERSION  = "v8"
NUM_EDGE_TYPES          = 11
NUM_NODE_TYPES          = 13
NUM_CLASSES             = 10
```

**NodeType IntEnum (13 types):**
```python
class NodeType(IntEnum):
    STATE_VAR, FUNCTION, MODIFIER, EVENT, FALLBACK, RECEIVE, CONSTRUCTOR, CONTRACT,
    CFG_NODE_CALL, CFG_NODE_WRITE, CFG_NODE_READ, CFG_NODE_CHECK, CFG_NODE_OTHER
```
Always use `NodeType.FUNCTION` etc. — never hardcode raw integer values.

**CFG subtypes (8–12):** Distinct type_ids give the GNN different initial embeddings for different statement roles. CFG_NODE_CALL (8) for external calls, CFG_NODE_WRITE (9) for state writes, CFG_NODE_READ (10) for state reads, CFG_NODE_CHECK (11) for require/assert/if conditions, CFG_NODE_OTHER (12) for all other statements.

**STRUCTURAL_PREFIX_TYPES:**
```python
STRUCTURAL_PREFIX_TYPES = frozenset({
    NodeType.FUNCTION, NodeType.MODIFIER, NodeType.CONSTRUCTOR,
    NodeType.FALLBACK, NodeType.RECEIVE
})
```
Used by `select_prefix_nodes()` in `sentinel_model.py` to identify declaration-level nodes eligible for GNN prefix injection.

**Node Features (11 dimensions):**

| Dim | Feature | Notes |
|-----|---------|-------|
| [0] | `type_id / 12.0` | NodeType enum value (0–12 → 0.0–1.0) |
| [1] | `visibility` | 0.0=public/external, 0.5=internal, 1.0=private |
| [2] | `uses_block_globals` | block.timestamp/number/difficulty reads |
| [3] | `view` | 0/1 |
| [4] | `payable` | 0/1 |
| [5] | `complexity` | log1p(CFG_block_count) / log1p(100) |
| [6] | `loc` | log1p(lines) / log1p(1000) |
| [7] | `return_ignored` | call return value unused downstream |
| [8] | `call_target_typed` | 0=dynamic/unknown, 1=typed interface |
| [9] | `has_loop` | 0/1 |
| [10] | `external_call_count` | log1p(count) / log1p(20) |

CFG nodes (CFG_NODE_CALL, CFG_NODE_WRITE, CFG_NODE_READ, CFG_NODE_CHECK, CFG_NODE_OTHER) inherit dims [1,3,4,5,9] from their parent FUNCTION node.

**Edge Types (11 types):**

| Type | Name | Description |
|------|------|-------------|
| 0 | CALLS | function → called function (internal) |
| 1 | READS | function → state variable |
| 2 | WRITES | function → state variable |
| 3 | EMITS | function → event |
| 4 | INHERITS | contract → parent contract |
| 5 | CONTAINS | contract/function → child node |
| 6 | CONTROL_FLOW | CFG block → CFG block |
| 7 | REVERSE_CONTAINS | flip of type 5 (generated at runtime, never stored) |
| 8 | CALL_ENTRY | call site → function entry CFG block |
| 9 | RETURN_TO | function exit CFG block → call-site continuation |
| 10 | DEF_USE | definition → use (data-flow) |

`edge_attr` is stored as 1-D int64 of shape `[E]` (PyG convention). `Embedding(11, 64)` in GNNEncoder consumes it.

### `graph_extractor.py`
**Core Graph Extraction Logic**

Implements `extract_contract_graph()` — the single source of truth for graph construction.

**Key functions:**
- `extract_contract_graph(contract_path, config)` — main entry point
- `node_features(node, parent_fn)` — computes 11-dim feature vector
- `build_edges(contract)` — constructs all 11 typed edge relations
- `GraphExtractionConfig` — configuration dataclass

**Architecture note:** imported by both:
- `ml/src/data_extraction/ast_extractor.py` (offline batch pipeline)
- `ml/src/inference/preprocess.py` (online inference)

## Schema Versioning

**Current Version: v8**

Schema history:
- v7: 11 features (dropped in_unchecked), EMITS(3) and INHERITS(4) now fire, CFG nodes inherit dims from FUNCTION, DEF_USE(10) added
- v8: CALL_ENTRY(8), RETURN_TO(9), DEF_USE(10) stored on disk (ICFG-Lite + data-flow edges)

When modifying the schema:
1. Update `FEATURE_SCHEMA_VERSION` in `graph_schema.py`
2. Re-extract all graphs: `poetry run python ml/scripts/reextract_graphs.py`
3. Rebuild cache: `poetry run python ml/scripts/create_cache.py`
4. Retrain model from scratch

## Usage

### Direct import
```python
from ml.src.preprocessing.graph_schema import (
    NODE_FEATURE_DIM, FEATURE_SCHEMA_VERSION, NUM_EDGE_TYPES,
    NodeType, STRUCTURAL_PREFIX_TYPES
)
from ml.src.preprocessing.graph_extractor import (
    extract_contract_graph, GraphExtractionConfig
)

config = GraphExtractionConfig(include_cfg=True, include_reverse_contains=True)
graph = extract_contract_graph(contract_path, config)
# graph.x: [N, 11], graph.edge_attr: [E] int64, graph.edge_index: [2, E]
```

### Via data extraction pipeline
```bash
poetry run python ml/scripts/reextract_graphs.py
```

### Via inference pipeline
```python
from ml.src.inference.preprocess import ContractPreprocessor
graph, tokens = ContractPreprocessor().process("Contract.sol")
```

## Testing

`ml/tests/test_preprocessing.py`:
- NodeType IntEnum (13 types, correct values)
- Feature dimension (11-dim)
- Edge type validation (11 types, NUM_EDGE_TYPES=11)
- STRUCTURAL_PREFIX_TYPES membership
- CFG feature inheritance
- Feature builder correctness

## Dependencies

- `slither-analyzer` ≥ 0.9.3 — AST/CFG extraction
- `torch-geometric` — PyG Data structures
- `torch` — tensor operations
