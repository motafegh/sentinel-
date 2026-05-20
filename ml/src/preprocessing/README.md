# ml/src/preprocessing — Graph Schema and Extraction

Core graph construction logic shared between training and inference pipelines.

## Purpose

This module contains the fundamental graph schema definitions and extraction logic that ensure consistency between offline data extraction and online inference. Both training and inference pipelines import from these modules to guarantee feature parity.

## Components

### `graph_schema.py`
**Graph Schema Definitions (v7)**

Defines the complete graph structure used throughout SENTINEL:

**Node Types (13 types):**
- CONTRACT, FUNCTION, PARAMETER, VARIABLE, EVENT, MODIFIER, STRUCT
- ENUM, STATE_VARIABLE, BLOCK, UNCHECKED_BLOCK, TMP_VARIABLE, ERROR

**Node Features (11 dimensions):**
- [0] `type_id / 12.0` — Normalized node type
- [1] `visibility` — 0.0=public/external, 0.5=internal, 1.0=private
- [2] `uses_block_globals` — Uses block.timestamp/number/difficulty
- [3] `view` — View function flag
- [4] `payable` — Payable function flag
- [5] `complexity` — log1p(CFG_block_count) / log1p(100)
- [6] `loc` — log1p(lines) / log1p(1000)
- [7] `return_ignored` — Call return value unused downstream
- [8] `call_target_typed` — 0=dynamic/unknown, 1=typed interface
- [9] `has_loop` — Contains loop flag
- [10] `external_call_count` — log1p(count) / log1p(20)

**Edge Types (8 types):**
- 0: CALLS — function → called function (internal)
- 1: READS — function → state variable
- 2: WRITES — function → state variable
- 3: EMITS — function → event
- 4: INHERITS — contract → parent contract
- 5: CONTAINS — contract/function → child node
- 6: CONTROL_FLOW — CFG block → CFG block
- 7: REVERSE_CONTAINS — flip of type 5 (generated at runtime)

**Key Constants:**
- `NODE_FEATURE_DIM = 11`
- `FEATURE_SCHEMA_VERSION = "v7"`
- `NUM_EDGE_TYPES = 8`

### `graph_extractor.py`
**Core Graph Extraction Logic**

Implements `extract_contract_graph()` — the single source of truth for graph construction:

**Key Functions:**
- `extract_contract_graph()` — Main extraction entry point
- `node_features()` — Compute 11-dim feature vectors
- `build_edges()` — Construct typed edge indices
- `GraphExtractionConfig` — Configuration dataclass

**Features:**
- Slither-based AST/CFG extraction
- Typed edge construction with 8 edge types
- Feature computation with CFG inheritance
- Custom exception types for error handling
- Schema validation

**Architecture Importance:**
This module is imported by both:
- `ml/src/data_extraction/ast_extractor.py` (offline batch)
- `ml/src/inference/preprocess.py` (online inference)

This ensures training and inference use identical feature computation, preventing silent accuracy regressions.

## Schema Versioning

**Current Version: v7**

When modifying the schema:
1. Update `FEATURE_SCHEMA_VERSION` in `graph_schema.py`
2. Re-extract all graphs using `ml/scripts/reextract_graphs.py`
3. Rebuild cache using `ml/scripts/create_cache.py`
4. Retrain model from scratch

**Locked Dimensions:**
- `NODE_FEATURE_DIM = 11` — Changing requires full re-extraction
- `NUM_CLASSES = 10` — Fixed vulnerability classes
- `fusion_output_dim = 128` — ZKML proxy dependency

## Usage

### Direct Import (for custom processing)
```python
from ml.src.preprocessing.graph_schema import (
    NODE_TYPES, EDGE_TYPES, NODE_FEATURE_DIM,
    FEATURE_SCHEMA_VERSION
)
from ml.src.preprocessing.graph_extractor import (
    extract_contract_graph, GraphExtractionConfig
)

config = GraphExtractionConfig(
    include_cfg=True,
    include_reverse_contains=True
)
graph = extract_contract_graph(contract_path, config)
```

### Via Data Extraction Pipeline
```bash
poetry run python ml/scripts/reextract_graphs.py
```

### Via Inference Pipeline
```python
from ml.src.inference.preprocess import preprocess_contract

graph, tokens = preprocess_contract(contract_source)
```

## Testing

Schema validation tests in `ml/tests/test_preprocessing.py`:
- Node type enumeration (13 types)
- Feature dimension validation (11-dim)
- Edge type validation (8 types)
- CFG feature inheritance
- Feature builder correctness

## Dependencies

- `slither-analyzer` >= 0.9.3 — AST/CFG extraction
- `torch-geometric` — PyG data structures
- `torch` — Tensor operations

## Design Principles

1. **Single Source of Truth** — Both training and inference import from here
2. **Schema Locking** — Versioned schema prevents silent incompatibilities
3. **Type Safety** — Custom exception types for clear error handling
4. **Feature Consistency** — Identical computation in all pipelines
5. **CFG Inheritance** — CFG nodes inherit relevant function features
