# ml/src/preprocessing — Shared Graph Schema & Extraction

**Purpose:** Single source of truth for graph construction. Both offline batch extraction (`ast_extractor.py`) and online inference (`preprocess.py`) import from this module to guarantee identical features.

## Modules

### `graph_schema.py` — Schema Constants

**What it defines:**
- Node feature dimensionality (11 dims)
- Node type vocabulary (13 types: 8 declaration + 5 CFG subtypes)
- Edge type vocabulary (11 types)
- Visibility encoding map
- Feature name registry

**Key constants:**
```python
NODE_FEATURE_DIM       = 11          # Input to GNNEncoder.conv1
FEATURE_SCHEMA_VERSION = "v8"        # Cache invalidation key
NUM_EDGE_TYPES         = 11          # Embedding table width
NUM_NODE_TYPES         = 13          # 0-12 inclusive
```

**NodeType IntEnum:**
```python
class NodeType(IntEnum):
    STATE_VAR      = 0   # State variable declaration
    FUNCTION       = 1   # Function declaration
    MODIFIER       = 2   # Modifier declaration
    EVENT          = 3   # Event declaration
    FALLBACK       = 4   # Fallback function
    RECEIVE        = 5   # Receive function
    CONSTRUCTOR    = 6   # Constructor
    CONTRACT       = 7   # Contract declaration
    CFG_NODE_CALL  = 8   # CFG statement with external call
    CFG_NODE_WRITE = 9   # CFG statement writing state
    CFG_NODE_READ  = 10  # CFG statement reading state
    CFG_NODE_CHECK = 11  # CFG statement with require/assert/if
    CFG_NODE_OTHER = 12  # All other CFG statements
```

**STRUCTURAL_PREFIX_TYPES:**
```python
STRUCTURAL_PREFIX_TYPES = frozenset({
    NodeType.FUNCTION, NodeType.MODIFIER, NodeType.CONSTRUCTOR,
    NodeType.FALLBACK, NodeType.RECEIVE
})
```
Used by `select_prefix_nodes()` in `sentinel_model.py` for GNN prefix injection (Phase 1). Only declaration-level nodes are eligible; CFG nodes carry aggregated signal via REVERSE_CONTAINS edges.

**Node Features (11 dimensions):**

| Index | Feature | Type | Range | Description |
|-------|---------|------|-------|-------------|
| 0 | `type_id / 12.0` | float | [0, 1] | Normalized NodeType ID |
| 1 | `visibility` | float | {0.0, 0.5, 1.0} | public=0.0, internal=0.5, private=1.0 |
| 2 | `uses_block_globals` | float | {0.0, 1.0} | Reads block.timestamp/number/difficulty/basefee |
| 3 | `view` | float | {0.0, 1.0} | Function.view attribute |
| 4 | `payable` | float | {0.0, 1.0} | Function.payable attribute |
| 5 | `complexity` | float | [0, 1] | log1p(CFG_blocks) / log1p(100) |
| 6 | `loc` | float | [0, 1] | log1p(lines) / log1p(1000) |
| 7 | `return_ignored` | float | {-1.0, 0.0, 1.0} | -1.0=IR unavailable, 0.0=captured, 1.0=discarded |
| 8 | `call_target_typed` | float | {-1.0, 0.0, 1.0} | -1.0=source unavailable, 0.0=raw addr, 1.0=typed |
| 9 | `has_loop` | float | {0.0, 1.0} | Contains loop construct |
| 10 | `external_call_count` | float | [0, 1] | log1p(count) / log1p(20), includes Transfer/Send |

**VISIBILITY_MAP:**
```python
VISIBILITY_MAP = {
    "public":   0.0,
    "external": 0.0,
    "internal": 0.5,
    "private":  1.0,
}
```

**Edge Types (11 types):**

| ID | Name | When Used | Stored on Disk? |
|----|------|-----------|-----------------|
| 0 | CALLS | function → internally-called function | Yes |
| 1 | READS | function → state variable read | Yes |
| 2 | WRITES | function → state variable written | Yes |
| 3 | EMITS | function → event emitted | Yes |
| 4 | INHERITS | contract → parent contract | Yes |
| 5 | CONTAINS | function → CFG child nodes | Yes |
| 6 | CONTROL_FLOW | CFG node → successor CFG node | Yes |
| 7 | REVERSE_CONTAINS | CFG node → parent function | No (runtime-only) |
| 8 | CALL_ENTRY | call site → callee ENTRYPOINT | Yes (v8 ICFG-Lite) |
| 9 | RETURN_TO | callee terminal → call-site successor | Yes (v8 ICFG-Lite) |
| 10 | DEF_USE | def node → use node (data-flow) | Yes (v8 DFG) |

**GNNEncoder Phase Routing:**
- **Phase 1 (Layers 1-2):** Edge types 0-5 (structural + CONTAINS forward)
- **Phase 2 (Layers 3-5):** Edge types 6, 8, 9, 10 (CONTROL_FLOW + ICFG-Lite + DEF_USE)
- **Phase 3 (Layers 6-8):** Edge type 7 only (REVERSE_CONTAINS upward)

### `graph_extractor.py` — Graph Construction

**What it does:**
Parses a Solidity file using Slither and constructs a PyG `Data` object with:
- Node features (11-dim per node)
- Typed directed edges (11 edge types)
- Metadata (`contract_name`, `node_metadata` list)

**Public API:**
```python
from ml.src.preprocessing.graph_extractor import (
    extract_contract_graph,
    GraphExtractionConfig,
    GraphExtractionError,
    SolcCompilationError,
    SlitherParseError,
    EmptyGraphError,
)

config = GraphExtractionConfig(
    multi_contract_policy="most_derived",  # Pick most-specific contract
    include_edge_attr=True,                # Attach edge_type IDs
    solc_binary="/path/to/solc-0.8.19",   # Version-pinned binary
    solc_version="0.8.19",                 # For --allow-paths gating
    allow_paths="/project/root",           # Import resolution
)

graph = extract_contract_graph(Path("contract.sol"), config)
# Returns: PyG Data with x=[N,11], edge_index=[2,E], edge_attr=[E]
```

**Exception hierarchy:**
```
GraphExtractionError (base)
├── SolcCompilationError   → HTTP 400 (user's bad Solidity)
├── SlitherParseError      → HTTP 500 (infrastructure failure)
└── EmptyGraphError        → HTTP 400 (zero analyzable nodes)
```

**Graph construction steps:**
1. Run Slither on contract (with version-pinned solc if provided)
2. Select target contract using `multi_contract_policy` heuristic
3. Build declaration nodes: CONTRACT → STATE_VAR → FUNCTION (+ CFG children) → MODIFIER → EVENT
4. Compute 11-dim features for each node
5. Build edges: CALLS, READS, WRITES, EMITS, INHERITS, CONTAINS, CONTROL_FLOW
6. Add ICFG-Lite edges: CALL_ENTRY, RETURN_TO (cross-function control flow)
7. Add DEF_USE edges (intra-function data flow)
8. Return PyG Data object

**CFG Node Feature Inheritance (BUG-C3 fix):**
CFG nodes inherit function-scoped features from their parent FUNCTION node:
- visibility [1], view [3], payable [4], complexity [5], has_loop [9]

This ensures CFG nodes carry contextual signal even though they represent individual statements.

**Contract Selection Heuristics:**
- `"most_derived"` (default): Pick contract inheriting from most other in-file contracts (~92% accurate on BCCC)
- `"last"`: Pick last non-interface contract (87.4% accurate)
- `"most_funcs"`: LEGACY — pick contract with most functions (47.4% accurate, worse than random)
- `"by_name"`: Pick contract matching `target_contract_name`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              graph_extractor.py                              │
│                                                              │
│  extract_contract_graph(sol_path, config)                    │
│    │                                                         │
│    ├─► Slither.parse() → AST                                 │
│    ├─► _select_contract() → target Contract                  │
│    ├─► _add_node() × N → declaration nodes                   │
│    ├─► _build_control_flow_edges() → CFG nodes + edges       │
│    ├─► _add_icfg_edges() → CALL_ENTRY, RETURN_TO             │
│    ├─► _add_def_use_edges() → DEF_USE                        │
│    └─► Data(x, edge_index, edge_attr, ...)                   │
│                                                              │
│  Feature builders:                                           │
│    _build_node_features(obj, type_id) → [11]                 │
│    _build_cfg_node_features(node, func, cfg_type, parent) → [11]  │
│                                                              │
│  Feature computations:                                       │
│    _compute_return_ignored(func) → {-1, 0, 1}                │
│    _compute_call_target_typed(func) → {-1, 0, 1}             │
│    _compute_uses_block_globals(func) → {0, 1}                │
│    _compute_has_loop(func) → {0, 1}                          │
│    _compute_external_call_count(func) → [0, 1]               │
└─────────────────────────────────────────────────────────────┘
```

## Schema Version History

| Version | Changes | Action Required |
|---------|---------|-----------------|
| v1 | 8 features, 5 edge types, 8 node types | Obsolete |
| v2 | 12 features, 7 edge types, 13 node types (added CFG) | Obsolete |
| v3 | +REVERSE_CONTAINS(7) runtime edge | Obsolete |
| v4 | Semantic changes: uses_block_globals, log-normalized loc | Obsolete |
| v5 | BUG fixes: log-normalized complexity, Send detection, most_derived heuristic | Obsolete |
| v6 | visibility normalized to [0,1] (was 0-2 int) | Obsolete |
| v7 | 11 features (dropped in_unchecked), EMITS/INHERITS fire, CFG inheritance | Obsolete |
| v8 | CALL_ENTRY(8), RETURN_TO(9), DEF_USE(10) stored on disk | **Current** |

**Breaking changes require:**
1. Increment `FEATURE_SCHEMA_VERSION` in `graph_schema.py`
2. Re-extract all graphs: `python ml/src/data_extraction/ast_extractor.py --force`
3. Rebuild cache: `python ml/scripts/create_cache.py`
4. Retrain model from scratch

## Usage Patterns

**Offline batch extraction:**
```python
# ast_extractor.py imports this module
from ml.src.preprocessing.graph_extractor import extract_contract_graph, GraphExtractionConfig

config = GraphExtractionConfig(
    solc_binary=solc_bin,
    solc_version=version,
    allow_paths=str(project_root),
)
graph = extract_contract_graph(Path(contract_path), config)
```

**Online inference:**
```python
# preprocess.py imports this module
from ml.src.preprocessing.graph_extractor import extract_contract_graph, GraphExtractionConfig

config = GraphExtractionConfig()  # Defaults for online (system solc)
graph = extract_contract_graph(Path(temp_file), config)
```

**Direct usage:**
```python
from ml.src.preprocessing import extract_contract_graph, GraphExtractionConfig

config = GraphExtractionConfig(multi_contract_policy="most_derived")
graph = extract_contract_graph("my_contract.sol", config)

print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
print(f"Feature shape: {graph.x.shape}")  # [N, 11]
print(f"Edge attr shape: {graph.edge_attr.shape}")  # [E]
```

## Testing

`ml/tests/test_preprocessing.py` validates:
- NodeType IntEnum has 13 members with correct values
- `len(FEATURE_NAMES) == NODE_FEATURE_DIM == 11`
- `len(EDGE_TYPES) == NUM_EDGE_TYPES == 11`
- `STRUCTURAL_PREFIX_TYPES` contains exactly 5 declaration types
- CFG nodes inherit features [1,3,4,5,9] from parent FUNCTION
- Feature builders return exactly 11 floats per node
- Visibility encoding produces values in {0.0, 0.5, 1.0}

## Dependencies

```toml
slither-analyzer = "^0.9.3"   # AST parsing, requires >=0.9.3 for NodeType.STARTUNCHECKED
torch-geometric = "^2.0"      # PyG Data structures
torch = "^2.0"                # Tensor operations
```

## Related Modules

- `ml/src/data_extraction/ast_extractor.py` — Batch orchestration using this module
- `ml/src/inference/preprocess.py` — Online inference using this module
- `ml/src/models/sentinel_model.py` — Consumes graph.x [N, 11] in GNNEncoder
- `ml/tests/test_preprocessing.py` — Schema validation tests
