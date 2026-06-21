# ml/src/preprocessing — Graph Schema and Extraction

Graph schema constants (re-exported from shared `sentinel_data` package) and the Slither-based graph extractor.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `graph_schema.py` | 22 | Thin re-export shim from `sentinel_data.representation.graph_schema` |
| `graph_extractor.py` | 2056 | `extract_contract_graph()` — Slither AST/CFG -> PyG Data |
| `__init__.py` | 0 | Empty |

---

## graph_schema.py

Re-exports all schema constants from `sentinel_data.representation.graph_schema`:
- `FEATURE_SCHEMA_VERSION` (= `"v9"`)
- `NODE_FEATURE_DIM` (= `12`)
- `NUM_NODE_TYPES` (= `14`)
- `NUM_EDGE_TYPES` (= `12`)
- `NODE_TYPES` — dict mapping type name -> integer ID (0-13)
- `EDGE_TYPES` — dict mapping edge name -> integer ID (0-11)
- `VISIBILITY_MAP` — visibility string -> float
- `FEATURE_NAMES` — list of human-readable feature names
- `CLASS_NAMES` — list of vulnerability class names
- `NUM_CLASSES` — number of classes
- `_MAX_TYPE_ID` — max node type ID
- `NodeType` — enum
- `STRUCTURAL_PREFIX_TYPES` — list of types eligible for GNN prefix

**To change schema constants:** Edit `sentinel_data/representation/graph_schema.py`, then bump `FEATURE_SCHEMA_VERSION`.

---

## graph_extractor.py

### extract_contract_graph()

The canonical graph extraction function. Converts a Solidity source file into a PyG `Data` object.

**Inputs:** `.sol` file path + `GraphExtractionConfig`

**Outputs:** PyG `Data` with:
- `x: [N, 12]` — node features (v9 schema)
- `edge_index: [2, E]` — graph connectivity
- `edge_attr: [E]` — edge type IDs (0-11)
- `node_metadata: list[dict]` — per-node metadata (name, source_lines, etc.)

**Dependencies:** Requires `slither-analyzer` and a compatible `solc` binary.

**Configuration:**
```python
GraphExtractionConfig(
    solc_version="0.8.31",
    solc_binary=Path("..."),  # optional override
)
```

**Exception hierarchy:**
- `GraphExtractionError` (base)
  - `SolcCompilationError` — invalid Solidity (maps to HTTP 400)
  - `EmptyGraphError` — no analyzable contract nodes (maps to HTTP 400)
  - `SlitherParseError` — Slither infrastructure failure (maps to HTTP 500)

**Node types (14):**
| ID | Name | Description |
|----|------|-------------|
| 0 | STATE_VAR | State variable |
| 1 | FUNCTION | Function declaration |
| 2 | MODIFIER | Modifier declaration |
| 3 | FALLBACK | Fallback function |
| 4 | RECEIVE | Receive function |
| 5 | CONSTRUCTOR | Constructor |
| 6 | CFG_NODE_CALL | CFG: function call |
| 7 | CFG_NODE_WRITE | CFG: state write |
| 8 | CFG_NODE_READ | CFG: state read |
| 9 | CFG_NODE_CHECK | CFG: conditional check |
| 10 | CFG_NODE_OTHER | CFG: other operations |
| 11 | EVENT | Event declaration |
| 12 | STRUCT | Struct definition |
| 13 | CONTRACT | Contract declaration |

**Edge types (12):**
| ID | Name | Direction |
|----|------|-----------|
| 0 | CALLS | function -> function |
| 1 | READS | function -> state_var |
| 2 | WRITES | function -> state_var |
| 3 | EMITS | function -> event |
| 4 | INHERITS | contract -> contract |
| 5 | CONTAINS | parent -> child |
| 6 | CONTROL_FLOW | CFG block -> CFG block |
| 7 | REVERSE_CONTAINS | runtime-only flip of type 5 |
| 8 | CALL_ENTRY | call site -> function entry |
| 9 | RETURN_TO | function exit -> call-site continuation |
| 10 | DEF_USE | definition -> use |
| 11 | EXTERNAL_CALL | CFG call -> external target |

**Feature normalization:**
- type_id stored as `float(id) / 13.0` (dim[0])
- complexity: `log1p(cfg_block_count) / log1p(100)` (dim[5])
- loc: `log1p(lines) / log1p(1000)` (dim[6])
- external_call_count: `log1p(count) / log1p(20)` (dim[10])

CFG nodes inherit visibility, view, payable, complexity, has_loop from parent FUNCTION.
