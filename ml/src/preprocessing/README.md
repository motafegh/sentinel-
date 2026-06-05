# preprocessing — SENTINEL Graph Feature Extraction Pipeline

Solidity source → PyG `Data` object. The single authoritative AST-to-graph conversion used by both the **offline batch pipeline** (`ml/scripts/reextract_graphs.py`, ~41K training graphs) and the **online inference API** (`ml/src/inference/preprocess.py`, one contract per request).

---

## Module Map

| File | Purpose |
|------|---------|
| `graph_schema.py` | Single source of truth for all graph constants — node types, edge types, feature layout, schema version |
| `graph_extractor.py` | Canonical Solidity-to-PyG graph extraction: Slither analysis → node/edge construction → `Data` object |
| `__init__.py` | Re-exports all schema constants, the extraction function, and exception classes |

---

## Quick Start

```python
from ml.src.preprocessing import extract_contract_graph, GraphExtractionConfig

# Default config — picks the most-derived contract in multi-contract files
config = GraphExtractionConfig(multi_contract_policy="most_derived")
graph = extract_contract_graph(Path("contract.sol"), config)

# graph is a PyG Data object:
#   graph.x           [N, 11]  float32  — node feature matrix
#   graph.edge_index  [2, E]   int64    — COO edge connectivity
#   graph.edge_attr   [E]      int64    — edge type IDs
#   graph.contract_name  str            — analysed contract name
#   graph.num_nodes      int            — N
#   graph.num_edges      int            — E
#   graph.has_cei_path   int            — 0/1 CEI violation label
#   graph.node_metadata  list[dict]     — index-aligned {name, type, source_lines}
```

---

## graph_schema.py — Single Source of Truth

Every constant governing the node feature vector, node types, edge types, visibility encoding, and schema version lives here. Any change requires re-extraction of all training graphs and model retraining.

### Schema Version

| Constant | Value | Purpose |
|----------|-------|---------|
| `FEATURE_SCHEMA_VERSION` | `"v8"` | Appended to inference cache keys; bumping invalidates stale caches |

### Structural Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `NODE_FEATURE_DIM` | `11` | Scalar features per graph node (v8 schema) |
| `NUM_NODE_TYPES` | `13` | Distinct node type IDs (0–12) |
| `NUM_EDGE_TYPES` | `11` | Distinct edge relation types (IDs 0–10) |

### Feature Layout (v8 — 11 dimensions)

| Index | Name | Semantics | Range |
|-------|------|-----------|-------|
| 0 | `type_id` | `NODE_TYPES[kind]/12.0`, normalised | [0, 1] |
| 1 | `visibility` | `VISIBILITY_MAP` ordinal encoding | {0.0, 0.5, 1.0} |
| 2 | `uses_block_globals` | 1.0 if reads block.timestamp/number/etc. | {0.0, 1.0} |
| 3 | `view` | Read-only state | {0.0, 1.0} |
| 4 | `payable` | Ether entry point | {0.0, 1.0} |
| 5 | `complexity` | `log1p(CFG blocks)/log1p(100)`, normalised | [0, 1] |
| 6 | `loc` | `log1p(lines)/log1p(1000)`, normalised | [0, 1] |
| 7 | `return_ignored` | 0.0=captured / 1.0=discarded / -1.0=IR unavailable | {-1.0, 0.0, 1.0} |
| 8 | `call_target_typed` | 0.0=raw addr / 1.0=typed / -1.0=source unavailable | {-1.0, 0.0, 1.0} |
| 9 | `has_loop` | Function contains a loop | {0.0, 1.0} |
| 10 | `external_call_count` | `log1p(n)/log1p(20)`, includes Transfer/Send | [0, 1] |

### Node Types

Declaration-level node types (v1 — stable IDs 0–7):

| ID | Name | Description |
|----|------|-------------|
| 0 | `STATE_VAR` | Contract state variable declaration |
| 1 | `FUNCTION` | Function declaration (includes free functions) |
| 2 | `MODIFIER` | Function modifier declaration |
| 3 | `EVENT` | Event declaration |
| 4 | `FALLBACK` | Fallback function (unnamed, receives plain ETH) |
| 5 | `RECEIVE` | Receive function (ETH-only entry) |
| 6 | `CONSTRUCTOR` | Constructor function |
| 7 | `CONTRACT` | Top-level contract node (root of the graph) |

CFG subtypes (v2 — IDs 8–12):

| ID | Name | Description |
|----|------|-------------|
| 8 | `CFG_NODE_CALL` | Statement containing an external call (highest priority) |
| 9 | `CFG_NODE_WRITE` | Statement writing a state variable |
| 10 | `CFG_NODE_READ` | Statement reading a state variable |
| 11 | `CFG_NODE_CHECK` | require / assert / if condition |
| 12 | `CFG_NODE_OTHER` | All other statement types (synthetic nodes, etc.) |

When a single IR node spans multiple operations, `_cfg_node_type()` assigns the **highest-priority** type: CALL > WRITE > READ > CHECK > OTHER.

### Edge Types

| ID | Name | Phase | On Disk? | Description |
|----|------|-------|----------|-------------|
| 0 | `CALLS` | Phase 1 | Yes | Function → internally-called function |
| 1 | `READS` | Phase 1 | Yes | Function → state variable it reads |
| 2 | `WRITES` | Phase 1 | Yes | Function → state variable it writes |
| 3 | `EMITS` | Phase 1 | Yes | Function → event it emits |
| 4 | `INHERITS` | Phase 1 | Yes | Contract → parent contract (linearised MRO) |
| 5 | `CONTAINS` | Phase 1 & 3 | Yes | Function node → its CFG_NODE children |
| 6 | `CONTROL_FLOW` | Phase 2 | Yes | CFG_NODE → successor CFG_NODE, directed |
| 7 | `REVERSE_CONTAINS` | Phase 3 | **No — runtime only** | CFG_NODE → parent function (generated in GNNEncoder) |
| 8 | `CALL_ENTRY` | Phase 2 | Yes (v8) | Calling CFG_NODE → ENTRYPOINT of callee function |
| 9 | `RETURN_TO` | Phase 2 | Yes (v8) | Terminal CFG_NODE of callee → successor of the call site |
| 10 | `DEF_USE` | Phase 2 | Yes (v8) | CFG_NODE defining a LocalVariable → CFG_NODE reading it |

### NodeType IntEnum

```python
from ml.src.preprocessing.graph_schema import NodeType

# Typed aliases for NODE_TYPES integer IDs — always use these instead of raw integers
mask = node_type_ids == NodeType.CFG_NODE_CALL   # not == 8
```

All 13 members (`STATE_VAR` through `CFG_NODE_OTHER`) mirror `NODE_TYPES` values. Derived at module load time so they cannot drift.

### STRUCTURAL_PREFIX_TYPES

```python
STRUCTURAL_PREFIX_TYPES: frozenset[NodeType] = frozenset({
    NodeType.FUNCTION, NodeType.MODIFIER, NodeType.CONSTRUCTOR,
    NodeType.FALLBACK, NodeType.RECEIVE,
})
```

Node types eligible for GNN prefix injection (Phase 1). Declaration-level only because after Phase 3 REVERSE_CONTAINS, FUNCTION nodes carry aggregated CFG signal — CFG nodes would inflate the prefix budget for minimal additional signal. Statistics: mean=20.3 declaration nodes per contract, P95=47 → K=48 covers 95.5%.

### VISIBILITY_MAP

```python
VISIBILITY_MAP = {"public": 0.0, "external": 0.0, "internal": 0.5, "private": 1.0}
```

Normalised ordinal encoding preserving private > internal > public ordering. Changed from raw int encoding `{0,1,2}` in v6 (BUG-3 fix — private=2 exceeded [0,1] feature range for 17.7% of graphs).

### Import-Time Assertions

Four assertions fire at import time to catch schema drift:
1. `len(FEATURE_NAMES) == NODE_FEATURE_DIM`
2. `len(EDGE_TYPES) == NUM_EDGE_TYPES`
3. `len(NODE_TYPES) == 13`
4. `max(NODE_TYPES.values()) == 12`

If any assertion fails, the import itself fails — preventing silent misalignment between schema constants and model expectations.

### Slither Version Guard

At import time, asserts `slither-analyzer >= 0.9.3` for `NodeType.STARTUNCHECKED` support. Raises `RuntimeError` if too old; passes silently if Slither is not installed (inference-only deployment).

---

## graph_extractor.py — Canonical Solidity-to-PyG Graph Extraction

### Exception Hierarchy

```
GraphExtractionError (base)
├── SolcCompilationError    — user error (bad Solidity) → HTTP 400
├── SlitherParseError       — infra error (Slither bug) → HTTP 500
└── EmptyGraphError         — zero AST nodes extracted
```

### GraphExtractionConfig (dataclass)

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `multi_contract_policy` | `str` | `"most_derived"` | Which contract to pick in multi-contract files |
| `target_contract_name` | `str\|None` | `None` | Used only with `"by_name"` policy |
| `include_edge_attr` | `bool` | `True` | Attach `graph.edge_attr [E]` int64 |
| `solc_binary` | `str\|Path\|None` | `None` | Override solc binary path |
| `solc_version` | `str\|None` | `None` | Solidity version string |
| `allow_paths` | `str\|None` | `None` | `--allow-paths` for solc compilation |

**Contract selection policies:**

| Policy | Logic | Accuracy |
|--------|-------|----------|
| `"most_derived"` (default) | Picks contract that inherits from the most other in-file candidates | ~92%+ |
| `"last"` | Last-defined contract in the file | ~87.4% |
| `"most_funcs"` | Contract with the most functions (legacy) | ~52.6% — **avoid** |
| `"by_name"` | Exact name match via `target_contract_name` | 100% when name given |

### Public API: `extract_contract_graph()`

```python
def extract_contract_graph(
    sol_path: Path,
    config: GraphExtractionConfig | None = None,
) -> Data
```

**Pipeline steps:**

1. Verify Slither is installed (raises `RuntimeError` if not)
2. Build solc_args from config
3. Instantiate Slither (routes errors: `SolcCompilationError` vs `SlitherParseError`)
4. Select target contract via `_select_contract()`
5. Add declaration nodes in fixed order: CONTRACT → parent CONTRACTs → STATE_VARs → MODIFIERs → EVENTs
6. For each function: add FUNCTION node, then CFG_NODE children via `_build_control_flow_edges()`
7. Add ICFG cross-function edges via `_add_icfg_edges()` (CALL_ENTRY + RETURN_TO)
8. Add DEF_USE data-flow edges via `_add_def_use_edges()`
9. Add declaration-level edges (CALLS, READS, WRITES, EMITS, INHERITS)
10. Contract-size normalization of complexity (dim 5)
11. OOR feature validation (logs warning, does not raise)
12. Compute `has_cei_path` label via BFS
13. Assemble PyG `Data` object

**Returned `Data` attributes:**

| Attribute | Shape/Type | Description |
|-----------|------------|-------------|
| `x` | `[N, 11]` float32 | Node feature matrix |
| `edge_index` | `[2, E]` int64 | COO edge connectivity |
| `edge_attr` | `[E]` int64 | Edge type IDs (if `config.include_edge_attr`) |
| `node_metadata` | `list[dict]` | Index-aligned dicts: `{name, type, source_lines}` |
| `contract_name` | `str` | Name of analysed contract |
| `num_nodes` | `int` | N |
| `num_edges` | `int` | E |
| `has_cei_path` | `int` | 0/1 CEI violation label |

### Feature Computation Helpers (module-private)

These functions compute individual feature dimensions from Slither IR analysis:

| Function | Returns | Description |
|----------|---------|-------------|
| `_compute_return_ignored(func)` | 0.0/1.0/-1.0 | Checks if any external call's return is discarded (sequential IR scan) |
| `_compute_call_target_typed(func)` | 0.0/1.0/-1.0 | Whether all calls use typed interfaces vs raw address (with source-scan fallback) |
| `_compute_has_loop(func)` | 0.0/1.0 | Detects loop constructs via Slither `NodeType` or `is_loop_present` |
| `_compute_external_call_count(func)` | [0,1] | `log1p(n)/log1p(20)` including Transfer/Send ops |
| `_compute_uses_block_globals(func)` | 0.0/1.0 | Checks if IR reads block.timestamp/number/difficulty/basefee/prevrandao |

### CFG Node Helpers (module-private)

| Function | Returns | Description |
|----------|---------|-------------|
| `_cfg_node_type(slither_node)` | 8–12 | Classifies CFG node by priority: CALL > WRITE > READ > CHECK > OTHER |
| `_build_cfg_node_features(node, func, cfg_type, parent_features)` | 11 floats | Builds CFG node feature vector; inherits function-scoped dims from parent |
| `_build_control_flow_edges(func, ...)` | tuple | Builds CONTAINS(5) and CONTROL_FLOW(6) edges for one function |
| `_add_icfg_edges(contract, ...)` | — | Adds CALL_ENTRY(8) and RETURN_TO(9) cross-function edges |
| `_add_def_use_edges(contract, ...)` | — | Adds DEF_USE(10) data-flow edges with two-tier scope |
| `_compute_has_cei_path(metadata, edge_index, edge_attr)` | 0/1 | BFS from CFG_NODE_CALL to CFG_NODE_WRITE via CONTROL_FLOW |

### CFG Node Feature Inheritance

CFG nodes inherit specific feature dimensions from their parent FUNCTION node because CFG statements share the function's scope-level properties. Inherited dimensions: `visibility` [1], `view` [3], `payable` [4], `complexity` [5], `has_loop` [9]. Statement-level features (`return_ignored`, `call_target_typed`, `uses_block_globals`, `external_call_count`) are computed per-CFG-node independently.

### ICFG-Lite: Cross-Function Edges

The v8 schema adds two inter-procedural edge types that connect caller and callee CFG nodes without requiring full interprocedural analysis:

- **CALL_ENTRY(8)**: The calling CFG node → the ENTRYPOINT of the callee function. This creates a direct link from the call site to the beginning of the called function's control flow.
- **RETURN_TO(9)**: The terminal CFG node of the callee → the successor of the call site in the caller. This closes the control-flow loop, enabling the model to trace execution through a call and back.

Terminal nodes are identified by CFG nodes with no outgoing CONTROL_FLOW edges (i.e., the last statement(s) in a function). Edge creation is limited to functions within the same contract (no cross-contract edges).

### DEF_USE Data-Flow Edges

**DEF_USE(10)** connects a CFG node that **defines** (writes) a local variable to CFG nodes that **use** (read) it later in the same function's control flow. This captures how values computed by arithmetic operations, call returns, and state reads flow into checks and state writes — crucial for integer overflow and return-value-ignored patterns.

Two-tier scope prevents false edges:
- **Tier 1 (same function)**: Standard intra-function def-use chains
- **Tier 2 (cross-function via parameters)**: When a variable is passed as a function argument, a DEF_USE edge connects the caller's argument definition to the callee's parameter usage

### CEI Label Computation

`_compute_has_cei_path()` performs BFS from every `CFG_NODE_CALL` node to every `CFG_NODE_WRITE` node following `CONTROL_FLOW` edges (max 8 hops). A contract is labelled `has_cei_path=1` if any such path exists — indicating a potential Checks-Effects-Interactions violation where an external call precedes a state write.

---

## Change Policy

Any modification to `NODE_TYPES`, `VISIBILITY_MAP`, `EDGE_TYPES`, or `FEATURE_NAMES` requires **all** of the following steps:

1. **Rebuild all ~41K .pt graph files:**
   ```bash
   python ml/scripts/reextract_graphs.py
   ```
2. **Rebuild token .pt files** (only if tokenizer logic changed):
   ```bash
   python ml/scripts/retokenize_windowed.py
   ```
3. **Retrain the model from scratch:**
   ```bash
   python ml/scripts/train.py
   ```
   (`GNNEncoder` reads `in_channels=NODE_FEATURE_DIM` at construction time)
4. **Increment `FEATURE_SCHEMA_VERSION`** to invalidate all inference caches:
   ```python
   FEATURE_SCHEMA_VERSION = "v9"  # next increment — currently v8
   ```

Skipping any of these steps will cause silent accuracy regression.

---

## Schema Version History

| Version | Changes |
|---------|---------|
| **v1** | 8 features, 5 edge types, 8 node types |
| **v2** | 12 features (+5 semantic), 7 edge types (+CONTAINS, +CONTROL_FLOW), 13 node types (+5 CFG subtypes) |
| **v4** | Changed feature semantics: pure→uses_block_globals, raw loc→log-norm loc, Transfer/Send in ext_call_count |
| **v5** | BUG-1/2/6/9 fixes: log-norm complexity, most_derived contract selection, Send return-ignored |
| **v6** | BUG-3: visibility normalised to [0,1] (was raw int 0/1/2) |
| **v7** | 11 features (dropped dead `in_unchecked`), EMITS/INHERITS fire, CFG nodes inherit parent features, REVERSE_CONTAINS(7) added |
| **v8** | 11 features, 11 edge types (+CALL_ENTRY, RETURN_TO, DEF_USE), ICFG-Lite cross-function edges, data-flow edges |

---

## Cross-Module Dependencies

```
graph_schema.py  ←──  graph_extractor.py  ←──  __init__.py
    (constants)       (imports schema       (re-exports
                       constants; defines    everything
                       extraction logic)     from both)
```

```
graph_schema.py ──→ (consumed by) ──→ gnn_encoder.py     (NODE_TYPES, EDGE_TYPES, NODE_FEATURE_DIM)
                                      sentinel_model.py   (_MAX_TYPE_ID assert, NODE_TYPES)
                                      inference/cache.py  (FEATURE_SCHEMA_VERSION for cache keys)
                                      inference/predictor.py (NODE_FEATURE_DIM, NODE_TYPES)
                                      training/trainer.py
```

```
graph_extractor.py ──→ (called by) ──→ ml/scripts/reextract_graphs.py     (offline batch)
                                         ml/src/inference/preprocess.py    (online inference)
```

No circular dependencies exist. `graph_schema.py` has no imports from `graph_extractor.py`. The dependency flows one way.
