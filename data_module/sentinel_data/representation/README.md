# `sentinel_data.representation` — Stage 2: Turning Source Code into Model-Consumable Artifacts

> **Status: ✅ Fully implemented (graph + tokens).** 3 deferred to v3.1 (call-graph, PDG, opcodes). Content-addressed cache + version registry prevent the "Run 8 v8-vs-v9 silent mix" failure mode.

## 1. Purpose

This is **the bridge between "source code on disk" and "tensors in GPU memory."** It takes preprocessed `.sol` files and produces:

1. **Graph representations** (`<sha256>.pt`) — PyG `Data` objects with node features, edge indices, and labels — the GNN's input
2. **Token representations** (`<sha256>.tokens.pt`) — windowed token sequences (`[max_windows, 512]`) for GraphCodeBERT — the Transformer's input
3. **Sidecar** (`<sha256>.rep.json`) — provenance, schema version, node/edge counts, compute time

This module is the most architecturally significant part of the pipeline because:

1. **It determines what the model can learn.** The graph schema (14 node types, 12 edge types, 12 features) is the model's vocabulary.
2. **It must be byte-identical to the old path.** The thin-adapter pattern ensures the new import produces the same output as the old `ml/src/preprocessing/` path.
3. **It must be cacheable and versionable.** Re-extracting 22K contracts takes hours; the content-addressed cache makes re-runs instant.

The v9 graph schema is **the source of truth for the model's classifier head**. The class order in `graph_schema.CLASS_NAMES` matches every existing Run 1–9 checkpoint — changing it would break all of them.

## 2. Source map

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 64 | Re-exports all public symbols from `graph_schema` + `graph_extractor`. |
| `graph_schema.py` | 148 | **Thin adapter** — re-exports v9 schema constants from `ml/src/preprocessing/graph_schema.py`. Defines `CLASS_NAMES` + `NUM_CLASSES` locally (10 classes, LOCKED to match existing checkpoints). |
| `graph_extractor.py` | 77 | **Thin adapter** — re-exports `extract_contract_graph` + error types from `ml/src/preprocessing/graph_extractor.py`. |
| `tokenizer.py` | 72 | **Thin adapter** — re-exports `tokenize_windowed_contract` from `ml/src/data_extraction/windowed_tokenizer.py` (NOT `tokenizer.py` — that one is single-window codebert-base, wrong). |
| `orchestrator.py` | 344 | **v2 manifest-driven orchestrator** — reads `data/preprocessed/<source>/<sha256>.meta.json`, runs graph_extractor + tokenizer, writes 3 files per contract. Honors content-addressed cache. |
| `cache_manager.py` | 119 | Content-addressed cache: `is_cached()`, `invalidate()`, `list_cached_sha256s()`, `stale_entries()`, `evict_stale()`. Cache key = `(sha256, schema_version, extractor_version)`. |
| `versioner.py` | 100 | Schema/extractor version registry at `data/representations/_version_registry.json`. Detects mismatches and evicts stale cache entries. |
| `cfg_builder.py` | 309 | **Opt-in** standalone CFG builder (`--emit-cfg`). Produces `CfgArtifact` JSON. Used by Stage 6 complexity_proxy_risk. |
| `call_graph.py` | 35 | **DEFERRED to v3.1** — placeholder. The Stage 2 ICFG-Lite (CALL_ENTRY/RETURN_TO edges in the main graph) already encodes the most important cross-function topology. |
| `opcode_extractor.py` | 36 | **DEFERRED to v3.1** — placeholder. Requires separate solc `--bin` compilation step. |
| `pdg_builder.py` | 34 | **DEFERRED to v3.1** — placeholder. Time better spent on Stage 4 verification. |

**Sub-total: 1,302 lines** across 11 files (3 of which are ~35-line v3.1 stubs).

## 3. Key concepts

### The v9 schema (LOCKED)

| Constant | Value | Source |
|----------|-------|--------|
| `FEATURE_SCHEMA_VERSION` | `"v9"` | `ml/src/preprocessing/graph_schema.py:161,175,218` (re-exported via `graph_schema.py:131`) |
| `NODE_FEATURE_DIM` | `12` | 12-dim per-node feature vector |
| `NUM_NODE_TYPES` | `14` | 14 distinct node types (`CFG_NODE_ARITH=13` added in v9) |
| `NUM_EDGE_TYPES` | `12` | 12 distinct edge types (`EXTERNAL_CALL=11` added in v9) |
| `_MAX_TYPE_ID` | `13.0` | Derived from `max(NODE_TYPES.values())` (re-exported for back-compat) |
| `NUM_CLASSES` | `10` | LOCKED |
| `EXTRACTOR_VERSION` | `"v2.1-windowed-gcb"` | Bumped when windowed tokenizer was added |

### The 10-class taxonomy (THE model order)

From `graph_schema.py:73-84`. **This is the order the model classifier head uses; do not change.**

| Idx | Class | Idx | Class |
|-----|-------|-----|-------|
| 0 | Reentrancy | 5 | DenialOfService |
| 1 | CallToUnknown | 6 | IntegerUO |
| 2 | Timestamp | 7 | UnusedReturn |
| 3 | ExternalBug | 8 | MishandledException |
| 4 | GasException | 9 | **NonVulnerable** |

> **⚠ The labeling taxonomy is DIFFERENT** — see `sentinel_data.labeling.schema.taxonomy.yaml`. The labeling one has `TransactionOrderDependence` at id=8 and `UnusedReturn` at id=9; no `NonVulnerable` slot. The model uses the representation order; everything else uses the labeling order. See `sentinel_data.labeling.schema/README.md` §3 for the full divergence.

### The thin-adapter pattern (CRITICAL)

`graph_schema.py`, `graph_extractor.py`, and `tokenizer.py` are **thin adapters** — they re-export the actual implementation from `ml/`:

```python
# sentinel_data/representation/graph_schema.py (simplified)
from ml.src.preprocessing.graph_schema import (
    FEATURE_SCHEMA_VERSION, NODE_FEATURE_DIM, NUM_NODE_TYPES, NUM_EDGE_TYPES,
    VISIBILITY_MAP, NODE_TYPES, EDGE_TYPES, FEATURE_NAMES, NodeType,
    STRUCTURAL_PREFIX_TYPES,
)
_MAX_TYPE_ID = float(max(NODE_TYPES.values()))   # derived, re-exported
CLASS_NAMES = [...]                                # defined locally
NUM_CLASSES = len(CLASS_NAMES)
```

**Why thin-adapter:**

- **Single source of truth** — bug fixes in `ml/` automatically propagate to the new path. `from sentinel_data.representation.graph_schema import NODE_TYPES` returns the **same dict object** as `from ml.src.preprocessing.graph_schema import NODE_TYPES` (`is` equality holds).
- **Byte-identical output** — same code, different import name. The `test_byte_identical_regression.py` test enforces this.
- **1-line seam swap in Stage 7** — delete the wrappers, rebind the import.
- **No `sys.path` hacks** — clean package boundaries (apart from the `cli.py` sys.path bootstrap).

`tokenizer.py` was the one that got wrong-imported early: it previously pointed at `ml/src/data_extraction/tokenizer.py` (codebert-base, single-window, wrong). It now correctly points at `ml/src/data_extraction/windowed_tokenizer.py` (graphcodebert-base, `[4,512]` sliding-window). See `tokenizer.py:11-15` for the bug history.

### Lazy import support (`__getattr__`)

Each thin adapter has a `__getattr__` that falls back to `importlib.import_module(_LIVE_*)` if `ml/` is not on `PYTHONPATH`. This lets `sentinel-data` be installed as a standalone PyPI package in the future without `ml/`. The eager `from ml... import ...` at the bottom of the file is the normal path; the lazy `__getattr__` is the fallback.

### The v2 orchestrator (`orchestrator.py:249-343`)

```python
def represent_source(
    source: str, cfg: dict, data_dir: Path,
    *, dry_run: bool = False, force: bool = False, limit: int | None = None,
    output_dir: Path | None = None, emit_cfg: bool = False,
) -> RepresentResult:
    """Run the representation pipeline for one source.
    
    Reads data/preprocessed/<source>/<sha256>.meta.json + .sol pairs,
    runs the graph_extractor + tokenizer on each, writes outputs to
    data/representations/<source>/<sha256>.{pt, tokens.pt, rep.json}.
    """
```

Per-contract flow (`_extract_one`, `orchestrator.py:112-233`):
1. Load companion `.meta.json` (Stage 1 sidecar).
2. Cache check — if `.rep.json` exists with matching `(schema_version, extractor_version)`, skip.
3. Resolve solc binary from `meta.solc_version` (uses `~/.solc-select/artifacts/solc-<v>/solc-<v>`).
4. Build `GraphExtractionConfig(allow_paths=[...], solc_binary=..., solc_version=...)`.
5. Call `extract_contract_graph(sol_path, config)`.
6. Call `tokenize_windowed_contract(sol_path)`.
7. Save `<sha256>.pt` (graph), `<sha256>.tokens.pt` (tokens), `<sha256>.rep.json` (sidecar).
8. If `--emit-cfg`, additionally call `build_cfg(...)` and save `<sha256>.cfg.json`.

### Content-addressed cache (`cache_manager.py` + `versioner.py`)

The cache is keyed on **3 values**: `(sha256_of_source, schema_version, extractor_version)`. Change any of the three → cache miss for that file. The cache lives in `data/representations/<source>/` as 3 files per contract.

The `versioner.py` module maintains `data/representations/_version_registry.json`:
```json
{
  "schema_version": "v9",
  "extractor_version": "v2.1-windowed-gcb",
  "updated_at": "2026-06-08T..."
}
```

On each `represent_source` call, `check_and_evict()` (line 66) compares the registry to the live versions; on mismatch, it evicts stale cache entries for the source before starting. This is the **fix for the Run 8 "v8 graphs mixed with v9 graphs" silent-failure mode**.

### Standalone CFG builder (`cfg_builder.py`, opt-in via `--emit-cfg`)

Not merged into the training graph. Produces a JSON-serializable `CfgArtifact` per contract:

```python
@dataclass
class CfgArtifact:
    sha256: str
    source: str
    solc_version: str
    schema_version: str
    extractor_version: str
    functions: list[CfgFunction]   # per-function CFG
    error: Optional[str]
    
@dataclass
class CfgFunction:
    canonical_name: str           # Slither's stable cross-run identifier
    nodes: list[CfgNode]
    edges: list[CfgEdge]
    num_loops: int                # DFS back-edge count
    max_depth: int                # BFS longest path from ENTRYPOINT
```

**Consumed by Stage 6's `feature_dist.complexity_proxy_risk`**, which needs a flat CFG view to spot loops, dead code, and deeply nested branches. NOT used for training in Run 11.

The CFG adds ~0.5s per contract. Default off; pass `--emit-cfg` to enable.

### v3.1 deferred placeholders

`call_graph.py:25-34`, `opcode_extractor.py:26-35`, `pdg_builder.py:24-33` all `raise NotImplementedError`. They exist to reserve the directory structure. The Stage 2 ICFG-Lite (CALL_ENTRY/RETURN_TO edges in the main graph) already encodes the most important cross-function topology for training.

## 4. Public API

### `graph_schema` exports — `graph_schema.py:148`

```python
FEATURE_SCHEMA_VERSION    # "v9"
NODE_FEATURE_DIM          # 12
NUM_NODE_TYPES            # 14
NUM_EDGE_TYPES            # 12
_MAX_TYPE_ID              # 13.0 (derived)
NUM_CLASSES               # 10 (LOCKED)
VISIBILITY_MAP            # dict
NODE_TYPES                # dict (int IDs for 14 node types)
EDGE_TYPES                # dict (int IDs for 12 edge types)
FEATURE_NAMES             # list of 12 feature names
CLASS_NAMES               # list of 10 class names (model order)
NodeType                  # type alias
STRUCTURAL_PREFIX_TYPES   # for K=48 GNN prefix injection
```

### `graph_extractor` exports — `graph_extractor.py:77`

```python
extract_contract_graph(sol_path: str, config: GraphExtractionConfig = None) -> Data
GraphExtractionConfig(...)
GraphExtractionError, SolcCompilationError, SlitherParseError, EmptyGraphError
```

### `tokenizer` exports — `tokenizer.py:72`

```python
tokenize_windowed_contract(contract_path: str, max_windows: int = 4) -> dict | None
init_worker()
TOKENIZER_MODEL   # "microsoft/graphcodebert-base"
WINDOW_SIZE       # 512
STRIDE            # 256
MAX_WINDOWS       # 4
```

### `orchestrator` exports — `orchestrator.py:344`

```python
@dataclass
class RepresentResult:
    source: str
    contracts_seen: int = 0
    graphs_written: int = 0
    graphs_cached: int = 0
    graphs_failed: int = 0
    tokens_written: int = 0
    tokens_cached: int = 0
    tokens_failed: int = 0
    duration_s: float = 0.0
    schema_version: str = ""
    extractor_version: str = "v2.0-thin-adapter"

def represent_source(source, cfg, data_dir, *, dry_run=False, force=False, limit=None,
                     output_dir=None, emit_cfg=False) -> RepresentResult: ...

EXTRACTOR_VERSION = "v2.1-windowed-gcb"
```

### `cache_manager` exports — `cache_manager.py:119`

```python
is_cached(output_dir, sha256, schema_version, extractor_version) -> bool
record_cache_hit(output_dir, sha256) -> None  # placeholder for v3.1 LRU
invalidate(output_dir, sha256) -> None
list_cached_sha256s(output_dir) -> list[str]
stale_entries(output_dir, schema_version, extractor_version) -> list[str]
evict_stale(output_dir, schema_version, extractor_version) -> int
```

### `versioner` exports — `versioner.py:100`

```python
read_registry(representations_root) -> dict
write_registry(representations_root, schema_v, extractor_v) -> None
check_and_evict(representations_root, source, schema_v, extractor_v) -> int
current_versions() -> tuple[str, str]   # (FEATURE_SCHEMA_VERSION, EXTRACTOR_VERSION)
```

### `cfg_builder` exports — `cfg_builder.py:309`

```python
build_cfg(sol_path, config, sha256="", source="") -> CfgArtifact
# Plus CfgArtifact / CfgFunction / CfgNode / CfgEdge dataclasses
```

## 5. Inputs → outputs

| Input | Where | What |
|-------|-------|------|
| `data/preprocessed/<source>/<sha256>.sol` | Stage 1b output | Cleaned + normalized Solidity |
| `data/preprocessed/<source>/<sha256>.meta.json` | Stage 1b sidecar | Provides `sha256` (cache key), `solc_version` (binary resolution), `original_path` (provenance) |
| `~/.solc-select/artifacts/solc-<v>/solc-<v>` | System | solc binaries (must match the contract's pragma) |
| `ml/src/preprocessing/graph_schema.py` + `graph_extractor.py` | Repo | The v9 implementation (thin-adapter target) |
| `ml/src/data_extraction/windowed_tokenizer.py` | Repo | The windowed GraphCodeBERT tokenizer (thin-adapter target) |

| Output | Where | What |
|--------|-------|------|
| `data/representations/<source>/<sha256>.pt` | `orchestrator.py:185` | PyG `Data` object — the GNN input |
| `data/representations/<source>/<sha256>.tokens.pt` | `orchestrator.py:186-196` | dict with `input_ids [4, 512]`, `attention_mask [4, 512]`, `num_windows`, `stride`, etc. |
| `data/representations/<source>/<sha256>.rep.json` | `orchestrator.py:215` | Sidecar: `sha256`, `source`, `original_path`, `schema_version`, `extractor_version`, `node_count`, `edge_count`, `window_count`, `compute_time_ms`, `cache_hit`, `pragma`, `solc_version` |
| `data/representations/<source>/<sha256>.cfg.json` | `orchestrator.py:229` | OPT-IN. Standalone CFG artifact for Stage 6. |
| `data/representations/_version_registry.json` | `versioner.py:62` | Schema + extractor version pin |

## 6. Pipeline interactions

| Stage | Direction | What |
|-------|-----------|------|
| Stage 1b (preprocessing) | ← | Reads `data/preprocessed/<source>/*.meta.json` |
| `ml/src/preprocessing/graph_schema.py` + `graph_extractor.py` | ← | Thin-adapter target (canonical source of truth) |
| `ml/src/data_extraction/windowed_tokenizer.py` | ← | Thin-adapter target (windowed tokenizer) |
| Stage 3 (labeling) | — | Doesn't read this stage; labeling reads `meta.json` from Stage 1b |
| Stage 4 (verification) | → | `semantic_checker.py`, `tool_validator.py`, `fp_estimator.py`, `negative_checker.py` all load `data/representations/<source>/<sha256>.pt` and `.rep.json` |
| Stage 5 (splitting) | → | Reads `data/representations/<source>/<sha256>.rep.json` for `node_count` and `edge_count` (used by `feature_dist` and `drift_monitor`) |
| Stage 6 (analysis) | → | `feature_dist._features_for_contract` and `drift_monitor._iter_label_features` read `.rep.json` |
| `ml/` training (SentinelDataset) | → | Reads `<sha256>.pt` and `<sha256>.tokens.pt` (post-seam-swap) |

## 7. Tests

**Location:** `data_module/tests/test_representation/` (no `__init__.py` — pytest still discovers)
- `test_thin_adapter.py` — `is` equality between `sentinel_data.X` and `ml.X` symbols (the byte-identical guarantee)
- `test_byte_identical_regression.py` — Schema-dim gate (`x.shape[-1] == NODE_FEATURE_DIM == 12`) + `old is new` guard
- `test_orchestrator.py` — Full orchestrator flow with synthetic Solidity
- `test_solidifi_fixes.py` — A-1, A-2, A-3 fix regression tests
- `test_13_issue_preservation.py` — 13 bug-fix regression tests

**Command:**
```bash
cd ~/projects/sentinel/data_module
poetry run pytest tests/test_representation/ -v
```

## 8. See also

- Previous stage: `sentinel_data/preprocessing/README.md`
- Next stage: `sentinel_data/labeling/README.md`
- CLI entry: `sentinel_data/cli.py` (`_run_represent` at line 160)
- The actual schema: `ml/src/preprocessing/graph_schema.py`
- The actual extractor: `ml/src/preprocessing/graph_extractor.py`
- The actual tokenizer: `ml/src/data_extraction/windowed_tokenizer.py`
- Thin-adapter design rationale: `docs/ml/adr/0007-representation-port-design.md`
- Critical v2 facts (do not re-derive): MEMORY.md §"Model Architecture (v9 schema — Run 9 current)"
- A-1 fix (strip_comments in tokenizer): `tests/test_representation/test_solidifi_fixes.py`
- Run 8 silent mix of v8/v9 graphs: `project_run8_audit_findings.md` (caused `versioner.py` to exist)
