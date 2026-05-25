# Session Log — Learning With Claude

Chronological record of all teaching sessions. Never delete — only append.
Each entry: what was taught, what was answered, gaps closed, audit flags raised.

---

## Session 1 — Phase 1: `graph_schema.py` + `hash_utils.py`

**Files covered:**
- `ml/src/preprocessing/graph_schema.py`
- `ml/src/utils/hash_utils.py`

**Concepts taught:**
- Single source of truth pattern — why feature drift between training and inference is a killer bug
- `FEATURE_SCHEMA_VERSION` — how it invalidates inference caches on schema change
- Node feature vector (11 dims, v7 schema) — each feature's purpose and normalization rationale
- Log normalization (`log1p`) — why raw values dominate dot products and how log fixes it
- Sentinel values (`-1.0`) — encoding "unknown" vs "safe" vs "unsafe" in a float
- Node types (13) — declaration-level vs CFG-level, priority ordering
- `NodeType` IntEnum — type safety over raw ints
- `STRUCTURAL_PREFIX_TYPES` — why declaration-level only for GNN prefix injection
- Edge types (11) — 3-phase routing, REVERSE_CONTAINS runtime-only invariant
- Compile-time assertions — fail-fast at import vs silent shape mismatch in training loop
- MD5 path hashing — deterministic file pairing, why path not content for training data
- Content hashing — inference cache key design (`content_hash + schema_version`)
- `validate_hash` — int vs regex, permissiveness tradeoff

**Challenge questions answered:** Q1–Q6

**Gaps found and closed:**
- Why hardcoding schema constants is safer than dynamic computation (train/inference mismatch)
- `nn.Embedding` shape contract: `[E,1]` produces `[E,1,d]` (3D) → shape mismatch in GATConv
- REVERSE_CONTAINS serialization: duplication (not conflict) → 2× edges → silent graph corruption
- Path hashing staleness + content hashing write-collision problem
- Graph-level vs node-level features: why GNN needs local signals for structural pattern detection
- `int(hex, 16)` accepts uppercase; `hexdigest()` always lowercase → validator is over-permissive

**Audit flags raised:** A1, A2

---

## Session 2 — Phase 2: `graph_extractor.py` (Chunk 1)

**File:** `ml/src/preprocessing/graph_extractor.py` (lines 1–213)

**Concepts taught:**
- Module purpose: canonical Solidity → PyG graph converter, shared by batch + inference
- Exception hierarchy: `GraphExtractionError` → `SolcCompilationError` / `SlitherParseError` /
  `EmptyGraphError` — typed exceptions map to HTTP 400 vs 500
- Why typed exceptions > `None` returns for dual-use (batch + API) code
- `GraphExtractionConfig` dataclass — all fields and their roles
- `multi_contract_policy` — 3 heuristics, accuracy comparison, "most_derived" rationale
- `_MAX_TYPE_ID` dynamic computation — contradiction with `graph_schema.py` hardcoded `/12.0`
- PyG COO format — why sparse over dense adjacency matrix
- `node_metadata` index-alignment invariant — what breaks if violated
- `include_edge_attr=False` — PyG `Data.__getattr__` hard crash vs silent degradation tradeoff

**Preferences added mid-session:** P4, P5, P6, P7, P8

**Challenge questions answered:** Q1–Q5

**Gaps found and closed:**
- Concrete example of `EmptyGraphError` (imports-only Solidity file)
- "Compatibility" = existing configs reference the policy, not just documentation
- Exact numerical shift when `_MAX_TYPE_ID` changes: type collision at 1.0 + all types shift
- `python -O` disables `assert` → production invariant checks must use `if/raise`
- PyG `Data` raises `AttributeError` for missing attrs (not `None`) → hard crash is good

**Audit flags raised:** A3, A4
