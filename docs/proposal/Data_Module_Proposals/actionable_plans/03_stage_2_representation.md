# Actionable Plan — Stage 2: Representation Extraction (port from `ml/`)

**Date:** 2026-06-23 (revised 2026-06-10 after v1 build audit)
**Stage:** 2 of 8 (Week 3: Jun 23–29)
**Owner:** SENTINEL data engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §3.3, §5 (Week 3)
**Audit ref:** [`AUDIT_PATCHES.md`](AUDIT_PATCHES.md) §0 (F1, F2, F3, F4, F5, F6, F7, F8, F11, F23), §1 (2-P1 through 2-P12), §2 (C-7 performance budget, C-9 complexity proxy)
**Exit criteria:** `sentinel-data represent --source solidifi` produces 283 graph `.pt` + 283 token `.pt` + 283 sidecar `.rep.json` files for the SolidiFI preprocessed output; a **byte-identical regression test** proves the new path produces the same PyG graphs as the existing `ml/src/preprocessing/` path; a **13-issue preservation test** confirms all 13 critical bug fixes are preserved; **3 SolidiFI-fix tests** confirm A-1 (comment stripping), A-2 (RETURN_TO edges), A-3 (interface injection) are fixed; **performance budget: 100-file extraction in < 1 min on 8 cores** (within ±10% of the old `ml/` path per 2-P11).

---

## Goal

Move the existing representation extraction code from `ml/src/preprocessing/` and `ml/src/data_extraction/` (graph extractor + graph schema + tokenizer) into `sentinel_data/representation/`, **without changing any extraction logic**. After this stage, the new `sentinel-data` import path produces **byte-identical** graphs and tokens for the same input contract. The regression test that proves this is the gate; Stage 7 (export + seam) is the stage that actually switches the ML module over.

Four new builders (CFG, PDG, call graph, opcode) are also added in this stage as **opt-in** representation channels. **Only the CFG builder ships; PDG/call-graph/opcode are deferred to v3.1** (per AUDIT_PATCHES 2-P9).

---

## Why this stage third

Stage 1 produced preprocessed `.sol` files. Stage 2 is the first stage that turns source code into **model-consumable artifacts** (PyG graphs + CodeBERT tokens). Doing representation as a dedicated stage (rather than folding it into preprocessing) is the design decision that lets the same source code produce multiple representations (graph for GNN, tokens for Transformer, CFG for Stage 6 complexity-proxy analysis, etc.) and lets each representation be **cached and versioned independently**.

The Stage-0 design choice to keep the original code in `ml/` and ship only a stub in `sentinel-data` means this stage is the first where the two code paths coexist. The regression test that proves byte-identical output is what makes that coexistence safe.

---

## Design decisions

### D-2.1 — No extraction logic changes in this stage (CORRECT, unchanged)

The port from `ml/src/preprocessing/` to `sentinel_data/representation/` is a *move*, not a *rewrite*. Every node feature, every edge type, every extraction detail is preserved bit-for-bit. The only changes are: (a) module paths, (b) `__init__.py` re-exports, (c) imports of shared utilities, (d) the addition of a new `cache_manager.py` and `versioner.py` around the moved code.

This decision is the key to the seam swap in Stage 7. If the moved code produces different output from the original, the ML module's active Run 9 training pipeline breaks. The regression test is the gate.

### D-2.2 — Schema version is frozen at v9 (CORRECT, unchanged; **stub needs fixing**)

Per the verified live state of `ml/src/preprocessing/graph_schema.py` (confirmed 2026-06-08), the active schema is **v9**:
- `FEATURE_SCHEMA_VERSION = "v9"`
- `NODE_FEATURE_DIM = 12`
- `NUM_NODE_TYPES = 14`
- `NUM_EDGE_TYPES = 12`
- `_MAX_TYPE_ID = 13.0`

**🔴 BUG IN STAGE 0 STUB (discovered 2026-06-10 audit):** `Data/sentinel_data/representation/graph_schema.py` has the constant dicts **backwards**:
- `NODE_TYPES: dict[int, str] = {0: "STATE_VAR", ...}` should be `dict[str, int] = {"STATE_VAR": 0, ...}`
- `EDGE_TYPES: dict[int, str] = {0: "CONTAINS", ...}` should be `dict[str, int] = {"CONTAINS": 0, ...}`
- `FEATURE_NAMES: list[str] = [...]` should be `tuple[str, ...] = (...)`

This is a **latent bug** — anyone calling `NODE_TYPES["STATE_VAR"]` would get a `KeyError`. **Task 2.1 explicitly fixes this as part of the port.**

The v2 build freezes at v9. The proposal §2 was wrong (it said v8); the Stage 0 stub from task 0.4 must use v9 constants, and the regression test in 2.6 compares v9 output to v9 output. No v8 vs v9 question remains — the decision is v9, period.

**The v9 schema additions** (feat[11] `in_unchecked_block`, CFG_NODE_ARITH, EXTERNAL_CALL edge type) ARE in scope for Stage 2 — they are part of the v9 schema, not a future change. The deferred-schema decision (proposal §7) is about NOT adding *new* schema features in v2; the v9 features are already in the active training pipeline and must be preserved.

### D-2.3 — Tokenization moves to `representation/tokenizer.py` unchanged (CORRECT, with adapter caveat)

The existing `ml/src/data_extraction/tokenizer.py` (CodeBERT windowed, stride=256, [4, 512] windows) moves to `sentinel_data/representation/tokenizer.py` with no changes. The new `sentinel-data represent` CLI subcommand orchestrates batch tokenization, but the underlying tokenizer function is the same code.

**Caveat:** `tokenizer.py` reads `contracts.parquet` (a v1-only input format) and uses `ml/src/utils/hash_utils.get_contract_hash` (MD5-based naming). The v2 build doesn't have `contracts.parquet` and uses SHA-256 from Stage 1's `meta.json`. **Task 2.5 refactors the parquet-orchestrator into a v2 manifest-driven orchestrator**, but **preserves the per-file tokenization function unchanged**. The MD5-based naming is dropped in favor of content-addressed SHA-256 from Stage 1 (matches the v2 cache key contract).

### D-2.4 — Four new builders; only CFG ships; PDG/call-graph/opcode DEFERRED to v3.1 (CORRECT, with scope clarification)

The four new builders are **additive** for the v2 baseline. They are exposed in the CLI as opt-in flags (`--emit-cfg`, `--emit-pdg`, `--emit-callgraph`, `--emit-opcode`) but the default `sentinel-data represent` run does not produce them. The v2 baseline (Run 11) trains on **AST + the in-graph CFG (CFG_NODE_* node types) + the existing CONTROL_FLOW edges** — the same as Run 9 did. The new standalone CFG builder is *not* needed for Run 11.

**Scope clarification (added 2026-06-10):** The new CFG is a **standalone, normalized CFG artifact** that lives alongside the graph, not added to it. Its output is consumed by **Stage 6's complexity_proxy_risk detector** (which needs a flat CFG view to spot loops, dead code, etc.). It's not used for training in Run 11.

**Per AUDIT_PATCHES 2-P9, 2-P12:** PDG, call-graph, and opcode builders are **DEFERRED to v3.1** (not Stage 2). Reasons:
- The "lightweight PDG" described in the original plan is a v3+ feature, not v2
- Time spent on PDG is better spent on Stage 4 verification
- The schema additions for standalone CFG / PDG would be a v2.1 schema change (not a v2 port)
- Run 11 doesn't need them

The **CFG builder is shipped in Stage 2** as the only new builder, because it's straightforward (uses Slither's IR, which is already imported in `graph_extractor.py`). All four builders must be gated by the cache invalidation logic (2.8 below) — the versioner must invalidate the cache when the Slither version changes.

For v3.1 (post-Run-11), PDG / call-graph / opcode are added as separate stages. The Stage 2 plan leaves the directory structure ready (`representation/{pdg_builder,call_graph,opcode_extractor}.py` are placeholders with a "DEFERRED to v3.1 — see AUDIT_PATCHES 2-P9" docstring) so the v3.1 work is a drop-in.

### D-2.5 — Content-addressed representation cache (CORRECT, unchanged)

Every representation is keyed by `(sha256 of source, schema_version, extractor_version)`. A change to *any* of those three values invalidates the cache for that file. The cache is stored under `data/representations/<source>/` with a per-source manifest recording which representations have been computed. Re-running `sentinel-data represent` skips already-computed files unless `--force` is passed.

This is the same pattern as `InferenceCache` in `ml/src/inference/cache.py` (content-addressed, schema-version-invalidated) — the data-side version is a superset that also includes the extractor version.

### D-2.6 — Sidecar `rep.json` records provenance per file (CORRECT, unchanged)

Every `.pt` graph is accompanied by a `.rep.json` containing: `schema_version`, `extractor_version`, `node_count`, `edge_count`, `window_count` (for tokens), `compute_time_ms`, `cache_hit` (bool). The sidecar is what makes "this graph was built by which version of which extractor" queryable for any later audit. The sidecar is also the input to the `analysis/feature_dist.py` tool in Stage 7 (Run 9's complexity-proxy finding was discovered by analyzing graph statistics — the sidecar makes that analysis automatic).

### D-2.7 — Thin-adapter pattern (REVISED 2026-06-10)

The new `sentinel_data/representation/` is a **thin adapter** that re-exports from `ml/src/preprocessing/`:

```python
# sentinel_data/representation/graph_extractor.py (Stage 2)
from ml.src.preprocessing.graph_extractor import (
    GraphExtractionConfig,
    GraphExtractionError,
    SolcCompilationError,
    SlitherParseError,
    EmptyGraphError,
    extract_contract_graph,
)
```

**One source of truth** (`ml/src/preprocessing/graph_extractor.py`), **two import names** (`ml.src.preprocessing.graph_extractor` and `sentinel_data.representation.graph_extractor`). Stage 7 deletes the wrapper and rebinds the active training pipeline to import from `sentinel_data` directly.

**Why thin-adapter beats copy-paste:**
- Bug fixes apply once (in `ml/`), automatically propagate to the new path
- The Stage 7 seam swap is a 1-line change (delete the wrapper) instead of a multi-file refactor
- The "byte-identical output" guarantee is **trivially true** (same code, different name)

**Why thin-adapter beats symlink-import:**
- No `sys.path` hack — clean package boundaries
- venv is clean: `Data/.venv` doesn't need to know about `ml/`
- Works in Docker (where the two packages might be in different layers)

### D-2.8 — `src.utils.hash_utils` is dropped; SHA-256 from Stage 1 (NEW 2026-06-10)

The v1 hash_utils uses MD5 of the full contract path. The v2 build uses **SHA-256 of the file content** (computed at Stage 1 ingest, stored in `ingestion_manifest.json` and each `.meta.json`).

**Decision:** the v2 representation cache uses `meta.sha256` as the cache key. The thin adapter re-uses Stage 1's SHA-256 (no re-hashing at Stage 2). `ml/src/utils/hash_utils.py` stays in `ml/` for backward compat with the v1 training pipeline, but is **not imported** by the v2 thin adapter.

---

## Tasks — ordered, each with verifiable exit condition

### 2.1 — Fix the Stage 0 stub + verify schema version (D-2.2)

**🔴 BUG FIX:** the Stage 0 stub has 3 structural errors that must be corrected before the port begins.

**Step 1: Fix the stub constants:**
- `NODE_TYPES: dict[int, str] = {0: "STATE_VAR", ...}` → `dict[str, int] = {"STATE_VAR": 0, ...}`
- `EDGE_TYPES: dict[int, str] = {0: "CONTAINS", ...}` → `dict[str, int] = {"CONTAINS": 0, ...}`
- `FEATURE_NAMES: list[str] = [...]` → `tuple[str, ...] = (...)`
- Reorder each dict to match `ml/src/preprocessing/graph_schema.py` line-for-line

**Step 2: Verify the stub constants match the live schema:**
- `FEATURE_SCHEMA_VERSION = "v9"` (NOT `"v8"` as the proposal §2 said)
- `NODE_FEATURE_DIM = 12` (NOT 11)
- `NUM_NODE_TYPES = 14` (NOT 13)
- `NUM_EDGE_TYPES = 12` (NOT 11)
- `_MAX_TYPE_ID = 13.0` (NOT 12.0)

**Why first:** the regression test in 2.6 will detect any inconsistency. If the stub is wrong, the byte-identical test fails because the old `ml/` path uses the right values.

**Exit condition:** schema version is v9 in the stub; all 5 constants above match `ml/src/preprocessing/graph_schema.py` line-for-line; all 3 structural bugs above are fixed.

**Commit:** `fix(data-rep): correct Stage 0 stub dict direction + tuple type (v9 audit)`

---

### 2.2 — Port `ml/src/preprocessing/graph_schema.py` to `sentinel_data/representation/graph_schema.py` (THIN ADAPTER)

**Step 1: Create the thin adapter.** Author `sentinel_data/representation/graph_schema.py` that re-exports every public symbol from `ml/src/preprocessing/graph_schema.py`:

```python
# sentinel_data/representation/graph_schema.py
from ml.src.preprocessing.graph_schema import (
    FEATURE_SCHEMA_VERSION,
    NODE_FEATURE_DIM,
    NUM_NODE_TYPES,
    NUM_EDGE_TYPES,
    _MAX_TYPE_ID,
    NUM_CLASSES,
    VISIBILITY_MAP,
    NODE_TYPES,
    EDGE_TYPES,
    FEATURE_NAMES,
    STUB,
    CLASS_NAMES,
    NodeType,
    EdgeType,
    # ... every other public symbol from the live schema
)
```

**Step 2:** Remove the old `STUB = True` flag content from the new file's docstring (the stub markers are now obsolete). Update the module docstring to point to the live `ml/src/preprocessing/graph_schema.py` as the source of truth.

**Step 3:** Add a `__getattr__` lazy import fallback for the case where the `ml/` package isn't on the Python path (e.g. when `sentinel_data` is installed as a standalone PyPI package in the future). The fallback raises a clear error pointing the user to the dependency.

**Why:** the schema is the simplest thing to port first; testing it independently catches import errors before the bigger extractor file is ported.

**Exit condition:**
- `from sentinel_data.representation.graph_schema import FEATURE_SCHEMA_VERSION, NODE_FEATURE_DIM, NUM_EDGE_TYPES` works
- All constants `is`-equal to the `ml/` originals (using `is` because thin adapter re-exports the same object)
- The dict-direction bug is gone (a smoke test calls `NODE_TYPES["STATE_VAR"] == 0` and `EDGE_TYPES["CONTAINS"] == 0`)

**Commit:** `feat(data-rep): port graph_schema.py via thin adapter (ml/src is source of truth)`

---

### 2.3 — Port `ml/src/preprocessing/graph_extractor.py` to `sentinel_data/representation/graph_extractor.py` (THIN ADAPTER)

**Step 1:** Author `sentinel_data/representation/graph_extractor.py` that re-exports the public surface from `ml/src/preprocessing/graph_extractor.py`:

```python
from ml.src.preprocessing.graph_extractor import (
    GraphExtractionConfig,
    GraphExtractionError,
    SolcCompilationError,
    SlitherParseError,
    EmptyGraphError,
    extract_contract_graph,
)
```

**Step 2:** Update internal imports inside the thin adapter file (e.g. if it imports `graph_schema`, route through `sentinel_data.representation.graph_schema` for consistency).

**Step 3 (AMENDED per solidifi_analysis_issues_for_v2.md):** The following fixes must be applied to the underlying `ml/src/preprocessing/graph_extractor.py` BEFORE the thin adapter is created, because the thin adapter re-exports the live code. These are bug fixes that improve the active training pipeline (Run 9) and are required for the v2 baseline:

- **A-2: RETURN_TO edges** — In `_build_icfg_edges()`, after adding CALL_ENTRY edges (type 8) for internal calls, add paired RETURN_TO edges (type 9) from callee terminal nodes (END_IF/RETURN/last CFG node) back to the call-site's successor node. Without this, the ICFG is one-directional across function boundaries.

- **A-3: Interface injection / concrete contract enumeration** — When processing a `.sol` file, enumerate ALL contracts with implemented function bodies (`len(func.nodes) > 0`), not just the "main" contract (last-defined). Build CFG nodes from all concrete contracts. This catches vulnerability code injected into interface/abstract contract bodies in SolidiFI benchmarks.

**Step 4:** Add a behavioral smoke test that calls `extract_contract_graph` on one of the SolidiFI preprocessed contracts and verifies the output has `x.shape[-1] == NODE_FEATURE_DIM` (schema-dim gate test). Also verify RETURN_TO edge count > 0 for contracts with internal calls.

**Why next:** the schema is the data; the extractor is the function that produces it. Testing the extractor against the schema catches import errors and any subtle differences in how the schema constants are used internally.

**Exit condition:**
- `from sentinel_data.representation.graph_extractor import extract_contract_graph, GraphExtractionConfig, GraphExtractionError` works
- Calling `extract_contract_graph(sol_path)` on a known-input contract produces a PyG `Data` object with `x.shape == (N, 12)` (matches `NODE_FEATURE_DIM = 12`)
- The `Data` object's `edge_attr` (if present) has values in range `[0, NUM_EDGE_TYPES - 1]` = `[0, 11]`
- **NEW:** For a contract with internal calls, `edge_attr` contains RETURN_TO (type 9) edges paired with CALL_ENTRY edges
- **NEW:** For a contract with vulnerability code in interface bodies, CFG nodes for those functions are extracted

**Commit:** `feat(data-rep): port graph_extractor.py via thin adapter + A-2/A-3 fixes (ml/src is source of truth)`

---

### 2.4 — Replace the v1 `ast_extractor.py` orchestrator with a v2 manifest-driven one

The v1 `ml/src/data_extraction/ast_extractor.py` (576 LoC) is a thin wrapper around `extract_contract_graph()` that:
1. Reads `contracts.parquet` (v1-only input format)
2. Resolves solc versions
3. Multiprocessing-Pool extracts graphs
4. Writes `<md5>.pt` to `ml/data/graphs/`

The v2 build has **no `contracts.parquet`** and uses **SHA-256 from Stage 1** (not MD5). We need a new v2 orchestrator that:
- Reads the per-source manifest from Stage 1 (`data/preprocessed/<source>/<sha256>.meta.json`)
- Iterates each `<sha256>.sol` file
- Calls `extract_contract_graph()` from the thin adapter
- Writes to `data/representations/<source>/<sha256>.pt` + `<sha256>.rep.json` + `<sha256>.tokens.pt`

**Author `sentinel_data/representation/orchestrator.py`** (NEW, not a port):

```python
def represent_source(
    source: str,
    data_dir: Path,
    n_workers: int = 1,
    force: bool = False,
) -> RepresentResult:
    """Run the full representation pipeline for one source.
    
    Reads data/preprocessed/<source>/*.sol + *.meta.json from Stage 1,
    runs the graph_extractor (thin adapter) on each, writes to
    data/representations/<source>/<sha256>.{pt,tokens.pt,rep.json}.
    
    Honors the content-addressed cache (D-2.5) and versioner (D-2.5).
    """
```

**Why not "port" the v1 orchestrator:** the v1 orchestrator's input format (parquet) and output path (ml/data/graphs) don't exist in the v2 build. A literal port would be a rewrite masquerading as a move. The clean approach is a **new** orchestrator for v2's data shape, calling the same `extract_contract_graph` function (via the thin adapter).

**Exit condition:**
- `sentinel-data represent --source solidifi` produces 283 graph `.pt` + 283 token `.pt` + 283 sidecar `.rep.json` files for the SolidiFI preprocessed output
- All 283 graphs have `x.shape == (N, 12)` and `edge_attr` values in `[0, 11]`
- A tokenized file's `input_ids.shape == (4, 512)` and `attention_mask.shape == (4, 512)`
- The orchestrator is **idempotent** — re-running skips already-computed files (cache hit)

**Commit:** `feat(data-rep): add v2 manifest-driven representation orchestrator`

---

### 2.5 — Port `ml/src/data_extraction/tokenizer.py` (THIN ADAPTER, same as 2.2-2.3)

The per-file tokenization function (`tokenize(source, version) -> input_ids, attention_mask`) is the **value** of v1's tokenizer. The parquet-orchestrator is the v1-specific part. We extract the per-file function via thin adapter:

**Step 1:** Author `sentinel_data/representation/tokenizer.py` that re-exports the per-file tokenization function. The v1 `ast_extractor.py`'s batching logic is **not** ported (the v2 orchestrator from 2.4 handles batching).

**Step 2 (AMENDED per solidifi_analysis_issues_for_v2.md):** The following fix must be applied to the underlying `ml/src/data_extraction/tokenizer.py` BEFORE the thin adapter is created:

- **A-1: Comment stripping** — Add a `_strip_comments(source: str) -> str` helper that removes single-line (`//`) and multi-line (`/* */`) comments from source before tokenization. Do NOT strip NatSpec tags (`@param`, `@notice`) — only free-form comment text. Add a `strip_comments: bool = True` flag to the tokenizer config so it can be disabled for debugging. **Regenerate all token files when this is applied — cache invalidation required.**

```python
import re

def _strip_comments(source: str) -> str:
    # Remove /* ... */ blocks (including multi-line)
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    # Remove // ... to end of line
    source = re.sub(r'//[^\n]*', '', source)
    return source
```

```python
# sentinel_data/representation/tokenizer.py (thin adapter)
from ml.src.data_extraction.tokenizer import (
    tokenize,
    TokenizerConfig,
    # ... whatever else is a per-file function
)
```

**Exit condition:**
- `from sentinel_data.representation.tokenizer import tokenize` works
- Tokenizing a known input produces the same `input_ids` and `attention_mask` arrays whether called via the old or new import path (verified by byte-identical test in 2.6)
- **NEW:** Tokenizing a contract with verbose comments (e.g. OpenZeppelin SafeMath docstrings) does NOT produce comment tokens in the output
- The thin adapter adds no new logic (just re-exports)

**Commit:** `feat(data-rep): port tokenizer.py via thin adapter + A-1 comment stripping (per-file function only)`

---

### 2.6 — Write the regression test suite (byte-identical + 13-issue preservation)

**Test File 1: `Data/tests/test_representation/test_byte_identical_regression.py`**

The byte-identical regression test. Takes a fixture of **10 hand-picked SolidiFI + DIVE preprocessed contracts** (covering all 5 Solidity eras 0.4.x, 0.5.x, 0.6.x, 0.7.x, 0.8.x + the `unchecked{}` 0.8.x case), runs `extract_contract_graph()` via:
- the old path: `from ml.src.preprocessing.graph_extractor import extract_contract_graph`
- the new path: `from sentinel_data.representation.graph_extractor import extract_contract_graph`

Asserts the resulting PyG `Data` objects are **byte-identical** (using `torch.equal` for tensors, deep equality for PyG Data). Test fails loud if any field differs.

**Test File 2: `Data/tests/test_representation/test_13_issue_preservation.py`**

The 13-issue regression test (per AUDIT_PATCHES 2-P2, F23, I-1). The 13 issues span **9 source files** in `ml/src/`, not just the 4 named in the original plan. The test file is organized by issue ID, with one test function per critical fix. Test fixtures are hand-written minimal `.sol` files in `tests/fixtures/solidifi_v2/`.

The 13 critical tests (per 2-P3 through 2-P6):

| Issue | Test |
|---|---|
| **A9** `now` keyword detection | Fixture: 0.4.x contract using `now`; assert `feat[2] = 1.0` for the function |
| **A15** def_map scope_key | Fixture: two functions with same var name in different scopes; assert no spurious cross-function DEF_USE edges |
| **A20** label=0 hardcode | Fixture: preprocessed .sol + meta.json; assert graph `.y` matches `meta.sha256` (proves labels aren't hardcoded to 0) |
| **A34** prefix sort dim | Fixture: contract with mixed `external_call_count`; assert `select_prefix_nodes` uses `raw_node_features[:, 10]` not post-GAT dim |
| **A38** NaN before backward | Fixture: deliberately-NaN injection; assert trainer guards before `loss.backward()` (test lives in `ml/`, not `Data/`, but we add a sentinel test in `Data/` that calls the trainer's guard) |
| **EMITS edge (Interp-6)** | Fixture: contract with `emit Event();`; assert EMITS edge exists in the graph |
| **CALL_ENTRY for external** (F7) | Fixture: contract with `HighLevelCall`; assert EXTERNAL_CALL self-loop edge exists |
| **LibraryCall classification** (F25) | Fixture: `SafeMath.add()` library call; assert NOT counted as cross-contract HighLevelCall |
| **Resume overwrite fix** (F6) | Fixture: checkpoint + resume command; assert default `resume_model_only=False` |
| **Return-ignored fix** (F29) | Fixture: function with unused return; assert `feat[7] = 1.0` |
| **A31 fusion BUG-C2** token_norm | Fixture: high-norm tokens; assert LayerNorm is applied before projection |
| **A18 ICFG map** | Fixture: contract with internal calls; assert CALL_ENTRY/RETURN_TO edges exist |
| **A10 _cfg_node_type** | Fixture: contract with diverse CFG ops; assert node types are not silently OTHER |

**Test File 3 (NEW per solidifi_analysis_issues_for_v2.md): `Data/tests/test_representation/test_solidifi_fixes.py`**

Three new regression tests for the SolidiFI-specific fixes applied in Tasks 2.3 and 2.5:

| Test | Fixture | Assert |
|---|---|---|
| **A-2: RETURN_TO edges** | Contract with internal calls (e.g. `buggy_4.sol`) | `edge_attr` contains RETURN_TO (type 9) edges; count > 0 |
| **A-3: Interface injection** | Contract with vulnerability code in interface/abstract body (e.g. `buggy_29.sol`) | CFG nodes for injected functions ARE extracted; graph is not empty stub |
| **A-1: Comment stripping** | Contract with verbose OpenZeppelin SafeMath docstrings | Tokenized `input_ids` do NOT contain tokens for comment text; `strip_comments=True` default |

**Performance regression test (2-P11, REVISED):** the test asserts that extracting 100 contracts via the new path takes within ±10% of the time of the old `ml/src/preprocessing/` path. **Budget revised: 100 files in < 1 min on 8 cores** (1s/file, was 5 min — 3s/file is unrealistic for solc + Slither per file).

**Exit condition:** all byte-identical tests pass for the 10-file fixture; all 13-issue tests pass; all 3 SolidiFI-fix tests pass; the performance regression test passes; covered in CI from this stage forward.

**Commit:** `test(data-rep): add byte-identical regression + 13-issue preservation + SolidiFI-fixes suite`

---

### 2.7 — Implement the CFG builder only (PDG / call-graph / opcode DEFERRED to v3.1)

**Per AUDIT_PATCHES 2-P9, D-2.4 above:** only the CFG builder is shipped in Stage 2. PDG, call-graph, and opcode are DEFERRED to v3.1.

**Author `sentinel_data/representation/cfg_builder.py`** (only). It is a self-contained module exposing a `build_cfg(source_path, source_text) -> CfgArtifact` function. Uses Slither's internal IR (already imported in `graph_extractor.py`); produces per-function CFG in a normalized form.

**Gated by CLI flag `--emit-cfg` (default off).** The CFG builder is NOT in the byte-identical regression test (it is new code, not a port). It IS in the 13-issue test's adjacent "smoke test" — calling it on a known input should produce a sensible output.

**For v3.1 (post-Run-11),** PDG / call-graph / opcode are added as separate stages. The Stage 2 plan leaves the directory structure ready (`representation/{pdg_builder,call_graph,opcode_extractor}.py` are placeholders with a "DEFERRED to v3.1 — see AUDIT_PATCHES 2-P9" docstring) so the v3.1 work is a drop-in.

**Exit condition:**
- CFG builder compiles and imports; can be called individually with `--emit-cfg` flag
- Produces a sensible output for a single test contract
- PDG/call-graph/opcode are placeholder files with deferral docstrings

**Commit:** `feat(data-rep): add CFG builder (opt-in, deferred PDG/call-graph/opcode to v3.1)`

---

### 2.8 — Implement `cache_manager.py` and `versioner.py`

Author `sentinel_data/representation/cache_manager.py` (the content-addressed cache) and `versioner.py` (the schema/extractor-version invalidation logic). The cache stores representations under `data/representations/<source>/<sha256>.<ext>` and the manifest under `data/representations/<source>/.cache_manifest.json`. The versioner maintains a global `data/representations/_version_registry.json` recording the current `(schema_version, extractor_version)` pair; a re-run checks each file's recorded version against the current and recomputes if mismatched.

**Why:** the cache makes re-runs fast (the 22K DIVE + 283 SolidiFI = 22,356 existing graphs would take ~6 hours to recompute); the versioner prevents the silent-mix-of-versions failure mode that bit us in Run 8 (graphs from v8 mixed with graphs from v9 in the same dataset).

**Exit condition:**
- `cache_manager` stores + loads representations correctly
- `versioner` invalidates a file when the schema version is bumped
- Re-running `sentinel-data represent` on 22,356 contracts takes < 5 min (cache hit)
- Regression test still passes

**Commit:** `feat(data-rep): add content-addressed cache + version registry`

---

### 2.9 — Wire the `sentinel-data represent` CLI subcommand

Connect `cli.py` `represent` subcommand to the new module. The CLI reads sources from `config.yaml`, iterates over preprocessed files, runs the graph extractor + tokenizer (and optionally the 4 new builders), writes outputs to `data/representations/<source>/`, and updates the cache manifest. Add `--force` to recompute regardless of cache hit, `--emit-*` flags for the 4 new builders.

**DVC: DEFERRED.** Exit criterion #9 in the original plan ("`dvc repro represent` runs end-to-end") is replaced with "manual `sentinel-data represent` works end-to-end". DVC setup is moved to Stage 7 (which actually does the seam swap and needs DVC working). This is a small scope-reduction that lets us ship Stage 2 without first wiring DVC.

**Exit condition:**
- `sentinel-data represent --source solidifi` produces 283 `.pt` (graph) + 283 `.pt` (tokens) + 283 `.rep.json` (sidecar) for the SolidiFI preprocessed output
- `sentinel-data represent --source dive --workers 4` produces 22,073 graph + 22,073 token + 22,073 sidecar files for the DIVE preprocessed output
- A dry-run shows the planned action without executing

**Commit:** `feat(data-rep): wire CLI for the represent stage`

---

### 2.10 — Add tests for the new representation code

Author `Data/tests/test_representation/` with:
- **Port regression test** (the 2.6 tests, live in this dir)
- **Cache tests** — store/load/invalidate round-trip
- **Versioner tests** — schema bump invalidates; extractor bump invalidates
- **New builder tests** — CFG builder produces a sensible output for a single test contract
- **Orchestrator tests** — `sentinel-data represent --source solidifi --dry-run` prints the planned action; idempotent re-run skips cache hits
- **Schema-dim gate test** — `x.shape[-1] == NODE_FEATURE_DIM` (catches the "silent shape mismatch" failure mode)

**Exit condition:** `pytest tests/test_representation -v` passes; coverage > 80%.

**Commit:** `test(data-rep): add full test suite for representation stage`

---

### 2.11 — Author `ADR-0003-representation-port-design.md`

Document the key design decisions: thin-adapter (D-2.7), v9 schema freeze (D-2.2 — confirmed, not a question), parallel import paths (D-2.7), additive new builders (D-2.4 with v3.1 deferral), content-addressed cache (D-2.5), sidecar provenance (D-2.6), `hash_utils` dropped (D-2.8).

**Exit condition:** file exists; cites the regression test as the gate; references the deferred schema decision; documents the Stage 0 stub bug fix as part of the port.

**Commit:** `docs(data): add ADR-0003 for representation port design`

---

## What NOT to fix (preservation list)

| Bug | Status | File:line | Stage 2 action |
|---|---|---|---|
| **A9** `now` keyword | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:587-605` | Do not re-fix. The 13-issue regression test (2.6) has a specific test for this. |
| **A15** def_map by name | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:1147-1179` | Do not re-fix. The 13-issue test has a scope_key test. |
| **A20** label=0 hardcode | ✅ FIXED | `ml/src/data_extraction/ast_extractor.py:290,342,395` | Do not re-fix. The 13-issue test has a CSV-label test. |
| **A34** prefix sort dim | ✅ FIXED | `ml/src/models/sentinel_model.py:356,367` | Do not re-fix. The 13-issue test has a prefix-sort test. |
| **A38** NaN before backward | ✅ FIXED | `ml/src/training/trainer.py` | Do not re-fix. The 13-issue test has a NaN-guard test. |
| Resume overwrite | ✅ FIXED | `ml/src/training/trainer.py:383,1184,1206,1212` | Do not re-fix. Stage 8 uses full-resume default. |
| **EMITS edge bug** | ⚠ OPEN (Interp-6) | `ml/src/preprocessing/graph_extractor.py` | Stage 7 seam swap must fix (per 7-P6). The 13-issue test asserts the bug exists pre-fix. |
| **CALL_ENTRY cross-function for external** | ⚠ PARTIAL FIX | `ml/src/preprocessing/graph_extractor.py:1001` | Self-loop only; full cross-function edge is post-Run-11. The 13-issue test asserts the self-loop is present. |
| **LibraryCall <: HighLevelCall** | ⚠ KNOWN | `ml/src/preprocessing/graph_extractor.py:1081` | `_compute_external_call_count` relies on isinstance; library calls counted as cross-contract. The 13-issue test asserts the current behavior (it's not a bug, it's a design choice). |
| **v8 schema** (in old proposal §2) | ❌ WRONG | (proposal §2 only) | **CORRECTED**: the active schema is v9. Stage 0 stub and Stage 2 port use v9 throughout. Per F1, this is now verified live. |
| **🔴 Stage 0 stub dict direction** | ❌ WRONG | `Data/sentinel_data/representation/graph_schema.py:42-73` | **FIXED in 2.1**: NODE_TYPES and EDGE_TYPES were `dict[int, str]` (id→name); should be `dict[str, int]` (name→id). FEATURE_NAMES was `list[str]`; should be `tuple[str, ...]`. |
| **Stage 0 stub sys.path hack** | ❌ WRONG | `ml/src/data_extraction/ast_extractor.py:71-72` | **NOT PORTED** — the v2 orchestrator (2.4) doesn't need it. We do the refactor inline in 2.4. |
| **A-1: Comment stripping** | ❌ NOT covered | `ml/src/data_extraction/tokenizer.py` | **NOW IN SCOPE (per solidifi_analysis_issues_for_v2.md)**: Task 2.5 adds `_strip_comments()` helper with `strip_comments=True` default. Applied to underlying ml/ code before thin adapter is created. |
| **A-2: RETURN_TO edges** | ⚠ OPEN | `ml/src/preprocessing/graph_extractor.py` | **NOW IN SCOPE (per solidifi_analysis_issues_for_v2.md)**: Task 2.3 adds RETURN_TO edge construction paired with CALL_ENTRY. Applied to underlying ml/ code before thin adapter is created. |
| **A-3: Interface injection** | ❌ NOT covered | `ml/src/preprocessing/graph_extractor.py` | **NOW IN SCOPE (per solidifi_analysis_issues_for_v2.md)**: Task 2.3 adds concrete-contract enumeration (process all contracts with implemented function bodies, not just main contract). Applied to underlying ml/ code before thin adapter is created. |

---

## Final exit criteria check (REVISED)

| # | Check | Status |
|---|---|---|
| 1 | `sentinel-data represent --source solidifi` produces 283 graph `.pt` + 283 token `.pt` + 283 sidecar `.rep.json` for the SolidiFI preprocessed output | NEW (was 30 ScaBench files; we use 283 SolidiFI as fixtures) |
| 2 | The byte-identical regression test passes against the 10-file SolidiFI+DIVE fixture for all 4 files: `graph_schema`, `graph_extractor`, `tokenizer`, `orchestrator` | CORRECT (uses thin-adapter for byte-identicality) |
| 3 | The **13-issue pre-Run-8 audit regression test** passes for all 13 critical issues (A9, A15, A20, A34, A38, EMITS, CALL_ENTRY, LibraryCall, Resume, Return-ignored, A31, A18, A10) | UNCHANGED (13 issues per the plan) |
| 4 | `from sentinel_data.representation.graph_extractor import extract_contract_graph` works; calling it on a known input produces the same `Data` object as the old import path | CORRECT (thin adapter makes this trivially true) |
| 5 | The CFG builder compiles and can be called with `--emit-cfg` flag; **PDG/call-graph/opcode are placeholders with deferral docstrings (NOT shipped in v2)** | UNCHANGED |
| 6 | The `cache_manager` stores + loads representations correctly; `versioner` invalidates on schema bump AND on Slither version bump | UNCHANGED |
| 7 | The active Run 9 training pipeline (using the old `ml/` import path) still works unchanged | UNCHANGED (no changes to `ml/` in Stage 2) |
| 8 | **Performance budget: 100-file extraction in < 1 min on 8 cores (within ±10% of old `ml/` path)** | **REVISED** (was 5 min — 3s/file unrealistic for solc+Slither per file) |
| 9 | `dvc repro represent` runs end-to-end | **DEFERRED to Stage 7** (DVC setup happens with the seam swap, not here) |
| 10 | `pytest tests/test_representation -v` passes with > 80% coverage | UNCHANGED |
| 11 | `ADR-0003-representation-port-design.md` is committed; **references the 13-issue audit, the Stage 0 stub bug fix, and the thin-adapter decision** | **EXPANDED** |
| 12 | **A-1 (comment stripping):** tokenizing a contract with verbose comments does NOT produce comment tokens in `input_ids`; `strip_comments=True` default is active | **NEW** (per solidifi_analysis_issues_for_v2.md) |
| 13 | **A-2 (RETURN_TO edges):** contracts with internal calls have RETURN_TO (type 9) edges paired with CALL_ENTRY edges; count > 0 | **NEW** (per solidifi_analysis_issues_for_v2.md) |
| 14 | **A-3 (interface injection):** contracts with vulnerability code in interface/abstract bodies have CFG nodes for those functions extracted; graph is not empty stub | **NEW** (per solidifi_analysis_issues_for_v2.md) |

All 14 pass → **Stage 2 complete**. Tag `data-stage-2`, proceed to Stage 3.

---

## Risk register

| Risk | Mitigation |
|---|---|
| The thin adapter has a typo'd import that breaks under certain conditions | The byte-identical regression test catches this immediately. The thin adapter is only ~10 lines of `from X import Y` statements; review is easy. |
| The Stage 0 stub bug fix (#1) introduces a regression because the broken dict direction was "working" for someone | Add a migration shim that supports BOTH `dict[int, str]` and `dict[str, int]` lookups for one release. Better: just fix it and the 13-issue test will catch any caller depending on the wrong direction. |
| `src.utils.hash_utils` is used elsewhere we didn't audit | `grep` for `hash_utils` imports across `ml/`. The audit (per this plan) found 2 callers (ast_extractor, tokenizer); the v2 orchestrator doesn't import it. The 3rd-party callers in `ml/scripts/` continue to use the v1 path until Stage 7. |
| The v2 orchestrator (2.4) takes 6+ hours for 22K DIVE contracts on first run | Run with `--workers 4` (3.3× speedup). The content-addressed cache makes re-runs instant. The performance test (2.6) bounds the single-threaded cost. |
| Tokenization is slower in the new path (e.g. import overhead) | Performance regression test in 2.10; if the new path is > 10% slower, profile and fix. Likely not an issue since the thin adapter adds ~0 cost. |
| Slither version differences between `ml/` and `sentinel-data` cause graph_extractor to produce different output | The Dockerfile pins the solc/slither versions; `pyproject.toml` pins `slither-analyzer >=0.10.0`. The byte-identical test catches any divergence. |
| The CFG builder breaks Slither's IR assumption (it's already imported in graph_extractor.py) | The CFG builder is opt-in (`--emit-cfg`); if it fails, the default `represent` run skips it. The 13-issue test's smoke test catches regressions. |
| PDG/call-graph/opcode stubs are accidentally used in Stage 3+ downstream code | The stubs raise `NotImplementedError` with a clear message. A test asserts each stub raises. |

---

## Schedule (7 working days, revised per solidifi_analysis_issues_for_v2.md)

| Day | Tasks | Output |
|---|---|---|
| Day 1 | 2.1 (fix stub), 2.2 (port schema), 2.3 (port graph_extractor) | 3 commits, byte-identical smoke test passes |
| Day 2 | 2.4 (v2 orchestrator), 2.5 (port tokenizer) | 2 commits, full pipeline runs on SolidiFI 283 contracts |
| Day 3 | **A-2 + A-3 fixes** applied to `ml/src/preprocessing/graph_extractor.py` (RETURN_TO edges, concrete-contract enumeration); thin adapter re-exports updated code | 1 commit, RETURN_TO edges present in graphs with internal calls; interface injection graphs are not empty stubs |
| Day 4 | **A-1 fix** applied to `ml/src/data_extraction/tokenizer.py` (comment stripping with `strip_comments=True` default); thin adapter re-exports updated code; regenerate all token files (cache invalidation) | 1 commit, comment tokens absent from tokenized output |
| Day 5 | 2.6 (regression tests) | 1 commit, byte-identical + 13-issue + SolidiFI-fixes tests pass for all 13+3 issues |
| Day 6 | 2.7 (CFG builder), 2.8 (cache + versioner) | 2 commits, full re-run hits cache instantly |
| Day 7 | 2.9 (CLI), 2.10 (orchestrator tests), 2.11 (ADR) | 3 commits, end-to-end `sentinel-data represent --source solidifi` works |

Total: 13 commits, ~2,000 LoC of new code (mostly thin adapters + tests + fixes), changes to `ml/src/preprocessing/graph_extractor.py` and `ml/src/data_extraction/tokenizer.py` for A-1, A-2, A-3 fixes.

---

**End of Stage 2 actionable plan (v2, post-audit 2026-06-10).**
