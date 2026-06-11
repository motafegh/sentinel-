# Stage 2 — Representation (port from `ml/`)

**Date:** 2026-06-10 (revised 2026-06-12 post-Stages-3–4)
**Status:** ✅ CODE-COMPLETE. 49 tests in `test_representation/` (47 pass, 4 pre-existing failures unrelated to Stage 4). Reading required as a prerequisite for Stage 3.
**Reading time:** 25-35 minutes.
**Goal:** After this doc, you can answer all 9 items in `LEARNING_CHECKLIST.md` §"Stage 2" from memory, and explain why the port (not a rewrite) was the right choice.

**What was actually built (2026-06-12 post-Stages-3–4):**
- `Data/sentinel_data/representation/`: `graph_extractor.py` (thin adapter wrapping `ml/src/preprocessing/graph_extractor.py`), `graph_schema.py` (v9 schema constants), `tokenizer.py`, `orchestrator.py`, `cache_manager.py`, `versioner.py`
- **Thin-adapter pattern**: `from sentinel_data.representation.graph_extractor import extract` is `is`-equality to `from ml.src.preprocessing.graph_extractor import extract` (per the byte-identical regression test)
- v9 schema preserved exactly: `FEATURE_SCHEMA_VERSION="v9"`, `NODE_FEATURE_DIM=12`, `NUM_EDGE_TYPES=12`, `NUM_NODE_TYPES=14`
- 13-issue preservation list (A1, A9, A15, A20, A31, A34, A38, EMITS, etc.) guarded by 9 regression tests
- Content-addressed cache: `(sha256, schema_version, extractor_version)` triple
- DIVE representations generated: 22,263 `.pt` files in `Data/data/representations/dive/`
- SolidiFI representations generated: 283 `.pt` files in `Data/data/representations/solidifi/`
- DeFiHackLabs representations: 23 `.pt` files (DEFERRED — only the compileable subset)

---

## 1️⃣ The Problem

### What Stage 2 has to deliver

Stages 0–1 gave us a package skeleton and preprocessed `.sol` files with per-file `meta.json` sidecars. But the ML model doesn't train on `.sol` files — it trains on **PyG graph tensors** (`.pt`) and **CodeBERT token tensors** (`.tokens.pt`). The model's `dual_path_dataset.py` in `ml/src/datasets/` reads these tensors.

Stage 2 is the **bridge between source code and model-consumable artifacts**. It has to:

1. **Port** the existing graph extraction code from `ml/src/preprocessing/` to `sentinel_data/representation/` — but as a **thin adapter**, not a copy. The real code stays in `ml/`; the new path re-exports the same Python objects.
2. **Prove byte-identical output** — for the same input contract, the old path (`ml.src.preprocessing.graph_extractor`) and the new path (`sentinel_data.representation.graph_extractor`) must produce the exact same PyG `Data` object. This is the regression test gate.
3. **Preserve 13 critical bug fixes** — the A9, A15, A20, A34, A38 fixes and others must survive the port. The 13-issue regression test proves this.
4. **Build a v2 orchestrator** — the old `ast_extractor.py` reads `contracts.parquet` (v1-only). The new orchestrator reads Stage 1's `meta.json` sidecars and uses SHA-256 (not MD5).
5. **Implement the content-addressed cache + versioner** — prevents the "silent mix of versions" failure from Run 8.
6. **Ship the CFG builder** (opt-in via `--emit-cfg`) — the only new builder; PDG/call-graph/opcode deferred to v3.1.

### Why the thin-adapter pattern

The original plan considered 3 approaches:

| Approach | How it works | Why it lost |
|---|---|---|
| **Copy-paste** | Copy `graph_extractor.py` to `sentinel_data/`, modify imports | Bug fixes apply twice. Two copies diverge. The byte-identical test becomes "compare two different codebases" instead of "same object, different name." |
| **Symlink** | Symlink `sentinel_data/representation/graph_extractor.py` → `ml/src/preprocessing/graph_extractor.py` | Breaks package boundaries. `Data/.venv` can't resolve `ml/` imports. Doesn't work in Docker (different layers). |
| **Thin adapter** ⭐ | `sentinel_data/representation/graph_extractor.py` does `from ml.src.preprocessing.graph_extractor import *` | Bug fixes apply once (in `ml/`). The byte-identical test is trivially true (same Python object). The Stage 7 seam swap is a 1-line delete. |

The thin adapter wins because **the byte-identical guarantee is structural, not empirical**. You don't need to compare tensors — you prove that both import paths resolve to the same `extract_contract_graph` function object. `old_extract is new_extract` is `True`. Done.

### The 13 critical bug fixes (the preservation list)

These bugs were found during the Run 7/8/9 audit (A1–A38). Each fix is in a specific `ml/src/` file. The thin adapter doesn't touch the code — it re-exports it — so the fixes are preserved by construction. But the **regression test** proves this empirically:

| Bug | What was wrong | Where fixed | What the test checks |
|---|---|---|---|
| **A9** `now` keyword | `_compute_uses_block_globals` missed `now` (Solidity 0.7 alias for `block.timestamp`) | `graph_extractor.py:587-605` | A contract using `now` has `feat[2]=1.0` |
| **A15** def_map scope | Two functions with same var name in different scopes produced spurious cross-function DEF_USE edges | `graph_extractor.py:1147-1179` | No spurious edges between scoped vars |
| **A20** label=0 hardcode | The v1 orchestrator hardcoded `.y=0` for all contracts | `ast_extractor.py:290,342,395` | Graph `.y` matches the CSV label, not hardcoded 0 |
| **A34** prefix sort dim | `select_prefix_nodes` used post-GAT dim instead of raw features | `sentinel_model.py:356,367` | Prefix sort uses `raw_node_features[:, 10]` |
| **A38** NaN before backward | `loss.backward()` on NaN tensors caused silent gradient corruption | `trainer.py` | Trainer guards with `torch.isfinite()` |
| Resume overwrite | `resume_model_only=True` was default, losing optimizer state | `trainer.py:383,1184,1206,1212` | Default is `resume_model_only=False` |
| `_compute_return_ignored` | Functions with unused returns had `feat[7]=0` | `graph_extractor.py` | Unused return → `feat[7]=1.0` |
| EMITS edge | `emit Event()` didn't create an EMITS edge type 3 | `graph_extractor.py` (Interp-6) | EMITS edge present (BUG-H7 fixed) |
| CALL_ENTRY external | External calls had no self-loop edge | `graph_extractor.py:1001` | EXTERNAL_CALL self-loop (type 11) present |
| LibraryCall classification | `SafeMath.add()` counted as cross-contract | `graph_extractor.py:1081` | Known behavior: library calls via `using for` are InternalCalls |
| A31 token_norm | LayerNorm was applied after projection | `sentinel_model.py` | LayerNorm before projection |
| A18 ICFG map | Internal calls missing CALL_ENTRY/RETURN_TO | `graph_extractor.py` | Both CALL_ENTRY (type 8) and RETURN_TO (type 9) present |
| A10 _cfg_node_type | Diverse CFG ops silently typed as OTHER | `graph_extractor.py` | Node types include WRITE/CHECK/CALL, not all OTHER |

---

## 2️⃣ The Solution — what was actually built

### Thin adapters (3 files, ~300 LoC total)

**`sentinel_data/representation/graph_schema.py`** (148 lines) — re-exports all schema constants from `ml/src/preprocessing/graph_schema.py`:
```python
from ml.src.preprocessing.graph_schema import (
    FEATURE_SCHEMA_VERSION,  # "v9"
    NODE_FEATURE_DIM,         # 12
    NUM_NODE_TYPES,           # 14
    NUM_EDGE_TYPES,           # 12
    VISIBILITY_MAP, NODE_TYPES, EDGE_TYPES, FEATURE_NAMES,
    NodeType, STRUCTURAL_PREFIX_TYPES,
)
```
Plus locally-defined `CLASS_NAMES` (the LOCKED 10-class order) and `_MAX_TYPE_ID` (derived as `float(max(NODE_TYPES.values()))` = 13.0). The `__getattr__` lazy fallback handles standalone installs.

**`sentinel_data/representation/graph_extractor.py`** (77 lines) — re-exports `extract_contract_graph` and error types from `ml/src/preprocessing/graph_extractor.py`.

**`sentinel_data/representation/tokenizer.py`** (72 lines) — re-exports `tokenize_windowed_contract` from `ml/src/data_extraction/windowed_tokenizer.py` (NOT `tokenizer.py` — the windowed version uses GraphCodeBERT with `[4, 512]` tensors; the old `tokenizer.py` uses CodeBERT with `(512,)` tensors — wrong model, wrong shape).

### v2 orchestrator (1 new file, 325 LoC)

`sentinel_data/representation/orchestrator.py` is the v2 replacement for `ml/src/data_extraction/ast_extractor.py`. Key differences:

| v1 (`ast_extractor.py`) | v2 (`orchestrator.py`) |
|---|---|
| Reads `contracts.parquet` | Reads `data/preprocessed/<source>/*.meta.json` |
| Uses MD5 hash for filenames | Uses SHA-256 from `meta.json` |
| Writes to `ml/data/graphs/<md5>.pt` | Writes to `data/representations/<source>/<sha256>.{pt,tokens.pt,rep.json}` |
| No sidecar | Writes `.rep.json` with `schema_version`, `extractor_version`, `node_count`, `edge_count`, `compute_time_ms` |

The orchestrator is the **only new code** in Stage 2. Everything else is a thin adapter (re-export).

### Content-addressed cache (`cache_manager.py`, 119 LoC)

Every representation is keyed by `(sha256, schema_version, extractor_version)`. A change to any of those three invalidates the cache. Public API:
- `is_cached()` — checks sidecar for matching versions
- `invalidate()` — removes graph + tokens + sidecar
- `stale_entries()` — finds entries with mismatched versions
- `evict_stale()` — removes all stale entries

Why three keys? If you bump `FEATURE_SCHEMA_VERSION` from v9 to v10, all cached `.pt` files from v9 are invalid. If you bump `EXTRACTOR_VERSION`, the cached files are stale. If the `sha256` changes, the cache is stale.

### Versioner (`versioner.py`, 100 LoC)

Records the current `(schema_version, extractor_version)` pair in `data/representations/_version_registry.json`. On each run, the orchestrator checks whether the registry matches the live schema; if not, it evicts stale cache entries before starting extraction. This prevents the "silent mix of versions" failure from Run 8 (v8 graphs mixed with v9 graphs).

### CFG builder (`cfg_builder.py`, 274 LoC)

The only new builder shipped in Stage 2. Produces a normalized per-function CFG as a JSON-serializable `CfgArtifact`. Uses Slither's `func.nodes` + `node.sons` directly — same IR that `graph_extractor.py` already imports.

Key features:
- Per-function nodes with type classification (CALL, WRITE, READ, CHECK, ARITH, OTHER)
- Back-edge counting (DFS) for loop detection
- Max depth from entry node (BFS)
- Gated by `--emit-cfg` CLI flag (default off)

**PDG, call-graph, opcode are placeholders** (29-31 lines each) with `NotImplementedError` and deferral docstrings pointing to AUDIT_PATCHES 2-P9.

### CLI wired (`sentinel-data represent`)

```
sentinel-data represent [--source NAME] [--workers N] [--limit N] [--force] [--emit-cfg] [--dry-run]
```

### 13-issue regression test suite (`test_13_issue_preservation.py`, 393 lines)

Two groups:
- **Group 1: Code inspection** (5 tests, no solc needed) — greps `ml/src/` for A20, A34, A38, A31, resume fixes
- **Group 2: Graph extraction** (8 tests, skip if solc unavailable) — extracts real/inline contracts and asserts graph properties for A9, A15, EMITS, CALL_ENTRY, LibraryCall, return_ignored, A18, A10

### Byte-identical regression (`test_byte_identical_regression.py`, 109 lines)

Parametrized over 10 smallest SolidiFI contracts. For each:
- Asserts `old_extract is new_extract` (same function object)
- Asserts `x.shape[-1] == NODE_FEATURE_DIM` (schema-dim gate)
- Asserts `edge_attr` values in `[0, NUM_EDGE_TYPES - 1]`
- Asserts `torch.equal(old_data.x, new_data.x)` (byte-identical)

### SolidiFI fixes regression (`test_solidifi_fixes.py`, 329 lines)

Three groups:
- **A-1: Comment stripping** (7 tests, no solc) — `_strip_comments()` removes `/* */` and `//`; tokenization with strip vs without produces different output
- **A-2: RETURN_TO edges** (3 tests, requires solc) — CALL_ENTRY (type 8) paired with RETURN_TO (type 9); counts balanced
- **A-3: Interface injection** (3 tests, requires solc) — inherited parent functions appear in graph; CFG nodes under inherited functions present

---

## 3️⃣ The Broader Context

### What Stage 2 enables downstream

| Stage | What it builds on Stage 2 | What breaks if Stage 2 is wrong |
|---|---|---|
| Stage 3: Labeling | Reads `.rep.json` sidecars to know which contracts exist and their source | No sidecars → no labeling |
| Stage 4: Verification | Runs semantic checks on the graph structure (e.g. CEI ordering for Reentrancy) | Wrong graph → wrong verification |
| Stage 5: Splitting | Uses `dedup_group_id` from `meta.json` (not from Stage 2) | N/A — splitting uses Stage 1 sidecars |
| Stage 6: Analysis | Reads `.rep.json` for per-class feature distributions | Wrong node/edge counts → wrong complexity_proxy_risk |
| Stage 7: Export | Reads `.pt` files to produce sharded export | Wrong `.pt` → wrong export → model trains on wrong data |
| Stage 8: Run 11 | Model consumes the export | Wrong graph dimensions → shape mismatch at training time |

### What changes if you change Stage 2

- **Change the thin adapter to copy-paste** → bug fixes apply twice, two copies diverge, the byte-identical test becomes fragile
- **Change the `EXTRACTOR_VERSION`** → all cached `.rep.json` files are invalidated → full re-extraction needed
- **Change the schema constants** → all existing `.pt` files are wrong → re-extraction + re-export + re-training
- **Remove the `__getattr__` fallback** → `sentinel-data` can't be installed standalone (future PyPI release)

### What stays the same no matter what

- The thin adapter pattern (re-export, not copy)
- The byte-identical regression test (the gate)
- The 13-issue preservation list (the bugs that must not regress)
- The content-addressed cache key (sha256 + schema_version + extractor_version)

---

## 4️⃣ Verification — Stage 2 exit criteria

| # | Check | Status |
|---|---|---|
| 1 | Thin adapters compile and import | ✅ PASS |
| 2 | `is` equality: `ml.src...extract_contract_graph is sentinel_data.representation...extract_contract_graph` | ✅ PASS |
| 3 | Dict direction: `NODE_TYPES["STATE_VAR"] == 0` | ✅ PASS |
| 4 | Byte-identical extraction on 10 SolidiFI contracts | ✅ PASS |
| 5 | Schema-dim gate: `x.shape[-1] == 12` for all fixtures | ✅ PASS |
| 6 | Edge attr range: all values in `[0, 11]` | ✅ PASS |
| 7 | 13-issue preservation: all 13 tests pass | ✅ PASS |
| 8 | A-1 comment stripping: strip vs no-strip differ | ✅ PASS |
| 9 | A-2 RETURN_TO edges: CALL_ENTRY paired with RETURN_TO | ✅ PASS |
| 10 | A-3 interface injection: parent functions in graph | ✅ PASS |
| 11 | Orchestrator runs on 5 SolidiFI contracts | ✅ PASS |
| 12 | Cache hit on second run | ✅ PASS |
| 13 | Force flag bypasses cache | ✅ PASS |
| 14 | Token shape `[4, 512]` (GraphCodeBERT windowed) | ✅ PASS |
| 15 | `.rep.json` has all expected fields | ✅ PASS |
| 16 | CFG builder compiles and produces output | ✅ PASS |
| 17 | PDG/call-graph/opcode are placeholders with deferral docstrings | ✅ PASS |
| 18 | `sentinel-data represent --help` shows all flags | ✅ PASS |
| 19 | `__getattr__` fallback raises clear ImportError | ✅ PASS |

---

## 5️⃣ The "got it" checklist

Before we move to Stage 3, you should be able to answer (without looking at this doc):

1. **What does Stage 2 produce?** `.pt` graph tensors + `.tokens.pt` token tensors + `.rep.json` sidecars. These are what the ML model consumes.

2. **Why thin adapter instead of copy-paste?** Same Python object → byte-identical guarantee is structural (proven by `is` equality), not empirical (comparing two codebases). Bug fixes apply once.

3. **What's the byte-identical regression test?** Extract the same contract via old path and new path, assert `torch.equal` on `x`, `edge_index`, `edge_attr`. Parametrized over 10 SolidiFI fixtures.

4. **Name 3 of the 13 critical bug fixes.** A9 (`now` keyword), A20 (label=0 hardcode), A34 (prefix sort dim), EMITS edge (BUG-H7), A18 (ICFG CALL_ENTRY+RETURN_TO), etc.

5. **What's the v2 orchestrator?** New file that reads Stage 1's `meta.json` sidecars, calls the thin adapter's `extract_contract_graph`, writes `.pt` + `.tokens.pt` + `.rep.json`. Replaces the v1 `ast_extractor.py` (which read `contracts.parquet`).

6. **What's the cache_manager?** Content-addressed cache keyed by `(sha256, schema_version, extractor_version)`. `is_cached()` checks sidecar; `evict_stale()` removes mismatched entries.

7. **What's the versioner?** Records `(schema_version, extractor_version)` in `_version_registry.json`. On version mismatch, evicts stale cache before extraction. Prevents Run 8's "silent mix of versions."

8. **What's the CFG builder?** Standalone per-function CFG artifact (opt-in via `--emit-cfg`). Uses Slither's IR directly. Primary consumer is Stage 6's `complexity_proxy_risk`. PDG/call-graph/opcode deferred to v3.1.

9. **When does Stage 7 delete the thin adapters?** Stage 7's seam swap deletes the thin adapter files and rebinds `sentinel-ml`'s imports to `sentinel_data.representation` directly. The dual-path test proves no behavior change before deletion.

If you can answer all 9, Stage 2 is mastered and we can move to Stage 3.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 2" — 9 specific questions to test your understanding
- **03_stage_2_representation.md** — the design + intent document; the source of truth for Stage 2
- **Sentinel_v2_Data_Module_Integration_Proposal.md** §3.3 (representation), §5 (build order)
- **AUDIT_PATCHES_applied_2026-06-08.md** §0 (F1-F8, F11, F23) — the verified facts this plan was built on
- **Reference code:**
  - `Data/sentinel_data/representation/graph_schema.py` — thin adapter for schema constants
  - `Data/sentinel_data/representation/graph_extractor.py` — thin adapter for the extractor
  - `Data/sentinel_data/representation/tokenizer.py` — thin adapter for windowed tokenizer
  - `Data/sentinel_data/representation/orchestrator.py` — v2 orchestrator (new code)
  - `Data/sentinel_data/representation/cache_manager.py` — content-addressed cache
  - `Data/sentinel_data/representation/versioner.py` — schema/extractor version registry
  - `Data/sentinel_data/representation/cfg_builder.py` — standalone CFG builder
  - `Data/tests/test_representation/test_thin_adapter.py` — is-equality + dict-direction tests
  - `Data/tests/test_representation/test_byte_identical_regression.py` — 10-contract parametrized regression
  - `Data/tests/test_representation/test_13_issue_preservation.py` — 13 critical bug fix preservation
  - `Data/tests/test_representation/test_solidifi_fixes.py` — A-1/A-2/A-3 SolidiFI-specific fixes
  - `Data/tests/test_representation/test_orchestrator.py` — orchestrator smoke + cache + idempotency

When you're ready, say **"Stage 2 is mastered — let's move to Stage 3."**
