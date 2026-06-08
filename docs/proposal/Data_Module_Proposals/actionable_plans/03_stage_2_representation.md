# Actionable Plan — Stage 2: Representation Extraction (port from ml/)

**Date:** 2026-06-23
**Stage:** 2 of 8 (Week 3: Jun 23–29)
**Owner:** SENTINEL data engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §3.3, §5 (Week 3)
**Audit ref:** [`AUDIT_PATCHES.md`](AUDIT_PATCHES.md) §0 (F1, F2, F3, F4, F5, F6, F7, F8, F11, F23), §1 (2-P1 through 2-P12), §2 (C-7 performance budget, C-9 complexity proxy)
**Exit criteria:** `sentinel-data represent --source scabench` produces 30 graph `.pt` + 30 token `.pt` + 30 sidecar `.rep.json` files for the ScaBench fixture; a **36-issue pre-Run-8 audit regression test** confirms the new path preserves all A1–A38 fixes and produces byte-identical output to the original `ml/` path; **performance budget: 100-file extraction in < 5 min on 8 cores** (within ±10% of the old `ml/` path per 2-P11).

---

## Goal

Move the existing representation extraction code from `ml/src/data_extraction/` and `ml/src/preprocessing/` (graph extractor + graph schema + tokenizer) into `sentinel_data/representation/`, **without changing any extraction logic**. After this stage, both the old `ml/` import path and the new `sentinel-data` import path produce byte-identical graphs and tokens for the same input contract. The regression test that proves this is the gate; Stage 7 (export + seam) is the stage that actually switches the ML module over.

Four new builders (CFG, PDG, call graph, opcode) are also added in this stage as **opt-in** representation channels (Stage 7's v2 baseline does not use them, but they are exposed for the v2.1 architecture).

---

## Why this stage third

Stage 1 produced preprocessed `.sol` files. Stage 2 is the first stage that turns source code into model-consumable artifacts. Doing representation as a dedicated stage (rather than folding it into preprocessing) is the design decision that lets the same source code produce multiple representations (graph for GNN, tokens for Transformer, CFG for call-flow analysis, etc.) and lets each representation be cached and versioned independently.

The Stage-0 design choice to keep the original code in `ml/` and ship only a stub in `sentinel-data` means this stage is the first where the two code paths coexist. The regression test that proves byte-identical output is what makes that coexistence safe.

---

## Design decisions

### D-2.1 — No extraction logic changes in this stage

The port from `ml/src/data_extraction/` to `sentinel_data/representation/` is a *move*, not a *rewrite*. Every node feature, every edge type, every extraction detail is preserved bit-for-bit. The only changes are: (a) module paths, (b) `__init__.py` re-exports, (c) imports of shared utilities, (d) the addition of a new `cache_manager.py` and `versioner.py` around the moved code.

This decision is the key to the seam swap in Stage 7. If the moved code produces different output from the original, the ML module's active Run 9 training pipeline breaks. The regression test is the gate.

### D-2.2 — Schema version is frozen at v9 (confirmed; the proposal §2 was wrong)

Per the verified live state of `ml/src/preprocessing/graph_schema.py:161,175,218` (confirmed 2026-06-08), the active schema is **v9**:
- `FEATURE_SCHEMA_VERSION = "v9"`
- `NODE_FEATURE_DIM = 12`
- `NUM_NODE_TYPES = 14`
- `NUM_EDGE_TYPES = 12`
- `_MAX_TYPE_ID = 13.0`

The v2 build freezes at v9. The proposal §2 was wrong (it said v8); the Stage 0 stub from task 0.4 must use v9 constants, and the regression test in 2.6 compares v9 output to v9 output. No v8 vs v9 question remains — the decision is v9, period.

**The v9 schema additions** (feat[11] `in_unchecked_block`, CFG_NODE_ARITH, EXTERNAL_CALL edge type) ARE in scope for Stage 2 — they are part of the v9 schema, not a future change. The deferred-schema decision (proposal §7) is about NOT adding *new* schema features in v2; the v9 features are already in the active training pipeline and must be preserved.

### D-2.3 — Tokenization moves to `representation/tokenizer.py` unchanged

The existing `ml/src/data_extraction/tokenizer.py` (CodeBERT windowed, stride=256, [4, 512] windows) moves to `sentinel_data/representation/tokenizer.py` with no changes. The new `scripts/retokenize_windowed.py` (which orchestrates batch tokenization) is replaced by the new `sentinel-data represent` CLI subcommand, but the underlying tokenizer function is the same code.

### D-2.4 — Four new builders (CFG, PDG, call graph, opcode) are additive; PDG/call-graph/opcode DEFERRED to v3.1

The four new builders are **additive** for the v2 baseline. They are exposed in the CLI as opt-in flags (`--emit-cfg`, `--emit-pdg`, `--emit-callgraph`, `--emit-opcode`) but the default `sentinel-data represent` run does not produce them. The v2 baseline (Run 11) trains on **AST + the in-graph CFG (CFG_NODE_* node types) + the existing CONTROL_FLOW edges** — the same as Run 9 did. The new standalone CFG builder is *not* needed for Run 11.

**Per AUDIT_PATCHES 2-P9, 2-P12:** PDG, call-graph, and opcode builders are **DEFERRED to v3.1** (not Stage 2). Reasons:
- The "lightweight PDG" described in the original plan is a v3+ feature, not v2
- Time spent on PDG is better spent on Stage 4 verification
- The schema additions for standalone CFG / PDG would be a v2.1 schema change (not a v2 port)
- Run 11 doesn't need them

The **CFG builder is shipped in Stage 2** as the only new builder, because it's straightforward (uses Slither's IR, which is already imported in `graph_extractor.py`). All four builders must be gated by the cache invalidation logic (2.7 below) — the versioner must invalidate the cache when the Slither version changes.

This decision is per the proposal §3.3: "Run 11 trains on AST+CFG as v1 did, and PDG / call graph / opcode are exposed for v3+ architectures." (v1's CFG = the in-graph CFG_NODE_* types, not the new standalone builder.)

### D-2.5 — Content-addressed representation cache

Every representation is keyed by `(sha256 of source, schema_version, extractor_version)`. A change to *any* of those three values invalidates the cache for that file. The cache is stored under `data/representations/<source>/` with a per-source manifest recording which representations have been computed. Re-running `sentinel-data represent` skips already-computed files unless `--force` is passed.

This is the same pattern as `InferenceCache` in `ml/src/inference/cache.py` (content-addressed, schema-version-invalidated) — the data-side version is a superset that also includes the extractor version.

### D-2.6 — Sidecar `rep.json` records provenance per file

Every `.pt` graph is accompanied by a `.rep.json` containing: `schema_version`, `extractor_version`, `node_count`, `edge_count`, `window_count` (for tokens), `compute_time_ms`, `cache_hit` (bool). The sidecar is what makes "this graph was built by which version of which extractor" queryable for any later audit. The sidecar is also the input to the `analysis/feature_dist.py` tool in Stage 7 (Run 9's complexity-proxy finding was discovered by analyzing graph statistics — the sidecar makes that analysis automatic).

### D-2.7 — Parallel import paths are intentional and temporary

From Stage 2 through Stage 7, the `sentinel-ml` import path `from src.preprocessing.graph_extractor import ...` and the new path `from sentinel_data.representation.graph_extractor import ...` both work and produce identical output. The active Run 9 training pipeline continues to use the old path. Stage 7 deletes the old path. This temporary redundancy is what makes the seam swap safe.

---

## Tasks — ordered, each with verifiable exit condition

### 2.1 — Resolve the schema version question (D-2.2)

**Schema version is confirmed v9** (per `ml/src/preprocessing/graph_schema.py:161,175,218` verified 2026-06-08, see AUDIT_PATCHES F1). The Stage 0 stub from task 0.4 must use v9 constants — this is the *first* task of Stage 2, before any code moves. Constants to verify in the stub:

- `FEATURE_SCHEMA_VERSION = "v9"` (NOT `"v8"` as the proposal §2 says)
- `NODE_FEATURE_DIM = 12` (NOT 11)
- `NUM_NODE_TYPES = 14` (NOT 13)
- `NUM_EDGE_TYPES = 12` (NOT 11)
- `_MAX_TYPE_ID = 13.0` (NOT 12.0)

The regression test in 2.6 will detect any inconsistency.

**Why first:** the stub in Stage 0 is supposed to reference v9. If it doesn't (e.g. if the proposal's v8 claim was taken at face value), the byte-identical regression test will fail because the old `ml/` path uses v9. The stub is the single most important file to verify before the port begins.

**Exit condition:** schema version is v9 in the stub; all 5 constants above match `ml/src/preprocessing/graph_schema.py` line-for-line.

**Commit:** `chore(data-rep): align stub schema version with active ML module (v9 confirmed)`

---

### 2.2 — Port `ml/src/preprocessing/graph_schema.py` to `sentinel_data/representation/graph_schema.py`

Move the file contents (NODE_TYPES, EDGE_TYPES, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, NODE_FEATURE_DIM, NUM_EDGE_TYPES, VISIBILITY_MAP) to the new location. Remove the `STUB = True` marker (the stub is being replaced with the real port). Update the module docstring to point to the new location for any future reader. Keep the `ml/src/preprocessing/graph_schema.py` file in place — it is still the active import path until Stage 7.

**Why:** the schema is the simplest move first; testing it independently catches import errors before the bigger extractor file is moved.

**Exit condition:** `from sentinel_data.representation.graph_schema import FEATURE_SCHEMA_VERSION, NODE_FEATURE_DIM, NUM_EDGE_TYPES` works; constants match the `ml/` originals.

**Commit:** `feat(data-rep): port graph_schema.py from ml/ to sentinel_data/`

---

### 2.3 — Port `ml/src/preprocessing/graph_extractor.py` to `sentinel_data/representation/graph_extractor.py`

Move the file contents (extract_contract_graph, GraphExtractionConfig, GraphExtractionError, and any helper functions) to the new location. Update internal imports (the file imports from `graph_schema` — those imports now point to the new sibling location). Keep the `ml/src/preprocessing/graph_extractor.py` file in place for now (Stage 7 deletes it).

**Why next:** the schema is the data; the extractor is the function that produces it. Testing the extractor against the schema catches import errors and any subtle differences in how the schema constants are used internally.

**Exit condition:** `from sentinel_data.representation.graph_extractor import extract_contract_graph, GraphExtractionConfig, GraphExtractionError` works; calling `extract_contract_graph()` on a known-input contract produces a PyG Data object with the expected shapes.

**Commit:** `feat(data-rep): port graph_extractor.py from ml/ to sentinel_data/`

---

### 2.4 — Port `ml/src/data_extraction/ast_extractor.py` to `sentinel_data/representation/ast_extractor.py`

Move the file contents to the new location. The file is a thin orchestrator (parquet → solc version resolve → multiprocessing Pool → .pt files) and the only real changes are: (a) import paths for `graph_extractor` and `graph_schema` (now from sibling modules in the new package), (b) the output directory changes from `ml/data/graphs/` to `data/representations/<source>/`.

**Why:** the AST extractor wraps the graph_extractor; moving it last lets us test the full extract pipeline end-to-end against the moved dependencies.

**Exit condition:** `sentinel-data represent` (real run, not stub) extracts 30 graphs from the 30 preprocessed ScaBench files; output `.pt` files loadable as PyG Data.

**Commit:** `feat(data-rep): port ast_extractor.py to sentinel_data/`

---

### 2.5 — Port `ml/src/data_extraction/tokenizer.py` to `sentinel_data/representation/tokenizer.py`

Move the CodeBERT windowed tokenizer to the new location. The file has more logic than the AST extractor (MD5 naming, multiprocessing, checkpoint/resume) and is its own self-contained module. Move it without changes except for the import paths and the output directory.

**Why:** tokenization is independent of graph extraction; moving it separately lets us test each in isolation.

**Exit condition:** tokenizing a known input produces the same `input_ids` and `attention_mask` arrays whether called via the old or new import path.

**Commit:** `feat(data-rep): port tokenizer.py to sentinel_data/`

---

### 2.6 — Write the regression test suite (byte-identical output gate + 36-issue pre-Run-8 audit)

Author `Data/tests/test_representation/test_byte_identical_regression.py` and a companion `test_36_issue_audit_preservation.py`. The two test files together are the gate for Stage 7.

**`test_byte_identical_regression.py`** — the byte-identical regression test. Takes a fixture of ~10 hand-picked ScaBench files (covering all 4 representative Solidity eras and the `unchecked{}` 0.8.x case), runs `extract_contract_graph()` + `tokenize()` via both the old `ml/` import path and the new `sentinel_data` import path, asserts the resulting PyG Data objects and token tensors are byte-identical (using `torch.equal` for tensors and deep equality for PyG Data). Test fails loud if any field differs.

**`test_36_issue_audit_preservation.py`** — the 36-issue regression test (per AUDIT_PATCHES 2-P2, F23, I-1). The 36 issues (A1–A38) span **9 source files** in `ml/src/`, not just the 4 named in the original plan. The test file is organized by issue ID, with one test function per critical fix. The critical tests (per 2-P3 through 2-P6):

| Issue | Test |
|---|---|
| **A9** `now` keyword detection | Fixture: 0.4.x contract using `now`; assert `feat[2]` = 1.0 for the function |
| **A15** def_map scope_key | Fixture: two functions with same var name in different scopes; assert no spurious cross-function DEF_USE edges |
| **A20** label=0 hardcode | Fixture: CSV with known labels; assert graph `.y` matches CSV (not 0) |
| **A34** prefix sort dim | Fixture: contract with mixed `external_call_count`; assert `select_prefix_nodes` uses `raw_node_features[:, 10]` not post-GAT dim |
| **A38** NaN before backward | Fixture: deliberately-NaN injection; assert trainer guards before `loss.backward()` |
| **EMITS edge (Interp-6)** | Fixture: contract with `emit Event();`; assert EMITS edge exists in the graph |
| **CALL_ENTRY for external** (F7) | Fixture: contract with `HighLevelCall`; assert EXTERNAL_CALL self-loop edge exists |
| **LibraryCall classification** (F25) | Fixture: `SafeMath.add()` library call; assert NOT counted as cross-contract HighLevelCall |
| **Resume overwrite fix** (F6) | Fixture: checkpoint + resume command; assert default `resume_model_only=False` |
| **Return-ignored fix** (F29) | Fixture: function with unused return; assert `feat[7] = 1.0` |
| **A31 fusion BUG-C2** token_norm | (already fixed) — fixture: high-norm tokens; assert LayerNorm is applied before projection |
| **A18 ICFG map** | Fixture: contract with internal calls; assert CALL_ENTRY/RETURN_TO edges exist |
| **A10 _cfg_node_type** | Fixture: contract with diverse CFG ops; assert node types are not silently OTHER |

The byte-identical test (which is what was originally in the plan) is a subset of the 36-issue test. The 36-issue test is the comprehensive gate.

**Why the 36-issue test is critical:** the pre-Run-8 audit `docs/pre-run-fixes/validated_audition.md` documents 36 confirmed code issues across 9 source files. Run 7/8/9 are working *because* these fixes are in place. The port must preserve all of them, not just the 4 named in the original plan. The 36-issue test is the proof that the port was a move, not a rewrite.

**Why per-issue test functions (not one big test):** when a regression happens, the failing test name points to the specific issue. Debug time is minutes, not hours.

**Performance regression test (2-P11):** the test also asserts that extracting 100 contracts via the new path takes within ±10% of the time of the old `ml/src/preprocessing/` path. If slower, profile and fix.

**Exit condition:** all byte-identical tests pass for the 10-file fixture; all 36-issue tests pass; the performance regression test passes; covered in CI from this stage forward.

**Commit:** `test(data-rep): add byte-identical regression + 36-issue pre-Run-8 audit preservation suite`

**Exit condition:** regression test passes against 10 fixture files; covered in CI from this stage forward.

**Commit:** `test(data-rep): add byte-identical regression test for ported extractors`

---

### 2.7 — Implement the CFG builder only (PDG / call-graph / opcode DEFERRED to v3.1)

**Per AUDIT_PATCHES 2-P9, D-2.4 above:** only the CFG builder is shipped in Stage 2. PDG, call-graph, and opcode are DEFERRED to v3.1.

Author `sentinel_data/representation/cfg_builder.py` (only). It is a self-contained module exposing a `build_cfg(source_path, source_text) -> CfgArtifact` function. Uses Slither's internal IR (already imported in `graph_extractor.py`); produces per-function CFG in a normalized form.

Gated by CLI flag `--emit-cfg` (default off). The CFG builder is NOT in the regression test (it is new code, not a port).

For v3.1 (post-Run-11), PDG / call-graph / opcode are added as separate stages. The Stage 2 plan leaves the directory structure ready (`representation/{pdg_builder,call_graph,opcode_extractor}.py` are placeholders with a "DEFERRED to v3.1 — see AUDIT_PATCHES 2-P9" docstring) so the v3.1 work is a drop-in.

**Why the CFG builder but not the others:** the CFG builder is straightforward — Slither's IR is already imported in the existing `graph_extractor.py`, so the import infrastructure is in place. PDG requires a new data-flow analysis library; call-graph requires `all_high_level_calls` API (which has changed between Slither versions); opcode requires a separate bytecode compilation step. These three are larger projects that justify their own stage.

**Why all four must be gated by the cache invalidation logic:** the versioner (2.8) must invalidate the cache when the Slither version changes. If a Slither API change breaks the CFG builder, the cache is invalidated and the new representations are recomputed; the regression test catches it.

**Exit condition:** CFG builder compiles and imports; can be called individually with `--emit-cfg` flag; produces a sensible output for a single test contract; PDG/call-graph/opcode are placeholder files with deferral docstrings.

**Commit:** `feat(data-rep): add CFG builder (opt-in, deferred PDG/call-graph/opcode to v3.1)`

---

### 2.8 — Implement `cache_manager.py` and `versioner.py`

Author `sentinel_data/representation/cache_manager.py` (the content-addressed cache) and `versioner.py` (the schema/extractor-version invalidation logic). The cache stores representations under `data/representations/<source>/<sha256>.<ext>` and the manifest under `data/representations/<source>/.cache_manifest.json`. The versioner maintains a global `data/representations/_version_registry.json` recording the current `(schema_version, extractor_version)` pair; a re-run checks each file's recorded version against the current and recomputes if mismatched.

**Why:** the cache makes re-runs fast (the 41K existing graphs would take days to recompute); the versioner prevents the silent-mix-of-versions failure mode that bit us in Run 8 (graphs from v8 mixed with graphs from v9 in the same dataset).

**Exit condition:** cache_manager stores + loads representations correctly; versioner invalidates a file when the schema version is bumped; regression test still passes.

**Commit:** `feat(data-rep): add content-addressed cache + version registry`

---

### 2.9 — Wire the `sentinel-data represent` CLI subcommand

Connect `cli.py` `represent` subcommand to the new module. The CLI reads sources from `config.yaml`, iterates over preprocessed files, runs the graph extractor + tokenizer (and optionally the 4 new builders), writes outputs to `data/representations/<source>/`, and updates the cache manifest. Add `--force` to recompute regardless of cache hit, `--emit-*` flags for the 4 new builders.

Update `dvc.yaml` stage `represent` to call `sentinel-data represent`.

**Exit condition:** `sentinel-data represent --source scabench` produces 30 `.pt` (graph) + 30 `.pt` (tokens) + 30 `.rep.json` (sidecar) for the ScaBench fixture.

**Commit:** `feat(data-rep): wire CLI + DVC for the represent stage`

---

### 2.10 — Add tests for the new representation code

Author `Data/tests/test_representation/` with:
- **Port regression test** (the 2.6 test, lives in this dir)
- **Cache tests** — store/load/invalidate round-trip
- **Versioner tests** — schema bump invalidates; extractor bump invalidates
- **New builder tests** — each of the 4 new builders produces a sensible output for a single test contract
- **CLI tests** — `sentinel-data represent --source scabench --dry-run` prints the planned action

**Exit condition:** `poetry run pytest tests/test_representation -v` passes; coverage > 80%.

**Commit:** `test(data-rep): add full test suite for representation stage`

---

### 2.11 — Author `ADR-0003-representation-port-design.md`

Document the key design decisions: no logic changes (D-2.1), **v9 schema freeze (D-2.2 — confirmed, not a question)**, parallel import paths (D-2.7), additive new builders (D-2.4 with v3.1 deferral), content-addressed cache (D-2.5), sidecar provenance (D-2.6).

**Exit condition:** file exists; cites the regression test as the gate; references the deferred schema decision.

**Commit:** `docs(data): add ADR-0003 for representation port design`

---

## What NOT to fix (preservation list)

| Bug | Status | File:line | Stage 2 action |
|---|---|---|---|
| **A9** `now` keyword | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:587-605` | Do not re-fix. The 36-issue regression test (2.6) has a specific test for this. |
| **A15** def_map by name | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py:1147-1179` | Do not re-fix. The 36-issue test has a scope_key test. |
| **A20** label=0 hardcode | ✅ FIXED | `ml/src/data_extraction/ast_extractor.py:290,342,395` | Do not re-fix. The 36-issue test has a CSV-label test. |
| **A34** prefix sort dim | ✅ FIXED | `ml/src/models/sentinel_model.py:356,367` | Do not re-fix. The 36-issue test has a prefix-sort test. |
| **A38** NaN before backward | ✅ FIXED | `ml/src/training/trainer.py` | Do not re-fix. The 36-issue test has a NaN-guard test. |
| Resume overwrite | ✅ FIXED | `ml/src/training/trainer.py:383,1184,1206,1212` | Do not re-fix. Stage 8 uses full-resume default. |
| **EMITS edge bug** | ⚠ OPEN (Interp-6) | `ml/src/preprocessing/graph_extractor.py` | Stage 7 seam swap must fix (per 7-P6). The 36-issue test asserts the bug exists pre-fix. |
| **CALL_ENTRY cross-function for external** | ⚠ PARTIAL FIX | `ml/src/preprocessing/graph_extractor.py:1001` | Self-loop only; full cross-function edge is post-Run-11. The 36-issue test asserts the self-loop is present. |
| **LibraryCall <: HighLevelCall** | ⚠ KNOWN | `ml/src/preprocessing/graph_extractor.py:1081` | `_compute_external_call_count` relies on isinstance; library calls counted as cross-contract. The 36-issue test asserts the current behavior (it's not a bug, it's a design choice). |
| **v8 schema** (in old proposal §2) | ❌ WRONG | (proposal §2 only) | **CORRECTED**: the active schema is v9. Stage 0 stub and Stage 2 port use v9 throughout. Per F1, this is now verified live. |

---

## Final exit criteria check

| # | Check |
|---|---|
| 1 | `sentinel-data represent --source scabench` produces 30 graph `.pt` + 30 token `.pt` + 30 `.rep.json` for the ScaBench fixture |
| 2 | The byte-identical regression test (`test_byte_identical_regression.py`) passes against the 10-file fixture for all 4 files: `graph_schema`, `graph_extractor`, `ast_extractor`, `tokenizer` |
| 3 | The **36-issue pre-Run-8 audit regression test** (`test_36_issue_audit_preservation.py`) passes for all 13 critical issues (A9, A15, A20, A34, A38, EMITS, CALL_ENTRY, LibraryCall, Resume, Return-ignored, A31, A18, A10) |
| 4 | `from sentinel_data.representation.graph_extractor import extract_contract_graph` works; calling it on a known input produces the same `Data` object as the old import path |
| 5 | The CFG builder compiles and can be called with `--emit-cfg` flag; **PDG/call-graph/opcode are placeholders with deferral docstrings (NOT shipped in v2)** |
| 6 | The cache_manager stores + loads representations correctly; versioner invalidates on schema bump AND on Slither version bump |
| 7 | The active Run 9 training pipeline (using the old `ml/` import path) still works unchanged |
| 8 | **Performance budget: 100-file extraction in < 5 min on 8 cores (within ±10% of old `ml/` path)** |
| 9 | `dvc repro represent` runs end-to-end |
| 10 | `poetry run pytest tests/test_representation -v` passes with > 80% coverage |
| 11 | `ADR-0003-representation-port-design.md` is committed; **references the 36-issue audit, not just the v9-schema additions** |

All 11 pass → **Stage 2 complete**. Tag `data-stage-2`, proceed to Stage 3.

---

## Risk register

| Risk | Mitigation |
|---|---|
| The port introduces a subtle difference (e.g. import order, default arg value) that breaks the byte-identical regression test | The test is run *first* against the moved code; if it fails, debug until it passes before continuing. No other stage work depends on Stage 2 until the regression test passes. |
| (RESOLVED) The Run 9 schema is v9 but the proposal §2 said v8 | D-2.2 is now resolved — the schema is v9, the Stage 0 stub uses v9, the regression test in 2.6 compares v9 to v9 |
| The 4 new builders (especially PDG) are large, untested code paths that could delay the stage | They are opt-in; the v2 baseline does not use them. A clean failure in PDG builder does not block Stage 3 onward — the default `represent` run skips them. |
| The cache invalidation logic has a bug that causes a partial recompute (some files v8, some v9 in the same dataset) | The versioner is unit-tested explicitly in 2.10; the `_version_registry.json` is the global source of truth that prevents mixed versions |
| Tokenization in the new path is slower than the old path (e.g. import overhead) | Performance regression test in 2.10; if the new path is > 10% slower, profile and fix |
| Slither version differences between `ml/` and `sentinel-data` cause graph_extractor to produce different output | The Dockerfile.pins the solc/slither versions; the `pyproject.toml` pins `slither-analyzer >=0.10.0`. If the versions differ, the byte-identical test catches it. |

---

**End of Stage 2 actionable plan. Total estimated time: 5 working days (Jun 23–27), with Jun 28–29 as buffer.**
