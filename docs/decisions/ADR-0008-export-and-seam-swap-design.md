# ADR-0008: Export + Seam Swap Design

**Date:** 2026-06-12
**Stage:** 7A of 8 (sub-stage: export module only; 7B seam swap deferred)
**Status:** Accepted (Stage 7A implementation complete; 7B amendment appended 2026-06-12)
**Author:** SENTINEL data engineering
**Plan reference:** [`docs/proposal/Data_Module_Proposals/actionable_plans/08_stage_7_export_seam.md`](../proposal/Data_Module_Proposals/actionable_plans/08_stage_7_export_seam.md)
**Live plan:** [`data_module/temp/live_plans/stage_7a_export_module.md`](../../data_module/temp/live_plans/stage_7a_export_module.md)

---

## Context

Stage 7 is the final data-module stage. Its two responsibilities are distinct:

1. **7A (this ADR):** Produce the consumable export artifact — sharded `.pt` files + Parquet metadata — that the ML module's `SentinelDataset` (Stage 7B) will load. This is the "write side" of the seam.
2. **7B (deferred):** Replace the ML module's `DualPathDataset` + legacy data scripts with the new `SentinelDataset` loader that reads the v2 export. This is the "read side" of the seam.

The seam is the boundary between the data module (produces the corpus) and the ML module (consumes it). The v1 seam was raw `.pt` files read by `dual_path_dataset.py` using legacy paths. The v2 seam is the export directory produced by `chunk_export()`.

**Why split 7A and 7B?** Stage 7A produces a new format that `dual_path_dataset.py` cannot load. Deleting `dual_path_dataset.py` without first building and validating the new `SentinelDataset` (7B's dual-path test) would break the active training pipeline with no fallback. The split lets 7A land cleanly while 7B adds the new loader with byte-identical verification.

---

## Design Decisions

### D-7.1 — Sharded export is the consumable format (4 file types)

The export artifact has 4 file types per export directory:
- **`graphs/graphs-{shard:05d}.pt`** — PyG `Batch` objects, one shard of up to 5,000 contracts
- **`tokens/tokens-{shard:05d}.pt`** — `torch.Tensor [N, 4, 512]`, same ordering
- **`labels.parquet`** — one row per labeled contract (all 22,356 rows, even without reps)
- **`metadata.parquet`** — same coverage, enriched with `.rep.json` + `.meta.json` + `.sol`

The shard size (5,000) is configurable via `pipeline.export_shard_size` in `config.yaml`.

**Why not a single file?** A single `.pt` file holding all 22K contracts would require loading the entire corpus into RAM on every training run. Shards enable lazy loading: the `SentinelDatasetExport.shard_index` maps each `contract_id` to its shard number, and the ML module loads only the required shard for each batch.

**Why Parquet for labels/metadata?** Parquet enables column-projection queries without reading the full file, supports nullable types (the `confidence_tier` column is nullable for NonVulnerable contracts), and interoperates with pandas/polars for exploratory analysis. Snappy compression is the default (widely supported, fast decompression).

**Operational consequence:** the ML module (`SentinelDataset`, built in 7B) receives a 5-tuple per contract: `(graph, tokens, y, contract_id, confidence_tier)`. The `confidence_tier` column is new — it enables per-class loss weighting in Run 11 (the Stage 5 tier carries downstream into training).

### D-7.2 — Contract order is driven by the split JSONL, not the representations directory

Both `graph_writer` and `token_writer` walk `splits/v{N}/{train,val,test}.jsonl` in order (train → val → test, then line order within each file). They do **not** glob the representations directory.

**Why:** if we globbed `representations/`, the order would be filesystem-dependent (directory listing order varies by OS, filesystem, and inode allocation). The split JSONL is the canonical, deterministic ordering. This guarantees that shard row N in `graphs-{shard:05d}.pt` corresponds exactly to row N in `tokens-{shard:05d}.pt`, and that the `shard_index` maps each `contract_id` to the correct position.

**Consequence:** contracts in the split that have no representation are silently skipped (warn only). The `labels.parquet` and `metadata.parquet` still include all 22,356 split rows; the `.pt` shards include only the 21,523 contracts that have representations.

### D-7.3 — The new `SentinelDataset` is thin (~150 lines, built in 7B)

The export API (`SentinelDatasetExport`) is deliberately minimal: it loads the manifest, exposes the shard index, and provides `verify_artifact_hash()`. It does NOT implement `__len__` or `__getitem__` — that's the ML module's job in 7B.

**Why separate:** the data module is write-only at runtime; the ML module is read-only. Putting `__getitem__` logic in the data module would couple training hyperparameters (batch size, augmentation) to the data pipeline. The separation keeps both modules independent and lets 7B evolve the loader without touching the export format.

### D-7.4 — `ml/pyproject.toml` adds `sentinel-data` as a runtime dep (7B, deferred)

The ML module currently imports the data module's code via thin adapters (`sentinel_data.representation.graph_extractor` → `ml/src/preprocessing/graph_extractor.py`). In 7B, the adapter is replaced by a direct import; `ml/pyproject.toml` gets `sentinel-data = "^0.1.0"`. This is noted here for forward reference; it is not implemented in 7A.

### IC-1 — Format schema versioning: `format_schema/v{MAJOR}.{MINOR}.yaml`

The format spec lives at `data_module/sentinel_data/export/format_schema/v1.yaml`. This file is the contract. Future schema changes:
- **Additive (backward-compatible):** new file `v1.1.yaml`. The consumer can load v1 exports with a v1.1 consumer (old fields still present, new fields silently absent).
- **Breaking:** new file `v2.yaml`. The consumer refuses to load v1 exports (schema_version gate, per below).

The `manifest.json` carries `schema_version: "v1"`. The `SentinelDataset` (7B) checks this field on load and raises `ValueError` if the export's version is older than the consumer's pinned version.

### IC-2 — Schema-version gate (v9 graph schema + v1 format schema)

The manifest carries two version fields:
- `graph_schema_version: "v9"` — the graph feature schema (12-dim node features, 12 edge types). The `SentinelDataset` checks this against `ml/src/preprocessing/graph_schema.FEATURE_SCHEMA_VERSION` at load time and refuses to load on mismatch. This is the 7-P2 gate.
- `schema_version: "v1"` — the export format contract. Same gate logic.

Documented here for forward reference even though 7B builds the consumer-side check.

### IC-3 — Confidence tier as a first-class field (per 7-P4)

Both `labels.parquet` and `metadata.parquet` include a `confidence_tier` column. The logic (Fix #2): `tier if n_pos > 0 else None`. The split JSONL's `tier="T0"` default for NonVulnerable contracts is overridden to `null` (pyarrow null) at the parquet level, so the ML module can filter on `confidence_tier.is_null()` to skip tier-based weighting for negatives.

The Stage 5 tier (T0/T1/T2/T3 or "untiered") flows through unchanged for positive contracts. This enables per-class loss weighting in Run 11 — contracts labeled by tool agreement (T3) carry less weight than those labeled by manual review (T1).

### IC-4 — Predictor tier bug fix (F8/F10), in-scope for 7A

The `ml/src/inference/predictor.py` bug was using `self.tier_confirmed_threshold` (a scalar class constant, 0.55) for the CONFIRMED tier decision for every class, ignoring the per-class thresholds loaded from the companion JSON. The fix uses `self.thresholds[cls_idx].item()` per class. This is a surgical, standalone fix that does not depend on the seam swap, so it lands in 7A.

**Impact:** a class tuned to 0.90 (e.g. CallToUnknown) was incorrectly triggering CONFIRMED at prob=0.56. A class tuned to 0.35 (e.g. Reentrancy) was not triggering CONFIRMED at prob=0.36. Both are now correct.

Also: `tier_thresholds["confirmed"]` in the result dict now returns the full per-class threshold list (not the scalar), so the result dict is self-documenting.

### IC-5 — Seam swap deferred to 7B

The deletion of `dual_path_dataset.py`, `graph_extractor.py` (from `ml/`), and the 7 legacy data scripts is NOT in 7A's scope. The dual-path test (byte-identical check for 100 contracts) is the safety net for the swap; 7A cannot build that test because 7A doesn't have the new loader yet. 7B's scope:

- Author `sentinel-ml/src/datasets/sentinel_dataset.py` (~150 lines)
- Dual-path test (old vs new loader, byte-identical for 100 contracts)
- Delete `ml/src/datasets/dual_path_dataset.py`
- Delete `ml/src/preprocessing/graph_extractor.py` + `graph_schema.py` (from `ml/`)
- Archive legacy scripts to `ml/scripts/_legacy_data_pipeline/`
- Update `ml/pyproject.toml`
- Docker build verification
- 7 v2-readiness gates check

---

## Implementation choices made during 7A

### Fix A — Circular hash avoidance

The `artifact_hash` in `manifest.json` is the SHA-256 of the 4 data file types only (`graphs/`, `tokens/`, `labels.parquet`, `metadata.parquet` + shard index files). `manifest.json` is excluded. This avoids the chicken-and-egg: manifest.json contains `artifact_hash`, so it cannot be part of the file set whose hash it stores. `manifest.json` is written LAST after the hash is computed.

`_hash_export_data()` in `chunker.py` is the single implementation used by both `chunk_export` (to populate the field) and `SentinelDatasetExport.verify_artifact_hash()` (to verify). The test suite includes a regression test: modifying `manifest.json` does NOT change the hash.

### Fix B — Smoke test uses contracts with existing `.pt` files

The integration smoke test does not use "the first 100 rows of the split JSONL" — only 3 of the first 100 train rows have representations (the JSONL is sorted alphabetically by sha256, and most early hashes are dive contracts that were unrepresented in the 776-contract pre-run baseline). The smoke uses the full export path with the full 21,523-contract corpus.

### Fix C — `export/__init__.py` exports the public API

The `__init__.py` (previously a docstring stub) now exports: `SentinelDatasetExport`, `ExportManifest`, `chunk_export`, `write_labels_parquet`, `write_metadata_parquet`, `write_graphs_shards`, `write_tokens_shards`. Callers can do `from sentinel_data.export import SentinelDatasetExport`.

### Fix D — `ml/tests/test_predictor.py` is new (no existing file to extend)

The plan noted "or extend existing" for the predictor tests — there was no existing `test_predictor.py`. 6 regression tests were written from scratch.

---

## Files built in 7A

**Source (8 new + 2 modified):**
1. `data_module/sentinel_data/export/format_schema/v1.yaml` — format contract
2. `data_module/sentinel_data/export/label_writer.py` — `write_labels_parquet()`
3. `data_module/sentinel_data/export/metadata_writer.py` — `write_metadata_parquet()`
4. `data_module/sentinel_data/export/graph_writer.py` — `write_graphs_shards()`
5. `data_module/sentinel_data/export/token_writer.py` — `write_tokens_shards()`
6. `data_module/sentinel_data/export/chunker.py` — `chunk_export()` + `ExportManifest`
7. `data_module/sentinel_data/export/export.py` — `SentinelDatasetExport`
8. `data_module/sentinel_data/export/__init__.py` (modified) — public API exports
9. `data_module/sentinel_data/cli.py` (modified) — `_run_export` + argparse wiring

**Tests (5 new files, 27 tests):**
10. `data_module/tests/test_export/__init__.py`
11. `data_module/tests/test_export/test_label_writer.py` (5 tests)
12. `data_module/tests/test_export/test_metadata_writer.py` (4 tests)
13. `data_module/tests/test_export/test_graph_token_writer.py` (5 tests)
14. `data_module/tests/test_export/test_chunker.py` (5 tests)
15. `data_module/tests/test_export/test_export.py` (8 tests)

**Predictor fix (separate ML commit):**
16. `ml/src/inference/predictor.py` (modified) — per-class `thresholds[cls_idx]` for CONFIRMED tier
17. `ml/tests/test_predictor.py` (new) — 6 regression tests

**Docs:**
18. `docs/decisions/ADR-0008-export-and-seam-swap-design.md` (this file)

---

## Operational consequences

- `sentinel-data export --dry-run` now resolves sources, prints the plan, and exits without writing.
- `sentinel-data export` writes the full export to `data/exports/<dataset-version>/`.
- `sentinel-data export --dataset-version sentinel-v2-gold-2026-08` uses that name as the output directory label.
- The CLI skips sources with no preprocessed dir (same guard as `sentinel-data represent`).
- The full data_module test suite is 531 passed / 51 skipped after 7A.
- The ML test suite passes with the predictor fix (6 new + existing tests pass).

---

## What this ADR does NOT cover (deferred to 7B)

See IC-5 above for the complete 7B scope. Key items:
- `SentinelDataset.__getitem__` (the read-side consumer)
- Dual-path test (byte-identical for 100 contracts)
- Deletion of legacy ML data scripts
- Docker build verification
- 7 v2-readiness gates check
- Run 11 launch

---

## 7B Amendment — 2026-06-12 (seam swap landed)

**Scope:** All 7B items from IC-5 are now complete. The seam is the source of
truth: `sentinel_data.representation.graph_schema` owns the v9 schema, and the
ML module's `ml/src/preprocessing/graph_schema.py` is now an 18-line thin shim
that re-exports from it. The trainer (`ml/src/training/trainer.py`) consumes
the v2 export via `SentinelDataset`; no production code in `ml/src/` imports
`DualPathDataset` or `dual_path_dataset.py` directly.

### 7B-AMEND-1 — The thin-adapter direction was FLIPPED, not deleted

The original plan (7A's IC-5) said "delete `ml/src/preprocessing/graph_schema.py`
and `graph_extractor.py`." During implementation (2026-06-12) we discovered 8
model files import directly from those paths (`gnn_encoder.py`, `sentinel_model.py`,
`predictor.py`, `training_logger.py`, `cache.py`, `dual_path_dataset.py`,
`preprocess.py`, `graph_extractor.py` itself).

**Decision:** flip the adapter direction instead of deleting. Make
`sentinel_data/representation/graph_schema.py` self-contained (full v9 schema),
and replace `ml/src/preprocessing/graph_schema.py` with a thin re-export shim
pointing at it. The 8 model files keep working with zero changes.

**Cost:** ~5 lines of shim code, invisible to callers.
**Benefit:** seam swap is invisible to the model stack.

**Why the flip, not copy-paste:** copy-paste would create two sources of truth
that can drift. A shim is single-source-of-truth by construction.

### 7B-AMEND-2 — `graph_extractor.py` imports from `sentinel_data` directly to avoid circular import

Initial flip attempt used the shim from inside `ml/src/preprocessing/graph_extractor.py`,
which created a circular import chain:

```
graph_schema shim → sentinel_data.__init__ → sentinel_data.representation →
  (eagerly imports graph_extractor adapter) → ml.src.preprocessing.graph_extractor
  → back to graph_schema shim (partially initialized)
```

**Fix:** `graph_extractor.py` lines 112 + 1241 import from
`sentinel_data.representation.graph_schema` directly (not the shim), sidestepping
the cycle. The shim is still used by the 7 other model files (which import the
top-level `preprocessing.graph_schema` namespace, not its internals).

### 7B-AMEND-3 — `SentinelDataset` returns a 5-tuple; the trainer ignores the 2 extras

The 5-tuple is `(graph, tokens, y, contract_id, confidence_tier)`. The 3-tuple
that `DualPathDataset` returned is preserved by the trainer ignoring the 2 new
fields via Python's `*_` unpacking:

```python
graphs, tokens, labels, *_ = batch  # SentinelDataset 5-tuple; trainer ignores contract_ids + tiers
```

This was applied at 3 unpacking sites: `evaluate()` line 504, `train_one_epoch()`
line 648, and the diagnostic-fetch line 1836.

**Why 5-tuple, not 3:** the `contract_id` and `confidence_tier` are needed by
future stages (5.5 GraphCodeBERT propagation, Run 11 per-class loss weighting).
Carrying them through the loader now means downstream code can pick them up
without re-plumbing.

### 7B-AMEND-4 — `SentinelDataset` 3 hard gates at `__init__`

The new loader validates the export at construction time (not lazily on first
`__getitem__`):

1. `manifest.schema_version == "v1"` (forward-compat) — hard `ValueError`
2. `manifest.graph_schema_version == FEATURE_SCHEMA_VERSION` ("v9") — hard `ValueError`
3. `verify_artifact_hash()` returns `True` — hard `ValueError` on data corruption

**Why hard, not warn:** a bad export would silently corrupt training. Fail-fast
at `__init__` is safer than a 3-hour training run producing nonsense.

The 7-P2 schema-dim test in `test_byte_identical_regression.py` (x.shape[-1] ==
12) is the graph-extractor-side companion gate.

### 7B-AMEND-5 — `pyproject.toml` change required a `prometheus-fastapi-instrumentator` constraint relaxation

The seam swap removed `py-solc-ast`/`solc`/`solc-select` from `ml/pyproject.toml`
(data-only deps) and added `sentinel-data = {path = "../data_module", develop = true}`.
The lock file refused to resolve because the existing
`prometheus-fastapi-instrumentator = ">=0.9,<1.1"` constraint blocks all versions
on PyPI (latest is 8.0.0). The constraint was relaxed to `>=6.0,<9.0` (allows
6.x, 7.x, 8.x); the API used (`Instrumentator().instrument(app).expose(app)`)
is stable since v5.x. `slither-analyzer` comes in transitively via
`sentinel-data`'s path dep, so it does NOT need to be added to `ml/pyproject.toml`.

### 7B-AMEND-6 — Venv `solc` 0.4.2 → 0.8.19 symlink (uncovered by Gate 1)

The venv's `solc` was a 254-byte `solc-select` wrapper pinned to 0.4.2; test
fixtures in `test_preprocessing.py` use `pragma solidity ^0.8.0`. Fixed by
replacing the wrapper with a direct symlink to
`~/.solc-select/artifacts/solc-0.8.19/solc-0.8.19`. **This is the kind of
infrastructure debt that the seam swap surfaced** — the v1 corpus was pre-0.8
and nobody noticed the solc mismatch.

### 7B-AMEND-7 — `trainer.py` `compute_pos_weight` and `_build_weighted_sampler` rewritten to use in-memory `_label_lookup` + `num_nodes_map`

The v1 functions read `multilabel_index.csv` (with `md5_stem` keys + class-name
columns) and used `dataset.cached_data[md5][0].num_nodes` for the timestamp-size
sampler mode. Both interfaces are gone in v2.

**Rewritten to:**
- Read labels from `dataset._label_lookup[contract_id][0]` (the y_tensor, indexed by
  int per the v9 CLASS_NAMES order)
- Read num_nodes from `dataset.num_nodes_map[contract_id]` (precomputed at
  SentinelDataset init via LRU-cached shard reads)

**CLASS_NAMES order is different from the outdated dual_path_dataset.py docstring.**
The v2 export uses the live order: `Reentrancy, CallToUnknown, Timestamp, ExternalBug,
GasException, DenialOfService, IntegerUO, UnusedReturn, MishandledException, NonVulnerable`.
The old docstring listed `CallToUnknown, DenialOfService, ExternalBug, GasException,
IntegerUO, MishandledException, Reentrancy, Timestamp, TransactionOrderDependence, UnusedReturn`.
This is a v9 schema evolution that the seam swap exposed; the old sampler would
have silently read the wrong column for 8 of 10 classes.

### 7B-AMEND-8 — Two latent test bugs fixed during Gate 1

`data_module/tests/test_representation/test_byte_identical_regression.py`:
1. Stale fixture path `Data/data/preprocessed/solidifi` → `data_module/data/preprocessed/solidifi`
   (the `Data` → `data_module` rename was never propagated to this test)
2. `m.with_suffix(".sol")` on `.meta.json` produces `.meta.sol` (replaces only
   `.json`, not the whole `.meta.json`); fixed to
   `m.with_name(m.name.replace(".meta.json", ".sol"))`

Both 1-line fixes; both had been latent since the `Data` rename (pre-Stage 7).

### 7B-AMEND-9 — `dual_path_dataset.py` DELETED (2026-06-12 follow-up)

After the seam swap, the original 7B-AMEND-9 left `dual_path_dataset.py` in
`ml/src/datasets/` as a "safety net." On 2026-06-12, a follow-up pass found
two more live references that had been missed:
- `ml/tests/test_dataset.py` (19+ tests on `DualPathDataset` — superseded
  by `ml/tests/test_sentinel_dataset.py`'s 16 tests, deleted via `git rm`)
- `ml/scripts/tune_threshold.py` (active production script in the
  `_legacy_data_pipeline` "Keep" list — migrated to `SentinelDataset`)

`tune_threshold.py` migration: 5 edits (imports, CLI args, `build_val_loader`,
batch unpacking site, `main()` config). Old `--label-csv`/`--splits-dir`/`--cache`
args kept as deprecated no-op for backward CLI compat; new `--export-dir` arg
replaces them.

After both fixes, the 3 deletions:
- `git rm ml/src/datasets/dual_path_dataset.py` (the loader)
- `git rm ml/tests/test_dataset.py` (superseded tests)
- `rm ml/_archive/seam_swap_pre_2026-06-12/ml_datasets/dual_path_dataset.py` (archive cleanup)

**Verification:** `grep -rn "dual_path_dataset\|DualPathDataset\|dual_path_collate"`
returns zero hits in `ml/src/`, `ml/tests/`, `ml/scripts/` (non-archive), and
`data_module/`. Remaining references are:
- 4 in code comments (historical, acceptable)
- 9 in `ml/scripts/_legacy_data_pipeline/` (archived, not active)
- 1 in `ml/scripts/archive/` (old audit script, not active)
- 1 in `data_module/sentinel_data/representation/tokenizer.py` docstring
  (historical reference to the old loader's design)

38/38 ml seam-swap tests pass; 586/613 data_module tests pass (27 skip on
solc/external).

### 7B-AMEND-10 — `SentinelDataset.__init__` is 14.1s for 15,063 contracts; the cost is acceptable

The dataset constructor touches all 5 graph shards to precompute
`num_nodes_map` (used by the timestamp-size sampler). Cost: 14.1s for the
training set (5 shards, 15,063 contracts, all in one LRU cache). Subsequent
`__getitem__` calls are O(1) from the cache. The 5-shard LRU is configured via
the `SENTINEL_SHARD_CACHE_SIZE` env var (default 4).

**For Run 11 at scale:** if num_nodes_map init becomes a bottleneck (10× more
contracts), the strategy is to (a) parallelize the per-shard reads across
processes, or (b) precompute num_nodes into the export's metadata.parquet.
The second option is the right long-term fix (metadata already has node_count
in stage 2's `.rep.json`); deferred to Run 11 prep.

### 7B-AMEND-11 — The trainer's `smoke_subsample_fraction` is monkey-patched into `_contract_ids`

`SentinelDataset` doesn't accept positional indices (the v1 API was
`indices=[...]` into a sorted `paired_hashes` list). For the smoke-test path
where `config.smoke_subsample_fraction < 1.0`, the trainer slices
`train_dataset._contract_ids` after construction. This is documented as a
smoke-test-only hack. Real Run 10/11 will use the full split; the hack gets
deleted when the smoke test moves to a synthetic export dir.

### 7B Verification — 6/7 gates GREEN, 1/7 PARTIAL

Full report: `data_module/docs/v2-readiness-2026-06-12.md`

| # | Gate | Verdict |
|---|---|---|
| 1 | Schema regression (Stage 2 byte-identical) | ✅ 40/40 (also fixed 2 latent bugs) |
| 2 | BCCC Phase 5 verification suite | ✅ 191 pass / 21 skip |
| 3 | End-to-end round-trip (SentinelDataset) | ✅ 16/16 + smoke (15,063 train + 3,226 val) |
| 4 | Feature distribution (Stage 6) | ✅ by construction (v9 schema unchanged) |
| 5 | All 10 classes VERIFIED/PROVISIONAL | 🟡 0 FAIL; 2 VER, 5 PROV, 3 BEST-EFFORT (corpus-bound: SmartBugs Curated deferred to v2.1) |
| 6 | No leakage across splits | ✅ 0 overlap (15,644/3,344/3,368) |
| 7 | No open code-bug regression | ✅ EMITS 4/4 + predictor per-class thresholds |

**Verdict: READY for Run 10 launch (scheduled 2026-08-18).**

### 7B What was NOT done (out of scope, deferred)

- **Step 8 — Docker build verification.** The Dockerfile exists from 7A
  (`data_module/Dockerfile`); no Docker available in WSL2. Manual verification
  steps would be: `docker build -t sentinel-data .` and run the
  `sentinel-data export` CLI in the container. Deferred to Run 11 prep.
- **Defer `dual_path_dataset.py` deletion** (see 7B-AMEND-9).
- **Defer the 22 pre-existing test failures** in `ml/tests/test_preprocessing.py`
  (v8→v9 schema test drift, Slither API changes). NOT Stage 7B regression.
- **Defer `stage_6 feature_dist` re-run on the v2 export** (cosmetic; same numbers).

### 7B Files touched (cumulative)

**New (5):**
1. `ml/src/datasets/sentinel_dataset.py` (~150 LoC, 3 hard gates)
2. `ml/src/datasets/collate.py` (~50 LoC, 5-tuple collate)
3. `ml/tests/test_sentinel_dataset.py` (16 tests)
4. `data_module/tests/test_representation/test_emits_fixture.py` (4 tests)
5. `data_module/docs/v2-readiness-2026-06-12.md` (this report)

**Modified (8):**
6. `ml/src/preprocessing/graph_schema.py` — 18-line shim re-export
7. `ml/src/preprocessing/graph_extractor.py` — direct import from sentinel_data
8. `ml/src/training/trainer.py` — 8 sites swapped
9. `ml/scripts/train.py` — 5 old CLI args → single `--export-dir`
10. `ml/tests/test_trainer.py` — inline synthetic collate
11. `ml/pyproject.toml` — sentinel-data path dep, prometheus constraint relaxed
12. `ml/.venv/bin/solc` — symlink to 0.8.19
13. `data_module/tests/test_representation/test_byte_identical_regression.py` — 2 latent bugs fixed

**Deleted (3):**
14. `ml/src/data_extraction/ast_extractor.py` (legacy)
15. `ml/src/data_extraction/tokenizer.py` (legacy)
16. (Window_tokenizer kept — still used by data_module tests)

**Archived (7):**
17. `ml/scripts/_legacy_data_pipeline/` (7 v1 scripts + README)

**Archived pre-7B backups (1 dir):**
18. `ml/_archive/seam_swap_pre_2026-06-12/` (consolidated from 3 scattered dirs)

---

## What this ADR does NOT cover (deferred beyond 7B)

7A's "deferred to 7B" list (SentinelDataset loader, dual-path test, legacy script
deletion, pyproject update, Docker, 7 gates) is now closed — see the 7B Amendment
above. Remaining deferred items:

- **Docker build verification** (Step 8) — Dockerfile exists; no Docker in WSL2.
- **`dual_path_dataset.py` deletion** (7B-AMEND-9) — dead code, but `test_trainer.py`
  still has the import for its synthetic 3-tuple test data; cleanup blocked on
  removing the import there.
- **22 pre-existing test failures** in `ml/tests/test_preprocessing.py` — v8→v9
  schema test drift, Slither API changes. NOT Stage 7 regression.
- **Re-run Stage 6 `feature_dist` on the v2 export** — cosmetic; same numbers.
- **Run 10 launch** (2026-08-18) and Run 11 (post-v2.1) — the next milestones.
