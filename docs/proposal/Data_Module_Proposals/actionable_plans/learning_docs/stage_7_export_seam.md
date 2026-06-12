# Stage 7 — Export + Seam Swap

**Date:** 2026-06-12
**Status:** ✅ 7A COMPLETE (export module + CLI + predictor fix). 7B deferred to next session.
**Reading time:** 25-30 minutes.
**Goal:** After this doc you can explain the 7A/7B split, the 5 implementation choices (IC-1 through IC-5), the 3-file hash scope for the artifact, why `manifest.json` is excluded, and what 7B needs to do before Run 11 can launch.

---

## 1️⃣ The Problem

### What Stage 7 has to deliver

Stages 1–6 produced verified, analyzed data — but none of it in a form the ML module can consume. Stage 7 produces the **sharded export** that `sentinel-ml`'s `SentinelDataset` reads, and performs the **seam swap** — replacing the old import paths with the new ones and deleting the legacy data scripts.

Stage 7 is split into:
- **7A (done):** Export module (4 writers + chunker + consumer API + CLI). Predictor tier fix (F8/F10). No seam swap.
- **7B (deferred):** `SentinelDataset` loader, dual-path test, legacy script deletion, Docker build, 7 v2-readiness gates.

### The 3 open bugs in scope

| Bug | Where | Status |
|---|---|---|
| **Predictor tier threshold** | `predictor.py:698,751` — was using scalar 0.55 for all classes | ✅ Fixed in 7A: `self.thresholds[cls_idx].item()` per class |
| **EMITS edge** | `graph_extractor.py` (Slither event-emission edge) | 7B (needs investigation: data-side vs code-side) |
| **CALL_ENTRY cross-function** | `graph_extractor.py:1001` (self-loop only) | Deferred post-Run-11 |

---

## 2️⃣ The Solution

### Sharded export (D-7.1)

The export artifact at `data/exports/<dataset-version>/`:
```
manifest.json            ← written LAST (contains artifact_hash)
labels.parquet           ← all 22,356 labeled contracts (including no-rep ones)
metadata.parquet         ← same coverage, enriched from sidecars
graphs/
  graphs-00000.pt        ← PyG Batch, up to 5,000 contracts
  graphs-00001.pt
  _shard_index.json      ← {sha256: {shard: int, pos_in_shard: int}}
tokens/
  tokens-00000.pt        ← torch.Tensor [N, 4, 512]
  tokens-00001.pt
  _shard_index.json
```

21,523 contracts have representations (21,247 dive + 276 solidifi). The parquet tables cover all 22,356 split rows; the `.pt` shards only cover the 21,523 with reps. Contracts missing reps get null node_count/edge_count in metadata.parquet.

### The 4 writers

| Writer | Input | Output |
|---|---|---|
| `label_writer.write_labels_parquet()` | splits JSONL | `labels.parquet` (14 cols) |
| `metadata_writer.write_metadata_parquet()` | splits JSONL + .rep.json + .meta.json + .sol | `metadata.parquet` (14 cols) |
| `graph_writer.write_graphs_shards()` | .pt files via splits JSONL order | `graphs/graphs-{n:05d}.pt` |
| `token_writer.write_tokens_shards()` | .tokens.pt files via splits JSONL order | `tokens/tokens-{n:05d}.pt` |

All 4 walk the **split JSONL** for ordering — not the representations directory — so shard row N aligns across all 4 file types (D-7.2).

### The artifact hash (Fix A — circular hash avoidance)

`manifest.json` contains `artifact_hash` — it cannot be part of the file set whose hash it stores. So:

1. Write all 4 data file types.
2. Compute `SHA-256` over those files (sorted by relative path, not including `manifest.json`).
3. Write `manifest.json` LAST with the hash field populated.

`_hash_export_data(export_dir)` is the single function that does this computation, shared by `chunk_export` (to write) and `SentinelDatasetExport.verify_artifact_hash()` (to verify). Modifying `manifest.json` does not change the hash; modifying any data file does.

### The consumer API

```python
from sentinel_data.export import SentinelDatasetExport

export = SentinelDatasetExport("data/exports/sentinel-v2-gold-2026-08/")
export.verify_artifact_hash()           # True on a clean export
export.get_split_contract_ids("train")  # [sha256, ...]
export.manifest.n_contracts             # 22356
export.manifest.n_shards               # 5
```

The `SentinelDataset` (7B) wraps this class and adds `__len__`/`__getitem__`.

### The CLI

```bash
sentinel-data export --dry-run
# → prints: sources, shard_size, output_dir, skipped_sources

sentinel-data export --dataset-version sentinel-v2-gold-2026-08
# → writes to data/exports/sentinel-v2-gold-2026-08/

sentinel-data export --dataset-version sentinel-v2-gold-2026-08 --shard-size 2000
```

---

## 3️⃣ Context

### Why loc=0 in the split JSONL is not the real LoC (Fix #3)

The splitter (`Stage 5`) writes `loc: 0` as a default — it never computed LoC because computing it requires reading the `.sol` file, which the splitter doesn't do. The `metadata_writer` re-computes `loc` and `n_functions` from the `.sol` source using the same `_loc()` and `_function_count()` regex helpers from `analysis/feature_dist.py`.

### Why confidence_tier is None for NonVulnerable (Fix #2)

The Stage 5 splitter assigns `tier="T0"` to all contracts by default, including NonVulnerable ones. But a NonVulnerable contract has no vulnerability label to verify, so the tier is meaningless for it. The export overrides `tier → None` (pyarrow null) when `n_pos == 0`. The ML module can then filter on `confidence_tier.is_null()` to skip tier-based loss weighting for negatives.

### The predictor tier fix (F8/F10)

`predictor.py:698` was:
```python
conf_thr = self.tier_confirmed_threshold  # scalar 0.55 for all classes
```

Now:
```python
conf_thr = self.thresholds[cls_idx].item()  # per-class from calibration JSON
```

A class tuned to `0.90` (e.g. CallToUnknown) no longer triggers CONFIRMED at `prob=0.56`. A class tuned to `0.35` (e.g. Reentrancy) now correctly triggers at `prob=0.36`.

### The 7A/7B split rationale

7A produces a v2 export that `dual_path_dataset.py` cannot load (different format). Deleting `dual_path_dataset.py` in 7A without first validating the replacement `SentinelDataset` (7B's byte-identical test) would break the active training pipeline with no fallback. The split preserves the ability to re-run Run 9 from its checkpoint if needed.

---

## 4️⃣ Verification

### Test suite (27 tests, all pass)

| Test file | What it covers |
|---|---|
| `test_label_writer.py` (5) | Column names, row count, confidence_tier=None for non-vuln, class values, missing split raises |
| `test_metadata_writer.py` (4) | Column names, loc computed from .sol, null node_count on missing rep, confidence_tier |
| `test_graph_token_writer.py` (5) | Shard count, shard index values, skips missing .pt, token shape, graph/token same order |
| `test_chunker.py` (5) | All 4 file types produced, manifest written last (mtime), manifest tamper doesn't change hash, data tamper changes hash, split counts |
| `test_export.py` (8) | Loads, verify_hash=True, verify_hash=False on tamper, manifest tamper doesn't break verify, get_split_ids, missing manifest raises, repr, public import (Fix C) |

Full suite: `python -m pytest data_module/tests/ -q` → 531 passed, 51 skipped.

### Stage 7A definition of done

- [x] `format_schema/v1.yaml` is the authoritative spec
- [x] All 4 writers produce valid output on synthetic fixtures
- [x] `chunk_export` runs end-to-end on synthetic data
- [x] `SentinelDatasetExport` roundtrips; `verify_artifact_hash` works
- [x] `sentinel-data export --dry-run` prints the plan
- [x] All 27 export tests pass; full suite still green (531 pass)
- [x] Predictor tier fix landed (separate commit, 6 regression tests pass)
- [x] ADR-0008 written
- [x] This learning doc updated; LEARNING_CHECKLIST.md updated

---

## 5️⃣ What's deferred to 7B

| Item | Why deferred |
|---|---|
| `SentinelDataset` loader (`~150 lines`) | Needs the export to exist first; build with test harness |
| Dual-path test (old vs new loader, byte-identical for 100 contracts) | 7B's safety net for the seam swap |
| Delete `dual_path_dataset.py` | Need dual-path test to pass first |
| Delete legacy ML data scripts (8 scripts) | Depends on dual-path test + seam swap |
| Update `ml/pyproject.toml` | Add `sentinel-data` dep, remove legacy solc deps |
| Docker build verification | Requires all ML-side changes to land |
| 7 v2-readiness gates check | Final before Run 11 |
| EMITS edge investigation | Data-side or code-side — needs triage |

---

## 6️⃣ Got it? Check yourself

| # | Question | Answer |
|---|---|---|
| 7.1 | What are the 4 file types in the export? | graphs/*.pt (PyG Batch), tokens/*.pt (Tensor [N,4,512]), labels.parquet, metadata.parquet |
| 7.2 | Why is manifest.json excluded from the artifact_hash? | It contains the hash — including it would be circular. Written LAST. |
| 7.3 | Why does the metadata_writer re-compute loc? | The split JSONL has loc=0 (splitter never computed it). Real loc comes from .sol source. |
| 7.4 | Why is confidence_tier None for NonVulnerable contracts? | The splitter's tier="T0" default is meaningless for negatives; null lets the ML module skip tier weighting. |
| 7.5 | What was the predictor tier bug and how was it fixed? | Used scalar 0.55 for all classes; now uses self.thresholds[cls_idx].item() per class. |
| 7.6 | Why is the seam swap in 7B not 7A? | 7A's format can't be loaded by dual_path_dataset.py; deleting it without the new SentinelDataset would break training with no fallback. |
| 7.7 | What does sentinel-data export --dry-run print? | Sources, shard_size, output_dir, skipped sources (those with no preprocessed dir). |
