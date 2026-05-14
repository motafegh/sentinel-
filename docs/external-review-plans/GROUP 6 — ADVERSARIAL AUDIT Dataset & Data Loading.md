Let me find and read the Group 6 source files — Dataset & Data Loading.
Now let me read the core Group 6 files in parallel:
Now let me read the remaining relevant files:
Let me also check the trainer to understand how the dataset is wired in, and the remaining scripts:
Now I have a thorough understanding of all the Group 6 files. Let me write the full adversarial audit.
Now let me perform the deep cross-referencing analysis. Let me check a few critical cross-file interactions:
Now I have a thorough understanding of all the Group 6 files and their cross-references. Let me deliver the full adversarial audit.

---

# 🔴 GROUP 6 — ADVERSARIAL AUDIT: Dataset & Data Loading

**Files in scope:** `dual_path_dataset.py`, `tokenizer.py`, `create_splits.py`, `verify_splits.py`, `dedup_multilabel_index.py`, `build_multilabel_index.py`, `create_label_index.py`, `create_cache.py`, `validate_graph_dataset.py`, `test_dataset.py`

**Cross-references:** `trainer.py` (dataset wiring), `hash_utils.py` (Group 1), `graph_extractor.py`/`ast_extractor.py` (Group 2), `graph_schema.py` (Group 1)

---

## 6.1 [CRITICAL] — `build_multilabel_index.py` uses `weights_only=False` — arbitrary code execution vector

**File:** `build_multilabel_index.py:162`

```python
graph = torch.load(pt_file, weights_only=False)
```

This iterates over **every** `.pt` file in `ml/data/graphs/` and unpickles it with zero safety. Compare to `dual_path_dataset.py` which was patched to `weights_only=True` with safe globals registered (audit fix #3). The `build_multilabel_index.py` script even registers safe globals at line 50:

```python
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
```

but **never uses them** — still passes `weights_only=False`. An attacker who can write a single malicious `.pt` file into the graphs directory gets remote code execution when this script runs. Same vulnerability exists in `create_label_index.py:30`, `validate_graph_dataset.py:70`, `compute_drift_baseline.py:56`, and `cache.py:93-94`.

**Fix:** Switch all to `weights_only=True`, add `GlobalStorage` to the allowlist in `build_multilabel_index.py` (it's missing — the dataset file registers it but this script doesn't), and verify the full PyG class set is registered.

---

## 6.2 [CRITICAL] — Deduped CSV and splits are never wired into the training pipeline — dedup is a dead letter

**Files:** `dedup_multilabel_index.py`, `trainer.py`

The dedup script was written to fix the 34.9% cross-split leakage (the exact problem identified in prior audit groups). It produces:
- `ml/data/processed/multilabel_index_deduped.csv` (44,420 rows vs 68,523)
- `ml/data/splits/deduped/train_indices.npy`, `val_indices.npy`, `test_indices.npy`

But `trainer.py` hardcodes:
```python
splits_dir: str = "ml/data/splits"           # NOT splits/deduped/
label_csv:  str = "ml/data/processed/multilabel_index.csv"  # NOT _deduped.csv
```

There is **no CLI flag, no config option, and no auto-detection** that switches to the deduped versions. A retraining run will silently use the leaky 68K-row CSV and the old leaky splits. The entire dedup effort — the most critical data-integrity fix in the project — is inert.

**Fix:** Add `--use-deduped` flag to `train.py` / `TrainConfig` that maps to the deduped CSV and `splits/deduped/` directory. Or better: make the deduped CSV the default and the old one a legacy fallback.

---

## 6.3 [HIGH] — `tokenizer.py` saves FULL dict (with metadata) but `dual_path_dataset.py` assumes tokens are `{input_ids, attention_mask}` only

**File:** `tokenizer.py:176-186` saves:
```python
result = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'contract_hash': contract_hash,       # str
    'contract_path': str(contract_path),   # str
    'num_tokens': num_real_tokens,         # int
    'truncated': truncated,                # bool
    'tokenizer_name': TOKENIZER_MODEL,     # str
    'max_length': MAX_LENGTH,              # int
    'feature_schema_version': FEATURE_SCHEMA_VERSION,  # int
}
torch.save(token_data, output_path)  # saves the ENTIRE dict
```

But `dual_path_dataset.py:288-295` validates:
```python
if tokens["input_ids"].shape != torch.Size([512]):
    raise ValueError(...)
if tokens["attention_mask"].shape != torch.Size([512]):
    raise ValueError(...)
```

And `dual_path_collate_fn:330-333` only extracts:
```python
batched_tokens = {
    "input_ids":      torch.stack([t["input_ids"]      for t in tokens]),
    "attention_mask": torch.stack([t["attention_mask"] for t in tokens]),
}
```

This works by accident today — extra dict keys are silently ignored. But it means:
1. Every `.pt` token file carries ~7 useless keys (strings, ints, bools), inflating the pickle cache from ~0.56 GB to ~0.7 GB
2. The `feature_schema_version` inside token files is **never read or validated** at training time — if you re-extract graphs with a new schema version but forget to re-tokenize, you'll train on mismatched data with no warning
3. If a future developer adds a key named `labels` or `y` to the token dict, it will silently shadow the graph label

**Fix:** `tokenizer.py` should save only `{input_ids, attention_mask}`. Schema version should be stored in a separate manifest file that `DualPathDataset.__init__` validates.

---

## 6.4 [HIGH] — `create_label_index.py` reads `graph.y` which is hardcoded to 0 — produces a 100%-safe label index

**File:** `create_label_index.py:36-37`
```python
if hasattr(graph_data, 'y'):
    label = int(graph_data.y.item())
```

From Group 2 audit (Finding 2.1 — CRITICAL), `ast_extractor.py` hardcodes `graph.y = torch.tensor(0)` for ALL contracts. This means `create_label_index.py` produces a CSV where 100% of rows are labelled "safe" (label=0). Any stratification based on this CSV is meaningless — it's equivalent to random splitting.

The `create_splits.py` docstring acknowledges this:
```
Note: label_index_path is IGNORED — binary labels are derived from
multilabel_index.csv (sum of class columns > 0) because ast_extractor.py
hardcodes graph.y=0 for all contracts
```

But `create_label_index.py` still exists and can still be run. If someone doesn't read the docstring carefully, they'll use the output `label_index.csv` for stratification, producing completely unstratified splits. The script should either be deleted, renamed to `_deprecated_`, or patched to derive labels from `multilabel_index.csv` like `create_splits.py` does.

**Fix:** Add a deprecation warning and a guard: if >99% of labels are 0, abort with a message explaining the `graph.y=0` hardcoding and pointing to `build_multilabel_index.py`.

---

## 6.5 [HIGH] — `dedup_multilabel_index.py` orphans kept with their own md5 as group key — may not actually dedup them

**File:** `dedup_multilabel_index.py:110-115`
```python
orphan_mask = df["_content_hash"].isna()
df.loc[orphan_mask, "_content_hash"] = df.loc[orphan_mask, "md5"]
```

Orphan rows (whose `.sol` source file wasn't found in any `SOURCE_DIRS`) get their path-MD5 used as the content hash group key. This means each orphan forms a 1-element group — no dedup happens for them. If two orphans are actually the same contract under different paths, they'll remain as duplicate rows in the deduped CSV.

The script logs: `"Orphan rows (no .sol found): {n_orphans:,} — kept as-is"`, but there's no indication of how many orphans exist or whether this is a problem. If non-BCCC data sources (SolidiFI, SmartBugs) have their own duplication, the dedup won't catch it because the source `.sol` files might not be found by the directory scanner (different directory structures, compressed archives, etc.).

**Fix:** For orphans that share the same `contract_path` stem (SHA256), attempt a secondary dedup. Or at minimum, log the number of orphans prominently and add a `--strict` mode that refuses to proceed if orphans exceed a threshold.

---

## 6.6 [HIGH] — `create_splits.py` and `dedup_multilabel_index.py` use completely different stratification methods — splits are not comparable

**File:** `create_splits.py:121-136` uses `sklearn.model_selection.train_test_split` with `stratify=all_labels` (binary safe/vulnerable).

**File:** `dedup_multilabel_index.py:150-163` uses manual bucket-based stratification by label count (0, 1, 2, 3+ positive classes).

These produce **different split distributions**. The deduped splits will have different class proportions in val/test than the original splits. Any experiment comparing metrics across the two split strategies is confounded — you can't tell whether improvement came from dedup or from the different stratification method.

Additionally, `dedup_multilabel_index.py` uses `np.random.default_rng(SEED)` while `create_splits.py` uses `sklearn` which uses `np.random.RandomState(SEED)` — different PRNG algorithms, so even with the same seed, the splits would differ.

**Fix:** Both scripts should use the same stratification method and PRNG. Ideally, `dedup_multilabel_index.py` should call `create_splits.create_splits()` after writing the deduped CSV instead of reimplementing its own split logic.

---

## 6.7 [HIGH] — `pickle.load(cache_path)` with no `weights_only` — same arbitrary code execution risk as .pt files

**File:** `dual_path_dataset.py:192-193`
```python
with open(cache_path, "rb") as f:
    self.cached_data = pickle.load(f)
```

The RAM cache is a pickle file created by `create_cache.py`. Pickle deserialization is equivalent to `torch.load(weights_only=False)` — arbitrary code execution. While the cache is presumably created by a trusted script, the same attack surface exists: if an attacker can write to `ml/data/cached_dataset.pkl`, they get code execution on the training node.

The `.pt` loading path was patched to `weights_only=True`, but the pickle cache was not, creating an inconsistency: the "faster" path is also the "less safe" path.

**Fix:** Use `torch.load(cache_path, weights_only=True)` for the cache file too (it contains only tensors and PyG Data objects). Or add a SHA256 checksum file alongside the cache that's verified before loading.

---

## 6.8 [MEDIUM] — `DualPathDataset` cache integrity check is spot-check only — stale cache passes if first hash matches

**File:** `dual_path_dataset.py:202-219`

The cache validation checks only `self.paired_hashes[0]` — a single spot check. If you:
1. Build a cache with 68K samples
2. Re-extract graphs with a new schema version (producing different tensors)
3. Some early samples (alphabetically first hashes) happen to be unchanged

The spot check passes, and the remaining 67,999 samples are stale — wrong features, wrong dimensions, wrong labels. Training will silently use garbage data.

The comment says "Audit #11: integrity check on the loaded cache" but a single-spot check is theatrical, not effective. A re-extraction that changes a subset of files will pass validation while delivering stale data for the rest.

**Fix:** After loading the cache, verify a random sample (e.g., 5% of entries, minimum 100) against on-disk files. Or store a manifest (hash → SHA256 of the cached pair) alongside the pickle and validate the full manifest.

---

## 6.9 [MEDIUM] — Tokenizer saves `feature_schema_version` but nobody reads it — silent schema drift between graphs and tokens

**File:** `tokenizer.py:185` saves `'feature_schema_version': FEATURE_SCHEMA_VERSION` into each `.pt` token file.

But at training time, `DualPathDataset` never reads or validates this field. If you:
1. Extract graphs with schema v4 (8-dim features)
2. Extract tokens with schema v4
3. Re-extract graphs with schema v5 (13-dim features, 7 edge types)
4. Forget to re-tokenize (or tokenizer is run against different source)

The schema version mismatch is invisible. The model will receive 13-dim graphs with token files that were created against the old schema. While the token files themselves don't depend on the graph schema, the **metadata** does — `num_real_tokens` and `truncated` counts are logged against a specific schema version.

More critically: `graph_schema.FEATURE_SCHEMA_VERSION` is imported in `tokenizer.py` but the graph extraction pipeline (`ast_extractor.py`) also uses `FEATURE_SCHEMA_VERSION` for cache invalidation. If the two get out of sync, you'll have graphs and tokens extracted at different schema versions with no runtime warning.

**Fix:** `DualPathDataset.__init__` should read the `feature_schema_version` from one token file and assert it matches the current `graph_schema.FEATURE_SCHEMA_VERSION`.

---

## 6.10 [MEDIUM] — `tokenizer.py` silently swallows ALL exceptions — corrupt/empty/tokenizer-crashing inputs return `None`

**File:** `tokenizer.py:190-193`
```python
except Exception as e:
    # Something went wrong - skip this contract
    # Main process will log it
    return None
```

This is the same "broad exception catch" anti-pattern found in Group 2 (`ast_extractor.py`). ANY failure — OOM, disk full, corrupted file, tokenizer bug, Python import error — silently returns `None` and the contract is skipped. The main process logs nothing about the specific error.

Even worse: the comment says "Main process will log it" but the main process **doesn't** log it — it only increments `stats['failed']` (line 353). The specific exception message is lost.

**Fix:** Return `(contract_path, str(e))` on failure instead of `None`, and log the first N error messages in the main loop.

---

## 6.11 [MEDIUM] — `tokenizer.py` checkpoint index-to-hash mapping is unreliable under `imap` — failed contracts get wrong hash

**File:** `tokenizer.py:348-352`
```python
try:
    failed_hash = get_contract_hash(contract_paths[i])
    failed_hashes.append(failed_hash)
except:
    pass
```

`pool.imap` preserves ordering, so `result` at position `i` corresponds to `contract_paths[i]`. This is correct. However, the `except: pass` is a bare except — even `KeyboardInterrupt` and `SystemExit` are swallowed. And if `get_contract_hash` fails (e.g., file deleted between listing and processing), the failed contract is not tracked at all, so the checkpoint can't skip it on resume — it will be retried forever.

**Fix:** Use `except Exception` instead of bare `except`, and track failed contracts by index rather than re-computing the hash.

---

## 6.12 [MEDIUM] — `create_splits.py --freeze-val-test` doesn't verify that existing val/test indices are valid for the CURRENT CSV

**File:** `create_splits.py:76-78`
```python
existing_val  = np.load(val_file)
existing_test = np.load(test_file)
existing_train = np.load(train_file) if train_file.exists() else np.array([], dtype=np.int64)
```

If `multilabel_index.csv` is rebuilt (e.g., after dedup), the row count changes and the existing `.npy` indices may be out of range. The code doesn't check `max(existing_val) < len(df)` or `max(existing_test) < len(df)`. An out-of-range index will cause an `IndexError` much later when `DualPathDataset` tries to use it, with a confusing error message.

Also: `existing_train` is loaded but only used to compute `original_n` for logging. It's not validated for range either.

**Fix:** Add `assert max(existing_val) < len(df)` and same for test after loading.

---

## 6.13 [MEDIUM] — `dedup_multilabel_index.py` uses `int()` rounding for train/val/test split — coverage gap

**File:** `dedup_multilabel_index.py:159-163`
```python
n_train = int(n_b * TRAIN_PCT)
n_val   = int(n_b * VAL_PCT)
train_idx.append(bucket_idx[:n_train])
val_idx.append(bucket_idx[n_train : n_train + n_val])
test_idx.append(bucket_idx[n_train + n_val:])
```

`int()` truncates, so `n_train + n_val + n_test = int(n_b*0.7) + int(n_b*0.15) + remainder`. The remainder (assigned to test) may be 1-2 samples larger or smaller than the ideal 15%. Over 4 buckets, this can accumulate. `create_splits.py` uses `sklearn` which handles this correctly.

The `rebuild_splits` function does assert no overlap and full coverage (lines 179-181), so correctness is maintained — but the split ratios can drift from the intended 70/15/15, especially for small buckets (e.g., bucket=3 with only 50 samples).

**Fix:** Use `sklearn.model_selection.train_test_split` like `create_splits.py`, or compute `n_test = n_b - n_train - n_val` to guarantee exact coverage.

---

## 6.14 [MEDIUM] — `verify_splits.py` Check 1 is opt-out via `--no-content-check` — leakage check should be mandatory

**File:** `verify_splits.py:294-297`

The content-hash leakage check (the most important check, which found the 34.9% cross-split leakage) can be skipped with `--no-content-check`. The comment says it's "slow" — it scans all source directories and hashes every `.sol` file. But if the leakage check is optional, CI pipelines will skip it for speed, and regressions will go undetected.

**Fix:** Make `--no-content-check` require an explicit `--i-understand-this-skips-leakage-check` confirmation, or at minimum, emit a WARNING-level log line that's visible in CI output.

---

## 6.15 [MEDIUM] — `build_multilabel_index.py` assumes SHA256 from `contract_path.stem` but some graphs may not have BCCC paths

**File:** `build_multilabel_index.py:171-172`
```python
contract_path = getattr(graph, "contract_path", "")
sha256 = Path(contract_path).stem if contract_path else ""
```

For non-BCCC sources (SmartBugs, SolidiFI, augmented), the `contract_path` embedded in the graph may not follow the BCCC naming convention where the stem is a SHA256 hash. If the stem is something like `0x1234...` or `contract_name`, the SHA256 lookup will fail, and the contract gets `label_vector = [0]*10` (treated as safe).

The script logs a WARNING for these cases (line 181-184), but the training pipeline silently trains on them as "safe" samples. If a significant fraction of the dataset comes from non-BCCC sources with real vulnerabilities, those vulnerabilities are erased from the label vector.

**Fix:** For non-BCCC sources, integrate their label sources (SmartBugs labels, SolidiFI vulnerability mappings) into the SHA256 lookup. Or at minimum, count and prominently report how many non-BCCC rows are labelled all-zeros.

---

## 6.16 [MEDIUM] — `DualPathDataset` loads `label_csv` into RAM as `dict[str, Tensor]` — no validation that class count matches model output

**File:** `dual_path_dataset.py:131-138`
```python
class_cols = [c for c in df.columns if c != "md5_stem"]
label_matrix = torch.tensor(df[class_cols].values.astype("float32"), dtype=torch.float32)
self._label_map = {stem: label_matrix[i] for i, stem in enumerate(stems)}
```

The number of classes is determined dynamically from the CSV column count. If someone provides the deduped CSV (10 classes) vs the original (which included `WeakAccessMod` = 11 columns before exclusion), the label tensor dimension will be 10 vs 11, and the collated batch shape will be `[B, 10]` vs `[B, 11]`.

The model's `num_classes` comes from `TrainConfig.num_classes` (default 10). There's no assertion that `len(class_cols) == num_classes`. A mismatch will cause a shape error deep inside the loss function, with a confusing `RuntimeError: The size of tensor a must match the size of tensor b` message.

**Fix:** In `DualPathDataset.__init__`, assert `len(class_cols) == expected_num_classes` and pass `expected_num_classes` as a parameter.

---

## 6.17 [MEDIUM] — `create_cache.py` uses `ThreadPoolExecutor` for I/O-bound loading but `torch.load` may release the GIL inconsistently

**File:** `create_cache.py:103`
```python
with ThreadPoolExecutor(max_workers=8) as pool:
```

`torch.load` with `weights_only=True` may or may not release the GIL depending on the PyTorch version and the underlying deserialization path. If the GIL is held, the thread pool provides zero parallelism — it degrades to sequential loading with overhead. For CPU-bound unpickling, `ProcessPoolExecutor` would be more reliable.

Additionally, the error handling in `create_cache.py:114` catches exceptions from `fut.result()` but doesn't track which stems failed — the cache will simply lack those entries, and `DualPathDataset` will crash on access.

**Fix:** Track failed stems in the cache building function and log them. Consider `ProcessPoolExecutor` if profiling shows the thread pool isn't providing speedup.

---

## 6.18 [LOW] — `tokenizer.py` suppresses ALL warnings globally — including deprecation and security warnings

**File:** `tokenizer.py:34`
```python
warnings.filterwarnings("ignore")  # Suppress HuggingFace warnings
```

This suppresses ALL warnings, not just HuggingFace's. Deprecation warnings (e.g., `torch.load` without `weights_only`), security warnings, and future-breaking-change notices are all silenced. The comment says it's for HuggingFace warnings, but the scope is global and unconditional.

**Fix:** Use `warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")` to target only the intended source.

---

## 6.19 [LOW] — `test_dataset.py` doesn't test cache loading, multi-label mode with actual labels, or edge_attr squeeze fix

**File:** `test_dataset.py`

The test suite covers:
- Basic construction, length, getitem
- Split indices (subset, empty, out-of-range)
- Unpaired files
- Binary and multi-label collation

Missing test coverage:
1. **Cache loading**: No test for `cache_path` parameter — the spot-check validation, stale cache detection, and malformed cache detection paths are untested
2. **Multi-label with real labels**: The multi-label test creates an all-zeros CSV — no test for actual positive labels
3. **edge_attr squeeze fix** (Fix #1 in `dual_path_dataset.py:302-304`): No test for the `[E, 1] → [E]` squeeze path
4. **Hash not in label_csv**: No test for the `KeyError` when `hash_id not in self._label_map`
5. **Missing `graph.y`**: No test for the `KeyError` when `graph.y` is absent in binary mode

**Fix:** Add tests for all five uncovered paths.

---

## 6.20 [LOW] — `validate_graph_dataset.py` uses `weights_only=False` and also replicates the `*12` magic number

**File:** `validate_graph_dataset.py:70, 135`
```python
data = torch.load(path, map_location="cpu", weights_only=False)
...
type_ids_raw = (data.x[:, 0] * 12).round().int()
```

The `weights_only=False` issue is tracked in Finding 6.1. The `* 12` magic number is the same one identified in Group 1 (Finding 1.2) and Group 4 (Finding 4.6) — it's disconnected from `graph_schema.NUM_NODE_TYPES` and will silently break if the schema ever changes.

**Fix:** Import and use `NUM_NODE_TYPES` from `graph_schema.py` instead of hardcoding 12.

---

## 6.21 [LOW] — `build_multilabel_index.py` docstring says output has 11 columns (including WeakAccessMod) but code excludes it to 10

**File:** `build_multilabel_index.py:21-22`
```
  Columns: md5_stem, CallToUnknown, DenialOfService, ExternalBug, GasException,
           IntegerUO, MishandledException, Reentrancy, Timestamp,
           TransactionOrderDependence, UnusedReturn, WeakAccessMod
```

But `CLASS_NAMES` (line 67-78) has 10 entries — `WeakAccessMod` is excluded with a detailed comment. The docstring was never updated. Anyone reading the docstring will expect an 11-column output and may write code that references a non-existent `WeakAccessMod` column.

**Fix:** Remove `WeakAccessMod` from the docstring and update the row count (68,555 may also be stale).

---

## 6.22 [LOW] — `dedup_multilabel_index.py` rebuild_splits uses `np.random.default_rng(42)` but `create_splits.py` uses `random_state=42` in sklearn — different PRNGs, different splits

**File:** `dedup_multilabel_index.py:145`, `create_splits.py:125-126`

Even if both scripts used the same stratification algorithm, the different PRNG implementations mean they produce different split assignments. This makes it impossible to do an apples-to-apples comparison between deduped and non-deduped training runs.

**Fix:** Use the same splitting library and PRNG. Import and call `create_splits()` from `create_splits.py` after writing the deduped CSV.

---

## Summary Table

| # | Severity | File | Finding |
|---|----------|------|---------|
| 6.1 | **CRITICAL** | build_multilabel_index.py (+ 5 other scripts) | `weights_only=False` — arbitrary code execution via malicious .pt |
| 6.2 | **CRITICAL** | dedup_multilabel_index.py + trainer.py | Deduped CSV/splits never wired into training — dedup is dead code |
| 6.3 | HIGH | tokenizer.py + dual_path_dataset.py | Token .pt saves 7 extra keys; `feature_schema_version` never validated |
| 6.4 | HIGH | create_label_index.py | Reads `graph.y=0` (hardcoded) → 100% "safe" label index |
| 6.5 | HIGH | dedup_multilabel_index.py | Orphans kept as 1-element groups — dedup doesn't cover them |
| 6.6 | HIGH | create_splits.py + dedup_multilabel_index.py | Different stratification + different PRNG → incomparable splits |
| 6.7 | HIGH | dual_path_dataset.py | `pickle.load(cache)` — no `weights_only`, code execution risk |
| 6.8 | MEDIUM | dual_path_dataset.py | Cache spot-check is single-hash — stale cache passes if first entry matches |
| 6.9 | MEDIUM | tokenizer.py + dual_path_dataset.py | `feature_schema_version` saved but never validated at training time |
| 6.10 | MEDIUM | tokenizer.py | Broad `except Exception` swallows all errors, no error logging |
| 6.11 | MEDIUM | tokenizer.py | Bare `except: pass` in checkpoint error tracking — even KeyboardInterrupt swallowed |
| 6.12 | MEDIUM | create_splits.py | `--freeze-val-test` doesn't validate index range against current CSV |
| 6.13 | MEDIUM | dedup_multilabel_index.py | `int()` truncation in split ratios — small buckets drift from 70/15/15 |
| 6.14 | MEDIUM | verify_splits.py | Leakage check is opt-out — CI will skip it, regressions undetected |
| 6.15 | MEDIUM | build_multilabel_index.py | Non-BCCC sources get all-zero labels — real vulnerabilities erased |
| 6.16 | MEDIUM | dual_path_dataset.py | No validation that CSV class count matches model `num_classes` |
| 6.17 | MEDIUM | create_cache.py | ThreadPoolExecutor + torch.load GIL uncertainty; failed stems not tracked |
| 6.18 | LOW | tokenizer.py | `warnings.filterwarnings("ignore")` suppresses ALL warnings globally |
| 6.19 | LOW | test_dataset.py | No tests for cache, real labels, edge_attr squeeze, hash-not-found, missing-y |
| 6.20 | LOW | validate_graph_dataset.py | `weights_only=False` + `*12` magic number disconnected from schema |
| 6.21 | LOW | build_multilabel_index.py | Docstring says 11 columns including WeakAccessMod; code outputs 10 |
| 6.22 | LOW | dedup + create_splits | Different PRNGs → different splits even with same seed |

**Critical/HIGH cluster analysis:** Findings 6.1, 6.2, and 6.7 form a devastating triad:
- **6.2** means the 34.9% cross-split leakage fix (dedup) is completely inert in production — the trainer always uses the leaky CSV
- **6.1 + 6.7** mean the entire data loading path (both disk `.pt` and pickle cache) has arbitrary code execution vulnerabilities that were partially fixed (dataset `.pt` loading) but left open in scripts and the cache path
- Together: you're training on leaky data AND the "fix" is dead code AND the data loading surface is still exploitable

**Connection to prior groups:**
- 6.4 connects to Group 2 Finding 2.1 (CRITICAL: `_add_node` type-id roundtrip bug + `graph.y=0` hardcoding)
- 6.3 and 6.9 connect to Group 1 Finding 1.2 (`FEATURE_SCHEMA_VERSION` not bumped / not validated)
- 6.20's `*12` magic number connects to Group 1 Finding 1.2 and Group 4 Finding 4.6
- 6.2 is the pipeline consequence of the 34.9% leakage identified across Groups 1-2
Group 6 audit complete. Ready for Group 7 (Inference & Production API) when you are. The two CRITICAL findings are the standouts:

1. **6.2 — Dedup is dead code**: The entire dedup pipeline (`dedup_multilabel_index.py`) was built to fix the 34.9% cross-split leakage, but `trainer.py` hardcodes the *leaky* CSV and old splits. No config flag switches to the deduped versions. The fix doesn't actually fix anything in production.

2. **6.1 — `weights_only=False` in 6+ scripts**: While `dual_path_dataset.py` was patched to `weights_only=True`, the surrounding scripts that iterate over the same `.pt` files (`build_multilabel_index.py`, `create_label_index.py`, `validate_graph_dataset.py`, `compute_drift_baseline.py`, `cache.py`) still use `weights_only=False`, leaving arbitrary code execution vectors open across the data pipeline.