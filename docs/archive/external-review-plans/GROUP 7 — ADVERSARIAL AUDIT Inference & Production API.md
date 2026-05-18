

---

# 🔴 GROUP 7 — ADVERSARIAL AUDIT: Inference & Production API

**Files in scope:** `api.py`, `predictor.py`, `preprocess.py`, `drift_detector.py`, `cache.py`
**Supporting files reviewed:** `test_api.py`, `test_cache.py`, `test_drift_detector.py`, `compute_drift_baseline.py`

**Cross-references:** `sentinel_model.py` (Group 4), `graph_extractor.py` (Group 2), `graph_schema.py` (Group 1), `hash_utils.py` (Group 1), `dual_path_dataset.py` (Group 6), `trainer.py` (Group 5)

---

## 7.1 [CRITICAL] — `cache.py` hardcodes `x.shape[1] != 8` — v5 model with 13-dim features is ALWAYS a cache miss

**File:** `cache.py:97-101`

```python
if not hasattr(graph, "x") or graph.x.shape[1] != 8:
    raise ValueError(
        f"Cached graph has x.shape={tuple(graph.x.shape) if hasattr(graph, 'x') else '?'}, "
        "expected [N, 8]. Schema may have changed — evicting."
    )
```

The v5 model uses `NODE_FEATURE_DIM=13` (from `graph_schema.py`). Every inference request for v5 produces graphs with `x.shape[1]=13`. The cache validation rejects them all as "stale" and evicts the entry. The cache **never returns a hit** for v5 models.

This means:
1. Every `/predict` call pays the full 3-5s Slither cost, even for repeated contracts
2. The `cache.put()` writes the entry, but `cache.get()` always evicts it on the next lookup
3. The entire T1-A optimization (InferenceCache) is inert for v5 — the most important architecture

The test suite (`test_cache.py:23`) uses `_make_graph` with `n_nodes=3` → `x = torch.randn(3, 8)`, so the 8-dim assertion passes by coincidence. No test uses v5 (13-dim) features.

**Fix:** Import `NODE_FEATURE_DIM` from `graph_schema.py` and validate against it. Or better: store `FEATURE_SCHEMA_VERSION` in the cache key (already done — the key includes `_v2` etc.) and skip the shape check entirely, since the schema version already guarantees correctness.

---

## 7.2 [CRITICAL] — `cache.py` uses `weights_only=False` — arbitrary code execution via cache poisoning

**File:** `cache.py:93-94`

```python
graph  = torch.load(graph_path,  map_location="cpu", weights_only=False)
tokens = torch.load(tokens_path, map_location="cpu", weights_only=False)
```

This is the same vulnerability identified in Group 6 (Finding 6.1) but now it's in the **production inference path**. The cache directory defaults to `~/.cache/sentinel/preprocess/` — a user-writable location. Any process running as the same user can write a malicious `.pt` file to this directory, and the next `/predict` request will execute arbitrary code.

The predictor itself notes in its docstring (line 12-13): "weights_only=False kept for checkpoint loading — LoRA state dict contains peft-specific classes." But the cache files contain only PyG Data objects and token dicts — they don't need `weights_only=False`. The safe globals are already registered by `dual_path_dataset.py` at import time.

**Fix:** Switch to `weights_only=True`. The PyG safe globals are already registered at module level by `dual_path_dataset.py` (which is imported by the training path, but may not be imported in the inference server). Add `torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])` to `cache.py`.

---

## 7.3 [HIGH] — `api.py` reads Fix #6 `thresholds` (list) but `PredictResponse` schema still has `threshold` (float) — guaranteed 422 on every response

**File:** `predictor.py:470-473` (Fix #6):
```python
return {
    "thresholds": self.thresholds.cpu().tolist(),  # list[float]
    ...
}
```

**File:** `api.py:140-148`:
```python
class PredictResponse(BaseModel):
    threshold: float    # ← singular float, not list
    ...
```

**File:** `api.py:249`:
```python
threshold=result["threshold"],  # result["thresholds"] is a list → key error or type mismatch
```

Predictor's `_format_result()` returns `"thresholds"` (plural, list). The API schema expects `"threshold"` (singular, float). Two failure modes:
1. If `_format_result` returns `{"thresholds": [...]}` and `api.py` does `result["threshold"]` → **KeyError** → HTTP 500
2. If someone "fixed" the key mismatch by reading `result["thresholds"]` → Pydantic validation fails because `list` is not `float` → **HTTP 422**

The test at `test_api.py:96` asserts `"threshold" in body` — it checks for the singular key. This test would pass only if the predictor still returns `"threshold"` (singular), meaning Fix #6 was never actually applied, or the test is against a stale schema.

This is a **production-breaking inconsistency**: the model code says BREAKING CHANGE, the API schema doesn't reflect it.

**Fix:** Update `PredictResponse` to `thresholds: list[float]` (matching the predictor), update `test_api.py`, and version the API (e.g., `/v2/predict`).

---

## 7.4 [HIGH] — `predictor.py` loads checkpoint with `weights_only=False` — arbitrary code execution via checkpoint file

**File:** `predictor.py:134`

```python
raw = torch.load(checkpoint, map_location=self.device, weights_only=False)
```

This is explicitly acknowledged in the docstring (line 12-13): "LoRA state dict contains peft-specific classes — weights_only=True blocks them." This is a legitimate constraint — PeftModel state dicts contain custom classes that aren't in PyTorch's safe globals list.

However, the checkpoint is loaded **once at startup** (in the lifespan handler) and the file path comes from an environment variable (`SENTINEL_CHECKPOINT`). If an attacker can modify this env var or replace the checkpoint file, they get code execution at server startup. This is a supply-chain attack vector.

**Fix:** Register the necessary `peft` classes in `add_safe_globals()` so `weights_only=True` works. The key classes are `peft.peft_model.PeftModel`, `peft.tuners.lora.model.LoRAModel`, etc. Alternatively, load the checkpoint with `weights_only=True` and reconstruct the LoRA model programmatically rather than deserializing the full PeftModel.

---

## 7.5 [HIGH] — `predictor.py` `_score_windowed` reuses the SAME graph Batch object across all windows — graph tensors accumulate on GPU

**File:** `predictor.py:394-402`

```python
batch = Batch.from_data_list([graph]).to(self.device)   # created ONCE
per_window_probs: list[torch.Tensor] = []
with torch.no_grad():
    for window in windows:
        input_ids = window["input_ids"].to(self.device)
        attention_mask = window["attention_mask"].to(self.device)
        logits = self.model(batch, input_ids, attention_mask)  # same batch every iteration
        per_window_probs.append(torch.sigmoid(logits.float()).squeeze(0))
```

The `batch` object is created once and passed to the model N times (once per window). This is correct for inference (the graph doesn't change between windows). However:

1. `per_window_probs` accumulates `num_classes`-dim tensors on GPU — one per window. For 8 windows × 10 classes × 4 bytes = 320 bytes — negligible. But `logits = self.model(batch, input_ids, attention_mask)` produces intermediate activations each call. With `torch.no_grad()`, these should be freed, but if the model has any internal caching (e.g., attention weight storage from Group 4 Finding 4.9), GPU memory could accumulate across windows.

2. More critically: `input_ids` and `attention_mask` are moved to GPU **inside the loop** but never explicitly freed. Python's GC may not collect them promptly. Under concurrent load, this could contribute to OOM.

3. If any window's `input_ids` has a different device than `batch`, this will silently fail or produce wrong results (though currently both should be on `self.device`).

**Fix:** Move window tensors to GPU inside the loop but add `del input_ids, attention_mask, logits` at the end of each iteration to accelerate GC. Or better: batch all windows into a single forward pass (stack input_ids and attention_mask, expand the graph batch N times) — this is more GPU-efficient.

---

## 7.6 [HIGH] — `preprocess.py` `process_source_windowed` calls `process_source` which writes to cache — then overwrites the cached single-window tokens with multi-window tokens

**File:** `preprocess.py:337-345`

```python
def process_source_windowed(self, source_code, name, stride=256, max_windows=8):
    graph, single_tokens = self.process_source(source_code, name)  # ← writes to cache!
    windows = self._tokenize_sliding_window(
        source_code,
        single_tokens["contract_hash"],
        stride=stride,
        max_windows=max_windows,
    )
    return graph, windows
```

`process_source()` at line 292-294 writes `(graph, single_tokens)` to the cache. On the next call with the same source code, `process_source()` returns the cached single-window result — but `predictor.predict_source()` calls `process_source_windowed()`, which is supposed to return a list of windows. The cache returns the single-window format, and the caller gets a mismatch.

This means:
- First call: `process_source_windowed` → builds graph + single tokens → caches them → creates windows → returns (graph, [window1, window2, ...]) ✓
- Second call (cache hit): `process_source_windowed` → `process_source` hits cache → returns (graph, single_tokens) → the single tokens dict is NOT a list → `predictor._score_windowed` expects a list of dicts → **crash or wrong result**

The cache stores single-window output but the windowed path expects multi-window output. The cache has no way to distinguish which format was stored.

**Fix:** Either (a) don't cache windowed results, or (b) use a different cache key for windowed vs single (e.g., append `_windowed`), or (c) have `process_source_windowed` skip the cache entirely and only write windowed results to the cache.

---

## 7.7 [HIGH] — `drift_detector.py` baseline JSON has no integrity check — corrupted/truncated file silently produces wrong KS results

**File:** `drift_detector.py:93-95`

```python
if bp.exists():
    with open(bp) as f:
        self._baseline = json.load(f)
```

If the baseline file is truncated (partial write, disk full, crash during creation), `json.load` will raise `JSONDecodeError`. But if it's a valid JSON file with wrong values (e.g., all zeros from a failed warm-up export, or an empty dict `{}`), the detector will silently run KS tests against garbage data. An empty dict means `self._baseline.items()` produces nothing, so `check()` always returns `{}` — drift detection is silently disabled.

Additionally, there's no validation that baseline values are reasonable (e.g., `num_nodes` should be > 0, `num_edges` should be >= 0). A baseline with `num_nodes: [0, 0, 0]` would make the KS test fire on every real contract.

**Fix:** Validate the loaded baseline: assert at least one stat key, assert each key has >= MIN_SAMPLES_FOR_KS values, assert values are positive for count-based stats. Log the number of stats and sample sizes.

---

## 7.8 [MEDIUM] — `predictor.py` `lora_target_modules` from checkpoint config still has the string→list deserialization trap from Group 4 Finding 4.1

**File:** `predictor.py:225-227`

```python
lora_target_modules=saved_cfg.get(
    "lora_target_modules", ["query", "value"]
),
```

If `saved_cfg["lora_target_modules"]` is the string `"query,value"` (which happens when MLflow or certain checkpoint serializers flatten lists), this passes a string to `SentinelModel.__init__` → `TransformerEncoder.__init__` → `LoraConfig(target_modules="query,value")`. The LoRA adapter will be created with zero target modules (no string matches any module name), silently producing a model with no LoRA gradients — identical to the bug in Group 4 Finding 4.1.

The predictor doesn't add the type guard recommended in 4.1. If a checkpoint stored `lora_target_modules` as a string, the predictor would load a broken model and the warmup forward pass would still succeed (it just runs a forward pass, it doesn't check trainable params).

**Fix:** Add the same type guard from 4.1: `if isinstance(v, str): v = [s.strip() for s in v.split(",")]` and `assert trainable_params > 0` after model construction.

---

## 7.9 [MEDIUM] — `preprocess.py` `_tokenize_sliding_window` doesn't add `[CLS]` and `[SEP]` tokens — window token sequences diverge from training

**File:** `preprocess.py:489-521`

The sliding-window code does:
```python
full_ids = self.tokenizer.encode(source_code, add_special_tokens=True)
...
chunk = full_ids[start:end]
pad_len = self.MAX_TOKEN_LENGTH - len(chunk)
input_ids = torch.tensor([chunk + [self.tokenizer.pad_token_id] * pad_len], dtype=torch.long)
```

`tokenizer.encode(add_special_tokens=True)` adds `[CLS]` at position 0 and `[SEP]` at the end of the **full** sequence. The sliding window then takes a chunk from the middle. Result:
- **Window 0**: Starts with `[CLS]` (good), but the chunk may include `[SEP]` in the middle if the full sequence is short enough. If the window ends before `[SEP]`, there's no `[SEP]` at all.
- **Window 1+**: Starts with a regular token (no `[CLS]`), may or may not end with `[SEP]`.

During training, every token sequence has exactly `[CLS] ... tokens ... [SEP] [PAD]...`. During inference with sliding windows, the format is `... mid-sequence tokens ... [PAD]...` for most windows. CodeBERT's positional embeddings and attention patterns are trained with `[CLS]` at position 0 and `[SEP]` as a boundary marker. Windows without these tokens will produce out-of-distribution attention patterns.

The single-window path (`_tokenize`) correctly uses `self.tokenizer(source_code, max_length=512, truncation=True, padding="max_length")` which handles `[CLS]`/`[SEP]` correctly. But the sliding-window path bypasses the tokenizer's padding/truncation logic entirely.

**Fix:** For each window chunk, manually prepend `[CLS]` and append `[SEP]`, then pad to 512. Or re-tokenize each window's source text substring independently using the tokenizer's standard pipeline.

---

## 7.10 [MEDIUM] — `api.py` drift update uses `float(result["num_nodes"])` but `_format_result` returns `int(graph.num_nodes)` — always works but loses precision for drift detection

**File:** `api.py:224-227`

```python
drift_detector.update_stats({
    "num_nodes": float(result["num_nodes"]),
    "num_edges": float(result["num_edges"]),
})
```

`result["num_nodes"]` is an `int` (from `predictor.py:477: int(graph.num_nodes)`). Converting to `float` is fine for small values but loses precision above 2^53. More importantly, the drift detector only tracks two crude stats (`num_nodes`, `num_edges`). It doesn't track the features that actually matter for model accuracy: average node degree, ratio of CFG edges to AST edges, type distribution of nodes, etc. A contract with 100 state-variable nodes has very different vulnerability characteristics than one with 100 CFG nodes, but both report `num_nodes=100` to the drift detector.

**Fix:** Add more informative stats to the drift detector: node type distribution (percentage of each type_id), average degree, CFG depth, etc. These are already available in the graph object.

---

## 7.11 [MEDIUM] — `predictor.py` `_score_windowed` uses max-aggregation across windows — a single high-probability window dominates, negating the benefit of multi-window analysis

**File:** `predictor.py:377-389`

```python
@staticmethod
def _aggregate_window_predictions(probs_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(probs_list).max(dim=0).values  # [num_classes]
```

Max-aggregation means that if window 3 assigns Reentrancy probability 0.8, the final result is 0.8 regardless of what the other 7 windows say. This is extremely sensitive to noise — a single window with an anomalous spike (e.g., due to a code pattern that looks superficially similar to a vulnerability) determines the outcome.

This also means that the effective threshold is lower than intended. If the per-class threshold is 0.5, and 8 windows each independently have a 10% chance of producing a spurious 0.5+ probability for a given class, the probability of at least one window exceeding 0.5 is `1 - 0.9^8 = 57%` — the false positive rate nearly doubles compared to single-window inference.

**Fix:** Consider using a more robust aggregation: e.g., second-highest (top-2 max), or mean with a minimum-count threshold (at least 2 windows must exceed threshold), or weighted aggregation where later windows get more weight (since later code often contains the actual vulnerability logic).

---

## 7.12 [MEDIUM] — `preprocess.py` `process()` reads source with `errors="ignore"` — silently drops invalid UTF-8 bytes, producing different tokens than `process_source()`

**File:** `preprocess.py:194`

```python
tokens = self._tokenize(
    sol_path.read_text(encoding="utf-8", errors="ignore"),
    contract_hash,
)
```

vs. `process_source()` which tokenizes the in-memory `source_code` string directly (no encoding issues). If a `.sol` file contains invalid UTF-8 bytes, `process()` silently drops them, producing a different token sequence than `process_source()` would produce for the same logical content. This means the same contract analyzed via file path vs. raw string can produce different vulnerability scores.

**Fix:** Use `errors="replace"` (consistent with `tokenizer.py:142`) or raise `ValueError` on encoding errors.

---

## 7.13 [MEDIUM] — `compute_drift_baseline.py` uses `weights_only=False` and stores the entire training set in RAM as a JSON array

**File:** `compute_drift_baseline.py:56, 119-127`

```python
data = torch.load(path, map_location="cpu", weights_only=False)
...
baseline: dict[str, list[float]] = defaultdict(list)
for path in pt_files:
    stats = _extract_stats_from_graph(path)
    for k, v in stats.items():
        baseline[k].append(v)
```

For 68K graph files, this builds a dict with two keys (`num_nodes`, `num_edges`) each containing 68K float values. The output JSON file will be ~1.5 MB — manageable. But the `weights_only=False` issue (Finding 6.1 applies here too) and the fact that the entire list is loaded into the `DriftDetector` at startup means the baseline JSON could be a multi-MB file in memory.

More importantly: if someone accidentally adds more stat keys (e.g., per-class probability distributions), the baseline JSON could grow to hundreds of MB, and the KS test over 68K samples is O(n log n) — potentially slow at check time.

**Fix:** Switch to `weights_only=True`. Add a cap on the number of baseline samples (e.g., 5000 randomly sampled values) since the KS test is well-calibrated with far fewer samples than 68K.

---

## 7.14 [MEDIUM] — `api.py` `request_count` is not thread-safe — concurrent requests can cause drift check to be skipped or double-fired

**File:** `api.py:228-229`

```python
request.app.state.request_count += 1
if request.app.state.request_count % DRIFT_CHECK_INTERVAL == 0:
    drift_detector.check()
```

FastAPI with `asyncio.to_thread` can handle concurrent requests. The `request_count += 1` operation is not atomic — two concurrent requests can both read the same count, both increment to the same value, and both trigger (or both skip) the drift check.

Under uvicorn with a single event loop, the `request_count += 1` in the async handler is actually safe because Python's GIL ensures atomic integer increments. However, if the API is deployed with multiple workers (`--workers N`), each worker has its own `request_count`, and drift checks will fire independently per worker at different intervals — which is actually fine. The real risk is if someone changes the deployment to use threading without the GIL (free-threaded Python 3.13+), where `+=` is NOT atomic.

**Fix:** Use `asyncio.Lock` around the counter increment and drift check, or use `threading.Lock` if deploying with threads.

---

## 7.15 [MEDIUM] — `preprocess.py` `_extract_graph` sets `graph.y = torch.tensor([0])` — same hardcoded-zero label as the training pipeline bug

**File:** `preprocess.py:394`

```python
graph.y = torch.tensor([0], dtype=torch.long)  # dummy; not used in forward
```

The comment says "not used in forward pass" and technically the label is not used during inference. However, `graph.y` is part of the cached graph object. If the cache is ever used as a training data source (e.g., for online learning), the hardcoded `y=0` label will silently label all inference samples as "safe" — the same bug as Group 2 Finding 2.1.

Additionally, some PyG `Batch` operations may propagate `y` into batch-level attributes, and if any future code path reads `batch.y` for logging or monitoring, it will always show "safe".

**Fix:** Remove `graph.y` entirely for inference, or set it to `None` / `-1` to make it clear this is unlabeled data.

---

## 7.16 [LOW] — `api.py` `PredictRequest.must_look_like_solidity` validator is trivially bypassable — `"pragma"` in any non-Solidity text passes

**File:** `api.py:126-131`

```python
@field_validator("source_code")
@classmethod
def must_look_like_solidity(cls, v: str) -> str:
    if "pragma" not in v.lower() and "contract" not in v.lower():
        raise ValueError(...)
    return v
```

Any text containing the word "pragma" or "contract" passes this check. Examples that would pass:
- `"pragma once\n#include <stdio.h>"` (C header)
- `"The contract was signed yesterday"` (natural language)
- `"contract\nhello world"` (garbage)

This wastes GPU resources on non-Solidity inputs. The validator should check for at least `.sol` file-like structure, e.g., requiring both `pragma solidity` AND `contract` (or `interface`/`library`).

**Fix:** Require at least two indicators: `"pragma solidity" in v.lower() or ("contract" in v.lower() and ("function" in v.lower() or "mapping" in v.lower()))`.

---

## 7.17 [LOW] — `test_api.py` doesn't test timeout, OOM, drift detection, or windowed inference

**File:** `test_api.py`

Missing test coverage:
1. **Timeout** (`PREDICT_TIMEOUT`): No test for HTTP 504
2. **OOM**: No test for HTTP 413 from `torch.cuda.OutOfMemoryError`
3. **Drift detection**: No test that drift alerts are fired/counted
4. **Windowed inference**: No test for long contracts (> 512 tokens) that trigger multi-window path
5. **Cache behavior**: No test that cache hit/miss affects response time or result
6. **Thresholds vs threshold**: No test for the breaking-change inconsistency (Finding 7.3)

**Fix:** Add integration tests for all six uncovered paths.

---

## 7.18 [LOW] — `cache.py` `_atomic_save` creates `.tmp` file but doesn't clean it up on failure

**File:** `cache.py:150-154`

```python
@staticmethod
def _atomic_save(obj: object, dest: Path) -> None:
    tmp = dest.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(dest)
```

If `torch.save` fails (disk full, permission error), the `.tmp` file remains on disk. Over time, these accumulate. There's no cleanup mechanism — no startup scan for `.tmp` files (unlike the sentinel temp file cleanup in `preprocess.py`).

**Fix:** Add a try/except that deletes the `.tmp` file on failure, and a startup scan for orphaned `.tmp` files.

---

## 7.19 [LOW] — `drift_detector.py` `dump_warmup_stats()` returns raw buffer — no dedup, no stat summary, hard to use for baseline construction

**File:** `drift_detector.py:175-177`

```python
def dump_warmup_stats(self) -> list[dict[str, float]]:
    return list(self._buffer)
```

This returns up to 200 raw dicts (one per request). To create a baseline, the operator needs to:
1. Call this method
2. Serialize to JSONL
3. Run `compute_drift_baseline.py --source warmup`

But there's no API endpoint to trigger this dump, no automatic baseline creation, and no documentation of the workflow. The `compute_drift_baseline.py` docstring mentions `/debug/warmup_dump` but no such endpoint exists in `api.py`.

**Fix:** Add a `/debug/warmup_dump` endpoint to `api.py`, or add an admin CLI command that connects to the running API and triggers the dump.

---

## 7.20 [LOW] — `predictor.py` `_ARCH_TO_NODE_DIM` hardcodes 12 and 8 — same magic-number problem as Groups 1, 4, 6

**File:** `predictor.py:87-92`

```python
_ARCH_TO_NODE_DIM: dict[str, int] = {
    "three_eye_v5":         12,    # v5: NODE_FEATURE_DIM=12
    "cross_attention_lora": 8,     # v4: NODE_FEATURE_DIM=8
    "legacy":               8,
    "legacy_binary":        8,
}
```

This is the same `8`/`12` magic number problem identified across Groups 1 (Finding 1.2), 4 (Finding 4.6), and 6 (Finding 6.20). If `NODE_FEATURE_DIM` changes in `graph_schema.py`, this dict must be manually updated — there's no import or validation.

**Fix:** Import `NODE_FEATURE_DIM` from `graph_schema.py` and use it for the current architecture. Only hardcode legacy values.

---

## Summary Table

| # | Severity | File | Finding |
|---|----------|------|---------|
| 7.1 | **CRITICAL** | cache.py | Hardcoded `x.shape[1] != 8` — v5 (13-dim) graphs ALWAYS cache-miss |
| 7.2 | **CRITICAL** | cache.py | `weights_only=False` on cache files — arbitrary code execution |
| 7.3 | HIGH | api.py + predictor.py | Fix #6 returns `thresholds` (list) but `PredictResponse` expects `threshold` (float) → HTTP 500/422 |
| 7.4 | HIGH | predictor.py | `weights_only=False` on checkpoint loading — supply-chain attack vector |
| 7.5 | HIGH | predictor.py | Windowed inference reuses same Batch — GPU memory risk under concurrent load |
| 7.6 | HIGH | preprocess.py | Windowed path writes single-window to cache, then reads single-window on cache hit — format mismatch |
| 7.7 | HIGH | drift_detector.py | No baseline validation — empty/corrupted JSON silently disables drift detection |
| 7.8 | MEDIUM | predictor.py | `lora_target_modules` string→list trap from G4 F4.1 not guarded |
| 7.9 | MEDIUM | preprocess.py | Sliding windows don't add `[CLS]`/`[SEP]` — out-of-distribution token sequences |
| 7.10 | MEDIUM | api.py | Drift stats too crude (only node/edge count) — misses meaningful distribution shifts |
| 7.11 | MEDIUM | predictor.py | Max-aggregation across windows inflates false positive rate |
| 7.12 | MEDIUM | preprocess.py | `errors="ignore"` vs `errors="replace"` inconsistency between process() and process_source() |
| 7.13 | MEDIUM | compute_drift_baseline.py | `weights_only=False` + no sample cap for baseline |
| 7.14 | MEDIUM | api.py | `request_count` not thread-safe under free-threaded Python |
| 7.15 | MEDIUM | preprocess.py | `graph.y = torch.tensor([0])` — same hardcoded-zero label as training bug |
| 7.16 | LOW | api.py | Solidity validator trivially bypassable |
| 7.17 | LOW | test_api.py | No tests for timeout, OOM, drift, windowed inference, cache, or threshold schema |
| 7.18 | LOW | cache.py | `_atomic_save` leaves `.tmp` files on failure |
| 7.19 | LOW | drift_detector.py | No API endpoint for warmup dump despite docstring mentioning one |
| 7.20 | LOW | predictor.py | `_ARCH_TO_NODE_DIM` hardcodes 8/12 — disconnected from `graph_schema.py` |

**Critical cluster analysis:**

Findings 7.1 + 7.6 form a devastating pair for production:
- **7.1**: The cache never works for v5 — every request pays 3-5s Slither cost
- **7.6**: Even if 7.1 were fixed, the windowed path would cache single-window results and return them when multi-window results are expected, causing a crash in `_score_windowed`

Finding 7.3 means the API is likely **broken in production right now** if Fix #6 was actually applied — every `/predict` response would fail Pydantic validation.

Finding 7.9 means that for long contracts (> 512 tokens), the model receives token sequences that are structurally different from what it was trained on, potentially degrading detection quality for the most complex (and often most vulnerable) contracts.

**Connection to prior groups:**
- 7.1 directly connects to Group 1 Finding 1.2 (NODE_FEATURE_DIM=8 vs 13, schema versioning)
- 7.2 connects to Group 6 Findings 6.1 + 6.7 (weights_only=False across the pipeline)
- 7.8 is the same bug as Group 4 Finding 4.1 (lora_target_modules string→list)
- 7.15 connects to Group 2 Finding 2.1 (graph.y=0 hardcoding)
- 7.20 connects to Group 1 Finding 1.2, Group 4 Finding 4.6, Group 6 Finding 6.20 (8/12 magic numbers)
Group 7 audit complete. Ready for Group 8 (Operational Scripts & Augmentation) when you are.

**Top-line: Three CRITICAL/HIGH findings that would break production:**

1. **7.1 [CRITICAL]** — The inference cache has `x.shape[1] != 8` hardcoded. Since v5 uses 13-dim features, **every** cache lookup is a miss. The T1-A optimization is completely inert.

2. **7.3 [HIGH]** — Predictor Fix #6 renamed `threshold` → `thresholds` (list), but `PredictResponse` still expects `threshold` (float). This means **every API response** either KeyError's internally or fails Pydantic validation → HTTP 422/500.

3. **7.6 [HIGH]** — The windowed inference path (`process_source_windowed`) calls `process_source` which writes single-window tokens to cache. On cache hit, the single-window dict is returned instead of a list of windows → crash in `_score_windowed`.