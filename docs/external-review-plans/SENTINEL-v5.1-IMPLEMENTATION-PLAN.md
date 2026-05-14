# SENTINEL v5.1 — Consolidated Implementation Plan

**Date:** 2026-05-13
**Source:** Adversarial Audit Groups 1–7 + v5.1 Root Cause Analysis
**Status:** READY FOR IMPLEMENTATION
**Total Findings:** ~90+ across 7 audit groups
**Estimated Total Effort:** 55–70 hours (code fixes) + 12 hrs training runtime

---

## How This Plan Is Organized

Findings from all 7 audit groups are consolidated into **6 implementation phases**, ordered by dependency and severity. Each phase lists concrete tasks with:
- **Source** (which audit group / finding ID)
- **Severity** (CRITICAL / HIGH / MEDIUM / LOW)
- **File(s)** affected
- **Estimated effort**
- **Dependency** (what must complete first)

The phases are designed so that each phase is independently testable before moving to the next.

---

## Phase 0 — Security & Critical Bug Fixes

**Goal:** Eliminate all RCE vectors and broken-code-path bugs. No training or inference should run until Phase 0 is complete.
**Estimated Effort:** 4–5 hours
**Dependency:** None (can start immediately)

### 0.1 [CRITICAL] `weights_only=False` — Eliminate All RCE Vectors

**Source:** G2#1, G2#8, G2#15, G2#20, G5#1, G5#9, G5#16, G6#7, G6#13, G6#20, G7-summary
**Files (10+ locations):**

| File | Line(s) | Current Code | Fix |
|------|---------|-------------|-----|
| `dual_path_dataset.py` | ~108, 155, 225 | `torch.load(path, ...)` | Add `weights_only=True` |
| `create_label_index.py` | ~45 | `torch.load(args.dataset_path)` | Add `weights_only=True` |
| `build_multilabel_index.py` | ~89, 142 | `torch.load(...)` | Add `weights_only=True` |
| `validate_graph_dataset.py` | ~67, 89 | `torch.load(...)` | Add `weights_only=True` |
| `tune_threshold.py` | ~188 | `weights_only=False` | Change to `True` |
| `promote_model.py` | ~67 | `weights_only=False` | Change to `True` |
| `compute_drift_baseline.py` | ~56 | `weights_only=False` | Change to `True` |
| `predictor.py` | ~124 | `weights_only=False` | Change to `True` + hash validation |
| `cache.py` | ~88-89 | `weights_only=False` | Change to `True` + integrity check |
| `preprocess.py` | ~88-89 (cache load) | `weights_only=False` | Change to `True` |

**Special Cases:**
- `predictor.py` and `cache.py` load model checkpoints with extra keys. For `weights_only=True`, ensure checkpoint dicts use only tensor/primitive types. If checkpoints contain Lambda or custom objects, add a `safe_load` wrapper that loads with `weights_only=True` and reconstructs config from metadata.
- Add `SENTINEL_CHECKPOINT_HASH` environment variable check for production: verify SHA-256 of checkpoint before loading.

**Effort:** 1.5 hours

### 0.2 [CRITICAL] `_add_node` Type-ID Roundtrip Bug

**Source:** G2 Critical finding
**File:** `ml/src/preprocessing/graph_extractor.py`
**Bug:** `int(normalized_float)` where `normalized_float = type_id / 12.0` always yields 0 for all type_ids 0–11.
**Impact:** All type_id information is lost after normalization — `_add_node` receives type_id=0 for every node.
**Fix:** Store raw type_id before normalization, or denormalize: `int(round(normalized_float * 12.0))`.

```python
# In _add_node or wherever type_id is read back from features:
raw_type_id = int(round(node_features[0] * 12.0))
```

**Effort:** 30 minutes

### 0.3 [CRITICAL] API `thresholds` vs `threshold` Schema Mismatch

**Source:** G7#7.3 (from conversation context)
**File:** `ml/src/inference/api.py` — `PredictResponse` model
**Bug:** `PredictResponse` expects `threshold: float` but predictor returns `thresholds: list[float]` → every response is HTTP 500/422.
**Fix:** Update `PredictResponse` to include `thresholds: list[float]` field, or add both for backward compatibility.

```python
class PredictResponse(BaseModel):
    # ... existing fields ...
    threshold: float = Field(description="Legacy: first threshold value")
    thresholds: list[float] = Field(description="Per-class thresholds")
```

**Effort:** 30 minutes

### 0.4 [CRITICAL] Cache Dimension Check Hardcodes 8 → v5 13-dim Always Miss

**Source:** G7#7.1 (from conversation context)
**File:** `ml/src/inference/cache.py`
**Bug:** `if graph.x.shape[1] != 8` — v5 uses 13-dimensional features, so every cache lookup is a miss. The T1-A optimization is completely inert.
**Fix:** Read expected dimension from schema or checkpoint config instead of hardcoding.

```python
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM
# ...
if graph.x.shape[1] != NODE_FEATURE_DIM:
    logger.warning("Cached graph feature dim mismatch...")
    return None
```

**Effort:** 30 minutes

### 0.5 [CRITICAL] Dedup Never Wired Into Training

**Source:** G6#6.2 (from conversation context)
**Bug:** `dedup_multilabel_index.py` produces `multilabel_index_deduped.csv` and deduped splits, but `trainer.py` hardcodes the leaky CSV and old splits. No CLI flag or auto-detection switches to deduped versions.
**Fix:** Add `--use-dedup` flag to `train.py` and `trainer.py`, or auto-detect `_deduped.csv` file existence.

```python
# In train.py:
parser.add_argument("--use-dedup", action="store_true",
                    help="Use deduplicated CSV and splits (recommended)")
# In trainer.py or dataset:
if config.use_dedup:
    csv_path = csv_path.replace(".csv", "_deduped.csv")
    splits_dir = Path(str(splits_dir).replace("/splits", "/splits_dedup"))
```

**Effort:** 1 hour

### 0.6 [CRITICAL] `graph.y=0` Hardcoding → 100% "safe" Label Index

**Source:** G6#6.4 (from conversation context)
**File:** `ml/scripts/create_label_index.py`
**Bug:** Reads `graph.y` which is always 0 (hardcoded during extraction), producing a label index where all contracts are "safe".
**Fix:** Read labels from `contract_labels.csv` / `multilabel_index.csv` instead of from graph objects.

**Effort:** 1 hour (or delete the redundant script per G2#9)

---

## Phase 1 — v5.1 Architecture Fixes

**Goal:** Fix the three root causes of v5.0's behavioral test failure (15% detection, 0% specificity).
**Estimated Effort:** 4 hours
**Dependency:** Phase 0 complete

### 1.1 [CRITICAL] Fix `_select_contract()` — Interface Selection Bug

**Source:** v5.1 Plan §0a, G2 finding
**File:** `ml/src/preprocessing/graph_extractor.py`, `_select_contract()` ~line 582
**Bug:** `candidates[0]` picks the first non-dependency contract, which is often an interface → ghost graphs (0 nodes, 0 edges).
**Impact:** ~10% of training graphs are ghosts, 4 of 20 behavioral test contracts produce ghost graphs.
**Fix (from v5.1 plan):**

```python
def _select_contract(sl, config):
    candidates = [c for c in sl.contracts if not c.is_from_dependency()]
    if not candidates:
        raise EmptyGraphError(...)

    non_iface = [c for c in candidates if not c.is_interface]
    if non_iface:
        candidates = sorted(non_iface, key=lambda c: len(c.functions), reverse=True)

    if config.multi_contract_policy == "by_name" and config.target_contract_name:
        matching = [c for c in candidates if c.name == config.target_contract_name]
        if matching:
            return matching[0]
        logger.warning("Contract %r not found; falling back.", ...)

    return candidates[0]
```

**Verification:** Run `11_external_bug.sol`, `13_multilabel_complex.sol`, `20_unused_return_minimal.sol` through extractor — expect nodes ≥ 10, edges ≥ 5.

**Effort:** 30 minutes (code) + 15 minutes (verification)

### 1.2 [CRITICAL] Function-Level GNN Pooling

**Source:** v5.1 Plan §0b, G3 finding (gradient collapse), G7 finding (CFG_RETURN flood)
**File:** `ml/src/models/sentinel_model.py`, `_build_gnn_eye()`
**Bug:** `global_max_pool + global_mean_pool` over ALL nodes — CFG_RETURN nodes (77% of CFG mass) dominate, drowning out CFG_CALL/CFG_WRITE signal.
**Impact:** GNN eye gradient share collapsed to 6.7% by epoch 43. This is the primary cause of v5.0 failure.
**Fix (from v5.1 plan):**

```python
POOL_NODE_TYPES = {NODE_TYPES[t] for t in
                   ("FUNCTION", "FALLBACK", "RECEIVE", "CONSTRUCTOR", "MODIFIER")}

def _build_gnn_eye(self, node_embs, node_types, batch):
    pool_mask = torch.zeros(node_embs.size(0), dtype=torch.bool,
                            device=node_embs.device)
    for t in POOL_NODE_TYPES:
        pool_mask |= (node_types == t)

    if not pool_mask.any():
        pooled_embs, pooled_batch = node_embs, batch
    else:
        pooled_embs = node_embs[pool_mask]
        pooled_batch = batch[pool_mask]

    gnn_pooled = torch.cat([
        global_max_pool(pooled_embs, pooled_batch),
        global_mean_pool(pooled_embs, pooled_batch),
    ], dim=-1)
    return self.gnn_eye_proj(gnn_pooled)
```

**Must also:** Pass `node_types` into `forward()` from `data.x[:, 0]`:
```python
node_types = (data.x[:, 0] * 12.0).round().long()
```

**Effort:** 2 hours (code + testing)

### 1.3 [HIGH] Raise `aux_loss_weight` 0.1 → 0.3

**Source:** v5.1 Plan §0c, G3/G4 findings
**Files:** `ml/scripts/train.py`, `ml/src/training/trainer.py`
**Fix:**
```python
# train.py argparse
parser.add_argument("--gnn-aux-weight", type=float, default=0.3)  # was 0.1
```
Also verify TrainConfig default matches.

**Effort:** 5 minutes

### 1.4 [MEDIUM] CFG Exception Counter (Non-Silent Failure Tracking)

**Source:** v5.1 Plan §0e, G2 finding
**File:** `ml/src/preprocessing/graph_extractor.py`, `_build_control_flow_edges()`
**Fix:** Add module-level counter and warn if exception rate exceeds 5% of contracts processed.

```python
_cfg_failure_count = 0
_cfg_attempt_count = 0

def _build_control_flow_edges(...):
    global _cfg_failure_count, _cfg_attempt_count
    _cfg_attempt_count += 1
    try:
        # existing code
    except Exception as e:
        _cfg_failure_count += 1
        if _cfg_attempt_count > 0 and _cfg_failure_count / _cfg_attempt_count > 0.05:
            logger.error(
                f"CFG extraction failure rate: {_cfg_failure_count}/{_cfg_attempt_count} "
                f"({100*_cfg_failure_count/_cfg_attempt_count:.1f}%) — exceeds 5% threshold"
            )
        # existing fallback
```

**Effort:** 30 minutes

---

## Phase 2 — Data Pipeline Fixes

**Goal:** Fix data integrity, validation gaps, and pipeline correctness.
**Estimated Effort:** 6–8 hours
**Dependency:** Phase 0 complete (security fixes); Phase 1 helps but not strictly required

### 2.1 [HIGH] Wire Dedup Into Training Pipeline

**Source:** Phase 0.5 above, G6#6.2
**Depends on:** Phase 0.5 (flag infrastructure)
**Additional work:**
- Update `create_splits.py` to generate deduped split files
- Verify `verify_splits.py` works with deduped data
- Add smoke test: train 1 epoch with `--use-dedup`, verify data loads correctly

**Effort:** 2 hours (including testing)

### 2.2 [HIGH] Token `.pt` Saves 7 Extra Keys

**Source:** G6#6.3
**File:** `ml/src/data_extraction/tokenizer.py`
**Bug:** Token `.pt` files save 7 extra keys beyond `input_ids` / `attention_mask`, bloating storage and causing confusion. `feature_schema_version` is never validated at training time.
**Fix:**
- Strip extraneous keys when saving token files (keep only `input_ids`, `attention_mask`)
- Add `feature_schema_version` validation in `DualPathDataset.__init__`

**Effort:** 1.5 hours

### 2.3 [HIGH] Different Stratification Methods Between Scripts

**Source:** G6#6.6
**Files:** `create_splits.py`, `dedup_multilabel_index.py`
**Bug:** `create_splits.py` uses one stratification method + PRNG, `dedup_multilabel_index.py` uses a different one. Split distributions diverge.
**Fix:** Extract shared stratification utility into a single module, use consistent PRNG seeding.

**Effort:** 1.5 hours

### 2.4 [HIGH] Orphan Rows in Dedup Get 1-Element Groups

**Source:** G6#6.5
**File:** `ml/scripts/dedup_multilabel_index.py`
**Bug:** Rows with no content-hash match get their own 1-element group — not actually deduped.
**Fix:** After grouping by content-hash, merge singletons into the nearest group or flag them for manual review.

**Effort:** 1 hour

### 2.5 [MEDIUM] Add Graph Feature Dimension Validation

**Source:** G2#5, G3#1
**Files:** `dual_path_dataset.py`, `validate_graph_dataset.py`
**Fix:**
```python
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM

# In DualPathDataset.__getitem__:
if graph.x.shape[1] != NODE_FEATURE_DIM:
    raise ValueError(
        f"Graph feature dim {graph.x.shape[1]} != expected {NODE_FEATURE_DIM}. "
        f"File: {graph_path}"
    )
```
Also add to `validate_graph_dataset.py`:
- NaN/Inf check
- Edge index bounds check
- Self-loop detection

**Effort:** 1.5 hours

### 2.6 [MEDIUM] Class Distribution Validation After Splits

**Source:** G2#13
**File:** `ml/scripts/create_splits.py`
**Fix:** Add post-split validation that no class has zero positive samples in any split.

**Effort:** 30 minutes

---

## Phase 3 — Inference & API Fixes

**Goal:** Fix production inference path bugs.
**Estimated Effort:** 6–8 hours
**Dependency:** Phase 0 complete

### 3.1 [HIGH] Windowed Inference Cache Format Mismatch

**Source:** G7#7.6 (conversation context)
**File:** `ml/src/inference/predictor.py` / `cache.py`
**Bug:** Windowed path writes single-window tokens to cache, reads them back on cache hit → format mismatch crash in `_score_windowed`.
**Fix:** Cache the aggregated result, not individual windows. Or add window-count metadata to cached entries.

**Effort:** 2 hours

### 3.2 [HIGH] Drift Detector Baseline Integrity

**Source:** G6#7.7, G5#17
**File:** `ml/src/inference/drift_detector.py`, `ml/scripts/compute_drift_baseline.py`
**Bugs:**
- Baseline JSON has no integrity check — empty/corrupted file silently disables drift detection
- MIN_SAMPLES_FOR_KS = 30 is statistically insufficient
- No validation of stat distributions for outliers
**Fixes:**
- Add checksum to baseline file; validate on load
- Increase MIN_SAMPLES_FOR_KS to 100
- Add outlier detection (IQR method)
- Add `--force` flag for training data source in compute_drift_baseline.py

**Effort:** 2 hours

### 3.3 [HIGH] Windowed Inference Reuses Batch Object

**Source:** G7#7.5 (conversation context)
**File:** `ml/src/inference/predictor.py`
**Bug:** Same Batch object reused across windows → GPU memory risk.
**Fix:** Create a fresh Batch per window, or explicitly detach/free between windows.

**Effort:** 1 hour

### 3.4 [MEDIUM] Sliding Windows Missing [CLS]/[SEP]

**Source:** G7#7.9 (conversation context)
**File:** `ml/src/inference/preprocess.py`
**Bug:** Sliding windows don't add [CLS]/[SEP] tokens → out-of-distribution token sequences.
**Fix:** Add [CLS] at start and [SEP] at end of each window.

**Effort:** 1 hour

### 3.5 [MEDIUM] No Rate Limiting on API

**Source:** G6#2
**File:** `ml/src/inference/api.py`
**Fix:** Add slowapi or similar rate limiter.

**Effort:** 1 hour

### 3.6 [MEDIUM] No Cache Size Limit / LRU Eviction

**Source:** G6#16
**File:** `ml/src/inference/cache.py`
**Fix:** Add `max_entries` parameter and LRU eviction.

**Effort:** 1.5 hours

### 3.7 [LOW] No Inference Latency Metrics

**Source:** G6#11
**File:** `ml/src/inference/predictor.py`
**Fix:** Add Prometheus histogram for inference latency.

**Effort:** 30 minutes

---

## Phase 4 — Training Pipeline Hardening

**Goal:** Make training robust, reproducible, and observable.
**Estimated Effort:** 10–12 hours
**Dependency:** Phases 0–1 complete before training; some items can be done in parallel

### 4.1 [HIGH] Checkpoint Format Versioning

**Source:** G4#1
**File:** `ml/src/training/trainer.py`
**Bug:** Checkpoint dict structure is rigid — any key change breaks backward compatibility.
**Fix:** Add `checkpoint_version` field and migration function.

**Effort:** 2 hours

### 4.2 [HIGH] Atomic Sidecar Writes

**Source:** G4#2
**File:** `ml/src/training/trainer.py`
**Bug:** JSON sidecar written without atomic rename → corruption risk.
**Fix:** Use tmp file + `os.rename()` pattern.

**Effort:** 30 minutes

### 4.3 [HIGH] Centralize CLASS_NAMES

**Source:** G4#4, G7#2
**Files:** `trainer.py`, `analyse_truncation.py`, `predictor.py`, `tune_threshold.py`
**Bug:** CLASS_NAMES duplicated in 4+ places.
**Fix:** Define once in `graph_schema.py`, import everywhere.

**Effort:** 1.5 hours

### 4.4 [HIGH] Pre-flight Validation in train.py

**Source:** G4#13
**File:** `ml/scripts/train.py`
**Fix:** Add checks for directory existence, label CSV validity, GPU memory sufficiency.

**Effort:** 1.5 hours

### 4.5 [HIGH] Gradient Accumulation Support

**Source:** G4#11
**File:** `ml/src/training/trainer.py`
**Fix:** Add `grad_accum_steps` parameter.

**Effort:** 2 hours

### 4.6 [MEDIUM] FocalLoss Alpha Per-Class Weighting

**Source:** G4#28
**File:** `ml/src/training/focalloss.py`
**Bug:** Scalar alpha treats all classes the same; multi-label needs per-class weighting.
**Fix:** Add `pos_weights` parameter.

**Effort:** 1.5 hours

### 4.7 [MEDIUM] FocalLoss Numerical Stability

**Source:** G3#21
**File:** `ml/src/training/focalloss.py`
**Bug:** `torch.log(pt)` produces `-inf` when `pt → 0`.
**Fix:** Add epsilon clipping: `torch.log(torch.clamp(pt, min=1e-7))`.

**Effort:** 15 minutes

### 4.8 [MEDIUM] TrainConfig Validation

**Source:** G4#8
**File:** `ml/src/training/trainer.py`
**Fix:** Add `__post_init__` validation for invalid parameter combinations.

**Effort:** 30 minutes

### 4.9 [LOW] RAM Cache Memory Leak

**Source:** G4#5
**File:** `ml/src/training/trainer.py` (DualPathDataset interaction)
**Fix:** Add `cache.clear()` between experiment runs in auto_experiment.py.

**Effort:** 30 minutes

---

## Phase 5 — Code Quality & Maintenance

**Goal:** Reduce technical debt, improve maintainability.
**Estimated Effort:** 6–8 hours
**Dependency:** None (can be done in parallel with Phases 2–4)

### 5.1 [MEDIUM] Replace Magic Numbers with Schema Constants

**Source:** G1, G3, G4, G6, G7
**Files:** Multiple
**Pattern:** `*12.0` / `*8.0` / hardcoded dimensions scattered across codebase.
**Fix:**
- Import `NODE_FEATURE_DIM`, `NUM_NODE_TYPES`, `NUM_EDGE_TYPES` from `graph_schema.py`
- Replace all `12.0` and `8.0` magic numbers with schema constants
- Add runtime assertions that `NODE_FEATURE_DIM` matches actual data

**Effort:** 2 hours

### 5.2 [MEDIUM] Empty `__init__.py` Files — Add Convenience Exports

**Source:** G7#9, G6#__init__
**Files:** `ml/src/utils/__init__.py`, `ml/src/inference/__init__.py`, `ml/src/datasets/__init__.py`, `ml/src/models/__init__.py`
**Fix:** Add docstrings and re-export key symbols.

**Effort:** 1 hour

### 5.3 [MEDIUM] Delete Redundant `create_label_index.py`

**Source:** G2#9
**File:** `ml/scripts/create_label_index.py`
**Fix:** Verify no external dependencies, then delete. Add README note about removal.

**Effort:** 30 minutes

### 5.4 [MEDIUM] Hash Utils — Relative Path Portability

**Source:** G7#7
**File:** `ml/src/utils/hash_utils.py`
**Bug:** `get_contract_hash()` hashes full absolute path → same contract gets different hash on different machines.
**Fix:** Hash relative path from repo root, use forward slashes for cross-platform consistency.

**Effort:** 1 hour

### 5.5 [LOW] Feature Schema Version Bump & Validation

**Source:** G1 (FEATURE_SCHEMA_VERSION)
**Files:** `graph_schema.py`, `tokenizer.py`
**Fix:**
- Bump `FEATURE_SCHEMA_VERSION` to reflect v5 changes
- Add version check in `DualPathDataset.__init__`

**Effort:** 30 minutes

### 5.6 [LOW] Validate Graph Dataset Enhancement

**Source:** G2#21, G2#22
**File:** `ml/scripts/validate_graph_dataset.py`
**Fix:** Add NaN/Inf checks, edge semantic validity, disconnected component detection.

**Effort:** 1 hour

---

## Phase 6 — Re-Extraction, Augmentation & Retrain v5.1

**Goal:** Apply all fixes to data and retrain the model.
**Estimated Effort:** 2 days work + 12 hrs training runtime
**Dependency:** ALL previous phases complete

### 6.1 Re-Extract All 68K Graphs

**Source:** v5.1 Plan Phase 1
**Steps:**
1. Verify Phase 0/1 fixes on known ghost contracts first:
   ```bash
   # Test 3 known ghost contracts
   for name in 11_external_bug.sol 13_multilabel_complex.sol 20_unused_return_minimal.sol; do
       src=$(cat ml/scripts/test_contracts/$name)
       # Run through extractor, expect nodes >= 10, edges >= 5
   done
   ```
2. Run `validate_graph_dataset.py` for baseline
3. Full re-extraction: `python -m ml.src.preprocessing.batch_extractor --force --output ml/data/graphs/`
4. Run `validate_graph_dataset.py` again — gate: PASS ≥ 95%, ghost ≤ 1%

**Effort:** 3 hours runtime + 30 min verification

### 6.2 Contrastive CEI Pair Injection (~50 pairs)

**Source:** v5.1 Plan §2a
**Steps:**
1. Create 50 call-before-write (Reentrancy) and write-before-call (safe) minimal pairs
2. Verify each pair's graph structure (same node set, different CFG edge ordering)
3. Extract and add to `multilabel_index.csv` train split only
4. Run `create_splits.py --freeze-val-test`

**Effort:** 2–3 hours

### 6.3 DoS Augmentation (+300 contracts)

**Source:** v5.1 Plan §2b
**Steps:**
1. Source from SmartBugs curated (SWC-128), template unbounded-loop variants
2. Label all with Slither; only add confirmed findings
3. Add to training split only

**Effort:** Half-day

### 6.4 Recompute pos_weight

**Source:** v5.1 Plan §2c
**Steps:**
```bash
python ml/scripts/build_multilabel_index.py  # regenerate from updated labels
# Verify distribution and recompute pos_weight tensor
# pos_weight formula: sqrt(neg_count / pos_count) per class
```

**Effort:** 30 minutes

### 6.5 Smoke Run v5.1 (2 epochs, 10% data)

**Source:** v5.1 Plan Phase 3
**Gates (ALL must pass before full run):**

| Gate | Check | Pass Criteria |
|------|-------|---------------|
| GNN eye alive | B2900 epoch 1 gnn share | ≥ 15% |
| Ghost contracts resolved | Test contracts 11/13/20 | nodes ≥ 10, edges ≥ 5 |
| No OOM | GPU memory | No CUDA OOM |

**Effort:** 30 min setup + 20 min runtime

### 6.6 Full Retrain v5.1 (60 epochs)

**Source:** v5.1 Plan Phase 3
**Command:**
```bash
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 nohup python ml/scripts/train.py \
  --run-name v5.1-full \
  --experiment-name sentinel-v5.1 \
  --epochs 60 \
  --batch-size 16 \
  --lr 2e-4 \
  --lora-r 16 --lora-alpha 32 \
  --gnn-hidden-dim 128 --gnn-layers 4 --gnn-heads 8 \
  --gnn-dropout 0.2 \
  --gnn-aux-weight 0.3 \
  --warmup-pct 0.06 \
  --early-stop-patience 10 \
  > ml/logs/train_v5.1_full.log 2>&1 &
```

**Effort:** ~12 hours runtime

### 6.7 Validate v5.1

**Source:** v5.1 Plan Phase 4
**Steps:**
1. `tune_threshold.py` on best checkpoint
2. `manual_test.py` on all 20 behavioral contracts
3. Check all 6 acceptance gates:

| Gate | v5.1 Target |
|------|-------------|
| F1-macro tuned | >0.60 |
| DoS tuned F1 | >0.45 |
| Behavioral detection | >70% |
| Behavioral safe specificity | >66% |
| Mandatory A (reentrancy) | detected |
| Mandatory B (CEI-safe) | clean |

4. `promote_model.py` if gates cleared
5. DVC commit + git push

**Effort:** 1–2 hours

---

## Cross-Phase Dependencies

```
Phase 0 (Security) ──→ Phase 1 (Architecture) ──→ Phase 6 (Retrain)
       │                      │
       └──→ Phase 2 (Data) ──┘
       │
       └──→ Phase 3 (Inference)
       │
       └──→ Phase 4 (Training Hardening) ──→ Phase 6
       │
       └──→ Phase 5 (Code Quality) [parallel with 2-4]
```

**Critical Path:** Phase 0 → Phase 1 → Phase 6 (shortest path to a working model)

---

## Consolidated Findings Count by Severity

| Severity | Count | Phases |
|----------|-------|--------|
| CRITICAL | 7 | Phase 0 (5), Phase 1 (2) |
| HIGH | 14 | Phase 1 (1), Phase 2 (4), Phase 3 (3), Phase 4 (5), Phase 6 (1) |
| MEDIUM | 12 | Phase 2 (2), Phase 3 (4), Phase 4 (4), Phase 5 (4) |
| LOW | 6 | Phase 3 (1), Phase 4 (1), Phase 5 (2), cross-cutting (2) |

---

## Effort Summary by Phase

| Phase | Description | Effort | Can Parallel? |
|-------|-------------|--------|---------------|
| 0 | Security & Critical Bugs | 4–5 hrs | No (blocker) |
| 1 | v5.1 Architecture Fixes | 4 hrs | After Phase 0 |
| 2 | Data Pipeline Fixes | 6–8 hrs | After Phase 0, overlaps with 1 |
| 3 | Inference & API Fixes | 6–8 hrs | After Phase 0, overlaps with 1-2 |
| 4 | Training Hardening | 10–12 hrs | After Phase 0, overlaps with 2-3 |
| 5 | Code Quality | 6–8 hrs | Fully parallel with 2-4 |
| 6 | Re-extract & Retrain | 2 days + 12 hrs | After ALL phases |
| **Total** | | **~55–70 hrs + 12 hrs training** | |

---

## Recommended Execution Order

### Day 1: Phases 0 + 5 (parallel where possible)
1. **Morning:** Phase 0.1 (weights_only sweep) — the single highest-impact security fix
2. **Midday:** Phase 0.2–0.6 (remaining critical bugs)
3. **Afternoon:** Phase 5 items (can be done by a second person while Phase 0 is tested)

### Day 2: Phases 1 + 2 + 3
1. **Morning:** Phase 1.1 (_select_contract) + Phase 1.2 (function-level pooling)
2. **Afternoon:** Phase 2.1–2.6 (data pipeline), start Phase 3.1–3.2

### Day 3: Phases 3 + 4
1. **Morning:** Complete Phase 3 (inference fixes)
2. **Afternoon:** Phase 4 (training hardening)

### Day 4: Phase 6 — Re-extraction
1. Re-extract graphs (3 hrs runtime)
2. Create contrastive pairs
3. Run smoke test

### Day 5: Phase 6 — Retrain + Validate
1. Launch full 60-epoch training
2. Validate results when complete
3. Promote if gates cleared

---

## Verification Checkpoints

| Checkpoint | After Phase | Test | Pass Criteria |
|------------|-------------|------|---------------|
| Security sweep | 0 | `rg "weights_only=False" ml/` | Zero matches |
| Ghost graph fix | 1 | Run 3 known ghost contracts through extractor | nodes ≥ 10, edges ≥ 5 |
| Pooling fix | 1 | Smoke run epoch 1 gnn share | ≥ 15% |
| Dedup wired | 2 | Train 1 epoch with `--use-dedup` | Data loads, no errors |
| API schema | 3 | `curl /predict` with test contract | HTTP 200, valid JSON |
| Checkpoint versioning | 4 | Resume from old checkpoint | Migrates successfully |
| Full validation | 6 | Behavioral test suite | >70% detection, >66% specificity |

---

## Files Change Map

| File | Phases Touching It | Key Changes |
|------|--------------------|-------------|
| `graph_extractor.py` | 0.2, 1.1, 1.4 | type_id fix, interface filter, CFG counter |
| `sentinel_model.py` | 1.2 | Function-level pooling, node_types param |
| `train.py` | 1.3, 0.5, 4.4 | aux_weight, --use-dedup, preflight |
| `trainer.py` | 1.3, 4.1–4.5, 4.8–4.9 | aux_weight, checkpoint versioning, CLASS_NAMES, grad accum, config validation |
| `dual_path_dataset.py` | 0.1, 2.5, 2.2 | weights_only, feature validation, schema version |
| `cache.py` | 0.1, 0.4, 3.6 | weights_only, dim check, LRU eviction |
| `predictor.py` | 0.1, 3.1, 3.3, 3.7 | weights_only, windowed cache, batch reuse, metrics |
| `api.py` | 0.3, 3.5 | threshold→thresholds, rate limiting |
| `drift_detector.py` | 3.2 | integrity check, min samples |
| `compute_drift_baseline.py` | 0.1, 3.2 | weights_only, min samples, --force |
| `focalloss.py` | 4.6, 4.7 | per-class alpha, epsilon clipping |
| `create_splits.py` | 2.3, 2.6 | shared stratification, distribution validation |
| `dedup_multilabel_index.py` | 2.4 | orphan row handling |
| `tokenizer.py` | 2.2 | strip extra keys on save |
| `validate_graph_dataset.py` | 0.1, 2.5 | weights_only, enhanced validation |
| `tune_threshold.py` | 0.1 | weights_only |
| `promote_model.py` | 0.1 | weights_only |
| `graph_schema.py` | 5.1, 5.5 | schema constants, version bump |
| `hash_utils.py` | 5.4 | relative path hashing |
| All `__init__.py` | 5.2 | Add convenience exports |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| `weights_only=True` breaks LoRA checkpoint loading | Medium | High | Test with actual v5 checkpoint before sweeping; if fails, add hash validation as alternative |
| Function-level pooling degrades val F1 | Low | Medium | Run smoke test first; fallback to hybrid pool if F1 drops >5pp by epoch 10 |
| Re-extraction corrupts existing valid graphs | Low | High | Backup graph hashes before force-re-extract; validate before and after |
| DoS gate still not reached with +300 samples | Medium | Medium | Gate relaxed to >0.45; document as known limitation |
| GNN share still <15% after pooling fix | Low | Critical | Investigate grad normalisation, weight init, LoRA interference before full run |
