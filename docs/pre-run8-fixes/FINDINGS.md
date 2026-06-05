# Pre-Run-8 Deep Investigation — Findings

**Date:** 2026-06-04  
**Source files audited:**
- `ml/src/models/gnn_encoder.py` (628 lines)
- `ml/src/models/sentinel_model.py` (646 lines)
- `ml/src/training/trainer.py` (2051 lines)
- `ml/scripts/train.py` (337 lines)
- `ml/src/inference/predictor.py` (339+ lines)
- `ml/src/training/training_logger.py` lines 295–316

---

## F1 — CRITICAL BLOCKER: `--drop-complexity-feature` NOT IMPLEMENTED

**Status:** 🐛 Bug (missing implementation)  
**Files affected:** gnn_encoder.py, sentinel_model.py, trainer.py, train.py, predictor.py  
**Risk if not fixed:** Run 8 cannot launch with the intended core change

### What needs to exist

| File | Change needed |
|------|--------------|
| `gnn_encoder.py:GNNEncoder.__init__` | Add `drop_complexity: bool = False` parameter |
| `gnn_encoder.py:GNNEncoder.forward` | Zero-out `x[:, 5]` when `drop_complexity=True` |
| `sentinel_model.py:SentinelModel.__init__` | Accept `drop_complexity_feature: bool = False`, pass to GNNEncoder |
| `trainer.py:TrainConfig` | Add `drop_complexity_feature: bool = False` field |
| `train.py` | Add `--drop-complexity-feature` argparse flag, wire to TrainConfig |
| `predictor.py` | Read `drop_complexity_feature` from `saved_cfg` when reconstructing SentinelModel |

### Critical implementation detail — WHERE to zero feat[5]

In `gnn_encoder.py:forward()`, the correct insertion point is **after** the dtype normalization (line ~425) and **before** the type embedding concat (line ~552):

```python
# After: if x.dtype != self._param_dtype: x = x.to(self._param_dtype)
if self.drop_complexity:
    x = x.clone()   # MUST clone — .to() returns original tensor if dtype already matches
    x[:, 5] = 0.0
```

The `.clone()` is essential because if `x.dtype == self._param_dtype`, the `.to()` call returns the original tensor (not a copy). Zeroing in-place would then corrupt the cached graph data for all future batches using the same sample.

**Why feat[5]?** `graph_schema.py` FEATURE_SCHEMA maps feat[5] to `complexity` = `log1p(cfg_block_count)/log1p(100)`, normalized [0,1]. L4 experiment confirmed this single feature dominates gradient signal at 34–36% share for ALL 10 classes — the model learned "more complex = more vulnerable" as a complexity-proxy instead of structural reasoning.

### Why the predictor fix matters

If `drop_complexity_feature=True` at training time and the predictor reconstructs `SentinelModel` without that flag, then:
- Training: feat[5] = 0 (model learned with this constraint)
- Inference: feat[5] = real value (e.g., 0.7 for a complex contract)

The model sees input distribution it never trained on. The bias formerly captured by feat[5] weight is now "homeless" and will manifest as unpredictable output shifts. The predictor fix is one line:
```python
# In predictor.py, SentinelModel() constructor call:
drop_complexity_feature=saved_cfg.get("drop_complexity_feature", False),
```

---

## F2 — `use_weighted_sampler` Default Mismatch

**Status:** ⚠️ Correctness issue  
**Files:** `trainer.py:376`, `train.py:224`

```python
# trainer.py TrainConfig dataclass default:
use_weighted_sampler: str = "positive"   # ← "positive"

# train.py argparse default:
p.add_argument("--weighted-sampler", default="timestamp-size", ...)   # ← "timestamp-size"
```

**Impact:**  
- CLI training (`python train.py`) → uses "timestamp-size" (correct for Run 7+)
- Direct `TrainConfig()` instantiation (tests, notebooks, smoke scripts) → uses "positive"
- This means any test that uses `TrainConfig()` directly trains with a different sampler than production

**Why it matters:**  
"timestamp-size" oversamples large Timestamp+ contracts (4×) and undersamples large negatives (0.5×). "positive" just oversamples any-vulnerability rows 3×. They produce different training dynamics and gradient flows. A smoke run using `TrainConfig()` directly will not reproduce the production setup.

**Fix:** Change `TrainConfig` default to match argparse:
```python
use_weighted_sampler: str = "timestamp-size"  # align with train.py argparse default
```

---

## F3 — `aux_cei_loss_weight` Argparse Arg Silently Discarded

**Status:** 💡 Improvement (silent dead arg)  
**Files:** `train.py:196–201`, `trainer.py` (not present)

```python
# train.py parses this:
p.add_argument("--aux-cei-loss-weight", type=float, default=0.0,
               dest="aux_cei_loss_weight", ...)

# train.py TrainConfig constructor call — this is NOT passed:
config = TrainConfig(
    run_name = ...,
    ...  # aux_cei_loss_weight is absent from this call
)
```

**Impact:**  
If a user passes `--aux-cei-loss-weight 0.1`, the value is parsed, stored in `args.aux_cei_loss_weight`, and then silently discarded. `TrainConfig` doesn't have a `aux_cei_loss_weight` field and doesn't receive the value. The default of 0.0 always wins.

**Fix options:**
1. Remove the argparse argument (cleanest — it's documented as "activate only after Phase 7")
2. Add `aux_cei_loss_weight: float = 0.0` to TrainConfig and wire it

Since RC7 CEI supervision is deferred, option 1 is cleaner for now. But the arg exists as a placeholder, so it should at minimum have a note in the help text saying it's currently non-functional.

---

## F4 — BUG-SL-1 Fix Confirmed ✅

**Status:** ✅ Fixed (confirmed in code)  
**File:** `ml/src/training/training_logger.py:304`

The fix is in place:
```python
def check_aux_head(self, model, epoch):
    head = getattr(model, "aux_phase2", None)
    if head is None:
        return result
    head = getattr(head, "_orig_mod", head)  # ← fix line 304
    final_linear = head[-1]
```

After `torch.compile`, `model.aux_phase2` is an `OptimizedModule` (wraps `nn.Sequential`). `OptimizedModule` doesn't support subscript indexing (`[-1]`). Without the fix, `head[-1]` silently raised a `TypeError` that was caught by the outer `try/except` block in the epoch loop, causing all structured epoch data to be empty for all 40 Run 7 epochs.

The `_orig_mod` unwrap gives back the original `nn.Sequential`, which does support `[-1]`.

---

## F5 — BUG-C4 Quantified

**Status:** ⚠️ Confirmed (VRAM impact assessed)  
**Data source:** Direct count from `ml/data/cached_dataset_v10.pkl`

| Metric | Value |
|--------|-------|
| Total graphs | 41,577 |
| Graphs >1024 nodes | **227 (0.55%)** |
| Graphs >2048 nodes | **0 (0.00%)** |
| Maximum graph size | **1,735 nodes** |
| P99 node count | 625 nodes |
| P95 node count | 322 nodes |
| Top 10 sizes | 1735, 1567, 1557, 1479, 1469, 1442, 1430, 1430, 1429, 1414 |

**Verdict:** `fusion_max_nodes=2048` covers 100% of v10 contracts. No graph exceeds 2048.

**Decision for Run 8:**  
The docs/training/RUN8-ULTRACODE.md mentions a VRAM gate test before enabling 2048. The gate requires: `worst-case max_nodes=2048, batch=16, full backward+step < 7.5 GB on RTX 3070`.

At training batch_size=8 (not 16), the risk is lower. Recommendation:
- Set `--fusion-max-nodes 2048` to fix the 0.55% truncation
- If VRAM OOM occurs, fall back to 1536 (still covers all graphs)
- Do NOT set >1735 (unnecessary) but 2048 is the clean power-of-2 choice

---

## F6 — `gnn_to_bert_proj` + `prefix_type_embedding` Missing from torch.compile List

**Status:** 💡 Performance miss  
**File:** `trainer.py:1410–1415`

The compile list:
```python
for name in ("gnn", "fusion", "classifier",
             "gnn_eye_proj", "cfg_eye_proj", "transformer_eye_proj", "window_pooler",
             "aux_gnn", "aux_transformer", "aux_fused", "aux_phase2"):
```

`gnn_to_bert_proj` and `prefix_type_embedding` are NOT in this list. For Run 7 with `gnn_prefix_k=0`, these modules don't exist — no issue. For Run 8 with `--gnn-prefix-k 48`, these modules will exist and run in eager mode (unccompiled).

**Impact:** The prefix projection (Linear 256→768) runs eager. Minor performance overhead (~5–10% per epoch for the prefix path). Not a correctness issue.

**Fix:** Add to the compile list, gated by `if config.gnn_prefix_k > 0`:
```python
if config.gnn_prefix_k > 0:
    for name in ("gnn_to_bert_proj", "prefix_type_embedding"):
        sub = getattr(model, name, None)
        if sub is not None:
            setattr(model, name, torch.compile(sub, dynamic=True))
```

---

## F7 — JK `_orig_mod` Access Through OptimizedModule (Works Correctly)

**Status:** ✅ No bug found (clarification)

The trainer accesses `model.gnn.jk.last_weights` (line 1761). After `torch.compile`, `model.gnn` is an `OptimizedModule`. `OptimizedModule.__getattr__` forwards attribute lookups to `_orig_mod`. So:
- `model.gnn.jk` → `model.gnn._orig_mod.jk` → the original `_JKAttention` module ✅
- `model.gnn.jk.last_weights` → the registered buffer ✅

This is different from BUG-SL-1 which used subscript indexing (`head[-1]`). Attribute access works fine through `OptimizedModule`.

---

## F8 — Phase 2 GNN Assertion at Import Time

**Status:** ✅ Robust defensive guard  
**File:** `sentinel_model.py:75`

```python
assert _MAX_TYPE_ID == 12.0, (
    f"_MAX_TYPE_ID is {_MAX_TYPE_ID} but expected 12.0. ..."
)
```

This assertion fires at module IMPORT if anyone adds a node type to `NODE_TYPES` without updating the normalisation divisor. Guards against a subtle silent bug where the type recovery in `forward()` (line 443: `(graphs.x[:, 0].float() * _MAX_TYPE_ID).round().long()`) would silently misalign type IDs.

---

## F9 — `label_smoothing` Dead Code Path

**Status:** 💡 Dead code  
**File:** `trainer.py:648–650`

```python
if class_eps is not None:
    labels = labels * (1.0 - class_eps) + 0.5 * class_eps
elif label_smoothing > 0.0:          # ← dead: class_eps is always non-None
    labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
```

`_class_eps` is always built from `config.class_label_smoothing` (which has non-zero defaults for all classes) and always passed as `class_eps`. The `elif` branch is never reached. The `label_smoothing` parameter in `train_one_epoch` serves no purpose — it's overridden by `class_eps`.

Not harmful (the correct path runs). But the dead code is misleading: passing `--label-smoothing 0.1` via CLI would appear to work but actually do nothing.

---

## F10 — Calibration Pipeline Gap

**Status:** ⚠️ Pre-run action needed  
**Folder:** `ml/calibration/`

Contents: `temperatures_run4.json`, `temperatures_run4_ece_comparison.png`, `temperatures_run4_stats.json`

There is **no `temperatures_run7.json`** and **no `calibrate_thresholds.py`**. The available script is:
- `ml/scripts/calibrate_temperature.py` — temperature scaling
- `ml/scripts/tune_threshold.py` — per-class threshold sweep

For Run 7 pre-run calibration, the needed step is running `tune_threshold.py` against the Run 7 ep39 checkpoint to get per-class thresholds. The trainer already runs threshold tuning every 10 epochs (and at the final epoch), so `_cached_tuned_thresholds` is stored in the checkpoint at ep40 (the last computed tuning epoch before run stop at ep41). This means the Run 7 checkpoint already contains tuned thresholds — they can be extracted directly without re-running the sweep.

**How to extract:** Load `ml/checkpoints/GCB-P1-Run7-v10-20260603_best.pt` and read `ckpt["tuned_thresholds"]`.

---

## F11 — `_is_final_epoch` Heuristic (Minor Logic Issue)

**Status:** 💡 Minor (conservative false-trigger)  
**File:** `trainer.py:1731`

```python
_is_final_epoch = (epoch == config.epochs) or (patience_counter + 1 >= config.early_stop_patience)
```

This is checked BEFORE `patience_counter` is updated at end of epoch. If `patience_counter + 1 >= early_stop_patience`, the threshold sweep runs. But if the current epoch DOES improve (resets patience to 0), the sweep was unnecessary (though harmless). Conversely, if the epoch doesn't improve, the sweep correctly runs before the stop.

Not a correctness bug — threshold tuning is idempotent and the cost (19×10 eval) is acceptable. The conservative trigger is intentional.

---

## F12 — `train.py` Docstring Stale

**Status:** 💡 Documentation drift  
**File:** `train.py:6`

The module docstring says `"v7 three-eye + JK, 7-layer GNN, LoRA"`. The actual architecture is 4-eye, 8-layer GNN, JK attention, LoRA. The docstring examples also show `--experiment-name sentinel-v7` which would mislabel Run 8 MLflow experiments.

---

## F13 — Phase 3 Downward CONTAINS Pass Is Correct

**Status:** ✅ Architecture verified  
**File:** `gnn_encoder.py:607`

The IMP-G3 downward pass (Layer 8, conv4c) uses `fwd_contains_ei` (original FUNCTION→CFG direction) with type-5 (CONTAINS) embeddings. This is correct: after two upward hops lifting CFG signal to FUNCTION level, the downward pass distributes the enriched FUNCTION context back to CFG children. The result is that ALL nodes have Phase 3 context before CrossAttentionFusion.

---

## F14 — Phase 2 Sub-masks Are Correct (NF-6 Fix Confirmed)

**Status:** ✅ No bug  
**File:** `gnn_encoder.py:504–517`

The `_cf_mask` and `_icfg_mask` sub-masks are applied to `phase2_raw` (integer type IDs from the already-filtered `cfg_mask` slice), NOT to the embedded `phase2_ea` tensor. This is correct — comparing float embeddings to integer type IDs would produce wrong results. The NF-6 fix is working.

---

## F15 — `import math` Inside Training Loop

**Status:** 💡 Minor performance  
**File:** `trainer.py:723`

```python
if jk_entropy_reg_lambda > 0.0:
    _jk_ent = aux.get("jk_entropy")...
    if _jk_ent is not None:
        import math   # ← inside inner loop
        _H_max = math.log(3)
```

`import math` inside the training loop is called tens of thousands of times per run. Python caches module lookups so it's O(1) dictionary access, not a real file import — but it's still unnecessary overhead. `math` should be imported at module level. `_H_max = math.log(3)` is a constant and can be precomputed once.

---

## Summary Table

| ID | Finding | Severity | Status | Action Required |
|----|---------|---------|--------|----------------|
| F1 | `--drop-complexity-feature` not implemented | 🔴 CRITICAL | 🐛 Bug | Implement in 5 files |
| F2 | `use_weighted_sampler` default mismatch | 🟡 Medium | ⚠️ Issue | Fix TrainConfig default |
| F3 | `aux_cei_loss_weight` arg silently discarded | 🟡 Medium | 💡 Dead code | Remove or wire |
| F4 | BUG-SL-1 fix confirmed | ✅ Done | ✅ | No action |
| F5 | BUG-C4: 227 graphs (0.55%) >1024 nodes, 0 >2048 | 🟡 Medium | ⚠️ Data | Set `--fusion-max-nodes 2048` |
| F6 | `gnn_to_bert_proj` not in torch.compile | 🟢 Low | 💡 Perf | Add when `gnn_prefix_k > 0` |
| F7 | JK `_orig_mod` access correct | ✅ No bug | ✅ | No action |
| F8 | `_MAX_TYPE_ID` assert correct | ✅ No bug | ✅ | No action |
| F9 | `label_smoothing` is dead code | 🟢 Low | 💡 Dead | Note only |
| F10 | No `temperatures_run7.json` | 🟡 Medium | ⚠️ Missing | Extract from checkpoint |
| F11 | `_is_final_epoch` is conservative | 🟢 Low | 💡 Minor | No action |
| F12 | `train.py` docstring stale | 🟢 Low | 💡 Docs | Update |
| F13 | IMP-G3 downward pass is correct | ✅ No bug | ✅ | No action |
| F14 | NF-6 sub-mask fix is correct | ✅ No bug | ✅ | No action |
| F15 | `import math` inside loop | 🟢 Low | 💡 Perf | Move to module level |

---

## Implementation Order for Run 8 Launch

1. **F1**: Implement `--drop-complexity-feature` (all 5 files) — BLOCKER
2. **F5**: Decide on `--fusion-max-nodes 2048` (no code change — just set the flag)
3. **F2**: Fix `use_weighted_sampler` default in TrainConfig
4. **F3**: Fix `aux_cei_loss_weight` dead arg
5. **F6**: Add prefix modules to compile list (low priority)
6. **F10**: Extract Run 7 thresholds from checkpoint
7. **F15**: Move `import math` to module level
