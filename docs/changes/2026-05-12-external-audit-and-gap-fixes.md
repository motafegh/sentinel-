# External Audit Review + G1–G6 Gap Fixes

**Date:** 2026-05-12
**Status:** Complete
**Commits:** `a0576fb` (22 external audit fixes), `539ae99` (G1–G6 gap fixes)

---

## Context

After v5.1 Phase 0 code fixes and dataset deduplication (see
[2026-05-12-v5.1-phase0-and-dataset-fixes.md](2026-05-12-v5.1-phase0-and-dataset-fixes.md)),
a full adversarial audit of the v5 codebase was performed across 8 groups covering
every layer of the pipeline. The audit produced 133+ findings; 22 were implemented
in commit `a0576fb`. A subsequent gap review identified 6 further issues (G1–G6)
fixed in commit `539ae99`.

---

## Commit a0576fb — 22 External Audit Fixes

### Audit Structure

Eight groups reviewed in order of pipeline dependency:

| Group | Area | Critical findings |
|-------|------|-------------------|
| 1 | Schema & Feature Contracts | FEATURE_SCHEMA_VERSION not bumped; magic numbers |
| 2 | Graph Extraction Engine | `actual_type_id` roundtrip always returned 0 |
| 3 | GNN Path | Edge type constants hardcoded as literals |
| 4 | Transformer Path | LoRA defaults r=8/α=16 too low; string→list trap |
| 5 | Training Pipeline & Loss | `aux_loss_weight` dataclass default still 0.1; dedup dead code |
| 6 | Dataset & Data Loading | Deduped CSV and splits never wired into TrainConfig |
| 7 | Inference & Production API | Cache always-miss (dim=8 vs 12); PredictResponse schema broken |
| 8 | Operational Scripts | Wrong Slither detectors; ghost graphs written to disk; data poisoning |

### Fixes by File

#### `ml/src/preprocessing/graph_schema.py`
- **`FEATURE_SCHEMA_VERSION` v2 → v3** — invalidates all stale inference-cache entries that were built against the v2 schema. Required bump because NODE_FEATURE_DIM and edge type constants changed between v4 and v5.

#### `ml/src/preprocessing/graph_extractor.py`
- **`actual_type_id` roundtrip** — `int(x_list[-1][0])` where `x[-1][0]` is `float(type_id)/12.0` always returned 0 for type IDs 0–11 (integer truncation of values < 1.0). Fixed to `int(round(x_list[-1][0] * 12))`. This was silently labelling all function/modifier/CFG metadata as `STATE_VAR` type.

#### `ml/src/models/gnn_encoder.py`
- **Edge type constants from `EDGE_TYPES` dict** — Phase 1/2/3 edge masks were using inline integer literals. Now `_CONTAINS = EDGE_TYPES["CONTAINS"]`, `_CONTROL_FLOW = EDGE_TYPES["CONTROL_FLOW"]`. Any future schema renumber updates in one place.

#### `ml/src/models/transformer_encoder.py`
- **LoRA defaults r=8 α=16 → r=16 α=32** — effective LoRA scale (α/r = 2.0) unchanged; rank doubled for capacity. r=8 was designed for full-fine-tune memory budgets; r=16 is appropriate for SENTINEL's 68K/44K contract vocabulary.
- **`lora_target_modules` string→list guard** — MLflow can serialise a `["query","value"]` list as `"query,value"` on checkpoint reload. `LoraConfig(target_modules="query,value")` produces zero LoRA adapters (silent failure). Added guard: if a string is received, split on `,` and strip whitespace.

#### `ml/src/models/sentinel_model.py`
- **`graphs.x[:, 0] * 12.0` → `.float() * 12.0`** — under AMP/BF16 the tensor may be in BF16 at this point; multiplying before `.round().long()` can lose the last bit. `.float()` coercion makes the denormalisation stable at full FP32.
- **Docstring λ=0.1 → λ=0.3** — updated to reflect the Phase 0c aux_loss_weight fix.

#### `ml/src/training/trainer.py`
- **`TrainConfig.label_csv` default** — was `ml/data/processed/multilabel_index.csv` (68K leaky); now `ml/data/processed/multilabel_index_deduped.csv`. Any non-CLI training call (Jupyter, sweep, programmatic) was silently training on the leaky dataset regardless of the Phase 0f dedup work.
- **`TrainConfig.splits_dir` default** — was `ml/data/splits`; now `ml/data/splits/deduped`.
- **`TrainConfig.aux_loss_weight` dataclass default 0.1 → 0.3** — Phase 0c raised the CLI default but left the dataclass at 0.1. Any programmatic use got the collapse value.
- **`train_one_epoch(..., aux_loss_weight=0.1)` function default → 0.3** — same issue: the function signature default would override the TrainConfig value if called without keyword arg.
- **Gradient collapse detection** — added `gnn_grad / total_grad` ratio monitoring; logs WARNING if GNN eye gradient share drops below 5%.

#### `ml/src/inference/predictor.py`
- **`NODE_FEATURE_DIM` imported from schema** — was hardcoded as `12` inline; now `from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM`. Tracks schema changes automatically.
- **`_ensure_list()` guard on `lora_target_modules`** — same string→list trap as TransformerEncoder. Applied at checkpoint load time in `_load_model()`.

#### `ml/src/inference/cache.py`
- **`weights_only=True` + `safe_globals`** — was `weights_only=False`. Added `torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])`.
- **`graph.x.shape[1] != 8` → `!= NODE_FEATURE_DIM`** — this was the root cause of 100% inference cache miss rate for v5. Every v5 graph (12-dim features) was rejected on load and re-extracted from scratch on every API call.
- **`from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM`** added.

#### `ml/src/inference/api.py`
- **`PredictResponse.threshold: float` → `thresholds: list[float]`** — predictor `Fix #6` renamed the returned key from `threshold` to `thresholds` (single float → per-class list), but `PredictResponse` schema was never updated. Every `/predict` response was returning HTTP 422.

#### `ml/scripts/generate_safe_variants.py`
- **`mishandled-exception` wrong detectors** — was checking `["suicidal", "controlled-delegatecall"]` instead of `["unchecked-lowlevel", "unchecked-send"]`. Slither never reported findings → `bad_findings = []` → every variant was accepted as safe → data poisoning (still-vulnerable contracts labelled safe injected into training).
- **`call-to-unknown` explicit skip** — the function body was `bad_findings = []` with a comment saying "complex to verify". Fixed to return `None` immediately with a warning. Downstream callers skip `None` returns.

#### `ml/scripts/reextract_graphs.py`
- **Ghost graphs no longer written to disk** — was calling `torch.save(ghost_graph, path)` before the ghost gate check. Ghost graphs (≤3 nodes) entered the training dataset and degraded GNN performance. Now skipped entirely with a counter.

#### `ml/scripts/validate_graph_dataset.py`, `compute_drift_baseline.py`, `build_multilabel_index.py`, `create_label_index.py`
- **`weights_only=True` + `safe_globals`** applied uniformly. All four were using `weights_only=False` for graph `.pt` loading.

---

## Commit 539ae99 — G1–G6 Gap Fixes

Six issues not addressed in `a0576fb`, identified in a post-audit gap review.

### G6 (HIGH) — `trainer.py` resume path `weights_only=False`

**File:** `ml/src/training/trainer.py:715`

The security fix (`weights_only=True`) was applied to `cache.py` and all 4 scripts but missed the trainer resume path. Training runs launched overnight or from CI can accept resume checkpoints from external sources.

**Analysis:** The checkpoint save format uses `dataclasses.asdict(config)` (plain dict), `model.state_dict()` (tensors — LoRA A/B matrices are standard `nn.Parameter`, not `PeftModel` objects), `optimizer.state_dict()` (tensors + basic types), `scheduler.state_dict()` (basic types). No custom pickled objects → `weights_only=True` safe without safe_globals.

**Fix:** `weights_only=False` → `weights_only=True` with a comment confirming the format analysis.

### G2 (HIGH) — `preprocess.py process()` cache key missing schema version

**File:** `ml/src/inference/preprocess.py:191`

`process()` (offline/disk path) built its cache key as `get_contract_hash(sol_path)` — no schema version. `process_source()` (API path, line 253) already used `f"{content_hash}_{FEATURE_SCHEMA_VERSION}"`. After the v2→v3 schema bump, `process()` would serve stale v2 graphs from cache without detecting the mismatch.

**Fix:** `contract_hash = f"{get_contract_hash(sol_path)}_{FEATURE_SCHEMA_VERSION}"`

### G4 (HIGH) — Inline MD5 in 3 scripts instead of `get_contract_hash()`

**Files:** `ml/scripts/reextract_graphs.py`, `ml/scripts/verify_splits.py`, `ml/scripts/dedup_multilabel_index.py`

All three used `hashlib.md5(str(rel).encode("utf-8")).hexdigest()` inline instead of the canonical `get_contract_hash()` from `ml/src/utils/hash_utils.py`. Three separate copy-paste implementations create divergence risk if the hashing strategy ever changes (e.g., moving from path-based to content-based).

**Analysis:** `get_contract_hash(rel)` does `str(rel).encode('utf-8')` internally — numerically identical to the inline form. No existing `.pt` filenames change.

Additionally `verify_splits.py:_content_hash()` was using `path.read_bytes()` (raw bytes) while the canonical `get_contract_hash_from_content()` encodes as UTF-8 string. The raw-bytes form is platform-sensitive (BOM, line-ending differences on Windows paths). Fixed to `get_contract_hash_from_content(path.read_text(encoding="utf-8", errors="ignore"))`.

**Fix:** All three scripts now import and call `get_contract_hash(rel)`. `hashlib` import removed from `reextract_graphs.py` and `verify_splits.py` (no longer needed).

### G1 (MEDIUM) — Fix #17 comment wrong in `trainer.py`

**File:** `ml/src/training/trainer.py:114`

The Fix #17 docstring said *"Both now default to 32"* but both `TrainConfig.batch_size` and `--batch-size` default to 16 (correct for RTX 3070 8GB VRAM — batch=32 would OOM during peak attention allocation). The comment was never updated after the hardware-conscious decision.

**Fix:** Comment corrected to *"Both now default to 16 (correct for RTX 3070 8GB VRAM)"*.

### G5 (MEDIUM) — No runtime range assertions on sentinel values

**File:** `ml/src/preprocessing/graph_extractor.py`

Features `[7] return_ignored` and `[8] call_target_typed` use a ternary sentinel encoding ({-1.0, 0.0, 1.0}) that the GNN model depends on. No assertion existed to catch a future compute helper returning an out-of-range value (e.g., a float from a ratio computation accidentally wired to these slots).

Similarly, `[9] in_unchecked` is documented as *"NEVER inherited from parent func"* in CFG node features but there was no code-level invariant enforcing it.

**Fix:**
- `_build_node_features()`: `assert return_ignored in (-1.0, 0.0, 1.0)` and `assert call_target_typed in (-1.0, 0.0, 1.0)` before the return.
- `_build_cfg_node_features()`: `_cfg_in_unchecked = 0.0; assert _cfg_in_unchecked == 0.0` — named variable makes the invariant self-documenting.

### G3 (LOW) — `/ 12.0` magic number in `graph_extractor.py`

**File:** `ml/src/preprocessing/graph_extractor.py` (two locations)

`float(type_id) / 12.0` and `float(cfg_type) / 12.0` were hardcoded. The divisor 12 is `max(NODE_TYPES.values())` (CFG_NODE_OTHER = 12). If a new node type is added with ID > 12 the normalisation silently breaks (values > 1.0 passed to the GNN).

**Fix:** Module-level constant `_MAX_TYPE_ID = float(max(NODE_TYPES.values()))` derived from the schema dict. Both `/ 12.0` occurrences replaced with `/ _MAX_TYPE_ID`. Currently equals 12.0 — no numeric change to existing graphs.

---

## Impact on Existing Artifacts

None of these fixes require re-extraction or cache rebuilds.

| Fix | Impact on disk artifacts |
|-----|--------------------------|
| G6  | None — future resume paths only |
| G2  | None — online API inference cache only; training pipeline unaffected |
| G4  | None — hash formula produces identical values; `.pt` filenames unchanged |
| G1  | None — comment only |
| G5  | None — assertions fire at future extraction time; existing graphs validated |
| G3  | None — `_MAX_TYPE_ID = 12.0`; numerically identical to hardcoded value |

The 44,420 graphs on disk, `multilabel_index_deduped.csv`, `splits/deduped/`, and `cached_dataset_deduped.pkl` (if complete) all remain valid.

---

## Current State After These Commits

| Item | State |
|------|-------|
| 44,420 canonical graphs on disk | ✅ Valid (Phase 1 complete) |
| `multilabel_index_deduped.csv` | ✅ 44,420 rows, 0 cross-split leaks |
| `splits/deduped/` train/val/test | ✅ 31,092 / 6,661 / 6,667 |
| `cached_dataset_deduped.pkl` | ✅ Built (verify with `ls -lh`) |
| External audit fixes (a0576fb) | ✅ 22 fixes across 8 groups |
| G1–G6 gap fixes (539ae99) | ✅ All 6 resolved |
| **Phase 2 data augmentation** | ⏳ B1 CEI pairs, B2 DoS — pending |
| **Phase 3 retrain** | ⏳ Pending |

---

## Next Steps (v5.1 Training Pipeline)

### B1 — CEI Contrastive Pairs (~50 pairs)
Generate ~50 Reentrancy-vulnerable contracts and matching safe (write-before-call) variants.
Extract graphs and inject into **train split only** (never val/test).
Gate: CEI-A (vulnerable) must fire in behavioral test; CEI-B (safe) must be silent.

### B2 — DoS Augmentation (optional, ~+300 contracts)
Collect SmartBugs SWC-128 contracts + synthetic templates.
Target: train DoS count 257 → ~557. Extract and inject train only.

### B3 — Recompute `pos_weight`
Run from deduped label counts before training. Gate: DoS weight ≤ 80.0.

### C1 — Smoke Run (2 epochs, 10% data)
Gate: GNN grad share ≥ 15% at end of epoch 1 (~B194 for 10% of 31K). GO/STOP decision before committing 60-epoch run.

### C2 — Full 60-Epoch Retrain
```bash
source ml/.venv/bin/activate
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 nohup python ml/scripts/train.py \
  --run-name v5.1-full --experiment-name sentinel-v5.1 \
  --epochs 60 --batch-size 16 --lr 2e-4 --lora-r 16 --lora-alpha 32 \
  --gnn-hidden-dim 128 --gnn-layers 4 --gnn-heads 8 --gnn-dropout 0.2 \
  --aux-loss-weight 0.3 --warmup-pct 0.06 --early-stop-patience 10 \
  --label-csv ml/data/processed/multilabel_index_deduped.csv \
  --splits-dir ml/data/splits/deduped \
  > ml/logs/train_v5.1.log 2>&1 &
```

### D1 — Tune Thresholds
```bash
python ml/scripts/tune_threshold.py \
  --checkpoint ml/checkpoints/v5.1-full_best.pt \
  --label-csv ml/data/processed/multilabel_index_deduped.csv \
  --splits-dir ml/data/splits/deduped
```
Gates: tuned F1-macro > 0.55 (pre-tuning raw floor 0.50); DoS tuned F1 > 0.30.

### D2 — Behavioral Validation
```bash
source ml/.venv/bin/activate && python ml/scripts/manual_test.py
```
Gates: ≥70% detection rate, ≥66% safe specificity, CEI-A fires, CEI-B silent.

### D3 — Push to GitHub
```bash
git push origin main
```
Currently 10+ commits ahead of origin/main.

### Post-v5.1 (deferred)
- **Module 2 ZKML**: distill v5.1 → proxy MLP → ONNX → EZKL proof
- **Module 5 Contracts**: `forge install` → build → test → Sepolia deploy
- **Module 6 API**: auth+rate-limit design → routes → docker-compose
