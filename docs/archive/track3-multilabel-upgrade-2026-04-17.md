# Track 3 — Multi-Label ML Upgrade Session Log
**Date:** 2026-04-17  
**Branch:** main  
**Status:** Phases 0–7 complete — ready for retrain (Phase 8)

---

## What Changed and Why

### Context
The SENTINEL binary model outputs a single vulnerability score [0,1].
The BCCC-SCsVul-2024 dataset has 12 subfolders: 11 vulnerability types + NonVulnerable.
41.2% of contracts genuinely appear in multiple vulnerability folders (multi-label).
Goal: upgrade to output 10 independent per-class probabilities instead of one binary score.

---

## Phase 0 — Build multilabel_index.csv
**File:** `ml/scripts/build_multilabel_index.py` (NEW)  
**Output:** `ml/data/processed/multilabel_index.csv` (68,555 rows × 11 columns)

**Why:** Graph .pt files store binary labels in `graph.y`. Rather than patching 68K files,
labels are loaded externally from CSV. This keeps .pt files unchanged.

**Two hash systems:**
- BCCC SHA256: content hash of .sol files → folder names in BCCC SourceCodes/
- Internal MD5: path hash → .pt filenames in ml/data/graphs/
- Bridge: `graph.contract_path` → `Path(...).stem` = SHA256

**WeakAccessMod exclusion decision (2026-04-17):**  
1,918 .sol files exist in BCCC SourceCodes/WeakAccessMod/ but ZERO .pt files were extracted
during the original Slither graph-extraction pass. Including a class with zero training examples
would produce NaN gradients and a permanently dead output node.
Decision: drop WeakAccessMod, train with 10 classes not 11.
If WeakAccessMod .pt files are extracted in a future run, add it at index 9 (appended),
rebuild CSV, retrain. Existing indices 0-8 remain valid.

**CSV statistics:**
- Total rows: 68,555
- Multi-label rows (sum > 1): 38,187
- Safe rows (all zeros): 24,456
- Unknown (not in BCCC): 0
- pos_weight range: 0.92 (IntegerUO) to 67.90 (DenialOfService)

**10 output classes (alphabetical, indices 0–9):**
```
0  CallToUnknown
1  DenialOfService
2  ExternalBug
3  GasException
4  IntegerUO
5  MishandledException
6  Reentrancy
7  Timestamp
8  TransactionOrderDependence
9  UnusedReturn
```

---

## Phase 1 — DualPathDataset update
**File:** `ml/src/datasets/dual_path_dataset.py` (MODIFIED)

**Changes:**
- Added `label_csv: Optional[Path] = None` to `__init__`
- When provided: loads CSV into `self._label_map: Dict[str, Tensor[10]]`
- `__getitem__`: two branches — multi-label (from `_label_map`) vs binary (from `graph.y`)
- `dual_path_collate_fn`: multi-label path stacks to `[B, num_classes] float32`; binary path keeps `[B] long`

**Backward compat:** `label_csv=None` restores binary behavior. Old tests/checkpoints still work.

---

## Phase 2 — SentinelModel update
**File:** `ml/src/models/sentinel_model.py` (MODIFIED)

**Changes:**
- Added `num_classes: int = 1` parameter
- Classifier: `nn.Sequential(Linear(64,1), Sigmoid())` → `nn.Linear(fusion_output_dim, num_classes)` (NO Sigmoid)
- `forward()`: raw logits returned; `squeeze(1)` only applied when `num_classes==1`

**Why remove Sigmoid from model:**  
BCEWithLogitsLoss applies sigmoid internally via the log-sum-exp trick.  
External sigmoid before the loss causes underflow on logits > ±15 (binary FocalLoss was safe  
only because it was clipped internally). With 10 output classes the distribution of extreme  
logits is wider — numerically stable path is required.

**Inference:** apply `torch.sigmoid()` explicitly after forward pass (done in predictor.py).

---

## Phase 3 — trainer.py rewrite
**File:** `ml/src/training/trainer.py` (REWRITTEN)

**Changes:**
- `FocalLoss` removed → `nn.BCEWithLogitsLoss(pos_weight=...)` added
- `pos_weight[c] = neg_count[c] / pos_count[c]` computed from training split only (no leakage)
- Labels: `[B, 10] float32` — passed directly to loss, no cast needed
- `evaluate()`: sigmoid → threshold → f1_macro (PRIMARY), f1_micro, hamming_loss, per-class F1 × 10
- `TrainConfig`: `num_classes=10`, `label_csv="ml/data/processed/multilabel_index.csv"`,
  `threshold=0.5`, removed `focal_gamma`/`focal_alpha`
- MLflow experiment: `"sentinel-multilabel"` (separate from `"sentinel-training"`)
- Checkpoint format: unchanged (`{model, optimizer, epoch, best_f1, config}`)
- `CLASS_NAMES` and `NUM_CLASSES = 10` exported as module-level constants (shared by predictor + api)

---

## Phase 4 — predictor.py rewrite
**File:** `ml/src/inference/predictor.py` (REWRITTEN)

**Changes:**
- `__init__`: reads `num_classes` from checkpoint config (defaults to `len(CLASS_NAMES)`)
- Instantiates `SentinelModel(num_classes=num_classes)` — not hardcoded to 1
- `_score()`: applies `torch.sigmoid()` post-forward, builds `vulnerabilities` list
- Return schema:
  ```python
  {
    "label": "vulnerable" | "safe",
    "vulnerabilities": [{"class": "Reentrancy", "probability": 0.81}, ...],  # desc sorted
    "threshold": 0.50,
    "truncated": bool,
    "num_nodes": int,
    "num_edges": int,
  }
  ```
- "safe" = empty vulnerabilities list (no `score` field anymore)
- Binary backward compat: `num_classes=1` → single-entry list keyed `"BinaryScore"`

---

## Phase 5 — api.py update
**File:** `ml/src/inference/api.py` (MODIFIED)

**Changes:**
- Added `VulnerabilityResult(BaseModel)` with `vulnerability_class: str` and `probability: float`
- `PredictResponse`: removed `confidence: float` → added `vulnerabilities: list[VulnerabilityResult]`
- `/predict` endpoint: maps `result["vulnerabilities"]` → `VulnerabilityResult` list
- Default checkpoint updated to `multilabel-v1_best.pt`
- API version bumped: `"1.0.0"` → `"2.0.0"`

---

## Phase 6 — inference_server.py update
**File:** `agents/src/mcp/servers/inference_server.py` (MODIFIED)

**Changes:**
- `_mock_prediction()`: returns Track 3 schema (`label`, `vulnerabilities` list, no `confidence`)
  - Safe contract → `{"label": "safe", "vulnerabilities": [], ...}`
  - Reentrancy contract → `{"label": "vulnerable", "vulnerabilities": [{"vulnerability_class": "Reentrancy", "probability": 0.72}, ...], ...}`
- `_call_inference_api()` docstring: updated to reflect new return schema
- `_handle_predict()` logging: removed `confidence` reference → now logs `detected={N} class(es)`
- A-13 fix confirmed: `"mock": True` key stays absent (was already removed in A-13 fix)

---

## Phase 7 — Tests updated
**Files:** `ml/tests/test_api.py`, `agents/tests/test_inference_server.py` (MODIFIED)

**ml/tests/test_api.py:**
- `test_predict_valid_contract`: now asserts `vulnerabilities` list + entry shape, NOT `confidence`
- Added `test_predict_no_confidence_field`: asserts old `confidence` field is absent (regression guard)
- Added `test_predict_safe_contract`: shape test for safe-path response
- `test_predict_consistent_on_same_input`: compares `vulnerabilities` not `confidence`

**agents/tests/test_inference_server.py:**
- `MOCK_PREDICTION_RESULT`: updated to Track 3 schema
- `test_mock_prediction_*`: assert new schema fields; assert `"mock"` key absent
- All `"risk_score"` assertions → `"label"` or `"vulnerabilities"`
- **Result: 18/18 tests passing**

---

## What Did NOT Change (locked)
- GNNEncoder architecture (3-layer GAT, 8-dim input, 64-dim output)
- TransformerEncoder (frozen CodeBERT CLS token)
- FusionLayer (832→256→64 MLP)
- Graph .pt files (68,555 files — labels loaded externally)
- Token .pt files (68,570 files)
- Data splits (train/val/test indices)
- Checkpoint format keys (`model`, `optimizer`, `epoch`, `best_f1`, `config`)

---

## Remaining Steps

### Phase 8 — Retrain (user-initiated)
```bash
nohup poetry run python ml/scripts/train.py > ml/logs/multilabel_train.log 2>&1 &
# Monitor: tail -f ml/logs/multilabel_train.log
```
Target: val macro-F1 > 0.50 baseline; aim for > 0.65.

### Phase 9 — Threshold tuning (post-retrain)
Update `ml/scripts/tune_threshold.py` for multi-label.

### Phase 10 — ZKML rebuild (post-retrain)
The existing EZKL circuit was built for [1]-output model.
New [10]-output model requires full circuit rebuild:
1. Rebuild knowledge distillation proxy MLP (10 outputs)
2. Re-export to ONNX
3. Re-run `setup_circuit.py` + `run_proof.py`
