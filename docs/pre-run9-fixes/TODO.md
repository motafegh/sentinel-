# Pre-Run-9 Fixes — Actionable TODO & Progress

**Status legend:**
- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[!]` Blocked
- `[-]` Cancelled / not needed

**Progress:** 2/8 fixes complete (Fix #1 + Fix #8 are pre-applied; #6-#8 are bonus)

---

## Fix #1 — Timestamp relabel [x DONE]

- [x] Apply `parents[2]` → `parents[3]` fix to `ml/scripts/archive/dedup_multilabel_index.py:64`
- [x] Run `dedup_multilabel_index.py --relabel-timestamp` to produce `ml/data/processed/multilabel_index_deduped.csv`
- [x] Validate: `ml/data/splits/deduped/{train,val,test}_indices.npy` exist (41,576 rows)
- [x] Verify Timestamp count dropped from 1,901 → 948
- [ ] **Pending:** confirm Run 8 checkpoint was trained on deduped splits (check train.py call site for `--label-csv` / `--splits-dir` flags)

---

## Fix #2 — Block-globals extraction [ ] NOT STARTED

**Order:** After #1, before #3. Required: 30 min coding + ~45 min re-extract on 8 workers.

### Code changes
- [ ] Edit `ml/src/preprocessing/graph_extractor.py:459-492` (`_compute_uses_block_globals`):
  - [ ] Add `rv_name` fallback check for `now`, `block.timestamp`, `block.number`
  - [ ] Add `rv_type` fallback check containing `"block"` in type name
  - [ ] Preserve existing `_SolidityVariableComposed` isinstance check
- [ ] Edit `ml/src/preprocessing/graph_extractor.py:552-567` (`_node_uses_block_globals`):
  - [ ] Apply same three-tier fallback pattern

### Validation
- [ ] Spot-check: `roulette.sol` (pre-0.8, uses `now`) → feat[2] > 0.5 (was 0.0)
- [ ] Run: `PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py --workers 8`
- [ ] Run: `python ml/scripts/validate_graph_dataset.py --check-block-globals`
- [ ] Confirm: feat[2] fires >9,500/41,576 (was 3,799)

---

## Fix #3 — External CALL_ENTRY edge [ ] NOT STARTED

**Order:** After #1, #2. Required: 2-3 hr coding + ~45 min re-extract.

### Schema changes
- [ ] Edit `ml/src/preprocessing/graph_schema.py:208`: `NUM_EDGE_TYPES = 12`
- [ ] Edit `ml/src/preprocessing/graph_schema.py:382-398` (EDGE_TYPES): add `"EXTERNAL_CALL": 11`

### Code changes
- [ ] Edit `ml/src/preprocessing/graph_extractor.py:825-888` (`_add_icfg_edges`):
  - [ ] After internal-calls loop, add external-calls loop iterating `node.high_level_calls` and `node.low_level_calls`
  - [ ] Emit self-loop edge `[caller_idx, caller_idx]` with `edge_type=11` per unique call site
- [ ] Edit `ml/scripts/train.py:165-166` (`--phase2-edge-types` default): include 11
- [ ] Edit `ml/src/models/gnn_encoder.py:471-483` (Phase 2 cfg_mask default): include 11

### Validation
- [ ] Add `--check-external-call-edges` flag to `ml/scripts/validate_graph_dataset.py`
- [ ] Run: `reextract_graphs.py` and `create_cache.py`
- [ ] Confirm: >2,000/41,576 graphs have edge_attr == 11 (was 0)

---

## Fix #4 — IntegerUO schema gap [ ] NOT STARTED

**Order:** After #2, #3 (most invasive schema change). Required: 4 hr coding + ~45 min re-extract.

### Schema changes
- [ ] Edit `ml/src/preprocessing/graph_schema.py:205`: `NUM_NODE_TYPES = 14`
- [ ] Edit `ml/src/preprocessing/graph_schema.py:250-269` (NODE_TYPES): add `"CFG_NODE_ARITH": 13`
- [ ] Edit `ml/src/preprocessing/graph_schema.py:174`: `NODE_FEATURE_DIM = 12`
- [ ] Edit `ml/src/preprocessing/graph_schema.py:422-435` (FEATURE_NAMES): add `"in_unchecked_block"` at index 11
- [ ] Edit `ml/src/preprocessing/graph_schema.py:160`: `FEATURE_SCHEMA_VERSION = "v9"`

### Code changes
- [ ] Edit `ml/src/preprocessing/graph_extractor.py:393-403` (`_compute_in_unchecked`):
  - [ ] Replace `NotImplementedError` body with real implementation (check `op.in_unchecked_block` and `expr.in_unchecked_block`)
  - [ ] Add per-function aggregation (any node with `in_unchecked_block=True` → 1.0)
- [ ] Edit `ml/src/preprocessing/graph_extractor.py:587-652` (`_cfg_node_type`):
  - [ ] Add Priority 3.5: any IR op is `Binary` with `type` in `BinaryType.{ADD,SUB,MUL,DIV,MOD,EXP,SHIFT_LEFT,SHIFT_RIGHT}` → return `CFG_NODE_ARITH` (id 13)
  - [ ] Verify Slither BinaryType names against installed Slither version (`python -c "from slither.slithir.operations import BinaryType; print(dir(BinaryType))"`)
- [ ] Edit `ml/src/preprocessing/graph_extractor.py:1078-1181` (`_build_node_features`):
  - [ ] Append `uses_unchecked = _compute_in_unchecked(obj)` after existing features
  - [ ] Return 12-dim list with new feature at index 11
- [ ] Edit `ml/src/preprocessing/graph_extractor.py:655-720` (`_build_cfg_node_features`):
  - [ ] Update return list to 12 dims (append `uses_unchecked` at index 11)
- [ ] Edit `ml/src/models/gnn_encoder.py:160-220` (GNNEncoder input projection):
  - [ ] Input projection must accept `NODE_FEATURE_DIM=12`

### Validation
- [ ] Add `--check-arith-nodes` and `--check-unchecked-feature` flags to `validate_graph_dataset.py`
- [ ] Spot-check: `17_integer_simple.sol` (uses `unchecked{}`) → feat[11] > 0.5
- [ ] Run: `reextract_graphs.py` and `create_cache.py`
- [ ] Confirm: CFG_NODE_ARITH (id 13) nodes present; feat[11] fires on 0.8+ contracts

---

## Fix #5 — Slither-derived labels [ ] NOT STARTED

**Order:** After #2, #3, #4 (uses new schema features for richer detector context). Required: 1 day (mostly Slither invocation time).

### Pre-condition
- [ ] Note: `ml/data/slither_results/` is empty. Slither must be invoked fresh.

### Code changes
- [ ] Create `ml/scripts/derive_slither_labels.py` (NEW, ~150 lines):
  - [ ] `DETECTOR_TO_CLASS` dict mapping Slither detector slugs → BCCC class names (see doc 05)
  - [ ] For each contract: `subprocess.run(["slither", str(sol_path), "--json", "-", "--detect", ALL_DETECTORS], timeout=120)`
  - [ ] Parse JSON output `results.detectors[].check` to populate 10-class multi-hot label
  - [ ] Disambiguation rules for `unchecked-send` (→ MishandledException) vs `unchecked-lowlevel` (→ UnusedReturn)
  - [ ] Manual heuristic for TOD: function reads `block.number` AND has external call after
  - [ ] Output: `ml/data/processed/multilabel_index_slither.csv` with `provenance_json` column
- [ ] Snapshot existing labels: `cp ml/data/processed/multilabel_index.csv ml/data/processed/multilabel_index_bccc.csv`
- [ ] Update `ml/scripts/create_cache.py` `--label-csv` default to `multilabel_index_slither.csv`
- [ ] Create `docs/label-provenance.md` documenting per-class mapping decisions

### Validation
- [ ] Run: `python ml/scripts/derive_slither_labels.py --workers 8` (~30-60 min)
- [ ] Run comparison:
  ```python
  python -c "
  import pandas as pd
  old = pd.read_csv('ml/data/processed/multilabel_index_bccc.csv')
  new = pd.read_csv('ml/data/processed/multilabel_index_slither.csv')
  for cls in [...]: print(f'{cls}: BCCC={old[cls].sum()} Slither={new[cls].sum()}')"
  ```
- [ ] Spot-check 10 contracts per class: do Slither's findings match the new label?
- [ ] Confirm: per-class counts drop as expected (Timestamp: 1,901 → 700-900; IntegerUO: 13,559 → 8,000-10,000)

---

## Fix #6 — Predictor tier-threshold bug [ ] NOT STARTED

**Order:** Independent (display-only). Required: 30 min.

### Code changes
- [ ] Edit `ml/src/inference/predictor.py:660-757` (`_format_result`):
  - [ ] Replace hardcoded `self.tier_confirmed_threshold` and `self.tier_suspicious_threshold` with per-class-aware comparison
  - [ ] For each class: compute `confirmed_cutoff = max(tier_confirmed_threshold, tuned + 0.20)` and `suspicious_cutoff = max(tier_suspicious_threshold, tuned)` where `tuned = self.thresholds.get(cls_name, DEFAULT_THRESHOLD)`
  - [ ] Preserve the 3-tier output structure (CONFIRMED, SUSPICIOUS, NOTEWORTHY)

### Validation
- [ ] Run: `python ml/scripts/archive/manual_test.py --checkpoint ml/checkpoints/GCB-P1-Run8-v10-20260605_best.pt --contract ml/scripts/test_contracts/12_safe_contract.sol`
- [ ] Confirm: 0 CONFIRMED detections (was 5+ classes > 0.55)

---

## Fix #7 — SmartBugs benchmark [ ] NOT STARTED

**Order:** Independent (eval methodology). Required: 1 hr.

### Pre-condition
- [ ] Verified: `ml/data/smartbugs-curated/dataset/` contains 143 .sol files across 10 category subdirs

### Code changes
- [ ] Create `ml/scripts/manual_test_smartbugs.py` (NEW, ~100 lines):
  - [ ] Iterate over `ml/data/smartbugs-curated/dataset/*/` category subdirs
  - [ ] Map SmartBugs category → BCCC class via `SB_CAT_TO_BCCC` dict (see doc 06)
  - [ ] For each contract: run model, record (true_class, top_pred, top_prob)
  - [ ] Compute per-category accuracy, macro-F1
  - [ ] Report confusion matrix

### Validation
- [ ] Run: `python ml/scripts/manual_test_smartbugs.py --checkpoint ml/checkpoints/GCB-P1-Run8-v10-20260605_best.pt`
- [ ] Compare against Run 7 baseline per-category (in `docs/run7-audit.md` if exists)

---

## Fix #8 — Complexity-proxy doc [x DONE in code, doc PENDING]

- [x] `ml/src/models/gnn_encoder.py:168, 208, 435-437` — `drop_complexity` flag zeros feat[5] in forward pass
- [x] `ml/scripts/train.py:219-220` — `--drop-complexity-feature` CLI flag exists
- [ ] **Pending:** when `docs/architecture-decisions.md` is created, add a one-paragraph note about:
  - [ ] Why complexity was dropped (feat[5] correlated with node count, model was using it as size proxy per L4 finding)
  - [ ] Current settings (`--drop-complexity-feature` in Run 8)
  - [ ] Future options: increase `--appnp-alpha`, add size normalization in graph_extractor

---

## Cross-cutting tasks [ ] NOT STARTED

After fixes #2, #3, #4 are applied:

- [ ] **Schema version bump:** `ml/src/preprocessing/graph_schema.py:160` → `FEATURE_SCHEMA_VERSION = "v9"`
- [ ] **Delete stale cache:** `rm ml/data/cached_dataset_v10.pkl` (2.5 GB, v8 schema)
- [ ] **Rebuild cache:** `python ml/scripts/create_cache.py --label-csv ml/data/processed/multilabel_index_deduped.csv --splits-dir ml/data/splits/deduped/ --output ml/data/cached_dataset_deduped.pkl`
- [ ] **Invalidate old checkpoints:** v8 checkpoints will fail to load. Either:
  - [ ] Train Run 9 from scratch, OR
  - [ ] Write a `migrate_checkpoint.py` to map v8 → v9 weights (most weights transfer cleanly; only the new `in_unchecked_block` projection and type-11/13 embeddings are fresh)
- [ ] **Update `ml/checkpoints/` manifest:** add a `MANIFEST.md` documenting which checkpoint corresponds to which schema version
- [ ] **Verify `ml/src/datasets/dual_path_dataset.py:221-229` cache-version check fires correctly:** schema mismatch should raise RuntimeError, not silently return stale labels

---

## Run 9 launch gating criteria [ ] NOT STARTED

From `00-overview.md` — must pass all 5 before launching Run 9:

- [ ] **G1:** Re-derive IntegerUO labels from Slither `integer-overflow` detector + structural guard (post Fix #5). Drop false positives.
- [ ] **G2:** Re-derive Timestamp labels via `--relabel-timestamp` + `now`/library extraction fix (post Fix #1 + #2).
- [ ] **G3:** Re-validate CALL_ENTRY/RETURN_TO coverage: at least 5% of training graphs have an EXTERNAL_CALL edge (post Fix #3). Was 0%.
- [ ] **G4:** SmartBugs Curated baseline: per-class precision > 0.3 on 6/10 classes (post Fix #7). Currently 1/10.
- [ ] **G5:** Manual safe contracts: no class above 0.50 on `12_safe_contract.sol` (post Fix #6). Currently 5/10 classes fire > 0.50.

---

## Execution order (recommended)

```
Phase 1 (parallel, no deps):
  Fix #6 (30 min) + Fix #7 (1 hr)        ← display + eval, don't affect training

Phase 2 (sequential, must be in order):
  Fix #2 → Fix #3 → Fix #4              ← each invalidates v8 schema
  (bump FEATURE_SCHEMA_VERSION after #4)
  Fix #5                                 ← uses new schema for richer Slither context

Phase 3 (post-validation):
  Cross-cutting: rebuild cache, retrain, gate against Run 7 baseline
  Fix #8 doc write                       ← when architecture-decisions.md exists
```

**Estimated total time:**
- Phase 1: 1.5 hr
- Phase 2: ~2.5 days (mostly re-extract + Slither invocation)
- Phase 3: 1 day (retrain + validation)

---

## Notes

- This file should be updated as work progresses — toggle checkboxes to `[x]` when each item is verified done.
- Each fix doc in this folder has the full spec; this TODO is the execution checklist.
- Do NOT commit the v9 schema changes until ALL of Fix #2, #3, #4 are applied together (avoids leaving the repo in an intermediate broken state).
