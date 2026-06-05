# Pre-Run-9 Pipeline — Fresh Build from Scratch

**Principle:** Everything from graphs to cache gets rebuilt. No old artifacts survive.

---

## What stays (immutable inputs)

| Path | What | Why |
|------|------|-----|
| `BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv` | 111,897-row BCCC label source | Authoritative ground truth |
| `BCCC-SCsVul-2024/SourceCodes/` | 12 category folders, SHA256-named .sol files | Source contracts |
| `ml/data/smartbugs-curated/dataset/` | 143 real .sol files | SmartBugs benchmark |
| `ml/data/smartbugs-results-master/results/slither/` | Pre-computed Slither JSONs | Fix #7 validation |
| `ml/scripts/test_contracts/` | 20 OOD synthetic .sol files | Manual testing |

## What gets deleted before re-extraction

```bash
rm -rf ml/data/graphs/*.pt           # all graph .pt files
rm -rf ml/data/tokens_windowed/*.pt   # all token .pt files
rm -f  ml/data/processed/multilabel_index.csv
rm -f  ml/data/processed/multilabel_index_deduped.csv
rm -f  ml/data/splits/deduped/*.npy
rm -f  ml/data/splits/v10_deduped/*.npy
rm -f  ml/data/splits/v9_deduped/*.npy
rm -f  ml/data/cached_dataset_*.pkl
```

## What gets created (in order)

```
Step 0: Apply code fixes (#2 → #3 → #4)
        ↓
Step 1: reextract_graphs.py   →  ml/data/graphs/*.pt        (41,576 graph files)
        ↓
Step 2: retokenize_windowed.py → ml/data/tokens_windowed/*.pt (41,576 token files)
        ↓
Step 3: build_multilabel_index.py → ml/data/processed/multilabel_index.csv
        ↓
Step 4: dedup_multilabel_index.py --relabel-timestamp
        → ml/data/processed/multilabel_index_deduped.csv
        → ml/data/splits/deduped/{train,val,test}_indices.npy
        ↓
Step 5: derive_slither_labels.py  → ml/data/processed/multilabel_index_slither.csv  (Fix #5)
        ↓
Step 6: create_cache.py --label-csv ml/data/processed/multilabel_index_deduped.csv
        → ml/data/cached_dataset_v9.pkl
        ↓
Step 7: train.py (Run 9)
```

---

## Step 0 — Apply code fixes (sequential, each breaks schema)

### Fix #2 — Block-globals extraction
- **File:** `ml/src/preprocessing/graph_extractor.py:459-492`
- **Change:** Add `now`/`blockhash`/`block.difficulty`/`block.basefee`/`block.prevrandao` fallback
- **Also:** Same fix to `_node_uses_block_globals` at line 552
- **Schema bump:** NO (fixes existing feature, doesn't add new one)
- **Smoke test:** `poetry run python ml/scripts/smoke/run_all.py --fix 2`

### Fix #3 — External CALL_ENTRY edge
- **Files:**
  - `ml/src/preprocessing/graph_schema.py:208` → `NUM_EDGE_TYPES = 12`
  - `ml/src/preprocessing/graph_schema.py:382-398` → add `"EXTERNAL_CALL": 11`
  - `ml/src/preprocessing/graph_extractor.py:825-888` → emit self-loop type 11
  - `ml/src/models/gnn_encoder.py:471-483` → add type 11 to Phase 2 cfg_mask
  - `ml/scripts/train.py:165-166` → default `--phase2-edge-types 6 8 9 10 11`
- **Schema bump:** YES → v9
- **Smoke test:** `poetry run python ml/scripts/smoke/run_all.py --fix 3`

### Fix #4 — IntegerUO schema gap
- **Files:**
  - `ml/src/preprocessing/graph_schema.py:205` → `NUM_NODE_TYPES = 14`
  - `ml/src/preprocessing/graph_schema.py:250-269` → add `"CFG_NODE_ARITH": 13`
  - `ml/src/preprocessing/graph_schema.py:174` → `NODE_FEATURE_DIM = 12`
  - `ml/src/preprocessing/graph_schema.py:422-435` → add `"in_unchecked_block"` at index 11
  - `ml/src/preprocessing/graph_schema.py:160` → `FEATURE_SCHEMA_VERSION = "v9"`
  - `ml/src/preprocessing/graph_extractor.py:393-403` → re-implement `_compute_in_unchecked`
  - `ml/src/preprocessing/graph_extractor.py:587-652` → add CFG_NODE_ARITH to `_cfg_node_type`
  - `ml/src/preprocessing/graph_extractor.py:1078-1181` → append feat[11] at index 11
  - `ml/src/preprocessing/graph_extractor.py:655-720` → 12-dim return in `_build_cfg_node_features`
  - `ml/src/models/gnn_encoder.py:160-220` → input projection accepts 12 dims
- **Schema bump:** YES → v9 (must be done together with #3)
- **Smoke test:** `poetry run python ml/scripts/smoke/run_all.py --fix 4`

**CRITICAL:** After all three are applied, verify schema:
```bash
poetry run python -c "from ml.src.preprocessing.graph_schema import FEATURE_SCHEMA_VERSION, NODE_FEATURE_DIM, NUM_NODE_TYPES, NUM_EDGE_TYPES; print(f'version={FEATURE_SCHEMA_VERSION} dim={NODE_FEATURE_DIM} node_types={NUM_NODE_TYPES} edge_types={NUM_EDGE_TYPES}')"
# Expected: version=v9 dim=12 node_types=14 edge_types=12
```

---

## Step 1 — Re-extract graphs

```bash
cd ~/projects/sentinel
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 poetry run python ml/scripts/reextract_graphs.py --workers 8
```

**Expected:** ~41,576 .pt files written to `ml/data/graphs/`
**Time:** ~30-60 min on 8 workers
**Gate out:** validate_graph_dataset.py passes all checks

```bash
poetry run python ml/scripts/validate_graph_dataset.py \
  --check-contains-edges \
  --check-control-flow \
  --check-block-globals
```

---

## Step 2 — Retokenize

```bash
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 poetry run python ml/scripts/retokenize_windowed.py --workers 8
```

**Expected:** ~41,576 .pt files written to `ml/data/tokens_windowed/`
**Time:** ~20-40 min

---

## Step 3 — Build multilabel index

```bash
PYTHONPATH=. poetry run python ml/scripts/build_multilabel_index.py
```

**Expected:** `ml/data/processed/multilabel_index.csv` with 41,576 rows
**Reads:** `BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv` + graph .pt contract_path

---

## Step 4 — Dedup + relabel Timestamp

```bash
PYTHONPATH=. poetry run python ml/scripts/archive/dedup_multilabel_index.py --relabel-timestamp
```

**Expected:**
- `ml/data/processed/multilabel_index_deduped.csv` (41,576 rows, 0 content-hash dups)
- `ml/data/splits/deduped/{train,val,test}_indices.npy` (70/15/15 split)
- Timestamp count drops from ~1,901 to ~948

**Gate out:** `poetry run python ml/scripts/smoke/run_all.py --fix 1` passes

---

## Step 5 — Slither-derived labels (Fix #5)

```bash
# Create the script first (per doc 05)
# Then run:
PYTHONPATH=. poetry run python ml/scripts/derive_slither_labels.py --workers 8
```

**Expected:** `ml/data/processed/multilabel_index_slither.csv` with provenance
**Time:** ~30-60 min (Slither invocation)
**Gate out:** `poetry run python ml/scripts/smoke/run_all.py --fix 5` passes

**Decision point:** Use `multilabel_index_deduped.csv` (BCCC labels) or `multilabel_index_slither.csv` (Slither labels) for training? Start with deduped BCCC; switch to Slither if BCCC labels prove too noisy.

---

## Step 6 — Build cache

```bash
PYTHONPATH=. poetry run python ml/scripts/create_cache.py \
  --label-csv ml/data/processed/multilabel_index_deduped.csv \
  --output ml/data/cached_dataset_v9.pkl
```

**Expected:** `ml/data/cached_dataset_v9.pkl` (~2.5 GB)
**Gate out:** cache loads without error, spot-check passes

---

## Step 7 — Train Run 9

```bash
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 poetry run python ml/scripts/train.py \
  --label-csv ml/data/processed/multilabel_index_deduped.csv \
  --splits-dir ml/data/splits/deduped/ \
  --cache ml/data/cached_dataset_v9.pkl \
  --drop-complexity-feature \
  --phase2-edge-types 6 8 9 10 11 \
  --appnp-alpha 0.0 \
  --epochs 50 \
  --patience 10
```

**Gate in (must pass ALL before training):**
- [ ] Smoke tests Fix #1-#5 all PASS
- [ ] validate_graph_dataset.py exits 0
- [ ] Cache loads without error
- [ ] SentinelModel constructs with in_channels=12, num_edge_types=12

**Gate out (Run 9 success criteria):**
- [ ] Test F1 > 0.3423 (Run 7 baseline)
- [ ] No class has precision < 0.10 (no degenerate class)
- [ ] Safe contracts: no class > 0.50

---

## What NOT to do

1. **Do NOT** reuse old graphs/tokens/splits/cache — they were extracted under v8 schema
2. **Do NOT** train before ALL of #2, #3, #4 are applied — schema must be v9
3. **Do NOT** skip deduplication — BCCC has 34.9% duplicate rows across categories
4. **Do NOT** use `multilabel_index.csv` (raw) for training — always use `_deduped.csv`
5. **Do NOT** commit intermediate v9 schema state — commit all three fixes together
