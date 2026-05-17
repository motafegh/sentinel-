# v7 Pipeline Complete — Ready for Training
**Date:** 2026-05-18  
**Commits:** `46a8f9d` (all 27 bugs fixed) · `deddca0` (6 audit fixes) · `4fb61be` (doc sync) · `b4ad806` (pipeline fixes)  
**Status:** All data pipeline steps complete. Pre-training verification passed. v7.0 training ready.

---

## Summary

This session completed the full v7 re-extraction pipeline and ran a thorough pre-training
verification. Four additional bugs were found and fixed during verification.

---

## Part 1 — All 27 Bugs Fixed (schema v7)

Completed in two phases (previous sessions), committed as `46a8f9d` + `deddca0`.

### Phase 3 — Loss/Training fixes
| Bug | Fix |
|-----|-----|
| BUG-M3 | AsymmetricLoss `gamma_neg`/`gamma_pos`/`clip` accept `Union[float, Tensor[C]]`; registered as buffers |
| BUG-M8 | `evaluate()` sweeps 19 per-class thresholds per epoch; logs `val_f1_macro_tuned` to MLflow |

### Phase 4 — Extractor fixes
| Bug | Fix |
|-----|-----|
| BUG-M1 | `return_ignored`: `id(lval)` → `lval.name` for stable string identity |
| BUG-C3 | CFG nodes inherit dims [1,3,4,5,9] from parent FUNCTION (visibility/view/payable/complexity/has_loop) |
| BUG-H1 | `conv3c` added to GNNEncoder Phase 2 (3rd CF hop); `gnn_layers` 6→7 |
| BUG-H7 | EMITS edges: EventCall IR scan fallback for old Solidity 0.4.x event syntax |
| BUG-H8 | INHERITS edges: parent contracts added as CONTRACT nodes before declaration phase |
| BUG-L2 | `in_unchecked` removed from feature vector; NODE_FEATURE_DIM 12→11; `FEATURE_SCHEMA_VERSION` "v6"→"v7" |
| BUG-L4 | OOR validation logged at extraction time after `x = torch.tensor(x_list)` |

### Additional audit fixes (commit `deddca0`)
| Fix | File |
|-----|------|
| C1 | `create_cache.py` default tokens-dir corrected to `tokens_windowed` |
| C2 | DoS gradient no longer leaks through 3 aux heads |
| C3 | `--gnn-layers` CLI default 6→7 |
| C4 | `--dos-loss-weight` CLI flag added and wired |
| C6 | Shape guard in `GNNEncoder.forward()` for stale 12-dim files |
| F4 | `"positive"` added to `--weighted-sampler` choices |

---

## Part 2 — Data Pipeline Execution

### Archive (v6 data moved out)
```
ml/data/archive/graphs_v6/    44,472 graphs  (schema v6, 12-dim)
ml/data/archive/tokens_v6/    44,472 tokens
ml/data/archive/cached_dataset_windowed.pkl  2.47 GB
```

### DoS augmentation
- `generate_dos_pairs.py` generated 60 contracts (30 `dos_vuln_*.sol` + 30 `dos_safe_*.sol`)
- `dos_safe_12`, `dos_safe_19`, `dos_vuln_12`, `dos_vuln_19`, `dos_vuln_20`, `dos_vuln_24` fail compilation (interface nested inside contract — Solidity syntax error); accepted, skipped

### Graph re-extraction
- Script: `ml/scripts/reextract_graphs.py --workers 8`
- Result: **41,522 graphs**, 0 failures, 0.2% ghost rate
- Schema: v7, NODE_FEATURE_DIM=11, FEATURE_SCHEMA_VERSION="v7"

### Windowed tokenization
- Script: `ml/scripts/retokenize_windowed.py`
- Result: **44,470 tokens**, 0 failures, shape [4,512] each
- Note: crashed once during first run (WSL); full re-run from scratch produced clean output

### inject_augmented.py
- **Bug fixed this session**: `_check_tokens()` always returned False for augmented contracts (not in CSV yet → not tokenized by retokenize). Replaced with inline `_tokenize_contract()` using CodeBERT directly.
- Result: **+54 CSV rows** (+26 DoS-vuln, 28 DoS-safe), 6 compile-fail skipped
- CEI contracts (50) were already in CSV from prior session; all 50 graphs confirmed present
- Train split: 31,128 → 31,182

### label_cleaner.py
- Input: 44,524 rows (44,470 original + 54 injected)
- Removed: **17,722 noisy labels** across 6 classes (BCCC folder-level labeling without structural verification)
- Output: `ml/data/processed/multilabel_index_cleaned.csv`

| Class | Labels removed |
|-------|---------------|
| IntegerUO | −9,897 |
| CallToUnknown | −2,198 |
| MishandledException | −2,376 |
| UnusedReturn | −1,665 |
| Reentrancy | −1,163 |
| Timestamp | −423 |

### Label counts after cleaning
| Class | Count |
|-------|-------|
| IntegerUO | 3,900 |
| GasException | 4,957 |
| Reentrancy | 3,335 |
| MishandledException | 1,810 |
| ExternalBug | 3,009 |
| TOD | 3,028 |
| CallToUnknown | 1,058 |
| UnusedReturn | 1,051 |
| Timestamp | 538 |
| DenialOfService | 372 |

### create_cache.py
- **Bug fixed this session**: `weights_only=True` on graph `.pt` files caused all 41k loads to silently fail with empty exception messages (PyG `DataEdgeAttr`/`DataTensorAttr` not in torch safe globals). Fixed to `weights_only=False`.
- Result: **41,577 pairs, 2.28 GB** → `ml/data/cached_dataset_deduped.pkl`

---

## Part 3 — Pre-Training Verification

### Graph integrity (500 random sample)
- Wrong dim (≠11): **0**
- Wrong edge_attr shape: **0**
- OOR features: **0**

### Token integrity (500 random sample)
- Wrong shape (≠[4,512]): **0**
- Bad values: **0**

### Splits
- CSV rows: 44,524 | Train: 31,182 | Val: 6,669 | Test: 6,673 | Sum: 44,524 ✓
- OOB indices: 0 | Train∩Val: 0 | Train∩Test: 0 ✓

### Model forward pass
- `SentinelModel(gnn_num_layers=7)` — output `[2, 10]` ✓
- Log confirms: `layers=7 use_jk=True jk_mode=attention` ✓

### Manual spot-checks (graph features vs .sol source)
**DoS augmented (dos_vuln_01..03):**
- `distribute()` / `payAll()` / `refundAll()`: loop=1 ✓, ext_calls=0.228 (log1p(1)/log1p(20)) ✓
- Edge types [1,2,5,6] (READS+WRITES+CONTAINS+CF) correct — `.transfer()`/`.send()`/`.call{}` are low-level, no typed CALLS(0) ✓
- `join()` payable=1 ✓, `deposit()` payable=1 ✓

**BCCC Reentrancy (PallyCoin ERC20):**
- ext_calls=0.228/0.361 for functions making 1–2 external calls ✓
- call_target_typed=1.0 (typed ERC20 interface calls) ✓
- INHERITS(4) edge type present ✓

---

## Part 4 — Bugs Found During Verification (all fixed, commit `b4ad806`)

| ID | File | Bug | Fix |
|----|------|-----|-----|
| V1 | `create_cache.py:61` | `weights_only=True` for graph files — PyG objects not in safe globals; silently fails all 41k loads | `weights_only=False` with comment |
| V2 | `inject_augmented.py` | `_check_tokens()` always returns False for new augmented contracts (not in CSV → not tokenized) | Replaced with `_tokenize_contract()` — inline CodeBERT windowed tokenization |
| V3 | `train.py:80` | `--label-csv` default pointed to `multilabel_index_deduped.csv` (noisy) | Changed default to `multilabel_index_cleaned.csv` |
| V4 | `sentinel_model.py:126` | `gnn_num_layers` default was `6` (stale v6 value) — silently builds 6-layer model if flag not passed | Changed default to `7` (2+3+2 phase structure) |

---

## Documentation Updates

| File | Change |
|------|--------|
| `docs/ACTIVE_BUGS.md` | Rev 5: pipeline completion status block added |
| `ml/src/models/gnn_encoder.py` | All v6/12-dim comments updated to v7/11-dim |
| `ml/src/models/sentinel_model.py` | Version string v6→v7; `gnn_num_layers` docstring updated |
| `ml/src/preprocessing/graph_extractor.py` | SHAPE CONTRACT updated to v7 (11-dim, 8 edge types); schema history rewritten |
| `ml/src/preprocessing/graph_schema.py` | CHANGE POLICY next-version v3→v8; v7 entry added to history |

---

## Ready to Train

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v7.0 --experiment-name sentinel-v7 \
    --epochs 100 --gradient-accumulation-steps 8 --compile --num-workers 4
```

Expected VRAM: ~6.9/8.0 GB (batch_size=8, MAX_WINDOWS=4).  
Previous baseline: v6.0 best F1=0.1717 epoch 9, collapsed to all-zeros by epoch 16.  
Fallback checkpoint: `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt` (F1=0.5422).
