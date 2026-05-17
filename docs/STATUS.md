# SENTINEL — Current Status

Last updated: 2026-05-18 (v7 pipeline complete — ready for v7.0 training)

---

## v7.0 Training — READY TO START

All 27 pre-training bugs fixed. Full data pipeline verified. No blockers.

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v7.0 --experiment-name sentinel-v7 \
    --epochs 100 --gradient-accumulation-steps 8 --compile --num-workers 4
```

Expected VRAM: ~6.9 / 8.0 GB (batch_size=8, MAX_WINDOWS=4, grad_accum=8 → effective batch=64)

---

## v7 Pipeline — COMPLETE (2026-05-18)

| Step | Status | Key Output |
|------|--------|-----------|
| Archive v6 data | ✅ DONE | `ml/data/archive/graphs_v6/` (44,472) + `tokens_v6/` (44,472) |
| DoS augmentation | ✅ DONE | 60 .sol files (30 vuln + 30 safe); 6 compile-fail skipped |
| Graph re-extraction (schema v7) | ✅ DONE | 41,522 graphs; NODE_FEATURE_DIM=11; FEATURE_SCHEMA_VERSION="v7" |
| Windowed tokenization | ✅ DONE | 44,470 tokens; shape [4,512] each; 0 failures |
| inject_augmented.py | ✅ DONE | +54 rows (+26 DoS-vuln); CSV 44,470→44,524; train 31,128→31,182 |
| label_cleaner.py | ✅ DONE | −17,722 noisy labels → `multilabel_index_cleaned.csv` |
| create_cache.py | ✅ DONE | 41,577 pairs; 2.28 GB → `cached_dataset_deduped.pkl` |
| Pre-training verification | ✅ DONE | Feature ranges clean; splits valid; forward pass [B,10] layers=7 ✓ |

---

## v7 Architecture

```
GNNEncoder (v7):       3-phase, 7-layer GAT; NODE_FEATURE_DIM=11; edge_emb Embedding(8,64)
                       Phase 1: structural+CONTAINS (layers 1+2, heads=8, add_self_loops=True)
                       Phase 2: CONTROL_FLOW directed (conv3+conv3b+conv3c, 3 hops, heads=1)
                         conv3b = 2nd CF hop → CALL→TMP→WRITE (CEI pattern)
                         conv3c = 3rd CF hop → ENTRY→CALL→TMP→WRITE (BUG-H1 fix)
                       Phase 3: REVERSE_CONTAINS type-7 (layers 6+7, heads=1)
                       JK attention aggregation; Per-phase LayerNorm; hidden_dim=256

TransformerEncoder:    CodeBERT (124M frozen) + LoRA r=16 α=32 on Q+V of all 12 layers
                       Windowed: input [B,4,512]; MAX_WINDOWS=4

CrossAttentionFusion:  attn_dim=256, output=128 LOCKED (ZKML proxy constraint)
                       LayerNorm(768) on token input (BUG-C2 fix)

Three-Eye Classifier:  GNN eye + TF eye + Fused → [B,384] → Linear(384,192) → Linear(192,10)

Loss (v7):             AsymmetricLoss(γ⁻=2.0, γ⁺=1.0, clip=0.01); per-class Tensor supported
                       DoS column detached from loss (dos_loss_weight=0.0 — BUG-H6)
                       WeightedRandomSampler "positive" mode (3× weight for any-vuln rows)

Schema:                FEATURE_SCHEMA_VERSION="v7"; NODE_FEATURE_DIM=11
```

---

## Label Counts (cleaned CSV — training targets)

| Class | Count | Notes |
|-------|-------|-------|
| IntegerUO | 3,900 | Reduced from 13,797 by label_cleaner |
| GasException | 4,957 | Unchanged (structural filter has no rule) |
| ExternalBug | 3,009 | Unchanged |
| Reentrancy | 3,335 | Reduced from 4,498 |
| TransactionOrderDependence | 3,028 | Unchanged |
| MishandledException | 1,810 | Reduced from 4,186 |
| CallToUnknown | 1,058 | Reduced from 3,256 |
| UnusedReturn | 1,051 | Reduced from 2,716 |
| Timestamp | 538 | Reduced from 961 |
| **DenialOfService** | **372** | +26 from augmentation; still data-starved |

Total: 44,524 rows (train=31,182 / val=6,669 / test=6,673)

---

## Key Data Files

| File | Size | Contents |
|------|------|----------|
| `ml/data/graphs/` | 41,522 .pt | v7 graphs, 11-dim |
| `ml/data/tokens_windowed/` | 44,470 .pt | windowed tokens [4,512] |
| `ml/data/cached_dataset_deduped.pkl` | 2.28 GB | 41,577 paired (graph, tokens) |
| `ml/data/processed/multilabel_index_cleaned.csv` | 44,524 rows | cleaned labels (used by train.py) |
| `ml/data/splits/deduped/` | 3 .npy | train/val/test indices |

---

## Active Checkpoints

```
ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt   ← ACTIVE FALLBACK
  tuned F1:   0.5422
  status:     ✅ Active fallback

(v6.0 best_pt was epoch 9 F1=0.1717 — obsolete, not worth resuming)
(v5.0/v5.1/v5.2/v5.3 — see docs/changes/2026-05-16-all-training-runs-summary.md)
```

---

## After v7.0 Training Completes

1. **Threshold tuning** — `ml/scripts/tune_thresholds.py` on val set
2. **Behavioral tests** — `ml/scripts/manual_test.py` — gate: ≥80% detection, ≥80% specificity
3. **Per-class F1** — compare vs targets below
4. If gate passes → `promote_model.py` → ZKML pipeline (EZKL/Groth16)
5. If gate fails → diagnose (DoS augmentation? label noise? threshold calibration?)

### v7 Targets (vs v5.2 baseline)

| Class | v5.2 Tuned F1 | v7.0 Target | Key fix |
|-------|--------------|-------------|---------|
| IntegerUO | 0.732 | ≥ 0.75 | Cleaner labels (−9,897 noisy) |
| GasException | 0.407 | ≥ 0.45 | 7-layer GNN + 3rd CF hop |
| Reentrancy | 0.322 | ≥ 0.45 | CEI pattern (conv3c) + CFG inheritance |
| MishandledException | 0.342 | ≥ 0.50 | return_ignored fixed (BUG-M1) |
| UnusedReturn | 0.238 | ≥ 0.45 | return_ignored fixed + clean labels |
| Timestamp | 0.174 | ≥ 0.30 | uses_block_globals + cleaner labels |
| DenialOfService | 0.329 | ≥ 0.35 | +26 augmented vuln + DoS detached from loss |
| CallToUnknown | 0.284 | ≥ 0.35 | INHERITS/EMITS edges now fire |
| TOD | 0.283 | ≥ 0.30 | uses_block_globals |
| ExternalBug | 0.262 | ≥ 0.30 | CFG node features (BUG-C3) |
| **Macro avg** | **0.3422** | **≥ 0.45** | |

---

## Module Completion

| Module | Status | Notes |
|--------|--------|-------|
| M1 ML Core — models | ✅ v7 complete | 7-layer GNN, 11-dim schema, all 27 bugs fixed |
| M1 ML Core — inference | ✅ Complete | api.py, predictor.py, preprocess.py |
| M1 ML Core — training | ✅ Complete | ASL per-class, per-epoch threshold tuning, DoS detach, weighted sampler |
| M1 ML Core — data pipeline | ✅ Complete | v7 extractor, windowed tokenizer, inject+clean pipeline |
| M2 ZKML | ✅ Source complete | NEVER run — awaiting v7.0 checkpoint |
| M3 MLOps | ✅ Complete | MLflow `sqlite:///mlruns.db`, Dagster, promote_model.py |
| M4 Agents | ✅ Complete | LangGraph AuditState; MCP :8010/:8011/:8012 |
| M5 Contracts | ❌ Never built | forge install/build/test never run; contracts/lib/ empty |
| M6 API | ❌ Does not exist | api/ directory missing; design auth/rate-limit first |
