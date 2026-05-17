# SENTINEL — Current Status

Last updated: 2026-05-17 (v6.0 training STOPPED — data fixes applied, cache rebuilding)

---

## v6.0 Training — STOPPED (data fixes in progress)

Training was killed at epoch ~16 (best F1-macro = **0.1717** at epoch 9, stalled).

**Why stopped:** Fresh full validation found **22.4% of graphs (9,973 / 44,470)** had out-of-range features:
- BUG-1: 2,856 graphs with raw `loc` values (max=2,167 vs expected [0,1])
- BUG-2: 37 graphs with raw `complexity` values (max=48 vs expected [0,1])
- BUG-3: 7,854 graphs with `visibility=2` (out of [0,1] range)

**Fixes applied (commit 8c8ce8c):**
- All 44,470 graphs patched in-place — 0 OOR nodes confirmed post-patch
- `FEATURE_SCHEMA_VERSION` bumped v5 → v6
- `VISIBILITY_MAP` changed: `{0,1,2}` → `{0.0, 0.5, 1.0}`
- Cache rebuilding now (PID 110135)

**Decisions pending before restart:**
- ASL γ⁻: keep 4 or reduce to 2? (60.1% zero-label rows + γ⁻=4 may cause all-zeros collapse)
- DoS class: keep (3 pure train samples), drop, or augment?
- Resume from epoch 9 checkpoint or restart from scratch?

**Best checkpoint:** `ml/checkpoints/v6.0-20260517_best.pt` (epoch 9, F1=0.1717)

**Resume command (after cache rebuild + config decisions):**
```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. nohup python ml/scripts/train.py \
    --resume ml/checkpoints/v6.0-20260517_best.pt \
    --run-name v6.0-20260517-resumed \
    --experiment-name sentinel-v6 \
    --epochs 100 \
    --gradient-accumulation-steps 8 \
    --compile --num-workers 0 \
    > ml/logs/train_v6.0_20260517.log 2>&1 &
```

---

## v6 Pipeline — COMPLETE

All 8 pipeline steps completed. Training running. 

| Step | Status | Key Output |
|------|--------|-----------|
| Phase 0 — Feature schema v4 | ✅ DONE (bef1f2a, 310e738) | 12-dim v4: return_ignored, uses_block_globals, ext_call_count, loc |
| Phase 1 — Windowed tokenization | ✅ DONE (b38c9da, 74e968c) | `retokenize_windowed.py`, TransformerEncoder multi-window, max_windows=4 |
| Phase 2 — GNN arch (256-dim, 6-layer) | ✅ DONE (2bb0e16) | hidden_dim=256, 6 layers, conv3b/conv4b, WindowAttentionPooler, classifier 384→192→10 |
| Phase 3 — Training config | ✅ DONE (64dfc5a) | AsymmetricLoss, 100ep, patience=30, LoRA LR 0.3× |
| Audit + BUG fixes (schema v5) | ✅ DONE (0d11e18) | 9 bugs found/fixed; loc/complexity/contract-selection/return_ignored all fixed; schema v4→v5 |
| Graph re-extraction (v7) | ✅ DONE | 41,521 ok / 74 ghost / 2,875 skipped; stale graphs patched in-place |
| Windowed retokenization | ✅ DONE | 44,470/44,470; ml/data/tokens_windowed/ [4,512] per file |
| Dedup + Timestamp relabeling | ✅ DONE (a75ae67) | 972 Timestamp labels removed (50.3% unverified); 1,933→961 |
| Full data validation | ✅ DONE | 22.4% graphs had OOR features; BUG-1/2/3 confirmed |
| Graph feature patch (schema v6) | ✅ DONE (8c8ce8c) | BUG-1/2/3 patched; 0 OOR nodes; schema v5→v6 |
| Cache rebuild (post-patch) | ✅ DONE | 2.3 GB, 44,470 pairs, all features verified in-range |
| v6.0 training | ⏸ **STOPPED** | Killed ep~16 for data fixes; best F1=0.1717 (ep9) |

---

## Updated Label Counts (post Timestamp relabeling — 2026-05-17)

| Class | Count | Train | Val | Test |
|-------|-------|-------|-----|------|
| IntegerUO | 13,797 | 9,697 | 2,047 | 2,053 |
| GasException | 4,957 | 3,448 | 773 | 736 |
| Reentrancy | 4,498 | 3,126 | 691 | 681 |
| MishandledException | 4,186 | 2,957 | 626 | 603 |
| CallToUnknown | 3,256 | 2,261 | 499 | 496 |
| TransactionOrderDependence | 3,028 | 2,112 | 445 | 471 |
| ExternalBug | 3,009 | 2,123 | 427 | 459 |
| UnusedReturn | 2,716 | 1,899 | 409 | 408 |
| **Timestamp** | **961** | **679** | **155** | **127** |
| **DenialOfService** | **346** | **215** | **68** | **63** |

Total: 44,470 rows (train=31,128 / val=6,669 / test=6,673)

---

## v6 Architecture (committed, training on this)

```
GNNEncoder (v6):       3-phase, 6-layer GAT; NODE_FEATURE_DIM=12; edge_emb Embedding(8,64)
                       Phase 1: structural+CONTAINS (layers 1+2, heads=8, add_self_loops=True)
                       Phase 2: CONTROL_FLOW directed (layers 3+4+3b+4b, heads=1)
                         conv3b = 2nd CF hop → CALL→TMP→WRITE (CEI pattern)
                         conv4b = 2nd RC hop → CONTRACT carries deeper CFG signal
                       Phase 3: REVERSE_CONTAINS type-7 (layers 5+6, heads=1)
                       JK attention aggregation; Per-phase LayerNorm; hidden_dim=256

TransformerEncoder:    CodeBERT (124M frozen) + LoRA r=16 α=32 on Q+V all 12 layers
                       Windowed: input [B,4,512] → reshape [B*4,512] → CodeBERT → [B,4*512,768]

WindowAttentionPooler: Extracts CLS from each window → Linear(768,1) attention → weighted sum → [B,768]

CrossAttentionFusion:  attn_dim=256, output=128 LOCKED (ZKML proxy constraint)

Three-Eye Classifier:  GNN (pool→128) + TF (pooler→128) + Fused (→128)
                       Concat [B,384] → Linear(384,192) → ReLU → Dropout → Linear(192,10)
                       Aux heads per eye (λ=0.3 after warmup); NO Sigmoid inside model

Loss:                  AsymmetricLoss(γ⁻=4.0, γ⁺=1.0, clip=0.05) — down-weights easy negatives
Schema:                FEATURE_SCHEMA_VERSION="v5" (bumped from v4 this session)
```

---

## v6 Targets vs v5.2 Baselines

| Class | v5.2 Tuned F1 | v6.0 Target | Primary fix |
|-------|--------------|-------------|-------------|
| IntegerUO | 0.732 | ≥ 0.75 | Windowed tokens + hidden_dim=256 |
| GasException | 0.407 | ≥ 0.45 | 6-layer GNN + CF signal depth |
| Reentrancy | 0.322 | ≥ 0.40 | ASL + 2nd CF layer |
| MishandledException | 0.342 | ≥ 0.50 | return_ignored fix (was always 0) |
| UnusedReturn | 0.238 | ≥ 0.45 | return_ignored fix (was always 0) |
| Timestamp | 0.174 | ≥ 0.30 | uses_block_globals + clean labels |
| DenialOfService | 0.329 | ≥ 0.35 | Transfer/Send fix + ext_call_count |
| CallToUnknown | 0.284 | ≥ 0.35 | Windowed tokenization |
| TOD | 0.283 | ≥ 0.30 | uses_block_globals + windowed |
| ExternalBug | 0.262 | ≥ 0.30 | Deeper GNN + contract selection fix |
| **Macro avg** | **0.3422** | **≥ 0.43** | |

**Behavioral gates (primary pass/fail):**
- Detection rate ≥ 80% on `ml/scripts/manual_test.py` (v5.2: 36%)
- Safe specificity ≥ 80% (v5.2: 33%)

---

## Active Checkpoints

```
ml/checkpoints/v6.0-20260517_best.pt          ← TRAINING (not yet complete)
  epoch:      (in progress)
  status:     🔄 Running — epoch 1/100

ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt   ← ACTIVE FALLBACK
  tuned F1:   0.5422
  status:     ✅ Active fallback; behavioral gate not tested

ml/checkpoints/v5-full-60ep_best.pt           ← v5.0 (behavioral gate FAILED)
  tuned F1:   0.5828
  status:     ❌ Behavioral: 36% detection / 33% specificity

(v5.1/v5.2/v5.3 checkpoints also on disk — see docs/changes/2026-05-16-all-training-runs-summary.md)
```

---

## After v6.0 Training Completes

1. **Threshold tuning** — `ml/scripts/tune_thresholds.py` on val set (deduped splits)
2. **Behavioral tests** — `ml/scripts/manual_test.py` — gate: ≥80% detection, ≥80% specificity
3. **Per-class F1 check** — compare vs v6 targets table above
4. If gate passes → `promote_model.py`
5. If gate fails → Phase 4 (DoS/Timestamp data augmentation) and retrain

---

## Module Completion

| Module | Status | Notes |
|--------|--------|-------|
| M1 ML Core — models | ✅ Complete (v6) | Three-phase GNNEncoder (12-dim, 8 edge types), TransformerEncoder (LoRA r=16, windowed), CrossAttentionFusion, three-eye SentinelModel |
| M1 ML Core — inference | ✅ Complete | api.py, predictor.py, preprocess.py, cache.py |
| M1 ML Core — training | ✅ Complete | AsymmetricLoss, AMP, full-resume CLI, patience sidecar, all Fix #1–#28 applied |
| M1 ML Core — data extraction | ✅ Complete | graph_extractor.py schema v5 (9 bugs fixed), retokenize_windowed.py, dual_path_dataset.py |
| M1 ML Scripts | ✅ Complete | train.py, tune_threshold.py, reextract_graphs.py, validate_graph_dataset.py, create_cache.py, dedup_multilabel_index.py |
| M2 ZKML | ✅ Source complete | Proxy MLP, ONNX, EZKL circuit — pipeline NEVER run. Awaiting v6 checkpoint. |
| M3 MLOps | ✅ Complete | MLflow + DVC + Dagster wired; promote_model.py |
| M4 Agents | ✅ Complete | LangGraph AuditState pipeline; MCP servers :8010/:8011/:8012 |
| M5 Contracts | ❌ Never built | forge install/build/test never run; contracts/lib/ empty |
| M6 API | ❌ Does not exist | api/ directory missing; design auth/rate-limit before writing routes |

---

## Known Limitations (Not Fixed in v6)

| Issue | Impact | Plan |
|-------|--------|------|
| BUG-7: EMITS edges never created (old event syntax) | 0 EMITS edges in all 44K graphs | v7 if v6 behavioral fails |
| BUG-8: INHERITS edges never created | 0 INHERITS edges | v7 if v6 behavioral fails |
| BUG-3: visibility=2 for private | Two values (0+2) for "not external" | v7 if v6 behavioral fails |
| DoS: 7 pure-label contracts (98.1% Reentrancy co-occurrence) | Model cannot distinguish DoS from Reentrancy | Phase 4 augmentation |
| in_unchecked dead (0.8.x = 0.1% of dataset) | Feature dimension wasted | Next schema bump |
| 48 retryable Slither failures | Stale BUG-6 for 48 graphs | Low priority; re-try with --solc-args |

---

## Open Loops

| Item | Status |
|------|--------|
| M6 auth design | Blocked — design before building routes |
| ZKML execution | Blocked on v6 checkpoint completion |
| Phase 4 DoS/Timestamp augmentation | Planned but not started |
| Fix #6 downstream (threshold → thresholds) | Breaking rename; downstream consumers not updated |
| Preprocess temp file cleanup on SIGKILL (audit item #9) | Low priority hardening |
