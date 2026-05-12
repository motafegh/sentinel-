# SENTINEL — Current Status

Last updated: 2026-05-12 (v5.1 Phase 3 retrain running — fix #26/#27/#28 applied)

---

## v5.1 Plan Status (2026-05-12) ← CURRENT

v5.0 behavioral gate failed (15% detection, 0/3 safe specificity — identical to v4).
Three confirmed root causes fixed in v5.1. See
[docs/proposals/2026-05-12-v5.1-analysis-and-plan.md](proposals/2026-05-12-v5.1-analysis-and-plan.md).

| Phase | Status | Key Output |
|-------|--------|-----------|
| Phase 0a — Fix `_select_contract` interface filter | ✅ Complete (bf57069) | Ghost ~10% → <1% after re-extraction |
| Phase 0b — Function-level GNN pooling | ✅ Complete (bf57069) | Pools FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR only |
| Phase 0c — aux_loss_weight 0.1 → 0.3 | ✅ Complete (bf57069) | Prevents GNN eye gradient collapse from epoch 23 |
| Phase 0e — CFG failure counter | ✅ Complete (bf57069) | Non-silent: logger.error if failure rate >5% |
| Phase 0f — Dataset deduplication | ✅ Complete (472e042) | 68,523 → 44,420 rows; 34.9% cross-split leakage eliminated |
| Phase 1 — Re-extract graphs | ✅ Complete (65617e1) | 44,140/44,420 fresh (99.4%); 44,420/44,420 validate PASS |
| Phase 2a — CEI contrastive pairs (~50) | ⏳ Pending | Teach call-before-write vs write-before-call distinction |
| Phase 2b — DoS augmentation (+300) | ⏳ Pending | DoS: 257 train → ~557; augment SmartBugs SWC-128 |
| Phase 3 — Retrain v5.1 (60 ep, fresh) | 🔄 **In Progress** | Run: `v5.1-fix28` — epoch 1 running, gnn_eye=0.601 ✅ healthy |
| Phase 4 — Validate + promote | ⏳ Pending | All gates → promote_model.py |

**v5.0 final checkpoint (NOT promoted):** `ml/checkpoints/v5-full-60ep_best.pt`
(epoch 43, raw F1=0.5736, tuned F1=0.5828 — validation gates cleared, behavioral gates failed)

**Active fallback:** `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt` (tuned F1=0.5422)

---

## v5.1 Training Run — Live (2026-05-12)

| Item | Value |
|------|-------|
| Run name | `v5.1-fix28` |
| Launched | 2026-05-12 23:46 UTC+3:30 |
| Epochs | 60 (fresh, no resume) |
| Effective batch | 64 (batch=16 × accum=4) |
| AMP / TF32 | ✅ / ✅ |
| LoRA | r=16, alpha=32, modules=[query, value] — 589,824 trainable / 124M frozen |
| Grad norm @ step 100 | gnn_eye=**0.601** tf_eye=0.170 fused_eye=0.191 — all eyes live ✅ |
| Log | `ml/logs/train_v5.1_fix28.log` |

---

## v5.0 Final Results (2026-05-12)

| Phase | Status | Key Output |
|-------|--------|-----------|
| Phase 0–4 — Full rebuild | ✅ Complete | NODE_FEATURE_DIM=12, 68K graphs, three-phase GNN |
| Phase 5A — Smoke | ✅ Complete | No errors |
| Phase 5B — 10-ep check | ✅ Complete | F1=0.3856, GNN re-engagement ep9–10 |
| Phase 5C — Full 60-ep run | ✅ Complete (ep 44) | Best F1=0.5736 (ep 43) |
| Phase 6a — tune_threshold | ✅ Complete | Tuned F1-macro=0.5828; DoS=0.449 |
| Phase 6b — Behavioral test | ✅ Complete | **FAILED: 15% detection, 0/3 safe** |
| Phase 6c — Promote | ❌ Blocked | Behavioral gate failed; v5.1 required |

See [docs/changes/2026-05-12-v5-evaluation-and-v5.1-analysis.md](changes/2026-05-12-v5-evaluation-and-v5.1-analysis.md)
for full evaluation record and root cause analysis.

See [docs/changes/2026-05-11-v5-implementation-record.md](changes/2026-05-11-v5-implementation-record.md)
for the v5.0 architecture build record.

---

## Module Completion

| Module | Status | Notes |
|--------|--------|-------|
| M1 ML Core — models | ✅ Complete | **v5**: three-phase GNNEncoder (12-dim, 7 edge types), TransformerEncoder (LoRA r=16), CrossAttentionFusion, three-eye SentinelModel |
| M1 ML Core — inference | ✅ Complete | api.py, predictor.py (windowed, full arch args, thresholds list), preprocess.py (cached), cache.py |
| M1 ML Core — training | ✅ Complete | TrainConfig with full arch fields, AMP, FocalLoss (FP32 cast fixed), early stopping, full-resume CLI, patience sidecar, focal_gamma/alpha logged to MLflow. Fix #25 (2026-05-09): model-only resume now resets epoch counter + patience + best_f1 (fine-tune mode); strict=False for lora_r mismatch. Autoresearch harness (`auto_experiment.py`, `program.md`, `README.md`) built and committed. Fix #26/#27/#28 applied 2026-05-12. |
| M1 ML Core — data extraction | ✅ Complete | ast_extractor.py V4.3, tokenizer.py (schema version metadata), dual_path_dataset.py (edge_attr shape guard) |
| M1 ML Core — shared preprocessing | ✅ Complete | graph_schema.py, graph_extractor.py (typed exceptions) |
| M1 ML Scripts | ✅ Complete | train.py (full-resume + reset-optimizer flags), tune_threshold.py (full arch args + fusion_dim lookup, prefetch guard), analyse_truncation.py, build_multilabel_index.py |
| M1 ML — known limitation | ⚠️ Tracked | **Single-contract scope**: only the first non-dependency contract per file is analysed. `GraphExtractionConfig.multi_contract_policy` scaffold exists; `"all"` policy not yet implemented. See Move 9 in ROADMAP. |
| M2 ZKML | ✅ Source complete | Z1/Z2/Z3 bugs fixed; pipeline not yet run. **No resolution path in ROADMAP — needs explicit move or descope decision.** |
| M3 MLOps | ✅ Complete | MLflow + DVC + Dagster wired; `promote_model.py` CLI added for Staging/Production promotion |
| M4 Agents/RAG | ✅ Complete | Core complete; LLM synthesizer upgraded (T3-A, qwen3.5-9b-ud, rule-based fallback); cross-encoder reranking added (T3-B, off by default) |
| M5 Contracts | ✅ Source complete | Foundry tests written; forge not yet run (not installed in env) |
| M6 Integration API | ❌ Not built | api/ directory does not exist. Auth/rate-limit design required before building routes. |

---

## Recent Changes (2026-05-12 — v5.1 Training Fixes #26/#27/#28)

- **Fix #26 — `need_weights=False` on MHA in `fusion_layer.py`** (commit df75466):
  Both `node_to_token` and `token_to_node` MHA calls now pass `need_weights=False`.
  Unlocks PyTorch's fused efficient-attention kernel; saves ~12.6 MB VRAM per forward
  pass. Zero behaviour change — the returned weight tensors were never used.

- **Fix #27 — `gc.collect()` + `torch.cuda.empty_cache()` between epochs** (commit df75466):
  Called after each `evaluate()` return inside the epoch loop in `trainer.py`.
  Releases the CUDA caching allocator's free-block pool back to CUDA before the next
  epoch's training starts. `import gc` added at module level.

- **Fix #28 — Grad norm logging moved before `zero_grad()`** (commit db2277f):
  `_grad_norm()` was called after `zero_grad(set_to_none=True)`, so `.grad` was always
  `None` → always logged `0.000` for all three eyes.
  Fix: grad norms are now captured inside the `is_accum_step` block, after
  `scaler.unscale_()` (fp32, post-clip) but **before** `zero_grad()` wipes them.
  A new `optimizer_step` counter makes `log_interval` count optimizer steps (not
  micro-batches), which is the correct granularity with gradient accumulation.
  **Confirmed working at step 100:** gnn_eye=0.601, tf_eye=0.170, fused_eye=0.191.

---

## Recent Changes (2026-05-12 — v5.1 Phase 0 complete; dataset dedup applied)

- **v4 experiment 1 — GATE CLEARED**: Fine-tune from v3 weights, lr=1e-4, 30 epochs, batch=16, lora_r=8. Best checkpoint at epoch 26.
  - Raw F1-macro: **0.5064** (v3 raw: 0.4715, +0.0349)
  - **Tuned F1-macro: 0.5422** (gate: >0.5069 ✅)
  - All 10 classes improved; none dropped below floor.
  - Patience=4/7 at epoch 30 — model was still learning (not early-stopped).
  - Checkpoint: `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt`
  - Thresholds: `ml/checkpoints/multilabel-v4-finetune-lr1e4_best_thresholds.json`
  - Confirms LR exhaustion diagnosis: fresh LR cycle from v3 weights recovered all classes.

---

## Recent Changes (2026-05-04 — Post-Training Audit, Second Pass)

### Inference & Data Pipeline Fixes (#1–#7, #9)

- **Fix #1 — `dual_path_dataset.py` edge_attr shape guard**: Old graph `.pt` files stored
  `edge_attr` as `[E, 1]`; `GNNEncoder` requires `[E]`. A `squeeze(-1)` guard in
  `__getitem__` normalises legacy files transparently without breaking current-format
  files.

- **Fix #2 — `predictor.py` missing `SentinelModel` args on load**: All arch fields
  (`dropout`, `gnn_dropout`, `lora_target_modules`, etc.) are now forwarded from the
  saved checkpoint config. Non-default LoRA checkpoints no longer crash on API startup.

- **Fix #3 — `tune_threshold.py` missing `SentinelModel` args + fusion_output_dim lookup**:
  Same arch arg additions as Fix #2. Additionally, `fusion_output_dim` now prefers
  `ckpt_config.get("fusion_output_dim")` over the hardcoded `_ARCH_TO_FUSION_DIM`
  mapping, matching `predictor.py` Fix #7 behaviour.

- **Fix #4 — `predictor.py` warmup dummy graph missing `edge_attr`**: Warmup graph
  now includes a 1-D long zero tensor for `edge_attr` when `use_edge_attr=True`, so the
  embedding path is exercised during startup.

- **Fix #5 — `tune_threshold.py` `prefetch_factor` PyTorch 2.x warning**: DataLoader
  kwargs are built conditionally; `prefetch_factor`, `pin_memory`, and
  `persistent_workers` are only included when `num_workers > 0`.

- **Fix #6 — `predictor.py` API response `threshold` → `thresholds` (breaking)**:
  `_format_result()` now returns `"thresholds": self.thresholds.cpu().tolist()` — a
  list of per-class thresholds. ⚠️ **Breaking key rename**: downstream consumers must
  update.

- **Fix #7 — `predictor.py` `fusion_output_dim` fallback order**: Now prefers
  `saved_cfg.get("fusion_output_dim")` first; falls back to `_ARCH_TO_FUSION_DIM`
  only for legacy checkpoints.

- **Fix #9 (MLflow) — `focal_gamma` / `focal_alpha` now logged**: Both params appear
  unconditionally in the MLflow run params dict, enabling comparison of Focal Loss
  sweeps in the UI.

---

## Recent Changes (2026-05-04 — Batch 3: Resume & Robustness Fixes (#23–#25))

- **Fix #23 — Patience sidecar for full persistence**: A `{checkpoint}.state.json`
  sidecar is written after **every** epoch with the real `patience_counter`. On resume,
  this sidecar overrides the checkpoint's saved counter (which is always 0 when a
  new best was found). If the sidecar is absent, a clear warning is logged.

- **Fix #24 — Warning on missing optimizer key during full resume**: When
  `--no-resume-model-only` is used but the checkpoint lacks the `"optimizer"` key,
  a prominent `WARNING` is now logged instead of silent fallback.

- **Fix #25 — Explicit `total_steps` existence check in scheduler restore**: The
  scheduler guard now distinguishes between a missing `total_steps` key (older PyTorch)
  and a genuine mismatch, logging the appropriate reason.

---

## Open Half-Open Loops

| Item | Missing piece |
|------|--------------|
| M6 auth design | Bearer token + rate-limit design must be written before building `api/` routes |
| ZKML resolution | M2 has no scheduled move to run the pipeline or formally descope it (see ROADMAP S5.5) |
| Multi-contract parsing | `GraphExtractionConfig.multi_contract_policy` scaffold exists (`"first"`, `"by_name"`). `"all"` policy not implemented. Single-contract limit documented in `ml/README.md` Known Limitation #2. See ROADMAP Move 9. |
| API breaking change (Fix #6) | Code complete — response key renamed to `"thresholds"`. Pending downstream consumer updates. |
| Move 8 Audit Item #9 | Preprocess temp file not cleaned on SIGKILL — not yet implemented. |
| v5.1 Phase 2a/2b | CEI contrastive pairs + DoS augmentation — deferred until Phase 3 first epoch results confirm model is healthy. |

### Closed loops (completed 2026-05-12)

| Item | Resolution |
|------|-----------|
| Fix #26 — MHA need_weights=False | `need_weights=False` on both MHA calls in `fusion_layer.py`; ~12.6 MB VRAM saved per forward pass |
| Fix #27 — CUDA cache between epochs | `gc.collect()` + `torch.cuda.empty_cache()` after `evaluate()` in `trainer.py` |
| Fix #28 — Grad norm always 0.000 | Moved `_grad_norm()` before `zero_grad()`; `optimizer_step` counter for correct interval tracking |

### Closed loops (completed 2026-05-02 / 2026-05-03 / 2026-05-04 / 2026-05-05)

| Item | Resolution |
|------|-----------|
| Fix #1 — dataset edge_attr shape `[E,1]` crash | `squeeze(-1)` guard in `dual_path_dataset.py.__getitem__`; backward compatible |
| Fix #2 — predictor SentinelModel incomplete args | All arch fields forwarded from saved config in `predictor.py` |
| Fix #3 — tune_threshold SentinelModel incomplete args + fusion_dim | Full arch args + `ckpt_config.get("fusion_output_dim")` in `tune_threshold.py` |
| Fix #4 — warmup dummy graph missing edge_attr | 1-D zero tensor added when `use_edge_attr=True` |
| Fix #5 — prefetch_factor PyTorch 2.x warning | Conditional DataLoader kwargs in `tune_threshold.py` |
| Fix #6 — API `threshold` key (single float, wrong) | Renamed to `thresholds` (list, per-class) — **breaking** |
| Fix #7 — fusion_output_dim hardcoded dict precedence | Prefers `saved_cfg` value; falls back to `_ARCH_TO_FUSION_DIM` |
| Fix #9 (training) — config.architecture AttributeError | `ARCHITECTURE` constant extracted; all usages unified |
| Fix #9 (MLflow) — focal_gamma/alpha not logged | Both params added to MLflow run params dict unconditionally |
| Fix #10 — OneCycleLR overflow on epoch extension | load_state_dict guard: skip scheduler state when total_steps mismatches |
| Fix #11 — patience_counter reset on resume | Saved + restored from checkpoint; JSON sidecar (Fix #23) for full persistence |
| Fix #12 — batch-size change on full resume | Warning emitted; `--resume-reset-optimizer` flag added |
| Fix #13 — pos_weight consistency on full resume | Warning logged explaining when mismatch is safe vs. risky |
| Fix #23 — patience sidecar per epoch | JSON written after every epoch; resume reads it |
| Fix #24 — missing optimizer key warning | Warning logged on full resume when key absent |
| Fix #25 — scheduler total_steps explicit check | Existence check with distinct warning for missing key |
| Full-resume CLI gap | `--no-resume-model-only` flag added to `train.py` |
| Duplicate constants | `NUM_EDGE_TYPES` imported, `MAX_SOURCE_BYTES` imported, `ARCHITECTURE` merged |
| `tune_threshold.py` hardcoded arch→dim mapping | Uses `_ARCH_TO_FUSION_DIM` as fallback, but now prefers config value |
| `--focal-gamma` / `--focal-alpha` CLI args | Wired end-to-end to `TrainConfig` |
| T2-A Prometheus | `prometheus-fastapi-instrumentator` added to `api.py` |
| T2-B Drift detection | `drift_detector.py` + `compute_drift_baseline.py` added |
| T2-C MLflow registry | `promote_model.py` CLI added |
| T3-A LLM synthesizer | Upgraded to qwen3.5-9b-ud with rule-based fallback |
| T3-B Cross-encoder reranking | `rerank=False` param added; uses `ms-marco-MiniLM-L-6-v2` |
| Graph dataset edge_attr validation | `validate_graph_dataset.py` added |
| Unit test plan | `test_cache.py`, `test_drift_detector.py`, `test_promote_model.py`, `test_gnn_encoder.py`, `test_fusion_layer.py` added |
| Training pipeline regeneration | All training inputs rebuilt fresh: graphs (68,523), tokens (68,568), multilabel_index.csv, splits (47,966/10,278/10,279) |
| Post-training arch alignment | `tune_threshold.py` + `predictor.py` now pass all GNN/LoRA params from checkpoint config |
| Checkpoint name alignment | All references updated to `multilabel_crossattn_v2_best.pt` |
| v3 retrain (60ep) | `multilabel-v3-fresh-60ep_best.pt` — best raw F1=0.4715, tuned F1=0.5069. Threshold JSON saved. ✅ |

---

## Active Checkpoint

ml/checkpoints/v5-full-60ep_best.pt   ← v5.0 best (NOT promoted — behavioral gate failed)
  epoch:      43
  raw F1:     0.5736
  tuned F1:   0.5828
  status:     ❌ Behavioral gate failed (15% detection, 0/3 safe)

ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt   ← ACTIVE FALLBACK
  tuned F1:   0.5422
  status:     ✅ Active fallback; superseded once v5.1 clears gate

---

## Next Actions

In priority order:

1. **v5.1 Phase 3 — watch epoch 1 complete** (run `v5.1-fix28` in progress):
   - Confirm F1-macro > 0.15 at epoch 1 (model learning, not stuck)
   - Confirm grad norms stay balanced (gnn_eye / tf_eye > 0.05) throughout
   - If healthy → let 60-epoch run complete uninterrupted

2. **v5.1 Phase 3 — after full run**:
   - Run `tune_threshold.py` on best checkpoint
   - Target: tuned F1-macro > 0.55, behavioral 70%/66%
   - If gate cleared → `promote_model.py`

3. **v5.1 Phase 2a/2b (optional pre-Phase 3 boost)**:
   - CEI contrastive pairs (~50 contracts) — call-before-write vs write-before-call
   - DoS augmentation (+300 from SmartBugs SWC-128)
   - Only worthwhile if Phase 3 epoch 1 shows DoS F1 near zero

4. **Fix #6 downstream update** — Update any API consumer that parsed `"threshold"` (single float)
   to use `"thresholds"` (list of floats).

5. **ZKML resolution** — decide Option A (run EZKL pipeline) or Option B (descope to S10).
   See ROADMAP S5.5.

6. **M6 Integration API** — design auth/rate-limit before writing any routes.

---

## Retrain Evaluation Protocol

| Parameter | Value |
|-----------|-------|
| Baseline checkpoint | `multilabel_crossattn_best.pt` — epoch 34, val F1-macro **0.4679** |
| Previous best (paused) | `multilabel_crossattn_v2_best.pt` — epoch 37, val F1-macro **0.4629** (raw), **0.4884** (tuned) |
| **v3 result** | `multilabel-v3-fresh-60ep_best.pt` — raw **0.4715**, tuned **0.5069** ✅ |
| **v4 result** | `multilabel-v4-finetune-lr1e4_best.pt` — tuned **0.5422** ✅ |
| **v5.0 result** | `v5-full-60ep_best.pt` — raw **0.5736**, tuned **0.5828** ✅ val / ❌ behavioral |
| Held-out split | Fixed — `ml/data/splits/val_indices.npy` (same seed, do NOT regenerate) |
| **v5.1 success gate** | tuned val F1-macro > **0.55** AND behavioral: 70% detection, 66% safe specificity |

---

## Drift Detection Baseline Note

⚠️ **Do not use training data as the KS drift baseline.**

`ml/data/graphs/` contains BCCC-SCsVul-2024 contracts (historical snapshot).
Computing a drift baseline from this data will cause the KS test to fire on almost
any 2026 production contract, making the alert useless.

Correct strategy:
- **Phase 1 (warm-up):** collect feature statistics from the first 500 real audit requests;
  emit no alerts during this period.
- **Phase 2 (active):** write `drift_baseline.json` from warm-up data; enable KS alerts.
- `compute_drift_baseline.py` must accept `--source [warmup|training]` with a
  prominent warning when `--source training` is used.
