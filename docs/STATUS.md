# SENTINEL — Current Status

Last updated: 2026-05-12 (v5.1 Phase 0 complete; dataset dedup applied; Phase 1 extraction running)

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
| Phase 1 — Re-extract 44K graphs | 🔄 Running (PID 435806) | ~22 c/s; ETA ~51 min; gate: ghost <1% |
| Phase 2a — CEI contrastive pairs (~50) | ⏳ Pending | Teach call-before-write vs write-before-call distinction |
| Phase 2b — DoS augmentation (+300) | ⏳ Pending | DoS: 257 train → ~557; augment SmartBugs SWC-128 |
| Phase 3 — Retrain v5.1 (60 ep, fresh) | ⏳ Pending | Target: tuned F1 > 0.55, behavioral 70%/66% |
| Phase 4 — Validate + promote | ⏳ Pending | All gates → promote_model.py |

**v5.0 final checkpoint (NOT promoted):** `ml/checkpoints/v5-full-60ep_best.pt`
(epoch 43, raw F1=0.5736, tuned F1=0.5828 — validation gates cleared, behavioral gates failed)

**Active fallback:** `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt` (tuned F1=0.5422)

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
| M1 ML Core — training | ✅ Complete | TrainConfig with full arch fields, AMP, FocalLoss (FP32 cast fixed), early stopping, full-resume CLI, patience sidecar, focal_gamma/alpha logged to MLflow. Fix #25 (2026-05-09): model-only resume now resets epoch counter + patience + best_f1 (fine-tune mode); strict=False for lora_r mismatch. Autoresearch harness (`auto_experiment.py`, `program.md`, `README.md`) built and committed. |
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

## Recent Changes (2026-05-10 — v4 Experiment 1 Complete)

- **v4 experiment 1 — GATE CLEARED**: Fine-tune from v3 weights, lr=1e-4, 30 epochs, batch=16, lora_r=8. Best checkpoint at epoch 26.
  - Raw F1-macro: **0.5064** (v3 raw: 0.4715, +0.0349)
  - **Tuned F1-macro: 0.5422** (gate: >0.5069 ✅)
  - All 10 classes improved; none dropped below floor.
  - Patience=4/7 at epoch 30 — model was still learning (not early-stopped).
  - Checkpoint: `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt`
  - Thresholds: `ml/checkpoints/multilabel-v4-finetune-lr1e4_best_thresholds.json`
  - Confirms LR exhaustion diagnosis: fresh LR cycle from v3 weights recovered all classes.

## Recent Changes (2026-05-09/10 — v3 Analysis, v4 Harness, Experiment 1 Launch)

- **v3 deep analysis** — inspected actual MLflow curves (sqlite:///mlruns.db) and thresholds JSON. Key findings: (1) plateau was LR exhaustion, not capacity ceiling — train loss still decreasing at ep60; (2) DoS is a data problem (137 samples, noisy val F1 every epoch, no hyperparameter fix); (3) dominant failure mode is over-prediction (low precision) in 6/10 classes caused by aggressive BCE pos_weight. See `docs/changes/2026-05-09-v3-analysis-and-v4-direction.md`.
- **Fix #25 — model-only resume resets epoch counter**: `trainer.py` resume block now distinguishes fine-tune (`resume_model_only=True` → reset start_epoch/patience/best_f1) from full continuation (`resume_model_only=False` → restore from checkpoint). Previously, fine-tuning from v3 would set start_epoch=55, making any run with epochs<55 return immediately without training.
- **Fix — strict=False for lora_r mismatch**: `load_state_dict` now uses `strict=False`; LoRA key shape mismatches (from lora_r change) are warned; non-LoRA mismatches raise RuntimeError.
- **Autoresearch harness built**: `ml/scripts/auto_experiment.py`, `ml/autoresearch/program.md`, `ml/autoresearch/README.md` committed. Loop redesigned from grid search to analysis-first (Karpathy-style: read results → propose one targeted change → run fixed budget → keep/revert).
- **MLflow tracking URI corrected**: Use `sqlite:///mlruns.db` — the `file:///mlruns` backend has corrupt experiment entries (1, 2, 3) and does not contain the v3 run.

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
  this sidecar overrides the checkpoint’s saved counter (which is always 0 when a
  new best was found). If the sidecar is absent, a clear warning is logged.

- **Fix #24 — Warning on missing optimizer key during full resume**: When
  `--no-resume-model-only` is used but the checkpoint lacks the `"optimizer"` key,
  a prominent `WARNING` is now logged instead of silent fallback.

- **Fix #25 — Explicit `total_steps` existence check in scheduler restore**: The
  scheduler guard now distinguishes between a missing `total_steps` key (older PyTorch)
  and a genuine mismatch, logging the appropriate reason.

---

## Recent Changes (2026-05-04 — Training Audit, First Pass)

### Resume Fixes: patience_counter, batch-size guard, pos_weight warning

- **Fix #11 — `patience_counter` not saved/restored on resume**: Counter was reset to 0
  on every resume. Now saved in checkpoint dict and restored. Backward compatible with
  old checkpoints.

- **Fix #12 — batch-size change on full resume (stale Adam moments)**: A warning is now
  emitted when batch size mismatches on full resume. New `--resume-reset-optimizer`
  flag discards optimizer/scheduler while preserving model weights and patience.

- **Fix #13 — `pos_weight` consistency warning on full resume**: A `logger.warning()`
  now explains when restored Adam moments may be calibrated to a different loss scale.

### Training Second Audit Pass (#14–#18)

- **`--focal-gamma` / `--focal-alpha` CLI args**: Wired end-to-end to `TrainConfig`.
- **`patience_counter` JSON sidecar** – already covered by Fix #23 above.
- **Duplicate `ARCHITECTURE` constant removed**: Merge artefact fixed.
- **`validate_graph_dataset.py` duplicate constant eliminated**: `NUM_EDGE_TYPES` now
  imported from `graph_schema.py`.
- **`tune_threshold.py` hardcoded arch→dim mapping removed**: Now uses
  `_ARCH_TO_FUSION_DIM` from `predictor.py` (superseded by Fix #3 fusion_dim lookup).
- **`api.py` duplicate `MAX_SOURCE_BYTES` removed**: Now imported from
  `ContractPreprocessor.MAX_SOURCE_BYTES`.

### Resume Fixes and Retrain Extended to 60 Epochs

- **Fix #9 — `trainer.py` AttributeError on resume**: `config.architecture` does not
  exist. Fixed by extracting `ARCHITECTURE = "cross_attention_lora"` as a module-level
  constant.
- **Fix #10 — `trainer.py` OneCycleLR overflow on epoch extension**: Guard added:
  compare checkpoint `total_steps` vs new schedule; skip scheduler state on mismatch.
- **`--no-resume-model-only` CLI flag added**: Exposes full-resume from the command line.
- **Retrain extended: 40 → 60 epochs**: Resumed from epoch-37 checkpoint
  (`best_f1=0.4629`). Epoch 43 was stopped after batch-size mismatch (batch=32 vs
  original 16) caused loss spikes. Retrain is currently paused pending clean resume
  decision.

---

## Recent Changes (2026-05-03)

### Graph Dataset Re-Extraction
- **Full re-extraction completed**: All 68,523 graph `.pt` files regenerated using the
  unified `graph_extractor.py` pipeline. New files have `edge_attr` shape `[E]` (1-D),
  required by `GNNEncoder.edge_emb` (P0-B).
- **32 orphaned files removed**: Files not present in `contracts_metadata.parquet`.
- **`validate_graph_dataset.py` exit 0**: 68,523/68,523 PASS, 0 shape errors.
- **`ml/pyproject.toml` completed**: Added `fastapi`, `uvicorn[standard]`, `loguru`,
  `httpx`, `scipy` — were installed manually but undeclared.

---

## Recent Changes (2026-05-02)

### Pre-Retrain Architecture Improvements
- **P0-A LoRA externalized**: `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`
  in `TrainConfig`.
- **P0-B edge_attr added**: `GNNEncoder` now embeds edge relation types via `nn.Embedding(5, 16)`.
- **P0-C architecture fields**: `gnn_hidden_dim`, `gnn_heads`, `gnn_dropout`, `use_edge_attr`,
  `gnn_edge_emb_dim`, `fusion_output_dim` in `SentinelModel` and `TrainConfig`.
- **P0-D tokenizer metadata**: `feature_schema_version` stored in output `.pt` files.
- **T1-C complete**: windowed inference for contracts > 512 tokens; `windows_used` in response.
- **T1-A inference cache**: `cache.py` (`InferenceCache`) added; TTL via file mtime.

---

## Open Half-Open Loops

| Item | Missing piece |
|------|--------------|
| M6 auth design | Bearer token + rate-limit design must be written before building `api/` routes |
| ZKML resolution | M2 has no scheduled move to run the pipeline or formally descope it (see ROADMAP S5.5) |
| Multi-contract parsing | `GraphExtractionConfig.multi_contract_policy` scaffold exists (`"first"`, `"by_name"`). `"all"` policy not implemented. Single-contract limit documented in `ml/README.md` Known Limitation #2. See ROADMAP Move 9. |
| Autoresearch | ⚠️ **Harness built, loop redesign in progress** — `auto_experiment.py`, `ml/autoresearch/program.md`, `ml/autoresearch/README.md` committed (2026-05-09). Analysis-first loop (Karpathy-style) adopted over grid search. v4 experiment 1 running overnight (fine-tune from v3, lr=1e-4). |
| API breaking change (Fix #6) | Code complete — response key renamed to `"thresholds"`. Pending downstream consumer updates. |
| Move 8 Audit Item #9 | Preprocess temp file not cleaned on SIGKILL — not yet implemented. |

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

## Recent Changes (2026-05-11 — v5 Complete Rebuild)

- **v5 architecture**: three-phase 4-layer GAT (Phase 1 structural, Phase 2 CONTROL_FLOW directed, Phase 3 reverse-CONTAINS); three-eye classifier (GNN+TF+Fused + aux heads); 12-dim node features; 7 edge types
- **4 pre-flight bugs fixed**: CFG_NODE_WRITE mapping detection, type_id normalisation (/12.0), function name matching, cosine threshold miscalibration
- **Phase 4 re-extraction complete**: 68,523 v5 graphs; RAM cache 1.02 GB
- **10-epoch check run**: F1 0.19→0.38, GNN re-engagement confirmed at epoch 9–10 (gnn grad norm rising within epoch, B2900 gnn=0.294 > tf=0.112)
- **Full 60-epoch run launched**: PID 244442, model-only resume from epoch-10 weights, warmup-pct=0.06, patience=10

---

## Active Checkpoint

ml/checkpoints/v5-check-15ep_best.pt   ← current best (v5 training in progress)
  epoch:      10
  raw F1:     0.3856
  status:     ✅ In use as seed for full 60-ep run

ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt   ← fallback (v4 complete)
  tuned F1:   0.5422
  status:     ✅ Active fallback; superseded once v5 clears gate

---

## Legacy Checkpoints

ml/checkpoints/multilabel_crossattn_best.pt   ← baseline (pre-edge_attr)
  epoch:        34
  val F1-macro: 0.4679
  architecture: cross_attention_lora (pre-P0-B)

ml/checkpoints/multilabel_crossattn_v2_best.pt   ← previous v2 run (paused)
  run:          multilabel-v2-edge-attr-60ep
  experiment:   sentinel-retrain-v2
  last resumed: 2026-05-04 (stopped at epoch 43 batch 903 — batch-size mismatch)
  best_f1:      0.4629 (epoch 37)
  edge_attr:    True (P0-B active)
  status:       ❌ Superseded by v3.

ml/checkpoints/multilabel-v3-fresh-60ep_best.pt   ← **ACTIVE (v3 complete)**
  run:          multilabel-v3-fresh-60ep
  experiment:   sentinel-retrain-v3
  completed:    2026-05-05
  batch_size:   32
  epochs:       60/60 (no early stop; patience counter=6 at end)
  best epoch:   ~52–53
  best raw F1:  0.4715
  edge_attr:    True (P0-B active)
  status:       ✅ Complete

ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json   ← tuned thresholds
  tuned F1-macro:  0.5069  ✅ (gate: > 0.4884)
  tuned F1-micro:  0.5608
  Hamming loss:    0.2342
  Exact-match:     0.2763

---

## Retrain Evaluation Protocol

| Parameter | Value |
|-----------|-------|
| Baseline checkpoint | `multilabel_crossattn_best.pt` — epoch 34, val F1-macro **0.4679** |
| Previous best (paused) | `multilabel_crossattn_v2_best.pt` — epoch 37, val F1-macro **0.4629** (raw), **0.4884** (tuned) |
| **v3 result** | `multilabel-v3-fresh-60ep_best.pt` — raw **0.4715**, tuned **0.5069** ✅ |
| Held-out split | Fixed — `ml/data/splits/val_indices.npy` (same seed, do NOT regenerate) |
| MLflow experiment | `sentinel-retrain-v3` |
| **v4 success gate** | tuned val F1-macro > **0.5069** on same `val_indices.npy` split |
| Per-class floor for v4 | No single class should drop > 0.05 F1 from v3 tuned values |

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

---

## Next Actions

In priority order:

1. **v4 experiment 1 — read results** (running overnight 2026-05-09→10):
   - Run: `multilabel-v4-finetune-lr1e4` — fine-tune from v3 weights, lr=1e-4, 30 epochs, batch=16
   - Hypothesis: v3 plateau was OneCycleLR exhaustion, not capacity ceiling. Fresh LR cycle should lift classes still improving at ep60 (CTU, ExternalBug, Timestamp, TOD).
   - Log: `ml/autoresearch/runs/v4-finetune-lr1e4.log`
   - After run: `tune_threshold.py` on best checkpoint, compare per-class F1 to v3.
   - **Success:** raw F1 > 0.4715 → hypothesis confirmed, continue fine-tuning.
   - **Failure:** raw F1 ≤ 0.4715 → capacity bottleneck, try lora_r=16 next.
   - Full analysis: `docs/changes/2026-05-09-v3-analysis-and-v4-direction.md`

2. **v4 experiment 2 (after reading exp 1 results)**:
   - If exp 1 confirmed: fine-tune with lora_r=16 from v3 (more capacity for CTU, ExternalBug).
   - If exp 1 failed: train from scratch with lora_r=16, lr=3e-4, 60 epochs.
   - DoS weighted sampler: add after primary LR/capacity question is answered.
   - Focal loss: requires α tuning; do NOT use α=0.25 (designed for binary, hurts rare multi-label classes).

3. **Fix #6 downstream update** — Update any API consumer that parsed `”threshold”` (single float)
   to use `”thresholds”` (list of floats).

4. **ZKML resolution** — decide Option A (run EZKL pipeline) or Option B (descope to S10).
   See ROADMAP S5.5.

5. **M6 Integration API** — design auth/rate-limit before writing any routes.

6. **Move 9 (post-M6)** — multi-contract parsing (`multi_contract_policy=”all”`).
