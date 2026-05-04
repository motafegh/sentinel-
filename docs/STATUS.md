# SENTINEL — Current Status

Last updated: 2026-05-04

---

## Module Completion

| Module | Status | Notes |
|--------|--------|-------|
| M1 ML Core — models | ✅ Complete | GNNEncoder (edge_attr, configurable arch), TransformerEncoder (LoRA configurable), CrossAttentionFusion |
| M1 ML Core — inference | ✅ Complete | api.py, predictor.py (windowed), preprocess.py (cached), cache.py |
| M1 ML Core — training | ✅ Complete | TrainConfig with full arch fields, AMP, FocalLoss (FP32 cast fixed), early stopping, full-resume CLI |
| M1 ML Core — data extraction | ✅ Complete | ast_extractor.py V4.3, tokenizer.py (schema version metadata) |
| M1 ML Core — shared preprocessing | ✅ Complete | graph_schema.py, graph_extractor.py (typed exceptions) |
| M1 ML Scripts | ✅ Complete | train.py (full-resume flag), tune_threshold.py, analyse_truncation.py, build_multilabel_index.py |
| M1 ML — known limitation | ⚠️ Tracked | **Single-contract scope**: only the first non-dependency contract per file is analysed. `GraphExtractionConfig.multi_contract_policy` scaffold exists; `"all"` policy not yet implemented. See Move 9 in ROADMAP. |
| M2 ZKML | ✅ Source complete | Z1/Z2/Z3 bugs fixed; pipeline not yet run. **No resolution path in ROADMAP — needs explicit move or descope decision.** |
| M3 MLOps | ✅ Complete | MLflow + DVC + Dagster wired; `promote_model.py` CLI added for Staging/Production promotion |
| M4 Agents/RAG | ✅ Complete | Core complete; LLM synthesizer upgraded (T3-A, qwen3.5-9b-ud, rule-based fallback); cross-encoder reranking added (T3-B, off by default) |
| M5 Contracts | ✅ Source complete | Foundry tests written; forge not yet run (not installed in env) |
| M6 Integration API | ❌ Not built | api/ directory does not exist. Auth/rate-limit design required before building routes. |

---

## Recent Changes (2026-05-04)

### Resume Fixes and Retrain Extended to 60 Epochs

- **Fix #9 — `trainer.py` AttributeError on resume**: `config.architecture` does not exist
  on `TrainConfig` (never declared as a field). The resume cross-check crashed with
  `AttributeError` on every resume attempt. Fixed by extracting `ARCHITECTURE = "cross_attention_lora"`
  as a module-level constant and replacing all three scattered usages (resume check,
  checkpoint save dict, MLflow params). This bug was dormant since it lives inside
  `if config.resume_from:` — never triggered during the original fresh training run.

- **`--no-resume-model-only` CLI flag added to `train.py`**: `TrainConfig.resume_model_only`
  existed and `trainer.py` already had full-resume logic (optimizer + scheduler state restore),
  but the CLI never exposed this field. Added `--no-resume-model-only` flag so full-resume
  can be triggered from the command line without editing source code.

- **Retrain extended: 40 → 60 epochs, full-resume**: With both fixes applied, the
  interrupted epoch-37 checkpoint was resumed correctly:
  - Optimizer Adam m/v accumulators restored (37 epochs of gradient history preserved)
  - Scheduler LR curve continues from epoch-37 position (no spike back to `max_lr=3e-4`)
  - Training confirmed running: epoch 38/60 started at ~3.78 batch/s
  - **🔄 IN PROGRESS** — expected ~5 hours overnight

---

## Recent Changes (2026-05-03)

### Graph Dataset Re-Extraction
- **Full re-extraction completed**: All 68,523 graph `.pt` files regenerated using the
  unified `graph_extractor.py` pipeline. New files have `edge_attr` shape `[E]` (1-D),
  required by `GNNEncoder.edge_emb` (P0-B). Old files had shape `[E, 1]` which crashes
  `nn.Embedding` at the first training step.
- **32 orphaned files removed**: Files not present in `contracts_metadata.parquet`
  (couldn't be regenerated). Represent ~0.05% of the dataset.
- **`validate_graph_dataset.py` exit 0**: 68,523/68,523 PASS, 0 shape errors.
  Retrain is now unblocked.
- **`ml/pyproject.toml` completed**: Added `fastapi`, `uvicorn[standard]`, `loguru`,
  `httpx`, `scipy` — were installed manually but undeclared. `poetry lock --no-update`
  run to regenerate lock file.
- **`graph_schema.py` docstrings corrected**: Two comments stating "GNNEncoder ignores
  edge_attr" updated to reflect P0-B reality (embedding is now active).
- **Local data paths confirmed**:
  - Contracts metadata: `ml/data/processed/_cache/contracts_metadata.parquet`
  - Graphs output: `ml/data/graphs/`
  - DVC remote: `/mnt/d/sentinel-dvc-remote` (Windows D drive via WSL2)
- **Re-extraction command** (for future reference):
  ```bash
  rm -f ml/data/graphs/checkpoint.json   # clear checkpoint to force full run
  TRANSFORMERS_OFFLINE=1 ml/.venv/bin/python ml/data_extraction/ast_extractor.py \
    --input ml/data/processed/_cache/contracts_metadata.parquet \
    --output ml/data/graphs --workers 11 --verbose
  ```

---

## Recent Changes (2026-05-02)

### Pre-Retrain Architecture Improvements
- **P0-A LoRA externalized**: `TrainConfig` now has `lora_r`, `lora_alpha`, `lora_dropout`,
  `lora_target_modules`. `TransformerEncoder.__init__()` accepts these directly.
  Previously hardcoded in a module-level `LORA_CONFIG` constant.
- **P0-B edge_attr added**: `GNNEncoder` now embeds edge relation types (CALLS/READS/WRITES/
  EMITS/INHERITS → `nn.Embedding(5, 16)`) and passes them to all GATConv layers.
  Graceful degradation: if `edge_attr is None` (old .pt files), falls back to zero-vectors
  so existing data still runs — but without edge-type signal.
  Previously these were computed by `graph_extractor.py` but silently discarded.
- **P0-C architecture fields**: `gnn_hidden_dim`, `gnn_heads`, `gnn_dropout`, `use_edge_attr`,
  `gnn_edge_emb_dim`, `fusion_output_dim` now in `SentinelModel.__init__()` and `TrainConfig`.
  All logged to MLflow on each training run.
- **P0-D tokenizer metadata**: `tokenizer.py` now stores `feature_schema_version` in output `.pt` files.

### Inference Improvements
- **T1-C complete**: `predictor.py` now routes long contracts (> 512 tokens) through
  `process_source_windowed()` and aggregates probabilities via max across windows.
  `windows_used: int` added to `PredictResponse`.
- **T1-A inference cache**: `cache.py` (`InferenceCache`) added. `ContractPreprocessor`
  accepts an optional cache; `process_source()` checks before running Slither, writes on miss.
  Cache key = `content_md5_FEATURE_SCHEMA_VERSION`; TTL via file mtime (default 24h).

---

## Open Half-Open Loops

| Item | Missing piece |
|------|--------------|
| M6 auth design | Bearer token + rate-limit design must be written before building `api/` routes |
| ZKML resolution | M2 has no scheduled move to run the pipeline or formally descope it (see ROADMAP S5.5) |
| Multi-contract parsing | `GraphExtractionConfig.multi_contract_policy` scaffold exists (`"first"`, `"by_name"`). `"all"` policy not implemented. Single-contract limit documented in `ml/README.md` Known Limitation #2. See ROADMAP Move 9. |
| Retrain | 🔄 **IN PROGRESS** — `multilabel-v2-edge-attr-60ep`, epochs 38–60, full-resume from epoch 37 (best_f1=0.4629). Started 2026-05-04 00:35. Success gate: tuned val F1-macro > 0.4884. |
| Autoresearch | 📋 **PLANNED** — `auto_experiment.py` + `ml/autoresearch/program.md` not yet built. Unblocked after retrain completes. See `docs/changes/2026-05-04-resume-fixes-and-autoresearch.md`. |

### Closed loops (completed 2026-05-02 / 2026-05-03 / 2026-05-04)

| Item | Resolution |
|------|-----------|
| Fix #9 — `config.architecture` AttributeError | `ARCHITECTURE` constant extracted; resume check, checkpoint dict, MLflow params all use it |
| Full-resume CLI gap | `--no-resume-model-only` flag added to `train.py`; `resume_model_only` wired to `TrainConfig` |
| T2-A Prometheus | `prometheus-fastapi-instrumentator` added to `api.py`; custom gauges for model load + GPU memory |
| T2-B Drift detection | `drift_detector.py` + `compute_drift_baseline.py` added; KS test + rolling buffer + warm-up mode |
| T2-C MLflow registry | `promote_model.py` CLI added (Staging/Production, dry-run, git tags) |
| T3-A LLM synthesizer | `synthesizer()` node upgraded — calls qwen3.5-9b-ud; rule-based fallback on timeout/unavailable |
| T3-B Cross-encoder reranking | `retriever.py` `rerank=False` param added; `_rerank()` uses `ms-marco-MiniLM-L-6-v2` |
| Audit #9 | `preprocess.py` SIGKILL-safe temp files (atexit registry + startup purge) |
| Audit #11 | `dual_path_dataset.py` RAM cache integrity check (type, hash, graph.x, tokens shape) |
| Graph dataset edge_attr | `validate_graph_dataset.py` added — checks presence, shape `[E]`, values in `[0, 5)` |
| Unit test plan | `test_cache.py`, `test_drift_detector.py`, `test_promote_model.py`, `test_gnn_encoder.py`, `test_fusion_layer.py` all added |
| Training pipeline regeneration | All training inputs rebuilt fresh: graphs (68,523), tokens (68,568), multilabel_index.csv, splits (47,966/10,278/10,279; 64.3% vuln stratified) |
| `create_splits.py` stratification fix | Binary labels now derived from `multilabel_index.csv` (sum > 0); `graph.y` is hardcoded 0 by extractor so `label_index.csv` is obsolete |
| Post-training arch alignment | `tune_threshold.py` + `predictor.py` now pass all GNN/LoRA params from checkpoint config when constructing `SentinelModel`; previously only `num_classes`+`fusion_output_dim` were passed (silent wrong-model risk) |
| Checkpoint name alignment | `api.py` default, `TrainConfig`, `tune_threshold.py`, `promote_model.py` examples all updated to `multilabel_crossattn_v2_best.pt` |

---

## Active Checkpoint

```
ml/checkpoints/multilabel_crossattn_best.pt   ← baseline (pre-edge_attr)
  epoch:        34
  val F1-macro: 0.4679
  architecture: cross_attention_lora (pre-P0-B)

ml/checkpoints/multilabel_crossattn_v2_best.pt   ← retrain IN PROGRESS (full-resume)
  run:          multilabel-v2-edge-attr-60ep
  experiment:   sentinel-retrain-v2
  resumed:      2026-05-04 00:35 (from epoch 37, best_f1=0.4629)
  target:       epoch 60
  edge_attr:    True (P0-B active)
  resume_type:  FULL (optimizer + scheduler state restored)
```

Retrain is running. Baseline checkpoint remains active until the new run
completes and clears the tuned val F1-macro > 0.4884 success gate.

---

## Retrain Evaluation Protocol

| Parameter | Value |
|-----------|-------|
| Baseline checkpoint | `multilabel_crossattn_best.pt` — epoch 34, val F1-macro **0.4679** |
| Previous best (interrupted) | `multilabel_crossattn_v2_best.pt` — epoch 37, val F1-macro **0.4629** (raw), **0.4884** (tuned thresholds) |
| Success threshold | tuned val F1-macro > **0.4884** on fixed `val_indices.npy` split |
| Held-out split | Fixed — `ml/data/splits/val_indices.npy` (same seed, do NOT regenerate) |
| MLflow experiment | `sentinel-retrain-v2` |
| Rollback rule | If tuned F1 < 0.4884 after 60 epochs: revert to current checkpoint; investigate edge_attr signal or try `loss_fn=focal` before re-running |
| Per-class floor | No single class should drop > 0.05 F1 from pre-retrain value — check per-class F1 in MLflow |

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

1. **Retrain** — 🔄 IN PROGRESS. Wait for completion overnight. Run `tune_threshold.py`
   in the morning; compare tuned F1 against 0.4884. See post-training protocol in
   `docs/changes/2026-05-04-resume-fixes-and-autoresearch.md`.

2. **Autoresearch setup** — After retrain completes:
   - Implement `ml/scripts/auto_experiment.py` (thin CLI wrapper printing `SENTINEL_SCORE`)
   - Write `ml/autoresearch/program.md` (metric, constraints, allowed knobs)
   - Use to drive hyperparameter search without manual overnight script editing

3. **ZKML resolution** — decide Option A (run EZKL pipeline) or Option B (descope to S10).
   See ROADMAP S5.5.

4. **M6 Integration API** — design auth/rate-limit before writing any routes.
   See ROADMAP M6 section.

5. **Move 9 (post-M6)** — multi-contract parsing (`multi_contract_policy="all"`
   in `graph_extractor.py`). See ROADMAP Move 9.
