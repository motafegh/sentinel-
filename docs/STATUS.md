# SENTINEL — Current Status

Last updated: 2026-05-03

---

## Module Completion

| Module | Status | Notes |
|--------|--------|-------|
| M1 ML Core — models | ✅ Complete | GNNEncoder (edge_attr, configurable arch), TransformerEncoder (LoRA configurable), CrossAttentionFusion |
| M1 ML Core — inference | ✅ Complete | api.py, predictor.py (windowed), preprocess.py (cached), cache.py |
| M1 ML Core — training | ✅ Complete | TrainConfig with full arch fields, AMP, FocalLoss (FP32 cast fixed), early stopping |
| M1 ML Core — data extraction | ✅ Complete | ast_extractor.py V4.3, tokenizer.py (schema version metadata) |
| M1 ML Core — shared preprocessing | ✅ Complete | graph_schema.py, graph_extractor.py (typed exceptions) |
| M1 ML Scripts | ✅ Complete | train.py, tune_threshold.py, analyse_truncation.py, build_multilabel_index.py |
| M1 ML — known limitation | ⚠️ Tracked | **Single-contract scope**: only the first non-dependency contract per file is analysed. `GraphExtractionConfig.multi_contract_policy` scaffold exists; `"all"` policy not yet implemented. See Move 9 in ROADMAP. |
| M2 ZKML | ✅ Source complete | Z1/Z2/Z3 bugs fixed; pipeline not yet run. **No resolution path in ROADMAP — needs explicit move or descope decision.** |
| M3 MLOps | ✅ Complete | MLflow + DVC + Dagster wired; `promote_model.py` CLI added for Staging/Production promotion |
| M4 Agents/RAG | ✅ Complete | Core complete; LLM synthesizer upgraded (T3-A, qwen3.5-9b-ud, rule-based fallback); cross-encoder reranking added (T3-B, off by default) |
| M5 Contracts | ✅ Source complete | Foundry tests written; forge not yet run (not installed in env) |
| M6 Integration API | ❌ Not built | api/ directory does not exist. Auth/rate-limit design required before building routes. |

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
| Retrain | 🔄 **IN PROGRESS** — `multilabel-v2-edge-attr`, 40 epochs, MLflow experiment `sentinel-retrain-v2` (2026-05-03). Success gate: val F1-macro > 0.4679. |

### Closed loops (completed 2026-05-02 / 2026-05-03)

| Item | Resolution |
|------|-----------|
| T2-A Prometheus | `prometheus-fastapi-instrumentator` added to `api.py`; custom gauges for model load + GPU memory |
| T2-B Drift detection | `drift_detector.py` + `compute_drift_baseline.py` added; KS test + rolling buffer + warm-up mode |
| T2-C MLflow registry | `promote_model.py` CLI added (Staging/Production, dry-run, git tags) |
| T3-A LLM synthesizer | `synthesizer()` node upgraded — calls qwen3.5-9b-ud; rule-based fallback on timeout/unavailable |
| T3-B Cross-encoder reranking | `retriever.py` `rerank=False` param added; `_rerank()` uses `ms-marco-MiniLM-L-6-v2` |
| Audit #9 | `preprocess.py` SIGKILL-safe temp files (atexit registry + startup purge) |
| Audit #11 | `dual_path_dataset.py` RAM cache integrity check (type, hash, graph.x, tokens shape) |
| Graph dataset edge_attr | `validate_graph_dataset.py` added — checks presence, shape `[E]`, values in `[0, 5)` |
| Unit test plan | `test_cache.py`, `test_drift_detector.py`, `test_promote_model.py`, `test_gnn_encoder.py`, `test_fusion_layer.py` all added |
| Training pipeline regeneration | All training inputs rebuilt fresh: graphs (68,523), tokens (68,568), multilabel_index.csv, splits (64.3% vuln stratified) |
| `create_splits.py` stratification fix | Binary labels now derived from `multilabel_index.csv` (sum > 0); `graph.y` is hardcoded 0 by extractor so `label_index.csv` is obsolete |

---

## Active Checkpoint

```
ml/checkpoints/multilabel_crossattn_best.pt   ← baseline (pre-edge_attr)
  epoch:        34
  val F1-macro: 0.4679
  architecture: cross_attention_lora (pre-P0-B)

ml/checkpoints/multilabel_crossattn_v2_best.pt   ← retrain in progress
  run:          multilabel-v2-edge-attr
  experiment:   sentinel-retrain-v2
  started:      2026-05-03
  epochs:       40 (running)
  edge_attr:    True (P0-B active)
```

Retrain is running. Baseline checkpoint remains active until the new run
completes and clears the val F1-macro > 0.4679 success gate.

---

## Retrain Evaluation Protocol

This protocol must be confirmed before launching the retrain. Do not re-randomize the split.

| Parameter | Value |
|-----------|-------|
| Baseline checkpoint | `multilabel_crossattn_best.pt` — epoch 34, val F1-macro **0.4679** |
| Held-out split | Fixed — `ml/data/splits/val_indices.npy` (same seed, do NOT regenerate) |
| Success threshold | val F1-macro > 0.4679 on the same held-out split |
| MLflow experiment | `sentinel-retrain-v2` (compare against `sentinel-multilabel` baseline run) |
| Rollback rule | If new checkpoint F1 < 0.4679 after 40 epochs: revert to current checkpoint; investigate P0-B `edge_emb_dim` (try 8 instead of 16) before re-running |
| Per-class floor | No single class should drop > 0.05 F1 from its pre-retrain value — log per-class F1 in MLflow and compare |

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

## Next Action

All ROADMAP Moves 0–8 and T3-A/T3-B complete. Remaining work in priority order:

1. **Retrain** — 🔄 IN PROGRESS (`multilabel-v2-edge-attr`, 40 epochs). Wait for completion; compare val F1-macro against 0.4679 baseline. See `docs/changes/2026-05-03-training-pipeline-fix.md` for post-training steps.
2. **ZKML resolution** — decide Option A (run EZKL pipeline) or Option B (descope to S10). See ROADMAP S5.5.
3. **M6 Integration API** — design auth/rate-limit before writing any routes. See ROADMAP M6 section.
4. **Move 9 (post-M6)** — multi-contract parsing (`multi_contract_policy="all"` in `graph_extractor.py`). See ROADMAP Move 9.
