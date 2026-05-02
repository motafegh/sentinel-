# SENTINEL — Current Status

Last updated: 2026-05-02

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
| M3 MLOps | ⚠️ Partial | MLflow + DVC + Dagster wired; model registry promotion script missing |
| M4 Agents/RAG | ⚠️ Partial | Core complete; LLM synthesizer not yet upgraded; cross-encoder not wired |
| M5 Contracts | ✅ Source complete | Foundry tests written; forge not yet run (not installed in env) |
| M6 Integration API | ❌ Not built | api/ directory does not exist. Auth/rate-limit design required before building routes. |

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
| T2-A Prometheus | `prometheus-fastapi-instrumentator` not yet added to `api.py` |
| T2-B Drift detection | `drift_detector.py` not yet created. **Drift baseline strategy must be resolved before implementing** — see note below. |
| T2-C MLflow registry | `promote_model.py` not yet created |
| T3-A LLM synthesizer | `nodes.py:synthesizer()` still uses rule-based output |
| T3-B Cross-encoder reranking | `retriever.py` rerank param not yet added |
| Audit items #9/#11 | Temp file SIGKILL (`preprocess.py`), RAM cache integrity check (`dual_path_dataset.py`) |
| Audit item #13 | **Already fixed** — FocalLoss `.float()` cast committed 2026-05-01 (Fix #2 + Fix #6 in focalloss.py and _FocalFromLogits). Remove from open items. |
| Graph dataset edge_attr | No script confirms `edge_attr` tensors are present in `ml/data/graphs/*.pt` files. P0-B degrades gracefully to zero-vectors if absent — but **signal is lost silently**. `validate_graph_dataset.py` needed before retrain. |
| M6 auth design | Bearer token + rate-limit design must be written before building `api/` routes |
| ZKML resolution | M2 has no scheduled move to run the pipeline or formally descope it |
| Unit test plan | `cache.py`, `drift_detector.py`, `promote_model.py` have no test coverage planned |
| Multi-contract parsing | `GraphExtractionConfig.multi_contract_policy` scaffold exists (`"first"`, `"by_name"`). `"all"` policy not implemented. Single-contract limit documented in `ml/README.md` Known Limitation #2. Cache key strategy and `PredictResponse` schema extension must be decided before implementing. See ROADMAP Move 9. |

---

## Active Checkpoint

```
ml/checkpoints/multilabel_crossattn_best.pt
  epoch:      34
  val F1-macro: 0.4679
  architecture: cross_attention_lora (pre-edge_attr)
```

**This checkpoint was trained WITHOUT edge_attr support (pre-P0-B).**
The next retraining run will incorporate edge relation type embeddings.

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

See `docs/ROADMAP.md` for the ordered list of remaining work.
Before proceeding with Moves 3–8, confirm:
1. Audit item #13 is closed (already done — verify in `focalloss.py` and `trainer.py`)
2. `validate_graph_dataset.py` is run — confirm `edge_attr` presence in `.pt` files
3. Retrain evaluation protocol above is agreed
4. Drift baseline strategy is decided
