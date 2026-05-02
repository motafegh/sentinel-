# SENTINEL — Current Status

Last updated: 2026-05-02

---

## Module Completion

| Module | Status | Notes |
|--------|--------|-------|
| M1 ML Core — models | ✅ Complete | GNNEncoder (edge_attr), TransformerEncoder (LoRA configurable), CrossAttentionFusion |
| M1 ML Core — inference | ✅ Complete | api.py, predictor.py (windowed), preprocess.py (cached), cache.py |
| M1 ML Core — training | ✅ Complete | TrainConfig with full arch fields, AMP, FocalLoss, early stopping |
| M1 ML Core — data extraction | ✅ Complete | ast_extractor.py V4.3, tokenizer.py (schema version metadata) |
| M1 ML Core — shared preprocessing | ✅ Complete | graph_schema.py, graph_extractor.py (typed exceptions) |
| M1 ML Scripts | ✅ Complete | train.py, tune_threshold.py, analyse_truncation.py, build_multilabel_index.py |
| M2 ZKML | ✅ Source complete | Z1/Z2/Z3 bugs fixed; pipeline not yet run (GPU + local data needed) |
| M3 MLOps | ⚠️ Partial | MLflow + DVC + Dagster wired; model registry promotion script missing |
| M4 Agents/RAG | ⚠️ Partial | Core complete; LLM synthesizer not yet upgraded; cross-encoder not wired |
| M5 Contracts | ✅ Source complete | Foundry tests written; forge not yet run (not installed in env) |
| M6 Integration API | ❌ Not built | api/ directory does not exist |

---

## Recent Changes (2026-05-02)

### Pre-Retrain Architecture Improvements
- **P0-A LoRA externalized**: `TrainConfig` now has `lora_r`, `lora_alpha`, `lora_dropout`,
  `lora_target_modules`. `TransformerEncoder.__init__()` accepts these directly.
  Previously hardcoded in a module-level `LORA_CONFIG` constant.
- **P0-B edge_attr added**: `GNNEncoder` now embeds edge relation types (CALLS/READS/WRITES/
  EMITS/INHERITS → `nn.Embedding(5, 16)`) and passes them to all GATConv layers.
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
| T2-B Drift detection | `drift_detector.py` not yet created |
| T2-C MLflow registry | `promote_model.py` not yet created |
| T3-A LLM synthesizer | `nodes.py:synthesizer()` still uses rule-based output |
| T3-B Cross-encoder reranking | `retriever.py` rerank param not yet added |
| Audit items #9/#11/#13 | Temp file SIGKILL, RAM cache integrity, FocalLoss cast |

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

## Next Action

See `docs/ROADMAP.md` for the ordered list of remaining work.
The next immediate items are T2-A (Prometheus), T2-C (MLflow registry script),
T3-A (LLM synthesizer upgrade).
