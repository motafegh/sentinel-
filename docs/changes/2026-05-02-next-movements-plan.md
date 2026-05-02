# 2026-05-02 — SENTINEL Next Movements Strategic Plan

## Scope

This document records the agreed implementation order for all remaining work before
the next model retraining. It answers: what to build, in what order, and why.

---

## Assessment Answers

### Q: Is `ml/data_extraction/` separation from `ml/src/` correct?

**Yes — keep it.** The two directories have fundamentally different concerns:

| Aspect | `ml/data_extraction/` | `ml/src/` |
|--------|----------------------|-----------|
| Invocation | Offline batch CLI | Runtime package imported by API |
| Error policy | Skip-and-log (returns `None`) | Raise typed exceptions |
| Output | `.pt` files on disk | In-memory `Data` objects |
| Dependencies | solc-select, multiprocessing Pool | Slither system binary, FastAPI |

Both already import from `ml/src/preprocessing/` for shared graph logic.
Moving data extraction into `ml/src/` would blur the batch/runtime boundary.

### Q: Is `ml/data_extraction/tokenizer.py` needed?

**Yes — it is essential for training.** `DualPathDataset` loads `{hash}.pt` token files
from `ml/data/tokens/`. These files are created exclusively by `tokenizer.py`.
`preprocess.py` tokenizes at inference time in memory and does NOT write training token
files. Both use the same CodeBERT model and `MAX_LENGTH=512`.

---

## Phase 0 — Pre-Retrain Production Readiness

These changes affect what the model *learns* on the next training run. They must be done
before retraining, not after.

### P0-A — Externalize LoRA Hyperparameters (45 min)

**Problem:** `transformer_encoder.py` hardcodes `LORA_CONFIG` with `r=8, lora_alpha=16`.
Cannot tune LoRA rank without a code change.

**Changes:**
- `ml/src/models/transformer_encoder.py`: Remove module-level `LORA_CONFIG`; add
  `lora_r, lora_alpha, lora_dropout, lora_target_modules` params to `__init__()`
- `ml/src/models/sentinel_model.py`: Add same 4 fields to `SentinelModel.__init__()`;
  pass to `TransformerEncoder.__init__()`
- `ml/src/training/trainer.py` `TrainConfig`: Add `lora_r: int = 8`,
  `lora_alpha: int = 16`, `lora_dropout: float = 0.1`,
  `lora_target_modules: list[str]` (default `["query", "value"]`); pass to
  `SentinelModel(...)` at model construction

### P0-B — Add `edge_attr` Support to GNNEncoder (1.5h)

**Problem:** `graph_extractor.py` computes 5-class edge type IDs (`edge_attr`, shape `[E]`).
`GNNEncoder.forward()` ignores them — relation type information (CALLS vs READS vs WRITES
vs EMITS vs INHERITS) is completely lost.

**Changes:**
- `ml/src/models/gnn_encoder.py`: Add `use_edge_attr: bool = True`,
  `edge_emb_dim: int = 16`; import `NUM_EDGE_TYPES` from `graph_schema`;
  add `nn.Embedding(NUM_EDGE_TYPES, edge_emb_dim)`; pass `edge_dim=edge_emb_dim` to
  all GATConv layers; update `forward()` to accept and embed `edge_attr`
- `ml/src/models/sentinel_model.py`: Pass `graphs.edge_attr` to `self.gnn.forward()`

### P0-C — Add Architecture Fields to SentinelModel + TrainConfig (30 min)

**Problem:** `gnn_hidden_dim=64`, `gnn_heads=8`, `gnn_dropout=0.2` are hardcoded.
Cannot run a hyperparameter search.

**Changes:**
- `ml/src/models/sentinel_model.py`: Add `gnn_hidden_dim, gnn_heads, gnn_dropout,
  use_edge_attr, gnn_edge_emb_dim` to `__init__()`, pass to `GNNEncoder`; derive
  `node_dim=gnn_hidden_dim` for `CrossAttentionFusion`
- `ml/src/models/gnn_encoder.py`: Accept `hidden_dim: int = 64, heads: int = 8`
  to make conv channels computed (`out_channels=hidden_dim//heads`)
- `ml/src/training/trainer.py` `TrainConfig`: Add all above fields; update model
  construction to pass them all

### P0-D — Align tokenizer.py Schema Version (15 min)

**Change:** Add `FEATURE_SCHEMA_VERSION` as a stored metadata field in the token dict
so token files carry their schema version for debugging and integrity verification.

- `ml/data_extraction/tokenizer.py`: Import `FEATURE_SCHEMA_VERSION` from
  `ml/src/preprocessing/graph_schema` and store it in the result dict

---

## Phase 1 — Close Half-Open Inference Loops

### Move 1 — Complete T1-C: Sliding-Window Predictor (30 min)

`process_source_windowed()` exists in `preprocess.py` but `predictor.py` never calls it.
Long contracts (>512 tokens) have tails silently truncated.

**Files:**
- `ml/src/inference/predictor.py`: Add `_aggregate_window_predictions(probs_list)` →
  `torch.stack(probs_list).max(dim=0).values`; update `predict_source()` to call
  `process_source_windowed()` when token count > 512; add `windows_used` to return dict
- `ml/src/inference/api.py`: Add `windows_used: int` to `PredictResponse`

### Move 2 — T1-A: Inference Cache (2h)

**Files:**
- **New** `ml/src/inference/cache.py`: `InferenceCache(cache_dir, ttl_seconds=86400)`;
  `get(key) → (Data, dict) | None`; `put(key, graph, tokens)`; TTL via `mtime`;
  stores `{key}_graph.pt` + `{key}_tokens.pt`
- `ml/src/inference/preprocess.py`: Optional `cache` in `ContractPreprocessor.__init__()`;
  cache lookup at top of `process_source()`, write on miss
- Reuse: `ml/src/utils/hash_utils.py:get_contract_hash_from_content()`

---

## Phase 2 — Observability + MLOps (Before M6)

| Move | Item | Time | Key Files |
|------|------|------|-----------|
| 3 | T2-A: Prometheus metrics | 1h | `api.py`, `pyproject.toml` |
| 4 | T2-C: MLflow registry script | 2h | `promote_model.py` (new) |
| 5 | T3-A: LLM synthesizer upgrade | 2h | `nodes.py`, `state.py` |
| 6 | T3-B: Cross-encoder reranking | 1h | `retriever.py`, `agents/pyproject.toml` |
| 7 | T2-B: Drift detection | 3h | `drift_detector.py` (new), `compute_drift_baseline.py` (new), `api.py` |
| 8 | Audit items #9/#11/#13 | 1h | `preprocess.py`, `dual_path_dataset.py`, `trainer.py` |

---

## Phase 3 — Doc Restructuring

- `docs/STATUS.md`: Current module completion table + half-open loops
- `docs/ROADMAP.md`: Future work ordered by priority; migrate `Additional skills, tools.md`
- `docs/changes/INDEX.md`: One-line summary per changelog
- Trim `SENTINEL_ACTIVE_IMPROVEMENT_LEDGER_UPDATED_2026-04-28.md` to audit items only

---

## Full Execution Order

```
Phase 0 (pre-retrain — ~3h total):
  P0-A  LoRA config externalized        trainer.py, transformer_encoder.py, sentinel_model.py
  P0-B  GNNEncoder edge_attr support    gnn_encoder.py, sentinel_model.py
  P0-C  Architecture fields exposed     sentinel_model.py, gnn_encoder.py, trainer.py
  P0-D  tokenizer schema version        ml/data_extraction/tokenizer.py

Phase 1 (~2.5h):
  Move 1  Complete T1-C               predictor.py, api.py
  Move 2  T1-A inference cache        cache.py (new), preprocess.py

Phase 2 (~10h):
  Move 3  T2-A Prometheus             api.py, pyproject.toml
  Move 4  T2-C MLflow registry        promote_model.py (new)
  Move 5  T3-A LLM synthesizer        nodes.py, state.py
  Move 6  T3-B cross-encoder rerank   retriever.py, agents/pyproject.toml
  Move 7  T2-B drift detection        drift_detector.py (new), compute_drift_baseline.py (new)
  Move 8  Audit items #9/#11/#13      preprocess.py, dual_path_dataset.py, trainer.py

Phase 3 (~1h):
  Doc-A  docs/STATUS.md
  Doc-B  docs/ROADMAP.md
  Doc-C  docs/changes/INDEX.md
  Doc-D  Trim LEDGER file

→ Retrain with new architecture (P0-A/B/C applied)
→ M6 Integration API sprint
```

---

## Verification Checklist

| Item | Test |
|------|------|
| P0-A | `SentinelModel(lora_r=16)` — LoRA rank change propagates to TransformerEncoder |
| P0-B | `GNNEncoder(use_edge_attr=True).forward(x, ei, batch, edge_attr=t)` runs without error; `edge_attr=None` also safe |
| P0-C | `SentinelModel(gnn_hidden_dim=128)` — CrossAttentionFusion receives `node_dim=128` |
| P0-D | Token `.pt` file contains `feature_schema_version` key |
| Move 1 | 600-token contract → `windows_used >= 2` in prediction response |
| Move 2 | Second call to `process_source()` on same contract returns in < 50ms |
