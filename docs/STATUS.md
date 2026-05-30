# SENTINEL — Current Status

**Last updated:** 2026-05-30 (post Phase 1 A1–A5)

---

## Module Summary

| Module | Status | Notes |
|--------|--------|-------|
| M1 ML — models | ✅ DONE | v8 three-eye, 8L GNN, GraphCodeBERT+LoRA+prefix K=48 |
| M1 ML — training | ✅ DONE | Run 4 complete; best checkpoint ep32 F1=0.3362 |
| M1 ML — inference | ✅ DONE | Three-tier output schema; `/health` with model metadata |
| M1 ML — data pipeline | ✅ DONE | 41,576 v8 graphs, 44,470 token files, v8 cache 2.2 GB |
| M2 ZKML | ❌ NOT RUN | Source complete; awaiting stable checkpoint gate |
| M3 MLOps | ✅ DONE | MLflow registry, promote_model.py gate, drift detector |
| M4 Agents — Phase 0 | ✅ DONE | LangGraph topology, SqliteSaver, MCP :8010/:8011/:8012 |
| M4 Agents — Step A–E | ✅ DONE | Three-tier schema, ExternalBug fix, graph_explain, cross_validator |
| M4 Agents — Phase 1 A1 | ✅ DONE | `/hotspots` ML endpoint + `predict_with_hotspots()` in predictor |
| M4 Agents — Phase 1 A2 | ✅ DONE | graph_inspector Phase 2: real GNN embedding-norm hotspots (not Slither proxy) |
| M4 Agents — Phase 1 A3 | ✅ DONE | `quick_screen` Tier-0 node (Slither+Aderyn on every contract, two-signal gate) |
| M4 Agents — Phase 1 A4 | ✅ DONE | Aderyn added to deep-path `static_analysis` node (tool="aderyn" findings alongside Slither) |
| M4 Agents — Phase 1 A5 | ✅ DONE | End-to-end smoke test: 7 tests covering all graph paths (deep/fast/screen-escalated/ML-failure) |
| M5 Contracts | ❌ NOT BUILT | forge never run; contracts/lib/ empty |
| M6 Integration API | ❌ NOT BUILT | No POST /v1/audit endpoint; auth/rate-limit not designed |

---

## Current Best Model

| Field | Value |
|-------|-------|
| Run | GCB-P1-Run4 |
| Architecture | 8-layer GAT, GraphCodeBERT+LoRA r=16, prefix K=48 |
| Best epoch | 32 |
| Val F1-macro (raw) | **0.3362** |
| Checkpoint | `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` |
| Training log | `ml/logs/graphcodebert-p1-run4-20260525.log` |
| Status | Killed ep44 (capacity ceiling); ep32 preserved |
| Prior best | PLAN-3A tuned F1=0.2877 |
| Gain | +0.0485 vs prior best |

**Top classes (ep32):** IntegerUO=0.647, Timestamp=0.329, GasException=0.317  
**Hard classes (ep32):** TOD=0.235, ExternalBug=0.246, CallToUnknown=0.247

---

## Test Suite

**Total: ~226 agent tests + ~86 ML tests**

| Suite | Count | Status |
|-------|-------|--------|
| `agents/tests/` (all) | **219** | ✅ (212 pre-A4/A5 + 7 smoke tests) |
| — `test_routing_phase0.py` | 46 | ✅ routing rules, verdict logic |
| — `test_graph_routing.py` | 59 | ✅ includes quick_screen + routing escalation |
| — `test_smoke_e2e.py` | 7 | ✅ full graph paths (deep/fast/escalated/ML-failure) |
| — `test_audit_server.py` + `test_inference_server.py` | ~30 | ✅ |
| — `test_chunker.py` + `test_retriever_filters.py` + others | ~77 | ✅ |
| `ml/tests/test_model.py` | ~20 | ✅ |
| `ml/tests/test_preprocessing.py` | ~40 | ✅ |
| `ml/tests/test_trainer.py` | ~28 | ✅ |
| `ml/tests/test_api.py` | 18 | ✅ includes /hotspots endpoint tests |

---

## Architecture Summary

### ML Model (v8, three-eye)

```
GNNEncoder (v8+IMP):   3-phase, 8-layer GAT; NODE_FEATURE_DIM=11; NUM_EDGE_TYPES=11
                       Phase 1 (L1+L2): structural edges + IMP-G2 skip Linear(11,256)
                       Phase 2 (L3+L4+L5): CF-only → ICFG-only → CF+ICFG joint
                       Phase 3 (L6+L7+L8): REVERSE_CONTAINS up + CONTAINS down (IMP-G3)
                       JK attention aggregation; per-phase LayerNorm; hidden_dim=256
                       JK entropy regulariser: λ=0.005

TransformerEncoder:    GraphCodeBERT (124M frozen) + LoRA r=16 α=32 on Q+V, all 12 layers
                       Windowed input [B,4,512]; WindowAttentionPooler

CrossAttentionFusion:  attn_dim=256, output=128 LOCKED (ZKML proxy constraint)
                       LayerNorm(768) on token input

GNN Prefix (K=48):     gnn_to_bert_proj Linear(256,768); prefix_type_embedding Embedding(5,768)
                       Warmup suppressed until gnn_prefix_warmup_epochs=15

Three-Eye Classifier:  GNN eye + TF eye + Fused → [B,384] → Linear(384,192) → Linear(192,10)

Schema:                FEATURE_SCHEMA_VERSION="v8"; NUM_CLASSES=10 LOCKED
```

### Agent Topology (Phase 1, A3 complete)

```
START → ml_assessment → quick_screen → evidence_router
                        (Tier 0: always)       │
                                 ┌─────────────┴──────────────────┐
                                 │ deep path                       │ fast path
                    ┌────────────┼───────────────┐                 │
               rag_research  static_analysis  graph_explain        │
                    └────────────┼───────────────┘                 │
                            audit_check                            │
                                 │                                 │
                          cross_validator   (deep path only)       │
                                 │                                 │
                            synthesizer ←────────────────────────-─┘ → END
```

**Two-signal fast-path gate (A3):**
- Fast path: ML safe (all probs < DEEP_THRESHOLDS) AND quick_screen clean (0 High/Critical hits)
- Screen-escalated deep path: ML safe BUT quick_screen fired → `static_analysis + graph_explain`
- Normal deep path: ML flagged → existing fan-out

**MCP servers:**
| Server | Port | Transport | Status |
|--------|------|-----------|--------|
| inference_server | 8010 | SSE | ✅ `/predict` + `/hotspots` |
| rag_server | 8011 | SSE | ✅ |
| audit_server | 8012 | SSE | ✅ |
| graph_inspector_server | 8013 | SSE | ✅ Phase 2 (real GNN embedding-norm hotspots via `/hotspots` endpoint) |

**LM Studio:** Windows host port 1234 (local LLM for agent calls)

---

## ML Inference — Three-Tier Output Schema

```python
{
  "confirmed": ["IntegerUO", "Reentrancy"],          # prob >= 0.55
  "suspicious": ["Timestamp"],                        # 0.25 <= prob < 0.55
  "probabilities": {"IntegerUO": 0.71, ...},          # all 10 classes
  "tier_thresholds": {"confirmed": 0.55, "suspicious": 0.25},
  "label": "confirmed_vulnerable"                     # safe | suspicious | confirmed_vulnerable
}
```

---

## Data

| Asset | Path | Size / Count |
|-------|------|-------------|
| Graphs (schema v8) | `ml/data/graphs/` | 41,576 .pt |
| Token files | `ml/data/tokens_windowed/` | 44,470 .pt, [4,512] each |
| Cache | `ml/data/cached_dataset_v8.pkl` | 2.2 GB, 41,576 pairs |
| Cleaned label CSV | `ml/data/processed/multilabel_index_cleaned.csv` | 44,524 rows |
| Splits | `ml/data/splits/deduped/` | train=29,103 / val=6,236 / test=6,237 |
| MLflow DB | `mlruns.db` | SQLite |
| Agent checkpoints | `agents/data/checkpoints.db` | SqliteSaver |

---

## What Works End-to-End vs What Is Not Yet Integrated

### Works end-to-end

- ML training pipeline: graph extraction → cache → train → tune_threshold → checkpoint
- ML inference: `predictor.py` loads checkpoint, returns three-tier result for a `.sol` file
- ML API: FastAPI `/predict` and `/health` endpoints with three-tier schema
- MLOps: MLflow logging, `promote_model.py` with F1 gate, drift detector smoke test
- Agent orchestration: LangGraph topology compiles and runs with SqliteSaver
- Agent tools: `rag_research`, `static_analysis`, `graph_explain` all call MCP tools
- `cross_validator`: LLM verdicts injected into synthesizer context

### Not yet integrated

- MCP servers are not all running simultaneously in a tested end-to-end flow — each tested individually (A5 smoke test still pending)
- ZKML: source complete but EZKL pipeline never run against any checkpoint
- Contracts: `contracts/lib/` empty; forge build never run
- M6 Integration API (`POST /v1/audit`): does not exist
- Phase 1 remaining: A6 (HIGH_VALUE_RAG_CLASSES routing distinction) — low priority, deferred to Phase 2 prep
- Phase 2 work requires Run 5 data quality fixes first (see ROADMAP.md)

---

## Key File Paths

| Asset | Path |
|-------|------|
| Model | `ml/src/models/sentinel_model.py` |
| GNN | `ml/src/models/gnn_encoder.py` |
| Transformer | `ml/src/models/transformer_encoder.py` |
| Fusion | `ml/src/models/fusion_layer.py` |
| Losses | `ml/src/training/losses.py` |
| Schema | `ml/src/preprocessing/graph_schema.py` |
| Trainer | `ml/src/training/trainer.py` |
| Train script | `ml/scripts/train.py` |
| Predictor | `ml/src/inference/predictor.py` |
| Inference API | `ml/src/inference/api.py` |
| Promote model | `ml/scripts/promote_model.py` |
| Drift detector exercise | `ml/scripts/exercise_drift_detector.py` |
| Agent nodes | `agents/src/orchestration/nodes.py` |
| Agent routing | `agents/src/orchestration/routing.py` |
| Agent state | `agents/src/orchestration/state.py` |
| Agent graph | `agents/src/orchestration/graph.py` |
| Graph inspector server | `agents/src/mcp/servers/graph_inspector_server.py` |
| Inference MCP server | `agents/src/mcp/servers/inference_server.py` |
| Run 4 checkpoint | `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` |
| Run 4 log | `ml/logs/graphcodebert-p1-run4-20260525.log` |
| v4 fallback checkpoint | `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt` (F1=0.5422, leaky) |
