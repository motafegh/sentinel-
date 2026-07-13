# 05 — ML model and inference

**Read this when:** you need to understand the four-eye network, preprocessing, checkpoint loading, HTTP inference, or drift monitoring.

**Skip this if:** you only need to train/evaluate the model; continue with [training and quality](06_ml_training_quality.md).

**Estimated reading time:** 14 minutes.

## 30-second summary

SENTINEL’s teacher is a four-eye multi-label classifier. An eight-layer, three-phase GAT reads the v9 program graph; GraphCodeBERT+LoRA reads four token windows; bidirectional cross-attention creates a 128-dimensional fusion vector; four projected views contribute to ten logits. FastAPI on port 8001 exposes health, prediction, hotspots, and the fusion embedding used by ZKML.

## Just-enough mental model

```text
v9 graph ──→ 8-layer GAT ─┬─→ GNN eye ───────────────┐
                          ├─→ CFG eye ───────────────┤
4×512 tokens → GCB+LoRA ─┼─→ transformer eye ───────┼→ 10 logits
                          └─→ cross-attention fusion ┘
                                      └→ 128-vector ZK boundary
```

“Four eyes” are correlated learned views, not four independent auditors. Their outputs improve diagnosis and attribution but do not create cryptographic independence.

## Actual runtime/source walkthrough

1. [`preprocess.py`](../../ml/src/inference/preprocess.py) — `ml/src/inference/preprocess.py::ContractPreprocessor` compiles source, extracts a v9 graph, creates token windows, and caches by content plus schema version.
2. [`gnn_encoder.py`](../../ml/src/models/gnn_encoder.py) — `::GNNEncoder` applies eight GAT layers across structural, CFG, and propagation phases with edge-type embeddings and JK attention.
3. [`transformer_encoder.py`](../../ml/src/models/transformer_encoder.py) — `::TransformerEncoder` runs GraphCodeBERT with LoRA over flattened windows and returns token embeddings.
4. [`fusion_layer.py`](../../ml/src/models/fusion_layer.py) — `::CrossAttentionFusion` exchanges information between graph nodes and tokens and returns 128 values.
5. [`sentinel_model.py`](../../ml/src/models/sentinel_model.py) — `::SentinelModel` produces GNN, transformer, fusion, and CFG eyes and combines them into ten logits.
6. [`predictor.py`](../../ml/src/inference/predictor.py) — `::Predictor` loads checkpoint/config, validates architecture/schema, loads companion thresholds, hashes the checkpoint, and exposes source-level prediction methods.
7. [`api.py`](../../ml/src/inference/api.py) — `::app` owns request limits, timeouts, error mapping, Prometheus metrics, deterministic startup, and drift updates.

`/fusion-embedding` intentionally returns no verdict. It supplies the exact 128-value proxy input plus graph counts, window count, and teacher checkpoint hash.

## Interfaces, data shapes, and configuration

| Route | Request | Important response |
|---|---|---|
| `GET /health` | none | load state, checkpoint metadata, thresholds, model hash |
| `POST /predict` | `{source_code}` | probabilities, tiers, eye predictions, shapes, model hash |
| `POST /hotspots` | `{source_code}` | per-function GNN hotspots plus prediction summary |
| `POST /fusion-embedding` | `{source_code}` | `fusion_embedding[128]`, graph counts, windows, model hash |

Input graph shapes are `x[N,12]`, `edge_index[2,E]`, `edge_attr[E]`; tokens are `input_ids[4,512]` and mask. Output is ten independent logits/probabilities, not a softmax-exclusive class.

Operational configuration includes `SENTINEL_CHECKPOINT`, `SENTINEL_DRIFT_BASELINE`, `SENTINEL_DETERMINISTIC`, source-size/time limits, device choice, and companion threshold JSON. Environment variable names belong in docs; their values do not.

[`drift_detector.py`](../../ml/src/inference/drift_detector.py) — `::DriftDetector` maintains rolling statistics and performs KS checks only when a real baseline has loaded. A missing/placeholder baseline is degraded monitoring, not “no drift.”

## Failure modes and current limitations

- The teacher checkpoint is not supplied by a fresh clone of this checkout.
- Solidity compiler/Slither failures are request failures and must remain distinguishable from clean predictions.
- GPU OOM maps to a size error after cache cleanup; timeouts map to 504.
- Hotspot scores are model signals, not proof of vulnerable source lines.
- Drift monitoring currently catches selected distribution shifts, not semantic concept drift or accuracy loss without labels.
- The API catches drift-metric update exceptions at debug level; this monitoring path should be reviewed against Rule 5C before production reliance.
- A checkpoint, thresholds file, schema, and architecture must be promoted as one compatibility set.

## Common change recipe

To change inference behavior:

1. Identify whether the change affects preprocessing, architecture, checkpoint, thresholds, or response schema.
2. Preserve training/inference preprocessing parity.
3. For schema/architecture changes, regenerate DATA and retrain; never force-load incompatible weights.
4. Recompute thresholds, calibration, model hash, and drift baseline.
5. Update all consumers: inference MCP, graph inspector, audit submitter, and handbook metadata.
6. Run targeted API tests, the full ML suite, then a live GPU request.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest ml/tests/test_inference_api.py -q   # smoke/targeted
ml/.venv/bin/python -m pytest ml/tests -q                         # module
curl -fsS http://127.0.0.1:8001/health                            # live
```

Current counts are in [current status](16_current_status.md).

## Optional deep references

- [`sentinel_model.py`](../../ml/src/models/sentinel_model.py) — `::SentinelModel.forward`
- [`predictor.py`](../../ml/src/inference/predictor.py) — `::Predictor._load_checkpoint`, `::predict_source`
- [`docs/ml/adr`](../../docs/ml/adr) — architecture decisions, after checking source
- [ZKML boundary](07_zkml.md)
- [Cross-module contracts](11_cross_module_contracts.md)

## Technical mastery layer

### Prerequisite knowledge

Know logits/sigmoid, graph batching/pooling, transformer attention, LoRA, and HTTP validation.

### Source map and reading order

Follow preprocessing → `GNNEncoder.forward` → GraphCodeBERT encoder → `SentinelModel.forward` → `Predictor.predict_source` → FastAPI `predict`/`fusion_embedding`. [T03](technical/03_ml_model_inference_internals.md) annotates the full tensor trace.

### Execution trace and worked example

Tokens `[B,4,512]` and graph nodes `[N,12]` create four `[B,128]` eyes (GNN, transformer, fusion, CFG). Concatenation `[B,512]` feeds ten logits; calibration and per-class thresholds become API decisions. `/fusion-embedding` exposes the 128-value proxy input.

### Implementation practice

[L03](labs/03_ml_tensor_api_trace.md) adds test-only hooks at each eye. Architecture changes must update shape tests, checkpoint restoration, predictor guards, and downstream proxy compatibility.

### Review and ownership check

Can you calculate every tensor width and locate the exact boundary between model score and product verdict?
