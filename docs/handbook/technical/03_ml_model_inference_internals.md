# T03 — Four-eye model and inference internals

## Learning outcome

You can trace tensors through all four eyes, explain prefix injection and auxiliary heads, and follow one `/predict` request from Solidity text to thresholded class results.

## Prerequisites

Read [ML model and inference](../05_ml_model_inference.md). Know tensor dimensions, logits versus probabilities, batching, attention, and graph pooling.

## Source map and reading order

1. `ml/src/preprocessing/preprocessor.py` and data-extraction adapters.
2. `ml/src/models/gnn_encoder.py::GNNEncoder.forward` — phased message passing and JK output.
3. `ml/src/models/codebert_encoder.py` — GraphCodeBERT/LoRA and optional GNN prefix.
4. `ml/src/models/sentinel_model.py::SentinelModel.forward` — four eyes and heads.
5. `ml/src/inference/predictor.py::{__init__,predict_source,_score_windowed}`.
6. `ml/src/inference/api.py::{lifespan,predict,hotspots,fusion_embedding}`.

## Entry point and complete call chain

FastAPI lifespan constructs `Predictor`, which restores architecture/configuration and checkpoint weights. `POST /predict` validates Solidity-like text, calls `Predictor.predict_source`, preprocesses source into a graph and token windows, batches a single graph, runs `SentinelModel.forward`, applies calibrated sigmoid probabilities and per-class thresholds, then builds confirmed/potential/safe results plus model hash. `/fusion-embedding` follows the same teacher boundary but exposes the 128-dimensional embedding consumed by the proxy path.

## Important symbols and configuration

- Input graph: node features `[N,12]`; token IDs/mask `[B,W,L]`, normally four windows of 512.
- GNN returns final and phase-2 node embeddings `[N,256]`.
- Function pooling and CFG pooling each concatenate max+mean `[B,512]` then project to `[B,128]`.
- Transformer window pooling projects `[B,768]` to `[B,128]`; cross-attention fusion yields `[B,128]`.
- Four eyes concatenate to `[B,512]`; classifier maps `512→256→10` raw logits.
- During training, GNN, transformer, fused, and phase-2 auxiliary heads provide additional logits. Inference avoids this branch.

## Annotated source excerpt

Source: `ml/src/models/sentinel_model.py::SentinelModel.forward`

```python
transformer_eye = self.transformer_eye_proj(
    self.window_pooler(token_embs)
)
fused_eye = self.fusion(node_embs, batch, token_embs, flat_mask)
combined = torch.cat(
    [gnn_eye, transformer_eye, fused_eye, cfg_eye], dim=1
)
logits = self.classifier(combined)
```

This is the architecture seam: four independent 128-wide views become one 512-wide decision vector. The returned values are logits; thresholding happens in inference, not inside the model.

## Worked example

For `B=2`, tokens `[2,4,512]` produce flattened attention mask `[2,2048]` and token embeddings `[2,2048,768]`. Suppose 73 total graph nodes produce `[73,256]` embeddings. Batch-aware max/mean pooling returns two rows. Four `[2,128]` eyes concatenate to `[2,512]`; classifier emits `[2,10]`. Sigmoid maps a Reentrancy logit `1.39` to about `0.80`; the result is confirmed only if the configured Reentrancy threshold is at most that probability.

## Success trace

Checkpoint architecture matches instantiated modules; preprocessing yields valid graph/tokens; batch dimensions remain aligned; model returns finite logits; calibration and class thresholds are loaded; API returns all class probabilities and stable model hash.

## Failure trace

Checkpoint/config mismatch fails restoration. Invalid Solidity-like input is rejected by request validation. Empty graph batches return correctly shaped zero logits. Missing function or CFG nodes produce safe zero pooled rows. Hotspot extraction is non-fatal and can return the base prediction without hotspots. Missing local teacher checkpoint makes real inference unavailable in a fresh clone.

## Design reasoning and rejected alternatives

Separate eyes prevent token semantics from erasing structural signals and provide inspectable auxiliary supervision. Function-only pooling avoids dominance by abundant CFG-return nodes; CFG phase-2 pooling creates a direct learning path for control-flow structure. GNN prefix injection lets selected structure influence transformer context. Returning probabilities without class-specific thresholds was rejected because classes have materially different operating points.

## Safe change walkthrough

For a new eye or changed width, first add a synthetic shape test, update constructor/config restoration, forward and auxiliary heads, checkpoint compatibility, predictor loading, and architecture guard tests. For threshold-only changes, do not edit model weights; regenerate validation evidence and threshold artifacts, then run predictor/API behavior tests.

## Guided lab

Complete [L03 — ML tensor and API trace](../labs/03_ml_tensor_api_trace.md).

## Tests and expected results

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp ml/.venv/bin/python -m pytest \
  ml/tests/test_model.py ml/tests/test_predictor.py ml/tests/test_api.py -q
```

Expected prerequisites: synthetic forward and threshold tests run without a production checkpoint; real API behavior requires the configured checkpoint. The current focused command also exposes the existing auxiliary-head expectation mismatch documented in [current status](../16_current_status.md); do not remove `phase2` merely to make the stale assertion green.

## Review questions

Why are logits returned? Which mask is flattened? What distinguishes the GNN eye from CFG eye? Where does a class threshold act? What must change when `eye_dim` changes?

## Ownership checklist

- I can calculate every major tensor shape.
- I can explain ghost/empty graph behavior.
- I can separate model, calibration, threshold, and API responsibilities.
- I can diagnose checkpoint architecture mismatch before changing code.
