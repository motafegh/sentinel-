# 05 — ML model and inference

**Read this when:** you need to understand the four-eye network, current Run12 inference, HTTP APIs, or the future retrain compatibility boundary.

**Skip this if:** you only need DATA semantics; read [DATA artifacts](04_data_artifacts.md).

**Estimated reading time:** 14 minutes.

## 30-second summary

SENTINEL’s teacher architecture remains a four-eye ten-output model: graph, transformer, fusion, and CFG views produce ten logits and a 128-value fusion embedding. **Run12 is still the historical operational checkpoint served by the current inference stack.** R4 has not promoted a retrained teacher yet. Therefore current inference is useful for historical comparison/runtime continuity, but its learned weights, thresholds, and calibration reflect the older DATA semantics and must not be treated as the repaired vNext model.

## Just-enough mental model

```text
v9 graph ──→ GNN/CFG views ───────┐
4×512 tokens → GraphCodeBERT ─────┼→ four 128-d views → 10 logits
cross-attention fusion[128] ──────┘
                       ↓
                ZK proxy boundary
```

Current serving model:

```text
Run12 checkpoint + historical companion thresholds
= historical operational baseline
≠ DATA-vNext-retrained teacher
```

## Actual runtime/source walkthrough

1. [`preprocess.py`](../../ml/src/inference/preprocess.py) builds v9 graph/token inference inputs from source.
2. [`gnn_encoder.py`](../../ml/src/models/gnn_encoder.py) and [`transformer_encoder.py`](../../ml/src/models/transformer_encoder.py) implement structural/code encoders.
3. [`fusion_layer.py`](../../ml/src/models/fusion_layer.py) produces the 128-value fusion representation.
4. [`sentinel_model.py`](../../ml/src/models/sentinel_model.py) produces the four eyes and ten logits.
5. [`predictor.py`](../../ml/src/inference/predictor.py) loads the current checkpoint/config/threshold companions and validates compatibility.
6. [`api.py`](../../ml/src/inference/api.py) exposes health, prediction, hotspots, and fusion embedding.

`/fusion-embedding` returns the teacher’s 128-value fusion vector and checkpoint identity; it does not return or prove an AGENTS verdict.

### R4 meaning of the current checkpoint

Run12 remains intentionally preserved because it is the baseline against which repaired retraining will be compared. It was trained before R4 established that many historical zero cells were unknown/unsupported rather than confirmed negatives.

R4 therefore freezes the **architecture**, not the learned weights or historical decision policy. Phase 8 is expected to retrain the existing architecture against the repaired DATA vNext semantics after G7.

## Interfaces, data shapes, and configuration

| Route | Request | Important response |
|---|---|---|
| `GET /health` | none | predictor/checkpoint/threshold/model-hash state |
| `POST /predict` | `{source_code}` | probabilities, tiers, eye predictions, model hash |
| `POST /hotspots` | `{source_code}` | GNN hotspot signals + prediction summary |
| `POST /fusion-embedding` | `{source_code}` | `fusion_embedding[128]`, graph/window counts, model hash |

Graph/tokens remain `x[N,12]` and `[4,512]`. The output class order remains the locked ten-class order.

The current predictor may load historical threshold companions for Run12 runtime compatibility. Those thresholds are **not authorized as threshold policy for a future DATA-vNext retrain**.

## Failure modes and current limitations

- Current probabilities reflect a model trained on the historical label contract.
- Run12 thresholds/calibration cannot be copied to a retrained vNext model.
- R4 Phase 6 has no trustworthy confirmed-negative threshold/calibration role, so Phase 9 cannot simply repeat historical fitting procedures on unknowns.
- GasException and UnusedReturn remain output positions even though vNext policy v1 disables supervised training for them.
- A current inference response is learned evidence, not proof or ground truth.
- Drift monitoring does not replace outcome-labeled quality evaluation.
- A fresh clone may not contain the Run12 checkpoint.

## Common change recipe

For inference changes:

1. classify whether the change is historical Run12 compatibility or future repaired-model behavior;
2. preserve v9/class-order compatibility unless explicitly versioned;
3. never force-load a checkpoint against different DATA/model semantics;
4. bind checkpoint, training config, DATA vNext artifact, class schema, and any future threshold/calibration policy together;
5. update AGENTS/ZKML consumers if fusion or response meaning changes;
6. retain Run12 as rollback/comparison evidence rather than overwriting it.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest ml/tests/test_inference_api.py -q
ml/.venv/bin/python -m pytest ml/tests -q
curl -fsS http://127.0.0.1:8001/health
```

Passing inference tests establish implementation compatibility, not repaired-model quality.

## Optional deep references

- [ML training and quality](06_ml_training_quality.md)
- [DATA artifacts](04_data_artifacts.md)
- [Evaluation](13_evaluation.md)
- [ZKML boundary](07_zkml.md)
- [`docs/plan/ml-R4`](../plan/ml-R4)

## Technical mastery layer

### Prerequisite knowledge

Know logits/sigmoid, graph batching, attention, checkpoint identity, thresholding, and the distinction between model architecture and learned parameters.

### Source map and reading order

Follow preprocessing → four-eye model → predictor → API. Then read R4 Phase 5/6 policy and Phase 8 retraining plan before changing training/inference semantics.

### Execution trace and worked example

A current request produces ten Run12 probabilities and a fusion[128] from the historical checkpoint. Later, the same architecture can be retrained against vNext targets/masks/strength; that new checkpoint must receive a new identity and cannot inherit Run12 thresholds merely because tensor shapes match.

### Implementation practice

Treat the current API as a versioned model-serving boundary. Repaired retraining changes checkpoint semantics even if the HTTP response shape remains identical.

### Review and ownership check

Can you distinguish the frozen architecture, historical Run12 weights, historical thresholds, vNext DATA semantics, and the future retrained checkpoint as five separate compatibility/evidence objects?
