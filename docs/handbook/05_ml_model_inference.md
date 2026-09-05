# 05 — ML model and inference

**Read this when:** you need to understand the four-eye network, current Run12 inference, HTTP APIs, or the future repaired-checkpoint compatibility boundary.

**Skip this if:** you only need DATA semantics; read [DATA artifacts](04_data_artifacts.md).

**Estimated reading time:** 14 minutes.

## 30-second summary

SENTINEL’s teacher architecture remains a four-eye ten-output model: graph, transformer, fusion, and CFG views produce ten logits and a 128-value fusion embedding. **Run12 is still the historical operational checkpoint served by the current inference stack.** R4 has not promoted a repaired teacher yet. Current inference is therefore useful for runtime continuity and historical comparison, but its learned weights, thresholds, calibration, and historical representation seam must not be treated as the repaired R4 model.

The R4-D-011 V10 V2.6 physical representation is accepted for a possible future repaired run, while D-012 requires a fresh guarded-selector successor token lineage and separate acceptance. Those artifacts do **not** currently power the Run12 inference service.

## Just-enough mental model

```text
current runtime / historical compatibility:
v9-compatible graph + 4×512 tokens
        ↓
Run12 four-eye teacher
        ↓
10 logits + fusion[128]
        ↓
AGENTS evidence / retained proxy seam

separate R4 future-training path:
logical V3 + D-011 V10 graph + pending D-012 guarded-token successor
        ↓
later repaired teacher only after explicit authorization
```

Current serving model:

```text
Run12 checkpoint + historical companion thresholds
= historical operational baseline
≠ DATA-vNext/R4-retrained teacher
```

## Actual runtime/source walkthrough

1. [`preprocess.py`](../../ml/src/inference/preprocess.py) builds the current historical-runtime graph/token inference inputs from source.
2. [`gnn_encoder.py`](../../ml/src/models/gnn_encoder.py) and [`transformer_encoder.py`](../../ml/src/models/transformer_encoder.py) implement structural/code encoders.
3. [`fusion_layer.py`](../../ml/src/models/fusion_layer.py) produces the 128-value fusion representation.
4. [`sentinel_model.py`](../../ml/src/models/sentinel_model.py) produces the four eyes and ten logits.
5. [`predictor.py`](../../ml/src/inference/predictor.py) loads the Run12 checkpoint/config/threshold companions and validates compatibility.
6. [`api.py`](../../ml/src/inference/api.py) exposes health, prediction, hotspots, and fusion embedding.

`/fusion-embedding` returns the teacher’s 128-value fusion vector and checkpoint identity; it does not return or prove an AGENTS verdict.

### R4 meaning of the current checkpoint

Run12 remains intentionally preserved because it is the historical operational baseline against which repaired training can later be compared. It was trained before R4 established that many historical zero cells were unknown/unsupported rather than confirmed negatives.

R4 therefore preserves the **architecture and historical checkpoint for compatibility**, while separately governing the DATA/representation lineage for a future repaired checkpoint. The current required future-training path includes accepted logical V3 authority, D-011 V10 V2.6 physical graphs, and a D-012 guarded-selector successor that has not yet been separately physically accepted. Full repaired training remains unauthorized.

### Runtime versus future representation identity

The current Run12 preprocessing/serving seam is historical compatibility. It should not be silently switched to D-011/D-012 artifacts merely because the four-eye model shapes still fit.

A future repaired model may preserve the same ten outputs and fusion width while changing:

- training/evaluation semantics;
- graph/token lineage;
- checkpoint hash;
- DATA/role identity;
- threshold/calibration availability;
- proxy agreement / V3 model-data identities.

That is a model-version migration even if the HTTP JSON shape remains stable.

## Interfaces, data shapes, and configuration

| Route | Request | Important response |
|---|---|---|
| `GET /health` | none | predictor/checkpoint/threshold/model-hash state |
| `POST /predict` | `{source_code}` | probabilities, tiers, eye predictions, model hash |
| `POST /hotspots` | `{source_code}` | GNN hotspot signals + prediction summary |
| `POST /fusion-embedding` | `{source_code}` | `fusion_embedding[128]`, graph/window counts, model hash |

For the **current Run12 runtime**, graph/token preprocessing remains the historical compatible seam and token shape remains `[4,512]`. The output class order remains the locked ten-class order.

The current predictor may load historical threshold companions for Run12 runtime compatibility. Those thresholds are **not authorized threshold policy for a repaired R4 checkpoint**.

## Failure modes and current limitations

- Current probabilities reflect a model trained on the historical label contract.
- Run12 thresholds/calibration cannot be copied to a repaired model.
- Treating the accepted D-011 V10 lineage as already connected to the live Run12 service is false architecture.
- Treating D-012 selector promotion as already applied to live inference is false; its successor has not been separately accepted.
- Confirmed negatives remain zero, so the repaired path cannot simply repeat historical binary threshold/calibration fitting.
- GasException and UnusedReturn remain output positions even though policy v1 disables repaired supervised training for them.
- An inference response is learned evidence, not proof or ground truth.
- Drift monitoring does not replace outcome-labeled quality evaluation.
- A fresh clone may not contain the Run12 checkpoint.

## Common change recipe

For inference changes:

1. classify whether the change is Run12 historical compatibility or a future repaired-model rollout;
2. preserve the current Run12 preprocessing/checkpoint seam unless an explicit versioned migration changes it;
3. never force-load a checkpoint against different DATA/representation semantics;
4. bind any repaired checkpoint to exact logical roles, graph/token artifact identity, training config, class schema, and evaluation policy;
5. update AGENTS/ZKML/V3 consumers if fusion or model identity/meaning changes;
6. retain Run12 as rollback/comparison evidence rather than overwriting it;
7. update [Architecture](01_architecture.md) and [Cross-module contracts](11_cross_module_contracts.md) when the runtime seam actually changes.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest ml/tests/test_inference_api.py -q
ml/.venv/bin/python -m pytest ml/tests -q
curl -fsS http://127.0.0.1:8001/health
```

Passing inference tests establishes implementation compatibility, not repaired-model quality.

## Optional deep references

- [Architecture](01_architecture.md)
- [ML training and quality](06_ml_training_quality.md)
- [DATA artifacts](04_data_artifacts.md)
- [Cross-module contracts](11_cross_module_contracts.md)
- [Evaluation](13_evaluation.md)
- [ZKML boundary](07_zkml.md)
- [Current status](16_current_status.md)

## Technical mastery layer

### Prerequisite knowledge

Know logits/sigmoid, graph batching, attention, checkpoint identity, thresholding, and the distinction between model architecture, physical input lineage, and learned parameters.

### Source map and reading order

Follow current runtime preprocessing → four-eye model → predictor → API. Then read current R4 status, D-009 logical V3, D-011, D-012, and Phase-8 training mechanics before changing repaired-model semantics. Do not infer the future inference seam from the historical Run12 loader.

### Execution trace and worked example

A current request produces ten Run12 probabilities and fusion `[128]` from the historical checkpoint. Separately, the R4 repair path can later produce a new checkpoint using accepted repaired semantics/representations. That checkpoint needs a new identity and explicit rollout; it cannot inherit Run12 thresholds or become the live model merely because its output shape is still ten values plus fusion `[128]`.

### Implementation practice

Treat the current API as a versioned model-serving boundary. Repaired training changes checkpoint/input/evidence semantics even if HTTP response shapes remain identical.

### Review and ownership check

Can you distinguish the frozen four-eye architecture, current Run12 runtime input/checkpoint seam, accepted R4 physical DATA/representation authority, pending guarded-token successor, and a future repaired checkpoint as separate compatibility/evidence objects?