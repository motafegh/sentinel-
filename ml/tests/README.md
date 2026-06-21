# ml/tests — SENTINEL ML Test Suite

Pytest test suite covering model architecture, training, inference, preprocessing, and data integrity.

---

## Files

| File | Tests | What it covers |
|------|-------|---------------|
| `test_model.py` | ~15 | SentinelModel forward pass shapes, aux output, empty batch, prefix path |
| `test_gnn_encoder.py` | ~10 | GNNEncoder shapes, phase routing, JK aggregation, edge masks |
| `test_fusion_layer.py` | ~8 | CrossAttentionFusion shapes, padding, token norm, device mismatch |
| `test_trainer.py` | ~12 | TrainConfig validation, ASL loss, gradient flow, prefix warmup |
| `test_api.py` | 18 | API endpoint tests (predict, health, hotspots, validation) |
| `test_api_config.py` | ~5 | API config loading from mlops_config.json |
| `test_predictor.py` | ~8 | Checkpoint loading, architecture detection, warmup |
| `test_drift_detector.py` | ~8 | DriftDetector warm-up, KS test, Prometheus counter |
| `test_preprocessing.py` | ~10 | Schema constants, feature builders, CFG inheritance |
| `test_sentinel_dataset.py` | ~6 | SentinelDataset loading, collation, batch shapes |
| `test_cache.py` | ~6 | Cache key format, TTL expiry, schema validation, atomic write |
| `test_promote_model.py` | ~4 | MLflow staging promotion |
| `test_framework_gates.py` | ~10 | Testing framework gate validation |
| `test_cfg_embedding_separation.py` | ~4 | CFG vs function node embedding separation |
| `conftest.py` | — | Shared fixtures (model instances, dummy graphs, temp dirs) |

---

## Running Tests

```bash
cd ml
poetry run pytest tests/ -v
```

Or from project root:
```bash
poetry run pytest ml/tests/ -v
```

---

## Test Categories

- **Architecture tests** (`test_model.py`, `test_gnn_encoder.py`, `test_fusion_layer.py`): Verify tensor shapes, module composition, and forward pass correctness
- **Training tests** (`test_trainer.py`): Config validation, loss computation, gradient flow, early stopping
- **Inference tests** (`test_api.py`, `test_predictor.py`, `test_api_config.py`): HTTP endpoints, checkpoint loading, warmup
- **Data tests** (`test_sentinel_dataset.py`, `test_preprocessing.py`): Dataset loading, schema validation, feature correctness
- **Infrastructure tests** (`test_cache.py`, `test_drift_detector.py`, `test_promote_model.py`): Caching, drift monitoring, MLflow integration
- **Framework tests** (`test_framework_gates.py`): Testing framework gate validation

---

## Shared Fixtures (conftest.py)

Common fixtures used across test files:
- Dummy PyG graphs with correct feature dimensions
- Small SentinelModel instances for fast testing
- Temporary directories for checkpoint/cache tests
- Mock Solidity source strings
