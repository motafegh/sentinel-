# ml/tests — Test Suite

> **Status:** ✅ Current — v9 schema, 14 test files, verified 2026-06-14

Comprehensive test suite for the SENTINEL ML pipeline.

## Purpose

This directory contains pytest-based tests for validating the correctness, robustness, and performance of all ML pipeline components.

## Test Files

### Core Component Tests

- **`test_model.py`** — Model architecture and forward pass validation
  - Forward pass shape verification
  - Auxiliary output testing
  - Output dimension validation [B, 10]
  - Four-eye classifier functionality

- **`test_preprocessing.py`** — Graph schema and feature validation
  - Schema validation (NODE_FEATURE_DIM=12, 14 types)
  - Feature builder correctness
  - CFG feature inheritance
  - Edge type validation (12 types)

- **`test_sentinel_dataset.py`** — Dataset loading and batching
  - SentinelDataset loading from v2 export artifacts
  - Collate function correctness
  - Batch shape validation
  - 5-tuple return verification

- **`test_trainer.py`** — Training pipeline validation
  - TrainConfig parameter validation
  - Loss function correctness (ASL)
  - Gradient flow verification
  - Training loop integration

### Infrastructure Tests

- **`test_cache.py`** — Cache system validation
  - Cache key generation
  - Schema version invalidation
  - Atomic write operations
  - Cache consistency

- **`test_api.py`** — Inference API testing
  - FastAPI endpoint functionality
  - Request/response validation
  - Error handling
  - Performance characteristics

### Specialized Tests

- **`test_gnn_encoder.py`** — GNN encoder specific tests
  - 8-layer GAT architecture (2+3+3 phases)
  - Three-phase attention
  - JK aggregation
  - Edge type handling

- **`test_fusion_layer.py`** — Cross-attention fusion tests
  - Bidirectional cross-attention
  - Node-token interaction
  - Output dimension validation (128)
  - Compile safety verification

- **`test_drift_detector.py`** — Drift detection tests
  - Statistical drift detection
  - Threshold validation
  - Alert generation

- **`test_cfg_embedding_separation.py`** — CFG embedding tests
  - CFG node separation
  - Embedding computation
  - Feature isolation

- **`test_predictor.py`** — Predictor loading and inference tests
  - Checkpoint loading
  - Architecture detection
  - Per-class threshold loading

### Additional Tests

- **`test_promote_model.py`** — Model promotion workflow and checkpoint management

## Configuration

### `conftest.py`
Pytest configuration file with:
- Shared fixtures
- Test data setup
- Common test utilities
- Environment configuration

## Running Tests

### Run All Tests
```bash
cd ml
poetry run pytest tests/ -v
```

### Run Specific Test File
```bash
poetry run pytest tests/test_model.py -v
```

### Run Specific Test Function
```bash
poetry run pytest tests/test_model.py::test_forward_pass -v
```

### Run with Coverage
```bash
poetry run pytest tests/ --cov=ml/src --cov-report=html
```

### Run in Parallel
```bash
poetry run pytest tests/ -n auto
```

## Test Coverage

| Module | Coverage Area |
|--------|---------------|
| `test_preprocessing.py` | Schema, features, CFG inheritance |
| `test_model.py` | Forward pass, aux output, shapes |
| `test_trainer.py` | TrainConfig, ASL loss, gradients |
| `test_cache.py` | Cache key, schema invalidation, atomic write |
| `test_dataset.py` | Dataset loading, collate, batch shapes |
| `test_api.py` | API endpoints, error handling |
| `test_gnn_encoder.py` | GNN architecture, edge types |
| `test_fusion_layer.py` | Cross-attention, compile safety |

## Test Data

Tests use:
- Mock graph data with v9 schema (12-dim features, 12 edge types)
- Synthetic token sequences
- Minimal test contracts
- Cached test fixtures

## Best Practices

When adding new tests:
1. Follow existing test structure
2. Use descriptive test names
3. Include edge cases
4. Test error conditions
5. Add fixtures to `conftest.py` for shared setup
6. Update this README with new test descriptions

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- Fast execution (< 5 minutes)
- No external dependencies
- Deterministic results
- Clear failure messages

## Dependencies

- `pytest` — Test framework
- `pytest-cov` — Coverage reporting
- `pytest-xdist` — Parallel execution
- `torch` — PyTorch for model testing
- `torch-geometric` — PyG for graph testing
