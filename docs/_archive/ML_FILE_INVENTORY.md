# SENTINEL ML — Complete File Inventory

All Python files in `ml/` excluding `.venv`. Grouped by role.

---

## Production pipeline (active)

### Model architecture
| File | Lines | Role | Doc |
|---|---|---|---|
| `ml/src/models/sentinel_model.py` | ~60 | Top-level model: GNN + Transformer + Fusion + classifier | ML_ARCHITECTURE.md |
| `ml/src/models/gnn_encoder.py` | ~70 | 3-layer GAT, global_mean_pool → [B,64] | ML_ARCHITECTURE.md |
| `ml/src/models/transformer_encoder.py` | ~60 | Frozen CodeBERT CLS token → [B,768] | ML_ARCHITECTURE.md |
| `ml/src/models/fusion_layer.py` | ~40 | concat(832) → MLP → [B,64] | ML_ARCHITECTURE.md |

### Dataset and training
| File | Lines | Role | Doc |
|---|---|---|---|
| `ml/src/datasets/dual_path_dataset.py` | ~364 | Paired graph+token dataset, collate_fn | ML_TRAINING.md |
| `ml/src/training/focalloss.py` | ~35 | Focal Loss for imbalanced binary classification | ML_TRAINING.md |
| `ml/src/training/trainer.py` | ~451 | Full training loop, MLflow, checkpointing, resume | ML_TRAINING.md |

### Inference
| File | Lines | Role | Doc |
|---|---|---|---|
| `ml/src/inference/preprocess.py` | ~480 | Solidity → (graph, tokens) for SentinelModel | ML_INFERENCE.md |
| `ml/src/inference/predictor.py` | ~130 | Loads checkpoint, runs inference | ML_INFERENCE.md |
| `ml/src/inference/api.py` | ~236 | FastAPI POST /predict + GET /health | ML_INFERENCE.md |

### Training scripts
| File | Lines | Role | Doc |
|---|---|---|---|
| `ml/scripts/train.py` | ~90 | CLI entry point for trainer.py | ML_SCRIPTS.md |
| `ml/scripts/tune_threshold.py` | ~289 | Threshold sweep on val set, F1-macro criterion | ML_SCRIPTS.md |
| `ml/scripts/run_overnight_experiments.py` | ~243 | 4-experiment sequential launcher | ML_SCRIPTS.md |

### Shared utilities
| File | Lines | Role | Doc |
|---|---|---|---|
| `ml/src/utils/hash_utils.py` | ~198 | MD5 hash functions for contract identification | ML_DATASET_PIPELINE.md |

---

## Offline data pipeline (run once — data already built)

| File | Lines | Role | Doc |
|---|---|---|---|
| `ml/scripts/ast_extractor_v4_production.py` | ~547 | Slither → PyG Data graphs, 11 workers, checkpointing | ML_DATASET_PIPELINE.md |
| `ml/scripts/tokenizer_v1_production.py` | ~546 | CodeBERT tokenization, 11 workers, checkpointing | ML_DATASET_PIPELINE.md |
| `ml/scripts/create_label_index.py` | ~66 | Scan graphs → label_index.csv (hash→label) | ML_DATASET_PIPELINE.md |
| `ml/scripts/create_splits.py` | ~99 | Stratified 70/15/15 split → .npy indices | ML_DATASET_PIPELINE.md |
| `ml/scripts/fix_labels_from_csv.py` | ~75 | **Historical** — one-time label correction, do not re-run | ML_DATASET_PIPELINE.md |

---

## Tests

| File | Lines | Role | Doc |
|---|---|---|---|
| `ml/tests/conftest.py` | ~27 | FastAPI TestClient fixture, session scope | ML_SCRIPTS.md |
| `ml/tests/test_api.py` | ~120 | Integration tests: /health, /predict, error cases, determinism | ML_SCRIPTS.md |
| `ml/scripts/test_sentinel_model.py` | ~80 | End-to-end: DataLoader → full forward pass | ML_SCRIPTS.md |
| `ml/scripts/test_dataloader.py` | ~50 | DataLoader + collate_fn sanity check | ML_SCRIPTS.md |
| `ml/scripts/test_dataset.py` | ~50 | DualPathDataset sanity check | ML_SCRIPTS.md |
| `ml/scripts/test_fusion_layer.py` | ~40 | FusionLayer forward pass | ML_SCRIPTS.md |
| `ml/scripts/test_gnn_encoder.py` | ~40 | GNNEncoder forward pass | ML_SCRIPTS.md |

---

## Data validation

| File | Lines | Role | Doc |
|---|---|---|---|
| `ml/scripts/comprehensive_data_validation.py` | ~200 | Full data validation before training | ML_SCRIPTS.md |
| `ml/scripts/analyze_token_stats.py` | ~40 | Truncation rate + token length statistics | ML_SCRIPTS.md |
| `ml/analysis/data_quality_validation.py` | ~120 | `DataQualityValidator` class — structural + statistical checks | ML_SCRIPTS.md |

---

## Archived / superseded (do not use in production)

| File | Status | Why not used |
|---|---|---|
| `ml/src/data/graphs/ast_extractor.py` | Stale import | Produces ASTNode/ASTEdge objects (text-based), not PyG Data. Imported in `preprocess.py` but never called. |
| `ml/src/data/graphs/graph_builder.py` | Archived | Produces 17-dim one-hot features. `GNNEncoder` requires 8-dim. Incompatible with trained model. |
| `ml/src/tools/slither_wrapper.py` | Superseded | Full Slither wrapper with vulnerability detection, version management. Not used in training pipeline. |
| `ml/src/tools/slither_wrapper_turbo.py` | Superseded | Faster variant of slither_wrapper. Not used in training pipeline. |
| `ml/src/data/bccc_dataset.py` | Not used | Loads BCCC-SCsVul-2024.csv (241 features). Production uses DualPathDataset (PyG graphs + CodeBERT tokens). |
| `ml/src/data/solidifi_dataset.py` | Not used | SolidiFI vulnerability injection dataset. Not part of production training data. |
| `ml/src/data/validate_dataset.py` | Superseded | Early validation script. Use `comprehensive_data_validation.py` instead. |
| `ml/src/data/validate_solidifi.py` | Not used | Validates SolidiFI dataset. Not relevant to production pipeline. |
| `ml/src/validation/models.py` | Not used in training | Pydantic schemas for Slither result validation. Used during data collection exploration. |
| `ml/src/validation/models_v2.py` | Not used in training | Updated Slither result schemas. |
| `ml/src/validation/statistical_validation.py` | Not used in training | Statistical analysis during data collection. |
| `ml/src/validation/test_full_dataset_final.py` | Superseded | Replaced by `comprehensive_data_validation.py`. |
| `ml/src/validation/test_models.py` | Not used in training | Early model validation scripts. |
| `ml/src/validation/test_real_data.py` | Not used in training | Early real-data validation scripts. |
| `ml/data/archive/old_extractors/ast_extractor_v2.py` | Archived | Superseded by v4 |
| `ml/data/archive/old_extractors/ast_extractor_v3.py` | Archived | Superseded by v4 |
| `ml/data/archive/old_extractors/slither_wrapper_backup_*.py` | Archived | Backup copy |

---

## External data source tooling (not part of training pipeline)

| File | Source | Role |
|---|---|---|
| `ml/data/smartbugs-wild/script/get_contracts.py` | SmartBugs | Downloads .sol contracts from Etherscan |
| `ml/data/smartbugs-wild/script/get_balance.py` | SmartBugs | Checks ETH balance for contract filtering |
| `ml/data/SolidiFI/solidifi.py` | SolidiFI | Injects synthetic vulnerabilities into safe contracts |
| `ml/data/SolidiFI/inject_file.py` | SolidiFI | Per-file vulnerability injection |
| `ml/data/SolidiFI/inspection.py` | SolidiFI | Dataset inspection utilities |
| `ml/data/SolidiFI/evaluator.py` | SolidiFI | Evaluation metrics for injected dataset |
| `ml/data/SolidiFI/performance.py` | SolidiFI | Performance benchmarking |
| `ml/data/SolidiFI/setup.py` | SolidiFI | Setup script |
| `ml/data/SolidiFI-benchmark/scripts/inspection.py` | SolidiFI-benchmark | Benchmark inspection |

---

## Key contracts between pipeline and inference

The following must stay in sync. Any change to one requires changing all others **and retraining the model**:

| Parameter | Where defined | Where used |
|---|---|---|
| Node feature dim = 8 | `ast_extractor_v4_production.py:node_features()` | `preprocess.py:_extract_graph()`, `GNNEncoder(in_channels=8)` |
| Token max_length = 512 | `tokenizer_v1_production.py:MAX_LENGTH` | `preprocess.py:_tokenize(MAX_TOKEN_LENGTH=512)`, `TransformerEncoder` |
| Tokenizer = `microsoft/codebert-base` | `tokenizer_v1_production.py:TOKENIZER_MODEL` | `preprocess.py:TOKENIZER_NAME` |
| Token shape = `[512]` in files | `tokenizer_v1_production.py` (`.squeeze(0)`) | `dual_path_dataset.py` (expects `[512]`) |
| Padding = `max_length` | `tokenizer_v1_production.py:PADDING` | `preprocess.py:_tokenize()` |
| Hash = MD5 of absolute path | `hash_utils.get_contract_hash()` | `preprocess.py:process()` |
