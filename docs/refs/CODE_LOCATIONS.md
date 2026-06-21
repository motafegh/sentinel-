# SENTINEL — Complete Source Code Reference Map

> Single source of truth for all source files across every module.
> Last updated: 2026-06-18

---

## 1. ROOT FILES

```
pyproject.toml             Poetry workspace root
poetry.lock                Locked deps
README.md                  Project README
CLAUDE.md                  Claude configuration & rules
ONBOARDING.md              Plan execution teaching style
report.json                Static analysis report
slither_output.json        Slither analysis dump
.env                       Environment variables
.gitignore / .dvcignore    Ignore rules
```

---

## 2. ML MODULE (`ml/`)

### 2.1 ML — Config & Root

| File | Purpose |
|------|---------|
| `ml/pyproject.toml` | Poetry project config |
| `ml/mlops_config.json` | **Active MLOps config** — checkpoint, thresholds, classes |
| `ml/.env` | Environment vars |
| `ml/CLAUDE.md` | Module-level Claude instructions |

### 2.2 ML — Core Model (`ml/src/models/`)

| File | Purpose |
|------|---------|
| `sentinel_model.py` | **Main model** — GNN + Transformer dual-path + CrossAttentionFusion |
| `gnn_encoder.py` | GNN encoder (GAT-based, JK connections) |
| `transformer_encoder.py` | Transformer encoder (CodeBERT + LoRA) |
| `fusion_layer.py` | Cross-attention fusion layer |

### 2.3 ML — Training (`ml/src/training/`)

| File | Purpose |
|------|---------|
| `trainer.py` | **Training loop** — config, epoch loop, eval |
| `losses.py` | Loss functions |
| `focalloss.py` | Focal loss for class imbalance |
| `training_logger.py` | Structured JSONL logging (StructuredLogger) |

### 2.4 ML — Inference (`ml/src/inference/`)

| File | Purpose |
|------|---------|
| `api.py` | **FastAPI inference server** |
| `predictor.py` | Model prediction + threshold loading |
| `preprocess.py` | Inference-time preprocessing |
| `drift_detector.py` | Data drift detection |
| `cache.py` | Inference cache |

### 2.5 ML — Datasets (`ml/src/datasets/`)

| File | Purpose |
|------|---------|
| `sentinel_dataset.py` | Main dataset class |
| `collate.py` | Custom collation for DataLoader |

### 2.6 ML — Preprocessing (`ml/src/preprocessing/`)

| File | Purpose |
|------|---------|
| `graph_extractor.py` | **28-line shim** → proxies to `sentinel_data/representation/graph_extractor.py` |
| `graph_schema.py` | Schema constants (NODE_FEATURE_DIM=12, NUM_NODE_TYPES=14, etc.) |

### 2.7 ML — Data Extraction (`ml/src/data_extraction/`)

| File | Purpose |
|------|---------|
| `windowed_tokenizer.py` | Windowed/tiled tokenization for long contracts |

### 2.8 ML — Utils (`ml/src/utils/`)

| File | Purpose |
|------|---------|
| `hash_utils.py` | Hashing utilities |

### 2.9 ML — Training Entry & Scripts (`ml/scripts/`)

| File | Purpose |
|------|---------|
| `train.py` | **Training entry point** |
| `promote_model.py` | Model checkpoint promotion |
| `set_active_checkpoint.py` | Set active checkpoint for inference |
| `tune_threshold.py` | Per-class threshold tuning |
| `calibrate_temperature.py` | Temperature scaling (legacy) |
| `compute_drift_baseline.py` | Drift baseline computation |
| `vram_gate_test.py` | VRAM usage gate |
| `compile_smoke_test.py` | Compilation smoke test |
| `auto_reproducibility_check.py` | Reproducibility verification |
| `check_stale_checkpoints.py` | Stale checkpoint detection |
| `check_contamination.py` | Contamination check (legacy v9/v10) |
| `build_warmup_baseline.py` | Warmup baseline builder |
| `session_close.py` | Session close utility |
| `dedup_multilabel_index.py` | Multi-label dedup |
| `benchmark_run9_solidifi.py` | Run9 Solidifi benchmark |
| `benchmark_run9_smartbugs.py` | Run9 SmartBugs benchmark |
| `diag_per_eye_solidifi.py` | Per-eye diagnostic on Solidifi |

### 2.10 ML — Shell Scripts (`ml/scripts/`)

| File | Purpose |
|------|---------|
| `run8_watcher.sh` / `run8_monitor.sh` | Run8 monitoring |
| `run9_launch.sh` / `run9_watcher.sh` | Run9 launch & watch |
| `check_run12_status.sh` | Run12 status check |
| `push_log_snapshot.sh` | Log snapshot push |
| `util/launch_eval.sh` | Launch evaluation |
| `util/watch_smartbugs_eval.sh` | Watch SmartBugs eval |
| `interpretability/run_training_ablation.sh` | Training ablation runner |

### 2.11 ML — Eval Scripts (`ml/scripts/eval/`)

| File | Purpose |
|------|---------|
| `evaluate_run12_on_v0.py` | Run12 on v0.1 benchmark |
| `smartbugs_wild_full_eval.py` | Full 47K eval on SmartBugs Wild |
| `smartbugs_wild_speed_test.py` | Speed test N=100/1000 |
| `verify_wild_predictions.py` | Prediction verification |
| `famous_contracts_test.py` | Famous contracts eval |
| `round_trip_v3.py` | Round-trip v3 inference test |

### 2.12 ML — Smoke Tests (`ml/scripts/smoke/`)

| File | Purpose |
|------|---------|
| `_common.py` | Common utilities |
| `run_all.py` | Run all smoke tests |
| `env_check.py` | Environment check |
| `test_phase_a_final.py` | Phase A final |
| `test_run12_loads_a5.py` | Run12 load test |
| `test_drift_detector_a1.py` | Drift detector test |
| `verify_c21_path.py` | C21 path verification |
| `move_checkpoints_to_archive.py` | Archive old checkpoints |
| `smoke_fix1.py`–`smoke_fix8.py` | Individual fix tests |

### 2.13 ML — Audit Scripts (`ml/scripts/audit/`)

| File | Purpose |
|------|---------|
| `check_contamination_v3.py` | Contamination check v3 |
| `check_contamination_wild.py` | Wild data contamination |
| `analyze_wild_ood.py` | OOD analysis on wild |
| `calibrate_temperature_v3.py` | Temperature scaling v3 |
| `bccc/full_me_audit.py` | Full BCCC audit |
| `bccc/deep_dive_v2.py` | BCCC deep dive v2 |
| `bccc/both_tools_audit.py` | Slither + Aderyn comparison |
| `bccc/verify_2tool_agreement.py` | Tool agreement verify |
| `bccc/edge_cases.py` | BCCC edge cases |
| `bccc/post_audit_analysis.py` | Post-audit analysis |
| `bccc/aderyn_retry.py` | Aderyn retry logic |

### 2.14 ML — Interpretability (`ml/scripts/interpretability/`)

**34 scripts** organized by experiment family:
| Prefix | Count | Scope |
|--------|-------|-------|
| `exp_a1_`–`a4_` | 4 | Pooling audit, CFG inheritance, JK entropy, aux contribution |
| `exp_b1_`–`b4_` | 4 | Gradient norms, per-eye ECE, JK weights, unused-return saliency |
| `exp_e1_`–`e4_` | 4 | Receptive field, WL distinguishability, message prop, direction sensitivity |
| `exp_l1_`–`l10_` | 10 | JK weight, edge ablation, attention vis, gradient saliency, probing, counterfactuals, calibration, permutation, rollout, ablation |
| `exp_s1_`–`s4_` | 4 | Structural trace, edge enrichment, feature dist, ICFG path |
| `val_finding*` | 3 | Validation findings |

### 2.15 ML — Legacy Data Pipeline (`ml/scripts/_legacy_data_pipeline/`)

`validate_graph_dataset.py`, `reextract_graphs.py`, `retokenize_windowed.py`, `create_splits.py`, `create_cache.py`, `build_multilabel_index.py`, `archive_v8_data.py`

### 2.16 ML — Test Contracts (`ml/scripts/test_contracts/`)

20 `.sol` files: `01_reentrancy_classic.sol`–`20_unused_return_minimal.sol`

### 2.17 ML — Tests (`ml/tests/`)

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest fixtures |
| `test_model.py` | Model tests |
| `test_gnn_encoder.py` | GNN encoder |
| `test_fusion_layer.py` | Fusion layer |
| `test_sentinel_dataset.py` | Dataset |
| `test_trainer.py` | Trainer |
| `test_preprocessing.py` | Preprocessing |
| `test_predictor.py` | Predictor |
| `test_api.py` / `test_api_config.py` | API + config |
| `test_cache.py` | Cache |
| `test_drift_detector.py` | Drift detector |
| `test_promote_model.py` | Promotion |
| `test_cfg_embedding_separation.py` | CFG separation |
| `test_framework_gates.py` | Framework gates (36 tests, 91% coverage) |

### 2.18 ML — Testing Specs / Validation (`ml/testing_specs/`)

**Spec docs (`.md`):** `00_rules.md`, `A_benchmark_runs.md`–`L_release_readiness.md` (12 spec docs)

**Framework Python (`framework/`):**
| File | Purpose |
|------|---------|
| `cli.py` | CLI runner |
| `gates.py` | Gate check definitions (9 gates) |
| `config.py` | Framework config |
| `reporters.py` | Report generation |
| `templates/sentinel_v2.yaml` | Gate template |

**Standalone gate scripts:**
`synthetic_probes.py`, `cross_tool.py`, `label_quality.py`, `threshold_sensitivity.py`

### 2.19 ML — Deploy (`ml/deploy/`)

| File | Purpose |
|------|---------|
| `Dockerfile.inference` | Inference API Docker image |
| `docker-compose.yml` | Inference + Prometheus |
| `prometheus.yml` | 15s scrape config |
| `.env.example` | 8 env vars template |

### 2.20 ML — Docker (`ml/docker/`)

`Dockerfile.slither`, `.dockerignore`

### 2.21 ML — Calibration (`ml/calibration/`)

`temperatures_run4.json`, `_run7.json`, `_run9.json`, `_run12.json` (+ `_stats.json`, `_ece_comparison.png` per run)

### 2.22 ML — Audit Docs (`ml/audit_docs/`)

| File | Purpose |
|------|---------|
| `ISSUES.md` | Master issues log |
| `2026-06-17_ml_Run12_testing_spec_suite_gap.md` | Testing gap analysis |
| `2026-06-17_ml_Run12_externalbug_false_positive_root_cause.md` | ExternalBug FP RCA |
| `2026-06-17_testing_suite_overhaul_plan.md` | Overhaul plan |
| `archive/` | 7 legacy audit docs |

### 2.23 ML — Checkpoints (`ml/checkpoints/`)

| File | Purpose |
|------|---------|
| `GCB-P1-Run12-v3dospatched-20260613_FINAL.pt` | **Active production model** (~280MB, DVC-tracked) |
| `*_FINAL.state.json` | Training state |
| `*_FINAL_thresholds.json` | Final thresholds |
| `*_behavioral_probes.json`, `*_cross_tool.json`, `*_reproducibility.json`, `*_stale_checkpoints.json`, `*_threshold_sensitivity.json` | Gate results |
| `_archive/` | 25 historical checkpoints (Run5–Run12) |

### 2.24 ML — Data Directory (`ml/data/`)

| Path | Purpose |
|------|---------|
| `smartbugs-wild/` | SmartBugs Wild contracts (~47K) |
| `smartbugs-curated/` | SmartBugs Curated benchmark |
| `SolidiFI/` | SolidiFI benchmark tools + contracts |
| `SolidiFI-benchmark/` | SolidiFI benchmark repo |
| `SolidiFI-processed/` | Processed SolidiFI variants |
| `augmented/` | Synthetic contracts (110 `.sol`) |
| `drift_baseline_run12.json` | **Active drift baseline** |
| `warmup_run12.jsonl` | Warmup data |

---

## 3. DATA MODULE (`data_module/`)

### 3.1 DM — Config & Root

| File | Purpose |
|------|---------|
| `data_module/pyproject.toml` | Poetry project |
| `data_module/config.yaml` | Pipeline config |
| `data_module/dvc.yaml` | DVC pipeline |
| `data_module/pytest.ini` | Pytest config |

### 3.2 DM — Ingestion (`sentinel_data/ingestion/`)

| File | Purpose |
|------|---------|
| `ingest.py` | Main ingestion orchestrator |
| `freshness.py` | Data freshness checks |
| `manifest.py` | Manifest generation |
| `label_folderize.py` | Organize labels by folder |
| `connectors/base.py` | Abstract connector base |
| `connectors/git_connector.py` | Git source connector |
| `connectors/etherscan_connector.py` | Etherscan API connector |
| `connectors/huggingface_connector.py` | HuggingFace dataset connector |
| `connectors/zenodo_connector.py` | Zenodo connector |
| `connectors/manual_connector.py` | Manual file connector |

### 3.3 DM — Preprocessing (`sentinel_data/preprocessing/`)

| File | Purpose |
|------|---------|
| `pipeline.py` | Main preprocessing pipeline |
| `preprocess.py` | Core preprocessing logic |
| `parallel.py` | Parallel execution |
| `segmenter.py` | Contract segmentation |
| `flattener.py` | Source flattening |
| `compiler.py` | Solidity compilation |
| `normalizer.py` | Source normalization |
| `deduplicator.py` | Deduplication logic |
| `_transitive_strip.py` | Transitive dependency stripping |

### 3.4 DM — Representation (`sentinel_data/representation/`)

| File | Purpose |
|------|---------|
| `orchestrator.py` | Representation pipeline orchestrator |
| `graph_extractor.py` | **Canonical graph extractor** (2056 lines, post-seam-flip) |
| `call_graph.py` | Call graph builder |
| `cfg_builder.py` | CFG builder |
| `pdg_builder.py` | PDG builder |
| `opcode_extractor.py` | Opcode extraction |
| `graph_schema.py` | Schema constants (v9: NODE_FEATURE_DIM=12, etc.) |
| `tokenizer.py` | Tokenization for CodeBERT |
| `cache_manager.py` | Representation cache |
| `versioner.py` | Schema version management |
| `_schema_version_registry.json` | Version registry |

### 3.5 DM — Labeling (`sentinel_data/labeling/`)

| File | Purpose |
|------|---------|
| `merger.py` | Multi-source label merger |
| `gate.py` | Label quality gate |
| `schema/taxonomy.yaml` | Vulnerability taxonomy (10 classes) |
| `parsers/dive.py` | DIVE label parser |
| `parsers/solidifi.py` | SolidiFI label parser |
| `parsers/smartbugs_curated.py` | SmartBugs Curated parser |
| `crosswalks/dive.yaml` | DIVE → Sentinel crosswalk |
| `crosswalks/solidifi.yaml` | SolidiFI → Sentinel crosswalk |
| `crosswalks/smartbugs_curated.yaml` | SmartBugs → Sentinel crosswalk |
| `crosswalks/defihacklabs.yaml` | DeFiHackLabs → Sentinel crosswalk |

### 3.6 DM — Splitting (`sentinel_data/splitting/`)

| File | Purpose |
|------|---------|
| `splitters.py` | Split strategies |
| `dedup_enforcer.py` | Cross-split dedup enforcement |
| `leakage_auditor.py` | Leakage detection |
| `nonvulnerable_cap.py` | Non-vulnerable class cap |

### 3.7 DM — Export (`sentinel_data/export/`)

| File | Purpose |
|------|---------|
| `export.py` | Main export orchestrator |
| `chunker.py` | Dataset sharding |
| `metadata_writer.py` | Manifest/metadata writing |
| `label_writer.py` | Label file writing |
| `graph_writer.py` | Graph serialization |
| `token_writer.py` | Token serialization |
| `format_schema/v1.yaml` | Export format spec |

### 3.8 DM — Verification (`sentinel_data/verification/`)

| File | Purpose |
|------|---------|
| `gate.py` | Verification gate |
| `report_generator.py` | Report generation |
| `negative_checker.py` | Negative class verification |
| `semantic_checker.py` | Semantic consistency |
| `fp_estimator.py` | False positive estimation |
| `tool_validator.py` | Cross-tool validation |
| `class_auditor.py` | Per-class auditor |
| `slither_runner.py` | Slither execution wrapper |
| `probe_trivials.py` | Trivial probe contracts |
| `probe_dataset.py` | Dataset probing |
| `patterns/` | 10 YAML pattern files (Reentrancy, Timestamp, etc.) |

### 3.9 DM — Registry (`sentinel_data/registry/`)

`catalog.py`, `dataset_diff.py`, `lineage_tracker.py`

### 3.10 DM — Analysis (`sentinel_data/analysis/`)

`balance_viz.py`, `cooccurrence.py`, `drift_monitor.py`, `feature_dist.py`, `overlap_detector.py`, `probe_dataset.py`

### 3.11 DM — CLI (`sentinel_data/cli.py`)

### 3.12 DM — Benchmarks (`data_module/benchmarks/`)

| File | Purpose |
|------|---------|
| `evaluate.py` | Benchmark evaluation |
| `contamination_check.py` | Contamination gate |
| `build_benchmark.py` | Benchmark builder |
| `sources/tier_a_existing_ood/build.py` | Tier A builder |
| `sources/tier_b_defihacklabs_heldout/build.py` | Tier B builder |
| `sources/tier_c_bccc_2tool/consensus.py` | Tier C consensus |
| `sources/tier_d_mutation/build.py` | Tier D mutation |
| `sources/tier_d_mutation/patterns/tx_origin.py` | Mutation pattern |
| `sources/tier_e_safe/build.py` | Tier E safe contracts |

### 3.13 DM — Tests (`data_module/tests/`)

**44 test files** organized by subpackage:
- `test_ingestion/` (3 tests) — connector, manifest, label_folderize
- `test_preprocessing/` (2 tests) — pipeline, retry_failed
- `test_labeling/` (7 tests) — merger, taxonomy, gate, parsers, crosswalks
- `test_representation/` (6 tests) — orchestrator, thin_adapter, emits, issue preservation, regression, fixes
- `test_splitting/` (1 test) — splitters
- `test_export/` (5 tests) — export, chunker, writers
- `test_verification/` (11 tests) — patterns, gate, class_auditor, fp_estimator, etc.
- `test_analysis/` (1 test)
- `test_registry/` (1 test)
- `test_integration_*.py`, `test_skeleton.py`

### 3.14 DM — Data Directory (`data_module/data/`)

| Path | Purpose |
|------|---------|
| `raw/` | Raw source data |
| `raw_staging/` | Staging area |
| `preprocessed/` | Preprocessed output |
| `labels/` | Label files |
| `exports/` | v3 export (active: `sentinel-v3-smartbugs-2026-06-13/`) |
| `representations/` | Graph representation cache |
| `splits/` | v3 splits (18,596/1,983/1,914) |
| `registry/` | Dataset registry |
| `verification/` | Verification outputs |
| `analysis/` | Analysis outputs |

---

## 4. AGENTS MODULE (`agents/`)

### 4.1 AG — Config & Root

| File | Purpose |
|------|---------|
| `agents/pyproject.toml` | Poetry project |
| `agents/README.md` | Module docs |
| `agents/AGENTS_STATE_AND_REDESIGN_2026-06-14.md` | State doc |

### 4.2 AG — Orchestration (`agents/src/orchestration/`)

| File | Purpose |
|------|---------|
| `graph.py` | **LangGraph graph definition** — wires all nodes (1,415 lines) |
| `nodes.py` | Node implementations (analyze_code, gen_audit, router, etc.) |
| `state.py` | Graph state schema (typed dict) |
| `routing.py` | Conditional routing logic (phase-0, phase-1) |

### 4.3 AG — LLM Client (`agents/src/llm/`)

`client.py` — OpenAI-compatible LLM client (`get_strong_llm()`, `get_fast_llm()`)

### 4.4 AG — RAG (`agents/src/rag/`)

| File | Purpose |
|------|---------|
| `chunker.py` | Document chunking |
| `embedder.py` | Embedding generation |
| `retriever.py` | Hybrid retriever (FAISS + BM25) |
| `build_index.py` | Index building pipeline |
| `fetchers/base_fetcher.py` | Abstract fetcher base |
| `fetchers/github_fetcher.py` | GitHub source fetcher |

### 4.5 AG — MCP Servers (`agents/src/mcp/servers/`)

| File | Purpose |
|------|---------|
| `inference_server.py` | Inference MCP server |
| `rag_server.py` | RAG MCP server |
| `audit_server.py` | Security audit MCP server |
| `graph_inspector_server.py` | Call-graph DOT inspection MCP server |

### 4.6 AG — Ingestion (`agents/src/ingestion/`)

| File | Purpose |
|------|---------|
| `pipeline.py` | Ingestion pipeline |
| `deduplicator.py` | Seen-hash dedup |
| `feedback_loop.py` | Audit outcome feedback loop |
| `scheduler_cron.py` | Cron-based scheduler |
| `scheduler_dagster.py` | Dagster-based scheduler |

### 4.7 AG — Scripts (`agents/scripts/`)

| File | Purpose |
|------|---------|
| `run_real_audit.py` | **Full real audit harness** (CLI, per-node timing) |
| `smoke_audit_mcp.py` | Audit MCP smoke test |
| `smoke_inference_mcp.py` | Inference MCP smoke test |
| `smoke_rag_mcp.py` | RAG MCP smoke test |
| `smoke_langgraph.py` | LangGraph orchestration smoke test |
| `test_k_cap.py` | K-capture test utility |

### 4.8 AG — Tests (`agents/tests/`)

**10 test files:**
`test_audit_server.py`, `test_chunker.py`, `test_deduplicator.py`, `test_github_fetcher.py`, `test_graph_routing.py`, `test_inference_server.py`, `test_retriever_filters.py`, `test_routing_phase0.py`, `test_smoke_e2e.py`

### 4.9 AG — Test Contracts (`agents/test_contracts/`)

`safe_storage.sol`, `vulnerable_reentrant.sol` (+ 4 `.dot` call-graph files)

### 4.10 AG — Data (`agents/data/`)

| Path | Purpose |
|------|---------|
| `checkpoints.db` | LangGraph checkpoint store |
| `feedback_state.json` | Feedback loop state |
| `index/faiss.index` | FAISS vector index |
| `index/bm25.pkl` | BM25 keyword index |
| `index/chunks.pkl` | Document chunks |
| `index/seen_hashes.json` | Dedup seen hashes |
| `reports/` | Generated audit reports (JSON) |
| `defihacklabs/` | DeFiHackLabs git submodule |

---

## 5. CONTRACTS MODULE (`contracts/`)

### 5.1 CT — Core Contracts (`contracts/src/`)

| File | Purpose |
|------|---------|
| `AuditRegistry.sol` | **On-chain audit registry** |
| `SentinelToken.sol` | SENTINEL token |
| `IZKMLVerifier.sol` | ZKML verifier interface |

### 5.2 CT — Forge Tests (`contracts/test/`)

| File | Purpose |
|------|---------|
| `SentinelToken.t.sol` | Token unit tests |
| `AuditRegistry.t.sol` | Registry unit tests |
| `SentinelTest.t.sol` | General test base |
| `InvariantAuditRegistry.t.sol` | Invariant fuzz tests |
| `mocks/MockZKMLVerifier.sol` | Mock verifier |

### 5.3 CT — Scripts & Standalone

| File | Purpose |
|------|---------|
| `script/Deploy.s.sol` | Deployment script |
| `standalone/ZKMLVerifier.sol` | Standalone verifier impl |
| `standalone/out/Halo2Verifier.bin` | Compiled verifier bytecode |
| `standalone/out/Halo2Verifier.abi` | Verifier ABI |

### 5.4 CT — Config

`foundry.toml`, `remappings.txt`, `.env`, `lib/forge-std/`, `lib/openzeppelin-contracts/`, `lib/openzeppelin-contracts-upgradeable/`

---

## 6. ZKML MODULE (`zkml/`)

### 6.1 ZK — EZKL Pipeline (`zkml/src/ezkl/`)

| File | Purpose |
|------|---------|
| `setup_circuit.py` | EZKL circuit setup |
| `run_proof.py` | Proof generation |
| `extract_calldata.py` | Calldata extraction for on-chain |

### 6.2 ZK — Distillation (`zkml/src/distillation/`)

| File | Purpose |
|------|---------|
| `proxy_model.py` | Proxy model for ZK |
| `train_proxy.py` | Proxy model training |
| `export_onnx.py` | ONNX export |
| `generate_calibration.py` | Calibration data gen |

### 6.3 ZK — Artifacts (`zkml/ezkl/`)

`proof.json`, `witness.json`, `settings.json`, `calibration.json`, `verifier_abi.json`, `srs.params`, `proving_key.pk`, `verification_key.vk`, `model.compiled`

### 6.4 ZK — Models (`zkml/models/`)

`proxy.onnx`, `proxy_best.pt`

---

## 7. OTHER

### Test Contracts (`test_contracts/`)

`simple_reentrancy.sol`

### Root `test` binary

`test` — compiled test executable (ELF binary)

### Top-level config

`pyproject.toml`, `poetry.lock`, `.env`, `.gitignore`, `.dockerignore`, `.dvcignore`
