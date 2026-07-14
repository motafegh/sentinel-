# SENTINEL D2 ML source-engineering audit appendix

**Audit date:** 2026-07-14
**Runtime baseline:** `4b5bd333c`
**Audit worktree HEAD:** `b8e577fcd` (baseline plus recovered documentation only)
**Scope:** `ml/` production source, ML-facing `data_module` representation source, deployment manifests, tracked model evidence, and `ml/tests`
**Mode:** audit only; no runtime, configuration, test, or model artifact was modified
**Track status:** `track-reproduced`; all accepted P0/P1 items still require independent primary reproduction

## 1. Executive result

The recovered 15-line note correctly reported the suite totals and the order of magnitude of the findings, but did not preserve evidence. This reconstruction independently supports **15 P1 findings and 7 P2/P3 findings**. There is no ML-local P0: the ML defects become cross-system P0 candidates only when another subsystem converts an ML failure into successful-looking deterministic evidence.

The highest-risk ML facts are:

1. the GNN uses the wrong divisor when reconstructing categorical node IDs, shifting the embedding index for every semantic type from `CONTRACT` through `CFG_NODE_OTHER`;
2. training and serving do not apply identical comment removal or padded-window tensors;
3. the active checkpoint is neither obtainable from a clean clone nor authenticated before unsafe pickle deserialization;
4. clean container/host serving lacks required Python dependencies, offline Hugging Face assets, and Solidity compilers;
5. timeout, promotion, and reproducibility controls do not fail closed; and
6. the API describes embedding magnitude as “attention,” while the recorded Run12 temperature calibration is not used by serving.

The ML suite was reproduced exactly: **20 failed, 159 passed, 16 skipped, 22 errors, 6 warnings**. Those results must not be flattened into 42 product defects: 22 errors share one missing checkpoint fixture; 12 of the 13 extractor integration failures share a missing compiler; and several remaining assertions are stale after seam/API changes. Section 4 classifies them.

## 2. Architecture and ownership inventory

### 2.1 Runtime data flow

`Solidity source` → `ContractPreprocessor` → canonical `sentinel_data` graph extractor + online tokenizer → `Predictor` → `SentinelModel` (GNN, GraphCodeBERT/LoRA, four-eye fusion) → per-class probabilities/tiers, hotspots, or 128-dimensional fusion embedding → FastAPI response. Training reads a chunked `SentinelDatasetExport`, reconstructs masks from token IDs, collates graphs/windows, and trains the same model family. Promotion/calibration/drift scripts sit outside the serving process and are only partly enforced by it.

### 2.2 Production source inventory

| Owner | Source surface | Responsibility and persistent/external state | Audit disposition |
|---|---|---|---|
| DATA representation | `data_module/sentinel_data/representation/{graph_schema,graph_extractor,tokenizer}.py` | Canonical 12-feature graph schema, Slither/solc extraction, window tokenizer adapter | P1-01, P1-02, P1-07, P1-12, P2-07 |
| ML data | `ml/src/data_extraction/windowed_tokenizer.py` | Offline GraphCodeBERT windows; comment removal; four-window padding | P1-02, P1-03 |
| ML dataset | `ml/src/datasets/{sentinel_dataset,collate}.py` | Export hash/schema gates, shard LRU caches, label lookup, batch shapes | P1-03, P1-04, P1-06, P2-07 |
| ML model | `ml/src/models/{gnn_encoder,transformer_encoder,fusion_layer,sentinel_model}.py` | Type embedding, GAT routing, LoRA encoder, window pooling, four-eye logits/fusion embedding | P1-01, P1-03, P1-15 |
| ML preprocessing | `ml/src/inference/preprocess.py` | Compiler selection, temporary source files, graph/token cache boundary | P1-02, P1-11, P1-12 |
| ML inference | `ml/src/inference/{predictor,api}.py` | Artifact load, model construction, padding, tiering, endpoints, timeout/error mapping | P1-03–P1-11, P1-13–P1-15; P2-01, P2-04, P2-05, P2-06 |
| ML cache | `ml/src/inference/cache.py` | Disk graph/token cache under user home; TTL and safe loads | P2-03 |
| ML drift | `ml/src/inference/drift_detector.py` | In-memory rolling window; baseline file; KS alert counter | P2-02, P2-05 |
| ML training | `ml/src/training/{trainer,training_logger,focalloss,losses}.py`, `ml/scripts/train.py` | Training loop, optimizer/checkpoint state, structured metrics | P1-04, P1-06, P2-07 |
| ML lifecycle | `ml/scripts/{promote_model,auto_reproducibility_check,set_active_checkpoint,tune_threshold,calibrate_temperature,compute_drift_baseline}.py` | Registry promotion, active-config mutation, calibration and evidence artifacts | P1-04–P1-06, P1-13, P1-14; P2-01, P2-02 |
| Deployment | `ml/deploy/{Dockerfile.inference,docker-compose.yml,prometheus.yml}` | Python/image dependency installation, artifact mounts, GPU/API/Prometheus topology | P1-07, P1-12, P2-05 |

### 2.3 Public interfaces and artifacts

- HTTP: `GET /health`, `POST /predict`, `POST /hotspots`, `POST /fusion-embedding`, and instrumented `GET /metrics` on port 8001.
- Input limit: 1 MiB UTF-8 Solidity text. Advertised compiler series: 0.4–0.8 via five pinned patch choices in source.
- Primary outputs: 10 class probabilities, confirmed/suspicious tiers, optional four-eye clues, model SHA-256, graph sizes, token-window count, hotspots, or fusion vector.
- Persistent inputs: `ml/mlops_config.json`, ignored checkpoint + sibling thresholds, tracked temperature/drift JSON, Hugging Face cache, solc binaries, optional MLflow registry, optional preprocessing cache.
- Training inputs: v1-format `SentinelDatasetExport` with v9 graph schema, labels/shards/splits and artifact hash.
- Truth boundary: the classifier and hotspot outputs are probabilistic clues. The 128-dimensional fusion vector is a teacher-model input to the ZK proxy; neither it nor a proxy proof establishes that a Solidity vulnerability is objectively present.

## 3. Exact evidence and commands

### 3.1 Suite run

The system shell lacks `python`; system `python3` lacks pytest; the root workspace virtualenv lacks PEFT. The reproducible command therefore used the existing ML virtualenv read-only and added the repository DATA package to `PYTHONPATH`:

```text
PYTHONPATH=.:data_module \
  /home/motafeq/projects/sentinel/ml/.venv/bin/python \
  -m pytest ml/tests -q -s \
  > /tmp/d2_ml_pytest_20260714.log 2>&1
```

Result:

```text
217 collected
20 failed, 159 passed, 16 skipped, 6 warnings, 22 errors in 26.67s
```

The raw log is an audit-host temporary artifact, not a canonical repository artifact: `/tmp/d2_ml_pytest_20260714.log`.

### 3.2 Deterministic probes

- Node-ID round trip: computed `round((id / 13) * 14)` for every canonical `NODE_TYPES` member. IDs 7–12 shift to 8–13; ID 13 calculates 14 and is clamped to 13.
- Live temp deletion: held an open `sentinel_prep_*.sol` file, invoked `_purge_orphaned_sentinel_temps`, and observed the pathname deleted while the FD remained valid.
- No-contract schema: validated the predictor's exact `windows_used=0` shape against `PredictResponse`; Pydantic rejected it because the field requires `>=1`.
- Run12 artifact metadata: workstation artifact SHA-256 is `6a220c6b085a8e0b6b8ae8f5b7610d22bee931d56721000d17e3e304b2daa6cb`; its config lacks feature/tokenizer/export digests.
- Reproducibility probe: Run12 stores 347 model parameters under `raw["model"]`, but the checker selected only top-level `rng_state` and `cuda_rng_state`; with no reference or inference it printed `PASS — reproducible` and exited zero.

## 4. Suite-result classification

| Result group | Count | Classification | Evidence/action |
|---|---:|---|---|
| Passing | 159 | Valid partial coverage | Primarily hermetic model, drift, cache, gates, predictor-format, and trainer tests |
| API setup errors | 22 | Environment/artifact gate; supports P1-05 | Session fixture starts real API; ignored Run12 checkpoint is absent |
| Dataset skips | 16 | Missing export coverage; supports P1-05/P2-07 | Fresh worktree has no training export artifact |
| Extractor integration failures | 12 | Environment/compiler gate; supports P1-12 | All fail because `solc` is absent |
| Feature sentinel failure | 1 | Stale seam test, not source defect | Test patches the ML re-export shim, but `_build_node_features.__globals__` belong to canonical DATA module; canonical helper returns `-1.0` as intended |
| CFG embedding failures | 2 | Blocked by missing compiler | Assertions never reach embedding comparison |
| API config failure | 1 | Same artifact issue as 22 errors | Configured checkpoint is absent |
| Auxiliary-key failure | 1 | Stale test | Source intentionally added `fusion_embedding`; expected key set was not updated |
| Promotion failures | 2 | Stale fixtures plus lifecycle risk | Tests do not create newly required behavioral-probe sidecars; P1-13 covers independent fail-open paths |

No failure or skip is counted as a pass. D2 ML acceptance remains blocked on a fresh-clone artifact/compiler acquisition path, then a clean rerun of all 217 tests and separate GPU/container suites.

## 5. P1 findings

### D2-ML-001 — GNN categorical node types are decoded with the wrong divisor

- **Classification / severity / status:** correctness, scientific validity / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/models/gnn_encoder.py::GNNEncoder.forward` (`_type_int`); producer is `data_module/sentinel_data/representation/graph_extractor.py::_build_node_features`.
- **Invariant and evidence:** feature 0 is `type_id / _MAX_TYPE_ID` (`13`), but the GNN multiplies by `_NUM_NODE_TYPES` (`14`). The exhaustive probe maps `CONTRACT→CFG_NODE_CALL`, `CFG_NODE_CALL→CFG_NODE_WRITE`, ..., `CFG_NODE_OTHER→CFG_NODE_ARITH`.
- **Impact / affected modules:** wrong learned embedding rows enter every GAT prediction for IDs 7–12; affects DATA→ML semantics, hotspots, fusion embedding and downstream ZK proxy inputs.
- **Recommendation / rejected alternative:** decode with schema `_MAX_TYPE_ID` and assert exhaustive round-trip. Do not bless/retrain around an undocumented wrong mapping.
- **Compatibility / migration / rollback:** determine whether Run12 training used the same defect; if yes retrain after correction, otherwise restore intended inference. Shadow-compare frozen contracts; retain current image/checkpoint as rollback only.
- **Dependencies / required tests / owner / duplicates:** graph schema, extractor, GNN, checkpoint; exhaustive IDs, frozen logits, retraining provenance; ML Modeling + DATA Representation; no duplicate.

### D2-ML-002 — Training removes comments while serving tokenizes them

- **Classification / severity / status:** train/serve skew / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/data_extraction/windowed_tokenizer.py::tokenize_windowed_contract,_strip_comments`; `ml/src/inference/preprocess.py::ContractPreprocessor._tokenize_sliding_window`.
- **Invariant and evidence:** offline defaults `strip_comments=True`; online passes raw source. Therefore identical source does not produce identical token tensors.
- **Impact / affected modules:** comments change BPE tokens, window boundaries and linspace-selected windows; late vulnerability code can be displaced. GNN sees full source while transformer sees a serving-only distribution.
- **Recommendation / rejected alternative:** one shared string tokenizer and golden tensor equivalence. Documenting the mismatch is insufficient because calibration assumes the training distribution.
- **Compatibility / migration / rollback:** choose the training policy as canonical initially, shadow logits, then recalibrate/retrain if policy changes. Roll back to current tokenizer/image.
- **Dependencies / required tests / owner / duplicates:** tokenizer, DATA orchestrator, preprocessor, checkpoint; commented/Unicode/long-source goldens; ML Data/Serving; no duplicate.

### D2-ML-003 — Training and serving use different padded-window IDs and pool synthetic windows

- **Classification / severity / status:** train/serve skew / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/data_extraction/windowed_tokenizer.py::tokenize_windowed_contract`; `ml/src/inference/predictor.py::_score_windowed,predict_fusion_embedding`; `ml/src/models/transformer_encoder.py::WindowAttentionPooler.forward`.
- **Invariant and evidence:** offline uses RoBERTa pad ID 1; serving creates zero token IDs. Both masks are zero, but `WindowAttentionPooler` receives no real-window mask and softmaxes all four CLS embeddings.
- **Impact / affected modules:** short-contract logits and 128-D fusion embeddings depend on synthetic-window states different from training; proof input reproducibility is affected.
- **Recommendation / rejected alternative:** use tokenizer pad ID and real-window mask in pooling. Attention-mask-only is insufficient because the pooler does not consume it.
- **Compatibility / migration / rollback:** behavior change requires 1–4-window shadow vectors and calibration/retraining decision; keep old path behind a temporary manifest version for rollback.
- **Dependencies / required tests / owner / duplicates:** tokenizer, predictor, transformer pooler, ZKML consumer; tensor/logit/fusion goldens; ML Modeling/Serving; no duplicate.

### D2-ML-004 — Checkpoints and graph shards use unrestricted pickle deserialization

- **Classification / severity / status:** supply-chain/security / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/inference/predictor.py::Predictor.__init__`; `ml/src/datasets/sentinel_dataset.py::_load_graph_shard`; `ml/scripts/promote_model.py::_load_checkpoint_meta`; `ml/scripts/auto_reproducibility_check.py::_model_state_hash`.
- **Invariant and evidence:** each calls `torch.load(..., weights_only=False)` on transported artifacts. PyTorch pickle may execute constructors during load.
- **Impact / affected modules:** substituted checkpoint/export can execute code in serving, training, promotion, or audit contexts.
- **Recommendation / rejected alternative:** safe state tensors/safetensors plus validated JSON and an allowlisted PyG representation. “Trusted directory” is not a control without authenticated acquisition.
- **Compatibility / migration / rollback:** dual-format staging reader; convert known artifact after digest verification. Legacy loader only in an isolated process for an exact allowlisted digest.
- **Dependencies / required tests / owner / duplicates:** artifact store, DATA export, promotion; malicious-pickle negative and format-round-trip tests; ML Platform + Security; related to but not duplicate of D2-ML-005.

### D2-ML-005 — Model identity is computed after load and never authenticated

- **Classification / severity / status:** supply-chain/security / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/inference/predictor.py::Predictor.__init__,_compute_file_hash`; `ml/src/inference/api.py::health`.
- **Invariant and evidence:** unrestricted load, model construction and warmup precede SHA-256 computation; no expected digest/signature is configured or compared. `/health` merely reports the observed digest.
- **Impact / affected modules:** code execution or silent model substitution happens before identity telemetry; operators cannot distinguish approved from unknown artifact.
- **Recommendation / rejected alternative:** verify signed manifest/digest before deserialization. Reporting a self-computed hash is observability, not authentication.
- **Compatibility / migration / rollback:** add approved digest alongside existing path, first warn then enforce in staging/production. Rollback pins the known workstation digest temporarily.
- **Dependencies / required tests / owner / duplicates:** config, artifact store, image admission; wrong-byte/truncated/signature tests; ML Platform + Security; related to D2-ML-004/006.

### D2-ML-006 — The active checkpoint is absent from clean-clone artifact control

- **Classification / severity / status:** reproducibility/deployability / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/mlops_config.json::checkpoint`; `ml/.gitignore`; `ml/src/inference/api.py::lifespan`.
- **Invariant and evidence:** config points to Run12 FINAL, the whole checkpoints directory is ignored, and `git ls-files ml/checkpoints` returns zero—including no DVC pointer. Clean suite yields 22 API errors and one config failure.
- **Impact / affected modules:** API, tests, promotion, container and scientific reproduction depend on a workstation-only 269 MiB file.
- **Recommendation / rejected alternative:** version an OCI/DVC manifest outside the ignored payload path with digest, size and acquisition command. “Run dvc pull” is invalid without a tracked pointer.
- **Compatibility / migration / rollback:** additive manifest first; populate cache and verify before cutover. Existing local file remains emergency rollback.
- **Dependencies / required tests / owner / duplicates:** artifact registry/DVC, CI secrets; fresh clone acquire/verify/offline-start tests; ML Platform; related to D2-ML-005, not duplicate.

### D2-ML-007 — The documented inference container is not clean-build runnable

- **Classification / severity / status:** packaging/deployment / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/deploy/Dockerfile.inference`; root `pyproject.toml`/`poetry.lock`; `ml/pyproject.toml`/lock.
- **Invariant and evidence:** Docker installs the root lock, which lacks PEFT, `prometheus-fastapi-instrumentator` and `sentinel-data`, instead of the ML lock. Source is copied after install. Offline HF mode is enabled although the claimed pre-baked `ml/.venv` is not copied.
- **Impact / affected modules:** model/API imports or GraphCodeBERT load fail before readiness; deployment is host-state dependent.
- **Recommendation / rejected alternative:** use one deploy lock including installable DATA dependency; pin and prefetch HF snapshot; multi-stage import/startup smoke. Bind-mounting an entire developer environment is not reproducible.
- **Compatibility / migration / rollback:** publish parallel immutable image and canary; old host process remains rollback.
- **Dependencies / required tests / owner / duplicates:** package manifests, HF registry, DATA wheel; no-network build/import/readiness/SBOM tests; ML Platform; distinct from compiler D2-ML-012.

### D2-ML-008 — No-contract success cannot serialize through `/predict`

- **Classification / severity / status:** API correctness / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/inference/predictor.py::_format_no_contracts_response`; `ml/src/inference/api.py::PredictResponse,predict`.
- **Invariant and evidence:** predictor returns `windows_used=0`; response model requires `ge=1`. Direct validation reproduced the exact error.
- **Impact / affected modules:** intended structured outcome becomes server-side response-validation failure; AGENTS cannot distinguish no contract from service fault.
- **Recommendation / rejected alternative:** discriminated `no_contract` outcome allowing zero windows, or typed 4xx. Returning fabricated zero probabilities under the normal prediction schema is ambiguous.
- **Compatibility / migration / rollback:** add versioned outcome while preserving normal fields; rollback maps to HTTP 400.
- **Dependencies / required tests / owner / duplicates:** API consumers/AGENTS; endpoint no-contract/interface/library tests; ML Serving; no duplicate.

### D2-ML-009 — Timeout does not cancel work and in-flight inference is unbounded

- **Classification / severity / status:** availability/concurrency / P1 / `track-reproduced`; primary load verification pending.
- **Location:** `ml/src/inference/api.py::predict,hotspots,fusion_embedding`.
- **Invariant and evidence:** each wraps `asyncio.to_thread` in `wait_for`; no semaphore, queue or worker budget exists. Cancelling the await does not terminate thread work.
- **Impact / affected modules:** repeated requests/timeouts accumulate Slither/CPU/GPU work against one model, causing queue growth, OOM and cascading outage.
- **Recommendation / rejected alternative:** bounded admission, explicit 429/503, stage budgets and cancellable subprocesses/cooperative cancellation. A 60-second response timeout alone is not resource control.
- **Compatibility / migration / rollback:** start with one GPU inference and bounded preprocessing pool; tune from measured load. Operator-only bypass is rollback.
- **Dependencies / required tests / owner / duplicates:** API server, Slither, GPU; saturation, timeout-after-work, OOM recovery tests; ML Serving/SRE; no duplicate.

### D2-ML-010 — Hotspot “attention” is unvalidated embedding magnitude

- **Classification / severity / status:** evidence semantics / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/inference/predictor.py::predict_with_hotspots`; `ml/src/inference/api.py::FunctionHotspot,HotspotsResponse,hotspots`.
- **Invariant and evidence:** implementation ranks L2 norm of final node embeddings, then min-max scales it; docs repeatedly call it GNN attention and ground-truth model signal.
- **Impact / affected modules:** AGENTS/RAG may treat a non-causal activation heuristic as attribution; a top node becomes 1.0 even when norm differences are negligible.
- **Recommendation / rejected alternative:** rename/type as `embedding_norm_hotspot` with uncertainty, or validate actual attention/gradient/ablation attribution. Narrative caveats alone do not fix machine-readable semantics.
- **Compatibility / migration / rollback:** deprecate `score`/`attention_source` aliases across one API version; rollback retains old endpoint clearly marked heuristic.
- **Dependencies / required tests / owner / duplicates:** AGENTS graph inspector; invariance, all-equal, ablation correlation tests; ML Interpretability + AGENTS; no duplicate.

### D2-ML-011 — Startup cleanup deletes other live inference inputs

- **Classification / severity / status:** concurrency/filesystem / P1 / `track-reproduced`; primary multi-process verification pending.
- **Location:** `ml/src/inference/preprocess.py::_purge_orphaned_sentinel_temps,ContractPreprocessor.__init__`.
- **Invariant and evidence:** every shared-temp `sentinel_prep_*.sol` is unlinked without PID, age, owner, lock or liveness check. Probe deleted an open live file.
- **Impact / affected modules:** one process startup can remove another process's Solidity file before Slither opens it, causing intermittent failures.
- **Recommendation / rejected alternative:** per-process private temp directory and scoped cleanup. Filename prefix is not proof of orphanhood.
- **Compatibility / migration / rollback:** additive directory layout; disable global purge as immediate rollback.
- **Dependencies / required tests / owner / duplicates:** OS temp/Slither; two-process barrier, crash cleanup tests; ML Serving; no duplicate.

### D2-ML-012 — Clean serving does not provision supported Solidity compilers

- **Classification / severity / status:** deployability/correctness / P1 / `track-reproduced`; primary container verification pending.
- **Location:** `ml/src/inference/preprocess.py::_SOLC_ARTIFACTS,_solc_binary,_make_extraction_config`; `ml/deploy/Dockerfile.inference`.
- **Invariant and evidence:** source expects five versions under `ml/.venv/.solc-select/artifacts`, then falls back to PATH. Clean suite has 12 extractor failures plus 2 blocked CFG tests from missing `solc`; Docker never installs compiler binaries.
- **Impact / affected modules:** every real prediction fails even if model loads; pragma-dependent behavior is not reproducible.
- **Recommendation / rejected alternative:** bake verified compiler inventory and fail readiness if incomplete. Runtime network installation is mutable and unsafe.
- **Compatibility / migration / rollback:** mount existing compiler set during parallel-image migration; rollback to host deployment.
- **Dependencies / required tests / owner / duplicates:** solc-select/Slither/image; representative pragma, range, missing-version tests; ML Platform + DATA Representation; distinct from D2-ML-007.

### D2-ML-013 — Production promotion gates fail open on missing evidence

- **Classification / severity / status:** governance/release safety / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/scripts/promote_model.py::_get_current_production_f1,_check_label_quality,promote,main`.
- **Invariant and evidence:** registry errors become `None` and skip F1 comparison; missing label quality/thresholds only warn; Production baseline is checked only if the optional argument is supplied despite help calling it required.
- **Impact / affected modules:** Production may be promoted without regression comparison, calibration, label-quality evidence or active drift baseline.
- **Recommendation / rejected alternative:** stage policy with mandatory evidence digests; unknown registry/evidence blocks. A dry-run/audit mode should be explicit, not implicit warning behavior.
- **Compatibility / migration / rollback:** enforce in Staging first; archive and retain previous Production version for atomic rollback.
- **Dependencies / required tests / owner / duplicates:** MLflow, artifact manifest, calibration/drift; unavailable-registry/missing-evidence/full-pass tests; ML Platform/Governance; no duplicate.

### D2-ML-014 — Reproducibility check passes without hashing Run12 weights or rerunning inference

- **Classification / severity / status:** scientific reproducibility / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/scripts/auto_reproducibility_check.py::_model_state_hash,run_reproducibility,main`.
- **Invariant and evidence:** Run12 key is `model`, unsupported by hash selection, so only two top-level RNG tensors are hashed; reference F1 is trusted, no benchmark is run, no reference means PASS, and mismatches return zero unless opt-in `--exit-on-fail` is used.
- **Impact / affected modules:** changed weights/predictions can receive a PASS report and promotion evidence is non-probative.
- **Recommendation / rejected alternative:** hash exact `model` state, require signed benchmark/reference, repeat deterministic logits/F1, fail nonzero by default. File hash alone does not test execution reproducibility.
- **Compatibility / migration / rollback:** new report schema/version; mark legacy reports untrusted rather than deleting them.
- **Dependencies / required tests / owner / duplicates:** checkpoint, benchmark export, compiler/tokenizer/environment; mutated-weight/logit/nondeterminism tests; ML Validation; no duplicate.

### D2-ML-015 — Checkpoint does not bind preprocessing and feature provenance

- **Classification / severity / status:** compatibility/reproducibility / P1 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/inference/predictor.py::Predictor.__init__`; Run12 `checkpoint["config"]`; `data_module/sentinel_data/representation/graph_schema.py`.
- **Invariant and evidence:** config includes architecture/hyperparameters/path strings but no feature-schema/vocabulary digest, extractor version, tokenizer revision, comment policy, window policy or export artifact digest. Predictor checks architecture/class order only.
- **Impact / affected modules:** shape-compatible semantic changes silently load; D2-ML-001–003 cannot be prevented at startup.
- **Recommendation / rejected alternative:** signed execution manifest binding all transforms/artifacts. Inferring compatibility from architecture name or tensor shapes is insufficient.
- **Compatibility / migration / rollback:** create a pinned legacy compatibility profile for Run12, require full manifest for new models; rollback selects the legacy profile explicitly.
- **Dependencies / required tests / owner / duplicates:** DATA export, tokenizer, graph schema, model registry, ZKML identity; mismatch rejection and shared Python/manifest vectors; ML Platform + DATA + ZKML; related to D2-ML-005/006, not duplicate.

## 6. P2/P3 findings

### D2-ML-016 — Recorded Run12 calibration is not applied by production serving

- **Classification / severity / status:** calibration/scientific validity / P2 / `track-reproduced`; primary verification pending.
- **Location:** `ml/calibration/*run12*`; `ml/src/inference/predictor.py::_score_windowed,_format_result`; `ml/mlops_config.json`.
- **Invariant and evidence:** README says temperatures are active and evaluations apply them; recorded mean ECE changes 0.195→0.035. Predictor never loads temperatures. Configured threshold path is ignored in favor of a derived sibling and fallbacks are not provenance/range checked.
- **Impact / affected modules:** served probability/tier semantics differ from evaluated reports and AGENTS confidence inputs.
- **Recommendation / rejected alternative:** manifest-bound temperatures applied to logits before sigmoid/threshold; validate complete finite class mapping. Mounting calibration files without consuming them is not integration.
- **Compatibility / migration / rollback:** recalibrate after P1 fixes, shadow responses, expose calibration ID; rollback labels raw output uncalibrated.
- **Dependencies / required tests / owner / duplicates:** calibration set, predictor, AGENTS; API-vs-eval golden; ML Validation/Serving; no duplicate.

### D2-ML-017 — Drift monitoring can become permanently silent-disabled

- **Classification / severity / status:** monitoring / P2 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/inference/drift_detector.py::update_stats,check`; `ml/src/inference/api.py::predict`.
- **Invariant and evidence:** absent/invalid baseline leaves `_baseline=None`; after N requests warmup becomes done, but `check()` returns `{}` forever. API discards results and logs detector failures only at debug.
- **Impact / affected modules:** operators may believe warmup completed while no alert can ever fire.
- **Recommendation / rejected alternative:** explicit warming/baseline-required/active/faulted state machine and readiness/metrics. Logging warmup completion is not activation.
- **Compatibility / migration / rollback:** additive health/metric fields; rollback disables claims rather than fabricating active status.
- **Dependencies / required tests / owner / duplicates:** baseline builder, Prometheus; missing/corrupt/restart/activation tests; ML SRE; no duplicate.

### D2-ML-018 — Inference cache is dead in serving and pair updates are not transactional

- **Classification / severity / status:** caching/performance / P2 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/inference/cache.py::InferenceCache`; `ml/src/inference/predictor.py::Predictor.__init__`.
- **Invariant and evidence:** Predictor constructs `ContractPreprocessor()` without a cache, so advertised cache benefit is unused. If enabled, graph and token files commit separately without pair manifest/lock.
- **Impact / affected modules:** repeated Slither cost persists; crash/concurrency can expose mixed graph/token generations.
- **Recommendation / rejected alternative:** either remove/deprecate or configure a SHA-256, transform-digest-bound, transactional pair cache with metrics/size control. Keeping unowned dormant complexity is not free.
- **Compatibility / migration / rollback:** optional canary enablement; disable cache on any mismatch.
- **Dependencies / required tests / owner / duplicates:** preprocessor/filesystem; concurrent writer, crash-between-renames, eviction tests; ML Serving; no duplicate.

### D2-ML-019 — Execution policy is duplicated across serving paths

- **Classification / severity / status:** maintainability/correctness risk / P2 / `track-reproduced`; primary verification pending.
- **Location:** `Predictor._warmup,_score_windowed,predict_fusion_embedding`; three POST endpoints in `api.py`.
- **Invariant and evidence:** padding/stacking exists three times; request limits/error handling twice or three times; threshold/calibration policy spans config, constants and companions.
- **Impact / affected modules:** fixes can land in prediction but not fusion/proof path, recreating train/serve/proof skew.
- **Recommendation / rejected alternative:** one typed batch builder and inference service operation with modes; one exception mapper. Comments claiming alignment do not enforce it.
- **Compatibility / migration / rollback:** internal refactor with frozen response/logit goldens; revert commit is rollback.
- **Dependencies / required tests / owner / duplicates:** API, predictor, ZKML; predict-vs-fusion preprocessing identity; ML Serving; no duplicate.

### D2-ML-020 — Readiness and metrics cannot support an inference SLO or audit

- **Classification / severity / status:** observability/operations / P2 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/inference/api.py::lifespan,health`; `ml/src/inference/drift_detector.py` metrics.
- **Invariant and evidence:** only loaded flag, current GPU bytes and drift-alert counter are custom metrics. Health omits compiler inventory, manifest approval, calibration/drift state, queue and cache state.
- **Impact / affected modules:** a loaded model can report healthy while real preprocessing fails; overload and post-timeout work are invisible.
- **Recommendation / rejected alternative:** separate liveness/readiness; bounded metrics for stage latency, queue/in-flight, timeouts, OOM, compiler/schema/model/calibration IDs, drift and cache. Logs alone cannot drive reliable alerts.
- **Compatibility / migration / rollback:** additive endpoint/metric fields; preserve `/health` while add `/ready`.
- **Dependencies / required tests / owner / duplicates:** Prometheus/deployment; readiness fault matrix and metric-label tests; ML SRE; no duplicate.

### D2-ML-021 — Performance evidence is not acceptance-grade

- **Classification / severity / status:** performance evidence / P3 / `candidate`; measurement pending.
- **Location:** `ml/reports/Run12_smartbugs_wild_speed_N{100,1000}.json`; absence in `ml/tests`/deployment gates.
- **Invariant and evidence:** tracked reports state about 3.79/s and 2.91/s, but no suite binds latency/VRAM/concurrency numbers to commit, full checkpoint digest, hardware, compiler/cache state or load profile.
- **Impact / affected modules:** capacity, quorum overhead and timeout values cannot be planned safely.
- **Recommendation / rejected alternative:** reproducible cold/warm p50/p95/p99, throughput, VRAM, maximum graph/window, saturation and recovery suite. Anecdotal single-host throughput is insufficient.
- **Compatibility / migration / rollback:** evidence-only addition; no runtime rollback needed.
- **Dependencies / required tests / owner / duplicates:** GPU, checkpoint/compiler acquisition; benchmark protocol above; ML Performance/SRE; no duplicate.

### D2-ML-022 — API and test schemas are weak, inconsistent, and non-hermetic

- **Classification / severity / status:** schema hygiene/test architecture / P2 / `track-reproduced`; primary verification pending.
- **Location:** `ml/src/inference/api.py` response models; `ml/tests/conftest.py` and affected suite files.
- **Invariant and evidence:** label/tier are free strings; threshold/hotspot stats are untyped dicts; hotspot response omits model/threshold/window/edge provenance despite claiming full prediction. One real session fixture makes 22 endpoint tests artifact-bound; compiler/export tests are not separately marked; several assertions are stale after seam/API changes.
- **Impact / affected modules:** clients cannot exhaustively handle outcomes and fresh-clone failures obscure unit regressions.
- **Recommendation / rejected alternative:** versioned enums/discriminated outcomes and shared evidence metadata; hermetic unit tests plus explicit artifact/compiler/GPU/container suites. Skipping everything in clean CI is not a test strategy.
- **Compatibility / migration / rollback:** one deprecation version for v3 fields; test refactor is non-runtime. Rollback preserves old schema adapter.
- **Dependencies / required tests / owner / duplicates:** AGENTS/ZKML clients, CI artifact service; OpenAPI compatibility and suite-marker gates; ML Serving/QA; no duplicate.

## 7. Scientific and performance assessment

- **Leakage and split reproducibility:** the dataset constructor verifies an export artifact hash and reads declared splits, but the required export is absent; D2 cannot independently rerun contamination, split-identity or label-distribution checks from a clean clone. Existing reports are evidence inputs, not a replacement.
- **Calibration:** Run12 temperature evidence exists and reports major ECE improvement, but serving does not apply it. GasException has zero validation support in the threshold report yet receives threshold `0.05` and 1,831 predicted positives; this must be treated as an unsupported class policy, not high-confidence evidence.
- **Teacher/proxy agreement:** ML exposes a 128-D fusion embedding and model hash, but the checkpoint lacks execution-manifest provenance and P1-02/P1-03 can change the vector. ZK proof semantics must bind preprocessing, teacher digest and vector hash; proof still covers proxy computation only.
- **Reproducibility:** deterministic mode seeds PyTorch at API startup, but compiler selection, external artifacts, HF snapshot, preprocessing policy and actual output repetition are not bound. The current reproducibility script is not evidence.
- **Performance:** historical throughput exists; no acceptance-quality latency/VRAM/concurrency probe was possible because the clean worktree lacks checkpoint and compiler. This is a blocked requirement, not a pass or skip.

## 8. Migration and verification order

1. **Stop trust-boundary regressions:** pin/acquire/authenticate the checkpoint and compiler/HF artifacts; replace unsafe loads; fix clean image.
2. **Correct semantics:** fix node-ID decode, unify comment/token padding/pool masks, add execution manifest, then determine retraining/recalibration using shadow vectors.
3. **Correct API evidence:** typed no-contract outcome, hotspot semantics, bounded concurrency and cancellable work.
4. **Fail closed:** repair promotion and reproducibility gates; bind calibration/drift artifacts.
5. **Stabilize operations:** explicit readiness/drift state, cache decision, observability and hermetic suite layers.
6. **Measure:** clean CPU/GPU/container runs, teacher/proxy agreement, calibration, contamination, and performance evidence bound to the final manifest.

Required rollback unit is an immutable bundle—not an independent file set—containing image digest, code commit, checkpoint digest, compiler inventory, HF snapshot, preprocessing/schema manifest, thresholds, temperatures and drift baseline. Rollback must never mix old weights with new transforms.

## 9. Acceptance decision

**ML track verdict: `REVIEW_REQUIRED / IMPLEMENTATION_BLOCKED`.**

The source audit appendix is complete enough for registry normalization, but ML acceptance is blocked until:

- primary audit independently reproduces all 15 P1s or records evidence-backed rejection/downgrade;
- clean artifact/compiler acquisition exists and the full 217-test suite reruns without hidden skips/errors;
- corrected preprocessing/model behavior is bound to a checkpoint/retraining and calibration decision;
- clean container, GPU, performance, drift and teacher/proxy probes are captured; and
- Ali reviews the integrated D2 package.

No runtime fix was made during this track.
