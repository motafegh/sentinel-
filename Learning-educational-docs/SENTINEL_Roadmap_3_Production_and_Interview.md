# SENTINEL ML — Roadmap 3 of 3: Production, MLOps & Interview Prep

> **Covers:** Inference API · Cache · Drift detection · Model registry · DVC · Test suite · Interview preparation
> **Weeks:** 7–8 of the 8-week plan
> **Previous:** ← [Roadmap 2: Architecture & Training](SENTINEL_Roadmap_2_Architecture_and_Training.md)
> **Start:** ← [Roadmap 1: Foundations & Data Pipeline](SENTINEL_Roadmap_1_Foundations_and_Data.md)

---

## Quick Reference

**Depth signals:** 🔴 Master (2–8h) · 🟡 Understand (1–2h) · 🟢 Survey (15–30min)

**The Senior's Angle — apply to every file:**
1. Why this architecture and not the alternative?
2. What are the input and output shapes?
3. What would break if this changed?
4. What is this component protecting against?
5. How does this connect to the file I read before?

---

## Phase 6 — Inference: API, Cache, Production Hardening

**Theme:** Full inference path from HTTP request to JSON response.
**Goal:** Describe the complete path from memory; explain every HTTP status code; describe the sliding window algorithm and when it activates.
**Time:** 3–4 hours

---

### Concept Injection — Before Opening Any File

**`asyncio.to_thread` — the one concept that unlocks `api.py`**
FastAPI is async. Model inference (`predictor.predict_source()`) is synchronous and blocking — it can take 1–2 seconds. If you run it directly inside an async endpoint, the entire event loop freezes for that duration — no other requests can be processed. `asyncio.to_thread(fn, *args)` runs `fn` in a thread pool, releasing the event loop while inference runs. Open `api.py` and find this call immediately. The entire async design of the endpoint is explained by this one fact.

**Atomic file writes — why `tmp.rename(dest)` beats `torch.save(obj, dest)`**
`torch.save(obj, dest)` writes directly to the destination. If the process is killed mid-write, the destination file is partially written and corrupted — future cache reads silently get bad data. `tmp.rename(dest)` writes to a temp file first, then renames it to the destination. Rename is atomic at the OS level — it either completes fully or not at all. No partial state is possible. Open `cache.py` and find the `tmp.rename()` call. This is the correct production pattern for any file-based caching.

---

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/inference/predictor.py` | 🔴 Master | Checkpoint loading; warmup; sliding window; per-class thresholds |
| `ml/src/inference/cache.py` | 🔴 Master | Atomic writes; TTL; schema-version invalidation |
| `ml/src/inference/api.py` | 🔴 Master | FastAPI lifespan; async threading; error codes; Prometheus |
| `ml/src/inference/preprocess.py` | 🟡 Understand | Online extraction; temp file lifecycle; cache integration |
| `docker/Dockerfile.slither` | 🟢 Survey | Slither runs in Docker for graph extraction; skip Dockerfile syntax |
| `ml/tests/test_api.py` | 🔴 Master | /health + /predict schema; determinism; error codes |
| `ml/tests/test_cache.py` | 🟡 Understand | miss/hit/TTL/schema-version |

### Questions to answer

**predictor.py:**
- `_warmup()` uses a 2-node 1-edge graph (Audit Fix #5). What would NOT be exercised by the 0-edge warmup? What bug type would only surface at the first real request?
- If `{checkpoint.stem}_thresholds.json` is missing, all classes fall back to 0.5. What is the concrete production risk given DenialOfService pos_weight=68?
- Sliding window uses `max` aggregation. With a 1200-token contract with `withdraw()` at token 800+: why is `max` correct and `mean` wrong?

**cache.py:**
- `tmp.rename(dest)` vs `torch.save(obj, dest)` directly. What failure mode does atomic rename prevent?
- `get()` validates `graph.x.shape[1] == 8` on every hit. Under what sequence of events could the correct schema-versioned key still contain a graph with the wrong feature dimension?
- What is the TTL and what production scenario does it protect against?

**api.py:**
- `/predict` uses `asyncio.wait_for(asyncio.to_thread(...))`. Why must `to_thread()` be used? What happens to the event loop if inference runs synchronously?
- The `must_look_like_solidity` validator checks for `pragma` or `contract`. Security gate or UX convenience? Where is the real security boundary?
- Prometheus metrics: what gauges/counters are exposed, and what alert rule would you write for each?

### Code Directing Exercise

Write the prompt you would give an AI to generate the `/predict` endpoint in `api.py`. Your prompt must specify: why `to_thread` is required, why `wait_for` wraps it (timeout protection), what Pydantic validation must happen before inference, what error codes map to which failure types, and what Prometheus metrics must be updated after each prediction. A correct prompt means you own the inference API design.

### Teach-Back Exercise

Walk through one API request for a 1500-token contract: Pydantic validation → size check → `predict_source()` → `process_source_windowed()` → graph extraction (including temp file) → sliding window tokenisation (stride=256, max_windows=8) → three forward passes → `max` aggregation → per-class threshold application → JSON response → drift detector update → Prometheus gauge update.

---

## Phase 7 — MLOps: Drift, Registry, Data Versioning

**Theme:** The operational layer that keeps the model honest after deployment.
**Goal:** Explain the warm-up drift strategy, walk through model promotion, understand DVC data versioning.
**Time:** 3–4 hours

---

### Concept Injection — Before Opening Any File

**`scipy.stats.ks_2samp` — the entire statistical machinery you need**
The KS test takes two arrays of samples (e.g., prediction confidence scores from training data vs recent production requests) and returns a statistic and a p-value.

- The statistic = maximum difference between the two empirical CDFs
- The p-value answers: "if these two samples came from the same distribution, how likely is this level of difference?"
- Low p-value (< 0.05) = the samples are unlikely to come from the same distribution = drift detected

That is the complete statistical model behind `drift_detector.py`. Open the file and find the `ks_2samp` call. Read the parameters and confirm you can identify which arrays correspond to the baseline and which to the live production window.

**DVC mental model — before reading any DVC-related file**
Git tracks code. DVC tracks data. The link between them is the `.dvc` pointer file.

```
ml/data/graphs.dvc  ← lives in git, contains a hash of the graphs/ directory
ml/data/graphs/     ← lives in DVC remote storage, NOT in git
```

When you `git checkout` a commit, the `.dvc` files change to point to the data version that existed at that commit. Then `dvc pull` downloads that exact data version. This is how reproducibility works — code + data are pinned together.

Commands to own:
- `dvc status` — are local files in sync with `.dvc` pointers?
- `dvc pull` — download data to match current git commit
- `dvc push` — upload new data after retraining
- `git add *.dvc && git commit` — pin new data version to the code commit

---

### Files

| File | Depth | What it teaches |
|------|-------|----------------|
| `ml/src/inference/drift_detector.py` | 🔴 Master | KS test; warm-up suppression; rolling buffer; Prometheus |
| `ml/scripts/compute_drift_baseline.py` | 🟡 Understand | Warm-up vs training source; why 30-record minimum |
| `ml/scripts/promote_model.py` | 🟡 Understand | MLflow registry; archive-on-Production; checkpoint validation |
| `ml/scripts/validate_graph_dataset.py` | 🟡 Understand | Pre-retrain gate; edge_attr shape check |
| `ml/tests/test_drift_detector.py` | 🟡 Understand | Warm-up suppression; KS fires on drift; rolling buffer eviction |
| `ml/tests/test_promote_model.py` | 🟢 Survey | Dry-run; stage validation; MLflow tag stubs |

### Deep study: DVC data versioning 🔴

**What DVC solves:**
Git cannot version 68K `.pt` files (~3.4 GB total). DVC tracks large binary artifacts via `.dvc` pointer files in git and actual data in remote storage.

**In this project specifically:**
- DVC remote: `/mnt/d/sentinel-dvc-remote` (Windows D drive via WSL2)
- Tracked: `ml/data/graphs/`, `ml/data/tokens/`, `ml/data/processed/`, `ml/data/splits/`, `ml/checkpoints/`
- If a colleague pulls the repo and runs `dvc pull`, they get the exact 68,523 graph files you trained on

**Why not git LFS?** Cost, performance at 68K files, no content-addressing. DVC is purpose-built for ML data versioning.

### Questions to answer

**drift_detector.py:**
- Using BCCC-2024 training corpus as the KS baseline causes `sentinel_drift_alerts_total` to fire constantly on 2026 DeFi contracts. Explain precisely why, and what that makes the alerting.
- KS test fires at `p < 0.05`. What does the p-value represent here, and why does a low p-value NOT tell you which direction the distribution shifted?
- The rolling buffer has a fixed size. What is evicted and why is FIFO the right eviction policy?

**promote_model.py:**
- `archive_existing_versions=(stage == "Production")` archives old Production but not Staging. What production incident does this prevent?
- If a developer promotes a `legacy_binary` checkpoint (num_classes=1), trace the failure to the specific check in `predictor.py` that surfaces it.

**DVC:**
- A teammate regenerates 500 graph files with a new feature engineering change and runs `dvc push`. What must happen in git to make this reproducible for everyone else?
- You roll back to a previous git commit. What DVC command makes the data match that commit?

### Code Directing Exercise

Write the prompt you would give an AI to generate `drift_detector.py`. Your prompt must specify: what statistical test is used and why, why the baseline must come from warm-up requests and not the training corpus, what the rolling buffer eviction policy is and why, what Prometheus metrics are exposed, and what warm-up suppression means (no alerts during the first N requests after deployment). A correct prompt means you own the drift detection design.

### Teach-Back Exercise

Describe the complete MLOps cycle for a retrain:
`validate_graph_dataset.py` → training with MLflow + WandB → `tune_threshold.py` → `promote_model.py --stage Staging` → warm-up period (500 real requests) → `compute_drift_baseline.py --source warmup` → `promote_model.py --stage Production`.
Explain `--dry-run` and when you use it. Explain what `dvc push` does after the checkpoint is saved.

### Beyond the Codebase: CI/CD and Advanced Deployment Patterns 🟡

SENTINEL does not implement CI/CD or advanced deployment strategies, but these are tested at senior levels. Be able to describe them:

**CI/CD for ML:**
- A complete CI pipeline for this project would run `pytest ml/tests/ -v`, `dvc status` (data in sync?), and a smoke test calling `/predict` with a known contract after building the Docker image.
- CD would trigger on a new checkpoint being promoted to Staging via `promote_model.py`, run integration tests against the Staging API, then promote to Production automatically if they pass.

**Shadow mode / canary deployment:**
- Shadow mode: route real traffic to both old and new model; log both predictions; compare offline before any live cutover. Detects silent regressions without user impact.
- Canary: route 5% of real traffic to the new model, monitor Prometheus metrics (latency, error rate, drift alerts) before full rollout. More aggressive than shadow but still controlled.
- SENTINEL's `promote_model.py` stages (None → Staging → Production) are the mechanism, but they don't implement shadow mode or traffic splitting — that would require an API gateway (e.g., Nginx, Envoy) or a feature flag system.

**When asked in an interview:** "The current system uses MLflow stages for controlled promotion, but for a higher-stakes deployment I'd add shadow mode inference — running both old and new models in parallel, logging predictions, and comparing aggregate metrics over 24 hours before promoting."

---

## Phase 8 — The Test Suite as a Learning Tool

**Theme:** Tests encode the intended contract for each component. Read each test file BEFORE the source it tests — tests are the spec.
**Goal:** Understand the testing strategy well enough to write a new test for any component.
**Time:** 2–3 hours total (spread across phases — read each test file alongside its phase)

### Concept Injection — Before Reading Any Test File

**Read `conftest.py` first — it is the fastest way to learn input/output contracts**
`conftest.py` contains shared fixtures that define valid input shapes for every component. Before reading any test, read `conftest.py`. The synthetic graph it generates tells you exactly what `GNNEncoder` expects. The synthetic token tensors tell you what `TransformerEncoder` expects. This is the fastest path to shape fluency in the entire codebase.

**Run `pytest --cov=ml/src --cov-report=term-missing` after each phase**
Coverage reports show which lines in source files have no test coverage — those are the complex branching paths to trace manually. Any uncovered branch in `trainer.py` or `fusion_layer.py` is a risk you need to reason about by hand.

### Complete test file map

| File | Depth | Read with phase | What it confirms |
|------|-------|-----------------|-----------------|
| `ml/tests/conftest.py` | 🔴 Master | Phase 1 | Shared fixtures; fastest way to learn input/output contracts |
| `ml/tests/test_gnn_encoder.py` | 🔴 Master | Phase 3 | edge_attr shapes; graceful degradation; head-divisibility |
| `ml/tests/test_fusion_layer.py` | 🔴 Master | Phase 4 | Masked pooling correctness; attn_dim divisibility; device detection |
| `ml/tests/test_model.py` | 🟡 Understand | Phase 4 | Full forward pass with stub TransformerEncoder |
| `ml/tests/test_preprocessing.py` | 🟡 Understand | Phase 2 | Mocked Slither + CodeBERT; temp file handling |
| `ml/tests/test_dataset.py` | 🟡 Understand | Phase 2 | Pairing; split indices; collate function; binary vs multi-label label shapes |
| `ml/tests/test_trainer.py` | 🟡 Understand | Phase 5 | pos_weight; evaluate(); FocalLoss BF16 fix |
| `ml/tests/test_api.py` | 🔴 Master | Phase 6 | /health + /predict schema; determinism; error codes |
| `ml/tests/test_cache.py` | 🟡 Understand | Phase 6 | miss/hit/TTL/schema-version |
| `ml/tests/test_drift_detector.py` | 🟡 Understand | Phase 7 | Warm-up suppression; KS fires on drift; rolling buffer eviction |
| `ml/tests/test_promote_model.py` | 🟢 Survey | Phase 7 | Dry-run; stage validation; MLflow tag stubs |

### Key testing patterns to master

**Stub TransformerEncoder (`test_model.py`):** The model test doesn't load actual CodeBERT — it is 500MB. The stub returns random tensors of the correct shape `[B, 512, 768]`. This lets you test GNN, fusion, and classifier without any HuggingFace dependency. This pattern — replacing expensive components with shape-correct stubs — is standard in production ML testing.

**Mocking Slither (`test_preprocessing.py`):** Slither requires a full compiler + analysis pass (3–5 seconds per contract). Tests mock `subprocess.run` to return pre-built PyG objects. This isolates the unit being tested from the tool dependency.

**Parametrised device testing:** `pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")` lets the CI suite pass on CPU-only machines.

---

## What to Skip or Fast-Forward

| File | Decision | Reason |
|------|----------|--------|
| `ml/scripts/run_overnight_experiments.py` | 🟡 Understand (template only) | Convenience wrapper; use as experiment management template |
| `ml/scripts/create_label_index.py` | 🟢 Skip | Documented obsolete in STATUS.md |
| `ml/scripts/analyse_truncation.py` | 🟡 Understand (output only) | Read output section only; confirms 96.6% truncation |
| `ml/tests/test_promote_model.py` | 🟢 Survey | MLflow stubs; understand the pattern, not the implementation |
| `docker/Dockerfile.slither` | 🟢 Survey | Know Slither exists and runs in Docker; skip Dockerfile syntax |
| All `__init__.py` files | 🟢 Survey | Quick scan for namespace exports only |
| `poetry.lock` | 🟢 Survey | Know what Poetry does; skip lock file internals |

---

## Interview Question Bank

Practice answering out loud, timed under 2 minutes each. The answer lives in the anchored file(s) noted.

### Part A — SENTINEL-Specific Questions (25)

| # | Question | Anchor |
|---|----------|--------|
| Q1 | Walk me through SENTINEL architecture end-to-end. | `sentinel_model.py` — all 4 components |
| Q2 | Why does the GNN path not pool before fusion? What specifically would you lose? | `gnn_encoder.py` docstring + `fusion_layer.py` |
| Q3 | Explain LoRA mathematically. Why r=8 and not r=64? What is the general principle for rank selection? | Phase 0 Topic C + `transformer_encoder.py` |
| Q4 | Why BCEWithLogitsLoss and not CrossEntropyLoss? | Phase 0 Topic E + `trainer.py` |
| Q5 | There's a production bug: inference scores are worse than val metrics. Same checkpoint. Where do you look first? | `graph_schema.py` CHANGE POLICY + `cache.py` + two hash systems |
| Q6 | What is pos_weight and when would you switch to FocalLoss? | `trainer.py` compute_pos_weight() + `focalloss.py` |
| Q7 | How does the inference cache work, and what prevents it serving stale data after a feature change? | `cache.py` + `preprocess.py` — content hash + schema version |
| Q8 | Your model has Dropout. How do you ensure inference is deterministic? | `predictor.py` `model.eval()` + `test_api.py` determinism test |
| Q9 | A user submits a 3000-token contract. Walk me through exactly what happens. | `predictor.py` `predict_source()` + `preprocess.py` windowing |
| Q10 | We want to add an 11th vulnerability class. Minimum change required, and what breaks? | `trainer.py` CLASS_NAMES + `predictor.py` Bug 5 fix |
| Q11 | Why is the KS drift baseline built from warm-up requests rather than training data? | `drift_detector.py` + `compute_drift_baseline.py` |
| Q12 | Explain the GAT edge attribute mechanism. What does SENTINEL's GATConv see that a GCN doesn't? | `gnn_encoder.py` + `graph_schema.py` EDGE_TYPES |
| Q13 | An engineer re-extracts a subset of contracts but skips `validate_graph_dataset.py`. What happens at training time? | `validate_graph_dataset.py` + `gnn_encoder.py` graceful degradation |
| Q14 | `OneCycleLR` was resuming incorrectly after a checkpoint. Describe the bug and fix. | `trainer.py` Audit Fix #8 |
| Q15 | What does the startup warmup protect against, and why did the 0-edge warmup fail? | `predictor.py` `_warmup()` Audit Fix #5 |
| Q16 | How do you version 68K binary `.pt` files alongside your code? | DVC section in Phase 7 |
| Q17 | Why did you choose BF16 over FP16 for mixed-precision training on this hardware? | Phase 5 AMP section + `trainer.py` |
| Q18 | Explain `Batch.from_data_list()`. Why can't you just `torch.stack()` PyG graphs? | Phase 3 Concept Injection — PyG Batch mechanics |
| Q19 | The original code had `SentinelModel(num_classes=1)` as the default. Walk me through the failure this caused. | `sentinel_model.py` Fix #3 |
| Q20 | I see this codebase was AI-assisted. Walk me through the bugs you found and fixed in the fusion layer. | Cross-Cutting §5 audit fixes narration |
| Q21 | Why is F1-macro the early stopping metric and not accuracy or F1-micro? | Cross-Cutting §6 Evaluation Metrics |
| Q22 | Explain what `build_multilabel_index.py` does. Why GROUP BY SHA256 with max()? What does WeakAccessMod exclusion tell you? | Phase 2 + `build_multilabel_index.py` |
| Q23 | Walk me through what happens to a Reentrancy contract from raw Solidity to the classifier's sigmoid output. Use concrete node/edge/token examples. | Phases 0 + 2 + 3 + 4 |
| Q24 | The model is deployed. In 6 months, DeFi contracts look very different from your training set. How do you detect and respond? | `drift_detector.py` + `promote_model.py` + Phase 7 |
| Q25 | Describe SENTINEL's per-class threshold tuning. Why 0.5 is wrong for every class, and how the tie-breaking rule reflects the purpose of the system. | `tune_threshold.py` + Phase 5 |

---

### Part B — Generic Senior ML Questions (7)

These are the questions companies ask BEFORE going into your specific project. Practice these first.

| # | Question | What to cover |
|---|----------|--------------|
| Q26 | Explain backpropagation and the chain rule. What breaks in very deep networks without residual connections? | Gradient flow; vanishing gradient; how residuals provide identity shortcuts |
| Q27 | What is the difference between batch normalisation and layer normalisation? When would you use each? | BatchNorm: normalises across batch dimension (bad for small batches, RNNs); LayerNorm: normalises across feature dimension (preferred in Transformers and GNNs where batch stats are noisy) |
| Q28 | Walk me through the transformer self-attention mechanism mathematically. What is the role of the scaling factor 1/√dk? | QKV; softmax(QKᵀ/√dk)V; scaling prevents dot products from growing too large in high dimensions, which causes vanishing gradients through softmax |
| Q29 | What is data leakage? Give three ways it can happen in an ML pipeline without the team noticing. | Feature computed on full dataset before split; target encoding using test set statistics; temporal data shuffled before split; evaluation metric computed on data used for hyperparameter tuning |
| Q30 | How do you approach debugging a model that trains well but fails badly in production? | Distribution shift (train/prod mismatch); feature drift; preprocessing difference between training and inference; label quality; threshold calibration; check production input distribution first |
| Q31 | Explain the bias-variance tradeoff. In a 10-class multi-label classification with extreme class imbalance, which is the more common failure mode and why? | High bias (underfitting) on rare classes — the model learns to predict the majority class (safe contract) almost always; this is not "high variance" but systematic suppression of minority classes; addressed by pos_weight / FocalLoss / oversampling |
| Q32 | What is a feature store? Have you worked with one or something equivalent? | Centralised repository for ML features; provides consistent feature computation for training and serving; prevents train/serve skew; SENTINEL's `graph_schema.py` + offline `.pt` files + online `preprocess.py` is a lightweight equivalent — describe this connection |

---

## Master Timeline — 8 Weeks

| Week | Phases | Focus | Deliverable |
|------|--------|-------|-------------|
| **Week 1** | Day 0 + Phase 0 | Setup, all 7 conceptual foundations, 10 vulnerability classes, EVM primer | Phase 0 teach-back passed without notes |
| **Week 2** | Phase 1 + Phase 2 first half | Locked contracts, graph_schema, hash systems, graph extraction, AST extractor | Trace one contract from `.sol` to `graph.x` verbally |
| **Week 3** | Phase 2 second half | `build_multilabel_index`, `dual_path_dataset`, truncation analysis, splits | Full data pipeline narration end-to-end |
| **Week 4** | Phase 3 | Both encoders. Run PyG Batch exercise in Python shell first. | Tensor shape diagram for the full forward pass — drawn from memory |
| **Week 5** | Phase 4 | Fusion. Run `to_dense_batch` exercise in Python shell first. Read `test_fusion_layer.py` before `fusion_layer.py`. | Fusion whiteboard teach-back with all 8 fixes narrated |
| **Week 6** | Phase 5 | Training. Draw OneCycleLR curve before reading trainer.py. | Training loop explanation + audit fix narration |
| **Week 7** | Phase 6 + Phase 7 | Inference API + MLOps. Run the API locally. Call `/predict`. Run `dvc status`. | Live API demo + full MLOps cycle teach-back |
| **Week 8** | Phase 8 + Review | Tests + full review. Run `pytest --cov`. Answer all 32 questions timed out loud. | All 32 questions under 2 minutes each without notes |

---

## Final Verification Checklist

Before considering this study plan complete, verify each item:

**Architecture:**
- [ ] Can explain SENTINEL end-to-end in under 3 minutes without notes
- [ ] Can write LoRA formula from scratch on a whiteboard, AND state the generalizable rank selection principle
- [ ] Can place GAT in the broader GNN landscape (GCN / GraphSAGE / GIN / Graph Transformer)
- [ ] Can draw PyG Batch mechanics (`Batch.from_data_list` + `to_dense_batch`) from memory
- [ ] Ran the `to_dense_batch` Python exercise and read the output
- [ ] Ran the toy SENTINEL graph construction exercise

**Domain & Data:**
- [ ] Can name all 10 vulnerability classes and describe one Solidity example each
- [ ] Can trace a `.sol` file to model input naming every function and file
- [ ] Can explain why 96.6% truncation mandates sliding windows, AND state when any new project should use the same approach

**Training & Evaluation:**
- [ ] Can explain all key audit fixes using "original did X, caused Y, I changed it to Z"
- [ ] Can explain why F1-macro is the right early stopping metric for this task
- [ ] Can describe the threshold tie-break rule AND the general principle behind it (cost asymmetry)
- [ ] Every 🔴 file has been annotated (comments above every non-trivial block)

**Production:**
- [ ] Can narrate the full MLOps cycle (validate → train → tune → stage → baseline → promote)
- [ ] Can describe the DVC model: what it solves, how `.dvc` files work, what `dvc pull` does
- [ ] Can describe shadow mode and canary deployment, and explain what SENTINEL would need to implement them

**Interpretability:**
- [ ] Can describe how you would extract attention weights from `CrossAttentionFusion` to explain a Reentrancy prediction to a developer
- [ ] Can explain what SHAP values would measure on the final classifier layer

**Code Ownership:**
- [ ] Can write a Code Directing prompt for `CrossAttentionFusion`, `GNNEncoder`, and the training loop
- [ ] Can write a Code Directing prompt for `drift_detector.py` and `api.py`

**Interview Readiness:**
- [ ] Passed all 25 SENTINEL-specific questions out loud, timed, without notes
- [ ] Passed all 7 generic senior ML questions out loud, timed, without notes
- [ ] `pytest ml/tests/ -v` passes cleanly on your machine
- [ ] `dvc status` shows clean (data in sync with git)

---

*This is Roadmap 3 of 3 of the SENTINEL Master Learning Plan. Together the three roadmaps replace the original single document, split at natural phase boundaries: Foundations & Data (Weeks 1–3) → Architecture & Training (Weeks 4–6) → Production & Interview Prep (Weeks 7–8). Key enhancements over the original: principle-vs-project-specific framing throughout; GNN landscape context in Phase 0; generalizable LoRA rank selection principle; interpretability guidance in Cross-Cutting; CI/CD and shadow mode note in Phase 7; 7 generic senior ML interview questions (Q26–Q32); and an expanded Final Verification Checklist.*
