# SENTINEL — Retrain Evaluation Protocol, Open Audits & Improvement Backlog

Load for: planning, sprint prioritisation, retrain decisions, open audit findings.

---

## Retrain Evaluation Protocol — v5.2 (ACTIVE)

### Current Run

| Parameter | Value |
|-----------|-------|
| Run ID | `v5.2-jk-20260515c-r3` |
| PID | 43784 |
| Checkpoint | `ml/checkpoints/v5.2-jk-20260515c-r3_best.pt` |
| MLflow experiment | `sentinel-v5.2` (sqlite:///mlruns.db) |
| Dataset | `ml/data/processed/multilabel_index_deduped.csv` (44,420 rows, deduped) |
| Splits | `ml/data/splits/deduped/` — train 31,092 / val 6,661 / test 6,667 |
| eval_threshold | 0.35 (patience tracking only — NOT inference threshold) |
| Inference threshold | Per-class, produced by `tune_threshold.py` after training completes |
| Patience | 20 epochs |
| Architecture | v5_three_eye + JK attention + per-phase LayerNorm |

### F1-Macro Trajectory (r3)

| Epoch | Val F1-macro (eval_threshold=0.35) | Notes |
|-------|-------------------------------------|-------|
| 21 | 0.3130 | First new best after eval_threshold fix |
| 22 | 0.3202 | |
| 24 | 0.3203 | |
| 27 | 0.3282 | |
| 28 | 0.3290 | Best so far at time of writing |

Note: eval_threshold=0.35 is used only for patience tracking. Macro-F1 reported here reflects
lower recall recall recall per-class predictions compared to what tune_threshold.py will produce.
The tuned inference F1 (post-tune_threshold.py) is the only number that counts for the success gate.

---

## v5.2 Success Gate

**All five conditions must hold before v5.2 is declared complete.**

1. **Behavioral gate**: `ml/scripts/manual_test.py` — vulnerable test contracts flagged correctly,
   safe contracts not flagged. This is the primary judge. F1-macro alone is insufficient.

2. **Per-class F1 floors**: Every class meets its floor (v4 tuned F1 − 0.05) after tune_threshold.py.
   See table below.

3. **Tuned macro F1 ≥ 0.5422**: v4 fallback baseline (`multilabel-v4-finetune-lr1e4_best.pt`).

4. **GNN gradient share > 15%** throughout training (no collapse). Verified from MLflow grad share logs.

5. **NaN counter = 0** at every epoch. Any epoch with NaN is an automatic failure.

---

## v4 Per-Class Floors

Floor = v4 tuned F1 − 0.05. v5.2 must exceed floor on tuned inference metrics.

| Class | v4 Tuned F1 | v5.2 Floor |
|-------|-------------|------------|
| CallToUnknown | 0.447 | 0.397 |
| DenialOfService | 0.434 | 0.384 |
| ExternalBug | 0.484 | 0.434 |
| GasException | 0.557 | 0.507 |
| IntegerUO | 0.826 | 0.776 |
| MishandledException | 0.509 | 0.459 |
| Reentrancy | 0.569 | 0.519 |
| Timestamp | 0.528 | 0.478 |
| TransactionOrderDependence | 0.522 | 0.472 |
| UnusedReturn | 0.545 | 0.495 |

Active fallback if v5.2 fails any gate: `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt`
(v4 architecture, tuned macro F1 = 0.5422).

---

## Post-Training Checklist

Execute in order after r3 (or any continuation run) completes.

```
1. Confirm training terminated cleanly (epoch count, NaN counter = 0)
2. Run tune_threshold.py on val split:
       python ml/scripts/tune_threshold.py \
           --checkpoint ml/checkpoints/v5.2-jk-20260515c-r3_best.pt \
           --split ml/data/splits/deduped/val_indices.npy
   Output: thresholds JSON file
3. Evaluate tuned per-class F1 against floors table above
4. If any class fails its floor → analyse class-level confusion before deciding to promote or continue
5. Run behavioral tests:
       source ml/.venv/bin/activate
       python ml/scripts/manual_test.py
   Verify: all vulnerable contracts in ml/scripts/test_contracts/ are flagged,
           all safe contracts are NOT flagged
6. If all five success gate conditions pass → run promote_model.py:
       python ml/scripts/promote_model.py \
           --checkpoint ml/checkpoints/v5.2-jk-20260515c-r3_best.pt \
           --stage production
7. Push to GitHub (currently 12+ commits ahead of origin/main):
       git push origin main
8. Update MEMORY.md: active checkpoint, final F1 values, training status
```

---

## Open Audit Findings

Findings from adversarial audit session 2026-05-15 (27 fixes applied; findings below remain open).

### High Priority

| # | Severity | Description | File |
|---|----------|-------------|------|
| A-H1 | High | process_source() temp files not cleaned on SIGKILL; leaked .sol files accumulate in /tmp | `ml/src/inference/preprocess.py` |
| A-H2 | High | RAM cache loaded via pickle.load() without integrity check; tampered cache file executes on load | `ml/src/inference/preprocess.py` |
| A-H3 | High | File-path vs content hashing — two incompatible MD5 namespaces in use; hash collisions silent | `ml/src/utils/hash_utils.py` |

### Medium Priority

| # | Severity | Description | File |
|---|----------|-------------|------|
| A-M1 | Medium | hash_utils.py contains partially dead code; hashing logic diverges between offline pipeline and online inference | `ml/src/utils/hash_utils.py` |
| A-M2 | Medium | evaluate() in trainer uses fixed threshold=0.5 for per-class F1 reporting during training; not the same as inference thresholds | `ml/src/training/trainer.py` |
| A-M3 | Medium | DenialOfService class has only 377 total samples (train≈257); model is effectively untrained on this class | Data — no single-file fix |

### Low Priority / Info

| # | Severity | Description | File |
|---|----------|-------------|------|
| A-L1 | Low | Shape logging always silent; add SENTINEL_TRACE=1 env flag for debug tensor shapes | Various |
| A-L2 | Low | FocalLoss docstring describes binary classification; class is used for multi-label | `ml/src/training/focalloss.py` |
| A-L3 | Low | CLASS_NAMES imported from trainer.py into predictor.py; creates a hidden coupling | `ml/src/training/trainer.py`, `ml/src/inference/predictor.py` |
| A-L4 | Info | peft availability check at module import time makes unit tests verbose (prints warning on every import) | `ml/src/models/transformer_encoder.py` |
| A-L5 | Info | Checkpoint files loaded with weights_only=False (required for LoRA peft objects); this is documented and intentional, not a bug, but should be noted in deployment security review | `ml/src/training/trainer.py`, `ml/src/inference/predictor.py` |

---

## Completed Moves (2026-05-02 through 2026-05-16)

### Foundational Moves (2026-05-02 – 2026-05-05)

| Move | Description |
|------|-------------|
| Move 0 | `validate_graph_dataset.py` exits 0; graph integrity baseline established |
| Move 1 | FocalLoss FP32 cast confirmed (Audit #13 closed) |
| Move 2 | T1-A Inference cache (`cache.py`) |
| Move 3 | T2-A Prometheus metrics in `api.py` |
| Move 4 | T2-C MLflow registry (`promote_model.py`) |
| Move 5 | T3-A LLM synthesizer (qwen3.5-9b-ud + rule-based fallback) |
| Move 6 | T3-B Cross-encoder reranking (`rerank=False` param, opt-in) |
| Move 7 | T2-B Drift detection (`drift_detector.py` + baseline script) |
| Move 8 (partial) | Audit #11 RAM cache HMAC integrity check added; Audit #9 SIGKILL temp cleanup still open |
| Retrain v3 | 60-epoch fresh run; tuned F1 0.5069; v3 baseline established |
| Fix #1–#7, #9–#13, #23–#25 | Inference / training / resume stability fixes (2026-05-02 – 2026-05-05) |

### v4 Retrain & Architecture Prep (2026-05-05 – 2026-05-12)

| Move | Description |
|------|-------------|
| v4 retrain | FocalLoss gamma=2.0, LoRA r=16, DoS weighted sampler; tuned macro F1 = 0.5422 |
| NODE_FEATURE_DIM 8→12 | Added 4 new structural features; FEATURE_SCHEMA_VERSION bumped v2→v3 (2026-05-12, commit a0576fb) |
| NUM_EDGE_TYPES 7→8 | REVERSE_CONTAINS=7 added as runtime-only edge type (2026-05-14); edge_emb Embedding resized 7→8 |
| Dataset deduplication | 68,523 → 44,420 rows; 34.9% cross-split leakage identified and eliminated; new splits in `ml/data/splits/deduped/` |
| Ghost graph audit | 66 ghost graphs post-extraction (0.1%); gate passed; 280 stale v5.0 graphs identified (0.6%, uncompilable 0.4.x contracts) |

### v5.2 Adversarial Audit Fixes — Session 1 + Session 2 (2026-05-14 – 2026-05-15, commit 35028f9)

27 fixes applied across two sessions:

| Fix Group | Description |
|-----------|-------------|
| GNN architecture | 4-layer GAT (3 phases), per-phase LayerNorm, JK attention aggregation, REVERSE_CONTAINS runtime flip |
| GNN pooling fix (v5.1) | Pool FUNCTION/FALLBACK/RECEIVE/CONSTRUCTOR/MODIFIER nodes only; eliminates 77% CFG_RETURN noise |
| Gradient scale fix | Separate LR groups: GNN×2.5, LoRA×0.5, Other×1.0; prevents GNN grad collapse |
| Residual fix | x + dropout(x2) NOT dropout(x2 + x); critical correctness fix |
| Auxiliary loss weight | λ=0.1 → λ=0.3; keeps GNN eye and TF eye gradients alive |
| Effective batch | gradient_accumulation_steps=4, micro_batch=16; prevents RTX 3070 VRAM fragmentation |
| JK aggregation | Custom _JKAttention replacing PyG JumpingKnowledge(lstm); verifiable gradient flow |
| Dataset cache key | "{content_md5}_{FEATURE_SCHEMA_VERSION}" format; auto-invalidation on schema change |
| Checkpoint atomic save | Write to .tmp then os.replace(); prevents corrupt checkpoints on SIGKILL |
| GNN eye projection | concat(max_pool, mean_pool) → [B,256]; richer per-graph representation |
| Aux loss function | Plain BCEWithLogitsLoss on aux heads; no pos_weight amplification |
| eval_threshold fix | eval_threshold=0.35 for patience, separate from inference threshold=0.5 |
| tune_threshold DataLoader fix | DataLoader shuffle=False during threshold tuning; reproducible threshold search |

---

## Upcoming Work (post v5.2 behavioral gate)

### ZKML (Module 2)

- Proxy MLP source is complete: Linear(128→64→32→10), ONNX opset=11, EZKL scale=8192 little-endian
- Status: NEVER executed; awaiting v5.2 checkpoint
- Option A: run proxy distillation against v5.2 checkpoint → ONNX export → EZKL pipeline → Groth16 proof
- Option B: formally defer ZKML to a later milestone; document decision in ADR

### Contracts (Module 5)

- `contracts/lib/` is empty; `forge install` has never been run
- Required: `forge install` → `forge build` → `forge test`
- Fix any AuditRegistry source gaps found during build
- Sepolia deploy prep: environment variables, funded deployer wallet, deployment script

### API (Module 6)

- `api/` directory does not exist
- Design auth and rate-limiting scheme BEFORE writing any routes
- Then: FastAPI + Celery + Redis; expose inference, audit submission, proof verification endpoints

### Agents (Module 4)

- Close static_analysis agent gap (tool stubs present, logic incomplete)
- Close submitAudit agent gap (LangGraph → contract call path not wired)
- MCP client connection pool (currently single connection per request)
- Durable agent state across restarts (LangGraph persistence backend)

### MLOps (Module 3)

- Automated retraining trigger from drift detector (currently manual)
- Optuna hyperparameter search (LR multipliers, λ, patience, accumulation steps)
- Dagster lineage for deduped dataset pipeline

### Observability

- Prometheus + Grafana dashboards (inference latency, class score distributions, drift metrics)
- Jaeger / OpenTelemetry tracing across API → agent → model → ZK pipeline

### CI/CD

- GitHub Actions: pytest, forge test, slither, coverage gates
- Semgrep + Bandit for Python; Grype for dependency CVEs
- Gate: no merge to main without all checks passing

### Data & Training

- DoS class: only 377 total samples (train≈257); synthetic augmentation strategy required
- Multi-contract policy="all" for protocol-scale audits (Move 9; scaffold exists in GraphExtractionConfig)
- Online learning: incremental model updates for new vulnerability patterns without full retrain
- Investigate whether 280 stale v5.0 graphs (uncompilable 0.4.x contracts) are recoverable via solc 0.4.x pinning
