# SENTINEL — Retrain Evaluation Protocol, Open Audits & Improvement Backlog

Load for: planning, sprint prioritisation, retrain decisions, open audit findings.

---

## Retrain Evaluation Protocol

### Active Baseline: v3 (gate for v4)

| Parameter | Value |
|-----------|-------|
| Baseline checkpoint | `multilabel-v3-fresh-60ep_best.pt` |
| Raw F1-macro (best epoch ~52–53) | 0.4715 |
| **Tuned F1-macro (v4 gate)** | **0.5069** |
| Held-out split | Fixed — `ml/data/splits/val_indices.npy` (same seed, do NOT regenerate) |
| MLflow experiment | `sentinel-retrain-v4` |
| Success gate | tuned val F1-macro > **0.5069** |
| Per-class floor | No class drops > 0.05 F1 from v3 tuned values |
| Rollback | If tuned F1 < 0.5069 after completion, revert to v3 checkpoint |

### v3 Per-Class Floor Values (minimum F1 for v4 to pass)

| Class | v3 Tuned F1 | v4 Floor |
|-------|-------------|---------|
| CallToUnknown | 0.3936 | 0.3436 |
| DenialOfService | 0.4000 | 0.3500 |
| ExternalBug | 0.4345 | 0.3845 |
| GasException | 0.5501 | 0.5001 |
| IntegerUO | 0.8214 | 0.7714 |
| MishandledException | 0.4916 | 0.4416 |
| Reentrancy | 0.5362 | 0.4862 |
| Timestamp | 0.4789 | 0.4289 |
| TransactionOrderDependence | 0.4770 | 0.4270 |
| UnusedReturn | 0.4860 | 0.4360 |

### Retrain Gate Checklist

```

Before ANY retrain:

1. Run validate_graph_dataset.py — must exit 0
2. Confirm val_indices.npy is the fixed split (do NOT regenerate)
3. Create MLflow experiment "sentinel-retrain-v4"
4. Compare against "sentinel-retrain-v3" in MLflow after run
5. Run tune_threshold.py on val split before computing tuned F1
6. If tuned F1 < 0.5069: rollback to v3 checkpoint immediately

```

---

## Open Audit Findings (Deferred)

| # | Severity | Description | File |
|---|----------|-------------|------|
| 9 | Medium | process_source() temp files not cleaned on SIGKILL | ml/src/inference/preprocess.py |
| 10 | Medium | File-path vs content hashing — two incompatible namespaces | ml/src/utils/hash_utils.py |
| 11 | Medium | RAM cache loaded via pickle.load() without integrity check | ml/src/inference/preprocess.py |
| 12 | Medium | hash_utils.py partially dead code; hashing diverges from pipeline | ml/src/utils/hash_utils.py |
| 14 | Low | Shape logging always silent; add SENTINEL_TRACE=1 env flag | — |
| 15 | Low | FocalLoss docstring binary/multi-label mismatch | ml/src/training/focalloss.py |
| 16 | Low | CLASS_NAMES imported from trainer.py into predictor.py | — |
| 17 | Info | peft check at module import time makes unit tests verbose | — |
| 18 | Info | Checkpoint pickle (weights_only=False) still present in trainer.py/predictor.py | — |

Note: Audit #3 (evaluate() uses fixed 0.5 threshold) and #6 (PredictResponse.threshold)
are tracked; #6 was partially addressed by Fix #6 (renamed to "thresholds" list).
Full per-class threshold return in evaluate() remains open.

---

## Improvement Backlog

Current state (module completion, open loops) lives in `docs/STATUS.md`.
Ordered remaining work lives in `docs/ROADMAP.md`.

### Currently In Progress

| Item | What | Notes |
|------|------|-------|
| v4 retrain | Focal loss (gamma=2.0), LoRA r=16, DoS weighted sampler | Gate: tuned F1 > 0.5069 |
| Autoresearch setup | `auto_experiment.py` + `ml/autoresearch/program.md` | Now unblocked; v3 baseline established |
| ZKML resolution | Option A (run pipeline) or Option B (descope to S10) | No scheduled move yet |
| M6 Integration API | Design auth/rate-limit before writing routes | api/ does not exist |

### Completed Moves (2026-05-02 through 2026-05-05)

| Move | Description | Done |
|------|-------------|------|
| Move 0 | `validate_graph_dataset.py` exits 0 | ✅ |
| Move 1 | Confirm FocalLoss FP32 cast (Audit #13) closed | ✅ |
| Move 2 | T1-A Inference cache (`cache.py`) | ✅ |
| Move 3 | T2-A Prometheus metrics in `api.py` | ✅ |
| Move 4 | T2-C MLflow registry (`promote_model.py`) | ✅ |
| Move 5 | T3-A LLM synthesizer (qwen3.5-9b-ud + fallback) | ✅ |
| Move 6 | T3-B Cross-encoder reranking (`rerank=False` param) | ✅ |
| Move 7 | T2-B Drift detection (`drift_detector.py`, baseline script) | ✅ |
| Move 8 | Audit #11 (RAM cache) fixed; Audit #9 (temp SIGKILL) still open | ⚠️ Partial |
| Retrain v3 | 60-epoch fresh run; tuned F1 0.5069 | ✅ |
| Fix #1–#7, #9–#13, #23–#25 | Various inference/training/resume fixes | ✅ |

### Upcoming (Post v4 Retrain)

| Sprint | Goal |
|--------|------|
| Fix #6 downstream | Update API consumers that used old `"threshold"` key |
| Move 8 remaining | preprocess.py temp file SIGKILL cleanup (Audit #9) |
| M6 Integration | FastAPI + Celery + Redis; auth design first |
| Move 9 | Multi-contract parsing (`multi_contract_policy="all"`) |
| S6 Observability | Prometheus, OpenTelemetry/Jaeger, Loki |
| S7 CI/CD | GitHub Actions, Semgrep, Bandit, Grype |
| S8 Advanced contracts | Echidna property fuzzing, Halmos symbolic proofs |
| S9 Advanced ML | Online drift, canary deployment, structured LLM output |
| S10 Advanced ZK | EZKL env config, circuit versioning on-chain, proof aggregation |

