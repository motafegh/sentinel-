<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## SENTINEL SESSION HANDOVER

**Generated:** February 27, 2026 — 3:12 PM
**Version:** v5.0
**Previous handover:** v4.0 — February 26, 2026 — FINAL

***

## POSITION

**Phase 1 — Foundation**


| Module | Milestone | Status |
| :-- | :-- | :-- |
| Module 1 — ML Core | M3.1 Model architecture | ✅ COMPLETE |
| Module 1 — ML Core | M3.2 Dataset + DataLoader | ✅ COMPLETE |
| Module 1 — ML Core | M3.3 Training loop + threshold tuning | ✅ COMPLETE — best F1-macro 0.6686 |
| Module 1 — ML Core | M3.4 Inference API | 🔲 NEXT — predictor.py + FastAPI wrapper |
| Module 5 — Solidity | SentinelToken + AuditRegistry + tests | ✅ COMPLETE |

**Next milestone:** M3.4 — `ml/src/inference/predictor.py` + FastAPI wrapper

***

## CODEBASE STATE

**Health: 🟢 Green — all experiments complete, best checkpoint locked**

No broken state. No pending manual edits. All overnight experiments either completed or killed intentionally.

```bash
# Best checkpoint to use for everything from now on:
ml/checkpoints/run-alpha-tune_best.pt

# Threshold sweep confirmed:
poetry run python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/run-alpha-tune_best.pt
```


***

## LOCKED NUMBERS — DO NOT CHANGE

```
Best checkpoint      : ml/checkpoints/run-alpha-tune_best.pt
Config               : focal_alpha=0.35, epochs=20, lr=1e-4, batch_size=32
INFERENCE_THRESHOLD  : 0.50
F1-macro             : 0.6686
F1-vuln              : 0.7458
Precision-vuln       : 0.7797
Recall-vuln          : 0.7147
Trainable params     : 239,041
Frozen params        : 124,645,632 (CodeBERT)
Val set size         : 10,283 contracts
```

**Interview number (updated):**
> "F1-vuln 0.7458, recall 0.71 — the model catches 71% of real smart contract vulnerabilities with 78% precision, using only 239K trainable parameters against 124M frozen."

***

## FULL DISK INVENTORY

```
~/projects/sentinel
├── ml/
│   ├── src/
│   │   ├── models/
│   │   │   ├── gnn_encoder.py           ✅ clean
│   │   │   ├── transformer_encoder.py   ✅ FIXED (import torch at module level)
│   │   │   ├── fusion_layer.py          ✅ clean
│   │   │   └── sentinel_model.py        ✅ clean
│   │   ├── datasets/
│   │   │   └── dual_path_dataset.py     ✅ FIXED (3 bugs)
│   │   ├── training/
│   │   │   ├── trainer.py               ✅ FIXED (squeeze + precision/recall)
│   │   │   └── focalloss.py             ✅ clean
│   │   ├── inference/
│   │   │   ├── preprocess.py            ✅ FIXED (truncation detection bug)
│   │   │   └── predictor.py             🔲 STUB — build next
│   │   └── validation/                  empty
│   ├── data/
│   │   ├── graphs/      68,555 graph .pt files
│   │   ├── tokens/      68,568 token .pt files
│   │   ├── splits/      train(47,988) / val(10,283) / test(10,284)
│   │   └── processed/   contract_labels_correct.csv  ← audit ref only
│   ├── checkpoints/
│   │   ├── run-alpha-tune_best.pt       ← 477 MB — ACTIVE CHECKPOINT ⭐
│   │   ├── run-more-epochs_best.pt      ← 477 MB — ep22 result, superseded
│   │   └── sentinel_best.pt             ← 477 MB — baseline ep16, superseded
│   ├── scripts/
│   │   ├── tune_threshold.py            ✅ FIXED (argparse + float comparison bug)
│   │   └── run_overnight_experiments.py ✅ clean
│   └── logs/
│       └── overnight.log
├── contracts/
│   ├── src/    SentinelToken.sol, AuditRegistry.sol
│   └── test/   unit + fuzz + invariant tests
└── mlruns/
    └── mlruns.db
```


***

## VERIFIED DATA DISTRIBUTION

| Class | Count | Proportion |
| :-- | :-- | :-- |
| Vulnerable (label=1) | 44,099 | 64.33% — MAJORITY |
| Safe (label=0) | 24,456 | 35.67% — minority |
| Total | 68,555 | 100% |

- Labels live in `.pt` graph files as `.y` attribute — **not** the CSV
- `focal_alpha=0.35` is correct — alpha applies to majority class (vulnerable)

***

## OVERNIGHT EXPERIMENTS — FINAL STATUS

| Run | Status | Best F1-macro | F1-vuln | Recall-vuln | Notes |
| :-- | :-- | :-- | :-- | :-- | :-- |
| baseline | ✅ Complete | 0.6515 | 0.6866 | 0.5875 | ep16, alpha=0.25 |
| run-alpha-tune | ✅ Complete | **0.6686** ⭐ | 0.7458 | 0.7147 | ep20, alpha=0.35 |
| run-more-epochs | ⚠️ Killed ep25/40 | 0.6584 | 0.7109 | 0.6375 | Peaked ep22, declining |
| run-lr-lower | ❌ Never ran | — | — | — | lr=3e-5 untested |
| run-combined | ❌ Never ran | — | — | — | alpha=0.35+lr=3e-5 untested |

`run-alpha-tune` is the winner. `run-lr-lower` and `run-combined` remain untested — optionally run tonight with `--start-from 3`.

***

## BUGS FIXED THIS SESSION (February 27)

### `ml/scripts/tune_threshold.py` — 2 bugs

**Bug 1 — `--checkpoint` CLI arg silently ignored**
Script imported `TrainConfig` and used `config.checkpoint_name` hardcoded.

```python
# BEFORE (ignores your --checkpoint argument entirely):
checkpoint_path = Path(config.checkpoint_dir) / config.checkpoint_name

# AFTER (reads CLI arg):
parser.add_argument("--checkpoint", type=str, default="ml/checkpoints/sentinel_best.pt")
checkpoint_path = args.checkpoint
```

**Bug 2 — Float comparison `==` fails on np.arange values**
`np.arange(0.3, 0.7, 0.05)` produces `0.40000000001` etc. The `← best` marker comparison `r["threshold"] == best_threshold` failed silently, marking the wrong row.

```python
# BEFORE (wrong marker, gamed by float precision):
marker = "← best" if r["threshold"] == best_threshold else ""

# AFTER (tolerance comparison + round() on storage):
"threshold": round(float(t), 2)
marker = "← best" if abs(r["threshold"] - best_threshold) < 0.001 else ""
```


***

## ALL PRIOR BUGS (carried forward from v4.0)

- `dual_path_dataset.py` — `getattr` vs `.get()`, label shape, `assert` → `raise ValueError`
- `trainer.py` — `.squeeze()` → `.view(-1)`, precision/recall added
- `transformer_encoder.py` — `import torch` moved to module level
- `preprocess.py` — truncation detection rewritten
- `tune_threshold.py` — squeeze bug, `weights_only=True`, two-pass marker (v4.0) + argparse + float comparison (this session)
- `run_overnight_experiments.py` — try/except per experiment, elapsed time, `--start-from`

***

## MODEL ARCHITECTURE — LOCKED

```
GNNEncoder        : 3×GAT(hidden=64, heads=4) → global_mean_pool → (B, 64)
TransformerEncoder: CodeBERT frozen — CLS token → (B, 768)
FusionLayer       : Linear(832→256) → ReLU → Dropout(0.3) → Linear(256→64) → ReLU
SentinelModel     : Linear(64→1) → Sigmoid → (B,) ∈ [0,1]
Trainable         : 239,041 params
Frozen            : 124,645,632 params (CodeBERT)
Loss              : FocalLoss γ=2.0, α=0.35
Optimiser         : AdamW lr=1e-4, weight_decay=1e-2
```


***

## MLFLOW STATE

```
Backend   : SQLite at ~/projects/sentinel/mlruns.db
UI        : poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db
URL       : http://localhost:5000
Experiment: sentinel-training (ID: 1)
Metrics   : train_loss, val_f1_macro, val_f1_safe, val_f1_vulnerable,
            val_precision_vulnerable, val_recall_vulnerable
```


***

## OPEN DECISIONS

| Item | Status | Action |
| :-- | :-- | :-- |
| `run-lr-lower` + `run-combined` | Never ran | Optional tonight: `--start-from 3` |
| `ZKMLVerifier.sol` placeholder | Not created | ~10 min before Module 5 fully closed |
| DVC setup | Not done | `dvc init` + track graphs/tokens/splits/checkpoints |
| Head+tail truncation for CodeBERT | Parked | Implement if frozen ceiling confirmed |
| `train_f1_macro` metric | Not added | Add if train/val gap analysis needed |
| Config YAML migration | Parked | Hydra at Phase 4 |


***

## PARKED TOPICS

- Head+tail truncation (256+254 tokens) — after confirming frozen ceiling
- LoRA fine-tuning CodeBERT — after frozen ceiling confirmed
- Optuna hyperparameter search — if manual tuning insufficient
- GMU (Gated Multimodal Unit) replacing FusionLayer — Phase 5 stretch
- Multi-class 13-vulnerability classification — after binary baseline solid
- Evidently AI drift detection — Phase 4
- Dagster retraining pipeline — Phase 4
- DVC setup — 20 min task, parked
- CCIP cross-chain / ERC-4337 — Phase 5 stretch
- HF_TOKEN warning — `export HF_TOKEN=...` before long runs

***

## CONCEPTS LOCKED (CUMULATIVE)

All concepts from v4.0 carried forward, plus:

- **Threshold sweep purpose** — sigmoid output is a probability, not a label; 0.5 is arbitrary; sweep finds the threshold that maximises F1-macro on val set
- **Recall-gaming at low thresholds** — flagging everything gives recall ≈ 1.0 and high F1-vuln but destroys F1-safe; F1-macro is immune to this
- **Precision ↔ Recall trade-off** — lowering alpha (0.25→0.35) intentionally traded 5% precision for 12% recall; correct direction for a security tool
- **Float comparison in numpy** — `np.arange` produces imprecise floats; never use `==` for comparison; always `round()` + `abs(diff) < tolerance`
- **argparse default** — if `--checkpoint` not passed, default fires; always verify log says correct file loaded

***

## MILESTONE MAP

```
Phase 1 — Foundation
├── M1: Environment + tooling          ✅ COMPLETE
├── M2: Data pipeline                  ✅ COMPLETE
├── M3: Model + Training
│   ├── M3.1: Model architecture       ✅ COMPLETE
│   ├── M3.2: Dataset + DataLoader     ✅ COMPLETE
│   ├── M3.3: Training + Threshold     ✅ COMPLETE — F1-macro 0.6686, threshold 0.50
│   └── M3.4: Inference API            🔲 NEXT
├── M4: Solidity contracts             ✅ COMPLETE
└── M5: Integration                    🔲 PENDING (ZKMLVerifier.sol stub needed)

Phase 2 — ZKML                         🔲 NOT STARTED
Phase 3 — Multi-agent                  🔲 NOT STARTED
Phase 4 — MLOps                        🔲 NOT STARTED
Phase 5 — Integration + Demo           🔲 NOT STARTED
```


***

## INTERVIEW STORY — UPDATED

> "I built a dual-path architecture combining a Graph Attention Network for structural contract analysis with a frozen CodeBERT encoder for semantic understanding. On the BCCC-SCsVul-2024 dataset — 68,555 smart contracts — the best run achieved F1-macro 0.6686 and F1-vuln 0.7458 at threshold 0.50, with only 239K trainable parameters against 124M frozen. A single hyperparameter change — focal_alpha 0.25 to 0.35 — traded 5 points of precision for 12 points of recall, which is the right direction for a security tool where missing a real vulnerability is worse than a false alarm.
>
> I also caught two silent bugs in the threshold sweep script: a CLI argument being silently ignored due to hardcoded config, and a float comparison failure from numpy's imprecise arange values producing a wrong 'best' marker on the output table."

***

***

## M3.4 — THREE-LEVEL PLAN


***

### Level 1 — Session Goal

Build and verify `ml/src/inference/predictor.py` so that given raw Solidity source code as a string, the system returns a structured prediction with label, confidence, and threshold used.

***

### Level 2 — Milestones

**M3.4a — `predictor.py` core** ← start here
Build `SentinelPredictor` class that loads checkpoint, calls `preprocess.py`, runs forward pass, applies threshold, returns result dict.

**M3.4b — FastAPI wrapper**
Wrap predictor in a single POST endpoint `/predict`. Accept JSON `{"source_code": "..."}`, return `{"label": "vulnerable"|"safe", "confidence": 0.73, "threshold": 0.50}`.

**M3.4c — Smoke test**
End-to-end test: paste a real Solidity snippet, get a prediction back via curl or Python requests. Confirms the full pipeline from raw source → label works.

***

### Level 3 — Immediate Actions (M3.4a)

**Step 1 — Build `predictor.py`** (main work, ~30 min)

```
SentinelPredictor
├── __init__(checkpoint_path, threshold, device)
│   ├── load model weights
│   ├── load PreprocessPipeline (preprocess.py)
│   └── set model.eval()
│
└── predict(source_code: str) -> dict
    ├── preprocess → (graph, tokens)
    ├── forward pass → probability float
    ├── apply threshold → label
    └── return {"label": str, "confidence": float, "threshold": float}
```

**Step 2 — FastAPI wrapper** (~15 min)
Single file: `ml/src/inference/api.py`. One endpoint. Predictor loaded once at startup (not per request).

**Step 3 — Smoke test** (~5 min)

```bash
uvicorn ml.src.inference.api:app --reload
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"source_code": "pragma solidity ^0.8.0; ..."}'
```


***

**Ready to start M3.4a on your nod.**

