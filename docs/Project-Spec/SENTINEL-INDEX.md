# SENTINEL Spec Index

This file is the entry point for AI assistants. Load ONLY the files relevant to the current task.
Do not load all files at once.

---

## File Map

| File | Contents | Size |
|------|----------|------|
| SENTINEL-OVERVIEW.md | What SENTINEL is, system data flow, module dep map, port map | Small |
| SENTINEL-CONSTRAINTS.md | Locked constants, shape constraints, rebuild triggers | Small–Medium |
| SENTINEL-ADR.md | Architecture Decision Records 001–038 | Large |
| SENTINEL-M1-ML.md | Module 1: ML model arch, inference API, training config, dataset facts, file inventory | Large |
| SENTINEL-M2-ZKML.md | Module 2: EZKL pipeline, proxy MLP, ONNX export, circuit status | Small |
| SENTINEL-M3-MLOPS.md | Module 3: MLflow experiments, DVC, Dagster RAG ingestion, drift detection | Small |
| SENTINEL-M4-AGENTS.md | Module 4: LangGraph orchestration, MCP servers, RAG retriever, LLM model map | Large |
| SENTINEL-M5-M6-PLATFORM.md | Module 5: Solidity contracts + Module 6: Integration API (not yet built) | Medium |
| SENTINEL-EVAL-BACKLOG.md | Retrain evaluation protocol, open audit findings, improvement backlog | Medium |
| SENTINEL-COMMANDS.md | All CLI commands, startup sequences, quick-reference | Small |

Current state (what is built, broken, next) lives in `docs/STATUS.md` and `docs/ROADMAP.md` — not here.

---

## Task → Files to Load

### Any implementation task
Always load: **SENTINEL-CONSTRAINTS.md**
Reason: locked constants, shape rules, rebuild triggers — violating these causes silent failures.

### System understanding / onboarding
Load: SENTINEL-OVERVIEW.md → SENTINEL-CONSTRAINTS.md

### ML inference / prediction work (preprocess.py, predictor.py, api.py)
Load: SENTINEL-CONSTRAINTS.md + SENTINEL-M1-ML.md

### ML training / retrain (trainer.py, focalloss.py, datasets)
Load: SENTINEL-CONSTRAINTS.md + SENTINEL-M1-ML.md + SENTINEL-EVAL-BACKLOG.md

### ZKML / proof pipeline work (ezkl, proxy model, distillation)
Load: SENTINEL-CONSTRAINTS.md + SENTINEL-M2-ZKML.md

### MLOps (MLflow, DVC, Dagster, drift detection)
Load: SENTINEL-M3-MLOPS.md

### Agent / LangGraph / MCP / RAG work
Load: SENTINEL-CONSTRAINTS.md + SENTINEL-M4-AGENTS.md

### Solidity contracts (AuditRegistry, SentinelToken, ZKMLVerifier)
Load: SENTINEL-M5-M6-PLATFORM.md + SENTINEL-CONSTRAINTS.md

### Integration API (M6 — not yet built)
Load: SENTINEL-M5-M6-PLATFORM.md

### Architecture decisions / design review
Load: SENTINEL-ADR.md + SENTINEL-OVERVIEW.md

### Planning / sprint prioritisation
Load: SENTINEL-EVAL-BACKLOG.md + SENTINEL-OVERVIEW.md

### Running / starting services
Load: SENTINEL-COMMANDS.md

---

## System Identity

**GitHub:** https://github.com/motafegh/sentinel-
**Environment:** WSL2 Ubuntu, RTX 3070 8GB VRAM, Python 3.12, Poetry
**Active checkpoint:** ml/checkpoints/multilabel-v3-fresh-60ep_best.pt
**Active threshold file:** ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json
**Baseline F1-macro (tuned):** 0.5069 — gate for v4 retrain

---

## Critical Cross-Cutting Rules (Always Apply)

```
fusion_output_dim = 128        — LOCKED; ZKML proxy depends on it
GNNEncoder in_channels = 8     — LOCKED; tied to 68,523 training graphs
NUM_CLASSES = 10               — WeakAccessMod excluded; append-only
edge_attr shape = [E] 1-D     — NOT [E, 1]
API response key = "thresholds" (list) — NOT "threshold" (single float; Breaking Fix #6)
NO "confidence" field anywhere — removed in Track 3; KeyError if accessed
TRANSFORMERS_OFFLINE = 1       — must be set at shell level, not in Python
weights_only policy:
  graph .pt files  → weights_only=True  (with add_safe_globals)
  checkpoint .pt   → weights_only=False (LoRA state dict)
```
