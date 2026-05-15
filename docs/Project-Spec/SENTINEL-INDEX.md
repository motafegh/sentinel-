# SENTINEL Spec Index

This file is the entry point for AI assistants. Load ONLY the files relevant to the current task.
Do not load all files at once.

---

## File Map

| File | Contents | Size |
|------|----------|------|
| SENTINEL-OVERVIEW.md | What SENTINEL is, system data flow, module dep map, port map | Small |
| SENTINEL-CONSTRAINTS.md | Locked constants, shape constraints, rebuild triggers — **load for every implementation task** | Medium |
| SENTINEL-ADR.md | Architecture Decision Records 001–052 | Large |
| SENTINEL-M1-ML.md | Module 1: ML model arch (v5.2 three-eye+JK), inference API, training config, dataset facts, file inventory | Large |
| SENTINEL-M2-ZKML.md | Module 2: EZKL pipeline, proxy MLP, ONNX export, circuit status + v2 upgrade proposals | Small |
| SENTINEL-M3-MLOPS.md | Module 3: MLflow experiments, DVC, Dagster RAG ingestion, drift detection + v2 upgrade proposals | Small |
| SENTINEL-M4-AGENTS.md | Module 4: LangGraph orchestration, MCP servers, RAG retriever, LLM model map + v2 upgrade proposals | Large |
| SENTINEL-M5-M6-PLATFORM.md | Module 5: Solidity contracts + Module 6: Integration API (not yet built) + v2 upgrade proposals | Medium |
| SENTINEL-EVAL-BACKLOG.md | v5.2 evaluation protocol, success gates, open audit findings, post-training checklist | Medium |
| SENTINEL-COMMANDS.md | All CLI commands, startup sequences, quick-reference (v5.2 current) | Small |
| SENTINEL-PROPOSAL-M1-v6.md | Forward-looking v6 ML architecture proposals (multi-contract, MC dropout, explainability, DoS augmentation) | Medium |

Current state (what is built, broken, next) lives in `docs/STATUS.md` and `docs/ROADMAP.md` — not here.

---

## Task → Files to Load

### Any implementation task
Always load: **SENTINEL-CONSTRAINTS.md**
Reason: locked constants (NODE_FEATURE_DIM=12, NUM_EDGE_TYPES=8, fusion_output_dim=128), shape rules,
rebuild triggers — violating these causes silent failures with no error signal.

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

### Planning / sprint prioritisation / upgrade proposals
Load: SENTINEL-EVAL-BACKLOG.md + SENTINEL-OVERVIEW.md + relevant module spec

### v6 ML planning
Load: SENTINEL-PROPOSAL-M1-v6.md + SENTINEL-M1-ML.md + SENTINEL-CONSTRAINTS.md

### Running / starting services
Load: SENTINEL-COMMANDS.md

---

## System Identity

**GitHub:** https://github.com/motafegh/sentinel-
**Environment:** WSL2 Ubuntu, RTX 3070 8GB VRAM, Python 3.12.1, Poetry
**Python venv:** `source ml/.venv/bin/activate`
**Active run:** v5.2-jk-20260515c-r3 (PID 43784, epoch 32+ best F1=0.3306, training in progress)
**Active checkpoint:** ml/checkpoints/v5.2-jk-20260515c-r3_best.pt
**v4 fallback:** ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt (tuned F1=0.5422)
**MLflow experiment:** sentinel-v5.2 — `sqlite:///mlruns.db`

---

## Critical Cross-Cutting Rules (Always Apply)

```
NODE_FEATURE_DIM   = 12          — LOCKED; rebuild 44K graphs + retrain if changed
NUM_EDGE_TYPES     = 8           — types 0–6 on disk; type 7 (REVERSE_CONTAINS) runtime-only
fusion_output_dim  = 128         — LOCKED; ZKML proxy input_dim depends on it
NUM_CLASSES        = 10          — WeakAccessMod excluded; APPEND-ONLY (never insert in middle)
edge_attr shape    = [E] 1-D    — NOT [E, 1]; nn.Embedding crashes on [E,1]
type_id normalised = float(id) / 12.0  — raw 0–12 dominates dot product without normalisation
FEATURE_SCHEMA_VERSION = "v3"   — cache key suffix; bump on ANY feature encoding change
gnn_hidden_dim     = 128         — LOCKED
MAX_TOKEN_LENGTH   = 512         — LOCKED; change requires token rebuild + retrain
ARCHITECTURE       = "v5_three_eye"

API response key = "thresholds" (list[float] per-class) — NOT "threshold" (Breaking Fix #6)
NO "confidence" field anywhere    — removed in Track 3; KeyError if accessed
request field = "source_code"     — NOT "contract_code"

weights_only policy:
  graph .pt files  → weights_only=True  (add_safe_globals: Data + PyG internals)
  checkpoint .pt   → weights_only=False (LoRA peft objects in state dict)

TRANSFORMERS_OFFLINE = 1          — must be set at SHELL level, not inside Python
```
