# SENTINEL

A decentralised AI security oracle for smart contracts. SENTINEL combines a dual-path GNN + CodeBERT vulnerability detector, zero-knowledge proof generation, and on-chain audit registration so that any result can be independently verified without trusting the agent that produced it.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Modules](#modules)
- [Current Model Performance](#current-model-performance)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Port Map](#port-map)
- [Testing](#testing)
- [Key Constraints](#key-constraints)
- [Development Workflow](#development-workflow)
- [Module & Architecture Documentation](#module--architecture-documentation)

---

## How It Works

```

User uploads .sol contract
│
▼
[M6  API Gateway]      POST /v1/audit → job_id   (FastAPI + Celery)   ← planned
│
▼
[M4/M5  LangGraph Orchestration]
├── ml_assessment   ──▶  [M1  FastAPI :8001]
│        │               GNN + CodeBERT + CrossAttention
│        │               → vulnerabilities[] with per-class probabilities
│        ▼
│   max(probability) ≥ threshold?
│        ├── YES (deep)  ──▶  rag_research  ──▶  audit_check  ──▶  synthesizer
│        └── NO  (fast)  ────────────────────────────────────────▶  synthesizer
│
└── synthesizer  →  final_report {label, vulnerabilities[], rag_evidence[], audit_history[]}
│
▼
[M2  ZKML Proof Generation]
proxy MLP(128→64→32→10) → EZKL/Groth16 → π + publicSignals[10 class scores]
│
▼
[M5  Blockchain — Sepolia]
AuditRegistry.submitAudit(proof, publicSignals)
ZKMLVerifier.verify() on-chain  →  AuditSubmitted event
│
▼
[M4  Feedback Loop]
Polls AuditRegistry, ingests findings back into RAG index

```

---

## Modules

| # | Path | What it does | Status |
|---|------|-------------|--------|
| M1 | `ml/` | Dual-path GNN (edge-type embeddings) + CodeBERT+LoRA + CrossAttention multi-label detector; per-class threshold tuning; windowed inference; FastAPI with Prometheus metrics and KS drift detection | ✅ Complete |
| M2 | `zkml/` | Proxy model distillation, EZKL circuit setup, per-audit Groth16 proof generation | ⚠️ Source complete — pipeline not yet run |
| M3 | `ml/` (mlops) | MLflow experiment tracking, DVC data versioning, Dagster RAG scheduling, model registry promotion | ✅ Complete |
| M4 | `agents/` | LangGraph orchestration, 3 MCP servers, RAG (hybrid BM25+dense+RRF), ingestion, feedback loop | ✅ Complete |
| M5 | `contracts/` | SentinelToken (ERC-20), AuditRegistry (UUPS), IZKMLVerifier interface — Foundry test suite written | ⚠️ Source complete — forge not yet run |
| M6 | `api/` | FastAPI + Celery gateway, Docker Compose full-stack | ❌ Planned |

---

## Current Model Performance

Active checkpoint: `ml/checkpoints/multilabel-v3-fresh-60ep_best.pt`  
Thresholds: `ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json`

Trained on BCCC-SCsVul-2024 (47,966 train / 10,278 val / 10,279 test).  
Architecture: `cross_attention_lora` — GNN(edge_attr) + CodeBERT + CrossAttention, LoRA r=8 α=16.

### Overall metrics (per-class threshold tuning applied)

| Metric | Value |
|--------|-------|
| F1-macro | **0.5069** |
| F1-micro | 0.5608 |
| Hamming loss | 0.2342 |
| Exact-match accuracy | 0.2763 |

### Per-class F1 (tuned thresholds)

| Vulnerability | F1 | Threshold | Support |
|---------------|----|-----------|---------|
| IntegerUO | 0.821 | 0.50 | 5,343 |
| GasException | 0.550 | 0.55 | 2,589 |
| Reentrancy | 0.536 | 0.65 | 2,501 |
| MishandledException | 0.492 | 0.60 | 2,207 |
| UnusedReturn | 0.486 | 0.70 | 1,716 |
| Timestamp | 0.479 | 0.75 | 1,077 |
| TransactionOrderDependence | 0.477 | 0.60 | 1,800 |
| ExternalBug | 0.435 | 0.65 | 1,622 |
| DenialOfService | 0.400 | 0.95 | 137 |
| CallToUnknown | 0.394 | 0.70 | 1,266 |

> DenialOfService and CallToUnknown are the weakest classes due to limited training data.  
> The v4 run targets both with focal loss and weighted sampling.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 – 3.12 | All Python modules |
| Poetry | ≥ 1.8 | Dependency management |
| Foundry (`forge`, `cast`) | latest | Solidity build and deploy |
| solc-select | latest | Solc version management |
| Docker + Compose | ≥ 24 | Full-stack local run |
| CUDA GPU (RTX 3070+) | ≥ 8 GB VRAM | ML inference and training |

---

## Repository Structure

```

sentinel-/
├── ml/                        # M1 + M3 — ML Core and MLOps
│   ├── src/
│   │   ├── models/            # SentinelModel, GNNEncoder, TransformerEncoder,
│   │   │                      # CrossAttentionFusion, FocalLoss
│   │   ├── inference/         # api.py, predictor.py, preprocess.py,
│   │   │                      # cache.py, drift_detector.py
│   │   ├── preprocessing/     # graph_extractor.py, graph_schema.py
│   │   ├── training/          # trainer.py, TrainConfig, CLASS_NAMES
│   │   └── datasets/          # dual_path_dataset.py, dual_path_collate_fn
│   ├── data_extraction/       # Offline Slither AST extractor + tokenizer (batch)
│   ├── scripts/               # train.py, tune_threshold.py, create_splits.py,
│   │                          # validate_graph_dataset.py, promote_model.py,
│   │                          # compute_drift_baseline.py, analyse_truncation.py
│   ├── tests/                 # 10 pytest modules (synthetic data only)
│   └── docker/                # Dockerfile.slither for offline extraction
│
├── zkml/                      # M2 — ZK-ML Proof Generation (EZKL)
│   └── src/
│       ├── distillation/      # ProxyMLP, train_proxy.py, export_onnx.py
│       └── ezkl/              # setup_circuit.py, run_proof.py, extract_calldata.py
│
├── agents/                    # M4 — Orchestration, MCP Servers, RAG
│   └── src/
│       ├── orchestration/     # LangGraph graph.py, nodes.py, state.py
│       ├── mcp/servers/       # inference_server.py (8010), rag_server.py (8011),
│       │                      # audit_server.py (8012)
│       ├── rag/               # HybridRetriever, chunker, embedder, build_index
│       ├── ingestion/         # pipeline, deduplicator, feedback_loop, dagster scheduler
│       └── llm/               # LM Studio client, model routing
│
├── contracts/                 # M5 — Solidity Contracts (Foundry)
│   └── src/
│       ├── AuditRegistry.sol  # UUPS upgradeable audit registry
│       ├── SentinelToken.sol  # ERC-20 + staking (MIN_STAKE = 1000 SNTL)
│       └── IZKMLVerifier.sol  # Interface for EZKL-generated verifier
│
├── docs/
│   ├── STATUS.md              # Current module completion and open loops
│   ├── ROADMAP.md             # Upcoming work in priority order
│   ├── changes/               # Dated changelogs for every significant change
│   └── project-spec/          # Split specification files (see below)
├── test_contracts/            # Sample .sol files for smoke testing
└── pyproject.toml             # Root Poetry workspace config

```

---

## Quick Start

### 1 — Install dependencies

```bash
# Root workspace
poetry install

# ML module
cd ml && poetry install && cd ..

# Agents module
cd agents && poetry install && cd ..
```

2 — Start the ML inference server

```bash
TRANSFORMERS_OFFLINE=1 \
SENTINEL_CHECKPOINT=ml/checkpoints/multilabel-v3-fresh-60ep_best.pt \
SENTINEL_THRESHOLDS=ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json \
ml/.venv/bin/uvicorn ml.src.inference.api:app --port 8001
```

Health check:

```bash
curl http://localhost:8001/health
# {
#   "status": "ok",
#   "predictor_loaded": true,
#   "checkpoint": "ml/checkpoints/multilabel-v3-fresh-60ep_best.pt",
#   "architecture": "cross_attention_lora",
#   "thresholds_loaded": true
# }
```

3 — Start the MCP servers

```bash
# In separate terminals:
cd agents && poetry run python -m src.mcp.servers.inference_server  # port 8010
cd agents && poetry run python -m src.mcp.servers.rag_server        # port 8011
cd agents && poetry run python -m src.mcp.servers.audit_server      # port 8012
```

4 — Run a full audit via LangGraph

```python
import asyncio
from agents.src.orchestration.graph import build_graph

graph = build_graph()
result = asyncio.run(graph.ainvoke(
    {
        "contract_code": open("test_contracts/simple_reentrancy.sol").read(),
        "contract_address": "0x0000000000000000000000000000000000000001",
    },
    config={"configurable": {"thread_id": "demo-001"}},
))
print(result["final_report"])
```

5 — Predict directly via the ML API

```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"source_code": "pragma solidity ^0.8.0; contract Foo { function bar() external payable {} }"}' \
  | python3 -m json.tool
```

Example response:

```json
{
  "label": "vulnerable",
  "vulnerabilities": [
    { "vulnerability_class": "Reentrancy",  "probability": 0.8943 },
    { "vulnerability_class": "IntegerUO",   "probability": 0.7102 }
  ],
  "thresholds": [0.70, 0.95, 0.65, 0.55, 0.50, 0.60, 0.65, 0.75, 0.60, 0.70],
  "truncated": false,
  "windows_used": 1,
  "num_nodes": 12,
  "num_edges": 18
}
```

thresholds is a list of 10 per-class values (index order matches CLASS_NAMES).

---

Port Map

Port Service
8000 M6 API Gateway (planned)
8001 M1 FastAPI inference
8010 sentinel-inference MCP
8011 sentinel-rag MCP
8012 sentinel-audit MCP
1234 LM Studio (Windows host)
3000 Dagster UI
5000 MLflow UI

---

Testing

```bash
# ML inference (10 test modules — synthetic data, no checkpoints required)
cd ml && poetry run pytest tests/ -v

# Agents (LangGraph, MCP servers, RAG, ingestion)
cd agents && poetry run pytest tests/ -v

# Contracts (requires forge)
cd contracts && forge test -vvv
```

Smoke scripts:

```bash
cd agents
poetry run python scripts/smoke_langgraph.py          # mock mode — no services needed
poetry run python scripts/smoke_langgraph.py --live   # live mode — all services must be running
poetry run python scripts/smoke_inference_mcp.py
poetry run python scripts/smoke_rag_mcp.py
poetry run python scripts/smoke_audit_mcp.py
```

---

## Key Constraints

Violating any of these without the matching rebuild or retrain produces silent failures.

Constraint Locked value Break condition
GNNEncoder in_channels 8 Rebuild all 68K graph .pt files + retrain
CodeBERT model microsoft/codebert-base Rebuild token files + retrain
MAX_TOKEN_LENGTH 512 Rebuild token files + retrain
CrossAttentionFusion output_dim 128 Rebuild ZKML circuit + redeploy verifier
CLASS_NAMES order indices 0–9 stable Silent wrong-class mapping
FEATURE_SCHEMA_VERSION "v1" Bump only alongside graph rebuild; invalidates inference cache
NUM_EDGE_TYPES 5 (CALLS/READS/WRITES/EMITS/INHERITS) Rebuild edge_emb layer + retrain
ONNX opset 11 EZKL requirement
solc for ZKMLVerifier.sol ≤ 0.8.17 Halo2 assembly constraint
solc for all other contracts 0.8.20 
weights_only=False on torch.load required LoRA state dict is not a plain dict
TRANSFORMERS_OFFLINE set at shell level Cannot be set inside Python

Full details in docs/project-spec/SENTINEL-CONSTRAINTS.md.

---

Development Workflow

```
main ← stable, deployed
  └── feature/* ← all development work

After landing a change that affects architecture, data contracts, or locked constants:
  1. Update the relevant file(s) in docs/project-spec/ (see SENTINEL-INDEX.md for which file)
  2. Add a dated entry under docs/changes/
  3. Update docs/STATUS.md and docs/ROADMAP.md as needed
  4. Re-run affected tests before merging
```

The old monolithic SENTINEL-SPEC.md is superseded by the split specifications in docs/project-spec/.

---

## Module & Architecture Documentation

Split specification (immutable project facts)

Located in docs/project-spec/. Load only what you need using the index:

· SENTINEL-INDEX.md — task → file routing
· SENTINEL-OVERVIEW.md — system design, data flow, ports
· SENTINEL-CONSTRAINTS.md — locked constants (must read before any implementation)
· SENTINEL-ADR.md — all Architecture Decision Records
· SENTINEL-M1-ML.md — ML model architecture, training, inference
· SENTINEL-M2-ZKML.md — ZK proof pipeline
· SENTINEL-M3-MLOPS.md — MLflow, DVC, Dagster, drift detection
· SENTINEL-M4-AGENTS.md — LangGraph, MCP servers, RAG
· SENTINEL-M5-M6-PLATFORM.md — Solidity contracts + Integration API
· SENTINEL-EVAL-BACKLOG.md — retrain protocol, audits, backlog
· SENTINEL-COMMANDS.md — quick‑reference CLI commands

## Per‑module READMEs

· ML Core — model architecture, training, inference, drift detection
· Agents / MCP / RAG — orchestration, servers, retrieval
· ZKML — proxy model, EZKL pipeline, proof generation
· Contracts — Foundry build, deploy, ZKMLVerifier handling

## Tracking

· docs/STATUS.md — what’s built, what’s broken, what’s next
· docs/ROADMAP.md — upcoming work in priority order
· docs/changes/ — dated changelogs for every significant change



