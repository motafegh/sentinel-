# SENTINEL

A decentralised AI security oracle for smart contracts. SENTINEL combines a dual-path GNN + CodeBERT vulnerability detector, ZK proof generation, and on-chain audit registration so that any result can be independently verified without trusting the agent that produced it.

---

## How It Works

```
User uploads .sol contract
         │
         ▼
[M6  API Gateway]      POST /v1/audit → job_id   (FastAPI + Celery)
         │
         ▼
[M4/M5  LangGraph Orchestration]
  ├── ml_assessment   ──▶  [M1  FastAPI :8001]
  │        │               GNN + CodeBERT + CrossAttention
  │        │               → vulnerabilities[] with per-class probabilities
  │        ▼
  │   max(probability) ≥ 0.70?
  │        ├── YES (deep)  ──▶  rag_research  ──▶  audit_check  ──▶  synthesizer
  │        └── NO  (fast)  ──────────────────────────────────────────▶  synthesizer
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
| M1 | `ml/` | Dual-path GNN (edge-type embeddings) + CodeBERT+LoRA + CrossAttention multi-label detector; windowed inference for long contracts; FastAPI server with Prometheus metrics and KS drift detection | Complete |
| M2 | `zkml/` | Proxy model distillation, EZKL circuit setup, per-audit proof generation | Scripts ready |
| M3 | `ml/` (mlops) | MLflow experiment tracking, DVC data versioning, Dagster RAG scheduling | Complete |
| M4 | `agents/` | LangGraph orchestration, 3 MCP servers, RAG, ingestion, feedback loop | Complete |
| M5 | `contracts/` | SentinelToken (ERC-20), AuditRegistry (UUPS), IZKMLVerifier interface | Written |
| M6 | `api/` | FastAPI + Celery gateway, Docker Compose full-stack | Planned |

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 – 3.12 | All Python modules |
| Poetry | ≥ 1.8 | Dependency management |
| Foundry (`forge`, `cast`) | latest | Solidity build and deploy |
| solc-select | latest | Solc version management |
| Docker + Compose | ≥ 24 | Full-stack local run |
| CUDA GPU (RTX 3070+) | 8 GB VRAM | ML inference and training |

---

## Repository Structure

```
sentinel-/
├── ml/                        # M1 — ML Core (GNN + CodeBERT model + inference API)
│   ├── src/
│   │   ├── models/            # SentinelModel, GNNEncoder, TransformerEncoder, CrossAttentionFusion, FocalLoss
│   │   ├── inference/         # api.py, predictor.py, cache.py, drift_detector.py, preprocess.py
│   │   ├── preprocessing/     # graph_extractor.py, graph_schema.py (edge-type vocab)
│   │   ├── training/          # trainer.py, CLASS_NAMES
│   │   └── datasets/          # dual_path_dataset.py, dual_path_collate_fn
│   ├── data_extraction/       # Offline Slither AST extractor (batch)
│   ├── scripts/               # train.py, create_splits.py, tune_threshold.py,
│   │                          # validate_graph_dataset.py, compute_drift_baseline.py,
│   │                          # promote_model.py
│   ├── tests/                 # pytest unit tests (10 modules)
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
│       ├── mcp/servers/       # inference_server.py (8010), rag_server.py (8011), audit_server.py (8012)
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
├── test_contracts/            # Sample .sol files for smoke testing
├── docs/changes/              # Changelog entries
├── SENTINEL-SPEC.md           # Timeless architecture spec (ADRs, constraints, data contracts)
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

### 2 — Start the ML inference server

```bash
TRANSFORMERS_OFFLINE=1 \
SENTINEL_CHECKPOINT=ml/checkpoints/multilabel_crossattn_v2_best.pt \
ml/.venv/bin/uvicorn ml.src.inference.api:app --port 8001

# Health check
curl http://localhost:8001/health
# → {
#      "status": "ok",
#      "predictor_loaded": true,
#      "checkpoint": "ml/checkpoints/multilabel_crossattn_v2_best.pt",
#      "architecture": "cross_attention_lora",
#      "thresholds_loaded": true
#    }
```

### 3 — Start the MCP servers

```bash
# In separate terminals:
cd agents && poetry run python -m src.mcp.servers.inference_server  # port 8010
cd agents && poetry run python -m src.mcp.servers.rag_server        # port 8011
cd agents && poetry run python -m src.mcp.servers.audit_server      # port 8012
```

### 4 — Run a full audit via LangGraph

```python
import asyncio
from agents.src.orchestration.graph import build_graph

graph = build_graph()
result = asyncio.run(graph.ainvoke(
    {"contract_code": open("test_contracts/simple_reentrancy.sol").read(),
     "contract_address": "0x0000000000000000000000000000000000000001"},
    config={"configurable": {"thread_id": "demo-001"}},
))
print(result["final_report"])
```

### 5 — Predict directly via the ML API

```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"source_code": "pragma solidity ^0.8.0; contract Foo { function bar() external payable {} }"}' \
  | python3 -m json.tool
```

---

## Port Map

| Port | Service |
|------|---------|
| 8000 | M6 API Gateway (planned) |
| 8001 | M1 FastAPI inference |
| 8010 | sentinel-inference MCP |
| 8011 | sentinel-rag MCP |
| 8012 | sentinel-audit MCP |
| 1234 | LM Studio (Windows host) |
| 3000 | Dagster UI |
| 5000 | MLflow UI |

---

## Testing

```bash
# Agents (LangGraph, MCP servers, RAG, ingestion)
cd agents && poetry run pytest tests/ -v
# → 41 tests, all green

# ML inference (10 test modules)
cd ml && poetry run pytest tests/ -v

# Contracts (requires forge)
cd contracts && forge test -vvv
```

Smoke scripts:

```bash
cd agents
poetry run python scripts/smoke_langgraph.py         # mock mode (no services needed)
poetry run python scripts/smoke_langgraph.py --live  # live mode (all services must be running)
poetry run python scripts/smoke_inference_mcp.py
poetry run python scripts/smoke_rag_mcp.py
poetry run python scripts/smoke_audit_mcp.py
```

---

## Key Constraints

Violating any of these without the matching rebuild/retrain produces silent failures.

| Constraint | Locked value |
|-----------|-------------|
| GNNEncoder `in_channels` | **8** — locked to 68 K graph `.pt` files |
| CodeBERT model | `microsoft/codebert-base` |
| `MAX_TOKEN_LENGTH` | **512** — matches training tokenisation |
| `CrossAttentionFusion output_dim` | **128** — ZKML proxy `input_dim` depends on this |
| `CLASS_NAMES` order | indices 0–9 must be stable; only append at end |
| ONNX opset | **11** — EZKL requirement |
| `solc` for ZKMLVerifier.sol | **≤ 0.8.17** — Halo2 assembly |
| `solc` for all other contracts | **0.8.20** |
| EZKL scale factor | **8192** (2¹³) |
| `weights_only=False` on `torch.load` | LoRA state dict requires it |
| `FEATURE_SCHEMA_VERSION` | **"v1"** — bump on any node/edge feature change; invalidates disk cache |
| `NUM_EDGE_TYPES` | **5** (CALLS/READS/WRITES/EMITS/INHERITS) — changing requires dataset rebuild |

---

## Development Workflow

```
main ← stable, deployed
  └── feature branches ← all development work

After landing a change that affects ML data contracts or the ZKML circuit:
  1. Update SENTINEL-SPEC.md (the timeless fact file)
  2. Add a dated entry to docs/changes/
  3. Re-run affected tests before merging
```

Spec file is NOT a status file — current status lives in session handovers. Do not treat spec sections as "done" without verifying against actual source files.

---

## Module Documentation

- [ML Core](ml/README.md) — model architecture, training, inference
- [Agents / MCP / RAG](agents/README.md) — orchestration, servers, retrieval
- [ZKML](zkml/README.md) — proxy model, EZKL pipeline, proof generation
- [Contracts](contracts/README.md) — Foundry build, deploy, ZKMLVerifier handling
