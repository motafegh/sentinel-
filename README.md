# SENTINEL

Smart contracts are immutable once deployed — a single vulnerability can result in irreversible loss of funds. Existing audit tools are either static analysers that miss context-dependent bugs, or expensive manual reviews that don't scale. SENTINEL is a **decentralised AI security oracle**: a dual-path GNN + CodeBERT vulnerability detector with zero-knowledge proof generation and on-chain audit registration so that any result can be independently verified without trusting the agent that produced it.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Modules](#modules)
- [Current Model](#current-model)
- [Output Classes](#output-classes)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
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
│        │               Three-eye SentinelModel v5.2
│        │               GNNEncoder (3-phase GAT + JK) + CodeBERT+LoRA + CrossAttentionFusion
│        │               → vulnerabilities[] with per-class probabilities (10 classes)
│        ▼
│   max(probability) ≥ 0.70?
│        ├── YES (deep)  ──▶  rag_research ──┐
│        │                                   ├──▶  audit_check ──▶  synthesizer
│        │                ├── static_analysis ──┘
│        └── NO  (fast)  ────────────────────────────────────────▶  synthesizer
│
└── synthesizer  →  final_report {overall_label, vulnerabilities[], rag_evidence[], audit_history[]}
│
▼
[M2  ZKML Proof Generation]
proxy MLP(128→64→32→10) → EZKL/Groth16 → proof π + publicSignals[10 class scores]
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
| M1 | `ml/` | Three-eye SentinelModel v5.2: three-phase GAT (JK connections) + CodeBERT LoRA + CrossAttentionFusion; 10-class multi-label; per-class threshold tuning; sliding-window inference; FastAPI with Prometheus metrics and KS drift detection | ✅ Complete |
| M2 | `zkml/` | ProxyMLP distillation (128→64→32→10), EZKL circuit setup, per-audit Groth16 proof generation | ⚠️ Source complete — pipeline not yet run end-to-end |
| M3 | `ml/` (mlops) | MLflow experiment tracking, DVC data versioning, Dagster RAG scheduling, model registry promotion | ✅ Complete |
| M4 | `agents/` | LangGraph orchestration (parallel fan-out), 3 MCP servers (SSE), hybrid RAG (FAISS+BM25+RRF), ingestion pipeline, feedback loop | ✅ Complete |
| M5 | `contracts/` | SentinelToken (ERC-20 + staking), AuditRegistry (UUPS upgradeable), IZKMLVerifier interface — Foundry test suite written | ⚠️ Source complete — forge not yet run on latest |
| M6 | `api/` | FastAPI + Celery gateway, Docker Compose full-stack | ❌ Planned |

---

## Current Model

**Architecture: SentinelModel v5.2** (`MODEL_VERSION = "v5.2"`)

Three-eye classifier: GNN eye (three-phase GAT with JK connections and per-phase LayerNorm) + Transformer eye (CodeBERT CLS token + LoRA r=16) + Fused eye (CrossAttentionFusion) → `cat [B, 384]` → `Linear(384, 10)`.

**Active baseline checkpoint:** `ml/checkpoints/multilabel-v3-fresh-60ep_best.pt`
Trained on BCCC-SCsVul-2024 · 47,966 train / 10,278 val / 10,279 test · per-class threshold tuning applied.

> The v5.2 architecture (three-phase GNN + JK connections + 12-dim node features) is the current source of truth. The active checkpoint was trained on the earlier v3 schema. A full v5.2 training run is the next milestone — see retrain gate in `docs/Project-Spec/SENTINEL-EVAL-BACKLOG.md`.

### Baseline per-class F1 (v3 checkpoint, tuned thresholds)

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

**Overall:** F1-macro 0.5069 · F1-micro 0.5608 · Hamming 0.2342 · Exact-match 0.2763

---

## Output Classes

10-class multi-label output. Index order is **locked** — reordering breaks all checkpoints and the ZKML circuit. Defined as `CLASS_NAMES` in `ml/src/training/trainer.py` — single source of truth.

| Index | Class |
|-------|-------|
| 0 | CallToUnknown |
| 1 | DenialOfService |
| 2 | ExternalBug |
| 3 | GasException |
| 4 | IntegerUO |
| 5 | MishandledException |
| 6 | Reentrancy |
| 7 | Timestamp |
| 8 | TransactionOrderDependence |
| 9 | UnusedReturn |

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12.1 (strict — ml) / ≥ 3.11 (agents) | All Python modules |
| Poetry | ≥ 1.8 | Dependency management |
| Foundry (`forge`, `cast`) | latest | Solidity build and deploy |
| solc-select | latest | Solc version management |
| Docker + Compose | ≥ 24 | Slither extraction environment |
| CUDA GPU (RTX 3070+) | ≥ 8 GB VRAM | ML training and inference |
| slither-analyzer | ≥ 0.9.3 | Graph extraction (hard minimum — older versions produce wrong `in_unchecked` features) |

---

## Environment Variables

All must be exported at **shell level** before any service starts. Setting them inside Python is too late — `TRANSFORMERS_OFFLINE` is read at `transformers` import time.

### ML Inference Server (M1 — port 8001)

| Variable | Default | Required | Notes |
|----------|---------|----------|-------|
| `TRANSFORMERS_OFFLINE` | — | **Yes** | Must be `1`. Set at shell level. |
| `HF_HUB_OFFLINE` | — | **Yes** | Set alongside `TRANSFORMERS_OFFLINE`. |
| `SENTINEL_CHECKPOINT` | `ml/checkpoints/multilabel_crossattn_v2_best.pt` | **Yes** | Path to `.pt` checkpoint. |
| `SENTINEL_PREDICT_TIMEOUT` | `60` | No | Seconds before HTTP 504. |
| `SENTINEL_DRIFT_BASELINE` | `ml/data/drift_baseline.json` | No | KS drift baseline. Alerts suppressed until file exists. |
| `SENTINEL_DRIFT_CHECK_INTERVAL` | `50` | No | Run KS test every N requests. |

### Agents / MCP Servers (M4)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio endpoint (WSL2: gateway IP changes on reboot — set explicitly) |
| `LM_STUDIO_MODEL` | — | Model name for LLM routing |
| `MCP_INFERENCE_URL` | `http://localhost:8010/sse` | Inference MCP endpoint |
| `MCP_RAG_URL` | `http://localhost:8011/sse` | RAG MCP endpoint |
| `MCP_AUDIT_URL` | `http://localhost:8012/sse` | Audit MCP endpoint |
| `AUDIT_RAG_K` | `5` | RAG chunks retrieved per deep-path query |
| `MODULE1_MOCK` | `false` | Force mock inference (dev/tests) |
| `AUDIT_MOCK` | `false` | Force mock audit history |
| `DAGSTER_HOME` | `agents/.dagster` | Dagster workspace |

### Contracts / Blockchain (M5)

| Variable | Purpose |
|----------|---------|
| `SEPOLIA_RPC_URL` | Sepolia RPC endpoint (Alchemy / Infura) |
| `DEPLOYER_PRIVATE_KEY` | Private key for deployment and `submitAudit` calls |
| `AUDIT_REGISTRY_ADDRESS` | Deployed `AuditRegistry` address (set after first deploy) |
| `ZKML_VERIFIER_ADDRESS` | Deployed `ZKMLVerifier` address |
| `ETHERSCAN_API_KEY` | For contract verification on deploy |

---

## Repository Structure

```
sentinel/
├── ml/                          # M1 + M3 — ML Core and MLOps
│   ├── src/
│   │   ├── preprocessing/       # graph_schema.py (single source of truth),
│   │   │                        # graph_extractor.py — shared by offline + online paths
│   │   ├── data_extraction/     # ast_extractor.py (offline batch, 11 workers),
│   │   │                        # tokenizer.py
│   │   ├── models/              # sentinel_model.py (v5.2 three-eye),
│   │   │                        # gnn_encoder.py (three-phase GAT + JK),
│   │   │                        # transformer_encoder.py (CodeBERT + LoRA),
│   │   │                        # fusion_layer.py (CrossAttentionFusion)
│   │   ├── training/            # trainer.py (TrainConfig, CLASS_NAMES, MODEL_VERSION),
│   │   │                        # focalloss.py
│   │   ├── datasets/            # dual_path_dataset.py, dual_path_collate_fn
│   │   ├── inference/           # api.py, predictor.py, preprocess.py,
│   │   │                        # cache.py, drift_detector.py
│   │   └── utils/               # hash_utils.py
│   ├── scripts/                 # train.py, tune_threshold.py, promote_model.py,
│   │                            # auto_experiment.py, build_multilabel_index.py,
│   │                            # create_splits.py, validate_graph_dataset.py,
│   │                            # generate_cei_pairs.py, compute_drift_baseline.py, …
│   ├── tests/                   # 12 pytest modules (~3,100 lines)
│   ├── docker/                  # Dockerfile.slither (Ubuntu 20.04 + slither 0.10.0)
│   └── data/
│       ├── graphs.dvc           # ~68K .pt graph files (DVC-tracked)
│       ├── tokens.dvc           # ~68K .pt token files (DVC-tracked)
│       ├── splits.dvc           # train/val/test .npy arrays (DVC-tracked)
│       └── processed/           # multilabel_index_deduped.csv
│
├── zkml/                        # M2 — ZK-ML Proof Generation (EZKL)
│   └── src/
│       ├── distillation/        # proxy_model.py (128→64→32→10), train_proxy.py,
│       │                        # export_onnx.py, generate_calibration.py
│       └── ezkl/                # setup_circuit.py (one-time), run_proof.py (per-audit),
│                                # extract_calldata.py
│
├── agents/                      # M4 — Orchestration, MCP Servers, RAG
│   └── src/
│       ├── orchestration/       # graph.py (StateGraph), nodes.py, state.py (AuditState)
│       ├── mcp/servers/         # inference_server.py (:8010), rag_server.py (:8011),
│       │                        # audit_server.py (:8012)
│       ├── rag/                 # HybridRetriever (FAISS+BM25+RRF), chunker, embedder,
│       │                        # build_index.py, fetchers/github_fetcher.py
│       ├── ingestion/           # pipeline.py, deduplicator.py, feedback_loop.py,
│       │                        # scheduler_dagster.py, scheduler_cron.py
│       └── llm/                 # client.py (LM Studio, model routing)
│
├── contracts/                   # M5 — Solidity Contracts (Foundry)
│   ├── foundry.toml             # solc 0.8.20 default; ZKMLVerifier uses 0.8.17
│   └── src/
│       ├── AuditRegistry.sol    # UUPS upgradeable registry (3-guard: stake, ZK, consistency)
│       ├── SentinelToken.sol    # ERC-20 + staking (MIN_STAKE = 1,000 SNTL)
│       └── IZKMLVerifier.sol    # Interface — ABI bridge to EZKL-generated verifier
│
└── docs/
    └── Project-Spec/            # SENTINEL-INDEX.md, SENTINEL-OVERVIEW.md,
                                 # SENTINEL-CONSTRAINTS.md, SENTINEL-ADR.md,
                                 # SENTINEL-M1–M6 spec files, SENTINEL-COMMANDS.md
```

---

## Quick Start

### 1 — Install dependencies

```bash
# ML module (Python 3.12.1 strict)
cd ml && poetry install && cd ..

# Agents module (Python ≥ 3.11)
cd agents && poetry install && cd ..
```

### 2 — Start the ML inference server

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SENTINEL_CHECKPOINT=ml/checkpoints/multilabel_crossattn_v2_best.pt

PYTHONPATH=. ml/.venv/bin/uvicorn ml.src.inference.api:app --host 0.0.0.0 --port 8001
```

Health check:

```bash
curl http://localhost:8001/health
# {
#   "status": "ok",
#   "model_loaded": true,
#   "architecture": "three_eye_v5",
#   "thresholds_loaded": true
# }
```

### 3 — Start the MCP servers

```bash
# In separate terminals (all from agents/):
cd agents
poetry run python -m src.mcp.servers.inference_server   # port 8010
poetry run python -m src.mcp.servers.rag_server         # port 8011
poetry run python -m src.mcp.servers.audit_server       # port 8012
```

### 4 — Run a full audit via LangGraph

```python
import asyncio
from src.orchestration.graph import build_graph

graph = build_graph()
result = asyncio.run(graph.ainvoke(
    {
        "contract_code":    open("ml/scripts/test_contracts/01_reentrancy_classic.sol").read(),
        "contract_address": "0x0000000000000000000000000000000000000001",
    },
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

Example response:

```json
{
  "vulnerabilities": [
    { "vulnerability_class": "Reentrancy",  "probability": 0.8943, "detected": true },
    { "vulnerability_class": "IntegerUO",   "probability": 0.7102, "detected": true }
  ],
  "thresholds": [0.70, 0.95, 0.65, 0.55, 0.50, 0.60, 0.65, 0.75, 0.60, 0.70],
  "num_nodes": 12,
  "num_edges": 18,
  "architecture": "three_eye_v5"
}
```

> `thresholds` is a list of 10 per-class values in `CLASS_NAMES` index order. There is no top-level `confidence` field — it was removed in Track 3.

---

## Port Map

| Port | Service |
|------|---------|
| 8000 | M6 API Gateway (planned) |
| 8001 | M1 FastAPI inference |
| 8010 | sentinel-inference MCP (SSE) |
| 8011 | sentinel-rag MCP (SSE) |
| 8012 | sentinel-audit MCP (SSE) |
| 1234 | LM Studio (Windows host) |
| 3000 | Dagster UI |
| 5000 | MLflow UI |

---

## Testing

```bash
# ML — 12 test modules, ~3,100 lines (synthetic data, no checkpoint required)
# Pre-flight GNN gate must pass before any training run
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. pytest ml/tests/ -v
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. pytest ml/tests/test_cfg_embedding_separation.py -v  # non-negotiable gate

# Agents — LangGraph routing, MCP servers, RAG, ingestion (41 routing tests alone)
cd agents && poetry run pytest tests/ -v

# Contracts — Foundry (requires forge)
cd contracts && forge test -vvv
cd contracts && forge test -vvv --gas-report
```

Smoke scripts (agents):

```bash
cd agents
poetry run python scripts/smoke_langgraph.py           # mock mode — no services needed
poetry run python scripts/smoke_langgraph.py --live    # live mode — all services must be up
poetry run python scripts/smoke_inference_mcp.py
poetry run python scripts/smoke_rag_mcp.py
poetry run python scripts/smoke_audit_mcp.py
```

---

## Key Constraints

Violating any of these without the corresponding rebuild or retrain produces silent failures.

| Constraint | Locked value | Break condition |
|-----------|-------------|----------------|
| `NODE_FEATURE_DIM` (GNNEncoder `in_channels`) | **12** (v2 schema) | Rebuild all ~68K graph `.pt` files + retrain |
| `FEATURE_SCHEMA_VERSION` | **`"v3"`** | Bump on any feature change; invalidates inference cache; requires graph rebuild + retrain |
| `NUM_EDGE_TYPES` | **8** (includes runtime-only `REVERSE_CONTAINS=7`) | GNNEncoder embedding table + retrain |
| `type_id` normalisation | `float(id) / 12.0` stored in `graph.x[:,0]` | Recover with `(x[:,0] * 12.0).round().long()` |
| CodeBERT model | `microsoft/codebert-base` | Rebuild token files + retrain |
| `MAX_TOKEN_LENGTH` | **512** | Rebuild token files + retrain |
| `CrossAttentionFusion output_dim` | **128** | ZKML circuit rebuild + ZKMLVerifier redeploy |
| `CLASS_NAMES` order | indices **0–9 stable** | Silent wrong-class mapping; never insert into middle |
| ZKML proxy input dim | **128** (matches fusion `output_dim`) | Full EZKL pipeline rebuild + redeploy |
| ONNX opset | **11** | EZKL 23.0.5 requirement |
| EZKL scale factor | **8192** (2¹³) | Wrong score decoded on-chain |
| `publicSignals` endianness | **little-endian** `int.from_bytes(..., 'little')` | Silent wrong score |
| solc — ZKMLVerifier.sol | **≤ 0.8.17** | Halo2 assembly deprecated opcodes |
| solc — all other contracts | **0.8.20** | Compilation failure |
| `weights_only` — graph `.pt` files | `True` (with `add_safe_globals`) | |
| `weights_only` — checkpoint `.pt` files | `False` | LoRA state dict contains peft-specific classes |
| `TRANSFORMERS_OFFLINE` | Set at **shell level** before any import | HuggingFace reads at import time |
| `add_self_loops` in GNN Phase 2 | **`False`** | Self-loops cancel directional CONTROL_FLOW signal |
| JK tensors in GNNEncoder | Collected **without** `.detach()` | Zero gradients to JK attention weights |
| `process_source()` | Writes a temp file — Slither requires a real path | Cannot pipe raw source to solc |
| No `torch.no_grad()` around `self.bert()` | LoRA A/B matrices live inside that call | Silently kills LoRA training |
| `confidence` field | **Removed** (Track 3) | `ml_result["confidence"]` → `KeyError` |
| API request field | `"source_code"` (not `"contract_code"`) | |
| RAG chunk config | `chunk_size=1536`, `overlap=128` | Change requires full index rebuild |
| slither-analyzer | **≥ 0.9.3** | Older versions silently produce wrong `in_unchecked` features |

Full details: [`docs/Project-Spec/SENTINEL-CONSTRAINTS.md`](docs/Project-Spec/SENTINEL-CONSTRAINTS.md)

---

## Development Workflow

```
main ← stable, deployed
  └── feature/* ← all development work

After landing a change that affects architecture, data contracts, or locked constants:
  1. Update the relevant file(s) in docs/Project-Spec/ (see SENTINEL-INDEX.md for routing)
  2. Bump FEATURE_SCHEMA_VERSION if node/edge schema changed
  3. Re-run affected tests before merging
  4. Run validate_graph_dataset.py before any retrain
```

---

## Module & Architecture Documentation

### Split specification (`docs/Project-Spec/`)

- [`SENTINEL-INDEX.md`](docs/Project-Spec/SENTINEL-INDEX.md) — task → file routing
- [`SENTINEL-OVERVIEW.md`](docs/Project-Spec/SENTINEL-OVERVIEW.md) — system design, data flow, ports
- [`SENTINEL-CONSTRAINTS.md`](docs/Project-Spec/SENTINEL-CONSTRAINTS.md) — locked constants (read before any implementation)
- [`SENTINEL-ADR.md`](docs/Project-Spec/SENTINEL-ADR.md) — all Architecture Decision Records
- [`SENTINEL-M1-ML.md`](docs/Project-Spec/SENTINEL-M1-ML.md) — ML model architecture, training, inference
- [`SENTINEL-M2-ZKML.md`](docs/Project-Spec/SENTINEL-M2-ZKML.md) — ZK proof pipeline
- [`SENTINEL-M3-MLOPS.md`](docs/Project-Spec/SENTINEL-M3-MLOPS.md) — MLflow, DVC, Dagster, drift detection
- [`SENTINEL-M4-AGENTS.md`](docs/Project-Spec/SENTINEL-M4-AGENTS.md) — LangGraph, MCP servers, RAG
- [`SENTINEL-M5-M6-PLATFORM.md`](docs/Project-Spec/SENTINEL-M5-M6-PLATFORM.md) — Solidity contracts + integration API
- [`SENTINEL-EVAL-BACKLOG.md`](docs/Project-Spec/SENTINEL-EVAL-BACKLOG.md) — retrain protocol, audits, backlog
- [`SENTINEL-COMMANDS.md`](docs/Project-Spec/SENTINEL-COMMANDS.md) — quick-reference CLI commands

### Per-module READMEs

- [`ml/README.md`](ml/README.md) — v5.2 architecture, data pipeline, training, inference, testing
- [`agents/README.md`](agents/README.md) — LangGraph orchestration, MCP servers, RAG, ingestion
- [`zkml/README.md`](zkml/README.md) — proxy model, EZKL pipeline, proof generation
- [`contracts/README.md`](contracts/README.md) — Foundry build, deploy, ZKMLVerifier handling