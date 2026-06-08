# SENTINEL

Smart contracts are immutable once deployed — a single vulnerability can result in irreversible loss of funds. Existing audit tools are either static analysers that miss context-dependent bugs, or expensive manual reviews that don't scale. SENTINEL is a **decentralised AI security oracle**: a dual-path GNN + GraphCodeBERT vulnerability detector with zero-knowledge proof generation and on-chain audit registration so that any result can be independently verified without trusting the agent that produced it.

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
│        │               Three-eye SentinelModel v8
│        │               GNNEncoder (3-phase GAT + JK) + GraphCodeBERT+LoRA+GNNPrefix
│        │               + CrossAttentionFusion
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
| M1 | `ml/` | Three-eye SentinelModel v8: three-phase 7-layer GAT (JK) + GraphCodeBERT LoRA + GNN prefix injection (K=48) + CrossAttentionFusion; 10-class multi-label; per-class threshold tuning; sliding-window inference; FastAPI with Prometheus metrics | ✅ Architecture complete — GCB-P1 training running overnight |
| M2 | `zkml/` | ProxyMLP distillation (128→64→32→10), EZKL circuit setup, per-audit Groth16 proof generation | ⚠️ Source complete — awaiting stable GCB-P1 checkpoint |
| M3 | `ml/` (mlops) | MLflow experiment tracking (`sqlite:///mlruns.db`), DVC data versioning, Dagster RAG scheduling, model registry promotion | ✅ Complete |
| M4 | `agents/` | LangGraph orchestration (parallel fan-out), 3 MCP servers (SSE), hybrid RAG (FAISS+BM25+RRF), ingestion pipeline, feedback loop | ✅ Complete — 46/46 tests pass |
| M5 | `contracts/` | SentinelToken (ERC-20 + staking), AuditRegistry (UUPS upgradeable), IZKMLVerifier interface — Foundry test suite written | ⚠️ Source complete — forge not yet run on latest |
| M6 | `api/` | FastAPI + Celery gateway, Docker Compose full-stack | ❌ Planned |

---

## Current Model

**Architecture: SentinelModel v8 + GraphCodeBERT + GNN Prefix Injection**

`FEATURE_SCHEMA_VERSION = "v8"` · `NODE_FEATURE_DIM = 11` · backbone: `microsoft/graphcodebert-base`

**Three-eye classifier:**
- **GNN eye:** three-phase 7-layer GAT with Jumping Knowledge connections and per-phase LayerNorm; `hidden_dim=256`; 11 edge types; `NodeType` IntEnum (13 types)
- **Transformer eye:** GraphCodeBERT (124M params, frozen) + LoRA r=16 α=32 on Q+V of all 12 layers; GNN prefix (K=48 declaration nodes projected [256→768]) prepended via `inputs_embeds` after warmup epoch 15
- **Fused eye:** CrossAttentionFusion — bidirectional node↔token cross-attention; `output_dim=128` LOCKED

Classifier head: `cat [B, 384]` → `Linear(384,192)` → `GELU` → `Linear(192,10)` → logits.

### Training Progress

| Run | Phase 2 edges | Best ep | Tuned F1 | Status |
|-----|---------------|---------|----------|--------|
| v7.0 | CF only | 23 | 0.2875 | Complete |
| PLAN-3A | CF+CALL_ENTRY+RETURN_TO | 41 | **0.2877** | **Best checkpoint** |
| v8.0-B | PLAN-3A + label clean | 10 | killed | Confirmed ~0.287 ceiling |
| GCB-P0 | GraphCodeBERT 5-ep gate | 3 | 0.2178 raw | Gate passed |
| GCB-P1 | CF+CE+RT, K=48, warmup=15 | running | — | **Overnight run** |
| **Run 9** | v9 schema + pre-Run-9 fixes | ep14 | 0.2586 | **In flight** — uses old noisy labels |
| **Run 10** | v1.3 verified labels | — | — | **Planned** — pending Run 9 completion |

**Ceiling conclusion:** all CodeBERT-backbone runs converge to ~0.287 tuned F1. GraphCodeBERT (GCB-P0 ep1–3) already shows ExternalBug and TOD non-zero — these were 0.000 in all CodeBERT runs. GCB-P1 is the active architectural intervention.

**⚠️ Label quality note:** BCCC labels had 69.8% false positive rate across 9 vulnerability classes. Phase 5 (2026-06-08) produced verified dataset `contracts_clean_v1.3.csv` with 46,977 labels dropped. Run 10 will use these verified labels.

**Active fallback checkpoint:** `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt`
Trained on deduplicated BCCC corpus · tuned F1-macro 0.5422.

---

## Output Classes

10-class multi-label output. Index order is **locked** — reordering breaks all checkpoints and the ZKML circuit.

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
| slither-analyzer | ≥ 0.9.3 | Graph extraction (hard minimum — older versions produce wrong features) |

---

## Environment Variables

All must be exported at **shell level** before any service starts. Setting them inside Python is too late — `TRANSFORMERS_OFFLINE` is read at `transformers` import time.

### ML Inference Server (M1 — port 8001)

| Variable | Default | Required | Notes |
|----------|---------|----------|-------|
| `TRANSFORMERS_OFFLINE` | — | **Yes** | Must be `1`. Set at shell level. |
| `HF_HUB_OFFLINE` | — | **Yes** | Set alongside `TRANSFORMERS_OFFLINE`. |
| `TRITON_CACHE_DIR` | — | **Yes (WSL2)** | Set to `/tmp/triton_cache` to avoid p9io crash. |
| `SENTINEL_CHECKPOINT` | `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt` | **Yes** | Path to `.pt` checkpoint. |
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
│   │   ├── preprocessing/       # graph_schema.py (NodeType IntEnum, STRUCTURAL_PREFIX_TYPES),
│   │   │                        # graph_extractor.py — Slither → v8 graphs
│   │   ├── models/              # sentinel_model.py (v8, GNN prefix),
│   │   │                        # gnn_encoder.py (three-phase GAT, Embedding(11,64)),
│   │   │                        # transformer_encoder.py (GraphCodeBERT + LoRA + prefix),
│   │   │                        # fusion_layer.py (CrossAttentionFusion, compile-safe)
│   │   ├── training/            # trainer.py (TrainConfig w/ prefix params, CLASS_NAMES),
│   │   │                        # losses.py (AsymmetricLoss)
│   │   ├── datasets/            # dual_path_dataset.py, dual_path_collate_fn
│   │   └── inference/           # api.py, predictor.py (prefix-aware), preprocess.py
│   ├── scripts/                 # train.py, tune_threshold.py, promote_model.py,
│   │                            # audit_prefix_node_counts.py, manual_test.py,
│   │                            # create_cache.py, create_splits.py, monitor.sh, …
│   ├── tests/                   # pytest modules (preprocessing, model, training, cache, dataset)
│   └── data/
│       ├── graphs/              # 41,576 .pt graph files (v8 schema, 11-dim)
│       ├── tokens_windowed/     # 44,470 .pt token files ([4,512], stride=256)
│       ├── splits/deduped/      # train=29,103 / val=6,236 / test=6,237 (.npy)
│       └── cached_dataset_v8.pkl  # 2.2 GB paired cache
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
│       │                        # build_index.py
│       ├── ingestion/           # pipeline.py, deduplicator.py, feedback_loop.py,
│       │                        # scheduler_dagster.py
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
    ├── ACTIVE_PLAN.md           # current phase status and gate tracking
    ├── CHANGELOG.md             # full project history
    ├── proposal/                # EXECUTION_PLAN.md, GCB+prefix proposal
    └── Project-Spec/            # SENTINEL-INDEX.md, SENTINEL-OVERVIEW.md, …
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
export TRITON_CACHE_DIR=/tmp/triton_cache
export SENTINEL_CHECKPOINT=ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt

PYTHONPATH=. ml/.venv/bin/uvicorn ml.src.inference.api:app --host 0.0.0.0 --port 8001
```

Health check:

```bash
curl http://localhost:8001/health
# {"status": "ok", "model_loaded": true, "architecture": "three_eye_v7", "thresholds_loaded": true}
```

### 3 — Start the MCP servers

```bash
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
| 5000 | MLflow UI (`sqlite:///mlruns.db`) |
| 5432 | PostgreSQL (planned — M6 Celery backend) |
| 6379 | Redis (planned — M6 task queue) |
| 9090 | Prometheus (planned) |
| 3001 | Grafana (planned) |

---

## Testing

```bash
# ML — preprocessing, model, training, cache, dataset
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. pytest ml/tests/ -v

# Agents — LangGraph routing, MCP servers, RAG, ingestion (46 tests)
cd agents && poetry run pytest tests/ -v

# Contracts — Foundry (requires forge)
cd contracts && forge test -vvv
```

Behavioral smoke (ML):

```bash
# 20 test contracts, 19 expected detections
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/manual_test.py \
    --checkpoint ml/checkpoints/<checkpoint>.pt
```

---

## Key Constraints

Violating any of these without the corresponding rebuild or retrain produces silent failures.

| Constraint | Locked value | Break condition |
|-----------|-------------|----------------|
| `NODE_FEATURE_DIM` | **11** | Rebuild all 41,576 graph `.pt` files + retrain |
| `FEATURE_SCHEMA_VERSION` | **`"v8"`** | Bump on schema change; invalidates inference cache |
| `NUM_EDGE_TYPES` | **11** | `Embedding(11,64)` in GNNEncoder + retrain |
| `NUM_CLASSES` | **10** | CLASS_NAMES order locked; ZKML circuit depends on it |
| `CrossAttentionFusion output_dim` | **128** | ZKML proxy MLP + ZKMLVerifier redeploy |
| `ZKML proxy input dim` | **128** | Must match fusion `output_dim` |
| ONNX opset | **11** | EZKL 23.0.5 requirement |
| EZKL scale factor | **8192** (2¹³) | Wrong score decoded on-chain |
| `publicSignals` endianness | **little-endian** | Silent wrong score |
| Backbone model | `microsoft/graphcodebert-base` | Token files rebuild + retrain |
| Checkpoint state dict | Strip `._orig_mod.` infix | `torch.compile` adds this; strip at save time |
| `weights_only` for graph `.pt` | `False` | PyG 2.7 metadata not safe-tensors serialisable |
| `weights_only` for checkpoint `.pt` | `False` | LoRA PEFT objects not safe-tensors serialisable |
| Prefix position IDs | prefix=1, code=3..466 | RoBERTa 0=BOS, 1=padding, 2=EOS reserved |
| `gnn_to_bert_proj` at inference | always active | `predictor.py` sets `_current_epoch=9999` |
| `add_self_loops` in Phase 2 | `False` | Self-loops cancel directional CF signal |
| `TRANSFORMERS_OFFLINE` | Set at **shell level** | Read at `transformers` import time |
| solc — ZKMLVerifier.sol | **≤ 0.8.17** | Halo2 assembly deprecated opcodes |
| solc — all other contracts | **0.8.20** | Compilation failure |

Full details: [`docs/Project-Spec/SENTINEL-CONSTRAINTS.md`](docs/Project-Spec/SENTINEL-CONSTRAINTS.md)

---

## Development Workflow

```
main ← stable, deployed
  └── feature/* ← all development work

After landing a change that affects architecture, data contracts, or locked constants:
  1. Update the relevant file(s) in docs/Project-Spec/ (see SENTINEL-INDEX.md for routing)
  2. Bump FEATURE_SCHEMA_VERSION in graph_schema.py if node/edge schema changed
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

- [`ml/README.md`](ml/README.md) — v8 + GCB architecture, GNN prefix injection, data pipeline, training, inference
- [`agents/README.md`](agents/README.md) — LangGraph orchestration, MCP servers, RAG, ingestion
- [`zkml/README.md`](zkml/README.md) — proxy model, EZKL pipeline, proof generation
- [`contracts/README.md`](contracts/README.md) — Foundry build, deploy, ZKMLVerifier handling

### Project history

- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — full history from v4 through current GCB-P1 run
- [`docs/ACTIVE_PLAN.md`](docs/ACTIVE_PLAN.md) — current phase gates and status
- [`docs/proposal/EXECUTION_PLAN.md`](docs/proposal/EXECUTION_PLAN.md) — Phase 3.6 gate-by-gate plan
