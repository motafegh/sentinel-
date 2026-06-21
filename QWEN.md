# SENTINEL — Qwen Code Context

> **Decentralised AI Security Oracle for Smart Contracts**
> Dual-path GNN + GraphCodeBERT vulnerability detector with ZK proof generation and on-chain audit registration.

---

## Current State (2026-06-18)

**🟢 Run 12 COMPLETE** — F1_tuned=**0.7004** (ep50), honest OOD F1=0.8743 (66 contracts). **In Staging** (MLflow `sentinel-vulnerability-detector` v1). NOT Production.
**🟢 MLOps Phases A+B+C COMPLETE** — Real drift baseline, Docker deployment stack (inference + Prometheus), drift monitoring active.
**🟢 Agents E2E tested** — 9-node graph runs end-to-end with real LLM+MCP+ML. **12 real bugs found + fixed.** GO with CAUTION.
**🟡 Run 13 PLANNED** (~3 weeks): Drop GasException → NUM_CLASSES=9, extend L4 to drop `loc`, strip Solidifi `bug_*` prefix, inject 658 BCCC ME contracts, ExternalBug label quality review.
**🟡 DIVE crosswalk fix IN PROGRESS (2026-06-18)** — ExternalBug + Reentrancy labels being cleaned via Slither corroboration (74% and 50.7% positive rates are spurious folder-membership mapping).
**🔴 Benchmark contamination:** SmartBugs 95.8% / SolidiFI 82.9% in v3 train. All future benchmarks must pass **0% contamination hard gate** (`ml/scripts/audit/check_contamination_v3.py`).
**🟢 Testing suite overhaul complete** — 9 gates wired into framework CLI (`python -m ml.testing_specs.framework.cli run`), 36 unit tests at 91% coverage.

---

## Project Overview

SENTINEL analyses Solidity smart contracts for vulnerability classes using a four-eye architecture:
- **GNN eye** — 8-layer three-phase Graph Attention Network (JK connections)
- **Transformer eye** — GraphCodeBERT + LoRA + GNN prefix injection (K=48)
- **Fused eye** — CrossAttentionFusion (bidirectional node↔token cross-attention)
- **CFG eye** — Phase 2 CFG-only pooling

Results flow through LangGraph orchestration (M4) with hybrid RAG (FAISS+BM25), static analysis (Slither/Aderyn), and on-chain ZK proof verification (EZKL/Groth16) on Sepolia.

### Modules

| # | Path | Status |
|---|------|--------|
| M1 | `ml/` | v8.1 four-eye architecture, v9 schema (12-dim nodes, 12 edge types) |
| M2 | `zkml/` | Proxy MLP distillation (128→64→32→10), EZKL circuit setup |
| M3 | `ml/` (mlops) | MLflow, DVC, Dagster, drift detection — Phases A+B+C complete |
| M4 | `agents/` | LangGraph orchestration, 4 MCP servers (SSE), RAG, ingestion — E2E tested |
| M5 | `contracts/` | Solidity (Foundry) — AuditRegistry, SentinelToken, ZKMLVerifier |
| M6 | `api/` | Planned — FastAPI + Celery gateway |

---

## Building and Running

### Install

```bash
# Root workspace
poetry install

# Agents (separate venv)
cd agents && poetry install && cd ..
```

### Start ML Inference Server (M1 — port 8001)

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export SENTINEL_CHECKPOINT=ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt

PYTHONPATH=. ml/.venv/bin/uvicorn ml.src.inference.api:app --host 0.0.0.0 --port 8001
```

Health check:
```bash
curl http://localhost:8001/health
```

### Start MCP Servers (M4)

```bash
cd agents
poetry run python -m src.mcp.servers.inference_server   # :8010
poetry run python -m src.mcp.servers.rag_server         # :8011
poetry run python -m src.mcp.servers.audit_server       # :8012
poetry run python -m src.mcp.servers.graph_inspector_server  # :8013
```

### Training

```bash
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. \
    python ml/scripts/train.py --run-name v13-$(date +%Y%m%d) --epochs 100
```

### Testing

```bash
# ML tests
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. pytest ml/tests/ -v

# Agents tests
cd agents && poetry run pytest tests/ -v

# Contracts (requires forge)
cd contracts && forge test -vvv

# Testing gates framework
python -m ml.testing_specs.framework.cli run
```

---

## Port Map

| Port | Service |
|------|---------|
| 8001 | M1 FastAPI inference |
| 8010 | sentinel-inference MCP |
| 8011 | sentinel-rag MCP |
| 8012 | sentinel-audit MCP |
| 8013 | graph-inspector MCP |
| 1234 | LM Studio (Windows host) |
| 3000 | Dagster UI |
| 5000 | MLflow UI |

---

## Key Invariants (DO NOT CHANGE)

| Constant | Value | Break condition |
|----------|-------|-----------------|
| `FEATURE_SCHEMA_VERSION` | `"v9"` | Bump on schema change; invalidates cache |
| `NODE_FEATURE_DIM` | **12** | Rebuild all graph `.pt` files + retrain |
| `NUM_NODE_TYPES` | **14** | Type embedding + retrain |
| `NUM_EDGE_TYPES` | **12** | GNNEncoder Embedding(12,64) + retrain |
| `NUM_CLASSES` | **10** | CLASS_NAMES order locked; ZKML circuit depends on it *(Run 13: → 9)* |
| `fusion_output_dim` | **128** | ZKML proxy MLP depends on this |
| `gnn_num_layers` | **8** | 2+3+3 three-phase architecture |
| `SENTINEL_GNN_NUM_LAYERS` | **8** | Raises ValueError if violated |
| `gnn_prefix_k` | **48** | Covers P95 of contracts |
| EZKL opset | **11** | EZKL 23.0.5 requirement |
| EZKL scale factor | **8192** (2¹³) | Wrong score decoded on-chain |
| `TRANSFORMERS_OFFLINE` | Shell-level `1` | Read at `transformers` import time |
| `add_self_loops` (Phase 2) | `False` | Self-loops cancel directional CFG signal |
| `weights_only` (graph/checkpoint `.pt`) | `False` | PyG/LoRA not safe-tensors serialisable |
| Checkpoint keys | Strip `._orig_mod.` infix | `torch.compile` adds this prefix |
| Checkpoint dtype | BF16 → call `.float()` | For diagnostic inference outside training loop |
| `gnn_to_bert_proj` at inference | Always active | `predictor.py` sets `_current_epoch=9999` |
| After dtype cast | Call `model.gnn.refresh_dtype_cache()` | A26 fix |

---

## Output Classes (multi-label, index order locked)

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

**⚠️ Run 13 will drop GasException → NUM_CLASSES=9.** This requires routing table updates and ZKML circuit redeploy.

---

## Active Checkpoint, Splits, MLflow

- **Active checkpoint:** `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt` (~280 MB)
  - Immutable copy: `*_FINAL.pt`
  - MLflow: `sentinel-vulnerability-detector` experiment, v1 (Staging)
- **Active splits:** `data_module/data/splits/v3/` (18,596 train / 1,983 val / 1,914 test; **0% leakage**)
  - v1 splits: **DO NOT USE** (45% contamination)
- **MLflow:** `sqlite:///mlruns.db`, experiment `sentinel-retrain-v2`
- **Drift baseline:** `ml/data/drift_baseline_run12.json` (real, synthetic warmup — replace with real traffic when available)
- **Logging:** `StructuredLogger("ml/logs/<run_name>")` — 3 JSONL streams. `f1_macro_tuned` in `epoch_summary.jsonl`.

---

## Development Conventions

### Workflow Rules
- **Plan before code.** Confirm a session plan before implementation. Surface a condensed plan, wait for acknowledgment.
- **Trust source code only.** `.py`, `.sol`, `.ts`, `.sh` files are canonical truth. Docs, READMEs, docstrings are stale until verified against code.
- **After landing changes affecting architecture/data contracts/locked constants:**
  1. Update `docs/Project-Spec/` docs
  2. Bump `FEATURE_SCHEMA_VERSION` if node/edge schema changed
  3. Re-run affected tests before merging
  4. Run `validate_graph_dataset.py` before any retrain

### Tooling
- **Python 3.12.1 strict** for ML module; ≥ 3.11 for agents
- **Poetry** for dependency management; **two separate venvs:**
  - Root `.venv` — most ML deps but NOT peft
  - `ml/.venv` — full ML stack including peft; training uses `ml/.venv/bin/python` directly
- **Black** (line-length 100), **isort** (profile=black), **mypy** (warn_return_any, disallow_untyped_defs)
- **pytest** with coverage (`--cov --cov-report=html --cov-report=term`)
- **Git workflow:** `main` ← stable; `feature/*` ← development branches

### WSL2 Notes
- Always use Linux paths (`~/projects/...`), never Windows paths
- `git config core.autocrlf false`
- Scripts won't run → `chmod +x script.sh` before debugging
- Run Python: `poetry run python <script>` or `ml/.venv/bin/python <script>` for peft-dependent scripts

---

## Environment

- **OS:** Linux (WSL2 Ubuntu on Windows 11)
- **Hardware:** RTX 3070 (8 GB VRAM), i7-12700H (14 cores, 20 threads), 64 GB RAM, Samsung 990 PRO SSD
- **Slither:** ≥ 0.9.3 (hard minimum — older versions produce wrong features)
- **Foundry:** `forge` + `cast` for Solidity build/deploy

---

## Important Paths

```
ml/src/models/              — sentinel_model.py, gnn_encoder.py, transformer_encoder.py, fusion_layer.py
ml/src/inference/           — api.py (:8001), predictor.py (prefix-aware), drift_detector.py
ml/src/preprocessing/       — graph_schema.py, graph_extractor.py (Slither → v9, 28-line shim → sentinel_data)
ml/src/training/            — trainer.py, training_logger.py, losses.py
ml/src/datasets/            — sentinel_dataset.py, collate.py
ml/scripts/train.py         — Training entry point
ml/scripts/interpretability/ — 21 experiment scripts
ml/data/                    — graphs/, tokens_windowed/, splits/, cached_dataset_v9.pkl
ml/deploy/                  — Dockerfile.inference, docker-compose.yml, prometheus.yml
ml/testing_specs/           — Gate attestations, framework CLI, test specs
ml/checkpoints/             — Active + _archive/

agents/src/orchestration/   — graph.py, nodes.py, state.py, routing.py
agents/src/mcp/servers/     — 4 MCP servers (inference, rag, audit, graph_inspector)
agents/src/rag/             — HybridRetriever (FAISS+BM25+RRF), chunker, embedder, build_index.py
agents/src/ingestion/       — pipeline.py, feedback_loop.py, scheduler_dagster.py
agents/src/llm/             — client.py (LM Studio, 4 model roles)
agents/scripts/run_real_audit.py — E2E test harness (--no-llm/--profile)
agents/test_audit_reports/  — E2E run reports + ANALYSIS_SUMMARY.md

contracts/src/              — AuditRegistry.sol, SentinelToken.sol, IZKMLVerifier.sol
zkml/src/distillation/      — proxy_model.py, train_proxy.py, export_onnx.py
zkml/src/ezkl/              — setup_circuit.py, run_proof.py, extract_calldata.py

data_module/data/exports/   — v3 export (active, artifact_hash 5cc5...)
data_module/data/splits/v3/ — Active splits (0% leakage)
data_module/benchmarks/     — benchmark_v0.1_quickstart (66 contracts)

docs/proposal/              — ML + Agents + MLOps proposals
docs/plans/                 — Run planning handoffs
docs/plan/agents/           — Agents implementation plans
```
