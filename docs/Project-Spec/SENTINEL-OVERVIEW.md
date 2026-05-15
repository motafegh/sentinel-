# SENTINEL — System Overview

Timeless project facts: what SENTINEL is, system data flow, module dependencies, port map.
Current state lives in `docs/STATUS.md` and `docs/ROADMAP.md`.

---

## 1. What SENTINEL Is

A decentralised AI security oracle. Smart contracts are analysed by a dual-path
GNN + CodeBERT model (v5.2: three-phase GAT with JK attention + CrossAttention fusion + LoRA).
Results are proved correct via a ZK circuit (EZKL / Groth16) and stored on-chain via AuditRegistry.
LangGraph orchestrates five specialised agents. The RAG knowledge base grounds findings
in historical DeFi exploits (DeFiHackLabs, 752 chunks).

**GitHub:** https://github.com/motafegh/sentinel-
**Environment:** WSL2 Ubuntu, RTX 3070 8GB VRAM, Python 3.12.1, Poetry
**Python venv:** `source ml/.venv/bin/activate`

---

## 2. System Data Flow

```
User uploads .sol contract
        │
        ▼
[M6 — API Gateway]  POST /v1/audit → job_id          (api/ dir does not yet exist)
        │  Celery task queue (Redis)
        ▼
[M4 — LangGraph Orchestration]
  agents/src/orchestration/graph.py
  ├── ml_assessment   → inference MCP (port 8010)
  │       │ POST /predict {"source_code": "..."}
  │       ▼
  │   [M1 — FastAPI port 8001]
  │       │ ContractPreprocessor → SentinelModel (v5.2) → per-class thresholds
  │       │   Short contracts (≤512 tokens): single-window path
  │       │   Long contracts (>512 tokens): sliding-window, stride=256, max_windows=8
  │       │     → aggregate via max() across windows
  │       ▼
  │   {label, vulnerabilities[{vulnerability_class, probability}],
  │    thresholds[],     ← per-class threshold list (NOT a single float)
  │    truncated, windows_used, num_nodes, num_edges}
  │       ← NO top-level "confidence" field. Removed in Track 3.
  │
  ├── rag_research    → RAG MCP (port 8011)
  │       │ search(query, k, filters)
  │       ▼
  │   [M4 RAG — FAISS+BM25+RRF over 752 DeFiHackLabs chunks]
  │
  ├── audit_check     → audit MCP (port 8012)
  │       │ get_audit_history(contract_address)
  │       ▼
  │   [AuditRegistry on Sepolia]
  │
  ├── static_analysis → Slither direct call (not via MCP)
  │
  └── synthesizer     → AuditReport
        │ LLM-based (qwen3.5-9b-ud) with rule-based fallback (T3-A complete)
        {overall_label, risk_probability, top_vulnerability,
         vulnerabilities[], rag_evidence[], audit_history[],
         static_findings[], narrative}
        │
        ▼
[M2 — ZKML Proof Generation]             (source complete; EZKL pipeline not yet run)
  proxy model (input_dim=128) → EZKL → proof π + publicSignals[10 class scores]
        │
        ▼
[M5 — Blockchain]  AuditRegistry.submitAudit(proof, signals)
  ZKMLVerifier.verify() on-chain → true/false              (contracts not yet compiled/deployed)
  Emit AuditSubmitted(contractAddress, proofHash, agent, score)
```

**Routing logic (LangGraph conditional):**

```python
def _is_high_risk(ml_result: dict) -> bool:
    vulns = ml_result.get("vulnerabilities", [])
    if not vulns:
        return False
    return max(v["probability"] for v in vulns) >= 0.70

# True  → deep path (parallel fan-out): ["rag_research", "static_analysis"]
#          both run concurrently in the same LangGraph superstep.
#          audit_check waits for BOTH (fan-in) before running.
#          synthesizer runs after audit_check.
# False → fast path: "synthesizer" directly (string, not list)
```

---

## 3. Module Dependency Map

```
M1 (ML) — no upstream deps
  → M2 needs trained v5.2 checkpoint for proxy distillation (input_dim=128)
  → M4/M5 agents call M1 /predict via inference MCP

M2 (ZKML) — depends on M1 trained checkpoint (v5.2 behavioral gate must pass first)
  → generates ZKMLVerifier.sol → deployed in M5
  → MUST rebuild when fusion_output_dim changes (currently 128, LOCKED)

M3 (MLOps) — depends on M1 (tracks training runs via MLflow; DVC versions checkpoints)
  → runs parallel to M1 training

M4 (Agents/MCP) — depends on M1 inference API + RAG index
  → orchestrates full audit pipeline

M5/M4 (LangGraph) — depends on M4 MCP servers + M5 Solidity contracts + M2 ZK proof
  → feeds M6

M5 (Solidity/Contracts) — depends on M2 ZKMLVerifier address
  → source complete; forge install/build/test not yet run

M6 (Integration) — depends on all modules
  → api/ directory does not exist; design auth/rate-limit before building routes
```

---

## 4. Port Map

| Port | Service | Notes |
|------|---------|-------|
| 8000 | M6 API gateway | FastAPI + Celery (api/ does not yet exist) |
| 8001 | M1 FastAPI inference | uvicorn, CUDA startup ~6s |
| 8010 | sentinel-inference MCP | SSE transport |
| 8011 | sentinel-rag MCP | SSE transport |
| 8012 | sentinel-audit MCP | SSE transport |
| 1234 | LM Studio | Windows host — WSL2 gateway IP changes on reboot |
| 3000 | Dagster UI | On demand |
| 5000 | MLflow UI | On demand |
| 5432 | PostgreSQL | Planned for M6 (not yet running) |
| 6379 | Redis | Planned for M6 Celery queue (not yet running) |
| 9090 | Prometheus | Planned for observability stack |
| 3001 | Grafana | Planned for dashboards |

---

## 5. Build Status Summary

| Module | Source Status | Runtime Status |
|--------|--------------|----------------|
| M1 ML model | Complete — v5.2 three-eye+JK | Training in progress (r3, epoch 28+) |
| M1 Inference API | Complete | Ready to start after behavioral gate passes |
| M2 ZKML | Source complete | EZKL pipeline NEVER run; blocked on M1 checkpoint |
| M3 MLOps | Complete (MLflow, DVC, Dagster, drift) | MLflow active; Dagster on demand |
| M4 Agents/MCP/RAG | Complete | Functional; MCP servers start individually |
| M5 Contracts | Source complete | forge NEVER run; lib/ empty |
| M6 Integration API | Does not exist | Design phase only |

---

## 6. Environment Variables

As these may change throughout the project, ask for current values when needed.
Never hardcode; always use env vars or `.env` files.

Key env vars:
```
TRANSFORMERS_OFFLINE=1              — SHELL level; Python setting has no effect
SENTINEL_CHECKPOINT=<path>          — ML inference checkpoint
SENTINEL_THRESHOLDS=<path>          — per-class threshold JSON
SENTINEL_PREDICT_TIMEOUT=60         — inference timeout (seconds)
SENTINEL_DRIFT_BASELINE=<path>      — drift baseline JSON
SENTINEL_DRIFT_CHECK_INTERVAL=50    — requests between drift checks
SEPOLIA_RPC_URL=<url>               — Ethereum Sepolia RPC (audit MCP)
AUDIT_MOCK=true                     — force mock mode for audit MCP (no RPC needed)
DAGSTER_HOME=agents/.dagster        — Dagster workspace
```
