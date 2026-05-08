# SENTINEL — System Overview

Timeless project facts: what SENTINEL is, system data flow, module dependencies, port map.
Current state lives in `docs/STATUS.md` and `docs/ROADMAP.md`.

---

## 1. What SENTINEL Is

A decentralised AI security oracle. Smart contracts are analysed by a dual-path
GNN + CodeBERT model with CrossAttention fusion and LoRA fine-tuning. Results are proved
correct via a ZK circuit (EZKL / Groth16) and stored on-chain via AuditRegistry.
LangGraph orchestrates five specialised agents. The RAG knowledge base grounds findings
in historical DeFi exploits.

**GitHub:** https://github.com/motafegh/sentinel-
**Environment:** WSL2 Ubuntu, RTX 3070 8GB VRAM, Python 3.12, Poetry

---

## 2. System Data Flow

```
User uploads .sol contract
        │
        ▼
[M6 — API Gateway]  POST /v1/audit → job_id
        │  Celery task queue (Redis)
        ▼
[M5 — LangGraph Orchestration]
  agents/src/orchestration/graph.py
  ├── ml_assessment   → inference MCP (port 8010)
  │       │ POST /predict {"source_code": "..."}
  │       ▼
  │   [M1 — FastAPI port 8001]
  │       │ ContractPreprocessor → SentinelModel → per-class thresholds
  │       │   Long contracts (>512 tokens): sliding-window path (T1-C)
  │       │   → aggregate via max() across windows
  │       ▼
  │   {label, vulnerabilities[{vulnerability_class, probability}],
  │    thresholds[],                ← per-class threshold list (Fix #6; was single float "threshold")
  │    truncated, windows_used, num_nodes, num_edges}
  │       ← NO top-level confidence field. confidence was binary-era. Removed.
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
[M2 — ZKML Proof Generation]
  proxy model (input_dim=128) → EZKL → proof π + publicSignals[10 class scores]
        │
        ▼
[M5 — Blockchain]  AuditRegistry.submitAudit(proof, signals)
  ZKMLVerifier.verify() on-chain → true/false
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
#          both nodes run concurrently in the same LangGraph superstep.
#          audit_check waits for BOTH to complete (fan-in) before running.
#          synthesizer runs after audit_check.
# False → fast path: "synthesizer" directly (string, not list)
```

---

## 3. Module Dependency Map

```
M1 (ML) — no upstream deps
  → M2 needs trained checkpoint for knowledge distillation (proxy input_dim=128)
  → M4/M5 agents call M1 /predict via inference MCP

M2 (ZKML) — depends on M1 trained checkpoint
  → generates ZKMLVerifier.sol → deployed in M5
  → MUST rebuild when fusion output_dim changes (currently 128)

M3 (MLOps) — depends on M1 (tracks training runs)
  → runs parallel to M1 training

M4 (Agents/MCP) — depends on M1 inference API + RAG index
  → feeds M5 LangGraph

M5 (LangGraph) — depends on M4 MCP servers + M5 Solidity contracts
  → feeds M6

M5 (Solidity/Contracts) — depends on M2 ZKMLVerifier address
  → feeds M6

M6 (Integration) — depends on all modules
```

---

## 4. Port Map

| Port | Service | Notes |
|------|---------|-------|
| 8000 | M6 API gateway | FastAPI + Celery (not yet built) |
| 8001 | M1 FastAPI inference | uvicorn, CUDA startup ~6s |
| 8010 | sentinel-inference MCP | SSE transport |
| 8011 | sentinel-rag MCP | SSE transport |
| 8012 | sentinel-audit MCP | SSE transport |
| 1234 | LM Studio | Windows host — WSL2 gateway IP changes on reboot |
| 3000 | Dagster UI | On demand |
| 5000 | MLflow UI | On demand |

---

## 5. Environment Variables

As these may change throughout the project, ask for current values when needed.
Never hardcode; always use env vars or `.env` files.
