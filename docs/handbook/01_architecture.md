# 01 — Architecture

## TL;DR

```
What: Decentralised AI security oracle — Solidity contracts analyzed by
      GNN+GraphCodeBERT, results ZK-proven and stored on-chain
Modules: DATA → ML → ZKML → CONTRACTS ← AGENTS
Ports: 8001 (ML API), 8010 (MCP inference), 8012 (MCP audit), 8545 (Anvil)
Tests: see §6 below
```

---

## 1. System diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   SENTINEL SYSTEM                     │
                    │                                                     │
  ┌──────────┐     │  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
  │  .sol    │─────┼─▶│  DATA    │───▶│   ML     │───▶│  ZKML    │       │
  │  source  │     │  │ module   │    │ module   │    │ module   │       │
  │  code    │     │  │          │    │          │    │          │       │
  └──────────┘     │  │ graph    │    │ GNN+     │    │ proxy    │       │
                    │  │ extractor│    │ CodeBERT │    │ (128→10) │       │
                    │  │ + tokens │    │ →fusion  │    │ EZKL     │       │
                    │  │          │    │ [128-dim]│    │ proof    │       │
                    │  └──────────┘    └────┬─────┘    └────┬─────┘       │
                    │                       │               │             │
                    │                       │  ┌──────────┐ │             │
                    │                       └─▶│  AGENTS  │─┘             │
                    │                          │ module   │               │
                    │                          │          │               │
                    │                          │ LangGraph│               │
                    │                          │ 14 nodes │               │
                    │                          │ + MCP    │               │
                    │                          │ + gateway│               │
                    │                          └────┬─────┘               │
                    │                               │                     │
                    │                               ▼                     │
                    │                          ┌──────────┐               │
                    │                          │CONTRACTS │               │
                    │                          │ module   │               │
                    │                          │          │               │
                    │                          │AuditReg  │               │
                    │                          │ V1 + V2  │               │
                    │                          │on-chain  │               │
                    │                          └──────────┘               │
                    └─────────────────────────────────────────────────────┘
```

**Data flow** (what happens to a contract being audited):

```
.sol source
    │
    ▼  DATA module: graph_extractor + windowed_tokenizer
PyG graph [N nodes, 12 features] + tokens [4 windows × 512]
    │
    ▼  ML module: GNN (8-layer GAT) + CodeBERT + CrossAttentionFusion
fusion_embedding [128-dim]  ←── the ZK boundary
    │
    ├──▶ AGENTS: probabilities, 3-tier label, verdicts (off-chain audit)
    │
    ▼  ZKML module: ProxyModel(fusion) → 10 logits → EZKL proof
proof.json [hex_proof + 138 publicSignals]
    │
    ▼  AGENTS MCP submit_audit → web3.py transact
AuditRegistry.submitAuditV2(classScores[10], proof, publicSignals[138], modelHash)
    │
    ▼  CONTRACTS module: on-chain storage
AuditSubmittedV2 event → feedback_loop → RAG (closes the loop)
```

**Service flow** (what processes run and talk to each other):

```
User ──HTTP──▶ Gateway (:8011)
                   │
                   ├──▶ LangGraph pipeline (in-process)
                   │        │
                   │        ├──▶ MCP inference (:8010) ──▶ ML API (:8001)
                   │        ├──▶ MCP audit (:8012) ─────▶ Anvil (:8545)
                   │        ├──▶ MCP rag
                   │        └──▶ MCP representation (:8014)
                   │
                   └──▶ SQLite JobStore (data/jobs.db)
```

---

## 2. Module map

| Module | Language / framework | Code dir | Test command | Tests |
|---|---|---|---|---|
| **DATA** | Python, PyG, Slither | `data_module/` | `cd data_module && .venv/bin/python -m pytest` | 625 |
| **ML** | Python, PyTorch, Transformers | `ml/src/` | `source ml/.venv/bin/activate && python -m pytest ml/tests/` | 217 |
| **ZKML** | Python, EZKL, Halo2 | `zkml/src/` | `source ml/.venv/bin/activate && python -m pytest zkml/tests/` | 37 |
| **CONTRACTS** | Solidity, Foundry | `contracts/src/` | `cd contracts && forge test` | 66 |
| **AGENTS** | Python, LangGraph, MCP | `agents/src/` | `cd agents && poetry run pytest` | 634 |
| **Total** | | | | **1,579** |

---

## 3. Port map

| Port | Service | Started by | Config location |
|---|---|---|---|
| 8001 | ML inference API (FastAPI + uvicorn) | `uvicorn ml.src.inference.api:app --port 8001` | `ml/src/inference/api.py:74-80` (checkpoint), `agents/.env` (MODULE1_INFERENCE_URL) |
| 8010 | MCP inference proxy | `python -m src.mcp.servers.inference_server` | `agents/.env` (MCP_INFERENCE_PORT=8010) |
| 8012 | MCP audit server (read + write) | `python -m src.mcp.servers.audit._server` | `agents/.env` (MCP_AUDIT_PORT=8012) |
| 8014 | MCP representation server | `python -m src.mcp.servers.representation_server` | `agents/.env` (MCP_REPRESENTATION_PORT=8014) |
| 8545 | Anvil local chain | `anvil --port 8545` | Anvil default |

---

## 4. The dual-verdict design

SENTINEL produces **two verdicts** per audit — one provable, one rich:

**verdict_provable** — deterministic only
- Computed from evidence where `deterministic=True`: ML predictions, Slither findings, Aderyn findings, fuse() math
- Reproducible across runs (same input → same output)
- This is what gets ZK-proved and anchored on-chain
- Source: `fuse.py:163` — `_fuse_for_evidence(det_items)` where `det_items` are deterministic-only

**verdict_full** — includes everything
- Computed from ALL evidence: adds LLM debate (`cross_validator`), RAG research, consensus engine
- Richer context, better narrative, but non-deterministic (LLM varies even at temp=0)
- This is what the human-readable audit report uses
- Source: `fuse.py:166` — `_fuse_for_evidence(items)` where `items` includes all evidence

**Why split:** The ZK circuit can only prove deterministic computations. The LLM debate is non-deterministic (model internals, GPU non-determinism, batching effects). So we prove the deterministic tier and report the full tier separately.

**Where in final_report:** `synthesizer.py:344-386` — both verdicts are included:
- `final_report["vulnerability_verdicts"]` — per-class verdicts
- `final_report["consensus_verdict"]` — overall
- `final_report["model_provenance"]` — model_hash for ZK anchoring
- `final_report["on_chain"]` — tx_hash after submission

---

## 5. The 14-node pipeline

The AGENTS module is a LangGraph with 14 nodes. The entry point is `ml_assessment`, the terminal node is `synthesizer` (or `reflection` → `explainer` → `visualizer` on the deep path).

```
ml_assessment → quick_screen → evidence_router ─┬─ (fast) ────────────→ synthesizer
                                                 │
                                                 └─ (deep) ─→ rag_research ─┐
                                                              static_analysis ─┤
                                                              graph_explain ────┤
                                                              formal_verification┤
                                                                              ↓
                                                    audit_check → consensus_engine → cross_validator → synthesizer
                                                                                                    │
                                                                                                    ▼
                                                                                              reflection
                                                                                                    │
                                                                                                    ▼
                                                                                              explainer → visualizer
```

Source: `agents/src/orchestration/graph.py:172-221` — all node registrations and edges.

**Fast path** (contract is clearly safe): `ml_assessment → quick_screen → evidence_router → synthesizer`
**Deep path** (suspicious findings): all nodes fire, including LLM debate and formal verification

---

## 6. Key configuration locations

| What | File | Key contents |
|---|---|---|
| All env vars | `agents/.env` | Ports, RPC URL, operator key, LM Studio URL, timeouts |
| L1 decision numbers | `agents/configs/verdicts_default.yaml` | Thresholds, weights, bands (hand-set, versioned) |
| L3 fitted weights | `agents/configs/reliability_v3.yaml` | Per-tool reliability (Bayesian shrinkage, α=5) |
| Solidity compiler | `contracts/foundry.toml` | solc=0.8.22, optimizer, OZ remappings |
| Frozen proxy architecture | `zkml/src/distillation/proxy_model.py:104-108` | 128/64/32/10 — changing these = new circuit |
| ML checkpoint path | `ml/src/inference/api.py:74-80` | 3-level: SENTINEL_CHECKPOINT env > mlops_config.json > default |
| EZKL circuit settings | `zkml/ezkl/settings.json` | model_instance_shapes, scale=13, logrows |
| Project rules | `CLAUDE.md` | Rule 5B (decision numbers), Rule 5C (no silent failures) |

---

## 7. Test commands (copy-pasteable)

```bash
# AGENTS (634 tests, ~28s)
cd ~/projects/sentinel/agents && poetry run pytest -q

# ML (213 passed, 4 failed — pre-existing fixture issues, ~60s)
cd ~/projects/sentinel && source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 SENTINEL_CHECKPOINT=ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt \
  python -m pytest ml/tests/ -q --tb=no

# ZKML (37 tests, ~15s)
cd ~/projects/sentinel && source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 python -m pytest zkml/tests/ -q

# DATA (616 passed, 9 failed — pre-existing, ~34s)
cd ~/projects/sentinel/data_module && .venv/bin/python -m pytest -q --tb=no

# CONTRACTS (66 tests, ~2s)
cd ~/projects/sentinel/contracts && forge test
```

---

## Deep reference

- → `docs/learning/01_orchestration_pipeline.md` — deep dive on the 14-node pipeline
- → `docs/learning/04_reproducibility_determinism.md` — deep dive on deterministic mode + ZK boundary
- → `docs/learning/10_decision_numbers.md` — L0→L3 maturity ladder, Rule 5B/5C
- → source: `graph.py`, `state.py`, `api.py`, `gateway.py`, `proxy_model.py`, `AuditRegistry.sol`
