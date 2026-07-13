# D1 — SENTINEL Developer Handbook

**Date:** 2026-07-13
**Phase:** D1 (Documentation — developer handbook covering all 5 modules)
**Pre-conditions:** P11 complete, ZKML+contracts+AGENTS integration working, 1,520 tests pass.
**Purpose:** A single entry point for any developer (including Ali after a month away) to understand, run, and modify the SENTINEL system — without reading source code or stale docs.

---

## 1. Why this exists

### Current documentation problem

| What | Where | Problem |
|---|---|---|
| Module architecture | 5 state docs (`*_STATE_AND_REDESIGN_2026-06-14.md`) | 25 days stale, written as redesign proposals not as reference |
| Deep technical detail | `docs/learning/01-10` (10 files, ~200 pages total) | Excellent depth but no quick entry point. Too much to read before doing anything. |
| Project state | `MEMORY.md` (118 lines) | Good snapshot, not a learning resource |
| Source code | 60+ `.py` files across 5 modules | Truth, but 15,000+ lines |
| Integration points | Nowhere documented | ML→ZKML boundary, ZKML→Contracts boundary, AGENTS→on-chain path |

### What the handbook fixes

A developer arriving fresh should be able to answer these in < 5 minutes each:
- "What does SENTINEL do and how do I run it?" → `01_architecture.md`
- "How do I start all the servers?" → `10_operations.md`
- "Where is the ZK circuit boundary?" → `04_zkml.md` + `07_cross_module.md`
- "How do I run the tests?" → `10_operations.md`
- "What ports are which services on?" → `01_architecture.md` + `10_operations.md`
- "How does an audit flow from source code to on-chain?" → `07_cross_module.md`

---

## 2. Handbook structure

```
docs/handbook/
  00_README.md              Navigation + 3 learning paths
  01_architecture.md         System overview, diagrams, module map, ports
  02_data_module.md          Contracts → graphs → tokens, v2 export, splits
  03_ml_module.md            Model architecture, checkpoint, inference API
  04_zkml_module.md          Proxy distillation, EZKL circuit, proof generation
  05_contracts_module.md     AuditRegistry V1/V2, SentinelToken, ZKMLVerifier
  06_agents_module.md        LangGraph pipeline, verdicts, MCP servers, gateway
  07_cross_module.md         Integration: ML→ZKML→Contracts→AGENTS
  08_security.md             Injection guards, routing isolation, provenance
  09_evaluation.md           Fbeta, reliability matrix, gates, decision numbers
  10_operations.md           Start/stop all services, Anvil, Sepolia, debugging
```

### Each doc's fixed structure (3 layers)

**Layer 1 — TL;DR (top of every doc, ~4 lines)**
```
What: One-sentence description
Runs on: ports, processes
Tests: exact command that proves it works
Key files: 3-5 paths
```

**Layer 2 — The tour (~1-2 pages, scannable)**
- Data flow diagram (ASCII art)
- Key functions and where they live (file:line)
- Configuration (which .env vars, which config files)
- How to modify something common (e.g. "add a new vulnerability class")

**Layer 3 — Deep reference (links, no duplication)**
- Links to `docs/learning/` for deep dives
- Links to relevant source files with line numbers
- Links to relevant ADRs or state docs

### Three learning paths

| Path | For | Docs | Time |
|---|---|---|---|
| **A — Run it** | Newcomer, operator | 01 → 06 → 07 → 10 | 30 min |
| **B — ML/ZK** | ML engineer | 02 → 03 → 04 → 07 | 45 min |
| **C — Blockchain** | Solidity dev | 05 → 07 → 10 | 30 min |

---

## 3. Doc-by-doc content plan

### 00_README.md
- How to use this handbook
- The 3 learning paths with estimated time
- Quick glossary: SENTINEL terms (fusion_embedding, proxy, EZKL, MCP, verdict_provable, etc.)
- File conventions used throughout

### 01_architecture.md
- Full system diagram (ASCII art: data_module → ML → AGENTS + ZKML → Contracts)
- Module map: what each module does, what language/framework, where its code lives
- Port map: 8001 (ML), 8010 (MCP inference), 8012 (MCP audit), 8545 (Anvil)
- Data flow: Solidity source → Graph extractor → GNN+CodeBERT → fusion(128) → proxy(10) → EZKL proof → AuditRegistry
- The dual-verdict design: `verdict_provable` vs `verdict_full`

### 02_data_module.md
- What: Converts Solidity contracts into PyTorch Geometric graphs + tokenized windows
- The v2 export: 22,493 contracts, graph shards, token shards, splits (train/val/test)
- Key files: `sentinel_data/representation/graph_schema.py`, exports directory
- How to: run a new export, understand the class label structure
- Test command: `cd data_module && .venv/bin/python -m pytest`

### 03_ml_module.md
- What: Dual-path GNN+GraphCodeBERT with CrossAttentionFusion, 10-class classifier
- Architecture: GNNEncoder (8-layer GAT) → TransformerEncoder (CodeBERT+LoRA) → CrossAttentionFusion(128) → 4-eye classifier(10)
- Run 12 checkpoint: `GCB-P1-Run12-v3dospatched-20260613_FINAL.pt` (269 MB)
- API endpoints: `/health`, `/predict`, `/hotspots`, `/fusion-embedding`
- How to: start the server, query it, understand the three-tier output
- Test command: `source ml/.venv/bin/activate && TRANSFORMERS_OFFLINE=1 SENTINEL_CHECKPOINT=... python -m pytest ml/tests/`

### 04_zkml_module.md
- What: Tiny proxy model (128→64→32→10, 10,666 params) proves via EZKL ZK circuit
- The ZK boundary: CrossAttentionFusion output [128] → proxy → class scores [10]
- Pipeline: `corpus_distill.py` → `export_onnx.py` → `generate_calibration.py` → `setup_circuit.py` → `run_proof.py`
- Key design decisions: why proxy (not full model), why architecture is frozen, circuit versioning
- How to: regenerate everything after retraining
- Test command: `source ml/.venv/bin/activate && python -m pytest zkml/tests/`

### 05_contracts_module.md
- What: On-chain storage of audit results via UUPS-upgradeable AuditRegistry
- V1 (single score) vs V2 (10-class): structs, events, guard differences
- SentinelToken: ERC20 staking for agent eligibility
- ZKMLVerifier: Auto-generated Halo2 verifier from EZKL
- How to: deploy to Anvil, deploy to Sepolia, verify
- Test command: `cd contracts && forge test`

### 06_agents_module.md
- What: LangGraph orchestration of 14 audit nodes + MCP servers + FastAPI gateway
- Pipeline nodes: quick_screen → static_analysis → ml_assessment → rag_research → cross_validator → synthesizer → etc.
- Evidence model: `Evidence` dataclass, 5 kinds, `fuse()` verdict producer
- MCP servers: inference (8010), audit/registry (8012), graph, rag, representation
- Gateway: POST `/audit`, GET `/audit/{job_id}`, SQLite JobStore
- How to: run an audit, interpret the final_report, understand verdicts
- Test command: `cd agents && poetry run pytest`

### 07_cross_module.md
- What: The integration glue — how ML output becomes a ZK proof and gets on-chain
- The full flow: Solidity source → `/fusion-embedding` → proxy → EZKL → `submitAuditV2`
- Key integration points: fusion_embedding in SentinelModel.aux_dict, MCP submit_audit tool, provenance manifest
- Decision numbers that cross modules: INPUT_OFFSET=128, NUM_CLASSES=10, SCALE=8192, EZKL_PARAM_LIMIT=12000
- How to: run the full pipeline end-to-end, debug a mismatch
- Common failure modes: score mismatch (EZKL vs PyTorch sigmoid), publicSignals layout, teacher/proxy disagreement

### 08_security.md
- What: 3-layer prompt injection defense, routing isolation, provenance manifest
- Defense layers: comment_strip → prompt_delimit → injection_detect (8 patterns)
- Routing isolation: evidence_router + routing.py — no LLM access to contract_code
- Provenance: EIP-191 signed manifest binding teacher model hash to fusion embedding
- Rule 5C: no silent failures — every tool_status carries ran/reason/detail
- How to: add a new injection pattern, test adversarial contracts

### 09_evaluation.md
- What: Fbeta(β=2), confusion matrix, gate assertions, Bayesian shrinkage (α=5)
- L0→L3 decision number maturity ladder
- Reliability matrix: per-tool TP/FP/FN/TN, drop-gate, fitted weights
- Gate system: 9 assertions in `gates.py` that block a release
- How to: run a benchmark, interpret results, calibrate a new tool

### 10_operations.md
- What: How to start, stop, and debug every service in the system
- Quickstart: `source ml/.venv/bin/activate && uvicorn ...` + Anvil + MCP servers
- Environment check: verify all prerequisites (solc, ezkl, forge, torch, GPU)
- Local testing (Anvil): deploy contracts, stake operator, run audit, verify on-chain
- Sepolia deployment: what needs to be configured, estimated gas costs
- Debugging: common failures and their symptoms
- Log file locations: `/tmp/ml_inference.log`, `/tmp/anvil.log`

---

## 4. Tasks

### D1.1 — Write 00_README.md + 01_architecture.md (~1.5h)
The two entry-point docs. Everything else references them.

### D1.2 — Write module docs 02-06 (~4h total, ~45min each)
One doc per module. Follow the 3-layer structure. Source-code verification for every claim (Rule 4).

### D1.3 — Write 07_cross_module.md (~1h)
The most important doc for understanding the system as a whole. No other doc covers this.

### D1.4 — Write 08_security.md + 09_evaluation.md (~1h)
Shorter docs — most content already exists in `docs/learning/`, just needs summarization.

### D1.5 — Write 10_operations.md (~1h)
Must be tested live — every command in this doc must work when copy-pasted.

### D1.6 — Final review pass (~1h)
- Verify all file:line references are still accurate
- Verify all test commands produce 0 failures
- Verify all paths resolve on a fresh checkout
- Add cross-references between docs

---

## 5. Acceptance gates

| Gate | What |
|---|---|
| G1 | Every doc has a TL;DR that a newcomer can understand in 30 seconds |
| G2 | Every test command in the handbook passes when copy-pasted |
| G3 | Every file:line reference points to actual code |
| G4 | The 3 learning paths cover all content with no gaps |
| G5 | A developer can start the full system using only `10_operations.md` |
| G6 | `07_cross_module.md` explains the full ML→ZK→Chain flow in one read |

---

## 6. What this deliberately does NOT do

- **Does not duplicate** `docs/learning/` — links to it for deep dives
- **Does not replace** `MEMORY.md` — that's project state, this is developer education
- **Does not explain** every function signature — that's what source code + docstrings are for
- **Does not cover** data_module in exhaustive detail — it's a supporting module, not a product module
- **Does not cover** the pre-June-2026 history — `memory/history_2026-06.md` handles that

---

## 7. Effort estimate

| Task | Est. |
|---|---|
| D1.1 — README + Architecture | 1.5h |
| D1.2 — Module docs (5 docs) | 4h |
| D1.3 — Cross-module | 1h |
| D1.4 — Security + Evaluation | 1h |
| D1.5 — Operations | 1h |
| D1.6 — Review pass | 1h |
| **Total** | **~9.5h (~1.5 days)** |

---

## 8. Ordering

The module docs (02-06) can be written in any order — they're independent. The cross-module doc (07) should be written last since it references all of them. The operations doc (10) should also be written last since every command must be verified live.
