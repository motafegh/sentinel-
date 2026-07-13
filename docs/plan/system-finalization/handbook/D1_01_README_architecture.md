# D1.1 — Handbook README + Architecture

**Doc targets:** `docs/handbook/00_README.md` + `docs/handbook/01_architecture.md`
**Estimated time:** 1.5h
**Rule:** Every claim verified against source code. Every file:line reference checked.

---

## 00_README.md

### Source files to read before writing
- `MEMORY.md` lines 1-18 — project description, environment
- `CLAUDE.md` lines 1-20 — who Ali is, rules

### Sections to write

**1. How to use this handbook** (3 paragraphs)
- What the 3 layers are: TL;DR (30 sec skim), tour (1-2 pages scannable), deep reference (links only)
- When to read which layer: TL;DR for orientation, tour for modifying code, deep reference for deep understanding
- How cross-references work: `→ 04_zkml.md §2` means "see doc 04, section 2"

**2. Three learning paths** (table)

| Path | For | Docs | Time |
|---|---|---|---|
| A — Run it | Newcomer, operator | 01→06→07→10 | 30 min |
| B — ML/ZK | ML engineer | 02→03→04→07 | 45 min |
| C — Blockchain | Solidity dev | 05→07→10 | 30 min |

**3. Glossary** (~20 terms, 2-3 words each)
- Each term must be verified to exist in at least one source file:
  - `fusion_embedding` — `sentinel_model.py:592`
  - `verdict_provable` — `fuse.py`
  - `EZKL` — `setup_circuit.py`
  - `MCP` — `agents/src/mcp/`
  - `CrossAttentionFusion` — `fusion_layer.py:198`
  - `proxy` — `proxy_model.py:76`
  - `AuditRegistry` — `AuditRegistry.sol:16`
  - `Guard 3` — `AuditRegistry.sol:96,163`
  - `publicSignals` — `AuditRegistry.sol:83,164`
  - `field element` — `extract_calldata.py:60`
  - `SCALE=8192` — `extract_calldata.py:46`
  - `SENTINEL_DETERMINISTIC` — `api.py:107`
  - `SentinelDataset` — `sentinel_dataset.py:57`
  - `tool_status` — `quick_screen.py:74`
  - `Rule 5C` — `CLAUDE.md` §C
  - `Bayesian shrinkage` — `reliability_fit.py`
  - `Fbeta` — `pipeline_metrics.py`
  - `reliability_v3.yaml` — `agents/configs/reliability_v3.yaml`
  - `CIRCUIT_VERSION` — `proxy_model.py:68`
  - `provenance manifest` — `_submit.py:219`

**4. File conventions** (1 paragraph)
- `file:line` references throughout (e.g. `sentinel_model.py:592`)
- Test commands are copy-pasteable (include `cd`, `source`, `activate`)
- ASCII diagrams use box-drawing characters

### Verification checklist
- [ ] Every glossary term appears in at least one source file at the cited line
- [ ] Learning path doc numbers match actual filenames in `docs/handbook/`
- [ ] No term is defined twice across the glossary

---

## 01_architecture.md

### Source files to read before writing (9 files)
1. `agents/src/orchestration/graph.py` — LangGraph definition: node names, edges, conditional routing
2. `agents/src/orchestration/state.py` — AuditState TypedDict: all fields with comments
3. `ml/src/inference/api.py:74-80` — CHECKPOINT path resolution
4. `ml/src/inference/api.py:380-510` — all 4 API endpoints (health, predict, hotspots, fusion-embedding)
5. `agents/src/api/gateway.py:212-322` — all gateway endpoints
6. `zkml/src/distillation/proxy_model.py:68,104-108` — CIRCUIT_VERSION, frozen dims
7. `contracts/src/AuditRegistry.sol:28-34,42-56` — V1 struct, V2 struct
8. `contracts/foundry.toml` — solc version, libs
9. `agents/.env` — all env vars and port numbers

### Sections to write

**1. TL;DR** (4 lines)
```
What: Decentralised AI security oracle — Solidity contracts analyzed by
      GNN+GraphCodeBERT, results ZK-proven and stored on-chain
Modules: DATA → ML → ZKML → CONTRACTS ← AGENTS
Ports: 8001 (ML API), 8010 (MCP inference), 8012 (MCP audit), 8545 (Anvil)
Tests: see 10_operations.md
```

**2. System diagram** (ASCII art, ~1 page)
- Data flow: `.sol source → graph_extractor → GNN+CodeBERT → fusion(128) → proxy(10) → EZKL proof → AuditRegistry`
- Service flow: `AGENTS gateway → MCP servers → ML API → Anvil`
- Verify: every arrow in the diagram corresponds to a real function call, HTTP request, or subprocess in source

**3. Module map** (table, 5 rows)

| Module | Language | Code dir | Test command | Tests |
|---|---|---|---|---|
| DATA | Python | `data_module/` | `cd data_module && .venv/bin/python -m pytest` | 569 |
| ML | Python/PyTorch | `ml/src/` | `source ml/.venv/bin/activate && python -m pytest ml/tests/` | 214 |
| ZKML | Python/EZKL | `zkml/src/` | `source ml/.venv/bin/activate && python -m pytest zkml/tests/` | 37 |
| CONTRACTS | Solidity | `contracts/src/` | `cd contracts && forge test` | 66 |
| AGENTS | Python/LangGraph | `agents/src/` | `cd agents && poetry run pytest` | 634 |

- Verify test counts by running `pytest --collect-only` for each module

**4. Port map** (table)
- Verify each port against `.env` and source:
  - 8001 — `ml/src/inference/api.py` (uvicorn)
  - 8010 — `agents/src/mcp/servers/inference_server.py` (MCP_AUDIT_PORT or custom)
  - 8012 — `agents/src/mcp/servers/audit/_server.py` (MCP_AUDIT_PORT)
  - 8014 — `agents/src/mcp/servers/representation_server.py` (MCP_REPRESENTATION_PORT)
  - 8545 — Anvil (default port)

**5. The dual-verdict design** (~1 page)
- `verdict_provable` — deterministic only (ML + static + fuse math), ZK-provable
- `verdict_full` — includes LLM debate + RAG, richer but non-deterministic
- Verify: `agents/src/orchestration/verdict/fuse.py` — both verdict keys exist
- Verify: `agents/src/orchestration/nodes/synthesizer.py:344-377` — final_report structure includes both
- Why this matters: only `verdict_provable` can be ZK-proved (LLM is non-deterministic even at temp=0)

**6. Key configuration locations** (table)
- `agents/.env` — all env vars (ports, RPC, operator key, LM Studio URL)
- `agents/configs/verdicts_default.yaml` — L1 decision numbers (thresholds, weights, bands)
- `agents/configs/reliability_v3.yaml` — L3 fitted reliability weights (Bayesian shrinkage)
- `contracts/foundry.toml` — Solidity compiler version (0.8.22), optimizer, remappings
- `zkml/src/distillation/proxy_model.py` — frozen architecture constants (128/64/32/10)
- `ml/src/inference/api.py:74-80` — checkpoint path (3-level precedence: env > config > default)

### Verification checklist
- [ ] Every port number matches `.env` or source
- [ ] Every module test count matches `pytest --collect-only` output
- [ ] Every arrow in the system diagram = a real import, HTTP call, or subprocess
- [ ] Dual-verdict keys exist in `fuse.py` source
- [ ] Config file paths resolve on disk
