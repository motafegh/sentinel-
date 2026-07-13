# SENTINEL Developer Handbook

> **What this is:** A practical reference for understanding, running, and modifying the SENTINEL system. Every claim is verified against source code. Every test command is copy-pasteable.

---

## How to use this handbook

Each doc has **three layers**:

1. **TL;DR** — 30-second skim. What this module does, where it runs, how to test it. Read this first.
2. **The tour** — 1-2 pages. Key files, data flow, configuration. Enough to modify code without reading source.
3. **Deep reference** — Links to `docs/learning/` deep dives and source files. Read only when you need to change something specific.

Cross-references look like this: `→ 04_zkml.md §2` means "see doc 04, section 2".

---

## Three learning paths

| Path | For who | Docs in order | Time |
|---|---|---|---|
| **A — Run it** | Newcomer, operator | `01` → `06` → `07` → `10` | 30 min |
| **B — ML/ZK** | ML engineer | `02` → `03` → `04` → `07` | 45 min |
| **C — Blockchain** | Solidity dev | `05` → `07` → `10` | 30 min |

---

## Glossary

| Term | Meaning | Where in source |
|---|---|---|
| **fusion_embedding** | 128-dim CrossAttentionFusion output — the ZK circuit's input | `sentinel_model.py:592` |
| **verdict_provable** | Deterministic-only verdict (ML + static + fuse math), ZK-provable | `fuse.py:163` |
| **verdict_full** | Full verdict including LLM debate + RAG, richer but non-deterministic | `fuse.py:166` |
| **EZKL** | Zero-knowledge ML toolkit — compiles ONNX to Halo2 circuit | `setup_circuit.py` |
| **MCP** | Model Context Protocol — how agents call external tools | `agents/src/mcp/` |
| **CrossAttentionFusion** | Bidirectional cross-attention merging GNN + CodeBERT embeddings → 128-dim | `fusion_layer.py:198` |
| **proxy** | Tiny model (128→64→32→10) that mimics the teacher, ZK-provable | `proxy_model.py:76` |
| **AuditRegistry** | On-chain UUPS upgradeable contract storing audit results | `AuditRegistry.sol:16` |
| **Guard 3** | On-chain check: publicSignals match submitted scores | `AuditRegistry.sol:96,163` |
| **publicSignals** | 138 public values in the ZK proof (128 features + 10 scores) | `AuditRegistry.sol:83,164` |
| **field element** | BN254 integer encoding — little-endian, scaled by 8192 | `extract_calldata.py:60` |
| **SCALE** | 8192 (2^13) — fixed-point multiplier for float→integer conversion | `extract_calldata.py:46` |
| **SENTINEL_DETERMINISTIC** | Env var enabling deterministic mode (disables LLM + RAG) | `api.py:109` |
| **SentinelDataset** | PyTorch dataset loading the v2 export (sharded .pt files) | `sentinel_dataset.py:57` |
| **tool_status** | Per-tool {ran, reason, detail} dict — Rule 5C: no silent failures | `quick_screen.py:69` |
| **Rule 5C** | Project rule: every tool failure must surface structured status | `CLAUDE.md` §C |
| **Bayesian shrinkage** | Per-tool reliability fitting with prior α=5 (small sample safety) | `reliability_fit.py` |
| **Fbeta** | F-measure with β=2 (recall weighted 4× over precision) | `pipeline_metrics.py` |
| **CIRCUIT_VERSION** | "v2.0" — tracks proxy architecture (bump = new EZKL keys) | `proxy_model.py:68` |
| **provenance manifest** | EIP-191 signed JSON binding teacher hash to fusion embedding | `_submit.py:219` |

---

## File conventions

- `file:line` references throughout (e.g. `sentinel_model.py:592` means line 592 of that file)
- Test commands are copy-pasteable — include `cd`, `source`, `activate` where needed
- ASCII diagrams use box-drawing characters, render correctly in any monospace terminal
- Paths are relative to the repo root (`~/projects/sentinel/`) unless stated otherwise

---

## Doc index

| # | Doc | What it covers |
|---|---|---|
| 01 | `01_architecture.md` | System overview, diagram, module map, ports, dual-verdict design |
| 02 | `02_data_module.md` | Solidity → graphs + tokens, v2 export, schema, splits |
| 03 | `03_ml_module.md` | GNN+CodeBERT model, inference API, 3-tier output, deterministic mode |
| 04 | `04_zkml_module.md` | Proxy distillation, EZKL pipeline, proof generation, calldata |
| 05 | `05_contracts_module.md` | AuditRegistry V1/V2, SentinelToken, ZKMLVerifier, deployment |
| 06 | `06_agents_module.md` | 14-node pipeline, evidence/fuse, MCP servers, gateway |
| 07 | `07_cross_module.md` | ML→ZKML→Contracts→Chain integration, 3 boundaries |
| 08 | `08_security.md` | 3-layer injection defense, routing isolation, provenance, Rule 5C |
| 09 | `09_evaluation.md` | Fbeta, 9 gates, Bayesian shrinkage, L0→L3 maturity ladder |
| 10 | `10_operations.md` | Start/stop services, Anvil testing, Sepolia deployment, debugging |
