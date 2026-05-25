# CLAUDE.md — Sentinel Project Context

This file is read automatically by Claude Code at the start of every session.
It gives Claude the project context needed to work correctly without re-explanation.

---

## What This Project Is

**SENTINEL** is a decentralised AI security oracle for Solidity smart contracts.
It combines a dual-path ML model (GNN + GraphCodeBERT) with zero-knowledge proof generation
and on-chain audit registration so results can be independently verified.

**Three-eye classifier (SentinelModel v8):**
- GNN eye: 3-phase 7-layer GAT (JK-attention) on the contract's AST/CFG graph
- Transformer eye: GraphCodeBERT (frozen, 124M params) + LoRA r=16 + GNN prefix injection
- Fused eye: CrossAttentionFusion (bidirectional node↔token cross-attention)
- Output: 10-class multi-label vulnerability prediction

**10 output classes (index order LOCKED — reordering breaks all checkpoints and ZKML circuit):**
`CallToUnknown(0)`, `DenialOfService(1)`, `ExternalBug(2)`, `GasException(3)`,
`IntegerUO(4)`, `MishandledException(5)`, `Reentrancy(6)`, `Timestamp(7)`,
`TransactionOrderDependence(8)`, `UnusedReturn(9)`

---

## ⚠️ Critical Locked Constants

Violating any of these without the corresponding rebuild/retrain produces **silent failures** — no crash, wrong predictions.

| Constant | Locked Value | What breaks if changed without full rebuild |
|----------|-------------|---------------------------------------------|
| `NODE_FEATURE_DIM` | **11** | All 41,576 graph `.pt` files invalid + model retrain required |
| `FEATURE_SCHEMA_VERSION` | **`"v8"`** | Bump on any schema change; invalidates inference cache |
| `NUM_EDGE_TYPES` | **11** | `Embedding(11,64)` in GNNEncoder; retrain required |
| `NUM_CLASSES` | **10** | CLASS_NAMES order locked; ZKML circuit breaks |
| `CrossAttentionFusion output_dim` | **128** | ZKML proxy MLP + on-chain ZKMLVerifier redeploy |
| `ZKML proxy input dim` | **128** | Must match fusion `output_dim` |
| Backbone model | `microsoft/graphcodebert-base` | Token files rebuild + retrain |
| `type_id` normalization divisor | **12.0** | All node features shift → silent accuracy regression |
| `TRANSFORMERS_OFFLINE` | Set at **shell level** before import | Read at `transformers` import time, too late inside Python |
| `add_self_loops` in Phase 2 | **False** | Self-loops cancel directional control-flow signal |
| `weights_only` for graph `.pt` | **False** | PyG 2.7 metadata not safe-tensors serialisable |
| `weights_only` for checkpoint `.pt` | **False** | LoRA PEFT objects not safe-tensors serialisable |

Full constraints reference: `docs/Project-Spec/SENTINEL-CONSTRAINTS.md`

---

## Repository Layout (key paths)

```
sentinel-/
├── CLAUDE.md                        ← this file
├── learning_with_claude/            ← ACTIVE LEARNING JOURNEY (read these every learning session)
│   ├── reference.md                 ← how the learning system works + current status
│   ├── preferences.md               ← all teaching preferences (P1–P10); MUST COMPLY
│   ├── audit_flags.md               ← all audit issues found (A1–A4)
│   └── session_log.md               ← chronological session record
│
├── ml/
│   ├── src/
│   │   ├── preprocessing/
│   │   │   ├── graph_schema.py      ← SINGLE SOURCE OF TRUTH: node types, edge types, feature dims
│   │   │   └── graph_extractor.py   ← canonical Solidity → PyG graph converter
│   │   ├── models/
│   │   │   ├── sentinel_model.py    ← assembles the three-eye classifier
│   │   │   ├── gnn_encoder.py       ← 3-phase GAT + JK-attention
│   │   │   ├── transformer_encoder.py ← GraphCodeBERT + LoRA + GNN prefix
│   │   │   └── fusion_layer.py      ← CrossAttentionFusion
│   │   ├── training/
│   │   │   ├── trainer.py           ← main training loop (1,633 lines)
│   │   │   └── losses.py            ← AsymmetricLoss, FocalLoss
│   │   ├── datasets/
│   │   │   └── dual_path_dataset.py ← serves (graph, tokens) pairs to training
│   │   ├── inference/
│   │   │   ├── api.py               ← FastAPI endpoint
│   │   │   ├── predictor.py         ← model loading + threshold application
│   │   │   ├── preprocess.py        ← inference-time graph + token extraction
│   │   │   ├── cache.py             ← inference feature cache
│   │   │   └── drift_detector.py    ← KS-test monitoring
│   │   └── utils/
│   │       └── hash_utils.py        ← MD5 contract identification + file pairing
│   ├── scripts/                     ← train.py, tune_threshold.py, manual_test.py, ...
│   ├── tests/                       ← pytest: preprocessing, model, training, dataset, api
│   └── data/
│       ├── graphs/                  ← 41,576 PyG graph .pt files (v8 schema, 11-dim)
│       ├── tokens_windowed/         ← 44,470 CodeBERT token .pt files ([4,512])
│       └── splits/deduped/          ← train/val/test split indices (.npy)
│
├── zkml/                            ← ZK-ML proof generation (EZKL + Groth16)
├── agents/                          ← LangGraph orchestration + MCP servers + RAG
├── contracts/                       ← Solidity (Foundry): AuditRegistry, SentinelToken
└── docs/                            ← full architecture docs, changelogs, proposals
```

---

## Development Conventions

- **Active branch:** `claude/busy-babbage-5R7q3` — all development goes here
- **Python version:** 3.12.1 strict for `ml/`; ≥ 3.11 for `agents/`
- **Dependency management:** Poetry (per-module `pyproject.toml`)
- **Test runner:** `TRANSFORMERS_OFFLINE=1 PYTHONPATH=. pytest ml/tests/ -v`
- **Manual smoke test:** `python ml/scripts/manual_test.py --checkpoint <path>` (19/20 expected detections)
- **Graph validation before retrain:** `python ml/scripts/validate_graph_dataset.py`
- **Schema change protocol:** bump `FEATURE_SCHEMA_VERSION` → re-extract all graphs → retokenize → retrain

---

## Key Invariants (never silently violate)

1. `graph_schema.py` is the single source of truth for NODE_TYPES, EDGE_TYPES, NODE_FEATURE_DIM.
   Both training and inference import from it. Never duplicate these constants elsewhere.

2. `extract_contract_graph()` in `graph_extractor.py` is the single canonical Solidity→graph converter.
   Both the batch pipeline (`ast_extractor.py`) and inference (`preprocess.py`) call it.

3. `graph.edge_attr` must be shape `[E]` (1-D), not `[E, 1]`. `nn.Embedding` requires 1-D indices.

4. `REVERSE_CONTAINS` (edge type 7) is RUNTIME-ONLY — generated inside GNNEncoder Phase 3 by
   reversing CONTAINS(5) edges. It is NEVER written to `.pt` files on disk.

5. `node_metadata` list must remain index-aligned with `graph.x` at all times.
   Violation causes wrong node metadata lookups with no error signal.

6. The 10 output class indices are LOCKED. Do not reorder CLASS_NAMES in `trainer.py`.

---

## Learning Journey

An active deep-dive study of the ML module is in progress.

**Every time a learning session starts, Claude MUST:**
1. Read `learning_with_claude/reference.md` — get current status, rules, and update protocol
2. Read `learning_with_claude/preferences.md` — apply ALL preferences (P1–P14) to every response
3. Read `learning_with_claude/audit_flags.md` — know what issues are already flagged
4. Read `learning_with_claude/session_log.md` — know what has been taught and what gaps were closed

**During learning sessions:**
- Raise `[AUDIT]` flags inline and immediately add to `audit_flags.md`
- Update `session_log.md` after each chunk is delivered
- Follow the **Spec File Update Protocol** in `reference.md` — defines exactly when/how/what to update
- Follow P10 (spaced repetition): warm-up recall at chunk start, lock-in summary at chunk end

Current status: Phase 2 — `graph_extractor.py` Chunks 1–3 complete, ready for Chunk 4.
