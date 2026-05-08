# SENTINEL — Critical Constraints

Load this file for ANY implementation task.
Violating any constraint without the corresponding rebuild/retrain produces silent failures.

---

## Locked Architecture Constants

```
GNNEncoder in_channels = 8
  — locked to 68,523 re-extracted .pt training graph files (edge_attr=[E] 1-D, P0-B format)
  — change requires: full graph dataset rebuild + retrain

CodeBERT model = "microsoft/codebert-base"
  — must match offline tokenizer used to build ml/data/tokens/

MAX_TOKEN_LENGTH = 512
  — matches training data; change requires token .pt rebuild + retrain

Node feature vector — 8-dim ordinal, fixed order (FEATURE_NAMES in graph_schema.py):
  [type_id, visibility, pure, view, payable, reentrant, complexity, loc]
  — any change: rebuild all 68,523 graph .pt files + retrain
  — bump FEATURE_SCHEMA_VERSION in graph_schema.py to invalidate inference caches

  type_id encoding (NODE_TYPES):
    STATE_VAR=0, FUNCTION=1, MODIFIER=2, EVENT=3,
    FALLBACK=4, RECEIVE=5, CONSTRUCTOR=6, CONTRACT=7

Node insertion order in graph builder (graph_extractor.extract_contract_graph):
  CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs
  — edge_index values are positional; reordering breaks edges

FEATURE_SCHEMA_VERSION = "v1"  (in graph_schema.py)
  — appended to InferenceCache keys: "{content_md5}_v1"
  — bump on ANY node/edge feature encoding change; automatically invalidates all disk-cached
    preprocessed graphs without touching the 68,523 training .pt files
  — single source of truth: ml/src/preprocessing/graph_schema.py

NUM_EDGE_TYPES = 5  (CALLS=0, READS=1, WRITES=2, EMITS=3, INHERITS=4)
  — stored in graph.edge_attr [E] int64, shape 1-D (NOT [E, 1])
  — GNNEncoder.edge_emb: nn.Embedding(5, 16) consumes this
  — changing NUM_EDGE_TYPES requires: graph dataset rebuild + retrain + ZKML rebuild
  — old .pt files without edge_attr: GNNEncoder degrades gracefully to zero-vectors

edge_attr shape: [E] 1-D int64 (NOT [E, 1])
  — all 68,523 re-extracted files (2026-05-03) have the correct [E] shape
  — old pre-refactor ast_extractor.py produced [E, 1]; unified graph_extractor.py produces [E]
  — dual_path_dataset.py: squeeze(-1) guard normalises any legacy [E,1] files transparently
  — validate_graph_dataset.py detects and reports shape mismatches — run before any retrain

Fusion architecture = CrossAttentionFusion
  — output_dim = 128 (changed from 64 when cross-attention replaced concat+MLP)
  — ZKML proxy input_dim depends on this — change requires ZKML rebuild + redeploy

CLASS_NAMES order in trainer.py — source of truth for all downstream code
  — never INSERT into the middle; only APPEND new classes at the end
  — indices 0–9 must remain stable across any future additions
```

---

## weights_only Policy — Do Not Mix

```
weights_only — TWO policies; do NOT mix:
  — DualPathDataset graph .pt files: weights_only=True
      (add_safe_globals([Data, DataEdgeAttr, DataTensorAttr]) registered at import)
      (future new PyG internal classes → add to safe_globals, do not revert to False)
  — Checkpoint .pt files (training/inference): weights_only=False
      (LoRA state dict contains peft-specific classes; weights_only=True rejects silently)
```

---

## Environment Variable Constraints

```
TRANSFORMERS_OFFLINE = 1 — must be set at shell level
  — cannot be set inside Python; if set inside Python it has no effect
```

---

## API Contract Constraints

```
Per-class thresholds companion file:
  ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json
  — must travel with the checkpoint; threshold values are sweep-derived per class
  — API response key is "thresholds" (list of floats), NOT "threshold" (single float)
    This was a breaking change (Fix #6, 2026-05-04). Update all downstream consumers.

NO "confidence" field anywhere in API responses or AuditState
  — removed in Track 3; any code using ml_result["confidence"] will KeyError silently

Request field name: "source_code" (not "contract_code")
  POST /predict {"source_code": "..."}
```

---

## Retrain Gate

```
DO NOT RETRAIN without running validate_graph_dataset.py first
  — P0-B GNNEncoder degrades gracefully to zero-vectors when edge_attr absent
  — training on zero-vectors means P0-B provides no benefit; silent quality regression
  — Retrain evaluation protocol (v4):
      baseline (v3 tuned): val F1-macro 0.5069 on ml/data/splits/val_indices.npy (fixed seed)
      success:   val F1-macro > 0.5069 on the SAME split (do NOT regenerate)
      floor:     no single class drops > 0.05 F1 from v3 tuned values
      rollback:  if tuned F1 < 0.5069 after completion: revert to v3 checkpoint
      MLflow:    experiment "sentinel-retrain-v4"; compare against "sentinel-retrain-v3"
```

---

## ZKML Constraints

```
ONNX opset version = 11
  — EZKL compatibility requirement; do not change

ZKML CIRCUIT_VERSION = "v2.0"
  — Linear(128→64→32→10) multi-label architecture
  — v1.0 was Linear(64→32→16→1) binary — all v1.0 EZKL artifacts are invalid
  — bumping requires: re-export ONNX, full EZKL pipeline rebuild, ZKMLVerifier.sol redeploy

EZKL scale factor = 8192 (2^13)
  — audit_server.py score decoding: score = field_element / 8192
  — baked into circuit settings; change requires full EZKL pipeline rebuild

Files never to commit:
  zkml/ezkl/proving_key.pk   (~10MB, gitignored)
  zkml/ezkl/srs.params       (gitignored)
```

---

## Solidity / Build Constraints

```
solc versions:
  ZKMLVerifier.sol: ≤0.8.17 (EZKL-generated assembly uses deprecated opcodes)
  All other contracts: 0.8.20
  solc-select use 0.8.17 before compiling ZKMLVerifier standalone
```

---

## RAG Index Constraints

```
RAG chunk configuration:
  chunk_size = 1536, chunk_overlap = 128
  — index was rebuilt at these values; changing requires full RAG rebuild
  — rebuild command: cd agents && poetry run python -m src.rag.build_index
```

---

## Hash Namespace Rules

```
BCCC SHA256 vs internal MD5 — never mix:
  SHA256 = hash of .sol file content → BCCC filename, CSV col 2
  MD5    = hash of .sol file path    → .pt filename in ml/data/graphs/
  bridge: graph.contract_path inside .pt → Path(...).stem = SHA256
```
