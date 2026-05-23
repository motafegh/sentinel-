# inference — Inference Pipeline

Converts a Solidity contract (file or raw string) into per-class vulnerability probabilities across 10 classes.

---

## Files

| File | Class / Purpose |
|------|----------------|
| `preprocess.py` | `ContractPreprocessor` — Slither + tokenizer → `(graph, tokens)` |
| `predictor.py` | `SentinelPredictor` — loads checkpoint, runs full pipeline |
| `api.py` | FastAPI app at `:8001` — HTTP wrapper around `SentinelPredictor` |

---

## ContractPreprocessor (`preprocess.py`)

Two public entry points — one for files on disk, one for raw strings from an API.

**Design:** instantiate once, reuse for many contracts. All expensive setup (GraphCodeBERT tokenizer load, Slither/graph extractor init) happens in `__init__`.

### Why two entry points?

Slither shells out to `solc` (the Solidity compiler), which requires a **real file path**. This is an external tool constraint that cannot be worked around.

| Caller context | Method |
|----------------|--------|
| Contract file already on disk (CLI, batch eval) | `process(sol_path)` |
| Raw string from HTTP request | `process_source(source_code)` |

### Instantiation

```python
from ml.src.inference.preprocess import ContractPreprocessor

preprocessor = ContractPreprocessor()
# Loads once: GraphCodeBERT tokenizer + graph extractor (Slither)
```

### Return value (both methods)

```python
graph, tokens = preprocessor.process("contracts/Vault.sol")
graph, tokens = preprocessor.process_source(open("Vault.sol").read(), name="Vault")

# graph: PyG Data object
#   graph.x              [N, 11]    float32 — node features (v8 schema)
#   graph.edge_index     [2, E]     int64   — directed edges
#   graph.edge_attr      [E]        int64   — edge type (1-D, values 0–10)
#   graph.contract_hash  str                — MD5 hash

# tokens: dict
#   tokens["input_ids"]      [4, 512]  long  — GraphCodeBERT token IDs, 4 windows
#   tokens["attention_mask"] [4, 512]  long  — 1=real token, 0=padding
#   tokens["contract_hash"]  str
```

Token shape is `[4, 512]` (4 windows). `SentinelPredictor` adds the batch dimension before the forward pass.

### Errors

| Exception | Cause |
|-----------|-------|
| `FileNotFoundError` | `sol_path` does not exist (`process()` only) |
| `ValueError` | Empty source, or Slither/graph extraction failed |

---

## Shape Contract

These shapes must match training data exactly:

| Tensor | Training (from dataset) | Inference (from preprocessor) |
|--------|-------------------------|-------------------------------|
| `graph.x` | `[N, 11]` | `[N, 11]` ✓ |
| `graph.edge_attr` | `[E]` 1-D int64 | `[E]` 1-D int64 ✓ |
| `tokens["input_ids"]` | `[B, 4, 512]` after collate | `[1, 4, 512]` ✓ |
| `tokens["attention_mask"]` | `[B, 4, 512]` after collate | `[1, 4, 512]` ✓ |

---

## SentinelPredictor (`predictor.py`)

Loads a checkpoint, reconstructs the model from its saved config, and runs the full pipeline.

### Instantiation

```python
from ml.src.inference.predictor import SentinelPredictor

predictor = SentinelPredictor(
    checkpoint="ml/checkpoints/graphcodebert-v1-prefix48_best.pt",
    device="cuda",
)
```

**Checkpoint loading:**
1. Reads `saved_cfg` from the checkpoint dict — reconstructs `TrainConfig` including `gnn_prefix_k`, `gnn_prefix_warmup_epochs`
2. Builds `SentinelModel` with those params
3. Strips `._orig_mod.` infix from state dict keys (torch.compile artifact)
4. Sets `model._current_epoch = 9999` — prefix always active regardless of warmup setting
5. Calls `model.eval()`

The architecture string `"three_eye_v7"` is used for `_ARCH_TO_FUSION_DIM` / `_ARCH_TO_NODE_DIM` validation — no architecture string change needed for GCB-P1 since the fusion output shape (128) and node feature dim (11) are unchanged.

### Prediction

```python
# From a file on disk
result = predictor.predict("contracts/Vault.sol")

# From a raw string
result = predictor.predict_source(source_code, name="Vault")

# result dict:
# {
#   "vulnerabilities": [
#     {"vulnerability_class": "Reentrancy",  "probability": 0.8943, "detected": true},
#     {"vulnerability_class": "IntegerUO",   "probability": 0.2101, "detected": false},
#     ...  # all 10 classes
#   ],
#   "thresholds": [0.45, 0.50, ...],   # per-class, from checkpoint
#   "num_nodes":  42,
#   "num_edges":  89,
#   "architecture": "three_eye_v7",
# }
```

**Thresholds:** per-class thresholds are stored in the checkpoint by `trainer.py` and loaded automatically. Defaults to 0.5 per class if not present (pre-tuning checkpoints).

### Backward compatibility

Checkpoints trained without GNN prefix (`gnn_prefix_k=0`) load correctly — `saved_cfg.get("gnn_prefix_k", 0)` defaults to 0, and the model is built without prefix components. No separate `_ARCH_TO_FUSION_DIM` entry needed.

---

## API (`api.py`) — FastAPI at port 8001

```bash
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache \
SENTINEL_CHECKPOINT=ml/checkpoints/graphcodebert-v1-prefix48_best.pt \
PYTHONPATH=. uvicorn ml.src.inference.api:app --host 0.0.0.0 --port 8001
```

**Endpoints:**

| Method | Path | Input | Output |
|--------|------|-------|--------|
| `POST` | `/predict` | `{"source_code": "..."}` | vulnerabilities + thresholds + metadata |
| `GET` | `/health` | — | `{"status": "ok", "model_loaded": true, ...}` |
| `GET` | `/metrics` | — | Prometheus text format |

**Environment variables:**

| Variable | Required | Notes |
|----------|----------|-------|
| `SENTINEL_CHECKPOINT` | Yes | Path to `.pt` checkpoint |
| `TRANSFORMERS_OFFLINE` | Yes | Must be `1` at shell level |
| `SENTINEL_PREDICT_TIMEOUT` | No | Default 60s |
| `SENTINEL_DRIFT_BASELINE` | No | KS drift baseline JSON |
