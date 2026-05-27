# SENTINEL ML Inference — Technical Reference

## Overview

The inference pipeline converts a raw Solidity contract into a vulnerability score and label. Three modules compose the pipeline:

```
ContractPreprocessor   (preprocess.py)  —  .sol → (graph, tokens)
Predictor              (predictor.py)   —  (graph, tokens) → score + label
FastAPI App            (api.py)         —  HTTP POST → JSON
```

The pipeline is split this way deliberately: preprocessing is the slow step (Slither + solc invocation), and the model forward pass is fast. Both `ContractPreprocessor` and `Predictor` are designed to be instantiated once and reused for many contracts.

---

## ContractPreprocessor (`ml/src/inference/preprocess.py`)

### Responsibility

Converts a Solidity contract into the exact two inputs `SentinelModel.forward()` expects:

| Output | Shape | Description |
|---|---|---|
| `graph` | PyG `Data(x=[N,8], edge_index=[2,E])` | AST/CFG structure → GNNEncoder |
| `tokens["input_ids"]` | `[1, 512]` long | CodeBERT token IDs → TransformerEncoder |
| `tokens["attention_mask"]` | `[1, 512]` long | 1=real token, 0=padding |
| `tokens["truncated"]` | bool | True if contract exceeded 512 tokens |
| `tokens["num_tokens"]` | int | Real (non-padding) token count |
| `tokens["contract_hash"]` | str | MD5 for caching |

### Two entry points

```python
preprocessor = ContractPreprocessor()   # loads tokenizer once — ~1s

graph, tokens = preprocessor.process("contracts/Vault.sol")
# OR
graph, tokens = preprocessor.process_source("pragma solidity ^0.8.0; contract Vault { ... }")
```

**`process(sol_path)`** — for contracts already on disk.

**`process_source(source_code)`** — for raw strings arriving via HTTP, stdin, or pipe. Slither shells out to `solc` which requires a real file path, so this method writes to a `NamedTemporaryFile`, runs extraction, then deletes the temp file in a `finally` block. The tokeniser receives the string directly — no redundant file read.

### Hashing convention

| Entry point | Hash input | Reason |
|---|---|---|
| `process()` | MD5 of resolved absolute path | Matches offline pipeline (`ast_extractor_v4_production.py`) |
| `process_source()` | MD5 of source content | Content-addressable — enables API-layer response caching |

### Graph extraction (`_extract_graph`)

**Critical:** This method replicates `node_features()` from `ast_extractor_v4_production.py` exactly. That script built all 68,555 training `.pt` files. Any divergence causes shape or distribution mismatch at inference time.

**Why not `graph_builder.GraphBuilder`?**

`graph_builder.py` produces 17-dim one-hot node features. The training data and `GNNEncoder(in_channels=8)` expect 8-dim raw float vectors. Using `graph_builder.py` would immediately raise:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (N×17 and 8×64)
```
`graph_builder.py` is kept for potential future retraining with richer features but **must not be used for inference with the current checkpoint**.

**Node insertion order matters.** Edge indices reference node positions. The insertion order must match the production script:
1. CONTRACT node (index 0)
2. State variables
3. Functions (with special-kind override: CONSTRUCTOR, FALLBACK, RECEIVE)
4. Modifiers
5. Events

**Node feature vector (8-dim, float32):**

| Index | Feature | Encoding |
|---|---|---|
| 0 | `type_id` | 0=STATE_VAR, 1=FUNCTION, 2=MODIFIER, 3=EVENT, 4=FALLBACK, 5=RECEIVE, 6=CONSTRUCTOR, 7=CONTRACT |
| 1 | `visibility` | 0=public/external, 1=internal, 2=private (ordinal, not one-hot) |
| 2 | `pure` | 1.0 if pure function, else 0.0 |
| 3 | `view` | 1.0 if view function, else 0.0 |
| 4 | `payable` | 1.0 if payable, else 0.0 |
| 5 | `reentrant` | 1.0 if Slither's `is_reentrant` flag, else 0.0 |
| 6 | `complexity` | `float(len(func.nodes))` — CFG node count |
| 7 | `loc` | `float(len(source_mapping.lines))` — lines of code |

**Edge types:**

| Type | Meaning |
|---|---|
| CALLS | Function A calls function B |
| READS | Function reads a state variable |
| WRITES | Function writes a state variable |
| EMITS | Function emits an event |
| INHERITS | Contract inherits from parent |

If a contract produces zero edges, `edge_index` is `torch.zeros((2, 0), dtype=torch.long)` — a valid empty graph, not an error.

**Slither options:**
- `detectors_to_run=[]` — disables all security detectors (we only need the AST, not Slither's own findings)
- `is_from_dependency()` — skips imported libraries (OpenZeppelin etc.) — analyses user code only
- `contracts[0]` — takes the first non-dependency contract

### Tokenisation (`_tokenize`)

Settings match `tokenizer_v1_production.py` exactly:
```python
tokenizer(source_code, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
```

**Truncation detection:** A second `encode()` call (without truncation) gets the true token count. Comparing decoded text is unreliable — the tokeniser normalises whitespace on the round-trip.

**Shape assertions:** After tokenisation, both `input_ids` and `attention_mask` are asserted to be `[1, 512]`. This fires immediately instead of producing a cryptic shape error deep in the model forward pass.

---

## Predictor (`ml/src/inference/predictor.py`)

### Responsibility

Loads a trained `SentinelModel` checkpoint and produces vulnerability scores for individual contracts.

```python
predictor = Predictor(
    checkpoint="ml/checkpoints/run-alpha-tune_best.pt",
    threshold=0.50,
)

result = predictor.predict("contracts/Vault.sol")
# {
#   "score":      0.823,        # sigmoid probability
#   "label":      "vulnerable", # >= threshold
#   "threshold":  0.50,
#   "truncated":  False,
#   "num_nodes":  12,
#   "num_edges":  9
# }
```

### Instantiation cost

`__init__` loads:
- `SentinelModel()` — architecture initialised with random weights
- `torch.load(checkpoint)` → `model.load_state_dict(state_dict)` — replaces with trained weights (~477 MB)
- `ContractPreprocessor()` — loads CodeBERT tokeniser from disk

**Total: ~4–10 seconds depending on storage speed.** Instantiate once; reuse for many contracts.

### Checkpoint format handling

```python
raw = torch.load(checkpoint, map_location=device, weights_only=True)
state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
```

| Format | Detection | Handling |
|---|---|---|
| New (April 2026+) | `isinstance(raw, dict) and "model" in raw` | Extract `raw["model"]`, log metadata (epoch, best_f1, run_name) |
| Old (pre-April 2026) | Plain `OrderedDict` | Use directly as state_dict |

Both formats are loaded with `weights_only=True`. The new format is a dict of tensors + basic Python types, so `weights_only=True` is safe for both.

### `eval()` mode

```python
self.model.eval()
```

**This is not optional.** `eval()` disables:
- `GNNEncoder` Dropout (p=0.2)
- `FusionLayer` Dropout (p=0.3)

Without `eval()`, every call to `predict()` returns a slightly different score for the same contract. The model stays in `eval()` permanently — there is no reason to call `train()` on a loaded inference predictor.

### Forward pass (`_score`)

```python
with torch.no_grad():
    batch = Batch.from_data_list([graph])   # single graph → Batch (required by global_mean_pool)
    batch = batch.to(device)
    input_ids      = tokens["input_ids"].to(device)       # [1, 512]
    attention_mask = tokens["attention_mask"].to(device)  # [1, 512]

    scores = self.model(batch, input_ids, attention_mask)  # [1] sigmoid-activated
    score: float = scores.item()

label = "vulnerable" if score >= self.threshold else "safe"
```

**Why `Batch.from_data_list([graph])` even for a single contract?**

`global_mean_pool(x, batch)` needs the `batch` vector to know which nodes belong to which graph. For B=1 this vector is all zeros, but it must be present. `Batch.from_data_list()` creates it automatically.

**`torch.no_grad()`** skips building the autograd computation graph — saves GPU memory and speeds up inference by ~15%.

---

## FastAPI App (`ml/src/inference/api.py`)

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check — returns `{"status": "ok", "predictor_loaded": true, ...}` |
| `/predict` | POST | Score a Solidity contract |

### Lifespan pattern

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor(checkpoint=CHECKPOINT)  # runs ONCE at startup
    yield                                                     # server live
    # shutdown cleanup here (none needed for PyTorch)
```

The `Predictor` is loaded once at startup and stored on `app.state.predictor`. Each request accesses it via `request.app.state.predictor` — microsecond lookup, not seconds. Loading at startup (not module import) means checkpoint errors surface immediately with a clear message instead of a cryptic `ImportError`.

**Checkpoint selection:** `SENTINEL_CHECKPOINT` environment variable overrides the default. This allows switching checkpoints in production without touching source code:
```bash
export SENTINEL_CHECKPOINT=ml/checkpoints/run-more-epochs_best.pt
uvicorn ml.src.inference.api:app
```

### Request/response schema

**`PredictRequest`:**
```json
{ "source_code": "pragma solidity ^0.8.0; contract Vault { ... }" }
```

- `min_length=10` — rejects obviously empty payloads before hitting the model
- `must_look_like_solidity` validator — checks for `pragma` or `contract` keyword; returns HTTP 422 with a helpful message if missing

**`PredictResponse`:**
```json
{
  "label":       "vulnerable",
  "confidence":  0.823,
  "threshold":   0.50,
  "truncated":   false,
  "num_nodes":   12,
  "num_edges":   9
}
```

Note: the predictor internally uses `"score"` but the API exposes it as `"confidence"` — a more meaningful name for API consumers. The mapping is explicit in the endpoint:
```python
return PredictResponse(confidence=result["score"], ...)
```

### Error handling

| Exception | Status | Meaning |
|---|---|---|
| `ValueError` from Slither | 400 | Bad input — contract failed to parse. Client should fix their request. |
| Any other `Exception` | 500 | Server error (GPU OOM, model crash, etc.). Generic message externally; full trace internally. |

HTTP 400 vs 500 matters: API consumers use status codes to decide whether to retry (5xx) or fix their request (4xx).

### Running the API

```bash
# From project root
poetry run uvicorn ml.src.inference.api:app --host 0.0.0.0 --port 8000

# Test health
curl http://localhost:8000/health

# Score a contract
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"source_code": "pragma solidity ^0.8.0; contract Test { uint x; function set(uint v) external { x = v; } }"}'
```

Interactive docs (auto-generated by FastAPI): `http://localhost:8000/docs`

---

## Shape invariants (must not change without retraining)

| Tensor | Shape | Where enforced |
|---|---|---|
| `graph.x` | `[N, 8]` | `GNNEncoder(in_channels=8)` will crash if N≠8 |
| `tokens["input_ids"]` | `[1, 512]` | Assert in `_tokenize()`; `TransformerEncoder` fixed at 512 |
| `tokens["attention_mask"]` | `[1, 512]` | Assert in `_tokenize()` |
| Model output | `[B]` sigmoid | Never apply sigmoid again — `BCELoss` only |

---

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (N×17 and 8×64)` | `graph_builder.GraphBuilder` used instead of `_extract_graph()` | Use only `ContractPreprocessor` for inference |
| Slither raises `ValueError: No non-dependency contracts found` | Contract file only contains imports (e.g. a pure interface file) | Use a file with actual contract logic |
| `Truncated=True` in result | Contract exceeded 512 tokens — tail was dropped | Treat result as partial; investigate with manual review |
| `num_nodes < 5` in result | Slither parsed a near-empty or stub contract | Check the source for valid Solidity logic |
| Score non-deterministic across calls | `model.train()` was called somewhere after loading | Always call `model.eval()` before inference |
