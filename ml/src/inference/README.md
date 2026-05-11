# inference — Inference Pipeline

Converts a Solidity contract (file or raw string) into a vulnerability score.

---

## Files

| File | Class | Status |
|---|---|---|
| `preprocess.py` | `ContractPreprocessor` | Complete |
| `predictor.py` | `Predictor` | Next to build (M3.4) |

---

## ContractPreprocessor (`preprocess.py`)

Two public entry points — one for files on disk, one for raw strings from an API.

**Design:** instantiate once, reuse for many contracts.
All expensive setup (tokenizer load, Slither/ASTExtractor init) happens in `__init__`.

### Why two entry points?

`ASTExtractor` calls Slither, which shells out to `solc` (the Solidity compiler).
`solc` requires a **real file path** — it cannot accept source as a string argument.
This is an external tool constraint that cannot be worked around.

| Caller context | Method to use |
|---|---|
| Contract file already on disk (CLI, batch eval) | `process(sol_path)` |
| Raw string from HTTP request / stdin | `process_source(source_code)` |

### Instantiation

```python
from ml.src.inference.preprocess import ContractPreprocessor

preprocessor = ContractPreprocessor()
# Loads once: CodeBERT tokenizer + ASTExtractor (Slither) + GraphBuilder
```

### process(sol_path) — file on disk

```python
graph, tokens = preprocessor.process("contracts/Vault.sol")
```

Hash = MD5 of the resolved file path (matches the offline pipeline convention).

### process_source(source_code) — raw string

```python
source = open("Vault.sol").read()   # or from HTTP request body
graph, tokens = preprocessor.process_source(source, name="Vault")
```

Internally: writes a `NamedTemporaryFile`, runs Slither on it, deletes it in `finally`.
The tokeniser is called directly with the string — no redundant file read.
Hash = MD5 of the source content (content-addressable; enables result caching).

### Return value (both methods)

```python
# graph: PyG Data object
#   graph.x              [N, 8]    float32 — node features
#   graph.edge_index     [2, E]    int64   — directed edges
#   graph.contract_hash  str               — MD5 hash
#   graph.y              tensor([0])       — dummy label (not used by model)

# tokens: dict
#   tokens["input_ids"]      [1, 512]  long  — CodeBERT token IDs
#   tokens["attention_mask"] [1, 512]  long  — 1=real token, 0=padding
#   tokens["contract_hash"]  str
#   tokens["num_tokens"]     int             — real (non-padding) tokens
#   tokens["truncated"]      bool            — True if source > 512 tokens
```

Token shape is `[1, 512]` (batch dim=1). Pass directly to `SentinelModel` — no collate needed.

### Errors

| Exception | Cause |
|---|---|
| `FileNotFoundError` | `sol_path` does not exist (`process()` only) |
| `ValueError` | Empty source (`process_source()`), or Slither/AST extraction failed |

---

## Shape Contract

These shapes must match training data exactly:

| Tensor | Training (from dataset) | Inference (from preprocessor) |
|---|---|---|
| `graph.x` | `[N, 8]` | `[N, 8]` ✓ |
| `graph.edge_index` | `[2, E]` | `[2, E]` ✓ |
| `tokens["input_ids"]` | `[B, 512]` after collate | `[1, 512]` ✓ |
| `tokens["attention_mask"]` | `[B, 512]` after collate | `[1, 512]` ✓ |

---

## Confirmed Inference Threshold

Threshold sweep was run on the val set (10,283 samples) using `run-alpha-tune_best.pt`.
Selection criterion: **F1-macro** (not F1-vuln — see [scripts/README.md](../../scripts/README.md) for why F1-vuln is gameable).

```
 Threshold |  F1-vuln |  Precision |   Recall |  F1-macro
------------------------------------------------------------
      0.40 |   0.8013 |     0.6701 |   0.9962 |    0.5036   ← degenerate (recall-gaming)
      0.50 |   0.7458 |     0.7797 |   0.7147 |    0.6686   ← best (selected)
      0.55 |   0.6446 |     0.8543 |   0.5176 |    0.6325
```

**INFERENCE_THRESHOLD = 0.50** (confirmed 2026-02-27, run-alpha-tune checkpoint)

At 0.50: of all flagged contracts 78% are genuinely vulnerable, and 71.5% of all
truly vulnerable contracts are caught. Precision/recall are well-balanced.

`run-more-epochs_best.pt` (ep 22, F1-macro 0.6584) needs a threshold sweep.
`run-lr-lower` and `run-combined` never ran — overnight process was killed at ep 25/40.
Compare `run-more-epochs` sweep result against `run-alpha-tune` (0.6686) to pick the
production checkpoint for `predictor.py`.

---

## Predictor (`predictor.py`) — Next to Build (M3.4)

`predictor.py` is the single remaining file for milestone M3.4.

**Contract:**
1. Accept `checkpoint` path and `threshold` (default `0.50`)
2. Load `SentinelModel` with `weights_only=True`
3. Instantiate `ContractPreprocessor` once in `__init__`
4. `predict(sol_path) -> dict` — full pipeline, single contract

**Target API:**
```python
from ml.src.inference.predictor import Predictor

predictor = Predictor(
    checkpoint="ml/checkpoints/run-alpha-tune_best.pt",
    threshold=0.50,
    device="cuda",
)

# From a file on disk (CLI / batch)
result = predictor.predict("contracts/Vault.sol")

# From a raw string (HTTP API)
result = predictor.predict_source(source_code, name="Vault")

# result dict (both methods):
# {
#   "score":      0.823,          # raw sigmoid probability [0, 1]
#   "label":      "vulnerable",   # "vulnerable" | "safe"
#   "threshold":  0.50,
#   "truncated":  False,          # source > 512 tokens?
#   "num_nodes":  147,
#   "num_edges":  203,
# }
```

**Implementation notes:**
- `predict()` → calls `preprocessor.process(sol_path)`
- `predict_source()` → calls `preprocessor.process_source(source_code, name)`
- Both routes produce identical `(graph, tokens)` shapes — same model forward pass
- Token shape `[1, 512]` passes directly to `SentinelModel` — no collate needed
