# data_extraction — Windowed Tokenization

> **Status:** ✅ Current — GraphCodeBERT windowed tokenization, verified 2026-06-14

Windowed tokenization for SENTINEL — produces `[W, 512]` token tensors from Solidity source files.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `windowed_tokenizer.py` | 175 | Per-file tokenization primitives — GraphCodeBERT, sliding window, comment stripping |
| `__init__.py` | — | Package init |
| `_backup_pre_seam_swap_2026-06-12/` | — | Backup of pre-seam-swap code (archived) |

---

## Config Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `TOKENIZER_MODEL` | `"microsoft/graphcodebert-base"` | Tokenizer model name |
| `WINDOW_SIZE` | `512` | Max sequence length per window |
| `STRIDE` | `256` | Overlap between consecutive windows |
| `MAX_WINDOWS` | `4` | Cap; linspace sub-sampling preserves start/mid/end |

---

## Public API

### `init_worker()`

Load the graphcodebert-base tokenizer into the process-level global. Call once per worker process before any `tokenize_windowed_contract()` call.

### `tokenize_windowed_contract(path, max_windows=4, strip_comments=True)`

Per-file entry point. Returns `(input_ids, attention_mask)` — each `[W, 512]` int64 tensors.

**Parameters:**
- `path`: Path to `.sol` file
- `max_windows`: Maximum number of windows (default 4)
- `strip_comments`: Remove `/* */` and `//` comments before tokenization (default True)

**Comment stripping (A-1 fix):**
- Removes `/* */` blocks and `//` line comments before tokenization
- Reclaims token budget for actual code tokens rather than documentation
- NatSpec tags are removed along with all other comment content

---

## Window Selection

When a contract produces more than `max_windows` windows, linspace sub-sampling selects windows that preserve start/mid/end coverage. This ensures the model sees a representative sample of the contract's code.

---

## Usage

```python
from ml.src.data_extraction.windowed_tokenizer import (
    init_worker,
    tokenize_windowed_contract,
    MAX_WINDOWS,
)

# Initialize tokenizer (once per process)
init_worker()

# Tokenize a contract
input_ids, attention_mask = tokenize_windowed_contract(
    Path("contract.sol"),
    max_windows=MAX_WINDOWS,
    strip_comments=True,
)
# input_ids:      [4, 512] int64
# attention_mask:  [4, 512] int64
```
