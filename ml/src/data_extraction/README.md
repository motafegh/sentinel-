# ml/src/data_extraction — Data Extraction Utilities

Windowed tokenization logic extracted from `ml/scripts/retokenize_windowed.py`.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `windowed_tokenizer.py` | 175 | Windowed tokenization for GraphCodeBERT |
| `__init__.py` | 0 | Empty |

---

## windowed_tokenizer.py

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `TOKENIZER_MODEL` | `"microsoft/graphcodebert-base"` | Tokenizer backbone |
| `WINDOW_SIZE` | `512` | Max sequence length per window |
| `STRIDE` | `256` | Overlap between consecutive windows |
| `MAX_WINDOWS` | `4` | Hard cap; linspace sub-sampling preserves start/mid/end |

### Functions

**`init_worker()`** — Load tokenizer into process-level global. Call once per worker process.

**`tokenize_windowed_contract(path, max_windows=4, strip_comments=True)`** — Tokenize one .sol file into `[max_windows, 512]` tensors.

Output:
- Short contracts (< 512 tokens): 1 real window; remaining are zero-padded
- Long contracts (> max_windows): sub-sampled via linspace
- Normal: padded with zero windows

**`_strip_comments(source)`** — Remove `/* */` and `//` comments before tokenization (reclaims token budget for code).

**`_select_windows(all_ids, all_masks, max_windows)`** — Sub-sample via `np.linspace` to preserve beginning, middle, and end.

### Key Design

This module produces the same shape as the offline training pipeline (`retokenize_windowed.py`). The v2 orchestrator (`sentinel_data/representation/orchestrator.py`) uses this module to ensure uniform tensor shapes for DataLoader collation.

Hash is NOT included — the caller (v2 orchestrator) sets the hash from Stage 1's SHA-256 in `meta.json`.
