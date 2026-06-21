# ml/src/utils — Utility Functions

Hash utilities for contract identification across the pipeline.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `hash_utils.py` | 315 | MD5 hashing for contract identification |
| `__init__.py` | 0 | Empty |

---

## hash_utils.py

MD5-based contract identification used by graph extraction, tokenization, and inference cache.

**Functions:**

| Function | Purpose |
|----------|---------|
| `get_contract_hash(path)` | MD5 of full file path (deterministic, for offline pipeline) |
| `get_contract_hash_from_content(content)` | MD5 of source text (content-addressable, for inference cache) |
| `validate_hash(s)` | Validate 32-char lowercase hex string |
| `get_filename_from_hash(h)` | Returns `"{hash}.pt"` |
| `get_filename_from_path(path)` | Combines path hash + filename |
| `extract_hash_from_filename(fn)` | Reverse lookup from `.pt` filename |

**Design:** MD5 is used for non-cryptographic uniqueness (file identification, not security). All pipeline components MUST use these functions for file pairing.
