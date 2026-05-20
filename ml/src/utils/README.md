# ml/src/utils — Utility Functions

Shared utility functions used across the SENTINEL ML pipeline.

## Purpose

This module contains common utility functions that are used by multiple components in the ML pipeline, ensuring consistency and reducing code duplication.

## Components

### `hash_utils.py`
**Contract Identification and Hashing**

Production-grade contract identification using MD5 hashing.

**Key Functions:**

- `get_contract_hash(contract_path)` — Generate MD5 hash for contract identification
  - Hashes the full contract path to create a unique identifier
  - Returns 32-character hexadecimal string
  - Used as filename for both graph and token files to ensure pairing

**Design Decision (Feb 15, 2026):**
- Use MD5 hash of full contract path for guaranteed uniqueness
- 32-character hexadecimal string (128 bits)
- Industry standard for non-cryptographic file identification
- Collision probability: ~0% for millions of files

**Performance:**
- 44,434 contracts: ~0.13 seconds
- Suitable for millions of contracts

**Usage Example:**
```python
from ml.src.utils.hash_utils import get_contract_hash

path = Path('BCCC-SCsVul-2024/SourceCodes/Reentrancy/contract_001.sol')
contract_hash = get_contract_hash(path)
# Returns: 'a1b2c3d4e5f6789012345678abcdef12'
```

## Critical Integration

**All pipeline components MUST use these functions:**
- Graph extraction (`ml/src/data_extraction/ast_extractor.py`)
- Tokenization (`ml/src/data_extraction/tokenizer.py`)
- Dataset loading (`ml/src/datasets/dual_path_dataset.py`)
- Inference preprocessing (`ml/src/inference/preprocess.py`)

This ensures consistent file naming and pairing across the entire pipeline.

## Technical Details

**Hash Properties:**
- Uses MD5 (not security-critical, just need uniqueness)
- Hashes full path string (not file content)
- UTF-8 encoding for Windows/Linux consistency
- Deterministic: same path always produces same hash

**Why MD5:**
- Fast computation
- Sufficient for non-cryptographic file identification
- Widely supported across platforms
- Collision resistance adequate for dataset scale

## Future Extensions

This module may be expanded to include:
- Additional hashing utilities
- File path manipulation helpers
- Configuration validation utilities
- Common data transformation functions

## Dependencies

- `hashlib` — Standard library for MD5 hashing
- `pathlib` — Path handling
