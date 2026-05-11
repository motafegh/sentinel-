# SENTINEL Encoding Reference — BN254 Field Elements & Score Encoding

This document is the single source of truth for how the model output travels from
a Python float inside the neural network to a `uint256` stored immutably on-chain.
Read this before writing any script that touches `proof.json` or calls `submitAudit`.

---

## The journey of a score

```
Neural network output (Python float)
    │
    │  0.5490  ← model's vulnerability probability for this contract
    │
    ▼
EZKL field element encoding
    │
    │  round(0.5490 × 8192) = round(4497.408) = 4497
    │  4497 = 0x1191
    │
    ▼
32-byte little-endian hex string (stored in proof.json)
    │
    │  "9111000000000000000000000000000000000000000000000000000000000000"
    │   ├─ bytes[0] = 0x91 = 145  (least significant)
    │   ├─ bytes[1] = 0x11 = 17
    │   └─ bytes[2..31] = 0x00 (zeros, value fits in 2 bytes)
    │
    ▼
Python uint256 (for Solidity ABI encoding)
    │
    │  int.from_bytes(bytes.fromhex("9111...00"), byteorder='little') = 4497
    │
    ▼
Solidity uint256 (stored in AuditRegistry)
    │
    │  scoreFieldElement = 4497
    │
    ▼
Human-readable probability (off-chain only)
    │
    │  4497 / 8192 = 0.5490
```

---

## Scale factor

The scale factor comes from EZKL's calibration step (Step 2 in setup_circuit.py).

For SENTINEL v1.0:
```
scale = 13
2^scale = 2^13 = 8192
```

This value is stored in `zkml/ezkl/settings.json` under the key `"scale"`.

**Encode:**
```python
field_element = round(probability * 8192)
# 0.5490 → 4497
# 0.0000 → 0
# 1.0000 → 8192
```

**Decode:**
```python
probability = field_element / 8192
# 4497 → 0.5490
# 8192 → 1.0000
```

---

## Endianness

### What is endianness?

When a multi-byte integer is stored in memory or a file, the bytes can be ordered two ways:

- **Big-endian (BE):** most significant byte first. Used by Ethereum EVM, TCP/IP, most network protocols.
- **Little-endian (LE):** least significant byte first. Used by x86 CPUs, Rust's internal representation, EZKL's JSON output.

For the number 4497 (= 0x1191):

| Format | Hex bytes (32 bytes) |
|---|---|
| Big-endian | `0000...00001191` |
| Little-endian | `9111000000...00` |

### Where EZKL uses little-endian

Every value in `proof.json["instances"][0]` is a 32-byte **little-endian** hex string:

```json
{
  "instances": [
    [
      "0000000000000000000000000000000000000000000000000000000000000000",
      "aa06000000000000000000000000000000000000000000000000000000000000",
      ...
      "9111000000000000000000000000000000000000000000000000000000000000"
    ]
  ]
}
```

The string `"aa06000000000000000000000000000000000000000000000000000000000000"` represents:
- Bytes: `[0xaa, 0x06, 0x00, ..., 0x00]`
- As little-endian uint256: `0x06aa` = 1706
- Not: `0xaa06...00` = a huge garbage number

### The conversion in Python

```python
# CORRECT
value = int.from_bytes(bytes.fromhex(hex_str), byteorder='little')

# WRONG — treats as big-endian
value = int(hex_str, 16)
```

**Rule of thumb:** If you get a value above 10,000 for `instances[64]`, you're using big-endian. The output score is always in `[0, 8192]` after correct little-endian decoding.

---

## What the on-chain verifier expects

The EZKL Halo2 verifier (`ZKMLVerifier.sol`) takes:

```solidity
function verifyProof(bytes calldata proof, uint256[] calldata instances) external returns (bool)
```

- `proof` — the raw proof bytes from `proof.json["hex_proof"]`. Pass as-is.
- `instances` — an array of `uint256` values, one per public signal. These must match the field elements embedded in the proof.

The verifier checks that the provided `instances` values match those embedded in the proof during proving. If you pass big-endian values, they won't match → `execution reverted`.

### Extract all 65 signals correctly

```python
import json
from pathlib import Path

proof = json.loads(Path("zkml/ezkl/proof.json").read_text())
instances = proof["instances"][0]  # list of 65 little-endian hex strings

# Correct decoding
signals = [
    int.from_bytes(bytes.fromhex(h), byteorder='little')
    for h in instances
]

# signals[0..63]: input feature field elements
# signals[64]:    output score field element
score_field_element = signals[64]
```

---

## What AuditRegistry stores

`AuditRegistry.submitAudit()` stores `scoreFieldElement` directly:

```solidity
_audits[contractAddress].push(AuditResult({
    scoreFieldElement: scoreFieldElement,  // raw field element, NOT divided by 8192
    ...
}));
```

To read a stored score off-chain:

```python
# via cast call
result = registry.getLatestAudit(contract_address)
score_field_element = result.scoreFieldElement   # e.g. 4497
probability = score_field_element / 8192         # e.g. 0.5490
```

To read from a transaction receipt (event log):
```python
# AuditSubmitted event data field (non-indexed params):
# bytes 0-31:  proofHash (bytes32)
# bytes 32-63: scoreFieldElement (uint256)
data = log["data"]
proof_hash = data[2:66]
score_hex = data[66:130]
score_field_element = int(score_hex, 16)  # ← safe here: ABI-encoded values are big-endian
probability = score_field_element / 8192
```

Note: values in ABI-encoded event data are big-endian (Ethereum standard). Only `proof.json` instances use little-endian. They are the same numeric value, different byte representations.

---

## Quick reference

| Context | Byte order | Conversion |
|---|---|---|
| `proof.json["instances"][0][i]` | Little-endian | `int.from_bytes(bytes.fromhex(h), 'little')` |
| `proof.json["hex_proof"]` | N/A — pass as-is | No conversion needed |
| ABI-encoded `uint256` (event log, cast output) | Big-endian | `int(hex_str, 16)` ✓ |
| Solidity `uint256` in memory | Big-endian | N/A |

---

## Why EZKL uses little-endian

EZKL is written in Rust. Rust's `ark-ff` library (used for BN254 field arithmetic) stores field elements in little-endian Montgomery form internally. When EZKL serialises to JSON, it uses the native Rust byte representation — little-endian. This is a deliberate choice in the EZKL codebase, not a bug.

The Solidity verifier generated by EZKL handles the endianness conversion internally when it reads the `instances` array from ABI calldata. You only need to worry about endianness when reading from `proof.json` directly in Python.
