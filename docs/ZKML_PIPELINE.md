# SENTINEL ZKML Pipeline — Technical Reference

## What this pipeline does

The ZKML pipeline bridges the ML model and the blockchain. It produces a ZK proof that lets anyone verify on-chain that:

> "A neural network with these specific weights, when run on these 64 contract features, produced this vulnerability score — without revealing the weights."

The proof is ~2KB. Verifying it on-chain costs ~250K gas. The weights are never revealed.

---

## Two-phase structure

```
PHASE 1 — One-time setup (per model version)
    proxy_model.py         → define tiny ZK-compatible network
    train_proxy.py         → distil from teacher to proxy
    export_onnx.py         → PyTorch → ONNX (EZKL's input format)
    generate_calibration.py → extract real features for scale calibration
    setup_circuit.py       → ONNX → ZK circuit + proving/verification keys

PHASE 2 — Per-audit proof generation
    run_proof.py           → gen_witness → prove → verify (off-chain)
    extract_calldata.py    → decode proof.json → cast commands
    cast send              → submitAudit on AuditRegistry.sol
```

Phase 1 runs once per model version. Phase 2 runs for every audit.

---

## Phase 1: One-time Setup

### Why a proxy model?

The full SENTINEL model has ~125M parameters. ZK circuits scale with parameter count — a 125M-parameter circuit would take hours to prove and require impossibly large cryptographic setup parameters.

The proxy is a tiny 3-layer network (2,625 params) trained to mimic the teacher's output. It operates on the teacher's 64-dimensional FusionLayer output — not raw Solidity. The teacher does all the hard work; the proxy just maps the teacher's compressed understanding to a scalar.

```
Raw Solidity
    ↓
[Teacher: GNN + CodeBERT + Fusion]  ← ~125M params, runs off-chain only
    ↓ 64-dim features
[Proxy: Linear 64→32→16→1]          ← 2,625 params, goes into ZK circuit
    ↓ risk score
[EZKL proof]                         ← proves proxy ran correctly
```

### `proxy_model.py` — architecture definition

```python
ProxyModel:
    Linear(64 → 32) → ReLU
    Linear(32 → 16) → ReLU
    Linear(16 → 1)  → Sigmoid
```

`CIRCUIT_VERSION = "v1.0"` tracks this architecture. If any layer size changes:
1. Bump `CIRCUIT_VERSION`
2. Retrain the proxy
3. Re-export ONNX
4. Rerun the full EZKL pipeline (Steps 1-5)
5. Redeploy `ZKMLVerifier.sol` with the new verification key

Architecture freeze is enforced by assertions in `__init__` — passing non-default dimensions raises `AssertionError` immediately.

### `train_proxy.py` — knowledge distillation

The proxy learns from the teacher using soft targets (the teacher's probability output), not hard labels. This preserves calibration — the proxy outputs similar probabilities to the teacher, not just the same 0/1 decisions.

Training stops when agreement with teacher ≥ 99.82% on the validation set.

### `export_onnx.py` — bridge to EZKL

EZKL cannot read `.pt` files — it only speaks ONNX. This script:
1. Loads `proxy_best.pt`
2. Calls `proxy.eval()` — removes Dropout from the computation graph (ZK circuits must be deterministic)
3. Exports to ONNX opset 11 (required by EZKL 23.x)
4. Verifies export: runs the same input through both PyTorch and ONNX, asserts `max_diff < 1e-5`

**Why opset 11?**
EZKL's circuit compiler translates specific ONNX operations into arithmetic constraints. These operations are stable in opset 11. Higher opsets introduce new operation variants that EZKL cannot yet compile.

### `generate_calibration.py` — data for scale calibration

EZKL encodes floating-point values as integers inside the circuit. The "scale factor" determines the precision:

```
field_element = round(float_value × 2^scale)
```

Too small a scale → overflow (values truncate to zero, wrong proofs).
Too large a scale → circuit is unnecessarily huge, slow proving.

`generate_calibration.py` extracts 200 real 64-dim feature vectors from the validation set and writes them to `calibration.json`. EZKL's Step 2 (`calibrate_settings`) observes the actual value distribution at each layer and picks the optimal scale.

Our calibration found scale=13 (2^13 = 8192) is optimal for this model.

### `setup_circuit.py` — EZKL Steps 1-5

This is the most important script to understand. It runs once and produces the keys that make proofs possible.

#### Step 1 — `gen_settings`
Reads the ONNX graph structure and produces an initial `settings.json` with a guessed scale factor.

#### Step 2 — `calibrate_settings`
Runs real data through the ONNX model, observes actual value ranges, and refines the scale in `settings.json`.

#### Step 3 — `compile_circuit`
Converts ONNX operations into R1CS arithmetic constraints. This is where the model becomes a "circuit" — a mathematical description of what computation the proof must verify.

Key property: **weights are separated from structure**. The compiled circuit captures the structure (layer shapes, operations). The weights become the "private witness" during proving. This is why:
- The circuit can be compiled once
- Retraining the proxy changes weights but not structure
- Retrained proxy = same circuit = same keys = no redeployment needed

#### Step 4 — `get_srs` (async)
Downloads the Structured Reference String (~4MB) from `kzg.ezkl.xyz`. This is a set of BN254 elliptic curve points that form the cryptographic foundation for Halo2 proofs.

The SRS is public and the same for all EZKL users with the same circuit size. It is cached locally after download.

**Why is this async?** `ezkl.get_srs()` wraps a Rust/tokio async runtime. Python's `asyncio.run()` provides the event loop that tokio requires.

#### Step 5 — `setup`
Derives two keys from the circuit + SRS:

| Key | Privacy | Contents | Used for |
|---|---|---|---|
| `proving_key.pk` | **PRIVATE** — never commit | Circuit structure + cryptographic trapdoor | Generating proofs |
| `verification_key.vk` | Public | Circuit fingerprint only | Verifying proofs; baked into `ZKMLVerifier.sol` |

**Why do keys survive proxy retraining?**
Keys are derived from the circuit structure, not weights. Retrain → weights change → structure unchanged → keys remain valid. Resize the architecture → structure changes → must rerun all of Steps 1-5 and redeploy the Solidity verifier.

---

## Phase 2: Per-audit Proof Generation

### `run_proof.py` — Steps 6-8

#### Step 6a — Feature extraction

The proxy never sees raw Solidity. The teacher runs first:

```python
gnn_out         = teacher.gnn(graphs.x, graphs.edge_index, graphs.batch)  # GNN path
transformer_out = teacher.transformer(input_ids, attention_mask)           # CodeBERT path
features        = teacher.fusion(gnn_out, transformer_out)                 # [1, 64]
```

These 64 features are the proxy's input.

#### Teacher/proxy agreement gate

Before generating a proof, the script checks that teacher and proxy agree on the binary classification:

```python
if (teacher_score >= 0.5) != (proxy_score >= 0.5):
    raise ValueError("Teacher/proxy disagreement — proof rejected")
```

**Why is this necessary?**
The ZK proof proves the proxy's output. If the proxy says "VULNERABLE" but the teacher says "SAFE", the proof is cryptographically valid — it correctly proves the proxy's output. But it would register a misleading result on-chain. This hard gate prevents that.

If you see this error, the contract is near the decision boundary. Either use a different contract or investigate whether the proxy needs retraining.

#### Step 6b — `gen_witness`

Encodes the 64 feature floats as BN254 field elements using the calibrated scale:

```
field_element = round(feature_float × 2^13)
Example: 0.131 × 8192 = 1073 → stored as 32-byte little-endian hex field element
```

The witness is an intermediate file (`witness.json`) containing all field elements that will appear in the proof.

#### Step 7 — `prove`

Takes:
- Private: `proving_key.pk` (circuit structure + trapdoor)
- Public: `witness.json` (field-encoded inputs + output)

Produces: `proof.json` (~2KB Halo2 proof)

The proof contains **zero information about the weights**. It only proves that some fixed set of weights, when applied to these specific inputs, produced this specific output.

#### Step 8 — `verify` (off-chain)

Off-chain verification using `verification_key.vk`. If this passes, the on-chain `ZKMLVerifier.verifyProof()` will also pass for the same proof and signals — they use the same verification key.

**Cleanup on failure:** If any step throws an exception, partially written `witness.json` and `proof.json` are deleted. Stale partial files would cause confusing errors on the next run.

### `extract_calldata.py` — decode proof for on-chain submission

After `run_proof.py` completes, this script decodes `proof.json` and generates the exact `cast` commands needed for on-chain submission.

**Critical:** applies the little-endian decode. See the next section.

---

## CRITICAL: BN254 Field Element Encoding

This is the most common source of errors in the ZKML pipeline.

### The problem

EZKL stores all public signals (input features + output score) in `proof.json["instances"][0]` as 32-byte little-endian hex strings.

"Little-endian" means the least significant byte comes first.

Ethereum's `uint256` uses big-endian order (most significant byte first).

These are **opposite byte orders**. You must convert explicitly.

### The correct conversion

```python
# instances[64] — the output risk score field element
hex_str = "9111000000000000000000000000000000000000000000000000000000000000"

# CORRECT — interpret as little-endian
score = int.from_bytes(bytes.fromhex(hex_str), byteorder='little')
# → 4497   (which is 0.5490 × 8192 — makes sense as a model output)

# WRONG — int() treats the string as big-endian
score = int(hex_str, 16)
# → 65615399444674858847734919285089764922285269...   (garbage, not a valid field element)
```

### How to verify you have it right

The score field element divided by 8192 should give a probability between 0 and 1:

```python
probability = score / 8192
assert 0.0 <= probability <= 1.0, "Score is wrong — check endianness"
```

If you get a value larger than 1, you used big-endian.

### Where this matters

| File | Usage |
|---|---|
| `extract_calldata.py` | Decodes all 65 signals for `cast send` |
| Any ad-hoc proof submission script | Must use `int.from_bytes(..., 'little')` |
| Any script reading `instances[i]` directly | Same |

Using `pretty_public_inputs` from proof.json does NOT avoid this — those hex strings have the same little-endian encoding.

### Where this does NOT matter

- `hex_proof` — the raw proof bytes — is already correctly formatted for the Solidity verifier; pass it as-is
- `proof.json` overall structure — only the individual hex strings inside `instances` have endianness

---

## Artifact reference

| File | Phase | Private? | Description |
|---|---|---|---|
| `zkml/models/proxy_best.pt` | 1 | No | Trained proxy weights (PyTorch) |
| `zkml/models/proxy.onnx` | 1 | No | Proxy in ONNX format (EZKL input) |
| `zkml/ezkl/calibration.json` | 1 | No | 200 real feature vectors for scale calibration |
| `zkml/ezkl/settings.json` | 1 | No | Circuit numerics (scale=13, etc.) |
| `zkml/ezkl/model.compiled` | 1 | No | ZK circuit (R1CS constraints) |
| `zkml/ezkl/srs.params` | 1 | No | BN254 structured reference string (~4MB) |
| `zkml/ezkl/proving_key.pk` | 1 | **YES** | How to construct proofs — never commit |
| `zkml/ezkl/verification_key.vk` | 1 | No | How to verify proofs — baked into Solidity |
| `zkml/ezkl/proof_input.json` | 2 | No | 64 feature floats for this specific audit |
| `zkml/ezkl/witness.json` | 2 | No | Field-encoded inputs + output |
| `zkml/ezkl/proof.json` | 2 | No | The ~2KB Halo2 ZK proof |

---

## Running the full pipeline

```bash
# --- Phase 1: one-time setup (already done for v1.0) ---

# Train proxy (if not already done)
poetry run python zkml/src/distillation/train_proxy.py

# Export to ONNX
poetry run python zkml/src/distillation/export_onnx.py

# Generate calibration data
poetry run python zkml/src/distillation/generate_calibration.py

# EZKL circuit setup (Steps 1-5, ~10 min)
poetry run python zkml/src/ezkl/setup_circuit.py

# --- Phase 2: per-audit ---

# Generate proof for a contract (Steps 6-8, ~1 min)
poetry run python zkml/src/ezkl/run_proof.py

# Decode calldata
poetry run python zkml/src/ezkl/extract_calldata.py

# Verify on-chain (should return true)
bash check_verify.sh

# Submit audit
export DEPLOYER_PRIVATE_KEY=0x...
bash submit_audit.sh
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Compiled circuit not found` | setup_circuit.py not run | Run `setup_circuit.py` |
| `Teacher/proxy disagreement` | Contract near decision boundary | Use different contract or retrain proxy |
| `verifyProof returned false` | Wrong signal encoding (big-endian) | Use `int.from_bytes(..., 'little')` |
| `execution reverted: invalid ZK proof` | Proof generated with different keys | Regenerate proof after setup_circuit.py |
| `execution reverted: score mismatch` | `scoreFieldElement` doesn't match `publicSignals[64]` | Use `extract_calldata.py` to extract both |
| `SRS size X MB outside expected range` | Corrupt SRS download | Delete `srs.params`, rerun setup_circuit.py |
| ONNX verification failed, max diff > 1e-5 | Wrong opset or train mode during export | Ensure `proxy.eval()` and `opset_version=11` |
