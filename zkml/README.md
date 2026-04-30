# M2 — ZKML Proof Generation

Bridges M1 (ML inference) and M5 (on-chain registry) by generating a ZK proof that a given Solidity contract scored a specific risk level — without revealing the model weights. Uses EZKL/Groth16 over the BN254 curve.

The proof is a cryptographic guarantee: anyone can verify on-chain that the score was produced by the committed model, without re-running the full 125M-parameter SentinelModel.

---

## Why a Proxy Model?

EZKL can prove circuits up to ~10 K parameters. SentinelModel has ~125 M parameters — far too large.

**Solution: Knowledge Distillation**

Train a tiny ProxyMLP to replicate SentinelModel's output on the CrossAttentionFusion embeddings. The proxy learns from the teacher's scores (MSE loss), never from ground truth labels. Target: ≥ 95% per-class agreement with the teacher.

```
SentinelModel forward()
  ├── GNNEncoder + TransformerEncoder
  └── CrossAttentionFusion  →  fused [1, 128]
                                    │
                          ┌─────────┘
                          │  (proof input)
                          ▼
                     ProxyMLP
          Linear(128→64) → ReLU
          Linear(64→32)  → ReLU
          Linear(32→10)  →  proxy_logits [1, 10]
```

---

## Proxy Model Architecture

```
Input:   128-dim fused embedding  (CrossAttentionFusion output — BEFORE classifier)
Layers:  Linear(128→64) → ReLU → Linear(64→32) → ReLU → Linear(32→10)
Params:  ~8 300
Target:  per-class agreement with teacher ≥ 95 %
Loss:    MSE(proxy_output, teacher_output)
```

**Circuit version: v2.0** — Architecture is frozen by ADR-007.
**Input dim = 128 is locked** to `CrossAttentionFusion output_dim`. If that changes, the ONNX export, EZKL circuit, `ZKMLVerifier.sol`, and `AuditRegistry` deployment must all be rebuilt.

---

## Full Pipeline

```
Step 0  Train proxy                 zkml/src/distillation/train_proxy.py
Step 1  Export to ONNX              zkml/src/distillation/export_onnx.py
Step 2  Gen EZKL settings           zkml/src/ezkl/setup_circuit.py  (step 1–2)
Step 3  Compile R1CS circuit        zkml/src/ezkl/setup_circuit.py  (step 3)
Step 4  Setup — gen keys            zkml/src/ezkl/setup_circuit.py  (step 4–5)
                                    → proving_key.pk  (gitignored, ~10 MB)
                                    → verification_key.vk
                                    → ZKMLVerifier.sol  (copy to contracts/src/)
        ── ONE TIME ─────────────────────────────────────────────────────────────
Step 5  Per-audit: gen proof        zkml/src/ezkl/run_proof.py
                                    → proof.json  (π ~2 KB)
                                    → publicSignals[10 class scores]
Step 6  Extract calldata            zkml/src/ezkl/extract_calldata.py
                                    → publicSignals[65] (64 features + 1 score)
Step 7  Submit on-chain             submit_audit.sh
```

Steps 0–4 are **one-time setup**. Steps 5–7 run **per audit**.

---

## Running the Pipeline

### Step 0 — Train the proxy

Requires: M1 trained checkpoint at `ml/checkpoints/multilabel_crossattn_best.pt`.

```bash
cd zkml
TRANSFORMERS_OFFLINE=1 \
SENTINEL_CHECKPOINT=../ml/checkpoints/multilabel_crossattn_best.pt \
poetry run python -m src.distillation.train_proxy \
  --graphs-dir ../ml/data/graphs/ \
  --tokens-dir ../ml/data/tokens/ \
  --output zkml/ezkl/proxy_best.pt
# Trains for up to 50 epochs; saves when agreement >= 95 %
```

### Step 1 — Export to ONNX

```bash
poetry run python -m src.distillation.export_onnx \
  --checkpoint zkml/ezkl/proxy_best.pt \
  --output zkml/ezkl/proxy.onnx
# Verifies PyTorch vs ONNX outputs (max diff tolerance 1e-5)
```

### Step 2–5 — Circuit setup (one-time, expensive)

```bash
poetry run python -m src.ezkl.setup_circuit
# Runs gen_settings → calibrate_settings → compile_circuit → get_srs → setup
# Generates:
#   zkml/ezkl/settings.json
#   zkml/ezkl/model.compiled
#   zkml/ezkl/proving_key.pk      ← gitignored (~10 MB)
#   zkml/ezkl/srs.params          ← gitignored
#   zkml/ezkl/verification_key.vk
#   zkml/ezkl/ZKMLVerifier.sol    ← copy to contracts/src/
```

After this step, copy the verifier contract and compile with solc 0.8.17:
```bash
cp zkml/ezkl/ZKMLVerifier.sol contracts/src/ZKMLVerifier.sol
solc-select use 0.8.17
cd contracts && forge build --contracts src/ZKMLVerifier.sol
solc-select use 0.8.20
```

### Step 5 — Generate proof per audit

```bash
cd zkml
poetry run python -m src.ezkl.run_proof \
  --contract test_contracts/simple_reentrancy.sol
# Takes 30–60 s on RTX 3070
# Writes: zkml/ezkl/proof.json
```

The script validates that teacher and proxy agree on the classification before accepting the proof.

### Step 6 — Extract calldata for on-chain submission

```bash
poetry run python -m src.ezkl.extract_calldata
# Reads: zkml/ezkl/proof.json
# Outputs:
#   check_verify.sh    test ZKMLVerifier.verify() via cast
#   submit_audit.sh    submit to AuditRegistry via cast
```

---

## Critical Encoding Details

EZKL stores field elements as **little-endian 32-byte hex strings**.

```python
# CORRECT
score = int.from_bytes(bytes.fromhex(instances[64]), byteorder='little') / 8192

# WRONG — treats as big-endian, produces garbage
score = int(instances[64], 16) / 8192
```

`publicSignals[64]` is the score field element index in `proof.json`.
Scale factor: `score_field_element = round(model_output * 8192)` (EZKL scale 2¹³).
`AuditRegistry` stores `scoreFieldElement` raw; divide by 8192 to get human-readable probability.

---

## Artifacts

| File | Status | Notes |
|------|--------|-------|
| `zkml/ezkl/proxy_best.pt` | generated | ProxyMLP weights |
| `zkml/ezkl/proxy.onnx` | generated | ONNX export (opset 11) |
| `zkml/ezkl/settings.json` | generated | EZKL quantisation config |
| `zkml/ezkl/calibration_data.json` | generated | Calibration inputs |
| `zkml/ezkl/model.compiled` | generated | Compiled R1CS circuit |
| `zkml/ezkl/verification_key.vk` | generated | Public verification key |
| `zkml/ezkl/proving_key.pk` | **gitignored** | Private proving key (~10 MB) |
| `zkml/ezkl/srs.params` | **gitignored** | BN254 SRS (~4 MB) |
| `zkml/ezkl/proof.json` | per-audit | Most recent proof artifact |
| `zkml/ezkl/ZKMLVerifier.sol` | generated | Copy to `contracts/src/` |

---

## EZKL Version Notes (23.0.5)

| Function | Behaviour |
|----------|-----------|
| `get_srs` | **async** — must wrap with `asyncio.run()` |
| `compile_circuit` | sync (was `compile_model` in older versions) |
| `calibrate_settings` | sync (was `calibrate` in older versions) |
| All other functions | sync |

Do not upgrade EZKL without verifying function signatures; names changed between releases.

---

## Deployment Addresses (Sepolia)

Populated after `setup_circuit.py` generates `ZKMLVerifier.sol` and it is deployed:

| Contract | Address |
|---------|---------|
| `ZKMLVerifier` | `0xB7093Be4958dd95438D6f53Ff7DF8659451CbD97` |
| `AuditRegistry` | `0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf` |

Update these in `audit_server.py` `AUDIT_REGISTRY_ADDRESS` env var and `extract_calldata.py` constants when redeploying.

---

## File Reference

```
zkml/src/
  distillation/
    proxy_model.py           ProxyMLP definition (circuit version v2.0)
    train_proxy.py           Knowledge distillation from SentinelModel
    export_onnx.py           PyTorch → ONNX (opset 11)
    generate_calibration.py  Calibration data from real embeddings

  ezkl/
    setup_circuit.py         One-time: gen_settings → calibrate → compile → setup
    run_proof.py             Per-audit: witness → prove → verify
    extract_calldata.py      proof.json → publicSignals + shell scripts

zkml/ezkl/                   Artifact directory (most files gitignored)
```

---

## Do Not Change Without Wider Plan

- **Never change `proxy_model.py` architecture** without incrementing circuit version, rerunning `setup_circuit.py`, regenerating `ZKMLVerifier.sol`, and redeploying on-chain.
- **Never change `CrossAttentionFusion output_dim`** without the full ZKML rebuild chain.
- **Never change ONNX opset** from 11 — EZKL 23.0.5 requires it.
- **Never commit `proving_key.pk` or `srs.params`** — gitignored for size reasons; losing `pk` requires re-running setup (expensive).
- **Do not expose `submit_audit`** via `audit_server.py` until Track 3 multi-label proof semantics are confirmed — the current `AuditResult` stores one `scoreFieldElement`; a 10-class output may need a different proof commitment design.
