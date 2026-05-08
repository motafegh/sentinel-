# SENTINEL — Module 2: ZKML

Load for: EZKL pipeline, proxy model, ONNX export, circuit status.
Always load alongside: **SENTINEL-CONSTRAINTS.md**

---

## Tech Stack

| Tool | Role |
|------|------|
| EZKL 23.0.5 | ZK circuit generation, proving, verification |
| PyTorch | Proxy model definition and distillation |
| ONNX opset 11 | Intermediate representation for EZKL |
| Foundry | ZKMLVerifier.sol deployment and testing |
| web3.py ^7.15 | Proof submission to AuditRegistry |

---

## Proxy Model Architecture

```
Full SentinelModel: ~315K trainable params → too large for EZKL (~10K limit)

Proxy MLP (CIRCUIT_VERSION = "v2.0"):
  Input:  128-dim fused embedding (CrossAttentionFusion output — BEFORE classifier)
  Layers: Linear(128→64) → ReLU → Linear(64→32) → ReLU → Linear(32→10)
  Params: ~8K
  Target: proxy agrees with full model ≥95% per class
  Loss:   MSE(proxy_output, full_model_output)

CIRCUIT_VERSION history:
  v1.0 — Linear(64→32→16→1) binary, old FusionLayer — OBSOLETE; all v1.0 EZKL artifacts invalid
  v2.0 — Linear(128→64→32→10) multi-label, CrossAttentionFusion — CURRENT

Bugs fixed (Z1/Z2/Z3, 2026-05-01):
  Z1 train_proxy.py:     Wrong checkpoint path (run-alpha-tune → multilabel_crossattn_best),
                         wrong feature dim comments (64→128), wrong distillation target shape
  Z2 export_onnx.py:     dummy_input was torch.randn(1, 64) → corrected to torch.randn(1, 128)
  Z3 generate_calibration.py: feature extraction updated 64-dim → 128-dim

ONNX export:
  opset_version=11        — EZKL requirement
  dynamic_axes: batch     — required for variable batch sizes
  dummy_input shape: (1, 128)   — must match CrossAttentionFusion output exactly

Status: source complete, pipeline not yet run.
  ⚠️ No resolution path scheduled — requires explicit decision (see docs/ROADMAP.md S5.5):
  Option A (run pipeline with GPU + EZKL env) or Option B (formally descope to S10).

  Note: active checkpoint is now multilabel-v3-fresh-60ep_best.pt. If Option A is chosen,
  proxy distillation must be run against this checkpoint (not the old multilabel_crossattn_best.pt).
```

---

## EZKL Pipeline

```
Step 1: gen_settings(model.onnx, settings.json)
Step 2: calibrate_settings(model.onnx, calibration_data.json, settings.json)
Step 3: compile_circuit(model.onnx, settings.json, model.compiled)
Step 4: setup()  ← ONE TIME, expensive, generates proving_key.pk + verification_key.vk
Step 5: prove()  ← PER AUDIT, ~30-60s
  outputs: proof π + publicSignals[10 class scores]
Step 6: verify()  ← on-chain or off-chain
  gas: ~250K on-chain

CRITICAL: use RuntimeError not assert in setup_circuit.py (ADR-019)
  python -O strips assert silently; EZKL cascade means silent failure corrupts circuit

EZKL 23.0.5 function names:
  compile_model → compile_circuit
  calibrate     → calibrate_settings
  get_srs requires asyncio.run() + await (PyO3 Rust future wrapping)
  all other EZKL functions are synchronous

EZKL instance value encoding:
  Values stored as little-endian 32-byte hex strings
  Correct decode: int.from_bytes(bytes.fromhex(x), byteorder='little')
  int(x, 16) treats as big-endian — produces garbage values

EZKL scale factor = 8192 (2^13)
  audit_server.py: score = field_element / 8192
```

---

## Locked Constants

```
ONNX opset version = 11          — EZKL compatibility; do not change
CIRCUIT_VERSION    = "v2.0"      — bump requires: re-export ONNX + full EZKL rebuild + ZKMLVerifier.sol redeploy
EZKL scale factor  = 8192 (2^13) — baked into circuit; change requires full rebuild
proxy input_dim    = 128         — must match CrossAttentionFusion output_dim (LOCKED)

Files never to commit:
  zkml/ezkl/proving_key.pk   (~10MB, gitignored)
  zkml/ezkl/srs.params       (gitignored)
```

---

## File Inventory

```
zkml/src/ezkl/
  setup_circuit.py         EZKL pipeline steps 1–4
  run_proof.py             Proof generation per audit (step 5)
  extract_calldata.py      Format proof for Solidity calldata

zkml/src/distillation/
  proxy_model.py           Proxy MLP definition (input_dim=128, CIRCUIT_VERSION="v2.0")
                           Guards: RuntimeError (not assert) on dim mismatch (ADR-019)
  train_proxy.py           Knowledge distillation training (Z1 fixed)
  export_onnx.py           ONNX export; dummy_input (1, 128) (Z2 fixed)
  generate_calibration.py  128-dim calibration data (Z3 fixed)

zkml/ezkl/
  proof.json               Most recent proof artifact
  proving_key.pk           PRIVATE — gitignored
  srs.params               PRIVATE — gitignored
  settings.json            Circuit settings
  model.compiled           Compiled circuit
```
