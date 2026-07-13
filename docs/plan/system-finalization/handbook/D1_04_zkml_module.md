# D1.2c — ZKML Module Doc

**Doc target:** `docs/handbook/04_zkml_module.md`
**Estimated time:** 0.75h
**Rule:** Every claim verified against source code.

---

## Source files to read before writing (8 files)

1. `zkml/src/distillation/proxy_model.py:68,73,104-108,156-162`
   - CIRCUIT_VERSION = "v2.0" (line 68)
   - EZKL_PARAM_LIMIT = 12_000 (line 73)
   - FROZEN_INPUT_DIM=128, FROZEN_HIDDEN1=64, FROZEN_HIDDEN2=32, FROZEN_NUM_CLASSES=10 (lines 104-108)
   - Network: Linear(128→64) → ReLU → Linear(64→32) → ReLU → Linear(32→10) (lines 156-162)
   - Freeze guards: RuntimeError on any dim mismatch (lines 126-148)

2. `zkml/src/distillation/train_proxy.py:72,75,96-110,156-174`
   - TEACHER_CHECKPOINT = Run 12 path (line 72)
   - EXPORT_DIR = v2 baseline export (line 75)
   - extract_features: uses model.forward(return_aux=True), gets fusion_embedding from aux (lines 96-110)
   - compute_agreement: per-class binary match, mean across B*10 pairs (lines 156-174)

3. `zkml/src/distillation/corpus_distill.py`
   - 61-contract fallback when full dataset unavailable
   - Uses Predictor directly (no SentinelDataset)
   - Document when to use this vs train_proxy.py

4. `zkml/src/distillation/export_onnx.py:111,119-139`
   - dummy_input shape (1, 128)
   - opset_version=11 (EZKL requirement)
   - dynamic_axes for batch size
   - do_constant_folding=True
   - Verification: PyTorch vs ONNX output diff < 1e-5

5. `zkml/src/distillation/generate_calibration.py:54,64-70`
   - TEACHER_CHECKPOINT = Run 12 path
   - EXPORT_DIR = v2 baseline export
   - Uses SentinelDataset val split
   - N_CALIBRATION_SAMPLES = 200
   - Output: {"input_data": [[128 floats], ...]}

6. `zkml/src/ezkl/setup_circuit.py:113-267`
   - Step 1: gen_settings → settings.json
   - Step 2: calibrate_settings → refined settings.json
   - Step 3: compile_circuit → model.compiled
   - Step 4: get_srs → srs.params (async, needs asyncio.run)
   - Step 5: setup → proving_key.pk (PRIVATE) + verification_key.vk (PUBLIC)
   - Each step has RuntimeError on failure with diagnostic message

7. `zkml/src/ezkl/run_proof.py:78,144-200,240-270`
   - TEACHER_CHECKPOINT = Run 12 path (line 78)
   - Uses Predictor + model.forward(return_aux=True) for fusion embedding
   - Tries multiple corpus contracts for best agreement (lines 240-270)
   - gen_witness → prove → verify (Steps 6-8)
   - Decodes 10 output field elements from witness
   - Parses proof.json instances → 138 publicSignals

8. `zkml/src/ezkl/extract_calldata.py:60-74,104`
   - _decode_field_element: little-endian hex → uint256
   - NUM_CLASSES=10, INPUT_OFFSET=128, TOTAL_SIGNALS=138
   - Validates len(instances) == 138
   - Splits: [0..127] = fusion features, [128..137] = class scores
   - Outputs check_verify.sh + submit_audit.sh (cast commands)

---

## Sections to write

**1. TL;DR** (5 lines)
```
What: Tiny proxy model (128→64→32→10, 10,666 params) ZK-proven via EZKL Halo2 circuit
Circuit: v2.0, scale=13 (8192), 138 publicSignals (128 inputs + 10 class scores)
Pipeline: distill → export_onnx → calibrate → setup_circuit → run_proof
Tests: source ml/.venv/bin/activate && python -m pytest zkml/tests/ (37 passed)
```

**2. The ZK boundary** (~1 page)
- What gets proved: "I know private weights W such that proxy(128-dim input) = 10 class scores"
- What does NOT get proved:
  - That the 128-dim input came from this specific teacher checkpoint
  - That the teacher's GNN+CodeBERT was correctly applied
  - That the source code analyzed matches the on-chain contract
- How the gap is bridged: provenance manifest (off-chain, EIP-191 signed) — see 07_cross_module.md
- Verify: `proxy_model.py:104-108` — FROZEN_INPUT_DIM=128, FROZEN_NUM_CLASSES=10
- Verify: `proxy_model.py:68` — CIRCUIT_VERSION = "v2.0"
- Verify: `proxy_model.py:73` — EZKL_PARAM_LIMIT = 12_000

**3. Distillation** (~1 page)
- `train_proxy.py`: uses SentinelDataset (17,877 train contracts from v2 export)
  - extract_features: calls model.forward(return_aux=True), gets fusion_embedding from aux_dict
  - Per-logit MSE loss (not scalar mean — P11 fix)
  - compute_agreement: per-class binary match at threshold 0.5
  - Target: 95% agreement, early stopping
- `corpus_distill.py`: 61-contract fallback
  - Uses Predictor directly (bypasses SentinelDataset)
  - Use when: full export unavailable or for quick testing
  - Agreement: 95.33% achieved on 61 contracts
- Verify: `train_proxy.py:96-110` — extract_features uses model.forward
- Verify: `train_proxy.py:156-174` — compute_agreement is per-class

**4. EZKL pipeline** (~1.5 pages)
- Steps 1-5 (one-time, `setup_circuit.py`):
  - Step 1 gen_settings: reads ONNX, produces initial scale factor (scale=13, 2^13=8192)
  - Step 2 calibrate_settings: runs real features through ONNX, refines scale
  - Step 3 compile_circuit: ONNX → R1CS constraints → model.compiled
  - Step 4 get_srs: downloads ~4MB SRS from kzg.ezkl.xyz (async, needs asyncio.run)
  - Step 5 setup: derives proving_key.pk (PRIVATE, ~132MB) + verification_key.vk (PUBLIC, ~65KB)
- Steps 6-8 (per-audit, `run_proof.py`):
  - Step 6 gen_witness: encodes 128 floats as BN254 field elements using scale factor
  - Step 7 prove: Halo2 proving protocol → proof.json (~37KB)
  - Step 8 verify: off-chain verification against vk
- Artifacts summary table:

| Artifact | Path | Size | Private? |
|---|---|---|---|
| settings.json | zkml/ezkl/ | ~1.5KB | No |
| model.compiled | zkml/ezkl/ | ~340KB | No |
| srs.params | zkml/ezkl/ | ~4MB | No (public) |
| proving_key.pk | zkml/ezkl/ | ~132MB | YES |
| verification_key.vk | zkml/ezkl/ | ~65KB | No (public) |
| proof.json | zkml/ezkl/ | ~37KB | No (per-audit) |

- Verify: `setup_circuit.py:113-267` — all 5 step names and error messages
- Verify: `run_proof.py:144-200` — witness generation and field element decoding

**5. Calldata extraction** (~0.5 page)
- `extract_calldata.py`: parses proof.json → Solidity calldata
- Little-endian decoding (CRITICAL):
  - CORRECT: `int.from_bytes(bytes.fromhex(hex_str), byteorder='little')`
  - WRONG: `int(hex_str, 16)` ← big-endian, produces garbage
- publicSignals layout: [0..127] = fusion features, [128..137] = class scores
- Verify: `extract_calldata.py:104` — `len(instances) != 138` check
- Verify: `extract_calldata.py:60-74` — `_decode_field_element` function

**6. How to regenerate everything** (~0.5 page)
- After teacher retrain (e.g., Run 13):
  1. `python zkml/src/distillation/train_proxy.py` — new proxy_best.pt
  2. `python zkml/src/distillation/export_onnx.py` — new proxy.onnx
  3. `python zkml/src/distillation/generate_calibration.py` — new calibration.json
  4. `python zkml/src/ezkl/setup_circuit.py` — new keys + compiled circuit
  5. `ezkl.create_evm_verifier(...)` — new ZKMLVerifier.sol
  6. Redeploy ZKMLVerifier on-chain, update AuditRegistry via upgradeToAndCall
- What changes: proxy_best.pt, proxy.onnx, all ezkl/ artifacts, ZKMLVerifier.sol
- What stays: srs.params (if logrows unchanged), SentinelToken, AuditRegistry proxy address

**7. Deep reference**
- → `zkml/ZKML_STATE_AND_REDESIGN_2026-06-14.md` (historical context)
- → `zkml/README.md`
- → source: `proxy_model.py`, `setup_circuit.py`, `run_proof.py`, `extract_calldata.py`

---

## Verification checklist
- [ ] CIRCUIT_VERSION = "v2.0" matches `proxy_model.py:68`
- [ ] Parameter count 10,666 matches `test_proxy_model.py::test_parameter_count_exact`
- [ ] EZKL_PARAM_LIMIT = 12_000 matches `proxy_model.py:73`
- [ ] EZKL step names match `setup_circuit.py` log messages (Step 1/5, Step 2/5, etc.)
- [ ] publicSignals count 138 matches `extract_calldata.py:104` and `settings.json` model_instance_shapes
- [ ] SCALE = 8192 matches `extract_calldata.py:46`
- [ ] Test command produces 37 passed
