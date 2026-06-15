# zkml Module — State Audit & Redesign Plan (2026-06-14)

> **Purpose:** Working reference for zkml module alignment with Run 12/13 model.
> Written from source-code read of `zkml/src/` and review of artifacts in `zkml/ezkl/` and `zkml/models/`.
> Do not re-read source files when starting this work — use this document.
>
> **Context:** Run 12 checkpoint trained on v3 data (f1_tuned=0.6941). Run 13 will change
> NUM_CLASSES from 10 to 9 (drop GasException). The zkml module has ALL code implemented
> but the artifacts (proxy model, circuit) were trained on an OLD teacher (pre-Run-12).

---

## 1. Is zkml Fully Implemented?

**Code: YES (100%). Artifacts: STALE (need regeneration for Run 13).**

### Code status

| File | Lines | Purpose | Status |
|---|---|---|---|
| `src/distillation/proxy_model.py` | ~150 | ProxyModel definition (128→64→32→NUM_CLASSES) | ✅ Complete |
| `src/distillation/train_proxy.py` | ~250 | Knowledge distillation training loop | ✅ Complete |
| `src/distillation/export_onnx.py` | ~100 | Export proxy.onnx + verify shapes | ✅ Complete |
| `src/distillation/generate_calibration.py` | ~100 | EZKL calibration JSON generation | ✅ Complete |
| `src/ezkl/setup_circuit.py` | ~200 | One-time EZKL pipeline Steps 1-5 | ✅ Complete |
| `src/ezkl/run_proof.py` | ~200 | Per-audit proof generation Steps 6-8 | ✅ Complete |
| `src/ezkl/extract_calldata.py` | ~100 | Convert proof.json → Solidity calldata | ✅ Complete |

### Artifact status

| Artifact | Path | Status | Issue |
|---|---|---|---|
| Proxy model checkpoint | `zkml/models/proxy_best.pt` | ⚠️ STALE | Trained on old teacher (pre-Run-12) |
| Proxy ONNX | `zkml/models/proxy.onnx` | ⚠️ STALE | Exported from stale proxy_best.pt |
| EZKL calibration | `zkml/ezkl/calibration.json` | ⚠️ STALE | Calibrated on old proxy ONNX |
| Compiled circuit | `zkml/ezkl/model.compiled` | ⚠️ STALE | Compiled from stale calibration |
| Proving key | `zkml/ezkl/proving_key.pk` | ⚠️ STALE | Tied to stale compiled circuit |
| Verification key | `zkml/ezkl/verification_key.vk` | ⚠️ STALE | Tied to stale compiled circuit |
| Proof (test) | `zkml/ezkl/proof.json` | ⚠️ STALE | Proof of old proxy on old input |
| SRS params | `zkml/ezkl/srs.params` | ✅ Reusable | SRS is fixed for BN254; only regen if degree changes |
| Verifier ABI | `zkml/ezkl/verifier_abi.json` | ⚠️ STALE | ABI of old verifier |

**Bottom line:** All pipeline code is complete and well-designed. Every artifact needs regeneration once the teacher model is finalised (Run 13, after GasException is dropped and NUM_CLASSES=9).

---

## 2. Architecture — How the Pipeline Works

### The key design decision (locked, correct)

The ZK proof does NOT prove the full 125M-param SentinelModel. It proves a tiny **proxy model** (8K params) that maps `CrossAttentionFusion output [128-dim] → class logits [NUM_CLASSES]`.

```
Full SentinelModel (125M params, too large for ZK):
  [Solidity source]
       ↓ graph_extractor  → PyG graph
       ↓ windowed_tokenizer → CodeBERT tokens
       ↓ GNNEncoder (8L GAT) → node embeddings [256-dim]
       ↓ TransformerEncoder (GraphCodeBERT + LoRA) → token embeddings [768-dim]
       ↓ CrossAttentionFusion → fusion embedding [128-dim]  ← THIS is the ZK boundary
       ↓ 4-eye classifier → [10|9 class logits]

Proxy model (8K params, ZK-provable):
  [128-dim fusion embedding] → [64] → [32] → [9 class logits]
```

**Why this is correct and future-proof:**
- The 128-dim CrossAttentionFusion output is LOCKED by ADR-025 — it doesn't change between runs
- Only the proxy weights change when we retrain on a new teacher
- The proxy architecture (layers, sizes) is FIXED — changing it requires new circuit keys

### EZKL pipeline in two phases

**Phase 1 — ONE-TIME setup (setup_circuit.py, already done once):**
```
Steps 1-5:
  1. gen_settings    → ezkl/settings.json       (circuit parameters)
  2. calibrate       → ezkl/calibration.json    (quantisation calibration)
  3. compile_model   → ezkl/model.compiled      (compiled EZKL circuit)
  4. get_srs         → ezkl/srs.params          (structured reference string, BN254)
  5. setup           → ezkl/proving_key.pk      (Prover key)
                     → ezkl/verification_key.vk (Verifier key — used to generate ZKMLVerifier.sol)
```

**Phase 2 — PER AUDIT (run_proof.py, runs for every new contract):**
```
Steps 6-8:
  6. gen_witness → ezkl/witness.json        (encode 128-dim input as field elements)
  7. prove       → ezkl/proof.json          (~2KB cryptographic proof π)
  8. verify      → confirm proof is valid locally
```

**After Phase 1:** Extract calldata → submit to AuditRegistry on-chain.

---

## 3. Critical Issue: publicSignals Index Mismatch

### The problem

`run_proof.py` comments state:
```
RECALL — what publicSignals contains:
    [features[0..63], risk_score]
    Index 0-63:  the 64 input features (public)
    Index 64:    the risk score output (public)
    AuditRegistry checks: publicSignals[64] == scoreFieldElement
```

But `proxy_model.py` takes `input_dim=128` (128 features), not 64.

**Discrepancy:** publicSignals[0..63] suggests 64 public inputs, but the proxy takes 128-dim input.

**Likely resolution:** EZKL may pack two float16 values per field element, making 128 floats fit into 64 field elements. OR the "64 features" comment is stale from an older proxy version that used 64-dim input. This needs verification when regenerating the circuit.

**Action:** When running setup_circuit.py for Run 13, print `len(publicSignals)` from the generated witness.json and update the index comment accordingly. Also update AuditRegistry's Guard 3 index (`publicSignals[N] == scoreFieldElement`) to match.

---

## 4. NUM_CLASSES Change: 10 → 9 Impact on zkml

When Run 13 drops GasException:

| Component | Change required | Effort |
|---|---|---|
| `proxy_model.py` | Change `out_dim=10` → `out_dim=9` (or read from config) | 10 min |
| `train_proxy.py` | No change — reads num_classes from teacher config | 0 min |
| Full proxy retrain | Retrain proxy on Run 13 teacher | ~2 hr (small model) |
| Re-export ONNX | `export_onnx.py` on new proxy_best.pt | 15 min |
| Re-generate calibration | `generate_calibration.py` on new ONNX | 15 min |
| Rerun setup_circuit.py | Steps 1-5, gets new pk/vk | ~30 min (depends on EZKL speed) |
| Regenerate ZKMLVerifier.sol | `ezkl gen-verifier --vk-path ...` | 5 min |
| Redeploy contracts | ZKMLVerifier + update AuditRegistry verifier address | 30 min |

**Total: ~4 hours of work.** The circuit architecture changes (output shape changes 10→9) so ALL artifacts must be regenerated. The SRS params (`srs.params`) can be reused if the circuit degree stays the same.

---

## 5. What Needs to Change in Proxy Architecture

### Current proxy (proxy_model.py)
```python
class ProxyModel(nn.Module):
    def __init__(self, input_dim: int = 128, out_dim: int = 10):
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_dim)  # out_dim=10 → must be 9 for Run 13
```

### Fix
Make `out_dim` configurable and default to the right value:
```python
# Option A: read from environment / config file
out_dim: int = int(os.getenv("SENTINEL_NUM_CLASSES", "9"))  # 9 after Run 13

# Option B: read from teacher checkpoint config at init time
# (train_proxy.py already does this — propagate to proxy_model.py __init__)
```

**Recommendation: Option B** — pass `num_classes` from the teacher checkpoint config into `ProxyModel.__init__`. This makes the proxy always consistent with the teacher.

---

## 6. Multi-Class Proof Output — The Open Design Question

Currently, EZKL is configured to prove a SCALAR output (one float → one field element). But the proxy outputs NUM_CLASSES=9 floats. The current design (as implied by `publicSignals[64]`) treats only one of them as the public output.

**Which class's logit is the public output?** Unclear from code. Likely the max logit or a risk aggregation. This is the same multi-class decision that affects contracts (see CONTRACTS_STATE report).

**Three options:**

| Option | publicSignals shape | AuditRegistry change | Gas cost |
|---|---|---|---|
| A: Prove all 9 class logits | `[128 inputs..., logit_0, ..., logit_8]` | Store `uint256[9]` | Higher |
| B: Prove single max-class logit | `[128 inputs..., max_logit, class_index]` | Store `(uint256 score, uint8 class)` | Same as now |
| C: Prove a weighted risk score | `[128 inputs..., risk_score]` | Store `uint256` (as now) | Same as now |

**Recommendation: Option A** — prove all 9 class logits. Rationale:
- SENTINEL's value proposition is multi-class detection, not a single risk score
- On-chain verifiability of individual class scores enables trustless per-class auditing
- Gas cost is ~9x per call but one audit costs on the order of 50k–200k gas anyway — manageable
- This is the only option that preserves full verifiability without information loss

Coordinate this decision with the contracts redesign (see CONTRACTS_STATE report — both must agree on the on-chain format).

---

## 7. Regeneration Runbook (for Run 13)

Run this after Run 13 training completes and the checkpoint is promoted:

```bash
cd /home/motafeq/projects/sentinel

# Step 0: Set env vars
export TEACHER_CHECKPOINT=ml/checkpoints/GCB-P1-Run13-v4-best.pt  # (example)
export NUM_CLASSES=9

# Step 1: Retrain proxy via knowledge distillation (uses Run 13 teacher)
# Input: teacher checkpoint + v3 export training split
# Output: zkml/models/proxy_best.pt
ml/.venv/bin/python zkml/src/distillation/train_proxy.py \
  --teacher-checkpoint $TEACHER_CHECKPOINT \
  --export-dir data_module/data/exports/sentinel-v3-smartbugs-2026-06-13 \
  --epochs 50 \
  --out zkml/models/proxy_best.pt

# Step 2: Export ONNX
# Output: zkml/models/proxy.onnx
ml/.venv/bin/python zkml/src/distillation/export_onnx.py \
  --checkpoint zkml/models/proxy_best.pt \
  --out zkml/models/proxy.onnx

# Step 3: Generate calibration data
# Output: zkml/ezkl/calibration.json
ml/.venv/bin/python zkml/src/distillation/generate_calibration.py \
  --onnx zkml/models/proxy.onnx \
  --out zkml/ezkl/calibration.json

# Step 4: Run EZKL setup (Steps 1-5, the expensive one-time part)
# Output: proving_key.pk, verification_key.vk, model.compiled, settings.json
ml/.venv/bin/python zkml/src/ezkl/setup_circuit.py \
  --onnx zkml/models/proxy.onnx \
  --calibration zkml/ezkl/calibration.json \
  --srs zkml/ezkl/srs.params \  # reuse existing SRS if degree unchanged
  --out-dir zkml/ezkl/

# Step 5: Generate ZKMLVerifier.sol
ezkl gen-verifier \
  --vk-path zkml/ezkl/verification_key.vk \
  --out contracts/src/ZKMLVerifier.sol

# Step 6: Compile ZKMLVerifier with solc 0.8.17 (EZKL requires deprecated opcodes)
solc-select use 0.8.17
cd contracts && forge build --contracts src/ZKMLVerifier.sol

# Step 7: Redeploy (see CONTRACTS_STATE report for deployment sequence)
```

---

## 8. test_proxy_distillation — No Tests Currently

`zkml/tests/` is EMPTY. There are no unit or integration tests for any zkml component.

Minimum tests to add before the Run 13 regeneration:
1. `test_proxy_shapes.py` — verify ProxyModel output shape matches NUM_CLASSES
2. `test_proxy_agreement.py` — verify proxy agrees with teacher > 95% of time on held-out set
3. `test_proof_roundtrip.py` — gen_witness → prove → verify on a single input; assert verify=True

These tests will catch the "wrong publicSignals index" bug and shape mismatches before they reach contracts.

---

## 9. Key Constants to Track

| Constant | Current value | After Run 13 |
|---|---|---|
| `proxy_model.py: input_dim` | 128 | 128 (unchanged — CrossAttentionFusion output locked) |
| `proxy_model.py: out_dim` | 10 | **9** |
| `CIRCUIT_VERSION` | (check file) | Bump after any architecture change |
| `publicSignals` score index | 64 (from run_proof.py comments) | TBD — verify after circuit regen |
| EZKL scale factor | 8192 (2¹³) | Unchanged (contracts hardcode 8192) |
| BN254 field size | ~2²⁵⁴ | Unchanged |

---

## 10. File Locations

| What | Path |
|---|---|
| Proxy model definition | `zkml/src/distillation/proxy_model.py` |
| Distillation training | `zkml/src/distillation/train_proxy.py` |
| ONNX export | `zkml/src/distillation/export_onnx.py` |
| Calibration generation | `zkml/src/distillation/generate_calibration.py` |
| EZKL one-time setup | `zkml/src/ezkl/setup_circuit.py` |
| Per-audit proof | `zkml/src/ezkl/run_proof.py` |
| Calldata extraction | `zkml/src/ezkl/extract_calldata.py` |
| Proxy checkpoint (stale) | `zkml/models/proxy_best.pt` |
| Proxy ONNX (stale) | `zkml/models/proxy.onnx` |
| EZKL artifacts (stale) | `zkml/ezkl/` |
| Old proposal/spec | `docs/archive/Project-Spec/SENTINEL-M2-ZKML.md` (treat as stale) |
| Historical contracts doc | `docs/archive/4-6-2026-docs/ZKML_PIPELINE.md` (treat as stale) |
