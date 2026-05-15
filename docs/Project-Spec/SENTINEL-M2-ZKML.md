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
  ⚠️  No resolution path scheduled — requires explicit decision (see docs/ROADMAP.md S5.5):
  Option A (run pipeline with GPU + EZKL env) or Option B (formally descope to S10).

  Note: active checkpoint is v5.2-jk-20260515c-r3_best.pt (training IN PROGRESS as of 2026-05-15).
  Proxy distillation MUST wait for v5.2 training to complete and behavioral gates to pass
  (ml/scripts/manual_test.py must exceed all v4 per-class floors before distillation begins).
  Do NOT distill against a mid-training checkpoint — proxy quality will be meaningless.
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

---

## Upgrade Proposals (v2)

### ZK-1: PLONK/FFLONK Migration (Replace Groth16)

**Problem:** Groth16 requires a per-circuit trusted setup (~2–4h, produces toxic waste that must be
securely destroyed). Every CIRCUIT_VERSION bump triggers a new ceremony. For a project that
anticipates fusion_dim or NUM_CLASSES changes, this is operationally expensive.

**Solution:** Migrate to PLONK (or FFLONK for smaller proofs). PLONK uses a universal SRS — the
SRS is generated once for a maximum circuit size and reused for any circuit of equal or smaller
size. FFLONK reduces proof size a further ~40% over PLONK via polynomial commitment folding.

**Tradeoffs:**
- PLONK proving time: ~2–3× slower than Groth16 (important for ZK-3 GPU proposal)
- Proof size: PLONK ~500 bytes vs Groth16 ~200 bytes; FFLONK ~300 bytes
- On-chain verification: PLONK verifier gas ~350K vs Groth16 ~250K (extra polynomial evaluations)
- Universality benefit: amortised cost acceptable if ZK-2 (recursive aggregation) is adopted

**Decision gate:** Adopt PLONK only if ZK-2 is also accepted; PLONK's universality only
pays off when circuits are batched. If ZK-2 is deferred, keep Groth16.

**Blocked by:** EZKL PLONK/FFLONK support — check `ezkl --help` and EZKL GitHub releases
before starting. As of EZKL 23.0.5, only Groth16/KZG is supported; track the roadmap.

---

### ZK-2: Recursive Proof Aggregation

**Problem:** Each audit costs ~30–60s proving time and ~250K gas on-chain. At 1,000 audits/day
that is ~250M gas/day — unscalable on mainnet (current block gas limit ~30M/block).

**Solution:** Accumulate N proofs into a single aggregate proof via recursive SNARKs. The
aggregator submits one on-chain verification covering a batch of N audits, reducing effective
per-audit on-chain gas to ~250K / N.

**Implementation:**
```
zkml/src/aggregation/
  proof_accumulator.py     Collects pending (proof, publicSignals) pairs in a local queue
                           Flushes when queue length == BATCH_SIZE (configurable, default 32)
  batch_prover.py          Calls EZKL recursive prover on the accumulated batch
                           Outputs aggregate_proof.json + batch_public_inputs.json

contracts/src/
  AggregatorRegistry.sol   Receives aggregate proof; emits BatchVerified(batchId, N, root)
```

**Aggregator contract interface (sketch):**
```solidity
function submitBatch(
    bytes calldata aggregateProof,
    uint256[] calldata publicInputs,   // flattened: N × 10 class scores
    bytes32 batchRoot                  // Merkle root of individual audit hashes
) external returns (bytes32 batchId);
```

**Gate:** Requires EZKL recursive proving support OR a separate KZG accumulator
(e.g., aztec-packages snark-lib). Depends on ZK-1 if PLONK is chosen (Halo2/Nova recursion
works naturally with PLONK; Groth16 recursion is possible but heavier).

---

### ZK-3: GPU-Accelerated Parallel Proving

**Problem:** EZKL proof generation is CPU-bound on BN254 multi-scalar multiplication (MSM),
taking ~30–60s per proof on the current dev machine. This is the primary throughput bottleneck.

**Solution:** Enable GPU acceleration via CUDA MSM. BN254 MSM is parallelisable; GPU
implementations achieve 4–10× speedup over CPU.

**Options (in priority order):**
1. **EZKL GPU backend** — check `ezkl` releases for a CUDA-enabled build. If available,
   set `EZKL_USE_GPU=1` and rebuild. No code changes required.
2. **rapidsnark** (GPU fork of snarkjs) — generates Groth16 proofs; EZKL would export
   witness + r1cs, rapidsnark handles proving. Requires a thin adapter in `run_proof.py`.
3. **bellman-cuda** — lower-level; more integration work but works with any Groth16 circuit.

**Expected speedup on RTX 3070 8GB (BN254 curve, current proxy circuit ~8K constraints):**
- CPU baseline: ~30–60s
- GPU (CUDA MSM): ~5–10s (4–6× for the constraint count of the proxy MLP)
- Note: for large recursive circuits (ZK-2) the GPU benefit increases further

**Blocked by:** EZKL GPU build availability + CUDA 12.x compatibility on WSL2
(check `nvidia-smi` inside WSL2 and confirm CUDA_HOME is set before attempting).

---

### ZK-4: On-Chain Circuit Version Registry

**Problem:** ZKMLVerifier.sol is redeployed whenever CIRCUIT_VERSION changes (fusion_dim
resize, NUM_CLASSES change, proxy architecture change). The current design hardcodes the
verifier address in AuditRegistry, so old proofs from prior circuit versions become
unverifiable after a redeploy.

**Solution:** Deploy CircuitRegistry.sol as an immutable routing layer. AuditRegistry reads
the correct verifier address from the registry at proof-submission time.

**New contracts:**
```solidity
// contracts/src/CircuitRegistry.sol
interface ICircuitRegistry {
    function registerCircuit(string calldata version, address verifier) external onlyOwner;
    function getVerifier(string calldata version) external view returns (address);
}

// AuditRegistry.submitAudit() updated signature:
function submitAudit(
    bytes calldata proof,
    uint256[] calldata publicSignals,
    string calldata circuitVersion    // e.g. "v2.0"
) external returns (bytes32 auditId);
// Routes: ICircuitRegistry(registry).getVerifier(circuitVersion)
```

**Migration path:**
1. Deploy CircuitRegistry.sol and register current v2.0 verifier address.
2. Redeploy AuditRegistry with `address registry` constructor arg.
3. For each future CIRCUIT_VERSION bump: deploy new verifier, call `registerCircuit()`.
   Old proofs remain verifiable via their original circuit version.

**Benefit:** Zero re-deployment of AuditRegistry on circuit upgrades; full historical
verifiability; clean separation of proof routing from proof storage.

---

### ZK-5: Distillation Quality Gate

**Problem:** The proxy agreement threshold (≥95% per class) is documented but not enforced
in code. A proxy that subtly misagreess on imbalanced classes (e.g., DoS with only 377 samples)
can pass a raw-percentage check while being unreliable in practice.

**Solution:** Add a mandatory evaluation step at the end of `train_proxy.py` using Cohen's
kappa per class, which accounts for class imbalance and is more meaningful than raw agreement
percentage for low-prevalence classes.

**Implementation:**
```python
# zkml/src/distillation/train_proxy.py — add after training loop

from sklearn.metrics import cohen_kappa_score

def evaluate_agreement(proxy, full_model, val_loader, threshold=0.5):
    """
    Computes per-class Cohen's kappa between proxy and full model predictions.
    Raises ProxyQualityError if any class kappa < KAPPA_FLOOR.
    Logs results to MLflow experiment 'sentinel-zkml'.
    """
    KAPPA_FLOOR = 0.80   # corresponds roughly to ≥90% agreement on balanced data
    CLASS_NAMES = [      # must match NUM_CLASSES order
        "CallToUnknown", "DoS", "ExternalBug", "GasException", "IntegerUO",
        "MishandledException", "Reentrancy", "Timestamp", "TOD", "UnusedReturn"
    ]
    ...
    for i, name in enumerate(CLASS_NAMES):
        kappa = cohen_kappa_score(full_preds[:, i], proxy_preds[:, i])
        mlflow.log_metric(f"proxy_kappa_{name}", kappa)
        if kappa < KAPPA_FLOOR:
            raise ProxyQualityError(
                f"Proxy kappa for {name} = {kappa:.3f} < floor {KAPPA_FLOOR}. "
                f"Re-distill or inspect class imbalance before exporting ONNX."
            )

class ProxyQualityError(RuntimeError):
    pass
```

**MLflow integration:** Log to experiment `"sentinel-zkml"` (create if absent). Tags:
`source_checkpoint`, `circuit_version`, `distillation_epochs`, `kappa_floor`.

**Gate ordering:** `train_proxy.py` → `evaluate_agreement()` passes → `export_onnx.py` →
`generate_calibration.py` → EZKL pipeline. Export must be blocked if quality gate fails.
