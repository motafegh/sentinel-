# SENTINEL ZKML

`zkml/` is the proof layer between the ML teacher and the on-chain audit registry.
It does **not** prove the full Sentinel teacher model. The tracked V2 circuit proves
a small student/proxy computation over a public 128-dimensional fusion embedding.

The current trust statement is intentionally narrow:

```text
Solidity source
    ↓
ML teacher (off-chain)
    ↓  /fusion-embedding
128 public fusion values
    ↓
ProxyModel 128 → 64 → 32 → 10
    ↓
10 public student scores
    ↓
EZKL proof
```

The proof establishes the proxy computation for the supplied public inputs. It
does not by itself prove that the fusion values came from a particular Solidity
contract, teacher checkpoint, chain, registry, or audit round. The V3 registry
protocol handles that context separately with a policy-signed EIP-712 request.

## Current executable contract

### ML boundary

`ml/src/inference/api.py::/fusion-embedding` returns:

- exactly 128 fusion values;
- the live teacher checkpoint SHA-256;
- graph/window metadata;
- structured execution status.

### Proxy model

`zkml/src/distillation/proxy_model.py` is frozen at:

```text
Linear(128 → 64) → ReLU → Linear(64 → 32) → ReLU → Linear(32 → 10)
```

Exact parameter count: **10,666**.

Circuit version: `v2.0`.

The student is trained by MSE directly against
`sigmoid(teacher_logits)`. Therefore `ProxyModel.forward()` is the canonical
student **score** vector for the current artifact semantics. Consumers must not
apply an additional sigmoid.

## Public-signal layout

The tracked V2 EZKL bundle uses public inputs and public outputs:

```text
indices   0..127  fusion inputs
indices 128..137  ten proxy output field elements
----------------
138 public signals total
```

Field elements in EZKL JSON artifacts are encoded as 32-byte little-endian hex.
Python decoding must use:

```python
int.from_bytes(bytes.fromhex(value), byteorder="little")
```

Do not decode the string as a normal big-endian integer.

## Distillation and DATA lineage

Future proxy training has **no implicit DATA export**. This is deliberate.
R4 is repairing DATA/label reality, so a new proxy must not silently train on an
old export.

```bash
python zkml/src/distillation/train_proxy.py \
  --export-dir <EXPLICIT_PROMOTED_DATA_EXPORT>
```

The saved checkpoint metadata binds, among other fields:

- teacher checkpoint SHA-256;
- DATA export manifest SHA-256;
- circuit version;
- output semantics;
- random seed;
- measured teacher/student agreement.

Calibration follows the same rule: the export must be selected explicitly.
Do not retrain/regenerate the production candidate until R4 has promoted the
intended DATA/ML lineage.

## Artifact chain

The intended regeneration chain is:

```text
promoted DATA export
      ↓
teacher checkpoint
      ↓
train_proxy.py
      ↓
zkml/models/proxy_best.pt
      ↓
export_onnx.py
      ↓
zkml/models/proxy.onnx (+ manifest)
      ↓
generate_calibration.py
      ↓
calibration data (+ lineage)
      ↓
setup_circuit.py
      ↓
settings / compiled circuit / SRS / PK / VK
      ↓
generated contracts/src/ZKMLVerifier.sol
      ↓
run_proof.py
      ↓
proof.json + 138 public signals
```

`zkml/src/validation/validate_bundle.py` checks the tracked bundle's structural,
identity, dimension, visibility, and protocol invariants. That validation is an
integrity check; it is not a replacement for cryptographic proof verification.

## Tracked V2 bundle

The repository currently retains a historical V2 proof bundle including the
proxy, ONNX artifacts, calibration/settings, compiled circuit, verification key,
proof fixture, and canonical generated Solidity verifier.

That bundle is useful for reproducibility and regression testing. It is **not an
eligible runtime audit submission protocol** because its proof scope is
`legacy_proxy_only_unbound`.

Remote CI exercises the tracked proof against the actual generated
`Halo2Verifier`. A mutation of a public output must fail closed (the generated
verifier may return `false` or revert).

## Per-proof helper

`zkml/src/ezkl/run_proof.py` is a deterministic proof-generation helper. It no
longer searches for an easier contract to prove and it does not claim that a V2
proof is eligible for chain finality.

`zkml/src/ezkl/extract_calldata.py` is intentionally **read-only**. It may decode
and inspect the historical V2 proof, but it does not emit `cast send`, accept a
private key, or generate a direct `submitAuditV2` write path. Runtime signing is
a separate security boundary.

## V3 relationship

The V3 registry protocol does not pretend the V2 neural proof suddenly proves
more than it does. It uses two independent checks:

1. the EZKL verifier validates the exact proxy proof and 138 public signals;
2. a dedicated policy signer authenticates the audit context with EIP-712.

The signed context binds the submitting agent, target address and runtime
bytecode hash, chain/registry domain, round, teacher model identity, proxy bundle
identity, DATA identity, class-schema identity, proof hash, public-signal hash,
score hash, and expiry.

See `agents/src/security/policy_signer.py` and
`contracts/src/AuditRegistry.sol` for executable V3 semantics.

## Testing

Dependency-light ZKML tests:

```bash
pytest -q zkml/tests
```

The dedicated root workflow `.github/workflows/system-alignment.yml` additionally
validates the tracked bundle and exercises the canonical generated Solidity
verifier with the tracked proof.

Local full proof regeneration still requires the ML/data environment and the
cryptographic setup artifacts that are intentionally not all reproduced in a
plain remote checkout.

## Important invariants

- Fusion input dimension is 128.
- Vulnerability class count is 10.
- V2 public-signal count is exactly 138.
- Proxy architecture changes require a circuit-version bump and complete artifact regeneration.
- Proxy output semantics are direct teacher-probability regression scores; no second sigmoid.
- DATA export selection for retraining/calibration must be explicit.
- Generated verifier identity must be tied to the exact circuit/VK bundle.
- Legacy V2 proof scope remains ineligible for runtime finality.
- No ZKML helper may bypass the isolated policy-signer boundary with a raw private key.

## Current status

The code/protocol alignment branch validates the historical V2 proof boundary and
implements the context-attested V3 registry protocol. New proxy/circuit artifacts
should be regenerated only after R4 promotes the appropriate DATA/ML candidate;
that future bundle must receive fresh lineage, proof, verifier, and integration
validation before deployment.
