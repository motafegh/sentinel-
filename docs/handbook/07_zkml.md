# 07 — ZKML proof boundary

**Read this when:** you need to distill the teacher, regenerate EZKL artifacts, interpret a proof, or submit class scores on-chain.

**Skip this if:** you only need off-chain gateway reports; that path does not invoke ZK submission.

**Estimated reading time:** 15 minutes.

## 30-second summary

The full teacher is not proved. A frozen 10,666-parameter proxy maps the teacher’s 128-value fusion embedding through `128→64→32→10`. EZKL proves that this proxy circuit produced the ten public class outputs for the 128 public inputs. It does not prove raw Solidity was analyzed, the teacher made the embedding, the agent verdict is correct, or `verdict_provable` was computed.

## Just-enough mental model

```text
Solidity → teacher → fusion[128] ───────── operator assertion
                         ↓
             proxy 128→64→32→10
                         ↓
                   outputs[10]
                         ╰── EZKL proves this computation only
```

The proof statement is precise: “for these public inputs and fixed circuit parameters, these public outputs satisfy the compiled proxy computation.” Everything before the fusion vector is outside the circuit.

## Actual runtime/source walkthrough

1. [`corpus_distill.py`](../../zkml/src/distillation/corpus_distill.py) or [`train_proxy.py`](../../zkml/src/distillation/train_proxy.py) produces teacher embeddings and trains the student for agreement.
2. [`proxy_model.py`](../../zkml/src/distillation/proxy_model.py) — `zkml/src/distillation/proxy_model.py::ProxyModel` freezes dimensions and enforces the parameter ceiling; `::CIRCUIT_VERSION` is `v2.0`.
3. [`export_onnx.py`](../../zkml/src/distillation/export_onnx.py) exports raw proxy logits to ONNX.
4. [`generate_calibration.py`](../../zkml/src/distillation/generate_calibration.py) creates representative calibration inputs.
5. [`setup_circuit.py`](../../zkml/src/ezkl/setup_circuit.py) performs settings generation, calibration, compilation, SRS acquisition, and setup to create proving/verification keys.
6. [`run_proof.py`](../../zkml/src/ezkl/run_proof.py) generates a witness, proves, verifies off-chain, and parses little-endian field elements.
7. [`_submit.py`](../../agents/src/mcp/servers/audit/_submit.py) — `agents/src/mcp/servers/audit/_submit.py::_run_submit` repeats the live proof lifecycle for a supplied contract and calls `AuditRegistry.submitAuditV2`.

The current [`settings.json`](../../zkml/ezkl/settings.json) records EZKL 23.0.5, `input_visibility="Public"`, `output_visibility="Public"`, `param_visibility="Fixed"`, scale 13, logrows 15, shapes `[1,128]` and `[1,10]`, and `check_mode="UNSAFE"`. Public signals are the 128 inputs followed by ten outputs: 138 total.

## Interfaces, data shapes, and configuration

| Boundary | Shape/meaning |
|---|---|
| teacher fusion | 128 floats |
| proxy output | 10 raw logits; sigmoid applied outside the PyTorch network |
| EZKL instances | 138 fixed-point field elements |
| contract offset | outputs begin at `INPUT_OFFSET=128` |
| fixed-point scale | `2^13 = 8192` |

Artifact ownership:

- tracked/public: proxy checkpoint, ONNX, settings, compiled circuit, verification key, generated verifier;
- regenerated: calibration/witness/proof inputs and proof outputs;
- ignored/private operational prerequisite: proving key;
- ignored/downloaded public prerequisite: `srs.params`.

The current repository also tracks sample proof/witness JSON. They are examples or mutable shared runtime paths, not per-request durable evidence.

## Failure modes and current limitations

- `check_mode="UNSAFE"` is not a production assurance setting. Review the exact guarantees and limitations against the [EZKL 23.0.5 Python bindings](https://pythonbindings.ezkl.xyz/en/stable/) and [official EZKL security guidance](https://docs.ezkl.xyz/security/) before relying on the proof in an adversarial setting.
- The provenance manifest is an off-chain operator assertion. It may be unsigned when the key/dependency is absent and does not prove the teacher generated the embedding.
- `verdict_provable` belongs to AGENTS evidence fusion and is outside this circuit.
- The submission path reports SHA-256 of proof bytes, while the contract stores `keccak256(proof)`.
- `_run_submit` reports the ML-returned model hash but passes the original function argument to the transaction, so callers must provide the correct hash.
- Shared `proof_input.json`, `witness.json`, and `proof.json` make concurrent calls unsafe.
- The proving key and SRS are absent from a fresh clone.

## Common change recipe

For any proxy architecture/weight or EZKL setting change:

1. Decide whether only weights changed or the circuit definition changed; bump `CIRCUIT_VERSION` for structural changes.
2. Rebuild distillation evidence against the approved teacher.
3. Export ONNX and regenerate calibration.
4. Regenerate settings, compiled circuit, SRS compatibility, proving key, and verification key with one pinned EZKL version.
5. Run witness/prove/off-chain verify and inspect all 138 instances.
6. Generate a new Solidity verifier and test gas/correctness.
7. Deploy/swap verifier through the controlled contract path.
8. Update artifact hashes, provenance expectations, metadata, and live evidence.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest zkml/tests -q                    # module
ml/.venv/bin/python -m zkml.src.distillation.proxy_model       # smoke
python3 docs/handbook/tools/verify_handbook.py static           # shape/settings contract
ml/.venv/bin/python -m zkml.src.ezkl.run_proof                 # live; needs proving artifacts
```

Current suite results are in [current status](16_current_status.md).

## Optional deep references

- [`setup_circuit.py`](../../zkml/src/ezkl/setup_circuit.py) — `::setup_circuit`
- [`run_proof.py`](../../zkml/src/ezkl/run_proof.py) — proof parsing and verification
- [Contracts](08_contracts.md)
- [Security and trust](12_security_and_trust.md)

## Technical mastery layer

### Prerequisite knowledge

Know distillation, ONNX, finite-field encoding, witnesses/keys, public inputs, and Solidity calldata.

### Source map and reading order

Read `proxy_model.py::ProxyModel`, distillation/export scripts, EZKL setup/settings, `run_proof.py::generate_proof`, calldata extraction, audit MCP `_run_submit`, and registry verifier boundary. See [T05](technical/05_zkml_proof_lifecycle.md).

### Execution trace and worked example

The teacher supplies 128 public proxy inputs; the fixed proxy produces ten public outputs at positions 128–137. Witness hex is little-endian and scores scale by 8192. Proof validity covers that fixed computation—not AGENTS `verdict_provable`, thresholding, or teacher provenance.

### Implementation practice

[L05](labs/05_zkml_witness_signals.md) checks architecture/endian/layout without pretending to run a live proof. Any architecture/settings/version change requires a complete circuit/key/verifier migration.

### Review and ownership check

Can you name all 138 signals, required private artifacts, and the production-review implication of `check_mode="UNSAFE"`?
