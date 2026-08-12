# 07 — ZKML proof boundary

**Read this when:** you need to understand proxy distillation, retained EZKL proof semantics, or the V3 proof/attestation boundary.

**Skip this if:** you only need off-chain gateway reports; that path does not require chain submission.

**Estimated reading time:** 15 minutes.

## 30-second summary

The full teacher is not proved. The retained 10,666-parameter proxy maps the teacher’s 128-value fusion embedding through `128→64→32→10`, and EZKL proves that fixed proxy computation for 128 public inputs and ten public outputs. This proof remains **legacy proxy-only scope** and uses `check_mode="UNSAFE"`. V3 does not expand the circuit. Instead, V3 separately binds proof/output identities to target bytecode, agent, chain/registry, round, teacher/proxy/DATA/schema identities, and deadline through an EIP-712 policy attestation.

## Just-enough mental model

```text
Solidity → teacher → fusion[128] ─────────────── outside ZK circuit
                         ↓
                proxy 128→64→32→10
                         ↓
                     outputs[10]
                         ↓
                 EZKL proxy proof
                         +
            V3 EIP-712 policy attestation
                         ↓
               context-bound registry record
```

Two claims remain separate:

1. the proxy computation is valid;
2. an authorized policy signer attested the fully bound audit context.

Neither proves the teacher, Solidity source analysis, LangGraph execution, or final verdict.

## Actual runtime/source walkthrough

### Retained proxy/circuit

- [`proxy_model.py`](../../zkml/src/distillation/proxy_model.py) owns the frozen 128→64→32→10 proxy and circuit version.
- ONNX/settings/compiled/VK artifacts remain tracked.
- [`run_proof.py`](../../zkml/src/ezkl/run_proof.py) owns witness/prove/verify/parsing mechanics.
- current settings retain 138 public signals: 128 inputs followed by ten outputs.
- score fixed-point scale remains 13 (`8192`).

### Current V3 trust integration

The historical audit submitter is no longer the live analysis MCP write surface. V3 integration is split across:

- [`policy_signer.py`](../../agents/src/security/policy_signer.py): builds/validates a fully bound unsigned V3 request/digest; deliberately has no private key, transaction construction, broadcast, or receipt handling;
- [`AuditRegistry.sol`](../../contracts/src/AuditRegistry.sol): verifies V3 policy signature, request reuse, target code, stake, 138-signal layout, output equality, and the proxy proof before storing V3 context.

The V3 request binds:

- agent;
- contract address + runtime bytecode hash;
- chain ID + registry address;
- round ID;
- teacher-model hash;
- proxy-bundle hash;
- DATA-version hash;
- class-schema hash;
- proof hash;
- public-signals hash;
- ten-score hash;
- deadline.

### Future regeneration boundary

The retained proxy/circuit artifacts belong to the historical teacher/fusion distribution. R4 intentionally defers any production proxy redistillation/regeneration until a repaired DATA-vNext teacher is trained and selected. Do not regenerate ZKML merely because DATA semantics changed; regenerate when the promoted teacher/fusion behavior actually changes.

## Interfaces, data shapes, and configuration

| Boundary | Current meaning |
|---|---|
| teacher fusion | 128 floats |
| proxy | 128→64→32→10, 10,666 params |
| public signals | 128 inputs + 10 outputs = 138 |
| contract output offset | 128 |
| fixed-point scale | `2^13 = 8192` |
| proof scope | `legacy_proxy_only_unbound` by itself |
| V3 submission protocol | `context_attested_v3` |
| EZKL check mode | `UNSAFE` |

Tracked proof/circuit artifacts are reproducibility inputs, not automatic production-eligibility claims. Proving key/SRS remain operational prerequisites where live proving is attempted.

## Failure modes and current limitations

- A valid proof does not prove source compilation, teacher execution, DATA provenance, AGENTS routing, or verdict correctness.
- A valid V3 policy signature does not expand the ZK statement.
- `check_mode="UNSAFE"` remains a production-assurance blocker requiring explicit review.
- A production signer/broadcast service is not currently claimed.
- Historical `_submit.py` must not be treated as the live analysis MCP entry point.
- The retained proxy may need redistillation after repaired teacher retraining; old agreement evidence cannot automatically transfer to a new teacher.
- proving artifacts may be absent from a fresh clone.

## Common change recipe

For a repaired-teacher ZKML update:

1. select/promote the repaired teacher candidate with bound DATA/config lineage;
2. regenerate distillation corpus from the approved teacher/fusion seam;
3. retrain/evaluate proxy agreement;
4. export ONNX and regenerate calibration/settings/compiled circuit/keys as required;
5. prove/verify exact 138-signal behavior;
6. regenerate/test Solidity verifier;
7. compute and bind the new proxy-bundle identity used by V3;
8. rotate V3 verifier only through explicit contract governance/tests;
9. preserve old verifier/proxy artifacts as historical lineage.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
ml/.venv/bin/python -m pytest zkml/tests -q
cd contracts && forge test -q
cd .. && python3 docs/handbook/tools/verify_handbook.py static
```

Live proving/signing/broadcast requires separate explicit prerequisites and is not implied by these checks.

## Optional deep references

- [Contracts](08_contracts.md)
- [Security and trust](12_security_and_trust.md)
- [Runtime flows](02_runtime_flows.md)
- [`policy_signer.py`](../../agents/src/security/policy_signer.py)

## Technical mastery layer

### Prerequisite knowledge

Know distillation, finite-field encoding, public inputs/outputs, EIP-712, artifact hashes, and the distinction between proof scope and authenticated provenance.

### Source map and reading order

Read proxy model/settings/run_proof for the cryptographic computation, then `policy_signer.py` for V3 request binding and `AuditRegistry.submitAuditV3` for contract enforcement. Treat historical `_submit.py` as compatibility/history, not current live service architecture.

### Execution trace and worked example

A 128-value fusion vector and ten score fields satisfy the fixed proxy circuit and produce a proof. V3 hashes that proof, the 138 public signals, the ten scores, target runtime code, teacher/proxy/DATA/schema identities, round/deadline, agent, chain, and registry into an EIP-712 digest. The signature authenticates that context; the verifier independently verifies the proxy proof.

### Implementation practice

Keep proof validation and policy/provenance authorization separately testable. If one changes, do not silently reinterpret the other.

### Review and ownership check

Can you state exactly what the retained proof proves, what V3 attests, what remains outside both, and why a new teacher may require proxy/circuit regeneration even when dimensions stay 128→10?
