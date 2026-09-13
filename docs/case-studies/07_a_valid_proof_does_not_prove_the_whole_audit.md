# Case Study 07 — A valid proof does not prove the whole audit

## 1. Decision in one sentence

SENTINEL keeps the EZKL proxy-proof statement, V3 context/provenance attestation, and transaction authority as separate trust boundaries instead of presenting one valid proof as proof of the entire audit pipeline.

## 2. Problem

The retained ZKML path verifies a compact neural proxy derived from the teacher fusion representation. It is tempting to describe a successful proof as proving the source audit, model execution, agent reasoning, and on-chain submission context.

That claim would be substantially stronger than the circuit actually establishes.

The system therefore needed an explicit way to bind surrounding audit context without pretending that the circuit proves facts it never receives as public inputs.

## 3. Evidence

The retained circuit proves the fixed 128→10 proxy computation:

- 128 fusion inputs;
- ten proxy outputs;
- 138 public signals in total.

It does not prove Solidity source semantics, teacher execution, LangGraph routing, DATA provenance, chain identity, target identity, audit round, or the final AGENTS verdict.

The retained bundle also records `check_mode="UNSAFE"`, which remains an explicit production-assurance limitation.

V3 therefore adds a separate EIP-712 policy/context attestation. The signed request binds identities such as the submitting agent, target runtime code hash, chain and registry, round, teacher hash, proxy-bundle hash, DATA-version hash, class-schema hash, proof/public-signal/score hashes, and expiry deadline.

## 4. Shortcut rejected

The project rejected three overclaims:

1. “valid EZKL proof” means the complete security audit was proved;
2. a policy signature expands the mathematical statement of the ZK circuit;
3. the analysis runtime should directly hold signing keys or broadcast registry transactions simply because V3 supports authenticated submission.

## 5. Decision

The trust model is intentionally split:

1. the EZKL verifier establishes the proxy computation;
2. the V3 EIP-712 signature authenticates the surrounding request/context;
3. transaction signing/broadcast remains an external authority domain.

The live audit MCP is read-only, and `policy_signer.py` owns request/digest semantics without private-key custody, transaction construction, broadcast, or receipt handling.

## 6. Implementation and validation

`AuditRegistry.submitAuditV3` verifies the V3 activation state, target runtime code, deadline, stake requirement, exact public-signal shape, score/output consistency, replay protection, policy-signer recovery, and proxy proof verification.

Historical V1/V2 reads remain available while new legacy writes are disabled after V3 activation.

The registry stores the bound context/provenance identities and signed digest, but neither the stored record nor the signature is reinterpreted as vulnerability ground truth.

## 7. Result

SENTINEL can make a precise statement about what each mechanism contributes:

- cryptographic proof: proxy computation integrity;
- policy attestation: authenticated context/provenance binding;
- registry: versioned persistence and replay-controlled submission;
- external signer/broadcaster: transaction authority.

This is weaker than claiming end-to-end proof of the audit, but it is technically defensible.

## 8. Remaining limitation

There is no claimed production signer/broadcaster today, and the retained EZKL bundle remains proxy-only with `UNSAFE` check mode.

A future repaired teacher would also require fresh proxy distillation, agreement measurement, circuit/verifier regeneration as needed, and new V3 identity binding. Existing proof evidence cannot automatically transfer to a new teacher lineage.

## 9. Evidence trail

Primary references:

- `zkml/README.md`;
- `contracts/README.md`;
- `contracts/src/AuditRegistry.sol`;
- `agents/src/security/policy_signer.py`;
- canonical ZKML, contracts, and security/trust handbook chapters.

Executable contracts/tests and retained proof artifacts remain higher authority than this case study.
