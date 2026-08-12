# 12 — Security, trust boundaries, and threat model

**Read this when:** you need to decide what evidence can be trusted, what is proved, or how hostile input, label uncertainty, signing authority, secrets, and failures are handled.

**Skip this if:** never for production/security claims.

**Estimated reading time:** 14 minutes.

## 30-second summary

SENTINEL combines untrusted Solidity, uncertain historical labels, learned predictions, external analyzers, retrieval/LLM evidence, a retained proxy ZK proof, an isolated V3 policy-attestation boundary, and owner-upgradeable contracts. None of those trust domains automatically upgrades another. R4 specifically repaired the DATA trust mistake where unknown/unsupported historical states became numeric negatives. V3 specifically repairs context binding around the proxy proof without pretending that the circuit proves teacher/source/AGENTS execution.

## Just-enough mental model

```text
untrusted source/data
   ↓
deterministic preprocessing / explicit evidence state
   ↓
learned + tool evidence → off-chain AGENTS report

separate chain trust path:
proxy proof (narrow computation)
   +
policy signature (authenticated context/provenance)
   +
transaction authority / stake
   ↓
V3 registry record
```

Trust is not transitive. A valid proof cannot make an unproved upstream claim true; a valid signature cannot expand the ZK statement; a model score cannot create ground truth; an unknown label cannot become negative merely because a tensor requires `0`.

## Actual runtime/source walkthrough

### Prompt-injection defense

[`prompt_sanitize.py`](../../agents/src/security/prompt_sanitize.py) detects on original source, strips comments with line preservation, then frames the source as data. Current detection categories include `comment`, `string`, `role-swap`, `extraction`, `identifier`, `NatSpec`, `multi`, and `import`.

Routing/policy must remain deterministic and independent from hostile source instructions. Detection is a defense signal, not proof that all prompt injection is impossible.

### Rule 5C / explicit failure

External tools/services must return precise errors or structured degraded status. `ran=false` must never become an empty finding list that looks clean. This rule applies equally to analyzers, feedback observation, DATA evidence, and production-readiness claims.

### DATA/ML trust boundary after R4

The current DATA policy distinguishes:

- source-native assertion;
- canonical outcome state;
- training signal/strength;
- role/metric eligibility;
- provenance/evidence.

Historical zero, source absence, unsupported class, dropped/out-of-taxonomy mapping, or tool silence does not establish a negative. The first repaired baseline has no blanket confirmed-negative source. GasException and UnusedReturn remain supervision-disabled; threshold/calibration/untouched-acceptance roles are empty/unsupported.

This means a future trainer must fail closed on missing vNext semantics rather than force partial knowledge back into old binary labels.

### ZK/V3 trust boundary

The retained EZKL proof verifies the fixed 128→10 proxy computation. V3 adds EIP-712 context/provenance authentication around that proof. The V3 digest binds agent, target/code hash, chain/registry, round/deadline, teacher/proxy/DATA/schema identities, proof/public-signal/score hashes.

[`policy_signer.py`](../../agents/src/security/policy_signer.py) contains no private key, signing, transaction construction, broadcast, or receipt logic. Signing belongs to an isolated service/domain. The live audit MCP is read-only.

### Contract/governance boundary

V3 verifies stake, target code, deadline, anti-replay digest, configured signer, proof, and output equality. Owner controls pause/unpause, policy-signer/verifier rotation, and UUPS upgrades. Those are explicit centralized trust points.

### Feedback trust boundary

Current V3 feedback observation does not automatically promote chain events into RAG/DATA truth. V3 policy is intentionally unavailable; observations can remain durable pending. Historical scalar feedback thresholds must not be reused as V3 truth policy without measured evidence.

## Interfaces, data shapes, and configuration

| Threat | Current control | Residual risk |
|---|---|---|
| prompt injection | detect → strip → delimit; routing isolation | novel attacks / LLM nondeterminism |
| missing analyzer/service | structured status/error | legacy/debug catches may still need review |
| historical label corruption | R4 ledger/state policy/masks/roles | limited positive/negative evidence remains |
| poisoned RAG/feedback | provenance, dedup, policy/review, pending V3 journal | source/content trust |
| artifact substitution | content hashes, schema/policy/partition bindings | artifact acquisition/operational handling |
| proof forgery | EZKL verification/on-chain verifier | retained `UNSAFE` check mode; upstream outside circuit |
| false provenance | V3 EIP-712 context binding | trust in authorized policy signer and its policy |
| compromised signer | isolated signer domain + on-chain signer rotation | signer/KMS/HSM governance not yet productionized |
| malicious operator | stake + proof/context checks | transaction authority and upstream artifact selection |
| upgrade compromise | owner-only UUPS controls/tests | owner-key/storage/governance risk |
| secret leakage | ignored env/secrets + docs policy | operational leakage outside repo controls |

## Failure modes and current limitations

- Sanitization reduces prompt influence but is not a sandbox.
- Learned/LLM/RAG outputs remain evidence, not ground truth.
- `verdict_provable` terminology does not mean the AGENTS verdict is inside the EZKL circuit.
- `check_mode="UNSAFE"` remains a production-assurance blocker.
- V3 context attestation is only as trustworthy as the authorized policy signer/policy.
- No production signer/broadcast service is claimed today.
- Owner controls upgrades/trust-root rotation.
- First repaired DATA baseline lacks trustworthy confirmed-negative, threshold/calibration, and untouched-acceptance support.
- Run12 remains trained on historical semantics until repaired retraining occurs.

## Common change recipe

For a new evidence/source/tool/signing capability:

1. define the adversary and trusted output boundary;
2. specify what evidence state it can establish and what it cannot;
3. add bounds/timeouts/provenance/failure status;
4. keep untrusted text out of deterministic routing/policy;
5. decide whether evidence may affect training, model selection, final verdict, feedback mutation, or only explanation;
6. for signing, isolate keys and separate policy authorization from transaction broadcast;
7. add adversarial/replay/expiry/missing-evidence tests;
8. update R4/ADR/trust/current-status docs before increasing claims.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
cd agents
poetry run pytest tests/security -q
poetry run pytest -q -k 'routing_isolation or adversarial or policy_signer or feedback'
cd ..
cd contracts && forge test
cd .. && python3 docs/handbook/tools/verify_handbook.py static
```

## Optional deep references

- [Runtime flows](02_runtime_flows.md)
- [ZKML](07_zkml.md)
- [Contracts](08_contracts.md)
- [Evaluation](13_evaluation.md)
- [Current status](16_current_status.md)

## Technical mastery layer

### Prerequisite knowledge

Know prompt injection, least privilege, partial-label/unknown semantics, provenance, ZK statement scope, EIP-712, signing trust, failure semantics, and UUPS governance.

### Source map and reading order

Read injection/sanitization/routing isolation, then R4 policy/role artifacts, proxy/settings, `policy_signer.py`, `AuditRegistry.submitAuditV3`, read-only audit handlers, and V3 feedback observation/policy/runtime.

### Execution trace and worked example

A DIVE DoS historical positive can be masked instead of trained as truth; a missing analyzer is recorded as unavailable rather than clean; a proxy proof verifies only 128→10 computation; a V3 signature binds that proof/context; a chain observation remains pending feedback unless policy explicitly authorizes mutation.

### Implementation practice

When a component cannot establish a claim, encode that limitation as data/status—not as a comment. Unknown labels, unavailable tools, absent signer, missing calibration role, and untrusted feedback should all fail closed in machine-readable state.

### Review and ownership check

Can you state exactly what DATA evidence, model inference, analyzer output, ZK proof, V3 policy signature, transaction authority, registry storage, and feedback observation each establish—and what remains unproved?
