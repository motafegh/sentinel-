# 12 — Security, trust boundaries, and threat model

**Read this when:** you need to decide what evidence can be trusted, what is proved, or how hostile input/secrets/failures are handled.

**Skip this if:** never for production or security claims.

**Estimated reading time:** 14 minutes.

## 30-second summary

SENTINEL combines untrusted Solidity, learned predictions, external analyzers, retrieval, LLM reasoning, operator-supplied provenance, ZK proxy proofs, and owner-controlled contracts. Each has a different trust level. Prompt sanitization detects on original source, then strips comments, then frames the remaining code. Deterministic routing is isolated from LLM/source prompt influence. Rule 5C requires explicit failure status. The current proof and provenance do not establish the complete end-to-end audit claim.

## Just-enough mental model

```text
untrusted source/data → deterministic parsers/tools → learned evidence → optional RAG/LLM
                                ↓                         ↓
                         explicit tool status       untrusted explanation

operator chooses fusion/hash → proxy ZK proof → owner-upgradeable registry
        assertion                  narrow proof          chain persistence
```

Trust is not transitive: a valid on-chain proof cannot make an unproved upstream assertion true.

## Actual runtime/source walkthrough

### Prompt-injection defense

[`prompt_sanitize.py`](../../agents/src/security/prompt_sanitize.py) — `agents/src/security/prompt_sanitize.py::sanitize_for_prompt` executes in this exact order:

1. `detect_injections(source)` on the original source so attack evidence remains visible;
2. `strip_comments(source)` with line preservation;
3. `delimit_contract_source(stripped)` with an explicit “data, not instructions” boundary.

[`injection_detect.py`](../../agents/src/security/injection_detect.py) emits actual pattern names: `comment`, `string`, `role-swap`, `extraction`, `identifier`, `NatSpec`, `multi`, and `import`. Detection is a log/report canary, not a proof that no adversarial instruction remains.

[`routing.py`](../../agents/src/orchestration/routing.py) and [`evidence_router.py`](../../agents/src/orchestration/nodes/evidence_router.py) are protected by tests that forbid LLM imports and source-dependent prompt routing. This prevents hostile text from choosing which security tools run.

### Rule 5C

External tools/services must either raise a precise error or return a structured degraded result. `AuditState.tool_status` records `{ran, reason, detail...}`. `ran=false` must never be collapsed into `[]`, because that would make tool absence look like a clean scan and contaminate reliability fitting.

### ZK and provenance boundary

The EZKL circuit proves the fixed proxy computation over public fusion inputs and public class outputs with fixed parameters. It does not prove source compilation, teacher execution, evidence fusion, or either AGENTS verdict. Provenance binds hashes/embedding/scores as an operator statement; it may be unsigned and does not prove teacher authorship.

## Interfaces, data shapes, and configuration

Threats and controls:

| Threat | Current control | Residual risk |
|---|---|---|
| source prompt injection | detect → strip → delimit; routing isolation | code strings/semantics and novel attacks; LLM remains nondeterministic |
| missing analyzer/service | structured status/error | legacy/debug-only catches may still hide monitoring failures |
| poisoned RAG/data | provenance, dedup, metadata filters, review | source/license/content trust and embedding contamination |
| artifact substitution | schema gates, artifact/checkpoint hashes | hash propagation and artifact acquisition trust |
| proof forgery | EZKL verification and on-chain verifier | unsafe settings review, upstream computation outside circuit |
| malicious operator | staking, proof constraints, stored identity | operator chooses upstream inputs/hash; MVP owner centralization |
| secret leakage | ignored env files, named env registry | source/scripts may contain historical hard-coded endpoints; do not repeat them |
| upgrade compromise | owner-only UUPS authorization | key compromise/storage-layout error/centralized governance |

Secret names may be documented (`SENTINEL_OPERATOR_KEY`, RPC URL, registry address); values must never be copied from `.env`, shell history, deployment state, or source literals into Markdown or CI logs.

## Failure modes and current limitations

- Sanitization reduces prompt influence but is not a sandbox.
- RAG/LLM evidence belongs only in `verdict_full`; deterministic evidence belongs in `verdict_provable`.
- “Provable” is currently a tier name, not proof coverage.
- `UNSAFE` EZKL check mode requires an explicit security review against the pinned bindings and official guidance.
- Provenance can be unsigned in degraded operation.
- Owner controls slashing, pausing, and upgrades.
- Gateway reports and on-chain submission are disconnected, creating two separate audit identities unless an operator correlates them.

## Common change recipe

For a new external tool or input:

1. Define the adversary and the trusted output boundary.
2. Add bounds, timeouts, provenance, and explicit failure schema.
3. Keep routing/policy deterministic and separate from untrusted text.
4. Add adversarial fixtures and a “tool did not run” test.
5. Decide whether evidence enters deterministic fusion, full fusion, or explanation only.
6. Update threat model, reliability evaluation, and operations prerequisites.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
cd agents
poetry run pytest tests/security -q
poetry run pytest -q -k 'routing_isolation or adversarial or tool_status'
cd ..
python3 docs/handbook/tools/verify_handbook.py static
```

Use `rg --files agents/tests | rg 'security|injection|routing'` if test paths evolve. Current totals are in [current status](16_current_status.md).

## Optional deep references

- [`docs/learning/03_prompt_injection_defense.md`](../learning/03_prompt_injection_defense.md)
- [`docs/learning/04_reproducibility_determinism.md`](../learning/04_reproducibility_determinism.md)
- [EZKL security guidance](https://docs.ezkl.xyz/security/)
- [Current status](16_current_status.md)

## Technical mastery layer

### Prerequisite knowledge

Know prompt injection, least privilege, failure semantics, provenance, ZK statement scope, and secret handling.

### Source map and reading order

Read injection detection, comment stripping, delimiter framing, sanitizer orchestration, routing-isolation tests, Rule 5C evidence paths, reliability fitting, then ZK/provenance boundaries. See [T09](technical/09_security_evaluation_trust.md).

### Execution trace and worked example

Original Solidity is checked for comment/string/role-swap/extraction/identifier/NatSpec/multi/import patterns, then comments are removed and source is framed. Tool timeout records unavailable status and emits no negative evidence. Proxy proof and unsigned provenance remain separate trust claims.

### Implementation practice

[L09](labs/09_injection_rule5c_reliability.md) adds positive/benign fixtures and a failure case. Never copy `.env`, RPC credentials, mnemonics, or keys into a trace.

### Review and ownership check

Can you state who/what must be trusted at every boundary and identify claims that remain unproved even after valid on-chain verification?
