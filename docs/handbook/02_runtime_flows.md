# 02 — Runtime flows

**Read this when:** you need to distinguish off-chain audit execution, registry observation, V3 submission protocol, or feedback ingestion.

**Skip this if:** never when operating the system; confusing these flows creates false production claims.

**Estimated reading time:** 12 minutes.

## 30-second summary

SENTINEL currently has four distinct runtime/security flows. The gateway runs an asynchronous off-chain LangGraph audit and stores a report. The audit MCP is a read-only V1/V2/V3 registry observer. V3 defines the current context-attested on-chain submission protocol, but signing/broadcast is intentionally outside the analysis MCP and no production submitter is claimed. Feedback observation is separate and V3 automatic promotion remains disabled because no measured policy exists yet.

## Just-enough mental model

```text
Flow A — analyze
client → gateway :8000 → 14-node LangGraph → SQLite report

Flow B — observe chain
client/tool → audit MCP :8012 → read V1/V2/V3 registry history

Flow C — V3 protocol boundary
proxy proof + fully bound request → isolated policy attestation → submitAuditV3 → registry
(signing/broadcast service is outside the analysis MCP and not claimed production-ready)

Flow D — learn/feedback
registry/report observation → versioned feedback observation → policy check
V3 policy unavailable → durable pending journal → NO automatic RAG/data promotion
```

Flow A does not invoke Flow C. Flow B cannot create Flow C transactions.

## Actual runtime/source walkthrough

### A. Off-chain gateway audit

1. `POST /audit` validates source/address/metadata.
2. [`gateway.py`](../../agents/src/api/gateway.py) creates a queued SQLite job and schedules graph execution.
3. [`graph.py`](../../agents/src/orchestration/graph.py) runs the 14-node audit graph.
4. ML/tool/RAG evidence is accumulated with explicit tool status.
5. The synthesizer creates the final report; any on-chain field begins unsubmitted.
6. The gateway persists completed/failed status.
7. The client polls `GET /audit/{job_id}`.

No step signs or broadcasts a registry transaction.

### B. Read-only registry observation

The live audit MCP server in [`audit/_server.py`](../../agents/src/mcp/servers/audit/_server.py) imports [`_readonly_handlers.py`](../../agents/src/mcp/servers/audit/_readonly_handlers.py).

It exposes exactly:

- `get_latest_audit`
- `get_audit_history`
- `check_audit_exists`

These operations are version-aware across V1, V2, and V3 storage. Runtime submission names are rejected before the historical mutable submission module is imported.

### C. V3 context-attested submission protocol

The contract protocol exists in [`AuditRegistry.sol`](../../contracts/src/AuditRegistry.sol):

1. obtain the retained 128→10 proxy proof and 138 public signals;
2. construct a V3 context containing target, round, teacher hash, proxy-bundle hash, DATA-version hash, class-schema hash, deadline;
3. bind target runtime bytecode hash, proof hash, public-signal hash, class-score hash, agent, chain ID, and registry address into the EIP-712 request digest;
4. an isolated policy signer may attest an eligible request;
5. a transaction authority outside the analysis MCP may call `submitAuditV3`;
6. the contract verifies stake, code/deadline/layout, signature reuse, policy signer, proof, and output equality before storing the V3 result.

[`policy_signer.py`](../../agents/src/security/policy_signer.py) can build/validate the unsigned request/digest but intentionally contains no private key, transaction construction, broadcast, or receipt handling.

The V3 signature adds authenticated context/provenance. It does **not** expand the EZKL circuit statement.

### D. Feedback/learning boundary

The V3 runtime observation path records versioned submission/finality truth separately from the historical scalar feedback loop. Current V3 policy state is intentionally unavailable: V3 events may be journaled as pending, but they do not automatically promote findings into RAG or DATA/ML ground truth.

Historical V1 feedback behavior is retained for compatibility and must not be generalized to V3 by reusing old scalar thresholds.

## Interfaces, data shapes, and configuration

| Flow | Entry | Durable output | Current write authority |
|---|---|---|---|
| off-chain audit | gateway `POST /audit` | SQLite job + report | gateway local state only |
| registry observation | audit MCP read tools | read response | none |
| V3 protocol | `AuditRegistry.submitAuditV3` | V3 registry record/event | external transaction authority + policy signer required |
| V3 feedback | versioned event observation | pending/observation journal | no automatic RAG promotion |

V3 storage includes proof/request/public-signal/bytecode/model/proxy/DATA/schema identities, round, agent, signer, verifier, timestamp, and ten score field elements.

## Failure modes and current limitations

- Gateway completion is not proof generation, policy signing, or chain submission.
- The audit MCP is not a signer/broadcaster and must remain read-only.
- Historical `_submit.py` exists but is not the live analysis-service write path.
- V3 submission plumbing does not imply a production signing/broadcast service exists.
- The retained proxy proof is `legacy_proxy_only_unbound` by itself; V3 context is a separate attestation layer.
- `check_mode="UNSAFE"` remains a proof-system review blocker for production assurance.
- V3 feedback policy/version is intentionally unavailable, so automatic promotion is disabled.

## Common change recipe

To add a real V3 submission service safely:

1. keep it outside the analysis MCP trust domain;
2. define signer/KMS/HSM and transaction-authority boundaries explicitly;
3. consume the exact V3 request/digest schema from policy-signer code;
4. use per-request proof workspaces and deterministic artifact hashes;
5. implement idempotency, retries, confirmations/reorg handling, receipt persistence, and signer/verifier rotation behavior;
6. add integration tests against `submitAuditV3` and read-back through the audit MCP;
7. update security/status docs before calling it production-ready.

## Verification commands

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8012/health
python3 docs/handbook/tools/verify_handbook.py static
cd agents && poetry run pytest -q -k 'audit and (v3 or readonly or submission)'
```

Contract-level V3 verification uses the Foundry V3 tests; it is separate from live-service availability.

## Optional deep references

- [Architecture](01_architecture.md)
- [AGENTS services](10_agents_services.md)
- [ZKML](07_zkml.md)
- [Contracts](08_contracts.md)
- [Security and trust](12_security_and_trust.md)

## Technical mastery layer

### Prerequisite knowledge

Know async jobs, MCP tool exposure, EIP-712 request binding, proxy proofs, Ethereum transactions, and event-driven feedback.

### Source map and reading order

Off-chain: `gateway.py::create_app` → graph. Registry reads: `audit/_server.py::run_server` → `_readonly_handlers.py` → `_versioned_reads.py`. V3 protocol: `policy_signer.py` → `AuditRegistry.computeAuditDigestV3` / `submitAuditV3`. Feedback: versioned observation/policy/runtime modules under `agents/src/ingestion`.

### Execution trace and worked example

A `Vault.sol` gateway audit ends as a stored off-chain report. Independently, a fully bound V3 request could be policy-attested and submitted by an external transaction service; after inclusion, the read-only audit MCP can return the V3 record. If feedback observes that event today, V3 promotion remains pending because no measured V3 feedback policy has been authorized.

### Implementation practice

Keep analysis, signing, broadcasting, chain observation, and learning-policy mutation as separate trust domains. Never reintroduce transaction signing by importing historical `_submit.py` into the live audit MCP.

### Review and ownership check

Can you identify the first durable record, write authority, and trust claim for all four flows—and explain why none of them implies the others completed?
