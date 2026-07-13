# T10 — End-to-end implementation and debugging trace

## Learning outcome

You can trace one Solidity contract across DATA, ML, AGENTS, proxy, and registry boundaries, then localize failures by the first violated contract rather than guessing across modules.

## Prerequisites

Read [Architecture](../01_architecture.md), [Runtime flows](../02_runtime_flows.md), and [Cross-module contracts](../11_cross_module_contracts.md). Use the preceding technical guides for unfamiliar subsystems.

## Source map and reading order

1. Offline seam: DATA `represent_source` → export manifest → `SentinelDataset`.
2. Online off-chain seam: gateway `submit_audit` → `_run_job` → graph `build_graph` → ML `/predict` → report.
3. Direct-chain seam: ML `/fusion-embedding` → audit MCP `_run_submit` → proxy/EZKL → registry V2.
4. Feedback seam: registry event → `FeedbackIngester` → acquisition/curation boundary.
5. Status and artifact matrices before running commands.

## Entry point and complete call chain

There are two online chains, not one. Off-chain: client submits source to gateway, job persists, LangGraph gathers/fuses evidence, and a report returns with an initially unsubmitted on-chain placeholder. Direct-chain: a caller separately invokes audit MCP `submit_audit`; the service asks ML for a fusion embedding, runs proxy/proof, submits V2, and returns transaction state. Offline DATA/training created the teacher/artifact identities both paths rely on. Feedback observes chain records later; it does not retroactively prove original labels.

## Important symbols and configuration

- Identity tuple: source/content hash, DATA schema/export hash, teacher model hash, proxy/circuit version, verifier/registry address, transaction/proof hash.
- Shape seams: graph `[N,12]`, tokens `[W,512]`, labels `[10]`, fusion `[128]`, proxy outputs `[10]`, public instances `[138]`.
- Process seams: gateway 8000, ML 8001, MCP 8010–8014, Anvil 8545.
- Current known submit gaps: separate invocation, placeholder report, SHA-256 response versus contract `keccak256` proof hash, original model-hash propagation to transaction path, and shared proof filenames.

## Annotated source excerpt

Source: `agents/src/api/gateway.py::create_app.submit_audit`

```python
record = store.create(
    contract_code=req.contract_code,
    contract_address=contract_address,
    audit_timeout_s=req.audit_timeout_s,
    metadata=req.metadata,
)
task = asyncio.create_task(_run_job(...))
```

The gateway creates an off-chain job. There is no audit-MCP call in this path, which is why a completed job is not equivalent to an on-chain transaction.

## Worked example

Start with `Vault.sol`. Offline, SHA-256 `S` indexes its representation/export. Teacher checkpoint hash `M` predicts and emits fusion embedding `E[128]`. AGENTS report fuses deterministic and nondeterministic evidence. If direct submission is separately requested, proxy version `v2.0` maps `E` to ten outputs, proof `P` binds the public computation, verifier accepts, and registry appends an event for address `C`. A complete audit record should let an operator relate `S`, export hash, `M`, circuit/verifier identity, `C`, and transaction—even though current provenance is an operator assertion.

## Success trace

At each seam: schema/version matches, service health is semantically healthy, payload validates, tensor/public shapes match, artifact hashes resolve, tool status is explicit, proof verifies, stake/output guards pass, and transaction receipt/event/query agree.

## Failure trace

Debug in dependency order. A loader hash error is DATA/artifact, not model. HTTP 422 is payload validation, not inference. Gateway completed with placeholder is expected disconnected behavior, not a lost transaction. Missing proving key/SRS blocks direct proof. Verifier revert with valid off-chain proof points to calldata/layout/deployment identity. Concurrent proof submissions can collide on shared files.

## Design reasoning and rejected alternatives

Boundary-first debugging avoids changing multiple modules based on one symptom. Separate off-chain and direct-chain demonstrations preserve truth about current integration. Stable IDs and hashes are preferred over filenames/log prose because they can be compared across processes.

## Safe change walkthrough

For any cross-module change, write the compatibility statement first: producer symbol/shape/version, consumers, regeneration, tests, rollout, and rollback. Change the producer and its focused test, update each consumer in dependency order, regenerate artifacts, run boundary tests, then live smoke. Roll back all coupled artifacts/configurations as one versioned bundle.

## Guided lab

Complete [L10 — end-to-end ownership capstone](../labs/10_end_to_end_capstone.md).

## Tests and expected results

```bash
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/handbook/tools/verify_handbook.py lab --check L10
```

Inventory should print the discovered contracts. L10 preflight should explicitly report which local/private/live prerequisites are missing; it must not claim success by substituting mocks.

## Review questions

Where does each online path begin/end? What is the first invariant to inspect for a score mismatch? Which IDs prove byte identity versus operator attribution? How do you roll back a circuit change?

## Ownership checklist

- I can trace both online paths without conflating them.
- I can name every shape, hash, and service seam.
- I diagnose the first broken boundary before editing code.
- I report degraded/local-only behavior explicitly.
