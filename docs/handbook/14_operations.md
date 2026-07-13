# 14 — Operations and troubleshooting

**Read this when:** you need to set up artifacts, start services, run an off-chain audit, prove/submit on Anvil or Sepolia, or diagnose failures.

**Skip this if:** you are changing schemas or architecture; read [playbooks](15_change_playbooks.md) first.

**Estimated reading time:** 16 minutes.

## 30-second summary

Operate SENTINEL in layers: establish environment and artifact availability, run static/smoke checks, start ML, start selected MCP services, start gateway, run the off-chain flow, then separately opt into EZKL and chain submission. Never treat mock/degraded health as live success, and never put secret values in commands committed to the repository.

## Just-enough mental model

```text
setup → artifacts → smoke → ML:8001 → MCP:8010-8014 → gateway:8000
                                                ├→ off-chain audit
optional proving artifacts + Anvil:8545/deployment └→ direct chain submission
```

“Smoke” checks imports/contracts quickly. “Module” runs a full suite. “Live” touches real services, GPU, EZKL, analyzers, RPC, or chain state.

## Actual runtime/source walkthrough

### 1. Repository and environment

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
```

Use the module’s existing environment: `ml/.venv` for ML/ZKML, Poetry under `agents`, and Foundry under `contracts`. Install from the tracked dependency declarations; do not improvise version upgrades during an operational verification.

### 2. Acquire or regenerate artifacts

- ML: obtain the approved Run 12 checkpoint and companion state/threshold files. This checkout’s local `.dvc` pointers are ignored/untracked, so a fresh clone needs an approved pointer/remote handoff.
- DATA training: obtain or regenerate the approved export and split; runtime source inference does not need the full training export.
- ZK submission: verify tracked proxy/settings/compiled/VK artifacts and obtain/regenerate `proving_key.pk` and `srs.params` with the pinned EZKL toolchain.
- RAG: build indexes from approved sources if RAG evidence is required.

### 3. Start ML

```bash
cd "$REPO_ROOT"
SENTINEL_CHECKPOINT="<approved-local-checkpoint-path>" \
  ml/.venv/bin/uvicorn ml.src.inference.api:app --host 127.0.0.1 --port 8001
```

The placeholder is intentional; never replace it in documentation with a private machine path. Confirm `GET /health` reports `predictor_loaded=true` and the expected model hash.

### 4. Start MCP services

In separate terminals from `agents/`:

```bash
poetry run python -m src.mcp.servers.inference_server
poetry run python -m src.mcp.servers.rag_server
poetry run python -m src.mcp.servers.audit_server
poetry run python -m src.mcp.servers.graph_inspector_server
poetry run python -m src.mcp.servers.representation_server
```

Start only the services required for your test, but record unavailable services honestly.

### 5. Start gateway and demonstrate off-chain audit

```bash
cd "$REPO_ROOT/agents"
poetry run python -m src.api.gateway
```

Submit source to `POST /audit`, capture the returned job ID, and poll `GET /audit/{job_id}`. Verify report tool statuses, path, dual verdicts, model hash, security detections, and `on_chain.submitted=false`.

### 6. Direct ZK/on-chain path

For local work, start `anvil --port 8545`, deploy with Foundry using a locally supplied account, stake the operator, configure the audit MCP for the local RPC/registry, then invoke `submit_audit`. Validate the off-chain proof first, transaction receipt second, and `getLatestAuditV2` third. Sepolia repeats this sequence with externally managed secrets/funds and a reviewed deployment manifest.

## Interfaces, data shapes, and configuration

| Port | Expected process |
|---:|---|
| 8000 | gateway |
| 8001 | ML FastAPI |
| 8010 | inference MCP |
| 8011 | RAG MCP |
| 8012 | audit MCP |
| 8013 | graph-inspector MCP |
| 8014 | representation MCP |
| 8545 | local Anvil |

Environment names are cataloged in [reference](17_reference.md#environment-variable-registry). Local `.env` values, RPC credentials, operator keys, and deployed private state must remain outside docs and Git.

## Failure modes and current limitations

| Symptom | Check first |
|---|---|
| pytest capture/temp failure | export Linux `/tmp` variables shown above |
| ML health degraded | checkpoint path/artifacts, dependency load, CUDA/CPU memory, logs |
| compilation error | pragma, installed solc versions, import availability, surfaced exception |
| MCP health down | process/port conflict, startup artifact/index, configured upstream URL |
| gateway job failed | persisted error, `tool_status`, service health, graph timeout |
| no RAG results | index availability/schema/filter, not immediate “no historical analogue” |
| proof missing artifact | `proving_key.pk`, `srs.params`, compiled/settings/VK identity |
| proof verifies but tx fails | key/funds/stake/RPC/registry ABI/address, gas, chain ID |
| score mismatch revert | little-endian parsing, 138 signal layout, proof-derived output felts |
| report says unsubmitted | expected for gateway flow; invoke direct submission separately |

## Common change recipe

For an operational configuration change:

1. Change one layer at a time and capture before/after health/inventory.
2. Keep secrets external and record only names/artifact hashes.
3. Run smoke, then module, then relevant live checks.
4. Preserve raw failure output; do not add `|| true`.
5. Update metadata/reference/status if defaults or availability changed.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/handbook/tools/verify_handbook.py live --services
python3 docs/handbook/tools/verify_handbook.py live --module agents
python3 docs/handbook/tools/verify_handbook.py live --ezkl --anvil
```

Live flags fail when prerequisites are missing; they do not silently skip.

## Optional deep references

- [Runtime flows](02_runtime_flows.md)
- [Current status](16_current_status.md)
- [`ml/deploy`](../../ml/deploy)
- [`contracts/script/Deploy.s.sol`](../../contracts/script/Deploy.s.sol)

## Technical mastery layer

### Prerequisite knowledge

Know Python environments, service process supervision, curl/JSON, DVC/artifact acquisition, Foundry/Anvil, and secret-safe environment configuration.

### Source map and reading order

Use metadata artifact/service registries, module entry points, health routes, gateway job runner, audit submit path, EZKL prerequisites, and deployment scripts. [T10](technical/10_end_to_end_debugging.md) supplies boundary-first troubleshooting.

### Execution trace and worked example

Start dependencies in order: artifacts/environments → ML → MCP services → gateway; separately start Anvil/deployments for chain work. First demonstrate a gateway report. Then, as a distinct live exercise, preflight private proof artifacts/stake/config and invoke direct submission.

### Implementation practice

Run `verify_handbook.py lab --check L10` before live work and follow [L10](labs/10_end_to_end_capstone.md). Troubleshoot the first unhealthy boundary: artifact, import, port, payload, tensor, proof, calldata, stake, or receipt.

### Review and ownership check

Can you start/stop each process, preserve logs without secrets, and prove which checks were smoke, module, or live?
