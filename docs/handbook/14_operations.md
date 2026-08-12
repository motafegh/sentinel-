# 14 — Operations and troubleshooting

**Read this when:** you need to set up artifacts, start current services, run an off-chain audit, query registry history, validate R4/DATA artifacts, or exercise V3 contracts locally.

**Skip this if:** you are changing schemas/architecture; read [playbooks](15_change_playbooks.md) first.

**Estimated reading time:** 16 minutes.

## 30-second summary

Operate SENTINEL in layers and keep historical compatibility separate from current authority. Today the live runtime is ML + selected MCP services + gateway off-chain audits; audit MCP :8012 is read-only. Run12 remains the historical operational teacher. R4 G6 is the stable repaired DATA/ML control state, while Phase 7 DATA vNext still needs local physical representation binding before G7. V3 contract/protocol behavior can be tested locally, but no production signer/broadcast service is claimed.

## Just-enough mental model

```text
repo/artifacts
   ↓
static + R4 gates
   ↓
ML :8001 → selected MCP :8010–8014 → gateway :8000
                                      ↓
                                off-chain report

separate chain testing:
Foundry/Anvil → V3 registry/verifier/policy-digest tests

not current analysis-runtime behavior:
audit MCP signing/broadcast
```

“Smoke” checks imports/contracts quickly. “Module” runs a subsystem suite. “Live” touches real services, GPU, analyzers, proving/signing/RPC/chain state.

## Actual runtime/source walkthrough

### 1. Repository and documentation/R4 state

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py
```

Use the repository’s existing module environments. Do not upgrade dependencies during an operational verification unless that upgrade is the task.

### 2. Artifact availability

- **Run12 teacher:** historical checkpoint/companions may exist only on the working machine; obtain the approved historical artifact set when reproducing current inference.
- **R4 evidence/policy/roles:** tracked in Git and are the current DATA/ML semantic authority through G6.
- **DATA vNext Phase 7:** candidate semantic overlay exists on the Phase-7 branch; final G7 requires physical binding to the local graph/token/sidecar representation population.
- **ZKML retained artifacts:** tracked proxy/ONNX/settings/compiled/VK exist for historical proxy reproducibility; proving prerequisites may be local/private/regenerated.
- **RAG/runtime databases:** generated/local unless promoted separately.

### 3. Start current ML inference

```bash
cd "$REPO_ROOT"
SENTINEL_CHECKPOINT="<approved-local-Run12-checkpoint>" \
  ml/.venv/bin/uvicorn ml.src.inference.api:app --host 127.0.0.1 --port 8001
```

Confirm `/health` reports the intended checkpoint/model identity. Run12 is historical operational inference, not repaired-vNext model quality.

### 4. Start selected MCP services

From `agents/` in separate terminals:

```bash
poetry run python -m src.mcp.servers.inference_server
poetry run python -m src.mcp.servers.rag_server
poetry run python -m src.mcp.servers.audit_server
poetry run python -m src.mcp.servers.graph_inspector_server
poetry run python -m src.mcp.servers.representation_server
```

Audit MCP is **read-only**. Its live tools are latest/history/existence queries across V1/V2/V3. Do not expect or expose transaction submission from this service.

### 5. Start gateway / off-chain audit

```bash
cd "$REPO_ROOT/agents"
poetry run python -m src.api.gateway
```

Submit Solidity to `POST /audit`, capture the job ID, poll `GET /audit/{job_id}`, and inspect report/tool-status/provenance. `on_chain.submitted=false` is expected unless some separate external system later performs a V3 submission; gateway completion itself never does.

### 6. Query chain history

With a configured registry/RPC, use the audit MCP read tools or direct contract calls to inspect V1/V2/V3 history. Check `protocol_version` and V3 provenance fields before interpreting a result. A V2 record does not carry V3 context guarantees.

### 7. Exercise V3 contracts locally

For contract/protocol verification:

```bash
cd "$REPO_ROOT/contracts"
forge build
forge test
```

The tracked suite includes V3 behavior, digest parity, upgrade/storage, and real-proof verifier coverage. Anvil/deployment exercises are optional live integration checks. They do not create a production signing service.

A real V3 submitter would need an isolated policy signer plus transaction authority outside the analysis MCP. Do not improvise by wiring historical `_submit.py` into the live MCP.

### 8. Phase-7 local G7 binding

When working specifically on active R4 Phase 7, use the exact branch-bound command from the R4 Phase-7 handoff/current status. The gate scans the existing representation root read-only, binds the required physical graph/token/sidecar artifacts, and promotes the v2 manifest only transactionally after final validation.

Do not run this against `main` until Phase 7 is merged; use the dedicated Phase-7 worktree/branch.

## Interfaces, data shapes, and configuration

| Port | Expected process / authority |
|---:|---|
| 8000 | gateway / off-chain job API |
| 8001 | ML FastAPI / Run12 historical inference today |
| 8010 | inference MCP |
| 8011 | RAG MCP |
| 8012 | audit MCP / **read-only registry observation** |
| 8013 | graph inspector MCP |
| 8014 | representation MCP |
| 8545 | optional local Anvil |

Secrets remain external. Document variable names/prerequisites, never key/RPC credential values.

## Failure modes and current limitations

| Symptom | Correct first interpretation |
|---|---|
| Run12 checkpoint absent | historical local artifact unavailable; not DATA-vNext failure |
| Phase-7 G7 says representations pending | expected until local physical binding succeeds |
| vNext unknown target appears as `0` | semantic corruption; stop |
| threshold/calibration role requested | unsupported under current G6 evidence |
| audit MCP rejects submit name | expected current read-only policy |
| gateway report unsubmitted | expected off-chain behavior |
| V3 policy signature missing | no authorized V3 submit; do not fall back to historical write path |
| proof verifies but V3 contract rejects | inspect target code/deadline/digest/signature/replay/stake/score layout |
| feedback V3 remains pending | expected while V3 promotion policy is unavailable |
| proving artifact missing | live proving prerequisite unavailable; do not convert to skipped success |

## Common change recipe

For operational changes:

1. identify the security domain: analysis, DATA build, model training, proving, signing, broadcasting, or observation;
2. change one layer and capture exact artifact/commit identities;
3. keep secrets external;
4. run static/smoke → subsystem → relevant live checks;
5. preserve raw failure output;
6. update current status/metadata if availability or authority changed;
7. never use a historical compatibility path to bypass a current fail-closed boundary.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/handbook/tools/verify_handbook.py live --services
python3 docs/handbook/tools/verify_handbook.py live --module agents
```

Foundry/ZK/local G7 checks are separate explicit operations with their own prerequisites.

## Optional deep references

- [Runtime flows](02_runtime_flows.md)
- [Current status](16_current_status.md)
- [Security and trust](12_security_and_trust.md)
- [R4 master plan](../plan/ml-R4/00_MASTER_PLAN.md)
- [`contracts/test`](../../contracts/test)

## Technical mastery layer

### Prerequisite knowledge

Know Python environments, service supervision, Git worktrees, artifact hashes, HTTP/MCP, Foundry/Anvil, and least-privilege signing/transaction boundaries.

### Source map and reading order

Use current status first. For services, read gateway/ML/live MCP server entry points. For chain protocol, read `policy_signer.py` and V3 contract/tests. For DATA vNext local verification, follow the active R4 Phase-7 gate script rather than legacy DATA build commands.

### Execution trace and worked example

A normal operational demo starts Run12 ML, selected MCPs, and gateway, then ends with an off-chain report. A separate contract test can prove V3 digest/proof/storage invariants. A separate Phase-7 worktree can bind physical representations. These exercises have different authorities and must not be collapsed into one “end-to-end production” claim.

### Implementation practice

Troubleshoot the first failed boundary and preserve its exact state. Do not respond to a missing signer, missing negative corpus, missing representation, or unavailable tool by substituting a weaker historical path.

### Review and ownership check

Can you operate the off-chain runtime, query registry history, test V3 contracts, and run DATA-vNext validation while keeping analysis, signing/broadcast, training, and local protected-artifact responsibilities separate?
