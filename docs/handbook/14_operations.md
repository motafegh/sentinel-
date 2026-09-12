# 14 — Operations and troubleshooting

**Read this when:** you need to set up artifacts, start current services, run an off-chain audit, query registry history, validate R4/DATA artifacts, or exercise V3 contracts locally.

**Skip this if:** you are changing schemas/architecture; read [playbooks](15_change_playbooks.md) first.

**Estimated reading time:** 16 minutes.

## 30-second summary

Operate SENTINEL in layers and keep historical compatibility separate from current authority. The live runtime remains ML + selected MCP services + gateway off-chain audits; audit MCP :8012 is read-only. Run12 remains the historical operational teacher. Historical R4 G0–G7 are passed and immutable; Phase 8 is in progress. R4-D-011 accepts the exact V10 V2.6 physical representation lineage, while R4-D-012 requires a fresh guarded-selector successor before that selector has accepted physical authority. Full training remains unauthorized. V3 contract/protocol behavior can be tested locally, but no production signer/broadcast service is claimed.

## Just-enough mental model

```text
repo / tracked evidence / local artifacts
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

### 2. Artifact availability and DVC scope

Artifact availability is intentionally not collapsed into one “fresh clone contains everything” claim.

- **Git-tracked R4 authority:** current plans, ADRs, manifests, machine-readable policy/evidence records, and durable bounded evidence under `docs/plan/ml-R4/` are the controlling semantic/governance layer.
- **R4-D-011 physical lineage:** the exact accepted 22,540-identity V10 V2.6 representation root is a protected local physical artifact with recorded digest/evidence. Its acceptance record is tracked; the heavy physical root is not implied to be downloadable from a fresh clone.
- **R4-D-012 successor:** the guarded-selector lineage must be created as a new versioned candidate and separately bound/accepted. Do not mutate the D-011 root.
- **Run12 teacher:** historical checkpoint/companions may exist only on an approved working machine or historical artifact store. Run12 is compatibility/operational history, not repaired-model quality.
- **ZKML retained artifacts:** tracked proxy/ONNX/settings/compiled/VK material supports historical proxy reproducibility; proving prerequisites may still be local/private/regenerated.
- **RAG/runtime databases:** generated/local unless explicitly promoted.

The repository currently has two distinct DVC contexts and they must not be confused:

1. **Root `.dvc/` context** — repository-level/local artifact operations. The public config intentionally contains no committed remote. Machine-specific remotes belong in `.dvc/config.local`, which is ignored by Git.
2. **`data_module/.dvc/` context** — owns the DATA module’s historical `data_module/dvc.yaml` lifecycle (`ingest → preprocess → represent → ... → export`). It is a module-local pipeline boundary and is not evidence that the current R4 physical/evaluation lineage can be reconstructed by running `dvc repro` from a fresh clone.

For a private/local root remote, configure it without changing tracked config, for example:

```bash
cd "$REPO_ROOT"
dvc remote add --local -d localbackup /path/to/local/dvc/remote
```

For DATA-module DVC operations, run DVC from `data_module/` so the module-local `.dvc` root and `dvc.yaml` are selected intentionally.

Do not commit absolute laptop/WSL paths, credentials, private endpoint URLs, DVC cache/runtime locks, or `config.local` files.

### 3. Start current ML inference

```bash
cd "$REPO_ROOT"
SENTINEL_CHECKPOINT="<approved-local-Run12-checkpoint>" \
  ml/.venv/bin/uvicorn ml.src.inference.api:app --host 127.0.0.1 --port 8001
```

Confirm `/health` reports the intended checkpoint/model identity. Run12 is historical operational inference, not repaired-R4 model quality.

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

### 8. Current R4 physical/training boundary

For current Phase-8 DATA/ML work, read the exact restart authority before any physical-artifact operation:

- `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
- R4-D-011 physical V10 V2.6 acceptance;
- R4-D-012 guarded-selector promotion;
- the latest Phase-8 run/restart record referenced by the status matrix.

Operational rules:

- preserve the accepted R4-D-011 root and digest unchanged;
- do not regenerate/patch it in place;
- build any R4-D-012 guarded-token successor as a fresh lineage;
- require separate binding and physical acceptance;
- keep candidate #2 negative evidence unresolved until genuinely independent agreement exists;
- do not invent threshold/calibration/untouched-acceptance populations;
- do not launch the 100-epoch/full training run without explicit authorization.

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
| Run12 checkpoint absent | historical local artifact unavailable; not R4 semantic failure |
| R4-D-011 physical root absent locally | accepted physical artifact unavailable on this machine; do not reinterpret acceptance or regenerate casually |
| guarded-selector candidate absent | expected until a new R4-D-012 lineage is constructed and accepted |
| vNext unknown target appears as `0` | semantic corruption; stop |
| threshold/calibration role requested | unsupported under current evidence |
| `dvc pull` has no public default remote | expected current public-repo behavior; configure an authorized local/private remote explicitly |
| audit MCP rejects submit name | expected current read-only policy |
| gateway report unsubmitted | expected off-chain behavior |
| V3 policy signature missing | no authorized V3 submit; do not fall back to historical write path |
| proof verifies but V3 contract rejects | inspect target code/deadline/digest/signature/replay/stake/score layout |
| feedback V3 remains pending | expected while V3 promotion policy is unavailable |
| proving artifact missing | live proving prerequisite unavailable; do not convert to skipped success |

## Common change recipe

For operational changes:

1. identify the security/artifact domain: analysis, DATA build, model training, proving, signing, broadcasting, observation, or historical reproduction;
2. identify whether the required artifact is Git-tracked, local/private, reproducible, or historical-only;
3. change one layer and capture exact artifact/commit identities;
4. keep secrets and machine-specific remotes external;
5. run static/smoke → subsystem → relevant live checks;
6. preserve raw failure output;
7. update current status/metadata if availability or authority changed;
8. never use a historical compatibility path to bypass a current fail-closed boundary.

## Verification commands

```bash
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 docs/handbook/tools/verify_handbook.py live --services
python3 docs/handbook/tools/verify_handbook.py live --module agents
```

Foundry/ZK/R4 physical-artifact checks are separate explicit operations with their own prerequisites.

## Optional deep references

- [Runtime flows](02_runtime_flows.md)
- [Current status](16_current_status.md)
- [Security and trust](12_security_and_trust.md)
- [R4 master plan](../plan/ml-R4/00_MASTER_PLAN.md)
- [`data_module/dvc.yaml`](../../data_module/dvc.yaml)
- [`contracts/test`](../../contracts/test)

## Technical mastery layer

### Prerequisite knowledge

Know Python environments, service supervision, Git worktrees, DVC repository roots/local config, artifact hashes, HTTP/MCP, Foundry/Anvil, and least-privilege signing/transaction boundaries.

### Source map and reading order

Use current status first. For services, read gateway/ML/live MCP server entry points. For chain protocol, read `policy_signer.py` and V3 contract/tests. For DATA/R4 physical work, follow the current Phase-8 status/decision chain rather than historical DATA build commands.

### Execution trace and worked example

A normal operational demo starts Run12 ML, selected MCPs, and gateway, then ends with an off-chain report. A separate contract test can prove V3 digest/proof/storage invariants. A separate R4 physical-artifact workflow can inspect or construct versioned representation candidates. These exercises have different authorities and must not be collapsed into one “end-to-end production” claim.

### Implementation practice

Troubleshoot the first failed boundary and preserve its exact state. Do not respond to a missing signer, missing negative corpus, missing representation, missing DVC remote, or unavailable tool by substituting a weaker historical path.

### Review and ownership check

Can you operate the off-chain runtime, distinguish the two DVC contexts, identify which artifacts are Git-tracked vs local/private, query registry history, test V3 contracts, and run current R4 validation while keeping analysis, signing/broadcast, training, and protected-artifact responsibilities separate?
