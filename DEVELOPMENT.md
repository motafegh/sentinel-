# Developing SENTINEL

SENTINEL is a multi-environment research/engineering monorepo. A fresh clone is intentionally **not** installed as one universal Python environment.

Use the environment owned by the module you are working on. This keeps incompatible ML, AGENTS, DATA, Solidity, and proving dependencies from being collapsed into a misleading single setup.

## 1. Fresh-clone expectations

A fresh clone contains the source code, tests, current handbook, R4 governance/evidence records, committed dependency manifests/locks, contract sources, and retained lightweight ZKML artifacts that are tracked in Git.

A fresh clone does **not** guarantee the following heavy or machine-local artifacts:

- the historical Run12 teacher checkpoint and every companion runtime artifact;
- the accepted R4-D-011 physical V10 V2.6 representation root;
- a future R4-D-012 guarded-selector successor representation;
- local RAG indexes/runtime databases;
- private DVC remotes;
- private RPC/API credentials;
- every proving key/SRS/runtime artifact needed for live proof regeneration.

Missing local artifacts must remain explicit `unavailable`/`not configured` conditions. Do not substitute historical artifacts, regenerate protected R4 artifacts casually, or reinterpret missing data as a successful/clean state.

For the detailed artifact boundary, see [`docs/handbook/14_operations.md`](docs/handbook/14_operations.md).

## 2. Recommended host/tooling

The commands below assume a POSIX shell. Linux or WSL is the simplest environment for the Solidity/analyzer tooling used by the project.

Install as needed for the module you are working on:

- Git;
- Python;
- Poetry;
- Foundry (`forge`, and optionally `anvil`/`cast`) for `contracts/`;
- Solidity compiler tooling such as `solc-select`/Slither for workflows that actually require it;
- CUDA-capable PyTorch only for ML operations that need a GPU.

A practical common interpreter for the Python modules is **Python 3.12.1+ and <3.13**. It satisfies the current ML and DATA constraints and is also accepted by AGENTS. Keep separate virtual environments even when they use the same Python interpreter.

Current declared Python ranges:

| Scope | Declared Python range | Environment owner |
|---|---|---|
| `ml/` | `>=3.12.1,<3.13` | `ml/pyproject.toml` + `ml/poetry.lock` |
| `data_module/` | `>=3.12,<3.13` | `data_module/pyproject.toml` |
| `agents/` | `>=3.11,<3.15` | `agents/pyproject.toml` + `agents/poetry.lock` |
| repository root | compatibility/shared workspace metadata; not the universal runtime | root `pyproject.toml` + `poetry.lock` |
| `contracts/` | Foundry/Solidity, not Poetry | `contracts/foundry.toml` + `foundry.lock` |
| `zkml/` | currently exercised through the ML Python environment plus Foundry where required | ML + `contracts/` tooling |

`data_module/` currently has no committed `poetry.lock`; deterministic DATA dependency locking remains a reproducibility item to close rather than something this guide pretends is already solved.

## 3. Clone and verify repository/documentation state

The repository contains substantial historical engineering/evidence history. For a normal development clone, Git partial clone is the preferred lightweight entry point because it preserves commit identities while deferring unneeded historical blob transfer:

```bash
git clone --filter=blob:none https://github.com/motafegh/sentinel-.git
cd sentinel-

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp

python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
```

A conventional full clone remains valid when all historical blobs are required locally. Do not rewrite/shallow-republish project history merely to reduce portfolio clone size; historical commit identities are used by the R4 evidence/provenance chain. See [`docs/plan/portfolio-professionalization/2026-09-04_REPOSITORY_WEIGHT_AND_HISTORY_AUDIT.md`](docs/plan/portfolio-professionalization/2026-09-04_REPOSITORY_WEIGHT_AND_HISTORY_AUDIT.md).

These checks validate tracked documentation/source relationships. They do not prove that heavy local artifacts or external services are present.

## 4. ML environment

Use `ml/` for teacher architecture, historical Run12 inference compatibility, Phase-8 training mechanics, ML tests, and ZKML Python validation.

```bash
cd "$REPO_ROOT/ml"
poetry env use python3.12
poetry install

poetry run pytest tests -q
```

Equivalent existing-project invocations may use `ml/.venv/bin/python` when that environment is already provisioned.

Important current boundary: installing the ML environment does **not** authorize a repaired/full training run. Historical R4 G0-G7 remain passed/immutable, Phase 8 is in progress, and full training remains gated by the current R4 decision/evidence chain.

See [`ml/README.md`](ml/README.md).

## 5. DATA environment

Use `data_module/` for DATA vNext/R4 code, ingestion/preprocessing mechanics, representation work, and the module-local historical DVC lifecycle.

Core CLI/tests:

```bash
cd "$REPO_ROOT/data_module"
poetry env use python3.12
poetry install
```

Add heavy pipeline dependencies only when needed:

```bash
poetry install --with pipeline
```

Add graph/token ML dependencies only when needed:

```bash
poetry install --with ml
```

Or install both optional groups for work that crosses those boundaries:

```bash
poetry install --with pipeline,ml
```

The module-local `data_module/.dvc/` and `data_module/dvc.yaml` describe a historical DATA lifecycle. They are **not** proof that a fresh clone can reproduce the currently accepted R4-D-011 physical representation by running `dvc repro`.

See [`data_module/README.md`](data_module/README.md).

## 6. AGENTS environment

Use `agents/` for the 14-node LangGraph audit pipeline, gateway, RAG/evidence logic, MCP services, security controls, registry observation, and feedback boundaries.

```bash
cd "$REPO_ROOT/agents"
poetry env use python3.12
poetry install

poetry run pytest -q
```

The AGENTS pytest configuration intentionally belongs to `agents/pyproject.toml`; do not rely on the repository-root pytest configuration for this module.

Starting the full runtime additionally requires the relevant external services/artifacts. The gateway remains off-chain and the live audit MCP on port 8012 remains read-only.

See [`agents/README.md`](agents/README.md).

## 7. Contracts environment

Use Foundry from `contracts/`:

```bash
cd "$REPO_ROOT/contracts"
./scripts/bootstrap_deps.sh
forge build
forge test
```

Network deployment is a separate live operation. It requires explicit local credentials and configured V3 verifier/policy-signer trust roots. Never place private keys in tracked files.

The current `foundry.toml` references RPC/API values through environment variables such as `SEPOLIA_RPC_URL` and `ETHERSCAN_API_KEY`; the deployment script reads `DEPLOYER_PRIVATE_KEY` from the environment.

See [`contracts/README.md`](contracts/README.md) and [`SECURITY.md`](SECURITY.md).

## 8. ZKML environment

The retained proxy/proof code is currently validated with the ML Python environment:

```bash
cd "$REPO_ROOT"
ml/.venv/bin/python -m pytest zkml/tests -q
```

Contract-side verifier validation belongs to Foundry:

```bash
cd "$REPO_ROOT/contracts"
forge test
```

Live proof regeneration has additional local cryptographic/data prerequisites and is not part of the fresh-clone baseline. The retained proof statement remains the fixed 128-to-10 proxy computation; V3 context attestation is a separate trust claim.

See [`zkml/README.md`](zkml/README.md).

## 9. Root Poetry metadata is not the universal environment

The repository has a root `pyproject.toml`/`poetry.lock` because historical/shared ML-oriented workspace tooling exists at the root. Do not infer from that file that `poetry install` at repository root provisions every module.

In particular:

- AGENTS owns its own Poetry environment and lockfile;
- ML owns its own Poetry environment and lockfile;
- DATA owns a separate Python/package contract;
- Contracts use Foundry;
- ZKML crosses the ML/Foundry boundary.

For new development, enter the target module first and use that module's manifest/test commands.

## 10. DVC and local artifacts

Two DVC contexts exist:

1. root `.dvc/` for repository-level/local artifact operations;
2. `data_module/.dvc/` for the historical DATA module lifecycle.

The public root DVC config intentionally has no machine-specific default remote. Configure private/local remotes without modifying tracked config, for example:

```bash
cd "$REPO_ROOT"
dvc remote add --local -d localbackup /path/to/local/dvc/remote
```

Machine-local DVC configuration belongs in ignored local config, not in committed absolute WSL/laptop paths.

## 11. Secrets and environment variables

Do not commit:

- `.env` files containing real values;
- RPC/API credentials;
- private keys or mnemonics;
- PEM/PKCS/SSH private-key material;
- private endpoint URLs or machine-specific artifact remotes.

Safe `.env.example` templates may be committed when they contain names/placeholders only.

For vulnerability or accidental-secret reporting, follow [`SECURITY.md`](SECURITY.md).

## 12. What to run before a change is considered reviewable

Choose validation proportionately to the changed scope.

Always for current-facing documentation/governance changes:

```bash
cd "$REPO_ROOT"
python3 docs/handbook/tools/verify_handbook.py static
python3 docs/handbook/tools/verify_handbook.py inventory
python3 -m unittest discover -s docs/handbook/tools/tests -p 'test_*.py'
```

Then add the affected module suite:

```bash
# ML
cd "$REPO_ROOT/ml" && poetry run pytest tests -q

# DATA (when its environment is provisioned)
cd "$REPO_ROOT/data_module" && poetry run pytest -q

# AGENTS
cd "$REPO_ROOT/agents" && poetry run pytest -q

# Contracts
cd "$REPO_ROOT/contracts" && forge test
```

Do not convert missing GPU, model, analyzer, RPC, proving, or physical-DATA prerequisites into fake passing results. Record them as not run/unavailable and state the prerequisite explicitly.

## 13. Running the complete system

A complete live audit is not the first fresh-clone smoke test. It depends on historical/local model artifacts and external analyzer/RAG/RPC services.

When those prerequisites are intentionally provisioned, follow [`docs/handbook/14_operations.md`](docs/handbook/14_operations.md) for the current service order, ports, artifact rules, and V3/off-chain boundaries.

## 14. Current reproducibility limitations

The repository intentionally documents rather than hides the remaining gaps:

- DATA currently lacks a committed Poetry lockfile;
- heavy R4 physical DATA is not publicly reconstructed by a one-command fresh-clone path;
- Run12/proving/runtime artifacts may be local or historical;
- repository history is relatively large; partial clone is the preferred non-destructive mitigation;
- there is no supported universal monorepo environment;
- full repaired training remains unauthorized;
- no production signer/broadcaster is claimed.

These are project-state constraints, not reasons to weaken evidence boundaries. Future professionalization work may close them with explicit artifact distribution, lock/CI improvements, and a bounded runnable showcase.