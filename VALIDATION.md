# SENTINEL validation and CI

This page explains **what each validation surface proves, what it does not prove, and which checks are normal current CI versus retained historical R4 evidence workflows**.

SENTINEL is a multi-environment research/engineering repository. There is intentionally no claim that one root command exercises every DATA, ML, AGENTS, ZKML, and Solidity path.

## Current pull-request CI

| Check | Scope | What a pass establishes | What it does **not** establish |
|---|---|---|---|
| **Portfolio showcase** | fresh-clone source/config boundary | public architecture/trust claims still match committed source/config; unexecuted live capabilities remain `NOT_RUN` | vulnerability quality, live ML/analyzers, proof generation, signing/broadcast |
| **Handbook** | canonical docs + source/R4 authority | links/structure/source anchors are intact; historical G6/G7 compatibility facts still match; current D-009/D-011/D-012 authority matches committed acceptance evidence/ADRs | full module runtime behavior or heavy/local artifact availability |
| **SENTINEL system alignment** | contracts, ZKML, V3 policy, live audit-MCP trust boundary | retained proof bundle structure, fail-closed proof mutation rejection, V3 digest parity, contract tests, and read-only audit-MCP containment | complete AGENTS runtime, ML model quality, Phase-8 training completion |
| **Security hygiene** | current tracked PR state | newly introduced configured high-signal credential/private-key shapes are absent from the checked scope | a formal guarantee that no secret can exist or that a historical external credential has been revoked; full-history scanning is a separate main/scheduled/manual control |
| **DATA reproducibility** | `data_module/pyproject.toml` + committed `data_module/poetry.lock` | Poetry 2.1.3 can regenerate the committed DATA dependency resolution with zero lock drift | availability of heavy DATA artifacts, Solidity compiler matrix, GPU runtime, or accepted R4 physical roots |

The exact workflow definitions live under [`.github/workflows/`](.github/workflows/).

## Historical R4 workflows are not normal product CI

The repository also retains many workflows named `r4-phase*`. They were created to enforce specific DATA/ML research gates and preserve the evidence path that led to G0–G7 and later Phase-8 decisions.

They are **historical/gate-specific evidence machinery**, not a claim that every one should run on every pull request forever. Their presence is intentional because the R4 decision trail depends on reproducible gate logic and immutable evidence.

Current DATA/ML authority should be read from:

1. `docs/plan/ml-R4/PLAN_STATUS_MATRIX.md`;
2. accepted R4 ADRs/evidence, especially R4-D-009 / R4-D-011 / R4-D-012;
3. `docs/handbook/16_current_status.md`.

A green historical R4 workflow does not authorize a new full training run.

## Two-layer handbook authority validation

The handbook deliberately uses two machine-check layers rather than overwriting historical metadata:

1. `docs/handbook/tools/verify_handbook.py` checks structure/source anchors and the historical G6/G7/runtime-compatibility contract still needed by preserved artifacts and Run12-facing source;
2. `docs/handbook/tools/verify_current_r4.py` checks a separate `docs/handbook/_meta/current_r4.json` contract against the committed logical-V3 acceptance evidence, D-011 physical acceptance record, D-012 ADR/selector evidence, Phase-8 hold state, and current-facing documentation.

This separation prevents the old G7 compatibility contract from masquerading as the latest R4 authority while avoiding the opposite mistake of rewriting history to look current.

## Module validation matrix

| Area | Normal validation | Heavier / environment-dependent validation | Current limitation |
|---|---|---|---|
| **Documentation / authority** | `python3 docs/handbook/tools/verify_handbook.py static` + `python3 docs/handbook/tools/verify_current_r4.py` | handbook inventory/lab preflight | canonical prose can still require human review when semantics evolve beyond encoded checks |
| **Portfolio boundary** | `python3 tools/showcase_sentinel.py` | optional AGENTS LangGraph smoke after AGENTS install | default showcase intentionally does not run live services |
| **DATA** | DATA unit tests in its own Poetry environment + lock reproducibility CI | ingestion/Slither/representation pipelines, protected/local R4 artifacts | heavy physical data is not guaranteed in a fresh clone |
| **ML** | ML unit/focused tests in the ML environment | GPU/CUDA checks and explicitly authorized training/evaluation work | repaired full training remains unauthorized; Run12 is historical runtime |
| **AGENTS** | `cd agents && poetry run pytest -q` | live MCP services, LM Studio/LLM, external analyzers/formal tools | live external dependencies can be unavailable/degraded and must report that state explicitly |
| **ZKML** | dependency-light boundary tests + tracked bundle validation | live proving/regeneration with proving prerequisites | retained `check_mode="UNSAFE"` is a production-assurance limitation |
| **Contracts** | Foundry build/test + V3 digest/proof boundary checks | Anvil/testnet/deployment exercises | no production signer/broadcaster is claimed |

See [DEVELOPMENT.md](DEVELOPMENT.md) for exact environment setup and module commands.

## Evidence-state vocabulary

Validation output should preserve these distinctions:

- **PASS** — the stated check ran and established its bounded claim;
- **PASS_WITH_LIMITATION** — the claim is established but an explicit limitation remains;
- **NOT_RUN / unavailable / degraded** — the capability did not produce normal evidence;
- **FAIL** — the check ran and the required invariant was not established;
- **unsupported / unauthorized** — the repository intentionally lacks evidence or authority for the requested claim.

`NOT_RUN`, `unavailable`, `unsupported`, and `unknown` must never be silently converted into a clean/negative outcome.

## DATA dependency-lock policy

`data_module/` owns a separate Poetry environment. Its generated lockfile is now committed at `data_module/poetry.lock` and is part of the public reproducibility contract.

The lock was bootstrapped by the P5 CI path using **Poetry 2.1.3**, then the bootstrap write path was removed. The final workflow is read-only: it regenerates the resolution from `data_module/pyproject.toml` with the pinned Poetry version and requires `git diff --exit-code -- data_module/poetry.lock`.

Rules:

1. do not hand-edit or synthesize `poetry.lock`;
2. generate it with the pinned Poetry version used by the DATA reproducibility workflow;
3. CI regenerates the resolution and requires no diff from the committed lock;
4. changes to DATA dependency constraints must update the lock in the same change;
5. the lock only establishes dependency resolution—it does not make protected/local R4 physical artifacts available.

## Secret scanning policy

The project uses two complementary levels:

- **PR/current-tree scan:** lightweight and mandatory so newly introduced common secret shapes fail quickly;
- **history baseline scan:** full reachable-history scan on `main`, scheduled runs, or explicit manual execution. It is intentionally not required on every PR after the baseline is established because the repository history is large.

The repository-level scanner targets high-signal forms such as private-key assignments, PEM private keys, common provider credential URLs, GitHub tokens, AWS access-key IDs, and mnemonic assignments. It is a bounded engineering control, not a mathematical proof of absence.

### Known historical provider-RPC finding

The first P5 full-history scan inspected **92,243 reachable blobs / about 3.36 GB of blob content**. The current tracked tree passed, but the history scan found four historical blob identities containing the same credential-shaped provider RPC endpoint in obsolete ZKML helper/generated shell material.

Those exact `(kind, path, object-id)` identities are recorded in `tools/security/known_history_findings.json`. The current scanner treats only those reviewed identities as `KNOWN_HISTORICAL`; any new occurrence remains blocking. The credential value is intentionally not reproduced in current documentation.

The baseline-aware full-history scan subsequently passed: the four reviewed identities were classified as known historical findings and no additional configured high-signal occurrence appeared. Current-tree scanning also passed.

This baseline is **not evidence that the external provider credential was revoked or rotated**. Before the portfolio release, revocation/rotation must be confirmed externally if it has not already been completed. A known historical finding can remain reachable because this project intentionally avoids rewriting the evidence/provenance history merely for cleanup.

Suspected real credentials should be rotated/revoked immediately and handled according to [SECURITY.md](SECURITY.md), regardless of whether a scanner detected them.

## Portfolio release gate

Before PR #72 is merged and a portfolio release is created, the current branch should have successful results for the applicable current checks above. Historical R4 workflows remain evidence records and are not retroactively redefined as release checks.

The known historical provider-RPC finding adds one explicit release prerequisite outside Git: **confirm that the exposed historical provider credential has been revoked/rotated, or revoke/rotate it before release**. Do not infer that external state from repository cleanup alone.
