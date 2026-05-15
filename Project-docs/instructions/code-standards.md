# Code Standards & Technical Environment

## Environment

```
OS:           Windows 11, Lenovo Legion 5 Pro
Dev:          VS Code + WSL2 Ubuntu
Terminal:     WSL2 bash
Project root: ~/projects/sentinel  (main branch)
GitHub:       https://github.com/motafegh/sentinel
Branches:     feature/* → develop → main
```

---

## WSL2-Specific Gotchas

Silent failures that will waste time if you don't know them:

- **File paths**: Always use Linux paths inside WSL (`~/projects/sentinel`). Windows paths (`C:\Users\...`) cause silent failures.
- **Line endings**: `git config core.autocrlf false` in WSL — CRLF corrupts scripts silently.
- **Permissions**: Script won't run → `chmod +x script.sh` before debugging anything else.
- **Port forwarding**: WSL2 ports reach Windows browser at `localhost`. If not → Windows Firewall.
- **Python**: the Project uses Poetry and for creating files this commands: "touch " command then "code " command  and then for running : poetry run python 


### GPU / CUDA Verification

 Do  if needed everything it needs 
---

## Actual Current Project Structure

**This is the canonical reference. The structure below reflects what actually exists on disk.** and might be changed or updated by the time so needs to check

```
sentinel/                          ← project root
├── ml/                            ← ML module (standalone, own pyproject.toml)
│   ├── src/                       ← ML source root
│   │   ├── data/                  ← dataset loaders
│   │   │   ├── bccc_dataset.py
│   │   │   ├── solidifi_dataset.py
│   │   │   └── graphs/            ← graph extraction
│   │   │       ├── ast_extractor.py   ← CANONICAL version
│   │   │       └── graph_builder.py
│   │   ├── datasets/
│   │   │   └── dual_path_dataset.py
│   │   ├── models/                ← model architectures
│   │   │   ├── gnn_encoder.py
│   │   │   ├── transformer_encoder.py
│   │   │   └── fusion_layer.py
│   │   ├── tools/
│   │   │   └── slither_wrapper.py ← use slither_wrapper_turbo.py for production
│   │   ├── training/              ← EMPTY — not yet built
│   │   ├── inference/             ← EMPTY — not yet built
│   │   ├── utils/
│   │   │   └── hash_utils.py
│   │   └── validation/            ← temporary home for tests (see note below)
│   ├── data/                      ← raw + processed datasets (not in git)
│   │   ├── processed/             ← parquet files, label CSVs
│   │   ├── splits/                ← train/val/test indices (.npy)
│   │   ├── graphs/                ← extracted graph files
│   │   └── tokens/                ← tokenized contract files
│   ├── scripts/                   ← standalone runnable scripts (not importable)
│   ├── configs/                   ← EMPTY — config YAMLs go here
│   ├── models/                    ← saved model checkpoints (not in git)
│   ├── notebooks/                 ← exploration only, never production logic
│   ├── tests/                     ← EMPTY — tests should migrate here from src/validation/
│   ├── pyproject.toml
│   └── poetry.lock
│
├── contracts/                     ← Solidity / Foundry module
│   ├── src/                       ← EMPTY — contracts not yet written
│   ├── test/                      ← EMPTY — tests not yet written
│   ├── script/                    ← EMPTY — deploy scripts not yet written
│   ├── lib/                       ← forge-std, openzeppelin (installed)
│   └── foundry.toml
│
├── BCCC-SCsVul-2024/              ← raw dataset (not in git ideally)
├── Project-docs/                  ← project documentation files
├── docs/                          ← milestone docs, ADRs
├── logs/                          ← runtime logs (not in git)
├── pyproject.toml                 ← root level (orchestration)
└── poetry.lock
```

### Known Structural Issues (clean up as you go)

| Issue | Location | Fix |
|---|---|---|
| Multiple ast_extractor versions | `ml/src/data/graphs/` | `ast_extractor.py` is canonical — delete v2, v3. v4 in scripts is the production extractor, promote it when ready |
| Backup file in source | `ml/src/tools/slither_wrapper_backup_*.py` | Delete — git has history |
| Tests in wrong place | `ml/src/validation/` | Should live in `ml/tests/` mirroring `ml/src/` — migrate when building training module |
| Old graph directories | `ml/data/graphs_old_*` | Delete when current `ml/data/graphs/` is confirmed correct |
| Root `src/` | `sentinel/src/` | Empty — remove or populate when other modules (zkml, agents, api) are started |

### Where Future Modules Live

When you start building beyond ML and Solidity:

| Module | Root path | Notes |
|---|---|---|
| ZKML | `zkml/` (new top-level dir) | Mirror `ml/` structure — own src/, tests/, configs/ |
| Agents | `agents/` (new top-level dir) | Same pattern |
| API | `api/` (new top-level dir) | FastAPI app |
| Shared utils | `shared/` or inline per module | Decide when duplication appears |

### File Location Reference (current)

| What | Canonical path |
|---|---|
| GNN model architecture | `ml/src/models/gnn_encoder.py` |
| Transformer model | `ml/src/models/transformer_encoder.py` |
| Fusion layer (GMU) | `ml/src/models/fusion_layer.py` |
| BCCC dataset loader | `ml/src/data/bccc_dataset.py` |
| SolidiFI dataset loader | `ml/src/data/solidifi_dataset.py` |
| Graph builder | `ml/src/data/graphs/graph_builder.py` |
| AST extractor (canonical) | `ml/src/data/graphs/ast_extractor.py` |
| Dual-path dataset | `ml/src/datasets/dual_path_dataset.py` |
| Slither wrapper (prod) | `ml/src/tools/slither_wrapper_turbo.py` |
| Processed data | `ml/data/processed/` |
| Train/val/test splits | `ml/data/splits/` |
| Saved model checkpoints | `ml/models/` (not in git) |
| Solidity contracts | `contracts/src/` (not yet written) |
| Milestone docs | `docs/milestones/` |
| ADRs | `docs/decisions/` |


---

## Non-Negotiable Code Rules

Every line is portfolio code reviewed by engineers.
- No Emojis or so in Codes or Commands 
- No bare `except:` — catch specific exceptions with context
- No `print()` — use `logger` (loguru)
- No magic numbers — externalize to config YAML in `ml/configs/`
- No `TODO: fix later` — fix it now, or open a GitHub issue and reference the number inline
- No commented-out code — git has history, delete it
- No versioned filenames (`_v2`, `_backup_*`) — that's what branches are for
- Always: type hints, docstrings, error handling, logging at state transitions
- Coverage minimum: 80% overall, 100% on inference and proof generation paths

---

## Commit Message Scopes (SENTINEL-specific)

```
Types:  feat | fix | refactor | test | docs | chore
Scopes: ml | zkml | agents | api | contracts | utils | config

Examples:
  feat(ml): Add fusion layer GMU implementation
  fix(ml): Correct graph builder edge indexing for multi-edge contracts
  test(ml): Add unit tests for dual_path_dataset loader
  refactor(ml): Promote ast_extractor_v4 to canonical and delete old versions
  docs(milestone): Complete Milestone 1 data pipeline documentation
  chore(contracts): Install OpenZeppelin upgradeable library
```

---
