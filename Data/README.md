# sentinel-data

Data engineering module for the SENTINEL smart-contract security oracle.

Builds a **verified, multi-source, multi-label** Solidity contract dataset from 17 curated sources and exports it to `sentinel-ml` as versioned shards. The module exists because the BCCC corpus that trained Runs 1–9 had 89.4% Reentrancy false-positives and 86.9% CallToUnknown false-positives — the model was learning label noise, not vulnerability patterns. See `docs/legacy/bccc_deep_dive/` for the full Phase 1–5 investigation.

**One-way dependency:** `sentinel-data` → (used by) → `sentinel-ml`. This package never imports from `sentinel-ml`.

---

## Quick start

```bash
# Install (core deps, no GPU required)
cd Data/
poetry install

# Verify the installation
poetry run sentinel-data --help

# Dry-run the full pipeline
poetry run sentinel-data run --dry-run

# Dry-run from a specific stage
poetry run sentinel-data run --dry-run --from-stage verify

# Run a single stage (dry)
poetry run sentinel-data ingest --source scabench --dry-run

# Run all smoke tests
poetry run pytest tests/ -v
```

---

## Module map

```
Data/
├── pyproject.toml                     sentinel-data package definition
├── config.yaml                        ← single source of truth for sources + pipeline settings
├── dvc.yaml                           ← 9-stage DVC DAG
├── sentinel_data/                     ← installable Python package
│   ├── cli.py                         entry point: sentinel-data <stage> [--dry-run]
│   ├── ingestion/                     Stage 1 — per-source connectors + SHA-256 manifests
│   ├── preprocessing/                 Stage 1 — flatten + compile + dedup + normalize + bucket
│   ├── representation/                Stage 2 — graph (.pt) + token extraction (v9 schema stub)
│   ├── labeling/                      Stage 3 — 17 crosswalk YAMLs + label merger
│   ├── verification/                  Stage 4 — AST semantic checks + tool corroboration
│   ├── splitting/                     Stage 5 — deterministic splits + leakage audit
│   ├── registry/                      Stage 5 — SQLite artifact catalog
│   ├── analysis/                      Stage 6 — complexity proxy risk + co-occurrence matrix
│   └── export/                        Stage 7 — shard export to sentinel-ml seam
├── data/                              ← DVC-tracked pipeline outputs (not in git)
│   ├── raw/                           ingested .sol files + manifests
│   ├── preprocessed/                  flattened + compiled + deduped
│   ├── representations/               graph .pt + token files
│   ├── labels/                        per-contract multi-label CSVs
│   ├── verification/                  verification reports + dropped.csv
│   ├── splits/                        train/val/test split manifests
│   ├── registry/                      SQLite catalog
│   ├── analysis/                      complexity proxy report + co-occurrence matrix
│   └── exports/                       sharded .pt export for sentinel-ml
├── docs/
│   ├── architecture.md                data-flow + DAG diagrams
│   ├── decisions/                     ADRs (append-only)
│   └── legacy/bccc_deep_dive/        ← frozen Phase 1–5 BCCC investigation
└── docker/
    └── Dockerfile.data                python:3.12.1-bookworm + 6 baseline solc versions
```

---

## Pipeline stages

| Stage | Command | What it does | Status |
|---|---|---|---|
| 0 | — | Skeleton + DVC scaffold | ✅ Stage 0 (this) |
| 1 | `ingest` + `preprocess` | Pull sources → flatten → compile → dedup → normalize → bucket | ⏳ Stage 1 |
| 2 | `represent` | Port graph extractor from `ml/` — v9 schema, byte-identical test | ⏳ Stage 2 |
| 3 | `label` | 17 crosswalk YAMLs + label merger | ⏳ Stage 3 |
| 4 | `verify` | BCCC Phase 5 regression test + AST semantic checks | ⏳ Stage 4 |
| 5 | `split` + `register` | Deterministic splits + leakage audit + SQLite catalog | ⏳ Stage 5 |
| 6 | `analyze` | Complexity proxy risk + co-occurrence matrix | ⏳ Stage 6 |
| 7 | `export` | Shard export + predictor tier-threshold fix + EMITS edge fix | ⏳ Stage 7 |
| 8 | — | Run 11 launch (2026-08-18) | ⏳ Stage 8 |

---

## Data sources (17 enabled + BCCC deferred)

See `config.yaml` for the full list with pins and connector types.

**Tier 1 (gold — human-curated or mathematically certain):**
ScaBench, SmartBugs Curated, SolidiFI, DIVE (Nature 2025), FORGE (ICSE 2026), Web3Bugs

**Tier 2 (silver — expert-audited or tool-majority):**
ScrawlD (3/5 tool majority), Code4rena audit reports, DeFi Hacks REKT, DeFiVulnLabs, EVMbench, Bastet

**Tier 3 (bronze — tool-generated, conservative):**
SmartBugs Wild ⚠ (97% FP rate — use with care), Slither-Audited (HF), OpenZeppelin Contracts, Ethernaut

**Tier 4 (unlabeled pretraining):**
DISL (514K contracts), ReentrancyStudy (230K)

**BCCC (deferred):** 89.4% Reentrancy FP. Legacy outputs at `docs/legacy/bccc_deep_dive/`. Verified v1.4 labels (24,021 contracts) may be re-introduced as a gold supplement in v2.1.

---

## Schema version

The active graph schema is **v9** (verified 2026-06-08). See `sentinel_data/representation/_schema_constants.md` for the full constant table.

| Constant | Value |
|---|---|
| `FEATURE_SCHEMA_VERSION` | `"v9"` |
| `NODE_FEATURE_DIM` | `12` |
| `NUM_NODE_TYPES` | `14` |
| `NUM_EDGE_TYPES` | `12` |
| `NUM_CLASSES` | `10` (LOCKED) |

The proposal §2 incorrectly says v8. All stages use v9.

---

## WSL2 caveats

All commands in this module are run inside WSL2. Use `wsl -- bash -c '...'` from the Windows host when scripting. The PowerShell host throws errors on inline WSL commands. Example:

```powershell
# From Windows PowerShell
wsl -- bash -c 'cd /home/motafeq/projects/sentinel/Data && poetry run sentinel-data run --dry-run'
```

---

## MLflow backend

The v2 build uses `sqlite:///mlruns.db` only. The `file:///` backend is **corrupt** (experiments 1, 2, 3 are in the file backend and return empty results). Any plan or script that logs an experiment must set:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```

This is also declared in `config.yaml` under `pipeline.mlflow.uri`.

---

## Design decisions

See `docs/decisions/` for the full ADR register. The two most important:

- **ADR-0001:** Why `sentinel-data` is a separate package (the BCCC failure + one-way dependency rule)
- **ADR-0002:** Code bug state at build start (8 fixed bugs + 3 open bugs — what the Stage 2 regression test guards)

---

## Contributing

1. Read `docs/decisions/ADR-0001-sentinel-data-skeleton.md` for the architectural context
2. Check `docs/proposal/Data_Module_Proposals/00_INDEX.md` for the current stage plan
3. `poetry install` and `poetry run pytest tests/ -v` must pass before any PR
4. The Stage 2 regression test (`byte-identical output vs ml/`) is the merge gate for any extractor change
