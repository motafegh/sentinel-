# Phase 0 — Baseline Freeze and Previous-Evidence Location

**Status:** READY  
**Gate:** G0  
**Purpose:** establish exact current state and locate prior work without reviewing contracts.

## Scope

- repository/worktree identity;
- active DATA and ML bundle;
- artifact hashes and protections;
- distinct population counts;
- locations and availability of previous evidence.

## Explicit non-scope

- contract-level semantic review;
- label correction;
- crosswalk changes;
- export regeneration;
- retraining;
- threshold/calibration changes;
- architecture work.

## Work packages

### P0.1 Repository and instruction baseline

Record repository root, branch, full HEAD, remotes, dirty state, submodules, DVC status, applicable agent instructions, environment files, and package locks.

If unrelated uncommitted changes exist, do not clean/reset/stash them automatically.

### P0.2 Active DATA bundle

Resolve through executable configuration and manifests:

- source pins and enabled state;
- raw/preprocessed/representation locations;
- per-source and merged label artifacts;
- verification outputs;
- split version;
- export version;
- label and representation manifests;
- counts at every boundary.

### P0.3 Active ML bundle

Resolve:

- active checkpoint;
- checkpoint-embedded configuration;
- DATA export/split binding;
- threshold sidecar;
- calibration sidecar or explicit absence;
- MLOps config and environment overrides;
- tokenizer and graph schema;
- relevant MLflow/training run;
- drift/benchmark artifacts.

Do not select an artifact from its filename alone.

### P0.4 Previous-evidence location

Search repository history and local storage for:

- DIVE investigation files and samples;
- BCCC review/verification files;
- SolidiFI and SmartBugs source analyses;
- manual review CSV/JSON/Markdown;
- tool reports;
- exploit reproductions;
- source snippets;
- prior AI-review outputs;
- benchmark case studies;
- scripts and commits;
- referenced temporary outputs.

Record `UNAVAILABLE` rather than recreate missing history.

### P0.5 Protected artifact manifest

Protect current:

- labels;
- crosswalks;
- configs;
- splits;
- exports;
- checkpoints;
- thresholds;
- calibration;
- MLOps configuration;
- historical evidence files.

### P0.6 Count snapshot

Keep separate:

- source records;
- raw contracts;
- preprocessed contracts;
- label rows;
- merged-label contracts;
- split rows;
- export label rows;
- representation-complete contracts;
- actual ML-loaded train/validation/test rows.

## Outputs

- `manifests/baseline_manifest.json`
- `manifests/protected_artifacts.json`
- `manifests/availability_inventory.csv`
- `manifests/evidence_location_inventory.csv`
- `findings/01_baseline_and_evidence_location.md`
- helper scripts under `scripts/`

Update the operational registers.

## G0 pass criteria

- actual local baseline is explicit;
- active DATA/ML bundle is bound by evidence;
- prior evidence locations are inventoried;
- missing artifacts are explicit;
- population counts are separated;
- protected artifacts are recorded;
- manifests validate;
- no protected artifact changed.

Stop after G0 report.
