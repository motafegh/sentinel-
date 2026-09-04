# SENTINEL Repository Weight and History Audit

**Date:** 2026-09-04  
**Branch:** `portfolio/professionalization-2026-09-02`  
**Purpose:** close the portfolio-readiness repository-size audit without rewriting technical history or invalidating R4 evidence identities.

## Verdict

The repository is materially large at the GitHub-reported repository level, but the current professionalization branch does **not** contain a corresponding set of very large active artifacts. The dominant remaining weight is therefore historical Git object/history accumulation rather than an obviously bloated current working tree.

**Decision:** do not rewrite Git history as part of ordinary portfolio cleanup.

A history rewrite would change historical commit identities and would therefore be disproportionately risky for SENTINEL because R4 plans, evidence records, checkpoints, and decision chains intentionally cite exact commits. Reducing repository bytes is not worth invalidating those references merely for appearance.

## Measured repository-level signal

GitHub repository metadata reports approximately **405,887 KB** of repository size (about **396 MB** using 1024-based conversion).

This number reflects repository storage/history, not merely the checked-out files at the current branch tip.

## Current-tree checks

The current branch was inspected through GitHub tree/content metadata with emphasis on the file classes most likely to dominate repository weight.

### ZKML retained artifacts

The retained ZKML material is small enough to remain useful committed reproducibility evidence:

- `zkml/models/proxy.onnx` — ~44 KB;
- `zkml/models/proxy_best.pt` — ~45 KB;
- `zkml/ezkl/model.compiled` — ~349 KB;
- `zkml/ezkl/calibration.json` — ~332 KB;
- `zkml/ezkl/verification_key.vk` — ~67 KB;
- proof/witness/settings files are smaller still.

These do not explain a ~396 MB repository.

### R4 review bundles

The currently retained review archives are also small:

- `r4_gap002_blind_review_bundle_v1.zip` — ~382 KB;
- `r4_gap007_candidate2_independent_review_v1.zip` — ~4 KB.

The small review bundles are evidence artifacts and are not repository-size concerns.

### Current ML data surface

`ml/data/` currently contains lightweight tracked metadata/examples such as:

- `drift_baseline_run12.json` — ~20 KB;
- `warmup_run12.jsonl` — ~41 KB;
- documentation and ignore metadata.

Historical/generated graph/token/cache datasets are excluded by current ignore boundaries rather than carried as active Git content.

### Current model/checkpoint protection

`ml/.gitignore` explicitly excludes:

- `/checkpoints/`;
- `/models/`;
- `*.pt`;
- `*.pth`;
- `*.ckpt`;
- `*.safetensors`.

This is the correct forward-looking boundary for large ML training artifacts. The intentionally retained ZKML proxy artifacts live outside that ML-generated-artifact boundary and are small.

## Historical evidence of prior weight accumulation

Repository history shows earlier phases creating and later cleaning sizeable generated DATA/ML material, including tracked processed CSV/split-era artifacts and repeated archive/data cleanup. Examples include:

- a May content-dedup phase that produced a 44,420-row processed multilabel CSV and split artifacts;
- later cleanup that removed tracked stale processed CSV/config files and moved generated graph/token/cache material out of the active tree;
- subsequent ML/DATA work explicitly keeping multi-GB caches, checkpoints, logs, physical exports, and large source datasets untracked or DVC/local.

A historical check confirmed that `ml/data/processed/multilabel_index_deduped.csv` existed in Git at its historical commit, while a later 2.2 GB `cached_dataset_v9.pkl` did **not** exist as a Git-tracked file at the checked historical commit. This distinction matters: not every large local training artifact contributed to Git history, but historical tracked data/config artifacts did accumulate before current hygiene rules matured.

## Why history rewrite is rejected for normal professionalization

Tools such as `git filter-repo` could potentially reduce historical storage, but doing so would rewrite commit IDs throughout the affected history.

For SENTINEL that has unusually high cost because exact commit identities are used as evidence/provenance anchors across:

- R4 plans and status records;
- accepted ADRs and physical-lineage decisions;
- reproducibility checkpoints;
- historical audits and handoff records;
- review/evidence manifests.

A rewrite would require a controlled migration of every affected commit reference plus validation of the complete evidence chain. That is not a repository-hygiene task.

Therefore:

- **do not run `git filter-repo`, BFG, force-push rewritten history, or equivalent cleanup merely to reduce portfolio clone size;**
- only reconsider history surgery if repository size becomes an operational blocker large enough to justify a dedicated migration plan and complete evidence-reference rebinding.

## Portfolio mitigation without rewriting history

For fresh development clones, prefer Git partial-clone support:

```bash
git clone --filter=blob:none https://github.com/motafegh/sentinel-.git
```

This preserves commit/history identity while deferring unnecessary historical blob transfer until requested. A normal clone remains valid when full local history is desired.

For GitHub portfolio review, the web repository surface is unaffected by local clone strategy.

## Current classification

| Area | Result | Action |
|---|---|---|
| current tracked ML checkpoints/models | controlled | keep ignore policy |
| current physical DATA/large datasets | not carried as normal Git content | keep DVC/local/artifact boundary |
| retained ZKML artifacts | small + useful evidence | keep tracked |
| R4 review bundles | small + evidential | keep tracked |
| current generated/runtime cruft | already cleaned/ignored | no further deletion now |
| repository-level ~396 MB size | primarily historical concern | accept for now; use partial clone |
| Git history rewrite | high-risk to evidence identities | explicitly rejected for normal cleanup |

## Gate result

Portfolio item **M-011 (repository size/artifact/history audit)** is complete at the audit/decision level.

The correct outcome is **not** “make the repository tiny at any cost.” The correct outcome is:

1. current large/generated artifacts have explicit boundaries;
2. no obvious current-tree blob justifies destructive cleanup;
3. historical weight is acknowledged;
4. clone ergonomics have a non-destructive mitigation;
5. history rewriting is prohibited unless later authorized as a dedicated evidence-preserving migration.
