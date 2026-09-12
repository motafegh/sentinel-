# SENTINEL Repository Weight, Artifact, and History Audit

**Date:** 2026-09-04  
**Last reconciled:** 2026-09-05  
**Status:** **COMPLETE — canonical M-011 record**  
**Branch:** `portfolio/professionalization-2026-09-02`  
**Scope:** portfolio-professionalization item M-011

## 1. Conclusion

SENTINEL remains materially large at the GitHub repository-storage level, but the current professionalization branch does **not** contain a corresponding set of very large active artifacts.

GitHub reports repository size at approximately **405,887 KB** (about **396 MB** using 1024-based conversion). Inspection of the current branch shows that the retained active DATA, ZKML, review, lock, and training-snapshot artifacts are comparatively compact and have clear reproducibility/evidence value.

**Decision:** keep the current evidence-preserving tree and do **not** rewrite Git history as ordinary portfolio cleanup.

The dominant remaining size concern is historical Git-object accumulation. Rewriting it would change commit identities across a project whose R4 plans, evidence records, reviews, and decisions intentionally bind exact commits. Reducing repository bytes does not justify breaking that provenance chain merely for appearance.

## 2. Current-tree classification

### DATA / R4 tracked exports — KEEP

`data_module/data/exports/sentinel-r4-vnext-v1/` contains compact machine-readable semantic/compatibility authority, including files such as:

- `label_states.parquet` — about 1.9 MB;
- `ml_targets.parquet` — about 1.3 MB;
- associated manifest, validation, binding, source, and evidence records.

These are useful reproducibility/evidence artifacts and are not the cause of a ~396 MB repository.

### Retained ZKML artifacts — KEEP

Representative retained files are small:

- `zkml/ezkl/model.compiled` — about 349 KB;
- `zkml/ezkl/calibration.json` — about 332 KB;
- `zkml/ezkl/verification_key.vk` — about 67 KB;
- `zkml/models/proxy.onnx` — about 44 KB;
- `zkml/models/proxy_best.pt` — about 45 KB.

These support the retained proxy/proof reproducibility boundary and are not meaningful repository-bloat drivers.

### R4 review bundles — KEEP

Representative protected bundles are also small:

- GAP-002 blind-review ZIP — about 382 KB;
- GAP-007 candidate-2 independent-review ZIP — about 4 KB.

They are evidence artifacts and must not be removed for cosmetic size reduction.

### Dependency locks / structured snapshots — KEEP

Root, ML, and AGENTS lockfiles are sub-megabyte package-resolution artifacts. ML training snapshots inspected are structured metrics/metadata rather than large checkpoint binaries. Their reproducibility value outweighs their size contribution.

### Current generated/runtime material — KEEP OUT OF GIT

The forward policy is to keep normal Git history free of:

- teacher/model checkpoints under ML (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`);
- generated checkpoint/model-output directories;
- raw/full datasets and generated graph/token corpora;
- DVC cache/runtime state;
- machine-local runtime databases/logs;
- large proving keys/SRS or regeneratable proving material;
- unbounded generated reports unless explicitly promoted as compact evidence.

`ml/.gitignore` and root ignore policy have been hardened accordingly.

## 3. Historical-storage evidence

Repository history shows earlier development phases creating and later cleaning processed DATA/ML artifacts. Historical inspection confirmed that some processed artifacts (for example a deduplicated multilabel CSV) did exist in Git at earlier commits, while other very large local training artifacts (for example a later multi-GB cached dataset) were not Git-tracked at the checked historical commit.

Therefore not every large local ML/DATA artifact contributed to Git history, but historical tracked data/config/report artifacts accumulated before the current hygiene rules matured.

The important distinction is:

**large GitHub repository size ≠ large current checkout artifact set.**

## 4. Why history rewrite is rejected

Tools such as `git filter-repo`, BFG, or equivalent force-pushed history could potentially remove old blobs, but they also rewrite commit identities.

For SENTINEL, exact commit identities are used as provenance anchors across:

- R4 plans and status records;
- accepted ADRs and physical-lineage decisions;
- reproducibility checkpoints;
- audit/review handoffs;
- evidence manifests and historical analysis.

A safe rewrite would therefore require a dedicated migration that maps and repairs every affected reference, revalidates the evidence chain, and preserves a pre-rewrite archive. That is not ordinary portfolio hygiene.

**Prohibited for normal professionalization:**

- `git filter-repo` merely to make GitHub's size number smaller;
- BFG/history-surgery equivalents;
- force-pushing rewritten `main` without a separately approved evidence-preserving migration.

## 5. Non-destructive clone mitigation

For reviewers/developers who do not need every historical blob immediately, prefer Git partial clone:

```bash
git clone --filter=blob:none https://github.com/motafegh/sentinel-.git
```

This preserves real commit/history identity while deferring historical blob transfer until needed. A normal clone remains valid when full local history is desired.

The same option is documented in `DEVELOPMENT.md` and the public README.

## 6. If history compaction is reconsidered later

It is authorized only as a separate migration after all of the following:

1. a local mirror/object inventory identifies exact historical blobs and their contribution;
2. every still-required heavy artifact has a retained verified copy/checksum;
3. every affected R4/ADR/evidence commit reference is mapped;
4. a complete backup/tag/archive of pre-rewrite history exists;
5. the operational benefit materially outweighs migration risk;
6. all public/current references are repaired and revalidated afterward.

Unless repository size becomes a real operational blocker, this work is optional.

## 7. M-011 disposition

| Area | Result | Action |
|---|---|---|
| current tracked ML checkpoints/models | controlled | keep ignore policy |
| current physical DATA / generated corpora | not normal Git content | keep DVC/local/artifact boundary |
| compact DATA authority artifacts | justified | keep tracked |
| retained ZKML artifacts | small + evidential | keep tracked |
| R4 review bundles | small + evidential | keep tracked |
| runtime/generated cruft | cleaned/ignored | no further deletion now |
| repository-level ~396 MB | primarily historical concern | accept; use partial clone where useful |
| Git-history rewrite | high provenance risk | reject for normal cleanup |

**M-011: CLOSED.**

This file is the single canonical portfolio-professionalization record for repository weight/artifact/history policy. A second overlapping audit file was removed during the 2026-09-05 reconciliation to avoid competing near-authorities.
