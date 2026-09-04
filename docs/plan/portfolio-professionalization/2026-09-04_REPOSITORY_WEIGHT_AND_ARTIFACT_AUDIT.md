# SENTINEL Repository Weight and Artifact Audit

**Date:** 2026-09-04  
**Status:** COMPLETE — current-tree audit and history-risk classification  
**Branch:** `portfolio/professionalization-2026-09-02`  
**Scope:** portfolio-professionalization item M-011  

## 1. Conclusion

SENTINEL's remaining GitHub repository weight is primarily a **Git-history/storage concern**, not evidence that the current portfolio branch still contains hundreds of megabytes of active model/data binaries.

GitHub currently reports repository size at approximately **405,887 KB**. By contrast, inspection of the current professionalization branch found no active raw-dataset top-level tree and no large ML checkpoint/model directory. The largest intentionally tracked current artifacts inspected are low-single-digit MiB rather than hundreds of MiB.

This means the correct response is **not** to delete current R4 evidence or rewrite history casually. The current tree should stay evidence-preserving, while any later history compaction must be treated as a separate migration with backup, artifact-retention proof, and reference-impact analysis.

## 2. Current-tree observations

### DATA/R4 tracked export

`data_module/data/exports/sentinel-r4-vnext-v1/` intentionally contains compact machine-readable authority/compatibility artifacts. Examples:

- `label_states.parquet` — 1,900,672 bytes;
- `ml_targets.parquet` — 1,310,473 bytes;
- associated manifest, validation, binding, source, and evidence records are small metadata files.

These are compact, directly useful, and tied to reproducibility/evidence. **KEEP.**

### ZKML retained bundle

The retained ZKML material is also compact:

- `zkml/ezkl/model.compiled` — 348,587 bytes;
- `zkml/ezkl/calibration.json` — 332,227 bytes;
- `zkml/ezkl/verification_key.vk` — 66,823 bytes;
- `zkml/models/proxy.onnx` — 43,644 bytes;
- `zkml/models/proxy_best.pt` — 45,414 bytes.

These files support the retained proxy/proof reproducibility boundary and are not material repository-bloat drivers. **KEEP.**

### R4 review bundles

The protected review bundles are small:

- GAP-002 blind-review ZIP — 382,393 bytes;
- GAP-007 candidate-2 independent-review ZIP — 3,567 bytes;
- corresponding SHA-256 records are tiny.

They are evidence artifacts and must not be removed for cosmetic repository-size reduction. **KEEP.**

### Dependency locks

Root, ML, and AGENTS lockfiles are sub-megabyte package-resolution artifacts. Their reproducibility value outweighs their negligible size contribution. **KEEP.**

### ML training snapshots

The current `ml/training_snapshots/` content inspected is structured metrics/metadata rather than checkpoint binaries. For example, the Run12 snapshot contains JSONL metrics and metadata, with the largest inspected file (`epoch_summary.jsonl`) about 278 KB. **KEEP** as lightweight historical/evaluation evidence.

## 3. Current-tree versus historical storage

The repository's GitHub size (~405,887 KB) is far larger than the individual active artifacts above. GitHub repository size includes Git object/history storage; removing a file from the latest tree does not remove its historical blob from earlier commits.

The current branch has already removed or excluded many machine-local, generated, raw-data, cache, and runtime paths through earlier professionalization/hygiene work. Therefore a large residual GitHub-size number should not be interpreted as proof that those files are still present in the current checkout.

A complete object-by-object historical size ranking normally requires a local mirror/clone and commands such as `git rev-list --objects --all` plus `git cat-file`. That analysis could not be performed from the current remote-only execution environment because direct Git clone/network access is unavailable here. The GitHub API evidence is sufficient for the current-tree classification, but not for a safe historical rewrite plan.

## 4. Decisions

### KEEP in Git

Keep:

- protected `docs/plan/ml-R4/` evidence, ADRs, manifests, and review bundles;
- compact DATA semantic/export artifacts required for reproducibility and traceability;
- retained lightweight ZKML proxy/proof/verifier artifacts;
- source, tests, configs, handbook, plans, and dependency lockfiles;
- lightweight structured training/evaluation snapshots.

### KEEP OUT of future Git commits

Do not add:

- teacher/model checkpoints under `ml/` (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`);
- checkpoint/model-output directories;
- raw/full datasets or generated graph/token corpora;
- DVC cache/runtime state;
- proving keys/SRS or other large regeneratable/local proving material;
- runtime databases/logs/reports unless explicitly promoted as bounded evidence.

Heavy model/data artifacts should use an explicit artifact store, DVC/LFS/release asset, or another versioned external mechanism with recorded hashes and acquisition instructions before the public portfolio claims them as reproducible.

## 5. History rewrite decision

**Do not rewrite Git history during this professionalization chunk.**

A history rewrite would change commit SHAs across a repository whose R4 evidence, plans, decisions, and review records frequently bind to exact commits. It therefore has a much larger integrity cost than ordinary cosmetic cleanup.

History compaction may be considered later only if all of the following are satisfied:

1. a local mirror/object inventory identifies the exact large historical blobs and their contribution;
2. every still-required heavy artifact has a verified retained copy and checksum;
3. protected R4 evidence/reference implications are mapped;
4. a full backup/tag/archive of pre-rewrite history exists;
5. the size reduction is materially worth the migration cost;
6. all public/current references are repaired and validated afterward.

Until then, the professional repository should optimize the **current tree and future commit discipline**, not erase historical evidence.

## 6. Preventive control added with this audit

`ml/.gitignore` is strengthened so future ML checkpoint/model binaries and checkpoint output directories remain local/artifact-managed by default. This is prevention, not deletion of accepted evidence.

## 7. M-011 disposition

**M-011 current-tree/artifact audit: CLOSED.**

Remaining optional follow-up: a local full-history object inventory and, only if justified by its results, a separately authorized history-compaction migration. That follow-up is not required to continue the portfolio professionalization work because the current branch itself is already free of the major classes of generated/raw/checkpoint bloat identified by this audit.
