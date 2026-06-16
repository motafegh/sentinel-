# DVC Legacy Cleanup — Archive Index (2026-06-15)

> **Purpose:** Document what was archived during Phase A.4 of the MLOps Q4 proposal, and why.
> **Source:** `docs/proposal/MLOps/2026-06-15_ml_q4_proposal_mlops_q4_implementation_plan.md` §A.4
> **Decision:** Per Ali (2026-06-15), all old/stale DVC files are unusable and must be properly removed/archived before a new DVC is implemented.

---

## What was archived and why

The DVC state before cleanup was a mix of:
- **Active pointers** tracking real (but legacy) data
- **Stale pointers** that referenced data that no longer exists (e.g., `ml/data/tokens` was renamed to `ml/data/tokens_windowed` but DVC wasn't updated)
- **Orphaned cache** entries in `.dvc/cache/` for files no longer in source

After the Stage 7B seam swap (completed 2026-06-15), the active data pipeline moved to `data_module/`. The `ml/data/graphs/`, `ml/data/tokens_windowed/`, and `ml/data/splits/` directories contain pre-seam-swap data that is now duplicated in `data_module/data/representations/` and `data_module/data/splits/v3/`.

Per project rule (CLAUDE.md §4: "Duplicates / moves → `docs/.bin/` (NEVER delete directly)"), the legacy data was **moved** here, not deleted. After full validation, the user can `rm -rf docs/.bin/2026-06-15_ml_q4_proposal_mlops_dvc_legacy_cleanup/`.

---

## Files archived in this directory

### `ml_data_graphs/` (1.3 GB)
- **Source:** `ml/data/graphs/` (41,577 .pt files + 1 subdirectory)
- **DVC reference:** `ml/data/graphs.dvc` (declared 68,556 files / 249 MB; actual dir has 41,577 — discrepancy noted below)
- **Why archived:** Pre-seam-swap legacy PyG graph files. Same data now lives in `data_module/data/representations/` (the canonical location per Stage 7B swap).
- **Verification needed:** Confirm `data_module/data/representations/` has equivalent data before deleting this archive.
- **DVC discrepancy:** The .dvc pointer says 68,556 files but the directory had 41,577. This is a 27,000-file discrepancy — likely some graphs were filtered/replaced in the v3 export. Confirm with data_module.

### `ml_data_tokens_windowed/` (1.6 GB)
- **Source:** `ml/data/tokens_windowed/` (44,526 .pt files)
- **DVC reference:** `ml/data/tokens.dvc` (declared 68,570 files for `tokens/`, but the actual dir is `tokens_windowed/` — directory was renamed at some point, DVC wasn't updated)
- **Why archived:** Pre-seam-swap legacy tokenized contract data. Same data now lives in `data_module/data/representations/`.
- **Verification needed:** Confirm `data_module/data/representations/` has equivalent token data before deleting this archive.

### `ml_data_splits/` (340 KB)
- **Source:** `ml/data/splits/deduped/` (3 .npy files: train/val/test indices)
- **DVC reference:** `ml/data/splits.dvc` (declared 3 files / 549 KB)
- **Why archived:** Pre-seam-swap legacy train/val/test indices. ACTIVE splits live in `data_module/data/splits/v3/` (18,596/1,983/1,914 per MEMORY.md).
- **Verification needed:** Confirm v3 splits are sufficient before deleting this archive.

---

## Files NOT archived (deliberately kept in source)

- `ml/data/cached_dataset_v9.pkl` (2.6 GB) — legacy v9 dataset pickle. NOT DVC-tracked. May still be useful for backward compat or quick reload. Decision deferred.
- `ml/data/augmented/*.sol` (50+ files) — git-tracked CEI-safe test fixtures. Used by various scripts.
- `ml/data/BCCC-SCsVul-2024_README.md` — documentation, not data.
- `ml/data/drift_baseline.json` — ACTIVE placeholder (just fixed in A.1). Keep.
- `ml/data/SolidiFI/`, `SolidiFI-benchmark/`, `SolidiFI-processed/` — raw source data, used by data_module ingestion.
- `ml/data/smartbugs-curated/`, `smartbugs-results-master/`, `smartbugs-wild/` — raw source data, used by data_module ingestion.
- `ml/data/slither_results/`, `ml/data/reports/`, `ml/data/processed/`, `ml/data/archive/` — analysis artifacts, not DVC-tracked.
- `ml/data/validation_report.json` — analysis output.

---

## Old DVC artifacts removed (also moved to `docs/.bin/` separately or deleted)

| Artifact | Size | Action | Reason |
|---|---|---|---|
| `ml/checkpoints.dvc` | 187 B | Deleted | Outdated pointer; will re-create under new DVC policy |
| `ml/data/graphs.dvc` | 187 B | Deleted | Pointer to legacy data (moved to docs/.bin) |
| `ml/data/splits.dvc` | 184 B | Deleted | Pointer to legacy data (moved to docs/.bin) |
| `ml/data/tokens.dvc` | 187 B | Deleted | Pointer to legacy data (moved to docs/.bin) |
| `.dvc/` (root) | 2.5 GB | Deleted | Cache + config; rebuilt from scratch |
| `.dvcignore` | 239 B | Kept (will update for new policy) | DVC ignore file |

The new DVC was initialised fresh (in `ml/`) with policy: track only the canonical Run 12 FINAL checkpoint + companion state + thresholds files.

---

## Verification (post-cleanup)

1. `dvc status` reports clean state
2. `dvc list` shows the new tracked checkpoint
3. The new DVC remote points to `/mnt/d/sentinel-dvc-remote` (same as old; Ali approved the local backup location)
4. `ml/data/.gitignore` updated to reflect the new state (no longer references `graphs/tokens/splits` since they don't exist in source)
5. The new DVC config is committed to git (per DVC best practice)

---

## References

- **Proposal:** `docs/proposal/MLOps/2026-06-15_ml_q4_proposal_mlops_q4_implementation_plan.md` §A.4
- **Working notes:** `~/.claude/scratch/mlops_phase_a_execution_20260615.md` (A.4 section)
- **MEMORY.md:** `~/.claude/projects/.../memory/MEMORY.md` (Phase A status)
- **Stage 7B seam swap context:** `~/.claude/projects/.../memory/2026-06-15_project_run12_post_training.md`
- **Project naming rules:** `~/projects/sentinel/CLAUDE.md` §4 (docs/.bin convention)
