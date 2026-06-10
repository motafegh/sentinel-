"""Preprocessing service — run the 5-step pipeline for one or all enabled sources.

Modes:
  - default:          run the full pipeline over all .sol files in the manifest
  - --sample N:       run on the first N files (fast iteration)
  - --retry-failed:   run ONLY the files listed in the existing dropped.csv
                      (typically after installing a missing solc version or
                      fixing a config bug). The new run's results are MERGED
                      with the existing preprocessed output: files that now
                      succeed are added to preprocessed/, files that still
                      fail stay in dropped.csv with their (possibly updated)
                      error message.

The retry-failed mode is a build-system-style incremental: install solc 0.7.4
→ run retry-failed → 50 files move from dropped to preprocessed. No need to
reprocess 22,000 files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sentinel_data.ingestion.ingest import _enabled_sources
from sentinel_data.preprocessing.pipeline import PreprocessingPipeline


def preprocess_source(
    name: str,
    cfg: dict,
    data_dir: Path,
    dry_run: bool = False,
    n_workers: int = 1,
    sample: int | None = None,
    retry_failed: bool = False,
) -> None:
    sources = _enabled_sources(cfg)
    if name not in sources:
        raise ValueError(f"Source '{name}' not found or not enabled in config.yaml")

    raw_dir = data_dir / "raw" / name
    out_dir = data_dir / "preprocessed" / name
    manifest_path = raw_dir / "ingestion_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Ingestion manifest {manifest_path} not found. "
            f"Run `sentinel-data ingest --source {name}` first."
        )

    sol_files: list[Path] = []
    if retry_failed:
        sol_files, previously_dropped = _load_dropped_files(out_dir, raw_dir)
        if not sol_files:
            print(f"  retry-failed : no dropped.csv or empty; nothing to retry")
            return
        print(f"  retry-failed : {len(sol_files)} files from previous dropped.csv")
    else:
        # Read the manifest's file list — this respects include_subdirs / exclude_subdirs
        # that the connector honored at ingest time. Re-scanning the raw dir would defeat
        # the scoping (e.g. SolidiFI's results/ subdirs).
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["files"]:
            rel = entry["path"]
            # Manifest paths are relative to raw_dir (e.g. "repo/buggy_contracts/x.sol")
            sol_files.append(raw_dir / rel)

        if sample is not None and sample > 0 and sample < len(sol_files):
            sol_files = sol_files[:sample]
            print(f"  sample       : limited to first {sample} files (--sample)")

        print(f"  found       : {len(sol_files)} .sol files (from manifest)")

    if dry_run:
        print("  (dry-run — no files written)")
        return

    # Folderize runs only in the full mode (retry-failed already has labels set up)
    if not retry_failed:
        _maybe_folderize(raw_dir, sources[name], name=name)

    pipeline = PreprocessingPipeline(name, out_dir)
    if n_workers and n_workers > 1:
        from sentinel_data.preprocessing.parallel import run_preprocess_parallel
        result = run_preprocess_parallel(
            pipeline, sol_files, raw_base=raw_dir, n_workers=n_workers
        )
    else:
        result = pipeline.run(sol_files, raw_base=raw_dir)

    if retry_failed:
        _merge_retry_results(out_dir, result, previously_dropped)
    else:
        # The pipeline's run() already wrote dropped.csv; nothing more to do
        pass

    print(f"  processed   : {len(result.processed)}")
    print(f"  dropped     : {len(result.dropped)}")
    print(f"  duration    : {result.duration_s:.1f}s")
    print(f"  output      : {out_dir}")


def _load_dropped_files(
    out_dir: Path, raw_dir: Path,
) -> tuple[list[Path], list[dict]]:
    """Read the existing dropped.csv and return (paths, rows).

    Paths are reconstructed as raw_dir / original_path (the same convention
    the pipeline uses). If a path no longer exists on disk, it's silently
    skipped (the file may have been removed during a re-ingest).
    """
    dropped_csv = out_dir / "dropped.csv"
    if not dropped_csv.exists():
        return [], []
    with open(dropped_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    paths: list[Path] = []
    for row in rows:
        rel = row.get("original_path", "")
        if not rel:
            continue
        p = raw_dir / rel
        if p.exists():
            paths.append(p)
    return paths, rows


def _merge_retry_results(
    out_dir: Path,
    new_result,  # PipelineResult
    previously_dropped: list[dict],
) -> None:
    """Merge retry-failed results into the existing preprocessed state.

    Behavior:
      - Files that NOW succeed: their .sol + .meta.json are already written
        to out_dir/ by the pipeline. We just need to remove them from
        dropped.csv. We discover their `original_path` by reading each
        newly-written .meta.json (the pipeline writes that field).
      - Files that STILL fail: their new error message overwrites the old
        one in dropped.csv (so users see the latest diagnostic).
      - Files in the OLD dropped.csv that weren't retried (e.g. their path
        no longer exists on disk): preserved as-is.

    Implementation: rebuild dropped.csv from scratch by joining
    `previously_dropped` (the old rows) with `new_result.dropped` (the new
    failures), keyed by `original_path`.
    """
    # Build the set of "now succeeds" original_paths by reading each
    # newly-written meta.json in out_dir/.
    new_successes_by_path: set[str] = set()
    for processed_path in new_result.processed:
        # processed_path is out_dir/<sha256>.sol
        meta_path = processed_path.with_suffix(".meta.json")
        if not meta_path.exists():
            # No meta — shouldn't happen in normal flow, but be defensive
            continue
        try:
            meta = json.loads(meta_path.read_text())
            rel = meta.get("original_path", "")
            if rel:
                new_successes_by_path.add(rel)
        except (json.JSONDecodeError, OSError):
            continue

    new_drops_by_path: dict[str, dict] = {
        r.get("original_path", ""): r for r in new_result.dropped
    }

    merged: list[dict] = []
    for old_row in previously_dropped:
        rel = old_row.get("original_path", "")
        if rel in new_successes_by_path:
            # Now succeeds — drop it from dropped.csv
            continue
        if rel in new_drops_by_path:
            # Still failing — use the NEW error message (more up-to-date)
            merged.append(new_drops_by_path[rel])
        else:
            # Wasn't retried (file no longer exists, etc.) — keep old row
            merged.append(old_row)

    if merged:
        from sentinel_data.preprocessing.pipeline import _write_dropped
        _write_dropped(out_dir / "dropped.csv", merged)
    else:
        # All retries succeeded; remove the file entirely
        dropped_csv = out_dir / "dropped.csv"
        if dropped_csv.exists():
            dropped_csv.unlink()


def _maybe_folderize(raw_dir: Path, entry: dict, name: str) -> None:
    """Run label-aware folderization for sources that have a separate labels CSV.

    Triggered by `labels_csv` (path, relative to repo root or absolute) and
    `label_id_column` + `class_columns` in the source's config entry.
    Idempotent — re-running with the same labels is a no-op.
    """
    labels_csv_rel = entry.get("labels_csv")
    if not labels_csv_rel:
        return
    labels_csv = Path(labels_csv_rel)
    if not labels_csv.is_absolute():
        # Try relative to data_dir (i.e. /home/.../Data/); fall back to raw_dir
        candidates = [
            raw_dir.parent.parent / labels_csv,    # Data/labels.csv
            raw_dir / labels_csv,
        ]
        labels_csv = next((c for c in candidates if c.exists()), candidates[0])
    if not labels_csv.exists():
        print(f"  folderize   : SKIP — labels_csv not found at {labels_csv}")
        return

    from sentinel_data.ingestion.label_folderize import folderize_by_labels

    repo_dir = raw_dir / "repo"
    id_col = entry.get("label_id_column", "contractID")
    class_cols = entry.get("class_columns") or []
    source_subdir = entry.get("label_source_subdir", "__source__")

    if not class_cols:
        print(f"  folderize   : SKIP — no class_columns in config entry")
        return

    print(f"  folderize   : {labels_csv.name} → {len(class_cols)} class folders")
    result = folderize_by_labels(
        repo_dir=repo_dir,
        labels_csv=labels_csv,
        id_column=id_col,
        class_columns=class_cols,
        source_subdir=source_subdir,
    )
    print(f"                contracts: {result.contracts_seen}, "
          f"symlinks: {result.symlinks_created}, "
          f"multi-label: {result.multi_label}")
    print(f"                classes: {', '.join(result.classes_present)}")


def preprocess_all(
    cfg: dict,
    data_dir: Path,
    dry_run: bool = False,
    n_workers: int = 1,
    sample: int | None = None,
    retry_failed: bool = False,
) -> None:
    for name in _enabled_sources(cfg):
        print(f"\n[preprocess] {name}")
        try:
            preprocess_source(name, cfg, data_dir, dry_run=dry_run,
                              n_workers=n_workers, sample=sample,
                              retry_failed=retry_failed)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
