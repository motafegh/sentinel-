"""Preprocessing service — run the 5-step pipeline for one or all enabled sources."""

from __future__ import annotations

from pathlib import Path

from sentinel_data.ingestion.ingest import _enabled_sources
from sentinel_data.preprocessing.pipeline import PreprocessingPipeline


def preprocess_source(
    name: str,
    cfg: dict,
    data_dir: Path,
    dry_run: bool = False,
) -> None:
    sources = _enabled_sources(cfg)
    if name not in sources:
        raise ValueError(f"Source '{name}' not found or not enabled in config.yaml")

    raw_dir = data_dir / "raw" / name / "repo"
    out_dir = data_dir / "preprocessed" / name

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dir {raw_dir} not found. Run `sentinel-data ingest --source {name}` first."
        )

    sol_files = sorted(raw_dir.rglob("*.sol"))
    print(f"  found       : {len(sol_files)} .sol files in {raw_dir}")

    if dry_run:
        print("  (dry-run — no files written)")
        return

    pipeline = PreprocessingPipeline(name, out_dir)
    result = pipeline.run(sol_files, raw_base=raw_dir.parent)

    print(f"  processed   : {len(result.processed)}")
    print(f"  dropped     : {len(result.dropped)}")
    print(f"  duration    : {result.duration_s:.1f}s")
    print(f"  output      : {out_dir}")


def preprocess_all(
    cfg: dict,
    data_dir: Path,
    dry_run: bool = False,
) -> None:
    for name in _enabled_sources(cfg):
        print(f"\n[preprocess] {name}")
        try:
            preprocess_source(name, cfg, data_dir, dry_run=dry_run)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
