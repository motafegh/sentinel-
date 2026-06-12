"""Multiprocessing wrapper for PreprocessingPipeline.

The serial pipeline iterates one file at a time. For large sources
(DIVE = 22,330 files, Web3Bugs = ~3,500, SmartBugs Wild = 47K), serial
processing is 6+ hours. A multiprocessing pool cuts this by N where N
is the number of workers.

This module mirrors the pattern used by `ml/src/data_extraction/ast_extractor.py`:
  - Module-level picklable worker (lambdas and bound methods don't pickle)
  - `mp.Pool(processes=n_workers).imap(worker, batch, chunksize=...)`
  - chunksize auto-tuned: max(1, total//(n_workers*16))

Why this is correct (and not just "parallelize and pray"):
  - Each file is independent (compile + dedup are per-file; dedup is
    stateful but stateful dedup races are a known acceptable trade-off
    for this stage — duplicates are dropped, not corrupted)
  - The output dir is shared and writes use a content-addressed filename
    (sha256), so concurrent writers target different files (no conflict)
  - The temp-file (for stripped imports) is in the source's own dir, so
    no global temp namespace conflicts

What this is NOT:
  - Not a true dedup-aware parallel pipeline. Two workers might both
    process the same file in the first 10K before either has recorded
    the SHA. We accept this — the dedup is best-effort, not exact.
  - Not for the REPRESENTATION stage. That comes in Stage 2 and has its
    own parallelism story (per-graph extraction is GIL-bound; needs
    `concurrent.futures.ThreadPoolExecutor` or a different shape).

Added 2026-06-10 for the DIVE integration test (22K files at ~1s each
= 6+ hours serial). Mirrors the pattern in ast_extractor.py (the
Stage-2/Stage-0 reference implementation in ml/).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sentinel_data.preprocessing.pipeline import PreprocessingPipeline, PipelineResult


# Module-level worker — must be picklable. Bound methods and lambdas don't
# pickle cleanly across processes. The pipeline object is also stateful
# (it has a Deduplicator), so we create a fresh one per worker.
def _process_one_worker(args: tuple) -> dict:
    """Picklable multiprocessing worker.

    Accepts (source_name, out_dir_str, sol_path_str, raw_base_str) and
    returns a dict that's easy to aggregate in the parent process.
    """
    source_name, out_dir_str, sol_path_str, raw_base_str = args
    pipeline = PreprocessingPipeline(source_name, Path(out_dir_str))
    sol_path = Path(sol_path_str)
    raw_base = Path(raw_base_str)
    try:
        outcome = pipeline._process_one(sol_path, raw_base)
    except Exception as e:
        # Don't let a single file kill the whole pool
        return {"status": "error", "path": str(sol_path), "error": repr(e)}
    if outcome is None:
        return {"status": "skip", "path": str(sol_path)}
    out_path, meta, drop_row = outcome
    if drop_row:
        return {"status": "drop", "path": str(sol_path), "drop_row": drop_row}
    return {"status": "ok", "path": str(sol_path), "out_path": str(out_path), "meta": meta}


def _chunksize(total: int, n_workers: int) -> int:
    """Auto-tune pool.imap chunksize. Larger = less IPC overhead but less
    load balancing. 16 chunks per worker is a common sweet spot."""
    if total <= 0 or n_workers <= 0:
        return 1
    return max(1, total // (n_workers * 16))


def run_preprocess_parallel(
    pipeline: PreprocessingPipeline,
    sol_files: list[Path],
    raw_base: Path,
    n_workers: int | None = None,
) -> PipelineResult:
    """Run `pipeline._process_one` in parallel over `sol_files`.

    Args:
        pipeline: a PreprocessingPipeline instance (used for its config;
                  workers create their own pipeline instances).
        sol_files: list of .sol paths to process.
        raw_base:  base path for resolving relative original_path.
        n_workers: pool size. Defaults to min(os.cpu_count(), 8) — the
                  plan calls for 8 cores; cap at 8 even on bigger boxes
                  because solc subprocesses are themselves multi-threaded.

    Returns:
        PipelineResult aggregating across all workers.
    """
    import time

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 8)

    out_dir = pipeline.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    args_iter = (
        (pipeline.source_name, str(out_dir), str(p), str(raw_base))
        for p in sol_files
    )
    chunksize = _chunksize(len(sol_files), n_workers)

    processed: list[Path] = []
    dropped: list[dict] = []
    errors: list[dict] = []

    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap(_process_one_worker, args_iter, chunksize=chunksize):
            status = result.get("status")
            if status == "ok":
                # Write the meta.json in the parent process (the worker
                # returned the meta object). Using a side-channel avoids
                # sharing a file writer across processes.
                from sentinel_data.preprocessing.pipeline import _write_meta
                _write_meta(Path(result["out_path"]).with_suffix(".meta.json"), result["meta"])
                processed.append(Path(result["out_path"]))
            elif status == "drop":
                dropped.append(result["drop_row"])
            elif status == "error":
                errors.append(result)
            # "skip" is silently ignored (matches the serial pipeline behavior)

    if dropped:
        from sentinel_data.preprocessing.pipeline import _write_dropped
        _write_dropped(out_dir / "dropped.csv", dropped)

    if errors:
        # Surface worker errors in dropped.csv too (with a distinctive reason)
        for e in errors:
            dropped.append({
                "source": pipeline.source_name,
                "original_path": e["path"],
                "pragma": "",
                "reason": "worker_exception",
                "error": e["error"][:300],
            })
        from sentinel_data.preprocessing.pipeline import _write_dropped
        _write_dropped(out_dir / "dropped.csv", dropped)

    return PipelineResult(
        processed=processed,
        dropped=dropped,
        duration_s=time.monotonic() - t0,
    )
