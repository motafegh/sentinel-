#!/usr/bin/env python3
"""
reextract_graphs.py — Parallel re-extraction of all graph .pt files (v5.1 Phase 1).

After v5.1 Phase 0 fixes (_select_contract interface bug, CFG failure counter),
this script re-extracts every graph in multilabel_index.csv using
multiprocessing.Pool (one worker per CPU), overwriting existing .pt files.
Token .pt files and the CSV are left untouched.

Architecture
────────────
  Main process:  builds MD5→path map, dispatches work via imap_unordered,
                 collects results, writes checkpoint, shows tqdm bar.
  Worker process: stateless — detects pragma, resolves solc binary,
                 calls extract_contract_graph, writes atomically (.tmp→rename).
                 No shared state between workers; each spawns its own solc.

Safety
──────
  Atomic writes:    torch.save → .tmp → rename  (crash-safe)
  Checkpointing:    every 1000 contracts; resume skips already-done MD5s
  No race conditions: each MD5 maps to exactly one worker task; unique out paths

USAGE
─────
  source ml/.venv/bin/activate
  PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py
  PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py --resume
  PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py --max-contracts 200
  PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py --workers 8

AFTER THIS SCRIPT
─────────────────
  python ml/scripts/validate_graph_dataset.py --check-dim 12 --check-edge-types 7
  python ml/scripts/create_cache.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
_SOLC_ARTIFACTS = PROJECT_ROOT / "ml" / ".venv" / ".solc-select" / "artifacts"

SOURCE_DIRS = [
    PROJECT_ROOT / "BCCC-SCsVul-2024" / "SourceCodes",
    PROJECT_ROOT / "ml" / "data" / "SolidiFI-processed",
    PROJECT_ROOT / "ml" / "data" / "SolidiFI",
    PROJECT_ROOT / "ml" / "data" / "smartbugs-curated",
    PROJECT_ROOT / "ml" / "data" / "smartbugs-wild",
    PROJECT_ROOT / "ml" / "data" / "augmented",
]

DEFAULT_CSV        = PROJECT_ROOT / "ml" / "data" / "processed" / "multilabel_index.csv"
DEFAULT_GRAPHS     = PROJECT_ROOT / "ml" / "data" / "graphs"
DEFAULT_CHECKPOINT = DEFAULT_GRAPHS / "reextract_checkpoint.json"
CHECKPOINT_EVERY   = 1000   # flush checkpoint to disk every N completions

_PRAGMA_RE = re.compile(r'pragma\s+solidity\s+[\^~>=<\s]*(\d+\.\d+\.\d+)')

# Use the latest available patch per minor series.
# Contracts often declare an older minimum (e.g. ^0.4.18) but use syntax
# introduced in later patches (emit, constructor(), etc.). Using the latest
# patch of the same minor compiles those while staying in the declared range.
_LATEST_PATCH = {
    "0.4": "0.4.26",
    "0.5": "0.5.17",
    "0.6": "0.6.12",
    "0.7": "0.7.6",
    "0.8": "0.8.31",
}


# ── Helpers (used in both main and worker processes) ──────────────────────────

def _md5_path(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    return hashlib.md5(str(rel).encode("utf-8")).hexdigest()


def _detect_solc_version(sol_path: Path) -> str:
    """
    Detect solc version from pragma. Returns the LATEST PATCH of the declared
    minor series so that forward syntax (emit, constructor(), etc.) compiles.
    E.g. '^0.4.18' → '0.4.26', '^0.5.0' → '0.5.17'. Defaults to 0.8.31.
    """
    try:
        txt = sol_path.read_text(encoding="utf-8", errors="replace")
        m = _PRAGMA_RE.search(txt)
        if m:
            minor = ".".join(m.group(1).split(".")[:2])
            return _LATEST_PATCH.get(minor, m.group(1))
    except OSError:
        pass
    return "0.8.31"


def _solc_binary(version: str) -> Optional[Path]:
    """
    Return path to the versioned solc binary inside the venv's solc-select
    artifacts directory, or None if not found (Slither falls back to PATH).
    Layout: .solc-select/artifacts/solc-X.Y.Z/solc-X.Y.Z
    """
    binary = _SOLC_ARTIFACTS / f"solc-{version}" / f"solc-{version}"
    return binary if binary.exists() else None


def _build_md5_to_path(target_md5s: set[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    scanned = 0
    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            continue
        for sol in src_dir.rglob("*.sol"):
            scanned += 1
            md5 = _md5_path(sol)
            if md5 in target_md5s:
                mapping[md5] = sol
    logger.info(f"Scan complete — {scanned} files scanned, {len(mapping)}/{len(target_md5s)} matched")
    return mapping


def _load_target_md5s(csv_path: Path) -> set[str]:
    md5s: set[str] = set()
    with csv_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            md5s.add(line.split(",")[0].strip())
    return md5s


# ── Worker (runs in child process) ───────────────────────────────────────────
#
# Receives a (md5, sol_path_str, graphs_dir_str) tuple.
# Returns (md5, status, detail) where status is "ok" | "ghost" | "skip" | "fail".
# No shared mutable state — each call is fully independent.

def _worker(args: Tuple[str, str, str]) -> Tuple[str, str, str]:
    md5, sol_path_str, graphs_dir_str = args

    # Import inside worker so it works with both fork and spawn start methods.
    # With fork this is a no-op (already imported); explicit import is harmless.
    from ml.src.preprocessing.graph_extractor import (
        GraphExtractionConfig,
        extract_contract_graph,
        EmptyGraphError,
        SolcCompilationError,
        SlitherParseError,
    )

    sol_path   = Path(sol_path_str)
    graphs_dir = Path(graphs_dir_str)
    out_path   = graphs_dir / f"{md5}.pt"

    solc_ver = _detect_solc_version(sol_path)
    cfg = GraphExtractionConfig(
        solc_version=solc_ver,
        solc_binary=_solc_binary(solc_ver),
    )

    try:
        g = extract_contract_graph(sol_path, cfg)
    except (EmptyGraphError, SolcCompilationError, SlitherParseError) as exc:
        return md5, "skip", str(exc)[:80]
    except Exception as exc:
        return md5, "fail", str(exc)[:80]

    status = "ghost" if g.num_nodes <= 3 else "ok"

    tmp = out_path.with_suffix(".tmp")
    torch.save(g, tmp)
    tmp.rename(out_path)   # atomic on same filesystem

    return md5, status, f"nodes={g.num_nodes}"


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel re-extraction of graph .pt files (v5.1 Phase 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--multilabel-csv",  type=Path,  default=DEFAULT_CSV)
    p.add_argument("--graphs-dir",      type=Path,  default=DEFAULT_GRAPHS)
    p.add_argument("--checkpoint",      type=Path,  default=DEFAULT_CHECKPOINT)
    p.add_argument("--workers",         type=int,   default=max(1, mp.cpu_count() - 1),
                   help="Worker processes (default: cpu_count-1)")
    p.add_argument("--chunksize",       type=int,   default=1,
                   help="imap_unordered chunksize — 1 gives best load balance for slow Slither jobs")
    p.add_argument("--resume",          action="store_true",
                   help="Skip MD5s already recorded in checkpoint file")
    p.add_argument("--max-contracts",   type=int,   default=None, metavar="N")
    p.add_argument("--dry-run",         action="store_true",
                   help="Map and count only — no extraction, no writes")
    p.add_argument("--verbose",         action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    logger.info("v5.1 Phase 1 — Parallel re-extraction")
    logger.info(f"Workers: {args.workers}  chunksize: {args.chunksize}")
    if args.dry_run:
        logger.info("DRY RUN — no files written")

    if not args.multilabel_csv.exists():
        logger.error(f"CSV not found: {args.multilabel_csv}")
        sys.exit(1)

    # ── Load targets ──────────────────────────────────────────────────────────
    logger.info("Loading target MD5s from CSV...")
    target_md5s = _load_target_md5s(args.multilabel_csv)
    logger.info(f"  {len(target_md5s)} MD5s in index")

    # ── Checkpoint resume ────────────────────────────────────────────────────
    done_md5s: set[str] = set()
    if args.resume and args.checkpoint.exists():
        with args.checkpoint.open() as f:
            ckpt = json.load(f)
        done_md5s = set(ckpt.get("done", []))
        logger.info(f"Resuming — {len(done_md5s)} MD5s already done (skipping)")

    # ── Build MD5 → path map ─────────────────────────────────────────────────
    logger.info("Scanning source dirs to build MD5 → path map...")
    t0 = time.time()
    md5_to_path = _build_md5_to_path(target_md5s)
    logger.info(f"  Map built in {time.time()-t0:.1f}s")

    # Issue 1: unmapped MD5s that still have old .pt on disk
    unmapped = target_md5s - set(md5_to_path)
    if unmapped:
        stale = [m for m in unmapped if (args.graphs_dir / f"{m}.pt").exists()]
        logger.warning(
            f"{len(unmapped)} MD5s not found in source dirs"
            + (f" — {len(stale)} still have OLD .pt files that WON'T be re-extracted" if stale else "")
        )

    # ── Build work list ───────────────────────────────────────────────────────
    work = [
        (md5, str(sol), str(args.graphs_dir))
        for md5, sol in md5_to_path.items()
        if md5 not in done_md5s
    ]
    if args.max_contracts:
        work = work[:args.max_contracts]

    total = len(work)
    logger.info(f"Contracts to process: {total}"
                + (f" ({len(done_md5s)} already done)" if done_md5s else ""))

    if args.dry_run or total == 0:
        logger.info("Nothing to do." if total == 0 else "Dry run — exiting.")
        return

    # ── Parallel extraction ───────────────────────────────────────────────────
    counters = dict(ok=0, ghost=0, skip=0, fail=0)
    done_this_run: list[str] = []
    t_start = time.time()

    with mp.Pool(processes=args.workers) as pool:
        with tqdm(total=total, desc="re-extract", unit="contract",
                  dynamic_ncols=True) as bar:
            for md5, status, detail in pool.imap_unordered(
                _worker, work, chunksize=args.chunksize
            ):
                counters[status] += 1
                done_this_run.append(md5)
                done_md5s.add(md5)
                bar.update(1)

                elapsed = time.time() - t_start
                rate    = len(done_this_run) / elapsed if elapsed > 0 else 0
                bar.set_postfix(
                    ok=counters["ok"], ghost=counters["ghost"],
                    skip=counters["skip"], fail=counters["fail"],
                    rate=f"{rate:.1f}c/s",
                )

                if args.verbose and status in ("ghost", "fail"):
                    logger.debug(f"  {status.upper()} {md5[:8]} — {detail}")

                # Checkpoint flush
                if len(done_this_run) % CHECKPOINT_EVERY == 0:
                    with args.checkpoint.open("w") as f:
                        json.dump({"done": list(done_md5s)}, f)

    # Final checkpoint
    with args.checkpoint.open("w") as f:
        json.dump({"done": list(done_md5s)}, f)

    elapsed = time.time() - t_start
    ghost_pct = 100 * counters["ghost"] / total if total else 0

    logger.info("=" * 60)
    logger.info("Re-extraction complete")
    logger.info(f"  Total          : {total}")
    logger.info(f"  OK (nodes > 3) : {counters['ok']}")
    logger.info(f"  Ghost (≤3 node): {counters['ghost']}  ({ghost_pct:.1f}%)")
    logger.info(f"  Skipped (solc) : {counters['skip']}")
    logger.info(f"  Failed         : {counters['fail']}")
    logger.info(f"  Elapsed        : {elapsed/60:.1f} min  ({total/elapsed:.1f} c/s avg)")

    gate_ok = ghost_pct < 1.0
    if gate_ok:
        logger.info(f"Gate PASS: ghost rate {ghost_pct:.1f}% < 1%")
        logger.info("Next: python ml/scripts/validate_graph_dataset.py --check-dim 12 --check-edge-types 7")
        logger.info("Then: python ml/scripts/create_cache.py")
    else:
        # Issue 2: ghost .pt files ARE on disk — make it actionable
        logger.error(
            f"Gate FAIL: ghost rate {ghost_pct:.1f}% >= 1%  "
            f"({counters['ghost']} ghost files are on disk)"
        )
        logger.error(
            "Delete ghosts before training: "
            "python ml/scripts/validate_graph_dataset.py --check-dim 12 --delete-ghosts"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
