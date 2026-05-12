#!/usr/bin/env python3
"""
reextract_graphs.py — Re-extract all graph .pt files using the fixed extractor.

PURPOSE
───────
After v5.1 Phase 0 fixes (_select_contract interface bug, CFG failure counter),
this script re-extracts every graph in multilabel_index.csv, overwriting the
existing .pt files. Token .pt files and the CSV itself are left untouched.

The MD5 filename = md5(relative_path_from_project_root), so this script
rebuilds an MD5 → path map by hashing every .sol file found under the
known source directories, then re-extracts matches.

USAGE
─────
  source ml/.venv/bin/activate
  PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py
  PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py --dry-run
  PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py --max-contracts 200

AFTER THIS SCRIPT
─────────────────
  python ml/scripts/validate_graph_dataset.py --check-dim 12 --check-edge-types 7
  Gate: ghost graphs (<= 3 nodes) must be < 1% of dataset.
  Then rebuild the RAM cache:
  python ml/scripts/create_cache.py
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIRS = [
    PROJECT_ROOT / "BCCC-SCsVul-2024" / "SourceCodes",
    PROJECT_ROOT / "ml" / "data" / "SolidiFI-processed",
    PROJECT_ROOT / "ml" / "data" / "SolidiFI",
    PROJECT_ROOT / "ml" / "data" / "smartbugs-curated",
    PROJECT_ROOT / "ml" / "data" / "smartbugs-wild",
    PROJECT_ROOT / "ml" / "data" / "augmented",
]

DEFAULT_CSV    = PROJECT_ROOT / "ml" / "data" / "processed" / "multilabel_index.csv"
DEFAULT_GRAPHS = PROJECT_ROOT / "ml" / "data" / "graphs"

# solc-select artifacts are inside the venv at this path:
#   ml/.venv/.solc-select/artifacts/solc-X.Y.Z/solc-X.Y.Z
_SOLC_ARTIFACTS = PROJECT_ROOT / "ml" / ".venv" / ".solc-select" / "artifacts"


# ── Helpers ───────────────────────────────────────────────────────────────────

_PRAGMA_RE = re.compile(r'pragma\s+solidity\s+[\^~>=<\s]*(\d+\.\d+\.\d+)')


def _detect_solc_version(sol_path: Path) -> str:
    """Return the major.minor version from the pragma, defaulting to 0.8.20."""
    try:
        txt = sol_path.read_text(encoding="utf-8", errors="replace")
        m = _PRAGMA_RE.search(txt)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "0.8.20"


def _solc_binary(version: str) -> Optional[Path]:
    """
    Return the path to the solc binary for `version` inside the venv's
    solc-select artifacts dir, or None if not found (falls back to PATH solc).

    Layout: .solc-select/artifacts/solc-X.Y.Z/solc-X.Y.Z
    """
    binary = _SOLC_ARTIFACTS / f"solc-{version}" / f"solc-{version}"
    if binary.exists():
        return binary
    # Try without patch (e.g. pragma "0.5" → try "0.5.0" → already handled by caller)
    return None


def _md5_path(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    return hashlib.md5(str(rel).encode("utf-8")).hexdigest()


def _build_md5_to_path(target_md5s: set[str]) -> dict[str, Path]:
    """Hash every .sol in known source dirs; return only hits in target_md5s."""
    mapping: dict[str, Path] = {}
    scanned = 0

    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            logger.debug(f"Source dir absent (skip): {src_dir}")
            continue
        logger.info(f"Scanning {src_dir} ...")
        for sol in src_dir.rglob("*.sol"):
            scanned += 1
            md5 = _md5_path(sol)
            if md5 in target_md5s:
                mapping[md5] = sol
            if scanned % 20_000 == 0:
                logger.info(f"  scanned {scanned}  found {len(mapping)}/{len(target_md5s)} ...")

    logger.info(f"Scan complete — {scanned} files scanned, {len(mapping)}/{len(target_md5s)} matched")
    return mapping


def _load_target_md5s(csv_path: Path) -> set[str]:
    md5s: set[str] = set()
    with csv_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header
            md5s.add(line.split(",")[0].strip())
    return md5s


# ── Core re-extraction ────────────────────────────────────────────────────────

def reextract(
    md5_to_path:   dict[str, Path],
    graphs_dir:    Path,
    dry_run:       bool,
    max_contracts: Optional[int],
) -> dict:
    from ml.src.preprocessing.graph_extractor import (
        GraphExtractionConfig,
        extract_contract_graph,
        EmptyGraphError,
        SolcCompilationError,
        SlitherParseError,
    )

    items = list(md5_to_path.items())
    if max_contracts:
        items = items[:max_contracts]

    total   = len(items)
    ok      = 0
    ghost   = 0
    failed  = 0
    t_start = time.time()

    for i, (md5, sol_path) in enumerate(items, 1):
        if i % 500 == 0 or i == 1:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta  = (total - i) / rate if rate > 0 else 0
            logger.info(
                f"[{i}/{total}] {rate:.0f} c/s  ETA {eta/60:.0f} min"
                f" | ok={ok} ghost={ghost} fail={failed}"
            )

        out_path = graphs_dir / f"{md5}.pt"
        solc_ver = _detect_solc_version(sol_path)
        cfg = GraphExtractionConfig(
            solc_version=solc_ver,
            solc_binary=_solc_binary(solc_ver),
        )

        try:
            g = extract_contract_graph(sol_path, cfg)
        except (EmptyGraphError, SolcCompilationError, SlitherParseError) as exc:
            logger.debug(f"  SKIP {md5[:8]} — {type(exc).__name__}: {exc}")
            failed += 1
            continue
        except Exception as exc:
            logger.warning(f"  FAIL {md5[:8]} — unexpected: {exc}")
            failed += 1
            continue

        if g.num_nodes <= 3:
            ghost += 1
            logger.debug(
                f"  GHOST {md5[:8]} — nodes={g.num_nodes} edges={g.num_edges}"
                f" contract={g.contract_name}"
            )
        else:
            ok += 1

        if not dry_run:
            # Atomic write: save to .tmp then rename — safe against mid-write kills
            tmp = out_path.with_suffix(".tmp")
            torch.save(g, tmp)
            tmp.rename(out_path)

    elapsed = time.time() - t_start
    summary = dict(
        total=total, ok=ok, ghost=ghost, failed=failed,
        ghost_pct=100 * ghost / total if total else 0,
        elapsed_s=elapsed, dry_run=dry_run,
    )
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-extract graph .pt files with fixed extractor (v5.1 Phase 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--multilabel-csv",  type=Path, default=DEFAULT_CSV)
    p.add_argument("--graphs-dir",      type=Path, default=DEFAULT_GRAPHS)
    p.add_argument("--dry-run",         action="store_true",
                   help="Report matches and ghost estimate without writing files")
    p.add_argument("--max-contracts",   type=int, default=None, metavar="N",
                   help="Stop after N contracts (smoke test)")
    p.add_argument("--verbose",         action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    logger.info("v5.1 Phase 1 — Re-extraction with fixed _select_contract()")
    logger.info(f"CSV: {args.multilabel_csv}")
    logger.info(f"Graphs dir: {args.graphs_dir}")
    if args.dry_run:
        logger.info("DRY RUN — no files will be written")

    if not args.multilabel_csv.exists():
        logger.error(f"CSV not found: {args.multilabel_csv}")
        sys.exit(1)
    if not args.graphs_dir.exists():
        logger.error(f"Graphs dir not found: {args.graphs_dir}")
        sys.exit(1)

    logger.info("Loading target MD5s from CSV...")
    target_md5s = _load_target_md5s(args.multilabel_csv)
    logger.info(f"  {len(target_md5s)} MD5s loaded")

    logger.info("Building MD5 → path map (scanning source dirs)...")
    md5_to_path = _build_md5_to_path(target_md5s)

    # Issue 1 fix: warn about unmapped MD5s that still have OLD .pt files on disk
    unmapped = target_md5s - set(md5_to_path)
    if unmapped:
        unmapped_with_pt = [
            m for m in unmapped
            if (args.graphs_dir / f"{m}.pt").exists()
        ]
        logger.warning(
            f"{len(unmapped)} MD5s not found in source dirs "
            f"(token-only orphans or missing data)"
        )
        if unmapped_with_pt:
            logger.warning(
                f"  {len(unmapped_with_pt)} of those still have OLD .pt files on disk "
                f"— these WON'T be re-extracted and WILL be included in validation"
            )
        else:
            logger.info(f"  None of the {len(unmapped)} unmapped MD5s have .pt files — no exposure")

    if args.max_contracts:
        logger.info(f"Capping at {args.max_contracts} contracts (smoke mode)")

    logger.info(
        f"Starting re-extraction of {len(md5_to_path)} contracts"
        f"{' [DRY RUN]' if args.dry_run else ''}..."
    )

    summary = reextract(
        md5_to_path=md5_to_path,
        graphs_dir=args.graphs_dir,
        dry_run=args.dry_run,
        max_contracts=args.max_contracts,
    )

    logger.info("=" * 60)
    logger.info("Re-extraction complete")
    logger.info(f"  Total mapped   : {summary['total']}")
    logger.info(f"  OK (nodes > 3) : {summary['ok']}")
    logger.info(f"  Ghost (≤3 node): {summary['ghost']}  ({summary['ghost_pct']:.1f}%)")
    logger.info(f"  Failed (skip)  : {summary['failed']}")
    logger.info(f"  Elapsed        : {summary['elapsed_s']/60:.0f} min")

    if not args.dry_run:
        gate_ok = summary["ghost_pct"] < 1.0
        logger.info("")
        if gate_ok:
            logger.info(f"Gate PASS: ghost rate {summary['ghost_pct']:.1f}% < 1%")
            logger.info("Next steps:")
            logger.info("  python ml/scripts/validate_graph_dataset.py --check-dim 12 --check-edge-types 7")
            logger.info("  python ml/scripts/create_cache.py")
        else:
            # Issue 2 fix: ghost .pt files ARE on disk — name them explicitly
            logger.error(
                f"Gate FAIL: ghost rate {summary['ghost_pct']:.1f}% >= 1% "
                f"({summary['ghost']} ghost files are on disk)"
            )
            logger.error("Delete ghosts before training — run validate with --delete-ghosts:")
            logger.error(
                "  python ml/scripts/validate_graph_dataset.py "
                "--check-dim 12 --check-edge-types 7 --delete-ghosts"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
