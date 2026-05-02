#!/usr/bin/env python3
"""
validate_graph_dataset.py — Confirm edge_attr tensors are present in all graph .pt files.

WHY THIS EXISTS
───────────────
P0-B added edge relation-type embeddings to GNNEncoder. If a .pt file was
produced by the old ast_extractor.py (pre-P0-B), it will lack edge_attr and
GNNEncoder will silently fall back to zero-vectors, losing all edge-type signal
without any error. This script must be run before retraining to confirm that
all graph files have been regenerated with edge_attr.

Usage:
    python ml/scripts/validate_graph_dataset.py [--graphs-dir ml/data/graphs]

Exit codes:
    0  all files have edge_attr (safe to retrain)
    1  one or more files are missing edge_attr, load errors occurred, or
       the directory is empty
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def validate(graphs_dir: Path) -> int:
    pt_files = sorted(graphs_dir.glob("*.pt"))
    if not pt_files:
        print(f"ERROR: no .pt files found in {graphs_dir}", flush=True)
        return 1

    total = len(pt_files)
    has_attr: list[Path] = []
    missing_attr: list[Path] = []
    load_errors: list[tuple[Path, str]] = []

    for path in pt_files:
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            load_errors.append((path, str(exc)))
            continue

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            has_attr.append(path)
        else:
            missing_attr.append(path)

    print(f"\nGraph dataset validation — {graphs_dir}")
    print(f"  Total files      : {total}")
    print(f"  With edge_attr   : {len(has_attr)}")
    print(f"  Missing edge_attr: {len(missing_attr)}")
    print(f"  Load errors      : {len(load_errors)}")

    if load_errors:
        print("\nLoad errors:")
        for path, err in load_errors[:5]:
            print(f"  {path.name}: {err}")
        if len(load_errors) > 5:
            print(f"  ... and {len(load_errors) - 5} more")

    if missing_attr:
        print("\nFiles missing edge_attr (first 10):")
        for path in missing_attr[:10]:
            print(f"  {path.name}")
        if len(missing_attr) > 10:
            print(f"  ... and {len(missing_attr) - 10} more")
        print(
            "\nFIX: re-run ast_extractor.py to regenerate graphs with edge_attr.\n"
            "     See ml/src/preprocessing/graph_schema.py CHANGE POLICY."
        )
        return 1

    if load_errors:
        return 1

    print("\nPASS: all graph files have edge_attr. Safe to retrain.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that all graph .pt files contain edge_attr tensors."
    )
    parser.add_argument(
        "--graphs-dir",
        default=str(Path(__file__).parent.parent / "data" / "graphs"),
        help="Directory containing graph .pt files (default: ml/data/graphs)",
    )
    args = parser.parse_args()

    graphs_dir = Path(args.graphs_dir)
    if not graphs_dir.exists():
        print(f"ERROR: graphs directory does not exist: {graphs_dir}", flush=True)
        sys.exit(1)

    sys.exit(validate(graphs_dir))


if __name__ == "__main__":
    main()
