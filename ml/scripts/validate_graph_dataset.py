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

NUM_EDGE_TYPES = 5  # CALLS READS WRITES EMITS INHERITS — mirror of graph_schema.py


def validate(graphs_dir: Path) -> int:
    pt_files = sorted(graphs_dir.glob("*.pt"))
    if not pt_files:
        print(f"ERROR: no .pt files found in {graphs_dir}", flush=True)
        return 1

    total = len(pt_files)
    has_attr:    list[Path] = []
    missing_attr: list[Path] = []
    shape_errors: list[tuple[Path, str]] = []   # [E,1] instead of [E], etc.
    value_errors: list[tuple[Path, str]] = []   # values outside [0, NUM_EDGE_TYPES)
    load_errors:  list[tuple[Path, str]] = []

    for path in pt_files:
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            load_errors.append((path, str(exc)))
            continue

        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            missing_attr.append(path)
            continue

        ea = data.edge_attr

        # Shape check: must be 1-D [E].  Pre-refactor files may be [E, 1].
        if ea.dim() != 1:
            shape_errors.append((
                path,
                f"edge_attr.shape={tuple(ea.shape)} — expected 1-D [E], "
                f"got {ea.dim()}-D. Regenerate with ast_extractor.py.",
            ))
            continue

        # Value-range check: all IDs must be in [0, NUM_EDGE_TYPES).
        if ea.numel() > 0:
            mn, mx = int(ea.min().item()), int(ea.max().item())
            if mn < 0 or mx >= NUM_EDGE_TYPES:
                value_errors.append((
                    path,
                    f"edge_attr values [{mn}, {mx}] outside [0, {NUM_EDGE_TYPES}). "
                    "Check EDGE_TYPES in graph_schema.py.",
                ))
                continue

        has_attr.append(path)

    failed = len(missing_attr) + len(shape_errors) + len(value_errors) + len(load_errors)

    print(f"\nGraph dataset validation — {graphs_dir}")
    print(f"  Total files      : {total}")
    print(f"  PASS             : {len(has_attr)}")
    print(f"  Missing edge_attr: {len(missing_attr)}")
    print(f"  Shape errors     : {len(shape_errors)}  (expected [E], got [E,1] or other)")
    print(f"  Value errors     : {len(value_errors)}  (IDs outside [0, {NUM_EDGE_TYPES}))")
    print(f"  Load errors      : {len(load_errors)}")

    def _print_list(items: list, label: str) -> None:
        if not items:
            return
        print(f"\n{label}:")
        for path, msg in items[:5]:
            print(f"  {path.name}: {msg}")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")

    _print_list(load_errors,  "Load errors")
    _print_list(missing_attr, "Files missing edge_attr")  # type: ignore[arg-type]
    _print_list(shape_errors, "Shape errors")
    _print_list(value_errors, "Value errors")

    if missing_attr:
        print(
            "\nFIX: re-run ast_extractor.py to regenerate graphs with edge_attr.\n"
            "     See ml/src/preprocessing/graph_schema.py CHANGE POLICY."
        )

    if failed:
        return 1

    print("\nPASS: all graph files have valid edge_attr. Safe to retrain.")
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
