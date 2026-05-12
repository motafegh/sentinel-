#!/usr/bin/env python3
"""
validate_graph_dataset.py — Validate graph .pt files against schema expectations.

Checks performed (by default):
  1. File loads without error
  2. edge_attr is present and 1-D [E]
  3. edge_attr values are in [0, NUM_EDGE_TYPES)
  4. x.shape[1] == NODE_FEATURE_DIM  (feature dimension matches schema)

Additional checks enabled by flags (used after Phase 4 v5 re-extraction):
  --check-edge-types N   verify edge_attr max < N  (use 7 for v5 schema)
  --check-contains-edges verify at least one CONTAINS (id=5) edge exists per file
  --check-control-flow   verify at least one CONTROL_FLOW (id=6) edge exists per file

Usage:
    # v4 validation (default)
    python ml/scripts/validate_graph_dataset.py

    # v5 validation after Phase 4 re-extraction
    python ml/scripts/validate_graph_dataset.py \\
      --check-dim 13 \\
      --check-edge-types 7 \\
      --check-contains-edges \\
      --check-control-flow

Exit codes:
    0  all files pass all requested checks
    1  one or more files failed, directory empty, or load error occurred
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NUM_EDGE_TYPES

torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])


def validate(
    graphs_dir:           Path,
    expected_node_dim:    int  = NODE_FEATURE_DIM,
    expected_edge_types:  int  = NUM_EDGE_TYPES,
    check_contains_edges: bool = False,
    check_control_flow:   bool = False,
    check_cfg_subtypes:   bool = False,
) -> int:
    pt_files = sorted(graphs_dir.glob("*.pt"))
    if not pt_files:
        print(f"ERROR: no .pt files found in {graphs_dir}", flush=True)
        return 1

    total = len(pt_files)

    # Counters
    passed:          list[Path]             = []
    load_errors:     list[tuple[Path, str]] = []
    missing_attr:    list[Path]             = []
    shape_errors:    list[tuple[Path, str]] = []
    value_errors:    list[tuple[Path, str]] = []
    dim_errors:      list[tuple[Path, str]] = []
    missing_contains:  list[Path]            = []
    missing_cf:        list[Path]            = []
    missing_cfg_sub:   list[Path]            = []

    for path in pt_files:
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            load_errors.append((path, str(exc)))
            continue

        failed_this = False

        # ── Node feature dimension ────────────────────────────────────────
        if hasattr(data, "x") and data.x is not None:
            actual_dim = data.x.shape[1] if data.x.dim() == 2 else -1
            if actual_dim != expected_node_dim:
                dim_errors.append((
                    path,
                    f"x.shape[1]={actual_dim} — expected {expected_node_dim}. "
                    "Regenerate with updated graph_extractor.py.",
                ))
                failed_this = True

        # ── Edge attribute presence ───────────────────────────────────────
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            missing_attr.append(path)
            failed_this = True
            continue  # can't do value/type checks without edge_attr

        ea = data.edge_attr

        # Shape: must be 1-D [E]
        if ea.dim() != 1:
            shape_errors.append((
                path,
                f"edge_attr.shape={tuple(ea.shape)} — expected 1-D [E], "
                f"got {ea.dim()}-D. Regenerate with ast_extractor.py.",
            ))
            failed_this = True
            continue

        # Value range: all IDs in [0, expected_edge_types)
        if ea.numel() > 0:
            mn, mx = int(ea.min().item()), int(ea.max().item())
            if mn < 0 or mx >= expected_edge_types:
                value_errors.append((
                    path,
                    f"edge_attr values [{mn}, {mx}] outside [0, {expected_edge_types}). "
                    "Check EDGE_TYPES in graph_schema.py.",
                ))
                failed_this = True

        # ── CONTAINS edges (id=5, function→cfg_node) ─────────────────────
        if check_contains_edges and ea.numel() > 0:
            if not (ea == 5).any():
                missing_contains.append(path)
                failed_this = True

        # ── CONTROL_FLOW edges (id=6, cfg_node→cfg_node) ─────────────────
        if check_control_flow and ea.numel() > 0:
            if not (ea == 6).any():
                missing_cf.append(path)
                failed_this = True

        # ── CFG subtype nodes (type_ids 8–12) ─────────────────────────────
        # Verifies that _build_control_flow_edges() created CFG nodes.
        # A graph with CONTAINS edges but no CFG subtypes in x[:,0] means
        # the CFG node extraction was skipped or fell back to CFG_NODE_OTHER
        # for every node — likely a Slither version mismatch.
        if check_cfg_subtypes and hasattr(data, "x") and data.x is not None:
            type_ids_raw = (data.x[:, 0] * 12).round().int()
            has_cfg = bool(((type_ids_raw >= 8) & (type_ids_raw <= 12)).any().item())
            if not has_cfg:
                missing_cfg_sub.append(path)
                failed_this = True

        if not failed_this:
            passed.append(path)

    failed_count = (
        len(missing_attr) + len(shape_errors) + len(value_errors)
        + len(load_errors) + len(dim_errors)
        + len(missing_contains) + len(missing_cf) + len(missing_cfg_sub)
    )

    print(f"\nGraph dataset validation — {graphs_dir}")
    print(f"  Total files         : {total}")
    print(f"  PASS                : {len(passed)}")
    print(f"  Node dim errors     : {len(dim_errors)}  (expected x.shape[1]={expected_node_dim})")
    print(f"  Missing edge_attr   : {len(missing_attr)}")
    print(f"  Edge shape errors   : {len(shape_errors)}  (expected [E], got [E,1] or other)")
    print(f"  Edge value errors   : {len(value_errors)}  (IDs outside [0, {expected_edge_types}))")
    if check_contains_edges:
        print(f"  Missing CONTAINS(5) : {len(missing_contains)}")
    if check_control_flow:
        print(f"  Missing CTRL_FLOW(6): {len(missing_cf)}")
    if check_cfg_subtypes:
        print(f"  Missing CFG subtypes: {len(missing_cfg_sub)}  (type_ids 8–12 absent)")
    print(f"  Load errors         : {len(load_errors)}")

    def _print_samples(items: list, label: str) -> None:
        if not items:
            return
        print(f"\n{label}:")
        # items can be list[Path] or list[tuple[Path, str]]
        for item in items[:5]:
            if isinstance(item, tuple):
                path, msg = item
                print(f"  {path.name}: {msg}")
            else:
                print(f"  {item.name}")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")

    _print_samples(load_errors,      "Load errors")
    _print_samples(dim_errors,       "Node dim errors")
    _print_samples(missing_attr,     "Files missing edge_attr")  # type: ignore[arg-type]
    _print_samples(shape_errors,     "Edge shape errors")
    _print_samples(value_errors,     "Edge value errors")
    _print_samples(missing_contains, "Files missing CONTAINS edges")  # type: ignore[arg-type]
    _print_samples(missing_cf,       "Files missing CONTROL_FLOW edges")  # type: ignore[arg-type]
    _print_samples(missing_cfg_sub,  "Files missing CFG subtype nodes (type_ids 8-12)")  # type: ignore[arg-type]

    if missing_attr or shape_errors or dim_errors:
        print(
            "\nFIX: re-run ast_extractor.py --force to regenerate graphs with the "
            "current graph_schema.py (NODE_FEATURE_DIM, NUM_EDGE_TYPES, EDGE_TYPES)."
        )

    if failed_count:
        return 1

    print(
        f"\nPASS: all {total} graph files are valid "
        f"(node_dim={expected_node_dim}, edge_types<{expected_edge_types}). "
        "Safe to retrain."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate graph .pt files against schema expectations."
    )
    parser.add_argument(
        "--graphs-dir",
        default=str(Path(__file__).parent.parent / "data" / "graphs"),
        help="Directory containing graph .pt files (default: ml/data/graphs)",
    )
    parser.add_argument(
        "--check-dim",
        type=int,
        default=NODE_FEATURE_DIM,
        metavar="N",
        help=(
            f"Expected node feature dimension x.shape[1] "
            f"(default: {NODE_FEATURE_DIM} from graph_schema.NODE_FEATURE_DIM). "
            "Pass 12 for v5 schema (was 8 in v4)."
        ),
    )
    parser.add_argument(
        "--check-edge-types",
        type=int,
        default=NUM_EDGE_TYPES,
        metavar="N",
        help=(
            f"Expected upper bound on edge type IDs; all IDs must be in [0, N). "
            f"(default: {NUM_EDGE_TYPES} from graph_schema.NUM_EDGE_TYPES). "
            "Pass 7 after v5 re-extraction."
        ),
    )
    parser.add_argument(
        "--check-contains-edges",
        action="store_true",
        help=(
            "Verify every file has at least one CONTAINS edge (id=5, "
            "function→cfg_node). Requires v5 graph schema."
        ),
    )
    parser.add_argument(
        "--check-control-flow",
        action="store_true",
        help=(
            "Verify every file has at least one CONTROL_FLOW edge (id=6, "
            "cfg_node→cfg_node). Requires v5 graph schema."
        ),
    )
    parser.add_argument(
        "--check-cfg-subtypes",
        action="store_true",
        help=(
            "Verify every file has at least one CFG node (type_id in 8–12). "
            "Catches cases where _build_control_flow_edges() was skipped "
            "or produced no CFG nodes despite CONTAINS edges being present."
        ),
    )
    args = parser.parse_args()

    graphs_dir = Path(args.graphs_dir)
    if not graphs_dir.exists():
        print(f"ERROR: graphs directory does not exist: {graphs_dir}", flush=True)
        sys.exit(1)

    sys.exit(validate(
        graphs_dir,
        expected_node_dim=args.check_dim,
        expected_edge_types=args.check_edge_types,
        check_contains_edges=args.check_contains_edges,
        check_control_flow=args.check_control_flow,
        check_cfg_subtypes=args.check_cfg_subtypes,
    ))


if __name__ == "__main__":
    main()
