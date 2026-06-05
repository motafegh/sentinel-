#!/usr/bin/env python3
"""
validate_graph_dataset.py — Validate graph .pt files against schema expectations.

Checks performed (by default):
  1. File loads without error
  2. edge_attr is present and 1-D [E]
  3. edge_attr values are in [0, NUM_EDGE_TYPES)
  4. x.shape[1] == NODE_FEATURE_DIM  (feature dimension matches schema)
  5. No NaN/inf in x or edge_attr
  6. Feature values in expected ranges

Additional checks enabled by flags:
  --check-contains-edges     verify CONTAINS edge exists per file
  --check-control-flow       verify CONTROL_FLOW edge exists per file
  --check-cfg-subtypes       verify CFG subtype nodes exist (type_ids 8–12)
  --check-block-globals      verify uses_block_globals (feat[2]) fires
  --check-external-call-edges verify EXTERNAL_CALL self-loop edges (id=11, Fix #3)
  --check-arith-nodes        verify CFG_NODE_ARITH nodes exist (type_id=13, Fix #4)
  --check-unchecked-feature  verify in_unchecked_block feature (feat[11], Fix #4)
  --check-icfg-edges         verify ICFG-Lite edges (ids 8–10) exist
  --check-edge-index-valid   verify edge_index values < node count
  --check-feature-dtypes     verify x is float32, edge_attr is int64
  --check-node-type-range    verify node type_ids in [0, max(NODE_TYPES)]
  --check-all                enable all optional checks

Usage:
    python ml/scripts/validate_graph_dataset.py --check-all
    python ml/scripts/validate_graph_dataset.py --check-contains-edges --check-control-flow

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

from ml.src.preprocessing.graph_schema import (
    NODE_FEATURE_DIM,
    NODE_TYPES,
    NUM_EDGE_TYPES,
    EDGE_TYPES,
)

torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

_MAX_NODE_TYPE_ID: int = max(NODE_TYPES.values())


def validate(
    graphs_dir:                Path,
    expected_node_dim:         int    = NODE_FEATURE_DIM,
    expected_edge_types:       int    = NUM_EDGE_TYPES,
    check_contains_edges:      bool   = False,
    check_control_flow:        bool   = False,
    check_cfg_subtypes:        bool   = False,
    check_block_globals:       bool   = False,
    check_external_call_edges: bool   = False,
    check_arith_nodes:         bool   = False,
    check_unchecked_feature:   bool   = False,
    check_icfg_edges:          bool   = False,
    check_edge_index_valid:    bool   = False,
    check_feature_dtypes:      bool   = False,
    check_node_type_range:     bool   = False,
) -> int:
    pt_files = sorted(graphs_dir.glob("*.pt"))
    if not pt_files:
        print(f"ERROR: no .pt files found in {graphs_dir}", flush=True)
        return 1

    total = len(pt_files)

    # Counters
    passed:               list[Path]             = []
    load_errors:          list[tuple[Path, str]] = []
    missing_attr:         list[Path]             = []
    shape_errors:         list[tuple[Path, str]] = []
    value_errors:         list[tuple[Path, str]] = []
    dim_errors:           list[tuple[Path, str]] = []
    nan_inf_errors:       list[tuple[Path, str]] = []
    dtype_errors:         list[tuple[Path, str]] = []
    edge_index_errors:    list[tuple[Path, str]] = []
    node_type_errors:     list[tuple[Path, str]] = []
    missing_contains:     list[Path]             = []
    missing_cf:           list[Path]             = []
    missing_cfg_sub:      list[Path]             = []
    missing_ext_call:     list[Path]             = []
    missing_arith:        list[Path]             = []
    missing_unchecked:    list[Path]             = []
    missing_icfg:         list[Path]             = []
    block_globals_count:  int                    = 0

    for path in pt_files:
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            load_errors.append((path, str(exc)))
            continue

        failed_this = False

        # ── NaN/inf in node features ───────────────────────────────────
        if hasattr(data, "x") and data.x is not None:
            if torch.isnan(data.x).any() or torch.isinf(data.x).any():
                nan_inf_errors.append((path, "NaN or inf in node features (x)"))
                failed_this = True

        # ── NaN/inf in edge features ───────────────────────────────────
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            if torch.isnan(data.edge_attr.float()).any() or torch.isinf(data.edge_attr.float()).any():
                nan_inf_errors.append((path, "NaN or inf in edge_attr"))
                failed_this = True

        # ── Feature dtypes ─────────────────────────────────────────────
        if check_feature_dtypes:
            if hasattr(data, "x") and data.x is not None:
                if data.x.dtype != torch.float32:
                    dtype_errors.append((path, f"x.dtype={data.x.dtype}, expected float32"))
                    failed_this = True
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                if data.edge_attr.dtype != torch.int64:
                    dtype_errors.append((path, f"edge_attr.dtype={data.edge_attr.dtype}, expected int64"))
                    failed_this = True

        # ── Node feature dimension ─────────────────────────────────────
        if hasattr(data, "x") and data.x is not None:
            actual_dim = data.x.shape[1] if data.x.dim() == 2 else -1
            if actual_dim != expected_node_dim:
                dim_errors.append((
                    path,
                    f"x.shape[1]={actual_dim} — expected {expected_node_dim}. "
                    "Regenerate with updated graph_extractor.py.",
                ))
                failed_this = True

        # ── Node type range ────────────────────────────────────────────
        if check_node_type_range and hasattr(data, "x") and data.x is not None and data.x.dim() == 2:
            type_ids_raw = (data.x[:, 0] * _MAX_NODE_TYPE_ID).round().int()
            out_of_range = type_ids_raw[(type_ids_raw < 0) | (type_ids_raw > _MAX_NODE_TYPE_ID)]
            if out_of_range.numel() > 0:
                bad_vals = out_of_range.unique().tolist()
                node_type_errors.append((path, f"type_ids out of range: {bad_vals}"))
                failed_this = True

        # ── Edge attribute presence ────────────────────────────────────
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            missing_attr.append(path)
            failed_this = True
            continue  # can't do further edge checks without edge_attr

        ea = data.edge_attr

        # Shape: must be 1-D [E]
        if ea.dim() != 1:
            shape_errors.append((
                path,
                f"edge_attr.shape={tuple(ea.shape)} — expected 1-D [E], "
                f"got {ea.dim()}-D. Regenerate with reextract_graphs.py.",
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

        # ── Edge index validity ────────────────────────────────────────
        if check_edge_index_valid and hasattr(data, "edge_index") and data.edge_index is not None:
            num_nodes = data.x.shape[0] if hasattr(data, "x") and data.x is not None else -1
            if num_nodes > 0:
                ei = data.edge_index
                if ei.numel() > 0:
                    bad_src = ei[0][(ei[0] < 0) | (ei[0] >= num_nodes)]
                    bad_dst = ei[1][(ei[1] < 0) | (ei[1] >= num_nodes)]
                    if bad_src.numel() > 0 or bad_dst.numel() > 0:
                        edge_index_errors.append((
                            path,
                            f"edge_index references invalid node IDs "
                            f"(num_nodes={num_nodes})",
                        ))
                        failed_this = True

        # ── Empty-graph guard for edge-count checks ────────────────────
        # Some files may legitimately have 0 edges (e.g., state-var-only
        # contracts with no functions). Skip edge-count checks for those.
        if ea.numel() == 0:
            if not failed_this:
                passed.append(path)
            continue

        # ── CONTAINS edges ─────────────────────────────────────────────
        contains_id = EDGE_TYPES["CONTAINS"]
        if check_contains_edges:
            if not (ea == contains_id).any():
                missing_contains.append(path)
                failed_this = True

        # ── CONTROL_FLOW edges ─────────────────────────────────────────
        ctrl_id = EDGE_TYPES["CONTROL_FLOW"]
        if check_control_flow:
            if not (ea == ctrl_id).any():
                missing_cf.append(path)
                failed_this = True

        # ── CFG subtype nodes (type_ids 8–12) ──────────────────────────
        if check_cfg_subtypes and hasattr(data, "x") and data.x is not None:
            type_ids_raw = (data.x[:, 0] * _MAX_NODE_TYPE_ID).round().int()
            has_cfg = bool(((type_ids_raw >= 8) & (type_ids_raw <= 12)).any().item())
            if not has_cfg:
                missing_cfg_sub.append(path)
                failed_this = True

        # ── EXTERNAL_CALL self-loop edges (type_id=11, Fix #3) ────────
        if check_external_call_edges:
            ext_call_id = 11  # Fix #3 edge type
            if not (ea == ext_call_id).any():
                missing_ext_call.append(path)
                failed_this = True

        # ── CFG_NODE_ARITH nodes (type_id=13, Fix #4) ─────────────────
        if check_arith_nodes and hasattr(data, "x") and data.x is not None:
            type_ids_raw = (data.x[:, 0] * _MAX_NODE_TYPE_ID).round().int()
            if not (type_ids_raw == 13).any():
                missing_arith.append(path)
                failed_this = True

        # ── in_unchecked_block feature (feat[11], Fix #4) ─────────────
        if check_unchecked_feature and hasattr(data, "x") and data.x is not None:
            if data.x.shape[1] > 11:
                if not (data.x[:, 11] > 0.5).any():
                    missing_unchecked.append(path)
                    failed_this = True

        # ── ICFG-Lite edges (ids 8–10) ────────────────────────────────
        if check_icfg_edges:
            icfg_ids = torch.tensor([
                EDGE_TYPES["CALL_ENTRY"],
                EDGE_TYPES["RETURN_TO"],
                EDGE_TYPES["DEF_USE"],
            ])
            if not ea.unique().isin(icfg_ids).any():
                missing_icfg.append(path)
                failed_this = True

        # ── uses_block_globals feature (feat[2]) ──────────────────────
        if check_block_globals and hasattr(data, "x") and data.x is not None:
            if (data.x[:, 2] > 0.5).any():
                block_globals_count += 1

        if not failed_this:
            passed.append(path)

    # ── Summary ────────────────────────────────────────────────────────────
    failed_count = (
        len(missing_attr) + len(shape_errors) + len(value_errors)
        + len(load_errors) + len(dim_errors) + len(nan_inf_errors)
        + len(dtype_errors) + len(edge_index_errors) + len(node_type_errors)
        + len(missing_contains) + len(missing_cf) + len(missing_cfg_sub)
        + len(missing_ext_call) + len(missing_arith) + len(missing_unchecked)
        + len(missing_icfg)
    )

    print(f"\nGraph dataset validation — {graphs_dir}")
    print(f"  Total files         : {total}")
    print(f"  PASS                : {len(passed)}")
    print(f"  Node dim errors     : {len(dim_errors)}  (expected x.shape[1]={expected_node_dim})")
    print(f"  Missing edge_attr   : {len(missing_attr)}")
    print(f"  Edge shape errors   : {len(shape_errors)}  (expected [E], got [E,1] or other)")
    print(f"  Edge value errors   : {len(value_errors)}  (IDs outside [0, {expected_edge_types}))")
    print(f"  NaN/inf errors      : {len(nan_inf_errors)}")
    if check_feature_dtypes:
        print(f"  Dtype errors        : {len(dtype_errors)}")
    if check_edge_index_valid:
        print(f"  Edge index errors   : {len(edge_index_errors)}")
    if check_node_type_range:
        print(f"  Node type errors    : {len(node_type_errors)}  (IDs outside [0, {_MAX_NODE_TYPE_ID}])")
    if check_contains_edges:
        print(f"  Missing CONTAINS({EDGE_TYPES['CONTAINS']}) : {len(missing_contains)}")
    if check_control_flow:
        print(f"  Missing CTRL_FLOW({EDGE_TYPES['CONTROL_FLOW']}) : {len(missing_cf)}")
    if check_cfg_subtypes:
        print(f"  Missing CFG subtypes: {len(missing_cfg_sub)}  (type_ids 8–12 absent)")
    if check_external_call_edges:
        print(f"  Missing EXT_CALL(11): {len(missing_ext_call)}")
    if check_arith_nodes:
        print(f"  Missing ARITH(13)   : {len(missing_arith)}")
    if check_unchecked_feature:
        print(f"  Missing unchecked   : {len(missing_unchecked)}  (feat[11] absent)")
    if check_icfg_edges:
        print(f"  Missing ICFG edges  : {len(missing_icfg)}  (ids 8–10)")
    if check_block_globals:
        pct = 100 * block_globals_count / total if total else 0
        print(f"  Block globals (feat[2] fires): {block_globals_count}/{total}  ({pct:.1f}%)")
        if pct < 1.0:
            print(f"  WARNING: <1% of files have uses_block_globals activated.")
            print(f"           May indicate BUG-6 cascade or _compute_uses_block_globals() issue.")
    print(f"  Load errors         : {len(load_errors)}")

    def _print_samples(items: list, label: str) -> None:
        if not items:
            return
        print(f"\n{label}:")
        for item in items[:5]:
            if isinstance(item, tuple):
                it_path, msg = item
                print(f"  {it_path.name}: {msg}")
            else:
                print(f"  {item.name}")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")

    _print_samples(load_errors,       "Load errors")
    _print_samples(dim_errors,        "Node dim errors")
    _print_samples(nan_inf_errors,    "NaN/inf errors")
    _print_samples(dtype_errors,      "Dtype errors")
    _print_samples(edge_index_errors, "Edge index errors")
    _print_samples(node_type_errors,  "Node type errors")
    _print_samples(missing_attr,      "Files missing edge_attr")  # type: ignore[arg-type]
    _print_samples(shape_errors,      "Edge shape errors")
    _print_samples(value_errors,      "Edge value errors")
    _print_samples(missing_contains,  "Files missing CONTAINS edges")  # type: ignore[arg-type]
    _print_samples(missing_cf,        "Files missing CONTROL_FLOW edges")  # type: ignore[arg-type]
    _print_samples(missing_cfg_sub,   "Files missing CFG subtype nodes")  # type: ignore[arg-type]
    _print_samples(missing_ext_call,  "Files missing EXTERNAL_CALL edges")  # type: ignore[arg-type]
    _print_samples(missing_arith,     "Files missing CFG_NODE_ARITH nodes")  # type: ignore[arg-type]
    _print_samples(missing_unchecked, "Files missing in_unchecked_block feature")  # type: ignore[arg-type]
    _print_samples(missing_icfg,      "Files missing ICFG edges")  # type: ignore[arg-type]

    if missing_attr or shape_errors or dim_errors or nan_inf_errors:
        print(
            "\nFIX: re-run reextract_graphs.py to regenerate graphs with the "
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
        help=f"Expected x.shape[1] (default: {NODE_FEATURE_DIM} from graph_schema).",
    )
    parser.add_argument(
        "--check-edge-types",
        type=int,
        default=NUM_EDGE_TYPES,
        metavar="N",
        help=f"Upper bound on edge type IDs (default: {NUM_EDGE_TYPES} from graph_schema).",
    )
    parser.add_argument(
        "--check-contains-edges",
        action="store_true",
        help="Verify CONTAINS edge exists per file.",
    )
    parser.add_argument(
        "--check-control-flow",
        action="store_true",
        help="Verify CONTROL_FLOW edge exists per file.",
    )
    parser.add_argument(
        "--check-cfg-subtypes",
        action="store_true",
        help="Verify CFG subtype nodes (type_ids 8–12) exist per file.",
    )
    parser.add_argument(
        "--check-block-globals",
        action="store_true",
        help="Count files where uses_block_globals (feat[2]) fires.",
    )
    parser.add_argument(
        "--check-external-call-edges",
        action="store_true",
        help="Verify EXTERNAL_CALL self-loop edge (id=11) exists (Fix #3).",
    )
    parser.add_argument(
        "--check-arith-nodes",
        action="store_true",
        help="Verify CFG_NODE_ARITH nodes (type_id=13) exist (Fix #4).",
    )
    parser.add_argument(
        "--check-unchecked-feature",
        action="store_true",
        help="Verify in_unchecked_block feature (feat[11]) fires (Fix #4).",
    )
    parser.add_argument(
        "--check-icfg-edges",
        action="store_true",
        help="Verify ICFG-Lite edges (ids 8–10) exist.",
    )
    parser.add_argument(
        "--check-edge-index-valid",
        action="store_true",
        help="Verify edge_index values are in [0, num_nodes).",
    )
    parser.add_argument(
        "--check-feature-dtypes",
        action="store_true",
        help="Verify x is float32 and edge_attr is int64.",
    )
    parser.add_argument(
        "--check-node-type-range",
        action="store_true",
        help="Verify node type_ids in [0, max(NODE_TYPES)].",
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Enable all optional checks.",
    )
    args = parser.parse_args()

    graphs_dir = Path(args.graphs_dir)
    if not graphs_dir.exists():
        print(f"ERROR: graphs directory does not exist: {graphs_dir}", flush=True)
        sys.exit(1)

    all_checks = args.check_all

    sys.exit(validate(
        graphs_dir,
        expected_node_dim=args.check_dim,
        expected_edge_types=args.check_edge_types,
        check_contains_edges=args.check_contains_edges or all_checks,
        check_control_flow=args.check_control_flow or all_checks,
        check_cfg_subtypes=args.check_cfg_subtypes or all_checks,
        check_block_globals=args.check_block_globals or all_checks,
        check_external_call_edges=args.check_external_call_edges or all_checks,
        check_arith_nodes=args.check_arith_nodes or all_checks,
        check_unchecked_feature=args.check_unchecked_feature or all_checks,
        check_icfg_edges=args.check_icfg_edges or all_checks,
        check_edge_index_valid=args.check_edge_index_valid or all_checks,
        check_feature_dtypes=args.check_feature_dtypes or all_checks,
        check_node_type_range=args.check_node_type_range or all_checks,
    ))


if __name__ == "__main__":
    main()
