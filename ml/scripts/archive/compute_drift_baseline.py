#!/usr/bin/env python3
"""
compute_drift_baseline.py — Build drift_baseline.json for T2-B drift detection.

WARNING — read before using --source training:
──────────────────────────────────────────────
DO NOT use the training data (ml/data/graphs/) as the baseline.  The
BCCC-SCsVul-2024 corpus is a 2024 historical snapshot; using it will cause
the KS test to fire on virtually every modern 2026 production contract,
making drift alerts meaningless.

Recommended workflow:
  1. Start the API (Move 0–6 complete).
  2. Let it serve at least 500 real audit requests (warm-up phase).
  3. Export the warm-up buffer to a JSONL file using /debug/warmup_dump
     (or call DriftDetector.dump_warmup_stats() in a management command).
  4. Run:
       python ml/scripts/compute_drift_baseline.py \\
           --source warmup \\
           --warmup-log ml/data/warmup_stats.jsonl \\
           --output ml/data/drift_baseline.json

Usage:
    # From real request warm-up (recommended)
    python ml/scripts/compute_drift_baseline.py \\
        --source warmup \\
        --warmup-log <path_to_jsonl> \\
        --output ml/data/drift_baseline.json

    # From training graphs (NOT recommended — will cause false alerts)
    python ml/scripts/compute_drift_baseline.py \\
        --source training \\
        --graphs-dir ml/data/graphs \\
        --output ml/data/drift_baseline.json

Exit codes:
    0  baseline written successfully
    1  error (file not found, empty input, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

STAT_NAMES = ["num_nodes", "num_edges"]


def _extract_stats_from_graph(path: Path) -> dict[str, float] | None:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"  SKIP {path.name}: {exc}", file=sys.stderr)
        return None

    return {
        "num_nodes": float(data.num_nodes or 0),
        "num_edges": float(data.num_edges or 0),
    }


def from_warmup(warmup_log: Path, output: Path) -> int:
    if not warmup_log.exists():
        print(f"ERROR: warmup log not found: {warmup_log}", file=sys.stderr)
        return 1

    records: list[dict[str, float]] = []
    with open(warmup_log) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  SKIP line {lineno}: {exc}", file=sys.stderr)

    if len(records) < 30:
        print(
            f"ERROR: warmup log has only {len(records)} records — need at least 30 "
            "for a reliable baseline. Collect more requests first.",
            file=sys.stderr,
        )
        return 1

    baseline: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        for k, v in rec.items():
            baseline[k].append(float(v))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(dict(baseline), f, indent=2)

    print(f"\nBaseline written — {output}")
    print(f"  Records      : {len(records)}")
    print(f"  Stats        : {sorted(baseline.keys())}")
    return 0


def from_training(graphs_dir: Path, output: Path) -> int:
    print(
        "\n⚠️  WARNING: --source training uses BCCC-SCsVul-2024 data.\n"
        "   This baseline will fire alerts on most modern 2026 contracts.\n"
        "   Use --source warmup for production deployments.\n",
        file=sys.stderr,
    )

    pt_files = sorted(graphs_dir.glob("*.pt"))
    if not pt_files:
        print(f"ERROR: no .pt files found in {graphs_dir}", file=sys.stderr)
        return 1

    baseline: dict[str, list[float]] = defaultdict(list)
    skipped = 0
    for path in pt_files:
        stats = _extract_stats_from_graph(path)
        if stats is None:
            skipped += 1
            continue
        for k, v in stats.items():
            baseline[k].append(v)

    if not baseline:
        print("ERROR: no stats extracted — all files failed to load.", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(dict(baseline), f, indent=2)

    print(f"\nBaseline written — {output}")
    print(f"  Files processed: {len(pt_files) - skipped}")
    print(f"  Files skipped  : {skipped}")
    print(f"  Stats          : {sorted(baseline.keys())}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build drift_baseline.json for sentinel_drift_alerts_total."
    )
    parser.add_argument(
        "--source", required=True, choices=["warmup", "training"],
        help="Data source. 'warmup' is strongly preferred over 'training'.",
    )
    parser.add_argument(
        "--warmup-log",
        help="Path to JSONL file from DriftDetector.dump_warmup_stats() (--source warmup).",
    )
    parser.add_argument(
        "--graphs-dir",
        default=str(Path(__file__).parent.parent / "data" / "graphs"),
        help="Directory of graph .pt files (--source training only).",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "data" / "drift_baseline.json"),
        help="Output path for drift_baseline.json.",
    )
    args = parser.parse_args()
    output = Path(args.output)

    if args.source == "warmup":
        if not args.warmup_log:
            print("ERROR: --warmup-log is required when --source warmup", file=sys.stderr)
            sys.exit(1)
        sys.exit(from_warmup(Path(args.warmup_log), output))
    else:
        sys.exit(from_training(Path(args.graphs_dir), output))


if __name__ == "__main__":
    main()
