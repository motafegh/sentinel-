"""
audit_prefix_node_counts.py — PRE-4: Count eligible GNN prefix nodes per contract.

Reads every graph .pt file in the graphs directory, counts nodes whose type_id
(feature dim [0] * 12.0 rounded to int) falls in STRUCTURAL_PREFIX_TYPES, and
reports the distribution.  Gate: P95 ≤ 32 (K=32 covers 95% of contracts).

Usage:
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/audit_prefix_node_counts.py \
        --graphs-dir ml/data/graphs/ \
        --out ml/logs/prefix_node_count_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.src.preprocessing.graph_schema import STRUCTURAL_PREFIX_TYPES


def _eligible_count(graph_path: Path) -> int:
    data = torch.load(graph_path, weights_only=False)
    x = data.x  # [N, 11]
    type_ids = (x[:, 0] * 12.0).round().long()  # feature[0] = type_id / 12.0
    return int((type_ids.unsqueeze(1) == torch.tensor(list(STRUCTURAL_PREFIX_TYPES))).any(dim=1).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    graph_files = sorted(args.graphs_dir.glob("*.pt"))
    if not graph_files:
        print(f"No .pt files found in {args.graphs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Counting eligible prefix nodes across {len(graph_files):,} graphs...")
    counts: list[int] = []
    errors = 0
    for i, p in enumerate(graph_files):
        try:
            counts.append(_eligible_count(p))
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [WARN] {p.name}: {e}", file=sys.stderr)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,} / {len(graph_files):,} done...")

    arr = np.array(counts)
    percentiles = {str(p): float(np.percentile(arr, p)) for p in [50, 90, 95, 99, 100]}

    result = {
        "num_graphs": len(counts),
        "errors": errors,
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "percentiles": percentiles,
        "eligible_types": sorted(int(t) for t in STRUCTURAL_PREFIX_TYPES),
        "k_recommendations": {
            "K=8":  f"covers {100 * (arr <= 8).mean():.1f}% of contracts without truncation",
            "K=16": f"covers {100 * (arr <= 16).mean():.1f}% of contracts without truncation",
            "K=32": f"covers {100 * (arr <= 32).mean():.1f}% of contracts without truncation",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("\n── Prefix node count distribution ──────────────────")
    print(f"  Graphs audited : {len(counts):,}  (errors: {errors})")
    print(f"  Min            : {result['min']}")
    print(f"  Mean           : {result['mean']:.1f}")
    for p, v in percentiles.items():
        print(f"  P{p:<3}           : {v:.0f}")
    print()
    for k, desc in result["k_recommendations"].items():
        print(f"  {k}: {desc}")
    print()

    p95 = percentiles["95"]
    if p95 <= 32:
        print(f"PRE-4 PASSED — P95={p95:.0f} ≤ 32  (K=32 safe as upper bound)")
    else:
        print(f"PRE-4 FAILED — P95={p95:.0f} > 32  (reduce K default or prune node types)")
        sys.exit(1)

    print(f"\nFull results written to: {args.out}")


if __name__ == "__main__":
    main()
