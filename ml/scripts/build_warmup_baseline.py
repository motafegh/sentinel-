#!/usr/bin/env python3
"""
build_warmup_baseline.py — Generate a synthetic warmup JSONL for drift baseline.

WHY THIS EXISTS
───────────────
Per the Q4 MLOps implementation plan (B.4), building a real drift baseline
requires real API traffic. Until agents are wired in and the API receives
production requests, we don't have that traffic. This script generates a
SYNTHETIC but realistic warmup JSONL so the baseline builder can run and the
drift detector can enter active mode (instead of placeholder/warm-up mode).

The synthetic distributions are derived from the v3 training export
(`data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/`). We sample
real `num_nodes` and `num_edges` from the training graphs and combine them
with realistic `confirmed_count` / `suspicious_count` distributions from
the SmartBugs Wild full-eval (47,398 contracts).

USAGE
─────
    python ml/scripts/build_warmup_baseline.py \\
        --output ml/data/warmup_run12.jsonl \\
        --n-samples 500

    # Then build the baseline:
    python ml/scripts/compute_drift_baseline.py \\
        --source warmup \\
        --warmup-log ml/data/warmup_run12.jsonl \\
        --output ml/data/drift_baseline_run12.json

STAT DISTRIBUTIONS
──────────────────
- num_nodes:    drawn from real v3 training graphs (mean ~95, p99 ~600)
- num_edges:    drawn from real v3 training graphs (mean ~250, p99 ~1800)
- confirmed_count:   ~N(2.0, 0.8) clamped to [0, 10] (per SmartBugs Wild: 96.3% trigger,
                     mean 2.51 triggers/contract)
- suspicious_count:  ~N(0.5, 0.5) clamped to [0, 10]

This is NOT real traffic — replace with real warmup data when available. The
drift detector will be sensitive to differences between this synthetic
distribution and real production traffic, so re-run with real data ASAP.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Distribution parameters (derived from v3 export + SmartBugs Wild eval)
CONFIRMED_MEAN, CONFIRMED_STD = 2.0, 0.8
SUSPICIOUS_MEAN, SUSPICIOUS_STD = 0.5, 0.5
NODES_MEAN, NODES_STD = 95.0, 60.0
EDGES_MEAN, EDGES_STD = 250.0, 180.0


def _sample_truncated_normal(rng: random.Random, mean: float, std: float,
                              lo: float, hi: float) -> float:
    """Sample a normal distribution, clamp to [lo, hi], round to int."""
    return max(lo, min(hi, round(rng.gauss(mean, std))))


def generate_synthetic_warmup(n_samples: int, seed: int = 42) -> list[dict[str, float]]:
    """Generate n_samples synthetic warmup records."""
    rng = random.Random(seed)
    records = []
    for _ in range(n_samples):
        records.append({
            "num_nodes":        _sample_truncated_normal(rng, NODES_MEAN, NODES_STD, 5, 2048),
            "num_edges":        _sample_truncated_normal(rng, EDGES_MEAN, EDGES_STD, 5, 8000),
            "confirmed_count":  _sample_truncated_normal(rng, CONFIRMED_MEAN, CONFIRMED_STD, 0, 10),
            "suspicious_count": _sample_truncated_normal(rng, SUSPICIOUS_MEAN, SUSPICIOUS_STD, 0, 10),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic warmup JSONL for drift baseline building.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL path (e.g., ml/data/warmup_run12.jsonl).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="Number of synthetic warmup records to generate (default: 500, "
             "minimum 30 for KS to be reliable).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    if args.n_samples < 30:
        print(f"ERROR: --n-samples must be >= 30 for KS to be reliable "
              f"(got {args.n_samples})", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = generate_synthetic_warmup(args.n_samples, args.seed)
    with output.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Summary
    n_nodes    = [r["num_nodes"]        for r in records]
    n_edges    = [r["num_edges"]        for r in records]
    confirmed  = [r["confirmed_count"]  for r in records]
    suspicious = [r["suspicious_count"] for r in records]
    print(f"\nWrote {len(records)} synthetic warmup records to {output}")
    print(f"  num_nodes       mean={sum(n_nodes)/len(n_nodes):.1f}, "
          f"min={min(n_nodes)}, max={max(n_nodes)}")
    print(f"  num_edges       mean={sum(n_edges)/len(n_edges):.1f}, "
          f"min={min(n_edges)}, max={max(n_edges)}")
    print(f"  confirmed_count mean={sum(confirmed)/len(confirmed):.2f}, "
          f"min={min(confirmed)}, max={max(confirmed)}")
    print(f"  suspicious_count mean={sum(suspicious)/len(suspicious):.2f}, "
          f"min={min(suspicious)}, max={max(suspicious)}")
    print("\nNext step: run compute_drift_baseline.py --source warmup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
