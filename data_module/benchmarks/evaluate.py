"""evaluate.py — SENTINEL benchmark evaluator (replaces benchmark_run9_*).

Replaces the contaminated benchmark scripts (benchmark_run9_smartbugs.py +
benchmark_run9_solidifi.py) with a unified evaluator that:
  1. Loads any benchmark_v<N>/
  2. Loads any checkpoint (Run N agnostic)
  3. Loads per-class tuned thresholds from Phase 2 calibration
  4. Reports per-class F1 under BOTH tier + tuned modes
  5. Reports per-tier breakdown (Tier A vs B vs C vs D vs E)
  6. Reports per-source breakdown
  7. Reports OOD analysis (graph size, version mix)
  8. Verifies contamination status (0 overlap with v3 expected)

Usage:
    python -m data_module.benchmarks.evaluate \
        --checkpoint ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt \
        --benchmark-dir data_module/benchmarks/benchmark_v0.1_quickstart/ \
        --thresholds ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best_thresholds.json \
        --output ml/reports/Run12_benchmark_v0.1.json

    # Compare across runs
    python -m data_module.benchmarks.evaluate --compare \
        --reports ml/reports/Run{9,10,11,12}_benchmark_v0.1.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path("/home/motafeq/projects/sentinel")


def list_benchmark_contracts(bench_dir: Path) -> list:
    """Return list of {path, class, sha256, size_bytes, manifest_entry}."""
    contracts = []
    manifest_path = bench_dir / "manifest.json"
    if manifest_path.exists():
        # Use manifest if available
        m = json.loads(manifest_path.read_text())
        for entry in m.get("contracts", []):
            contracts.append(entry)
    else:
        # Fallback: scan contracts/by_class/*.sol
        contracts_root = bench_dir / "contracts" / "by_class"
        for cls_dir in sorted(contracts_root.iterdir()):
            if not cls_dir.is_dir():
                continue
            for sol in sorted(cls_dir.glob("*.sol")):
                contracts.append({
                    "dest_path": str(sol.relative_to(bench_dir)),
                    "sentinal_class": cls_dir.name,
                    "name": sol.name,
                })
    return contracts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--benchmark-dir", help="Path to benchmark_v<N>/")
    parser.add_argument("--thresholds", help="Path to <stem>_thresholds.json (from Phase 2 calibration)")
    parser.add_argument("--tier-confirmed", type=float, default=0.55)
    parser.add_argument("--tier-suspicious", type=float, default=0.25)
    parser.add_argument("--output", help="Path to write JSON report")
    parser.add_argument("--compare", action="store_true",
                        help="Compare mode: pass --reports with comma-separated paths")
    parser.add_argument("--reports", help="Comma-separated paths to previous reports for comparison")
    args = parser.parse_args()

    if args.compare:
        if not args.reports:
            print("ERROR: --reports required with --compare")
            sys.exit(1)
        paths = [Path(p) for p in args.reports.split(",")]
        print(f"\n{'='*70}")
        print(f"Comparing {len(paths)} benchmark reports")
        print(f"{'='*70}\n")
        for p in paths:
            if not p.exists():
                print(f"  {p}: NOT FOUND")
                continue
            r = json.loads(p.read_text())
            print(f"  {p.name}:")
            print(f"    F1 (tier):  {r.get('f1_tier', 'N/A')}")
            print(f"    F1 (tuned): {r.get('f1_tuned', 'N/A')}")
            if 'per_class_f1' in r:
                for cls, f1 in r['per_class_f1'].items():
                    print(f"    {cls:30s}: tier={f1.get('tier', 'N/A'):.3f} tuned={f1.get('tuned', 'N/A'):.3f}")
        return

    # Normal evaluation
    if not all([args.checkpoint, args.benchmark_dir, args.thresholds]):
        print("ERROR: --checkpoint, --benchmark-dir, --thresholds all required")
        sys.exit(1)

    checkpoint = Path(args.checkpoint)
    bench_dir = Path(args.benchmark_dir)
    thresholds_path = Path(args.thresholds)

    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        sys.exit(1)
    if not bench_dir.exists():
        print(f"ERROR: benchmark dir not found: {bench_dir}")
        sys.exit(1)
    if not thresholds_path.exists():
        print(f"ERROR: thresholds not found: {thresholds_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"SENTINEL Benchmark Evaluation")
    print(f"{'='*70}\n")
    print(f"Checkpoint:    {checkpoint}")
    print(f"Benchmark dir: {bench_dir}")
    print(f"Thresholds:    {thresholds_path}")
    print(f"Tier confirmed: {args.tier_confirmed}, suspicious: {args.tier_suspicious}")

    # Load benchmark contracts
    contracts = list_benchmark_contracts(bench_dir)
    print(f"\nBenchmark contracts: {len(contracts)}")
    class_counts = Counter(c.get("sentinal_class", "Unknown") for c in contracts)
    print(f"Per-class counts:")
    for cls, n in sorted(class_counts.items()):
        print(f"  {cls:30s}: {n}")

    # Check contamination status
    contam_path = bench_dir / "contamination_check.json"
    if contam_path.exists():
        contam = json.loads(contam_path.read_text())
        print(f"\nContamination check ({contam_path}):")
        print(f"  Overlap with v3: {contam.get('overlap_count', 'N/A')} / {contam.get('total_contracts', 'N/A')}")
    else:
        print(f"\nWARNING: no contamination_check.json in {bench_dir}")
        print(f"  Run: ml/.venv/bin/python -m data_module.benchmarks.contamination_check --version {bench_dir.name.replace('benchmark_', '')}")

    # TODO: actual evaluation loop using the inference API
    # For now, this is a SKELETON — the actual implementation would:
    # 1. Initialize the inference predictor (ml.src.inference.predictor.Predictor)
    # 2. For each contract, call .predict(source_code) → probabilities dict
    # 3. Apply thresholds (tier + tuned) to determine predictions
    # 4. Compute per-class F1 vs ground truth (sentinal_class)
    # 5. Report per-tier and per-source breakdowns
    print(f"\n{'='*70}")
    print(f"ACTUAL EVALUATION NOT YET IMPLEMENTED")
    print(f"{'='*70}")
    print(f"\nTODO:")
    print(f"  1. Initialize Predictor with --checkpoint {checkpoint}")
    print(f"  2. Load thresholds from {thresholds_path}")
    print(f"  3. For each of {len(contracts)} contracts, predict()")
    print(f"  4. Apply tier + tuned thresholds")
    print(f"  5. Compute per-class F1 (10 classes)")
    print(f"  6. Compute per-tier F1 (Tier A vs B vs C vs D vs E)")
    print(f"  7. Write report to {args.output or 'ml/reports/benchmark_<run_name>.json'}")


if __name__ == "__main__":
    main()
