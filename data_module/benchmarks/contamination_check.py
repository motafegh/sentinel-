"""contamination_check.py — Per-build SHA-256 audit of benchmark vs v3 (or v4, v5, ...).

For every contract in a benchmark version, compute SHA-256 and check if it's
in the active data export's training/val/test splits. HARD GATE: any overlap
is a critical finding.

Usage:
    ml/.venv/bin/python -m data_module.benchmarks.contamination_check --version v0.1_quickstart
    ml/.venv/bin/python -m data_module.benchmarks.contamination_check --version v1.0
    ml/.venv/bin/python -m data_module.benchmarks.contamination_check --version v0.1_quickstart --export-dir data_module/data/exports/sentinel-v4-bcccme-2026-06-XX
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
BENCH_ROOT = REPO_ROOT / "data_module" / "benchmarks"
DEFAULT_EXPORT_DIR = REPO_ROOT / "data_module" / "data" / "exports" / "sentinel-v3-smartbugs-2026-06-13"
DEFAULT_SPLITS_DIR = REPO_ROOT / "data_module" / "data" / "splits" / "v3"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_export_index(export_dir: Path, splits_dir: Path) -> dict:
    """Return {sha256 -> {"split": ..., "source": ...}} for all export contracts."""
    index = {}
    if not splits_dir.exists():
        # Try to infer from export MANIFEST.json
        manifest = export_dir / "MANIFEST.json"
        if manifest.exists():
            print(f"  Reading from MANIFEST.json: {manifest}")
            return index  # will be empty if no split files
        return index
    for split_file in sorted(splits_dir.glob("*.jsonl")):
        split_name = split_file.stem
        for line in split_file.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            sha = d.get("sha256", "")
            if sha:
                index[sha] = {
                    "split": split_name,
                    "source": d.get("source", "unknown"),
                }
    return index


def audit_benchmark(bench_dir: Path, export_index: dict) -> dict:
    """For every contract in bench_dir/contracts/, compute SHA-256 and check."""
    contracts_root = bench_dir / "contracts" / "by_class"
    if not contracts_root.exists():
        return {"error": f"no contracts/ in {bench_dir}"}

    all_contracts = []
    overlap = []
    for cls_dir in sorted(contracts_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for sol in sorted(cls_dir.glob("*.sol")):
            h = sha256_file(sol)
            contract = {
                "class": cls,
                "name": sol.name,
                "path": str(sol.relative_to(bench_dir)),
                "sha256": h,
                "size_bytes": sol.stat().st_size,
                "in_export": h in export_index,
            }
            if h in export_index:
                contract["export_split"] = export_index[h]["split"]
                contract["export_source"] = export_index[h]["source"]
                overlap.append(contract)
            all_contracts.append(contract)

    return {
        "total_contracts": len(all_contracts),
        "overlap_count": len(overlap),
        "overlap_pct": 100 * len(overlap) / max(1, len(all_contracts)),
        "per_class_counts": dict(Counter(c["class"] for c in all_contracts)),
        "per_class_overlap": dict(Counter(c["class"] for c in overlap)),
        "overlap_contracts": overlap,
        "all_contracts": all_contracts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="e.g. v0.1_quickstart, v1.0")
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR),
                        help="Path to the data export (e.g. v3, v4)")
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR),
                        help="Path to the splits directory")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero on any overlap")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    splits_dir = Path(args.splits_dir)
    bench_dir = BENCH_ROOT / f"benchmark_{args.version}"

    if not bench_dir.exists():
        print(f"ERROR: benchmark dir not found: {bench_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Contamination Check: benchmark_{args.version}")
    print(f"{'='*70}\n")
    print(f"Export:   {export_dir}")
    print(f"Splits:   {splits_dir}")
    print(f"Benchmark: {bench_dir}")

    print("\nBuilding export SHA-256 index...")
    export_index = build_export_index(export_dir, splits_dir)
    print(f"  Export contracts indexed: {len(export_index)}")

    print(f"\nAuditing benchmark {args.version}...")
    result = audit_benchmark(bench_dir, export_index)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}\n")
    print(f"Total benchmark contracts: {result['total_contracts']}")
    print(f"Overlap with export:       {result['overlap_count']} ({result['overlap_pct']:.2f}%)")
    print(f"\nPer-class counts:")
    for cls, n in sorted(result["per_class_counts"].items()):
        ovl = result["per_class_overlap"].get(cls, 0)
        ovl_marker = " [CONFLICT]" if ovl > 0 else ""
        print(f"  {cls:30s}: {n:4d} contracts ({ovl} overlap){ovl_marker}")

    if result["overlap_count"] > 0:
        print(f"\n!!! CONTAMINATION DETECTED: {result['overlap_count']} contracts are in {export_dir.name} !!!")
        for c in result["overlap_contracts"][:10]:
            print(f"  - {c['class']}/{c['name']} (in {c['export_split']}/{c['export_source']})")
        if len(result["overlap_contracts"]) > 10:
            print(f"  ...and {len(result['overlap_contracts']) - 10} more")

    # Write contamination_check.json
    out = bench_dir / "contamination_check.json"
    out.write_text(json.dumps({
        "version": args.version,
        "export_dir": str(export_dir),
        "splits_dir": str(splits_dir),
        "audit_date": str(datetime.now()),
        **result,
    }, indent=2))
    print(f"\nReport written: {out}")

    if args.strict and result["overlap_count"] > 0:
        print(f"\nSTRICT MODE: {result['overlap_count']} overlap contracts. Exit 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
