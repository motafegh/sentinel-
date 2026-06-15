"""Tier A builder — copy 6 SmartBugs + 60 SolidiFI honest OOD contracts.

These are contracts NOT in v3 training/val/test (verified by
contamination_v3_*.json from the v3-aware audit). Labels are trusted:
- SmartBugs = DASP-10 taxonomy (well-known real-world)
- SolidiFI = synthetic single-bug injection (known ground truth)

Output: benchmark_v<N>/contracts/by_class/<ClassName>/<source>_<n>.sol
        benchmark_v<N>/tier_a_manifest.json (per-contract metadata)
"""
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
SMARTBUGS_DIR = REPO_ROOT / "ml" / "data" / "smartbugs-curated" / "dataset"
SOLIDIFI_BENCH_DIR = REPO_ROOT / "ml" / "data" / "SolidiFI-benchmark" / "buggy_contracts"

# Category → SENTINEL class (from benchmark_run9_smartbugs.py + benchmark_run9_solidifi.py)
SMARTBUGS_CAT_TO_CLASS = {
    "reentrancy": "Reentrancy",
    "arithmetic": "IntegerUO",
    "unchecked_low_level_calls": "CallToUnknown",
    "denial_of_service": "DenialOfService",
    "front_running": "TransactionOrderDependence",
    "time_manipulation": "Timestamp",
    "access_control": "NonVulnerable",  # No SENTINEL equivalent; treat as safe for benchmark
    "bad_randomness": "NonVulnerable",
    "short_addresses": "NonVulnerable",
    "other": "NonVulnerable",
}

SOLIDIFI_CAT_TO_CLASS = {
    "Re-entrancy": "Reentrancy",
    "Overflow-Underflow": "IntegerUO",
    "TOD": "TransactionOrderDependence",
    "Timestamp-Dependency": "Timestamp",
    "Unchecked-Send": "CallToUnknown",
    "Unhandled-Exceptions": "MishandledException",
    "tx.origin": "NonVulnerable",  # No direct mapping; treat as safe
}


def get_v3_sha256_set() -> set:
    """Return SHA-256 set of all v3 contracts (train + val + test)."""
    v3_shas = set()
    splits_dir = REPO_ROOT / "data_module" / "data" / "splits" / "v3"
    for split_file in splits_dir.glob("*.jsonl"):
        for line in split_file.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            sha = d.get("sha256", "")
            if sha:
                v3_shas.add(sha)
    return v3_shas


def collect_honest_ood() -> list:
    """For each benchmark contract, compute SHA-256 and check vs v3."""
    import hashlib
    v3_shas = get_v3_sha256_set()
    print(f"  v3 contracts in training/val/test: {len(v3_shas)}")

    honest_ood = []

    # SmartBugs
    for cat_dir in sorted(SMARTBUGS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for sol in sorted(cat_dir.glob("*.sol")):
            raw = sol.read_bytes()
            h = hashlib.sha256(raw).hexdigest()
            if h not in v3_shas:
                honest_ood.append({
                    "source": "smartbugs_curated",
                    "category": cat_dir.name,
                    "name": sol.name,
                    "path": sol,
                    "sha256": h,
                    "size_bytes": len(raw),
                    "sentinal_class": SMARTBUGS_CAT_TO_CLASS.get(cat_dir.name, "NonVulnerable"),
                })

    # SolidiFI
    for cat_dir in sorted(SOLIDIFI_BENCH_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for sol in sorted(cat_dir.glob("*.sol")):
            raw = sol.read_bytes()
            h = hashlib.sha256(raw).hexdigest()
            if h not in v3_shas:
                honest_ood.append({
                    "source": "solidifi_benchmark",
                    "category": cat_dir.name,
                    "name": sol.name,
                    "path": sol,
                    "sha256": h,
                    "size_bytes": len(raw),
                    "sentinal_class": SOLIDIFI_CAT_TO_CLASS.get(cat_dir.name, "NonVulnerable"),
                })

    return honest_ood


def build_tier_a(output_dir: Path) -> dict:
    """Build Tier A: copy honest OOD contracts into output_dir/contracts/by_class/."""
    print(f"\n=== Building Tier A (existing honest OOD) ===")
    print(f"Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir = output_dir / "contracts" / "by_class"
    contracts_dir.mkdir(parents=True, exist_ok=True)

    print("\nCollecting honest OOD contracts (SHA-256 vs v3)...")
    contracts = collect_honest_ood()
    print(f"  Found: {len(contracts)} honest OOD contracts")

    # Group by class
    by_class = {}
    for c in contracts:
        cls = c["sentinal_class"]
        by_class.setdefault(cls, []).append(c)
    print(f"  Per class:")
    for cls, items in sorted(by_class.items()):
        print(f"    {cls:30s}: {len(items)} contracts")

    # Copy files
    manifest = []
    for cls, items in by_class.items():
        cls_dir = contracts_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for c in items:
            new_name = f"{c['source']}_{c['category']}_{c['name']}"
            # Sanitize name
            new_name = new_name.replace("/", "_").replace(" ", "_")
            dest = cls_dir / new_name
            shutil.copy2(c["path"], dest)
            manifest.append({
                "source": c["source"],
                "category": c["category"],
                "original_name": c["name"],
                "dest_path": str(dest.relative_to(output_dir)),
                "sha256": c["sha256"],
                "size_bytes": c["size_bytes"],
                "sentinal_class": cls,
                "tier": "A",
            })

    # Write manifest
    manifest_path = output_dir / "tier_a_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest written: {manifest_path}")

    return {
        "tier": "A",
        "total_contracts": len(contracts),
        "per_class": {cls: len(items) for cls, items in by_class.items()},
        "manifest_path": str(manifest_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Path to benchmark_v<N>/")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    result = build_tier_a(output_dir)
    print(f"\nTier A done: {result}")
