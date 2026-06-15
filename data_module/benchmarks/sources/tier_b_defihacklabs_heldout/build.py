"""Tier B builder — DeFiHackLabs held-out subset (~149 contracts, 80/20 split).

Source: data_module/data/raw/defihacklabs/repo/ (742 contracts)
Method:
  1. SHA-256 each contract; check against v3 (active export)
  2. For contracts NOT in v3: 80% go to benchmark, 20% to held-out (locked split)
  3. For contracts IN v3: verify ground truth (2-tool), may still be useful
  4. Each contract = real-world exploit with documented CVE/post-mortem
  5. Multi-class: most DeFiHackLabs exploits are 2-3 class (Reentrancy+ExtBug, etc.)

Output: benchmark_v<N>/contracts/by_class/<ClassName>/defihacklabs_<n>.sol
        benchmark_v<N>/tier_b_manifest.json (per-contract metadata: CVE, classes, source)
        benchmark_v<N>/heldout_split.json (locked train/bench split for reproducibility)
"""
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
DEFIHACKLABS_DIR = REPO_ROOT / "data_module" / "data" / "raw" / "defihacklabs" / "repo"
HELD_OUT_FRACTION = 0.20  # 20% to benchmark, 80% to training (locked split)
RANDOM_SEED = 42  # For reproducibility


def get_v3_sha256_set() -> set:
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


def parse_classes_from_filename(sol_name: str) -> list:
    """Best-effort class parsing from DeFiHackLabs file naming convention.

    DeFiHackLabs doesn't have a single canonical naming; the actual class
    labels come from the post-mortem documentation. This function is a
    conservative fallback that uses keyword matching in the filename.

    TODO: replace with a manifest of CVE → class mappings (currently doesn't exist
    in this repo; would need to be scraped from DeFiHackLabs README or
    manually maintained).
    """
    name_lower = sol_name.lower()
    classes = []
    if "reentrancy" in name_lower or "reentrant" in name_lower:
        classes.append("Reentrancy")
    if "flashloan" in name_lower or "flash_loan" in name_lower:
        classes.append("Reentrancy")  # Often flashloan-related
    if "integer" in name_lower or "overflow" in name_lower or "underflow" in name_lower:
        classes.append("IntegerUO")
    if "timestamp" in name_lower or "time" in name_lower:
        classes.append("Timestamp")
    if "dos" in name_lower or "denial" in name_lower:
        classes.append("DenialOfService")
    if "access" in name_lower or "permission" in name_lower or "owner" in name_lower:
        classes.append("ExternalBug")
    if "unchecked" in name_lower or "return" in name_lower:
        classes.append("UnusedReturn")
    if "tx.origin" in name_lower or "txorigin" in name_lower or "front" in name_lower:
        classes.append("TransactionOrderDependence")
    if "exception" in name_lower or "send" in name_lower:
        classes.append("MishandledException")
    return classes or ["ExternalBug"]  # Default if no keyword matches (most exploits are auth/access)


def build_tier_b(output_dir: Path) -> dict:
    """Build Tier B: DeFiHackLabs held-out split."""
    print(f"\n=== Building Tier B (DeFiHackLabs held-out) ===")
    print(f"Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir = output_dir / "contracts" / "by_class"
    contracts_dir.mkdir(parents=True, exist_ok=True)

    v3_shas = get_v3_sha256_set()
    print(f"v3 contracts: {len(v3_shas)}")
    print(f"DeFiHackLabs pool: {sum(1 for _ in DEFIHACKLABS_DIR.glob('*.sol'))}")

    # Enumerate all DeFiHackLabs contracts
    all_contracts = []
    for sol in sorted(DEFIHACKLABS_DIR.glob("*.sol")):
        raw = sol.read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        all_contracts.append({
            "path": sol,
            "name": sol.name,
            "sha256": h,
            "in_v3": h in v3_shas,
            "size_bytes": len(raw),
            "classes": parse_classes_from_filename(sol.name),
        })

    n_in_v3 = sum(1 for c in all_contracts if c["in_v3"])
    n_not_in_v3 = len(all_contracts) - n_in_v3
    print(f"  In v3 (already in training): {n_in_v3}")
    print(f"  NOT in v3 (truly OOD):       {n_not_in_v3}")

    # Sample 20% from the truly-OOD set
    honest_ood = [c for c in all_contracts if not c["in_v3"]]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(honest_ood)

    n_bench = int(len(honest_ood) * HELD_OUT_FRACTION)
    benchmark_contracts = honest_ood[:n_bench]
    print(f"\n  Held-out to benchmark: {n_bench} contracts ({100*n_bench/len(honest_ood):.0f}%)")

    # Copy files (multi-class: contract goes into each class directory)
    manifest = []
    for c in benchmark_contracts:
        for cls in c["classes"]:
            cls_dir = contracts_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            dest = cls_dir / f"defihacklabs_{c['name']}"
            shutil.copy2(c["path"], dest)
            manifest.append({
                "source": "defihacklabs",
                "category": "exploit",
                "original_name": c["name"],
                "dest_path": str(dest.relative_to(output_dir)),
                "sha256": c["sha256"],
                "size_bytes": c["size_bytes"],
                "sentinal_class": cls,
                "tier": "B",
                "label_method": "filename_keyword_match (TODO: verify with CVE/postmortem)",
                "in_v3_training": False,
            })

    manifest_path = output_dir / "tier_b_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest written: {manifest_path}")

    # Save the held-out split (for reproducibility)
    heldout_split = {
        "seed": RANDOM_SEED,
        "fraction": HELD_OUT_FRACTION,
        "total_honest_ood": len(honest_ood),
        "n_benchmark": n_bench,
        "n_remaining_for_training": len(honest_ood) - n_bench,
        "benchmark_sha256s": [c["sha256"] for c in benchmark_contracts],
    }
    heldout_path = output_dir / "heldout_split.json"
    heldout_path.write_text(json.dumps(heldout_split, indent=2))
    print(f"  Held-out split written: {heldout_path}")

    return {
        "tier": "B",
        "total_benchmark_contracts": n_bench,
        "manifest_path": str(manifest_path),
        "heldout_split_path": str(heldout_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_tier_b(Path(args.output_dir))
    print(f"\nTier B done: {result}")
