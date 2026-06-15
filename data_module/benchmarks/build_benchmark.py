"""build_benchmark.py — Orchestrate the SENTINEL comprehensive benchmark build.

Phases:
1. Tier A: 6 SmartBugs + 60 SolidiFI honest OOD (~1 hour)
2. Tier E: 100 NonVulnerable known-safe (~4-6 hours, parallel)
3. Tier B: 149 DeFiHackLabs held-out (~4-6 hours)
4. Tier C: ~2,000-3,000 BCCC 2-tool consensus (~3-5 days)
5. Tier D: 500-1000 mutation-based (~2-3 days)
6. Aggregate: combine all tiers into benchmark_v<N>/
7. Contamination check: SHA-256 vs v3 (must show 0 overlap)
8. Stats + manifest: per-class/per-source/per-tier counts
9. Version tag: save as benchmark_v<N>/

Usage:
    python -m data_module.benchmarks.build_benchmark --tier quickstart  # A + E only
    python -m data_module.benchmarks.build_benchmark --tier a
    python -m data_module.benchmarks.build_benchmark --tier all          # All tiers
    python -m data_module.benchmarks.build_benchmark --rebuild v0.1
    python -m data_module.benchmarks.build_benchmark --verify v0.1
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
BENCH_ROOT = REPO_ROOT / "data_module" / "benchmarks"
SOURCES_DIR = BENCH_ROOT / "sources"


def build_tier(tier: str, output_dir: Path) -> dict:
    """Build a single tier via its builder script."""
    builders = {
        "a": SOURCES_DIR / "tier_a_existing_ood" / "build.py",
        "b": SOURCES_DIR / "tier_b_defihacklabs_heldout" / "build.py",
        "c": SOURCES_DIR / "tier_c_bccc_2tool" / "consensus.py",
        "d": SOURCES_DIR / "tier_d_mutation" / "build.py",
        "e": SOURCES_DIR / "tier_e_safe" / "build.py",
    }
    if tier not in builders:
        raise ValueError(f"Unknown tier: {tier}")
    script = builders[tier]
    if not script.exists():
        return {"tier": tier, "status": "BUILDER_NOT_FOUND", "path": str(script)}

    print(f"\n>>> Building tier {tier.upper()}: {script}")
    result = subprocess.run(
        ["ml/.venv/bin/python", str(script), "--output-dir", str(output_dir)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"WARNING: tier {tier} build returned {result.returncode}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    return {"tier": tier, "status": "OK" if result.returncode == 0 else "FAIL", "returncode": result.returncode}


def build_quickstart(version: str = "v0.1") -> Path:
    """Build Tier A + E only (Day 1 quickstart)."""
    output_dir = BENCH_ROOT / f"benchmark_{version}_quickstart"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Building SENTINEL benchmark {version} (Tier A + E quickstart)")
    print(f"{'='*70}\n")

    a_result = build_tier("a", output_dir)
    e_result = build_tier("e", output_dir)

    # Run contamination check
    print(f"\n>>> Running contamination check on {output_dir}")
    subprocess.run(
        ["ml/.venv/bin/python", "-m", "data_module.benchmarks.contamination_check",
         "--version", f"{version}_quickstart"],
        cwd=str(REPO_ROOT)
    )

    return output_dir


def build_full(version: str = "v1.0") -> Path:
    """Build all tiers (Tier A + B + C + D + E)."""
    output_dir = BENCH_ROOT / f"benchmark_{version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Building SENTINEL benchmark {version} (ALL tiers)")
    print(f"{'='*70}\n")

    # Build each tier
    for tier in ["a", "b", "c", "d", "e"]:
        build_tier(tier, output_dir)

    # Contamination check
    print(f"\n>>> Running contamination check on {output_dir}")
    subprocess.run(
        ["ml/.venv/bin/python", "-m", "data_module.benchmarks.contamination_check",
         "--version", version],
        cwd=str(REPO_ROOT)
    )

    return output_dir


def write_manifest(output_dir: Path, tier_results: dict) -> None:
    """Write the overall manifest.json for this benchmark version."""
    manifest = {
        "benchmark_version": output_dir.name,
        "build_date": datetime.now().isoformat(),
        "v3_artifact_hash": "5cc5cfcbf42bef4ced58b963ef98241bcf3ec4ab3bea5d198f336ec763a4faa9",
        "tier_results": tier_results,
        "design_doc": str(BENCH_ROOT / "BENCHMARK_DESIGN.md"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["a", "b", "c", "d", "e", "all", "quickstart"],
                        default="quickstart")
    parser.add_argument("--version", default="v0.1", help="Benchmark version tag")
    args = parser.parse_args()

    if args.tier == "quickstart":
        output_dir = build_quickstart(args.version)
    elif args.tier == "all":
        output_dir = build_full(args.version)
    else:
        output_dir = BENCH_ROOT / f"benchmark_{args.version}"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = build_tier(args.tier, output_dir)
        write_manifest(output_dir, {args.tier: result})

    print(f"\n{'='*70}")
    print(f"DONE. Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
