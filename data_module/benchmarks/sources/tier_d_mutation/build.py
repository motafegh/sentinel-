"""Tier D builder — mutation-based synthetic contracts (for rare classes).

Methodology:
  1. Start with a known-clean contract (verified no 2-tool findings)
  2. Apply a known bug pattern from SWC Registry
  3. Re-compile; verify the bug is now detectable by at least one tool
  4. Result: synthetic contract with verified-vulnerable label

Patterns (one per file in patterns/):
  - unchecked_send.py         → MishandledException
  - unbounded_loop.py         → DenialOfService
  - tx_origin.py              → TransactionOrderDependence
  - low_level_call.py         → CallToUnknown
  - timestamp_comparison.py   → Timestamp
  - reentrancy_no_mutex.py    → Reentrancy
  - integer_overflow.py       → IntegerUO
  - return_value_ignored.py   → UnusedReturn
  - arbitrary_external_call.py → ExternalBug

Output: benchmark_v<N>/contracts/by_class/<ClassName>/mutation_<n>.sol
        benchmark_v<N>/tier_d_manifest.json
"""
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
PATTERNS_DIR = Path(__file__).parent / "patterns"


def build_tier_d(output_dir: Path, n_per_class: int = 50) -> dict:
    """Build Tier D: run all mutation patterns, compile + verify, collect."""
    print(f"\n=== Building Tier D (mutation-based) ===")
    print(f"Output: {output_dir}")
    print(f"Target: {n_per_class} contracts per class")

    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir = output_dir / "contracts" / "by_class"
    contracts_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    pattern_files = sorted(PATTERNS_DIR.glob("*.py"))
    print(f"\n  Found {len(pattern_files)} pattern files: {[p.stem for p in pattern_files]}")

    for pattern_file in pattern_files:
        pattern_name = pattern_file.stem
        print(f"\n  Pattern: {pattern_name}")
        # TODO: import the pattern module, call its `apply(source_contract) -> mutated`
        # For now, this is a SKELETON — patterns are TODOs.
        # Each pattern should:
        #   1. Take a known-clean contract
        #   2. Apply the bug
        #   3. Return the mutated source + class label
        # Then we compile + verify with 2-tool.

        manifest.append({
            "pattern": pattern_name,
            "status": "SKELETON — TODO: implement pattern",
            "n_contracts": 0,
        })

    manifest_path = output_dir / "tier_d_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest written: {manifest_path}")

    return {
        "tier": "D",
        "status": "SKELETON",
        "manifest_path": str(manifest_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-per-class", type=int, default=50)
    args = parser.parse_args()
    result = build_tier_d(Path(args.output_dir), args.n_per_class)
    print(f"\nTier D done: {result}")
