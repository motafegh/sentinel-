"""Tier E builder — sample known-safe contracts for negative examples.

Sources:
- DISL NonVulnerable pool: BCCC-SCsVul-2024/SourceCodes/NonVulnerable/ (26,914 files)
- OpenZeppelin audited: download from github.com/OpenZeppelin/openzeppelin-contracts (TODO)

Method:
- Sample 100 contracts from NonVulnerable (stratified by compiler version if possible)
- Verify with 2-tool: NO findings (if any tool fires, exclude)
- These serve as TRUE NEGATIVES (model should predict "safe" with high confidence)

Output: benchmark_v<N>/contracts/by_class/NonVulnerable/<source>_<n>.sol
        benchmark_v<N>/tier_e_manifest.json
"""
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
NONVULN_DIR = REPO_ROOT / "BCCC-SCsVul-2024" / "SourceCodes" / "NonVulnerable"
SLITHER = "/home/motafeq/.venv/bin/slither"
ADERYN = "/home/motafeq/.cargo/bin/aderyn"


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


def compile_with_solc(sol_path: Path) -> str | None:
    """Try to compile with our solc versions; return version string on success, None on failure."""
    import hashlib
    solc_versions = [
        "/home/motafeq/.solc-select/artifacts/solc-0.8.19/solc-0.8.19",
        "/home/motafeq/.solc-select/artifacts/solc-0.7.6/solc-0.7.6",
        "/home/motafeq/.solc-select/artifacts/solc-0.6.12/solc-0.6.12",
        "/home/motafeq/.solc-select/artifacts/solc-0.5.17/solc-0.5.17",
    ]
    for solc in solc_versions:
        if not Path(solc).exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = subprocess.run(
                    [solc, "--bin", str(sol_path)],
                    capture_output=True, timeout=30, cwd=tmp
                )
                if result.returncode == 0:
                    return Path(solc).parent.name  # e.g. "solc-0.8.19"
            except (subprocess.TimeoutExpired, Exception):
                continue
    return None


def two_tool_check(sol_path: Path) -> bool:
    """Returns True if contract compiles AND both tools find NO vulnerability findings."""
    if compile_with_solc(sol_path) is None:
        return False

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Slither
        try:
            r = subprocess.run(
                [SLITHER, str(sol_path), "--json", "-"],
                capture_output=True, timeout=60
            )
            if r.returncode == 0:
                output = r.stdout.decode("utf-8", errors="replace")
                if '"impact": "High"' in output or '"impact": "Medium"' in output:
                    return False  # slither found high/medium issues
        except (subprocess.TimeoutExpired, Exception):
            return False

        # Aderyn
        try:
            r = subprocess.run(
                [ADERYN, str(sol_path), "--output", str(tmp_path / "report.md")],
                capture_output=True, timeout=60
            )
            report = (tmp_path / "report.md").read_text() if (tmp_path / "report.md").exists() else ""
            # If any high/medium severity finding, exclude
            if "High" in report or "Medium" in report:
                return False
        except (subprocess.TimeoutExpired, Exception):
            return False

    return True


def build_tier_e(output_dir: Path, n_target: int = 100) -> dict:
    """Build Tier E: known-safe contracts (negatives)."""
    print(f"\n=== Building Tier E (known-safe, n_target={n_target}) ===")
    print(f"Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir = output_dir / "contracts" / "by_class" / "NonVulnerable"
    contracts_dir.mkdir(parents=True, exist_ok=True)

    v3_shas = get_v3_sha256_set()
    print(f"\nv3 contracts: {len(v3_shas)}")
    print(f"NonVulnerable pool: {sum(1 for _ in NONVULN_DIR.glob('*.sol'))}")

    import hashlib
    candidates = []
    for sol in sorted(NONVULN_DIR.glob("*.sol")):
        raw = sol.read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        if h not in v3_shas:
            candidates.append((sol, h, len(raw)))
    print(f"  Honest OOD NonVulnerable candidates (not in v3): {len(candidates)}")

    # Sample (take first N for now; could add stratification by size)
    print(f"\nCompiling + 2-tool checking candidates (target: {n_target} clean)...")
    accepted = []
    for i, (sol, h, size) in enumerate(candidates):
        if len(accepted) >= n_target:
            break
        if i % 20 == 0:
            print(f"  ...processed {i}/{len(candidates)}, accepted {len(accepted)}")
        if two_tool_check(sol):
            accepted.append((sol, h, size))

    print(f"\n  Accepted: {len(accepted)}/{n_target} candidates as known-safe")

    # Copy files
    manifest = []
    for sol, h, size in accepted:
        new_name = f"nonvulnerable_{sol.name}"
        dest = contracts_dir / new_name
        shutil.copy2(sol, dest)
        manifest.append({
            "source": "bccc_nonvulnerable",
            "category": "NonVulnerable",
            "original_name": sol.name,
            "dest_path": str(dest.relative_to(output_dir)),
            "sha256": h,
            "size_bytes": size,
            "sentinal_class": "NonVulnerable",
            "tier": "E",
            "label_method": "2-tool_clean (slither+aderyn no high/medium findings)",
        })

    manifest_path = output_dir / "tier_e_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest written: {manifest_path}")

    return {
        "tier": "E",
        "total_contracts": len(accepted),
        "manifest_path": str(manifest_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-target", type=int, default=100)
    args = parser.parse_args()
    result = build_tier_e(Path(args.output_dir), args.n_target)
    print(f"\nTier E done: {result}")
