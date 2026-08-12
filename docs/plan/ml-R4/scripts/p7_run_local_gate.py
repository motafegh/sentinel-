#!/usr/bin/env python3
"""Run the final local-only R4 Phase-7 representation-binding gate.

This wrapper deliberately does not use DVC.  It verifies the committed semantic
v2 overlay and frozen inputs are clean, checks lineage against the checked-out
code, and then invokes the transactional DATA vNext local gate against the
protected/local representation tree.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_MODULE = REPO_ROOT / "data_module"
EXPORT_DIR = DATA_MODULE / "data/exports/sentinel-r4-vnext-v1"
DEFAULT_REPRESENTATIONS = DATA_MODULE / "data/representations"

CRITICAL_TRACKED_PATHS = [
    "docs/plan/ml-R4/ledger/evidence_ledger_v1.parquet",
    "docs/plan/ml-R4/specs/data_vnext_policy_v1.json",
    "docs/plan/ml-R4/schemas/data_vnext_label_state_v1.schema.json",
    "docs/plan/ml-R4/manifests/p6_contract_role_manifest.jsonl",
    "docs/plan/ml-R4/manifests/p6_partition_manifest.json",
    "docs/plan/ml-R4/manifests/p6_unsupported_roles.json",
    "docs/plan/ml-R4/manifests/p6_untouched_acceptance_manifest.json",
    "data_module/sentinel_data/export/format_schema/v2.yaml",
    "data_module/sentinel_data/vnext/__init__.py",
    "data_module/sentinel_data/vnext/policy.py",
    "data_module/sentinel_data/vnext/builder.py",
    "data_module/sentinel_data/vnext/validator.py",
    "data_module/sentinel_data/vnext/publication.py",
    "data_module/sentinel_data/vnext/representations.py",
    "data_module/sentinel_data/vnext/loader.py",
    "data_module/sentinel_data/vnext/cli.py",
    "data_module/data/exports/sentinel-r4-vnext-v1/manifest.json",
    "data_module/data/exports/sentinel-r4-vnext-v1/source_registry.json",
    "data_module/data/exports/sentinel-r4-vnext-v1/crosswalk_registry.json",
    "data_module/data/exports/sentinel-r4-vnext-v1/evidence_snapshot.json",
    "data_module/data/exports/sentinel-r4-vnext-v1/representation_requirements.json",
    "data_module/data/exports/sentinel-r4-vnext-v1/label_states.parquet",
    "data_module/data/exports/sentinel-r4-vnext-v1/ml_targets.parquet",
    "data_module/data/exports/sentinel-r4-vnext-v1/validation_report.json",
]


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=check
    )


def require_clean_critical_paths() -> None:
    missing = [p for p in CRITICAL_TRACKED_PATHS if not (REPO_ROOT / p).is_file()]
    if missing:
        raise RuntimeError("critical tracked Phase-7 files are missing: " + ", ".join(missing))

    status = run_git("status", "--porcelain=v1", "--", *CRITICAL_TRACKED_PATHS).stdout.splitlines()
    dirty = [line for line in status if line.strip()]
    if dirty:
        raise RuntimeError(
            "critical frozen/semantic files have local Git changes; restore or commit them before G7:\n"
            + "\n".join(dirty)
        )


def require_generation_commit_is_ancestor() -> str:
    manifest = json.loads((EXPORT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "SEMANTIC_VALIDATED_REPRESENTATIONS_PENDING":
        raise RuntimeError(
            "local gate requires the committed semantic-only manifest state; got "
            f"{manifest.get('status')!r}"
        )
    generation_commit = str(manifest.get("generation_commit") or "")
    if len(generation_commit) != 40:
        raise RuntimeError("manifest generation_commit is not a full SHA")
    exists = run_git("cat-file", "-e", f"{generation_commit}^{{commit}}", check=False)
    if exists.returncode != 0:
        raise RuntimeError(f"manifest generation commit is not available locally: {generation_commit}")
    ancestor = run_git("merge-base", "--is-ancestor", generation_commit, "HEAD", check=False)
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"semantic overlay generation commit {generation_commit} is not an ancestor of current HEAD"
        )
    return generation_commit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representations-root",
        type=Path,
        default=DEFAULT_REPRESENTATIONS,
        help="Existing local protected representation tree (no DVC fetch is performed).",
    )
    args = parser.parse_args()

    representations_root = args.representations_root.expanduser().resolve()
    if not representations_root.is_dir():
        print(json.dumps({
            "passed": False,
            "stage": "preflight",
            "error": f"representation root does not exist: {representations_root}",
        }, indent=2))
        return 1

    try:
        require_clean_critical_paths()
        generation_commit = require_generation_commit_is_ancestor()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"passed": False, "stage": "preflight", "error": str(exc)}, indent=2))
        return 1

    sys.path.insert(0, str(DATA_MODULE))
    from sentinel_data.vnext.cli import main as vnext_main

    print(json.dumps({
        "preflight": "PASS",
        "repo_head": run_git("rev-parse", "HEAD").stdout.strip(),
        "generation_commit": generation_commit,
        "export_dir": str(EXPORT_DIR),
        "representations_root": str(representations_root),
    }, indent=2))

    rc = int(vnext_main([
        "local-gate",
        "--output", str(EXPORT_DIR),
        "--representations-root", str(representations_root),
    ]))
    if rc != 0:
        return rc

    manifest = json.loads((EXPORT_DIR / "manifest.json").read_text(encoding="utf-8"))
    rep_report = EXPORT_DIR / "representation_binding_report.json"
    g7_report = EXPORT_DIR / "g7_validation_report.json"
    if manifest.get("status") != "VALIDATED_G7_CANDIDATE":
        print(json.dumps({"passed": False, "stage": "postflight", "error": "manifest not promoted"}, indent=2))
        return 1
    if not rep_report.is_file() or not g7_report.is_file():
        print(json.dumps({"passed": False, "stage": "postflight", "error": "required G7 reports missing"}, indent=2))
        return 1

    print("\n=== PHASE 7 LOCAL G7 CANDIDATE PASS ===")
    print(f"manifest:               {EXPORT_DIR / 'manifest.json'}")
    print(f"representation report:  {rep_report}")
    print(f"final G7 report:         {g7_report}")
    print("\nPublish these three files to the Phase-7 branch:")
    print("git add -f \\")
    print("  data_module/data/exports/sentinel-r4-vnext-v1/manifest.json \\")
    print("  data_module/data/exports/sentinel-r4-vnext-v1/representation_binding_report.json \\")
    print("  data_module/data/exports/sentinel-r4-vnext-v1/g7_validation_report.json")
    print('git commit -m "data(ml-r4): bind local representations for G7"')
    print("git push origin HEAD:r4/phase7-data-vnext-implementation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
