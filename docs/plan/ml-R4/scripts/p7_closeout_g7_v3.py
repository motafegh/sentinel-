#!/usr/bin/env python3
"""Final G7 closeout wrapper: G7-aware handbook metadata + truth checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_PATH = HERE / "p7_closeout_g7_v2.py"
spec = importlib.util.spec_from_file_location("sentinel_p7_closeout_g7_v2", V2_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {V2_PATH}")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
base = v2.base

_original_update_handbook_validator = base.update_handbook_validator


def replace_required(body: str, old: str, new: str, label: str) -> str:
    if new in body:
        return body
    if old not in body:
        raise RuntimeError(f"missing expected text for {label}: {old[:140]!r}")
    return body.replace(old, new, 1)


def update_handbook_validator() -> None:
    # First apply the original G7 validator additions (G7 artifacts/status checks).
    _original_update_handbook_validator()

    validator = base.REPO / "docs/handbook/tools/verify_handbook.py"
    body = base.text(validator)
    old_truth = '        "16_current_status.md": ["91f795885", "G6", "95c339edf", "SEMANTIC_VALIDATED_REPRESENTATIONS_PENDING", "UNSUPPORTED_EMPTY_FROZEN"],'
    new_truth = f'        "16_current_status.md": ["81d9c547d", "G7", "VALIDATED_G7_CANDIDATE", "{base.EXPECTED_BINDING_DIGEST}", "UNSUPPORTED_EMPTY_FROZEN"],'
    body = replace_required(body, old_truth, new_truth, "current-status documented truth")

    old_verified = '    checks.append(Check("verified runtime commit", commit_ok and verified_commit == "91f795885", f"metadata={verified_commit}, exists={commit_ok}"))'
    new_verified = '    checks.append(Check("verified source/runtime commit", commit_ok and verified_commit == "81d9c547d", f"metadata={verified_commit}, exists={commit_ok}"))'
    body = replace_required(body, old_verified, new_verified, "verified source/runtime baseline")
    base.write(validator, body)

    meta = base.REPO / "docs/handbook/_meta/handbook.toml"
    meta_body = base.text(meta)
    meta_body = replace_required(meta_body, 'handbook_version = "D1-v4-r4-v3"', 'handbook_version = "D1-v5-r4-g7"', "handbook version")
    meta_body = replace_required(
        meta_body,
        '# Runtime/source baseline. Documentation-only reconciliation commits may be later.\nverified_commit = "91f795885"',
        '# Canonical source/runtime baseline after DATA vNext G7 implementation merge.\nverified_commit = "81d9c547d"',
        "verified baseline",
    )
    meta_body = replace_required(meta_body, 'r4_canonical_gate = "G6"', 'r4_canonical_gate = "G7"', "canonical R4 gate")

    # Register the canonical G7 DATA vNext artifacts in the handbook inventory.
    if 'name = "R4 DATA vNext G7 manifest"' not in meta_body:
        anchor = '''[[artifact]]
name = "R4 untouched acceptance manifest"
path = "docs/plan/ml-R4/manifests/p6_untouched_acceptance_manifest.json"
classification = "tracked"
fresh_clone = true
owner = "R4 DATA"
'''
        addition = anchor + '''
[[artifact]]
name = "R4 DATA vNext G7 manifest"
path = "data_module/data/exports/sentinel-r4-vnext-v1/manifest.json"
classification = "tracked"
fresh_clone = true
owner = "R4 DATA"

[[artifact]]
name = "R4 DATA vNext G7 label states"
path = "data_module/data/exports/sentinel-r4-vnext-v1/label_states.parquet"
classification = "tracked"
fresh_clone = true
owner = "R4 DATA"

[[artifact]]
name = "R4 DATA vNext G7 representation binding"
path = "data_module/data/exports/sentinel-r4-vnext-v1/representation_binding_report.json"
classification = "tracked"
fresh_clone = true
owner = "R4 DATA"
'''
        if anchor not in meta_body:
            raise RuntimeError("handbook artifact anchor missing")
        meta_body = meta_body.replace(anchor, addition, 1)
    base.write(meta, meta_body)


base.update_handbook_validator = update_handbook_validator

if __name__ == "__main__":
    raise SystemExit(base.main())
