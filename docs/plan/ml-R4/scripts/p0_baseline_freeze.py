#!/usr/bin/env python3
"""R4 Phase 0 — deterministic baseline freeze and hash validator.

Usage:
    python3 docs/plan/ml-R4/scripts/p0_baseline_freeze.py            # verify
    python3 docs/plan/ml-R4/scripts/p0_baseline_freeze.py --write     # regenerate

Reads the protected_artifacts.json manifest and verifies SHA-256 hashes
against the on-disk files.  Exits non-zero on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_DIR = REPO_ROOT / "docs" / "plan" / "ml-R4" / "manifests"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    write_mode = "--write" in sys.argv
    manifest_path = MANIFEST_DIR / "protected_artifacts.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    failures: list[str] = []

    for entry in manifest.get("protected_artifacts", []):
        rel = entry["path"]
        expected = entry.get("sha256")
        if expected is None:
            continue
        full = REPO_ROOT / rel
        if not full.exists():
            failures.append(f"MISSING: {rel}")
            continue
        actual = sha256_file(full)
        if actual != expected:
            if write_mode:
                entry["sha256"] = actual
                print(f"UPDATED: {rel} -> {actual}")
            else:
                failures.append(f"MISMATCH: {rel}\n  expected: {expected}\n  actual:   {actual}")
        else:
            print(f"OK: {rel}")

    if write_mode:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nWrote {manifest_path}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print("\nAll protected artifact hashes verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
