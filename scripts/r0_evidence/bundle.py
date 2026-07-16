"""Deterministic generator for the external R0 behavioral probe bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.r0_evidence.model import probe_bundle_digest, sha256_file


def build_probe_bundle(bundle: Path, *, entrypoint: str = "baseline_probes.py") -> str:
    """Regenerate the manifest and aggregate digest for a bundle directory."""
    bundle = bundle.resolve()
    entrypoint_path = (bundle / entrypoint).resolve()
    if not entrypoint_path.is_relative_to(bundle) or not entrypoint_path.is_file():
        raise ValueError(f"bundle entrypoint is missing or outside bundle: {entrypoint}")
    manifest = {
        "bundle_version": "1",
        "entrypoint": entrypoint,
        "file_count": 1,
        "files": {entrypoint: sha256_file(entrypoint_path)},
        "kind": "r0_probe_bundle_manifest",
        "python": sys.executable,
        "python_version": sys.version,
    }
    digest = probe_bundle_digest(manifest)
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle / "aggregate_digest.txt").write_text(digest + "\n", encoding="utf-8")
    return digest


__all__ = ["build_probe_bundle"]
