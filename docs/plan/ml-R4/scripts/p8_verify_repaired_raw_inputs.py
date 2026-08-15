#!/usr/bin/env python3
"""Verify the protected Phase-8 raw manifests before repaired preprocessing.

The script is read-only.  It verifies every manifest-recorded byte length and
SHA-256 for the three active sources and checks that no manifest path escapes
its source root.  It intentionally records no machine-specific path in output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sentinel_data.preprocessing.r4_raw_verifier import verify_manifest_source

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module" / "data"
ACTIVE_SOURCES = ("dive", "smartbugs_curated", "solidifi")


def main() -> int:
    results = [
        verify_manifest_source(
            source,
            DATA_ROOT / "raw" / source,
            DATA_ROOT / "raw" / source / "ingestion_manifest.json",
        )
        for source in ACTIVE_SOURCES
    ]
    passed = all(row.get("passed") for row in results)
    print(
        json.dumps(
            {
                "schema": "sentinel-r4-repaired-raw-provenance-gate-v1",
                "passed": passed,
                "sources": results,
                "expected_historical_manifest_records": {
                    "dive": 22330,
                    "smartbugs_curated": 143,
                    "solidifi": 350,
                    "total": 22823,
                },
                "claim_scope": (
                    "byte-level agreement with existing ingestion manifests only; "
                    "this does not prove portable reacquisition"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
