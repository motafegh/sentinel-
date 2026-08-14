#!/usr/bin/env python3
"""Materialize the repaired R4-v2 contract×class evidence ledger locally.

The historical Phase-3 ledger remains immutable.  This script builds a new
role-independent ledger from repaired source claims + final leakage grouping +
physical representation availability.  It writes only generated, Git-ignored
local artifacts and never invents negative targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_BUILD_ROOT = DATA_ROOT / "r4-v2-build"
DEFAULT_REPRESENTATIONS = DATA_ROOT / "representations-r4-v2"
DEFAULT_POLICY = REPO_ROOT / "docs/plan/ml-R4/specs/data_vnext_policy_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument(
        "--representations-root", type=Path, default=DEFAULT_REPRESENTATIONS
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from sentinel_data.vnext.r4_builder import build_semantic_cells
    except ImportError as exc:
        print(f"repaired ledger dependency unavailable: {exc}", file=sys.stderr)
        return 2

    claims_path = args.build_root / "source_claims.jsonl"
    grouping_path = args.build_root / "grouping.json"
    ledger_path = args.build_root / "evidence_ledger_v2.parquet"
    manifest_path = args.build_root / "evidence_ledger_v2_manifest.json"
    if ledger_path.exists() or manifest_path.exists():
        print(
            "repaired evidence-ledger outputs already exist; use a fresh build root",
            file=sys.stderr,
        )
        return 2
    for path in (claims_path, grouping_path, args.policy):
        if not path.is_file():
            print(f"missing repaired ledger input: {path}", file=sys.stderr)
            return 2

    claims = _load_jsonl(claims_path)
    grouping = json.loads(grouping_path.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    rows, artifact_info = build_semantic_cells(
        claims,
        grouping,
        policy,
        args.representations_root,
    )
    rows.sort(key=lambda row: (row["contract_id"], row["class_index"]))
    if any(row["target_value"] == 0 for row in rows):
        raise AssertionError("repaired ledger contains target zero")

    args.build_root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), ledger_path, compression="zstd")
    targets = Counter(
        "None" if row["target_value"] is None else str(row["target_value"])
        for row in rows
    )
    strengths = Counter(row["training_strength"] for row in rows)
    represented = sum(
        info["representation_available"] for info in artifact_info.values()
    )
    manifest = {
        "schema": "sentinel-r4-repaired-evidence-ledger-manifest-v2",
        "ledger_version": "evidence-ledger-r4-v2",
        "status": "LOCAL_REBUILD_CANDIDATE_NOT_CANONICAL",
        "contracts": len(artifact_info),
        "contract_class_rows": len(rows),
        "represented_contracts": represented,
        "target_counts": dict(sorted(targets.items())),
        "training_strength_counts": dict(sorted(strengths.items())),
        "confirmed_negative_rows": 0,
        "artifacts": {
            "ledger": {
                "path": "evidence_ledger_v2.parquet",
                "sha256": _sha256(ledger_path),
            },
            "source_claims": {
                "path": "source_claims.jsonl",
                "sha256": _sha256(claims_path),
            },
            "grouping": {
                "path": "grouping.json",
                "sha256": _sha256(grouping_path),
            },
            "policy": {
                "path": str(args.policy.relative_to(REPO_ROOT)),
                "sha256": _sha256(args.policy),
            },
        },
        "limitations": [
            "Role assignment is intentionally absent from this ledger; roles are frozen only after final grouping/representation availability.",
            "Physical representation binding and repaired-data acceptance remain separate gates."
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
