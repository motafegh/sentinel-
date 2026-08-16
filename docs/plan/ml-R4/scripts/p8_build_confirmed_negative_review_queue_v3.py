#!/usr/bin/env python3
"""Build the confirmed-negative pilot queue from corrected logical lineage V3.

This replaces the obsolete V2 queue. Candidates remain UNKNOWN and are only
reserved for class-specific human/evidence review; no target 0 is created.
Leakage groups are globally unique across the entire class-balanced queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

from sentinel_data.vnext.confirmed_negative_evaluation import (
    build_review_queue,
    minimum_zero_false_positive_sample_size,
)
from sentinel_data.vnext.policy import CLASS_NAMES
from sentinel_data.vnext.r4_v3_versions import (
    DATASET_VERSION_V3,
    ROLE_PARTITION_VERSION_V3,
)

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_OVERLAY = DATA_ROOT / "exports/sentinel-r4-vnext-v3"
DEFAULT_POLICY = REPO_ROOT / "docs/plan/ml-R4/specs/data_vnext_policy_v1.json"
DEFAULT_OUTPUT = DATA_ROOT / "r4-v3-logical-build/confirmed_negative_review_queue_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--max-fpr-planning-bound", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest_path = args.overlay / "manifest.json"
    ml_targets_path = args.overlay / "ml_targets.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != DATASET_VERSION_V3:
        raise ValueError("confirmed-negative V3 queue requires sentinel-r4-vnext-v3")
    if manifest.get("partition_version") != ROLE_PARTITION_VERSION_V3:
        raise ValueError("confirmed-negative V3 queue requires r4-vnext-roles-v3")
    if manifest.get("confirmed_negative_rows") != 0:
        raise ValueError("V3 queue expects zero existing confirmed-negative rows")
    if (
        manifest.get("status")
        != "LOGICAL_V3_REPRESENTATION_BOUND_LOCAL_REVIEW_REQUIRED"
    ):
        raise ValueError("V3 queue requires a physically bound logical publication")

    expected_ml_sha = ((manifest.get("artifacts") or {}).get("ml_targets") or {}).get(
        "sha256"
    )
    if not expected_ml_sha or _sha256(ml_targets_path) != expected_ml_sha:
        raise ValueError("V3 queue ml_targets.parquet hash mismatch")

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    enabled = [
        name
        for name in CLASS_NAMES
        if policy["class_supervision"][name]["status"] == "ENABLED"
    ]

    import pyarrow.parquet as pq

    rows = pq.read_table(ml_targets_path).to_pylist()
    queue = build_review_queue(
        rows,
        dataset_version=DATASET_VERSION_V3,
        partition_version=ROLE_PARTITION_VERSION_V3,
        publication_manifest_sha256=_sha256(manifest_path),
        enabled_class_names=enabled,
        per_class=args.per_class,
    )
    if queue.get("group_uniqueness_scope") != "GLOBAL_ACROSS_ENABLED_CLASSES":
        raise AssertionError("V3 negative queue did not enforce global group uniqueness")
    if len(queue.get("reserved_group_ids") or []) != int(queue.get("queued_cells", -1)):
        raise AssertionError("V3 negative queue contains a reused group reservation")

    queue["source_commit"] = _source_commit()
    queue["supersedes_queue"] = {
        "dataset_version": "sentinel-r4-vnext-v2",
        "partition_version": "r4-vnext-roles-v2",
        "reason": "V2 address-authority grouping was over-broad; V2 reservations are obsolete.",
    }
    queue["planning_only_zero_false_positive_bound"] = {
        "max_false_positive_rate": args.max_fpr_planning_bound,
        "confidence": args.confidence,
        "minimum_confirmed_negatives_per_class_if_zero_false_positives": (
            minimum_zero_false_positive_sample_size(
                max_false_positive_rate=args.max_fpr_planning_bound,
                confidence=args.confidence,
            )
        ),
        "limitations": [
            "This is a simple one-sided binomial planning bound, not a final quality gate.",
            "Leakage groups, review selection bias, threshold fitting, and multiple classes require separate treatment.",
            "The default 25-per-class queue is a pilot to estimate adjudication yield, not final evaluation size.",
        ],
    }

    text = json.dumps(queue, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
