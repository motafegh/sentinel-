#!/usr/bin/env python3
"""Build a deterministic confirmed-negative pilot review queue from repaired-v2.

The queue contains *candidates for class-specific review*, not negative labels.
Only currently unlabeled leakage groups are eligible.  The script never writes
target 0 into the accepted publication and never authorizes training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_OVERLAY = DATA_ROOT / "exports/sentinel-r4-vnext-v2"
DEFAULT_POLICY = REPO_ROOT / "docs/plan/ml-R4/specs/data_vnext_policy_v1.json"
DEFAULT_OUTPUT = DATA_ROOT / "r4-v2-build/confirmed_negative_review_queue_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument(
        "--max-fpr-planning-bound",
        type=float,
        default=0.05,
        help="Planning-only zero-FP upper-bound target; does not define G8.",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest_path = args.overlay / "manifest.json"
    ml_targets_path = args.overlay / "ml_targets.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not ml_targets_path.is_file():
        raise FileNotFoundError(ml_targets_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != "sentinel-r4-vnext-v2":
        raise ValueError("confirmed-negative queue requires repaired-v2 publication")
    if manifest.get("confirmed_negative_rows") != 0:
        raise ValueError("queue builder expects zero existing confirmed-negative rows")

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
        dataset_version=str(manifest["dataset_version"]),
        partition_version=str(manifest["partition_version"]),
        publication_manifest_sha256=_sha256(manifest_path),
        enabled_class_names=enabled,
        per_class=args.per_class,
    )
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
