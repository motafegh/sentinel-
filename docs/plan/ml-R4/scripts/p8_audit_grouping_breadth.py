#!/usr/bin/env python3
"""Profile repaired-v2 leakage-group breadth and address-driven connectivity.

This is a read-only audit.  Flags request review; they do not mutate accepted
grouping, roles, or DATA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

from sentinel_data.preprocessing.r4_grouping_audit import audit_grouping_payload

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_GROUPING = DATA_ROOT / "r4-v2-build/grouping.json"
DEFAULT_OUTPUT = DATA_ROOT / "r4-v2-build/grouping_breadth_audit_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouping", type=Path, default=DEFAULT_GROUPING)
    parser.add_argument("--address-threshold", type=int, default=20)
    parser.add_argument("--large-group-threshold", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.grouping.read_text(encoding="utf-8"))
    report = audit_grouping_payload(
        payload,
        high_frequency_address_threshold=args.address_threshold,
        large_group_threshold=args.large_group_threshold,
        top_n=args.top_n,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
