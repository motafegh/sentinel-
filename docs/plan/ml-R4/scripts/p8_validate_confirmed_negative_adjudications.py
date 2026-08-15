#!/usr/bin/env python3
"""Validate explicit confirmed-negative adjudications for evaluation-only use.

Input adjudications are JSON Lines.  A CONFIRMED_NEGATIVE row must satisfy the
fail-closed dual-review contract implemented by
``sentinel_data.vnext.confirmed_negative_evaluation``.  Accepted rows remain
evaluation-only and do not mutate repaired-v2 or authorize target 0 for training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data_module"))

from sentinel_data.vnext.confirmed_negative_evaluation import validate_adjudications

DATA_ROOT = REPO_ROOT / "data_module/data"
DEFAULT_QUEUE = DATA_ROOT / "r4-v2-build/confirmed_negative_review_queue_v1.json"
DEFAULT_ADJUDICATIONS = (
    DATA_ROOT / "r4-v2-build/confirmed_negative_adjudications_v1.jsonl"
)
DEFAULT_OUTPUT = (
    DATA_ROOT / "r4-v2-build/confirmed_negative_evaluation_v1.json"
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument(
        "--adjudications",
        type=Path,
        default=DEFAULT_ADJUDICATIONS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    adjudications = _load_jsonl(args.adjudications)
    report = validate_adjudications(queue, adjudications)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
