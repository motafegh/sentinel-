"""
audit_labels.py — Audit // expect: headers across the manual contracts corpus.

Walks manual_hand_written_contracts/, parses every // expect: header,
and reports:
  - Total contracts parsed / found
  - Per-class support counts
  - Any contract with a missing / unparseable header
  - Ground truth split (vulnerable vs safe)

Exit 0 on success (all headers valid), 1 if any issue found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SIDECAR_FIELDS = {"labels", "ground_truth", "source"}
SIDECAR_SOURCE = "expect_header"


def parse_expect_header(sol_path: Path) -> tuple[list[str], str] | None:
    """
    Parse // expect: labels from the first lines of a .sol file.

    Format:
      // expect: ClassName            — vulnerable, one class
      // expect: ClassName1,ClassName2 — vulnerable, multi-class
      // expect:                      — safe (empty body)

    Returns (labels, ground_truth) or None if no header found.
    """
    try:
        lines = sol_path.read_text(encoding="utf-8", errors="replace").splitlines()[:30]
    except OSError:
        return None

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("//"):
            continue
        text = stripped[2:].strip()
        lower = text.lower()
        if not lower.startswith("expect"):
            continue
        if ":" not in text:
            continue
        body = text.split(":", 1)[1].strip()
        if not body:
            return [], "safe"
        labels = [t.strip() for t in body.replace(",", " ").split() if t.strip()]
        return labels, "vulnerable" if labels else "safe"
    return None


def audit_corpus(corpus_dir: Path) -> dict:
    results: dict = {
        "total_contracts": 0,
        "missing_header": [],
        "unparseable": [],
        "per_class": {},
        "ground_truth": {"vulnerable": 0, "safe": 0},
        "by_class_dir": {},
    }

    for class_dir in sorted(corpus_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for sol_file in sorted(class_dir.glob("*.sol")):
            results["total_contracts"] += 1
            header = parse_expect_header(sol_file)
            if header is None:
                results["missing_header"].append(str(sol_file.resolve()))
                continue
            labels, gt = header
            results["ground_truth"][gt] += 1
            if class_name not in results["by_class_dir"]:
                results["by_class_dir"][class_name] = {"count": 0, "labels": {}}
            results["by_class_dir"][class_name]["count"] += 1
            for label in labels:
                results["per_class"][label] = results["per_class"].get(label, 0) + 1
                if label not in results["by_class_dir"][class_name]["labels"]:
                    results["by_class_dir"][class_name]["labels"][label] = 0
                results["by_class_dir"][class_name]["labels"][label] += 1

    return results


def print_table(results: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  LABEL AUDIT REPORT")
    print(f"{'=' * 60}")
    print(f"\n  Total contracts:       {results['total_contracts']}")
    print(f"  Ground truth — vulnerable: {results['ground_truth']['vulnerable']}")
    print(f"  Ground truth — safe:       {results['ground_truth']['safe']}")
    print(f"  Missing headers:       {len(results['missing_header'])}")
    for p in results["missing_header"]:
        print(f"    !  {p}")
    print(f"  Unparseable:           {len(results['unparseable'])}")

    total_labeled = sum(results["per_class"].values())
    print(f"\n{'─' * 60}")
    print(f"  Per-class support (from headers, {total_labeled} total labels):")
    print(f"{'─' * 60}")
    for cls in sorted(results["per_class"]):
        print(f"    {cls:30s} {results['per_class'][cls]:3d} contracts")

    print(f"\n{'─' * 60}")
    print(f"  By class directory:")
    print(f"{'─' * 60}")
    for cls in sorted(results["by_class_dir"]):
        info = results["by_class_dir"][cls]
        labels_str = ", ".join(
            f"{k}({v})" for k, v in sorted(info["labels"].items())
        )
        print(f"    {cls:25s} {info['count']:2d} contracts  labels: [{labels_str}]")


def generate_sidecars(corpus_dir: Path) -> tuple[int, list[str]]:
    """
    Write a <stem>.json sidecar next to each .sol file with the parsed
    // expect: header. Idempotent — re-running produces identical output.

    Sidecar shape: {"labels": [...], "ground_truth": "vulnerable"|"safe",
                     "source": "expect_header"}

    Returns (count_written, errors).
    """
    errors: list[str] = []
    written = 0
    for class_dir in sorted(corpus_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for sol_file in sorted(class_dir.glob("*.sol")):
            header = parse_expect_header(sol_file)
            if header is None:
                errors.append(f"No header found: {sol_file}")
                continue
            labels, ground_truth = header
            sidecar = {
                "labels": labels,
                "ground_truth": ground_truth,
                "source": SIDECAR_SOURCE,
            }
            sidecar_path = sol_file.with_suffix(".json")
            sidecar_path.write_text(
                json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written += 1
    return written, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit // expect: headers in the contracts corpus"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manual_hand_written_contracts"),
        help="Path to corpus directory (default: manual_hand_written_contracts)",
    )
    parser.add_argument(
        "--generate-sidecars",
        action="store_true",
        help="Generate <stem>.json sidecars from // expect: headers",
    )
    args = parser.parse_args()

    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"ERROR: corpus dir not found: {corpus}", file=sys.stderr)
        sys.exit(1)

    if args.generate_sidecars:
        n_written, errors = generate_sidecars(corpus)
        for err in errors:
            print(f"  !  {err}", file=sys.stderr)
        if errors:
            print(f"\n  FAILED: {len(errors)} error(s) generating sidecars.", file=sys.stderr)
            sys.exit(1)
        print(f"\n  Generated {n_written} sidecar JSON files.")
        sys.exit(0)

    results = audit_corpus(corpus)
    print_table(results)

    n_issues = len(results["missing_header"]) + len(results["unparseable"])
    if n_issues > 0:
        print(f"\n  FAILED: {n_issues} issue(s) found.")
        sys.exit(1)
    print(f"\n  PASSED: all {results['total_contracts']} contracts have valid headers.")
    sys.exit(0)


if __name__ == "__main__":
    main()
