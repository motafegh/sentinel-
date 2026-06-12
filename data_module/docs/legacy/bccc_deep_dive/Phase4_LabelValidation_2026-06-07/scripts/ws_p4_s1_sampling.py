"""
Stage 1 — Stratified sampling from the v1.1+12 sampling frame.

Strategy (per user D-P4-2 + D-P4-x choice):
- For primary_class with n >= SMALL_CLASS_THRESHOLD in eligible pool: take 15%
- For primary_class with n <  SMALL_CLASS_THRESHOLD: take 100% (preserve all)
- Exclude reviewed and Oraclize-dup contracts (already filtered in input)

Output: ws_p4_s1_sample.csv
"""
import argparse
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/home/motafeq/projects/sentinel")
FRAME = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s03_exclude_reviewed.csv"
OUT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_sample.csv"

LARGE_CLASS_FRACTION = 0.15
SMALL_CLASS_THRESHOLD = 500
RANDOM_SEED = 42


def deterministic_shuffle(items: list, seed: int) -> list:
    """SHA-256-based deterministic shuffle for reproducibility."""
    pairs = []
    for item in items:
        h = hashlib.sha256(f"{seed}:{item}".encode()).hexdigest()
        pairs.append((h, item))
    pairs.sort()
    return [item for _, item in pairs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=FRAME)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fraction", type=float, default=LARGE_CLASS_FRACTION,
                    help="Fraction for large classes (default 0.15)")
    ap.add_argument("--threshold", type=int, default=SMALL_CLASS_THRESHOLD,
                    help="Small class threshold (default 500)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"ERROR: input not found: {args.inp}", file=sys.stderr)
        return 1

    with args.inp.open() as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = r.fieldnames
    print(f"Loaded {len(rows)} contracts from {args.inp.name}")

    eligible = [r for r in rows
                if r["is_oraclize_dup"] == "0" and r["reviewed_in_phase3"] == "0"]
    print(f"Eligible (not oraclize_dup, not reviewed): {len(eligible)}")

    by_class = defaultdict(list)
    for r in eligible:
        by_class[r["primary_class"]].append(r["id"])

    sample_ids = []
    sample_log = []
    for primary_class, ids in sorted(by_class.items()):
        ids_sorted = deterministic_shuffle(ids, args.seed)
        n = len(ids_sorted)
        if n < args.threshold:
            take = n
            rule = f"100% (n<{args.threshold})"
        else:
            take = max(1, int(round(n * args.fraction)))
            rule = f"{args.fraction*100:.0f}% (n>={args.threshold})"
        sample_ids.extend(ids_sorted[:take])
        sample_log.append((primary_class, n, take, rule))

    print(f"\nSampling plan:")
    print(f"  {'primary_class':<25s} {'eligible':>8s} {'take':>6s} {'rule'}")
    total = 0
    for cls, n, take, rule in sample_log:
        print(f"  {cls:<25s} {n:>8d} {take:>6d} {rule}")
        total += take
    print(f"  {'TOTAL':<25s} {len(eligible):>8d} {total:>6d}")

    sample_set = set(sample_ids)
    sample_rows = [r for r in rows if r["id"] in sample_set]
    if "in_stage1_sample" not in fieldnames:
        fieldnames = fieldnames + ["in_stage1_sample"]
    for r in rows:
        r["in_stage1_sample"] = "1" if r["id"] in sample_set else "0"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out} ({total} sampled, {len(rows)-total} not sampled)")

    sample_class_counter = Counter(r["primary_class"] for r in sample_rows)
    print(f"\nSample primary_class distribution (in sample):")
    for c, n in sample_class_counter.most_common():
        print(f"  {c:25s}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
