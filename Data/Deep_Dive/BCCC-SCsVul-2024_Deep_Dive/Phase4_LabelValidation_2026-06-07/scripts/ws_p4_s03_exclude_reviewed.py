"""
Stage 0.3 — Exclude 32 reviewed contracts from the sampling frame.

The 32 contracts are the 30 worst-disagreement (from
`ws_i_worst_30_for_review.csv`) plus the 5 n_pos=8 maxing contracts
(found in `contracts_clean.csv` by n_pos=8). Union of these = 33 unique IDs
(the worst_30 includes 2 of the 5 maxing contracts, hence 30+5-2=33).

Add `reviewed_in_phase3=1` flag to those contracts and exclude them from
the sampling frame. They are still present in the dataset but marked.

Output: ws_p4_s03_exclude_reviewed.csv  (sampling frame, v1.1, dups already flagged)
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path("/home/motafeq/projects/sentinel")
FRAME_IN = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s02_sampling_frame.csv"
WORST_30 = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_worst_30_for_review.csv"
V11 = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01_d11_applied.csv"
OUT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s03_exclude_reviewed.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=FRAME_IN)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--worst-30", type=Path, default=WORST_30)
    ap.add_argument("--v11", type=Path, default=V11)
    args = ap.parse_args()

    if not args.inp.exists() or not args.worst_30.exists() or not args.v11.exists():
        print(f"ERROR: missing input file", file=sys.stderr)
        return 1

    with args.worst_30.open() as f:
        worst_ids = {r["id"] for r in csv.DictReader(f)}
    with args.v11.open() as f:
        maxing_ids = {r["id"] for r in csv.DictReader(f) if int(r["n_pos"]) == 8}
    reviewed = worst_ids | maxing_ids
    print(f"32 reviewed contracts:")
    print(f"  worst_30: {len(worst_ids)}")
    print(f"  maxing (n_pos=8): {len(maxing_ids)}")
    print(f"  union: {len(reviewed)} unique contracts")

    with args.inp.open() as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = r.fieldnames

    if "reviewed_in_phase3" not in fieldnames:
        fieldnames = fieldnames + ["reviewed_in_phase3"]
    n_flagged = 0
    for row in rows:
        if row["id"] in reviewed:
            row["reviewed_in_phase3"] = "1"
            n_flagged += 1
        else:
            row["reviewed_in_phase3"] = "0"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out} ({n_flagged} flagged, expected {len(reviewed)})")

    n_eligible = sum(1 for r in rows
                     if r["reviewed_in_phase3"] == "0"
                     and r["is_oraclize_dup"] == "0")
    n_oraclize_dup = sum(1 for r in rows if r["is_oraclize_dup"] == "1")
    n_reviewed = sum(1 for r in rows if r["reviewed_in_phase3"] == "1")
    n_both = sum(1 for r in rows
                 if r["reviewed_in_phase3"] == "1" and r["is_oraclize_dup"] == "1")
    print(f"\nSampling frame composition:")
    print(f"  Total: {len(rows)}")
    print(f"  Oraclize duplicates: {n_oraclize_dup}")
    print(f"  Reviewed in Phase 3: {n_reviewed}")
    print(f"  Both: {n_both}")
    print(f"  Eligible for Stage 1 sampling: {n_eligible}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
