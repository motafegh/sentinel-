"""
Stage 0.1b — Apply D-I-12: drop Class12:NonVulnerable when it co-occurs with
Class10:IntegerUO (and the contract is still review_pending=1).

This is a strict generalization of D-I-11 for the one class that wasn't
included (IntegerUO). Resolves the remaining 41 review_pending contracts.

Decision doc: ../decisions/D-I-12_drop_nv_with_integeruo.md
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path("/home/motafeq/projects/sentinel")
V11 = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01_d11_applied.csv"
OUT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01b_d12_applied.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=V11)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"ERROR: input not found: {args.inp}", file=sys.stderr)
        return 1

    with args.inp.open() as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = r.fieldnames
    print(f"Loaded {len(rows)} contracts from {args.inp.name}")

    n_dropped = 0
    samples = []
    for row in rows:
        if int(row["review_pending"]) != 1:
            continue
        nv = int(row["Class12:NonVulnerable"])
        integeruo = int(row["Class10:IntegerUO"])
        if nv != 1 or integeruo != 1:
            continue
        n_dropped += 1
        if len(samples) < 10:
            samples.append({
                "id": row["id"],
                "primary": row["primary_class"],
                "n_pos_before": row["n_pos"],
            })
        row["Class12:NonVulnerable"] = "0"
        row["review_pending"] = "0"
        new_pos = sum(int(row[c]) for c in fieldnames
                      if c.startswith("Class") and c != "Class12:NonVulnerable")
        row["n_pos"] = str(new_pos)
        if new_pos == 0:
            row["is_pure_nv"] = "0"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out} (NV dropped: {n_dropped})")

    n_review_pending_remaining = sum(1 for r in rows if int(r["review_pending"]) == 1)
    n_pure_nv = sum(1 for r in rows if int(r["is_pure_nv"]) == 1)
    n_nv = sum(1 for r in rows if int(r["Class12:NonVulnerable"]) == 1)
    print(f"After D-I-12:")
    print(f"  review_pending remaining: {n_review_pending_remaining}")
    print(f"  NV=1 contracts: {n_nv}")
    print(f"  is_pure_nv=1 contracts: {n_pure_nv}")
    print(f"\nFirst 10 corrections (all should be IntegerUO+NV, n_pos=2):")
    for s in samples:
        print(f"  {s['id'][:16]}.. primary={s['primary']} n_pos_before={s['n_pos_before']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
