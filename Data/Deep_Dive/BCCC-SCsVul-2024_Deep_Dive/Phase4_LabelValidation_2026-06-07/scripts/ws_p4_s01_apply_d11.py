"""
Stage 0.1 — Apply D-I-11: drop Class12:NonVulnerable when it co-occurs with
any of {CallToUnknown, Reentrancy, GasException, MishandledException,
DenialOfService, Timestamp}.

Scope: review_pending only by default (D-P4-1 default).
Override: --broad flag applies to all 67,311 contracts.

Decision doc: ../Phase3_DeepAnalysis_2026-06-06/decisions/D-I-11_drop_nv_with_vuln.md
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path("/home/motafeq/projects/sentinel")
V10 = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/outputs/contracts_clean.csv"
OUT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01_d11_applied.csv"
REPORT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01_d11_report.md"

NV_COL = "Class12:NonVulnerable"
TRIGGER_COLS = [
    "Class08:CallToUnknown",
    "Class11:Reentrancy",
    "Class02:GasException",
    "Class03:MishandledException",
    "Class09:DenialOfService",
    "Class04:Timestamp",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--broad", action="store_true",
                    help="Apply D-I-11 to all 67,311 contracts (D-P4-1 override).")
    ap.add_argument("--in", dest="inp", type=Path, default=V10)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"ERROR: input not found: {args.inp}", file=sys.stderr)
        return 1

    with args.inp.open() as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = r.fieldnames

    print(f"Loaded {len(rows)} contracts from {args.inp.name}")
    if "review_pending" not in fieldnames:
        print("ERROR: review_pending column missing", file=sys.stderr)
        return 1

    n_total = len(rows)
    n_in_scope = 0
    n_dropped = 0
    n_dropped_review_pending = 0
    n_dropped_non_review_pending = 0
    samples = []

    for row in rows:
        nv = int(row[NV_COL])
        if nv != 1:
            continue
        triggered = [c for c in TRIGGER_COLS if int(row[c]) == 1]
        if not triggered:
            continue
        if not args.broad and int(row["review_pending"]) != 1:
            continue
        n_in_scope += 1
        n_dropped += 1
        is_rp = int(row["review_pending"]) == 1
        if is_rp:
            n_dropped_review_pending += 1
        else:
            n_dropped_non_review_pending += 1
        if len(samples) < 10:
            samples.append({
                "id": row["id"],
                "folder": row["bccc_folder"],
                "primary": row["primary_class"],
                "triggered": ",".join(t.replace("Class", "") for t in triggered),
                "review_pending": row["review_pending"],
                "n_pos_before": row["n_pos"],
            })
        row[NV_COL] = "0"
        if is_rp:
            row["review_pending"] = "0"
        new_pos = sum(int(row[c]) for c in fieldnames
                      if c.startswith("Class") and c != NV_COL)
        row["n_pos"] = str(new_pos)
        if new_pos == 0 and int(row[NV_COL]) == 0:
            row["is_pure_nv"] = "0"
        if int(row[NV_COL]) == 1 and new_pos == 0:
            row["is_pure_nv"] = "1"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_review_pending_remaining = sum(1 for r in rows if int(r["review_pending"]) == 1)
    n_pure_nv = sum(1 for r in rows if int(r["is_pure_nv"]) == 1)
    n_nv_pos = sum(1 for r in rows if int(r[NV_COL]) == 1)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    scope = "BROAD (all 67,311 contracts)" if args.broad else "NARROW (review_pending only, default)"
    with REPORT.open("w") as f:
        f.write(f"# D-I-11 Application Report\n\n")
        f.write(f"**Date:** 2026-06-07\n")
        f.write(f"**Scope:** {scope}\n")
        f.write(f"**Input:** `{args.inp.relative_to(REPO)}` ({n_total} contracts)\n")
        f.write(f"**Output:** `{args.out.relative_to(REPO)}`\n\n")
        f.write("## Rule Applied\n\n")
        f.write("For every contract where `Class12:NonVulnerable=1` AND at least one of "
                "{CallToUnknown, Reentrancy, GasException, MishandledException, "
                "DenialOfService, Timestamp}=1 → set `Class12:NonVulnerable=0` and "
                "(`review_pending=0` if it was 1).\n\n")
        f.write("## Counts\n\n")
        f.write(f"| Metric | Value |\n|---|---:|\n")
        f.write(f"| Total contracts in input | {n_total} |\n")
        f.write(f"| Contracts with NV=1 + ≥1 trigger (in scope) | {n_in_scope} |\n")
        f.write(f"| NV labels dropped (renumbered) | {n_dropped} |\n")
        f.write(f"|   – in review_pending | {n_dropped_review_pending} |\n")
        f.write(f"|   – in non-review_pending | {n_dropped_non_review_pending} |\n")
        f.write(f"| review_pending remaining after | {n_review_pending_remaining} |\n")
        f.write(f"| NV=1 contracts remaining | {n_nv_pos} |\n")
        f.write(f"| is_pure_nv=1 contracts | {n_pure_nv} |\n\n")
        f.write("## First 10 Corrections (for spot-check)\n\n")
        f.write("| id (prefix) | folder | primary | triggered classes | review_pending before | n_pos before |\n")
        f.write("|---|---|---|---|---:|---:|\n")
        for s in samples:
            f.write(f"| `{s['id'][:16]}…` | {s['folder']} | {s['primary']} | "
                    f"{s['triggered']} | {s['review_pending']} | {s['n_pos_before']} |\n")
        f.write("\n## Sanity Check\n\n")
        f.write("- All dropped contracts had NV=1 co-occurring with at least one "
                "vulnerability class (rule satisfied).\n")
        f.write("- n_pos recomputed for each row.\n")
        if n_review_pending_remaining < 100:
            f.write(f"- review_pending reduced to **{n_review_pending_remaining}** — "
                    f"close to the predicted ~61. Stage 0.4 will handle these.\n")
        f.write(f"\n## Version\n\nv1.1 (D-I-11 applied, {scope.lower()})\n")
    print(f"Wrote {args.out} and {REPORT}")
    print(f"Dropped {n_dropped} NV labels ({n_dropped_review_pending} in review_pending).")
    print(f"review_pending remaining: {n_review_pending_remaining}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
