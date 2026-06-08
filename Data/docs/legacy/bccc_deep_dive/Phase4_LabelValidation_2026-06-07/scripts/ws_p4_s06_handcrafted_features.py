"""
Stage 0.6 — Compute 3 hand-crafted features on all 67,311 contracts in v1.1.

The 3 features:
- h01_nv_but_has_reentrancy_call: NV=1 AND (low-level call with value OR transfer OR send)
- h02_nv_but_has_external_call: NV=1 AND (any external contract call exists)
- h03_unsafe_arith_no_safemath: (arithmetic op +,-,* exists) AND (not inside SafeMath.add/sub/mul)

These are designed to be near-zero for the majority of contracts — they
flag specific patterns that contradict the BCCC NonVulnerable label or
indicate unsafe arithmetic.

Output: ws_p4_s06_handcrafted_features.csv (67311 x 3, plus id).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path("/home/motafeq/projects/sentinel")
V11 = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01_d11_applied.csv"
SRC = REPO / "BCCC-SCsVul-2024/SourceCodes"
OUT = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s06_handcrafted_features.csv"

NV_COL = "Class12:NonVulnerable"

LOW_LEVEL_VALUE = re.compile(r"\.call\s*\{[^}]*value\s*:")
TRANSFER = re.compile(r"\.transfer\s*\(")
SEND = re.compile(r"\.send\s*\(")
ANY_CALL = re.compile(r"\.call\s*\(|\.delegatecall\s*\(|\.staticcall\s*\(|\.transfer\s*\(|\.send\s*\(|address\([^)]*\)\.transfer")
ARITH_OP = re.compile(r"[+\-\*/](?!=)")
SAFEMATH_PRESENT = re.compile(r"\bSafeMath\b")
SAFEMATH_USED = re.compile(r"SafeMath\.(add|sub|mul|div|mod)\s*\(")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=V11)
    ap.add_argument("--src-root", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"ERROR: input not found: {args.inp}", file=sys.stderr)
        return 1

    with args.inp.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} contracts")

    out_fieldnames = ["id", "h01_nv_but_has_reentrancy_call", "h02_nv_but_has_external_call",
                      "h03_unsafe_arith_no_safemath"]
    out_rows = []
    h01 = h02 = h03 = 0
    h01_in_nv = h02_in_nv = 0
    n_missing = 0
    for i, row in enumerate(rows):
        rel = row["bccc_file_path"]
        rel_clean = rel
        if rel_clean.startswith("BCCC-SCsVul-2024/Source Codes/"):
            rel_clean = rel_clean.replace("BCCC-SCsVul-2024/Source Codes/", "")
        elif rel_clean.startswith("BCCC-SCsVul-2024/SourceCodes/"):
            rel_clean = rel_clean.replace("BCCC-SCsVul-2024/SourceCodes/", "")
        p = args.src_root / rel_clean
        if not p.exists():
            n_missing += 1
            out_rows.append({"id": row["id"], "h01_nv_but_has_reentrancy_call": 0,
                             "h02_nv_but_has_external_call": 0, "h03_unsafe_arith_no_safemath": 0})
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            n_missing += 1
            out_rows.append({"id": row["id"], "h01_nv_but_has_reentrancy_call": 0,
                             "h02_nv_but_has_external_call": 0, "h03_unsafe_arith_no_safemath": 0})
            continue
        nv = int(row[NV_COL])
        has_reentrant_call = bool(LOW_LEVEL_VALUE.search(src) or TRANSFER.search(src) or SEND.search(src))
        has_external_call = bool(ANY_CALL.search(src))
        has_arith = bool(ARITH_OP.search(src))
        uses_safemath_for_arith = bool(SAFEMATH_PRESENT.search(src) and SAFEMATH_USED.search(src))
        unsafe_arith = 1 if (has_arith and not uses_safemath_for_arith) else 0

        h01_v = 1 if (nv == 1 and has_reentrant_call) else 0
        h02_v = 1 if (nv == 1 and has_external_call) else 0
        h03_v = unsafe_arith

        h01 += h01_v
        h02 += h02_v
        h03 += h03_v
        if nv == 1:
            h01_in_nv += h01_v
            h02_in_nv += h02_v

        out_rows.append({"id": row["id"], "h01_nv_but_has_reentrancy_call": h01_v,
                         "h02_nv_but_has_external_call": h02_v, "h03_unsafe_arith_no_safemath": h03_v})
        if (i + 1) % 5000 == 0:
            print(f"  handcrafted features computed: {i+1}/{len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {args.out} ({len(out_rows)} rows, {n_missing} missing source)")

    n_nv = sum(1 for r in rows if int(r[NV_COL]) == 1)
    print(f"\nFeature prevalence:")
    print(f"  h01_nv_but_has_reentrancy_call: {h01} total ({h01_in_nv} in NV=1 contracts)")
    print(f"  h02_nv_but_has_external_call:   {h02} total ({h02_in_nv} in NV=1 contracts)")
    print(f"  h03_unsafe_arith_no_safemath:   {h03} total")
    print(f"  NV=1 contracts in dataset:      {n_nv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
