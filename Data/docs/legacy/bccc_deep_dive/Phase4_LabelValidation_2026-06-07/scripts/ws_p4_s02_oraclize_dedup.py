"""
Stage 0.2 — Oraclize cluster dedup.

Compute `source_stripped_sha256` for every contract in v1.1, group by hash,
identify clusters of >= 3 near-identical contracts (expected: ~100-200
Oraclize duplicates per friend extrapolation from 808 sample).

Stripping rules (v1):
- Remove Solidity comments (// and /* */)
- Collapse whitespace
- Lowercase everything

This is a lightweight dedup — full normalization (pragma, version, etc.)
will be done in Stage 0.2b if needed.

Output:
- ws_p4_s02_dedup_clusters.csv  (one row per cluster, with member IDs)
- ws_p4_s02_sampling_frame.csv  (v1.1 + is_oraclize_dup flag, members of dups flagged)
"""
import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/motafeq/projects/sentinel")
V11 = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s01b_d12_applied.csv"
SRC = REPO / "BCCC-SCsVul-2024/SourceCodes"
OUT_DEDUP = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s02_dedup_clusters.csv"
OUT_FRAME = REPO / "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s02_sampling_frame.csv"

CLUSTER_MIN_SIZE = 3


def strip_source(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", " ", src)
    src = re.sub(r"\s+", " ", src)
    return src.lower().strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=V11)
    ap.add_argument("--src-root", type=Path, default=SRC)
    ap.add_argument("--out-dedup", type=Path, default=OUT_DEDUP)
    ap.add_argument("--out-frame", type=Path, default=OUT_FRAME)
    args = ap.parse_args()

    if not args.inp.exists():
        print(f"ERROR: input not found: {args.inp}", file=sys.stderr)
        return 1

    with args.inp.open() as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = r.fieldnames
    print(f"Loaded {len(rows)} contracts from {args.inp.name}")

    hashes = {}
    missing = []
    n_stripped = 0
    for i, row in enumerate(rows):
        rel = row["bccc_file_path"]
        if not rel:
            missing.append(row["id"])
            continue
        rel_clean = rel
        if rel_clean.startswith("BCCC-SCsVul-2024/Source Codes/"):
            rel_clean = rel_clean.replace("BCCC-SCsVul-2024/Source Codes/", "")
        elif rel_clean.startswith("BCCC-SCsVul-2024/SourceCodes/"):
            rel_clean = rel_clean.replace("BCCC-SCsVul-2024/SourceCodes/", "")
        p = args.src_root / rel_clean
        if not p.exists():
            missing.append(row["id"])
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            missing.append(row["id"])
            continue
        s = strip_source(src)
        if not s:
            missing.append(row["id"])
            continue
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        hashes[row["id"]] = h
        n_stripped += 1
        if (i + 1) % 5000 == 0:
            print(f"  stripped {i+1}/{len(rows)}")

    print(f"Stripped sources for {n_stripped} contracts; {len(missing)} missing.")

    groups = defaultdict(list)
    for cid, h in hashes.items():
        groups[h].append(cid)
    clusters = {h: ids for h, ids in groups.items() if len(ids) >= CLUSTER_MIN_SIZE}
    print(f"Identified {len(clusters)} clusters of size >= {CLUSTER_MIN_SIZE} "
          f"(total members: {sum(len(v) for v in clusters.values())})")

    cluster_member_set = set()
    for h, ids in clusters.items():
        cluster_member_set.update(ids)
    for cid, h in hashes.items():
        if h in clusters:
            cluster_member_set.add(cid)

    args.out_dedup.parent.mkdir(parents=True, exist_ok=True)
    with args.out_dedup.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hash", "size", "representative_id", "member_ids", "folders"])
        for h, ids in sorted(clusters.items(), key=lambda x: -len(x[1])):
            member_rows = [r for r in rows if r["id"] in ids]
            folders = ",".join(sorted({r["bccc_folder"] for r in member_rows}))
            w.writerow([h, len(ids), ids[0], "|".join(ids), folders])
    print(f"Wrote {args.out_dedup}")

    if "is_oraclize_dup" not in fieldnames:
        fieldnames = fieldnames + ["is_oraclize_dup", "cluster_hash", "cluster_size"]
    for row in rows:
        h = hashes.get(row["id"])
        if h and h in clusters:
            row["is_oraclize_dup"] = "1"
            row["cluster_hash"] = h
            row["cluster_size"] = str(len(clusters[h]))
        else:
            row["is_oraclize_dup"] = "0"
            row["cluster_hash"] = ""
            row["cluster_size"] = "0"

    with args.out_frame.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out_frame} ({len(rows)} rows, "
          f"{sum(1 for r in rows if r['is_oraclize_dup']=='1')} flagged as dups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
