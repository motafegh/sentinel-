"""
Select 50 high-uncertainty contracts for manual review.
These are contracts where BCCC labels=1 but neither slither nor aderyn
found a matching detector hit — the tools disagree with BCCC.
"""
import csv, json, sys
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07")
SAMPLE = BASE / "outputs" / "ws_p4_s1_sample.csv"
SLITHER = BASE / "outputs" / "ws_p4_s1_slither_results.csv"
ADERYN = BASE / "outputs" / "ws_p4_s1_aderyn_results.csv"
SRC = Path("/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes")
OUT = BASE / "outputs" / "ws_p4_s1_manual_review_50.csv"

SLITHER_MAP = {
    "Class11:Reentrancy": ["reentrancy"],
    "Class10:IntegerUO": ["divide-before-multiply"],
    "Class06:UnusedReturn": ["unused-return"],
    "Class01:ExternalBug": ["low-level-calls", "controlled-delegatecall", "selfdestruct", "tx-origin", "arbitrary-send"],
    "Class08:CallToUnknown": ["controlled-delegatecall", "low-level-calls"],
    "Class03:MishandledException": ["unchecked-transfer", "unchecked-send", "unchecked-lowlevel", "unused-return", "missing-zero-check"],
    "Class02:GasException": ["costly-loop", "msg-value-loop"],
    "Class09:DenialOfService": ["calls-loop", "reentrancy-unlimited-gas"],
    "Class04:Timestamp": ["timestamp"],
}

ADERYN_MAP = {
    "Class11:Reentrancy": ["reentrancy"],
    "Class06:UnusedReturn": ["unused-return"],
    "Class01:ExternalBug": ["selfdestruct", "centralization-risk", "unsafe-erc20-operation"],
    "Class08:CallToUnknown": ["centralization-risk", "unsafe-erc20-operation"],
    "Class03:MishandledException": ["unchecked-return", "uninitialized-local-variable", "missing-zero-check", "boolean-equality"],
    "Class04:Timestamp": ["block-timestamp-dependency", "timestamp"],
}


def classify_hit(hits, cls, det_map):
    patterns = det_map.get(cls, [])
    if not patterns:
        return False
    hits_lower = [h.lower() for h in hits]
    return any(any(p in h for p in patterns) for h in hits_lower)


def fix_path(stored):
    stored = stored.replace("BCCC-SCsVul-2024/Source Codes/", "")
    stored = stored.replace("BCCC-SCsVul-2024/SourceCodes/", "")
    return str(SRC / stored)


def main():
    with open(SAMPLE) as f:
        bccc = {r["id"]: r for r in csv.DictReader(f)}
    sample = {cid: r for cid, r in bccc.items() if r.get("in_stage1_sample") == "1"}
    print(f"Sample: {len(sample)} contracts")

    slither = {}
    with open(SLITHER) as f:
        for r in csv.DictReader(f):
            hits = json.loads(r["hits_json"]) if r["hits_json"] else []
            slither[r["id"]] = {"status": r["status"], "hits": hits}

    aderyn = {}
    with open(ADERYN) as f:
        for r in csv.DictReader(f):
            hits = json.loads(r["hits_json"]) if r["hits_json"] else []
            aderyn[r["id"]] = {"status": r["status"], "hits": hits}

    fn_by_class = defaultdict(list)
    for cid, r in sample.items():
        for cls in SLITHER_MAP:
            if cls == "Class12:NonVulnerable":
                continue
            if r.get(cls, "0") != "1":
                continue
            s = slither.get(cid, {})
            a = aderyn.get(cid, {})
            s_hit = classify_hit(s.get("hits", []), cls, SLITHER_MAP) if s.get("status") == "OK" else False
            a_hit = classify_hit(a.get("hits", []), cls, ADERYN_MAP) if a.get("status") == "OK" else False
            if not s_hit and not a_hit:
                fn_by_class[cls].append(cid)

    print(f"\nFN counts (BCCC=1, neither tool found it):")
    total_fn = sum(len(v) for v in fn_by_class.values())
    for cls, ids in sorted(fn_by_class.items(), key=lambda x: -len(x[1])):
        print(f"  {cls}: {len(ids)}")
    print(f"  TOTAL: {total_fn}")

    # Sample 50 proportionally
    target = 50
    sampled = {}
    for cls, ids in sorted(fn_by_class.items(), key=lambda x: -len(x[1])):
        n = max(1, round(target * len(ids) / total_fn))
        if cls == "Class09:DenialOfService":
            n = 1
        import random
        random.seed(42)
        sampled[cls] = random.sample(ids, min(n, len(ids)))

    total_sampled = sum(len(v) for v in sampled.values())
    print(f"\nSampled for manual review: {total_sampled}")
    for cls, ids in sorted(sampled.items(), key=lambda x: -len(x[1])):
        print(f"  {cls}: {len(ids)}")

    rows = []
    for cls, ids in sampled.items():
        for cid in ids:
            r = sample[cid]
            fp = fix_path(r["bccc_file_path"])
            try:
                with open(fp) as f:
                    code = f.read()
            except:
                code = "# FILE NOT FOUND"
            rows.append({
                "id": cid,
                "class": cls,
                "bccc_file_path": r["bccc_file_path"],
                "slither_status": slither.get(cid, {}).get("status", "N/A"),
                "slither_hits": json.dumps(slither.get(cid, {}).get("hits", [])),
                "aderyn_status": aderyn.get(cid, {}).get("status", "N/A"),
                "aderyn_hits": json.dumps(aderyn.get(cid, {}).get("hits", [])),
                "loc": r.get("loc", ""),
                "n_functions": r.get("n_functions", ""),
                "source_code": code,
            })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "class", "bccc_file_path",
                                           "slither_status", "slither_hits",
                                           "aderyn_status", "aderyn_hits",
                                           "loc", "n_functions", "source_code"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT}")

    for cls, ids in sorted(sampled.items(), key=lambda x: -len(x[1])):
        print(f"\n--- {cls} ({len(ids)} contracts) ---")
        for cid in ids[:3]:
            r = sample[cid]
            fp = fix_path(r["bccc_file_path"])
            try:
                with open(fp) as f:
                    code = f.read()[:200]
            except:
                code = "# NOT FOUND"
            print(f"  {cid[:16]}  LOC={r.get('loc','?')}  NF={r.get('n_functions','?')}")
            print(f"    {code[:150]}...")
        if len(ids) > 3:
            print(f"  ... and {len(ids)-3} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
