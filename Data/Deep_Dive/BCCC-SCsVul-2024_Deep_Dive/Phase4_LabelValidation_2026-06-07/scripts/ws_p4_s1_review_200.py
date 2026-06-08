"""
Expanded manual review: 200 contracts. Samples TP (tool agrees) + FN (tool disagrees).
"""
import csv, json, re, random
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07")
SRC = Path("/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes")
OUT = BASE / "outputs" / "ws_p4_s1_review_200.csv"
REPORT = BASE / "outputs" / "ws_p4_s1_review_200_report.md"

S_MAP = {
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
A_MAP = {
    "Class11:Reentrancy": ["reentrancy"],
    "Class06:UnusedReturn": ["unused-return"],
    "Class01:ExternalBug": ["selfdestruct", "centralization-risk", "unsafe-erc20-operation"],
    "Class08:CallToUnknown": ["centralization-risk", "unsafe-erc20-operation"],
    "Class03:MishandledException": ["unchecked-return", "uninitialized-local-variable", "missing-zero-check", "boolean-equality"],
    "Class04:Timestamp": ["block-timestamp-dependency", "timestamp"],
}

VULN_CLASSES = [c for c in S_MAP if c != "Class12:NonVulnerable"]


def fix_path(p):
    return str(SRC / p.replace("BCCC-SCsVul-2024/Source Codes/", "").replace("BCCC-SCsVul-2024/SourceCodes/", ""))


def classify(hits, cls, dmap):
    pats = dmap.get(cls, [])
    if not pats:
        return False
    hl = [h.lower() for h in hits]
    return any(any(p in h for p in pats) for h in hl)


def review(code, cls):
    if cls == "Class11:Reentrancy":
        has_call = bool(re.search(r"\.call[\{\(]", code))
        has_send = ".send(" in code
        has_transfer = ".transfer(" in code
        has_state_change = bool(re.search(r"(balances?\[|totalSupply|amount|owner|supply).*\=", code))
        if has_call:
            return "KEEP" if has_state_change else "UNCERTAIN", "call found"
        elif has_send:
            return "UNCERTAIN", "send found (may be reentrancy)"
        elif has_transfer:
            return "DROP", "only transfer (safe, reverts)"
        return "DROP", "no external call"

    elif cls == "Class10:IntegerUO":
        has_safe = "SafeMath" in code
        has_arith = bool(re.search(r"(amount|balance|total|value|supply|wei)\s*[+\-*/]\s*\d", code))
        if has_arith and not has_safe:
            return "KEEP", "arithmetic without SafeMath"
        elif has_arith and has_safe:
            return "UNCERTAIN", "arithmetic with SafeMath"
        return "UNCERTAIN", "no clear pattern"

    elif cls == "Class06:UnusedReturn":
        has_unchecked = bool(re.search(r"\.call\{[^}]*\}\s*\([^)]*\)\s*;", code))
        if has_unchecked:
            return "KEEP", "unchecked call return"
        return "UNCERTAIN", "no unchecked return"

    elif cls == "Class01:ExternalBug":
        if "selfdestruct" in code or "suicide" in code:
            return "KEEP", "selfdestruct"
        if "tx.origin" in code:
            return "KEEP", "tx.origin"
        if ".delegatecall(" in code:
            return "KEEP", "delegatecall"
        if ".call(" in code:
            return "KEEP", "low-level call"
        return "DROP", "no external bug"

    elif cls == "Class08:CallToUnknown":
        if ".call(" in code or ".delegatecall(" in code or ".staticcall(" in code:
            return "KEEP", "external call"
        return "DROP", "no external call"

    elif cls == "Class03:MishandledException":
        if bool(re.search(r"\.call\{.*\}\s*\(", code)):
            return "KEEP", "unchecked call"
        if ".send(" in code:
            return "UNCERTAIN", "send found"
        return "UNCERTAIN", "no clear pattern"

    elif cls == "Class02:GasException":
        has_loop = bool(re.search(r"(for|while)\s*\(", code))
        has_ext_in_loop = bool(re.search(r"(for|while)\s*\(.*\.call\(", code, re.DOTALL))
        if has_ext_in_loop:
            return "KEEP", "ext call in loop"
        if has_loop:
            return "UNCERTAIN", "loop found"
        return "DROP", "no loop"

    elif cls == "Class09:DenialOfService":
        has_loop = bool(re.search(r"(for|while)\s*\(", code))
        if has_loop and bool(re.search(r"\.call\(", code)):
            return "KEEP", "ext call near loop"
        if has_loop:
            return "UNCERTAIN", "loop"
        return "DROP", "no loop"

    elif cls == "Class04:Timestamp":
        if "block.timestamp" in code or " now " in code:
            return "KEEP", "timestamp usage"
        return "DROP", "no timestamp"

    return "UNCERTAIN", "unclassified"


def main():
    random.seed(42)
    with open(BASE / "outputs/ws_p4_s1_sample.csv") as f:
        all_rows = {r["id"]: r for r in csv.DictReader(f)}
    sample = {k: v for k, v in all_rows.items() if v.get("in_stage1_sample") == "1"}
    print(f"Sample: {len(sample)}")

    slither = {}
    with open(BASE / "outputs/ws_p4_s1_slither_results.csv") as f:
        for r in csv.DictReader(f):
            slither[r["id"]] = {"status": r["status"], "hits": json.loads(r["hits_json"]) if r["hits_json"] else []}
    aderyn = {}
    with open(BASE / "outputs/ws_p4_s1_aderyn_results.csv") as f:
        for r in csv.DictReader(f):
            aderyn[r["id"]] = {"status": r["status"], "hits": json.loads(r["hits_json"]) if r["hits_json"] else []}

    # Classify each contract as TP or FN per class
    by_class_tp = defaultdict(list)
    by_class_fn = defaultdict(list)
    for cid, r in sample.items():
        for cls in VULN_CLASSES:
            if r.get(cls, "0") != "1":
                continue
            s = slither.get(cid, {})
            a = aderyn.get(cid, {})
            s_hit = classify(s.get("hits", []), cls, S_MAP) if s.get("status") == "OK" else False
            a_hit = classify(a.get("hits", []), cls, A_MAP) if a.get("status") == "OK" else False
            if s_hit or a_hit:
                by_class_tp[cls].append(cid)
            else:
                by_class_fn[cls].append(cid)

    # Sample 200: 100 TP + 100 FN, proportional to class size
    total_tp = sum(len(v) for v in by_class_tp.values())
    total_fn = sum(len(v) for v in by_class_fn.values())
    print(f"TP: {total_tp}, FN: {total_fn}")

    sampled = []
    for cls in VULN_CLASSES:
        tp_ids = by_class_tp.get(cls, [])
        fn_ids = by_class_fn.get(cls, [])
        n_tp = max(1, round(100 * len(tp_ids) / total_tp)) if total_tp else 0
        n_fn = max(1, round(100 * len(fn_ids) / total_fn)) if total_fn else 0
        if tp_ids:
            sampled.extend([(cid, cls, "TP") for cid in random.sample(tp_ids, min(n_tp, len(tp_ids)))])
        if fn_ids:
            sampled.extend([(cid, cls, "FN") for cid in random.sample(fn_ids, min(n_fn, len(fn_ids)))])

    print(f"Sampled: {len(sampled)} contracts")

    # Review
    results = []
    for cid, cls, group in sampled:
        r = sample[cid]
        fp = fix_path(r["bccc_file_path"])
        try:
            with open(fp) as f:
                code = f.read()
        except:
            code = ""
        decision, reason = review(code, cls) if code else ("UNCERTAIN", "file not found")
        results.append({
            "id": cid, "class": cls, "group": group, "decision": decision,
            "reason": reason, "loc": r.get("loc", ""),
        })

    # Summary
    print(f"\n{'='*80}")
    print("REVIEW 200 RESULTS")
    print(f"{'='*80}")
    keep = sum(1 for r in results if r["decision"] == "KEEP")
    drop = sum(1 for r in results if r["decision"] == "DROP")
    unc = sum(1 for r in results if r["decision"] == "UNCERTAIN")
    print(f"KEEP: {keep}  DROP: {drop}  UNCERTAIN: {unc}  Total: {len(results)}")

    print(f"\n{'Class':<30} {'Group':>5} {'KEEP':>5} {'DROP':>5} {'UNC':>5} {'Total':>6}")
    print("-" * 60)
    by_cg = defaultdict(lambda: defaultdict(int))
    for r in results:
        by_cg[(r["class"], r["group"])][r["decision"]] += 1
    for cls in VULN_CLASSES:
        for grp in ["TP", "FN"]:
            key = (cls, grp)
            if key in by_cg:
                d = by_cg[key]
                t = sum(d.values())
                print(f"{cls:<30} {grp:>5} {d.get('KEEP',0):>5} {d.get('DROP',0):>5} {d.get('UNCERTAIN',0):>5} {t:>6}")
        print()

    # Noise estimates
    print(f"\n=== Noise Estimates (DROP rate) ===")
    for cls in VULN_CLASSES:
        fn_results = [r for r in results if r["class"] == cls and r["group"] == "FN"]
        tp_results = [r for r in results if r["class"] == cls and r["group"] == "TP"]
        fn_drop = sum(1 for r in fn_results if r["decision"] == "DROP")
        tp_drop = sum(1 for r in tp_results if r["decision"] == "DROP")
        fn_n = len(fn_results)
        tp_n = len(tp_results)
        fn_rate = fn_drop / fn_n if fn_n else 0
        tp_rate = tp_drop / tp_n if tp_n else 0
        print(f"  {cls:<30} FN_noise={fn_rate:.0%} ({fn_drop}/{fn_n})  TP_noise={tp_rate:.0%} ({tp_drop}/{tp_n})")

    # Write CSV
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "class", "group", "decision", "reason", "loc"])
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {OUT}")

    # Write report
    lines = ["# Expanded Review: 200 Contracts", "",
             f"**Date:** 2026-06-08", "",
             f"## Summary", "",
             f"| Decision | Count | % |",
             f"|----------|-------|---|",
             f"| KEEP | {keep} | {keep/len(results)*100:.0f}% |",
             f"| DROP | {drop} | {drop/len(results)*100:.0f}% |",
             f"| UNCERTAIN | {unc} | {unc/len(results)*100:.0f}% |",
             "", "## Noise Estimates (DROP rate)", "",
             "| Class | FN Noise | TP Noise | Interpretation |",
             "|-------|----------|----------|----------------|"]
    for cls in VULN_CLASSES:
        fn_results = [r for r in results if r["class"] == cls and r["group"] == "FN"]
        tp_results = [r for r in results if r["class"] == cls and r["group"] == "TP"]
        fn_drop = sum(1 for r in fn_results if r["decision"] == "DROP")
        tp_drop = sum(1 for r in tp_results if r["decision"] == "DROP")
        fn_n = len(fn_results)
        tp_n = len(tp_results)
        fn_rate = fn_drop / fn_n if fn_n else 0
        tp_rate = tp_drop / tp_n if tp_n else 0
        interp = "HIGH noise" if fn_rate > 0.5 else "MODERATE noise" if fn_rate > 0.3 else "LOW noise"
        lines.append(f"| {cls} | {fn_rate:.0%} ({fn_drop}/{fn_n}) | {tp_rate:.0%} ({tp_drop}/{tp_n}) | {interp} |")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
