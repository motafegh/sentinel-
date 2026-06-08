"""
3-way consensus: BCCC vs Slither vs Aderyn.
Per-class F1 for each tool pair + majority vote pseudo-labels.
"""
import csv, json, sys, statistics
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07")
SAMPLE_CSV = BASE / "outputs" / "ws_p4_s1_sample.csv"
SLITHER_CSV = BASE / "outputs" / "ws_p4_s1_slither_results.csv"
ADERYN_CSV = BASE / "outputs" / "ws_p4_s1_aderyn_results.csv"
OUT = BASE / "outputs" / "ws_p4_s1_3way_agreement.csv"
REPORT = BASE / "outputs" / "ws_p4_s1_3way_agreement_report.md"

# BCCC class -> slither detector patterns (substring match)
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
    "Class12:NonVulnerable": [],
}

# BCCC class -> aderyn detector patterns (substring match)
ADERYN_MAP = {
    "Class11:Reentrancy": ["reentrancy"],
    "Class10:IntegerUO": [],  # aderyn has no integer overflow detector
    "Class06:UnusedReturn": ["unused-return"],
    "Class01:ExternalBug": ["selfdestruct", "centralization-risk", "unsafe-erc20-operation"],
    "Class08:CallToUnknown": ["centralization-risk", "unsafe-erc20-operation"],
    "Class03:MishandledException": ["unchecked-return", "uninitialized-local-variable", "missing-zero-check", "boolean-equality"],
    "Class02:GasException": [],
    "Class09:DenialOfService": [],
    "Class04:Timestamp": ["block-timestamp-dependency", "timestamp"],
    "Class12:NonVulnerable": [],
}

BCCC_CLASSES = list(SLITHER_MAP.keys())


def classify_hit(hits, cls, detector_map):
    patterns = detector_map.get(cls, [])
    if not patterns:
        return False
    hits_lower = [h.lower() for h in hits]
    return any(any(p in h for p in patterns) for h in hits_lower)


def main():
    with open(SAMPLE_CSV) as f:
        bccc = {r["id"]: r for r in csv.DictReader(f)}
    print(f"BCCC: {len(bccc)} contracts")

    slither = {}
    with open(SLITHER_CSV) as f:
        for r in csv.DictReader(f):
            hits = json.loads(r["hits_json"]) if r["hits_json"] else []
            slither[r["id"]] = {"status": r["status"], "hits": hits}
    print(f"Slither: {len(slither)} ({sum(1 for v in slither.values() if v['status']=='OK')} OK)")

    aderyn = {}
    with open(ADERYN_CSV) as f:
        for r in csv.DictReader(f):
            hits = json.loads(r["hits_json"]) if r["hits_json"] else []
            aderyn[r["id"]] = {"status": r["status"], "hits": hits}
    print(f"Aderyn: {len(aderyn)} ({sum(1 for v in aderyn.values() if v['status']=='OK')} OK)")

    results = []
    for cls in BCCC_CLASSES:
        metrics = {}
        for tool_name, tool_data, tool_map in [
            ("slither", slither, SLITHER_MAP),
            ("aderyn", aderyn, ADERYN_MAP),
        ]:
            tp = fp = fn = tn = skipped = 0
            for cid, br in bccc.items():
                bccc_has = br.get(cls, "0") == "1"
                td = tool_data.get(cid)
                if td is None or td["status"] != "OK":
                    skipped += 1
                    continue
                tool_has = classify_hit(td["hits"], cls, tool_map)
                if bccc_has and tool_has: tp += 1
                elif not bccc_has and tool_has: fp += 1
                elif bccc_has and not tool_has: fn += 1
                else: tn += 1
            total = tp + fp + fn + tn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            metrics[tool_name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                  "precision": prec, "recall": rec, "f1": f1,
                                  "support": tp + fn, "skipped": skipped}

        # Majority vote: 2/3 tools agree
        tp3 = fp3 = fn3 = tn3 = skipped3 = 0
        for cid, br in bccc.items():
            bccc_has = br.get(cls, "0") == "1"
            votes = 0
            valid = 0
            for tool_data, tool_map in [(slither, SLITHER_MAP), (aderyn, ADERYN_MAP)]:
                td = tool_data.get(cid)
                if td is None or td["status"] != "OK":
                    continue
                valid += 1
                if classify_hit(td["hits"], cls, tool_map):
                    votes += 1
            if valid < 2:
                skipped3 += 1
                continue
            tool_has = votes >= 2
            if bccc_has and tool_has: tp3 += 1
            elif not bccc_has and tool_has: fp3 += 1
            elif bccc_has and not tool_has: fn3 += 1
            else: tn3 += 1
        total3 = tp3 + fp3 + fn3 + tn3
        prec3 = tp3 / (tp3 + fp3) if (tp3 + fp3) > 0 else 0.0
        rec3 = tp3 / (tp3 + fn3) if (tp3 + fn3) > 0 else 0.0
        f13 = 2 * prec3 * rec3 / (prec3 + rec3) if (prec3 + rec3) > 0 else 0.0
        metrics["majority"] = {"tp": tp3, "fp": fp3, "fn": fn3, "tn": tn3,
                               "precision": prec3, "recall": rec3, "f1": f13,
                               "support": tp3 + fn3, "skipped": skipped3}

        row = {"class": cls}
        for tool_name, tool_metrics in metrics.items():
            for metric_name, metric_val in tool_metrics.items():
                row[f"{tool_name}_{metric_name}"] = metric_val
        results.append(row)

    # Print
    print(f"\n{'='*100}")
    print("PER-CLASS AGREEMENT: Slither vs Aderyn vs Majority (2/3)")
    print(f"{'='*100}")
    header = f"{'Class':<28} {'Slith_F1':>8} {'Ader_F1':>8} {'Maj_F1':>8} | {'S_TP':>5} {'S_FP':>5} {'A_TP':>5} {'A_FP':>5} {'M_TP':>5} {'M_FP':>5}"
    print(header)
    print("-" * len(header))

    slith_f1s = []
    ader_f1s = []
    maj_f1s = []
    for r in results:
        cls = r["class"]
        s_f1 = r.get("slither_f1", 0)
        a_f1 = r.get("aderyn_f1", 0)
        m_f1 = r.get("majority_f1", 0)
        if cls != "Class12:NonVulnerable":
            slith_f1s.append(s_f1)
            ader_f1s.append(a_f1)
            maj_f1s.append(m_f1)
        s_tp = r.get("slither_tp", 0)
        s_fp = r.get("slither_fp", 0)
        a_tp = r.get("aderyn_tp", 0)
        a_fp = r.get("aderyn_fp", 0)
        m_tp = r.get("majority_tp", 0)
        m_fp = r.get("majority_fp", 0)
        print(f"{cls:<28} {s_f1:>8.3f} {a_f1:>8.3f} {m_f1:>8.3f} | {s_tp:>5} {s_fp:>5} {a_tp:>5} {a_fp:>5} {m_tp:>5} {m_fp:>5}")
    print("-" * len(header))

    med_s = statistics.median(slith_f1s)
    med_a = statistics.median(ader_f1s)
    med_m = statistics.median(maj_f1s)
    print(f"{'Median F1 (vuln)':>28} {med_s:>8.3f} {med_a:>8.3f} {med_m:>8.3f}")

    for name, med in [("Slither", med_s), ("Aderyn", med_a), ("Majority", med_m)]:
        gate = "PASS" if med >= 0.5 else "FAIL"
        print(f"  {name} gate: {gate}")

    # Write report
    with open(REPORT, "w") as f:
        f.write(f"# 3-Way Agreement: BCCC vs Slither vs Aderyn\n\n")
        f.write(f"**Date:** 2026-06-08\n\n")
        f.write(f"**Sample:** {len(bccc)} contracts\n")
        f.write(f"**Slither OK:** {sum(1 for v in slither.values() if v['status']=='OK')}\n")
        f.write(f"**Aderyn OK:** {sum(1 for v in aderyn.values() if v['status']=='OK')}\n\n")
        f.write(f"## Per-Class F1\n\n")
        f.write(f"| Class | Slither F1 | Aderyn F1 | Majority F1 | Support |\n")
        f.write(f"|-------|-----------|----------|------------|---------|\n")
        for r in results:
            cls = r["class"]
            s_f1 = r.get("slither_f1", 0)
            a_f1 = r.get("aderyn_f1", 0)
            m_f1 = r.get("majority_f1", 0)
            sup = r.get("slither_support", 0)
            f.write(f"| {cls} | {s_f1:.3f} | {a_f1:.3f} | {m_f1:.3f} | {sup} |\n")
        f.write(f"\n**Median F1:** Slither={med_s:.3f}  Aderyn={med_a:.3f}  Majority={med_m:.3f}\n\n")
        for name, med in [("Slither", med_s), ("Aderyn", med_a), ("Majority", med_m)]:
            gate = "PASS" if med >= 0.5 else "FAIL"
            f.write(f"**{name} gate:** {gate}\n\n")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "slither_tp", "slither_fp", "slither_f1",
                     "aderyn_tp", "aderyn_fp", "aderyn_f1",
                     "majority_tp", "majority_fp", "majority_f1", "support"])
        for r in results:
            cls = r["class"]
            w.writerow([cls,
                        r.get("slither_tp", 0), r.get("slither_fp", 0), f"{r.get('slither_f1',0):.4f}",
                        r.get("aderyn_tp", 0), r.get("aderyn_fp", 0), f"{r.get('aderyn_f1',0):.4f}",
                        r.get("majority_tp", 0), r.get("majority_fp", 0), f"{r.get('majority_f1',0):.4f}",
                        r.get("slither_support", 0)])

    print(f"\nWrote {OUT}")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
