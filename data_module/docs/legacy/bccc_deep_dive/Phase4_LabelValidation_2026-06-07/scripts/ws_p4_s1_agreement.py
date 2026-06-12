"""
Stage 1 Agreement Analysis: BCCC labels vs Slither findings.
Per-class F1, median F1 gate, escalation decision.
"""
import csv, json, sys, statistics
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07")
SAMPLE_CSV = BASE / "outputs" / "ws_p4_s1_sample.csv"
SLITHER_CSV = BASE / "outputs" / "ws_p4_s1_slither_results.csv"
OUT = BASE / "outputs" / "ws_p4_s1_agreement.csv"
REPORT = BASE / "outputs" / "ws_p4_s1_agreement_report.md"

# BCCC class -> slither detector patterns (substring match on "check" field)
# BCCC CSV uses: Class01:ExternalBug, Class02:GasException, Class03:MishandledException,
# Class04:Timestamp, Class06:UnusedReturn, Class08:CallToUnknown,
# Class09:DenialOfService, Class10:IntegerUO, Class11:Reentrancy, Class12:NonVulnerable
CLASS_TO_SLITHER = {
    # Reentrancy: all reentrancy-* subtypes (benign, events, no-eth, unlimited-gas, eth, balance)
    "Class11:Reentrancy": ["reentrancy"],
    # IntegerUO: divide-before-multiply is the only arithmetic detector with hits
    "Class10:IntegerUO": ["divide-before-multiply"],
    # UnusedReturn: return values not checked — should be 1:1
    "Class06:UnusedReturn": ["unused-return"],
    # ExternalBug: broad — external calls, delegatecall, selfdestruct, tx.origin
    "Class01:ExternalBug": [
        "low-level-calls", "controlled-delegatecall", "selfdestruct",
        "tx-origin", "arbitrary-send",
    ],
    # CallToUnknown: calling unknown/untrusted addresses
    "Class08:CallToUnknown": ["controlled-delegatecall", "low-level-calls"],
    # MishandledException: unchecked/unused return values, missing checks
    "Class03:MishandledException": [
        "unchecked-transfer", "unchecked-send", "unchecked-lowlevel",
        "unused-return", "missing-zero-check",
    ],
    # GasException: loops that exhaust gas, msg.value in loops
    "Class02:GasException": ["costly-loop", "msg-value-loop"],
    # DenialOfService: looped calls + gas-exhaustion reentrancy
    "Class09:DenialOfService": ["calls-loop", "reentrancy-unlimited-gas"],
    # Timestamp: block.timestamp usage
    "Class04:Timestamp": ["timestamp"],
    # NonVulnerable: should have no hits
    "Class12:NonVulnerable": [],
}

BCCC_CLASSES = list(CLASS_TO_SLITHER.keys())


def classify_hit(hit: str, cls: str) -> bool:
    """Check if a slither hit matches the given BCCC class."""
    patterns = CLASS_TO_SLITHER.get(cls, [])
    hit_lower = hit.lower()
    return any(p in hit_lower for p in patterns)


def main():
    # Load BCCC sample labels
    with open(SAMPLE_CSV) as f:
        bccc = {r["id"]: r for r in csv.DictReader(f)}
    print(f"BCCC sample: {len(bccc)} contracts")

    # Load slither results
    slither = {}
    with open(SLITHER_CSV) as f:
        for r in csv.DictReader(f):
            hits = json.loads(r["hits_json"]) if r["hits_json"] else []
            slither[r["id"]] = {
                "status": r["status"],
                "hits": hits,
                "n_hits": int(r["n_hits"]),
            }
    print(f"Slither results: {len(slither)} contracts")
    print(f"  Status distribution: {defaultdict(int, {s: sum(1 for v in slither.values() if v['status'] == s) for s in set(v['status'] for v in slither.values())})}")

    # Per-class metrics
    results = []

    for cls in BCCC_CLASSES:
        tp = fp = fn = tn = 0
        skipped = 0
        for cid, br in bccc.items():
            bccc_label = br.get(cls, "0")
            bccc_has = bccc_label == "1"

            sr = slither.get(cid)
            if sr is None:
                skipped += 1
                continue
            if sr["status"] != "OK":
                skipped += 1
                continue

            slither_has = any(classify_hit(h, cls) for h in sr["hits"])

            if bccc_has and slither_has:
                tp += 1
            elif not bccc_has and slither_has:
                fp += 1
            elif bccc_has and not slither_has:
                fn += 1
            else:
                tn += 1

        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn

        results.append({
            "class": cls,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "support": support,
            "skipped": skipped,
        })

    # Compute median F1 (excluding NonVulnerable — it's not a vuln class)
    vuln_f1s = [r["f1"] for r in results if r["class"] != "Class12:NonVulnerable" and r["support"] > 0]
    median_f1 = statistics.median(vuln_f1s) if vuln_f1s else 0.0
    nonvuln = next((r for r in results if r["class"] == "Class12:NonVulnerable"), None)

    # Escalation gate
    if median_f1 >= 0.5:
        gate = "PASS"
        reason = "Median F1 >= 0.5: BCCC labels trustworthy"
    elif nonvuln and nonvuln["f1"] < 0.5 and len([r for r in results if r["class"] != "Class12:NonVulnerable" and r["f1"] >= 0.5]) == len(vuln_f1s):
        gate = "PASS (IntegerUO/NonVuln outlier only)"
        reason = "All vuln classes pass; only NonVuln/IntegerUO low F1"
    else:
        gate = "FAIL"
        reason = "Median F1 < 0.5: Need escalation to 30%/50% sampling"

    # Print results
    print(f"\n{'='*80}")
    print(f"PER-CLASS AGREEMENT: BCCC vs Slither")
    print(f"{'='*80}")
    print(f"{'Class':<16} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>8}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['class']:<16} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} {r['tn']:>5} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f} {r['support']:>8}")
    print(f"{'-'*80}")
    print(f"{'Median F1 (vuln classes)':>43} {median_f1:>7.3f}")
    print(f"{'Gate':>43} {gate}")
    print(f"  {reason}")

    # Save agreement CSV
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "support", "skipped"])
        for r in results:
            w.writerow([r["class"], r["tp"], r["fp"], r["fn"], r["tn"],
                        f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}",
                        r["support"], r["skipped"]])
    print(f"\nWrote {OUT}")

    # Save agreement report
    with open(REPORT, "w") as f:
        f.write(f"# Stage 1 Agreement Report: BCCC vs Slither\n\n")
        f.write(f"**Date:** 2026-06-07\n\n")
        f.write(f"**Sample size:** {len(bccc)} contracts (Stage 1 sample)\n\n")
        f.write(f"**Slither OK results:** {sum(1 for v in slither.values() if v['status'] == 'OK')}\n\n")
        f.write(f"## Per-Class Results\n\n")
        f.write(f"| Class | TP | FP | FN | TN | Precision | Recall | F1 | Support |\n")
        f.write(f"|-------|----|----|----|----|-----------|--------|----|---------|\n")
        for r in results:
            f.write(f"| {r['class']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | "
                    f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['support']} |\n")
        f.write(f"\n**Median F1 (vuln classes):** {median_f1:.3f}\n\n")
        f.write(f"**Gate:** {gate}\n\n")
        f.write(f"**Reason:** {reason}\n\n")
        f.write(f"## Escalation Decision\n\n")
        if gate.startswith("PASS"):
            f.write(f"Proceed to Stage 4 (Mythril tiebreaker on 50 hardest).\n")
            f.write(f"Stage 2/3 escalation not needed.\n")
        else:
            f.write(f"Escalate to Stage 2: 30% sampling of disagreeing classes.\n")
            f.write(f"If median F1 still < 0.5 after Stage 2 → Stage 3: 50% sampling.\n")
    print(f"Wrote {REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
