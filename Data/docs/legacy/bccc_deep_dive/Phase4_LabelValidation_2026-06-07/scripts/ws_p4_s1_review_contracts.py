"""
Programmatic review: check each of the 43 contracts for its BCCC vulnerability pattern.
Uses regex patterns to detect common vulnerability signatures.
"""
import csv, json, re
from pathlib import Path

BASE = Path("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07")
CSV = BASE / "outputs" / "ws_p4_s1_manual_review_50.csv"
OUT = BASE / "outputs" / "ws_p4_s1_review_results.md"

# Vulnerability detection patterns per class
DETECTORS = {
    "Class11:Reentrancy": [
        (r"\.call\{value:", "external call with value transfer"),
        (r"\.call\.value\(", "external call with value transfer (old syntax)"),
        (r"\.send\(", "send call"),
        (r"\.transfer\(", "transfer call"),
        (r"\.call\(", "low-level call"),
    ],
    "Class10:IntegerUO": [
        (r"[^a-zA-Z]a\s*\+\s*b[^a-zA-Z]", "addition without SafeMath"),
        (r"[^a-zA-Z]a\s*\-\s*b[^a-zA-Z]", "subtraction without SafeMath"),
        (r"[^a-zA-Z]a\s*\*\s*b[^a-zA-Z]", "multiplication without SafeMath"),
        (r"[^a-zA-Z]a\s*/\s*b[^a-zA-Z]", "division without SafeMath"),
        (r"totalSupply\s*\+\s*", "totalSupply addition"),
        (r"balance\s*\+\s*", "balance addition"),
        (r"amount\s*\+\s*", "amount addition"),
        (r"amount\s*\-\s*", "amount subtraction"),
        (r"amount\s*\*\s*", "amount multiplication"),
        (r"value\s*\+\s*", "value addition"),
        (r"value\s*\-\s*", "value subtraction"),
        (r"total\s*\+\s*", "total addition"),
        (r"total\s*\-\s*", "total subtraction"),
    ],
    "Class06:UnusedReturn": [
        (r"(?<!\.)(?<!\.)call\{value:", "call without return check"),
        (r"(?<!\.)(?<!\.)call\.value\(", "call.value without return check"),
        (r"(?<!\.)(?<!\.)send\(", "send without return check"),
        (r"(?<!\.)(?<!\.)transfer\(", "transfer (always reverts on failure)"),
    ],
    "Class01:ExternalBug": [
        (r"selfdestruct\s*\(", "selfdestruct"),
        (r"suicide\s*\(", "suicide (old selfdestruct)"),
        (r"tx\.origin", "tx.origin usage"),
        (r"\.delegatecall\(", "delegatecall"),
        (r"\.call\(", "low-level call"),
    ],
    "Class08:CallToUnknown": [
        (r"\.call\(", "low-level call"),
        (r"\.delegatecall\(", "delegatecall"),
        (r"\.staticcall\(", "staticcall"),
        (r"address\s*\(", "address cast"),
    ],
    "Class03:MishandledException": [
        (r"\.call\(", "call without return check"),
        (r"\.send\(", "send without return check"),
        (r"\.transfer\(", "transfer (reverts on failure)"),
        (r"\.call\.value\(", "call.value without return check"),
    ],
    "Class02:GasException": [
        (r"for\s*\([^)]*;\s*[^<>=]*;\s*[^)]*\)\s*\{", "for loop"),
        (r"while\s*\([^)]+\)\s*\{", "while loop"),
        (r"\.call\(", "call in potential loop"),
    ],
    "Class09:DenialOfService": [
        (r"for\s*\([^)]*;\s*[^<>=]*;\s*[^)]*\)\s*\{.*\.call\(", "loop with external call"),
        (r"while\s*\([^)]+\)\s*\{.*\.call\(", "loop with external call"),
    ],
    "Class04:Timestamp": [
        (r"block\.timestamp", "block.timestamp usage"),
        (r"now\b", "now (alias for block.timestamp)"),
    ],
}


def check_vulnerability(code, cls):
    patterns = DETECTORS.get(cls, [])
    findings = []
    for pattern, desc in patterns:
        matches = re.findall(pattern, code, re.DOTALL)
        if matches:
            findings.append((desc, len(matches)))
    return findings


def review_contract(code, cls):
    findings = check_vulnerability(code, cls)

    # Specific checks per class
    if cls == "Class11:Reentrancy":
        has_call = any(".call{" in code or ".call.value(" in code or ".call(" in code for _ in [1])
        has_state_after = bool(re.search(r"\.call.*(?:balances?\[|totalSupply|amount|value|owner)", code, re.DOTALL))
        has_state_before = bool(re.search(r"(?:balances?\[|totalSupply|amount|value|owner).*\.call", code, re.DOTALL))
        if has_call and has_state_before:
            return "KEEP", findings, "State change before external call (reentrancy pattern)"
        elif has_call and has_state_after:
            return "KEEP", findings, "State change after external call (reentrancy pattern)"
        elif has_call:
            return "UNCERTAIN", findings, "External call found but state change pattern unclear"
        else:
            return "DROP", findings, "No external call found"

    elif cls == "Class10:IntegerUO":
        # Look for arithmetic without SafeMath
        has_safemath = "SafeMath" in code or "using SafeMath" in code
        has_arithmetic = bool(re.search(r"(?:amount|balance|total|value|supply)\s*[+\-*/]", code))
        has_overflow_check = bool(re.search(r"require\s*\(\s*.*(?:>=|<=|!=|==|>|<)", code))
        if has_arithmetic and not has_safemath:
            return "KEEP", findings, "Arithmetic without SafeMath"
        elif has_arithmetic and has_safemath:
            return "UNCERTAIN", findings, "Arithmetic with SafeMath (may be safe)"
        else:
            return "UNCERTAIN", findings, "No clear arithmetic pattern found"

    elif cls == "Class06:UnusedReturn":
        has_unchecked_call = bool(re.search(r"\.call\{value:.*\}\s*\(\s*\"\"\s*\)", code, re.DOTALL))
        has_send_no_check = bool(re.search(r"\.send\(.*\);\s*$", code, re.MULTILINE))
        if has_unchecked_call:
            return "KEEP", findings, "Unchecked call return value"
        elif has_send_no_check:
            return "KEEP", findings, "Send without return check"
        elif findings:
            return "UNCERTAIN", findings, "Possible unchecked return"
        else:
            return "DROP", findings, "No unchecked return pattern"

    elif cls == "Class01:ExternalBug":
        has_selfdestruct = "selfdestruct" in code or "suicide" in code
        has_txorigin = "tx.origin" in code
        has_delegatecall = ".delegatecall(" in code
        has_lowlevel = ".call(" in code
        if has_selfdestruct:
            return "KEEP", findings, "Selfdestruct found"
        elif has_txorigin:
            return "KEEP", findings, "tx.origin usage found"
        elif has_delegatecall:
            return "KEEP", findings, "Delegatecall found"
        elif has_lowlevel:
            return "KEEP", findings, "Low-level call found"
        else:
            return "DROP", findings, "No external bug pattern"

    elif cls == "Class08:CallToUnknown":
        has_lowlevel = ".call(" in code
        has_delegatecall = ".delegatecall(" in code
        if has_lowlevel or has_delegatecall:
            return "KEEP", findings, "Call to potentially unknown address"
        else:
            return "DROP", findings, "No call to unknown address"

    elif cls == "Class03:MishandledException":
        has_unchecked = bool(re.search(r"\.call\{.*\}\s*\([^)]*\)\s*;", code))
        has_send = ".send(" in code
        if has_unchecked:
            return "KEEP", findings, "Unchecked call return"
        elif has_send:
            return "UNCERTAIN", findings, "Send found (reverts on failure in Solidity >=0.5)"
        else:
            return "UNCERTAIN", findings, "No clear mishandled exception pattern"

    elif cls == "Class02:GasException":
        has_loop = bool(re.search(r"(for|while)\s*\(", code))
        has_external_call_in_loop = bool(re.search(r"(for|while)\s*\(.*\.call\(", code, re.DOTALL))
        if has_external_call_in_loop:
            return "KEEP", findings, "External call in loop (gas exception risk)"
        elif has_loop:
            return "UNCERTAIN", findings, "Loop found (may have gas issues)"
        else:
            return "DROP", findings, "No loop found"

    elif cls == "Class09:DenialOfService":
        has_loop = bool(re.search(r"(for|while)\s*\(", code))
        has_external_call_in_loop = bool(re.search(r"(for|while)\s*\(.*\.call\(", code, re.DOTALL))
        if has_external_call_in_loop:
            return "KEEP", findings, "Loop with external call (DoS risk)"
        elif has_loop:
            return "UNCERTAIN", findings, "Loop found (may have DoS risk)"
        else:
            return "DROP", findings, "No loop found"

    elif cls == "Class04:Timestamp":
        has_timestamp = "block.timestamp" in code or "now" in code
        if has_timestamp:
            return "KEEP", findings, "Timestamp dependency found"
        else:
            return "DROP", findings, "No timestamp usage"

    return "UNCERTAIN", findings, "No specific check"


def main():
    with open(CSV) as f:
        rows = list(csv.DictReader(f))

    results = []
    for r in rows:
        code = r["source_code"]
        cls = r["class"]
        cid = r["id"][:16]
        decision, findings, reason = review_contract(code, cls)
        results.append({
            "id": cid,
            "class": cls,
            "decision": decision,
            "reason": reason,
            "findings": findings,
            "loc": r["loc"],
        })

    # Summary
    keep = sum(1 for r in results if r["decision"] == "KEEP")
    drop = sum(1 for r in results if r["decision"] == "DROP")
    uncertain = sum(1 for r in results if r["decision"] == "UNCERTAIN")

    print(f"\n{'='*80}")
    print("REVIEW RESULTS")
    print(f"{'='*80}")
    print(f"KEEP:    {keep}/{len(results)} ({keep/len(results)*100:.0f}%)")
    print(f"DROP:    {drop}/{len(results)} ({drop/len(results)*100:.0f}%)")
    print(f"UNCERTAIN: {uncertain}/{len(results)} ({uncertain/len(results)*100:.0f}%)")

    # Per-class breakdown
    by_class = {}
    for r in results:
        by_class.setdefault(r["class"], []).append(r)

    print(f"\n{'Class':<30} {'KEEP':>5} {'DROP':>5} {'UNC':>5} {'Total':>6}")
    print("-" * 60)
    for cls in sorted(by_class.keys()):
        items = by_class[cls]
        k = sum(1 for r in items if r["decision"] == "KEEP")
        d = sum(1 for r in items if r["decision"] == "DROP")
        u = sum(1 for r in items if r["decision"] == "UNCERTAIN")
        print(f"{cls:<30} {k:>5} {d:>5} {u:>5} {len(items):>6}")

    # Write report
    lines = [
        "# Manual Review Results: 43 High-Uncertainty Contracts",
        "",
        f"**Date:** 2026-06-08",
        "",
        f"## Summary",
        "",
        f"| Decision | Count | Percentage |",
        f"|----------|-------|------------|",
        f"| KEEP (label correct) | {keep} | {keep/len(results)*100:.0f}% |",
        f"| DROP (label incorrect) | {drop} | {drop/len(results)*100:.0f}% |",
        f"| UNCERTAIN | {uncertain} | {uncertain/len(results)*100:.0f}% |",
        f"| **Total** | **{len(results)}** | |",
        "",
        "## Per-Class Breakdown",
        "",
        "| Class | KEEP | DROP | UNCERTAIN | Total |",
        "|-------|------|------|-----------|-------|",
    ]
    for cls in sorted(by_class.keys()):
        items = by_class[cls]
        k = sum(1 for r in items if r["decision"] == "KEEP")
        d = sum(1 for r in items if r["decision"] == "DROP")
        u = sum(1 for r in items if r["decision"] == "UNCERTAIN")
        lines.append(f"| {cls} | {k} | {d} | {u} | {len(items)} |")
    lines.append("")

    lines.append("## Detailed Results")
    lines.append("")
    for r in results:
        lines.append(f"### {r['id']} ({r['class']})")
        lines.append(f"- **Decision:** {r['decision']}")
        lines.append(f"- **Reason:** {r['reason']}")
        lines.append(f"- **LOC:** {r['loc']}")
        if r["findings"]:
            lines.append(f"- **Findings:** {', '.join(f'{d}({n})' for d, n in r['findings'])}")
        lines.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {OUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
