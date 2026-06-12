"""
Generate readable markdown review document from the 50-contract CSV.
"""
import csv, json
from pathlib import Path

BASE = Path("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07")
CSV = BASE / "outputs" / "ws_p4_s1_manual_review_50.csv"
OUT = BASE / "outputs" / "ws_p4_s1_manual_review_50.md"

CLASS_DESCRIPTIONS = {
    "Class11:Reentrancy": "Reentrant external call (state change after external call)",
    "Class10:IntegerUO": "Integer overflow/underflow (pre-SafeMath arithmetic)",
    "Class06:UnusedReturn": "Return value of external call not checked",
    "Class01:ExternalBug": "Unsafe external call (low-level, delegatecall, selfdestruct)",
    "Class08:CallToUnknown": "Call to untrusted/unknown address",
    "Class03:MishandledException": "Exception not handled (unchecked call/transfer/return)",
    "Class02:GasException": "Gas-dependent loops or operations that may exceed block gas limit",
    "Class09:DenialOfService": "Denial of service (looped calls, gas exhaustion)",
    "Class04:Timestamp": "Block timestamp dependency",
}

with open(CSV) as f:
    rows = list(csv.DictReader(f))

by_class = {}
for r in rows:
    by_class.setdefault(r["class"], []).append(r)

lines = [
    "# Manual Review: 43 High-Uncertainty Contracts",
    "",
    "**Purpose:** Validate BCCC labels where neither slither nor aderyn found a matching detector hit.",
    "**Method:** Read each contract's source code. For each, determine:",
    "  - **KEEP**: BCCC label is correct (the vulnerability exists)",
    "  - **DROP**: BCCC label is incorrect (no vulnerability of this type)",
    "  - **UNCERTAIN**: Can't determine from source alone",
    "",
    f"**Total contracts:** {len(rows)}",
    "",
    "| # | Class | ID | LOC | NF | Decision | Notes |",
    "|---|-------|----|----|----|----------|-------|",
]

idx = 0
for cls in sorted(by_class.keys()):
    for r in by_class[cls]:
        idx += 1
        desc = CLASS_DESCRIPTIONS.get(cls, "")
        lines.append(f"| {idx} | {cls} | {r['id'][:16]} | {r['loc']} | {r['n_functions']} | | |")
    lines.append("")

lines.append("")
lines.append("---")
lines.append("")

for cls in sorted(by_class.keys()):
    desc = CLASS_DESCRIPTIONS.get(cls, "")
    lines.append(f"## {cls} ({len(by_class[cls])} contracts)")
    lines.append(f"**What to look for:** {desc}")
    lines.append("")
    for r in by_class[cls]:
        lines.append(f"### Contract {r['id'][:16]}")
        lines.append(f"- **LOC:** {r['loc']} | **Functions:** {r['n_functions']}")
        lines.append(f"- **Slither:** {r['slither_status']} | **Aderyn:** {r['aderyn_status']}")
        lines.append(f"- **Source:** `{r['bccc_file_path']}`")
        lines.append("")
        lines.append("```solidity")
        lines.append(r["source_code"][:3000])
        if len(r["source_code"]) > 3000:
            lines.append(f"\n// ... truncated ({len(r['source_code'])} chars total)")
        lines.append("```")
        lines.append("")
        lines.append("**Decision:** KEEP / DROP / UNCERTAIN")
        lines.append("**Notes:**")
        lines.append("")
    lines.append("---")
    lines.append("")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {OUT} ({len(lines)} lines)")
