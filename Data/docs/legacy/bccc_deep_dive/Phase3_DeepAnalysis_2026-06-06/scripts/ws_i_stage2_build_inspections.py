"""WS-I Stage 2 final piece: build inspection input doc.

For the 30 worst-disagreement + 2 maxing contracts, extract:
- BCCC labels (all 10 vuln classes + NV)
- Slither findings (grouped by detector)
- Contract source code (read from BCCC path)
- BCCC folder origin (which of 12 BCCC folders it's in)

Output: ws_i_inspections_input.md — single file the user can scroll through
and decide label-change recommendations.
"""
import sys
import json
import pandas as pd
from pathlib import Path

BCCC_ROOT = Path("BCCC-SCsVul-2024")
SAMPLE_IN = "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_sample_818.csv"
SLITHER_RESULTS = "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_slither_results.csv"
WORST_30_CSV = "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/outputs/ws_i_worst_30_for_review.csv"
OUT_MD = "Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase3_DeepAnalysis_2026-06-06/labels/ws_i_inspections_input.md"

CLASS_COLS = [f"Class{i:02d}:{name}" for i, name in [
    (1, "ExternalBug"), (2, "GasException"), (3, "MishandledException"),
    (4, "Timestamp"), (5, "BlockNumDep"), (6, "UnusedReturn"),
    (7, "Authorization"), (8, "CallToUnknown"), (9, "DenialOfService"),
    (10, "IntegerUO"), (11, "Reentrancy"), (12, "NonVulnerable"),
] if i not in (5, 7, 12)] + ["Class12:NonVulnerable"]


def find_bccc_path(contract_id, bccc_path_fixed):
    """Use the bccc_path_fixed column from the sample CSV (already resolved)."""
    if bccc_path_fixed and isinstance(bccc_path_fixed, str) and bccc_path_fixed != "nan":
        candidate = Path(bccc_path_fixed)
        if candidate.exists():
            return candidate
    # Fallback: scan all BCCC folders
    for sub in BCCC_ROOT.iterdir():
        if not sub.is_dir():
            continue
        candidate = sub / f"{contract_id}.sol"
        if candidate.exists():
            return candidate
    return None


def read_source(path, max_lines=200):
    """Read contract source, truncate if huge."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"<ERROR reading file: {e}>"
    lines = text.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n\n... [truncated, {len(lines) - max_lines} more lines]"
    return text


def get_all_bccc_paths(contract_id):
    """Return list of (folder, full_path) for a contract that lives in multiple BCCC folders."""
    paths = []
    source_root = BCCC_ROOT / "SourceCodes"
    for sub in source_root.iterdir():
        if not sub.is_dir():
            continue
        candidate = sub / f"{contract_id}.sol"
        if candidate.exists():
            paths.append((sub.name, candidate))
    return paths


def group_findings_by_detector(slither_hits_str):
    """Parse slither_hits (list of detector names) and count each."""
    if not slither_hits_str or not isinstance(slither_hits_str, str):
        return {}
    try:
        hits = json.loads(slither_hits_str)
    except Exception:
        return {}
    counts = {}
    for h in hits:
        counts[h] = counts.get(h, 0) + 1
    return counts


def main():
    Path(OUT_MD).parent.mkdir(parents=True, exist_ok=True)

    # Load sample + results
    sample = pd.read_csv(SAMPLE_IN)
    results = pd.read_csv(SLITHER_RESULTS)
    for col in ["slither_status", "slither_hits", "slither_elapsed_sec", "slither_solc", "slither_n_detectors"]:
        if col in results.columns:
            sample[col] = sample["id"].map(results.set_index("id")[col].to_dict())

    # Worst 30 (from CSV) + 2 maxing (which are the 2 worst, but explicit for clarity)
    worst_df = pd.read_csv(WORST_30_CSV)
    print(f"Loaded {len(worst_df)} worst-disagreement contracts")
    print(f"  - n_pos range: {worst_df['n_pos'].min()}-{worst_df['n_pos'].max()}")
    print(f"  - sample_reason counts: {dict(worst_df['sample_reason'].value_counts())}")

    # Group by disagreement pattern for easier review
    # Pattern A: high n_pos (>=6) with high slither hits — these are the maxing
    # Pattern B: low n_pos (1-2) with high slither hits — over-zealous BCCC
    # Pattern C: high n_pos (3-5) with low slither hits — slither missing things
    # Pattern D: zero n_pos (NV) with high slither hits — BCCC false-negative

    out = []
    out.append("# WS-I Inspections Input — Manual Review Material\n")
    out.append("**Purpose:** Single file you can scroll through to decide label-change recommendations for the 30 worst-disagreement + 2 maxing contracts.\n")
    out.append("**For each contract, you see:**\n")
    out.append("- BCCC path + folder + pragma\n")
    out.append("- BCCC labels (which classes are marked = 1)\n")
    out.append("- Slither findings grouped by detector\n")
    out.append("- Full source code (truncated to 200 lines if huge)\n")
    out.append("\n**At the bottom of each entry is a 'Decision' line with 4 checkboxes:**\n")
    out.append("- `[ ] KEEP` — BCCC labels are correct, leave as-is\n")
    out.append("- `[ ] MODIFY: <reason>` — recommend label change (add/remove class, add to dropped set, etc.)\n")
    out.append("- `[ ] REVIEW-NEEDED` — ambiguous, need Aderyn or human second opinion\n")
    out.append("- `[ ] FALSE-POSITIVE-CONTRACT` — entire contract is templated/junk\n")
    out.append("\n---\n")

    # Sort worst_df: maxing first, then by disagreement_score
    maxing = worst_df[worst_df["sample_reason"] == "nine_folder_maxing"].copy()
    non_maxing = worst_df[worst_df["sample_reason"] != "nine_folder_maxing"].copy()
    non_maxing = non_maxing.sort_values("disagreement_score", ascending=False)

    ordered = pd.concat([maxing, non_maxing], ignore_index=True)
    print(f"  - {len(maxing)} maxing + {len(non_maxing)} non-maxing = {len(ordered)} total to inspect")

    for idx, row in ordered.iterrows():
        contract_id = row["id"]
        bccc_path_fixed = row.get("bccc_path_fixed", "")
        n = idx + 1
        n_pos = row["n_pos"]
        reason = row["sample_reason"]
        primary = row.get("primary_class", "?")
        bccc_classes = row.get("bccc_classes", "")
        pragma = row.get("pragma", "")
        sli_status = row.get("slither_status", "?")
        sli_hits = row.get("slither_hits", "")
        score = row.get("disagreement_score", 0)

        out.append(f"\n## Contract {n}/{len(ordered)}: `{contract_id[:24]}`\n")
        out.append(f"**Sample bucket:** {reason}  |  **Primary class:** {primary}  |  **n_pos:** {n_pos}  |  **Pragma:** `{pragma}`  |  **Disagreement score:** {score:.3f}\n")

        # Find BCCC path
        path = find_bccc_path(contract_id, bccc_path_fixed)
        all_paths = get_all_bccc_paths(contract_id)
        folders = [f for f, _ in all_paths]

        if path:
            out.append(f"**BCCC folders containing this contract:** {folders}\n")
            rel = path.relative_to(BCCC_ROOT.parent) if BCCC_ROOT.parent in path.parents else path
            out.append(f"**Showing source from:** `{rel}`\n")
            source = read_source(path, max_lines=200)
            out.append(f"\n### Source code:\n```solidity\n{source}\n```\n")
        else:
            out.append(f"**BCCC folders containing this contract:** {folders} (all not found on disk!)\n")
            out.append(f"\n### Source code: <missing>\n")

        # BCCC labels
        out.append(f"### BCCC labels ({n_pos} positive):\n")
        if bccc_classes:
            out.append(f"- {bccc_classes}\n")
        else:
            out.append(f"- (none positive — all zero, contract is in BCCC folder by class but not labeled in CSV?)\n")

        # Slither findings
        out.append(f"\n### Slither findings ({sli_status}):\n")
        if sli_status != "OK":
            out.append(f"- Slither status: **{sli_status}** (findings not available)\n")
        else:
            counts = group_findings_by_detector(sli_hits)
            if not counts:
                out.append(f"- **No slither findings** (rare for this corpus!)\n")
            else:
                out.append(f"- **Total findings:** {sum(counts.values())}  |  **Unique detectors:** {len(counts)}\n")
                out.append(f"- **Top detectors (by frequency):**\n")
                for det, ct in sorted(counts.items(), key=lambda x: -x[1])[:15]:
                    out.append(f"  - `{det}` × {ct}\n")
                if len(counts) > 15:
                    out.append(f"  - ... and {len(counts) - 15} more detectors\n")

        # Decision line
        out.append(f"\n### Decision:\n")
        out.append(f"- [ ] **KEEP** — BCCC labels are correct, no change needed\n")
        out.append(f"- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>\n")
        out.append(f"- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion\n")
        out.append(f"- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)\n")
        out.append(f"\n---\n")

        if (n) % 10 == 0:
            print(f"  Wrote {n}/{len(ordered)} contracts...")

    text = "\n".join(out)
    Path(OUT_MD).write_text(text, encoding="utf-8")
    print(f"\n✅ Wrote {OUT_MD}")
    print(f"   {len(text)} bytes, {len(text.splitlines())} lines")
    print(f"\nNext: read the doc, fill in the Decision checkboxes, save as ws_i_disagreement_inspections.md")
    print(f"Then summarize label-change recommendations for v1.2 of contracts_clean.csv")


if __name__ == "__main__":
    main()
