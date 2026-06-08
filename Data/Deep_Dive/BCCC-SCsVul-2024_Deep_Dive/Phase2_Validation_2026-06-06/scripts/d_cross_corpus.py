"""BCCC Phase 2 — WS-D: Cross-Corpus Overlap (BCCC vs SmartBugs-curated).

Detects byte-identical contracts that appear in BOTH corpora. If a contract is in both,
we need a conflict-resolution rule (e.g., drop from OOD test set, or relabel).

Inputs:
  - ../integrity/sha256_all_files.tsv (BCCC sha256s)
  - ../../../../ml/data/smartbugs-curated/dataset/ (SmartBugs .sol files)

Outputs (under ../cross_corpus/):
  - smartbugs_sha256.tsv           (SmartBugs sha256s)
  - bccc_vs_smartbugs_overlap.csv  (shared contracts with both labels)
  - overlap_report.md              (analysis)
"""
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

INTEG = Path(__file__).resolve().parent.parent / "integrity"
BCCC_SHA_TSV = INTEG / "sha256_all_files.tsv"
SMARTBUGS_ROOT = Path("ml/data/smartbugs-curated")
SMARTBUGS_DATASET = SMARTBUGS_ROOT / "dataset"
SMARTBUGS_VULN_JSON = SMARTBUGS_ROOT / "vulnerabilities.json"
OUT = Path(__file__).resolve().parent.parent / "cross_corpus"
OUT.mkdir(parents=True, exist_ok=True)


# Mapping from SmartBugs vuln category to closest BCCC class
SMARTBUGS_TO_BCCC = {
    "access_control":              "Class07:WeakAccessMod",
    "arithmetic":                  "Class10:IntegerUO",
    "bad_randomness":              "Class04:Timestamp",  # closest (uses block vars)
    "denial_of_service":           "Class09:DenialOfService",
    "front_running":               "Class05:TransactionOrderDependence",
    "other":                       "",  # no good BCCC match
    "reentrancy":                  "Class11:Reentrancy",
    "short_addresses":             "",  # no good BCCC match
    "time_manipulation":           "Class04:Timestamp",
    "unchecked_low_level_calls":   "Class06:UnusedReturn",  # or CallToUnknown
}


def hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("WS-D: Cross-Corpus Overlap (BCCC vs SmartBugs-curated)")
    print("=" * 70)

    # 1. Load BCCC sha256s
    print("\n[1/4] Loading BCCC sha256s...")
    bccc_by_sha: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with BCCC_SHA_TSV.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            bccc_by_sha[row["sha256"]].append((row["folder"], row["filename"]))
    print(f"  BCCC unique sha256s: {len(bccc_by_sha)}")
    print(f"  BCCC total files: {sum(len(v) for v in bccc_by_sha.values())}")

    # 2. Hash SmartBugs files
    print("\n[2/4] Hashing SmartBugs-curated .sol files...")
    sb_files: list[tuple[Path, str, str]] = []
    for p in sorted(SMARTBUGS_DATASET.glob("**/*.sol")):
        sha = hash_file(p)
        # Extract category from relative path
        rel = p.relative_to(SMARTBUGS_DATASET)
        category = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        sb_files.append((p, sha, category))
    print(f"  SmartBugs .sol files: {len(sb_files)}")
    print(f"  Unique SmartBugs sha256s: {len(set(s for _, s, _ in sb_files))}")

    # Write SmartBugs sha256 list
    sb_tsv = OUT / "smartbugs_sha256.tsv"
    with sb_tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["smartbugs_path", "category", "sha256"])
        for p, sha, cat in sb_files:
            w.writerow([str(p), cat, sha])
    print(f"  Wrote {sb_tsv}")

    # 3. Compute overlap
    print("\n[3/4] Computing overlap...")
    sb_shas = set(s for _, s, _ in sb_files)
    bccc_shas = set(bccc_by_sha.keys())
    overlap = sb_shas & bccc_shas
    print(f"  Overlap (sha256 intersection): {len(overlap)} unique contracts")
    overlap_pct_sb = 100 * len(overlap) / len(sb_shas) if sb_shas else 0
    print(f"  % of SmartBugs contracts also in BCCC: {overlap_pct_sb:.1f}%")

    # 4. For each overlap contract, list (SmartBugs path+category, BCCC folders)
    print("\n[4/4] Building overlap detail...")
    rows = []
    for p, sha, cat in sb_files:
        if sha in overlap:
            bccc_locs = bccc_by_sha[sha]
            bccc_folders = sorted(set(f for f, _ in bccc_locs))
            bccc_ids = sorted(set(fn for _, fn in bccc_locs))
            rows.append({
                "sha256": sha,
                "smartbugs_path": str(p),
                "smartbugs_category": cat,
                "smartbugs_to_bccc_mapping": SMARTBUGS_TO_BCCC.get(cat, ""),
                "bccc_folders": ";".join(bccc_folders),
                "bccc_canonical_ids": ";".join(bccc_ids),
            })
    print(f"  Overlap rows: {len(rows)}")

    overlap_csv = OUT / "bccc_vs_smartbugs_overlap.csv"
    with overlap_csv.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        else:
            w = csv.writer(f)
            w.writerow(["sha256", "smartbugs_path", "smartbugs_category", "smartbugs_to_bccc_mapping", "bccc_folders", "bccc_canonical_ids"])
            w.writerow(["(no overlap)", "", "", "", "", ""])
    print(f"  Wrote {overlap_csv}")

    # Load SmartBugs vulnerabilities.json for label context
    vuln_data = []
    if SMARTBUGS_VULN_JSON.exists():
        with SMARTBUGS_VULN_JSON.open() as f:
            vuln_data = json.load(f)
    sb_vuln_map: dict[str, list[str]] = defaultdict(list)
    for v in vuln_data:
        if v.get("path"):
            sb_vuln_map[v["path"]] = [c["category"] for c in v.get("vulnerabilities", [])]

    # Distribution by SmartBugs category
    cat_dist: dict[str, int] = defaultdict(int)
    for r in rows:
        cat_dist[r["smartbugs_category"]] += 1
    print(f"\n  Overlap by SmartBugs category:")
    for cat, n in sorted(cat_dist.items(), key=lambda x: -x[1]):
        bccc_class = SMARTBUGS_TO_BCCC.get(cat, "(no good BCCC mapping)")
        print(f"    {cat:30s} n={n:>3d}  →  BCCC: {bccc_class}")

    # Build overlap report
    lines = [
        "# WS-D: Cross-Corpus Overlap (BCCC vs SmartBugs-curated) — Report",
        "",
        "**Date:** 2026-06-06",
        "**Status:** Complete",
        "",
        "## Summary",
        "",
        f"- **BCCC unique contracts:** {len(bccc_shas):,}",
        f"- **SmartBugs-curated contracts:** {len(sb_files):,} ({len(sb_shas):,} unique by sha256)",
        f"- **Byte-identical overlap:** {len(overlap):,} unique contracts ({overlap_pct_sb:.1f}% of SmartBugs)",
        "",
        "## Method",
        "",
        "1. Compute SHA-256 of every BCCC `.sol` file (111,897 files) and every SmartBugs-curated `.sol` file (143 files).",
        "2. Set-intersect the SHA-256 hashes. A contract is in the overlap iff its byte content is identical in both corpora.",
        "3. For each overlap, list both labels and compute a category mapping.",
        "",
        "## Class Mapping (SmartBugs → Closest BCCC class)",
        "",
        "| SmartBugs category | Closest BCCC class | Notes |",
        "|---|---|---|",
    ]
    for sb_cat, bccc_cls in SMARTBUGS_TO_BCCC.items():
        lines.append(f"| `{sb_cat}` | `{bccc_cls}` | " + (
            "Exact match" if bccc_cls and sb_cat == bccc_cls.split(":")[1].lower() else
            "Best approximation" if bccc_cls else
            "**No good BCCC match** — would need a v2 schema"
        ))

    lines += [
        "",
        "## Overlap by SmartBugs Category",
        "",
        "| SmartBugs category | n overlap | Closest BCCC class |",
        "|---|---:|---|",
    ]
    for cat, n in sorted(cat_dist.items(), key=lambda x: -x[1]):
        bccc_cls = SMARTBUGS_TO_BCCC.get(cat, "(no good match)")
        lines.append(f"| `{cat}` | {n} | `{bccc_cls}` |")

    lines += [
        "",
        "## Decision Required: How to Handle Overlap",
        "",
        f"**{len(overlap)} contracts are in both BCCC and SmartBugs-curated.** This is a data leak risk if we use SmartBugs as an OOD test set (per ADR-0005).",
        "",
        "Options:",
        "",
        f"1. **Drop overlap from SmartBugs OOD** — {len(overlap)} contracts removed; remaining SmartBugs serves as a clean OOD test set.",
        "2. **Relabel overlap to BCCC's labels** — use BCCC's 12-class label as canonical; SmartBugs label discarded. Risk: BCCC and SmartBugs may have different annotation conventions, so a 1:1 relabel is not always correct.",
        "3. **Keep overlap in SmartBugs OOD, ignore in metrics** — measure test F1 on overlap contracts but don't count them in headline numbers.",
        "",
        f"**Recommendation: (1) Drop from SmartBugs OOD** — simplest, safest, and respects the OOD premise. After drop, SmartBugs has {len(sb_files) - len(overlap)} contracts for OOD evaluation.",
        "",
    ]

    if len(overlap) == 0:
        lines += [
            "**No overlap detected.** SmartBugs-curated is a fully disjoint corpus from BCCC (good for OOD).",
            "",
        ]
    else:
        lines += [
            "## Sample Overlap Contracts",
            "",
            "| sha256 (first 16) | SmartBugs path | SB category | BCCC folders | BCCC canonical ID |",
            "|---|---|---|---|---|",
        ]
        for r in rows[:15]:
            lines.append(
                f"| `{r['sha256'][:16]}…` | `{r['smartbugs_path']}` | "
                f"`{r['smartbugs_category']}` | `{r['bccc_folders']}` | "
                f"`{r['bccc_canonical_ids'][:60]}{'…' if len(r['bccc_canonical_ids']) > 60 else ''}` |"
            )

    lines += [
        "",
        "## Files",
        "",
        "- `smartbugs_sha256.tsv` — 143 SmartBugs file hashes",
        "- `bccc_vs_smartbugs_overlap.csv` — overlap detail (one row per SmartBugs file in overlap)",
        "- `overlap_report.md` — this file",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "cd /home/motafeq/projects/sentinel",
        "source ml/.venv/bin/activate",
        "python Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/scripts/d_cross_corpus.py",
        "```",
        "",
    ]
    (OUT / "overlap_report.md").write_text("\n".join(lines))
    print(f"  Wrote {OUT / 'overlap_report.md'}")


if __name__ == "__main__":
    main()
