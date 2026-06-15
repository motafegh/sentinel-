"""v3-aware contamination audit for SENTINEL benchmark suites.

Replaces the legacy ml/scripts/check_contamination.py (which only checks SmartBugs
vs BCCC, using v9/v10 paths). This version:

  1. Builds the index of v3 training/val/test contracts by SHA-256 from
     data_module/data/splits/v3/{train,val,test}.jsonl
  2. For each benchmark contract (SmartBugs Curated + SolidiFI-benchmark):
     - Tier 1: exact SHA-256 match in v3 splits
     - Tier 2: normalised (comments stripped, whitespace collapsed, lowercased) SHA-256
     - Tier 3: token Jaccard >= 0.75 vs top-50 v3 candidates (length-filtered)
  3. Reports: per-benchmark overlap, per-category overlap, per-split breakdown
  4. Emits a JSON report: /tmp/contamination_v3_<date>.json
  5. Exits 0 if overlap is acceptable per user-defined policy; non-zero otherwise

Usage (from repo root, with venv active):
    ml/.venv/bin/python /mnt/c/Users/lenovo/AppData/Local/Temp/opencode/check_contamination_v3.py
    ml/.venv/bin/python /mnt/c/Users/lenovo/AppData/Local/Temp/opencode/check_contamination_v3.py --strict
    ml/.venv/bin/python /mnt/c/Users/lenovo/AppData/Local/Temp/opencode/check_contamination_v3.py --jaccard-threshold 0.80

Output: console report + /tmp/contamination_v3_<date>.json
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
V3_SPLITS_DIR = REPO_ROOT / "data_module" / "data" / "splits" / "v3"
SMARTBUGS_DIR = REPO_ROOT / "ml" / "data" / "smartbugs-curated" / "dataset"
SOLIDIFI_BENCH_DIR = REPO_ROOT / "ml" / "data" / "SolidiFI-benchmark" / "buggy_contracts"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE = re.compile(r"//[^\n]*")
_WHITESPACE = re.compile(r"\s+")


def normalise(src: str) -> str:
    s = _COMMENT_BLOCK.sub(" ", src)
    s = _COMMENT_LINE.sub(" ", s)
    s = _WHITESPACE.sub(" ", s)
    return s.strip().lower()


def token_set(src: str) -> set:
    return set(re.split(r"[^a-zA-Z0-9_]+", src)) - {""}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def build_v3_index() -> dict:
    """Return {sha256 -> {"split": ..., "source": ...}} for all v3 contracts."""
    index = {}
    for split_file in sorted(V3_SPLITS_DIR.glob("*.jsonl")):
        split_name = split_file.stem
        for line in split_file.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            sha = d.get("sha256", "")
            if sha:
                index[sha] = {
                    "split": split_name,
                    "source": d.get("source", "unknown"),
                    "primary_class": d.get("primary_class", ""),
                }
    return index


def collect_smartbugs() -> list:
    contracts = []
    for cat_dir in sorted(SMARTBUGS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for sol in sorted(cat_dir.glob("*.sol")):
            raw = sol.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            contracts.append({
                "benchmark": "smartbugs",
                "category": cat_dir.name,
                "name": sol.name,
                "h_raw": sha256_bytes(raw),
                "h_norm": sha256_bytes(normalise(text).encode()),
                "toks": token_set(normalise(text)),
                "len": len(text),
            })
    return contracts


def collect_solidifi() -> list:
    contracts = []
    if not SOLIDIFI_BENCH_DIR.exists():
        return contracts
    for cat_dir in sorted(SOLIDIFI_BENCH_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for sol in sorted(cat_dir.glob("*.sol")):
            raw = sol.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            contracts.append({
                "benchmark": "solidifi",
                "category": cat_dir.name,
                "name": sol.name,
                "h_raw": sha256_bytes(raw),
                "h_norm": sha256_bytes(normalise(text).encode()),
                "toks": token_set(normalise(text)),
                "len": len(text),
            })
    return contracts


def audit(benchmark_contracts: list, v3_index: dict, v3_h_norm_index: dict,
          v3_entries: list, jaccard_threshold: float) -> dict:
    """For each benchmark contract, run tiers 1-3."""
    results = []
    for c in benchmark_contracts:
        r = {
            "benchmark": c["benchmark"],
            "category": c["category"],
            "name": c["name"],
            "tier1_exact": None,  # (split, source) if found
            "tier2_normalised": None,
            "tier3_jaccard_max": 0.0,
            "tier3_jaccard_top": None,
        }
        if c["h_raw"] in v3_index:
            v = v3_index[c["h_raw"]]
            r["tier1_exact"] = (v["split"], v["source"])
        if c["h_norm"] in v3_h_norm_index:
            v3_idx_list = v3_h_norm_index[c["h_norm"]]
            r["tier2_normalised"] = [v3_index[s]["split"] for s in v3_idx_list if s in v3_index]
        # Tier 3: top-50 candidates by length proximity
        candidates = sorted(v3_entries, key=lambda e: abs(e["len"] - c["len"]))[:50]
        best = 0.0
        best_match = None
        for cand_sha, cand_toks, cand_len, cand_split, cand_source in candidates:
            j = jaccard(c["toks"], cand_toks)
            if j > best:
                best = j
                best_match = (cand_sha[:12], cand_split, cand_source, j)
        r["tier3_jaccard_max"] = best
        r["tier3_jaccard_top"] = best_match
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jaccard-threshold", type=float, default=0.75)
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if ANY contamination found")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("SENTINEL v3-Aware Contamination Audit (SmartBugs + SolidiFI)")
    print(f"{'='*70}\n")

    print("Building v3 training index...")
    v3_index = build_v3_index()
    print(f"  v3 contracts indexed: {len(v3_index)}")
    v3_h_norm_index = defaultdict(list)
    v3_entries = []
    for split_file in sorted(V3_SPLITS_DIR.glob("*.jsonl")):
        split_name = split_file.stem
        for line in split_file.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            sha = d.get("sha256", "")
            if not sha:
                continue
            text = d.get("source_text", "") or d.get("raw_source", "")
            if text:
                norm = normalise(text)
                h_norm = sha256_bytes(norm.encode())
                v3_h_norm_index[h_norm].append(sha)
                v3_entries.append((sha, token_set(norm), len(text), split_name, d.get("source", "")))
    print(f"  v3 normalised entries: {len(v3_h_norm_index)}")

    print("\nLoading benchmark contracts...")
    sb = collect_smartbugs()
    sf = collect_solidifi()
    print(f"  SmartBugs: {len(sb)}")
    print(f"  SolidiFI-benchmark: {len(sf)}")

    print("\nAuditing SmartBugs vs v3...")
    sb_results = audit(sb, v3_index, v3_h_norm_index, v3_entries, args.jaccard_threshold)
    print("Auditing SolidiFI vs v3...")
    sf_results = audit(sf, v3_index, v3_h_norm_index, v3_entries, args.jaccard_threshold)

    # Per-benchmark summary
    def summarise(results, name):
        n = len(results)
        t1 = sum(1 for r in results if r["tier1_exact"] is not None)
        t2 = sum(1 for r in results if r["tier2_normalised"] is not None)
        t3 = sum(1 for r in results if r["tier3_jaccard_max"] >= args.jaccard_threshold)
        print(f"\n{'='*70}")
        print(f"{name}: {n} contracts total")
        print(f"  Tier 1 (exact SHA-256):     {t1} in v3 ({100*t1/n:.1f}%)")
        print(f"  Tier 2 (normalised SHA-256): {t2} in v3 ({100*t2/n:.1f}%)")
        print(f"  Tier 3 (Jaccard >= {args.jaccard_threshold}):     {t3} near-dup ({100*t3/n:.1f}%)")
        if t1 > 0 or t2 > 0:
            print(f"\n  CONTAMINATION: {name} contracts are IN v3 training/val/test.")
            print(f"  The {name} benchmark F1 is INFLATED (testing on training data, not OOD).")
            print(f"  RECOMMENDATION: use only the {n - t1} non-contaminated contracts for OOD evaluation,")
            print(f"  OR remove {name} from training (requires v3.1 export),")
            print(f"  OR use a different OOD benchmark (e.g., DeFiHackLabs wild contracts).")
        return {"total": n, "tier1": t1, "tier2": t2, "tier3": t3}

    sb_summary = summarise(sb_results, "SmartBugs Curated")
    sf_summary = summarise(sf_results, "SolidiFI-benchmark")

    # Per-category breakdown for SmartBugs
    from collections import Counter
    print(f"\n{'='*70}")
    print("Per-category overlap (Tier 1, exact SHA-256):")
    print(f"{'='*70}")
    cat_overlap_sb = Counter()
    cat_total_sb = Counter()
    for r in sb_results:
        cat_total_sb[r["category"]] += 1
        if r["tier1_exact"] is not None:
            cat_overlap_sb[r["category"]] += 1
    for cat in sorted(cat_total_sb):
        n = cat_total_sb[cat]
        k = cat_overlap_sb[cat]
        print(f"  SmartBugs/{cat:30s}: {k:3d}/{n:3d} in v3 ({100*k/n:.0f}%)")

    cat_overlap_sf = Counter()
    cat_total_sf = Counter()
    for r in sf_results:
        cat_total_sf[r["category"]] += 1
        if r["tier1_exact"] is not None:
            cat_overlap_sf[r["category"]] += 1
    print()
    for cat in sorted(cat_total_sf):
        n = cat_total_sf[cat]
        k = cat_overlap_sf[cat]
        print(f"  SolidiFI-benchmark/{cat:25s}: {k:3d}/{n:3d} in v3 ({100*k/n:.0f}%)")

    # Per-split breakdown
    print(f"\n{'='*70}")
    print("Per-split breakdown of contaminated contracts (Tier 1):")
    print(f"{'='*70}")
    split_count_sb = Counter()
    for r in sb_results:
        if r["tier1_exact"]:
            split_count_sb[r["tier1_exact"][0]] += 1
    print(f"  SmartBugs in: {dict(split_count_sb)}")
    split_count_sf = Counter()
    for r in sf_results:
        if r["tier1_exact"]:
            split_count_sf[r["tier1_exact"][0]] += 1
    print(f"  SolidiFI in: {dict(split_count_sf)}")

    # Write JSON report
    report = {
        "date": datetime.now().isoformat(),
        "v3_contracts_total": len(v3_index),
        "jaccard_threshold": args.jaccard_threshold,
        "smartbugs": {
            "summary": sb_summary,
            "per_category": {cat: {"total": cat_total_sb[cat], "in_v3": cat_overlap_sb[cat]}
                              for cat in cat_total_sb},
            "results": sb_results,
        },
        "solidifi": {
            "summary": sf_summary,
            "per_category": {cat: {"total": cat_total_sf[cat], "in_v3": cat_overlap_sf[cat]}
                              for cat in cat_total_sf},
            "results": sf_results,
        },
    }
    out = Path("/tmp") / f"contamination_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report written to: {out}")

    # Honest-OOD: list the contracts NOT in v3 (the only ones that count as OOD)
    sb_honest = [r for r in sb_results if r["tier1_exact"] is None]
    sf_honest = [r for r in sf_results if r["tier1_exact"] is None]
    print(f"\n{'='*70}")
    print(f"HONEST OOD SUBSET (contracts NOT in v3 training/val/test):")
    print(f"{'='*70}")
    print(f"  SmartBugs honest OOD: {len(sb_honest)} / {len(sb_results)}")
    for r in sb_honest:
        print(f"    {r['category']}/{r['name']}")
    print(f"  SolidiFI honest OOD:  {len(sf_honest)} / {len(sf_results)}")
    if len(sf_honest) <= 20:
        for r in sf_honest:
            print(f"    {r['category']}/{r['name']}")

    # Exit code
    total_contam = sb_summary["tier1"] + sf_summary["tier1"]
    if args.strict and total_contam > 0:
        print(f"\nSTRICT MODE: contamination detected ({total_contam} contracts). Exit 1.")
        sys.exit(1)
    print(f"\nTotal contaminated contracts: {total_contam}")
    print("Use --strict to fail on any contamination. Use the HONEST OOD SUBSET for OOD benchmark reporting.")


if __name__ == "__main__":
    main()
