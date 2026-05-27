"""
Task 18: Solidity Version Distribution and Feature Implications
---------------------------------------------------------------
Samples 2000 .sol files from BCCC sources, extracts pragma version,
buckets into 0.4.x / 0.5.x / ... / 0.8.x, and cross-tabulates with labels.

Run:
    python task_18_solidity_version.py
"""
import re
import random
from pathlib import Path
from collections import defaultdict
from common import get_dirs, load_csv, load_graph, sol_from_graph, print_header

N = 2000
PRAGMA_RE = re.compile(
    r'pragma\s+solidity\s+[^0-9]*(\d+)\.(\d+)\.?(\d*)', re.IGNORECASE
)

def extract_version(source: str):
    """Returns (major, minor) or None."""
    m = PRAGMA_RE.search(source)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def main():
    print_header(18, "Solidity Version Distribution and Feature Implications")
    _, _, graphs_dir, _, _, bccc_dir = get_dirs()

    df = load_csv()
    all_stems = df["md5_stem"].astype(str).tolist()

    if bccc_dir is None:
        print("[ERROR] BCCC directory not found.")
        return

    random.seed(42)
    sample = random.sample(all_stems, min(N, len(all_stems)))

    # Version bucket → list of stem records
    buckets = defaultdict(list)
    label_cols = [c for c in df.columns if c != "md5_stem"]
    label_by_stem = df.set_index("md5_stem")

    processed = 0
    for stem in sample:
        gpath = graphs_dir / f"{stem}.pt"
        if not gpath.exists():
            continue
        try:
            g = load_graph(gpath)
        except Exception:
            continue
        sol = sol_from_graph(g, bccc_dir)
        if sol is None:
            continue
        source = sol.read_text(errors="replace")
        ver = extract_version(source)
        if ver is None:
            bucket = "no_pragma"
        else:
            major, minor = ver
            if major == 0:
                if minor == 4: bucket = "0.4.x"
                elif minor == 5: bucket = "0.5.x"
                elif minor == 6: bucket = "0.6.x"
                elif minor == 7: bucket = "0.7.x"
                elif minor == 8: bucket = "0.8.x"
                else: bucket = f"0.{minor}.x"
            else:
                bucket = f"{major}.x"

        if stem in label_by_stem.index:
            labels = label_by_stem.loc[stem, label_cols].to_dict()
        else:
            labels = {c: 0 for c in label_cols}

        buckets[bucket].append({"stem": stem, "source": source, **labels})
        processed += 1

    print(f"\nProcessed {processed} files.\n")

    # ── Version distribution table ────────────────────────────────────────────
    total = sum(len(v) for v in buckets.values())
    print(f"{'Bucket':<12} {'Count':>7} {'%':>6}")
    print("-" * 28)
    for bucket in sorted(buckets.keys()):
        n = len(buckets[bucket])
        print(f"  {bucket:<10} {n:>7,}  {n/total*100:>5.1f}%")
    print(f"  {'TOTAL':<10} {total:>7,}")

    # ── Per-bucket label distribution ─────────────────────────────────────────
    print(f"\nPer-version positive rate (%) per label:")
    header = f"{'Bucket':<10}" + "".join(f" {c[:6]:>7}" for c in label_cols)
    print(header)
    print("-" * (10 + 8 * len(label_cols)))
    for bucket in sorted(buckets.keys()):
        recs = buckets[bucket]
        n    = len(recs)
        if n == 0:
            continue
        rates = []
        for c in label_cols:
            pos = sum(r[c] for r in recs)
            rates.append(f"{pos/n*100:>7.1f}")
        print(f"  {bucket:<10}" + "".join(rates))

    # ── 0.8.x special analysis ───────────────────────────────────────────────
    v08 = buckets.get("0.8.x", [])
    if v08:
        print(f"\n0.8.x special analysis (n={len(v08)}):")
        unchecked_pat = re.compile(r'\bunchecked\s*\{')
        safemath_pat  = re.compile(r'\bSafeMath\b')
        n_unchecked   = sum(1 for r in v08 if unchecked_pat.search(r["source"]))
        n_safemath    = sum(1 for r in v08 if safemath_pat.search(r["source"]))
        n_intUO       = sum(r.get("IntegerUO", 0) for r in v08)
        print(f"  With unchecked{{}} blocks: {n_unchecked} ({n_unchecked/len(v08)*100:.1f}%)")
        print(f"  Still using SafeMath:     {n_safemath} ({n_safemath/len(v08)*100:.1f}%)")
        print(f"  IntegerUO=1:              {n_intUO}    ({n_intUO/len(v08)*100:.1f}%)")
    else:
        print("\n0.8.x: No contracts found in sample.")

    # ── Finding ───────────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────────────")
    v04_pct = len(buckets.get("0.4.x", [])) / total * 100
    v08_pct = len(buckets.get("0.8.x", [])) / total * 100
    print(f"  0.4.x: {v04_pct:.1f}% of sample")
    print(f"  0.8.x: {v08_pct:.1f}% of sample")
    if v08_pct < 1.0:
        print("  [CONFIRMED] BUG-5 (in_unchecked dead): 0.8.x is <1% — "
              "feature provides essentially zero signal.")
    else:
        print(f"  [NEW FINDING] 0.8.x is {v08_pct:.1f}% — larger than expected. "
              "Reconsider BUG-5 severity.")

if __name__ == "__main__":
    main()
