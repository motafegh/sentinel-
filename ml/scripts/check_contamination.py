"""
check_contamination.py — Data contamination audit: SmartBugs vs BCCC training corpus

Checks whether SmartBugs Curated contracts were seen during Run 9 training.

Three detection tiers (in order of increasing cost):

  Tier 1 — Exact content hash
    SHA256 of raw source bytes. Instant lookup once the BCCC hash index is built.
    Also checks if the BCCC filename IS the content SHA256 (common pattern when
    corpus was de-duped by content hash at collection time).

  Tier 2 — Normalised content hash
    Strip single-line (//) and block (/* */) comments, collapse all whitespace to
    single spaces, lowercase. Recompute SHA256. Catches reformatted / re-annotated
    copies of the same contract.

  Tier 3 — Token Jaccard similarity
    Split on non-alphanumeric characters, compute word-level Jaccard.
    Only run against the top-K BCCC candidates selected by normalised-length
    proximity to avoid O(143 × 111K) full scan.
    Reports any pair with Jaccard ≥ 0.75 as a near-duplicate.

  Tier 4 — Structural graph fingerprint (training graphs only)
    For contracts that survived Slither extraction: compare (num_nodes, num_edges,
    sorted function-name list from node_metadata) against all 41,576 training .pt
    files. Only the training split matters for contamination; val/test are also
    flagged for completeness.

Usage (from repo root, venv active):
    python -m ml.scripts.check_contamination [--jaccard-threshold 0.75]
                                             [--top-k-candidates 50]
                                             [--no-tier4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SMARTBUGS_DIR = REPO_ROOT / "ml" / "data" / "smartbugs-curated" / "dataset"
BCCC_DIR      = REPO_ROOT / "BCCC-SCsVul-2024" / "SourceCodes"
GRAPHS_DIR    = REPO_ROOT / "ml" / "data" / "graphs"
SPLITS_DIR    = REPO_ROOT / "ml" / "data" / "splits" / "deduped"

# ── helpers ───────────────────────────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def md5_content(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

_COMMENT_BLOCK = re.compile(r'/\*.*?\*/', re.DOTALL)
_COMMENT_LINE  = re.compile(r'//[^\n]*')
_WHITESPACE    = re.compile(r'\s+')

def normalise(src: str) -> str:
    """Strip comments and collapse whitespace — order matters."""
    s = _COMMENT_BLOCK.sub(' ', src)
    s = _COMMENT_LINE.sub(' ', s)
    s = _WHITESPACE.sub(' ', s)
    return s.strip().lower()

def token_set(src: str) -> set[str]:
    """Split on non-alphanumeric (keeps hex literals, identifiers)."""
    return set(re.split(r'[^a-zA-Z0-9_]+', src)) - {''}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


# ── index builders ────────────────────────────────────────────────────────────

def build_bccc_index(verbose: bool) -> tuple[dict, dict, list]:
    """
    Scan all 111,897 BCCC .sol files and build:
      exact_index  : {sha256_of_content → [path, ...]}
      norm_index   : {sha256_of_normalised_content → [path, ...]}
      bccc_entries : list of (path, sha256_raw, sha256_norm, token_set, len_chars)
                     — full list for Jaccard and filename checks
    """
    print("Building BCCC content index (111,897 files × ~5 KB = ~560 MB reads)...")
    exact_index: dict[str, list[Path]] = defaultdict(list)
    norm_index:  dict[str, list[Path]] = defaultdict(list)
    bccc_entries: list[tuple] = []

    sol_files = list(BCCC_DIR.rglob("*.sol"))
    n = len(sol_files)
    for i, p in enumerate(sol_files, 1):
        if i % 10000 == 0:
            print(f"  {i}/{n} files indexed...")
        try:
            raw = p.read_bytes()
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue

        h_raw  = sha256_bytes(raw)
        h_norm = sha256_bytes(normalise(text).encode())
        toks   = token_set(normalise(text))

        exact_index[h_raw].append(p)
        norm_index[h_norm].append(p)
        bccc_entries.append((p, h_raw, h_norm, toks, len(text)))

    print(f"  Done. {len(exact_index)} unique raw hashes, {len(norm_index)} normalised.\n")
    return dict(exact_index), dict(norm_index), bccc_entries


def load_training_split_ids() -> dict[str, str]:
    """Return {md5_stem → split} for all training .pt files."""
    split_map: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        f = SPLITS_DIR / f"{split_name}_indices.npy"
        if not f.exists():
            continue
        import numpy as np
        indices = np.load(str(f))
        # indices are positional — map them via the full sorted graph list
        all_graphs = sorted(GRAPHS_DIR.glob("*.pt"))
        for idx in indices:
            if idx < len(all_graphs):
                split_map[all_graphs[idx].stem] = split_name
    return split_map


def build_graph_fingerprint_index(
    split_map: dict[str, str]
) -> dict[tuple, list[tuple]]:
    """
    Build {(num_nodes, num_edges) → [(md5_stem, split, fn_names), ...]}
    Loading all 41K .pt files but only reading 3 attributes per graph.
    """
    import torch
    print("Building graph structural fingerprint index (41,576 .pt files)...")
    fp_index: dict[tuple, list] = defaultdict(list)
    all_graphs = sorted(GRAPHS_DIR.glob("*.pt"))
    for i, gp in enumerate(all_graphs, 1):
        if i % 5000 == 0:
            print(f"  {i}/{len(all_graphs)}...")
        try:
            g = torch.load(str(gp), weights_only=False)
            nn_ = int(g.num_nodes)
            ne_ = int(g.num_edges)
            # collect function names from node_metadata
            fn_names = []
            if hasattr(g, "node_metadata") and g.node_metadata:
                fn_names = sorted(
                    m.get("name", "") for m in g.node_metadata
                    if m.get("type") in ("FUNCTION", "MODIFIER", "FALLBACK",
                                         "RECEIVE", "CONSTRUCTOR")
                )
            fp_index[(nn_, ne_)].append(
                (gp.stem, split_map.get(gp.stem, "unknown"), fn_names)
            )
        except Exception:
            pass
    print(f"  Done. {len(fp_index)} unique (num_nodes, num_edges) fingerprints.\n")
    return dict(fp_index)


# ── main ──────────────────────────────────────────────────────────────────────

def run_check(jaccard_threshold: float, top_k: int, skip_tier4: bool) -> None:
    sys.path.insert(0, str(REPO_ROOT))

    print(f"\n{'='*70}")
    print("SENTINEL — SmartBugs × BCCC Contamination Audit")
    print(f"{'='*70}")
    print(f"  Jaccard threshold : {jaccard_threshold}")
    print(f"  Top-K for Jaccard : {top_k}")
    print()

    # ── collect SmartBugs ────────────────────────────────────────────────────
    sb_contracts = []
    for cat_dir in sorted(SMARTBUGS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for sol in sorted(cat_dir.glob("*.sol")):
            try:
                raw  = sol.read_bytes()
                text = raw.decode("utf-8", errors="replace")
                sb_contracts.append({
                    "path":     sol,
                    "category": cat_dir.name,
                    "raw":      raw,
                    "text":     text,
                    "h_raw":    sha256_bytes(raw),
                    "h_norm":   sha256_bytes(normalise(text).encode()),
                    "toks":     token_set(normalise(text)),
                    "len":      len(text),
                })
            except Exception as exc:
                print(f"  WARNING: could not read {sol.name}: {exc}")

    print(f"SmartBugs contracts loaded: {len(sb_contracts)}\n")

    # ── build BCCC index ─────────────────────────────────────────────────────
    exact_index, norm_index, bccc_entries = build_bccc_index(verbose=True)

    # Also build a set of BCCC filenames (stems) — they may BE the content SHA256
    bccc_filename_sha256_set = {p.stem for p, *_ in bccc_entries}
    print(f"BCCC filename stems sampled (first 3): {list(bccc_filename_sha256_set)[:3]}\n")

    # ── run tiers per SmartBugs contract ─────────────────────────────────────
    results: list[dict] = []

    print(f"{'='*70}")
    print("TIER 1+2 — Exact & Normalised Content Hash")
    print(f"{'='*70}")

    exact_hits = 0
    norm_hits  = 0
    filename_hits = 0

    for c in sb_contracts:
        r = {
            "name":     c["path"].name,
            "category": c["category"],
            "exact":    [],
            "norm":     [],
            "filename": [],
            "jaccard":  [],
            "graph":    [],
        }

        # Tier 1a — exact raw content hash in BCCC index
        if c["h_raw"] in exact_index:
            r["exact"] = [str(p) for p in exact_index[c["h_raw"]]]
            exact_hits += 1

        # Tier 1b — SmartBugs content SHA256 == a BCCC filename stem
        # (BCCC filenames may be sha256 of Ethereum address, not content;
        #  this checks whether the contract's content hash was ever used as a BCCC name)
        if c["h_raw"] in bccc_filename_sha256_set:
            r["filename"].append(f"BCCC filename match: {c['h_raw']}.sol")
            filename_hits += 1

        # Tier 2 — normalised content hash
        if not r["exact"] and c["h_norm"] in norm_index:
            r["norm"] = [str(p) for p in norm_index[c["h_norm"]]]
            norm_hits += 1

        hit = bool(r["exact"] or r["norm"] or r["filename"])
        status = "EXACT" if r["exact"] else ("NORM" if r["norm"] else ("FNAME" if r["filename"] else "clean"))
        if hit:
            print(f"  *** {status} HIT *** {c['category']}/{c['path'].name}")
            for p in (r["exact"] or r["norm"] or r["filename"])[:3]:
                print(f"        → {p}")

        results.append(r)

    print(f"\n  Exact hits    : {exact_hits}/{len(sb_contracts)}")
    print(f"  Normalised    : {norm_hits}/{len(sb_contracts)}")
    print(f"  Filename stem : {filename_hits}/{len(sb_contracts)}")

    # ── Tier 3: Jaccard near-duplicate ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TIER 3 — Token Jaccard Similarity (threshold={jaccard_threshold})")
    print(f"{'='*70}")
    print(f"  Strategy: for each SmartBugs contract, pre-filter BCCC to top-{top_k}")
    print(f"  candidates by |token_count_diff|, then compute exact Jaccard.\n")

    jaccard_hits = 0
    for c, r in zip(sb_contracts, results):
        if r["exact"] or r["norm"]:
            continue  # already confirmed contaminated — no need for Jaccard

        c_len  = c["len"]
        c_toks = c["toks"]

        # Pre-filter: keep BCCC entries whose char length is within 50% of SmartBugs
        candidates = [
            (p, toks) for p, _, _, toks, blen in bccc_entries
            if abs(blen - c_len) / max(c_len, 1) <= 0.50
        ]

        # Sort by |token_set size difference| and keep top-K
        candidates.sort(key=lambda x: abs(len(x[1]) - len(c_toks)))
        candidates = candidates[:top_k]

        best_j = 0.0
        best_match = None
        for bp, btoks in candidates:
            j = jaccard(c_toks, btoks)
            if j > best_j:
                best_j = j
                best_match = bp

        r["jaccard_best"] = best_j
        r["jaccard_match"] = str(best_match) if best_match else None

        if best_j >= jaccard_threshold:
            jaccard_hits += 1
            r["jaccard"].append(f"J={best_j:.3f} → {best_match}")
            print(f"  NEAR-DUP  J={best_j:.3f}  {c['category']}/{c['path'].name}")
            print(f"            → {best_match}")

    if jaccard_hits == 0:
        print(f"  No near-duplicates found at threshold {jaccard_threshold}.")
    print(f"\n  Near-duplicate hits : {jaccard_hits}/{len(sb_contracts)}")

    # ── Tier 4: Structural graph fingerprint ─────────────────────────────────
    if not skip_tier4:
        print(f"\n{'='*70}")
        print("TIER 4 — Structural Graph Fingerprint (num_nodes, num_edges, fn_names)")
        print(f"{'='*70}")
        import torch

        split_map = load_training_split_ids()
        fp_index  = build_graph_fingerprint_index(split_map)

        # Extract graphs for SmartBugs contracts (re-use predictor preprocessor)
        from ml.src.inference.preprocess import ContractPreprocessor
        preprocessor = ContractPreprocessor()

        graph_hits = 0
        for c, r in zip(sb_contracts, results):
            try:
                graph, _ = preprocessor.process_source_windowed(c["text"])
            except Exception as exc:
                r["graph_error"] = str(exc)
                continue

            nn_ = int(graph.num_nodes)
            ne_ = int(graph.num_edges)
            fn_names = sorted(
                m.get("name", "") for m in (graph.node_metadata or [])
                if m.get("type") in ("FUNCTION", "MODIFIER", "FALLBACK",
                                     "RECEIVE", "CONSTRUCTOR")
            )

            # Look up by (num_nodes, num_edges) — very specific fingerprint
            candidates = fp_index.get((nn_, ne_), [])
            fn_matches = [
                (stem, split, cand_fn)
                for stem, split, cand_fn in candidates
                if cand_fn == fn_names
            ]

            r["graph"] = [(stem, split) for stem, split, _ in fn_matches]

            if fn_matches:
                graph_hits += 1
                splits_found = set(s for _, s, _ in fn_matches)
                print(f"  STRUCT HIT  {c['category']}/{c['path'].name}"
                      f"  nodes={nn_} edges={ne_}  fn_match={fn_names[:3]}")
                print(f"    Found in splits: {splits_found}  ({len(fn_matches)} candidates)")
                for stem, split, _ in fn_matches[:3]:
                    print(f"      md5={stem}  split={split}")
            elif candidates:
                # Same shape but different function names — partial structural match
                print(f"  shape-only  {c['category']}/{c['path'].name}"
                      f"  nodes={nn_} edges={ne_}  (fn names differ)")

        print(f"\n  Full struct hits (nodes+edges+fn_names) : {graph_hits}/{len(sb_contracts)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONTAMINATION SUMMARY")
    print(f"{'='*70}")

    confirmed_contaminated = [
        r for r in results if r["exact"] or r["norm"] or r["filename"]
    ]
    near_dups = [r for r in results if r.get("jaccard")]
    struct_hits = [r for r in results if r.get("graph")]

    total_flagged = len(set(
        r["name"] for r in results
        if r["exact"] or r["norm"] or r["filename"] or r.get("jaccard") or r.get("graph")
    ))

    print(f"  SmartBugs contracts checked : {len(sb_contracts)}")
    print(f"  Exact content match         : {len(confirmed_contaminated)}")
    print(f"  Near-duplicate (J≥{jaccard_threshold})     : {len(near_dups)}")
    if not skip_tier4:
        print(f"  Structural graph match      : {len(struct_hits)}")
    print(f"  Total flagged (any tier)    : {total_flagged}")
    print()

    if total_flagged == 0:
        print("  ✓ NO CONTAMINATION DETECTED — benchmark results are clean.")
        print("    SmartBugs contracts appear independent of the BCCC training corpus.")
    else:
        print("  ⚠ CONTAMINATION DETECTED — benchmark results may be inflated.")
        print("  Flagged contracts:")
        for r in results:
            hits = []
            if r["exact"]:   hits.append("EXACT")
            if r["norm"]:    hits.append("NORM")
            if r["filename"]: hits.append("FNAME")
            if r.get("jaccard"): hits.append(f"J={r.get('jaccard_best', 0):.2f}")
            if r.get("graph"):   hits.append(f"GRAPH(split={r['graph'][0][1]})")
            if hits:
                print(f"    {r['category']}/{r['name']:40s}  {'+'.join(hits)}")

    print()

    # ── Distribution of best Jaccard scores (diagnostic) ────────────────────
    best_jaccards = [r.get("jaccard_best", 0.0) for r in results]
    if best_jaccards:
        import statistics
        print(f"  Jaccard score distribution (vs top-{top_k} BCCC candidates per contract):")
        print(f"    min={min(best_jaccards):.3f}  median={statistics.median(best_jaccards):.3f}"
              f"  mean={statistics.mean(best_jaccards):.3f}  max={max(best_jaccards):.3f}")
        buckets = [(0.9, 1.0), (0.75, 0.9), (0.5, 0.75), (0.25, 0.5), (0.0, 0.25)]
        for lo, hi in buckets:
            cnt = sum(1 for j in best_jaccards if lo <= j < hi)
            if cnt:
                print(f"    [{lo:.2f}–{hi:.2f}): {cnt} contracts")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jaccard-threshold", type=float, default=0.75)
    parser.add_argument("--top-k-candidates", type=int, default=50,
                        help="BCCC candidates per SmartBugs contract for Jaccard")
    parser.add_argument("--no-tier4", action="store_true",
                        help="Skip structural graph fingerprint check (saves ~3 min)")
    args = parser.parse_args()

    if not BCCC_DIR.exists():
        print(f"ERROR: BCCC source dir not found: {BCCC_DIR}", file=sys.stderr)
        sys.exit(1)
    if not SMARTBUGS_DIR.exists():
        print(f"ERROR: SmartBugs dir not found: {SMARTBUGS_DIR}", file=sys.stderr)
        sys.exit(1)

    run_check(args.jaccard_threshold, args.top_k_candidates, args.no_tier4)


if __name__ == "__main__":
    main()
