"""Contamination audit: SmartBugs Wild 47K vs v3 training data.

Checks whether any of the 47,398 SmartBugs Wild contracts are present in
v3 train/val/test splits. Uses three detection tiers:

  Tier 1 — Exact SHA-256 of raw Wild .sol bytes vs v3 sha256 field
  Tier 2 — SHA-256 of normalised content (comments stripped, whitespace
            collapsed, lowercased) for both Wild and preprocessed v3 .sol files
  Tier 3 — Token Jaccard >= threshold vs top-K v3 candidates (length-filtered)

Outputs:
  ml/reports/Run12_smartbugs_wild_contamination_index.json  (per-address verdict)
  ml/reports/Run12_smartbugs_wild_contamination_summary.json (aggregate stats)

Usage:
    ml/.venv/bin/python ml/scripts/audit/check_contamination_wild.py
    ml/.venv/bin/python ml/scripts/audit/check_contamination_wild.py --no-tier3
    ml/.venv/bin/python ml/scripts/audit/check_contamination_wild.py \\
        --jaccard-threshold 0.80 --top-k 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]  # sentinel/
WILD_DIR = REPO_ROOT / "ml" / "data" / "smartbugs-wild" / "contracts"
V3_SPLITS_DIR = REPO_ROOT / "data_module" / "data" / "splits" / "v3"
PREPROCESSED_DIR = REPO_ROOT / "data_module" / "data" / "preprocessed"
REPORTS_DIR = REPO_ROOT / "ml" / "reports"

SOURCES = ("dive", "solidifi", "smartbugs_curated")

_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE = re.compile(r"//[^\n]*")
_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalise(src: str) -> str:
    s = _COMMENT_BLOCK.sub(" ", src)
    s = _COMMENT_LINE.sub(" ", s)
    s = _WHITESPACE.sub(" ", s)
    return s.strip().lower()


def token_set(text: str) -> frozenset:
    return frozenset(re.split(r"[^a-zA-Z0-9_]+", text)) - {""}


# ---------------------------------------------------------------------------
# V3 index builders
# ---------------------------------------------------------------------------

def build_v3_index() -> dict[str, dict]:
    """Return {sha256 -> {split, source, primary_class}} from v3 splits JSONL."""
    index: dict[str, dict] = {}
    for split_file in sorted(V3_SPLITS_DIR.glob("*.jsonl")):
        split_name = split_file.stem
        for line in split_file.read_text(encoding="utf-8").splitlines():
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


def build_v3_norm_index(v3_index: dict) -> tuple[dict[str, list[str]], list[tuple]]:
    """Build normalised-SHA-256 index and token-set list from preprocessed .sol files.

    Returns:
        norm_index: {h_norm -> [sha256, ...]}
        entries:    [(sha256, token_frozenset, char_len, split, source), ...]
    """
    norm_index: dict[str, list[str]] = defaultdict(list)
    entries: list[tuple] = []
    missing = 0

    for sha, meta in v3_index.items():
        source = meta["source"]
        sol_path = PREPROCESSED_DIR / source / f"{sha}.sol"
        if not sol_path.exists():
            missing += 1
            continue
        raw = sol_path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        norm = normalise(text)
        h_norm = sha256_bytes(norm.encode())
        norm_index[h_norm].append(sha)
        entries.append((sha, token_set(norm), len(text), meta["split"], source))

    if missing:
        print(f"  [WARN] {missing} v3 contracts missing preprocessed .sol file")
    return dict(norm_index), entries


# ---------------------------------------------------------------------------
# Wild contract loader
# ---------------------------------------------------------------------------

def iter_wild_contracts(wild_dir: Path):
    """Yield (address, raw_bytes, text, norm_text) for each Wild .sol file."""
    sol_files = sorted(wild_dir.glob("*.sol"))
    total = len(sol_files)
    for i, sol in enumerate(sol_files):
        address = sol.stem  # 0x...
        try:
            raw = sol.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            norm = normalise(text)
        except Exception as exc:
            yield address, None, None, None, str(exc)
            continue
        if i % 5000 == 0:
            pct = 100 * i / total
            print(f"  [{i:>6}/{total}] {pct:.1f}%  {address[:20]}…", flush=True)
        yield address, raw, text, norm, None


# ---------------------------------------------------------------------------
# Audit core
# ---------------------------------------------------------------------------

def audit_wild(
    v3_index: dict,
    v3_norm_index: dict,
    v3_entries: list,
    jaccard_threshold: float,
    top_k: int,
    skip_tier3: bool,
) -> list[dict]:
    """Run all tiers against every Wild contract. Returns per-address verdicts."""
    results: list[dict] = []

    for address, raw, text, norm, read_err in iter_wild_contracts(WILD_DIR):
        record: dict = {
            "address": address,
            "read_error": read_err,
            "tier_hit": 0,
            "v3_split": None,
            "v3_sha256": None,
            "v3_source": None,
            "tier1_exact": False,
            "tier2_norm": False,
            "tier3_jaccard": None,
        }

        if read_err:
            results.append(record)
            continue

        # Tier 1 — exact raw SHA-256
        h_raw = sha256_bytes(raw)
        if h_raw in v3_index:
            v = v3_index[h_raw]
            record.update({
                "tier_hit": 1,
                "tier1_exact": True,
                "v3_split": v["split"],
                "v3_sha256": h_raw,
                "v3_source": v["source"],
            })
            results.append(record)
            continue

        # Tier 2 — normalised SHA-256
        h_norm = sha256_bytes(norm.encode())
        if h_norm in v3_norm_index:
            matched_shas = v3_norm_index[h_norm]
            sha = matched_shas[0]
            v = v3_index.get(sha, {})
            record.update({
                "tier_hit": 2,
                "tier2_norm": True,
                "v3_split": v.get("split"),
                "v3_sha256": sha,
                "v3_source": v.get("source"),
            })
            results.append(record)
            continue

        # Tier 3 — token Jaccard (optional, expensive)
        if skip_tier3:
            results.append(record)
            continue

        wild_toks = token_set(norm)
        wild_len = len(text)
        # Pre-filter by length ±50%
        candidates = [
            e for e in v3_entries
            if 0.5 * wild_len <= e[2] <= 2.0 * wild_len
        ][:top_k]

        best_j = 0.0
        best_match = None
        for sha, v3_toks, v3_len, v3_split, v3_src in candidates:
            u = len(wild_toks | v3_toks)
            if u == 0:
                continue
            j = len(wild_toks & v3_toks) / u
            if j > best_j:
                best_j = j
                best_match = (sha, v3_split, v3_src)

        record["tier3_jaccard"] = round(best_j, 4)
        if best_j >= jaccard_threshold and best_match:
            sha, v3_split, v3_src = best_match
            record.update({
                "tier_hit": 3,
                "v3_split": v3_split,
                "v3_sha256": sha,
                "v3_source": v3_src,
            })

        results.append(record)

    return results


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def make_summary(results: list[dict], args: argparse.Namespace) -> dict:
    total = len(results)
    read_errors = sum(1 for r in results if r["read_error"])
    checkable = total - read_errors

    flagged = [r for r in results if r["tier_hit"] > 0]
    ood = [r for r in results if r["tier_hit"] == 0 and not r["read_error"]]

    by_split: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    by_tier: dict[int, int] = defaultdict(int)

    for r in flagged:
        by_split[r["v3_split"] or "unknown"] += 1
        by_source[r["v3_source"] or "unknown"] += 1
        by_tier[r["tier_hit"]] += 1

    in_train = by_split.get("train", 0)
    in_val = by_split.get("val", 0)
    in_test = by_split.get("test", 0)

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "jaccard_threshold": args.jaccard_threshold,
        "top_k": args.top_k,
        "skip_tier3": args.no_tier3,
        "n_wild_total": total,
        "n_read_errors": read_errors,
        "n_checkable": checkable,
        "n_flagged": len(flagged),
        "n_ood": len(ood),
        "contamination_rate_pct": round(100 * len(flagged) / checkable, 2) if checkable else 0,
        "ood_rate_pct": round(100 * len(ood) / checkable, 2) if checkable else 0,
        "by_v3_split": dict(by_split),
        "in_train": in_train,
        "in_val_test": in_val + in_test,
        "by_v3_source": dict(by_source),
        "by_tier": {str(k): v for k, v in by_tier.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wild-dir", type=Path, default=WILD_DIR,
                        help="Directory of SmartBugs Wild .sol files")
    parser.add_argument("--v3-splits-dir", type=Path, default=V3_SPLITS_DIR)
    parser.add_argument("--preprocessed-dir", type=Path, default=PREPROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--jaccard-threshold", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K v3 candidates by length proximity for Jaccard")
    parser.add_argument("--no-tier3", action="store_true",
                        help="Skip Tier 3 Jaccard (faster, lower bound on contamination)")
    parser.add_argument("--output-prefix", default="Run12_smartbugs_wild_contamination",
                        help="Prefix for output filenames in --reports-dir")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("SENTINEL — SmartBugs Wild Contamination Audit vs v3 Training Data")
    print(f"{'='*70}\n")

    # Validate paths
    for p, name in [
        (args.wild_dir, "Wild contracts dir"),
        (args.v3_splits_dir, "V3 splits dir"),
        (args.preprocessed_dir, "Preprocessed dir"),
    ]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}", file=sys.stderr)
            sys.exit(1)

    n_wild = len(list(args.wild_dir.glob("*.sol")))
    print(f"Wild contracts:  {n_wild:,}")

    # Build v3 index (Tier 1)
    print("\nBuilding v3 index (Tier 1 — exact SHA-256)...")
    v3_index = build_v3_index()
    print(f"  v3 contracts indexed: {len(v3_index):,}")

    # Build v3 normalised index (Tier 2 + 3)
    print("\nBuilding v3 normalised index (Tier 2/3 — reading preprocessed .sol files)...")
    v3_norm_index, v3_entries = build_v3_norm_index(v3_index)
    print(f"  normalised entries: {len(v3_entries):,}")

    tier3_msg = "SKIP (--no-tier3)" if args.no_tier3 else f"threshold={args.jaccard_threshold}, top_k={args.top_k}"
    print(f"\nAuditing {n_wild:,} Wild contracts...")
    print(f"  Tier 3 Jaccard: {tier3_msg}\n")

    results = audit_wild(
        v3_index=v3_index,
        v3_norm_index=v3_norm_index,
        v3_entries=v3_entries,
        jaccard_threshold=args.jaccard_threshold,
        top_k=args.top_k,
        skip_tier3=args.no_tier3,
    )

    summary = make_summary(results, args)

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Wild contracts total:    {summary['n_wild_total']:>8,}")
    print(f"  Read errors:             {summary['n_read_errors']:>8,}")
    print(f"  Checkable:               {summary['n_checkable']:>8,}")
    print(f"  Flagged (in v3):         {summary['n_flagged']:>8,}  ({summary['contamination_rate_pct']:.1f}%)")
    print(f"  True OOD:                {summary['n_ood']:>8,}  ({summary['ood_rate_pct']:.1f}%)")
    print(f"\n  By v3 split:")
    for split, n in sorted(summary["by_v3_split"].items()):
        print(f"    {split:<12} {n:>6,}")
    print(f"\n  By v3 source:")
    for src, n in sorted(summary["by_v3_source"].items()):
        print(f"    {src:<24} {n:>6,}")
    print(f"\n  By detection tier:")
    for tier, n in sorted(summary["by_tier"].items()):
        print(f"    Tier {tier}  {n:>6,}")
    print()

    # Write outputs
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.reports_dir / f"{args.output_prefix}_index.json"
    summary_path = args.reports_dir / f"{args.output_prefix}_summary.json"

    with open(index_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Index  → {index_path}")
    print(f"Summary→ {summary_path}")
    print()


if __name__ == "__main__":
    main()
