"""
patch_graph_features.py — In-place fix for three known feature bugs in graph .pt files.

Bugs fixed
──────────
BUG-1  dim[6] loc — 2,856 graphs (6.4%) store raw line counts (max observed: 2,167)
       instead of log1p(N)/log1p(1000) ∈ [0,1].  Root cause: those graphs were not
       re-extracted during the v7 run and were not correctly normalised by the earlier
       in-place patch (which only targeted CFG nodes, missing FUNCTION, MODIFIER, etc.).

BUG-2  dim[5] complexity — 37 graphs (0.08%) store raw CFG block counts (max: 48)
       instead of log1p(N)/log1p(100) ∈ [0,1].  Same root cause as BUG-1.

BUG-3  dim[1] visibility — 7,854 graphs (17.7%) have values of 2 for private functions,
       which exceeds the nominal [0,1] range.  Root cause: VISIBILITY_MAP used an
       ordinal encoding {public=0, internal=1, private=2} but the schema declared [0,1].
       Fix: remap to {public=0.0, internal=0.5, private=1.0} preserving the ordinal
       ordering (private > internal > public) while staying in range.

       NOTE: This patch must be applied AFTER updating VISIBILITY_MAP in graph_schema.py
       to {public:0, external:0, internal:0.5, private:1.0}.  New graphs extracted after
       that change will already use the correct values.  This patch only fixes existing
       graphs on disk.

Verification
────────────
After the patch, run a full validation to confirm zero out-of-range values:
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/patch_graph_features.py --verify-only

Usage
─────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/patch_graph_features.py
    PYTHONPATH=. python ml/scripts/patch_graph_features.py --graphs-dir ml/data/graphs
    PYTHONPATH=. python ml/scripts/patch_graph_features.py --verify-only  # check without patching

After patching, rebuild the RAM cache:
    PYTHONPATH=. python ml/scripts/create_cache.py \\
        --tokens-dir ml/data/tokens_windowed \\
        --output ml/data/cached_dataset_windowed.pkl
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── normalization helpers (same formulas as graph_extractor.py) ──────────────

def _norm_loc(raw: float) -> float:
    """log1p(raw) / log1p(1000), clamped [0, 1]."""
    return min(math.log1p(raw) / math.log1p(1000), 1.0)


def _norm_complexity(raw: float) -> float:
    """log1p(raw) / log1p(100), clamped [0, 1]."""
    return min(math.log1p(raw) / math.log1p(100), 1.0)


# Visibility remap: old schema v5 int encoding → new schema v6 float encoding
# Old: {public=0, internal=1, private=2}   (private exceeded [0,1] range)
# New: {public=0.0, internal=0.5, private=1.0}  (ordinal preserved, in-range)
# Applied in descending value order in patch_graph() to avoid collision.


def patch_graph(path: Path, verify_only: bool = False) -> dict:
    """
    Load one graph, detect and (optionally) fix BUG-1/2/3, save back.

    Returns a dict with counts of what was found and fixed:
        {"bug1": int, "bug2": int, "bug3": int, "modified": bool}
    """
    g = torch.load(path, weights_only=False)
    x = g.x  # [N, 12]

    bug1_nodes = (x[:, 6] > 1.0).sum().item()    # loc out of range
    bug2_nodes = (x[:, 5] > 1.0).sum().item()    # complexity out of range
    bug3_nodes = (x[:, 1] > 1.0).sum().item()    # visibility=2 (private, OOR)
    # internal=1 must also be remapped to 0.5 under schema v6 even though
    # it was not technically OOR — apply to all graphs for consistency
    vis_remap_nodes = ((x[:, 1] == 1.0) | (x[:, 1] == 2.0)).sum().item()

    result = {"bug1": bug1_nodes, "bug2": bug2_nodes, "bug3": bug3_nodes, "modified": False}

    needs_patch = bug1_nodes > 0 or bug2_nodes > 0 or vis_remap_nodes > 0
    if verify_only or not needs_patch:
        return result

    x = x.clone()

    # ── BUG-1: normalize loc (dim[6]) ────────────────────────────────────────
    if bug1_nodes > 0:
        mask = x[:, 6] > 1.0
        raw_vals = x[mask, 6].tolist()
        normed = torch.tensor(
            [_norm_loc(v) for v in raw_vals],
            dtype=x.dtype
        )
        x[mask, 6] = normed

    # ── BUG-2: normalize complexity (dim[5]) ─────────────────────────────────
    if bug2_nodes > 0:
        mask = x[:, 5] > 1.0
        raw_vals = x[mask, 5].tolist()
        normed = torch.tensor(
            [_norm_complexity(v) for v in raw_vals],
            dtype=x.dtype
        )
        x[mask, 5] = normed

    # ── BUG-3 + visibility remap (dim[1]) ────────────────────────────────────
    # Old schema v5: {public=0, internal=1, private=2}
    # New schema v6: {public=0.0, internal=0.5, private=1.0}
    #
    # Process in descending value order to avoid collision:
    #   private: 2.0 → 1.0  (done first — 1.0 is new internal, not yet assigned)
    #   internal: 1.0 → 0.5 (done after private is already moved to 1.0)
    if vis_remap_nodes > 0:
        mask_private = (x[:, 1] == 2.0)
        if mask_private.any():
            x[mask_private, 1] = 1.0

        # After private remap, any remaining 1.0 nodes are genuine internal
        mask_internal = (x[:, 1] == 1.0) & ~mask_private
        if mask_internal.any():
            x[mask_internal, 1] = 0.5

    g.x = x
    torch.save(g, path)
    result["modified"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch BUG-1/2/3 in graph .pt files")
    parser.add_argument("--graphs-dir", default="ml/data/graphs",
                        help="Directory containing graph .pt files")
    parser.add_argument("--verify-only", action="store_true",
                        help="Scan and report without modifying any files")
    args = parser.parse_args()

    graphs_dir = Path(args.graphs_dir)
    files = sorted(graphs_dir.glob("*.pt"))

    if not files:
        print(f"ERROR: No .pt files found in {graphs_dir}")
        sys.exit(1)

    mode = "VERIFY-ONLY" if args.verify_only else "PATCH"
    print(f"{'='*60}")
    print(f"Graph feature patch — mode: {mode}")
    print(f"Directory: {graphs_dir}")
    print(f"Files: {len(files):,}")
    print(f"{'='*60}")
    print()
    print("Bugs being checked:")
    print("  BUG-1  dim[6] loc > 1.0  → normalize with log1p(N)/log1p(1000)")
    print("  BUG-2  dim[5] complexity > 1.0  → normalize with log1p(N)/log1p(100)")
    print("  BUG-3  dim[1] visibility = 2  → remap {internal:1→0.5, private:2→1.0}")
    print()

    total_bug1 = total_bug2 = total_bug3 = 0
    graphs_bug1 = graphs_bug2 = graphs_bug3 = 0
    graphs_modified = 0
    errors = []

    for path in tqdm(files, desc=mode, unit="graph"):
        try:
            r = patch_graph(path, verify_only=args.verify_only)
            if r["bug1"] > 0:
                total_bug1 += r["bug1"]
                graphs_bug1 += 1
            if r["bug2"] > 0:
                total_bug2 += r["bug2"]
                graphs_bug2 += 1
            if r["bug3"] > 0:
                total_bug3 += r["bug3"]
                graphs_bug3 += 1
            if r["modified"]:
                graphs_modified += 1
        except Exception as e:
            errors.append((path.name, str(e)))

    print()
    print(f"{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total graphs scanned: {len(files):,}")
    print()
    print(f"BUG-1  loc > 1.0:         {graphs_bug1:,} graphs ({100*graphs_bug1/len(files):.2f}%),  {total_bug1:,} nodes")
    print(f"BUG-2  complexity > 1.0:  {graphs_bug2:,} graphs ({100*graphs_bug2/len(files):.2f}%),  {total_bug2:,} nodes")
    print(f"BUG-3  visibility > 1.0:  {graphs_bug3:,} graphs ({100*graphs_bug3/len(files):.2f}%),  {total_bug3:,} nodes")
    print()
    if not args.verify_only:
        print(f"Graphs modified:          {graphs_modified:,}")
    if errors:
        print(f"Errors:                   {len(errors)}")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    print()
    if args.verify_only:
        total_oor = graphs_bug1 + graphs_bug2 + graphs_bug3
        if total_oor == 0:
            print("✅  All features in range — no patches needed.")
        else:
            print(f"❌  {total_oor:,} graphs have out-of-range features. Run without --verify-only to patch.")
    else:
        print("✅  Patch complete.")
        print()
        print("Next step — rebuild the RAM cache:")
        print("    PYTHONPATH=. python ml/scripts/create_cache.py \\")
        print("        --tokens-dir ml/data/tokens_windowed \\")
        print("        --output ml/data/cached_dataset_windowed.pkl")


if __name__ == "__main__":
    main()
