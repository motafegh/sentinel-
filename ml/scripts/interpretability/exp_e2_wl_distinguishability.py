"""
exp_e2_wl_distinguishability.py — Layer 2, P1: Typed Directed WL Distinguishability Test

PURPOSE
───────
Verify that the GNN's graph representation has enough structural resolution to
distinguish vulnerability-positive from vulnerability-negative contracts using
Typed Directed Weisfeiler-Lehman (WL) graph isomorphism tests.  High WL
equivalence between positive/negative pairs at round 8 implies a fundamental
expressivity ceiling for the GNN on that vulnerability class.

LAYER / PRIORITY
─────────────────
Layer 2, Priority 1 — GNN expressivity ceiling analysis.

HOW IT WORKS
────────────
Standard 1-WL ignores edge types and direction; SENTINEL graphs are directed and
typed, so we use Typed Directed WL:

    hash_r(v) = hash(
        hash_{r-1}(v),
        tuple(sorted((edge_type, hash_{r-1}(u)) for u->v edges)),  # incoming
        tuple(sorted((edge_type, hash_{r-1}(w)) for v->w edges)),  # outgoing
    )

    Initial: hash_0(v) = hash(node_type_id_of_v)

    Graph hash at round r = sorted tuple of all node hashes at round r.

Two graphs are WL-equivalent at round r if their graph hashes are equal.

For each class in [Reentrancy, IntegerUO, Timestamp, CallToUnknown]:
    1. Sample up to 50 positive + 50 negative val contracts.
    2. Pair them by closest node count (absolute difference).
    3. Run typed directed WL for 8 rounds.
    4. Report % of pairs that are WL-equivalent at each round.

PASS CRITERIA
─────────────
  Reentrancy:  <30% equivalent at round 8.
  IntegerUO:   <30% equivalent at round 8.
  Any class >50% equivalent at round 8 → warning about expressivity ceiling.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_e2_wl_distinguishability.py \\
        --cache ml/data/cached_dataset_v8.pkl \\
        --label-csv ml/data/processed/multilabel_index_cleaned.csv \\
        --splits-dir ml/data/splits/deduped \\
        --out ml/logs/interpretability/e2_wl_distinguishability.json \\
        --n-contracts 100

    # No --checkpoint needed.

OUTPUT
──────
Per-round equivalence rates table, pass/fail summary, JSON report, PNG heatmap.

EXIT CODES
──────────
    0  all pass criteria satisfied
    1  one or more pass criteria failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    add_common_args,
    load_val_split,
    CLASS_NAMES,
    get_node_type_tensor,
)
from ml.src.preprocessing.graph_schema import NODE_TYPES, EDGE_TYPES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TARGET_CLASSES = ["Reentrancy", "IntegerUO", "Timestamp", "CallToUnknown"]

WL_ROUNDS = 8


# ── Typed Directed WL ─────────────────────────────────────────────────────────

def _stable_hash(obj) -> int:
    """Deterministic hash of any tuple/int via SHA-256 (no PYTHONHASHSEED dependency)."""
    data = repr(obj).encode()
    return int(hashlib.sha256(data).hexdigest()[:16], 16)


def compute_wl_hashes(
    node_types: list[int],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    n_rounds: int = 8,
) -> list[list[int]]:
    """
    Compute typed directed WL node hashes for all rounds.

    Args:
        node_types: List of integer node type IDs, length N.
        edge_index: [2, E] tensor (src, dst).
        edge_attr:  [E] tensor of integer edge type IDs.
        n_rounds:   Number of WL refinement rounds.

    Returns:
        List of length n_rounds.  Each element is a list of N hashes (one per node)
        at that round (0-indexed: result[0] = round 1, result[-1] = round n_rounds).
    """
    n = len(node_types)
    if n == 0:
        return [[] for _ in range(n_rounds)]

    # Initial hash: round 0 (type-only)
    current_hashes: list[int] = [_stable_hash(("node", t)) for t in node_types]

    # Build in-edge and out-edge adjacency lists per node
    # Each entry: (edge_type, neighbour_index)
    in_edges:  list[list[tuple]] = [[] for _ in range(n)]
    out_edges: list[list[tuple]] = [[] for _ in range(n)]

    if edge_index.numel() > 0 and edge_attr is not None:
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        ets  = edge_attr.tolist()
        for s, d, et in zip(srcs, dsts, ets):
            s, d, et = int(s), int(d), int(et)
            if 0 <= s < n and 0 <= d < n:
                out_edges[s].append((et, d))
                in_edges[d].append((et, s))

    round_hashes: list[list[int]] = []
    for _ in range(n_rounds):
        new_hashes: list[int] = []
        for v in range(n):
            in_sig  = tuple(sorted((et, current_hashes[u]) for et, u in in_edges[v]))
            out_sig = tuple(sorted((et, current_hashes[w]) for et, w in out_edges[v]))
            new_hashes.append(_stable_hash((current_hashes[v], in_sig, out_sig)))
        current_hashes = new_hashes
        round_hashes.append(list(current_hashes))

    return round_hashes


def graph_wl_hash(node_hashes: list[int]) -> tuple:
    """Graph-level WL hash = sorted tuple of all node hashes."""
    return tuple(sorted(node_hashes))


# ── Pairing ───────────────────────────────────────────────────────────────────

def _pair_by_node_count(
    pos_graphs: list,
    neg_graphs: list,
) -> list[tuple]:
    """
    Greedily pair each positive graph with the negative graph closest in node count.
    Returns list of (pos_graph, neg_graph) tuples.
    """
    if not pos_graphs or not neg_graphs:
        return []

    neg_by_count: defaultdict = defaultdict(list)
    for g in neg_graphs:
        neg_by_count[g.x.shape[0]].append(g)

    neg_all_counts = sorted(neg_by_count.keys())
    neg_pool       = list(neg_graphs)

    pairs: list[tuple] = []
    used_neg = [False] * len(neg_pool)

    for pg in pos_graphs:
        pn = pg.x.shape[0]
        best_idx  = -1
        best_diff = float("inf")
        for i, ng in enumerate(neg_pool):
            if used_neg[i]:
                continue
            diff = abs(ng.x.shape[0] - pn)
            if diff < best_diff:
                best_diff = diff
                best_idx  = i
        if best_idx >= 0:
            used_neg[best_idx] = True
            pairs.append((pg, neg_pool[best_idx]))

    return pairs


# ── Per-class analysis ────────────────────────────────────────────────────────

def _analyse_class(
    cls: str,
    stems: list[str],
    df_split: pd.DataFrame,
    cache: dict,
    n_per_class: int,
    seed: int = 42,
) -> dict:
    """Run WL distinguishability for one vulnerability class."""
    rng = np.random.default_rng(seed)
    stem_to_row = {row["md5_stem"]: row for _, row in df_split.iterrows()}

    pos_stems = [s for s in stems if s in stem_to_row and int(stem_to_row[s].get(cls, 0)) == 1]
    neg_stems = [s for s in stems if s in stem_to_row and int(stem_to_row[s].get(cls, 0)) == 0]

    if len(pos_stems) > n_per_class:
        pos_stems = rng.choice(pos_stems, size=n_per_class, replace=False).tolist()
    if len(neg_stems) > n_per_class:
        neg_stems = rng.choice(neg_stems, size=n_per_class, replace=False).tolist()

    def _load_graphs(stem_list: list[str]) -> list:
        graphs = []
        for s in stem_list:
            if s not in cache:
                continue
            entry = cache[s]
            if not isinstance(entry, tuple):
                continue
            g, _ = entry
            if g is None or g.x is None or g.x.shape[0] == 0:
                continue
            graphs.append(g)
        return graphs

    pos_graphs = _load_graphs(pos_stems)
    neg_graphs = _load_graphs(neg_stems)

    log.info(f"  {cls}: {len(pos_graphs)} pos / {len(neg_graphs)} neg loaded")

    pairs = _pair_by_node_count(pos_graphs, neg_graphs)
    log.info(f"  {cls}: {len(pairs)} pairs formed")

    if not pairs:
        return {
            "n_pairs": 0,
            "equiv_by_round": {},
            "pass": True,
            "warning": "no pairs formed",
        }

    # For each pair, compute WL hashes and check equivalence at each round
    equiv_counts = {r: 0 for r in range(1, WL_ROUNDS + 1)}

    for pg, ng in pairs:
        p_types = get_node_type_tensor(pg).tolist()
        n_types = get_node_type_tensor(ng).tolist()
        p_ea = pg.edge_attr if pg.edge_attr is not None else torch.zeros(pg.edge_index.shape[1], dtype=torch.long)
        n_ea = ng.edge_attr if ng.edge_attr is not None else torch.zeros(ng.edge_index.shape[1], dtype=torch.long)

        p_rounds = compute_wl_hashes(p_types, pg.edge_index, p_ea, WL_ROUNDS)
        n_rounds = compute_wl_hashes(n_types, ng.edge_index, n_ea, WL_ROUNDS)

        for r in range(1, WL_ROUNDS + 1):
            p_gh = graph_wl_hash(p_rounds[r - 1])
            n_gh = graph_wl_hash(n_rounds[r - 1])
            if p_gh == n_gh:
                equiv_counts[r] += 1

    n_pairs = len(pairs)
    equiv_rates = {r: round(100.0 * equiv_counts[r] / n_pairs, 2) for r in range(1, WL_ROUNDS + 1)}
    final_equiv  = equiv_rates[WL_ROUNDS]

    passed = final_equiv < 30.0
    warn   = final_equiv > 50.0

    return {
        "n_pairs":       n_pairs,
        "equiv_by_round": equiv_rates,
        "final_equiv_pct": final_equiv,
        "pass":          passed,
        "warning":       f"GNN may have fundamental expressivity ceiling for {cls}" if warn else None,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(all_results: dict) -> None:
    print(f"\n{'═'*72}")
    print("  EXP-E2: Typed Directed WL Distinguishability Test")
    print(f"{'═'*72}")

    header = f"  {'Class':<22}" + "".join(f"  r{r:>2}" for r in range(1, 9))
    print(f"\n{header}")
    print(f"  {'-'*22}" + "  -----" * 8)

    pass_classes = []
    fail_classes = []
    warn_classes = []

    for cls, res in all_results.items():
        if res["n_pairs"] == 0:
            print(f"  {cls:<22}  (no pairs)")
            continue
        rates_str = "".join(
            f"  {res['equiv_by_round'].get(r, 0.0):>5.1f}" for r in range(1, 9)
        )
        status = "PASS" if res["pass"] else "FAIL"
        print(f"  {cls:<22}{rates_str}   [{status}]")
        if res["pass"]:
            pass_classes.append(cls)
        else:
            fail_classes.append(cls)
        if res.get("warning"):
            warn_classes.append(cls)
            print(f"    WARNING: {res['warning']}")

    print(f"\n  {'─'*68}")
    print(f"  Pass: {pass_classes}")
    print(f"  Fail: {fail_classes}")
    if warn_classes:
        print(f"  Expressivity ceiling warnings: {warn_classes}")
    overall = not fail_classes
    print(f"\n  EXP-E2 RESULT: {'PASS' if overall else 'FAIL'}")
    print(f"{'═'*72}\n")


def _save_plot(all_results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    rounds = list(range(1, WL_ROUNDS + 1))
    for cls, res in all_results.items():
        if res["n_pairs"] == 0:
            continue
        rates = [res["equiv_by_round"].get(r, 0.0) for r in rounds]
        ax.plot(rounds, rates, marker="o", label=cls)

    ax.axhline(30, color="red", linestyle="--", linewidth=0.8, label="30% pass threshold")
    ax.axhline(50, color="orange", linestyle=":", linewidth=0.8, label="50% expressivity warning")
    ax.set_xlabel("WL round")
    ax.set_ylabel("% equivalent pairs")
    ax.set_title("E2: Typed Directed WL Equivalence Rates")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "e2_wl_distinguishability.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved: {plot_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP-E2: Typed Directed WL Distinguishability Test"
    )
    add_common_args(p, require_checkpoint=False)
    p.set_defaults(n_contracts=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_path = Path(args.out) if args.out else Path("ml/logs/interpretability/e2_wl_distinguishability.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stems, df_split, cache = load_val_split(
            cache_path  = Path(args.cache),
            label_csv   = Path(args.label_csv),
            splits_dir  = Path(args.splits_dir),
            split       = args.split,
        )
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1

    n_per_class = args.n_contracts  # per side (pos / neg)
    all_results: dict = {}
    for cls in TARGET_CLASSES:
        log.info(f"Analysing class: {cls}")
        all_results[cls] = _analyse_class(
            cls         = cls,
            stems       = stems,
            df_split    = df_split,
            cache       = cache,
            n_per_class = n_per_class,
            seed        = args.seed,
        )

    _print_report(all_results)
    _save_plot(all_results, out_path.parent)

    overall_pass = all(
        res["pass"] for res in all_results.values() if res["n_pairs"] > 0
    )
    report = {
        "args": {
            "cache":       str(args.cache),
            "label_csv":   str(args.label_csv),
            "splits_dir":  str(args.splits_dir),
            "split":       args.split,
            "n_contracts": args.n_contracts,
            "seed":        args.seed,
        },
        "results":      all_results,
        "overall_pass": overall_pass,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved: {out_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
