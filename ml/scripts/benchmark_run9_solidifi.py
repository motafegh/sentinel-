"""
benchmark_run9_solidifi.py — Evaluate Run 9 model on SolidiFI (350 contracts, 7 categories)

Usage (from repo root, with venv active):
    python -m ml.scripts.benchmark_run9_solidifi [--checkpoint PATH] [--verbose]

SolidiFI dataset:
  350 standalone .sol files — 50 per category × 7 categories.
  All have pragma >=0.4.22 <0.6.0, use address payable (0.5.x syntax) → solc 0.5.17.
  341/350 confirmed clean of BCCC contamination (Tier 1–3 check, 9 near-dups in
  Unchecked-Send only — included with a contamination flag in the output).

SolidiFI category → SENTINEL class mapping:
  Re-entrancy           → Reentrancy
  Overflow-Underflow    → IntegerUO
  TOD                   → TransactionOrderDependence
  Timestamp-Dependency  → Timestamp
  Unchecked-Send        → CallToUnknown
  Unhandled-Exceptions  → MishandledException
  tx.origin             → (no mapping — FP probe)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT     = Path(__file__).resolve().parents[2]
SOLIDIFI_DIR  = REPO_ROOT / "Data" / "data" / "raw" / "solidifi" / "repo" / "buggy_contracts"
DEFAULT_CKPT  = REPO_ROOT / "ml" / "checkpoints" / "GCB-P1-Run9-v11-20260606_best.pt"
THRESHOLDS_JSON = REPO_ROOT / "ml" / "calibration" / "GCB-P1-Run9-v11-20260606_thresholds.json"

# 9 SolidiFI contracts identified as near-duplicate with BCCC training data
NEAR_DUP_STEMS = {
    "buggy_1", "buggy_11", "buggy_13", "buggy_16", "buggy_25",
    "buggy_27", "buggy_35", "buggy_37", "buggy_43",
}

SOLIDIFI_TO_SENTINEL: dict[str, str] = {
    "Re-entrancy":         "Reentrancy",
    "Overflow-Underflow":  "IntegerUO",
    "TOD":                 "TransactionOrderDependence",
    "Timestamp-Dependency": "Timestamp",
    "Unchecked-Send":      "CallToUnknown",
    "Unhandled-Exceptions": "MishandledException",
}
UNMAPPED = {"tx.origin"}

TRAINING_MEDIAN_NODES = 90
TIER_CONFIRMED  = 0.55
TIER_SUSPICIOUS = 0.25


def load_tuned_thresholds(path: Path) -> dict[str, float]:
    with path.open() as f:
        data = json.load(f)
    return data.get("thresholds", {})


def classify_tuned(probs: dict[str, float], thresholds: dict[str, float]) -> dict[str, bool]:
    return {cls: prob >= thresholds.get(cls, 0.325) for cls, prob in probs.items()}


def run_benchmark(checkpoint: Path, verbose: bool, include_near_dups: bool) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from ml.src.inference.predictor import Predictor

    print(f"\n{'='*70}")
    print("SENTINEL Run 9 — SolidiFI Benchmark")
    print(f"{'='*70}")
    print(f"Checkpoint  : {checkpoint}")
    print(f"SolidiFI    : {SOLIDIFI_DIR}")
    print(f"Thresholds  : {THRESHOLDS_JSON}")
    print(f"Near-dups   : {'included' if include_near_dups else 'excluded (9 Unchecked-Send contracts)'}")
    print()

    tuned_thresholds: dict[str, float] = {}
    if THRESHOLDS_JSON.exists():
        tuned_thresholds = load_tuned_thresholds(THRESHOLDS_JSON)
        print(f"Tuned thresholds: {tuned_thresholds}")
    else:
        print(f"WARNING: {THRESHOLDS_JSON} not found — tuned pass skipped")
    print()

    print("Loading predictor...")
    predictor = Predictor(checkpoint=str(checkpoint))
    print()

    # Collect contracts
    contracts: list[tuple[str, Path, bool]] = []  # (category, path, is_near_dup)
    for cat_dir in sorted(SOLIDIFI_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for sol in sorted(cat_dir.glob("*.sol")):
            near_dup = sol.stem in NEAR_DUP_STEMS and cat_dir.name == "Unchecked-Send"
            if near_dup and not include_near_dups:
                continue
            contracts.append((cat_dir.name, sol, near_dup))

    total = len(contracts)
    print(f"Found {total} contracts across {len(set(c[0] for c in contracts))} categories\n")

    results: dict[str, list[dict]] = defaultdict(list)
    node_counts: list[int] = []
    errors: list[tuple[str, str]] = []

    for i, (category, sol_path, near_dup) in enumerate(contracts, 1):
        if verbose:
            dup_tag = " [near-dup]" if near_dup else ""
            print(f"[{i:3d}/{total}] {category}/{sol_path.name}{dup_tag}", end=" ... ", flush=True)
        try:
            result = predictor.predict(str(sol_path))
            probs    = result["probabilities"]
            num_nodes = result["num_nodes"]
            node_counts.append(num_nodes)

            confirmed_cls = {v["vulnerability_class"] for v in result.get("confirmed", [])}
            suspicious_cls = {v["vulnerability_class"] for v in result.get("suspicious", [])}
            tuned_preds   = classify_tuned(probs, tuned_thresholds) if tuned_thresholds else {}

            results[category].append({
                "path": sol_path.name,
                "probs": probs,
                "confirmed": confirmed_cls,
                "suspicious": suspicious_cls,
                "tuned_preds": tuned_preds,
                "num_nodes": num_nodes,
                "near_dup": near_dup,
                "error": None,
            })
            if verbose:
                mapped = SOLIDIFI_TO_SENTINEL.get(category)
                if mapped:
                    p = probs.get(mapped, 0)
                    in_conf = mapped in confirmed_cls
                    in_susp = mapped in suspicious_cls
                    tier = "CONF ✓" if in_conf else ("SUSP ✓" if in_susp else f"miss (p={p:.3f})")
                    tuned = "HIT" if tuned_preds.get(mapped) else "miss"
                    print(f"nodes={num_nodes:3d}  p={p:.3f}  tier={tier:<14} tuned={tuned}")
                else:
                    top = sorted(probs.items(), key=lambda x: -x[1])[:2]
                    print(f"nodes={num_nodes:3d}  top: {', '.join(f'{c}={v:.3f}' for c,v in top)}")

        except Exception as exc:
            errors.append((str(sol_path), str(exc)))
            results[category].append({"path": sol_path.name, "error": str(exc), "near_dup": near_dup})
            if verbose:
                print(f"ERROR: {exc}")

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    # OOD stats
    if node_counts:
        import statistics
        med  = statistics.median(node_counts)
        mean = statistics.mean(node_counts)
        mn, mx = min(node_counts), max(node_counts)
        print(f"\n[OOD Graph Size] median={med:.0f}  mean={mean:.0f}  min={mn}  max={mx}")
        print(f"  Training median: {TRAINING_MEDIAN_NODES} nodes")
        pct_small = sum(1 for n in node_counts if n < 30) / len(node_counts) * 100
        print(f"  Contracts with <30 nodes: {pct_small:.1f}%")
    print()

    print(f"{'Category':<24} {'SENTINEL class':<28} {'N':>4}  {'Tier P/R/F1':>20}  {'Tuned P/R/F1':>18}")
    print("-" * 104)

    ov_tp_t = ov_fn_t = ov_tp_u = ov_fn_u = 0

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    for cat in sorted(SOLIDIFI_TO_SENTINEL.keys()):
        mapped = SOLIDIFI_TO_SENTINEL[cat]
        cat_ok = [r for r in results.get(cat, []) if r.get("error") is None]
        n = len(cat_ok)
        if n == 0:
            err_count = sum(1 for r in results.get(cat, []) if r.get("error"))
            print(f"  {cat:<22} {mapped:<28} {0:>4}  (all {err_count} errored)")
            continue

        tp_t = sum(1 for r in cat_ok if mapped in r["confirmed"] or mapped in r["suspicious"])
        fn_t = n - tp_t
        tp_u = sum(1 for r in cat_ok if r["tuned_preds"].get(mapped, False))
        fn_u = n - tp_u

        p_t, r_t, f_t = prf(tp_t, 0, fn_t)
        p_u, r_u, f_u = prf(tp_u, 0, fn_u)
        ov_tp_t += tp_t; ov_fn_t += fn_t
        ov_tp_u += tp_u; ov_fn_u += fn_u

        err_count = sum(1 for r in results.get(cat, []) if r.get("error"))
        err_tag = f"  ({err_count} err)" if err_count else ""
        print(
            f"  {cat:<22} {mapped:<28} {n:>4}  "
            f"P={p_t:.2f} R={r_t:.2f} F1={f_t:.2f}  |  "
            f"P={p_u:.2f} R={r_u:.2f} F1={f_u:.2f}{err_tag}"
        )

        if verbose:
            for r in cat_ok:
                prob = r["probs"].get(mapped, 0)
                tier_lbl = ("CONF" if mapped in r["confirmed"]
                            else "SUSP" if mapped in r["suspicious"] else "miss")
                tuned_lbl = "HIT " if r["tuned_preds"].get(mapped) else "miss"
                dup_tag = " [nd]" if r["near_dup"] else ""
                print(f"      {r['path'][:42]:<42}  p={prob:.3f}  tier={tier_lbl}  tuned={tuned_lbl}  "
                      f"nodes={r['num_nodes']}{dup_tag}")

    N = ov_tp_t + ov_fn_t
    print()
    print(f"  {'MAPPABLE TOTAL':<52}  "
          f"R(tier)={ov_tp_t/N:.2f}  R(tuned)={ov_tp_u/N:.2f}"
          if N else "  No results")

    # FP probe for tx.origin (unmapped)
    print(f"\n{'='*70}")
    print("FALSE POSITIVE PROBE  (tx.origin — no SENTINEL equivalent)")
    print(f"{'='*70}")
    for cat in sorted(UNMAPPED):
        cat_ok = [r for r in results.get(cat, []) if r.get("error") is None]
        n = len(cat_ok)
        if n == 0:
            print(f"  {cat}: no successful predictions")
            continue
        any_conf  = sum(1 for r in cat_ok if len(r["confirmed"]) > 0)
        any_susp  = sum(1 for r in cat_ok if len(r["confirmed"]) > 0 or len(r["suspicious"]) > 0)
        any_tuned = sum(1 for r in cat_ok if any(r["tuned_preds"].values()))
        print(f"  {cat:<20} n={n}  CONF={any_conf}/{n}  CONF+SUSP={any_susp}/{n}  Tuned={any_tuned}/{n}")
        if verbose:
            for r in cat_ok:
                top = sorted(r["probs"].items(), key=lambda x: -x[1])[:3]
                print(f"    {r['path'][:40]:<40}  top: {', '.join(f'{c}={v:.3f}' for c,v in top)}")

    if errors:
        print(f"\n{'='*70}")
        print(f"ERRORS ({len(errors)})")
        print(f"{'='*70}")
        for path, err in errors[:20]:
            print(f"  {Path(path).name}: {err[:140]}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")

    ok_count = total - len(errors)
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Contracts processed : {ok_count}/{total}")
    print(f"  Errors              : {len(errors)}")
    if node_counts:
        import statistics
        print(f"  Median graph size   : {statistics.median(node_counts):.0f} nodes  "
              f"(training median={TRAINING_MEDIAN_NODES})")
    print(f"\n  SolidiFI is 341/350 unseen (9 Unchecked-Send near-dups flagged).")
    print(f"  All have pragma >=0.4.22 <0.6.0 → compiled with solc 0.5.17.")
    print(f"  Run 9 trained on noisy BCCC labels — these are first genuine OOD results.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Run 9 on SolidiFI")
    parser.add_argument("--checkpoint",    type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--include-near-dups", action="store_true",
                        help="Include the 9 Unchecked-Send near-dups (excluded by default)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not SOLIDIFI_DIR.exists():
        print(f"ERROR: SolidiFI dataset not found: {SOLIDIFI_DIR}", file=sys.stderr)
        sys.exit(1)

    run_benchmark(args.checkpoint, args.verbose, args.include_near_dups)


if __name__ == "__main__":
    main()
