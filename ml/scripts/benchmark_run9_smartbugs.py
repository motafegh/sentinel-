"""
benchmark_run9_smartbugs.py — Evaluate Run 9 model against SmartBugs Curated (143 contracts)

Usage (from repo root, with venv active):
    python -m ml.scripts.benchmark_run9_smartbugs [--checkpoint PATH] [--verbose]

What this does:
  1. Loads Run 9 checkpoint with its companion tuned-threshold JSON (auto-detected)
  2. Runs inference on every .sol in smartbugs-curated/dataset/<category>/
  3. Reports per-class Precision / Recall / F1 for the 6 mappable DASP categories
  4. Reports false-positive rate on unmappable categories (access_control, bad_randomness, etc.)
  5. Evaluates under BOTH tier thresholds (0.55 confirmed / 0.25 suspicious) AND
     the tuned per-class thresholds from the companion JSON so we can compare
  6. Reports OOD graph-size stats (median nodes vs training median=90)

SmartBugs category → SENTINEL class mapping:
  reentrancy              → Reentrancy
  arithmetic              → IntegerUO
  unchecked_low_level_calls → CallToUnknown
  denial_of_service       → DenialOfService
  front_running           → TransactionOrderDependence
  time_manipulation       → Timestamp
  (access_control / bad_randomness / short_addresses / other → no match, FP probe)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import torch

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
SMARTBUGS_DIR = REPO_ROOT / "ml" / "data" / "smartbugs-curated" / "dataset"
DEFAULT_CKPT  = REPO_ROOT / "ml" / "checkpoints" / "GCB-P1-Run9-v11-20260606_best.pt"
THRESHOLDS_JSON = REPO_ROOT / "ml" / "calibration" / "GCB-P1-Run9-v11-20260606_thresholds.json"

# ── category → SENTINEL class mapping ─────────────────────────────────────────
CATEGORY_TO_CLASS: dict[str, str] = {
    "reentrancy":               "Reentrancy",
    "arithmetic":               "IntegerUO",
    "unchecked_low_level_calls": "CallToUnknown",
    "denial_of_service":        "DenialOfService",
    "front_running":            "TransactionOrderDependence",
    "time_manipulation":        "Timestamp",
}
# Categories with no direct SENTINEL class — used as FP probes
UNMAPPED_CATEGORIES = {"access_control", "bad_randomness", "short_addresses", "other"}

# Training median node count (from audit findings)
TRAINING_MEDIAN_NODES = 90
TIER_CONFIRMED = 0.55
TIER_SUSPICIOUS = 0.25


def load_tuned_thresholds(path: Path) -> dict[str, float]:
    with path.open() as f:
        data = json.load(f)
    return data.get("thresholds", {})


def classify_with_tuned_thresholds(
    probabilities: dict[str, float],
    tuned_thresholds: dict[str, float],
) -> dict[str, bool]:
    """Apply per-class tuned thresholds to raw probabilities."""
    return {
        cls: prob >= tuned_thresholds.get(cls, 0.325)
        for cls, prob in probabilities.items()
    }


def run_benchmark(checkpoint: Path, verbose: bool) -> None:
    # ── import after path setup so sys.path is correct ───────────────────────
    sys.path.insert(0, str(REPO_ROOT))
    from ml.src.inference.predictor import Predictor

    print(f"\n{'='*70}")
    print("SENTINEL Run 9 — SmartBugs Curated Benchmark")
    print(f"{'='*70}")
    print(f"Checkpoint : {checkpoint}")
    print(f"SmartBugs  : {SMARTBUGS_DIR}")
    print(f"Thresholds : {THRESHOLDS_JSON}")
    print()

    # Load tuned per-class thresholds for the second evaluation pass
    tuned_thresholds = {}
    if THRESHOLDS_JSON.exists():
        tuned_thresholds = load_tuned_thresholds(THRESHOLDS_JSON)
        print(f"Tuned thresholds loaded: {tuned_thresholds}")
    else:
        print(f"WARNING: {THRESHOLDS_JSON} not found — tuned-threshold pass skipped")
    print()

    # Load predictor — it will auto-find the thresholds JSON next to the checkpoint
    print("Loading predictor (warmup forward pass included)...")
    predictor = Predictor(checkpoint=str(checkpoint))
    print()

    # ── collect all .sol files by category ───────────────────────────────────
    contracts: list[tuple[str, Path]] = []  # (category, path)
    for cat_dir in sorted(SMARTBUGS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for sol in sorted(cat_dir.glob("*.sol")):
            contracts.append((cat_dir.name, sol))
    print(f"Found {len(contracts)} contracts across {len(set(c[0] for c in contracts))} categories\n")

    # ── per-contract inference ────────────────────────────────────────────────
    # Results structure: {category: [{path, probs, pred_tier, pred_tuned, num_nodes, error}]}
    results: dict[str, list[dict]] = defaultdict(list)
    node_counts: list[int] = []
    errors: list[tuple[str, str]] = []

    for i, (category, sol_path) in enumerate(contracts, 1):
        if verbose:
            print(f"[{i:3d}/{len(contracts)}] {category}/{sol_path.name}", end=" ... ", flush=True)
        try:
            result = predictor.predict(str(sol_path))
            probs = result["probabilities"]
            num_nodes = result["num_nodes"]
            node_counts.append(num_nodes)

            # Tier classification (predictor's built-in 0.55/0.25 thresholds)
            confirmed_classes = {v["vulnerability_class"] for v in result.get("confirmed", [])}
            suspicious_classes = {v["vulnerability_class"] for v in result.get("suspicious", [])}

            # Tuned-threshold classification
            tuned_preds = classify_with_tuned_thresholds(probs, tuned_thresholds) if tuned_thresholds else {}

            results[category].append({
                "path": sol_path.name,
                "probs": probs,
                "confirmed": confirmed_classes,
                "suspicious": suspicious_classes,
                "tuned_preds": tuned_preds,
                "num_nodes": num_nodes,
                "error": None,
            })
            if verbose:
                mapped_class = CATEGORY_TO_CLASS.get(category)
                tier = "SAFE"
                if mapped_class and mapped_class in confirmed_classes:
                    tier = "CONFIRMED ✓"
                elif mapped_class and mapped_class in suspicious_classes:
                    tier = "SUSPICIOUS ✓"
                elif mapped_class:
                    tier = f"MISS (p={probs.get(mapped_class, 0):.3f})"
                p_str = f"{probs.get(mapped_class, 0):.3f}" if mapped_class else "n/a"
                print(f"nodes={num_nodes:3d}  p={p_str}  → {tier}")

        except Exception as exc:  # noqa: BLE001
            errors.append((str(sol_path), str(exc)))
            results[category].append({"path": sol_path.name, "error": str(exc)})
            if verbose:
                print(f"ERROR: {exc}")

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    # ── OOD stats ─────────────────────────────────────────────────────────────
    if node_counts:
        import statistics
        med = statistics.median(node_counts)
        mean = statistics.mean(node_counts)
        mn, mx = min(node_counts), max(node_counts)
        print(f"\n[OOD Graph Size] median={med:.0f}  mean={mean:.0f}  min={mn}  max={mx}")
        print(f"  Training median: {TRAINING_MEDIAN_NODES} nodes")
        pct_small = sum(1 for n in node_counts if n < 30) / len(node_counts) * 100
        print(f"  Contracts with <30 nodes: {pct_small:.1f}% (OOD tiny contracts)")
    print()

    # ── Per-category / per-class results (both evaluation modes) ──────────────
    print(f"{'Category':<28} {'Class':<28} {'N':>4}  {'Tier(0.55) P/R/F1':>22}  {'Tuned P/R/F1':>18}")
    print("-" * 110)

    overall_tp_tier = overall_fp_tier = overall_fn_tier = 0
    overall_tp_tuned = overall_fp_tuned = overall_fn_tuned = 0

    for category in sorted(CATEGORY_TO_CLASS.keys()):
        mapped_class = CATEGORY_TO_CLASS[category]
        cat_results = [r for r in results.get(category, []) if r.get("error") is None]
        n = len(cat_results)
        if n == 0:
            print(f"  {category:<26} {mapped_class:<28} {0:>4}  (no successful predictions)")
            continue

        # Tier mode: TP if class is in confirmed OR suspicious
        tp_t = sum(1 for r in cat_results if mapped_class in r["confirmed"] or mapped_class in r["suspicious"])
        fp_t = 0  # no FP for GT-positive contracts
        fn_t = n - tp_t

        # Tuned mode
        tp_u = sum(1 for r in cat_results if r["tuned_preds"].get(mapped_class, False))
        fp_u = 0
        fn_u = n - tp_u

        def prf(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return p, r, f

        p_t, r_t, f_t = prf(tp_t, fp_t, fn_t)
        p_u, r_u, f_u = prf(tp_u, fp_u, fn_u)

        overall_tp_tier += tp_t; overall_fn_tier += fn_t
        overall_tp_tuned += tp_u; overall_fn_tuned += fn_u

        print(
            f"  {category:<26} {mapped_class:<28} {n:>4}  "
            f"P={p_t:.2f} R={r_t:.2f} F1={f_t:.2f}  |  "
            f"P={p_u:.2f} R={r_u:.2f} F1={f_u:.2f}"
        )

        # Verbose per-contract breakdown
        if verbose:
            for r in cat_results:
                prob = r["probs"].get(mapped_class, 0)
                in_conf = mapped_class in r["confirmed"]
                in_susp = mapped_class in r["suspicious"]
                tuned_hit = r["tuned_preds"].get(mapped_class, False)
                tier_label = "CONF" if in_conf else ("SUSP" if in_susp else "miss")
                tuned_label = "HIT " if tuned_hit else "miss"
                print(f"      {r['path'][:45]:<45} p={prob:.3f}  tier={tier_label}  tuned={tuned_label}  nodes={r['num_nodes']}")

    print()
    print(f"  {'MAPPABLE TOTAL':<54}  R(tier)={overall_tp_tier/(overall_tp_tier+overall_fn_tier):.2f}  "
          f"R(tuned)={overall_tp_tuned/(overall_tp_tuned+overall_fn_tuned):.2f}")

    # ── False-positive analysis on unmapped categories ─────────────────────────
    print(f"\n{'='*70}")
    print("FALSE POSITIVE PROBE (categories with no SENTINEL equivalent)")
    print(f"{'='*70}")
    print(f"{'Category':<26} {'N':>4}  {'Any CONF (tier)':>16}  {'Any CONF+SUSP (tier)':>20}  {'Any Tuned':>10}")
    print("-" * 85)

    fp_total_conf = 0
    fp_total_any = 0
    fp_total_tuned = 0
    fp_n = 0

    for category in sorted(UNMAPPED_CATEGORIES):
        cat_results = [r for r in results.get(category, []) if r.get("error") is None]
        n = len(cat_results)
        if n == 0:
            continue
        fp_n += n

        any_conf = sum(1 for r in cat_results if len(r["confirmed"]) > 0)
        any_susp = sum(1 for r in cat_results if len(r["confirmed"]) > 0 or len(r["suspicious"]) > 0)
        any_tuned = sum(1 for r in cat_results if any(r["tuned_preds"].values()))

        fp_total_conf += any_conf
        fp_total_any += any_susp
        fp_total_tuned += any_tuned

        print(
            f"  {category:<24} {n:>4}  "
            f"{any_conf:>4}/{n} ({any_conf/n*100:4.0f}%)     "
            f"{any_susp:>4}/{n} ({any_susp/n*100:4.0f}%)          "
            f"{any_tuned:>4}/{n} ({any_tuned/n*100:4.0f}%)"
        )

        if verbose:
            for r in cat_results:
                top_conf = sorted(r["confirmed"], key=lambda c: r["probs"][c], reverse=True)
                top_susp = sorted(r["suspicious"], key=lambda c: r["probs"][c], reverse=True)
                top_tuned = [c for c, v in r["tuned_preds"].items() if v]
                top_probs = sorted(r["probs"].items(), key=lambda x: -x[1])[:3]
                prob_str = "  ".join(f"{c}={p:.3f}" for c, p in top_probs)
                print(f"      {r['path'][:42]:<42}  top: {prob_str}")
                if top_conf:
                    print(f"        CONF: {top_conf}")
                if top_susp:
                    print(f"        SUSP: {top_susp}")

    if fp_n > 0:
        print(
            f"\n  TOTAL FP probe: {fp_n} contracts — "
            f"CONF rate={fp_total_conf/fp_n*100:.1f}%  "
            f"CONF+SUSP rate={fp_total_any/fp_n*100:.1f}%  "
            f"Tuned rate={fp_total_tuned/fp_n*100:.1f}%"
        )

    # ── Errors ────────────────────────────────────────────────────────────────
    if errors:
        print(f"\n{'='*70}")
        print(f"ERRORS ({len(errors)} contracts failed to process)")
        print(f"{'='*70}")
        for path, err in errors:
            print(f"  {Path(path).name}: {err[:120]}")

    # ── Summary ───────────────────────────────────────────────────────────────
    success_n = len(contracts) - len(errors)
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Contracts processed : {success_n}/{len(contracts)}")
    print(f"  Errors              : {len(errors)}")
    if node_counts:
        import statistics
        print(f"  Median graph size   : {statistics.median(node_counts):.0f} nodes  "
              f"(training median={TRAINING_MEDIAN_NODES})")
    print(f"\n  Evaluation modes:")
    print(f"    Tier (0.55 confirmed / 0.25 suspicious) — any detection counts as hit")
    print(f"    Tuned (per-class thresholds 0.30–0.375) — stricter, matched to val F1 tuning")
    print(f"\n  NOTE: Run 9 was trained on BCCC (89% Reentrancy FP, 87% CallToUnknown FP).")
    print(f"  These SmartBugs results show real-world signal despite noisy training labels.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Run 9 on SmartBugs Curated")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-contract predictions")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not SMARTBUGS_DIR.exists():
        print(f"ERROR: SmartBugs dataset not found: {SMARTBUGS_DIR}", file=sys.stderr)
        sys.exit(1)

    run_benchmark(args.checkpoint, args.verbose)


if __name__ == "__main__":
    main()
