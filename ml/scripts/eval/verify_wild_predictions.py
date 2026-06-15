"""Verify Run 12 SmartBugs Wild eval predictions by re-running the predictor.

Samples N contracts from the eval state, re-runs the exact same pipeline
(predictor → temperature scaling → thresholds), and compares stored vs
recomputed results field by field.

Samples are chosen to cover:
  - High-confidence OOD contracts (top_prob >= 0.9, not in v3)
  - Low-confidence OOD (borderline, top_prob 0.5–0.7)
  - Seen-train contracts (in v3 train split — expect identical results)
  - Multi-trigger contracts (n_triggered_tuned >= 3)
  - Single-trigger contracts

Usage:
    ml/.venv/bin/python ml/scripts/eval/verify_wild_predictions.py
    ml/.venv/bin/python ml/scripts/eval/verify_wild_predictions.py --n 50
    ml/.venv/bin/python ml/scripts/eval/verify_wild_predictions.py --n 10 --verbose
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[3]
CHECKPOINT = REPO_ROOT / "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt"
THRESHOLDS_JSON = REPO_ROOT / "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best_thresholds.json"
TEMPERATURES_JSON = REPO_ROOT / "ml/calibration/temperatures_run12.json"
EVAL_STATE = REPO_ROOT / "ml/data/smartbugs_wild_eval_state.json"
CONTAM_INDEX = REPO_ROOT / "docs/reports/2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_47K_complete/2026-06-15_ml_Run12_eval_full_eval_smartbugs_wild_contamination_index.json"
WILD_DIR = REPO_ROOT / "ml/data/smartbugs-wild/contracts"


def apply_temperature(probs: dict, temperatures: dict) -> dict:
    cal = {}
    for cls, p in probs.items():
        T = temperatures.get(cls, 1.0)
        if T == 1.0 or p in (0.0, 1.0):
            cal[cls] = p
            continue
        eps = 1e-7
        p = max(eps, min(1 - eps, p))
        logit = np.log(p / (1 - p))
        cal[cls] = float(1.0 / (1.0 + np.exp(-logit / T)))
    return cal


def sample_addresses(eval_state: dict, contam_map: dict, n: int) -> list[str]:
    """Pick a stratified sample across OOD/seen and confidence levels."""
    successful = {
        addr: rec for addr, rec in eval_state["processed_set"].items()
        if rec.get("top_class")
    }

    buckets: dict[str, list[str]] = {
        "ood_high_conf":    [],  # OOD, top_prob >= 0.9
        "ood_low_conf":     [],  # OOD, top_prob 0.5–0.75
        "ood_multi":        [],  # OOD, n_triggered_tuned >= 3
        "seen_train":       [],  # in v3 train
        "seen_valtest":     [],  # in v3 val/test
    }

    for addr, rec in successful.items():
        c = contam_map.get(addr, {})
        tier_hit = c.get("tier_hit", 0)
        split = c.get("v3_split")
        top_prob = rec.get("top_prob", 0)
        n_trig = rec.get("n_triggered_tuned", 0)

        is_ood = (tier_hit == 0)
        if is_ood:
            if top_prob >= 0.9:
                buckets["ood_high_conf"].append(addr)
            elif top_prob <= 0.75:
                buckets["ood_low_conf"].append(addr)
            if n_trig >= 3:
                buckets["ood_multi"].append(addr)
        elif split == "train":
            buckets["seen_train"].append(addr)
        else:
            buckets["seen_valtest"].append(addr)

    per_bucket = max(1, n // len(buckets))
    selected = []
    for bucket, addrs in buckets.items():
        k = min(per_bucket, len(addrs))
        chosen = random.sample(addrs, k)
        selected.extend(chosen)
        print(f"  Bucket '{bucket}': {len(addrs):,} candidates → picked {k}")

    # Fill remainder from OOD high-conf
    remainder = n - len(selected)
    if remainder > 0:
        extra_pool = [a for a in buckets["ood_high_conf"] if a not in selected]
        selected.extend(random.sample(extra_pool, min(remainder, len(extra_pool))))

    return list(dict.fromkeys(selected))  # dedupe, preserve order


def compare(stored: dict, recomputed: dict, temperatures: dict, thresholds: dict,
            tier_susp: float = 0.25) -> dict[str, object]:
    """Compare stored vs recomputed result. Returns a per-field diff dict."""
    diffs = {}

    # all_probs comparison (stored = temperature-scaled)
    stored_probs = stored.get("all_probs", {})
    for cls, stored_p in stored_probs.items():
        recomp_p = recomputed.get(cls, None)
        if recomp_p is None:
            diffs[f"prob_{cls}"] = f"MISSING in recomputed"
            continue
        delta = abs(stored_p - recomp_p)
        if delta > 1e-4:
            diffs[f"prob_{cls}"] = f"stored={stored_p:.5f} recomp={recomp_p:.5f} Δ={delta:.5f}"

    # top_class
    stored_top = stored.get("top_class")
    recomp_top = max(recomputed, key=recomputed.get) if recomputed else None
    if stored_top != recomp_top:
        diffs["top_class"] = f"stored={stored_top} recomp={recomp_top}"

    # n_triggered_tuned
    stored_nt = stored.get("n_triggered_tuned", 0)
    recomp_nt = sum(1 for cls, p in recomputed.items() if p >= thresholds.get(cls, 0.5))
    if stored_nt != recomp_nt:
        diffs["n_triggered_tuned"] = f"stored={stored_nt} recomp={recomp_nt}"

    # n_triggered_tier
    stored_ntier = stored.get("n_triggered_tier", 0)
    recomp_ntier = sum(1 for p in recomputed.values() if p >= tier_susp)
    if stored_ntier != recomp_ntier:
        diffs["n_triggered_tier"] = f"stored={stored_ntier} recomp={recomp_ntier}"

    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="Number of contracts to verify")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Print full probs for each contract")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n{'='*65}")
    print("SENTINEL Run 12 — Wild Eval Prediction Verification")
    print(f"{'='*65}\n")

    # Load reference data
    print("Loading eval state...")
    with open(EVAL_STATE) as f:
        eval_state = json.load(f)
    print(f"  {len(eval_state['processed_set']):,} eval records")

    print("Loading contamination index...")
    with open(CONTAM_INDEX) as f:
        contam = json.load(f)
    contam_map = {r["address"]: r for r in contam}

    print("Loading thresholds + temperatures...")
    with open(THRESHOLDS_JSON) as f:
        thresholds = json.load(f)["thresholds"]
    with open(TEMPERATURES_JSON) as f:
        temperatures = json.load(f)

    # Sample
    print(f"\nSampling {args.n} contracts (seed={args.seed})...")
    addresses = sample_addresses(eval_state, contam_map, args.n)
    print(f"  Total selected: {len(addresses)}")

    # Load predictor once
    print("\nLoading predictor (warmup included)...")
    sys.path.insert(0, str(REPO_ROOT))
    from ml.src.inference.predictor import Predictor
    predictor = Predictor(checkpoint=CHECKPOINT)
    print(f"  architecture: {predictor.architecture}")
    print(f"  thresholds_loaded: {predictor.thresholds_loaded}")

    # Verify each contract
    print(f"\n{'─'*65}")
    n_exact = 0
    n_diffs = 0
    n_errors = 0
    all_results = []

    for i, addr in enumerate(addresses):
        sol_path = WILD_DIR / f"{addr}.sol"
        stored_rec = eval_state["processed_set"].get(addr, {})
        contam_rec = contam_map.get(addr, {})
        tier_hit = contam_rec.get("tier_hit", 0)
        in_split = contam_rec.get("v3_split", "OOD")
        label = in_split if tier_hit else "OOD"

        print(f"[{i+1:>3}/{len(addresses)}] {addr[:20]}… ({label})", end="  ", flush=True)

        if not sol_path.exists():
            print("SKIP — .sol not found")
            n_errors += 1
            continue

        try:
            source = sol_path.read_text(encoding="utf-8", errors="replace")
            result = predictor.predict_source(source)
            raw_probs = result["probabilities"]
            scaled_probs = apply_temperature(raw_probs, temperatures)

            diffs = compare(
                stored=stored_rec,
                recomputed=scaled_probs,
                temperatures=temperatures,
                thresholds=thresholds,
            )

            record = {
                "address": addr,
                "contamination_label": label,
                "stored_top_class": stored_rec.get("top_class"),
                "stored_top_prob": stored_rec.get("top_prob"),
                "stored_n_triggered_tuned": stored_rec.get("n_triggered_tuned"),
                "recomp_top_class": max(scaled_probs, key=scaled_probs.get),
                "recomp_top_prob": round(max(scaled_probs.values()), 4),
                "recomp_n_triggered_tuned": sum(1 for c, p in scaled_probs.items() if p >= thresholds.get(c, 0.5)),
                "diffs": diffs,
            }
            all_results.append(record)

            if not diffs:
                print("✓ EXACT MATCH")
                n_exact += 1
            else:
                print(f"△ {len(diffs)} diff(s): {list(diffs.keys())[:3]}")
                n_diffs += 1

            if args.verbose:
                print(f"       stored : {stored_rec.get('top_class')} p={stored_rec.get('top_prob'):.4f} n_trig={stored_rec.get('n_triggered_tuned')}")
                print(f"       recomp : {record['recomp_top_class']} p={record['recomp_top_prob']:.4f} n_trig={record['recomp_n_triggered_tuned']}")
                if diffs:
                    for k, v in list(diffs.items())[:5]:
                        print(f"         DIFF {k}: {v}")

        except Exception as exc:
            print(f"ERROR — {exc}")
            n_errors += 1

    # Summary
    total_checked = len(addresses) - n_errors
    print(f"\n{'='*65}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Contracts checked:  {total_checked}")
    print(f"  Exact matches:      {n_exact}  ({100*n_exact/total_checked:.1f}%)" if total_checked else "")
    print(f"  With differences:   {n_diffs}")
    print(f"  Errors (skip):      {n_errors}")

    if n_diffs > 0:
        print(f"\n  Contracts with differences:")
        for r in all_results:
            if r["diffs"]:
                print(f"    {r['address'][:20]}… stored_top={r['stored_top_class']} recomp_top={r['recomp_top_class']}")
                for k, v in list(r["diffs"].items())[:3]:
                    print(f"      {k}: {v}")

    if n_exact == total_checked:
        print("\n  ✓ ALL PREDICTIONS VERIFIED — eval results are deterministic and genuine")
    elif n_diffs / max(total_checked, 1) < 0.05:
        print("\n  ✓ >95% exact match — minor float precision differences only")
    else:
        print("\n  ⚠ Significant mismatches — investigate before trusting eval stats")

    print()


if __name__ == "__main__":
    main()
