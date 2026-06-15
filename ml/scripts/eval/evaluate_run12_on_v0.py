"""evaluate_v0_quickstart.py — Evaluate Run 12 on the v0.1 quickstart benchmark (66 contracts).

This is the FIRST truly honest OOD evaluation of any SENTINEL run, because:
- The 66 contracts are NOT in v3 training/val/test (verified by check_contamination_v3.py)
- The benchmark was built from honest OOD of SmartBugs + SolidiFI directories
- Per-class tuned thresholds (Phase 2.1) are applied
- Per-class temperature scaling (Phase 2.2) is applied for calibration

Outputs:
  - Console: per-class F1, tier F1, tuned F1, FP probe rate
  - ml/reports/Run12_benchmark_v0.1.json (machine-readable)

Limitations of v0.1:
  - Only 6 classes covered (CtU, ME, NonVuln, Reentrancy, Timestamp, ToD)
  - Other 4 classes (DoS, ExtBug, IntegerUO, UnusedReturn) have 0 contracts → F1 undefined
  - v1.0 (with Tier B/C/D/E) will have all 8+ classes; not built yet
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
sys.path.insert(0, str(REPO_ROOT))

# Re-use the v3-aware CLASS_NAMES from canonical schema
from data_module.sentinel_data.representation.graph_schema import CLASS_NAMES, NUM_CLASSES
from ml.src.inference.predictor import Predictor

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def load_benchmark_contracts(bench_dir: Path) -> list:
    """Load all contracts from the v0.1 quickstart benchmark."""
    contracts = []
    contracts_root = bench_dir / "contracts" / "by_class"
    if not contracts_root.exists():
        raise FileNotFoundError(f"No contracts/ in {bench_dir}")

    for cls_dir in sorted(contracts_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for sol in sorted(cls_dir.glob("*.sol")):
            contracts.append({
                "path": str(sol),
                "name": sol.name,
                "class": cls,
                "source": sol.read_text(),
            })
    return contracts


def load_thresholds(thresholds_path: Path) -> dict:
    """Load per-class tuned thresholds from Phase 2.1 output."""
    data = json.loads(thresholds_path.read_text())
    return data["thresholds"]


def load_temperatures(temps_path: Path) -> dict:
    """Load per-class temperatures from Phase 2.2 output."""
    return json.loads(temps_path.read_text())


def apply_temperature(probs: dict, temperatures: dict) -> dict:
    """Apply per-class temperature scaling.

    Since we have probabilities (not logits), apply T via:
      p_cal = sigmoid(logit(p) / T)
    Equivalent: p_cal = p^(1/T) / (p^(1/T) + (1-p)^(1/T))  [temperature scaling on probs]
    """
    calibrated = {}
    for cls, p in probs.items():
        T = temperatures.get(cls, 1.0)
        if T == 1.0 or p in (0.0, 1.0):
            calibrated[cls] = p
            continue
        # Inverse sigmoid to get logit
        eps = 1e-7
        p = max(eps, min(1 - eps, p))
        logit = np.log(p / (1 - p))
        scaled_logit = logit / T
        # Sigmoid
        calibrated[cls] = 1.0 / (1.0 + np.exp(-scaled_logit))
    return calibrated


def tier_mode(probs: dict, tier_confirmed: float, tier_suspicious: float) -> set:
    """Return classes with prob >= tier_suspicious (lenient mode)."""
    return {c for c, p in probs.items() if p >= tier_suspicious}


def tuned_mode(probs: dict, thresholds: dict) -> set:
    """Return classes with prob >= per-class tuned threshold."""
    return {c for c, p in probs.items() if p >= thresholds.get(c, 0.5)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-dir", default="data_module/benchmarks/benchmark_v0.1_quickstart")
    parser.add_argument("--thresholds", required=True, help="Path to <stem>_thresholds.json from Phase 2.1")
    parser.add_argument("--temperatures", required=True, help="Path to temperatures_run12.json from Phase 2.2")
    parser.add_argument("--tier-confirmed", type=float, default=0.55)
    parser.add_argument("--tier-suspicious", type=float, default=0.25)
    parser.add_argument("--output", help="Path to write JSON report")
    args = parser.parse_args()

    bench_dir = Path(args.benchmark_dir)
    print(f"Benchmark: {bench_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Temperatures: {args.temperatures}")

    # Load resources
    print("\nLoading benchmark contracts...")
    contracts = load_benchmark_contracts(bench_dir)
    print(f"  loaded: {len(contracts)} contracts")
    by_class = Counter(c["class"] for c in contracts)
    for cls, n in sorted(by_class.items()):
        print(f"    {cls:30s}: {n}")

    print("\nLoading thresholds + temperatures...")
    thresholds = load_thresholds(Path(args.thresholds))
    temperatures = load_temperatures(Path(args.temperatures))
    print(f"  thresholds: {thresholds}")
    print(f"  temperatures: {temperatures}")

    print("\nInitialising Predictor (this loads the model + warms up)...")
    t0 = time.time()
    predictor = Predictor(
        checkpoint=args.checkpoint,
        threshold=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"  predictor ready in {time.time()-t0:.1f}s")
    print(f"  thresholds_loaded: {getattr(predictor, 'thresholds_loaded', 'N/A')}")
    print(f"  architecture: {getattr(predictor, 'architecture', 'N/A')}")

    # Run inference on each contract
    print(f"\nRunning inference on {len(contracts)} contracts...")
    y_true_per_class = {c: [] for c in CLASS_NAMES}
    y_pred_tier = {c: [] for c in CLASS_NAMES}
    y_pred_tuned = {c: [] for c in CLASS_NAMES}
    raw_probs = []
    per_contract = []

    for i, c in enumerate(contracts):
        try:
            result = predictor.predict_source(c["source"], name=c["name"])
            probs = result["probabilities"]
            # Apply temperature
            probs_cal = apply_temperature(probs, temperatures)

            # Per-class predictions
            tier_pred = tier_mode(probs_cal, args.tier_confirmed, args.tier_suspicious)
            tuned_pred = tuned_mode(probs_cal, thresholds)

            # True class
            true_class = c["class"]
            for cls in CLASS_NAMES:
                y_true_per_class[cls].append(1 if cls == true_class else 0)
                y_pred_tier[cls].append(1 if cls in tier_pred else 0)
                y_pred_tuned[cls].append(1 if cls in tuned_pred else 0)

            raw_probs.append({"name": c["name"], "true": true_class, "probs": probs, "probs_cal": probs_cal})
            per_contract.append({
                "name": c["name"], "true_class": true_class,
                "tier_pred": sorted(tier_pred), "tuned_pred": sorted(tuned_pred),
                "probs": {k: round(v, 4) for k, v in probs.items()},
            })
        except Exception as e:
            print(f"  ERROR on {c['name']}: {e}")
            continue
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(contracts)}")

    # Compute per-class F1
    print(f"\n=== PER-CLASS F1 (with temperature + tuned thresholds) ===")
    print(f"{'Class':<28} | {'Support':>7} | {'Tier F1':>8} | {'Tuned F1':>9} | Notes")
    print("-" * 78)
    per_class_f1 = {}
    for cls in CLASS_NAMES:
        y_t = np.array(y_true_per_class[cls])
        y_p_tier = np.array(y_pred_tier[cls])
        y_p_tuned = np.array(y_pred_tuned[cls])
        support = int(y_t.sum())
        if support == 0:
            per_class_f1[cls] = {"tier_f1": None, "tuned_f1": None, "support": 0, "in_benchmark": False}
            print(f"{cls:<28} | {support:>7} | {'N/A':>8} | {'N/A':>9} | NOT IN BENCHMARK (0 honest OOD contracts)")
            continue
        f1_tier = float(f1_score(y_t, y_p_tier, zero_division=0))
        f1_tuned = float(f1_score(y_t, y_p_tuned, zero_division=0))
        per_class_f1[cls] = {"tier_f1": f1_tier, "tuned_f1": f1_tuned, "support": support, "in_benchmark": True}
        notes = "OVERFIT" if support <= 10 and f1_tuned > 0.9 else ("small sample" if support < 30 else "")
        print(f"{cls:<28} | {support:>7} | {f1_tier:>8.3f} | {f1_tuned:>9.3f} | {notes}")

    # Overall metrics (only on classes that are IN the benchmark)
    in_bench_classes = [c for c in CLASS_NAMES if per_class_f1[c]["in_benchmark"]]
    if in_bench_classes:
        # Micro F1: flatten all per-class predictions
        y_t_all = np.concatenate([y_true_per_class[c] for c in in_bench_classes])
        y_p_tier_all = np.concatenate([y_pred_tier[c] for c in in_bench_classes])
        y_p_tuned_all = np.concatenate([y_pred_tuned[c] for c in in_bench_classes])
        f1_tier_micro = float(f1_score(y_t_all, y_p_tier_all, average="micro", zero_division=0))
        f1_tuned_micro = float(f1_score(y_t_all, y_p_tuned_all, average="micro", zero_division=0))
        # Macro F1: mean of per-class F1 (only in-bench classes)
        f1_tier_macro = float(np.mean([per_class_f1[c]["tier_f1"] for c in in_bench_classes]))
        f1_tuned_macro = float(np.mean([per_class_f1[c]["tuned_f1"] for c in in_bench_classes]))
        print(f"\n=== OVERALL (in-benchmark classes only: {len(in_bench_classes)}) ===")
        print(f"  F1-macro (tier):  {f1_tier_macro:.4f}")
        print(f"  F1-macro (tuned): {f1_tuned_macro:.4f}")
        print(f"  F1-micro (tier):  {f1_tier_micro:.4f}")
        print(f"  F1-micro (tuned): {f1_tuned_micro:.4f}")
    else:
        f1_tier_macro = f1_tuned_macro = f1_tier_micro = f1_tuned_micro = 0.0

    # FP probe check (unmapped categories → NonVulnerable in our mapping)
    nonvuln_tier = []
    nonvuln_tuned = []
    n_nonvuln_errors = 0
    for c in contracts:
        if c["class"] == "NonVulnerable":
            try:
                result = predictor.predict_source(c["source"], name=c["name"])
                probs_cal = apply_temperature(result["probabilities"], temperatures)
                tier_pred = tier_mode(probs_cal, args.tier_confirmed, args.tier_suspicious)
                tuned_pred = tuned_mode(probs_cal, thresholds)
                # If model says "vulnerable" on a NonVulnerable contract, that's a false positive
                nonvuln_tier.append(1 if tier_pred and tier_pred != {"NonVulnerable"} else 0)
                nonvuln_tuned.append(1 if tuned_pred and tuned_pred != {"NonVulnerable"} else 0)
            except Exception as e:
                n_nonvuln_errors += 1
                print(f"  (FP probe) ERROR on {c['name']}: {e}")
                continue
    print(f"\n=== FP PROBE (NonVulnerable contracts) ===")
    print(f"  Tier FP rate:  {100*np.mean(nonvuln_tier):.1f}%  ({sum(nonvuln_tier)}/{len(nonvuln_tier)} NonVulnerable contracts had >=1 tier trigger)")
    print(f"  Tuned FP rate: {100*np.mean(nonvuln_tuned):.1f}%  ({sum(nonvuln_tuned)}/{len(nonvuln_tuned)} NonVulnerable contracts had >=1 tuned trigger)")

    # Write report (always, even if some contracts errored)
    if args.output:
        try:
            report = {
                "benchmark": str(bench_dir),
                "checkpoint": str(args.checkpoint),
                "thresholds_file": str(args.thresholds),
                "temperatures_file": str(args.temperatures),
                "n_contracts": len(contracts),
                "n_evaluated": len(per_contract),
                "n_errors": sum(1 for c in contracts if c not in [r for r in per_contract]),
                "per_class_counts": dict(by_class),
                "per_class_f1": per_class_f1,
                "in_benchmark_classes": in_bench_classes,
                "f1_tier_macro": f1_tier_macro,
                "f1_tuned_macro": f1_tuned_macro,
                "f1_tier_micro": f1_tier_micro,
                "f1_tuned_micro": f1_tuned_micro,
                "fp_probe_tier_rate": float(np.mean(nonvuln_tier)) if nonvuln_tier else None,
                "fp_probe_tuned_rate": float(np.mean(nonvuln_tuned)) if nonvuln_tuned else None,
                "per_contract": per_contract,
            }
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(f"\nReport written → {args.output}")
        except Exception as e:
            print(f"ERROR writing report: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
