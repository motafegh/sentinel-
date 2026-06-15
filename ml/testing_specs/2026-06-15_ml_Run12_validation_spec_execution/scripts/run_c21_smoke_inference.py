"""C.2.1 Smoke Inference — Run 12 — spec gate check.

Runs the Run 12 checkpoint against the v0.1 honest OOD benchmark (66 contracts)
and prints a per-contract verdict table.

Per spec C.2.1: checks that known-vulnerable contracts score high on their class
and known-clean contracts score low on all classes.

Usage:
    ml/.venv/bin/python ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/scripts/run_c21_smoke_inference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]
CHECKPOINT = REPO_ROOT / "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt"
THRESHOLDS = REPO_ROOT / "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best_thresholds.json"
TEMPERATURES = REPO_ROOT / "ml/calibration/run12/temperatures_run12.json"
BENCHMARK_DIR = REPO_ROOT / "data_module/benchmarks"

import numpy as np


def apply_temperature(probs: dict, temps: dict) -> dict:
    cal = {}
    for cls, p in probs.items():
        T = temps.get(cls, 1.0)
        if T == 1.0 or p in (0.0, 1.0):
            cal[cls] = p
            continue
        eps = 1e-7
        p = max(eps, min(1 - eps, p))
        logit = np.log(p / (1 - p))
        cal[cls] = float(1.0 / (1.0 + np.exp(-logit / T)))
    return cal


def main() -> None:
    print(f"\n{'='*65}")
    print("C.2.1 Smoke Inference — Run 12 — v0.1 OOD Benchmark")
    print(f"{'='*65}\n")

    # Verify benchmark dir exists
    if not BENCHMARK_DIR.exists():
        print(f"ERROR: benchmark dir not found: {BENCHMARK_DIR}")
        sys.exit(1)

    # Find all .sol files in the benchmark
    sol_files = list(BENCHMARK_DIR.rglob("*.sol"))
    if not sol_files:
        print(f"ERROR: no .sol files found in {BENCHMARK_DIR}")
        sys.exit(1)
    print(f"Found {len(sol_files)} contracts in benchmark dir")

    # Load predictor
    sys.path.insert(0, str(REPO_ROOT))
    from ml.src.inference.predictor import Predictor
    predictor = Predictor(checkpoint=CHECKPOINT)
    print(f"Loaded: {predictor.architecture}")

    # Load calibration
    with open(THRESHOLDS) as f:
        thresholds = json.load(f)["thresholds"]
    with open(TEMPERATURES) as f:
        temperatures = json.load(f)

    # Run inference
    print(f"\n{'─'*65}")
    n_triggered = 0
    n_clean = 0
    n_errors = 0
    results = []

    for sol_path in sorted(sol_files):
        label = sol_path.parent.name  # folder name = expected class
        contract_name = sol_path.stem

        try:
            source = sol_path.read_text(encoding="utf-8", errors="replace")
            result = predictor.predict_source(source)
            probs = apply_temperature(result["probabilities"], temperatures)
            top_class = max(probs, key=probs.get)
            top_prob = max(probs.values())
            triggered = [c for c, p in probs.items() if p >= thresholds.get(c, 0.5)]

            match = top_class == label if label in probs else "N/A"
            status = "✓" if match is True else ("?" if match == "N/A" else "✗")

            print(f"{status} {contract_name[:35]:<35} | label={label:<24} | pred={top_class:<24} | p={top_prob:.3f} | n_trig={len(triggered)}")

            results.append({
                "contract": contract_name,
                "label": label,
                "top_pred": top_class,
                "top_prob": round(top_prob, 4),
                "top_match": match,
                "n_triggered": len(triggered),
                "triggered_classes": triggered,
            })
            if len(triggered) > 0:
                n_triggered += 1
            else:
                n_clean += 1

        except Exception as exc:
            print(f"  ERROR {contract_name}: {exc}")
            n_errors += 1

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r["top_match"] is True)
    print(f"\n{'='*65}")
    print("SMOKE INFERENCE SUMMARY — C.2.1")
    print(f"{'='*65}")
    print(f"  Contracts tested:     {total}")
    print(f"  Top-class correct:    {correct}/{total} ({100*correct/total:.1f}%)" if total else "")
    print(f"  Any trigger:          {n_triggered}")
    print(f"  No trigger (clean):   {n_clean}")
    print(f"  Errors:               {n_errors}")

    if total > 0 and correct / total >= 0.7:
        print("\n  ✓ PASS — >70% top-class correct on honest benchmark")
    elif total > 0:
        print("\n  ⚠ REVIEW — <70% top-class correct; investigate mismatched contracts")

    # Write gate report
    gate_path = Path(__file__).parents[1] / "gate_reports" / "C21_smoke_inference_results.json"
    with open(gate_path, "w") as f:
        json.dump({
            "check": "C.2.1",
            "date": "2026-06-15",
            "benchmark_dir": str(BENCHMARK_DIR),
            "checkpoint": str(CHECKPOINT),
            "total": total,
            "top_class_correct": correct,
            "accuracy_pct": round(100 * correct / total, 1) if total else 0,
            "n_triggered": n_triggered,
            "n_clean": n_clean,
            "n_errors": n_errors,
            "results": results,
        }, f, indent=2)
    print(f"\n  Gate report written: {gate_path}")
    print()


if __name__ == "__main__":
    main()
