"""round_trip_v3.py — V3-aware round-trip test for Run 12 checkpoint.

Tests:
  1. Per-class known-positive contracts: predict and verify expected class is in CONFIRMED
  2. Known-safe contract: predict and verify NO CONFIRMED triggers
  3. FP probe: predict on NonVulnerable contracts; count false-positive rate

Uses ml/scripts/test_contracts/ hand-crafted contracts (20 contracts, covering
all 10 classes + safe + multilabel complex).

Output:
  - Console report
  - ml/reports/Run12_round_trip.json (machine-readable)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
sys.path.insert(0, str(REPO_ROOT))

from ml.src.inference.predictor import Predictor

# Hand-crafted test contracts (verified labels)
TEST_CONTRACTS_DIR = REPO_ROOT / "ml" / "scripts" / "test_contracts"

# Per-contract expected primary class (from test_contracts/ filename conventions)
TEST_CASES = [
    ("01_reentrancy_classic.sol", "Reentrancy"),
    ("02_reentrancy_tricky.sol", "Reentrancy"),
    ("03_integer_overflow.sol", "IntegerUO"),
    ("04_timestamp_dependence.sol", "Timestamp"),
    ("05_denial_of_service.sol", "DenialOfService"),
    ("06_mishandled_exception.sol", "MishandledException"),
    ("07_tx_order_dependence.sol", "TransactionOrderDependence"),
    ("08_unused_return.sol", "UnusedReturn"),
    ("09_call_to_unknown.sol", "CallToUnknown"),
    ("10_gas_exception.sol", "GasException"),
    ("11_external_bug.sol", "ExternalBug"),
    ("12_safe_contract.sol", None),  # Safe — no expected vuln
    ("13_multilabel_complex.sol", None),  # Multi-class — no single expected
    ("14_reentrancy_minimal.sol", "Reentrancy"),
    ("15_tod_minimal.sol", "TransactionOrderDependence"),
    ("16_gas_minimal.sol", "GasException"),
    ("17_integer_simple.sol", "IntegerUO"),
    ("18_safe_no_calls.sol", None),  # Safe
    ("19_safe_with_transfer.sol", None),  # Safe
    ("20_unused_return_minimal.sol", "UnusedReturn"),
]


def run_round_trip(checkpoint: str, thresholds: str, temperatures: str) -> dict:
    print(f"\n=== ROUND-TRIP TEST (Run 12) ===")
    print(f"Checkpoint: {checkpoint}")
    print(f"Thresholds: {thresholds}")
    print(f"Temperatures: {temperatures}")

    print("\nInitialising Predictor...")
    t0 = time.time()
    predictor = Predictor(
        checkpoint=checkpoint, threshold=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"  ready in {time.time()-t0:.1f}s")
    print(f"  architecture: {getattr(predictor, 'architecture', 'N/A')}")
    print(f"  thresholds_loaded: {getattr(predictor, 'thresholds_loaded', 'N/A')}")

    # Load tuned thresholds and temperatures
    tdata = json.loads(Path(thresholds).read_text())
    per_class_thresholds = tdata["thresholds"]
    temperatures_dict = json.loads(Path(temperatures).read_text())

    import numpy as np
    def apply_temperature(probs):
        calibrated = {}
        for cls, p in probs.items():
            T = temperatures_dict.get(cls, 1.0)
            if T == 1.0 or p in (0.0, 1.0):
                calibrated[cls] = p
                continue
            eps = 1e-7
            p = max(eps, min(1 - eps, p))
            logit = np.log(p / (1 - p))
            scaled_logit = logit / T
            calibrated[cls] = 1.0 / (1.0 + np.exp(-scaled_logit))
        return calibrated

    results = []
    n_positive_correct = 0
    n_positive_total = 0
    n_safe_clean = 0
    n_safe_total = 0
    n_fp = 0

    for filename, expected_class in TEST_CASES:
        path = TEST_CONTRACTS_DIR / filename
        if not path.exists():
            results.append({"name": filename, "status": "MISSING"})
            continue
        source = path.read_text()
        try:
            result = predictor.predict_source(source, name=filename)
            probs_raw = result["probabilities"]
            probs_cal = apply_temperature(probs_raw)
        except Exception as e:
            results.append({"name": filename, "status": "ERROR", "error": str(e)})
            continue

        # CONFIRMED tier: class with prob >= tier_confirmed_threshold (default 0.55)
        confirmed = [c for c, p in probs_cal.items() if p >= predictor.TIER_CONFIRMED_THRESHOLD]
        # Tuned threshold: class with prob >= per_class_thresholds[c]
        tuned_pred = [c for c, p in probs_cal.items() if p >= per_class_thresholds.get(c, 0.5)]

        # Score
        if expected_class is not None:
            n_positive_total += 1
            tier_hit = expected_class in confirmed
            tuned_hit = expected_class in tuned_pred
            if tier_hit or tuned_hit:
                n_positive_correct += 1
            status = "PASS" if (tier_hit or tuned_hit) else "MISS"
        else:
            # Safe contract — expect no CONFIRMED triggers
            n_safe_total += 1
            clean = len(confirmed) == 0
            if clean:
                n_safe_clean += 1
                status = "PASS"
            else:
                n_fp += 1
                status = "FP"

        results.append({
            "name": filename,
            "expected": expected_class,
            "confirmed": sorted(confirmed),
            "tuned_pred": sorted(tuned_pred),
            "status": status,
            "top_probs": dict(sorted(probs_cal.items(), key=lambda x: -x[1])[:3]),
        })

    # Report
    print(f"\n=== RESULTS ===")
    print(f"\nPositive round-trip ({n_positive_total} contracts with expected class):")
    print(f"  PASS: {n_positive_correct} / {n_positive_total} ({100*n_positive_correct/n_positive_total:.0f}%)")
    print(f"\nSafe round-trip ({n_safe_total} known-safe contracts):")
    print(f"  PASS: {n_safe_clean} / {n_safe_total} ({100*n_safe_clean/n_safe_total:.0f}%)")
    print(f"  FP (false positive): {n_fp}")

    print(f"\nPer-contract details:")
    for r in results:
        marker = "✓" if r["status"] == "PASS" else "✗" if r["status"] in ("MISS", "FP") else "?"
        expected = r.get("expected", "—")
        print(f"  {marker} {r['name']:38s} expected={str(expected):30s} status={r['status']}")
        if r["status"] in ("MISS", "FP"):
            print(f"      top probs: {r.get('top_probs', {})}")

    return {
        "n_positive_total": n_positive_total,
        "n_positive_correct": n_positive_correct,
        "positive_rate": n_positive_correct / n_positive_total if n_positive_total else 0,
        "n_safe_total": n_safe_total,
        "n_safe_clean": n_safe_clean,
        "fp_rate": n_fp / n_safe_total if n_safe_total else 0,
        "n_fp": n_fp,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt")
    parser.add_argument("--thresholds", default="ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best_thresholds.json")
    parser.add_argument("--temperatures", default="ml/calibration/temperatures_run12.json")
    parser.add_argument("--output", default="ml/reports/Run12_round_trip.json")
    args = parser.parse_args()

    report = run_round_trip(args.checkpoint, args.thresholds, args.temperatures)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\nReport written → {args.output}")


if __name__ == "__main__":
    main()
