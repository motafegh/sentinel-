"""famous_contracts_test.py — Predict on famous compromised contracts.

These are well-known contracts with public vulnerability histories:
  - The DAO (0xbb9bc...) — reentrancy bug, $60M hack 2016
  - EtherDelta (0x8d12a...) — reentrancy-like bug, hacked 2017
  - Parity MultiSigWalletWithDailyLimit (0x242a...) — reentrancy (initWallet/initMultiowned)
  - WithdrawDAO (0xbf4e...) — related to DAO hack
  - DAOToken (0x543f...)
  - Hacken (0x9e6b...) — security firm (likely clean)

For each: predict, then compare to known vulnerability history.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/motafeq/projects/sentinel")
sys.path.insert(0, str(REPO_ROOT))

from ml.src.inference.predictor import Predictor

# Famous contracts (well-known in security research)
# Format: (address, expected_primary_vuln, description)
FAMOUS_CONTRACTS = [
    ("0xbb9bc244d798123fde783fcc1c72d3bb8c189413", "Reentrancy", "The DAO (2016 reentrancy hack, $60M)"),
    ("0x8d12a197cb00d4747a1fe03395095ce2a5cc6819", "Reentrancy", "EtherDelta (2017 hack)"),
    ("0x242aa8c63aab36df59ce19aaccd020fe4114c349", "Reentrancy", "Parity MultiSigWalletWithDailyLimit (initWallet bug)"),
    ("0xbf4ed7b27f1d666546e30d74d50d173d20bca754", "Reentrancy", "WithdrawDAO (DAO-related)"),
    ("0x543ff227f64aa17ea132bf9886cab5db55dcaddf", "Reentrancy", "DAOToken (DAO-related)"),
    ("0x4c0ff1b2c1ef5e2b3dd2c74023cbf7ae36f01391", "ExternalBug", "Generic Wallet (access control)"),
    ("0x6f6deb5db0c4994a8283a01d6cfeeb27fc3bbe9c", "Reentrancy", "SmartBillions (exploited)"),
    ("0x9e6b2b11542f2bc52f3029077ace37e8fd838d7f", None, "Hacken (security firm, likely clean)"),
    ("0x0705ba621e97bef9e857b18d260726e99099a2a4", "ExternalBug", "UserWallet (access control pattern)"),
    ("0x2ff99437f01dbc064426aafe21769de93de74ec6", "ExternalBug", "Generic Wallet (pre-0.4)"),
]

CONTRACTS_DIR = REPO_ROOT / "ml" / "data" / "smartbugs-wild" / "contracts"


def run_famous_test():
    print(f"\n=== FAMOUS CONTRACTS TEST ===")
    print(f"Source: {CONTRACTS_DIR}")

    print("\nInitialising Predictor...")
    t0 = time.time()
    predictor = Predictor(
        checkpoint="ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt",
        threshold=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"  ready in {time.time()-t0:.2f}s")

    # Load tuned thresholds + temperatures
    thresholds = json.loads(Path("ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best_thresholds.json").read_text())["thresholds"]
    temperatures = json.loads(Path("ml/calibration/temperatures_run12.json").read_text())

    def apply_temperature(probs):
        calibrated = {}
        for cls, p in probs.items():
            T = temperatures.get(cls, 1.0)
            if T == 1.0 or p in (0.0, 1.0):
                calibrated[cls] = p
                continue
            eps = 1e-7
            p = max(eps, min(1 - eps, p))
            logit = np.log(p / (1 - p))
            calibrated[cls] = 1.0 / (1.0 + np.exp(-logit / T))
        return calibrated

    print(f"\n=== FAMOUS CONTRACTS PREDICTION ===")
    results = []
    n_pass = 0
    for addr, expected, desc in FAMOUS_CONTRACTS:
        path = CONTRACTS_DIR / f"{addr}.sol"
        if not path.exists():
            print(f"  SKIP: {addr}.sol not found")
            continue
        try:
            source = path.read_text()
            result = predictor.predict_source(source, name=path.name)
            probs = apply_temperature(result["probabilities"])

            # Top 3
            top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
            top_class = top3[0][0]
            top_prob = top3[0][1]

            # Tuned prediction (>= per-class threshold)
            tuned_pred = [c for c, p in probs.items() if p >= thresholds.get(c, 0.5)]
            tier_pred = [c for c, p in probs.items() if p >= predictor.TIER_SUSPICIOUS_THRESHOLD]

            # Score
            status = "—"
            if expected is not None:
                if expected in tuned_pred:
                    status = "✓ PASS"
                    n_pass += 1
                elif expected in tier_pred:
                    status = "⚠ tier-only"
                else:
                    status = "✗ MISS"

            print(f"\n  {addr}  ({desc})")
            print(f"    expected: {expected or 'clean'}")
            print(f"    top: {top_class} ({top_prob:.3f})")
            print(f"    top 3: {[(c, f'{p:.3f}') for c, p in top3]}")
            print(f"    tier (>=0.25): {tier_pred}")
            print(f"    tuned (per-class): {tuned_pred}")
            print(f"    status: {status}")

            results.append({
                "address": addr,
                "description": desc,
                "expected": expected,
                "top_class": top_class,
                "top_prob": top_prob,
                "top_3": top3,
                "tier_pred": tier_pred,
                "tuned_pred": tuned_pred,
                "status": status,
            })
        except Exception as e:
            print(f"  ERROR on {addr}: {e}")
            results.append({"address": addr, "status": "ERROR", "error": str(e)[:200]})

    print(f"\n=== SUMMARY ===")
    n_total = sum(1 for r in results if r.get("expected") is not None)
    print(f"  {n_pass}/{n_total} famous vulnerable contracts correctly flagged (tuned)")
    print(f"  {n_total - n_pass}/{n_total} missed")

    return {"results": results, "n_pass": n_pass, "n_total": n_total}


if __name__ == "__main__":
    report = run_famous_test()
    Path("ml/reports/Run12_famous_contracts.json").parent.mkdir(parents=True, exist_ok=True)
    Path("ml/reports/Run12_famous_contracts.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written → ml/reports/Run12_famous_contracts.json")
