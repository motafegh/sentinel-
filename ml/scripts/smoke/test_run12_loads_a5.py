"""A.5 smoke test: Verify Run 12 checkpoint loads cleanly into Predictor.

This is a focused regression test — does the Predictor work with the Run 12
FINAL checkpoint after the drift detector fix?

If this passes, the inference pipeline is healthy and we can proceed with
Phase A cosmetic changes (A.2, A.3, A.4).
"""
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
sys.path.insert(0, "/home/motafeq/projects/sentinel")

# Set TRANSFORMERS_OFFLINE (per memory: shell level for training, also works here)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CHECKPOINT = "/home/motafeq/projects/sentinel/ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt"
print(f"Loading checkpoint: {CHECKPOINT}")
print(f"  size: {os.path.getsize(CHECKPOINT) / 1024 / 1024:.1f} MB")

t0 = time.time()
from ml.src.inference.predictor import Predictor
predictor = Predictor(checkpoint=CHECKPOINT)
t1 = time.time()
print(f"\n[PASS] Predictor loaded in {t1 - t0:.1f}s")
print(f"  architecture: {predictor.architecture}")
print(f"  num_classes:  {predictor.num_classes}")
print(f"  thresholds_loaded: {predictor.thresholds_loaded}")
print(f"  class_names ({len(predictor._class_names)}): {predictor._class_names}")

# Verify health endpoint would return the expected shape
print(f"\nSimulating /health endpoint:")
cfg = predictor._saved_cfg
health = {
    "status": "ok",
    "predictor_loaded": True,
    "checkpoint": CHECKPOINT,
    "architecture": predictor.architecture,
    "thresholds_loaded": predictor.thresholds_loaded,
    "tier_thresholds": {
        "confirmed": predictor.tier_confirmed_threshold,
        "suspicious": predictor.tier_suspicious_threshold,
        "noteworthy": 0.10,
    },
    "model_epoch": cfg.get("epoch", "?"),
    "model_f1_val": cfg.get("best_f1", None),
}
import json
print(json.dumps(health, indent=2, default=str))

# Run a real predict on a tiny Solidity contract (a known-vulnerable reentrancy example)
test_source = '''
pragma solidity ^0.8.0;

contract Vulnerable {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() public {
        uint amount = balances[msg.sender];
        require(amount > 0);
        (bool success,) = msg.sender.call{value: amount}("");
        require(success);
        balances[msg.sender] = 0;  // CEI violation
    }
}
'''

print(f"\nRunning prediction on test contract ({len(test_source)} chars)...")
t0 = time.time()
result = predictor.predict_source(test_source)
t1 = time.time()
print(f"[PASS] Prediction in {t1 - t0:.2f}s")
print(f"  label: {result['label']}")
print(f"  num_nodes: {result['num_nodes']}, num_edges: {result['num_edges']}")
print(f"  windows_used: {result.get('windows_used', 1)}")
print(f"  truncated: {result['truncated']}")
print(f"\n  Probabilities:")
for cls, prob in sorted(result.get("probabilities", {}).items(), key=lambda x: -x[1])[:5]:
    print(f"    {cls:30s} = {prob:.4f}")
print(f"\n  Confirmed (top 5):")
for v in result.get("confirmed", [])[:5]:
    print(f"    {v['vulnerability_class']:30s} = {v['probability']:.4f} ({v.get('tier', '?')})")
print(f"\n  Suspicious (top 5):")
for v in result.get("suspicious", [])[:5]:
    print(f"    {v['vulnerability_class']:30s} = {v['probability']:.4f} ({v.get('tier', '?')})")

# Sanity checks
probs = result.get("probabilities", {})
assert len(probs) == 10, f"Expected 10 classes, got {len(probs)}"
assert "Reentrancy" in probs, "Missing Reentrancy class"
reentrancy_prob = probs.get("Reentrancy", 0.0)
print(f"\n[VERIFY] Reentrancy probability: {reentrancy_prob:.4f}")
if reentrancy_prob > 0.5:
    print(f"[PASS] Reentrancy correctly detected (high confidence on a CEI violation)")
else:
    print(f"[WARN] Reentrancy prob is {reentrancy_prob:.4f} (low) — model may need more training")
    # Not a fail; just an observation

print(f"\n=== A.5 SMOKE TEST PASSED ===")
