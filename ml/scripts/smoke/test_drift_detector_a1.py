"""Smoke test for A.1 — drift detector baseline validation fix."""
import json
import logging
import os
import sys
import tempfile

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.insert(0, "/home/motafeq/projects/sentinel")

from ml.src.inference.drift_detector import DriftDetector, _KNOWN_STAT_NAMES


def test(name, expected_baseline_is_none, expected_warmup_done, baseline_path, baseline_data=None):
    print(f"\n=== {name} ===")
    if baseline_data is not None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(baseline_data, f)
            path = f.name
    else:
        path = baseline_path
    d = DriftDetector(baseline_path=path)
    actual_baseline_is_none = d._baseline is None
    actual_warmup_done = d._warmup_done
    pass_baseline = actual_baseline_is_none == expected_baseline_is_none
    pass_warmup = actual_warmup_done == expected_warmup_done
    print(f"  _baseline=None: expected={expected_baseline_is_none}, got={actual_baseline_is_none} -> {'PASS' if pass_baseline else 'FAIL'}")
    print(f"  _warmup_done  : expected={expected_warmup_done}, got={actual_warmup_done} -> {'PASS' if pass_warmup else 'FAIL'}")
    if baseline_data is not None and os.path.exists(path):
        os.unlink(path)
    return pass_baseline and pass_warmup


print(f"KNOWN_STAT_NAMES: {sorted(_KNOWN_STAT_NAMES)}")

results = []
# Test 1: Placeholder baseline (the actual file on disk)
results.append(test(
    "Test 1: Placeholder baseline (current state)",
    expected_baseline_is_none=True,
    expected_warmup_done=False,
    baseline_path="/home/motafeq/projects/sentinel/ml/data/drift_baseline.json",
))

# Test 2: Real warmup baseline
results.append(test(
    "Test 2: Real warmup baseline (synthetic)",
    expected_baseline_is_none=False,
    expected_warmup_done=True,
    baseline_path=None,
    baseline_data={
        "source": "warmup",
        "num_nodes": [100.0, 200.0, 150.0] * 10,
        "num_edges": [50.0, 80.0, 70.0] * 10,
        "confirmed_count": [0.0, 1.0, 2.0] * 10,
        "suspicious_count": [1.0, 2.0, 3.0] * 10,
    },
))

# Test 3: Non-dict baseline
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    f.write("[1, 2, 3]")
    list_path = f.name
print(f"\n=== Test 3: Non-dict baseline (edge case) ===")
d3 = DriftDetector(baseline_path=list_path)
print(f"  _baseline=None: expected=True, got={d3._baseline is None} -> {'PASS' if d3._baseline is None else 'FAIL'}")
print(f"  _warmup_done  : expected=False, got={d3._warmup_done} -> {'PASS' if not d3._warmup_done else 'FAIL'}")
results.append(d3._baseline is None and not d3._warmup_done)
os.unlink(list_path)

# Test 4: Missing file
print(f"\n=== Test 4: Missing file (existing behavior preserved) ===")
d4 = DriftDetector(baseline_path="/tmp/does_not_exist_xyz.json")
print(f"  _baseline=None: expected=True, got={d4._baseline is None} -> {'PASS' if d4._baseline is None else 'FAIL'}")
print(f"  _warmup_done  : expected=False, got={d4._warmup_done} -> {'PASS' if not d4._warmup_done else 'FAIL'}")
results.append(d4._baseline is None and not d4._warmup_done)

# Test 5: Empty dict baseline
results.append(test(
    "Test 5: Empty dict baseline",
    expected_baseline_is_none=True,
    expected_warmup_done=False,
    baseline_path=None,
    baseline_data={},
))

print(f"\n=== SUMMARY ===")
print(f"Passed: {sum(results)}/{len(results)}")
if all(results):
    print("ALL TESTS PASSED ✓")
    sys.exit(0)
else:
    print("SOME TESTS FAILED ✗")
    sys.exit(1)
