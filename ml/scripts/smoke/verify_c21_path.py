"""Verify C.2.1 path resolves after the move."""
from pathlib import Path

SCRIPT = Path("ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/scripts/run_c21_smoke_inference.py")
REPO_ROOT = SCRIPT.parents[4]
TEMPERATURES = REPO_ROOT / "ml/calibration/temperatures_run12.json"

print(f"REPO_ROOT: {REPO_ROOT}")
print(f"TEMPERATURES: {TEMPERATURES}")
print(f"Exists: {TEMPERATURES.exists()}")
print(f"Size: {TEMPERATURES.stat().st_size} bytes")
