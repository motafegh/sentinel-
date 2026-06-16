"""Final regression check after all Phase A changes."""
import sys
sys.path.insert(0, "/home/motafeq/projects/sentinel")

# Reload the modules to make sure no syntax errors or import issues
import ml.src.inference.drift_detector
import ml.src.inference.api
import ml.src.inference.predictor
import ml.src.inference.preprocess
import ml.src.inference.cache

print("All inference modules import cleanly")
print(f"  drift_detector has _KNOWN_STAT_NAMES: {hasattr(ml.src.inference.drift_detector, '_KNOWN_STAT_NAMES')}")
print(f"  api CHECKPOINT default: {ml.src.inference.api.CHECKPOINT}")
print(f"  api DRIFT_BASELINE_PATH default: {ml.src.inference.api.DRIFT_BASELINE_PATH}")
print(f"  predictor has _ARCH_TO_FUSION_DIM: {hasattr(ml.src.inference.predictor, '_ARCH_TO_FUSION_DIM')}")
print(f"  cache NODE_FEATURE_DIM import: {ml.src.inference.cache.NODE_FEATURE_DIM}")

# Verify docstring updates
import ml.src.inference.api as api_module
api_doc = api_module.__doc__
print()
print("Doc check:")
print(f"  api.py docstring has 'NUM_CLASSES-class': {'NUM_CLASSES-class' in api_doc}")
print(f"  api.py docstring still has '10-class': {'10-class' in api_doc}")
print(f"  api.py docstring has 'mlops_config.json': {'mlops_config.json' in api_doc}")
