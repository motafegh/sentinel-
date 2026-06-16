"""Quick environment check for A.5 smoke test."""
import sys
sys.path.insert(0, "/home/motafeq/projects/sentinel")

import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
    print(f"cuda mem (free/total): {torch.cuda.mem_get_info()}")

# Check the Run 12 checkpoint file
from pathlib import Path
ckpt = Path("/home/motafeq/projects/sentinel/ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt")
print(f"\nRun 12 FINAL checkpoint exists: {ckpt.exists()}")
if ckpt.exists():
    print(f"  size: {ckpt.stat().st_size / 1024 / 1024:.1f} MB")

# Check transformers / uvicorn / fastapi
try:
    import fastapi
    print(f"fastapi: {fastapi.__version__}")
except ImportError:
    print("fastapi: NOT INSTALLED")

try:
    import uvicorn
    print(f"uvicorn: {uvicorn.__version__}")
except ImportError:
    print("uvicorn: NOT INSTALLED")

try:
    import transformers
    print(f"transformers: {transformers.__version__}")
except ImportError:
    print("transformers: NOT INSTALLED")
