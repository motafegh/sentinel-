"""pytest configuration for the Data/ test suite.

Adds the project root to sys.path so the thin-adapter re-exports from
ml.src.preprocessing.* can be loaded. Without this, pytest cannot import
the v9 schema constants and graph_extractor from the new path.

Alternative considered: add `ml` to install_requires. Rejected because:
  - It would create a circular install dep (Data depends on ml)
  - It would force the v2 build to ship the v1 code as a transitive dep
  - The conftest.py approach is local to tests, doesn't affect production
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # /home/.../sentinel/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
