"""pytest configuration for the data_module/ test suite.

Adds the project root + ml/src to sys.path so the thin-adapter re-exports
from ml.src.preprocessing.* and ml.src.data_extraction.* can be loaded.
Without this, pytest cannot import the v9 schema constants, graph_extractor,
or tokenizer from the new path.

Alternative considered: add `ml` to install_requires. Rejected because:
  - It would create a circular install dep (data_module depends on ml)
  - It would force the v2 build to ship the v1 code as a transitive dep
  - The conftest.py approach is local to tests, doesn't affect production

The ml/src path is needed because the v1 tokenizer.py uses
`sys.path.insert(0, ml/src)` to import `src.utils.hash_utils`. We add
ml/ to sys.path so the relative path resolves correctly.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # /home/.../sentinel/
MODULE_ROOT = REPO_ROOT / "data_module"
ML_ROOT = REPO_ROOT / "ml"
for p in (REPO_ROOT, ML_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def data_dir():
    """Point to the real data directory for integration tests."""
    return MODULE_ROOT / "data"
