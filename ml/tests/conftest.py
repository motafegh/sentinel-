# ml/tests/conftest.py
import os
import pytest
from fastapi.testclient import TestClient

# Set BEFORE the app imports — HuggingFace checks this at import time.
# Without it, the app tries to reach binaries.soliditylang.org on every
# startup, fails silently in WSL2, and pollutes test output with noise.
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Import app AFTER env vars are set
from ml.src.inference.api import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    Single TestClient for the entire test session.

    scope="session" means the model loads ONCE — not once per test.
    The SENTINEL model (~500MB) takes ~10s to load.
    Without session scope: 4 tests × 10s = 40s of wasted loading.
    With session scope:    load once → 4 tests run in ~2s total.
    """
    with TestClient(app) as c:
        yield c  # 'yield' means: set up → run tests → tear down