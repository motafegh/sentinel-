"""
_common.py — Shared utilities for per-fix smoke tests.

All smoke tests should:
  1. Use `pass_("message")` and `fail_("message")` for output (not print).
  2. Raise AssertionError with descriptive messages on any check failure.
  3. Be idempotent and < 30s wall-clock.
  4. Exit 0 on pass, non-zero on any AssertionError.

Import contract:
  from smoke._common import REPO_ROOT, GRAPHS_DIR, ..., pass_, fail_, smoke_header
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
ML_ROOT: Path = REPO_ROOT / "ml"
DATA_DIR: Path = ML_ROOT / "data"
GRAPHS_DIR: Path = DATA_DIR / "graphs"
TOKENS_DIR: Path = DATA_DIR / "tokens_windowed"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_DIR / "splits"
SPLITS_DEDUPED: Path = SPLITS_DIR / "deduped"
CHECKPOINTS_DIR: Path = ML_ROOT / "checkpoints"
SMARTBUGS_DIR: Path = DATA_DIR / "smartbugs-curated" / "dataset"
SLITHER_RESULTS_DIR: Path = DATA_DIR / "slither_results"
TEST_CONTRACTS_DIR: Path = ML_ROOT / "scripts" / "test_contracts"
SRC_DIR: Path = ML_ROOT / "src"

# Schema constants (updated as fixes bump them).
EXPECTED_SCHEMA_AFTER_FIX2_3_4: str = "v9"
EXPECTED_NODE_FEATURE_DIM_AFTER_FIX4: int = 12
EXPECTED_NUM_NODE_TYPES_AFTER_FIX4: int = 14
EXPECTED_NUM_EDGE_TYPES_AFTER_FIX3: int = 12

# Class columns in order (must match ml/data/processed/multilabel_index.csv).
CLASS_COLUMNS: list[str] = [
    "Reentrancy",
    "Timestamp",
    "IntegerUO",
    "Delegatecall",
    "MishandledException",
    "DenialOfService",
    "ExternalBug",
    "TOD",
    "UnusedReturn",
    "GasException",
]

T = TypeVar("T")


def smoke_header(fix_id: int, fix_name: str) -> None:
    """Print a clear test header to stderr (won't pollute test result parsing)."""
    print(f"=== SMOKE FIX #{fix_id}: {fix_name} ===", file=sys.stderr, flush=True)


def pass_(message: str) -> None:
    """Print a green-styled pass marker."""
    print(f"  [PASS] {message}", file=sys.stderr, flush=True)


def fail_(message: str) -> None:
    """Print a red-styled fail marker (also raises via the caller)."""
    print(f"  [FAIL] {message}", file=sys.stderr, flush=True)


def check(assertion: bool, message: str) -> None:
    """Assert with a clean message. On fail, prints FAIL marker before raising."""
    if not assertion:
        fail_(message)
        raise AssertionError(message)
    pass_(message)


def timed(label: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: time a check, print duration. Useful for sub-30s enforcement."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                print(f"  [TIME] {label}: {elapsed_ms:.0f}ms", file=sys.stderr, flush=True)
        return wrapper
    return decorator


def load_graph_sample(n: int = 50, seed: int | None = None) -> list[Path]:
    """Return up to n .pt files from GRAPHS_DIR, deterministically sampled."""
    if not GRAPHS_DIR.exists():
        raise AssertionError(f"GRAPHS_DIR missing: {GRAPHS_DIR} — re-extract needed")
    all_files = sorted(GRAPHS_DIR.glob("*.pt"))
    if not all_files:
        raise AssertionError(f"No .pt files in {GRAPHS_DIR}")
    if len(all_files) <= n:
        return all_files
    if seed is not None:
        import random
        rng = random.Random(seed)
        return rng.sample(all_files, n)
    return all_files[:n]


def safe_load_graph(path: Path) -> Any:
    """Load a .pt graph file with weights_only=False, surfacing errors clearly."""
    import torch
    try:
        return torch.load(path, weights_only=False, map_location="cpu")
    except Exception as exc:
        raise AssertionError(f"Failed to load graph {path.name}: {exc}") from exc


def check_schema_version(expected: str) -> None:
    """Import graph_schema and assert its FEATURE_SCHEMA_VERSION matches."""
    sys.path.insert(0, str(SRC_DIR))
    try:
        from preprocessing.graph_schema import FEATURE_SCHEMA_VERSION
    except Exception as exc:
        raise AssertionError(f"Could not import FEATURE_SCHEMA_VERSION: {exc}") from exc
    if FEATURE_SCHEMA_VERSION != expected:
        raise AssertionError(
            f"FEATURE_SCHEMA_VERSION={FEATURE_SCHEMA_VERSION!r}, expected {expected!r}"
        )
    pass_(f"FEATURE_SCHEMA_VERSION == {expected!r}")


def find_checkpoint(pattern: str = "Run8-v10") -> str | None:
    """Find a checkpoint matching the substring pattern. Returns path or None."""
    if not CHECKPOINTS_DIR.exists():
        return None
    candidates = sorted(CHECKPOINTS_DIR.glob(f"*best.pt"))
    for c in candidates:
        if pattern in c.name:
            return str(c)
    return None
