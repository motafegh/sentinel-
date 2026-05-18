"""
common.py — Shared utilities for SENTINEL audit scripts.
Every task script imports from here.
"""
import os
import sys
from pathlib import Path

import torch
import torch.serialization

# ── Project root detection ────────────────────────────────────────────────────

def get_project_root() -> Path:
    env = os.environ.get("SENTINEL_ROOT")
    if env:
        root = Path(env)
        if root.exists():
            return root
        print(f"[WARN] SENTINEL_ROOT={env} does not exist; trying known paths.")
    candidates = [
        Path("/home/motafeq/projects/sentinel"),
        Path("/home/z/my-project/sentinel/sentinel--main"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Cannot find project root. Set the SENTINEL_ROOT environment variable:\n"
        "  export SENTINEL_ROOT=/path/to/sentinel"
    )

ROOT        = None  # lazy-initialised the first time a script calls get_dirs()
ML_DIR      = None
GRAPHS_DIR  = None
TOKENS_DIR  = None
CSV_PATH    = None
BCCC_DIR    = None

def get_dirs():
    global ROOT, ML_DIR, GRAPHS_DIR, TOKENS_DIR, CSV_PATH, BCCC_DIR
    if ROOT is None:
        ROOT       = get_project_root()
        ML_DIR     = ROOT / "ml"
        GRAPHS_DIR = ML_DIR / "data" / "graphs"
        TOKENS_DIR = ML_DIR / "data" / "tokens_windowed"
        CSV_PATH   = ML_DIR / "data" / "processed" / "multilabel_index_deduped.csv"
        # BCCC source: try a few common locations
        for bdir in [
            ROOT / "BCCC-SCsVul-2024" / "SourceCodes",
            ROOT / "data" / "BCCC-SCsVul-2024" / "SourceCodes",
            ML_DIR / "data" / "BCCC-SCsVul-2024" / "SourceCodes",
        ]:
            if bdir.exists():
                BCCC_DIR = bdir
                break
        if BCCC_DIR is None:
            print("[WARN] BCCC source directory not found; tasks that read .sol files will fail.")
    return ROOT, ML_DIR, GRAPHS_DIR, TOKENS_DIR, CSV_PATH, BCCC_DIR

# ── PyTorch / PyG safe loading ───────────────────────────────────────────────

_pyg_registered = False

def register_pyg_globals():
    global _pyg_registered
    if _pyg_registered:
        return
    try:
        from torch_geometric.data import Data
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        torch.serialization.add_safe_globals(
            [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]
        )
    except Exception as e:
        print(f"[WARN] register_pyg_globals: {e}")
    _pyg_registered = True

def load_graph(path):
    """Load a graph .pt file safely."""
    register_pyg_globals()
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)

def load_token(path):
    """Load a token .pt file safely."""
    register_pyg_globals()
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return torch.load(path, weights_only=False)

# ── Label columns ─────────────────────────────────────────────────────────────

LABEL_COLS = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn",
]

FEATURE_NAMES = [
    "[0] type_id", "[1] visibility", "[2] uses_block_globals", "[3] view",
    "[4] payable", "[5] complexity", "[6] loc", "[7] return_ignored",
    "[8] call_target_typed", "[9] in_unchecked", "[10] has_loop",
    "[11] ext_call_count",
]

# Node type threshold: type_id [0] values
# Declaration nodes: type_id_raw in 0..7 → normalised < 8/12 ≈ 0.583
# CFG nodes:         type_id_raw in 8..12 → normalised >= 8/12
DECL_THRESHOLD = 8.0 / 12.0

# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_csv():
    import pandas as pd
    _, _, _, _, csv, _ = get_dirs()
    df = pd.read_csv(csv, dtype={"md5_stem": str})
    return df

def stem_to_sol(md5_stem, bccc_dir):
    """
    Find the .sol source file for a given md5_stem.
    Tries BCCC/SourceCodes/*/{md5_stem}*.sol  (the filename is the sha256 hash,
    but the md5_stem is used as the .pt filename; search by glob).
    Returns Path or None.
    """
    if bccc_dir is None:
        return None
    # The .sol files are named by SHA256, the stems are MD5; we need to scan.
    # Look for a file whose stem starts with md5_stem (unlikely) OR use the
    # contract_path stored in the graph .pt (preferred).
    # This helper is a fallback full scan — callers should prefer g.contract_path.
    for sol in bccc_dir.rglob("*.sol"):
        import hashlib
        md5 = hashlib.md5(sol.read_bytes()).hexdigest()
        if md5 == md5_stem:
            return sol
    return None

def sol_from_graph(g, bccc_dir):
    """Return Path to the .sol source for graph g, using g.contract_path."""
    cp = getattr(g, "contract_path", None)
    if cp is None:
        return None
    # contract_path is relative to some root — try a few bases
    for base in [bccc_dir.parent if bccc_dir else None,
                 get_dirs()[0],  # ROOT
                 get_dirs()[0].parent]:
        if base is None:
            continue
        candidate = base / cp
        if candidate.exists():
            return candidate
    # Last resort: search by filename
    filename = Path(cp).name
    if bccc_dir:
        for sol in bccc_dir.rglob(filename):
            return sol
    return None

# ── Misc ──────────────────────────────────────────────────────────────────────

def random_pt_sample(directory: Path, n: int, seed: int = 42):
    """Return up to n random .pt paths from directory."""
    import random
    files = list(directory.glob("*.pt"))
    random.seed(seed)
    if len(files) <= n:
        return files
    return random.sample(files, n)

def print_header(task_num, task_name):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Task {task_num}: {task_name}")
    print(bar)
