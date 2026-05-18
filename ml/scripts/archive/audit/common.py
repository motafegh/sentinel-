"""
common.py — Shared utilities for SENTINEL v6 audit scripts.

All audit scripts import from this module for:
  - Project path constants
  - Safe .pt file loading (weights_only=True with PyG safe globals)
  - Label CSV loading
  - MD5→path resolution (same method as reextract_graphs.py)
  - Report saving
  - Sampling helpers
"""

from __future__ import annotations

import csv
import hashlib
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.serialization

# PyG safe globals — required for weights_only=True with PyG Data objects
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

# ── Project paths ──────────────────────────────────────────────────────────────
# Auto-detect project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_CANDIDATES = [
    _SCRIPT_DIR.parents[2],  # ml/scripts/audit/ → project root
    Path("/home/motafeq/projects/sentinel"),
    Path("/home/z/my-project/sentinel/sentinel--main"),
]

PROJECT_ROOT: Optional[Path] = None
for candidate in _PROJECT_CANDIDATES:
    if (candidate / "ml" / "src" / "preprocessing" / "graph_schema.py").exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    print("WARNING: Could not auto-detect project root. Set PROJECT_ROOT manually.")
    PROJECT_ROOT = _SCRIPT_DIR.parents[2]

# Data directories
GRAPHS_DIR = PROJECT_ROOT / "ml" / "data" / "graphs"
GRAPHS_LEGACY_DIR = PROJECT_ROOT / "ml" / "data" / "graphs_legacy"
TOKENS_DIR = PROJECT_ROOT / "ml" / "data" / "tokens"
TOKENS_WINDOWED_DIR = PROJECT_ROOT / "ml" / "data" / "tokens_windowed"
LABEL_CSV = PROJECT_ROOT / "ml" / "data" / "processed" / "multilabel_index_deduped.csv"
REPORT_DIR = PROJECT_ROOT / "ml" / "scripts" / "audit" / "reports"

# BCCC source directories (same as reextract_graphs.py)
SOURCE_DIRS = [
    PROJECT_ROOT / "BCCC-SCsVul-2024" / "SourceCodes",
    PROJECT_ROOT / "ml" / "data" / "SolidiFI-processed",
    PROJECT_ROOT / "ml" / "data" / "SolidiFI",
    PROJECT_ROOT / "ml" / "data" / "smartbugs-curated",
    PROJECT_ROOT / "ml" / "data" / "smartbugs-wild",
    PROJECT_ROOT / "ml" / "data" / "augmented",
]

# Feature layout (v4 schema)
FEATURE_NAMES = [
    "type_id",              # [0]
    "visibility",           # [1]
    "uses_block_globals",   # [2]
    "view",                 # [3]
    "payable",              # [4]
    "complexity",           # [5]
    "loc",                  # [6]
    "return_ignored",       # [7]
    "call_target_typed",    # [8]
    "in_unchecked",         # [9]
    "has_loop",             # [10]
    "external_call_count",  # [11]
]

EDGE_TYPE_NAMES = {
    0: "CALLS",
    1: "READS",
    2: "WRITES",
    3: "EMITS",
    4: "INHERITS",
    5: "CONTAINS",
    6: "CONTROL_FLOW",
    7: "REVERSE_CONTAINS",
}

VULN_CLASSES = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn",
]


# ── Safe loading ───────────────────────────────────────────────────────────────

def load_graph(path: Path):
    """Load a graph .pt file with weights_only=True."""
    return torch.load(path, weights_only=True)


def load_token(path: Path):
    """Load a token .pt file with weights_only=True."""
    return torch.load(path, weights_only=True)


# ── Label CSV ──────────────────────────────────────────────────────────────────

def load_label_csv(csv_path: Path = LABEL_CSV) -> Dict[str, Dict[str, int]]:
    """
    Load multilabel_index_deduped.csv.
    Returns: {md5_stem: {class_name: 0/1, ...}, ...}
    """
    labels: Dict[str, Dict[str, int]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row["md5_stem"]
            labels[stem] = {cls: int(row[cls]) for cls in VULN_CLASSES}
    return labels


def load_label_csv_as_rows(csv_path: Path = LABEL_CSV) -> List[Dict]:
    """Load CSV as list of row dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── MD5→path resolution ───────────────────────────────────────────────────────

def get_contract_hash(contract_path) -> str:
    """MD5 hash of the full contract path (same as hash_utils.get_contract_hash)."""
    path_string = str(contract_path)
    return hashlib.md5(path_string.encode("utf-8")).hexdigest()


def build_md5_to_path(target_md5s: Set[str]) -> Dict[str, Path]:
    """
    Build MD5→path mapping by scanning all BCCC source directories.
    Same method as reextract_graphs.py.
    """
    mapping: Dict[str, Path] = {}
    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            continue
        for sol in src_dir.rglob("*.sol"):
            try:
                rel = sol.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = sol
            md5 = get_contract_hash(rel)
            if md5 in target_md5s:
                mapping[md5] = sol
    return mapping


def find_sol_for_stem(stem: str, md5_to_path: Dict[str, Path]) -> Optional[Path]:
    """Find .sol file for a given md5_stem."""
    return md5_to_path.get(stem)


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_stems(n: int, from_set: Optional[Set[str]] = None, seed: int = 42) -> List[str]:
    """Sample n md5_stems from available graph files."""
    if from_set is None:
        from_set = {p.stem for p in GRAPHS_DIR.glob("*.pt")}
    rng = random.Random(seed)
    sample = list(from_set)
    rng.shuffle(sample)
    return sample[:n]


def get_paired_stems() -> Set[str]:
    """Get stems that exist in both graphs/ and tokens_windowed/."""
    graph_stems = {p.stem for p in GRAPHS_DIR.glob("*.pt")}
    token_stems = {p.stem for p in TOKENS_WINDOWED_DIR.glob("*.pt")}
    return graph_stems & token_stems


def get_stems_with_label(label_name: str, label_value: int = 1,
                         pure: bool = False) -> List[str]:
    """
    Get md5_stems with a specific label.
    If pure=True, require label_value=1 for target class and 0 for ALL other classes.
    """
    labels = load_label_csv()
    result = []
    for stem, lbls in labels.items():
        if lbls[label_name] == label_value:
            if pure:
                if all(lbls[cls] == 0 for cls in VULN_CLASSES if cls != label_name):
                    result.append(stem)
            else:
                result.append(stem)
    return result


# ── Report saving ──────────────────────────────────────────────────────────────

def save_report(task_name: str, content: str) -> Path:
    """Save a report file to the reports directory."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_name}_{timestamp}.md"
    path = REPORT_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"\nReport saved to: {path}")
    return path


def format_table(headers: List[str], rows: List[List], col_widths: Optional[List[int]] = None) -> str:
    """Format a simple text table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    lines = []
    header_line = "|".join(str(h).center(w) for h, w in zip(headers, col_widths))
    sep_line = "-".join("-" * w for w in col_widths)
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        line = "|".join(str(v).center(w) for v, w in zip(row, col_widths))
        lines.append(line)
    return "\n".join(lines)


# ── Solidity helpers ──────────────────────────────────────────────────────────

import re
_PRAGMA_RE = re.compile(r'pragma\s+solidity\s+[\^~>=<\s]*(\d+)\.(\d+)')


def extract_pragma_version(sol_path: Path) -> Optional[str]:
    """Extract Solidity major.minor version from pragma."""
    try:
        text = sol_path.read_text(encoding="utf-8", errors="replace")
        m = _PRAGMA_RE.search(text)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    except OSError:
        pass
    return None


# ── Node type helpers ─────────────────────────────────────────────────────────

def is_cfg_node(type_id_value: float) -> bool:
    """Check if a type_id feature value corresponds to a CFG node (type_id >= 8/12)."""
    return type_id_value >= 8.0 / 12.0


# ── Print helper ──────────────────────────────────────────────────────────────

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    print(f"\n--- {title} ---")
