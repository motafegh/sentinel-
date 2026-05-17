#!/usr/bin/env python3
"""
Deep data audit investigation script.
Runs Tasks 1-8 from the audit spec.
"""
import sys
import os
import random
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, "/home/motafeq/projects/sentinel")

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Safe loading for graph .pt files
from torch_geometric.data import Data
try:
    from torch_geometric.data.storage import GlobalStorage
    from torch_geometric.data import DataEdgeAttr, DataTensorAttr
    SAFE_GLOBALS = [Data, GlobalStorage]
    try:
        SAFE_GLOBALS.append(DataEdgeAttr)
        SAFE_GLOBALS.append(DataTensorAttr)
    except:
        pass
except ImportError:
    SAFE_GLOBALS = [Data]

def safe_load_graph(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return None

def safe_load_token(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return None

# Paths
GRAPHS_DIR = Path("/home/motafeq/projects/sentinel/ml/data/graphs")
TOKENS_DIR = Path("/home/motafeq/projects/sentinel/ml/data/tokens_windowed")
CSV_PATH = Path("/home/motafeq/projects/sentinel/ml/data/processed/multilabel_index_deduped.csv")
BCCC_DIR = Path("/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes")

CLASSES = [
    "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
    "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
    "TransactionOrderDependence", "UnusedReturn"
]

FEATURE_NAMES = [
    "type_id/12", "visibility", "uses_block_globals", "view",
    "payable", "complexity", "loc", "return_ignored",
    "call_target_typed", "in_unchecked", "has_loop", "ext_call_count"
]

EDGE_TYPE_NAMES = {0: "CALLS", 1: "READS", 2: "WRITES", 3: "EMITS",
                   4: "INHERITS", 5: "CONTAINS", 6: "CF"}

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"CSV shape: {df.shape}, columns: {list(df.columns)}")

# Rename TransactionOrderDependence if needed
col_map = {}
for col in df.columns:
    if "Transaction" in col or "TOD" in col or "TransactionOrder" in col:
        col_map[col] = "TransactionOrderDependence"
df = df.rename(columns=col_map)
print(f"Columns after rename: {list(df.columns)}")

# ============================================================
# TASK 8 FIRST: Label co-occurrence matrix (fast, no graph loading)
# ============================================================
print("\n" + "="*70)
print("TASK 8: Label Co-occurrence Matrix")
print("="*70)

label_cols = [c for c in CLASSES if c in df.columns]
print(f"Label columns found: {label_cols}")

label_df = df[label_cols].fillna(0).astype(int)

# Total counts per class
class_counts = label_df.sum()
print("\nClass counts:")
for cls, cnt in class_counts.items():
    print(f"  {cls}: {cnt}")

# Co-occurrence matrix: P(B|A) = count(A=1 & B=1) / count(A=1)
print("\nCo-occurrence matrix P(col_B=1 | row_A=1) [%]:")
co_matrix = {}
for clsA in label_cols:
    a_rows = label_df[label_df[clsA] == 1]
    row = {}
    for clsB in label_cols:
        if clsA == clsB:
            row[clsB] = 100.0
        else:
            row[clsB] = 100.0 * a_rows[clsB].sum() / len(a_rows) if len(a_rows) > 0 else 0.0
    co_matrix[clsA] = row

# Print as table
header = "          " + "".join(f"{c[:6]:>8}" for c in label_cols)
print(header)
for clsA in label_cols:
    row_str = f"{clsA[:10]:10}" + "".join(f"{co_matrix[clsA][clsB]:8.1f}" for clsB in label_cols)
    print(row_str)

# Highlight high co-occurrences
print("\nHigh co-occurrences (>50%, excluding diagonal):")
for clsA in label_cols:
    for clsB in label_cols:
        if clsA != clsB and co_matrix[clsA][clsB] > 50.0:
            n_A = class_counts[clsA]
            n_AB = label_df[(label_df[clsA] == 1) & (label_df[clsB] == 1)].shape[0]
            print(f"  P({clsB}=1 | {clsA}=1) = {co_matrix[clsA][clsB]:.1f}%  ({n_AB}/{n_A})")

# Pure label counts
print("\nPure single-label counts:")
pure_counts = {}
for cls in label_cols:
    mask = (label_df[cls] == 1) & (label_df.drop(columns=[cls]).sum(axis=1) == 0)
    pure_counts[cls] = mask.sum()
    print(f"  {cls}: {pure_counts[cls]} pure")

# ============================================================
# Helper: get pure samples for a class
# ============================================================
def get_pure_samples(cls, n=20, seed=42):
    if cls not in label_df.columns:
        return []
    other_cols = [c for c in label_cols if c != cls]
    mask = (label_df[cls] == 1) & (label_df[other_cols].sum(axis=1) == 0)
    stems = df[mask]["md5_stem"].tolist()
    random.seed(seed)
    random.shuffle(stems)
    return stems[:n]

# ============================================================
# TASK 1: Per-class feature activation rates
# ============================================================
print("\n" + "="*70)
print("TASK 1: Per-class feature activation rates")
print("="*70)

FEATURES_OF_INTEREST = {
    "uses_block_globals": 2,
    "return_ignored": 7,
    "has_loop": 10,
    "ext_call_count": 11,
    "in_unchecked": 9,
}

task1_results = {}

for cls in label_cols:
    samples = get_pure_samples(cls, n=20)
    if not samples:
        print(f"  {cls}: NO pure samples found")
        task1_results[cls] = None
        continue

    feat_vals = defaultdict(list)
    cf_counts = []
    graphs_loaded = 0

    for stem in samples:
        gpath = GRAPHS_DIR / f"{stem}.pt"
        if not gpath.exists():
            continue
        g = safe_load_graph(gpath)
        if g is None or not hasattr(g, 'x') or g.x is None:
            continue

        graphs_loaded += 1
        x = g.x  # [N, 12]

        for fname, fidx in FEATURES_OF_INTEREST.items():
            if fidx < x.shape[1]:
                vals = x[:, fidx].numpy()
                feat_vals[fname].extend(vals.tolist())

        # CF edge count
        if hasattr(g, 'edge_index') and g.edge_index is not None and \
           hasattr(g, 'edge_attr') and g.edge_attr is not None:
            ea = g.edge_attr
            if ea.dim() > 1:
                ea = ea.squeeze()
            cf_mask = (ea == 6)
            cf_counts.append(cf_mask.sum().item())
        else:
            cf_counts.append(0)

    row = {}
    for fname in FEATURES_OF_INTEREST:
        vals = np.array(feat_vals[fname])
        if len(vals) > 0:
            row[fname] = {
                "mean": float(np.mean(vals)),
                "nonzero_rate": float(np.mean(vals != 0)),
                "pos_rate": float(np.mean(vals > 0)),
            }
        else:
            row[fname] = {"mean": 0, "nonzero_rate": 0, "pos_rate": 0}
    row["cf_mean"] = float(np.mean(cf_counts)) if cf_counts else 0
    row["cf_zero_frac"] = float(np.mean(np.array(cf_counts) == 0)) if cf_counts else 0
    row["n_loaded"] = graphs_loaded
    task1_results[cls] = row

# Print table
print(f"\n{'Class':28} {'n':>3} {'blk_glb':>8} {'ret_ign':>8} {'has_loop':>9} {'ext_call':>9} {'unchecked':>10} {'CF_mean':>8} {'CF_zero':>8}")
print("-"*100)
for cls in label_cols:
    r = task1_results[cls]
    if r is None:
        print(f"  {cls:26} NO PURE SAMPLES")
        continue
    print(f"  {cls:26} {r['n_loaded']:>3} "
          f"  {r['uses_block_globals']['nonzero_rate']:>6.2%} "
          f"  {r['return_ignored']['nonzero_rate']:>6.2%} "
          f"  {r['has_loop']['nonzero_rate']:>7.2%} "
          f"  {r['ext_call_count']['nonzero_rate']:>7.2%} "
          f"  {r['in_unchecked']['nonzero_rate']:>8.2%} "
          f"  {r['cf_mean']:>7.1f} "
          f"  {r['cf_zero_frac']:>7.2%}")

print("\n  (nonzero_rate = fraction of nodes with feature != 0)")

# Mean values too
print(f"\n{'Class':28} {'blk_glb_m':>10} {'ret_ign_m':>10} {'has_loop_m':>11} {'ext_call_m':>11} {'unchecked_m':>12}")
print("-"*80)
for cls in label_cols:
    r = task1_results[cls]
    if r is None:
        continue
    print(f"  {cls:26} "
          f"  {r['uses_block_globals']['mean']:>8.3f} "
          f"  {r['return_ignored']['mean']:>8.3f} "
          f"  {r['has_loop']['mean']:>9.3f} "
          f"  {r['ext_call_count']['mean']:>9.3f} "
          f"  {r['in_unchecked']['mean']:>10.3f}")

# ============================================================
# TASK 5: Edge type distribution per class
# ============================================================
print("\n" + "="*70)
print("TASK 5: Edge type distribution per class")
print("="*70)

task5_results = {}
for cls in label_cols:
    samples = get_pure_samples(cls, n=20)
    edge_counts = defaultdict(list)
    n_loaded = 0

    for stem in samples:
        gpath = GRAPHS_DIR / f"{stem}.pt"
        if not gpath.exists():
            continue
        g = safe_load_graph(gpath)
        if g is None or not hasattr(g, 'edge_attr') or g.edge_attr is None:
            continue

        n_loaded += 1
        ea = g.edge_attr
        if ea.dim() > 1:
            ea = ea.squeeze()

        for etype, ename in EDGE_TYPE_NAMES.items():
            cnt = (ea == etype).sum().item()
            edge_counts[ename].append(cnt)

    row = {}
    for ename in EDGE_TYPE_NAMES.values():
        vals = edge_counts[ename]
        row[ename] = float(np.mean(vals)) if vals else 0.0
    row["n"] = n_loaded
    task5_results[cls] = row

etypes = list(EDGE_TYPE_NAMES.values())
print(f"\n{'Class':28} {'n':>3} " + "".join(f"{e:>8}" for e in etypes))
print("-"*90)
for cls in label_cols:
    r = task5_results[cls]
    print(f"  {cls:26} {r['n']:>3} " + "".join(f"{r[e]:>8.1f}" for e in etypes))

# ============================================================
# TASK 7: Sentinel value check (-1.0)
# ============================================================
print("\n" + "="*70)
print("TASK 7: Sentinel value (-1.0) analysis across 500 graphs")
print("="*70)

all_stems = [p.stem for p in GRAPHS_DIR.glob("*.pt")]
random.seed(42)
sample_500 = random.sample(all_stems, min(500, len(all_stems)))

sentinel_stats = {
    "return_ignored_sentinel": 0,
    "call_target_sentinel": 0,
    "total_nodes": 0,
    "total_graphs": 0,
    "graphs_with_return_sentinel": 0,
    "graphs_with_call_sentinel": 0,
}

for stem in sample_500:
    gpath = GRAPHS_DIR / f"{stem}.pt"
    if not gpath.exists():
        continue
    g = safe_load_graph(gpath)
    if g is None or not hasattr(g, 'x') or g.x is None:
        continue

    x = g.x
    sentinel_stats["total_graphs"] += 1
    sentinel_stats["total_nodes"] += x.shape[0]

    if x.shape[1] > 7:
        ret_ign = x[:, 7].numpy()
        n_sentinel_ret = (ret_ign == -1.0).sum()
        sentinel_stats["return_ignored_sentinel"] += int(n_sentinel_ret)
        if n_sentinel_ret > 0:
            sentinel_stats["graphs_with_return_sentinel"] += 1

    if x.shape[1] > 8:
        call_tgt = x[:, 8].numpy()
        n_sentinel_call = (call_tgt == -1.0).sum()
        sentinel_stats["call_target_sentinel"] += int(n_sentinel_call)
        if n_sentinel_call > 0:
            sentinel_stats["graphs_with_call_sentinel"] += 1

total_n = sentinel_stats["total_nodes"]
total_g = sentinel_stats["total_graphs"]
print(f"\nAnalyzed {total_g} graphs, {total_n} total nodes")
print(f"  return_ignored sentinel nodes: {sentinel_stats['return_ignored_sentinel']} "
      f"({100*sentinel_stats['return_ignored_sentinel']/max(total_n,1):.2f}% of nodes)")
print(f"  call_target_typed sentinel nodes: {sentinel_stats['call_target_sentinel']} "
      f"({100*sentinel_stats['call_target_sentinel']/max(total_n,1):.2f}% of nodes)")
print(f"  graphs with any return_ignored=-1: {sentinel_stats['graphs_with_return_sentinel']} "
      f"({100*sentinel_stats['graphs_with_return_sentinel']/max(total_g,1):.1f}%)")
print(f"  graphs with any call_target=-1: {sentinel_stats['graphs_with_call_sentinel']} "
      f"({100*sentinel_stats['graphs_with_call_sentinel']/max(total_g,1):.1f}%)")

# Check for other unusual values
print("\nFeature value range check (500 graphs, all 12 features):")
feat_min = np.full(12, np.inf)
feat_max = np.full(12, -np.inf)
feat_neg_counts = np.zeros(12)
feat_total = 0

for stem in sample_500[:200]:  # Use 200 for speed
    gpath = GRAPHS_DIR / f"{stem}.pt"
    if not gpath.exists():
        continue
    g = safe_load_graph(gpath)
    if g is None or not hasattr(g, 'x') or g.x is None:
        continue
    x = g.x.numpy()
    feat_total += x.shape[0]
    for fi in range(min(12, x.shape[1])):
        feat_min[fi] = min(feat_min[fi], x[:, fi].min())
        feat_max[fi] = max(feat_max[fi], x[:, fi].max())
        feat_neg_counts[fi] += (x[:, fi] < 0).sum()

print(f"\n{'Feature':25} {'min':>8} {'max':>8} {'n_negative':>12}")
print("-"*60)
for fi, fname in enumerate(FEATURE_NAMES):
    print(f"  {fname:23} {feat_min[fi]:>8.3f} {feat_max[fi]:>8.3f} {feat_neg_counts[fi]:>12.0f}")

# ============================================================
# TASK 4: Ghost graph analysis
# ============================================================
print("\n" + "="*70)
print("TASK 4: Ghost graph analysis (no CF edges)")
print("="*70)

ghost_stems = []
ghost_sample = random.sample(all_stems, min(2000, len(all_stems)))
random.seed(123)

n_checked = 0
for stem in ghost_sample:
    gpath = GRAPHS_DIR / f"{stem}.pt"
    if not gpath.exists():
        continue
    g = safe_load_graph(gpath)
    if g is None or not hasattr(g, 'edge_attr') or g.edge_attr is None:
        n_checked += 1
        ghost_stems.append(stem)
        continue

    ea = g.edge_attr
    if ea.dim() > 1:
        ea = ea.squeeze()
    n_checked += 1
    if (ea == 6).sum().item() == 0:
        ghost_stems.append(stem)

print(f"\nChecked {n_checked} graphs, found {len(ghost_stems)} ghosts ({100*len(ghost_stems)/max(n_checked,1):.1f}%)")

# Classify ghost graphs by their labels
ghost_label_counts = defaultdict(int)
ghost_class_dist = defaultdict(int)
for stem in ghost_stems:
    row = df[df["md5_stem"] == stem]
    if len(row) == 0:
        ghost_label_counts["NOT_IN_CSV"] += 1
        continue
    row = row.iloc[0]
    labels = [cls for cls in label_cols if cls in row.index and row[cls] == 1]
    if not labels:
        ghost_class_dist["NonVulnerable"] += 1
    else:
        for lbl in labels:
            ghost_class_dist[lbl] += 1
    if len(labels) == 0:
        ghost_label_counts["unlabeled"] += 1
    else:
        ghost_label_counts["|".join(sorted(labels))] += 1

print("\nGhost graphs by vulnerability class:")
for cls in sorted(ghost_class_dist.keys()):
    print(f"  {cls}: {ghost_class_dist[cls]}")

# ============================================================
# TASK 3: Token coverage analysis
# ============================================================
print("\n" + "="*70)
print("TASK 3: Token coverage analysis (50 windowed tokens)")
print("="*70)

token_files = list(TOKENS_DIR.glob("*.pt"))
random.seed(42)
token_sample = random.sample(token_files, min(50, len(token_files)))

window_dist = Counter()
num_tokens_list = []
num_windows_list = []
token_data = []

for tp in token_sample:
    t = safe_load_token(tp)
    if t is None:
        continue

    stem = tp.stem
    nw = t.get("num_windows", 1) if isinstance(t, dict) else getattr(t, "num_windows", 1)
    nt = t.get("num_tokens", None) if isinstance(t, dict) else getattr(t, "num_tokens", None)

    # Check shape
    if isinstance(t, dict):
        ids = t.get("input_ids", None)
    else:
        ids = getattr(t, "input_ids", None)

    if ids is not None:
        shape = ids.shape if hasattr(ids, 'shape') else None
        if shape is not None and len(shape) >= 1:
            if len(shape) == 2:
                actual_windows = shape[0]
            else:
                actual_windows = 1
            window_dist[actual_windows] += 1
            num_windows_list.append(actual_windows)

    if nw is not None:
        pass  # use actual shape
    if nt is not None:
        num_tokens_list.append(int(nt))

    # Get labels for this token file
    row = df[df["md5_stem"] == stem]
    labels = []
    if len(row) > 0:
        row = row.iloc[0]
        labels = [cls for cls in label_cols if cls in row.index and row[cls] == 1]

    token_data.append({
        "stem": stem,
        "num_windows": actual_windows if ids is not None else None,
        "num_tokens": nt,
        "is_vulnerable": len(labels) > 0,
        "labels": labels,
    })

print(f"\nWindow distribution (W=1/2/3/4):")
for w in sorted(window_dist.keys()):
    print(f"  W={w}: {window_dist[w]} contracts ({100*window_dist[w]/len(token_data):.1f}%)")

if num_tokens_list:
    arr = np.array(num_tokens_list)
    print(f"\nnum_tokens: mean={arr.mean():.0f}, p50={np.median(arr):.0f}, p95={np.percentile(arr,95):.0f}, max={arr.max()}")

# Vulnerable vs safe
vuln_windows = [d["num_windows"] for d in token_data if d["is_vulnerable"] and d["num_windows"]]
safe_windows = [d["num_windows"] for d in token_data if not d["is_vulnerable"] and d["num_windows"]]
print(f"\n  Mean windows — vulnerable: {np.mean(vuln_windows):.2f} (n={len(vuln_windows)}), "
      f"safe: {np.mean(safe_windows):.2f} (n={len(safe_windows)})")

# Also check token file structure
print("\nToken file key inspection (first 5):")
for d in token_data[:5]:
    tp = TOKENS_DIR / f"{d['stem']}.pt"
    t = safe_load_token(tp)
    if isinstance(t, dict):
        print(f"  {d['stem'][:12]}: keys={list(t.keys())}, input_ids={t['input_ids'].shape if 'input_ids' in t else 'N/A'}")
    elif hasattr(t, '__dict__'):
        print(f"  {d['stem'][:12]}: type={type(t)}, attrs={[a for a in dir(t) if not a.startswith('_')][:8]}")

# ============================================================
# TASK 6: return_ignored and call_target_typed sanity check
# ============================================================
print("\n" + "="*70)
print("TASK 6: return_ignored and call_target_typed sanity check")
print("="*70)

# Find contracts where any node has return_ignored = 1.0
ret_ign_candidates = []
call_tgt_zero_candidates = []  # call_target_typed = 0.0 (dynamic/unknown)

for stem in sample_500[:300]:
    gpath = GRAPHS_DIR / f"{stem}.pt"
    if not gpath.exists():
        continue
    g = safe_load_graph(gpath)
    if g is None or not hasattr(g, 'x') or g.x is None:
        continue
    x = g.x
    if x.shape[1] <= 8:
        continue

    ret_ign = x[:, 7].numpy()
    call_tgt = x[:, 8].numpy()

    if (ret_ign == 1.0).any() and len(ret_ign_candidates) < 5:
        # Find the contract path
        contract_path = getattr(g, 'contract_path', None)
        if contract_path is None:
            contract_path = getattr(g, 'path', None)
        ret_ign_candidates.append({
            "stem": stem,
            "n_ret_ign_nodes": int((ret_ign == 1.0).sum()),
            "contract_path": contract_path,
        })

    if (call_tgt == 0.0).any() and len(call_tgt_zero_candidates) < 5:
        contract_path = getattr(g, 'contract_path', None)
        if contract_path is None:
            contract_path = getattr(g, 'path', None)
        call_tgt_zero_candidates.append({
            "stem": stem,
            "n_call_zero_nodes": int((call_tgt == 0.0).sum()),
            "contract_path": contract_path,
        })

print(f"\nContracts with return_ignored=1.0 nodes (found {len(ret_ign_candidates)}):")
for c in ret_ign_candidates:
    print(f"  stem={c['stem']}, n_nodes_with_ret_ign={c['n_ret_ign_nodes']}, path={c['contract_path']}")

print(f"\nContracts with call_target_typed=0.0 nodes (found {len(call_tgt_zero_candidates)}):")
for c in call_tgt_zero_candidates:
    print(f"  stem={c['stem']}, n_nodes={c['n_call_zero_nodes']}, path={c['contract_path']}")

# Read actual source for validation
def read_sol_source(contract_path, max_lines=200):
    if contract_path is None:
        return None
    # Try multiple path resolutions
    paths_to_try = [
        Path(contract_path),
        Path("/home/motafeq/projects/sentinel") / contract_path.lstrip("/"),
    ]
    # Also try BCCC
    fname = Path(contract_path).name
    for cls_dir in BCCC_DIR.iterdir():
        potential = cls_dir / fname
        if potential.exists():
            paths_to_try.append(potential)

    for p in paths_to_try:
        if p.exists():
            try:
                lines = p.read_text(errors='replace').splitlines()[:max_lines]
                return lines, str(p)
            except:
                pass
    return None

print("\n--- Checking return_ignored=1.0 contracts ---")
for c in ret_ign_candidates[:3]:
    result = read_sol_source(c["contract_path"])
    if result:
        lines, actual_path = result
        print(f"\n  stem: {c['stem']}")
        print(f"  path: {actual_path}")
        # Look for .call/.send/.delegatecall with unchecked returns
        for i, line in enumerate(lines):
            if any(kw in line for kw in ['.call(', '.send(', '.delegatecall(', '.transfer(']):
                start = max(0, i-1)
                end = min(len(lines), i+3)
                print(f"  Line {i+1}: {lines[i].strip()}")
    else:
        print(f"\n  stem: {c['stem']} — source not found at {c['contract_path']}")

print("\n--- Checking call_target_typed=0.0 contracts ---")
for c in call_tgt_zero_candidates[:3]:
    result = read_sol_source(c["contract_path"])
    if result:
        lines, actual_path = result
        print(f"\n  stem: {c['stem']}")
        print(f"  path: {actual_path}")
        for i, line in enumerate(lines):
            if any(kw in line for kw in ['.call(', 'call{', 'address(']):
                print(f"  Line {i+1}: {lines[i].strip()}")
    else:
        print(f"\n  stem: {c['stem']} — source not found at {c['contract_path']}")

# ============================================================
# TASK 2: Spot-check specific vulnerability patterns
# ============================================================
print("\n" + "="*70)
print("TASK 2: Spot-check specific vulnerability patterns")
print("="*70)

VULN_PATTERNS = {
    "Reentrancy": [".call(", "call{value", "msg.value", "balance[", "withdrawal"],
    "DenialOfService": ["for (", "for(", "while (", "while(", ".length", "gasleft"],
    "Timestamp": ["block.timestamp", "block.number", "now "],
    "IntegerUO": ["unchecked {", "unchecked{", "+ ", "- ", "* ", "overflow", "SafeMath"],
    "MishandledException": [".send(", ".call(", ".transfer(", "call{"],
    "UnusedReturn": ["return ", "; //", ".call(", ".send("],
    "GasException": ["gasleft()", "gas:", "gas =", ".call{gas"],
    "CallToUnknown": ["address(", ".call(", "call{", "low-level"],
    "TransactionOrderDependence": ["block.timestamp", "block.number", "tx.gasprice", "block.difficulty"],
    "ExternalBug": ["interface ", "IToken", ".transfer(", "external"],
}

for cls in label_cols[:5]:  # First 5 classes for brevity
    print(f"\n--- {cls} ---")
    samples = get_pure_samples(cls, n=5, seed=99)
    found = 0
    for stem in samples:
        if found >= 2:
            break
        gpath = GRAPHS_DIR / f"{stem}.pt"
        if not gpath.exists():
            continue
        g = safe_load_graph(gpath)
        if g is None:
            continue

        x = g.x if hasattr(g, 'x') else None
        ea = g.edge_attr if hasattr(g, 'edge_attr') else None

        n_nodes = x.shape[0] if x is not None else 0
        n_edges = ea.shape[0] if ea is not None else 0

        if ea is not None:
            ea_flat = ea.squeeze() if ea.dim() > 1 else ea
            edge_types_present = sorted(set(ea_flat.tolist()))
        else:
            edge_types_present = []

        contract_path = getattr(g, 'contract_path', getattr(g, 'path', None))

        print(f"  stem: {stem}")
        print(f"  nodes={n_nodes}, edges={n_edges}, edge_types={edge_types_present}")
        if x is not None:
            nonzero_per_feat = (x != 0).sum(dim=0).tolist()
            print(f"  nonzero_per_feature: {[f'{v:.0f}' for v in nonzero_per_feat]}")
        print(f"  contract_path: {contract_path}")

        result = read_sol_source(contract_path)
        if result:
            lines, actual_path = result
            patterns = VULN_PATTERNS.get(cls, [])
            pattern_lines = []
            for i, line in enumerate(lines):
                if any(p in line for p in patterns):
                    pattern_lines.append((i+1, line.strip()))

            if pattern_lines:
                print(f"  CONFIRMED: Found {len(pattern_lines)} pattern lines in source:")
                for lno, ltext in pattern_lines[:5]:
                    print(f"    L{lno}: {ltext[:120]}")
            else:
                print(f"  WARNING: No expected patterns found in source ({len(lines)} lines)")
        else:
            print(f"  SOURCE NOT FOUND")

        found += 1

# Do last 5 classes
for cls in label_cols[5:]:
    print(f"\n--- {cls} ---")
    samples = get_pure_samples(cls, n=5, seed=99)
    found = 0
    for stem in samples:
        if found >= 2:
            break
        gpath = GRAPHS_DIR / f"{stem}.pt"
        if not gpath.exists():
            continue
        g = safe_load_graph(gpath)
        if g is None:
            continue

        x = g.x if hasattr(g, 'x') else None
        ea = g.edge_attr if hasattr(g, 'edge_attr') else None
        n_nodes = x.shape[0] if x is not None else 0
        n_edges = ea.shape[0] if ea is not None else 0

        if ea is not None:
            ea_flat = ea.squeeze() if ea.dim() > 1 else ea
            edge_types_present = sorted(set(ea_flat.tolist()))
        else:
            edge_types_present = []

        contract_path = getattr(g, 'contract_path', getattr(g, 'path', None))

        print(f"  stem: {stem}")
        print(f"  nodes={n_nodes}, edges={n_edges}, edge_types={edge_types_present}")
        if x is not None:
            nonzero_per_feat = (x != 0).sum(dim=0).tolist()
            print(f"  nonzero_per_feature: {[f'{v:.0f}' for v in nonzero_per_feat]}")
        print(f"  contract_path: {contract_path}")

        result = read_sol_source(contract_path)
        if result:
            lines, actual_path = result
            patterns = VULN_PATTERNS.get(cls, [])
            pattern_lines = []
            for i, line in enumerate(lines):
                if any(p in line for p in patterns):
                    pattern_lines.append((i+1, line.strip()))

            if pattern_lines:
                print(f"  CONFIRMED: Found {len(pattern_lines)} pattern lines in source:")
                for lno, ltext in pattern_lines[:5]:
                    print(f"    L{lno}: {ltext[:120]}")
            else:
                print(f"  WARNING: No expected patterns found in source")
        else:
            print(f"  SOURCE NOT FOUND")

        found += 1

print("\n" + "="*70)
print("INVESTIGATION COMPLETE")
print("="*70)
