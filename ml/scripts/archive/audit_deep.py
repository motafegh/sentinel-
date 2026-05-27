#!/usr/bin/env python3
"""
Deep follow-up investigation.
"""
import sys, os, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/motafeq/projects/sentinel")

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

GRAPHS_DIR = Path("/home/motafeq/projects/sentinel/ml/data/graphs")
TOKENS_DIR = Path("/home/motafeq/projects/sentinel/ml/data/tokens_windowed")
CSV_PATH = Path("/home/motafeq/projects/sentinel/ml/data/processed/multilabel_index_deduped.csv")
BCCC_DIR = Path("/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes")

def safe_load_graph(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except:
        return None

def safe_load_token(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except:
        return None

df = pd.read_csv(CSV_PATH)
CLASSES = ['CallToUnknown','DenialOfService','ExternalBug','GasException',
           'IntegerUO','MishandledException','Reentrancy','Timestamp',
           'TransactionOrderDependence','UnusedReturn']
label_df = df[[c for c in CLASSES if c in df.columns]].fillna(0).astype(int)

# -----------------------------------------------------------------------
# DEEP CHECK 1: in_unchecked feature — is it ALWAYS 0?
# -----------------------------------------------------------------------
print("="*70)
print("DEEP CHECK 1: in_unchecked [9] — is it truly always zero?")
print("="*70)

all_stems = [p.stem for p in GRAPHS_DIR.glob("*.pt")]
random.seed(42)
sample_1000 = random.sample(all_stems, min(1000, len(all_stems)))

unchecked_nonzero = 0
total_nodes = 0
graphs_with_unchecked = 0
unchecked_max_val = 0.0
example_graph = None

for stem in sample_1000:
    g = safe_load_graph(GRAPHS_DIR / f"{stem}.pt")
    if g is None or not hasattr(g,'x') or g.x is None:
        continue
    x = g.x
    if x.shape[1] <= 9:
        continue
    total_nodes += x.shape[0]
    vals = x[:, 9].numpy()
    n_nz = (vals != 0).sum()
    unchecked_nonzero += n_nz
    if n_nz > 0:
        graphs_with_unchecked += 1
        if vals.max() > unchecked_max_val:
            unchecked_max_val = vals.max()
            example_graph = (stem, vals.max(), n_nz)

print(f"  Checked 1000 graphs ({total_nodes} nodes)")
print(f"  in_unchecked nonzero nodes: {unchecked_nonzero} ({100*unchecked_nonzero/max(total_nodes,1):.3f}%)")
print(f"  Graphs with any in_unchecked != 0: {graphs_with_unchecked}")
print(f"  Max in_unchecked value seen: {unchecked_max_val}")
if example_graph:
    print(f"  Example: {example_graph}")

# Look at the extractor code to understand in_unchecked
extractor_path = Path("/home/motafeq/projects/sentinel/ml/src/preprocessing/graph_extractor.py")
if extractor_path.exists():
    text = extractor_path.read_text()
    # Find in_unchecked related code
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'unchecked' in line.lower() or 'in_unchecked' in line.lower():
            start = max(0, i-2)
            end = min(len(lines), i+5)
            print(f"\n  Extractor L{i+1}: {line.strip()}")

# -----------------------------------------------------------------------
# DEEP CHECK 2: uses_block_globals — why is it near zero even for Timestamp?
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("DEEP CHECK 2: uses_block_globals [2] — near-zero for Timestamp?")
print("="*70)

# Find pure Timestamp contracts
ts_mask = (label_df['Timestamp'] == 1) & (label_df.drop(columns=['Timestamp']).sum(axis=1) == 0) if 'Timestamp' in label_df.columns else pd.Series([False]*len(df))
ts_stems = df[ts_mask]['md5_stem'].tolist()
random.seed(42)
random.shuffle(ts_stems)
ts_sample = ts_stems[:20]

print(f"\nPure Timestamp contracts: {len(ts_stems)}, sampling {len(ts_sample)}")
ts_bglobal_nodes = 0
ts_total_nodes = 0
ts_bglobal_graphs = 0

for stem in ts_sample:
    g = safe_load_graph(GRAPHS_DIR / f"{stem}.pt")
    if g is None or not hasattr(g,'x') or g.x is None:
        continue
    x = g.x
    if x.shape[1] <= 2:
        continue
    ts_total_nodes += x.shape[0]
    vals = x[:, 2].numpy()
    n_nz = (vals != 0).sum()
    ts_bglobal_nodes += n_nz
    if n_nz > 0:
        ts_bglobal_graphs += 1

print(f"  Pure Timestamp: {ts_bglobal_nodes}/{ts_total_nodes} nodes have uses_block_globals=1")
print(f"  Graphs with block globals: {ts_bglobal_graphs}/{len(ts_sample)}")

# Read 2 Timestamp .sol files and count block.timestamp manually
print("\n  Manual check of Timestamp .sol files:")
for stem in ts_sample[:5]:
    g = safe_load_graph(GRAPHS_DIR / f"{stem}.pt")
    if g is None:
        continue
    cp = getattr(g, 'contract_path', None)
    if cp is None:
        continue
    sol_path = Path("/home/motafeq/projects/sentinel") / cp.lstrip("/")
    if not sol_path.exists():
        # Try BCCC dir
        fname = Path(cp).name
        sol_path = BCCC_DIR / "Timestamp" / fname
    if not sol_path.exists():
        continue
    text = sol_path.read_text(errors='replace')
    n_timestamp = text.count('block.timestamp') + text.count(' now ')
    n_blocknumber = text.count('block.number')
    x = g.x
    n_bglobal = (x[:,2].numpy() != 0).sum() if x.shape[1] > 2 else 0
    print(f"    stem={stem[:16]}: block.timestamp={n_timestamp}, block.number={n_blocknumber} in source | graph uses_block_globals nodes={n_bglobal}")

# Check what "uses_block_globals" actually maps to in the extractor
print("\n  Searching extractor for uses_block_globals logic:")
if extractor_path.exists():
    text = extractor_path.read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'block' in line.lower() and ('global' in line.lower() or 'timestamp' in line.lower() or 'number' in line.lower()):
            print(f"    L{i+1}: {line.rstrip()}")

# -----------------------------------------------------------------------
# DEEP CHECK 3: Token file num_windows — ALL are W=4, why?
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("DEEP CHECK 3: Token files — all W=4 shape?")
print("="*70)

# Check a wider sample and look at actual content
token_files = list(TOKENS_DIR.glob("*.pt"))
random.seed(42)
sample_100 = random.sample(token_files, min(100, len(token_files)))

w_dist = Counter()
actual_content_windows = []

for tp in sample_100:
    t = safe_load_token(tp)
    if t is None:
        continue
    ids = t.get('input_ids', None) if isinstance(t, dict) else getattr(t, 'input_ids', None)
    nw = t.get('num_windows', None) if isinstance(t, dict) else getattr(t, 'num_windows', None)
    if ids is not None:
        shape = tuple(ids.shape)
        w_dist[shape] += 1
        # Check if last windows are all-padding
        if len(shape) == 2 and shape[0] == 4:
            # Count non-padding windows: a window is padding if all tokens are 0 or 1 (pad token)
            non_pad = 0
            for wi in range(shape[0]):
                row = ids[wi]
                # If the row has only pad tokens (typically 1 for BERT or 0)
                unique = set(row.tolist())
                if unique not in [{0}, {1}, {0,1}]:
                    non_pad += 1
                else:
                    # Also check if all are just padding (all 1s = PAD for BERT)
                    if row[0].item() != 0:  # non-zero first token = real content
                        non_pad += 1
            actual_content_windows.append((tp.stem, non_pad, nw))

print(f"\nShape distribution across {len(sample_100)} token files:")
for shape, cnt in sorted(w_dist.items()):
    print(f"  shape={shape}: {cnt} files ({100*cnt/len(sample_100):.1f}%)")

print(f"\nnum_windows field vs actual content (first 20):")
for stem, non_pad, nw_field in actual_content_windows[:20]:
    print(f"  stem={stem[:16]}: non-padding windows={non_pad}, num_windows field={nw_field}")

# Check one token file in detail
print("\nDetailed inspection of first token file:")
t = safe_load_token(sample_100[0])
if isinstance(t, dict):
    print(f"  Keys: {list(t.keys())}")
    for k, v in t.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, (int, float, str)):
            print(f"  {k}: {v}")

# Check the windowed tokenizer script
retok_path = Path("/home/motafeq/projects/sentinel/ml/scripts/retokenize_windowed.py")
if retok_path.exists():
    text = retok_path.read_text()
    lines = text.splitlines()
    print(f"\nRetokenize windowed script ({len(lines)} lines):")
    for i, line in enumerate(lines):
        if 'num_windows' in line or 'stride' in line or 'MAX_WINDOW' in line or 'padding' in line.lower():
            print(f"  L{i+1}: {line.rstrip()}")

# -----------------------------------------------------------------------
# DEEP CHECK 4: ext_call_count normalization check
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("DEEP CHECK 4: ext_call_count [11] max values and normalization")
print("="*70)

max_ext_calls = 0.0
high_ext_examples = []

for stem in sample_1000[:500]:
    g = safe_load_graph(GRAPHS_DIR / f"{stem}.pt")
    if g is None or not hasattr(g,'x') or g.x is None:
        continue
    x = g.x
    if x.shape[1] <= 11:
        continue
    vals = x[:, 11].numpy()
    mx = vals.max()
    if mx > max_ext_calls:
        max_ext_calls = mx
    if mx > 0.5:
        high_ext_examples.append((stem, float(mx), int((vals > 0).sum())))

print(f"  Max ext_call_count seen: {max_ext_calls:.4f}")
print(f"  Graphs with any ext_call_count > 0.5: {len(high_ext_examples)}")
for ex in high_ext_examples[:5]:
    print(f"    {ex[0][:16]}: max={ex[1]:.4f}, nonzero_nodes={ex[2]}")
    # What does 0.816 mean? If normalized as count/MAX
    estimated_raw = ex[1] * 100  # Wild guess
    print(f"      If normalized /100: raw_count≈{estimated_raw:.1f}")

# Check extractor for ext_call_count normalization
print("\n  Searching extractor for ext_call_count:")
if extractor_path.exists():
    text = extractor_path.read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'ext_call' in line.lower():
            start = max(0, i-1)
            end = min(len(lines), i+3)
            for j in range(start, end):
                print(f"    L{j+1}: {lines[j].rstrip()}")
            print()

# -----------------------------------------------------------------------
# DEEP CHECK 5: complexity and loc max values — normalization bug
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("DEEP CHECK 5: complexity [5] and loc [6] — raw values (known bug)")
print("="*70)

complexity_vals = []
loc_vals = []

for stem in sample_1000[:500]:
    g = safe_load_graph(GRAPHS_DIR / f"{stem}.pt")
    if g is None or not hasattr(g,'x') or g.x is None:
        continue
    x = g.x
    if x.shape[1] <= 6:
        continue
    complexity_vals.extend(x[:, 5].numpy().tolist())
    loc_vals.extend(x[:, 6].numpy().tolist())

ca = np.array(complexity_vals)
la = np.array(loc_vals)
print(f"  complexity: min={ca.min():.0f}, p50={np.percentile(ca,50):.0f}, p90={np.percentile(ca,90):.0f}, p99={np.percentile(ca,99):.0f}, max={ca.max():.0f}")
print(f"  loc:        min={la.min():.0f}, p50={np.percentile(la,50):.0f}, p90={np.percentile(la,90):.0f}, p99={np.percentile(la,99):.0f}, max={la.max():.0f}")
print(f"  complexity > 1.0: {(ca > 1.0).sum()} nodes ({100*(ca>1.0).mean():.1f}%) — UNNORMALIZED")
print(f"  loc > 1.0: {(la > 1.0).sum()} nodes ({100*(la>1.0).mean():.1f}%) — UNNORMALIZED")

# -----------------------------------------------------------------------
# DEEP CHECK 6: visibility feature values — 0,1,2 or something else?
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("DEEP CHECK 6: visibility [1] — value distribution")
print("="*70)

vis_vals = []
for stem in sample_1000[:300]:
    g = safe_load_graph(GRAPHS_DIR / f"{stem}.pt")
    if g is None or not hasattr(g,'x') or g.x is None:
        continue
    x = g.x
    vis_vals.extend(x[:, 1].numpy().tolist())

vis_counter = Counter([round(v, 3) for v in vis_vals])
print(f"  Unique visibility values (top 10): {vis_counter.most_common(10)}")
print(f"  Max: {max(vis_vals):.3f}, Min: {min(vis_vals):.3f}")

# Is 2.0 normalized to 1.0 or left as raw?
if max(vis_vals) > 1.0:
    print(f"  BUG CONFIRMED: visibility is NOT normalized (max={max(vis_vals):.3f})")
else:
    print(f"  OK: visibility appears normalized (max={max(vis_vals):.3f})")

# -----------------------------------------------------------------------
# DEEP CHECK 7: Reentrancy — CEI violation detection
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("DEEP CHECK 7: Reentrancy source pattern check")
print("="*70)

# Get more pure Reentrancy samples
reen_mask = (label_df['Reentrancy'] == 1) & (label_df.drop(columns=['Reentrancy']).sum(axis=1) == 0) if 'Reentrancy' in label_df.columns else pd.Series([False]*len(df))
reen_stems = df[reen_mask]['md5_stem'].tolist()
random.seed(42)
random.shuffle(reen_stems)

found = 0
for stem in reen_stems[:30]:
    g = safe_load_graph(GRAPHS_DIR / f"{stem}.pt")
    if g is None:
        continue
    cp = getattr(g, 'contract_path', None)
    if cp is None:
        continue
    sol_path = Path("/home/motafeq/projects/sentinel") / cp.lstrip("/")
    if not sol_path.exists():
        fname = Path(cp).name
        sol_path = BCCC_DIR / "Reentrancy" / fname
    if not sol_path.exists():
        continue

    text = sol_path.read_text(errors='replace')
    lines = text.splitlines()

    # Look for CEI violations: .call before state update
    call_lines = [i for i, l in enumerate(lines) if '.call(' in l or 'call{value' in l or '.send(' in l or '.transfer(' in l]
    if call_lines:
        print(f"\n  stem: {stem}")
        print(f"  path: {sol_path}")
        # Check if there's a state variable assignment AFTER the call
        for cl in call_lines[:3]:
            start = max(0, cl-3)
            end = min(len(lines), cl+5)
            for i in range(start, end):
                print(f"    L{i+1}: {lines[i].rstrip()[:120]}")
        found += 1
        if found >= 3:
            break
    else:
        print(f"\n  stem: {stem} — NO .call/.send/.transfer found!")
        print(f"  path: {sol_path}")
        # Show first 30 lines
        for i, l in enumerate(lines[:20]):
            print(f"    L{i+1}: {l.rstrip()[:120]}")
        found += 1
        if found >= 2:
            break

# -----------------------------------------------------------------------
# DEEP CHECK 8: EMITS edge type — always 0?
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("DEEP CHECK 8: EMITS edge type (3) — always absent?")
print("="*70)

emits_total = 0
emits_graphs = 0
for stem in sample_1000[:500]:
    g = safe_load_graph(GRAPHS_DIR / f"{stem}.pt")
    if g is None or not hasattr(g,'edge_attr') or g.edge_attr is None:
        continue
    ea = g.edge_attr.squeeze() if g.edge_attr.dim() > 1 else g.edge_attr
    n_emits = (ea == 3).sum().item()
    emits_total += n_emits
    if n_emits > 0:
        emits_graphs += 1

print(f"  EMITS edges in 500 graphs: {emits_total} total, {emits_graphs} graphs with EMITS edges")
if emits_graphs == 0:
    print("  BUG: EMITS edges are never present — event emission not captured in CFG")

# Check INHERITS edges too
inherits_total = 0
inherits_graphs = 0
for stem in sample_1000[:500]:
    g = safe_load_graph(GRAPHS_DIR / f"{stem}.pt")
    if g is None or not hasattr(g,'edge_attr') or g.edge_attr is None:
        continue
    ea = g.edge_attr.squeeze() if g.edge_attr.dim() > 1 else g.edge_attr
    n_inh = (ea == 4).sum().item()
    inherits_total += n_inh
    if n_inh > 0:
        inherits_graphs += 1

print(f"  INHERITS edges in 500 graphs: {inherits_total} total, {inherits_graphs} graphs with INHERITS edges")

# -----------------------------------------------------------------------
# DEEP CHECK 9: contract_path=None prevalence
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("DEEP CHECK 9: contract_path=None prevalence")
print("="*70)

no_path = 0
has_path = 0
no_path_examples = []

for stem in sample_1000[:500]:
    g = safe_load_graph(GRAPHS_DIR / f"{stem}.pt")
    if g is None:
        continue
    cp = getattr(g, 'contract_path', None)
    if cp is None:
        no_path += 1
        no_path_examples.append(stem)
    else:
        has_path += 1

total = no_path + has_path
print(f"  contract_path=None: {no_path}/{total} ({100*no_path/max(total,1):.1f}%)")
print(f"  contract_path set: {has_path}/{total}")
print(f"  Examples of no-path graphs: {no_path_examples[:5]}")

# For no-path graphs, can we find the source anyway?
print("\n  Attempting to find source for no-path graphs:")
for stem in no_path_examples[:5]:
    # Try finding in any BCCC folder
    found_path = None
    for cls_dir in BCCC_DIR.iterdir():
        if cls_dir.is_dir():
            for sol_file in cls_dir.glob("*.sol"):
                if sol_file.stem[:16] == stem[:16]:
                    found_path = sol_file
                    break
        if found_path:
            break
    # Also check if stem matches MD5 -> SHA256 mapping
    row = df[df['md5_stem'] == stem]
    labels = []
    if len(row) > 0:
        row = row.iloc[0]
        labels = [c for c in CLASSES if c in row.index and row[c] == 1]
    print(f"  stem={stem[:16]}: labels={labels}, found_sol={found_path is not None}")

# -----------------------------------------------------------------------
# DEEP CHECK 10: Token file num_windows field — what is stride?
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("DEEP CHECK 10: Token num_windows field — actual values stored")
print("="*70)

nw_dist = Counter()
stride_dist = Counter()
for tp in sample_100[:50]:
    t = safe_load_token(tp)
    if t is None or not isinstance(t, dict):
        continue
    nw = t.get('num_windows', None)
    stride = t.get('stride', None)
    if nw is not None:
        nw_dist[int(nw)] += 1
    if stride is not None:
        stride_dist[int(stride) if isinstance(stride, (int, float)) else str(stride)] += 1

print(f"  num_windows distribution: {dict(nw_dist)}")
print(f"  stride distribution: {dict(stride_dist)}")

# Show some files where num_windows < 4
print("\n  Files where num_windows < 4:")
small_nw = []
for tp in token_files[:200]:
    t = safe_load_token(tp)
    if t is None or not isinstance(t, dict):
        continue
    nw = t.get('num_windows', None)
    if nw is not None and nw < 4:
        small_nw.append((tp.stem, nw, t.get('num_tokens', None)))
    if len(small_nw) >= 10:
        break

if small_nw:
    for s, nw, nt in small_nw[:10]:
        print(f"  stem={s[:16]}: num_windows={nw}, num_tokens={nt}")
else:
    print("  None found in first 200 — checking larger sample...")
    for tp in token_files[:2000]:
        t = safe_load_token(tp)
        if t is None or not isinstance(t, dict):
            continue
        nw = t.get('num_windows', None)
        if nw is not None and nw < 4:
            small_nw.append((tp.stem, nw, t.get('num_tokens', None)))
        if len(small_nw) >= 5:
            break
    if small_nw:
        for s, nw, nt in small_nw[:5]:
            print(f"  stem={s[:16]}: num_windows={nw}, num_tokens={nt}")
    else:
        print("  BUG/FINDING: ALL token files appear to have num_windows=4 (always W=4)")
        # Is MAX_WINDOWS being used? Check the tokenizer code

# -----------------------------------------------------------------------
# DEEP CHECK 11: Check for mismatched CSV/graph stems
# -----------------------------------------------------------------------
print("\n" + "="*70)
print("DEEP CHECK 11: CSV-graph coverage mismatch")
print("="*70)

graph_stems = set(p.stem for p in GRAPHS_DIR.glob("*.pt"))
csv_stems = set(df['md5_stem'].tolist())
token_stems = set(p.stem for p in TOKENS_DIR.glob("*.pt"))

print(f"  Graphs on disk: {len(graph_stems)}")
print(f"  Stems in CSV: {len(csv_stems)}")
print(f"  Token files: {len(token_stems)}")
print(f"  CSV stems with graphs: {len(csv_stems & graph_stems)}")
print(f"  CSV stems without graphs: {len(csv_stems - graph_stems)}")
print(f"  Graphs not in CSV: {len(graph_stems - csv_stems)}")
print(f"  CSV stems with tokens: {len(csv_stems & token_stems)}")
print(f"  CSV stems without tokens: {len(csv_stems - token_stems)}")

# What labels do the missing-graph entries have?
missing_graph_stems = csv_stems - graph_stems
if missing_graph_stems:
    missing_df = df[df['md5_stem'].isin(missing_graph_stems)]
    print(f"\n  Missing graphs — class distribution:")
    for cls in CLASSES:
        if cls in missing_df.columns:
            cnt = missing_df[cls].sum()
            print(f"    {cls}: {cnt:.0f}")

print("\n" + "="*70)
print("DEEP INVESTIGATION COMPLETE")
print("="*70)
