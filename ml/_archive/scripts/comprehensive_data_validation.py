#!/usr/bin/env python3
"""
SENTINEL - Comprehensive Data Validation Suite
Validates ALL data before training to prevent downstream issues.

Run this BEFORE Module 3 to ensure data integrity.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import random
import json
from datetime import datetime

print("=" * 80)
print("SENTINEL - COMPREHENSIVE DATA VALIDATION")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
GRAPH_DIR = Path('ml/data/graphs')
TOKEN_DIR = Path('ml/data/tokens')
LABELS_CSV = Path('ml/data/processed/contract_labels_correct.csv')
SAMPLE_SIZE = 1000  # For detailed validation
FULL_SCAN = True    # Set False for quick check

validation_results = {
    "timestamp": datetime.now().isoformat(),
    "passed": [],
    "warnings": [],
    "errors": [],
    "statistics": {}
}

def log_pass(msg):
    print(f"  ✅ {msg}")
    validation_results["passed"].append(msg)

def log_warning(msg):
    print(f"  ⚠️  {msg}")
    validation_results["warnings"].append(msg)

def log_error(msg):
    print(f"  ❌ {msg}")
    validation_results["errors"].append(msg)

# ============================================================================
# CATEGORY 1: FILE-LEVEL INTEGRITY
# ============================================================================
print("\n" + "="*80)
print("CATEGORY 1: FILE-LEVEL INTEGRITY")
print("="*80)

print("\n1.1 File Counts")
graph_files = list(GRAPH_DIR.glob('*.pt'))
token_files = list(TOKEN_DIR.glob('*.pt'))

print(f"  Graph files: {len(graph_files):,}")
print(f"  Token files: {len(token_files):,}")

if len(graph_files) == 0:
    log_error("No graph files found!")
elif len(graph_files) < 50000:
    log_warning(f"Only {len(graph_files):,} graphs (expected 60K+)")
else:
    log_pass(f"Graph count OK: {len(graph_files):,}")

if abs(len(graph_files) - len(token_files)) > 20:
    log_warning(f"Graph/token count mismatch: {abs(len(graph_files) - len(token_files))}")
else:
    log_pass(f"Graph/token counts aligned (diff: {abs(len(graph_files) - len(token_files))})")

print("\n1.2 File Size Checks")
graph_sizes = [f.stat().st_size for f in graph_files[:1000]]
token_sizes = [f.stat().st_size for f in token_files[:1000]]

zero_byte_graphs = [f for f in graph_files[:1000] if f.stat().st_size == 0]
zero_byte_tokens = [f for f in token_files[:1000] if f.stat().st_size == 0]

if zero_byte_graphs:
    log_error(f"Found {len(zero_byte_graphs)} zero-byte graph files!")
else:
    log_pass("No zero-byte graph files")

if zero_byte_tokens:
    log_error(f"Found {len(zero_byte_tokens)} zero-byte token files!")
else:
    log_pass("No zero-byte token files")

print(f"  Graph size: {np.mean(graph_sizes)/1024:.1f} KB (avg)")
print(f"  Token size: {np.mean(token_sizes)/1024:.1f} KB (avg)")

print("\n1.3 Hash Format Validation")
invalid_hashes = []
for f in random.sample(graph_files, min(100, len(graph_files))):
    hash_str = f.stem
    if len(hash_str) != 32:
        invalid_hashes.append((f.name, len(hash_str)))
    elif not all(c in '0123456789abcdef' for c in hash_str):
        invalid_hashes.append((f.name, "non-hex"))

if invalid_hashes:
    log_error(f"Found {len(invalid_hashes)} invalid MD5 hashes: {invalid_hashes[:3]}")
else:
    log_pass("All sampled hashes are valid MD5 (32 hex chars)")

# ============================================================================
# CATEGORY 2: GRAPH QUALITY
# ============================================================================
print("\n" + "="*80)
print("CATEGORY 2: GRAPH QUALITY")
print("="*80)

print("\n2.1 Graph Loading Test")
sample_graphs = random.sample(graph_files, min(SAMPLE_SIZE, len(graph_files)))

load_errors = []
graph_stats = {
    "num_nodes": [],
    "num_edges": [],
    "node_features": [],
    "labels": [],
    "has_contract_path": [],
    "has_contract_hash": [],
}

for gf in sample_graphs:
    try:
        graph = torch.load(gf, map_location='cpu', weights_only=False)
        
        # Basic attributes
        graph_stats["num_nodes"].append(graph.x.shape[0])
        graph_stats["num_edges"].append(graph.edge_index.shape[1])
        graph_stats["node_features"].append(graph.x.shape[1])
        graph_stats["labels"].append(graph.y.item() if hasattr(graph, 'y') else None)
        graph_stats["has_contract_path"].append(hasattr(graph, 'contract_path'))
        graph_stats["has_contract_hash"].append(hasattr(graph, 'contract_hash'))
        
    except Exception as e:
        load_errors.append((gf.name, str(e)))

if load_errors:
    log_error(f"Failed to load {len(load_errors)}/{SAMPLE_SIZE} graphs")
    print(f"    Sample errors: {load_errors[:3]}")
else:
    log_pass(f"All {SAMPLE_SIZE} sampled graphs loaded successfully")

print("\n2.2 Graph Structure Validation")

# Node features
feature_dims = set(graph_stats["node_features"])
if len(feature_dims) == 1 and list(feature_dims)[0] == 8:
    log_pass("All graphs have 8-dimensional node features")
else:
    log_error(f"Inconsistent feature dimensions: {feature_dims}")

# Check for degenerate graphs
min_nodes = min(graph_stats["num_nodes"])
max_nodes = max(graph_stats["num_nodes"])
mean_nodes = np.mean(graph_stats["num_nodes"])

print(f"  Nodes per graph: min={min_nodes}, max={max_nodes}, mean={mean_nodes:.1f}")

if min_nodes == 0:
    log_error("Found graphs with 0 nodes!")
elif min_nodes < 2:
    log_warning(f"Found graphs with only {min_nodes} node (degenerate)")
else:
    log_pass(f"All graphs have >= {min_nodes} nodes")

if max_nodes > 1000:
    log_warning(f"Found very large graph with {max_nodes} nodes")

# Edge validation
min_edges = min(graph_stats["num_edges"])
max_edges = max(graph_stats["num_edges"])
mean_edges = np.mean(graph_stats["num_edges"])

print(f"  Edges per graph: min={min_edges}, max={max_edges}, mean={mean_edges:.1f}")

disconnected_count = sum(1 for e in graph_stats["num_edges"] if e == 0)
if disconnected_count > 0:
    log_warning(f"{disconnected_count}/{SAMPLE_SIZE} graphs have no edges (disconnected)")
else:
    log_pass("All graphs have edges (connected)")

print("\n2.3 Node Feature Quality")

# Load a few graphs and check feature ranges
feature_samples = []
nan_inf_count = 0

for gf in random.sample(graph_files, min(100, len(graph_files))):
    graph = torch.load(gf, map_location='cpu', weights_only=False)
    
    if torch.isnan(graph.x).any():
        nan_inf_count += 1
    if torch.isinf(graph.x).any():
        nan_inf_count += 1
    
    feature_samples.append(graph.x)

if nan_inf_count > 0:
    log_error(f"Found {nan_inf_count}/100 graphs with NaN or Inf features")
else:
    log_pass("No NaN or Inf values in node features (sample)")

# Feature range check
all_features = torch.cat(feature_samples, dim=0)
feature_mins = all_features.min(dim=0).values
feature_maxs = all_features.max(dim=0).values

print(f"  Feature ranges (8 dims):")
for i in range(8):
    print(f"    Dim {i}: [{feature_mins[i]:.2f}, {feature_maxs[i]:.2f}]")

# Check for all-zero features
zero_dims = (feature_maxs == 0).sum().item()
if zero_dims > 0:
    log_warning(f"{zero_dims}/8 feature dimensions are all-zero")
else:
    log_pass("All feature dimensions have non-zero values")

print("\n2.4 Label Quality")

label_counts = Counter([l for l in graph_stats["labels"] if l is not None])
print(f"  Label distribution (sample):")
for label, count in sorted(label_counts.items()):
    print(f"    Label {label}: {count} ({count/SAMPLE_SIZE*100:.1f}%)")

if None in graph_stats["labels"]:
    log_error("Some graphs missing labels (y attribute)")
elif set(label_counts.keys()) != {0, 1}:
    log_error(f"Labels not binary: {set(label_counts.keys())}")
else:
    log_pass("All graphs have binary labels (0 or 1)")

# Check class balance
if len(label_counts) == 2:
    ratio = label_counts[1] / label_counts[0]
    if ratio > 10 or ratio < 0.1:
        log_warning(f"Severe class imbalance: {ratio:.1f}:1")
    else:
        log_pass(f"Class balance is trainable: {ratio:.2f}:1 ratio")

print("\n2.5 Metadata Presence")

missing_path = sum(1 for x in graph_stats["has_contract_path"] if not x)
missing_hash = sum(1 for x in graph_stats["has_contract_hash"] if not x)

if missing_path > 0:
    log_error(f"{missing_path}/{SAMPLE_SIZE} graphs missing contract_path")
else:
    log_pass("All graphs have contract_path attribute")

if missing_hash > 0:
    log_error(f"{missing_hash}/{SAMPLE_SIZE} graphs missing contract_hash")
else:
    log_pass("All graphs have contract_hash attribute")

# ============================================================================
# CATEGORY 3: TOKEN QUALITY
# ============================================================================
print("\n" + "="*80)
print("CATEGORY 3: TOKEN QUALITY")
print("="*80)

print("\n3.1 Token Loading Test")
sample_tokens = random.sample(token_files, min(SAMPLE_SIZE, len(token_files)))

token_load_errors = []
token_stats = {
    "input_ids_shape": [],
    "attention_mask_shape": [],
    "num_tokens": [],
    "truncated": [],
    "has_contract_path": [],
    "has_contract_hash": [],
}

for tf in sample_tokens:
    try:
        token = torch.load(tf, map_location='cpu', weights_only=False)
        
        token_stats["input_ids_shape"].append(token['input_ids'].shape)
        token_stats["attention_mask_shape"].append(token['attention_mask'].shape)
        token_stats["num_tokens"].append(token.get('num_tokens', -1))
        token_stats["truncated"].append(token.get('truncated', None))
        token_stats["has_contract_path"].append('contract_path' in token)
        token_stats["has_contract_hash"].append('contract_hash' in token)
        
    except Exception as e:
        token_load_errors.append((tf.name, str(e)))

if token_load_errors:
    log_error(f"Failed to load {len(token_load_errors)}/{SAMPLE_SIZE} tokens")
else:
    log_pass(f"All {SAMPLE_SIZE} sampled tokens loaded successfully")

print("\n3.2 Token Format Validation")

# Check shapes
shapes = set(token_stats["input_ids_shape"])
if len(shapes) == 1 and list(shapes)[0] == torch.Size([512]):
    log_pass("All tokens have shape [512] (CodeBERT format)")
else:
    log_error(f"Inconsistent token shapes: {shapes}")

# Check attention masks match
mask_shapes = set(token_stats["attention_mask_shape"])
if mask_shapes == shapes:
    log_pass("Attention masks match input_ids shapes")
else:
    log_error("Attention mask shape mismatch")

print("\n3.3 Token Content Validation")

# Load sample and check token ID ranges
token_id_errors = []
all_padding_count = 0

for tf in random.sample(token_files, min(100, len(token_files))):
    token = torch.load(tf, map_location='cpu', weights_only=False)
    
    input_ids = token['input_ids']
    attention_mask = token['attention_mask']
    
    # Check token IDs in valid range (CodeBERT vocab: 0-50264)
    if input_ids.min() < 0 or input_ids.max() > 50264:
        token_id_errors.append(tf.name)
    
    # Check for all-padding sequences
    if attention_mask.sum() == 0:
        all_padding_count += 1
    
    # Check special tokens
    if input_ids[0] != 0:  # [CLS] token
        token_id_errors.append(f"{tf.name} (missing [CLS])")

if token_id_errors:
    log_error(f"Found {len(token_id_errors)} tokens with invalid IDs or format")
else:
    log_pass("All token IDs in valid CodeBERT range (0-50264)")

if all_padding_count > 0:
    log_error(f"Found {all_padding_count} all-padding sequences!")
else:
    log_pass("No all-padding sequences")

# Truncation rate
truncated_count = sum(1 for t in token_stats["truncated"] if t)
truncation_rate = truncated_count / len(token_stats["truncated"]) * 100
print(f"  Truncation rate: {truncation_rate:.1f}%")

if truncation_rate > 98:
    log_warning(f"Very high truncation rate: {truncation_rate:.1f}%")
else:
    log_pass(f"Truncation rate is reasonable: {truncation_rate:.1f}%")

print("\n3.4 Token Metadata")

missing_path_token = sum(1 for x in token_stats["has_contract_path"] if not x)
missing_hash_token = sum(1 for x in token_stats["has_contract_hash"] if not x)

if missing_path_token > 0:
    log_error(f"{missing_path_token}/{SAMPLE_SIZE} tokens missing contract_path")
else:
    log_pass("All tokens have contract_path")

if missing_hash_token > 0:
    log_error(f"{missing_hash_token}/{SAMPLE_SIZE} tokens missing contract_hash")
else:
    log_pass("All tokens have contract_hash")

# ============================================================================
# CATEGORY 4: PAIRING INTEGRITY
# ============================================================================
print("\n" + "="*80)
print("CATEGORY 4: PAIRING INTEGRITY")
print("="*80)

print("\n4.1 Hash Pairing")
graph_hashes = {f.stem for f in graph_files}
token_hashes = {f.stem for f in token_files}

intersection = graph_hashes & token_hashes
only_graphs = graph_hashes - token_hashes
only_tokens = token_hashes - graph_hashes

pairing_rate = len(intersection) / max(len(graph_hashes), len(token_hashes)) * 100

print(f"  Paired (both): {len(intersection):,}")
print(f"  Only graph: {len(only_graphs):,}")
print(f"  Only token: {len(only_tokens):,}")
print(f"  Pairing rate: {pairing_rate:.2f}%")

if pairing_rate < 95:
    log_error(f"Low pairing rate: {pairing_rate:.2f}%")
elif pairing_rate < 99:
    log_warning(f"Pairing rate below 99%: {pairing_rate:.2f}%")
else:
    log_pass(f"Excellent pairing rate: {pairing_rate:.2f}%")

print("\n4.2 Contract Path Consistency")

path_mismatches = []
for md5_hash in random.sample(list(intersection), min(100, len(intersection))):
    graph = torch.load(GRAPH_DIR / f'{md5_hash}.pt', map_location='cpu', weights_only=False)
    token = torch.load(TOKEN_DIR / f'{md5_hash}.pt', map_location='cpu', weights_only=False)
    
    if graph.contract_path != token['contract_path']:
        path_mismatches.append(md5_hash)

if path_mismatches:
    log_error(f"Found {len(path_mismatches)} path mismatches: {path_mismatches[:3]}")
else:
    log_pass("All paired files have matching contract_path (sample)")

print("\n4.3 Hash Consistency")

hash_mismatches = []
for md5_hash in random.sample(list(intersection), min(100, len(intersection))):
    graph = torch.load(GRAPH_DIR / f'{md5_hash}.pt', map_location='cpu', weights_only=False)
    token = torch.load(TOKEN_DIR / f'{md5_hash}.pt', map_location='cpu', weights_only=False)
    
    if hasattr(graph, 'contract_hash') and 'contract_hash' in token:
        if graph.contract_hash != token['contract_hash']:
            hash_mismatches.append(md5_hash)

if hash_mismatches:
    log_error(f"Found {len(hash_mismatches)} hash mismatches")
else:
    log_pass("All paired files have matching contract_hash (sample)")

# ============================================================================
# CATEGORY 5: LABEL INTEGRITY
# ============================================================================
print("\n" + "="*80)
print("CATEGORY 5: LABEL INTEGRITY")
print("="*80)

if not LABELS_CSV.exists():
    log_warning("contract_labels_correct.csv not found - skipping CSV validation")
else:
    print("\n5.1 CSV Label Validation")
    labels_df = pd.read_csv(LABELS_CSV)
    
    print(f"  CSV rows: {len(labels_df):,}")
    print(f"  CSV columns: {list(labels_df.columns)[:5]}...")
    
    if 'file_hash' not in labels_df.columns:
        log_error("CSV missing 'file_hash' column")
    elif 'binary_label' not in labels_df.columns:
        log_error("CSV missing 'binary_label' column")
    else:
        csv_hashes = set(labels_df['file_hash'].values)
        
        print(f"  Unique hashes in CSV: {len(csv_hashes):,}")
        
        # Check graph label vs CSV label
        print("\n5.2 Graph vs CSV Label Match")
        
        label_mismatches = []
        for md5_hash in random.sample(list(intersection), min(100, len(intersection))):
            graph = torch.load(GRAPH_DIR / f'{md5_hash}.pt', map_location='cpu', weights_only=False)
            
            # Extract SHA256 from path
            sha256_hash = graph.contract_path.split('/')[-1].replace('.sol', '')
            
            # Lookup in CSV
            csv_row = labels_df[labels_df['file_hash'] == sha256_hash]
            
            if len(csv_row) > 0:
                csv_label = csv_row['binary_label'].values[0]
                graph_label = graph.y.item()
                
                if csv_label != graph_label:
                    label_mismatches.append((md5_hash, graph_label, csv_label))
        
        if label_mismatches:
            log_error(f"Found {len(label_mismatches)} label mismatches!")
            print(f"    Sample: {label_mismatches[:3]}")
        else:
            log_pass("All graph labels match CSV (sample)")

print("\n5.3 Full Dataset Label Distribution")

all_labels = []
for gf in graph_files:
    try:
        graph = torch.load(gf, map_location='cpu', weights_only=False)
        all_labels.append(graph.y.item())
    except:
        pass

full_label_counts = Counter(all_labels)
print(f"  Total graphs loaded: {len(all_labels):,}")
for label, count in sorted(full_label_counts.items()):
    print(f"  Label {label}: {count:,} ({count/len(all_labels)*100:.2f}%)")

validation_results["statistics"]["label_distribution"] = dict(full_label_counts)

expected_ratio = full_label_counts[1] / full_label_counts[0]
if 1.5 < expected_ratio < 2.5:
    log_pass(f"Label distribution matches expected: {expected_ratio:.2f}:1")
else:
    log_warning(f"Label distribution differs from expected: {expected_ratio:.2f}:1")

# ============================================================================
# CATEGORY 6: STATISTICAL SANITY
# ============================================================================
print("\n" + "="*80)
print("CATEGORY 6: STATISTICAL SANITY")
print("="*80)

print("\n6.1 Outlier Detection")

node_counts = [g.x.shape[0] for g in [torch.load(f, map_location='cpu', weights_only=False) for f in random.sample(graph_files, min(1000, len(graph_files)))]]

q1, q3 = np.percentile(node_counts, [25, 75])
iqr = q3 - q1
outlier_threshold = q3 + 3 * iqr

outliers = [n for n in node_counts if n > outlier_threshold]

if outliers:
    log_warning(f"Found {len(outliers)} graphs with extreme node counts: max={max(outliers)}")
else:
    log_pass("No extreme outliers in node counts")

print(f"  Node count stats: Q1={q1:.0f}, Q3={q3:.0f}, Outlier threshold={outlier_threshold:.0f}")

# ============================================================================
# CATEGORY 7: FUTURE-PROOFING
# ============================================================================
print("\n" + "="*80)
print("CATEGORY 7: FUTURE-PROOFING (Module 3+)")
print("="*80)

print("\n7.1 PyTorch DataLoader Compatibility")
try:
    from torch_geometric.data import Data
    log_pass("PyTorch Geometric installed")
except ImportError:
    log_error("PyTorch Geometric not installed!")

print("\n7.2 Batching Test")
try:
    from torch_geometric.loader import DataLoader
    
    # Load 10 graphs and try batching
    sample = [torch.load(f, map_location='cpu', weights_only=False) for f in graph_files[:10]]
    loader = DataLoader(sample, batch_size=4)
    
    batch = next(iter(loader))
    log_pass(f"Graph batching works (batch size: {batch.num_graphs})")
except Exception as e:
    log_error(f"Graph batching failed: {e}")

print("\n7.3 Train/Val/Test Split Feasibility")

total_samples = len(intersection)
min_class_count = min(full_label_counts.values())

print(f"  Total paired samples: {total_samples:,}")
print(f"  Minimum class count: {min_class_count:,}")

# Check if stratified split is possible
train_size = int(0.7 * min_class_count)
val_size = int(0.15 * min_class_count)
test_size = min_class_count - train_size - val_size

print(f"  Proposed split (per class):")
print(f"    Train: {train_size:,} samples")
print(f"    Val:   {val_size:,} samples")
print(f"    Test:  {test_size:,} samples")

if val_size < 100:
    log_warning(f"Validation set would be very small: {val_size}")
elif val_size < 1000:
    log_warning(f"Validation set is small: {val_size}")
else:
    log_pass(f"Sufficient samples for stratified split")

print("\n7.4 Multi-Class Compatibility (Module 4)")

if LABELS_CSV.exists():
    labels_df = pd.read_csv(LABELS_CSV)
    class_columns = [c for c in labels_df.columns if c.startswith('Class')]
    
    if len(class_columns) >= 12:
        log_pass(f"Multi-class labels available ({len(class_columns)} classes)")
    else:
        log_warning("Multi-class labels not fully available")
else:
    log_warning("Cannot verify multi-class compatibility (no CSV)")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

print(f"\n✅ Passed: {len(validation_results['passed'])}")
print(f"⚠️  Warnings: {len(validation_results['warnings'])}")
print(f"❌ Errors: {len(validation_results['errors'])}")

if validation_results['errors']:
    print("\n❌ CRITICAL ERRORS (Must fix before training):")
    for err in validation_results['errors']:
        print(f"  • {err}")

if validation_results['warnings']:
    print("\n⚠️  WARNINGS (Review recommended):")
    for warn in validation_results['warnings']:
        print(f"  • {warn}")

# Overall verdict
print("\n" + "="*80)
if len(validation_results['errors']) == 0:
    if len(validation_results['warnings']) == 0:
        print("🎉 VERDICT: PERFECT - Ready for training!")
        print("   All checks passed. Data quality is production-grade.")
    elif len(validation_results['warnings']) <= 3:
        print("✅ VERDICT: GOOD - Ready for training with minor notes")
        print("   Warnings are non-blocking. Safe to proceed.")
    else:
        print("⚠️  VERDICT: ACCEPTABLE - Review warnings before training")
        print("   Multiple warnings detected. Review recommended.")
else:
    print("❌ VERDICT: ISSUES FOUND - Fix errors before training")
    print("   Critical errors detected. Training may fail.")

print("="*80)

# Save report
report_path = Path('ml/data/validation_report.json')
with open(report_path, 'w') as f:
    json.dump(validation_results, f, indent=2)

print(f"\n📄 Full report saved: {report_path}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

