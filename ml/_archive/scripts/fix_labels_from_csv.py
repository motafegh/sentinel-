"""Fix graph labels using CSV ground truth."""
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("=" * 70)
print("FIXING GRAPH LABELS FROM CSV GROUND TRUTH")
print("=" * 70)
print()

# Load label mapping
labels_df = pd.read_csv('ml/data/processed/contract_labels_correct.csv')
print(f"Loaded labels for {len(labels_df):,} contracts")
print()

# Create hash -> label mapping
hash_to_binary = dict(zip(labels_df['file_hash'], labels_df['binary_label']))
hash_to_class = dict(zip(labels_df['file_hash'], labels_df['class_label']))

print(f"Created mappings for {len(hash_to_binary):,} unique hashes")
print()

# Process all graphs
graph_dir = Path('ml/data/graphs')
graph_files = list(graph_dir.glob('*.pt'))

print(f"Found {len(graph_files):,} graph files")
print()

stats = {
    'total': 0,
    'fixed': 0,
    'unmapped': 0,
    'errors': 0
}

label_counts = {0: 0, 1: 0}

print("Processing graphs...")
for graph_file in tqdm(graph_files):
    stats['total'] += 1
    
    try:
        # Load graph
        graph = torch.load(graph_file, weights_only=False)
        
        # Extract hash from contract path
        file_hash = graph.contract_path.split('/')[-1].replace('.sol', '')
        
        # Check if we have label
        if file_hash in hash_to_binary:
            # Update label
            binary_label = hash_to_binary[file_hash]
            graph.y = torch.tensor([binary_label])
            
            # Save back
            torch.save(graph, graph_file)
            
            stats['fixed'] += 1
            label_counts[binary_label] += 1
        else:
            stats['unmapped'] += 1
            
    except Exception as e:
        stats['errors'] += 1
        print(f"\nError processing {graph_file.name}: {e}")

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Total graphs processed: {stats['total']:,}")
print(f"Labels fixed: {stats['fixed']:,}")
print(f"Unmapped (no CSV label): {stats['unmapped']:,}")
print(f"Errors: {stats['errors']}")
print()

if stats['fixed'] > 0:
    print("Label distribution in fixed graphs:")
    print(f"  Safe (0): {label_counts[0]:,} ({label_counts[0]/stats['fixed']*100:.1f}%)")
    print(f"  Vulnerable (1): {label_counts[1]:,} ({label_counts[1]/stats['fixed']*100:.1f}%)")
    print(f"  Balance ratio: {min(label_counts.values()) / max(label_counts.values()):.3f}")
    print()

print("=" * 70)
print("✅ LABEL FIXING COMPLETE")
print("=" * 70)
print()
print("Next steps:")
print("  1. Run validation again: poetry run python ml/analysis/data_quality_validation.py")
print("  2. Should now see 55% safe, 45% vulnerable")
print("  3. Ready for training!")
