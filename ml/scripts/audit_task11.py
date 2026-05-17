"""Task 11: File Count Triple-Alignment"""
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path('/home/motafeq/projects/sentinel')
CSV_PATH = PROJECT_ROOT / 'ml/data/processed/multilabel_index_deduped.csv'
GRAPHS_DIR = PROJECT_ROOT / 'ml/data/graphs'
TOKENS_DIR = PROJECT_ROOT / 'ml/data/tokens_windowed'

# 1. CSV rows
df = pd.read_csv(CSV_PATH)
csv_stems = set(df['md5_stem'].tolist())
print(f"1. CSV rows (excluding header): {len(csv_stems)}")

# 2. Graph stems
graph_stems = set(p.stem for p in GRAPHS_DIR.glob('*.pt'))
print(f"2. Graph .pt stems: {len(graph_stems)}")

# 3. Token stems (exclude checkpoint.json, failed_contracts.json)
token_stems = set()
for p in TOKENS_DIR.iterdir():
    if p.suffix == '.pt' and p.name not in {'checkpoint.json', 'failed_contracts.json'}:
        token_stems.add(p.stem)
print(f"3. Token .pt stems: {len(token_stems)}")

# 4. Intersections
csv_and_graphs = csv_stems & graph_stems
csv_and_tokens = csv_stems & token_stems
graphs_and_tokens = graph_stems & token_stems
all_three = csv_stems & graph_stems & token_stems

print(f"\n4. Intersections:")
print(f"   |CSV ∩ graphs|         = {len(csv_and_graphs)}")
print(f"   |CSV ∩ tokens|         = {len(csv_and_tokens)}")
print(f"   |graphs ∩ tokens|      = {len(graphs_and_tokens)}")
print(f"   |CSV ∩ graphs ∩ tokens| = {len(all_three)}")

# 5. In CSV but NOT in graphs
csv_not_in_graphs = csv_stems - graph_stems
print(f"\n5. In CSV but NOT in graphs ({len(csv_not_in_graphs)} stems):")
for s in sorted(csv_not_in_graphs)[:20]:
    print(f"   {s}")
if len(csv_not_in_graphs) > 20:
    print(f"   ... and {len(csv_not_in_graphs) - 20} more")

# 6. In graphs but NOT in CSV
graphs_not_in_csv = graph_stems - csv_stems
print(f"\n6. In graphs but NOT in CSV ({len(graphs_not_in_csv)} stems):")
for s in sorted(graphs_not_in_csv)[:20]:
    print(f"   {s}")
if len(graphs_not_in_csv) > 20:
    print(f"   ... and {len(graphs_not_in_csv) - 20} more")

# 7. In tokens but NOT in graphs
tokens_not_in_graphs = token_stems - graph_stems
print(f"\n7. In tokens but NOT in graphs ({len(tokens_not_in_graphs)} stems):")
for s in sorted(tokens_not_in_graphs)[:20]:
    print(f"   {s}")
if len(tokens_not_in_graphs) > 20:
    print(f"   ... and {len(tokens_not_in_graphs) - 20} more")

# 8. In graphs but NOT in tokens (tokenization gaps)
graphs_not_in_tokens = graph_stems - token_stems
print(f"\n8. In graphs but NOT in tokens ({len(graphs_not_in_tokens)} stems):")
for s in sorted(graphs_not_in_tokens)[:20]:
    print(f"   {s}")
if len(graphs_not_in_tokens) > 20:
    print(f"   ... and {len(graphs_not_in_tokens) - 20} more")

# Summary findings
print("\n=== FINDINGS SUMMARY ===")
if len(csv_not_in_graphs) > 0:
    print(f"FINDING: {len(csv_not_in_graphs)} CSV entries missing graphs")
if len(graphs_not_in_csv) > 0:
    print(f"FINDING: {len(graphs_not_in_csv)} orphan graphs not in CSV")
if len(tokens_not_in_graphs) > 0:
    print(f"FINDING: {len(tokens_not_in_graphs)} tokens without corresponding graphs")
if len(graphs_not_in_tokens) > 0:
    print(f"FINDING: {len(graphs_not_in_tokens)} graphs missing tokens (tokenization gap)")
if len(csv_not_in_graphs) == 0 and len(graphs_not_in_csv) == 0 and len(tokens_not_in_graphs) == 0 and len(graphs_not_in_tokens) == 0:
    print("CONFIRMED: Perfect alignment — no gaps")
