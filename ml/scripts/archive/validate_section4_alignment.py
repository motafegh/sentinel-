"""
SECTION 4: File alignment — CSV vs graphs vs tokens
"""
import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = Path("/home/motafeq/projects/sentinel/ml/data/processed/multilabel_index_deduped.csv")
GRAPHS_DIR = Path("/home/motafeq/projects/sentinel/ml/data/graphs")
TOKENS_DIR = Path("/home/motafeq/projects/sentinel/ml/data/tokens_windowed")

def main():
    df = pd.read_csv(CSV_PATH)
    total_csv = len(df)

    # Get md5 column
    md5_col = None
    for c in ['md5_stem', 'md5', 'hash', 'contract_hash', 'id']:
        if c in df.columns:
            md5_col = c
            break

    print(f"CSV rows: {total_csv:,}")
    print(f"MD5 column: {md5_col}")

    # Build sets
    csv_md5s = set(df[md5_col].astype(str).str.strip()) if md5_col else set()

    graph_files = {f.stem for f in GRAPHS_DIR.glob("*.pt")}
    token_files = {f.stem for f in TOKENS_DIR.glob("*.pt")}

    print(f"Graph files:  {len(graph_files):,}")
    print(f"Token files:  {len(token_files):,}")

    print("\n" + "=" * 70)
    print("SECTION 4: FILE ALIGNMENT")
    print("=" * 70)

    print(f"\n--- Counts ---")
    print(f"  CSV md5s:     {len(csv_md5s):,}")
    print(f"  Graph .pt:    {len(graph_files):,}")
    print(f"  Token .pt:    {len(token_files):,}")

    # CSV vs graphs
    csv_missing_graphs = csv_md5s - graph_files
    csv_missing_tokens = csv_md5s - token_files
    graphs_not_in_csv = graph_files - csv_md5s
    tokens_not_in_csv = token_files - csv_md5s

    print(f"\n--- Alignment checks ---")
    print(f"  CSV md5s missing graph .pt:   {len(csv_missing_graphs):,} ({'FAIL' if csv_missing_graphs else 'PASS'})")
    if csv_missing_graphs:
        sample = list(csv_missing_graphs)[:5]
        print(f"    Examples: {sample}")

    print(f"  CSV md5s missing token .pt:   {len(csv_missing_tokens):,} ({'FAIL' if csv_missing_tokens else 'PASS'})")
    if csv_missing_tokens:
        sample = list(csv_missing_tokens)[:5]
        print(f"    Examples: {sample}")

    print(f"  Graph .pt NOT in CSV:          {len(graphs_not_in_csv):,} ({'WARN' if graphs_not_in_csv else 'PASS'})")
    if graphs_not_in_csv:
        sample = list(graphs_not_in_csv)[:5]
        print(f"    Examples: {sample}")

    print(f"  Token .pt NOT in CSV:          {len(tokens_not_in_csv):,} ({'WARN' if tokens_not_in_csv else 'PASS'})")
    if tokens_not_in_csv:
        sample = list(tokens_not_in_csv)[:5]
        print(f"    Examples: {sample}")

    # Check graph/token alignment
    graph_missing_tokens = graph_files - token_files
    token_missing_graphs = token_files - graph_files
    print(f"\n--- Graph vs Token alignment ---")
    print(f"  Graphs missing token .pt:     {len(graph_missing_tokens):,} ({'FAIL' if graph_missing_tokens else 'PASS'})")
    print(f"  Tokens missing graph .pt:     {len(token_missing_graphs):,} ({'WARN' if token_missing_graphs else 'PASS'})")

    perfect = (len(csv_missing_graphs) == 0 and len(csv_missing_tokens) == 0 and
               len(graphs_not_in_csv) == 0 and len(tokens_not_in_csv) == 0)
    print(f"\n  Overall alignment: {'PASS — perfect 3-way match' if perfect else 'FAIL — mismatches found'}")

if __name__ == "__main__":
    main()
