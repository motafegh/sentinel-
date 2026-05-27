"""Task 10: Graph-Token Contract Alignment (fixed for dict token files)"""
import sys
import hashlib
import torch
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path('/home/motafeq/projects/sentinel')
CSV_PATH = PROJECT_ROOT / 'ml/data/processed/multilabel_index_deduped.csv'
GRAPHS_DIR = PROJECT_ROOT / 'ml/data/graphs'
TOKENS_DIR = PROJECT_ROOT / 'ml/data/tokens_windowed'

df = pd.read_csv(CSV_PATH)
md5_stems = df['md5_stem'].tolist()[:30]
print(f"Processing first 30 md5_stems from CSV")

results = []
for md5 in md5_stems:
    row = {'md5': md5, 'graph_ok': False, 'token_ok': False, 'hash_match': '-',
           'g_sol_exists': '-', 't_sol_exists': '-',
           'path_md5_match': '-', 'issues': []}

    graph_path = GRAPHS_DIR / f"{md5}.pt"
    token_path = TOKENS_DIR / f"{md5}.pt"

    # Load graph
    g = None
    if graph_path.exists():
        try:
            g = torch.load(graph_path, weights_only=False)
            row['graph_ok'] = True
        except Exception as e:
            row['issues'].append(f"graph_load_err: {e}")
    else:
        row['issues'].append("graph_missing")

    # Load token
    t = None
    if token_path.exists():
        try:
            t = torch.load(token_path, weights_only=False)
            row['token_ok'] = True
        except Exception as e:
            row['issues'].append(f"token_load_err: {e}")
    else:
        row['issues'].append("token_missing")

    # Check contract_hash (dict access)
    if t is not None:
        t_hash = t.get('contract_hash') if isinstance(t, dict) else getattr(t, 'contract_hash', None)
        row['hash_match'] = 'OK' if t_hash == md5 else f'MISMATCH({t_hash})'
        if t_hash != md5:
            row['issues'].append(f"BUG:hash_mismatch expected={md5} got={t_hash}")

    # Check g.contract_path
    if g is not None:
        g_cp = getattr(g, 'contract_path', None)
        if g_cp:
            g_cp_path = Path(g_cp)
            g_sol_ok = g_cp_path.exists()
            row['g_sol_exists'] = 'YES' if g_sol_ok else 'NO'
            if not g_sol_ok:
                row['issues'].append(f"BUG:g_sol_missing")

            # Compute path MD5
            path_md5 = hashlib.md5(str(g_cp_path).encode()).hexdigest()
            row['path_md5_match'] = 'OK' if path_md5 == md5 else 'DIFF'

    # Check t.contract_path
    if t is not None:
        t_cp = t.get('contract_path') if isinstance(t, dict) else getattr(t, 'contract_path', None)
        if t_cp:
            t_cp_path = Path(t_cp)
            t_sol_ok = t_cp_path.exists()
            row['t_sol_exists'] = 'YES' if t_sol_ok else 'NO'
            if not t_sol_ok:
                row['issues'].append(f"BUG:t_sol_missing")
        else:
            row['t_sol_exists'] = 'NONE'

        # Verify g and t point to same file
        if g is not None:
            g_cp = getattr(g, 'contract_path', None)
            if g_cp and t_cp and str(g_cp) != str(t_cp):
                row['issues'].append(f"NOTE:g_path≠t_path")

    results.append(row)

# Print table
print(f"\n{'MD5':<36} {'G_OK':<6} {'T_OK':<6} {'HASH':<8} {'G_SOL':<8} {'T_SOL':<8} {'P_MD5':<8} {'ISSUES'}")
print('-' * 130)
bugs = 0
for r in results:
    issues_str = '; '.join(r['issues']) if r['issues'] else 'OK'
    if any('BUG' in i for i in r['issues']):
        bugs += 1
    print(f"{r['md5']:<36} {str(r['graph_ok']):<6} {str(r['token_ok']):<6} {str(r['hash_match']):<8} {str(r['g_sol_exists']):<8} {str(r['t_sol_exists']):<8} {str(r['path_md5_match']):<8} {issues_str}")

print(f"\nTotal rows with BUGs: {bugs} / 30")
