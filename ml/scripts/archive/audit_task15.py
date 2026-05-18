"""Task 15: in_unchecked False Positive Check"""
import sys
import re
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path('/home/motafeq/projects/sentinel')
GRAPHS_DIR = PROJECT_ROOT / 'ml/data/graphs'
BCCC_ROOT = PROJECT_ROOT / 'BCCC-SCsVul-2024/SourceCodes'

# Check if any graph has in_unchecked > 0
print("=== Part A: Scan all graphs for in_unchecked > 0 ===")
all_files = list(GRAPHS_DIR.glob('*.pt'))
print(f"Total graph files: {len(all_files)}")

found_unchecked = []
sample_size = min(2000, len(all_files))
import random
random.seed(42)
# Scan a sample first, then all if needed
scan_files = random.sample(all_files, sample_size)

for fpath in scan_files:
    try:
        g = torch.load(fpath, weights_only=False)
        x = g.x
        if x is not None and x.shape[1] >= 10:
            in_unc = x[:, 9]  # feature index 9 = in_unchecked
            if (in_unc > 0).any():
                found_unchecked.append(fpath.name)
    except Exception:
        pass

print(f"Graphs with in_unchecked > 0 in sample of {sample_size}: {len(found_unchecked)}")
if found_unchecked:
    for f in found_unchecked[:5]:
        print(f"  {f}")

# Now look for Solidity 0.8 contracts in BCCC
print(f"\n=== Part B: Find ^0.8 contracts in BCCC IntegerUO ===")
integer_uo_dir = BCCC_ROOT / 'IntegerUO'
if not integer_uo_dir.exists():
    print(f"IntegerUO dir not found: {integer_uo_dir}")
    # Try to find it
    if BCCC_ROOT.exists():
        classes = list(BCCC_ROOT.iterdir())
        print(f"Available classes: {[c.name for c in classes[:20]]}")
    else:
        print(f"BCCC_ROOT not found: {BCCC_ROOT}")
        sys.exit(0)
else:
    sol_files = list(integer_uo_dir.glob('*.sol'))
    print(f"Total IntegerUO .sol files: {len(sol_files)}")

    v08_files = []
    for fpath in sol_files:
        try:
            content = fpath.read_text(encoding='utf-8', errors='replace')
            if re.search(r'pragma\s+solidity\s+[^;]*\^?0\.8', content):
                v08_files.append((fpath, content))
        except Exception:
            pass

    print(f"Files with ^0.8 pragma: {len(v08_files)}")

    if not v08_files:
        print("CONFIRMED: No 0.8+ Solidity files found in IntegerUO. Feature is dead.")
    else:
        print(f"\nProcessing {len(v08_files[:10])} 0.8 files:")
        for sol_path, content in v08_files[:10]:
            # Check for unchecked {} block
            has_unchecked_block = bool(re.search(r'\bunchecked\s*\{', content))
            has_unchecked_comment = bool(re.search(r'//.*unchecked', content))
            has_unchecked_string = bool(re.search(r'["\']unchecked["\']', content))

            print(f"\n  File: {sol_path.name}")
            print(f"    has unchecked{{}} block: {has_unchecked_block}")
            print(f"    has // unchecked comment: {has_unchecked_comment}")
            print(f"    has 'unchecked' string: {has_unchecked_string}")

            # Find corresponding graph
            import hashlib
            path_md5 = hashlib.md5(str(sol_path).encode()).hexdigest()
            graph_path = GRAPHS_DIR / f"{path_md5}.pt"
            if graph_path.exists():
                try:
                    g = torch.load(graph_path, weights_only=False)
                    x = g.x
                    if x is not None and x.shape[1] >= 10:
                        in_unc_vals = x[:, 9].tolist()
                        any_nonzero = any(v > 0 for v in in_unc_vals)
                        print(f"    graph in_unchecked any>0: {any_nonzero}")
                        if any_nonzero:
                            print(f"    -> Expected since has unchecked block: {has_unchecked_block}")
                            if not has_unchecked_block:
                                print(f"    -> FALSE POSITIVE!")
                except Exception as e:
                    print(f"    graph load error: {e}")
            else:
                print(f"    graph not found (path_md5={path_md5})")

# Final verdict
print(f"\n=== VERDICT ===")
if not found_unchecked:
    print("CONFIRMED: in_unchecked feature is effectively dead in the current dataset.")
    print("  No nodes with in_unchecked > 0 found in a sample of 2000 graphs.")
    print("  This is expected — BCCC is ~94% Solidity 0.4.x which has no 'unchecked {}' blocks.")
