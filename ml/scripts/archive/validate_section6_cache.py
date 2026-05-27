"""
SECTION 6: Cache integrity
"""
import pickle
import torch
import numpy as np
import random
from pathlib import Path

CACHE_PATH = Path("/home/motafeq/projects/sentinel/ml/data/cached_dataset_windowed.pkl")
SAMPLE_N = 10
SEED = 42

def main():
    print(f"Loading cache from {CACHE_PATH}...")
    print(f"File size: {CACHE_PATH.stat().st_size / 1e9:.2f} GB")

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    print(f"\nType: {type(cache)}")
    all_keys = list(cache.keys())

    # Exclude metadata keys
    meta_keys = [k for k in all_keys if k.startswith('__')]
    data_keys = [k for k in all_keys if not k.startswith('__')]

    print(f"Total keys: {len(all_keys)}")
    print(f"Metadata keys: {meta_keys}")
    print(f"Data entries: {len(data_keys)}")

    if meta_keys:
        for mk in meta_keys:
            print(f"  {mk} = {cache[mk]}")

    print("\n" + "=" * 70)
    print("SECTION 6: CACHE INTEGRITY")
    print("=" * 70)

    expected_length = 44470
    length = len(data_keys)
    print(f"\n--- Length check ---")
    print(f"  Data entries:  {length:,} (expected {expected_length:,}) {'PASS' if length == expected_length else 'FAIL'}")

    # Sample 10 random items
    rng = random.Random(SEED)
    sample_keys = rng.sample(data_keys, SAMPLE_N)
    print(f"\n--- Sampling {SAMPLE_N} random items ---")

    errors = 0
    schema_versions = set()
    for k in sample_keys:
        item = cache[k]
        try:
            graph, token_dict = item
            # Graph check
            graph_ok = hasattr(graph, 'x') and graph.x is not None
            g_shape = list(graph.x.shape) if graph_ok else "NO X"

            # Token check - input_ids shape
            input_ids = token_dict.get('input_ids')
            token_ok = isinstance(input_ids, torch.Tensor) and input_ids.shape == torch.Size([4, 512])
            t_shape = list(input_ids.shape) if isinstance(input_ids, torch.Tensor) else "MISSING"

            # Schema version
            sv = token_dict.get('feature_schema_version', 'UNKNOWN')
            schema_versions.add(sv)

            # Edge attr check
            ea = graph.edge_attr if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
            ea_info = f"edge_attr={list(ea.shape)}" if ea is not None else "no edge_attr"

            status = "PASS" if (graph_ok and token_ok) else "FAIL"
            if not (graph_ok and token_ok):
                errors += 1

            num_windows = token_dict.get('num_windows', '?')
            print(f"  [{k[:8]}...] graph.x={g_shape} {ea_info} token={t_shape} windows={num_windows} schema={sv} → {status}")
        except Exception as e:
            print(f"  [{k[:8]}...] ERROR: {e}")
            errors += 1

    print(f"\n  Errors in sample: {errors}/{SAMPLE_N} ({'FAIL' if errors > 0 else 'PASS'})")
    print(f"  Schema versions found: {schema_versions} ({'PASS' if schema_versions == {'v5'} else 'WARN'})")

if __name__ == "__main__":
    main()
