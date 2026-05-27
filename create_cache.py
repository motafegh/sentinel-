import torch
import pickle
from pathlib import Path
from tqdm import tqdm

graphs_dir = Path('ml/data/graphs')
tokens_dir = Path('ml/data/tokens')
# Get stems as strings (MD5 hashes)
stems = [p.stem for p in graphs_dir.glob('*.pt')]
print(f"Found {len(stems)} samples")

cached = {}
for stem in tqdm(stems, desc="Caching to RAM"):
    graph = torch.load(graphs_dir / f'{stem}.pt', weights_only=False)
    tokens = torch.load(tokens_dir / f'{stem}.pt', weights_only=True)
    cached[stem] = (graph, tokens)

with open('ml/data/cached_dataset.pkl', 'wb') as f:
    pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)

cache_size = Path('ml/data/cached_dataset.pkl').stat().st_size / 1e9
print(f"\n✅ Cached {len(cached)} samples to ml/data/cached_dataset.pkl")
print(f"   Cache size: {cache_size:.2f} GB")
