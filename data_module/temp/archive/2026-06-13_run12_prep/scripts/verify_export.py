import json
from pathlib import Path

export_dir = Path("data/exports/sentinel-v3-smartbugs-2026-06-13")

cache_path = export_dir / ".hash_cache.json"
print("hash_cache exists:", cache_path.exists())

m = json.loads((export_dir / "manifest.json").read_text())
sample = list(m["shard_index"].items())[:3]
print("shard_index sample entries:")
for sha, entry in sample:
    print(" ", sha[:16], entry)

print("splits: train=%d val=%d test=%d" % (
    len(m["splits"]["train"]), len(m["splits"]["val"]), len(m["splits"]["test"])))
