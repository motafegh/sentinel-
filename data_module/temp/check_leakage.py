import json
from pathlib import Path

split_dir = Path("data/splits/v3")
train_shas = {json.loads(l)["sha256"] for l in (split_dir / "train.jsonl").read_text().splitlines() if l.strip()}
val_shas   = {json.loads(l)["sha256"] for l in (split_dir / "val.jsonl").read_text().splitlines() if l.strip()}
test_shas  = {json.loads(l)["sha256"] for l in (split_dir / "test.jsonl").read_text().splitlines() if l.strip()}

tv = train_shas & val_shas
tt = train_shas & test_shas
vt = val_shas & test_shas
print(f"train: {len(train_shas)}, val: {len(val_shas)}, test: {len(test_shas)}")
print(f"train∩val: {len(tv)}, train∩test: {len(tt)}, val∩test: {len(vt)}")
print("Leakage:", "NONE" if not tv and not tt and not vt else "DETECTED")
