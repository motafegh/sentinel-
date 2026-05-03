# Graph Dataset Re-Extraction (2026-05-03)

## Why this was needed

P0-B (2026-05-02) added edge relation-type embeddings to `GNNEncoder`:

```python
self.edge_emb = nn.Embedding(NUM_EDGE_TYPES, gnn_edge_emb_dim)  # shape in: [E] → out: [E, 16]
```

`nn.Embedding` requires a **1-D integer tensor** of shape `[E]`. All existing
~68 K graph `.pt` files in `ml/data/graphs/` were produced by the pre-refactor
`ast_extractor.py`, which wrote `edge_attr` with shape `[E, 1]`. Passing `[E, 1]`
to `nn.Embedding` raises a dimension error at the first training step, not at
data loading — a silent-until-crash issue. The graceful degradation in
`GNNEncoder` (fall back to zero-vectors) fires only when `edge_attr is None`;
it does **not** catch the wrong shape.

A second reason: the old `ast_extractor.py` contained its own copy of the node
feature logic. It now delegates entirely to `graph_extractor.py` (the shared
preprocessing package introduced 2026-05-01), ensuring train/inference feature
parity is structurally guaranteed.

---

## What was re-extracted

| Data | Action | Reason |
|------|--------|--------|
| `ml/data/graphs/*.pt` | **Re-extracted** | All had `edge_attr=[E,1]`; new extractor writes `[E]` |
| `ml/data/tokens/*.pt` | Kept as-is | CodeBERT tokenisation; no edge features involved |
| `ml/data/splits/*.npy` | Kept as-is | Based on label CSV row order, not graph content |
| `ml/data/processed/multilabel_index.csv` | Kept as-is | Unchanged |
| `ml/checkpoints/multilabel_crossattn_best.pt` | Kept as baseline | Still the F1 baseline to beat |

---

## Local environment

| Item | Path |
|------|------|
| Project root | `~/projects/sentinel` (WSL2) |
| DVC remote | `/mnt/d/sentinel-dvc-remote` (Windows D drive via WSL2) |
| Contracts metadata | `ml/data/processed/_cache/contracts_metadata.parquet` |
| Graph output | `ml/data/graphs/` |
| ML venv | `ml/.venv/bin/python` |

> **Note on contracts_metadata path:** The `ast_extractor.py` CLI default is
> `ml/data/processed/contracts_metadata.parquet` but the actual file lives in
> `ml/data/processed/_cache/contracts_metadata.parquet`. Always pass `--input`
> explicitly.

---

## Steps performed

### 1 — Fix lock file (new deps added to pyproject.toml)

`ml/pyproject.toml` was missing inference dependencies that had been installed
manually but never declared. Added: `fastapi`, `uvicorn[standard]`, `loguru`,
`httpx`, `scipy`. The lock file was then stale and needed regeneration:

```bash
cd ml && poetry lock --no-update && poetry install && cd ..
```

`--no-update` regenerates the lock without upgrading already-pinned packages.

### 2 — Clear checkpoint to force full re-extraction

The extractor writes `ml/data/graphs/checkpoint.json` tracking processed
contract hashes. If this file exists, the extractor skips those hashes even
without `--resume`. To re-extract everything:

```bash
rm -f ml/data/graphs/checkpoint.json
```

> **Important:** `ast_extractor.py` does NOT have a `--force` flag (the
> `graph_schema.py` change-policy comment mentioned it speculatively). The
> correct approach is to delete `checkpoint.json`.

### 3 — Run full extraction

```bash
TRANSFORMERS_OFFLINE=1 \
  ml/.venv/bin/python ml/data_extraction/ast_extractor.py \
    --input ml/data/processed/_cache/contracts_metadata.parquet \
    --output ml/data/graphs \
    --workers 11 \
    --verbose
```

Runtime: ~40 minutes (11 workers, 35 Solidity version groups, RTX 3070).

**Expected "Skipped" noise in verbose output:** Slither fails to parse a small
number of contracts (recursion depth exceeded, IR generation failures on old
SafeMath patterns). These are logged and skipped — the same contracts were
skipped in the original extraction. This is normal and not an error in the
extractor.

Result:
```
✅ Successfully processed 68,523 NEW graphs
📁 Total graphs now: 68,523
```

68,568 contracts in the parquet → 68,523 extracted (45 skipped by Slither).

### 4 — Validate

```bash
python ml/scripts/validate_graph_dataset.py --graphs-dir ml/data/graphs
```

First run result:
```
Total files      : 68555
PASS             : 68523
Shape errors     : 32   ← orphaned old files not in parquet
```

The 32 shape-error files had `edge_attr=[E,1]` and were NOT present in
`contracts_metadata.parquet` — they were orphaned leftovers from a previous
extraction run. They cannot be regenerated. Removed:

```python
import torch
from pathlib import Path

for p in sorted(Path('ml/data/graphs').glob('*.pt')):
    data = torch.load(p, map_location='cpu', weights_only=False)
    if (hasattr(data, 'edge_attr') and data.edge_attr is not None
            and data.edge_attr.dim() != 1):
        p.unlink()
```

Second validation:
```
Total files      : 68523
PASS             : 68523
Shape errors     : 0
Value errors     : 0
Load errors      : 0

PASS: all graph files have valid edge_attr. Safe to retrain.
```

---

## Companion source changes (same session)

| File | Change |
|------|--------|
| `ml/src/preprocessing/graph_schema.py` | Corrected two stale docstrings that said "GNNEncoder ignores edge_attr" — untrue since P0-B |
| `ml/pyproject.toml` | Added `fastapi`, `uvicorn[standard]`, `loguru`, `httpx`, `scipy` as declared dependencies |

---

## Next step

Dataset is validated. Retrain is unblocked:

```bash
TRANSFORMERS_OFFLINE=1 \
  ml/.venv/bin/python ml/scripts/train.py \
    --run-name multilabel-v2-edge-attr \
    --experiment-name sentinel-retrain-v2 \
    --epochs 40 \
    --batch-size 16 \
    --graphs-dir ml/data/graphs \
    --tokens-dir ml/data/tokens \
    --splits-dir ml/data/splits \
    --checkpoint-dir ml/checkpoints \
    --checkpoint-name multilabel_crossattn_v2_best.pt
```

Success gate: val F1-macro > **0.4679** on the fixed `val_indices.npy` split.
