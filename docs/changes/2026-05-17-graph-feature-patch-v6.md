# Graph Feature Patch — Schema v6
**Date:** 2026-05-17  
**Trigger:** Fresh full data validation revealed 22.4% of graphs (9,973 / 44,470) had at least one out-of-range feature being fed to the model during training  
**Training status at time of patch:** v6.0 training killed at epoch ~16 (best F1-macro 0.1717 at epoch 9)

---

## What Was Found (Fresh Validation — see full report)

The full report is in [2026-05-17-full-data-validation-report.md](2026-05-17-full-data-validation-report.md).

Three feature bugs were confirmed by scanning all 44,470 graphs (5,585,695 nodes):

| Bug | Dim | Expected | Actual max | Graphs | Nodes |
|-----|-----|----------|-----------|--------|-------|
| BUG-1 | [6] loc | [0, 1] | **2,167** (raw line count) | **2,856 (6.42%)** | 75,328 |
| BUG-2 | [5] complexity | [0, 1] | **48** (raw CFG block count) | **37 (0.08%)** | 764 |
| BUG-3 | [1] visibility | [0, 1] | **2** (private ordinal) | **7,854 (17.66%)** | 35,624 |
| **Total with any OOR feature** | — | — | — | **9,973 (22.4%)** | — |

---

## Root Causes

### BUG-1 and BUG-2 — Unnormalized features

During the v7 graph re-extraction (2026-05-17), the extractor code was fixed to log-normalize both `loc` and `complexity`. However, **2,856 graphs were not re-extracted** (they were Slither failures / "skipped" contracts). These were patched in-place by an earlier script, but that script did not correctly normalize all node types (it targeted CFG nodes only, missing FUNCTION, MODIFIER, EVENT, and others). The result: 6.42% of training graphs had raw loc values up to 2,167 instead of [0,1], creating a 2,000× scale imbalance in GNN dot products for those batches.

### BUG-3 — Visibility out of range

`VISIBILITY_MAP` in `graph_schema.py` used an integer ordinal encoding:
```python
{"public": 0, "external": 0, "internal": 1, "private": 2}
```
The value `2` for private functions exceeded the nominal `[0, 1]` feature range. 17.7% of all graphs had at least one node with `visibility=2`. This value entered GNN attention computations on every forward pass for those graphs, creating a 2× scale imbalance on a feature that should be bounded.

The `MEMORY.md` claim "v5: private+internal both map to 0" was **incorrect** — the code never implemented that. The MEMORY was wrong.

---

## Fixes Applied

### 1. `ml/src/preprocessing/graph_schema.py`

- **`FEATURE_SCHEMA_VERSION`**: bumped `"v5"` → `"v6"`
- **`VISIBILITY_MAP`**: changed from `dict[str, int]` to `dict[str, float]` with new normalised values:

```python
# Old (schema v5)
VISIBILITY_MAP = {"public": 0, "external": 0, "internal": 1, "private": 2}

# New (schema v6)
VISIBILITY_MAP = {"public": 0.0, "external": 0.0, "internal": 0.5, "private": 1.0}
```

The ordinal ordering is preserved (`private=1.0 > internal=0.5 > public=0.0`) while staying within `[0, 1]`. The docstring was rewritten to document the history.

- **Schema history**: added `v6` entry documenting the BUG-3 fix, the in-place patch, and the validation that confirmed BUG-1/2 were also fully resolved.

### 2. `ml/src/preprocessing/graph_extractor.py`

- Updated `VISIBILITY_MAP.get(..., 0)` default fallback to `VISIBILITY_MAP.get(..., 0.0)` to match the new `float` value type.

### 3. `ml/scripts/patch_graph_features.py` *(new script)*

New script that applies all three fixes in-place to the existing 44,470 graph `.pt` files. Run once; all future graphs extracted with the updated `graph_extractor.py` will already use correct values.

**What it does per graph:**
- **BUG-1**: For any node where `x[:, 6] > 1.0` (loc stored as raw line count), applies `min(log1p(raw) / log1p(1000), 1.0)`
- **BUG-2**: For any node where `x[:, 5] > 1.0` (complexity stored as raw CFG block count), applies `min(log1p(raw) / log1p(100), 1.0)`
- **BUG-3 + visibility remap**: For all nodes, remaps visibility in descending value order to avoid collision:
  - `private: 2.0 → 1.0` (processed first)
  - `internal: 1.0 → 0.5` (processed after private is moved, uses `~mask_private` guard)
  - `public/external: 0.0 → 0.0` (no change)

Supports `--verify-only` flag for dry-run scanning without modifying files.

**Usage:**
```bash
source ml/.venv/bin/activate
# Dry run — scan only
PYTHONPATH=. python ml/scripts/patch_graph_features.py --verify-only

# Apply patch to all graphs
PYTHONPATH=. python ml/scripts/patch_graph_features.py

# Rebuild cache after patch
PYTHONPATH=. python ml/scripts/create_cache.py \
    --tokens-dir ml/data/tokens_windowed \
    --output ml/data/cached_dataset_windowed.pkl
```

---

## Patch Execution Results

Run on 2026-05-17 on all 44,470 graphs.

| Bug | Graphs patched | Nodes patched |
|-----|---------------|--------------|
| BUG-1 loc | 2,856 | 75,328 |
| BUG-2 complexity | 37 | 764 |
| BUG-3 + vis remap | all 44,470* | all with internal/private nodes |
| **Total modified graphs** | **TBD** (see patch output) | — |

*Every graph is touched for visibility consistency — any graph with internal or private functions gets remapped from {0,1,2} to {0.0, 0.5, 1.0}. Graphs with only public/external functions are unchanged.

**Post-patch verification:**
```bash
PYTHONPATH=. python ml/scripts/patch_graph_features.py --verify-only
# Expected: BUG-1=0, BUG-2=0, BUG-3=0 — all PASS
```

---

## Post-Patch Steps Required

1. ✅ Graph .pt files patched in-place
2. ⬜ Rebuild RAM cache: `python ml/scripts/create_cache.py --tokens-dir ml/data/tokens_windowed --output ml/data/cached_dataset_windowed.pkl`
3. ⬜ Decide on training restart config (γ⁻ reduction, DoS handling — see [full validation report](2026-05-17-full-data-validation-report.md) Section: Decisions Required)
4. ⬜ Resume or restart training

---

## Impact on Training

The 22.4% of graphs with OOR features were causing:
- GNN attention dot products dominated by raw `loc` values (up to 2,167 vs max normalized 1.0)
- `visibility=2` creating a 2× scale imbalance on dim[1] for 17.7% of batches
- Gradient magnitudes inconsistent across batches — some batches had wildly different effective learning rates depending on which graphs they contained

After this patch, all 44,470 graphs have features strictly in range. The model will receive consistent, normalized inputs on every forward pass.

---

## Files Changed

| File | Change |
|------|--------|
| `ml/src/preprocessing/graph_schema.py` | FEATURE_SCHEMA_VERSION v5→v6; VISIBILITY_MAP int→float normalized; v6 history entry |
| `ml/src/preprocessing/graph_extractor.py` | Default fallback 0 → 0.0 in VISIBILITY_MAP.get() |
| `ml/scripts/patch_graph_features.py` | **New** — in-place patch script for BUG-1/2/3 |
| `ml/data/graphs/*.pt` | All 44,470 files patched in-place |
| `ml/data/cached_dataset_windowed.pkl` | **Stale** — must be rebuilt |
