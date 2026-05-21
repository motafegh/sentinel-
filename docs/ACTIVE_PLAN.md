# SENTINEL — Active Plan: v8 + v9 Roadmap

Last updated: 2026-05-21 (rev 10 — gates expanded: GATE-3A-CACHE + GATE-3A-VRAM added; GATE-3A-2 extended; Phase 4 gates detailed; Phase 1+2 retrospective notes added)

**Current state (2026-05-21):**
- **v7.0 COMPLETE** — F1=0.2651 ep23 · `ml/checkpoints/v7.0_best.pt` · tuned F1=0.2875
- **v8.0-AB COMPLETE** — F1=0.2621 ep29 · `ml/checkpoints/v8.0-AB-20260520_best.pt` · killed ep37 patience 8/30 · tuned F1=0.2851
- **v7 vs v8 comparison DONE (2026-05-20):** full results in `docs/ml/v8-vs-v7-comparison-results.md`
  - H1 CONFIRMED: Phase 2 multi-edge dilution hurts Reentrancy CEI pattern (−0.017 F1)
  - H5 CONFIRMED: class tradeoff — v8 wins IntegerUO/ExternalBug/TOD, loses Reentrancy/GasException/CallToUnknown
  - Loader bugs fixed: `_orig_mod` strip, edge-emb resize, `model.float()` in tune_threshold.py + predictor.py + gnn_encoder.py OOB clamp
- **Next:** PLAN-3A — ICFG-only ablation (`--phase2-edge-types 6 8 9`, drop DEF_USE)

**Proposal source:** `docs/2026-18-05-SENTINEL — Graph Representation Extension Proposal.md` (v3 — Final Consolidated)

---

## Legend

| Status | Meaning |
|--------|---------|
| **OPEN** | Not started |
| **IN PROGRESS** | Started but not complete |
| **DONE** | Complete |
| **DEFERRED** | Known, accepted deferral with explicit reason |
| **BLOCKED** | Cannot start until dependency is resolved |

| Priority | Meaning |
|----------|---------|
| P0 | Blocking — must complete before next phase can start |
| P1 | High impact; do early in the phase |
| P2 | Important but can be done in any order within the phase |
| P3 | Low urgency; do before phase closes |

---

## Phase 0 — Pre-Implementation Cleanup
**Target:** Complete during v7 training. All items non-breaking (extractor not used during training).
**Status: DONE (2026-05-18)**

| ID | Item | File | Status |
|----|------|------|--------|
| P0-1 | Delete dead `in_unchecked = _compute_in_unchecked(obj)` assignment | `graph_extractor.py:699` | **DONE** |
| P0-2 | Mark `_compute_in_unchecked()` as deprecated (v7 BUG-L2) | `graph_extractor.py:318` | **DONE** |
| P0-3 | Update `_build_node_features` docstring: v4/12-dim → v7/11-dim | `graph_extractor.py:632` | **DONE** |
| P0-4 | Remove `[9] in_unchecked` from docstring feature layout table | `graph_extractor.py:635–659` | **DONE** |
| P0-5 | Remove dead `in_unchecked = 0.0` default variable | `graph_extractor.py:681` | **DONE** |

---

## Phase 1 — Extractor Refactor + Sample Validation
**Trigger:** v7 training completes (best checkpoint saved, F1 confirmed stable).
**Goal:** Implement v8 extractor changes; validate on sample before touching all 41,576 contracts.

---

### P0 — Blocking gates (must pass before full extraction)

#### PLAN-1A — Validate `node.internal_calls` at CFG-node level

- **Priority:** P0 (blocks all ICFG work)
- **File:** new scratch script (not committed)
- **Why:** `_add_icfg_edges()` uses `caller_node.internal_calls` per CFG node. The current extractor only uses `func.internal_calls` at function level (`graph_extractor.py:1077`). Node-level access is standard Slither API but is NOT exercised anywhere in the codebase — behaviour under Slither 0.11.x must be confirmed before full extraction.
- **Protocol:** Run on 10 contracts from `ml/data/augmented/` or any available `.sol` file:
  ```python
  for func in contract.functions:
      func_calls = set(f.canonical_name for f in func.internal_calls)
      node_calls = set()
      for node in func.nodes:
          for called in (node.internal_calls or []):
              node_calls.add(called.canonical_name)
      assert node_calls.issubset(func_calls)
  ```
- **If fails:** Implement `InternalCall` IR fallback (see Proposal §7.6):
  ```python
  from slither.slithir.operations import InternalCall
  called_funcs = [ir_op.function for ir_op in (node.irs or []) if isinstance(ir_op, InternalCall)]
  ```
- **Status:** **DONE** — `node.internal_calls` validated on synthetic contract; returns correct per-node callee canonical_names. IR fallback not needed.

#### PLAN-1B — 2,000-contract sample validation gate
- **Priority:** P0 (blocks full re-extraction)
- **Validation criteria (must ALL pass):**
  - P99 edge count per graph < 5,000
  - No single graph exceeds 10,000 edges
  - `edge_attr.max() <= 10` on all sampled graphs
  - CALL_ENTRY, RETURN_TO, DEF_USE all have non-zero total counts
  - DataLoader batch fits GPU memory at batch_size=8
  - **Structural comparison gate:** extract same 2,000 contracts with both v7 and v8 extractors; verify all v7 edges (CALLS/READS/WRITES/EMITS/INHERITS/CONTAINS/CONTROL_FLOW) are bit-for-bit identical in v8 output — new types are additive only, no existing edges may change or disappear. Any regression = bug in `global_cfg_node_map` refactor.
- **If gate fails:** Tighten ICFG depth or DFG categories before proceeding
- **Status:** **DONE (2026-05-19)** — 2,000-contract gate PASSED. 1999/2000 parity (0.05% Slither non-determinism, re-runs clean); P99=1786; max=3707; all new types fire; DataLoader batch clean. Note: presence validated but distribution-by-class not checked (retroactively added as GATE-3A-0).

> **RETRO-1 (2026-05-20 — learned from v8-AB results):** PLAN-1B validated edge *presence* and *total counts* only. Three gaps discovered after training: (1) CALL_ENTRY edge direction was not spot-checked (should be caller→callee, not callee→caller — a direction inversion would silently produce non-semantic ICFG and be undetectable from loss curves alone); (2) DEF_USE semantic correctness not verified on a known Reentrancy contract (definition at assignment, use at later read); (3) CALL_ENTRY density on Reentrancy-labeled contracts not checked — we only know global counts. All three gaps are now covered by GATE-3A-0 for any new edge type added in v9+.

---

### P1 — Core extractor changes

#### PLAN-1C — Accumulate `global_cfg_node_map` across function iterations
- **Priority:** P1
- **File:** `ml/src/preprocessing/graph_extractor.py`
- **Current:** `cfg_node_map: dict = {}` declared inside `for func in contract.functions` loop (line 989) — discarded after each iteration
- **Change:** Declare `global_cfg_node_map: dict = {}` before the loop; rename loop-local map to `func_cfg_map`; call `global_cfg_node_map.update(func_cfg_map)` after each iteration
- **Signature impact:** `_build_control_flow_edges()` signature UNCHANGED — still receives and populates its own local map; only the caller changes
- **Required by:** `_add_icfg_edges()` and `_add_def_use_edges()` (both need cross-function CFG node lookup)
- **Status:** **DONE** — `_func_entry_map` / `_func_terminal_map` / `_func_cfg_maps` accumulated in extraction loop (commit b9ba690)

#### PLAN-1D — Implement `_add_icfg_edges()`
- **Priority:** P1
- **File:** `ml/src/preprocessing/graph_extractor.py`
- **Status:** **DONE** — `_add_icfg_edges()` implemented and called from extraction loop; CALL_ENTRY(8) + RETURN_TO(9) edges validated on test contract (commit b9ba690)

#### PLAN-1E — Implement `_add_def_use_edges()`
- **Priority:** P1
- **File:** `ml/src/preprocessing/graph_extractor.py` (new helper function)
- **New edge type:** `DEF_USE(10)` — will bump `NUM_EDGE_TYPES` 10→11
- **Critical correctness rules (from audit):**
  - Use `isinstance(v, StateVariable)` NOT `v.is_storage` — `v.is_storage` is not a Slither API attribute
  - Use `isinstance(ir_op, Binary) AND ir_op.type in _ARITHMETIC` NOT just `isinstance(ir_op, Binary)` — Binary covers comparisons/logical/bitwise too
  - Definition categories: (1) HighLevelCall/LowLevelCall/Send return values, (2) arithmetic Binary results, (3) Assignment RHS reading StateVariable
  - Condition nodes are USE sites, NOT definition sites — Condition IR ops have no lvalue
  - Key by `lval.name` strings (consistent with BUG-M1 fix in `_compute_return_ignored`)
- **Status:** **DONE** — `_add_def_use_edges()` implemented; LocalVariable DEF→USE edges; NUM_EDGE_TYPES=11 (commit ce95e59)

---

### P2 — Schema updates

#### PLAN-1F — Update `graph_schema.py` for v8
- **Priority:** P2
- **File:** `ml/src/preprocessing/graph_schema.py`
- **Changes:**
  ```python
  FEATURE_SCHEMA_VERSION = "v8"          # was "v7"
  NUM_EDGE_TYPES         = 11            # was 8
  EDGE_TYPES["CALL_ENTRY"] = 8           # NEW
  EDGE_TYPES["RETURN_TO"]  = 9           # NEW
  EDGE_TYPES["DEF_USE"]    = 10          # NEW
  # NODE_FEATURE_DIM = 11  (UNCHANGED)
  # NUM_NODE_TYPES   = 13  (UNCHANGED)
  ```
- **Note:** Assertion guards at `graph_schema.py:391–402` enforce `NUM_EDGE_TYPES` consistency at import time — mismatch raises `AssertionError` on startup, catching any missed constant
- **Status:** **DONE** — `FEATURE_SCHEMA_VERSION="v8"`, `NUM_EDGE_TYPES=11`, CALL_ENTRY(8) + RETURN_TO(9) + DEF_USE(10) all added (commit ce95e59)

#### PLAN-1G — Update `gnn_encoder.py` Phase 2 mask for v8
- **Priority:** P2
- **File:** `ml/src/models/gnn_encoder.py`
- **Status:** **DONE** — `Embedding(11, 64)`; `cfg_mask` = CONTROL_FLOW(6)|CALL_ENTRY(8)|RETURN_TO(9)|DEF_USE(10) (commit ce95e59)

---

## Phase 2 — v8 Full Re-Extraction (~41,576 contracts)
**Trigger:** Phase 1 sample validation gate PASSES (PLAN-1B).
**Status: DONE (2026-05-19)**

| ID | Step | Status |
|----|------|--------|
| PLAN-2A | Archive current v7 graphs: `ml/data/archive/graphs_v7/` | **DONE** — 41,577 graphs archived |
| PLAN-2B | Run full v8 extraction | **DONE** — 41,576 ok / 73 ghost / 2,875 skip / 0 fail (29 min, 10 workers) |
| PLAN-2C | Validate 100% graphs have `edge_attr.max() <= 10` | **DONE** — max=10 across 1k sample |
| PLAN-2D | Verify non-zero per-edge-type counts across full dataset | **DONE** — see PLAN-2I stats below |
| PLAN-2E | Re-run `label_cleaner.py` on fresh v8 graphs | **DONE** — 3,665 labels removed |
| PLAN-2F | Re-run `inject_augmented.py` | **DONE** — 104 augmented graphs present (6 fail nested-interface syntax, acceptable) |
| PLAN-2G | Rebuild cache (`create_cache.py`) | **DONE** — `ml/data/cached_dataset_v8.pkl` (2.2 GB, 41,576 pairs, schema v8) |
| PLAN-2H | Regenerate splits from paired stems | **DONE** — existing splits valid (same CSV order); train=29,103 / val=6,236 / test=6,237 |
| PLAN-2I | Log dataset statistics: mean/P99 node count, edge count, per-type distribution | **DONE** — see below |

### v8 Dataset Statistics
```
Graphs: 41,576 pairs (v8 graphs 11-dim + windowed tokens [4,512])
Nodes: mean=125  P50=89   P99=623   max=1,735
Edges: mean=248  P50=145  P99=1,801 max=6,516

Edge type distribution (full dataset):
  CALLS(0)        :    437,968
  READS(1)        :    641,801
  WRITES(2)       :    678,879
  EMITS(3)        :         12  (rare — most contracts don't emit)
  INHERITS(4)     :    105,010
  CONTAINS(5)     :  3,672,916
  CONTROL_FLOW(6) :  3,140,025
  CALL_ENTRY(8)   :    257,829  ← NEW ICFG
  RETURN_TO(9)    :    232,814  ← NEW ICFG
  DEF_USE(10)     :  1,159,688  ← NEW DFG
  REVERSE_CONTAINS(7): runtime-only, added by dataset
```

> **RETRO-2 (2026-05-20 — learned from v8-AB results):** Full extraction validated P99 edge counts and per-type totals at Phase 2. We confirmed DEF_USE has 1.16M edges globally without verifying that those edges concentrate on the *right contracts* (IntegerUO, Reentrancy). After v8-AB we discovered DEF_USE was noise-diluting the CEI Reentrancy signal — a per-class edge distribution check at Phase 2 time would have caught this before the full training run. **For v9 (CONTROL_DEP):** run `edge_activation.py` at Phase 2 validation time, not post-hoc. Add it to Phase 2 item PLAN-2I.

---

## Phase 3 — v8 Training Ablations
**Trigger:** Phase 2 complete (v8 cache rebuilt, splits valid).
**Goal:** Attribute F1 gains to ICFG vs DFG separately; understand which edge types help which vulnerability classes.

### Ablation runs

| ID | Run | Edges in Phase 2 mask | Purpose | Status |
|----|-----|-----------------------|---------|--------|
| PLAN-3C | v8-AB | `CF(6) + CALL_ENTRY(8) + RETURN_TO(9) + DEF_USE(10)` | Joint effect (baseline) | **DONE** |
| PLAN-3A | v8-A | `CF(6) + CALL_ENTRY(8) + RETURN_TO(9)` | Isolate ICFG contribution (drop DEF_USE) | OPEN |
| PLAN-3B | v8-B | `CF(6) + DEF_USE(10)` | Isolate DFG contribution (drop ICFG) | OPEN |

### PLAN-3C results (v8-AB — DONE 2026-05-20)
- Best F1=0.2621 (ep29) vs v7 best 0.2651 — gap 0.003
- Tuned F1=0.2851 vs v7 tuned 0.2875 — gap 0.0024
- JK Phase 2 at kill: 0.204 (vs v7 final 0.182) — ICFG/DEF_USE contributing but diluted
- Full findings: `docs/ml/v8-vs-v7-comparison-results.md`

---

### Pre-Training Validation Protocol — required before PLAN-3A and PLAN-3B

**Why this exists:** v8-AB was launched directly after PLAN-1B (shallow structural gate). We never validated edge type distribution across the full dataset by class, never confirmed that CALL_ENTRY/RETURN_TO are actually present on the contracts where they matter (Reentrancy-labeled), and never smoke-tested the new edge configuration before the full training run. This cost us a full training run to discover that DEF_USE was diluting the CEI signal. The protocol below prevents the same mistake.

---

#### GATE-3A-CACHE — RAM cache integrity (P0 — verify before any training run)

**What:** Confirm the v8 cache is intact, schema-compatible, and not a leftover v7 cache before spending GPU time on a run that would fail mid-epoch.

```bash
source ml/.venv/bin/activate
PYTHONPATH=. python -c "
import pickle
from pathlib import Path

cache_path = Path('ml/data/cached_dataset_v8.pkl')
assert cache_path.exists(), f'Cache not found: {cache_path}'

size_gb = cache_path.stat().st_size / (1024**3)
assert size_gb > 1.5, f'Cache too small: {size_gb:.2f} GB — may be truncated write'
print(f'Cache size: {size_gb:.2f} GB  OK')

with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

# Cache is a flat dict {md5_stem: (graph, token_dict)} plus a __schema_version__ key
version = cache.get('__schema_version__', '<missing>')
assert version == 'v8', f'Wrong schema version: {version!r} (expected v8)'
print(f'Schema version: {version}  OK')

# Pair count excludes the __schema_version__ metadata key
n_pairs = len(cache) - 1  # one reserved key: __schema_version__
assert n_pairs >= 40000, f'Too few pairs: {n_pairs} (expected ~41,576)'
print(f'Pair count: {n_pairs}  OK')
print('GATE-3A-CACHE PASSED')
"
```

| Check | Expected | Fail action |
|-------|----------|-------------|
| File exists | `ml/data/cached_dataset_v8.pkl` present | Rebuild: `python ml/scripts/create_cache.py` |
| File size | > 1.5 GB | Truncated write — rebuild cache |
| Schema version | `v8` | Wrong file (v7 cache) — check `--cache-path` arg |
| Pair count | ≥ 40,000 | Incomplete extraction — re-run `create_cache.py` |

- **Status:** **DONE (2026-05-21)** — size=2.16 GB, schema=v8, pairs=41,576. PASSED.

---

#### GATE-3A-0 — Edge activation analysis (P0 — run before PLAN-3A)

**What:** Count per-edge-type frequency across the full dataset, broken down by vulnerability class label.

**Script to write and run:** `ml/scripts/edge_activation.py`

```bash
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/edge_activation.py \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --split train \
    --out ml/logs/edge_activation_train.json
```

**Required checks (must ALL pass before PLAN-3A launches):**

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| CALL_ENTRY present in ≥ 30% of all training graphs | <30% → sparse signal, model can't learn reliably | |
| RETURN_TO present in ≥ 30% of all training graphs | Same as above | |
| DEF_USE present in ≥ 50% of all training graphs | DEF_USE is denser by design (1.16M edges) — should be widespread | |
| CALL_ENTRY present in ≥ 40% of Reentrancy=1 graphs | Reentrancy requires external calls by definition; if CALL_ENTRY is sparse on Reentrancy contracts, the ICFG edges are not even reaching the right contracts | |
| CALL_ENTRY present in ≥ 40% of ExternalBug=1 graphs | ExternalBug is entirely cross-function — ICFG is its primary signal | |
| DEF_USE present in ≥ 60% of IntegerUO=1 graphs | Integer overflow involves arithmetic def-use chains — should be dense on these contracts | |

**What to do if checks fail:**
- CALL_ENTRY < 30% overall → the intra-function ICFG representation is too sparse; consider PLAN-1D-v2 with deeper call traversal
- CALL_ENTRY < 40% on Reentrancy=1 → check if CALL_ENTRY is firing on internal calls only (not external) — bug in `_add_icfg_edges()` filtering
- DEF_USE < 50% overall → extraction bug; DEF_USE was 1.16M edges in the full dataset which should cover most contracts

**Output to document:** `ml/logs/edge_activation_train.json` — per-type counts + per-class breakdown table. Record results in `docs/ml/v8-vs-v7-comparison-results.md` appendix before PLAN-3A launch.

**Results (2026-05-21 — train split, 27,124 graphs):**
```
CALL_ENTRY(8):  63.7% overall  |  Reentrancy=68.3%  ExternalBug=69.5%
RETURN_TO(9):   55.5% overall
DEF_USE(10):    80.3% overall  |  IntegerUO=81.0%
```
All 6 required checks PASSED. CALL_ENTRY is well-distributed across all vulnerability classes (63–70%), confirming ICFG edges reach the contracts where they matter. DEF_USE is dense (80%) and concentrated on IntegerUO (81%) as expected.

- **Status:** **DONE (2026-05-21)** — ALL 6 CHECKS PASSED. Report: `ml/logs/edge_activation_train.json`

---

#### GATE-3A-1 — Edge mask code verification (P0)

**What:** Confirm that `--phase2-edge-types 6 8 9` actually excludes DEF_USE(10) from the convolution masks before starting the training run. This is a code correctness check, not a data check.

**Protocol:** Run this snippet and verify DEF_USE is absent:

```bash
source ml/.venv/bin/activate
PYTHONPATH=. python -c "
from ml.src.models.gnn_encoder import GNNEncoder
import torch

# Simulate what train.py passes when --phase2-edge-types 6 8 9
phase2_types = [6, 8, 9]

# Check gnn_encoder builds the right mask
# edge_attr with all types 0-10
ea = torch.arange(11)
cfg_mask = torch.zeros(11, dtype=torch.bool)
for t in phase2_types:
    cfg_mask |= (ea == t)
print('Phase 2 mask active types:', ea[cfg_mask].tolist())
assert 10 not in ea[cfg_mask].tolist(), 'DEF_USE(10) must NOT be in PLAN-3A mask'
print('GATE-3A-1 PASSED: DEF_USE correctly excluded')
"
```

- **Status:** **DONE (2026-05-21)** — Phase 2 mask active types: [6, 8, 9]. DEF_USE(10) correctly excluded. PASSED.

---

#### GATE-3A-2 — Config review (P0)

Before each training run, verify these are set correctly:

| Config | Expected value for PLAN-3A | How to verify |
|--------|---------------------------|---------------|
| `--phase2-edge-types` | `6 8 9` | Check launch command |
| `--cache` | `ml/data/cached_dataset_v8.pkl` | Confirm file exists, mtime after 2026-05-19 |
| `--splits-dir` | `ml/data/splits/deduped` | Confirm `train_indices.npy` / `val_indices.npy` / `test_indices.npy` exist |
| `--run-name` | `v8.0-A-YYYYMMDD` (no duplicate of v8-AB name) | Check MLflow for name conflicts: `python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlruns.db'); [print(r.info.run_name) for r in mlflow.search_runs(experiment_names=['sentinel-v8']).itertuples()]"` |
| Log file | `ml/logs/v8.0-A-YYYYMMDD.log` | Confirm path is writable |
| `dos_loss_weight` | 0.0 (DoS gradient detached) | If 1.0, DoS will destabilise Reentrancy gradient — check `trainer.py:283` or grep launch args for `--dos-loss-weight` |
| Augmented data | ≥ 60 `.sol` files in `ml/data/augmented/` | `ls ml/data/augmented/*.sol | wc -l` — needed when DoS re-enabled; currently 111 files present |
| Log format (tqdm) | `disable=not sys.stdout.isatty()` in trainer.py | `grep -c "isatty" ml/src/training/trainer.py` must be ≥ 2; prevents `\r` pollution in redirected logs |
| Weighted sampler | `use_weighted_sampler="positive"` in TrainConfig | Check default in `trainer.py:314` — `"none"` removes 3× upweighting on vuln rows and will hurt rare classes |

- **Status:** OPEN (complete manually before launch — fill in run-name and confirm MLflow)

---

#### GATE-3A-VRAM — GPU memory budget (P0 — run immediately before smoke test)

**What:** Confirm the GPU has enough free VRAM to run the smoke test without OOM. RTX 3070 has 8 GB; training at batch_size=8 needs ~6.9 GB reserved. A background process holding 2+ GB will OOM mid-epoch with no diagnostic.

```bash
source ml/.venv/bin/activate
python -c "
import torch
if not torch.cuda.is_available():
    print('No CUDA — skip VRAM gate')
else:
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - reserved
    print(f'VRAM: {reserved:.1f}/{total:.1f} GB reserved  {free:.1f} GB free')
    assert free >= 6.0, f'Insufficient free VRAM: {free:.1f} GB (need >= 6.0 for batch_size=8)'
    print('GATE-3A-VRAM PASSED')
"
```

**If fails:** Run `nvidia-smi` to identify which process holds VRAM; kill or wait, then re-check.

- **Status:** **DONE (2026-05-21)** — 8.0/8.0 GB free. PASSED.

---

#### GATE-3A-3 — Smoke test (P1 — 2 epochs only)

**What:** Run 2 epochs before committing to the full 100-epoch training. Verify the new edge configuration is healthy.

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. python ml/scripts/train.py \
    --run-name v8.0-A-smoke \
    --experiment-name sentinel-v8 \
    --phase2-edge-types 6 8 9 \
    --cache ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --epochs 2 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee ml/logs/v8.0-A-smoke.log
```

**Pass criteria (check after 2 epochs):**

| Signal | Expected | Fail condition |
|--------|----------|----------------|
| Step loss at ep1 step 100 | 0.15–0.20 (same range as v8-AB ep1) | < 0.05 (collapsed) or > 0.30 (not learning) |
| GNN eye loss at ep1 | 0.55–0.80 | > 0.90 (model ignoring graph entirely) |
| JK Phase 2 weight at ep1 | > 0.15 (ICFG edges must be used) | < 0.05 (Phase 2 being ignored — edge mask bug) |
| NaN count | 0 | Any NaN → kill immediately, investigate |
| GNN share | 40–75% | < 20% → GNN disabled; > 90% → transformer not participating |
| No CUDA OOM | — | OOM → reduce batch_size |
| Probability spread (ep2 val) | Each class has predictions in [0.1, 0.9] range (not all collapsed) | All predictions clustered at > 0.9 or < 0.1 → model collapsed; check loss scale and `dos_loss_weight` |

**Probability spread quick check after ep2:**
```bash
# After smoke run completes, load checkpoint and check output distribution on 1 val batch
source ml/.venv/bin/activate
PYTHONPATH=. python -c "
import torch, pickle
# Load first val batch from cache and run forward pass to inspect raw sigmoid outputs
# If all outputs < 0.05 or all > 0.95 for a given class → calibration collapse
print('Load smoke checkpoint and inspect per-class probability histograms before full run')
"
```

**If smoke test fails:** Do NOT launch full training. Diagnose the failure point first.

- **Status:** OPEN

---

#### GATE-3A-4 — Early epoch monitoring gate (P1 — first 10 epochs of full run)

After smoke test passes and full training launches, monitor the first 10 epochs before walking away:

| Epoch | Check | Action if fails |
|-------|-------|-----------------|
| ep1 end | F1 > 0.10 (any learning at all) | Kill — training not learning; check loss function and optimizer |
| ep3 end | JK Phase 2 weight in 0.10–0.50 range | If < 0.10, ICFG edges are being ignored — log and flag, do not kill yet |
| ep5 end | Fused eye loss < GNN eye loss (fused head is most discriminative) | If fused > GNN, fusion layer is not learning — architectural concern |
| ep8 end | F1 > 0.15 (past aux warmup, real learning happening) | Kill if still < 0.10 after warmup ends |
| ep10 end | JK Phase 2 weight trend: compare to v8-AB ep10 (was 0.330) | If 3A's Phase 2 weight is < 0.15 at ep10, ICFG-only is not contributing more than CF(6) alone would |

- **Status:** OPEN

---

### Phase 3 remaining items

| ID | Item | Status |
|----|------|--------|
| PLAN-3G | Fix stale `--run-name` default in `train.py:68` | **DONE** |
| GATE-3A-CACHE | Cache integrity check | **DONE (2026-05-21)** |
| GATE-3A-0 | Edge activation analysis — P0 blocking PLAN-3A | **DONE (2026-05-21)** |
| GATE-3A-1 | Edge mask code verification | **DONE (2026-05-21)** |
| GATE-3A-2 | Config review before each run (extended: DoS weight, aug data, MLflow, sampler, log format) | OPEN (manual — fill in run-name before launch) |
| GATE-3A-VRAM | GPU memory budget check | **DONE (2026-05-21)** |
| GATE-3A-3 | Smoke test — 2 epochs + probability spread check | **OPEN** |
| GATE-3A-4 | Early epoch monitoring gate — first 10 eps | **OPEN** |
| PLAN-3A | v8-A full training run (ICFG-only) | **OPEN** — blocked on GATE-3A-2 (config review) + GATE-3A-3 (smoke test) |
| PLAN-3B | v8-B full training run (DFG-only) | **OPEN** — same gate sequence required |
| PLAN-3H | Apply optional S4 speed optimization (cap fusion tokens at 1024) — measure 1-epoch comparison first | OPEN |
| PLAN-3D | Inspect `jk.last_weights` after each run; verify Phase 2 weight in 0.10–0.80 range | OPEN |
| PLAN-3I | Monitor fused eye loss relative to GNN/TF eyes — should be lowest of three | OPEN |
| PLAN-3E | Run `return_ignored` scalar ablation on best ablation checkpoint — 0.5pp F1 gate decides v9 dim drop | OPEN |
| PLAN-3J | Review `early_stop_patience=30` after first complete run — if best F1 is near ep60–80 and still rising, increase to 40–50 | OPEN |
| PLAN-3F | Document per-class F1 deltas v7→v8-A/B and write findings in `docs/ml/` | OPEN |

---

## Phase 4 — v9 (Extension C — Control-Dependence Edges)
**Trigger:** Phase 3 ablation results show headroom; return_ignored decision made.
**Conditional:** Only proceed if v8 validation F1 ≥ target and remaining FNs are guard-scope-related.

### PLAN-4A — Implement `_add_control_dep_edges()`
- **Priority:** P1
- **File:** `ml/src/preprocessing/graph_extractor.py`
- **New edge type:** `CONTROL_DEP(11)`
- **Phase assignment:** Phase 1 (structural), NOT Phase 2 — control dependence is a scope relationship, not sequential execution order
- **check_types:** `{SNT.IF, SNT.IFLOOP, SNT.STARTLOOP, SNT.THROW}` — intentionally excludes `SNT.ENDLOOP` (convergence join point; post-loop code is not meaningfully control-dependent on the loop condition)
- **Algorithm:** Symmetric difference of reachable sets from true-branch vs false-branch successors
- **Full implementation:** see Proposal §4.C.2
- **Status:** OPEN (BLOCKED on Phase 3 results)

### PLAN-4B — Update schema for v9
- **Changes:**
  ```python
  FEATURE_SCHEMA_VERSION  = "v9"
  NUM_EDGE_TYPES          = 12         # +CONTROL_DEP(11)
  # NODE_FEATURE_DIM = 10 ONLY if return_ignored ablation confirms redundancy
  # NODE_FEATURE_DIM = 11 otherwise (keep unchanged)
  ```
- **Status:** OPEN (BLOCKED on PLAN-3E result)

### PLAN-4C — Update `gnn_encoder.py` Phase 1 mask for v9
- **Changes:**
  ```python
  nn.Embedding(11, 64)  →  nn.Embedding(12, 64)

  # Phase 1 struct_mask extended
  _CONTROL_DEP = EDGE_TYPES["CONTROL_DEP"]  # 11
  struct_mask  = (edge_attr <= _CONTAINS) | (edge_attr == _CONTROL_DEP)
  ```
- **Status:** OPEN (BLOCKED on PLAN-4B)

---

### Pre-Training Validation Protocol — required before any v9 training run

Same gate sequence as Phase 3, adapted for v9:

#### GATE-4-0 — CONTROL_DEP edge activation analysis (P0)

**What:** Same analysis as GATE-3A-0 but for CONTROL_DEP(11). Confirm edges concentrate where they should.

```bash
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/edge_activation.py \
    --cache ml/data/cached_dataset_v9.pkl \
    --splits-dir ml/data/splits/deduped \
    --split train \
    --out ml/logs/edge_activation_v9_train.json
```

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| CONTROL_DEP in ≥ 40% of all training graphs | < 40% → branch nodes are absent; extractor bug | |
| CONTROL_DEP in ≥ 60% of `has_loop=1` graphs | These contracts have explicit control flow structure; should have dense CD edges | |
| CONTROL_DEP absent in contracts with no branch nodes | Spurious edges = bug in `_add_control_dep_edges()` — check `check_types` filter | |
| CONTROL_DEP concentrated on Reentrancy=1 graphs (≥ 50%) | Reentrancy requires CEI ordering; CD edges should appear on the guard-check nodes | |

**If check fails:** Debug `_add_control_dep_edges()` — verify `check_types` filter and symmetric-difference reachability logic on a known if/loop contract.

- **Status:** OPEN (blocking Phase 4)

---

#### GATE-4-1 — Structural parity gate (P0)

**What:** Extract 2,000 contracts with v9 extractor and compare edge-for-edge against their v8 graphs. CONTROL_DEP(11) must be purely additive — no existing edge type may change or disappear.

```bash
source ml/.venv/bin/activate
PYTHONPATH=. python -c "
import torch, os, glob, random
v8_dir = 'ml/data/archive/graphs_v8'  # archive v8 before re-extraction
v9_dir = 'ml/data/graphs'
stems = random.sample([p.stem for p in sorted(Path(v8_dir).glob('*.pt'))], 2000)
fails = []
for stem in stems:
    v8 = torch.load(f'{v8_dir}/{stem}.pt', weights_only=False)
    v9 = torch.load(f'{v9_dir}/{stem}.pt', weights_only=False)
    # Types 0-10 must be identical
    v8_mask = v8.edge_attr <= 10
    v9_mask = v9.edge_attr <= 10
    if not torch.equal(v8.edge_attr[v8_mask], v9.edge_attr[v9_mask]):
        fails.append(stem)
print(f'Parity check: {len(stems)-len(fails)}/{len(stems)} clean')
assert len(fails) == 0, f'Regressions in {len(fails)} graphs: {fails[:5]}'
print('GATE-4-1 PASSED')
"
```

| Check | Expected | Fail action |
|-------|----------|-------------|
| v8 edge types (0–10) unchanged in v9 | Bit-for-bit identical | Regression in extractor — debug `_add_control_dep_edges()` for side effects |
| P99 edge count | < 5,000 | CD edges too dense — tighten depth or `check_types` |
| Max edge count | < 10,000 | Single pathological graph — investigate and exclude |

- **Status:** OPEN (blocking Phase 4)

---

#### GATE-4-2 — Edge mask code verification (P0)

**What:** Confirm CONTROL_DEP(11) is in Phase 1 `struct_mask` and NOT in Phase 2 `cfg_mask`. Control dependence is a scope relationship, not sequential execution order.

```bash
source ml/.venv/bin/activate
PYTHONPATH=. python -c "
import torch
from ml.src.models.gnn_encoder import GNNEncoder

ea = torch.arange(12)
# Phase 1: struct_mask should include types 0,1,2,3,4,5,7,11 (structural + CONTROL_DEP)
# Phase 2: cfg_mask should include types 6,8,9 (or 6,10 for PLAN-3B variant) but NOT 11
# Exact masks come from gnn_encoder — instantiate and inspect
print('Verify CONTROL_DEP(11) in Phase 1 struct_mask: YES')
print('Verify CONTROL_DEP(11) in Phase 2 cfg_mask:   NO (it is structural, not sequential)')
print('GATE-4-2 requires reading gnn_encoder.py Phase masks after PLAN-4C is implemented')
"
```

After PLAN-4C is implemented, replace the above with a concrete assertion matching GATE-3A-1 format.

- **Status:** OPEN (BLOCKED on PLAN-4C)

---

#### GATE-4-3 — Smoke test (P1 — 2 epochs only)

**What:** Same smoke protocol as GATE-3A-3 adapted for 12 edge types and v9 schema.

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. python ml/scripts/train.py \
    --run-name v9.0-smoke-YYYYMMDD \
    --experiment-name sentinel-v9 \
    --cache-path ml/data/cached_dataset_v9.pkl \
    --splits-dir ml/data/splits/deduped \
    --epochs 2 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee ml/logs/v9.0-smoke-YYYYMMDD.log
```

| Signal | Expected | Fail condition |
|--------|----------|----------------|
| Step loss at ep1 step 100 | 0.15–0.20 (same range as v8 runs) | < 0.05 (collapsed) or > 0.30 |
| JK Phase 1 weight at ep1 | > JK Phase 1 from v8 smoke (Phase 1 now has CD edges) | Lower than v8 Phase 1 → CD edges not contributing |
| JK Phase 2 weight at ep1 | Similar to v8-A/B Phase 2 baseline | Drastic change → CD edges leaking into Phase 2 mask (mask bug) |
| NaN count | 0 | Any NaN → kill immediately |
| No CUDA OOM | — | 12 edge types → marginally larger embedding; batch_size=8 should still fit |

- **Status:** OPEN (BLOCKED on GATE-4-0, GATE-4-1, GATE-4-2)

---

#### GATE-4-4 — Early epoch monitoring (P1 — first 10 epochs of full run)

| Epoch | Check | Action if fails |
|-------|-------|-----------------|
| ep1 end | F1 > 0.10 | Kill — not learning; check loss and CD edge mask |
| ep3 end | JK Phase 1 weight > v8 baseline Phase 1 at ep3 | CD edges adding Phase 1 signal; if still lower → may not be Phase 1 placement |
| ep5 end | Reentrancy F1 > v8-A at ep5 (CD should help guard-scope detection) | If lower, CD edges may be noise-diluting Phase 1 just as DEF_USE diluted Phase 2 |
| ep8 end | F1 > 0.15 (past aux warmup) | Kill if < 0.10 after warmup |
| ep10 end | JK Phase 1 weight trend stable or growing | If Phase 1 weight collapsing → Phase 3 (CONTAINS) is absorbing CD signal |

**Additional Phase 4 watch:** Monitor per-class F1 on Timestamp and MishandledException — these are the FN-heavy classes where `CONTROL_DEP` was hypothesized to help (guard-scope mislabeling). If those don't improve over v8-A, Phase 4 is not addressing the right failure mode.

- **Status:** OPEN (BLOCKED on GATE-4-3)

---

## Open Bugs (carried from ACTIVE_BUGS.md)

These were OPEN in v7 and remain unresolved. Address during v8 data preparation.

### BUG-H4 — Timestamp labels: 48.2% have no source evidence
- **File:** `ml/data/processed/multilabel_index_deduped.csv`
- **Impact:** ~463 Timestamp=1 contracts with `uses_block_globals=0` — model learns to associate "no timestamp signal" with timestamp vulnerability (inverted signal)
- **Fix:** Filter Timestamp=1 rows where `uses_block_globals=0.0` across all nodes — can be added to `label_cleaner.py` as a new precondition rule
- **When:** During PLAN-2E (re-run label_cleaner on v8 graphs)
- **Status:** OPEN

### BUG-H5 — ~14% of Reentrancy=1 contracts have no external calls
- **File:** `ml/data/graphs/*.pt`
- **Impact:** ~630 mislabeled training samples; Reentrancy requires external call by definition
- **Fix:** Filter Reentrancy=1 rows where `external_call_count=0` across all FUNCTION nodes — add to `label_cleaner.py`
- **When:** During PLAN-2E
- **Status:** OPEN

### BUG-M5 — Brainmab contract mislabeled across 4 classes
- **File:** `ml/data/processed/multilabel_index_deduped.csv`
- **Impact:** Standard ERC20 labeled Reentrancy=1, CallToUnknown=1, IntegerUO=1, MishandledException=1 — worst-case OR-label noise
- **Fix:** Identify contract hash and remove from CSV; audit any contract with ≥3 co-occurring labels and zero supporting features
- **When:** Before PLAN-2E
- **Status:** OPEN

### BUG-M6 — Token files carry stale `feature_schema_version='v4'`
- **Files:** `ml/data/tokens_windowed/*.pt` metadata
- **Impact:** Cosmetic — not checked at load time; will be resolved automatically when tokens are regenerated for v8
- **When:** Resolved automatically by v8 re-tokenization (runs as part of re-extraction)
- **Status:** OPEN (auto-resolved in Phase 2)

### BUG-M7 — 8.5% of graphs have empty `contract_path`
- **File:** `ml/data/graphs/*.pt` metadata
- **Impact:** Cannot manually inspect or cross-reference against source for those contracts
- **Fix:** Build `ml/scripts/backfill_contract_paths.py` — MD5→path map from BCCC-SCsVul-2024/, patch metadata in-place
- **When:** Before Phase 2 archive (or during extraction if path tracking is added to extractor)
- **Status:** OPEN

### BUG-L3 — Hash-based graph-token pairing fragile to directory restructuring
- **File:** `ml/src/datasets/dual_path_dataset.py`
- **Impact:** Any BCCC reorganization invalidates all hashes, requiring full re-extraction
- **Fix:** Store both path-hash (pairing) and content-hash (dedup)
- **Status:** DEFERRED — low immediate impact; reconsider at v9 if dataset is restructured

---

## Operational Items

### OPS-1 — Install CUDA toolkit + Flash Attention 2
- **Status:** **DONE (2026-05-19)** — CUDA toolkit installed; flash-attn 2.8.3 installed via pre-built wheel
- **Current speed stack:**
  - S2 Fused AdamW: `trainer.py:1029` — `fused=True` ✓
  - S3 SDPA: `transformer_encoder.py:131–143` ✓ (active — see note below)
  - flash-attn 2.8.3: installed ✓
- **Note:** CodeBERT (RoBERTa architecture) does NOT support `attn_implementation="flash_attention_2"` via HuggingFace Transformers — only decoder models (LLaMA, Mistral etc.) have it. The `TransformerEncoder` falls back to SDPA, which is the correct ceiling for this architecture. flash-attn is available for any future backbone swap.

### OPS-2 — Monitor v7 training through convergence
- **Epoch 9 watch point: PASSED** — F1 jumped 0.2102→0.2317 with no collapse, no guardrail fires.
- **FINAL RESULT:** Best F1=0.2651 at epoch 23 (2026-05-19 05:49). Killed at epoch 34, patience 10/30.
- **Plateau analysis:** Val F1 flat epochs 24–33 (0.2432–0.2618) while train loss still declining. Structural ceiling confirmed by JK Phase3 weight drifting 0.572→0.784 — model learned to distrust intra-function CFG signal. No cross-function edges available to rescue.
- **v7 vs v6:** +54.4% F1 (0.2651 vs 0.1717). Improvement attributed to: 3-hop CFG (conv3c), AsymmetricLoss, LR schedule, weighted sampler, correct VRAM headroom.
- **Status:** DONE — best checkpoint at `ml/checkpoints/v7.0_best.pt`

### OPS-3 — Commit Phase 0 cleanup to git
- **Files changed:** `ml/src/preprocessing/graph_extractor.py`
- **Status:** DONE — included in commit 907e442

---

## Summary Table

| ID | Item | Phase | Priority | Blocker | Status |
|----|------|-------|----------|---------|--------|
| P0-1..5 | Phase 0 dead-code + docstring cleanup | 0 | P0 | — | **DONE** |
| PLAN-1A | Validate `node.internal_calls` at node level (10 contracts) | 1 | P0 | v7 training done | **DONE** |
| PLAN-1C | Accumulate `global_cfg_node_map` in extractor loop | 1 | P1 | PLAN-1A | **DONE** |
| PLAN-1D | Implement `_add_icfg_edges()` (CALL_ENTRY, RETURN_TO) | 1 | P1 | PLAN-1C | **DONE** |
| PLAN-1F | Update `graph_schema.py` to v8 constants | 1 | P2 | PLAN-1D/1E | **DONE** |
| PLAN-1G | Update `gnn_encoder.py` Phase 2 mask + embedding size | 1 | P2 | PLAN-1F | **DONE** |
| PLAN-1E | Implement `_add_def_use_edges()` (DEF_USE) | 1 | P1 | PLAN-1C | **DONE** |
| PLAN-1B | 2,000-contract sample validation gate | 1 | P0 | PLAN-1D/1E | **DONE** — presence validated, distribution-by-class not checked (retroactively added as GATE-3A-0) |
| PLAN-2A | Archive v7 graphs | 2 | P0 | Phase 1 done | **DONE** |
| PLAN-2B–2I | Full v8 re-extraction + validation + cache rebuild | 2 | P1 | PLAN-2A | **DONE** |
| PLAN-3G | Fix stale `--run-name` default in `train.py:68` | 3 | P0 | before any v8 run | **DONE** |
| PLAN-3C | v8-AB ablation run (both ICFG + DFG) | 3 | P1 | Phase 2 done | **DONE (2026-05-20, F1=0.2621)** |
| GATE-3A-CACHE | Cache integrity check (schema v8, size, pair count) | 3 | P0 | PLAN-3C done | **DONE (2026-05-21)** |
| GATE-3A-0 | Edge activation analysis — P0 blocking PLAN-3A | 3 | P0 | GATE-3A-CACHE | **DONE (2026-05-21)** |
| GATE-3A-1 | Edge mask code verification | 3 | P0 | GATE-3A-0 | **DONE (2026-05-21)** |
| GATE-3A-2 | Config review before each run (incl. DoS weight, aug data, MLflow, sampler) | 3 | P0 | GATE-3A-1 | OPEN (manual — complete before launch) |
| GATE-3A-VRAM | GPU memory budget check before smoke test | 3 | P0 | GATE-3A-2 | **DONE (2026-05-21)** — 8.0 GB free |
| GATE-3A-3 | Smoke test — 2 epochs + probability spread check | 3 | P1 | GATE-3A-VRAM | OPEN |
| GATE-3A-4 | Early epoch monitoring gate — first 10 eps | 3 | P1 | GATE-3A-3 | OPEN |
| PLAN-3A | v8-A ablation run (ICFG only) | 3 | P1 | GATE-3A-2, GATE-3A-3 | **OPEN — blocked on GATE-3A-2 (config review) + GATE-3A-3 (smoke test)** |
| PLAN-3B | v8-B ablation run (DFG only) | 3 | P1 | same gate sequence | **OPEN (same gate sequence required)** |
| PLAN-3H | Apply optional S4 (cap fusion tokens at 1024) — measure first | 3 | P2 | Phase 2 done | OPEN |
| PLAN-3D | Inspect `jk.last_weights` per run; check fused eye canary | 3 | P2 | PLAN-3A/B | OPEN |
| PLAN-3I | Monitor fused eye loss relative to GNN/TF eyes in v8 | 3 | P2 | PLAN-3A/B | OPEN |
| PLAN-3E | `return_ignored` scalar ablation on best ablation checkpoint | 3 | P2 | PLAN-3A/B | OPEN |
| PLAN-3J | Review `early_stop_patience` after first v8 run | 3 | P3 | PLAN-3A | OPEN |
| PLAN-3F | Document per-class F1 deltas v7→v8-A/B | 3 | P2 | PLAN-3A/B | OPEN |
| PLAN-4A | Implement `_add_control_dep_edges()` (CONTROL_DEP) | 4 | P1 | Phase 3 results | OPEN |
| PLAN-4B | Update `graph_schema.py` to v9 constants | 4 | P2 | PLAN-3E + 4A | OPEN |
| PLAN-4C | Update `gnn_encoder.py` Phase 1 mask + embedding size | 4 | P2 | PLAN-4B | OPEN |
| BUG-H4 | Filter Timestamp labels with no source evidence | 2 | P1 | v8 graphs ready | OPEN |
| BUG-H5 | Filter Reentrancy labels with no external calls | 2 | P1 | v8 graphs ready | OPEN |
| BUG-M5 | Remove Brainmab mislabeled contract | 2 | P2 | — | OPEN |
| BUG-M6 | Stale token schema version metadata | 2 | P3 | auto-resolved by retokenize | OPEN |
| BUG-M7 | 8.5% graphs have empty contract_path | 1 | P3 | — | OPEN |
| BUG-L3 | Path-hash pairing fragile to dir restructure | — | P3 | — | DEFERRED |
| OPS-1 | Install CUDA toolkit + Flash Attention 2 | — | P3 | — | **DONE** (SDPA is ceiling for CodeBERT; flash-attn ready for backbone swap) |
| OPS-2 | Monitor v7 training through convergence | — | P0 | — | **DONE** |
| OPS-3 | Commit Phase 0 cleanup | — | P1 | — | **DONE** |
