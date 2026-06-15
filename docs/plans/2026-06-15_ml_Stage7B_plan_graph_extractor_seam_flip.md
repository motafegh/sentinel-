# Plan: Stage 7B Graph Extractor Seam Flip

**Date:** 2026-06-15  
**Module:** ml / data_module  
**Phase:** Stage7B  
**Status:** READY FOR EXECUTION  
**Pre-requisite:** Run 12 complete ✅, schema flip complete ✅

---

## Context

Stage 7B seam swap is partially complete:

| File | State |
|------|-------|
| `data_module/sentinel_data/representation/graph_schema.py` | ✅ **CANONICAL** — full source (429 lines) |
| `ml/src/preprocessing/graph_schema.py` | ✅ **SHIM** — 22-line re-export from sentinel_data |
| `data_module/sentinel_data/representation/graph_extractor.py` | ❌ **OLD ADAPTER** — 77-line thin wrapper still pointing TO ml/ |
| `ml/src/preprocessing/graph_extractor.py` | ❌ **FULL SOURCE** — 2056-line canonical source (still in ml/) |

The extractor's *internal* imports have already been patched (lines 108–116 in the ml/ file
now use `from sentinel_data.representation.graph_schema import ...`). The file bodies have
NOT been swapped. This plan does that swap.

---

## Import Site Map

### Group A — already work via ml/ shim (NO changes needed after flip)

These files import from `ml.src.preprocessing.*`. After the flip, the ml/ files become
thin shims. The callers never change; they transparently get the sentinel_data version.

| File | Import | Via |
|------|--------|-----|
| `ml/src/models/gnn_encoder.py:52` | `NODE_FEATURE_DIM, NODE_TYPES, NUM_EDGE_TYPES, EDGE_TYPES` | `..preprocessing.graph_schema` (schema shim — already done) |
| `ml/src/models/sentinel_model.py:64` | `NODE_TYPES` | `..preprocessing.graph_schema` (schema shim — already done) |
| `ml/src/inference/predictor.py:73` | `NODE_FEATURE_DIM, NODE_TYPES` | `..preprocessing.graph_schema` (schema shim — already done) |
| `ml/src/inference/predictor.py:400` | `EDGE_TYPES` (deferred) | `ml.src.preprocessing.graph_schema` (schema shim — already done) |
| `ml/src/inference/cache.py:48` | `NODE_FEATURE_DIM` | `..preprocessing.graph_schema` (schema shim — already done) |
| `ml/src/training/training_logger.py:34` | `NODE_FEATURE_DIM` | `ml.src.preprocessing.graph_schema` (schema shim — already done) |
| `ml/src/preprocessing/__init__.py` | `graph_schema.*`, `graph_extractor.*` | Both become shims after flip — no change needed |
| `ml/src/inference/preprocess.py:80` | `..preprocessing.graph_extractor` | Hits extractor shim after flip — no change needed |
| `ml/src/inference/preprocess.py:88` | `FEATURE_SCHEMA_VERSION` | Via schema shim — already done |

**Result: ZERO ml/src files need modification.** The shim pattern absorbs all existing imports.

### Group B — data_module internal (fixed as part of the flip itself)

| File | Current state | After flip |
|------|---------------|------------|
| `data_module/sentinel_data/representation/graph_extractor.py` | 77-line adapter with `from ml.src.preprocessing.graph_extractor import ...` | Replaced wholesale with full 2056-line canonical source |
| `data_module/sentinel_data/representation/__init__.py:33` | `from sentinel_data.representation.graph_extractor import (...)` | Still correct — now resolves to the canonical source directly |

---

## Circular Import Analysis

After the flip the dependency graph is:

```
ml/src/models/gnn_encoder.py
  → ml/src/preprocessing/graph_schema (shim)
      → sentinel_data/representation/graph_schema (canonical) ✅ no cycle

ml/src/inference/preprocess.py
  → ml/src/preprocessing/graph_extractor (NEW shim after flip)
      → sentinel_data/representation/graph_extractor (canonical)
          → sentinel_data/representation/graph_schema (canonical) ✅ no cycle

data_module/sentinel_data/representation/__init__.py
  → sentinel_data/representation/graph_extractor (canonical)
      → sentinel_data/representation/graph_schema (canonical) ✅ no cycle
```

The canonical extractor uses `from sentinel_data.representation.graph_schema import ...`
(already patched at lines 108–116). It does NOT import from ml/. No cycle possible.

---

## Execution Steps

### Pre-flight (verify state before any change)

```bash
# 1. Confirm schema flip is done
head -5 ml/src/preprocessing/graph_schema.py
# Expected: "# Stage 7B seam swap: sentinel_data.representation.graph_schema is now source of truth"

# 2. Confirm extractor is NOT yet flipped
wc -l ml/src/preprocessing/graph_extractor.py
# Expected: ~2056 lines (full source, not a shim)

wc -l data_module/sentinel_data/representation/graph_extractor.py
# Expected: ~77 lines (old adapter)

# 3. Confirm extractor internal imports already patched
grep -n "from sentinel_data" ml/src/preprocessing/graph_extractor.py
# Expected: lines 108–116 show sentinel_data.representation.graph_schema imports

# 4. Confirm no ml/ path in extractor internals (other than the file path itself)
grep -n "from ml\." ml/src/preprocessing/graph_extractor.py
# Expected: 0 results (internal imports already cleaned up)
```

### Step 1 — Backup

```bash
# Safety copy (NOT in docs/.bin — this is a live backup, not a stale duplicate)
cp ml/src/preprocessing/graph_extractor.py \
   ml/src/preprocessing/graph_extractor.py.bak-stage7b-2026-06-15

cp data_module/sentinel_data/representation/graph_extractor.py \
   data_module/sentinel_data/representation/graph_extractor.py.bak-stage7b-2026-06-15
```

### Step 2 — Copy full source to sentinel_data (make it canonical)

```bash
cp ml/src/preprocessing/graph_extractor.py \
   data_module/sentinel_data/representation/graph_extractor.py
```

After this, `data_module/sentinel_data/representation/graph_extractor.py` is the 2056-line
canonical source. It already has the correct `from sentinel_data.representation.graph_schema`
imports internally (patched in a prior session).

### Step 3 — Write thin shim to ml/src/preprocessing/graph_extractor.py

Replace the 2056-line file with this shim (match the graph_schema shim pattern exactly):

```python
# Stage 7B seam swap: sentinel_data.representation.graph_extractor is now source of truth.
# This file is a thin re-export shim. Do not add logic here.
from sentinel_data.representation.graph_extractor import (  # noqa: F401
    EmptyGraphError,
    GraphExtractionConfig,
    GraphExtractionError,
    extract_contract_graph,
)
```

> **Exact symbols to re-export:** Confirm the set by running:
> `grep -n "^class\|^def\|^[A-Z_]\+ =" ml/src/preprocessing/graph_extractor.py | head -40`
> before Step 2. The shim must export every public name that `ml/src/preprocessing/__init__.py`
> currently imports from `.graph_extractor`. Check `__init__.py` to confirm the exact list.

### Step 4 — Verify sentinel_data __init__.py is clean

```bash
head -40 data_module/sentinel_data/representation/__init__.py
```

Confirm line 33 (or nearby) still reads:
```python
from sentinel_data.representation.graph_extractor import (extract_contract_graph, ...)
```
This is correct — it now resolves to the canonical source. **No change needed.**

### Step 5 — Remove backup files after smoke passes

```bash
rm ml/src/preprocessing/graph_extractor.py.bak-stage7b-2026-06-15
rm data_module/sentinel_data/representation/graph_extractor.py.bak-stage7b-2026-06-15
```

---

## Verification Gates

### Gate 1 — Import smoke (run before smoke suite)

```bash
ml/.venv/bin/python -c "
from sentinel_data.representation.graph_extractor import extract_contract_graph
from ml.src.preprocessing.graph_extractor import extract_contract_graph as ecg2
from ml.src.preprocessing import GraphExtractionConfig, EmptyGraphError
from ml.src.models.gnn_encoder import SentinelGNN
from ml.src.inference.predictor import Predictor
print('ALL IMPORTS OK')
"
```

Expected: `ALL IMPORTS OK` with no ImportError or circular import.

### Gate 2 — Smoke suite

```bash
ml/.venv/bin/python ml/scripts/smoke/run_all.py
```

Expected: all 12 smoke tests PASS.

### Gate 3 — C.2.1 inference regression (optional but recommended)

```bash
ml/.venv/bin/python ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/scripts/run_c21_smoke_inference.py
```

Expected: ≥70% top-class correct (same 76.9% as before flip, since no model weights change).

### Gate 4 — Confirm shim line count

```bash
wc -l ml/src/preprocessing/graph_extractor.py
# Expected: ~8 lines

wc -l data_module/sentinel_data/representation/graph_extractor.py
# Expected: ~2056 lines
```

---

## Rollback Procedure

If any gate fails:

```bash
# Restore from backup (Step 1 backups must still exist)
cp ml/src/preprocessing/graph_extractor.py.bak-stage7b-2026-06-15 \
   ml/src/preprocessing/graph_extractor.py

cp data_module/sentinel_data/representation/graph_extractor.py.bak-stage7b-2026-06-15 \
   data_module/sentinel_data/representation/graph_extractor.py
```

The system returns to its pre-flip state. No other files were changed.

---

## Post-Execution Attestation (fill in after gates pass)

Write attestation to:
`ml/testing_specs/2026-06-15_ml_Run12_validation_spec_execution/attestations/Stage7B_extractor_flip_attestation.md`

Template:

```markdown
# Stage 7B Graph Extractor Seam Flip Attestation

**Date:** 2026-06-15
**Executed by:** [session]

## Pre-conditions verified
- [ ] Schema flip confirmed (graph_schema.py shim 22 lines)
- [ ] Extractor internals already patched (sentinel_data.representation.graph_schema imports)
- [ ] ml/ extractor was full source (2056 lines) before flip

## Steps completed
- [ ] Step 1 — Backups created
- [ ] Step 2 — Full source copied to sentinel_data/representation/graph_extractor.py
- [ ] Step 3 — Shim written to ml/src/preprocessing/graph_extractor.py
- [ ] Step 4 — sentinel_data __init__.py verified clean

## Gates passed
- [ ] Gate 1 — Import smoke: ALL IMPORTS OK
- [ ] Gate 2 — Smoke suite: all 12 PASS
- [ ] Gate 3 — C.2.1 regression: ≥70% top-class
- [ ] Gate 4 — Line counts correct

## Stage 7B overall status
- graph_schema: CANONICAL in sentinel_data ✅ (done prior session)
- graph_extractor: CANONICAL in sentinel_data ✅ (done this session)
- Stage 7B seam swap: **COMPLETE**
```

---

## Files Changed Summary

| File | Change | Lines |
|------|--------|-------|
| `data_module/sentinel_data/representation/graph_extractor.py` | OLD 77-line adapter → CANONICAL 2056-line source | +1979 |
| `ml/src/preprocessing/graph_extractor.py` | FULL 2056-line source → THIN ~8-line shim | −2048 |
| All ml/src model files | **NO CHANGE** — shim absorbs existing imports | 0 |

---

## What This Completes

After this flip, Stage 7B is **fully done**:
- All source-of-truth code lives in `data_module/sentinel_data/representation/`
- `ml/src/preprocessing/` contains only thin re-export shims
- No model files, training files, or inference files need import changes
- `data_module` is a self-contained, importable package with the full graph logic

Run 13 data pipeline work can proceed against `sentinel_data` directly without
needing to reference `ml/src/preprocessing/` at all.
