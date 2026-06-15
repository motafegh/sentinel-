# Stage 7B Graph Extractor Seam Flip Attestation

**Date:** 2026-06-15  
**Executed by:** Claude Code session (2026-06-15 ~19:00–20:00 UTC)  
**Plan:** `docs/plans/2026-06-15_ml_Stage7B_plan_graph_extractor_seam_flip.md`

---

## Pre-conditions verified

| Check | Result |
|-------|--------|
| Schema flip confirmed (graph_schema.py shim 22 lines) | ✅ PASS |
| Extractor internals already patched (sentinel_data.representation.graph_schema imports at lines 112, 1290) | ✅ PASS |
| ml/ extractor was full source (2056 lines) before flip | ✅ PASS |
| No `from ml.` references in extractor internals | ✅ PASS |

---

## Steps completed

| Step | Action | Result |
|------|--------|--------|
| Step 1 | Backups created: `graph_extractor.py.bak-stage7b-2026-06-15` in both locations | ✅ Done |
| Step 2 | Full source (2056 lines) copied to `sentinel_data/representation/graph_extractor.py` | ✅ Done |
| Step 3 | Thin shim written to `ml/src/preprocessing/graph_extractor.py` (28 lines, with `__getattr__` proxy) | ✅ Done |
| Step 4 | `sentinel_data/representation/__init__.py` verified — already imports from canonical path | ✅ Clean |

**Note on shim:** The shim includes a `__getattr__` proxy to delegate any attribute lookups
(including private functions like `_compute_uses_block_globals`, `_compute_in_unchecked`)
to the canonical `sentinel_data.representation.graph_extractor` module. This prevents
regressions in legacy smoke tests that access private functions via `getattr(module, name)`.

---

## Gate results

| Gate | Check | Result |
|------|-------|--------|
| Gate 1 | Import smoke: all critical paths import OK; `extract_contract_graph` same object via both paths | ✅ PASS |
| Gate 1+ | `__getattr__` proxy: `_compute_uses_block_globals` and `_compute_in_unchecked` accessible via shim | ✅ PASS |
| Gate 2 | Smoke suite (Pre-Run-9): **3/8 PASS** vs **1/8 PASS before flip** (net improvement) | ✅ IMPROVED |
| Gate 3 | C.2.1 inference regression: 50/65 = 76.9% correct — identical to pre-flip result | ✅ PASS |
| Gate 4 | Line counts: shim=28 lines, canonical=2056 lines | ✅ PASS |
| Training smoke | Full 1-epoch training run into /tmp — exit code 0. ep1: f1_tuned=0.441, auc_roc=0.816, loss=0.141, 4507 steps, 0 spikes | ✅ PASS |

### Smoke suite detail (Gate 2)

| Test | Before flip | After flip | Notes |
|------|-------------|------------|-------|
| Fix #1 | PASS | PASS | No change |
| Fix #2 | FAIL (no module 'ml') | PASS | ✅ IMPROVED — now resolves via sentinel_data |
| Fix #3 | FAIL (no module 'ml') | PASS | ✅ IMPROVED — now resolves via sentinel_data |
| Fix #4 | FAIL (no module 'ml') → G4.6 still fails | PARTIAL | G4.2+G4.5 PASS; G4.6 (SentinelModel construction) still fails — pre-existing PYTHONPATH issue |
| Fix #5 | FAIL (slither not on PATH) | FAIL (same) | Pre-existing, unrelated |
| Fix #6 | FAIL (no module 'ml') | FAIL (same) | Pre-existing PYTHONPATH issue in smoke script itself |
| Fix #7 | FAIL (wrong script path) | FAIL (same) | Pre-existing, unrelated |
| Fix #8 | FAIL (missing doc) | FAIL (same) | Pre-existing, unrelated |

---

## Final file states

| File | State | Lines |
|------|-------|-------|
| `data_module/sentinel_data/representation/graph_extractor.py` | **CANONICAL** — full source | 2056 |
| `ml/src/preprocessing/graph_extractor.py` | **SHIM** — re-export + `__getattr__` proxy | 28 |
| `data_module/sentinel_data/representation/graph_schema.py` | CANONICAL (done prior session) | ~429 |
| `ml/src/preprocessing/graph_schema.py` | SHIM (done prior session) | 22 |

---

## Stage 7B overall status

- `graph_schema`: CANONICAL in sentinel_data ✅ (done 2026-06-12)
- `graph_extractor`: CANONICAL in sentinel_data ✅ (done 2026-06-15)
- **Stage 7B seam swap: COMPLETE**

All source-of-truth code now lives in `data_module/sentinel_data/representation/`.
`ml/src/preprocessing/` contains only thin re-export shims.
No model files, training files, or inference files required import changes.

---

## Backup cleanup — DONE (2026-06-15)

All temp files removed after training smoke confirmed success:
- `ml/src/preprocessing/graph_extractor.py.bak-stage7b-2026-06-15` — deleted ✅
- `data_module/sentinel_data/representation/graph_extractor.py.bak-stage7b-2026-06-15` — deleted ✅
- `/tmp/sentinel-smoke-ckpt/` — deleted ✅
