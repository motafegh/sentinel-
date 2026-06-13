# Seam Swap Completion — graph_extractor + windowed_tokenizer source-of-truth migration to data_module

**Date:** 2026-06-13 (post-Run 12 launch, ep1 in progress)
**Status:** PLAN — not yet executing
**Author:** Ali + Claude (verified against source code, not docs)
**Predecessor plans (now in `live_plans/archive/`):**
- `archive/stage_7a_export_module.md` — Stage 7A work, completed 2026-06-12
- `archive/stage_7b_seam_swap.md` — Stage 7B original plan, **prescribed the correct move** (this plan completes it)
- `archive/stage_7b_seam_swap_active.md` — Stage 7B implementation log; documents the partial implementation

---

## 1. Context

Stage 7B "seam swap" was supposed to flip the source of truth for the SENTINEL preprocessing pipeline from `ml/src/preprocessing/*` to `data_module/sentinel_data/representation/*`. The intent: make `sentinel-data` an independently testable, versionable, reusable package per `data_module/docs/architecture.md` §"One-way dependency" + `data_module/__init__.py:30-34`.

**What actually happened** (verified at file level, 2026-06-13 ~23:55 UTC):

| File | `ml/src/` (REAL impl) | `data_module/sentinel_data/` (adapter) | Seam-swap state |
|---|---|---|---|
| `graph_schema.py` | 22 lines (shim) | 251 lines (REAL) | ✅ Correct (data_module canonical) |
| `graph_extractor.py` | **2,056 lines (REAL)** | 77 lines (adapter) | ❌ **Inverted** (ml/src has real) |
| `tokenizer.py` (windowed) | **175 lines (REAL)** | 72 lines (adapter) | ❌ **Inverted** (ml/src has real) |

The original `stage_7b_seam_swap.md` plan PRESCRIBED the correct move:

> "`git rm ml/src/preprocessing/graph_extractor.py` (moved to `data_module/sentinel_data/representation/graph_extractor.py`)"
> "Replace with imports from `sentinel_data.representation`"

But the actual implementation did the OPPOSITE — it kept the real impl in `ml/src/` and made `data_module/sentinel_data/representation/` thin adapters pointing BACK to `ml/src/`. Result: **`sentinel-data` is NOT independently usable as a package** — it requires `ml/` on PYTHONPATH.

This plan completes the seam swap by moving the canonical implementations to `data_module/` and turning `ml/src/preprocessing/graph_extractor.py` + `ml/src/data_extraction/windowed_tokenizer.py` into thin shims (mirror of the existing `ml/src/preprocessing/graph_schema.py` shim pattern).

---

## 2. Current state (verified 2026-06-13 23:55 UTC)

### 2.1 The files we'll move

**`ml/src/preprocessing/graph_extractor.py` (2,056 lines)** — REAL implementation
- Header: "Canonical Solidity-to-PyG graph extraction (v8 schema)"
- Public surface: `GraphExtractionConfig`, `GraphExtractionError`, `SolcCompilationError`, `SlitherParseError`, `EmptyGraphError`, `extract_contract_graph(sol_path, config) → Data`
- 36-issue pre-Run-8 audit regression tests reference specific line numbers in this file (e.g., `:1656-1670` for EMITS edge)

**`ml/src/data_extraction/windowed_tokenizer.py` (175 lines)** — REAL implementation
- Header: "Windowed tokenization for SENTINEL — GraphCodeBERT, [W, 512] output"
- Public surface: `tokenize_windowed_contract`, `init_worker`, `TOKENIZER_MODEL`, `WINDOW_SIZE`, `STRIDE`, `MAX_WINDOWS`
- Note: this is the CORRECT module (graphcodebert-base). The old `ml/src/data_extraction/tokenizer.py` uses codebert-base and is **legacy only** (kept for v1 batch scripts).

### 2.2 The adapters we'll replace

**`data_module/sentinel_data/representation/graph_extractor.py` (77 lines)** — thin adapter
- Currently: `__getattr__` lazy re-export from `ml.src.preprocessing.graph_extractor`
- Will be replaced with: the 2,056-line real implementation (moved here)

**`data_module/sentinel_data/representation/tokenizer.py` (72 lines)** — thin adapter
- Currently: `__getattr__` lazy re-export from `ml.src.data_extraction.windowed_tokenizer`
- Will be replaced with: the 175-line real implementation (moved here)

### 2.3 The shims we'll create

**`ml/src/preprocessing/graph_extractor.py` (will become ~22 lines)** — thin re-export shim
- Pattern: identical to existing `ml/src/preprocessing/graph_schema.py:8-22`
- Re-exports every public symbol from `sentinel_data.representation.graph_extractor`

**`ml/src/data_extraction/windowed_tokenizer.py` (will become ~22 lines)** — thin re-export shim
- Same pattern as above
- Re-exports from `sentinel_data.representation.tokenizer`

### 2.4 The ~30 importers (verify-against, but no changes needed)

| Group | Count | Import path used |
|---|---|---|
| `ml/src/models/*` (gnn_encoder, sentinel_model) | 2 | `from ml.src.preprocessing.graph_schema import ...` |
| `ml/src/inference/*` (predictor, cache) | 2 | `from ml.src.preprocessing.graph_extractor import ...` |
| `ml/src/training/training_logger.py` | 1 | imports from `ml/src/data_extraction/windowed_tokenizer` |
| `ml/scripts/interpretability/*` (24 scripts) | 24 | various |
| `ml/scripts/_legacy_data_pipeline/*` (4 scripts) | 4 | various |
| `ml/scripts/archive/*` (~10 scripts) | 10 | various |
| `ml/scripts/compile_smoke_test.py`, `vram_gate_test.py` | 2 | various |
| `ml/tests/*` (test_model, test_gnn_encoder, test_preprocessing, test_cache, test_cfg_embedding_separation) | 5 | various |
| `data_module/tests/test_representation/*` (5 tests) | 5 | import from data_module (correct) |
| `data_module/sentinel_data/representation/cfg_builder.py` | 1 | imports from graph_extractor |

**Key insight:** All ~30 ml/ importers go through `from ml.src.preprocessing.graph_extractor import ...` (not through the data_module path). If we KEEP `ml/src/preprocessing/graph_extractor.py` as a shim that re-exports from `sentinel_data.representation.graph_extractor`, **zero importers need to change.**

---

## 3. Goals & Non-Goals

### 3.1 Goals

1. **G1:** `data_module/sentinel_data/representation/graph_extractor.py` contains the 2,056-line canonical implementation (not a thin adapter)
2. **G2:** `data_module/sentinel_data/representation/tokenizer.py` contains the 175-line canonical windowed_tokenizer (not a thin adapter)
3. **G3:** `ml/src/preprocessing/graph_extractor.py` is a thin ~22-line re-export shim (mirror graph_schema.py pattern)
4. **G4:** `ml/src/data_extraction/windowed_tokenizer.py` is a thin ~22-line re-export shim
5. **G5:** `data_module` is independently usable as a package (no PYTHONPATH dependency on `ml/`)
6. **G6:** All 598/27/0 data_module tests still pass
7. **G7:** All ~38 ml/ tests still pass
8. **G8:** Run 12 continues training unaffected (PID 230342, 30:46 elapsed at plan time)
9. **G9:** `architecture.md` is updated to match the corrected state
10. **G10:** A single coherent commit captures the whole change

### 3.2 Non-Goals

- **NG1:** NOT changing any public API (all imports continue to resolve)
- **NG2:** NOT changing graph_extractor's logic (pure file move + 22-line shim replacement)
- **NG3:** NOT changing windowed_tokenizer's logic (same)
- **NG4:** NOT touching `ml/src/inference/preprocess.py` (deferred — see §11)
- **NG5:** NOT touching `ml/scripts/*` (the shims make them work unchanged)
- **NG6:** NOT touching `ml/src/models/*` (they import from shim, shim still works)
- **NG7:** NOT touching `ml/src/training/trainer.py` (uses v3 export, not live extraction)
- **NG8:** NOT removing the `_archive/seam_swap_pre_2026-06-12/` backup (it documents the pre-seam-swap state for historical reference)

---

## 4. Architecture target (what "done" looks like)

### 4.1 File structure after completion

```
data_module/sentinel_data/representation/
├── graph_schema.py        (251 lines — REAL, source of truth for constants)
├── graph_extractor.py     (2,056 lines — REAL, source of truth for extraction)
├── tokenizer.py           (175 lines — REAL, source of truth for windowed tokenization)
├── orchestrator.py        (unchanged)
├── cfg_builder.py         (unchanged)
├── cache_manager.py       (unchanged)
├── versioner.py           (unchanged)
├── call_graph.py          (unchanged - v3.1 stub)
├── pdg_builder.py         (unchanged - v3.1 stub)
└── opcode_extractor.py    (unchanged - v3.1 stub)

ml/src/preprocessing/
├── __init__.py            (unchanged - 44 lines)
├── graph_schema.py        (22 lines — thin shim, re-exports from data_module)
├── graph_extractor.py     (22 lines — thin shim, re-exports from data_module)  ← CHANGED
└── graph_extractor_REAL_backup_2026-06-13.py  (2,056 lines — backup, will NOT be created in ml/src since git history preserves it)

ml/src/data_extraction/
├── __init__.py            (unchanged)
├── tokenizer.py           (unchanged - LEGACY codebert-base, v1-only)
└── windowed_tokenizer.py  (22 lines — thin shim, re-exports from data_module)  ← CHANGED
```

### 4.2 Import pattern after completion

**Before (current — broken):**
```python
# ml/src/preprocessing/graph_extractor.py is the REAL impl
# data_module/sentinel_data/representation/graph_extractor.py is a thin adapter
from ml.src.preprocessing.graph_extractor import extract_contract_graph  # works

# data_module CANNOT extract without ml/ on PYTHONPATH
from sentinel_data.representation.graph_extractor import extract_contract_graph  # requires ml/
```

**After (target — fixed):**
```python
# data_module/sentinel_data/representation/graph_extractor.py is the REAL impl
# ml/src/preprocessing/graph_extractor.py is a thin shim
from sentinel_data.representation.graph_extractor import extract_contract_graph  # works STANDALONE

# ml/ importers still work via the shim
from ml.src.preprocessing.graph_extractor import extract_contract_graph  # works via re-export
```

### 4.3 The shim template (mirror `ml/src/preprocessing/graph_schema.py`)

```python
# ml/src/preprocessing/graph_extractor.py (post-completion, ~22 lines)
"""graph_extractor.py — thin re-export shim.

Stage 7B seam-swap completion (2026-06-13): the canonical implementation
now lives in `data_module/sentinel_data/representation/graph_extractor.py`.
This file re-exports every public symbol for backward compatibility with
~30 existing importers (ml/src/models/*, ml/src/inference/*, ml/scripts/*).

To change extraction logic: edit the data_module file. The shim picks up
the change automatically.
"""
from sentinel_data.representation.graph_extractor import (  # noqa: F401
    GraphExtractionConfig,
    GraphExtractionError,
    SolcCompilationError,
    SlitherParseError,
    EmptyGraphError,
    extract_contract_graph,
)
```

(Same template for `ml/src/data_extraction/windowed_tokenizer.py`.)

---

## 5. Step-by-step execution plan

### Phase 0: Pre-flight checks (5 min)

**0.1 Verify Run 12 is still running**
```bash
ps -p 230342 -o pid,etime,cmd
# Expected: process alive, elapsed time growing
```

**0.2 Verify tests pass BEFORE the change (baseline)**
```bash
cd /home/motafeq/projects/sentinel
ml/.venv/bin/python -m pytest data_module/tests/ -q --tb=line 2>&1 | tail -3
# Expected: 598 passed, 27 skipped, 0 failed
ml/.venv/bin/python -m pytest ml/tests/ -q --tb=line 2>&1 | tail -3
# Expected: ~38 passed, 0 failed
```

**0.3 Verify byte-identical guarantee for graph_extractor**
- The original `test_byte_identical_regression.py` (data_module) tests that the v9 graph schema produces identical output to a reference. This test must STILL pass after the move (since we're not changing logic, just location).
- The `test_thin_adapter.py` (data_module) tests that the shim preserves all public symbols. We need to make sure the new shim in ml/src/ has the same public surface.

**0.4 Verify graph_extractor's public API surface**
```bash
grep -E '^def |^class ' ml/src/preprocessing/graph_extractor.py
# Record the list of public symbols
```
Expected public symbols (from header docstring): `GraphExtractionConfig`, `GraphExtractionError`, `SolcCompilationError`, `SlitherParseError`, `EmptyGraphError`, `extract_contract_graph`. The new shim must re-export ALL of these.

**0.5 Verify windowed_tokenizer's public API surface**
```bash
grep -E '^[A-Z_]+|^def ' ml/src/data_extraction/windowed_tokenizer.py | head -20
# Record the list of public symbols
```
Expected: `TOKENIZER_MODEL`, `WINDOW_SIZE`, `STRIDE`, `MAX_WINDOWS`, `tokenize_windowed_contract`, `init_worker`.

---

### Phase 1: Move graph_extractor (15 min, ZERO Run 12 impact)

**1.1 Copy real implementation to data_module**
```bash
cd /home/motafeq/projects/sentinel
cp ml/src/preprocessing/graph_extractor.py data_module/sentinel_data/representation/graph_extractor.py
wc -l data_module/sentinel_data/representation/graph_extractor.py
# Expected: 2,056 lines (was 77)
```

**1.2 Verify the copy is byte-identical to the source**
```bash
diff -q ml/src/preprocessing/graph_extractor.py data_module/sentinel_data/representation/graph_extractor.py
# Expected: no output (files identical)
```

**1.3 Replace the ml/src/ file with a thin shim**
Write the 22-line shim content (template in §4.3) to `ml/src/preprocessing/graph_extractor.py`. The shim must:
- Re-export every public symbol from the data_module canonical
- Match the docstring style of `ml/src/preprocessing/graph_schema.py`
- Use `noqa: F401` to silence "imported but unused" warnings

**1.4 Quick smoke test — verify the shim works**
```bash
cd /home/motafeq/projects/sentinel
PYTHONPATH=. ml/.venv/bin/python -c "
from ml.src.preprocessing.graph_extractor import extract_contract_graph, GraphExtractionConfig, GraphExtractionError
from sentinel_data.representation.graph_extractor import extract_contract_graph as canonical_extract
print('Shim works:', extract_contract_graph is canonical_extract)
"
# Expected: Shim works: True
```

**1.5 Run a quick test subset to catch import-time regressions**
```bash
cd /home/motafeq/projects/sentinel
ml/.venv/bin/python -m pytest data_module/tests/test_representation/test_thin_adapter.py -q --tb=short 2>&1 | tail -5
# Expected: thin adapter tests pass (they test the new shim)
```

---

### Phase 2: Move windowed_tokenizer (10 min, ZERO Run 12 impact)

**2.1 Copy real implementation to data_module**
```bash
cd /home/motafeq/projects/sentinel
cp ml/src/data_extraction/windowed_tokenizer.py data_module/sentinel_data/representation/tokenizer.py
wc -l data_module/sentinel_data/representation/tokenizer.py
# Expected: 175 lines (was 72)
```

**2.2 Verify the copy is byte-identical**
```bash
diff -q ml/src/data_extraction/windowed_tokenizer.py data_module/sentinel_data/representation/tokenizer.py
# Expected: no output
```

**2.3 Replace the ml/src/ file with a thin shim**
Same template as Phase 1.3. The shim must re-export:
- `TOKENIZER_MODEL`, `WINDOW_SIZE`, `STRIDE`, `MAX_WINDOWS` (constants)
- `tokenize_windowed_contract`, `init_worker` (functions)

**2.4 Quick smoke test — verify the shim works**
```bash
cd /home/motafeq/projects/sentinel
PYTHONPATH=. ml/.venv/bin/python -c "
from ml.src.data_extraction.windowed_tokenizer import (
    tokenize_windowed_contract, init_worker,
    TOKENIZER_MODEL, WINDOW_SIZE, STRIDE, MAX_WINDOWS,
)
from sentinel_data.representation.tokenizer import tokenize_windowed_contract as canonical
print('Shim works:', tokenize_windowed_contract is canonical)
print('Tokenizer model:', TOKENIZER_MODEL)
"
# Expected: Shim works: True, Tokenizer model: microsoft/graphcodebert-base
```

---

### Phase 3: Full test suite (10 min)

**3.1 Run data_module full test suite**
```bash
cd /home/motafeq/projects/sentinel
ml/.venv/bin/python -m pytest data_module/tests/ -q --tb=line 2>&1 | tail -5
# Expected: 598 passed, 27 skipped, 0 failed (no change from baseline)
```

**3.2 Run ml/ full test suite**
```bash
cd /home/motafeq/projects/sentinel
ml/.venv/bin/python -m pytest ml/tests/ -q --tb=line 2>&1 | tail -5
# Expected: ~38 passed, 0 failed (no change from baseline)
```

**3.3 Run the byte-identical regression test specifically (Gate 1 of v2-readiness)**
```bash
cd /home/motafeq/projects/sentinel
ml/.venv/bin/python -m pytest data_module/tests/test_representation/test_byte_identical_regression.py -q --tb=long 2>&1 | tail -10
# Expected: 40/40 pass — proves graph schema + extraction produces identical output
# CRITICAL: this is the test that proves the move didn't change extraction behavior
```

**3.4 If any test fails, STOP and revert (see §8)**

---

### Phase 4: Verify Run 12 is still training (1 min)

**4.1 Check process is alive**
```bash
ps -p 230342 -o pid,etime,cmd
# Expected: still alive, elapsed time growing
```

**4.2 Check the launch log shows continued training**
```bash
tail -10 /home/motafeq/projects/sentinel/ml/logs/run12_launch_2026-06-13.log
# Expected: training step output, no ImportError or crash
```

**4.3 Verify the structured log directory still being written**
```bash
ls -la /home/motafeq/projects/sentinel/ml/logs/GCB-P1-Run12-v3dospatched-20260613/
# Expected: alerts.jsonl, epoch_summary.jsonl, step_metrics.jsonl, all with recent mtime
```

**4.4 If Run 12 died, STOP and revert (see §8)**

---

### Phase 5: Update documentation (5 min)

**5.1 Update `data_module/docs/architecture.md`**
- The current header says "Stage 7B seam swap complete" but doesn't reflect that graph_extractor + tokenizer were inverted
- Add a note in the "What changed since this doc" or similar section:
  > "**2026-06-13 seam-swap completion:** graph_extractor.py (2,056 lines) and tokenizer.py (175 lines) moved to `data_module/sentinel_data/representation/` as canonical implementations. `ml/src/preprocessing/graph_extractor.py` and `ml/src/data_extraction/windowed_tokenizer.py` are now thin re-export shims (mirror the existing graph_schema.py shim pattern)."

**5.2 Add an entry to the v2 build timeline**
- Add a sub-bullet under "Stage 7B ✅ COMPLETE":
  > "**2026-06-13:** Seam swap completion — graph_extractor + windowed_tokenizer moved to data_module canonical. data_module is now independently usable."

**5.3 Update `~/.claude/projects/.../memory/MEMORY.md`**
- Add a note in the "Sentinel v2 Data Module Build" section
- Add to "Code Changes Made" table for the next commit

---

### Phase 6: Commit (5 min)

**6.1 Verify git status shows expected changes**
```bash
cd /home/motafeq/projects/sentinel
git status
# Expected:
# - data_module/sentinel_data/representation/graph_extractor.py (modified, 77 → 2056 lines)
# - data_module/sentinel_data/representation/tokenizer.py (modified, 72 → 175 lines)
# - ml/src/preprocessing/graph_extractor.py (modified, 2056 → 22 lines)
# - ml/src/data_extraction/windowed_tokenizer.py (modified, 175 → 22 lines)
# - data_module/docs/architecture.md (modified)
# - (this plan file is NOT committed — it's in temp/)
```

**6.2 Stage and commit**
```bash
cd /home/motafeq/projects/sentinel
git add data_module/sentinel_data/representation/graph_extractor.py
git add data_module/sentinel_data/representation/tokenizer.py
git add ml/src/preprocessing/graph_extractor.py
git add ml/src/data_extraction/windowed_tokenizer.py
git add data_module/docs/architecture.md
git commit -F <<'EOF'
refactor(stage7): complete seam swap — graph_extractor + tokenizer to data_module

Stage 7B seam swap was applied in a partial state: graph_schema.py was
flipped to data_module canonical, but graph_extractor.py (2,056 lines) and
windowed_tokenizer.py (175 lines) remained as the real implementation in
ml/src/ with thin adapters in data_module/. This made sentinel-data NOT
independently usable as a package (required ml/ on PYTHONPATH).

This commit completes the seam swap by:

  - Moving ml/src/preprocessing/graph_extractor.py (2,056 lines, REAL) →
    data_module/sentinel_data/representation/graph_extractor.py (REAL)
  - Replacing ml/src/preprocessing/graph_extractor.py with a 22-line
    thin re-export shim (mirror the existing graph_schema.py shim)
  - Moving ml/src/data_extraction/windowed_tokenizer.py (175 lines, REAL) →
    data_module/sentinel_data/representation/tokenizer.py (REAL)
  - Replacing ml/src/data_extraction/windowed_tokenizer.py with a 22-line
    thin re-export shim
  - Updating data_module/docs/architecture.md with the completion note

ZERO importer changes needed: all ~30 existing importers (ml/src/models/*,
ml/src/inference/*, ml/scripts/interpretability/*, ml/scripts/_legacy_data_pipeline/*,
ml/scripts/archive/*, ml/tests/*) continue to work via the shims. Their
imports `from ml.src.preprocessing.graph_extractor import ...` resolve
through the shim to the new canonical location in data_module.

VERIFIED:
  - data_module tests: 598 passed, 27 skipped, 0 failed (no change)
  - ml tests: ~38 passed, 0 failed (no change)
  - test_byte_identical_regression.py: 40/40 pass (proves extraction
    behavior unchanged after the move)
  - test_thin_adapter.py: passes (proves the new shim preserves all
    public symbols)
  - Run 12 training process (PID 230342): still alive, no interruption

Run 12 impact: ZERO. Training uses v3 export (already-built .pt shards),
not live extraction. The seam swap touches files that training doesn't
import at runtime.

Plan: data_module/temp/live_plans/seam_swap_completion_2026-06-13.md
Co-Authored-By: Claude <noreply@anthropic.com>
EOF
```

**6.3 Push**
```bash
cd /home/motafeq/projects/sentinel
git push origin main
```

---

## 6. Risk inventory

### 🔴 R1: Import circular dependency (HIGH severity, LOW probability)

**Risk:** If `data_module/sentinel_data/representation/graph_extractor.py` ever imports from `ml/`, we have a circular import (data_module should not depend on ml/).

**Mitigation:**
- The 2,056-line file's current imports are all stdlib + torch + torch_geometric. None from `ml/`.
- Verify before commit: `grep -E '^(from|import) ml' data_module/sentinel_data/representation/graph_extractor.py` should return empty.
- Same check for tokenizer.py.

### 🔴 R2: Run 12 dies (HIGH severity, LOW probability)

**Risk:** Even though the seam swap doesn't touch training files, if there's an import-time side effect that breaks the training process, Run 12 could die.

**Mitigation:**
- Phase 0.1 checks Run 12 is alive before starting
- Phase 4 verifies Run 12 is still alive after the change
- The shim approach means importers get the same objects (no behavior change)
- If Run 12 dies, immediately revert (see §8)

### 🟡 R3: test_byte_identical_regression fails (MEDIUM severity, VERY LOW probability)

**Risk:** If somehow the move changes extraction behavior, this test would fail.

**Mitigation:**
- The cp command preserves bytes exactly (diff -q should return empty)
- If test fails, the file is byte-identical to before, so the failure is in the test fixtures, not our change
- Revert and investigate

### 🟡 R4: Thin adapter circular import via lazy `__getattr__` (MEDIUM severity, LOW probability)

**Risk:** The current data_module/ adapters use `__getattr__` for lazy import. If the shim approach has issues with lazy loading, the adapters' `__getattr__` could fail.

**Mitigation:**
- We're not using the adapter approach anymore — we're putting the REAL impl in data_module
- The shim in ml/src/ uses `from sentinel_data.representation.graph_extractor import ...` (eager, at module load)
- If shim import fails, importers will fail at import time, not lazily → fast feedback

### 🟢 R5: Test count changes (LOW severity, EXPECTED)

**Risk:** Maybe 1-2 tests in the 598/27/0 will change count due to test_thin_adapter.py now testing the new shim differently.

**Mitigation:**
- The shim's behavior is identical to the adapter's behavior (same objects)
- If tests fail, the difference is in the test itself, not our change
- Acceptable to have 597 passed or 599 passed (within ±2)

### 🟢 R6: Long-line / PEP8 issues in the new shim (LOW severity)

**Risk:** The shim needs to match the existing graph_schema.py shim style.

**Mitigation:**
- Use the exact same template as graph_schema.py
- Use `noqa: F401` to silence unused import warnings

### 🟢 R7: The 36-issue pre-Run-8 audit tests reference specific line numbers (LOW severity)

**Risk:** `test_13_issue_preservation.py` and similar tests may reference line numbers in `ml/src/preprocessing/graph_extractor.py` (e.g., `:1656-1670` for EMITS edge). Moving the file changes the location of these lines.

**Mitigation:**
- The test file `ml/tests/test_preprocessing.py` may reference lines — but the test runs against the IMPORTS (which work via shim), not the file location
- The data_module's `test_13_issue_preservation.py` tests the graph_extractor via `sentinel_data.representation.graph_extractor` (already, per the shim approach)
- After the move, both test files test the SAME function (canonical in data_module)
- Line number references in test docstrings are documentation, not assertions

---

## 7. Verification protocol

After each phase, verify:

| Check | Command | Expected result |
|---|---|---|
| Phase 0.1 | `ps -p 230342 -o pid,etime` | Process alive, time growing |
| Phase 0.2 | `pytest data_module/tests/ ml/tests/ -q` | 598+38 passed, 27+0 skipped, 0 failed |
| Phase 1.1 | `wc -l data_module/.../graph_extractor.py` | 2056 |
| Phase 1.2 | `diff -q ml/src/...graph_extractor.py data_module/...graph_extractor.py` | No output (identical) |
| Phase 1.4 | Smoke test shim | `Shim works: True` |
| Phase 2.1 | `wc -l data_module/.../tokenizer.py` | 175 |
| Phase 2.4 | Smoke test shim | `Shim works: True, Tokenizer model: microsoft/graphcodebert-base` |
| Phase 3.1 | `pytest data_module/tests/ -q` | 598 passed, 27 skipped, 0 failed |
| Phase 3.2 | `pytest ml/tests/ -q` | ~38 passed, 0 failed |
| Phase 3.3 | `pytest test_byte_identical_regression.py -q` | 40/40 pass |
| Phase 4.1 | `ps -p 230342 -o pid,etime` | Process alive, time growing |
| Phase 4.2 | `tail -10 ml/logs/run12_launch_*.log` | Training step output, no errors |
| Phase 6.1 | `git status` | Only expected files modified |
| Phase 6.3 | `git push origin main` | Success |

---

## 8. Rollback strategy

If any verification step fails, rollback is straightforward because we're only moving files (no logic changes).

### Phase 1 rollback (graph_extractor)

```bash
cd /home/motafeq/projects/sentinel
# 1. Restore the original ml/src/ file (we can use git)
git checkout HEAD -- ml/src/preprocessing/graph_extractor.py
# 2. Restore the data_module/ file to the original 77-line adapter
git checkout HEAD -- data_module/sentinel_data/representation/graph_extractor.py
# 3. Verify tests pass
ml/.venv/bin/python -m pytest data_module/tests/ ml/tests/ -q --tb=line 2>&1 | tail -3
# 4. Verify Run 12 is still alive
ps -p 230342 -o pid,etime
```

### Phase 2 rollback (windowed_tokenizer)

Same as Phase 1 rollback, but for the tokenizer files.

### Phase 5/6 rollback (docs / commit)

```bash
# If commit was made but is bad:
cd /home/motafeq/projects/sentinel
git revert HEAD  # creates a new commit that undoes the seam swap
git push origin main
# Or, if not yet pushed:
git reset --soft HEAD~1
git reset HEAD <bad files>
```

### Full rollback (everything)

```bash
cd /home/motafeq/projects/sentinel
git reset --hard HEAD~1  # reverts the seam swap commit
git push --force-with-lease  # only if already pushed; safer to use git revert
```

---

## 9. Run 12 interaction analysis

### 9.1 Does training use graph_extractor at runtime?

**NO.** The training pipeline is:
```
trainer.py (ml/src/training/trainer.py)
  → SentinelDataset (ml/src/datasets/sentinel_dataset.py)
    → loads from ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt (or the v3 export)
      → reads .pt shards from data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/graphs/
```

The training reads **already-extracted** graphs from disk. It never invokes `extract_contract_graph()` or `tokenize_windowed_contract()`.

### 9.2 What does the trainer import at startup?

The trainer imports:
- `from ml.src.models.sentinel_model import SentinelModel` (model arch)
- `from ml.src.models.gnn_encoder import GNNEncoder` (model arch)
- `from ml.src.training.losses` (loss functions)
- `from ml.src.training.training_logger import StructuredLogger` (logging)
- `from ml.src.datasets.sentinel_dataset import SentinelDataset` (data loader)

**None of these import graph_extractor or windowed_tokenizer at module load time.** Some of them (like the model files) import `graph_schema` for constants, but that's a separate file (already correctly a shim).

### 9.3 What could go wrong?

The ONLY way Run 12 could be affected is if:
1. Some module the trainer imports triggers a transitive import of `ml.src.preprocessing.graph_extractor` or `ml.src.data_extraction.windowed_tokenizer`
2. AND that transitive import fails (e.g., circular import, syntax error in shim)

**Verification:** Run the import test BEFORE doing the actual move. If it works, the move is safe.

```bash
cd /home/motafeq/projects/sentinel
PYTHONPATH=. ml/.venv/bin/python -c "
# Test all the imports the trainer does (and their transitive deps)
import ml.src.training.trainer
import ml.src.datasets.sentinel_dataset
import ml.src.models.sentinel_model
import ml.src.models.gnn_encoder
import ml.src.training.training_logger
import ml.src.inference.predictor
import ml.src.inference.cache
import ml.src.models.transformer_encoder
import ml.src.models.fusion_layer
# Plus the seam swap targets (these are the ones we're moving)
import ml.src.preprocessing.graph_extractor
import ml.src.data_extraction.windowed_tokenizer
import ml.src.preprocessing.graph_schema
# Test that all symbols resolve
from ml.src.preprocessing.graph_extractor import extract_contract_graph, GraphExtractionConfig, GraphExtractionError
from ml.src.data_extraction.windowed_tokenizer import tokenize_windowed_contract, init_worker
print('All imports OK')
"
```

If this prints "All imports OK" before the move, it will print the same after the move (because the shims preserve the API).

### 9.4 Worst case scenario

If Run 12 dies:
- PID 230342 → process gone
- The launch log will show an ImportError or other exception
- The structured log directory will stop getting new JSONL rows

**Recovery:** Revert (see §8), restart Run 12 from the ep1 checkpoint (`ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt` will exist after ep1 completes; if ep1 is still in progress, the model state is lost — we'd need to restart from scratch).

---

## 10. Timeline & dependencies

### 10.1 Estimated duration

| Phase | Duration | Notes |
|---|---|---|
| 0. Pre-flight | 5 min | Verify Run 12 alive, baseline tests pass |
| 1. Move graph_extractor | 15 min | Copy + write shim + smoke test |
| 2. Move windowed_tokenizer | 10 min | Same as Phase 1 |
| 3. Full test suite | 10 min | Run both test suites |
| 4. Verify Run 12 | 1 min | Check process still alive |
| 5. Update docs | 5 min | Update architecture.md |
| 6. Commit + push | 5 min | One focused commit |
| **Total** | **~50 min** | |

### 10.2 When to execute

**Recommended timing:** NOW, before Run 12 completes its first epoch (so we have a checkpoint to fall back to if needed).

**Risk window:** Run 12 is currently in torch.compile tracing phase (30:46 elapsed, expected ep1 at ~25-30 min — so ep1 may be done or close to done by now).

**Best timing:** If Run 12 just completed ep1 (checkpoint saved), do the seam swap. The ep1 checkpoint gives us a recovery point.

**Worst timing:** If Run 12 crashes mid-epoch-2 (like Run 11 did), the recovery is harder (no checkpoint to fall back to).

### 10.3 Sequential dependencies

The phases are SEQUENTIAL — each depends on the previous:
- Phase 1 depends on Phase 0 (verified baseline + Run 12 alive)
- Phase 2 depends on Phase 1 (graph_extractor moved first, builds confidence)
- Phase 3 depends on Phases 1+2 (both files moved, then test)
- Phase 4 depends on Phase 3 (tests pass, then verify Run 12)
- Phase 5 depends on Phase 4 (Run 12 still alive, then update docs)
- Phase 6 depends on Phase 5 (docs updated, then commit + push)

If any phase fails, STOP and rollback (don't proceed).

---

## 11. Open questions & deferred work

### 11.1 Open questions for Ali (need decisions before executing)

**Q1: Do we want to delete the legacy `ml/src/data_extraction/tokenizer.py` (codebert-base, single-window) as part of this plan?**
- Pro: removes dead code, the docstring says "kept for v1 batch-script use only"
- Con: low priority, no functional impact
- **My recommendation:** No, defer to a separate cleanup. Don't mix with the seam swap.

**Q2: Do we want to also flip the seam swap in `ml/src/inference/preprocess.py` (625 lines)?**
- The inference preprocessor has its own slither orchestration + temp file management
- It's a wrapper, not a duplicate of graph_extractor
- Flipping it would require changing its import paths (but no logic change)
- **My recommendation:** Defer to a future plan. This plan focuses on graph_extractor + tokenizer (the clear duplicates).

**Q3: Should we keep the `data_module/.dvcignore` and `data_module/.gitignore` (data/**) rules as-is?**
- These ensure the 7 GB of data doesn't get committed
- The seam swap doesn't change this
- **My recommendation:** Keep as-is. No changes needed.

### 11.2 Deferred work (not part of this plan)

- **Stage 5.5 GCB propagation** (deferred post-Run 12) — see pre-run12-fixes Item 2
- **DeFiHackLabs extraction** (BLOCKED, PoC contracts) — see pre-run12-fixes Item 5
- **Data source additions** (CGT, HF audit-firms, Kaggle) — see `data-source-addition-plan-2026-06-13.md`
- **Stage 5 splitting + registry** (deferred August) — see pre-run12-fixes Item 6
- **`ml/src/data_extraction/tokenizer.py` cleanup** (legacy codebert-base)
- **`ml/src/inference/preprocess.py` shim** (its own follow-up plan)
- **`ml/_archive/seam_swap_pre_2026-06-12/` review** (historical artifacts, may need cleanup later)
- **Empty `ml/src/tools/` and `ml/src/validation/` directories** (legacy, may need cleanup)

---

## 12. References

### Pre-existing related plans (now archived)

- `data_module/temp/live_plans/archive/stage_7a_export_module.md` — Stage 7A work (completed 2026-06-12)
- `data_module/temp/live_plans/archive/stage_7b_seam_swap.md` — Stage 7B original plan (PRESCRIBED the correct move that this plan completes)
- `data_module/temp/live_plans/archive/stage_7b_seam_swap_active.md` — Stage 7B implementation log (documents the partial implementation)

### Architecture / decision docs

- `data_module/docs/architecture.md` — canonical architecture doc, §"v2 → v3 transition" + §"Appendix A" (will be updated in Phase 5)
- `docs/decisions/ADR-0008-export-and-seam-swap-design.md` — seam swap design + 7B Amendment (11 subsections)
- `docs/decisions/ADR-0009-canonical-class-vocabulary.md` — LABELING order

### Memory

- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` — current state (193 lines)
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_run12_launch.md` — Run 12 launch context
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_dos_patch_2026-06-13.md` — DoS patch audit (the previous task that revealed this inconsistency)
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_stage7b_handoff.md` — Stage 7B handoff story

### Data module docs (to be updated in Phase 5)

- `data_module/sentinel_data/representation/graph_extractor.py` (will become 2,056 lines, REAL)
- `data_module/sentinel_data/representation/tokenizer.py` (will become 175 lines, REAL)
- `data_module/sentinel_data/representation/graph_schema.py` (already canonical, 251 lines)

### ML code (to be modified in Phase 1+2)

- `ml/src/preprocessing/graph_extractor.py` (will become 22 lines, shim)
- `ml/src/data_extraction/windowed_tokenizer.py` (will become 22 lines, shim)
- `ml/src/preprocessing/graph_schema.py` (already shim, 22 lines — pattern to mirror)

### Test files (to be run in Phase 3)

- `data_module/tests/test_representation/test_byte_identical_regression.py` — 40/40 tests, Gate 1 of v2-readiness
- `data_module/tests/test_representation/test_thin_adapter.py` — validates shim preserves public API
- `data_module/tests/test_representation/test_13_issue_preservation.py` — 36 pre-Run-8 audit regression tests
- `ml/tests/test_preprocessing.py` — 85+ ml preprocessing tests
- `ml/tests/test_sentinel_dataset.py` — 16 SentinelDataset tests (Gate 3 of v2-readiness)

### Run 12 (currently training)

- `ml/logs/run12_launch_2026-06-13.log` — launch stdout
- `ml/logs/GCB-P1-Run12-v3dospatched-20260613/` — structured log dir
- `ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt` — checkpoint (after ep1)
- `ml/logs/GCB-P1-Run12-v3dospatched-20260613.log` — run's own log

---

## 13. Approval & sign-off

This plan is ready for Ali's review. To approve, Ali should:

1. Read this plan end-to-end
2. Verify the 3 critical claims:
   - Claim: graph_extractor + tokenizer are inverted (verified by `wc -l` and `diff -q`)
   - Claim: ~30 importers depend on ml/src/ paths (verified by grep)
   - Claim: Run 12 doesn't use graph_extractor at runtime (verified by import test in §9.3)
3. Answer the 3 open questions in §11.1
4. Confirm timing preference (Phase 10.2)
5. Say "go" or equivalent

Then execute Phases 0-6 in order, with verification at each step.

---

**Plan last updated:** 2026-06-13 23:55 UTC
**Status:** AWAITING APPROVAL
**Next step:** Ali's sign-off, then Phase 0 (5 min pre-flight)
