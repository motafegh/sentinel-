# D — Smoke Tests and Pre-Flight Gates

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.
>
> **Last revised: 2026-06-14** (post-Run-12 launch). Fixed broken `M_session_handoff.md` reference (was a typo for `L.5`). No path updates needed — `ml/scripts/smoke/run_all.py` and `ml/scripts/smoke/_common.py` still exist (verified).

---

## When This File Applies

Load this spec before:
- Launching any new training run
- Loading or resuming from a checkpoint
- Making any change to `ml/src/`
- Changing the graph schema (load alongside `J_schema_migration.md`)
- Any evaluation run requiring model inference

---

## D.1 — System Pre-Flight

Before running any smoke fix tests, run the system pre-flight check:

```
python ml/scripts/smoke/run_all.py --preflight
```

The pre-flight checks: Python executable, `torch`, `numpy`, `pandas` imports,
graph directory non-empty (>100 `.pt` files), and at least one checkpoint present.

Read `ml/scripts/smoke/_common.py` to understand:
- What paths it resolves (`GRAPHS_DIR`, `CHECKPOINTS_DIR`, `TOKENS_DIR`, etc.)
- What `EXPECTED_SCHEMA_AFTER_FIX2_3_4` and dim constants it asserts
  (these are the schema constants the smoke suite was written against —
  if the schema has advanced, re-read `_common.py` before interpreting
  any schema-related smoke failure)

If pre-flight fails, do not proceed to per-fix tests or training.
Document the failure as a gate assertion (Rule 2, Layer 1).

---

## D.2 — Per-Fix Smoke Suite

Run the full suite:

```
python ml/scripts/smoke/run_all.py
```

The suite runs in phase order — read `run_all.py` for the phase structure
before interpreting results. Phase order matters: Phase 2 fixes run sequentially
because each one invalidates cache. Do not reorder.

**Phases (read `run_all.py` for current definitions):**

| Phase | Fixes | Focus |
|---|---|---|
| 0 | Fix #1 | Regression check (already-applied fix — permanent gate) |
| 1 | Fix #6, #7, #8 | Display, benchmark, docs (no model change) |
| 2 | Fix #2, #3, #4 | Schema bump v8→v9 (sequential; each invalidates cache) |
| 3 | Fix #5 | Slither-derived label relabelling |

**Before interpreting any fix failure:**
1. Read the failing `smoke_fix<N>.py` file header — it names the specific
   historical bug it guards against
2. Confirm the failure is not a stale `_common.py` schema constant vs.
   a legitimately advanced schema — these are different problems
3. A `SKIP` result means the script file was missing, not that the fix passed

**Partial runs (when appropriate):**
- To run a single fix: `python ml/scripts/smoke/run_all.py --fix <N>`
- To run one phase: `python ml/scripts/smoke/run_all.py --phase <0|1|2|3>`
- To skip Fix #1 regression check: `--no-fix1` (document the reason)

All results are emitted to stderr. Exit code 0 = all passed. Non-zero = at
least one failure. Write a gate assertion for the overall result.

---

## D.3 — VRAM Pre-Flight Gate

Required before any training run or full-dataset inference run.

```
python ml/scripts/vram_gate_test.py
```

Read `vram_gate_test.py` before running to understand:
- What VRAM threshold it checks against (do not assume the threshold is static—
  it may reflect the active batch size and model config)
- What it reports: available VRAM, required VRAM, headroom

If the gate fails:
- Document the available vs. required VRAM in the run pre-flight log
- Do not proceed with the training run
- Record as a blocker in the session handoff (see `L_release_readiness.md` §L.5)

Write a gate assertion: `vram_gate: PASS | FAIL — <available>GB available,
<required>GB required, <headroom>GB headroom`

---

## D.4 — Compile Validation (after checkpoint load)

Required after loading any checkpoint into a `torch.compile`-enabled model.

```
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/compile_smoke_test.py
```

The script runs three checks:
1. `compile` — `torch.compile` succeeds on all targeted submodules
2. `numerical` — compiled output is numerically close to eager output
   (tolerance is defined inside the script — read it, do not assume)
3. `stability` — 2-epoch training loop with variable batch shapes completes
   without compile errors or excessive NaN loss steps

Read `compile_smoke_test.py` before running to understand:
- Which submodules it compiles (must match what `trainer.py` compiles)
- The `._orig_mod.` infix issue: compiled checkpoints prefix weights with
  `._orig_mod.` — confirm the load path in `trainer.py` strips this before
  weight access, or `compile_smoke_test.py` will surface the mismatch
- The `--device cpu` flag is available for environments without CUDA

If Check 1 (`compile`) fails, the script aborts and skips checks 2 and 3.
A compile failure means: set `use_compile=False` in `TrainConfig` and
file a fix item before proceeding.

Write a gate assertion for each of the three checks separately.

---

## D.5 — Completion Attestation

After completing this section, append to the relevant run pre-flight doc:

```
## Procedure Attestation — D_smoke_preflight — <ISO date>
Steps completed:   D.1 preflight: PASS/FAIL/UNVERIFIED
                   D.2 smoke suite: PASS/FAIL/UNVERIFIED (<N>/8 fixes passed)
                   D.3 vram gate: PASS/FAIL/UNVERIFIED
                   D.4 compile validation: compile PASS/FAIL, numerical PASS/FAIL, stability PASS/FAIL
Steps skipped:     [any skipped steps + explicit reason]
Unverified items:  [anything not confirmable from tool output]
New findings:      [link to audit doc entry, or "none"]
Written to:        [path of this attestation]
```

Do not proceed to training launch (`F_new_run_checklist.md`) until this
attestation is written and all required gates show PASS.
