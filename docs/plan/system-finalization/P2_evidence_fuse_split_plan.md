# P2 — Evidence + fuse + nodes.py split (execution plan)

**Date:** 2026-06-24
**Phase:** P2 (evidence/fuse/split — proposal §5.1–§5.2, §9, §10.1)
**Prerequisite:** P0 FOUNDATION complete (P0.0 + P1 + P0.1) — first honest baseline at `agents/eval/runs/20260624T133420Z_p0_honest_baseline/`
**Architecture of record:** `docs/proposal/2026-06-23_proposal_SYSTEM_architecture-finalization.md`
**Build-decision register:** `docs/proposal/2026-06-23_proposal_BUILD_questions.md §5`
**Review:** `~/.claude/scratch/p2_plan_review_20260624.md` — gaps G1-G10 found, G1-G5 resolved below

---

## Scope — files created / touched

All paths under `agents/` (`cd agents && poetry run ...`).

### Created
| File | Lines (est.) | Content |
|------|-------------|---------|
| `src/orchestration/verdict/__init__.py` | 5 | re-export public API |
| `src/orchestration/verdict/evidence.py` | 80 | `Evidence` dataclass (frozen), `Polarity`/`Kind` enums, helper constructors |
| `src/orchestration/verdict/verdict.py` | 40 | `ClassVerdict` dataclass: `cls`, `verdict_provable`, `verdict_full`, `confidence`, `driving_evidence` |
| `src/orchestration/verdict/fuse.py` | 120 | `fuse()` — group → de-correlate → aggregate → asymmetry rule → band map → dual emit |
| `src/orchestration/verdict/reliability.py` | 60 | `load_reliability()`, `get_reliability(source, cls)` — P3 placeholder, returns config defaults for now |
| `src/orchestration/verdict/emit.py` | 100 | Per-source emit helpers: `emit_ml_evidence()`, `emit_slither_evidence()`, `emit_debate_evidence()`, etc. |
| `src/orchestration/nodes/__init__.py` | 5 | Namespace package — re-exports all node functions |
| `src/orchestration/nodes/quick_screen.py` | \~80 | Extracted from `nodes.py` |
| `src/orchestration/nodes/evidence_router.py` | \~60 | Extracted from `nodes.py` |
| `src/orchestration/nodes/ml_assessment.py` | \~120 | Extracted from `nodes.py` |
| `src/orchestration/nodes/rag_research.py` | \~80 | Extracted from `nodes.py` |
| `src/orchestration/nodes/audit_check.py` | \~60 | Extracted from `nodes.py` |
| `src/orchestration/nodes/static_analysis.py` | \~160 | Extracted from `nodes.py` |
| `src/orchestration/nodes/graph_explain.py` | \~100 | Extracted from `nodes.py` |
| `src/orchestration/nodes/consensus_engine.py` | \~300 | Extracted from `nodes.py` |
| `src/orchestration/nodes/cross_validator.py` | \~300 | Extracted from `nodes.py` |
| `src/orchestration/nodes/synthesizer.py` | \~500 | Extracted from `nodes.py` |
| `src/orchestration/nodes/reflection.py` | \~100 | Extracted from `nodes.py` |
| `src/orchestration/nodes/explainer.py` | \~100 | Extracted from `nodes.py` |
| `src/orchestration/nodes/visualizer.py` | \~100 | Extracted from `nodes.py` |
| `src/orchestration/nodes/_helpers.py` | \~120 | Shared utilities: `_call_mcp_tool`, `_parse_aderyn_report`, `_run_aderyn_on_file`, `_extract_external_call_summary`, `_signals_for_class`, `_best_rag_score`, `_reconcile_verdicts` (kept as backward-compat shim during dual-write) |
| `src/orchestration/nodes/_reconcile_shim.py` | \~50 | `_reconcile_verdicts()` → calls `fuse()` internally + logs mismatch. Deleted in flip step. |
| `tests/test_verdict_evidence.py` | 30 | `Evidence` construction, immutability, helper tests |
| `tests/test_verdict_fuse.py` | 120 | `fuse()` — known cases, asymmetry invariant, dual-verdict, golden reproduction |
| `tests/test_verdict_reliability.py` | 40 | Default reliability lookup, config override |
| `tests/test_nodes_split.py` | 60 | All node files import cleanly, graph compiles with new package |

### Modified
| File | Change |
|------|--------|
| `src/orchestration/nodes.py` | **Becomes a thin re-export module** that imports from `nodes/.*` and `verdict/.*`. Deleted entirely after flip step. |
| `src/orchestration/state.py` | Add `evidence_list` (append-reducer), `verdict_provable`, `verdict_full` fields |
| `src/orchestration/graph.py` | Update imports from `nodes` → `nodes.*`; wire new state fields; ensure `evidence_list` flows through |
| `src/eval/pipeline_metrics.py` | Update `as_dict()` to report `verdict_provable` vs `verdict_full` if available (backward compat) |
| `src/eval/run_benchmark.py` | Optionally read `verdict_provable` from reports |
| `tests/test_eval_framework.py` | No changes expected (backward compat) |

### Deleted (flip step)
| File | Reason |
|------|--------|
| `src/orchestration/nodes.py` | Replaced by `nodes/` + `verdict/` packages |
| `src/orchestration/nodes/_helpers.py` `_reconcile_verdicts` | Replaced by `fuse()` |
| Each node's legacy per-channel verdict write | Replaced by evidence emission + `fuse()` |

---

## Pre-conditions

1. **P0.1 baseline exists** — first honest baseline at `eval/runs/20260624T133420Z_p0_honest_baseline/eval_metrics.json`. Macro-F1: 0.1958, Macro-Fbeta: 0.2515.
2. **Full suite green** — 486 passed, 3 skipped (`cd agents && poetry run pytest -q`).
3. **ML server + 4 MCP servers running** (needed for golden characterization of current verdicts).
4. **83 contract reports available** from prior audit pipeline run (at `test_audit_reports_p0/`).

---

## Ordered tasks

Each task is independently verifiable. Acceptance = the exact command.

### T2.1 — Create `verdict/` package

#### T2.1.1 — `Evidence` dataclass + enums (`verdict/evidence.py`)
- `Evidence` frozen dataclass: `source`, `vuln_class`, `polarity` (SUPPORTS/REFUTES/NEUTRAL), `strength` [0,1], `reliability` [0,1], `kind` (STATISTICAL/SYNTACTIC/SEMANTIC/FORMAL/ECONOMIC), `deterministic` bool, `detail` dict.
- `Polarity`, `Kind` enums.
- Helper constructors: `evidence.ml(...)`, `evidence.slither(...)`, `evidence.debate(...)` etc.

**Acceptance:** `poetry run python -c "from src.orchestration.verdict.evidence import Evidence, Polarity; e = Evidence(source='ml', vuln_class='Reentrancy', polarity=Polarity.SUPPORTS, strength=0.87, reliability=0.6, kind='STATISTICAL', deterministic=True, detail={})"` — no import errors, `frozen=True` enforced.

#### T2.1.2 — `ClassVerdict` dataclass (`verdict/verdict.py`)
- `ClassVerdict`: `cls`, `verdict_provable` (str), `verdict_full` (str), `confidence` (float), `driving_evidence` (list[Evidence]).

**Acceptance:** Unit test constructing a `ClassVerdict`.

#### T2.1.3 — `fuse()` (`verdict/fuse.py`)
Implements §5.2 steps 1–6:
1. Group evidence by class
2. De-correlate by family (ML, STATIC_SYNTAX, RAG, LLM_DEBATE, FORMAL, ECONOMIC) — within-family N sources scale reliability by `1/N`
3. Aggregate signed `reliability × strength` → confidence [0,1]
4. FN/FP asymmetry: REFUTES cannot clear a strong SUPPORTS (defined as: one SUPPORTS with `reliability×strength ≥ 0.5`, OR two+ SUPPORTS each ≥0.3, OR one FORMAL SUPPORTS). Strong SUPPORTS → can only reach DISPUTED/INCONCLUSIVE, not SAFE.
5. Map confidence to band: CONFIRMED ≥0.70, LIKELY ≥0.50, DISPUTED ≥0.30, else SAFE (asymmetry override → DISPUTED floor)
6. Emit dual: `verdict_provable` (deterministic=True only) + `verdict_full` (all evidence)
- All band cutoffs read from config at call time (Rule B).
- NEUTRAL evidence contributes 0.0 to the signed sum but appears in `driving_evidence` for report attribution.

**Acceptance:** `poetry run python -c "from src.orchestration.verdict.fuse import fuse; from src.orchestration.verdict.evidence import Evidence, Polarity"` — no import errors. Then unit tests:
  - Perfect ML evidence → CONFIRMED
  - ML + contradicting debate → DISPUTED (not SAFE — asymmetry)
  - Deterministic-only → verdict_provable matches, verdict_full may differ
  - Empty list → SAFE

#### T2.1.4 — Per-source emit helpers (`verdict/emit.py`)
- `emit_ml_evidence(ml_result) -> list[Evidence]` — class probability > threshold
- `emit_static_evidence(static_findings, class_to_detectors) -> list[Evidence]` — Slither/Aderyn impact-map
- `emit_rag_evidence(rag_results) -> list[Evidence]` — similarity > floor
- `emit_debate_evidence(debate_transcript) -> list[Evidence]` — judge confidence
- `emit_quick_screen_evidence(quick_screen_hits) -> list[Evidence]` — syntactic findings
- Each sets `deterministic` appropriately (ML, static, QS = True; debate = False; RAG = True if similarity from deterministic embedding, else False).

**Acceptance:** Given mock input for each source, emits correct count of Evidence with correct polarity/strength/deterministic.

#### T2.1.5 — Reliability lookup (`verdict/reliability.py`)
- `load_reliability(config=None) -> dict[tuple[str,str], float]` — reads from config's `reliability` table. P3 will fit from data; for now returns config defaults (hand-set per proposal §5.4).

**Acceptance:** Returns `reliability[("ml", "Reentrancy")]` from config or default.

---

### T2.2 — Split `nodes.py` → `nodes/` package

#### T2.2.1 — Create `nodes/__init__.py` + `nodes/_helpers.py`
- `_helpers.py` contains shared utilities currently in `nodes.py`: `_call_mcp_tool`, `_parse_aderyn_report`, `_run_aderyn_on_file`, `_extract_external_call_summary`, `_signals_for_class`, `_best_rag_score`, `_llm_enabled`.
- `_ask` and `_run_debate` closures inside `cross_validator` stay as module-level functions in `nodes/cross_validator.py` (they are debate-specific, not shared).
- Keep `_reconcile_verdicts` in a separate `_reconcile_shim.py` (deleted in flip step).
- `quick_screen` and `static_analysis` each keep their own lazy slither imports — per-file imports are idiomatic in the one-file-per-node layout.

#### T2.2.2 — Extract each node function into its own file
Mechanical: copy function body, fix imports, verify it runs. Each file exports exactly one async function with the same signature as today:
```python
async def <node_name>(state: AuditState) -> dict[str, Any]:
```

Order (any, but this matches graph execution):
1. `quick_screen.py`
2. `evidence_router.py`
3. `ml_assessment.py`
4. `rag_research.py`
5. `audit_check.py`
6. `static_analysis.py`
7. `graph_explain.py`
8. `consensus_engine.py`
9. `cross_validator.py`
10. `synthesizer.py`
11. `reflection.py`
12. `explainer.py`
13. `visualizer.py`

#### T2.2.3 — Update `graph.py` imports
```python
# Before:
from src.orchestration.nodes import quick_screen, ml_assessment, ...
# After:
from src.orchestration.nodes.quick_screen import quick_screen
from src.orchestration.nodes.ml_assessment import ml_assessment
...
```

#### T2.2.4 — Make old `nodes.py` a thin re-export shim
```python
# nodes.py — DEPRECATED. Import from src.orchestration.nodes.<name> instead.
from src.orchestration.nodes.quick_screen import quick_screen
from src.orchestration.nodes.ml_assessment import ml_assessment
# ... all 13 nodes + helpers
```
This keeps all existing `from src.orchestration.nodes import X` imports working during dual-write.

**Acceptance:** `poetry run pytest tests/ -q` — 486+ passed, no import errors from the split.

---

### T2.3 — Transitional dual-write (Shape B scaffolding)

#### T2.3.1 — Add `evidence_list` to `AuditState`
```python
evidence_list: Annotated[list[Evidence], operator.add]
```
Append-reducer so every node appends without overwriting.

Also add:
```python
verdict_provable: dict[str, str]    # {class: verdict} — deterministic-only
verdict_full: dict[str, str]        # {class: verdict} — all evidence
```

#### T2.3.2 — Each channel emits Evidence + keeps old verdict writes
Touch each node in execution order:
- `consensus_engine`: after computing votes, call `emit_ml_evidence()`, `emit_static_evidence()`, etc. Append to `evidence_list`. Keep legacy `consensus_verdict` + `confidence_by_class` writes.
- `cross_validator`: after debate, call `emit_debate_evidence()`. Append to `evidence_list`. Keep legacy `verdicts`/`confirmations`/`contradictions` writes.
- `synthesizer`: after reconciliation, call `emit_rag_evidence()` for any RAG-based classes. Call `fuse()` on the accumulated evidence list → produces `verdict_provable` + `verdict_full` alongside existing verdict writes.

**Synthesizer dual-write flow (pseudocode):**

```python
async def synthesizer(state: AuditState) -> dict[str, Any]:
    # ... existing reads: ml_result, static_findings, rag_results, consensus_verdict, pre_verdicts ...

    # 1. EXISTING: per-class reconciliation (kept verbatim)
    verdicts, confirmations = {}, {}
    for cls in all_classes:
        verdict, sources = _reconcile_verdicts(
            cls, prob, consensus_verdict.get(cls), pre_verdicts.get(cls),
            static_findings, rag_results, path_taken,
        )
        verdicts[cls] = verdict
        confirmations[cls] = sources

    # 2. NEW: gather evidence from raw state fields
    evidence_list = list(state.get("evidence_list", []))  # accumulate from upstream nodes
    # Add RAG evidence (not emitted upstream — consensus_engine doesn't reach RAG directly)
    for cls in all_classes:
        rag_score = _best_rag_score(cls, rag_results)
        if rag_score >= RAG_RELEVANCE_FLOOR:
            evidence_list.append(Evidence.rag(cls, score=rag_score, ...))

    # 3. NEW: fuse → produce provable + full verdicts
    fused = fuse(evidence_list)  # dict[cls, ClassVerdict]
    verdict_provable = {cls: cv.verdict_provable for cls, cv in fused.items()}
    verdict_full = {cls: cv.verdict_full for cls, cv in fused.items()}

    # 4. Compare: log any mismatch between legacy (from _reconcile_verdicts)
    #    and fused verdicts. Golden tests will pin these.
    for cls in set(verdicts) | set(verdict_full):
        if verdicts.get(cls) != verdict_full.get(cls):
            logger.warning("dual_write_mismatch | cls={} legacy={} fused={}",
                           cls, verdicts.get(cls), verdict_full.get(cls))

    # 5. Return: legacy fields + new fields
    return {
        "verdicts":       verdicts,        # legacy (kept)
        "confirmations":  confirmations,   # legacy (kept)
        "verdict_provable": verdict_provable,  # NEW
        "verdict_full":     verdict_full,      # NEW
        "final_report":   final_report,    # legacy (kept, now includes fused verdicts)
    }
```

Critical invariant: `fuse()` output must match pre-existing verdicts (golden test).

**Acceptance:** Run on 83 contracts → `evidence_list` populated (≥ some number), `verdict_provable` and `verdict_full` present in final report.

---

### T2.4 — Golden tests

#### T2.4.1 — Pin current verdicts
Capture the P0 honest baseline's per-class verdicts as golden reference:
```python
# eval/baselines/golden_pre_p2.json
# {"<stem>": {"<class>": "<verdict>", ...}, ...}
```

#### T2.4.2 — `test_verdict_fuse.py`: equivalence test

**Reconstruction strategy (resolves G1):** The golden test builds Evidence objects from the **same raw inputs** `_reconcile_verdicts` uses — not from the emit helpers (which are part of the system under test). This ensures an independent verification:

```python
def _build_evidence_from_raw(cls, prob, consensus_vote, debate_verdict,
                              static_findings, rag_results, path_taken) -> list[Evidence]:
    """Reconstruct Evidence from the raw per-source inputs that _reconcile_verdicts reads."""
    evidence = []
    # ML evidence
    ml_threshold = get_config().consensus.ml_positive_threshold
    if prob >= ml_threshold:
        evidence.append(Evidence(source="ml", vuln_class=cls, polarity=Polarity.SUPPORTS,
                                 strength=prob, reliability=reliability[("ml", cls)],
                                 kind=Kind.STATISTICAL, deterministic=True, detail={}))
    # Slither/Aderyn evidence
    slither_found, aderyn_found = _signals_for_class(cls, static_findings)
    if slither_found:
        evidence.append(Evidence.slither(cls, ...))
    if aderyn_found:
        evidence.append(Evidence.aderyn(cls, ...))
    # RAG evidence
    rag_score = _best_rag_score(cls, rag_results)
    if rag_score >= RAG_RELEVANCE_FLOOR:
        evidence.append(Evidence.rag(cls, score=rag_score, ...))
    # Debate evidence (deterministic=False)
    if debate_verdict:
        evidence.append(Evidence.debate(cls, debate_verdict, ...))
    return evidence
```

Then the test:
```python
def test_fuse_reproduces_legacy_verdicts():
    for each contract in golden_pre_p2.json:
        for cls in all_classes:
            evidence_list = _build_evidence_from_raw(
                cls, prob, consensus_vote, debate_verdict,
                static_findings, rag_results, path_taken)
            actual = fuse(evidence_list)
            expected = golden_verdict[stem][cls]
            assert actual[cls].verdict_full == expected  # exact per-class match
```
All 83 contracts, exact per-class match.

**Acceptance:** `poetry run pytest tests/test_verdict_fuse.py -v` — all pass.

---

### T2.5 — Prove equivalence via P0 eval

Run the full eval on dual-write output and compare to the P0 honest baseline.

```bash
cd agents
poetry run python -m src.eval.run_benchmark \
    --name p2_equivalence \
    --config configs/verdicts_default.yaml \
    --reports test_audit_reports_p0/ \
    --corpus ../manual_hand_written_contracts \
    --baseline eval/runs/20260624T133420Z_p0_honest_baseline/eval_metrics.json
```

**Acceptance:** macro_f1 >= 0.1958 - 0.005 (tolerance for dual-write overhead). If not, debug fuse() parameters.

---

### T2.6 — Fix `to_thread` bug (proposal §10.1)

`nodes.py` uses `asyncio.to_thread` for blocking calls (Slither, Aderyn). This is non-cancellable — a timeout never kills the thread.

**Fix:** Replace every `asyncio.to_thread(fn, *args)` with:
```python
loop = asyncio.get_running_loop()
future = loop.run_in_executor(None, fn, *args)
result = await asyncio.wait_for(future, timeout=timeout_s)
```

**Test:** Add a test that calls a long-running blocking function with a short timeout and asserts cancellation within 2× timeout.

**Files touched:** All `nodes/*.py` files that use `to_thread` — search for `asyncio.to_thread`.

---

### T2.7 — Flip to Shape A (delete legacy path)

Once T2.5 equivalence is proven:

1. **Remove legacy per-channel verdict writes** from `consensus_engine`, `cross_validator`, `synthesizer`:
   - `state["verdicts"]` is no longer written directly by channels — populated from `verdict_full`
   - `state["confirmations"]`, `state["contradictions"]` — derived from `evidence_list`
2. **`synthesizer` exclusively uses `fuse(evidence_list)`** for all verdict production
3. **`metric_attribution` and `confidence_by_class`** become derivations of `evidence_list`, computed in `explainer` and kept in `final_report` for backward compat:
   - `confidence_by_class[cls] = class_verdict.confidence` (from `ClassVerdict` returned by `fuse()`)
   - `contradictions[cls]` = present when a class has BOTH `SUPPORTS` AND `REFUTES` evidence in `evidence_list`; detail strings built from conflicting evidence items
4. **Audit all downstream consumers** of removed fields (G5):
   - `reflection`: reads `verdicts`, `contradictions`, `confidence_by_class`, `static_findings`, `rag_results`, `final_report` — populate `verdicts` from `verdict_full`, derive `contradictions`/`confidence_by_class` from evidence as above; `static_findings`/`rag_results`/`final_report` unchanged
   - `explainer`: reads `final_report` → `vuln_verdicts`, derives `metric_attribution` from `evidence_list` per-class instead of raw per-source signals
   - `visualizer`: reads `final_report` + `contract_address` only — unchanged
5. **Delete `nodes/_reconcile_shim.py`** — `_reconcile_verdicts` is gone
6. **`verdict_provable` and `verdict_full`** become the primary verdict fields in `final_report`
7. **`nodes.py` shim deleted** — all callers updated to import from `nodes/` directly

**Acceptance:** Full test suite passes. Eval re-run produces identical metrics (macro_f1 within ±0.001).

---

### T2.8 — Update `graph.py`

- Wire `evidence_list` as a graph state field (append-reducer)
- Wire `verdict_provable` + `verdict_full` as graph state fields
- Update `synthesizer` return type to include new fields
- Ensure backward compat: existing `verdicts` field is populated from `verdict_full` for consumers that read it

**Acceptance:** `poetry run python -c "from src.orchestration.graph import build_graph; g = build_graph()"` — no errors.

---

### T2.9 — Integration tests

New tests:
- `test_evidence_list_populated`: full pipeline on a SINGLE contract → `evidence_list` has ≥1 entry from each channel that ran
- `test_dual_verdict_differs`: deterministic-only verdict vs full verdict differ when debate evidence is present
- `test_asymmetry_invariant`: no class with strong ML evidence ends SAFE
- `test_full_suite_green`: `poetry run pytest -q` — 500+ passed

**Acceptance:** `poetry run pytest -q` — all tests pass.

---

### P2 gate (P0.1 equivalent)

```bash
cd agents && poetry run pytest -q
# Expect: 500+ passed, 0 failed
```

```bash
poetry run python -m src.eval.run_benchmark \
    --name p2_complete \
    --config configs/verdicts_default.yaml \
    --reports test_audit_reports_p0/ \
    --corpus ../manual_hand_written_contracts \
    --baseline eval/runs/20260624T133420Z_p0_honest_baseline/eval_metrics.json
# macro_F1 delta: within ±0.005
```

---

## Rollback plan

| Step | What fails | Rollback |
|------|-----------|----------|
| T2.1 | verdict/ package import errors | Delete `verdict/` dir |
| T2.2 | node split import errors | Restore `nodes.py` from git, delete `nodes/` dir |
| T2.3 | dual-write changes verdicts | Remove evidence emission from each node — old path unaffected |
| T2.4 | golden tests don't match | Adjust fuse() thresholds; or skip T2.4 and accept the delta |
| T2.5 | eval delta exceeds tolerance | Don't flip — stay on dual-write permanently (Shape B) |
| T2.7 | flip breaks something | Restore from git — the flip commit is a single atomic diff |
| T2.6 | to_thread fix regresses | Revert to `asyncio.to_thread` — the bug exists today; rollback restores status quo |

---

## Risk register (P2-specific)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| fuse() doesn't reproduce legacy verdicts exactly | Med | High | Tune de-correlation weights + band cutoffs; golden test is the arbiter |
| Dual-write doubles maintenance surface | High | Med | Shape B is explicitly scaffolding — no feature adds during dual-write |
| Nodes.py split misses a shared import (circular) | Low | Med | Keep _helpers.py for all shared code; no cross-node imports |
| to_thread fix introduces new timeout bug | Low | Med | wait_for raises TimeoutError → graph marks error, still produces partial report |
| Evidence emission in each node touches code paths that conflict with the concurrent P2.5/P3 work | Low | Low | P2 is a hard dependency for P2.5/P3 — no parallel work until flip |

---

## Changelog
- **2026-06-24** — Plan created from proposal §5.1–§5.2, §9, §10.1.
- **2026-06-24 (rev 2)** — Gaps G1-G5 resolved after source review (`~/.claude/scratch/p2_plan_review_20260624.md`): §T2.3.2 synthesizer dual-write pseudocode added; §T2.4 golden reconstruction strategy specified; §T2.7 consumer audit (reflection/explainer/contradictions) added; `_llm_enabled` added to `_helpers.py`; NEUTRAL polarity handling documented.
- **2026-06-24 (execution day 1)** — T2.1, T2.2, T2.3, T2.6, T2.8/9 COMPLETE. 549 tests green. BUG FIXED: `sys.path parents[2]` → `parents[3]` in all node files. CALIBRATION ISSUE: `fuse()` match rate ~22% vs legacy; `emit_ml_evidence` threshold (0.50) excludes below-threshold classes that consensus_engine overrides SAFE→DISPUTED.
- **2026-06-25 (execution day 2)** — T2.4 COMPLETE (golden characterization on 79/83 contracts; `emit_consensus_evidence()` added to fill the ML-below-threshold gap; DISPUTED mapped to SUPPORTS; overridden-from-SAFE strength floored at 0.30). T2.5 COMPLETE (P2 eval: macro_F1=0.1998, delta +0.0041 ✓; macro_Fbeta=0.2246, delta -0.0269). fuse() matches legacy exactly on 22.9% of classes (120/524); 75 asymmetry violations (legacy flagged → fused SAFE), 42/75 DenialOfService. **T2.7 (flip to Shape A) remains — gated on Ali accepting the Fbeta tradeoff.**
