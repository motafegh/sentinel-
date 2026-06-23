# Agents Redesign — WS0 through WS5 Complete (2026-06-22)

**Status:** All 6 workstreams in `docs/plan/agents/2026-06-21-agents-redesign/`
executed. **351 → 368 tests pass** (+17 new for WS3+WS4.2 retro-coverage). No
regressions; 3 pre-existing `solc`-unavailable failures unchanged.

## TL;DR

- **WS0** (gate infrastructure) — done before this session but reviewed
- **WS1** (verdict integrity) — done before this session
- **WS1.5** (8-case reconciliation) — done before this session
- **WS4.1** (debate max_tokens cap) — done before this session
- **WS2** (fake RAG removed) — done before this session
- **WS3** (hotspot excerpts + D4 eye clues) — **done 2026-06-22, this session**
- **WS4.2** (selective debate gating) — **done 2026-06-22, this session**
- **WS5** (data_module → MCP tools) — **done 2026-06-22, this session**

## What this session did (2026-06-22)

### 1. WS3 — What the debate SEES (Finding #1, #3, #7, #12)

**Problem:** cross_validator was sending the LLM a blind `contract_code[:2000]`
prefix. For real-sized contracts, the vulnerable function lives past the
cutoff, so the debate reasoned over the wrong code. Also, the 4 ML "eyes"
(`gnn`/`transformer`/`fused`/`phase2`) were computed in the forward pass but
thrown away by `_format_result` — rich signal sitting unused.

**Fix (D3 resolved):**
- `agents/src/orchestration/nodes.py:cross_validator` now reads
  `state.ml_hotspots` (populated by `graph_explain`) and extracts the flagged
  source lines per class into a "Focused code excerpts" block. The full
  source is appended as reference (capped at 4000 chars).
- Fallback when `ml_hotspots` empty: old `[:2000]` truncation + note about
  ML sliding-window count (no useful hotspot data to focus the excerpt).

**Fix (D4 resolved):**
- ML side: `ml/src/inference/predictor.py` now passes `return_aux=True` to
  the model, applies sigmoid to all 4 eye auxiliary heads, and includes them
  as `eye_predictions: {"gnn": {"Reentrancy": 0.81, ...}, ...}` in the
  inference response. Backward-compatible — `eye_predictions` is optional
  in the Pydantic `PredictResponse`.
- Agents side: cross_validator renders per-class "Eye clues: gnn=0.81
  transformer=0.32 fused=0.72 phase2=0.55 (gnn eye driving)" in the
  evidence block. **Discountable hints, NOT votes** — the 4 eyes are
  correlated views of one already-discounted model, so adding them to
  `consensus_engine` as votes would quadruple-count the ML signal
  (correctly rejected at the D4 decision).

**Tests:** `agents/tests/test_ws3_hotspot_excerpts.py` — **8 tests** (NEW):
hotspot excerpts replace raw prefix, grouped by class, full source as
reference, fallback when no hotspots, fallback note suppression for
single-window contracts, eye clues appear, eye clues absent when no
predictions, eye clues per class.

### 2. WS4.2 — Selective debate gating (Finding #5)

**Problem:** the 3-role debate ran unconditionally on every deep-path
contract — 30% of audit time. The cheap signals (ML/Slither/Aderyn) had
ALREADY done a multi-tool vote via `consensus_engine`. When the consensus
verdict was already CONFIRMED with multi-tool agreement, the debate was
redundant.

**Fix:** add an explicit gate before `if _debate_on:` in cross_validator:
- **Skip** the debate when ALL flagged classes are CONFIRMED by
  `consensus_engine` AND ≥2 of 3 tools (ML/Slither/Aderyn) voted positive
- **NEVER skip** because cheap signals say "safe" (FN/FP asymmetry
  principle — "looks safe" is the worst place to save time)
- When skipped, emit consensus verdicts directly with the same shape
  (`verdicts`/`confirmations`/`contradictions`) so downstream synthesis is
  unchanged

**Tests:** `agents/tests/test_ws4_2_selective_gating.py` — **9 tests** (NEW):
skip when all CONFIRMED with 2 tools, skip with 3 tools, skip with multiple
classes, run when one class is LIKELY, run when one class is DISPUTED, run
when consensus says SAFE, run when only 1 tool agrees, run when no
consensus state, skipped-debate emits confirmations.

### 3. WS5 — Wrap `data_module` as MCP tools

**Problem:** `data_module/sentinel_data/representation/` has tested code
(graph_extractor, cfg_builder, etc.) that the agents module wasn't reusing.
Phase B (symbolic execution, taint, call-graph reachability) would need the
same CFG scaffolding — better to expose it as MCP tools now.

**Fix:** new `agents/src/mcp/servers/representation_server.py` (port 8014)
wraps `cfg_builder.py::build_cfg` (the same code path as ML training) as
the `get_function_cfgs(contract_code)` MCP tool. Returns per-function CFG
with:
- Node-type counts (CALL/WRITE/READ/CHECK/ARITH/OTHER)
- Structural metrics (num_loops, max_depth)
- **Explicit CEI violation detection** via DFS-based search for
  write→call paths in the CFG (the classic reentrancy pattern)
- `cfg_complexity_score` heuristic for cheap ranking

Three modes: `data_module_cfgs` (real), `slither_error` (solc/Slither
failed), `mock` (testing). Reuses tested data_module code rather than
reinventing AST/CFG parsing.

Wired into `agents/.env` (`MCP_REPRESENTATION_PORT=8014`),
`agents/scripts/run_real_audit.py` (`--mcp-representation-url` CLI flag +
health check), `agents/src/mcp/README.md` updated.

**Tests:** `agents/tests/test_representation_server.py` — **18 tests** (NEW):
CEI detection (empty/no-writes/no-calls/classic reentrancy pattern/call-
before-write/multiple-writes-one-call/cycle guard), function summary (empty
function/with CEI/loops+arithmetic), mock mode, tool dispatcher (mocked +
real path with mocked `_call_build_cfg`), health endpoint, app routes.

### 4. Stale doc fixes (caught by Ali)

**Stale #1: `agents/src/llm/client.py:128`** — `get_fast_llm` docstring said
"Defaults: prosecutor/defender 512, judge 256" but actual code (nodes.py)
uses `768/768/0`. The 512 was from the start of the WS4.1 sweep; 768 is
the chosen sweet spot. Docstring now matches reality + includes the sweep
rationale and a pointer to where the actual env-var defaults live.

**Stale #2 (DANGEROUS): `agents/src/orchestration/README.md:184-187`** —
the "Verdict source priority" section documented the **broken** 3-tier chain
("1. debate → 2. consensus → 3. compute_verdict") that caused Finding #14
(debate silently SAFEd 13 consensus-flagged classes, 4 at confidence=1.0).
Anyone reading the docs would learn the wrong, dangerous mechanism. The
section is now replaced with:
- Explicit callout that the old 3-tier chain is the broken mechanism that
  caused Finding #14
- Description of the actual 8-case `_reconcile_verdicts()` (full table link)
- The 2 invariants that test_verdict_reconciliation.py enforces
- Link to the plan doc for the full case table

### 5. Test additions to fill the WS3+WS4.2 coverage gap

WS3 and WS4.2 were originally shipped without dedicated test coverage —
they were verified by the LLM-on gate (16-contract pass) but had no unit
tests. Caught during this session's audit. Added:
- `tests/test_ws3_hotspot_excerpts.py` — 8 tests
- `tests/test_ws4_2_selective_gating.py` — 9 tests

Total: **+17 tests, 0 regressions**. Test count: **351 → 368 pass**.

## Files changed this session

### Code
- `agents/src/orchestration/nodes.py` (WS3 hotspot excerpts, D4 eye clues, WS4.2 gating)
- `agents/src/mcp/servers/representation_server.py` (NEW, WS5)
- `ml/src/inference/predictor.py` (D4 — pass `return_aux=True` to model)
- `ml/src/inference/api.py` (D4 — `PredictResponse.eye_predictions`)
- `agents/scripts/run_real_audit.py` (WS5 — `--mcp-representation-url`)
- `agents/.env` (WS5 — `MCP_REPRESENTATION_PORT=8014`)
- `agents/src/llm/client.py` (stale doc fix)
- `agents/src/orchestration/README.md` (stale doc fix + WS3/WS4.2 sections)

### Tests
- `agents/tests/test_ws3_hotspot_excerpts.py` (NEW, 8 tests)
- `agents/tests/test_ws4_2_selective_gating.py` (NEW, 9 tests)
- `agents/tests/test_representation_server.py` (NEW, 18 tests, WS5)

### Plan / scratch
- `~/.claude/scratch/ws3_hotspot_excerpts_20260622.md`
- `~/.claude/scratch/ws5_representation_mcp_20260622.md`

## Test count progression
- Start of session: 351 pass, 3 pre-existing solc failures
- After WS3+WS4.2 retro-tests: 368 pass
- After WS5: 368 pass (no change to existing tests; only new test files added)

## What remains (per master plan)
- **WS6a** — Phase C: FastAPI gateway (C.1) + eval framework expansion (C.2)
  + prompt-injection guards (C.3) + monitoring (C.4). **2-3 weeks**. Promoted
  to first of the deferred phases per master plan because C.2 makes
  WS0-WS5 measurable at scale and C.3 closes a real security hole.
  **2026-06-22 status:** C.2's evaluation dataset decision recorded in
  `03_PHASE_C_EXECUTION_PLAN.md` — use the existing WS0 88-contract corpus
  for now; expand to 100-200 contracts is deferred (not abandoned).
- **WS6b** — Phase B: symbolic execution + bytecode + taint + access-control.
  **2026-06-22 status:** UNBLOCKED. Halmos+Z3 (pip into agents venv),
  Gigahorse (souffle binary + boost headers + souffle-addon hand-built, no
  root except `libsqlite3-dev`/`libncurses-dev` which Souffle hardcodes),
  all installed and verified working. Full invocation details at
  `~/tools/TOOLCHAIN_ENV.md`. Overlaps with WS5 (which built the CFG
  scaffolding it needs). Implementation not yet started.
- **WS6c** — Phase D: economic + on-chain. **2026-06-22 status:** UNBLOCKED.
  ItyFuzz prebuilt nightly binary, Anvil via Foundry (already present).
  All installed and verified working. Last. Implementation not yet started.

## References
- Master plan: `docs/plan/agents/2026-06-21-agents-redesign/01_MASTER_PLAN.md`
- Findings: `docs/plan/agents/2026-06-21-agents-redesign/00_FINDINGS.md`
- Verdict reconciliation (WS1.5): `docs/plan/agents/2026-06-21-agents-redesign/05_VERDICT_RECONCILIATION_PLAN.md`
- Ali's live baseline findings (#13-15): `docs/plan/agents/2026-06-21-agents-redesign/04_LIVE_BASELINE_FINDINGS.md`
- Phase C plan: `docs/plan/agents/2026-06-17-extended-capability/03_PHASE_C_EXECUTION_PLAN.md`
- Session working memory: `~/.claude/scratch/agents_redesign_exec_20260621.md`
