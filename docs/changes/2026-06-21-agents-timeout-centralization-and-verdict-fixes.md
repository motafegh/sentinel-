# Agents Module — Timeout Centralization, Live Timing, Verdict Fixes (2026-06-21)

**Follow-up to:** `docs/changes/2026-06-21-agents-manual-verification-real-bugs-found.md`
(the Slither/Aderyn registration bugs). After that fix, two LLM-debate timeouts were
analyzed precisely, leading to a 5-part follow-up: (1) centralize every timeout
default, CLI-overridable, (2) fix the two design gaps found during manual
verification, (3) add live per-step timing logs throughout the whole module, (4)
document, (5) re-test with timeouts removed entirely to see true per-step latency.

## 1. Centralized timeout configuration

**New file `src/orchestration/timeouts.py`** — the ONLY place a timeout default is
defined. Previously every LLM-call timeout (`90`, `120`, `240`...) was a bare magic
number duplicated at its call site; tracing or changing a default meant grepping the
whole module. Every other file now imports the named constant and reads it via
`get_timeout(ENV_VAR_NAME, DEFAULT)` at call time (never cached), so existing
override mechanisms (shell env var, test `monkeypatch.setenv`) keep working
unchanged — only WHERE the default lives changed, not the override contract.

**`scripts/run_real_audit.py`** gained one CLI flag per timeout
(`--lm-studio-timeout-s`, `--cross-validator-timeout-s`, `--debate-timeout-s`,
`--synthesizer-timeout-s`, `--reflection-timeout-s`, `--aderyn-timeout-s`) plus
`--unbounded-timeouts`, which sets all of them (and the script's own `--timeout-s`)
to 3600s in one shot — an explicit per-flag value still wins over the blanket
setting. `_resolve_timeouts()` pushes the resolved values into `os.environ` before
any agents module is imported, mirroring the existing `_resolve_urls()` pattern.

## 2. Two design gaps fixed (found during the manual-verification session)

**Verdict fallback gap:** `synthesizer` used to read only `cross_validator`'s
verdicts and fall straight to the older `compute_verdict()` rule when the debate
failed — completely bypassing `consensus_engine`'s already-computed, ML-discounted
vote. Verified live: on `safe_storage.sol`, `consensus_engine` correctly said
`ExternalBug: SAFE` while `compute_verdict()` said `DISPUTED` for the identical
evidence. Fixed: `synthesizer` now prefers `consensus_engine`'s vote as the
fallback; `compute_verdict()` only runs for classes `consensus_engine` never scored.

**Narrative hallucination:** the narrative prompt listed every ML-flagged class with
NO verdict attached, so the model had no signal a class had been cleared — observed
live, it described a "Reentrancy risk" on a contract whose Reentrancy verdict was
SAFE and which contains zero external calls, citing unlabeled RAG reference content
about an unrelated exploit pattern. Fixed: `vuln_lines` now attaches
`→ verdict: ...` to every line (mirroring `reflection`'s prompt, which already did
this correctly), the system prompt explicitly instructs the model to only discuss
CONFIRMED/LIKELY classes, and the RAG section is relabeled as general historical
reference, not site-specific evidence.

Both fixes are scoped to `synthesizer`'s verdict-selection loop and narrative
prompt-building block — no schema/data-flow changes.

## 3. Live per-step timing, everywhere

**New file `src/orchestration/timing.py`** — `step_timer()` is a context manager
logging a uniform `START` line and a `DONE | elapsed=Xs` line around any block, even
on exception. `timed_node()` wraps a LangGraph node coroutine with `step_timer`
automatically.

**`graph.py`'s `build_graph()`** now wraps every one of the 13 nodes with
`timed_node()` at registration time — one change, full coverage, zero risk to
internal node logic (no node body was reindented). This means timing logs appear in
EVERY context the graph runs in (production MCP-driven server, the test script,
ad-hoc REPL use), not only when a caller happens to add its own wrapper.

**`cross_validator`'s 3 debate roles are individually timed**
(`cross_validator.prosecutor`, `.defender`, `.judge`), not just the aggregate — a
slow run now shows exactly which of the 3 sequential calls is the bottleneck.
`synthesizer.narrative`, `reflection.llm_summary`, and the two MCP servers'
(`inference.predict`/`inference.batch_predict`, `graph_inspector.get_graph_hotspots`)
handlers got the same treatment.

**`scripts/run_real_audit.py`** dropped its own `_wrap_node`/`_build_instrumented_graph`
— the module now does this natively, so the harness no longer needs its own
ad-hoc timing wrapper (was previously the ONLY place this existed; would have
double-logged against the new built-in instrumentation).

## 4. Why `DEBATE_TIMEOUT_S=240` was wrong from the start

Investigated precisely at Ali's request. The number was set earlier in the same
session, generalized from ONE single-call observation (~60-90s) early in the
session, before LM Studio had served hours of back-to-back large calls. By the time
it shipped, two real debates had already timed out at 246-248s under the session's
accumulated GPU load — meaning the default was already too tight, not regressed
later. Documented honestly in `.env` and `timeouts.py` rather than presented as a
validated measurement. `--unbounded-timeouts` exists specifically so a real
measurement can replace this guess.

## 5. Re-tested with `--unbounded-timeouts`

See the live run results appended to
`~/.claude/scratch/agents_manual_tool_verification_20260621.md` — true per-step
timing on both `vulnerable_reentrant.sol` and `safe_storage.sol` with no artificial
truncation anywhere in the chain.

## Testing

- 6 new tests for the two verdict fixes (`test_graph_routing.py`:
  `test_consensus_engine_wins_over_compute_verdict_when_debate_fails`,
  `test_compute_verdict_is_last_resort_when_consensus_engine_didnt_score_class`,
  `test_narrative_prompt_grounds_each_class_with_its_verdict`).
- 10 new tests for the centralized config + timing modules (`test_timeouts_and_timing.py`).
- Full suite: **297 passed** (284 before this follow-up → +13... see note: 2 Fix1 +
  1 Fix2 + 10 timeouts/timing = 13).
