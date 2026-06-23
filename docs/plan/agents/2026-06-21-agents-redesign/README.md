# Agents Module Redesign — 2026-06-21

**Start here, then read in this order:**

| File | What it is |
|---|---|
| `00_FINDINGS.md` | The investigation — every claim verified against the actual code, not assumed. Raised when Ali questioned whether the debate's full-source-reading makes earlier pipeline stages pointless. 12 numbered findings, severity-rated. |
| `01_MASTER_PLAN.md` | The plan — 6 workstreams grouping the 12 findings (plus the deferred Phase B/C/D from `docs/plan/agents/2026-06-17-extended-capability/`), priority-ordered by false-negative risk first. All 4 open decisions (D1-D4) resolved. |
| `02_RAG_BUILD_PLAN.md` | Split out from the master plan — RAG is a slow data-acquisition project, not a code tweak. Currently RAG is fakes + one narrow real corpus; this is the plan to build it properly, gated behind an evaluation framework that doesn't exist yet. |
| `03_GATE_INFRASTRUCTURE_PLAN.md` | **WS0 — inserted before WS1.** The existing 297 tests assert graph plumbing, not verdict quality (mocked ML + toy contracts + LLM off). This builds the lightweight gate (corpus + runner + comparator + baseline + per-WS gate assertions) that makes WS1-WS5 measurable NOW. Reuses the 66-contract benchmark v0.1. ~2 days. Foundation for WS6a's full Phase C.2 eval framework — not a replacement. |
| `04_LIVE_BASELINE_FINDINGS.md` | **3 new findings (#13-#15), found empirically** by manually inspecting real `--llm-on` baseline reports against ground truth. #14 is CRITICAL: the debate's own verdict outranks `consensus_engine`, and on a real contract it silently overrode a correct, tool-corroborated LIKELY to a wrong SAFE — D1 is not actually enforced at the point where verdicts get reconciled. |
| `05_VERDICT_RECONCILIATION_PLAN.md` | **WS1.5 — the single fix for #13, #14, #15.** Rewrites the synthesizer's verdict-reconciliation loop + the debate's input filter + persists the debate transcript. 8-case reconciliation rules: the debate can upgrade but can only downgrade to DISPUTED, never to SAFE, when consensus voted non-SAFE. The core invariant: a flagged class reaches SAFE only if BOTH consensus AND debate agree. |

**Status (2026-06-22):** WS0 through WS5 all DONE + retro-tested. WS6a (Phase C) partly done — **C.1 gateway (`src/api/`, ~44 tests) and C.2 eval framework (`src/eval/`, 28 tests) DONE**; C.3 prompt-injection guards and C.4 monitoring still open (~1-2 weeks). WS6b (symbolic/bytecode) and WS6c (economic/on-chain) were blocked on missing tools (Halmos, Z3, Gigahorse, ItyFuzz) — **all 4 now installed and verified working** (Anvil was already present); neither workstream's implementation has started yet, but nothing is blocking them anymore. C.2's evaluation dataset will use the existing 88-contract WS0 corpus for now; expanding to 100-200 real-world contracts is deferred, not abandoned. **Execution order: WS0→WS1→WS1.5→WS4.1→WS2→WS3→WS4.2→WS5 (all done) → WS6a (next) → WS6b → WS6c.**

**Related (not duplicated here):**
- The work this redesign is reacting to (Phase A build + the Slither/Aderyn bug
  fixes + timeout/verdict fixes, all 2026-06-21):
  `docs/changes/2026-06-21-agents-{phase-a-extended-capability,manual-verification-real-bugs-found,timeout-centralization-and-verdict-fixes}.md`
- The original deferred-phase proposal this plan reorders:
  `docs/plan/agents/2026-06-17-extended-capability/`
