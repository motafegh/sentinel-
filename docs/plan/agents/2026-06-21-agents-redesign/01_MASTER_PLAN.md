# Agents Module — Redesign Master Plan (2026-06-21)

**Status (2026-06-22):** WS1-WS5 complete + retro-tested. WS6a is next (Phase C).
**WS6b/c tool installs DONE (2026-06-22)** — Halmos+Z3 (pip, agents venv), Gigahorse
(souffle binary + boost headers + souffle-addon hand-built, no root except
`libsqlite3-dev`/`libncurses-dev` which Ali installed via apt — souffle's compiled
output hardcodes absolute system lib paths for these two, no way around it), ItyFuzz
(prebuilt nightly binary), Anvil (already present via Foundry). All 4 verified by
actually running them, not just checking the binary exists — Gigahorse decompiled a
real contract end-to-end (0 errors, 19 functions found). Full paths/env vars needed to
invoke each: `~/tools/TOOLCHAIN_ENV.md`. **WS6b/c are no longer blocked** — see
"Master ordering" table for the status of each workstream.
**Owner:** Ali + Claude.

## How to read this doc

This is the single ordered plan. It does NOT restate the underlying facts — those
already live in two places and are *referenced*, not duplicated, per Ali's
instruction:

- **Findings & verification** (what's broken/missing and the code-level proof):
  `docs/plan/agents/2026-06-21-agents-redesign/00_FINDINGS.md`
  — referenced below as **[FINDINGS #N]** using that doc's numbered summary table.
- **Deferred capability phases** (the original B/C/D proposal, only Phase A was
  built): `docs/plan/agents/2026-06-17-extended-capability/0{2,3,4}_PHASE_{B,C,D}_EXECUTION_PLAN.md`
  — referenced below as **[PHASE B/C/D, item X.Y]**.

Everything is included (nothing deferred-away). Ordering is by **priority**, where
priority = correctness/safety risk first, then cost/quality, then new capability.

---

## Priority rationale (the rule the ordering follows)

Anchored to the FN/FP asymmetry Ali established and verified
([FINDINGS, "FN/FP asymmetry correction"]): **a missed vulnerability (false
negative) can cost millions; a wasted review (false positive) costs time.** So:

1. First, fix anything that can silently mark a real vulnerability SAFE.
2. Then, fix anything that produces *fabricated* or misleading evidence (a
   different way to get a wrong verdict).
3. Then, make the expensive parts efficient and scale-correct.
4. Then, add genuinely new capability (the deferred phases).

---

## WORKSTREAM 1 — Verdict integrity & the FN/FP safety net (HIGHEST)

**Why first:** this is the only workstream where a defect can directly cause a
false negative — the costly failure. Groups [FINDINGS #8, #9, #10, #11]. These are
four symptoms of one root problem: there is no single, guaranteed-correct authority
deciding a class's final verdict, and the fallback chain can silently auto-SAFE a
flagged-but-uncorroborated class.

**D1 RESOLVED → Option B:** `consensus_engine` becomes the single complete verdict
authority — vote on EVERY flagged class (remove its `prob<0.50 and no-tool` skip),
demote `compute_verdict()` to a last resort that never silently SAFEs.

**Tasks:**
1. ~~Make D1 decision~~ — done (Option B).
2. Implement: `consensus_engine` votes on every flagged class; guarantee no flagged
   class can reach a final SAFE
   verdict without a recorded reason (corroboration absent ≠ safe).
3. Introduce a distinct "uninvestigated / inconclusive" verdict state so
   "we checked and found nothing" is never conflated with "we couldn't check"
   (covers the debate-timeout case in [FINDINGS #10]).
4. Fix the quick_screen Slither-vs-Aderyn impact-level inconsistency
   [FINDINGS #11] — align Aderyn to High/Medium/Critical like Slither.
5. **Name the FN/FP asymmetry as an explicit, written design principle**
   ([FINDINGS #8]) in `agents/src/orchestration/README.md`, and add a test that
   asserts a flagged-but-uncorroborated class never silently becomes SAFE.

**Validation:** new regression tests for each path; re-run the 2 real contracts +
add a deliberately-crafted "borderline, no corroboration, debate disabled" case
that MUST NOT come out SAFE-with-no-flag.

---

## WORKSTREAM 2 — Remove fabricated evidence (HIGH)

**Why second:** fabricated evidence is a direct route to a wrong verdict (it
already caused one hallucination). Cheaper and more contained than WS1.
Groups [FINDINGS #2].

**Tasks:**
1. Remove the 5 synthetic RAG corpora (Code4rena/Sherlock/Solodit/Immunefi/SWC
   placeholder JSON) from the live index — an empty source is honest, a fake one
   is a liability. Keep the real DeFiHackLabs corpus (726 real chunks).
2. Keep the fetcher *interfaces* (they're real, tested code) but make them no-ops
   until a real data source is wired — so re-enabling later is a data task, not a
   code task.
3. **D2 RESOLVED:** RAG essentially doesn't exist today. The full real build is a
   separate sub-project, planned in `docs/plan/agents/2026-06-21-agents-redesign/02_RAG_BUILD_PLAN.md`
   — gated behind the Phase C evaluation framework (can't decide if RAG helps until
   we can measure it). WS2's job is only the *removal*; the build is that doc.
4. Audit the narrative/debate prompts so RAG content, when present, is labeled as
   general reference (this fix already partially shipped 2026-06-21 — see
   `docs/changes/2026-06-21-agents-timeout-centralization-and-verdict-fixes.md`;
   verify it holds after the corpus change).

**Validation:** re-run safe_storage.sol and confirm the prior Multicall/Reentrancy
hallucination cannot recur (the source chunk no longer exists in the index).

---

## WORKSTREAM 3 — What the debate SEES (scale correctness) (HIGH)

**Why third:** this is the design-untested-at-scale risk. A real (long) contract
today would have its vulnerable function silently truncated away. Groups
[FINDINGS #1, #3, #7, #12] — all one question: *"what should the debate read?"*

**Open decision D3:** the debate's source-access mechanism. Candidates, cheapest→
richest:
- *3a:* raise/remove the raw char limits (trivial, but doesn't fix scale — a
  10k-char contract still overflows context).
- *3b:* feed the debate the `ml_hotspots` excerpt (specific functions/lines
  flagged by `graph_explain`) instead of a blind prefix [FINDINGS #3] — targeted,
  length-independent. Requires `ml_hotspots` to actually be plumbed into the
  debate prompt (today it's only used by `visualizer`).
- *3c:* reuse the ML model's sliding-window tokenization [FINDINGS #7] as the
  fallback when hotspots are empty — proven, already shipping in
  `ml/src/inference/predictor.py`.
- *3d (depends on WS6):* use real GNN attention weights (graph_explain's own
  documented "Phase 2") instead of the Slither proxy.
- *Recommendation:* 3b primary + 3c fallback now; 3d later via WS6.

**Tasks:**
1. Make D3 decision.
2. Plumb `ml_hotspots` into `cross_validator` (and `synthesizer` narrative) prompts.
3. Implement chosen truncation/windowing fallback.
4. **4-eyes as CLUES [FINDINGS #12, D4 reframed]:** expose each eye's individual
   per-class prediction in the ML API schema (the aux heads are already computed at
   inference — `aux_gnn`/`aux_transformer`/`aux_fused`/`aux_phase2` — just discarded).
   Use them as discountable HINTS that reveal *which reasoning drives the model's
   suspicion* (cfg_eye → control-flow bug; transformer_eye → token pattern), to:
   (a) point the debate/hotspot targeting at the right code aspect, (b) enrich
   explanations. **Do NOT add them to `consensus_engine` as votes** — they're 4
   correlated views of one already-discounted model, not independent witnesses;
   voting them would quadruple-count the ML signal. A light offline check (does
   eye X predict class Y errors?) sets per-class hint weighting, but stakes are low
   since they're hints, not deciders.

**Validation:** run one realistically-sized (multi-function, hundreds of lines)
real contract through the pipeline — the test we never ran — and confirm the
debate reasons over the actual vulnerable code, not a truncated prefix.

---

## WORKSTREAM 4 — Debate cost & selectivity (MEDIUM-HIGH)

**Why fourth:** the debate is ~3/4 of audit time. These are efficiency wins, not
correctness — but WS1's "inconclusive" state and the asymmetry rule are
PREREQUISITES (you can't safely skip the debate until the fallback is safe).
Groups [FINDINGS #5, #6].

**Tasks:**
1. Add `max_tokens` cap + "be concise / N sentences" instruction to all 3 debate
   roles [FINDINGS #6] — cheapest standalone win, do early; safe even before WS1.
2. Gate the debate selectively [FINDINGS #5], **asymmetrically per WS1's principle**:
   skip ONLY when multiple tools already agree it's vulnerable; NEVER skip because
   cheap signals say "safe." Depends on WS1 being done first.

**Validation:** re-measure debate wall-time before/after the max_tokens cap;
confirm the selective gate never skips a "looks-safe-by-cheap-signals" case.

---

## WORKSTREAM 5 — Reuse `data_module` as agent tooling (MEDIUM)

**Why fifth:** opportunity, not a bug — but it's the foundation WS3's richer
options (3d) and several deferred-phase items build on. Groups [FINDINGS #4].

**Tasks:**
1. Inventory `data_module/sentinel_data/representation/` (graph_extractor,
   cfg_builder, call_graph, pdg_builder, opcode_extractor, tokenizer) +
   `preprocessing/` and pick which are worth exposing to agents.
2. Wrap the chosen ones as MCP tools (reusing tested code, not reinventing AST/
   graph parsing inside agents).
3. This unlocks: real attention for WS3-3d, and overlaps heavily with deferred
   Phase B's static-analysis items (taint, access-control, call-graph, bytecode)
   — see WS6.

---

## WORKSTREAM 6 — Deferred capability phases (NEW CAPABILITY, LOWER PRIORITY)

**Why last:** these add NEW detection paradigms — valuable, but none fixes a
current correctness defect, and several are blocked on uninstalled external tools.
Only **Phase A** of the original 2026-06-17 proposal was built. Full detail lives
in the original phase docs (referenced, not restated).

**Reordered by priority within this workstream** (the original A→B→C→D order is NOT
kept — reprioritized against this redesign's findings):

**6a — Phase C (Production infra) — promote to FIRST of the deferred phases.**
Ref: `03_PHASE_C_EXECUTION_PLAN.md`. Items C.1 FastAPI gateway, C.2 evaluation
framework, C.3 prompt-injection guards, C.4 monitoring. **Why promoted:** C.2
(evaluation framework — precision/recall/F1 on a labeled benchmark) is the tool
that would let us *measure* whether WS1-WS4 actually improved verdict quality
instead of guessing. C.3 (prompt-injection guards) is a real security hole for a
tool that feeds untrusted contract source into LLMs. These are infrastructure the
rest of the redesign needs, not optional extras.

**6b — Phase B (Symbolic execution + bytecode) — second.**
Ref: `02_PHASE_B_EXECUTION_PLAN.md`. B.1-B.3 Halmos/Z3 symbolic verification,
B.4-B.5 Gigahorse bytecode analysis, B.6 taint, B.7 access-control, B.8 call-graph
reachability, B.9 CVE matching. **No longer blocked (2026-06-22):** Halmos+Z3
installed via pip into `agents/.venv`; Gigahorse installed by hand-extracting a
prebuilt Souffle `.deb`, Boost headers (header-only, no build), and building
Gigahorse's own `souffle-addon` locally — required `sudo apt-get install
libsqlite3-dev libncurses-dev` (Ali ran this; Souffle's compiled output hardcodes
absolute system library paths for these two, no way around it without root). All
verified by actually running them — Gigahorse decompiled a real contract end-to-end
(0 errors, 19 functions, 282 reachable blocks). Full invocation paths/env vars:
`~/tools/TOOLCHAIN_ENV.md`. B.6-B.8 overlap with WS5 (`data_module` already has
call-graph/PDG code) — do WS5 first so these reuse it (WS5 is done). These add
genuinely independent evidence channels (formal proofs, not heuristics) — high
value for the FN side, but a large build. Implementation not yet started.

**6c — Phase D (Economic security + on-chain) — last.**
Ref: `04_PHASE_D_EXECUTION_PLAN.md`. D.1-D.4 ItyFuzz/Anvil/economic-attack
simulation, D.5 ZKML proofs, D.6 on-chain AuditRegistry submission, D.7-D.8
(optional) Echidna/severity. **No longer blocked (2026-06-22):** ItyFuzz installed
(prebuilt nightly binary, no build needed); Anvil was already present via Foundry.
Both verified working. Highest effort, most specialized, furthest from the current
correctness gaps — genuinely last, but no longer waiting on tooling. Implementation
not yet started.

---

## Master ordering (single list)

| Order | Workstream | What | Gating / depends on | Status (2026-06-22) |
|---|---|---|---|---|
| 1 | WS1 | Verdict integrity & FN/FP safety net | D1 resolved (consensus authority) | ✅ DONE |
| 2 | WS4.1 | Debate `max_tokens` cap (cheap, safe early) | none | ✅ DONE |
| 3 | WS2 | Remove fabricated RAG evidence | D2 resolved (remove now; build via 02_RAG_BUILD_PLAN.md) | ✅ DONE |
| 4 | WS3 | What the debate sees (scale) | D3 resolved (hotspot+windowing); D4 resolved (eyes as clues) | ✅ DONE (retro-tested 2026-06-22, 8 tests) |
| 5 | WS4.2 | Selective debate gating | WS1 done first | ✅ DONE (retro-tested 2026-06-22, 9 tests) |
| 6 | WS5 | data_module as MCP tools | none (enables WS3-3d, WS6b) | ✅ DONE (sentinel-representation port 8014, 18 tests) |
| 7 | WS6a | Phase C: **C.1 FastAPI gateway** + **C.2 eval framework** | — | ✅ **C.1 + C.2 DONE (2026-06-22). C.1: `src/api/` (5 endpoints, JobStore, 44 tests). C.2: `src/eval/` library (28 tests, script refactored).** C.3 guards + C.4 monitoring still open (~1-2 weeks). |
| 8 | WS6b | Phase B (symbolic + bytecode + taint + access-control) | WS5 done | ✅ UNBLOCKED (2026-06-22) — Halmos+Z3, Gigahorse all installed + verified working. Implementation not started. |
| 9 | WS6c | Phase D (economic + on-chain) | — | ✅ UNBLOCKED (2026-06-22) — ItyFuzz, Anvil installed + verified working. Implementation not started. |

**Summary:** 6 of 9 workstreams complete. Full session: `docs/changes/2026-06-22-agents-ws0-ws5-complete.md`.

---

## Open decisions — RESOLVED 2026-06-21

- **D1** (WS1): **RESOLVED → `consensus_engine` becomes the single complete verdict
  authority.** It votes on every flagged class (its `prob<0.50 and no-tool` skip is
  removed); `compute_verdict()` is demoted to a last resort that can never silently
  return SAFE. One component owns the verdict instead of three coexisting rules.
- **D2** (WS2): **RESOLVED → RAG essentially doesn't exist; remove the fakes now,
  build it properly later via a dedicated plan.** The 24 placeholder chunks are
  removed immediately (WS2); the full real RAG build is split into its own document:
  `docs/plan/agents/2026-06-21-agents-redesign/02_RAG_BUILD_PLAN.md`. RAG is gated behind the Phase
  C evaluation framework (can't decide if it helps until we can measure it).
- **D3** (WS3): **RESOLVED → hotspot-guided excerpt + ML sliding-window fallback.**
  Raising the char limit is rejected (doesn't fix scale). Debate reads the actually-
  flagged functions, length-independent.
- **D4** (WS3.4): **RESOLVED, reframed by Ali → use each eye's INDIVIDUAL output as
  CLUES, not the disagreement signal as a vote.** Critical correction: the 4 eyes
  are NOT independent witnesses (fused_eye combines the others; all from one model
  we already discount) — so they must NOT be added to `consensus_engine` as 4 more
  votes (that quadruple-counts the discounted ML signal). Instead, each eye reveals
  WHICH kind of reasoning drives the model's suspicion (cfg_eye → control-flow bug
  like reentrancy; transformer_eye → token pattern like tx.origin), used to point
  the debate at the right code aspect (feeds WS3's hotspot targeting) and enrich
  explanations — as discountable hints, never deciders. Still benefits from a light
  offline check to learn per-class which eye matters, but low-stakes since they're
  hints. See revised WS3.4 below.

---

## What this plan deliberately does NOT do

- Does not restate the code-level evidence — that's in the findings doc.
- Does not keep the original A→B→C→D phase order — Phase C is promoted because its
  evaluation framework is what makes the rest measurable.
- Does not fold the RAG build into a workstream — it's its own document
  (`02_RAG_BUILD_PLAN.md`), because it's a slow data project, not a code tweak.

**Decisions D1-D4: all RESOLVED 2026-06-21** (see "Open decisions — RESOLVED"
above). The plan is now decision-complete and ready to execute top-down when Ali
gives the go.
