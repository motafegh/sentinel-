# Agents Redesign — Gate Infrastructure Plan (WS0) (2026-06-21)

**Status:** planning only — nothing here implemented yet.
**Owner:** Ali + Claude.
**Inserts before:** WS1 of `01_MASTER_PLAN.md`.

## Why this doc exists (separately from the master plan)

Ali's directive (2026-06-21), after reviewing the master plan:

> *"first check and see if we need some actionable plans or not to have proper
> real gates, not just some trivial tests — instead the tests while we progress,
> we incrementally checking the whole system properly."*

The master plan put the evaluation framework at **order 7** (WS6a / Phase C.2),
arguing "C.2 is what lets us measure whether WS1-WS4 actually improved verdict
quality." That ordering has a hole: it asks us to ship WS1-WS5 (correctness
fixes to the verdict chain) **before** the tool that could prove they worked
exists. WS1-WS5 would then be validated only by the existing test suite —
which (verified below) asserts graph **plumbing**, not verdict **quality**.
This doc closes that hole: a lightweight gate that runs **first**, makes every
later WS measurable, and is reused (not replaced) by WS6a's full framework.

**This is WS0.** It is not optional and not deferred — it is the prerequisite
that makes WS1-WS5 trustworthy.

---

## The problem — what current "tests" actually verify

Verified by reading `agents/tests/conftest.py` + `agents/tests/test_smoke_e2e.py`
(the only system-level check, 7 tests) in full:

| Property | Current state | Consequence |
|---|---|---|
| ML signal | **Mocked** (`_mock_mcp` returns hardcoded `Reentrancy=0.82` or all-0.04) | Can't test borderline scores (0.35-0.49) — the exact WS1 silent-SAFE band |
| MCP servers | **Mocked** (search/audit/inspector return canned dicts) | Can't test real RAG retrieval — the WS2 hallucination path |
| Slither | **Mocked** or stubbed via `patch.dict("sys.modules", {"slither": None})` | Can't test real tool corroboration — the WS1 consensus logic |
| Contract size | **373-720 chars** (VAULT_CONTRACT ~15 lines, SAFE_CONTRACT ~15 lines) | Can't trigger the 2000/500-char truncation — the WS3 bug |
| LLM | **Disabled globally** (`conftest.py:15` sets `AGENTS_DISABLE_LLM=1`) | No debate/narrative/reflection runs — can't test WS4.1 cost or quality |
| Assertions | Field presence, path_taken, key existence, "Reentrancy" in decisions (only because the mock returned it) | Verifies **plumbing**, not **verdict correctness** |

**The 7 smoke tests cannot detect any of the 4 target bugs:**
- WS1 (`compute_verdict` SAFE-collapse on 0.35-0.49): mocked ML never returns borderline scores. Invisible.
- WS2 (RAG hallucination): RAG mocked to a fixed chunk. Invisible.
- WS3 (truncation on long contracts): all contracts <720 chars. Invisible.
- WS4.1 (debate cost/quality): no debate runs. Invisible.

The other 19 test files (`test_graph_routing.py` 63 tests, `test_routing_phase0.py`
46, `test_chunker.py` 18, `test_consensus_voting.py` 11, etc.) are **unit tests**
— they verify functions in isolation (does `consensus_vote(0.82, True, False, "RE")`
return the expected dict?), also not whole-system verdict quality. None runs the
full graph on a real, labelled, realistic-sized contract and asks "did we get
the right answer?"

This is exactly the "trivial tests" regime Ali named.

---

## The reusable asset — benchmark v0.1

`data_module/benchmarks/benchmark_v0.1_quickstart/` (already built, OOD-verified,
contamination-checked — see MEMORY.md §Comprehensive Benchmark v0.1):

| Property | Value |
|---|---|
| Contracts | **66 real `.sol` files** |
| Size range | **1.7 KB – 48 KB** (mean ~16 KB) — realistic, multi-function |
| Labelling | **By directory** (`contracts/by_class/<CLASS>/`) + per-contract `.json` sidecar |
| Sidecar schema | `{labels: [...], source, category, original_name, tier, ground_truth}` |
| Classes present | CallToUnknown (14), MishandledException (10), NonVulnerable (11), Reentrancy (11), Timestamp (10), TransactionOrderDependence (10) |
| `NonVulnerable` class | **Safe contracts** — the negative examples (11) |
| OOD status | Verified not in v3 training/val/test (SHA-256 audit) |
| Manifest | `tier_a_manifest.json` (662 lines, full sha256 + size + class per contract) |

This is a **drop-in corpus** for an agents gate. No new data acquisition needed
for WS0 — only the 5 crafted edge cases (below) are new.

---

## WS0 components

| # | Component | Effort | What it is |
|---|---|---|---|
| 1 | **Corpus** | ~1 d | (a) Reuse the 66 benchmark contracts as-is (their `.json` sidecars are the labels). (b) Craft **5 edge cases** the benchmark doesn't cover, each targeting a specific WS's bug: <br>• `edge_borderline_no_corroboration.sol` — prob ~0.40, no Slither/Aderyn hit (the WS1 silent-SAFE case). <br>• `edge_debate_timeout.sol` — a contract that produces a long debate, forces the INCONCLUSIVE path. <br>• `edge_long_contract_truncated.sol` — a >2000-char contract with the vulnerable function PAST the truncation cutoff (the WS3 bug). <br>• `edge_safe_rag_resembles_exploit.sol` — a safe contract that loosely matches a DeFiHackLabs exploit chunk (the WS2 hallucination case). <br>• `edge_multi_bug.sol` — a contract with 2+ real bugs in different classes (tests per-class verdict independence). <br>Each gets a `.json` sidecar in the same schema. Total: ~71 contracts. |
| 2 | **Runner** | ~0.5 d | Extend `agents/scripts/run_real_audit.py` with `--corpus DIR` mode: loop every `*.sol` under DIR, invoke the full graph per contract, write a per-contract JSON report to `<output-dir>/<stem>_report.json`. Default `--no-llm` (fast, deterministic, ~5s/contract → ~6 min for 71). `--llm-on` for spot checks (slow, ~60s/contract). Reuses the existing per-contract report format — no new schema. |
| 3 | **Comparator** | ~1 d | New `agents/scripts/eval_benchmark.py`: reads per-contract reports + sidecar labels, computes: <br>• **Per-class precision / recall / F1** (verdict vs label; positive = CONFIRMED+LIKELY, negative = SAFE+DISPUTED+WATCH+INCONCLUSIVE — tunable via config). <br>• **WS-specific gates** (the assertions below; each gate is a boolean pass/fail). <br>• **Macro-F1 + micro-F1** across classes. <br>• Markdown report (`<output-dir>/eval_report.md`) + JSON metrics (`<output-dir>/eval_metrics.json`) for diffing. <br>• Exit code: 0 if all gates pass + F1 ≥ baseline; 1 on regression. <br>~150 lines, no external deps beyond what agents already uses. |
| 4 | **Baseline run** | ~0.25 d | Run the gate on the **current pre-redesign** code → store `agents/eval/baselines/pre_redesign.json`. Every later WS's gate compares against this. This is the honest "before" snapshot. |
| 5 | **Per-WS gate assertions** | ~2-4 h per WS | Each later WS adds 1-3 specific gates to the comparator (see next section). The WS is not "done" until its gates pass AND macro-F1 ≥ baseline. |

**Total WS0 effort: ~2 days.** No external tool installs. No new ML training.
Reuses tested code (the runner) + an existing labelled corpus (the benchmark).

---

## Per-WS gate assertions (what each later WS must prove)

Each WS ships with its own gates. The comparator runs all gates; a WS is "done"
only when its gates pass AND `macro_F1 ≥ baseline.macro_F1` (no regression).

| WS | Gate assertions (all must pass) |
|---|---|
| **WS1** (verdict integrity) | (a) `silent_safe_on_flagged == 0` — no class with `prob ∈ [0.35, 0.50)` ends in verdict `SAFE` with no recorded reason. (b) `inconclusive_emitted_on_timeout == true` — the `edge_debate_timeout` case emits `INCONCLUSIVE`, not `SAFE`. (c) `macro_F1 ≥ baseline` — the verdict-authority change didn't break correct verdicts. |
| **WS4.1** (debate max_tokens) | (a) `debate_walltime_p50 ≤ baseline.debate_walltime_p50 × 0.6` — the cap cuts debate time ~40%. (b) `macro_F1 ≥ baseline` — shorter LLM responses didn't degrade verdicts. (LLM-on mode only; --no-llm skip this gate.) |
| **WS2** (remove fake RAG) | (a) `hallucination_on_safe_subset == 0` — on the `NonVulnerable` + `edge_safe_rag_resembles_exploit` contracts, zero verdicts reference RAG content not in the index. (b) `macro_F1 ≥ baseline`. |
| **WS3** (debate source access) | (a) `edge_long_contract_truncated` verdict == the planted bug's class (the vulnerable function past the old 2000-char cutoff is now seen + correctly identified). (b) `macro_F1 ≥ baseline`. |
| **WS4.2** (selective debate gating) | (a) `debate_skipped_on_multi_tool_agree == true` — a contract where ML + Slither + Aderyn all agree VULNERABLE skips the debate (and verdict is still correct). (b) `debate_NEVER_skipped_on_looks_safe` — the `edge_borderline_no_corroboration` case still gets a debate (or an INCONCLUSIVE), never silently SAFE. (c) `macro_F1 ≥ baseline`. |
| **WS5** (data_module as MCP tools) | (a) The new MCP tools run without error on the corpus. (b) `macro_F1 ≥ baseline`. (Light — WS5 is infrastructure, not a verdict change.) |

**The "F1 ≥ baseline" guard is the regression net.** A WS that makes its
specific gate pass but drops F1 is NOT done — it broke something else. This is
the "incrementally checking the whole system properly" Ali described.

---

## Sequencing (new ordering, WS0 inserted)

| Order | Workstream | What | Gating / depends on |
|---|---|---|---|
| **0** | **WS0** | **Gate infrastructure (this doc)** | **none — do FIRST** |
| 1 | WS1 | Verdict integrity + FN/FP safety net | WS0 done (gate proves no silent-SAFE) |
| 2 | WS4.1 | Debate `max_tokens` cap | WS0 done (gate proves no F1 regression) |
| 3 | WS2 | Remove fake RAG evidence | WS0 done (gate proves no hallucination) |
| 4 | WS3 | What the debate sees (scale) | WS0 done |
| 5 | WS4.2 | Selective debate gating | WS1 done |
| 6 | WS5 | `data_module` as MCP tools | none (enables WS3-3d, WS6b) |
| 7 | WS6a | Phase C (gateway, **full eval framework**, guards, monitoring) | builds on WS0 (reuses comparator + runner) |
| 8 | WS6b | Phase B (symbolic + bytecode) | tool installs; WS5 |
| 9 | WS6c | Phase D (economic + on-chain) | tool installs |

---

## How WS0 relates to WS6a (Phase C.2) — not a replacement, a foundation

WS6a's C.2 (per `03_PHASE_C_EXECUTION_PLAN.md`) proposes:
- `agents/src/eval/pipeline_metrics.py` (~150 lines) + `agents/src/eval/benchmarks.py` (~100 lines)
- 100-200 contracts, per-class F1 + **AUC-PR**, 6-8 days, 6 tests.

**WS0 is the lightweight version that makes WS1-WS5 measurable NOW; WS6a builds
on it, does not replace it:**
- The comparator (`eval_benchmark.py`) becomes `pipeline_metrics.py` — same logic, expanded (AUC-PR added).
- The runner's `--corpus` mode becomes `benchmarks.py`'s loader — same corpus format.
- The 66-contract benchmark expands to 100-200 (the v1.0 target already planned in `BENCHMARK_DESIGN.md`).
- The per-WS gate assertions become regression tests in the full framework.
- The baseline `pre_redesign.json` becomes the first entry in a longitudinal metrics history.

**Nothing in WS0 is throwaway.** It is the foundation C.2 was always going to
need, pulled forward because WS1-WS5 can't be trusted without it.

---

## Effort + dependencies

| Item | Value |
|---|---|
| WS0 total effort | ~2 days (corpus 1d + runner 0.5d + comparator 1d + baseline 0.25d, with overlap) |
| Per-WS gate effort (after WS0) | ~2-4 h per WS (add 1-3 gate assertions to the comparator + run before/after) |
| External tools needed | **None** — reuses `run_real_audit.py` (exists), benchmark v0.1 (exists), agents venv (exists) |
| New files created | `agents/scripts/eval_benchmark.py` (NEW), 5 `edge_*.sol` + 5 `edge_*.json` (NEW), `agents/eval/baselines/pre_redesign.json` (NEW, baseline output). `run_real_audit.py` EXTENDED (not rewritten). |
| Tests added | The 5 edge cases double as regression tests (the comparator checks them every run). Plus ~6 unit tests for the comparator's metric math. |

---

## What this plan deliberately does NOT do

- Does NOT build the full Phase C.2 (100-200 contracts, AUC-PR, production eval framework) — that stays at WS6a, building on WS0.
- Does NOT add new ML training or data acquisition — reuses the existing benchmark v0.1.
- Does NOT modify the master plan's WS1-WS6 content — only inserts WS0 before them and renumbers the ordering table.
- Does NOT gate WS6b/WS6c (the tool-blocked phases) — they still wait on Halmos/Z3/Gigahorse/ItyFuzz/Anvil installs.
- Does NOT replace the existing 297 unit tests — they still run (plumbing checks are valuable, just not sufficient). WS0 adds the verdict-quality layer on top.

---

## Status

- **Decision-complete.** No open design decisions in WS0 itself.
- The one downstream decision already made (Ali, 2026-06-21): the `INCONCLUSIVE`
  verdict state (WS1.3) will be a **6th verdict value**, not a flag on `SAFE` —
  so downstream consumers can't silently treat it as safe. The comparator's
  positive/negative mapping (above) already accounts for this.
- Ready to execute when Ali gives the go. The first action is corpus curation
  (the 5 edge cases) + runner extension, in parallel.
