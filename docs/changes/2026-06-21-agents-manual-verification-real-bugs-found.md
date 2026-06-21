# Agents Module — Manual Tool Verification: Real Bugs Found & Fixed (2026-06-21)

**Trigger:** Ali asked to "fully test agents module with real contracts," then —
after I asserted a wrong claim about solc-select without checking — corrected the
methodology: manually establish ground truth on 1-2 contracts first, run Slither/
Aderyn DIRECTLY and inspect raw output, only THEN validate the agents module against
both. This methodology found two critical, previously-invisible bugs that had made
Slither and Aderyn non-functional in the agents pipeline since inception.

**Full incremental log:** `~/.claude/scratch/agents_manual_tool_verification_20260621.md`
(every observation, assumption, and correction documented as it happened — including
the correction where Ali caught me asserting something wrong).

---

## Bug 1 — Slither never registered any detectors (3 call sites)

`Slither(tmp_path)` registers **zero** detectors on construction — `sl._detectors`
starts empty. The Slither CLI explicitly calls `slither.register_detector(cls)` for
every detector class before running; the agents code went straight from
construction to filtering an already-empty list. **Result: Slither has returned
zero findings on every contract, in every audit, since `static_analysis`,
`quick_screen`, and `graph_inspector_server`'s hotspot fallback were written** —
not a tool/dependency problem, a missing API call.

**Verified by direct reproduction:** running `slither contract.sol` from the CLI on
a textbook reentrant Vault found `reentrancy-eth`; calling the exact same node code
in-process found nothing, until adding the registration loop.

**Fixed in:** `agents/src/orchestration/nodes.py` (`static_analysis`, `quick_screen`),
`agents/src/mcp/servers/graph_inspector_server.py` (hotspot fallback path).

**Secondary fix (same investigation):** 3 of 11 names in `routing.py:CLASS_TO_DETECTORS`
referenced Slither detectors that don't exist in version 0.11.5 — `integer-overflow`
and `toctou` were removed upstream (no replacement), `reentrancy-events-and-order`
was renamed to `reentrancy-events`. Fixed in `routing.py` and the parallel map in
`graph_inspector_server.py`.

## Bug 2 — Aderyn invocation broken in 3 compounding ways

1. `_run_aderyn_on_file` passed a **file** path as Aderyn's `[ROOT]` argument.
   Aderyn 0.6.8 requires a **directory** — exit code 1, "Not a directory".
2. `--output json` passed the literal word `"json"` as the output file path.
   Aderyn's `--output` flag takes a real file path (e.g. `report.json`), not a
   format selector — even with bug 1 fixed, this would silently produce a
   markdown-formatted file misnamed `json`.
3. The JSON parser assumed a schema (`{"high": [...], "medium": [...], "low": [...]}`
   with `id`/`line`/`function_name` fields) that doesn't exist. Real schema
   (verified by inspecting an actual `report.json`): `{"high_issues": {"issues":
   [{"detector_name", "title", "description", "instances": [{"line_no", ...}]}]},
   "low_issues": {...}}`. No "medium" bucket; no function-name field anywhere.

All three combined meant exit-code-1 fired first and was silently swallowed by the
existing `if returncode != 0: return []` — **Aderyn has never produced a single
finding through the agents module, on any contract, ever.**

**Fixed in:** `agents/src/orchestration/nodes.py` — rewrote `_run_aderyn_on_file` to
write source into a real temp directory, invoke with a real `--output <dir>/report.json
<dir>`, and added `_parse_aderyn_report()` matching the verified schema. `quick_screen`'s
independent (identically broken) inline Aderyn call was replaced with a delegate to
the fixed helper. Also removed `_SCREEN_ADERYN_HIGH_IDS` — dead code, never referenced,
and based on a wrong assumption (it encoded "H-1".."C-3" position labels as if they
were stable rule IDs; they renumber per report).

## Why the existing test suite never caught this

Every existing test touching Slither/Aderyn mocks them entirely (`test_smoke_e2e.py`'s
`mock_slither_cls`). No test exercised the real libraries end-to-end. Added 6 new
regression tests using the **real, non-mocked** binaries against
`test_contracts/vulnerable_reentrant.sol` (positive control) and a trivial safe
contract (negative control):
- `tests/test_static_analysis_real_slither.py` (3 tests)
- `tests/test_static_analysis_real_aderyn.py` (3 tests)

Full suite: **284 passed** (278 baseline this session → +6).

## Live pipeline validation, before vs. after fix

Re-ran the real end-to-end pipeline (`scripts/run_real_audit.py`, real LM Studio,
real MCP servers, real ML inference) on both contracts after the fix:

### `vulnerable_reentrant.sol` (ground truth: TRUE Reentrancy — checks-effects-
interactions violated, external call before balance update)
| | Before fix | After fix |
|---|---|---|
| Slither findings | 0 | 1 (`reentrancy-eth`, High) |
| Aderyn findings | 0 | 7 (2 High incl. `reentrancy-state-change`, 5 Low) |
| `quick_screen` escalation | never fires | fires correctly |
| `consensus_engine` ExternalBug/Reentrancy | no tool corroboration possible | `ml_signal=1, slither_match=1, aderyn_match=1, confidence=1.0` |
| `overall_verdict` | DISPUTED (rule-based, no corroboration) | **CONFIRMED** (correct) |

### `safe_storage.sol` (ground truth: SAFE — zero external calls anywhere; the
ML's ExternalBug CONFIRMED p=0.818 is the documented Run 12 false positive)
| | Before fix | After fix |
|---|---|---|
| Slither findings (ExternalBug-scoped) | 0 (meaninglessly — registration was broken) | 0 (now a real, verified zero) |
| Aderyn findings | 0 | 4 (all Low, none High) |
| `consensus_engine` ExternalBug | n/a | `ml_signal=1, slither_match=0, aderyn_match=0, confidence=0.19, verdict=SAFE` — correct |
| `overall_verdict` | SAFE (debate succeeded, overrode ML from source) | DISPUTED (debate timed out under load; rule-based fallback disagrees with consensus_engine — see gap below) |

## Two design gaps found and documented (not fixed — flagging for a decision)

1. **`consensus_engine`'s vote is never wired as the verdict fallback.** When
   `cross_validator`'s debate fails, `synthesizer` falls through to its own,
   separate, pre-Phase-A `compute_verdict()` (routing.py) — bypassing
   `consensus_engine`'s already-computed vote entirely. On `safe_storage.sol` this
   run, `consensus_engine` correctly said SAFE while the reported `overall_verdict`
   was DISPUTED — two valid-but-different philosophies (cautious-flag-for-review
   vs. ML-discounted-clear) disagreeing in the same report, with no current wiring
   to reconcile them.
2. **The narrative can hallucinate vulnerabilities from RAG reference material.**
   On `safe_storage.sol`, the LLM narrative asserted a "Reentrancy" risk and
   "arbitrary external calls" — neither exists in this contract (zero external
   calls; the verdicts list explicitly says `Reentrancy: SAFE`). Root cause: the
   prompt includes RAG chunks without clearly labeling them as general background,
   not site-specific findings; the FAST model conflated the two.
3. `reflection` caught gap #1's verdict-level tension precisely (its LLM summary
   called out the DISPUTED-vs-failure-mode contradiction) but has no visibility
   into the narrative text, so it could not have caught gap #2.

Both gaps are architecture/prompt-design decisions, not clear-cut bugs — left for
Ali to decide direction rather than silently patched.
