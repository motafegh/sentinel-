# Verdict Reconciliation Rewrite Plan (WS1.5) (2026-06-21)

**Status:** planning only — nothing here implemented yet.
**Owner:** Ali + Claude.
**Fixes:** Findings #13, #14, #15 from `04_LIVE_BASELINE_FINDINGS.md` (verified independently by Claude against the actual report JSONs + run log).
**Precedes:** everything else in WS1 — this is the single fix at the root cause.

## Why this doc exists

WS1's 7 changes (already implemented) fixed `consensus_engine`'s **internal** logic —
its skip bar, its SAFE→DISPUTED override, `compute_verdict()`'s INCONCLUSIVE path.
Those are necessary and working (319 tests pass, 36 borderline classes now
DISPUTED not SAFE in the no-LLM gate).

But the live LLM-on baseline (16 contracts, Ali's manual inspection + Claude's
independent verification) exposed that WS1's changes are **insufficient**: D1
("`consensus_engine` is the sole verdict authority") was implemented as a local
guard inside `consensus_engine`, never as **enforcement at the reconciliation
point** — the synthesizer's per-class verdict loop. Three symptoms, one root
cause, all at `nodes.py:1377-1399` + `nodes.py:1059-1063`:

| Finding | Severity | Verified count | Root cause |
|---|---|---|---|
| #13 — debate class cap blind to tool corroboration | HIGH | 15+ classes | cross_validator sorts by raw ML prob, ignores consensus_verdict |
| #14 — debate overrides consensus, undoing D1 | CRITICAL | 13 instances (4 at consensus conf=1.0) | synthesizer checks `pre_verdicts` FIRST, consensus second |
| #15 — consensus votes silently dropped from final report | HIGH | 7 instances (2 are GT classes) | synthesizer loops over `all_flagged` (ML≥0.25), not `union(all_flagged, consensus_verdict.keys())` |

Plus Claude's additional findings: #16 (debate only downgrades, never upgrades —
0 FPs, 13 FNs), #17 (debate can silently return empty with no persisted
transcript), #18 (narrative-verdict disconnect — self-contradicting reports),
#19 (37.5% of contracts have label=vulnerable but verdict=DISPUTED), #20
(`04_opaque_contract_factory` as the perfect-storm case where all 3 findings
hit one contract).

**This doc is the single coherent fix.** It does NOT patch each symptom
separately — it rewrites the reconciliation logic as one piece, because all
three symptoms converge at the same code location.

---

## The design decision — how should the debate and consensus interact?

The debate reads the actual source and reasons about it. `consensus_engine`
computes a weighted vote from ML + Slither + Aderyn (ML discounted). Both can
be wrong:

- **The debate can be wrong** (verified: 13 downgrades of correct consensus
  votes, 4 at consensus confidence=1.0 where all tools agreed — the debate's
  Defender systematically wins on actually-vulnerable contracts).
- **`consensus_engine` can be wrong** (the historical incident that motivated
  the current debate-first priority: `consensus_engine` said ExternalBug=SAFE
  on `safe_storage.sol` while `compute_verdict()` said DISPUTED — but that was
  BEFORE D1's SAFE→DISPUTED override, which now prevents consensus from
  silently SAFE-ing a flagged class).

The current code resolves this by always trusting the debate. That's the
wrong default for a security tool: the debate's 13 FNs vs 0 FPs shows it's
biased toward clearing real bugs, and the FN/FP asymmetry principle (Ali's
rule, written into `README.md` in WS1) says a missed vulnerability costs
millions while a wasted review costs time.

**The new design: the debate can UPGRADE, but can only DOWNGRADE to DISPUTED,
never to SAFE, when consensus voted non-SAFE.** The only way a flagged class
reaches SAFE is if BOTH consensus AND debate agree it's safe. This preserves
the debate's value (it can find things the tools missed — case 5) while
preventing its systematic FN bias from silently clearing real bugs.

---

## Change A — Fix the debate's input filter (Finding #13)

**File:** `agents/src/orchestration/nodes.py:1059-1063` (`cross_validator`)
**Current:**
```python
_max_classes = int(os.getenv("CROSS_VALIDATOR_MAX_CLASSES", "5"))
if len(all_flagged) > _max_classes:
    all_flagged = sorted(
        all_flagged, key=lambda v: v.get("probability", 0.0), reverse=True
    )[:_max_classes]
```
**New:**
```python
_max_classes = int(os.getenv("CROSS_VALIDATOR_MAX_CLASSES", "5"))
# Read consensus_verdict from state — consensus_engine already ran and its
# confidence incorporates tool corroboration (Finding #13: the current sort
# by raw ML prob misses tool-corroborated classes below the ML top-5).
consensus_verdict = state.get("consensus_verdict", {}) or {}
if len(all_flagged) > _max_classes:
    # Sort by consensus confidence (tool-corroborated ranks higher), fall
    # back to ML prob for classes consensus didn't vote on.
    all_flagged = sorted(all_flagged, key=lambda v: (
        consensus_verdict.get(v.get("vulnerability_class", ""), {}).get("confidence", 0.0),
        v.get("probability", 0.0),
    ), reverse=True)
    # Guarantee any class with a tool hit is adjudicated, regardless of rank.
    tool_classes = {
        c for c, v in consensus_verdict.items()
        if v.get("slither_match") or v.get("aderyn_match")
    }
    top = all_flagged[:_max_classes]
    for v in all_flagged[_max_classes:]:
        if v.get("vulnerability_class") in tool_classes:
            top.append(v)
    all_flagged = top
```
**Why:** a class with Slither + Aderyn agreement but low ML prob (e.g.
CallToUnknown at 0.249 on `04_opaque_contract_factory`) is exactly the class
the debate SHOULD adjudicate. The current filter excludes it because ML prob
is the only signal it uses — even though `consensus_engine` already computed
the combined evidence one node earlier.

---

## Change B — Rewrite the synthesizer's reconciliation loop (Findings #14, #15)

**File:** `agents/src/orchestration/nodes.py:1377-1399` (`synthesizer`)
**Current:** iterates `all_flagged` only; `if cls in pre_verdicts: debate wins / elif cls in consensus_verdict: consensus / else: compute_verdict`
**New:** iterates `union(all_flagged, consensus_verdict.keys())`; 8-case reconciliation rules.

### The loop driver (fixes #15)
```python
# Iterate the UNION of ML-flagged classes and consensus-voted classes.
# Fixes Finding #15: consensus votes on tool hits regardless of ML score,
# so a class with ML < 0.25 + tool corroboration was voted on correctly
# but never reached this loop (all_flagged only has ML >= 0.25).
all_flagged_set = {v.get("vulnerability_class") for v in all_flagged}
consensus_classes = set(consensus_verdict.keys())
all_classes = all_flagged_set | consensus_classes
# Build a cls → (prob, vuln_dict) lookup for classes in all_flagged
flagged_by_cls = {v.get("vulnerability_class"): v for v in all_flagged}

for cls in sorted(all_classes):
    vuln = flagged_by_cls.get(cls, {"vulnerability_class": cls, "probability": 0.0})
    prob = vuln.get("probability", 0.0)
    consensus_vote = consensus_verdict.get(cls)
    debate_verdict = pre_verdicts.get(cls)  # may be None

    verdict, sources = _reconcile_verdicts(
        cls, prob, consensus_vote, debate_verdict,
        static_findings, rag_results, path_taken,
    )
    verdicts[cls] = verdict
    confirmations[cls] = sources
    vuln_verdicts.append({...})
```

### The reconciliation function (fixes #14)
```python
def _reconcile_verdicts(
    cls: str,
    prob: float,
    consensus_vote: dict | None,      # consensus_verdict[cls] or None
    debate_verdict: str | None,        # pre_verdicts[cls] or None
    static_findings: list,
    rag_results: list,
    path_taken: str,
) -> tuple[str, list[str]]:
    """
    Reconcile consensus_engine's vote and the debate's verdict into one
    final verdict. Implements the FN/FP asymmetry principle: the debate
    can upgrade but can only downgrade to DISPUTED, never to SAFE, when
    consensus voted non-SAFE.
    """
    cv_verdict = consensus_vote.get("verdict") if consensus_vote else None
    cv_conf = consensus_vote.get("confidence", 0.0) if consensus_vote else 0.0
    deep_threshold = DEEP_THRESHOLDS.get(cls, 0.40)

    # Case 7: no consensus vote → debate is the only signal
    if consensus_vote is None:
        if debate_verdict is not None:
            return debate_verdict, [f"ml:{prob:.3f}", "debate"]
        # Case 8: neither → compute_verdict (last resort)
        return compute_verdict(cls, prob, static_findings, rag_results, path_taken)

    # Case 6: consensus voted, debate was silent (empty/timeout) → consensus stands
    if debate_verdict is None:
        return cv_verdict, [f"ml:{prob:.3f}", f"consensus:confidence={cv_conf:.2f}"]

    # Both voted — apply the 8-case reconciliation
    sources = [f"ml:{prob:.3f}", f"consensus:{cv_verdict}(conf={cv_conf:.2f})", f"debate:{debate_verdict}"]

    # Case 1: consensus CONFIRMED (conf >= 0.70) + debate SAFE/DISPUTED → keep CONFIRMED
    # All tools agreed — the debate cannot override.
    if cv_verdict == "CONFIRMED" and debate_verdict in ("SAFE", "DISPUTED", "WATCH"):
        return cv_verdict, sources + ["rule:consensus_confirmed_debate_cannot_downgrade"]

    # Case 5: consensus DISPUTED + debate CONFIRMED/LIKELY → take debate (upgrade)
    if cv_verdict == "DISPUTED" and debate_verdict in ("CONFIRMED", "LIKELY"):
        return debate_verdict, sources + ["rule:debate_upgrade"]

    # Case 2: consensus LIKELY + debate SAFE → DISPUTED (surface disagreement)
    if cv_verdict == "LIKELY" and debate_verdict == "SAFE":
        return "DISPUTED", sources + ["rule:disagreement_surfaces_as_disputed"]

    # Case 3: consensus LIKELY + debate DISPUTED → DISPUTED (agreement on "not confirmed")
    if cv_verdict == "LIKELY" and debate_verdict == "DISPUTED":
        return "DISPUTED", sources + ["rule:likely_downgraded_to_disputed"]

    # Case 4: consensus DISPUTED + debate SAFE → DISPUTED (uncorroborated ≠ cleared)
    if cv_verdict == "DISPUTED" and debate_verdict == "SAFE":
        return "DISPUTED", sources + ["rule:disputed_not_cleared_by_debate"]

    # Both agree (any combination not covered above) → return the agreement
    if cv_verdict == debate_verdict:
        return cv_verdict, sources + ["rule:both_agree"]

    # Any other disagreement → take the more severe (higher rank)
    # This is the default for cases not explicitly above (e.g. consensus WATCH
    # + debate CONFIRMED → CONFIRMED; consensus SAFE + debate CONFIRMED → CONFIRMED)
    cv_rank = OVERALL_VERDICT_RANK.get(cv_verdict, 0)
    debate_rank = OVERALL_VERDICT_RANK.get(debate_verdict, 0)
    if debate_rank > cv_rank:
        return debate_verdict, sources + ["rule:more_severe_wins_debate"]
    return cv_verdict, sources + ["rule:more_severe_wins_consensus"]
```

### The 8 cases as a table (for the README + tests)

| Case | Consensus | Debate | Final | Rule |
|---|---|---|---|---|
| 1a | CONFIRMED (conf≥0.70) | SAFE/WATCH/INCONCLUSIVE | **CONFIRMED** | Debate cannot clear or ignore unanimous tool agreement |
| 1b | CONFIRMED (conf≥0.70) | DISPUTED | **DISPUTED** | Debate read source + is uncertain — surface it (not "cleared") |
| 2 | LIKELY | SAFE | **DISPUTED** | Surface the disagreement |
| 3 | LIKELY | DISPUTED | **DISPUTED** | Agreement on "not confirmed" |
| 4 | DISPUTED | SAFE | **DISPUTED** | Uncorroborated ≠ cleared |
| 5 | DISPUTED | CONFIRMED/LIKELY | **debate** | Debate found something tools didn't |
| 6 | any | (none) | **consensus** | Debate was empty/timeout |
| 7 | (none) | any | **debate** | No tools voted — debate is only signal |
| 8 | (none) | (none) | **compute_verdict()** | Last resort (INCONCLUSIVE for flagged) |

**Key invariant:** a flagged class reaches SAFE only if BOTH consensus AND
debate agree it's safe (or if neither voted and `compute_verdict()` says SAFE
for a below-threshold class). The debate alone can never clear a class that
consensus flagged.

### Known trade-off: DISPUTED looks calmer than CONFIRMED in the headline

**Case 1b** surfaces the debate's uncertainty as DISPUTED instead of keeping
CONFIRMED. This is correct (the debate read the source, the tools are
syntactic) but has a UX consequence worth tracking: on `01_unbounded_refund`/
Reentrancy and `02_push_payment_failure`/Reentrancy (both confirmed-real bugs
from the GT label audit), `overall_verdict` now shows a less alarming top-line
label for a real bug. A reviewer skimming the headline would see DISPUTED
("uncertain") instead of CONFIRMED ("all tools agree") — they'd still need to
read the per-class verdict list to see the class is flagged. Not a bug — the
class is still in the report, still contributes to overall_verdict (DISPUTED
ranks above SAFE), still triggers the deep path. But "the headline looks
calmer than the contract actually is" is exactly the kind of thing that's easy
to forget about three workstreams from now. Worth revisiting when WS6a
(Phase C gateway) designs the user-facing report format.

---

## Change C — Persist the debate transcript (Finding #17)

**File:** `agents/src/orchestration/nodes.py` (`cross_validator` return) + `agents/scripts/run_real_audit.py` (report JSON)
**Current:** `debate_transcript` is computed in graph state but never written to the persisted report — empty-verdict cases (`07_multivuln_call_reentrancy`: all 3 roles ran, 63s, but `verdicts={}`) are undiagnosable.
**New:**
1. `cross_validator` returns `{"verdicts": ..., "debate_transcript": debate_transcript}` (already computed at nodes.py:1097, just not returned)
2. `synthesizer` passes `debate_transcript` through to `final_report`
3. `run_real_audit.py` captures `debate_transcript` in the report JSON

**Why:** when the debate returns empty (JSON parse failure, model hallucinated non-JSON, all-SAFE debate), there's currently no way to see what the 3 roles actually said. The transcript is the only diagnostic. This is a small change — the data already exists in graph state, it just needs to flow to the persisted output.

---

## Tests

### New: `tests/test_verdict_reconciliation.py`
One test per case in the 8-case table:
- `test_case1_consensus_confirmed_debate_cannot_downgrade`
- `test_case2_likely_vs_safe_disputes`
- `test_case3_likely_vs_disputed_agrees_on_disputed`
- `test_case4_disputed_vs_safe_stays_disputed`
- `test_case5_disputed_vs_confirmed_upgrades`
- `test_case6_consensus_only_when_debate_silent`
- `test_case7_debate_only_when_consensus_silent`
- `test_case8_compute_verdict_when_neither`
Plus:
- `test_loop_includes_consensus_only_classes` (Finding #15: a class with ML < 0.25 + tool hit gets a final verdict)
- `test_debate_cannot_safe_a_consensus_flagged_class` (the core invariant)
- `test_confidence_1_0_never_downgraded` (Finding #14 worst case)

### Update: `tests/test_consensus_voting.py`
- Add: `test_tool_corroborated_class_below_ml_025_gets_final_verdict` (Finding #15)

### Update: `tests/test_smoke_e2e.py`
- Verify the reconciliation loop runs on the full graph (existing tests should pass — the fast path is unchanged; deep path now uses reconciliation)

---

## Gate assertions (refine `eval_benchmark.py`)

| Gate | What it checks | Fixes |
|---|---|---|
| **WS1a (refined)** | No flagged class ends SAFE with no recorded reason. Uses `consensus_verdict[cls]` from the report: a violation is `final_verdict == SAFE + consensus_vote non-SAFE + no debate agreement`. | #14 |
| **WS1b** | `edge_debate_timeout` emits INCONCLUSIVE (LLM-on only) | existing |
| **WS1c (new)** | No consensus vote is missing from final verdicts: `set(consensus_verdict.keys()) - set(vulnerability_verdicts.classes) == empty` | #15 |
| **WS1d (new)** | No consensus confidence=1.0 class ends in SAFE | #14 worst case |
| **WS1e (new)** | No contract has `overall_label=confirmed_vulnerable` + `overall_verdict=SAFE` | #19 |
| **WS2** | Zero FPs on safe subset | existing |
| **WS3** | Long-contract bug detected | existing |
| **macro_f1** | ≥ baseline | existing |

---

## Validation

1. **Full test suite:** `agents/.venv/bin/pytest agents/tests/ -q` → must stay green (319+ tests with new additions).
2. **No-LLM gate:** re-run the 88-contract `--no-llm` gate. Expected: WS1c PASS (no missing consensus votes), WS1d PASS (no conf=1.0 → SAFE), macro-F1 ≥ baseline (0.2455). The no-LLM gate can't test the debate-override cases (debate doesn't run) but validates the loop boundary fix (#15).
3. **LLM-on gate:** re-run the 88-contract LLM-on gate (~100 min with GPU). Expected:
   - The 13 debate-override FNs → now DISPUTED or kept at consensus verdict
   - The 2 missing GT classes (CallToUnknown on `02_dynamic_dispatch` + `04_opaque_contract_factory`) → now visible in final verdicts
   - macro-F1 improves (fewer FNs)
   - WS1d PASS (no conf=1.0 → SAFE)
   - WS1e PASS (no label=vulnerable + verdict=SAFE)
4. **Spot-check:** manually inspect `01_flash_loan_oracle_manipulation` + `04_opaque_contract_factory` reports — the ExternalBug/CallToUnknown verdicts should now be non-SAFE.

---

## What this does NOT change

- Does NOT remove the debate (it still runs, still reads source, still can UPGRADE via case 5)
- Does NOT make consensus_engine infallible (it's still ML-discounted; the debate can still upgrade DISPUTED→CONFIRMED)
- Does NOT change the ML model
- Does NOT touch RAG (WS2) or debate source access (WS3) or debate max_tokens (WS4.1)
- Does NOT change the fast path (classes below DEEP_THRESHOLD → SAFE is correct)
- Does NOT change `compute_verdict()` (already fixed in WS1 — INCONCLUSIVE for flagged)

---

## Sequencing

This is **WS1.5** — it goes after WS1's 7 changes (already done) and before
WS4.1/WS2. The order within the remaining WS1 work:

1. **WS1 (done):** consensus_engine internal logic + compute_verdict + Aderyn fix
2. **WS1.5 (this doc):** synthesizer reconciliation rewrite + debate input filter + transcript persistence
3. **WS4.1:** debate max_tokens cap (cheap, independent)
4. **WS2:** remove fake RAG

WS1.5 must complete before the LLM-on baseline is meaningful — without it, the
debate's systematic FN bias dominates every LLM-on run.

---

## Status

- **Decision-complete.** The 8 reconciliation rules are explicit; the code changes are concrete; the tests + gates are specified.
- The one design decision (debate can upgrade but only downgrade to DISPUTED, never SAFE) is the direct consequence of the FN/FP asymmetry principle Ali established — not a new decision, just its enforcement at the reconciliation point.
- Ready to execute when Ali gives the go.
