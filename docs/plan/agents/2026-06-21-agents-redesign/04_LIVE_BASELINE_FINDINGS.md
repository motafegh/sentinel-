# Live Baseline Findings — 2026-06-21

**Source:** manual, line-by-line inspection of the `--llm-on` baseline run (`agents/eval/runs/baseline_llm_pre_redesign_20260621T224353Z/`), started by Ali to see how the system actually behaves on real source + real debate, stopped partway through (16 of 88 contracts completed). For each report, ground truth was read directly from the contract's `// expect:` header in `manual_hand_written_contracts/`, then compared against `final_report.vulnerability_verdicts`, `consensus_verdict`, and the run log, line by line.

Every claim below is reproducible: contract path + report path + exact code lines are given. Nothing here is inferred from docstrings or comments — only from actual JSON output, log lines, and source code that was read directly.

This is additive to `00_FINDINGS.md` (continues its numbering at #13) and feeds `01_MASTER_PLAN.md`'s WS1.

---

## Finding #13 — The debate's class cap is blind to tool corroboration (Severity: HIGH)

**Where:** `agents/src/orchestration/nodes.py:1059-1061` (`cross_validator`)

```python
_max_classes = int(os.getenv("CROSS_VALIDATOR_MAX_CLASSES", "5"))
if len(all_flagged) > _max_classes:
    all_flagged = sorted(all_flagged, key=lambda v: v.get("probability", 0.0), reverse=True)[:_max_classes]
```

**What happens:** the debate only adjudicates the top-5 classes **by raw ML probability**. It runs *after* `consensus_engine` in the graph, but has no access to (and ignores) what `consensus_engine` already learned — specifically, whether Slither/Aderyn independently corroborated a class regardless of its ML score.

**Evidence (two real contracts, same pattern):**

| Contract | Ground truth | ML prob | In debate's top-5? | What rescued it |
|---|---|---|---|---|
| `manual_hand_written_contracts/CallToUnknown/01_proxy_delegatecall.sol` | CallToUnknown | 0.272 (ranked 8th/10) | No | `consensus_engine`'s tool-corroborated vote (Slither hit) |
| `manual_hand_written_contracts/CallToUnknown/06_tricky_call_in_fallback.sol` | CallToUnknown | 0.261 (ranked 8th/10) | No | Same — tool-corroborated `consensus_engine` vote |

**Why it matters:** the cap exists to bound debate cost on ambiguous contracts (see the comment directly above it, dated 2026-06-21, explaining the FAST-model-timeout risk that motivated it — that reasoning is sound). The bug is that it ranks by the *one signal the whole redesign agreed not to trust alone* (raw ML probability), with zero weight given to independent tool evidence that `consensus_engine` already computed one node earlier. Both contracts above were only rescued by chance — because a tool *also* happened to fire AND because the fallback chain (see Finding #14) happened to surface that vote correctly this time. If a real vulnerability is ML-underscored *and* missed by Slither/Aderyn, nothing in the current design would ever bring it in front of the debate.

**Suggested direction:** rank the debate's candidate-class list by something that already reflects `consensus_engine`'s combined evidence (e.g. its `score` or `confidence` field) rather than raw ML probability, or at minimum guarantee any class with a tool hit is included regardless of rank.

---

## Finding #14 — The debate's own verdict outranks `consensus_engine`, silently undoing D1 (Severity: CRITICAL)

**Where:** `agents/src/orchestration/nodes.py:1377-1392` (`synthesizer`)

```python
for vuln in all_flagged:
    cls  = vuln.get("vulnerability_class", "?")
    prob = vuln.get("probability", 0.0)
    if cls in pre_verdicts:                # 1. debate's verdict — checked FIRST
        verdict = pre_verdicts[cls]
        sources = pre_confirmations.get(cls, [f"ml:{prob:.3f}"])
    elif cls in consensus_verdict:         # 2. consensus_engine — only if debate silent
        verdict = consensus_verdict[cls]["verdict"]
        sources = [f"ml:{prob:.3f}", f"consensus:confidence={consensus_verdict[cls]['confidence']:.2f}"]
    else:                                  # 3. compute_verdict() rule fallback
        verdict, sources = compute_verdict(cls, prob, static_findings, rag_results, path_taken)
```

`pre_verdicts = state.get("verdicts", {})` is written by `cross_validator`'s debate. It is checked **before** `consensus_verdict`. This is the opposite of what `01_MASTER_PLAN.md`'s D1 resolved: *"consensus_engine becomes the sole verdict authority."* D1 was implemented as a guard *inside* `consensus_engine` (forcing its own SAFE→DISPUTED when uncorroborated — verified working in `00_FINDINGS.md`/WS1's regression tests), but that guard only protects `consensus_engine`'s own output. Nothing stops the debate, sitting at higher priority here, from overriding a correct `consensus_engine` vote with a wrong one of its own, and there is no equivalent "can't silently land on SAFE" guard on the debate's path.

**Live evidence — this is not hypothetical, it happened in the run Ali just executed:**

Contract: `manual_hand_written_contracts/ExternalBug/01_flash_loan_oracle_manipulation.sol` (ground truth: `ExternalBug` — a flash-loan oracle manipulation pattern).

- ML probability: **0.636** (well above the 0.40 investigation threshold for this class, and inside the debate's top-5 — Finding #13 did not apply here).
- `consensus_verdict.ExternalBug` (computed correctly, in the raw report JSON): `{"verdict": "LIKELY", "aderyn_match": 1, "ml_signal": 1, "confidence": 0.5745}` — Aderyn independently corroborated it.
- **Final verdict actually used** (`final_report.vulnerability_verdicts`, the field everything downstream reads): `{"verdict": "SAFE", "evidence_sources": ["ml:0.636"], "confidence": 0.6014}` — no consensus tag. This is the debate's own answer, and it is wrong.
- Report file: `agents/eval/runs/baseline_llm_pre_redesign_20260621T224353Z/01_flash_loan_oracle_manipulation_report.json`.

**Why it matters — this is the exact failure mode the whole redesign exists to prevent:** a real, ground-truth vulnerability, correctly flagged by ML and correctly corroborated by an independent static tool, was overruled to SAFE by the debate, and nothing in the current architecture stops that from happening or even flags it as a disagreement worth surfacing. This is a live false negative, not a theoretical gap.

Note on history: the comment block directly above this code (`nodes.py:1355-1368`) explains this priority order was deliberately chosen after a *different*, earlier incident (`consensus_engine` was wrong, `compute_verdict()` was right, on `safe_storage.sol`). That fix was reasonable at the time — it was made before D1 existed. It was never revisited after D1 was decided, so the two design decisions now directly contradict each other, and the contradiction was never caught because no test exercises "debate disagrees with a tool-corroborated consensus vote."

**Suggested direction:** when the debate's verdict and `consensus_engine`'s verdict disagree on whether a class clears the safety bar (e.g. debate says SAFE, consensus says LIKELY/CONFIRMED with real tool corroboration), the more severe of the two should win, or at minimum the disagreement itself should surface as a distinct state (this is exactly what `DISPUTED` is for) rather than silently picking the debate's answer. This needs a real decision, not a quick patch — flagging for `01_MASTER_PLAN.md`.

---

## Finding #15 — `consensus_engine`'s votes can be silently dropped from the final report (Severity: HIGH)

**Where:** `agents/src/orchestration/nodes.py:1377` (`synthesizer`) — the loop driver itself.

```python
all_flagged = confirmed + suspicious or ml_result.get("vulnerabilities", [])
...
for vuln in all_flagged:
```

`confirmed` = ML ≥0.55, `suspicious` = ML 0.25–0.54 (`nodes.py:1525`). So `all_flagged` only contains classes with **ML probability ≥ 0.25**. But `consensus_engine` votes whenever `prob >= DEEP_THRESHOLDS[cls]` **OR a tool hit exists** — meaning it can and does vote on classes below 0.25 purely on tool evidence. The `synthesizer` loop only ever visits `all_flagged`, never `consensus_verdict.keys()` directly — so a class consensus_engine voted on correctly, but whose ML score sits under 0.25, never gets a verdict entry in `vulnerability_verdicts`, never contributes to `overall_verdict`, and never reaches the narrative. The vote is computed right and then never read by anything downstream.

**Live evidence (three contracts, same pattern, all with ground-truth class missing entirely from `vulnerability_verdicts` — not SAFE, not DISPUTED, simply absent):**

| Contract | Ground truth | ML prob | Result |
|---|---|---|---|
| `manual_hand_written_contracts/CallToUnknown/02_dynamic_dispatch.sol` | CallToUnknown | 0.245 | Missing from final verdicts |
| `manual_hand_written_contracts/CallToUnknown/04_opaque_contract_factory.sol` | CallToUnknown | 0.249 | Missing from final verdicts |
| `manual_hand_written_contracts/DenialOfService/06_tricky_dos_in_constructor.sol` | DenialOfService | (below floor) | Missing from final verdicts |

Reports: same baseline run directory, `02_dynamic_dispatch_report.json`, `04_opaque_contract_factory_report.json`, `06_tricky_dos_in_constructor_report.json`.

**Why it matters:** this is structurally different from Finding #14 (which is about *which* verdict wins when both exist) — this is about a vote that exists correctly in the raw JSON (`consensus_verdict`) but is invisible to every consumer that reads the "official" verdict list (the narrative, the visualizer, on-chain submission, this very `eval_benchmark.py` comparator). A correct answer was computed and thrown away by a loop-boundary mismatch, not by a logic error in the vote itself.

**Suggested direction:** the loop driver needs to be the union of `all_flagged` and `consensus_verdict.keys()`, not `all_flagged` alone — otherwise `consensus_engine`'s "votes on tool hits regardless of ML score" design (which is deliberate and correct) is undermined by the very next node.

---

## Finding #16 — The "exceeded 512 tokens" warning is computed from the wrong window, not from a missing feature (Severity: MEDIUM)

**Where:** `ml/src/inference/predictor.py:683` (`_score_windowed`) and `ml/src/inference/preprocess.py:612` (`_tokenize_sliding_window`).

**Context — the windowing design is real and correctly built:** `predict_source()` (`predictor.py:585`) calls `process_source_windowed()` (`preprocess.py:338`), which splits long contracts into up to 4 overlapping 512-token windows (stride 256, ~50% overlap), and when there are more candidate windows than 4, deliberately subsamples via `np.linspace` (`preprocess.py:592`) so the selected windows cover the **beginning, middle, and end** of the contract. The model then scores all 4 windows in a single forward pass through `WindowAttentionPooler`'s real multi-window attention (`predictor.py:627-683`) — the same shape used in training, not a fallback or approximation. This is a genuine, deliberate design (the docstring at `preprocess.py:346-363` cites it as fix "T1-C"), and for most contracts it does cover the tail.

**The bug:** each window gets its own `truncated` flag — `(start + _CONTENT_CAP < total_content)` (`preprocess.py:612`) — meaning "did *this window alone* reach the end of the contract." Window 0 always starts at position 0, so its flag is `True` any time the contract exceeds ~510 tokens **at all**, regardless of whether windows 1-3 (placed at the middle/end by the linspace selection) actually covered the rest. Then:

```python
# predictor.py:683
return self._format_result(graph, probs, windows[0], n_real)
```

`_score_windowed` passes **only `windows[0]`** into `_format_result`, which reports that window's narrow, single-window-scoped flag as the contract-level "truncated" result. The downstream warning text (`agents/src/orchestration/nodes.py:1690-1691,1941`, *"Contract exceeded 512 CodeBERT tokens — tail code unanalysed"*) is generated directly from this flag.

**Net effect:** the warning fires on **every** contract that enters the multi-window path at all (>~510 tokens), even when the linspace-selected tail window genuinely covered the contract's end. The claim in the warning text ("tail unanalysed") is usually false when it fires — it should only fire when the contract's length exceeds the *combined* coverage of all 4 selected windows, not whenever it exceeds one window's individual capacity.

**Evidence:** observed live on real reports, e.g. `agents/test_audit_reports/01_approve_frontrun_report.json:654` and `06_tricky_timestamp_in_pricing_report.json:1007` — both well within the model's actual ~1280-token combined-window coverage budget, both flagged with the tail-unanalysed warning anyway.

**Why MEDIUM not HIGH:** the underlying analysis is not actually degraded — the model genuinely scores the tail in the cases checked. The harm is purely in the reported warning misleading a human reviewer (or a downstream automated consumer) into discounting a verdict, or manually re-reviewing content that was, in fact, already covered.

**Suggested fix:** compute the overall `truncated` flag from whichever selected window has the largest `start + _CONTENT_CAP` (i.e. the one nearest the tail — guaranteed to exist after the linspace selection), not from `windows[0]`. One-line change at the call site (`predictor.py:683`), no retraining or schema change required.

---

## Common root cause across #13, #14, #15

D1 ("`consensus_engine` is the sole verdict authority") was implemented as a **local guard inside `consensus_engine`** (its own SAFE→DISPUTED override, verified working). It was never implemented as an **enforcement rule at the point where competing signals actually get reconciled** — `synthesizer`'s per-class loop. That loop currently has three independent gaps, found empirically by running real contracts through the live pipeline, not by reading code in isolation:
- It lets the debate outrank `consensus_engine` instead of the reverse (#14).
- It iterates a list that doesn't cover everything `consensus_engine` voted on (#15).
- The debate it's deferring to has its own pre-existing blind spot in what it even gets to look at (#13).

All three point at the same fix location: `synthesizer`'s verdict-reconciliation logic (`nodes.py:1377-1399`) needs to be rewritten as a single coherent priority/merge rule that actually reflects D1, not patched a third time. Recommend this becomes its own WS1 sub-item, ahead of anything else in WS1, since it is the one place all three findings converge.

---

## Two additional, smaller observations from the same inspection (not yet full findings — noted for completeness)

- **No persisted debate transcript.** On `manual_hand_written_contracts/CallToUnknown/07_multivuln_call_reentrancy.sol`, all 3 debate roles completed cleanly within budget (63.0s total, no timeout, no exception logged) but `cross_validator complete | verdicts={}` — an empty result with no diagnosable cause, because `debate_transcript` is computed transiently in graph state but never written into `final_report` or the persisted JSON (`agents/data/reports/{address}.json`). Worth fixing alongside #13-#15 since debugging any of them benefits from being able to see what the debate actually said.
- **Corpus composition note (not a bug):** this baseline run used `manual_hand_written_contracts/` (83 hand-written contracts) + 5 edge cases, not the official OOD-verified `data_module/benchmarks/benchmark_v0.1_quickstart/` (66 contracts) that `03_GATE_INFRASTRUCTURE_PLAN.md` named as the reusable asset. Confirmed with Ali this was deliberate — he wants to see how the system behaves on these specific hand-written, deliberately tricky contracts first. Recorded here only so a future reader doesn't assume this baseline number is comparable to one run against the official benchmark.
