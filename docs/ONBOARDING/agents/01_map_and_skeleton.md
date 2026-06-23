# Session 1 — Map & Skeleton (walkthrough note)

**Date:** 2026-06-22
**Scope:** Agents module 13-node graph, `AuditState` schema, graph builder internals.

---

## What this session covered

1. The actual graph topology (lines and node order in source).
2. The full `AuditState` schema (26 fields, lifecycle of each).
3. The graph builder + lazy `audit_graph` + SqliteSaver checkpointer.
4. The conditional-edge function `_route_from_evidence_router`.
5. The evidence-gathering nodes: `ml_assessment`, `quick_screen`, `rag_research`, `static_analysis`, `graph_explain`, `audit_check`.
6. The first half of `cross_validator` (the LLM-adjudicated debate node).

---

## Files read

| File | Lines | What I got from it |
|------|-------|---------------------|
| `agents/src/orchestration/graph.py` | 280 | Graph topology, conditional edges, lazy `audit_graph`, SqliteSaver wiring |
| `agents/src/orchestration/state.py` | 202 | AuditState schema + per-field lifecycle comments |
| `agents/src/orchestration/nodes.py` (1-1175) | first 1175 of 2280 | Evidence-gathering nodes + start of cross_validator |

I did NOT yet read: `consensus_engine` (line 1987+), `synthesizer` (line 1571+), `reflection` (2058+), `explainer` (2208+), `visualizer` (2250+), or `routing.py`. Those are covered in Sessions 2 and 4-5.

---

## The 13 nodes — where they live in code

| # | Node | `nodes.py` line | Sets state field(s) | Calls |
|---|------|-----------------|---------------------|-------|
| 1 | `ml_assessment` | 361 | `ml_result`, `error` | MCP `:8010` `predict` |
| 2 | `quick_screen` | 191 | `quick_screen_hits` | direct Slither + Aderyn (Tier 0) |
| 3 | `evidence_router` | 316 | `routing_decisions` (append-reducer) | none — pure logging |
| 4 | `rag_research` | 434 | `rag_results`, `error` | MCP `:8011` `search` |
| 5 | `static_analysis` | 729 | `static_findings`, `external_call_summary`, `error` | direct Slither (scoped) + Aderyn |
| 6 | `graph_explain` | 914 | `ml_hotspots`, `graph_explanations` | MCP `:8013` `get_graph_hotspots` |
| 7 | `audit_check` | 518 | `audit_history`, `error` | MCP `:8012` `get_audit_history` |
| 8 | `consensus_engine` | 1987 | `consensus_verdict`, `confidence_by_class` | pure (weighted vote) |
| 9 | `cross_validator` | 1013 | `verdicts`, `confirmations`, `contradictions`, `debate_transcript` | LLM (FAST model, 3-role debate) |
| 10 | `synthesizer` | 1571 | `final_report`, `narrative`, `verdicts` | LLM (STRONG) for narrative only |
| 11 | `reflection` | 2058 | `reflection_notes` | LLM (optional) + rule-based |
| 12 | `explainer` | 2208 | `metric_attribution` (folds into `final_report`) | pure |
| 13 | `visualizer` | 2250 | `hotspot_visualization` | pure HTML render |

---

## The graph topology (literal edges from `graph.py`)

```
ml_assessment  →  quick_screen  →  evidence_router
                                       │
                       ┌───────────────┴───────────────┐
                       │ conditional:                  │
                       │  list[str] = deep path fan-out │
                       │  "synthesizer" = fast path     │
                       └───────────────┬───────────────┘
                                       │
         ┌──────────────┬──────────────┴──────────────┐
         ▼              ▼                             ▼
   rag_research   static_analysis              graph_explain
         │              │                             │
         └──────────────┼──────────────┬──────────────┘
                        ▼              ▼
                   audit_check  (fan-in here, all three)
                        │
                        ▼
                 consensus_engine     ← A.6/A.7
                        │
                        ▼
                 cross_validator      ← A.4 debate
                        │
                        ▼  (fast path skips straight here too)
                    synthesizer
                        │
                        ▼
                    reflection         ← A.3
                        │
                        ▼
                    explainer          ← A.8
                        │
                        ▼
                    visualizer         ← A.9
                        │
                        ▼
                       END
```

**Key design choices visible in the code:**

1. **Single fast-path entry, single deep-path entry.** `evidence_router` is the only conditional point. Every other node has a fixed edge in or out.
2. **Fan-in at `audit_check`.** The three deep-path nodes (rag_research, static_analysis, graph_explain) all converge at audit_check. LangGraph's fan-in semantics handle the "wait for all three" behavior automatically.
3. **The enrichment chain (`reflection → explainer → visualizer`) is unconditional.** Both fast and deep paths reach `synthesizer`, so the enrichment chain runs for every audit. This is why the reflection/explainer/visualizer outputs end up in every `final_report`.
4. **`_route_from_evidence_router` returns either `str` or `list[str]`.** Returning a list is LangGraph's fan-out signal. Returning a string is a single-edge signal. The function is pure (no state writes) — the evidence_router NODE logs decisions to state, the FUNCTION chooses the branch.

---

## The two-signal fast-path gate (lines 99-137)

```
fast path requires BOTH:
  1. ML all class probabilities < DEEP_THRESHOLDS
  2. quick_screen zero High/Critical Slither/Aderyn hits

If ML safe but quick_screen fired:
  → "screen-escalated" deep path: static_analysis only
  → graph_explain + rag_research still added via `sorted(set(active + ["graph_explain"]))`

If both safe:
  → fast path → synthesizer directly
```

**Why two signals and not one:** A contract that scores safe on ML but has Slither/Aderyn hits is exactly the case where ML is wrong (e.g. ExternalBug — typed interface calls look safe to GNN, but the static tool sees something). Two tools disagreeing warrants deeper scrutiny. The asymmetry (ML safe + screen hit → still deep) is the design choice that makes the system not over-trust ML.

---

## AuditState — 26 fields, 4 phases

| Phase | Field | Set by | Read by | Type |
|-------|-------|--------|---------|------|
| Input | `contract_code` | caller | every node | `str` |
| Input | `contract_address` | caller | `audit_check` | `str` |
| ML | `ml_result` | ml_assessment | rag_research, static_analysis, cross_validator, synthesizer, consensus_engine | `dict` |
| ML | `ml_hotspots` | graph_explain | synthesizer, visualizer | `list[dict]` |
| Routing | `routing_decisions` | evidence_router (+ append-reducer) | synthesizer | `list[str]` (append) |
| Screen | `quick_screen_hits` | quick_screen | evidence_router (logic), graph.py (route) | `dict` |
| Static | `static_findings` | static_analysis | cross_validator, synthesizer, reflection | `list[dict]` |
| Static | `external_call_summary` | static_analysis (ExternalBug only) | rag_research (query), synthesizer (prompt) | `list[dict]` |
| RAG | `rag_results` | rag_research | cross_validator, synthesizer, reflection | `list[dict]` |
| On-chain | `audit_history` | audit_check | cross_validator, synthesizer, reflection | `list[dict]` |
| Decision | `verdicts` | cross_validator (deep) OR synthesizer (fallback) | synthesizer | `dict[str,str]` |
| Decision | `confirmations` | cross_validator | synthesizer, explainer | `dict[str,list[str]]` |
| Decision | `contradictions` | cross_validator | synthesizer, reflection | `dict[str,list[str]]` |
| Output | `final_report` | synthesizer (+ folded by explainer) | caller | `dict` |
| Output | `narrative` | synthesizer (LLM success) | final_report | `str \| None` |
| Output | `error` | any node | synthesizer | `str \| None` |
| Phase A | `consensus_verdict` | consensus_engine | cross_validator (sort key) | `dict[str,dict]` |
| Phase A | `confidence_by_class` | consensus_engine | synthesizer (folded) | `dict[str,float]` |
| Phase A | `debate_transcript` | cross_validator | final_report | `dict[str,str]` |
| Phase A | `reflection_notes` | reflection | final_report | `dict` |
| Phase A | `metric_attribution` | explainer | final_report | `dict[str,dict]` |
| Phase A | `hotspot_visualization` | visualizer | file + final_report | `str \| None` |
| Phase B | `symbolic_findings` | (Halmos, TBD) | synthesizer, reflection | `list[dict]` |
| Phase B | `bytecode_analysis` | (Gigahorse, TBD) | synthesizer | `dict` |
| Phase B | `taint_flows` | (TBD) | synthesizer | `list[dict]` |
| Phase B | `permission_graph` | (access control, TBD) | synthesizer | `dict[str,list[str]]` |

**Reducer:** only `routing_decisions` uses `Annotated[list, operator.add]` — every other field is replaced on update (default LangGraph behavior). This means a node must NOT re-emit a list it wants preserved; it should only return the fields it actually changed.

**`total=False`:** every field is optional in the TypedDict. A node that doesn't touch `rag_results` just doesn't return it. The graph entry point validates `contract_code` and `contract_address` separately in `graph.py`.

---

## The lazy `audit_graph` pattern (`graph.py:270-280`)

```python
_audit_graph_singleton: Any = None

def __getattr__(name: str) -> Any:
    if name == "audit_graph":
        global _audit_graph_singleton
        if _audit_graph_singleton is None:
            _audit_graph_singleton = build_graph()
        return _audit_graph_singleton
    raise AttributeError(...)
```

**Why this exists (per the in-file comment):** The historic module-level `audit_graph = build_graph()` ran at import time, compiling the graph and opening a SqliteSaver connection on every `import` — even for callers that only wanted `build_graph` or a single node. That slowed test collection and forced I/O for callers that never used the default.

**PEP 562 module-level `__getattr__`:** defers the build to the first attribute access, then caches. Existing callers (`from src.orchestration.graph import audit_graph`) keep working unchanged; importers that never touch `audit_graph` pay nothing. Tests should still call `build_graph(use_checkpointer=False)` directly to skip the SqliteSaver I/O.

**Why not just always lazy:** Keeping a default singleton is convenient for scripts (`agents/scripts/run_real_audit.py` etc.). The PEP 562 trick gives you both: a singleton that doesn't cost anything until used.

---

## Surprises from the code (the Delta)

These are facts the code reveals that aren't stated in the README:

1. **`timed_node` wrapper on every node** — uniform START/DONE+elapsed logs in every invocation context (production, run_real_audit.py, ad-hoc scripts). Not just when a caller adds its own ad-hoc timing.

2. **Slither constructor registers ZERO detectors** — both `quick_screen` and `static_analysis` had the same silent-failure bug. Without `for cls in all_detector_classes: sl.register_detector(cls)`, the nodes find NOTHING on every contract. Confirmed by direct comparison: `slither contract.sol` CLI found reentrancy-eth on a textbook reentrant Vault; the node pre-fix found 0.

3. **Aderyn 0.6.8 quirks** — three silent-failure modes that were all fixed 2026-06-21:
   - Needs DIRECTORY as ROOT arg (`Not a directory (os error 20)` on a file path).
   - `--output` takes a real file path ending in `.json/.md/.sarif`, not a bare format word.
   - Schema is `high_issues`/`low_issues` only (no `medium` bucket).
   - Instances carry NO function-name field (only contract_path/line_no/src/src_char/hint).

4. **Cross-validator sorts by consensus confidence, not raw ML prob** — addresses a 2026-06-21 finding where classes with strong tool corroboration but low ML score (e.g. CallToUnknown at 0.249 with Slither+Aderyn agreeing) were dropped from the debate even though consensus_engine had them at CONFIRMED.

5. **Debate model is FAST, not STRONG** — STRONG (qwen3.5-9b-ud) took 94s+ for 9 classes and TIMED OUT at 90s. Switched 2026-06-17 to FAST (gemma-4-e2b-it), ~3× faster, quality sufficient for verdict picking. Override via `CROSS_VALIDATOR_LLM_MODEL=strong`.

6. **Per-role max_tokens for debate (768/768/0)** — empirically tuned 2026-06-22 on `vulnerable_reentrant.sol`. 384/512 returns empty content (model's preamble hits cap before producing output); 1024 is verbose with no quality improvement. Judge is uncapped because LM Studio returns empty when the judge is capped (judge generates reasoning before the JSON).

7. **ExternalBug structural gap** — GNN encodes `call_target_typed=1.00` for typed interface calls, making oracle price calls look safe. `_extract_external_call_summary` extracts the inter-contract call graph so `rag_research` can build a targeted query (e.g. "ExternalBug ChainlinkOracle getLatestPrice stale price manipulation") and `synthesizer` can include the call graph in the LLM prompt.

8. **`ml_result` schema is the three-tier schema (2026-05-27)** — fields are `label`, `probabilities` (full 10-class vector), `confirmed`, `suspicious`, `vulnerabilities` (legacy alias for `confirmed`), `tier_thresholds`, `thresholds` (per-class tuned), `truncated`, `windows_used`, `num_nodes`, `num_edges`. NO `confidence` field (removed in Track 3). Use `max(v["probability"])` instead.

---

## What I did NOT read (and why)

- `consensus_engine` internals — covered in Session 4.
- `synthesizer` internals — covered in Session 4.
- `reflection`, `explainer`, `visualizer` internals — covered in Session 5.
- `routing.py` (the actual `compute_active_tools`, `compute_verdict`, `DEEP_THRESHOLDS`, `CLASS_TO_DETECTORS`) — covered in Session 2.
- `timeouts.py` — referenced by every node but not the focus; will skim when needed.
- `timing.py` (the `timed_node` wrapper) — one-liner, will read in Session 10 (tests).

This is a deliberate "map first, leaves later" strategy. By the end of Sessions 2-5 you'll have read every orchestration file end-to-end.
