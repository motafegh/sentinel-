# Orchestration

LangGraph StateGraph that coordinates the SENTINEL audit workflow — from ML inference through multi-source evidence collection to a final vulnerability report.

## Architecture

```
START
  │
  ▼
ml_assessment          POST /predict via inference MCP :8010
  │
  ▼
quick_screen           Slither + Aderyn Tier-0 screen (every contract)
  │
  ▼
evidence_router        Per-class routing decisions, logs to state
  │
  ├─ [deep path] ───── rag_research ────────┐
  │                   static_analysis ───────┤
  │                   graph_explain ─────────┘
  │                                            │
  │                                      audit_check
  │                                            │
  │                                   consensus_engine      ← A.6/A.7 (weighted vote, ML discounted)
  │                                            │
  │                                    cross_validator      ← A.4 (Prosecutor/Defender/Judge debate)
  │                                            │
  ├─ [fast path] ───────────────────── synthesizer
  │                                            │
  │                                       reflection         ← A.3 (self-critique)
  │                                            │
  │                                       explainer          ← A.8 (LIME attribution)
  │                                            │
  │                                       visualizer         ← A.9 (interactive HTML)
  │                                            │
  ▼                                           ▼
  END ◄──────────────────────────────────── END
```

### Fast vs Deep Path

| Signal | Fast Path | Deep Path |
|--------|-----------|-----------|
| ML all classes below `DEEP_THRESHOLDS` | Yes | No |
| `quick_screen` zero High/Critical hits | Yes | No |
| Both signals agree it is safe | **Yes** | — |
| Either signal flags risk | — | **Yes** |

Two-signal gate: fast path requires ML **and** static analysis to agree. If Slither/Aderyn fire High/Critical findings while ML is below thresholds, the contract is still escalated to deep path.

## Files

| File | Purpose |
|------|---------|
| `state.py` | `AuditState` TypedDict — 26 fields flowing through the graph |
| `routing.py` | Per-class thresholds, tool routing, verdict computation |
| `nodes.py` | 13 node implementations (the core logic) |
| `graph.py` | StateGraph builder, conditional edges, lazy `audit_graph`, SqliteSaver checkpointing |
| `consensus.py` | **(A.6, 2026-06-21)** Weighted ML/Slither/Aderyn vote — ML weight discounted |
| `confidence.py` | **(A.7, 2026-06-21)** Bayesian staged confidence tracking |
| `attribution.py` | **(A.8, 2026-06-21)** LIME-style evidence-source attribution |
| `visualizer.py` | **(A.9, 2026-06-21)** Interactive hotspot HTML generator |
| `timeouts.py` | **(2026-06-21)** Single source of truth for every timeout default — see "Timeouts" below |
| `timing.py` | **(2026-06-21)** `step_timer()` / `timed_node()` — uniform live START/DONE+elapsed logs for every node and LLM sub-call |

## `state.py` — AuditState

`TypedDict` with `total=False` — every field is optional. Nodes return only the keys they updated; LangGraph merges partial dicts automatically.

### Fields

| Field | Type | Set By | Reducer |
|-------|------|--------|---------|
| `contract_code` | `str` | Caller | — (immutable) |
| `contract_address` | `str` | Caller | — (immutable) |
| `ml_result` | `dict` | `ml_assessment` | replace |
| `ml_hotspots` | `list[dict]` | `graph_explain` | replace |
| `routing_decisions` | `list[str]` | `evidence_router` + any node | `operator.add` (append) |
| `graph_explanations` | `dict` | `graph_explain` | replace |
| `quick_screen_hits` | `dict[str, list[str]]` | `quick_screen` | replace |
| `static_findings` | `list[dict]` | `static_analysis` | replace |
| `external_call_summary` | `list[dict]` | `static_analysis` | replace |
| `rag_results` | `list[dict]` | `rag_research` | replace |
| `econ_scenarios` | `list[dict]` | (Phase 3) | replace |
| `verdicts` | `dict[str, str]` | `cross_validator` / `synthesizer` | replace |
| `confirmations` | `dict[str, list[str]]` | `cross_validator` / `synthesizer` | replace |
| `contradictions` | `dict[str, list[str]]` | `cross_validator` | replace |
| `audit_history` | `list[dict]` | `audit_check` | replace |
| `final_report` | `dict` | `synthesizer` | replace |
| `narrative` | `str \| None` | `synthesizer` | replace |
| `error` | `str \| None` | Any node | replace |
| `consensus_verdict` | `dict[str, dict]` | `consensus_engine` | replace |
| `confidence_by_class` | `dict[str, float]` | `consensus_engine` | replace |
| `metric_attribution` | `dict[str, dict]` | `explainer` | replace |
| `reflection_notes` | `dict` | `reflection` | replace |
| `debate_transcript` | `dict[str, str]` | `cross_validator` (DEBATE_MODE=on) | replace |
| `hotspot_visualization` | `str \| None` | `visualizer` | replace |
| `symbolic_findings`, `bytecode_analysis`, `taint_flows`, `permission_graph` | — | *(Phase B placeholders — no node yet)* | replace |

## `routing.py` — Per-Class Routing

### DEEP_THRESHOLDS

Per-class probability thresholds that trigger deep analysis. Deliberately lower than the inference threshold (0.50) — borderline cases are investigated, not skipped.

```python
DEEP_THRESHOLDS = {
    "Reentrancy":          0.35,
    "IntegerUO":           0.35,
    "GasException":        0.40,
    "Timestamp":           0.35,
    "TransactionOrderDependence": 0.35,
    "ExternalBug":         0.40,
    "CallToUnknown":       0.40,
    "MishandledException": 0.40,
    "UnusedReturn":        0.45,
    "DenialOfService":     0.30,
}
```

### ROUTING_RULES

Which tool nodes activate per flagged class:

| Class | Tools |
|-------|-------|
| Reentrancy, IntegerUO, Timestamp, TransactionOrderDependence, ExternalBug, CallToUnknown, DenialOfService | `static_analysis` + `rag_research` |
| GasException, MishandledException, UnusedReturn | `static_analysis` only |

### CLASS_TO_DETECTORS

Maps each vulnerability class to the relevant Slither detector names. Used by `static_analysis` node for detector scoping (3-8x faster than running all 90+ detectors) and by `synthesizer` for matching Slither findings to ML-flagged classes.

### Verdict Computation

**Rule-based** (`compute_verdict`):
- `CONFIRMED` — prob >= 0.50 AND (Slither match OR RAG score >= 0.80)
- `LIKELY` — prob >= 0.50 AND RAG score >= 0.50
- `DISPUTED` — prob >= 0.50 AND no corroborating evidence
- `SAFE` — below threshold

**LLM-adjudicated** (`cross_validator` node, A.4): when `DEBATE_MODE=on` (default),
runs three sequential calls — Prosecutor (reads the source, argues vulnerable) →
Defender (argues false-positive) → Judge (renders verdicts as JSON). Single-pass
classification if `DEBATE_MODE=off`. Falls back silently to rule-based on LLM failure
or when `AGENTS_DISABLE_LLM=1`.

**Consensus vote** (`consensus_engine` node, A.6/A.7): per flagged class, weights ML/
Slither/Aderyn signals by per-class reliability (`consensus.py:ACCURACY_WEIGHTS`), with
ML's weight discounted by `ML_WEIGHT_SCALE` (default 0.5) — **ML alone cannot reach
CONFIRMED**, it needs at least one corroborating static-tool hit. Also computes a
Bayesian-updated `confidence_by_class` (`confidence.py:track_confidence`) that nudges
the ML probability up/down as Slither/Aderyn/RAG evidence arrives.

## `nodes.py` — Node Implementations

### Node Summary

| Node | MCP Server | Parallel? | Purpose |
|------|-----------|-----------|---------|
| `quick_screen` | — (direct Slither + Aderyn) | No | Tier-0 screen on every contract |
| `evidence_router` | — (pure function) | No | Logs per-class routing decisions |
| `ml_assessment` | `:8010` | No | ML vulnerability prediction |
| `rag_research` | `:8011` | Yes (deep) | Exploit pattern retrieval |
| `static_analysis` | — (direct Slither + Aderyn) | Yes (deep) | Detector-scoped static analysis |
| `graph_explain` | `:8013` | Yes (deep) | Function-level hotspot attribution |
| `audit_check` | `:8012` | After fan-in | On-chain audit history lookup |
| `consensus_engine` | — (pure logic) | After audit_check | **(A.6/A.7)** Weighted vote + Bayesian confidence |
| `cross_validator` | — (LLM call) | After consensus_engine | **(A.4)** Prosecutor/Defender/Judge debate |
| `synthesizer` | — (LLM call) | After cross_validator | Final report assembly |
| `reflection` | — (LLM call, optional) | After synthesizer | **(A.3)** Self-critique of the assembled audit |
| `explainer` | — (pure logic) | After reflection | **(A.8)** LIME-style attribution, folds enrichment into final_report |
| `visualizer` | — (pure logic) | After explainer | **(A.9)** Interactive hotspot HTML, last node before END |

### MCP Client Pattern

Each node opens a **short-lived SSE connection**, calls exactly one MCP tool, and closes the connection. The shared `_call_mcp_tool()` helper handles the full lifecycle:

```python
async with sse_client(server_url) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(tool_name, arguments)
```

Connection-per-call is intentional for M5 simplicity. Promotable to a module-level persistent client in M6 if RTT measurements show this is a bottleneck.

### quick_screen

Runs Slither and Aderyn on **every contract** before routing. Scoped to High/Critical-impact detectors only:

- Slither: 14 detectors (`reentrancy-eth`, `arbitrary-send-eth`, `controlled-delegatecall`, `integer-overflow`, `suicidal`, etc.)
- Aderyn: H-1 through H-5, C-1 through C-3

Non-fatal: if either tool is not installed, the screen proceeds with the remaining tool.

### static_analysis

Runs Slither directly (not via MCP — Slither is a Python library in this process). Scoping:
1. Collect classes above `DEEP_THRESHOLDS`
2. Map classes → detectors via `CLASS_TO_DETECTORS`
3. Filter `sl._detectors` to scoped set only

Also runs Aderyn on the same temp file for independent findings. Produces `external_call_summary` when ExternalBug is flagged (inter-contract call graph extraction for RAG query enrichment and LLM synthesis).

### consensus_engine (A.6/A.7, added 2026-06-21)

Runs after `audit_check`, before `cross_validator`. For each class with ML probability
≥ 0.50 or a static-tool hit, calls `consensus.consensus_vote()` — a weighted vote over
{ML, Slither, Aderyn} using per-class reliability weights (`ACCURACY_WEIGHTS`), with
ML's weight scaled by `ML_WEIGHT_SCALE` (default 0.5, env-tunable). Also calls
`confidence.track_confidence()` to Bayesian-update a per-class confidence as
Slither/Aderyn/RAG evidence arrives. Output: `consensus_verdict`, `confidence_by_class`.

**Why ML is discounted:** Run 12's ML model is known to over-predict certain classes
(e.g. ExternalBug — see MEMORY.md DIVE crosswalk audit, p=0.96 on a 26-line KV store
false positive). The agent layer treats ML as a *hint* and requires independent
corroboration — `consensus_vote(0.99, slither=False, aderyn=False)` always returns SAFE.

### cross_validator (upgraded to debate, A.4, 2026-06-21)

When `DEBATE_MODE=on` (default), runs three sequential LLM calls instead of one:
1. **Prosecutor** — reads per-class evidence AND the contract source (first 2000 chars),
   argues why the contract HAS the vulnerabilities.
2. **Defender** — given the prosecutor's case, argues why findings may be false
   positives (the ML model is explicitly named as over-prediction-prone in the prompt).
3. **Judge** — given both arguments, renders `{class: verdict}` as JSON, verdict ∈
   `CONFIRMED | LIKELY | DISPUTED | WATCH | SAFE`.

Set `DEBATE_MODE=off` to fall back to the original single classification call.
Default model: FAST (`CROSS_VALIDATOR_LLM_MODEL=fast`); transcript stored in
`debate_transcript`. Falls back silently to `{}` (rule-based verdicts downstream) on
any LLM failure or when `AGENTS_DISABLE_LLM=1`.

**Timeout (fixed 2026-06-21):** the entire 3-role debate is bounded by ONE outer
timeout, `DEBATE_TIMEOUT_S` (default 240s) — not per-call. An earlier version applied
`CROSS_VALIDATOR_TIMEOUT_S` (90s) to each of the 3 sequential calls, allowing up to
270s worst case; this exceeded external script timeouts during real-audit testing and
caused ungraceful process kills (the abandoned `asyncio.to_thread()` HTTP call kept
running in its OS thread since `to_thread` cannot be cancelled). `DEBATE_MODE=off`
single-pass calls still use `CROSS_VALIDATOR_TIMEOUT_S` (90s).

### synthesizer

Assembles the final report from all available state fields:
1. Selects each class's verdict by priority (fixed 2026-06-21 — see "Verdict
   fallback gap" below): `cross_validator`'s debate verdict → else
   `consensus_engine`'s vote → else rule-based `compute_verdict()` as last resort.
2. Computes `risk_probability`, `top_vulnerability`, `overall_verdict`
3. Generates rule-based recommendation (fallback) or LLM narrative (primary)
4. Persists report to `data/reports/{contract_address}.json` for the feedback loop bridge

LLM narrative: prompts the strong LLM with structured Markdown output (Severity, Vulnerability Summary, Exploit Pattern, Recommended Fix). Timeout: `SYNTHESIZER_TIMEOUT_S` (120s). Skipped entirely when `AGENTS_DISABLE_LLM=1`.

**Verdict fallback gap (fixed 2026-06-21):** `synthesizer` used to read ONLY
`state["verdicts"]` (written by `cross_validator`) and fall through directly to
`compute_verdict()` — a pre-Phase-A rule — whenever the debate failed, completely
bypassing `consensus_engine`'s already-computed, ML-discounted vote. Verified live on
`safe_storage.sol`: `consensus_engine` correctly said `ExternalBug: SAFE` while
`compute_verdict()` said `DISPUTED` for the identical evidence — two different
philosophies disagreeing in the same report. `consensus_engine`'s vote is now the
preferred fallback; `compute_verdict()` only runs for classes `consensus_engine`
never scored (weak SUSPICIOUS-tier noise below its scoring bar).

**Narrative hallucination (fixed 2026-06-21):** the narrative prompt's `vuln_lines`
used to list every ML-flagged class with NO verdict attached — the model had no
signal a class had been cleared and would describe it as a real risk anyway.
Verified live: a narrative described a "Reentrancy risk" on a contract whose
Reentrancy verdict was SAFE and which contains zero external calls, citing unlabeled
RAG reference content about an unrelated pattern (Multicall). `vuln_lines` now
attaches `→ verdict: <CONFIRMED|LIKELY|DISPUTED|SAFE|...>` to every line (mirroring
`reflection`'s prompt, which already did this correctly), the system prompt
explicitly instructs the model to only discuss CONFIRMED/LIKELY classes, and the RAG
section is relabeled as general historical reference, not site-specific evidence.

### reflection (A.3, added 2026-06-21)

Runs after `synthesizer`. Computes rule-based structured critique (always, no LLM
required): unused evidence (RAG/static findings not tied to any verdict), contradictions
(from `cross_validator`), uncertain verdicts (DISPUTED/WATCH or confidence < 0.70), and
known failure modes (truncated contract, ExternalBug ML over-prediction, missing static
findings). If the LLM is enabled, additionally asks the strong LLM for a 3-5 sentence
skeptical narrative summary; falls back to a rule-based summary otherwise.

### explainer (A.8, added 2026-06-21)

Runs after `reflection`. For each verdict, calls `attribution.attribute_verdict()` to
compute a LIME-style `{ml_pct, slither_pct, rag_pct}` breakdown (sums to ~100). Folds
`confidence_by_class`, `consensus_verdict`, and `reflection_notes` into `final_report`
so the enriched report is a single self-contained artifact.

### visualizer (A.9, added 2026-06-21)

Last node before `END`. Calls `visualizer.generate_hotspot_html()` to render a
dependency-free, self-contained HTML report (source code with hotspot-line highlighting
+ clickable verdict cards showing confidence and attribution bars). Writes to
`data/reports/{contract_address}_hotspot.html`. Never raises — returns `None` on
generation failure.

## `graph.py` — Graph Builder

### Compilation

```python
from src.orchestration.graph import build_graph

graph = build_graph()                    # with SqliteSaver (production)
graph = build_graph(use_checkpointer=False)  # no persistence (tests)
```

### Checkpointing

`SqliteSaver` persists state to `agents/data/checkpoints.db` after **every node**. If the process crashes mid-graph, it resumes from the last completed node by providing the same `thread_id`:

```python
result = await graph.ainvoke(
    None,  # state loaded from checkpoint by thread_id
    config={"configurable": {"thread_id": "audit-001"}},
)
```

Falls back to `MemorySaver` (in-process dict, lost on restart) if `langgraph-checkpoint-sqlite` is not installed.

### Module-Level Default (lazy, A.1)

```python
from src.orchestration.graph import audit_graph
# Ready-to-use compiled graph instance — built on FIRST ACCESS, then cached
```

`audit_graph` is no longer built at import time. A PEP 562 module-level `__getattr__`
defers `build_graph()` until the attribute is first accessed (importers that never touch
`audit_graph` pay nothing — no SqliteSaver connection opened). For tests, call
`build_graph(use_checkpointer=False)` directly rather than relying on the cached default.

## Timeouts (centralized, 2026-06-21)

Every timeout default in the pipeline lives in ONE place: `timeouts.py`. No other
module hardcodes a magic-number default — each reads its timeout via
`get_timeout(ENV_VAR_NAME, DEFAULT_CONST)` at call time (never cached), so the
existing override mechanisms all keep working: set the env var directly, set it via
a test's `monkeypatch.setenv(...)`, or pass the matching CLI flag to
`scripts/run_real_audit.py` (it sets the env var before any agents module is
imported — see that script's `_resolve_timeouts()`).

| Timeout | Env var | Default | CLI flag |
|---|---|---|---|
| LM Studio HTTP client (floor under every LLM call) | `LM_STUDIO_TIMEOUT` | 60s | `--lm-studio-timeout-s` |
| `cross_validator` single-pass (`DEBATE_MODE=off`) | `CROSS_VALIDATOR_TIMEOUT_S` | 90s | `--cross-validator-timeout-s` |
| `cross_validator` debate, ALL 3 roles as one budget | `DEBATE_TIMEOUT_S` | 240s | `--debate-timeout-s` |
| `synthesizer` narrative LLM call | `SYNTHESIZER_TIMEOUT_S` | 120s | `--synthesizer-timeout-s` |
| `reflection` self-critique LLM call | `REFLECTION_TIMEOUT_S` | 120s | `--reflection-timeout-s` |
| Aderyn subprocess | `ADERYN_TIMEOUT_S` | 90s | `--aderyn-timeout-s` |

`run_real_audit.py --unbounded-timeouts` sets every value above (and the script's own
`--timeout-s`) to 3600s in one shot — use this to observe TRUE per-step timing with
nothing artificially truncated; an explicit `--<x>-timeout-s` flag still wins for that
one value even with `--unbounded-timeouts` set.

**Honesty note:** `DEBATE_TIMEOUT_S=240` was an estimate from one early single-call
observation, not a measured 3-role-sequence budget — two later real debates both
timed out at ~246-248s under sustained session load. Don't treat shipped defaults as
validated; re-measure with `--unbounded-timeouts` if you need a real number.

## Timing / Live Logs (centralized, 2026-06-21)

`timing.py`'s `step_timer()` context manager logs a uniform `START` line and a `DONE
| elapsed=Xs` line around any block — every node is wrapped with it automatically via
`timed_node()` at registration time in `build_graph()` (no per-node code change
needed), so production logs always show exactly where time went, in every context
(production MCP-driven server, `run_real_audit.py`, ad-hoc scripts) — not only when a
caller happens to add its own ad-hoc wrapper. `cross_validator`'s 3 debate roles
(Prosecutor/Defender/Judge) are each individually timed (`cross_validator.prosecutor`
etc.), not just the aggregate, so a slow run shows exactly which role is the bottleneck.

```
16:35:52.919 | INFO | cross_validator.prosecutor | START | address=0xLIVE
16:37:14.221 | INFO | cross_validator.prosecutor | DONE | elapsed=81.30s | address=0xLIVE
```

## Usage

### Standalone

```python
import asyncio
from src.orchestration.graph import build_graph

async def audit():
    graph = build_graph(use_checkpointer=False)
    result = await graph.ainvoke(
        {
            "contract_code": "<solidity source>",
            "contract_address": "0x...",
        },
        config={"configurable": {"thread_id": "audit-001"}},
    )
    print(result["final_report"])

asyncio.run(audit())
```

### Smoke Test

```bash
cd agents
poetry run python scripts/smoke_langgraph.py          # mock — no services needed
poetry run python scripts/smoke_langgraph.py --live    # live — all services must be up
```
