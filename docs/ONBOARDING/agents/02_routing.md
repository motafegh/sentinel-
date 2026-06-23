# Session 2 — Routing + per-class matrix (walkthrough note)

**Date:** 2026-06-22
**Scope:** `agents/src/orchestration/routing.py` (282 lines) end-to-end.

---

## What this file is

The **single source of truth** for the per-class decision matrix. Three constants (thresholds, tool rules, detector mappings) and five pure functions (decide, log, verdict, severity, roll-up).

Both `graph.py` and `nodes.py` import from it. The graph uses `compute_active_tools()` to choose LangGraph branches; the nodes use the same function plus `build_routing_decisions()` and `compute_verdict()` to log and render verdicts. Centralizing here = those two callers can't disagree.

---

## The matrix (memorize this)

### `DEEP_THRESHOLDS` (lines 23-34) — per-class probability that triggers deep analysis

| Class | Threshold | Why this number |
|-------|-----------|-----------------|
| `DenialOfService` | 0.30 | Rare class, under-represented in training data |
| `Reentrancy` | 0.35 | |
| `IntegerUO` | 0.35 | |
| `Timestamp` | 0.35 | |
| `TransactionOrderDependence` | 0.35 | |
| `GasException` | 0.40 | |
| `ExternalBug` | 0.40 | |
| `CallToUnknown` | 0.40 | |
| `MishandledException` | 0.40 | |
| `UnusedReturn` | 0.45 | High false-positive rate; needs stricter gate |

All thresholds are **deliberately below** the 0.50 inference threshold — the routing layer wants to investigate borderline cases, not skip them. The inference layer decides "is this a real finding?"; the routing layer decides "is this worth deeper evidence?"

### `ROUTING_RULES` (lines 43-54) — which tools activate per class

| Tools | Classes |
|-------|---------|
| `static_analysis` + `rag_research` | Reentrancy, IntegerUO, Timestamp, TransactionOrderDependence, ExternalBug, CallToUnknown, DenialOfService |
| `static_analysis` only | GasException, MishandledException, UnusedReturn |

Five classes have documented exploit history in DeFiHackLabs → they trigger RAG. Three classes are mostly Slither-detectable (loops, missed returns) → RAG adds nothing. `graph_explain` is added to the deep path unconditionally via the `+["graph_explain"]` line in `graph.py:135`.

### `CLASS_TO_DETECTORS` (lines 62-119) — class → Slither detector names

| Class | Detectors |
|-------|-----------|
| `Reentrancy` | `reentrancy-eth`, `reentrancy-no-eth`, `reentrancy-events`, `reentrancy-benign` |
| `IntegerUO` | `unchecked-lowlevel` (only one — see Stale entries below) |
| `GasException` | `costly-loop`, `calls-loop`, `incorrect-exp` |
| `Timestamp` | `timestamp` |
| `TransactionOrderDependence` | `tx-origin`, `controlled-delegatecall`, `msg-value-loop` |
| `ExternalBug` | `arbitrary-send-eth`, `low-level-calls`, `unchecked-send`, `controlled-delegatecall` |
| `CallToUnknown` | `low-level-calls`, `controlled-delegatecall`, `delegatecall-loop` |
| `MishandledException` | `unchecked-send`, `unchecked-lowlevel`, `unchecked-transfer`, `return-bomb` |
| `UnusedReturn` | `unused-return` |
| `DenialOfService` | `calls-loop`, `costly-loop`, `msg-value-loop` |

Used by `static_analysis` to scope Slither runs (3-8× faster than running all 90+ detectors). Used by `synthesizer` to bucket Slither findings by class for verdict computation. Used by `cross_validator` to build the per-class evidence package for the LLM debate.

The inverted map `DETECTOR_TO_CLASSES` is built at import time (lines 122-126) for the inverse lookup.

---

## Stale entries that were silently dead (2026-06-21 sweep)

The matrix was audited against a live Slither 0.11.5 installation via direct detector enumeration. Three entries were removed because they did not exist as detector ARGUMENT values in the installed Slither:

| Removed entry | Why it was dead |
|---------------|-----------------|
| `"reentrancy-events-and-order"` | Renamed to `"reentrancy-events"` in current Slither |
| `"integer-overflow"` | Slither 0.11.5 dropped it; Solidity ≥0.8 has built-in checked arithmetic |
| `"toctou"` | Never existed in Slither 0.11.5 (was always wrong) |

Impact: pre-fix, `static_analysis` was running these detector names against Slither and Slither was returning nothing → `compute_verdict` saw "no Slither match" for these classes → could only reach `LIKELY` or `DISPUTED`, never `CONFIRMED`. Silent under-corroboration.

---

## The five functions

### 1. `_iter_class_probs` (lines 133-145) — yield (class, prob) pairs

```python
probabilities = ml_result.get("probabilities")
if probabilities:
    yield from probabilities.items()      # three-tier schema: all 10 classes
else:
    for vuln in ml_result.get("vulnerabilities", []):
        yield vuln.get("vulnerability_class", ""), vuln.get("probability", 0.0)
```

The three-tier `ml_result` schema (2026-05-27) provides the full 10-class probability vector. The legacy `vulnerabilities` list only contains above-threshold findings. The new schema is preferred because it lets routing decisions be made for ALL classes, not just the ones ML already flagged.

### 2. `compute_active_tools` (lines 148-163) — which tools fire

```python
active: set[str] = set()
for cls, prob in _iter_class_probs(ml_result):
    if prob >= DEEP_THRESHOLDS.get(cls, 0.40):
        active.update(ROUTING_RULES.get(cls, []))
return sorted(active)
```

**Two callers, two purposes:**
- `graph.py:_route_from_evidence_router` (line 114) — returns the list to LangGraph as the branch target
- `nodes.py:evidence_router` (line 336) — calls it to log to `routing_decisions`

**Set + sorted** ensures each tool appears at most once (e.g. Reentrancy + ExternalBug both want `static_analysis` + `rag_research`; the set collapses to one of each) and the order is deterministic across runs (helpful for log diffing).

### 3. `build_routing_decisions` (lines 166-191) — human-readable log strings

Returns a list of strings like:
- `"Reentrancy prob=0.872 >= threshold=0.35 → static_analysis+rag_research"`
- `"GasException prob=0.290 < threshold=0.40 → skip"`

Appended to `state["routing_decisions"]` by the `evidence_router` node. The `routing_decisions` field uses an append-reducer (`Annotated[list, operator.add]` per `state.py:77`), so multiple nodes can log to it without overwriting.

### 4. `compute_verdict` (lines 194-261) — single-class verdict

The verdict hierarchy (rule-based, no LLM cost):

| Verdict | Condition | Sources |
|---------|-----------|---------|
| `CONFIRMED` | prob ≥ 0.50 AND (slither match OR rag score ≥ 0.80) | ml + slither/rag |
| `LIKELY` | prob ≥ 0.50 AND rag score ≥ 0.50 | ml + rag |
| `DISPUTED` | prob ≥ 0.50 AND no corroborating evidence | ml only |
| `INCONCLUSIVE` | prob ≥ DEEP_THRESHOLD but no corroboration (WS1) | ml only |
| `SAFE` | prob < DEEP_THRESHOLDS | (shouldn't reach here) |

**The WS1 fix (2026-06-21):** pre-fix, the `prob >= DEEP_THRESHOLDS but no corroboration` case returned `SAFE`. Post-fix, it returns `INCONCLUSIVE`. The principle (per the in-file comment): **"The FN/FP asymmetry principle forbids silently marking a flagged class SAFE."** A security tool's worst failure is missing a real vulnerability; false positives are cheap (a human can review). So a class that warranted investigation must not be silently cleared just because Slither didn't fire and RAG didn't match.

The `path_taken == "fast"` branch (lines 220-221) is essentially unreachable in practice — fast-path contracts have no flagged classes to compute verdicts for. It exists for defensive completeness if a caller passes the parameter inconsistently.

### 5. `prob_to_severity` (lines 264-269) and `compute_overall_verdict` (lines 278-282)

`prob_to_severity` maps a probability to a CRITICAL/HIGH/MEDIUM/LOW/INFO band. Used by `synthesizer` to label findings in the final report.

`compute_overall_verdict` rolls per-class verdicts up to one. `OVERALL_VERDICT_RANK` (lines 272-275) defines the ordering: CONFIRMED(5) > LIKELY(4) > DISPUTED(3) > WATCH(2) > INCONCLUSIVE(1) > SAFE(0). `WATCH` appears in the rank but is never returned by `compute_verdict` — it's set by the LLM-adjudicated debate in `cross_validator`.

---

## Three verdict scales (how they interact)

There are three different verdict-producing paths in the system. The synthesizer's job is to pick the right one.

| Scale | Set by | Verdict states | When used |
|-------|--------|----------------|-----------|
| Rule-based | `compute_verdict` (this file) | CONFIRMED / LIKELY / DISPUTED / INCONCLUSIVE / SAFE | Fallback when LLM is disabled or fails |
| Consensus vote | `consensus_engine` (nodes.py:1987, A.6) | CONFIRMED / LIKELY / DISPUTED / SAFE | Always, deep path only |
| LLM-adjudicated debate | `cross_validator` (nodes.py:1013, A.4) | CONFIRMED / LIKELY / DISPUTED / WATCH / SAFE | Default when LLM enabled, deep path only |

The synthesizer reads the LLM verdicts if present, otherwise falls back to consensus verdicts, otherwise to rule-based verdicts. That fallback chain is what we'll see in Session 4.

---

## What I did NOT read (and why)

- `consensus_engine` internals (`nodes.py:1987-2057`) — covered in Session 4.
- `synthesizer` internals (`nodes.py:1571-1986`) — covered in Session 4.
- `cross_validator` post-line 1175 (the actual debate prompt construction and 3-role LLM call) — covered in Session 4.
- `timeouts.py` — the centralized timeout config; will skim when needed.
- Tests that exercise this file: `tests/test_routing_phase0.py` (345 lines, the matrix + verdict logic) and `tests/test_graph_routing.py` (931 lines, the full graph + the routing function in `graph.py`). Both are on the Session 10 list.
