# SENTINEL — Agent Module Technical Proposal

**Date:** 2026-05-27  
**Module:** M4 — Orchestration & Multi-Agent Reasoning  
**Status:** Design finalized — ready for implementation  
**Authors:** SENTINEL Engineering

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Context](#3-system-context)
4. [Design Principles](#4-design-principles)
5. [Architecture Overview](#5-architecture-overview)
6. [Graph Topology](#6-graph-topology)
7. [State Schema](#7-state-schema)
8. [ML Integration — Three-Tier Input](#8-ml-integration--three-tier-input)
9. [Node Specifications](#9-node-specifications)
10. [Evidence Routing](#10-evidence-routing)
11. [Verdict Model](#11-verdict-model)
12. [Report Format](#12-report-format)
13. [MCP Server Architecture](#13-mcp-server-architecture)
14. [Escalation Tiers](#14-escalation-tiers)
15. [Special Case: ExternalBug](#15-special-case-externalbug)
16. [Implementation Plan](#16-implementation-plan)
17. [Success Metrics](#17-success-metrics)
18. [Scope Boundaries](#18-scope-boundaries)

---

## 1. Executive Summary

SENTINEL's ML model produces a continuous probability vector across 10 vulnerability classes for
any Solidity smart contract. Probability is signal, not verdict. The agent module converts that
signal into a reasoned, traceable, actionable audit report by coordinating four independent
evidence sources: ML assessment, static analysis (Slither), semantic retrieval (RAG), and
on-chain audit history.

The agent layer is not post-processing. It is where the audit verdict actually happens.
The ML model does triage; the agents do verification; the final report documents the chain of
evidence. A contract flagged by ML at Reentrancy=0.62 that Slither also confirms at
`reentrancy-eth` with a high-severity finding is materially different from a contract flagged
at Reentrancy=0.62 where Slither finds nothing. Both share the same ML score. Only the
agent layer can tell them apart.

**The core deliverable:** An async LangGraph multi-agent pipeline that takes raw Solidity source
as input and produces a structured audit report with per-class verdicts, evidence citations,
severity ratings, and a natural-language narrative — all within 60 seconds at worst, under 30
for the majority of contracts.

---

## 2. Problem Statement

### 2.1 The Binary Verdict Trap

Smart contract security tools traditionally output pass/fail or risk score. These are operationally
useless at audit scale:

- A risk score of 0.72 does not tell the developer which function is vulnerable, what the exploit
  pattern is, whether any other tool agrees, or what to fix.
- A "HIGH RISK" label without evidence is marketing, not auditing.
- A SAFE verdict on a contract scoring Reentrancy=0.34 (just below threshold) could be dangerous —
  the model has non-trivial signal that the threshold discards.

### 2.2 The Single-Tool Blind Spot

Every individual tool has known blind spots:

| Tool | What it misses |
|------|---------------|
| ML (GNN+BERT) | Semantic patterns that don't appear in graph structure (e.g. oracle manipulation via typed interfaces), classes underrepresented in training data |
| Slither static analysis | Exploits requiring runtime state or cross-contract reasoning |
| RAG / exploit database | Novel vulnerability patterns not yet documented in known exploits |
| On-chain audit history | Contracts never audited before, or those audited under different source versions |

No single tool is reliable alone. The agent layer exists to reconcile disagreements,
elevate agreements, and flag ambiguities — not to pick one tool and trust it.

### 2.3 The Traceability Requirement

A credible audit report must answer: **why** was this vulnerability flagged, and **what evidence
supports or refutes it**? Post-hoc justification is unacceptable in security. The agent must
produce a traceable evidence chain at the time of verdict, not after the fact.

---

## 3. System Context

SENTINEL is a decentralized AI security oracle. The agent module sits between the ML model
(Module 1) and the final outputs (ZK proof, on-chain AuditRegistry, API response):

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SENTINEL Data Flow                            │
│                                                                      │
│  Raw Solidity Source                                                 │
│        │                                                             │
│        ▼                                                             │
│  ┌──────────────┐   10-class probability vector                      │
│  │  ML Module   │ ─────────────────────────────────────────────────┐ │
│  │ (GNN+BERT)   │   full probabilities dict + confirmed + suspicious│ │
│  └──────────────┘                                                   │ │
│                                                                      │ │
│  ┌──────────────────────────────────────────────────────────────┐   │ │
│  │                    AGENT MODULE (this doc)                    │ ◄─┘ │
│  │                                                              │     │
│  │  evidence_router → [rag_research ‖ static_analysis]          │     │
│  │                 → audit_check → synthesizer                  │     │
│  │                                                              │     │
│  └────────────────────────────┬─────────────────────────────────┘     │
│                               │                                       │
│              ┌────────────────┼────────────────┐                      │
│              ▼                ▼                ▼                      │
│       AuditReport       ZK Proof          On-Chain                   │
│       (JSON+MD)       (EZKL/Groth16)    AuditRegistry                │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**What the agent module receives from ML:**
- Full 10-class probability vector (always — no filtering)
- CONFIRMED list (prob ≥ 0.55) — classes the model is confident about
- SUSPICIOUS list (prob ≥ 0.25) — classes with non-trivial signal below commit threshold
- Per-class tuned thresholds and tier boundaries
- Graph metadata (node/edge counts, windows used)

**What the agent module produces:**
- Per-class verdicts: CONFIRMED / LIKELY / DISPUTED / WATCH / SAFE
- Evidence sources per verdict (which tools agreed, at what confidence)
- Structured JSON audit report
- Natural-language narrative (when LLM is available)
- Persisted report for feedback loop and on-chain indexing

---

## 4. Design Principles

### P1 — ML is triage, not the judge

Probabilities are direction signals. A contract with Reentrancy=0.87 and a confirmed
`reentrancy-eth` Slither finding is a different situation from Reentrancy=0.87 with no
static or RAG corroboration. The model assigns the same score to both. The agent module
must not.

### P2 — Every verdict must be traceable

The final report must cite which tools produced which evidence. No conclusion without citation.
"CONFIRMED: ml:0.872, slither:reentrancy-eth[High], rag:TheDAO-0.81" is an audit finding.
"HIGH RISK" is not.

### P3 — Cost proportional to risk

A contract where all probabilities are below investigation thresholds does not need an LLM
call. A confirmed high-severity vulnerability does. Design tiers explicitly and enforce them.
Maximum LLM calls per audit: 2 (adjudication + narrative). Most contracts: 0.

### P4 — No information loss

The ML model's probability vector is the most information-dense output it produces. The old
binary threshold system converted "I'm 36% confident this is DoS" into SAFE — discarding
signal the model actually has. The agent module preserves and uses the full vector.

### P5 — Fail gracefully, never silently

Any node can fail. Tool unavailability (Slither import error, MCP server down, LLM timeout)
must not abort the graph. Errors are captured into state and reported in the final output.
A partial report is better than a crash.

### P6 — The agents must not add false certainty

If the model has signal and Slither finds nothing, the verdict is DISPUTED, not CONFIRMED.
If the model sees no signal and Slither finds nothing, the verdict is SAFE, not "LIKELY SAFE".
The agent's job is to reconcile evidence honestly, including when it is inconclusive.

---

## 5. Architecture Overview

**Framework:** LangGraph `StateGraph` with `SqliteSaver` checkpointing  
**Execution model:** async Python coroutines; parallel fan-out on deep path  
**Transport:** MCP over SSE for all external tool calls (ML, RAG, audit registry)  
**State:** Single `AuditState` TypedDict flowing through all nodes  
**Persistence:** `SqliteSaver` at `agents/data/checkpoints.db` — every node checkpointed

**Why LangGraph:**
- Built-in parallel fan-out (`Send` API) for running rag_research and static_analysis
  simultaneously — the most expensive operations in the deep path
- `SqliteSaver` gives crash recovery (resume from last completed node by thread_id)
- `StateGraph` enforces the typed state schema — nodes cannot access undeclared fields
- Conditional edges express the routing logic explicitly and inspectably

**Why MCP for external tools:**
- MCP over SSE decouples the agents process from ML inference and RAG processes
- ML inference can be on GPU-attached host; agents on CPU host; same protocol
- Any MCP-compatible agent (Claude Desktop, Cursor, custom) can consume SENTINEL tools
- Health endpoints on each server enable monitoring without protocol changes

---

## 6. Graph Topology

### Full Topology (all phases)

```
START
  │
  ▼
┌─────────────────────┐
│    ml_assessment    │  ← calls sentinel-inference :8010 via MCP
│                     │    returns three-tier ml_result
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   graph_explain     │  ← Phase 1: calls graph_inspector_server :8013
│   (Phase 1)         │    GNN attention → hotspot node IDs + feature descriptions
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   evidence_router   │  ← reads probabilities dict from ml_result
│                     │    determines which tools activate per class
│                     │    logs routing_decisions to state
└──────────┬──────────┘
           │
    ┌──────┴──────────────────────────────┐
    │                                     │
    │ CONFIRMED classes:                  │ SUSPICIOUS-only or
    │ all relevant tools                  │ all below threshold:
    ▼                                     │
┌──────────────┐  ┌──────────────┐        │
│ rag_research │  │static_analyis│        │ (fast path)
│ per-class    │  │scoped detect.│        │
│ parallel     │  │+ext_call_sum.│        │
└──────┬───────┘  └──────┬───────┘        │
       │                 │                │
       └────────┬────────┘                │
                ▼                         │
    ┌───────────────────────┐             │
    │      audit_check      │             │
    │  AuditRegistry lookup │             │
    └───────────┬───────────┘             │
                │                         │
                └─────────────────────────┤
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   cross_validator     │ ← Phase 2
                              │  rule-based verdict   │
                              │  LLM for DISPUTED only│
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │      synthesizer      │
                              │  final report + LLM   │
                              │  narrative            │
                              └───────────┬───────────┘
                                          │
                                         END
```

### Phase 0 (current — implemented)

```
ml_assessment → evidence_router → [rag_research ‖ static_analysis] → audit_check → synthesizer
                                → synthesizer (fast path)
```

### Phase 1 (next — graph_explain node, graph_inspector_server)

```
ml_assessment → graph_explain → evidence_router → ...
```

### Phase 2 (cross_validator between audit_check and synthesizer)

```
... → audit_check → cross_validator → synthesizer
```

---

## 7. State Schema

`AuditState` is a single TypedDict that flows through every node. Each node receives the
full current state and returns only the keys it updated — LangGraph merges these back.
No node mutates state in-place.

```python
class AuditState(TypedDict, total=False):

    # ── Inputs (set by caller, never changed by nodes) ────────────────────────
    contract_code:    str           # raw Solidity UTF-8 source
    contract_address: str           # on-chain address (for registry lookup + report)

    # ── ML evidence ────────────────────────────────────────────────────────────
    ml_result: dict[str, Any]
    # Three-tier schema:
    #   label:           "safe" | "suspicious" | "confirmed_vulnerable"
    #   probabilities:   {class: float}  — all 10 classes, always present
    #   confirmed:       [{vulnerability_class, probability, tier="CONFIRMED"}, ...]
    #   suspicious:      [{vulnerability_class, probability, tier="SUSPICIOUS"}, ...]
    #   vulnerabilities: legacy alias for confirmed (backward compat)
    #   thresholds:      [float × 10]  — per-class tuned thresholds
    #   tier_thresholds: {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10}
    #   truncated:       bool
    #   windows_used:    int
    #   num_nodes:       int
    #   num_edges:       int

    ml_hotspots: list[dict[str, Any]]
    # Phase 1. Per-class attention hotspots from GNN + CodeBERT.
    # Each: {vulnerability_class, function_name, lines: [int], node_ids: [int], score: float}

    # ── Routing trace ──────────────────────────────────────────────────────────
    routing_decisions: Annotated[list[str], operator.add]
    # Append-reducer: every node can add entries; none overwrite prior entries.
    # Format per entry:
    #   "{Class} prob={p:.3f} [CONFIRMED|SUSPICIOUS] >= agent_threshold={t} → tools"
    #   "{Class} prob={p:.3f} [SUSPICIOUS] < agent_threshold={t} → WATCH (fast path)"

    # ── Graph explanation (Phase 1) ────────────────────────────────────────────
    graph_explanations: dict[str, Any]
    # {class: {subgraph_json, feature_descriptions, node_ids}}
    # Set by graph_explain node via graph_inspector_server :8013

    # ── Static analysis ────────────────────────────────────────────────────────
    static_findings: list[dict[str, Any]]
    # Each finding:
    #   {tool, detector, impact, confidence, description, lines: [int], function_names: [str]}
    # Optional key on ExternalBug findings:
    #   external_call_summary: [{caller_function, called_contract, called_function,
    #                            is_interface, is_in_value_path}]

    # ── RAG evidence ───────────────────────────────────────────────────────────
    rag_results: list[dict[str, Any]]
    # Ranked exploit knowledge chunks from FAISS+BM25+reranker.
    # Each: {content, metadata: {protocol, date, class, source}, score}

    # ── Economic simulation (Phase 3) ──────────────────────────────────────────
    econ_scenarios: list[dict[str, Any]]
    # {name, inputs, outcome, exploitable: bool, description}

    # ── Cross-validation ───────────────────────────────────────────────────────
    verdicts: dict[str, str]
    # {vulnerability_class: "CONFIRMED"|"LIKELY"|"DISPUTED"|"WATCH"|"SAFE"}

    confirmations: dict[str, list[str]]
    # {class: ["ml:0.872", "slither:reentrancy-eth", "rag:0.81"]}

    contradictions: dict[str, list[str]]
    # Phase 2. {class: ["ml_flagged", "slither_clean", "rag_nomatch"]}

    # ── On-chain history ───────────────────────────────────────────────────────
    audit_history: list[dict[str, Any]]
    # Prior AuditResult records from on-chain AuditRegistry.

    # ── Final output ───────────────────────────────────────────────────────────
    final_report: dict[str, Any]   # full structured audit report (see §12)
    narrative: str | None           # LLM Markdown narrative; None if LLM unavailable
    error: str | None               # first non-fatal error from any node
```

**Reducer note:** `routing_decisions` uses `Annotated[list, operator.add]` — the only field
with non-default merge semantics. All other fields follow LangGraph's default (replace on update).

---

## 8. ML Integration — Three-Tier Input

### Why Full Probability Vector Matters

The ML model's current inference output applies a hard per-class threshold and returns only
classes above it. Under the old schema, a contract scoring DoS=0.36 (threshold=0.45) is
reported as SAFE for DoS. But 0.36 is not "no signal." It means the model found something
DoS-like but wasn't confident enough to commit.

Evidence from the 20-contract evaluation (Run 4, epoch 32):

| Contract | Expected class | Raw prob | Threshold | Old output |
|----------|---------------|----------|-----------|------------|
| 05 | DenialOfService | 0.36 | 0.45 | SAFE (missed) |
| 10 | GasException | 0.36 | 0.40 | SAFE (missed) |
| 14 | Reentrancy | 0.34 | 0.40 | SAFE (missed) |
| 17 | IntegerUO | 0.41 | 0.50 | SAFE (missed) |

With SUSPICIOUS tier at prob ≥ 0.25, all four would be flagged for agent investigation.
The model has the signal. The binary threshold suppresses it.

### New ML Output Schema

```json
{
  "label": "suspicious",
  "probabilities": {
    "CallToUnknown":              0.638,
    "DenialOfService":            0.312,
    "ExternalBug":                0.261,
    "GasException":               0.302,
    "IntegerUO":                  0.314,
    "MishandledException":        0.292,
    "Reentrancy":                 0.620,
    "Timestamp":                  0.197,
    "TransactionOrderDependence": 0.281,
    "UnusedReturn":               0.249
  },
  "confirmed": [
    {"vulnerability_class": "CallToUnknown", "probability": 0.638, "tier": "CONFIRMED"},
    {"vulnerability_class": "Reentrancy",    "probability": 0.620, "tier": "CONFIRMED"}
  ],
  "suspicious": [
    {"vulnerability_class": "DenialOfService",            "probability": 0.312, "tier": "SUSPICIOUS"},
    {"vulnerability_class": "IntegerUO",                  "probability": 0.314, "tier": "SUSPICIOUS"},
    {"vulnerability_class": "GasException",               "probability": 0.302, "tier": "SUSPICIOUS"},
    {"vulnerability_class": "MishandledException",        "probability": 0.292, "tier": "SUSPICIOUS"},
    {"vulnerability_class": "TransactionOrderDependence", "probability": 0.281, "tier": "SUSPICIOUS"},
    {"vulnerability_class": "ExternalBug",                "probability": 0.261, "tier": "SUSPICIOUS"},
    {"vulnerability_class": "UnusedReturn",               "probability": 0.249, "tier": "SUSPICIOUS"}
  ],
  "vulnerabilities": [
    {"vulnerability_class": "CallToUnknown", "probability": 0.638},
    {"vulnerability_class": "Reentrancy",    "probability": 0.620}
  ],
  "tier_thresholds": {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10},
  "thresholds": [0.40, 0.45, 0.35, 0.40, 0.50, 0.40, 0.40, 0.30, 0.35, 0.35],
  "truncated": false,
  "windows_used": 2,
  "num_nodes": 47,
  "num_edges": 89
}
```

### What Changes in the Agents

The routing logic reads from `ml_result["probabilities"]` (always the full 10-class dict)
and applies agent-level investigation thresholds (`DEEP_THRESHOLDS`) independently from the
ML's tier thresholds. These are separate parameters with separate purposes:

- **ML tier threshold (0.55 / 0.25):** where the model draws its confidence tiers
- **Agent DEEP_THRESHOLD (0.30–0.45 per class):** at what probability the agent decides
  an investigation is warranted — can be tuned per deployment without retraining

---

## 9. Node Specifications

### Node 1: ml_assessment

**Purpose:** Obtain the ML model's probability assessment.  
**Calls:** `sentinel-inference :8010` via MCP  
**Input state keys:** `contract_code`  
**Output state keys:** `ml_result`, `error`  

**Behavior:**
1. POST `contract_code` to inference server tool `predict`
2. Receives three-tier `ml_result`
3. Logs top CONFIRMED class (or "no confirmed" if only SUSPICIOUS/safe)
4. On tool failure: returns `ml_result={}`, sets `error` — graph continues to synthesizer
   which will emit a partial report noting ML failure

**Key invariant:** This node never fails the graph. Any exception is caught and stored in
`error`. The synthesizer handles missing `ml_result` with a "manual review required" recommendation.

---

### Node 2: graph_explain (Phase 1)

**Purpose:** Explain which regions of the contract graph the model attended to most.  
**Calls:** `graph_inspector_server :8013` via MCP  
**Input state keys:** `ml_result`, `contract_code`  
**Output state keys:** `ml_hotspots`, `graph_explanations`, `error`  

**Behavior:**
1. Calls `/hotspots` endpoint with top CONFIRMED+SUSPICIOUS classes and the contract source
2. Server runs GNN attention extraction + CodeBERT gradient×attention
3. Returns per-class: `{function_name, lines, node_ids, score}`
4. Calls `/subgraph` and `/features` for the top-scoring nodes per class
5. Stores structured hotspot data for evidence_router and rag_research to use

**Fallback:** If graph_inspector_server is unavailable, node returns `ml_hotspots=[]`
and `graph_explanations={}`. All downstream nodes handle missing hotspot data gracefully.

**Impact on downstream:**
- `rag_research`: query becomes "Reentrancy: external call before state update in withdraw()"
  instead of just "Reentrancy vulnerability" — significantly more targeted
- `synthesizer`: LLM prompt includes function names and feature descriptions

---

### Node 3: evidence_router

**Purpose:** Compute per-class routing decisions and log them for auditability.  
**Calls:** nothing (pure computation)  
**Input state keys:** `ml_result`  
**Output state keys:** `routing_decisions`  

**Key design point:** This node does NOT make the actual routing branch — that is the job of
the conditional edge function `_route_from_evidence_router()` in `graph.py`. This node's sole
purpose is to log the routing decisions into state so they appear in the final report.
The function and the node both call `compute_active_tools()` — the routing module is the
single source of truth.

**Routing logic** (see §10 for full specification):
- Reads `ml_result["probabilities"]` — the full 10-class dict
- Applies `DEEP_THRESHOLDS` per class — returns which tool nodes activate
- CONFIRMED classes: all tools in `ROUTING_RULES[cls]`
- SUSPICIOUS classes above DEEP_THRESHOLD: tools based on class risk profile
- SUSPICIOUS below DEEP_THRESHOLD: no tools activated (WATCH verdict, fast path)
- Zero classes above any threshold: fast path to synthesizer

**Routing decision log format:**

```
"Reentrancy prob=0.620 [CONFIRMED] >= threshold=0.35 → static_analysis+rag_research"
"DenialOfService prob=0.312 [SUSPICIOUS] >= threshold=0.30 → static_analysis+rag_research"
"UnusedReturn prob=0.249 [SUSPICIOUS] < threshold=0.45 → WATCH (no tools activated)"
"Timestamp prob=0.197 [NOTEWORTHY] < threshold=0.35 → skip"
```

---

### Node 4: rag_research

**Purpose:** Retrieve documented exploit patterns matching the flagged vulnerability classes.  
**Calls:** `sentinel-rag :8011` via MCP  
**Input state keys:** `ml_result`, `ml_hotspots`, `contract_code`  
**Output state keys:** `rag_results`, `error`  

**Query construction:**

For each CONFIRMED class (up to 3, ranked by probability):
```
"{class} smart contract exploit: {hotspot_function_name} {feature_description}
 pattern: {contract_snippet_200_chars}"
```

If no CONFIRMED classes exist, for top SUSPICIOUS class:
```
"possible {class} vulnerability: {contract_snippet_200_chars}"
```

If hotspots not available (Phase 0), use:
```
"{class} smart contract vulnerability exploit attack pattern: {contract_snippet}"
```

**Result aggregation:** All RAG queries run sequentially (RAG server is single process);
results are merged and deduped by chunk ID. Final `rag_results` is a flat ranked list.

**High-value RAG classes** — rag_research activates for SUSPICIOUS tier (in addition to
CONFIRMED) for these classes due to rich exploit history:
`{Reentrancy, IntegerUO, DenialOfService, ExternalBug, TOD}`

---

### Node 5: static_analysis

**Purpose:** Run Slither static analysis scoped to the vulnerability classes flagged by ML.  
**Calls:** Slither Python library (in-process, no MCP)  
**Input state keys:** `ml_result`, `contract_code`, `contract_address`  
**Output state keys:** `static_findings`, `error`  

**Detector scoping:**

Slither supports 90+ detectors. Running all on every contract is slow and noisy.
The `CLASS_TO_DETECTORS` matrix maps each vulnerability class to the detectors most likely
to confirm or refute it:

```python
CLASS_TO_DETECTORS = {
    "Reentrancy":          ["reentrancy-eth", "reentrancy-no-eth",
                            "reentrancy-events-and-order", "reentrancy-benign"],
    "IntegerUO":           ["integer-overflow", "toctou", "unchecked-lowlevel"],
    "GasException":        ["costly-loop", "calls-loop", "incorrect-exp"],
    "Timestamp":           ["timestamp"],
    "TOD":                 ["tx-origin", "controlled-delegatecall", "msg-value-loop"],
    "ExternalBug":         ["arbitrary-send-eth", "low-level-calls",
                            "unchecked-send", "controlled-delegatecall"],
    "CallToUnknown":       ["low-level-calls", "controlled-delegatecall", "delegatecall-loop"],
    "MishandledException": ["unchecked-send", "unchecked-lowlevel",
                            "unchecked-transfer", "return-bomb"],
    "UnusedReturn":        ["unused-return"],
    "DenialOfService":     ["calls-loop", "costly-loop", "msg-value-loop"],
}
```

Only detectors for ML-flagged classes (above DEEP_THRESHOLD in `probabilities`) are activated.
Benchmark: running scoped detectors is 3–8× faster than running all detectors on large contracts.

**Finding schema:**
```python
{
    "tool":           "slither",
    "detector":       "reentrancy-eth",
    "impact":         "High",          # High | Medium | Low | Informational
    "confidence":     "High",          # High | Medium | Low
    "description":    "...",           # human-readable Slither description
    "lines":          [45, 46, 52],    # affected source lines
    "function_names": ["withdraw"],    # extracted from elements
}
```

**ExternalBug special handling** (see §15 for full specification):
When ExternalBug is in flagged classes, `static_analysis` also calls
`_extract_external_call_summary()` which uses `fn.high_level_calls` to enumerate
typed external calls — producing structured data the LLM can reason over.

---

### Node 6: audit_check

**Purpose:** Retrieve prior on-chain audit records for this contract.  
**Calls:** `sentinel-audit :8012` via MCP  
**Input state keys:** `contract_address`  
**Output state keys:** `audit_history`, `error`  

**Behavior:**
- Queries `AuditRegistry` by contract address (last 10 records)
- Returns reverse-chronological list of prior audit results
- Empty address → returns `audit_history=[]` (skipped gracefully)
- Enables: "this contract was audited 3 weeks ago and had no Reentrancy findings;
  the ML now flags Reentrancy at 0.62 — this is a regression"

---

### Node 7: cross_validator (Phase 2)

**Purpose:** Reconcile ML, static, and RAG evidence per flagged class into a structured verdict.  
**Calls:** LLM client (only for DISPUTED cases)  
**Input state keys:** `ml_result`, `static_findings`, `rag_results`  
**Output state keys:** `verdicts`, `confirmations`, `contradictions`  

**Rule-based verdict logic (no LLM cost for clear cases):**

```python
def _compute_class_verdict(cls, prob, tier, static_findings, rag_results):
    slither_match = any(
        f["detector"] in CLASS_TO_DETECTORS.get(cls, [])
        and f["impact"] in ("High", "Medium")
        for f in static_findings
    )
    rag_score = max((r["score"] for r in rag_results), default=0.0)

    if tier == "CONFIRMED" and slither_match:       return "CONFIRMED"
    if tier == "CONFIRMED" and rag_score >= 0.80:   return "CONFIRMED"
    if tier == "CONFIRMED" and rag_score >= 0.50:   return "LIKELY"
    if tier == "CONFIRMED":                          return "DISPUTED"
    if tier == "SUSPICIOUS" and slither_match:       return "LIKELY"
    if tier == "SUSPICIOUS" and rag_score >= 0.80:   return "LIKELY"
    if tier == "SUSPICIOUS":                         return "WATCH"
    return "SAFE"
```

**LLM adjudication (only for DISPUTED cases):**

DISPUTED cases are where the ML is confident (CONFIRMED tier) but nothing else agrees.
These are the highest-value LLM calls: a DISPUTED Reentrancy finding on a $50M DeFi protocol
either means the model is wrong, or there's a novel pattern that static analysis doesn't catch.

The LLM receives all available evidence (static findings, RAG chunks, hotspot data, code
snippet) and returns a structured verdict override with reasoning. One LLM call handles all
DISPUTED classes in the contract simultaneously.

**LLM prompt structure for DISPUTED adjudication:**
```
System: You are a senior smart contract security auditor adjudicating disputes between
        an ML vulnerability detector and static analysis tools.

User:   Contract: {address}
        Disputed classes: [{class, ml_prob, ml_tier}]
        ML reasoning (if hotspots available): {graph_explanations}
        Slither findings (all active detectors, no matches): {detector_list_run}
        RAG evidence (closest matches): {top_3_rag_chunks}
        Contract code (first 800 chars): {code_snippet}

        For each disputed class: return {"class": ..., "verdict": "LIKELY"|"DISPUTED",
        "reasoning": "one sentence"}.
        Output only valid JSON.
```

---

### Node 8: synthesizer

**Purpose:** Assemble the final audit report from all node outputs.  
**Calls:** LLM client (optional — for CONFIRMED/LIKELY verdicts)  
**Input state keys:** all of the above  
**Output state keys:** `final_report`, `narrative`, `verdicts`, `confirmations`  

**Two-phase synthesis:**

**Phase A — Structured report assembly (always runs, no LLM):**
- Aggregates verdicts from cross_validator (Phase 2) or computes rule-based verdicts (Phase 0/1)
- Computes `overall_verdict` as max-rank across all per-class verdicts
- Builds `vulnerability_verdicts` list with evidence_sources and severity per class
- Writes rule-based `recommendation` as fallback
- Includes routing trace, static findings, RAG chunks, audit history

**Phase B — LLM narrative (runs when CONFIRMED or LIKELY verdict exists):**
- Structured Markdown with four mandatory sections:
  - `## Severity` — one of: CRITICAL / HIGH / MEDIUM / LOW / INFORMATIONAL
  - `## Vulnerability Summary` — 2–3 sentences on what was detected and why it's dangerous
  - `## Exploit Pattern` — how an attacker could exploit this, referencing RAG evidence
  - `## Recommended Fix` — concrete, actionable mitigation steps
- Falls back to rule-based recommendation on any failure (timeout, LLM unavailable)
- Maximum wait: 45 seconds (asyncio timeout)

**Report persistence:** If `contract_address` is known, `final_report` is written to
`agents/data/reports/{contract_address}.json` for the feedback loop bridge.

---

## 10. Evidence Routing

### Per-Class Routing Matrix

`DEEP_THRESHOLDS` are agent investigation thresholds — tuned independently from the ML's
tier thresholds. These determine when the agents spend resources investigating.

```python
DEEP_THRESHOLDS = {
    "Reentrancy":             0.35,   # lower than inference threshold — catch borderlines
    "IntegerUO":              0.35,
    "GasException":           0.40,
    "Timestamp":              0.35,
    "TOD":                    0.35,
    "ExternalBug":            0.35,   # lower because model structurally misses this class
    "CallToUnknown":          0.40,
    "MishandledException":    0.40,
    "UnusedReturn":           0.45,   # higher — semantic gap means many FPs
    "DenialOfService":        0.30,   # lower — severely underrepresented class
}
```

`ROUTING_RULES` maps each class to the tools that should investigate it:

```python
ROUTING_RULES = {
    "Reentrancy":             ["static_analysis", "rag_research"],
    "IntegerUO":              ["static_analysis", "rag_research"],
    "GasException":           ["static_analysis"],
    "Timestamp":              ["static_analysis", "rag_research"],
    "TOD":                    ["static_analysis", "rag_research"],
    "ExternalBug":            ["static_analysis", "rag_research"],
    "CallToUnknown":          ["static_analysis", "rag_research"],
    "MishandledException":    ["static_analysis"],
    "UnusedReturn":           ["static_analysis"],
    "DenialOfService":        ["static_analysis", "rag_research"],
}
```

### Routing Algorithm

```python
def compute_active_tools(ml_result: dict) -> list[str]:
    active: set[str] = set()
    probs = ml_result.get("probabilities", {})   # full 10-class dict

    if not probs:
        # Legacy fallback for old binary schema
        for vuln in ml_result.get("vulnerabilities", []):
            cls, prob = vuln["vulnerability_class"], vuln["probability"]
            if prob >= DEEP_THRESHOLDS.get(cls, 0.40):
                active.update(ROUTING_RULES.get(cls, []))
        return sorted(active)

    for cls, prob in probs.items():
        if prob >= DEEP_THRESHOLDS.get(cls, 0.40):
            active.update(ROUTING_RULES.get(cls, []))

    return sorted(active)   # sorted for deterministic log output
```

### Routing Decision Log

Every routing decision is stored in `routing_decisions` for inclusion in the final report.
One entry per vulnerability class above `NOTEWORTHY` tier (prob ≥ 0.10):

```
Reentrancy    prob=0.620 [CONFIRMED] >= threshold=0.35 → static_analysis+rag_research
DoS           prob=0.312 [SUSPICIOUS] >= threshold=0.30 → static_analysis+rag_research
GasException  prob=0.302 [SUSPICIOUS] >= threshold=0.40 → skip (below threshold)
IntegerUO     prob=0.314 [SUSPICIOUS] >= threshold=0.35 → static_analysis+rag_research
UnusedReturn  prob=0.249 [SUSPICIOUS] < threshold=0.45 → WATCH (fast path for this class)
Timestamp     prob=0.197 [NOTEWORTHY] < threshold=0.35 → skip
```

---

## 11. Verdict Model

### Five Verdict Levels

| Verdict | Condition | Meaning | Analyst action |
|---------|-----------|---------|---------------|
| `CONFIRMED` | CONFIRMED tier + ≥1 tool corroboration (Slither High/Med OR RAG score ≥ 0.80) | Multiple independent sources agree | Priority review; ZK proof candidate |
| `LIKELY` | CONFIRMED tier + partial corroboration (RAG 0.50–0.79) OR SUSPICIOUS tier + Slither High/Med OR SUSPICIOUS tier + RAG ≥ 0.80 | Strong signal, not fully corroborated | Targeted review of flagged functions |
| `DISPUTED` | CONFIRMED tier + zero corroboration from any tool | ML confident, nothing else agrees | LLM adjudication; could be novel pattern or FP |
| `WATCH` | SUSPICIOUS tier + no tool corroboration above threshold | Non-trivial signal, insufficient evidence | Note in report; monitor in next audit |
| `SAFE` | Below DEEP_THRESHOLD for all classes | Nothing worth investigating | Standard due diligence |

### Verdict Severity Mapping

```python
def verdict_to_severity(verdict: str, probability: float) -> str:
    if verdict == "CONFIRMED":
        if probability >= 0.85: return "CRITICAL"
        if probability >= 0.70: return "HIGH"
        return "MEDIUM"
    if verdict == "LIKELY":
        if probability >= 0.70: return "HIGH"
        return "MEDIUM"
    if verdict == "DISPUTED": return "LOW"
    if verdict == "WATCH":    return "INFO"
    return "INFO"
```

### Overall Verdict

The contract-level `overall_verdict` is the maximum severity across all per-class verdicts:

```python
VERDICT_RANK = {"CONFIRMED": 5, "LIKELY": 4, "DISPUTED": 3, "WATCH": 2, "SAFE": 1}
overall_verdict = max(verdicts.values(), key=lambda v: VERDICT_RANK.get(v, 0))
```

### Evidence Source Notation

All evidence sources are cited in a standard format:

| Source | Format | Example |
|--------|--------|---------|
| ML model | `ml:{probability}` | `ml:0.872` |
| ML tier | `ml_tier:{tier}` | `ml_tier:CONFIRMED` |
| Slither | `slither:{detector}` | `slither:reentrancy-eth` |
| RAG | `rag:{score}` | `rag:0.81` |
| Audit history | `history:{verdict}` | `history:prior_clean` |
| LLM adjudication | `llm:adjudicated` | `llm:adjudicated` |

---

## 12. Report Format

### Per-Class Finding

```python
@dataclass
class VulnerabilityFinding:
    vulnerability_class: str
    ml_probability:      float
    ml_tier:             str           # "CONFIRMED" | "SUSPICIOUS" | "NOTEWORTHY"
    verdict:             str           # see §11
    severity:            str           # CRITICAL | HIGH | MEDIUM | LOW | INFO
    evidence_sources:    list[str]     # e.g. ["ml:0.872", "slither:reentrancy-eth", "rag:0.81"]
    slither_findings:    list[dict]    # matching Slither findings for this class (may be [])
    rag_matches:         list[dict]    # top RAG chunks for this class (may be [])
    location:            str | None    # "function withdraw() lines 45–72" (from hotspots/Slither)
    graph_hotspot:       dict | None   # Phase 1: {function_name, lines, score, node_ids}
    explanation:         str           # 1–2 sentence LLM or rule-based explanation
```

### Audit Report

```python
@dataclass
class AuditReport:
    # Identity
    contract_address:  str
    timestamp:         str            # ISO 8601
    sentinel_version:  str            # "SENTINEL-v1.0 Run4"

    # Overall assessment
    overall_verdict:   str            # max-rank verdict across all classes
    overall_severity:  str
    risk_score:        float          # max(probability) across CONFIRMED+LIKELY classes

    # Per-class findings
    vulnerability_findings: list[VulnerabilityFinding]

    # Observability
    tools_activated:   list[str]      # which tools ran
    routing_trace:     list[str]      # full routing decision log
    path_taken:        str            # "fast" | "deep"
    audit_duration_ms: int

    # Evidence summary
    rag_evidence:      list[dict]     # all retrieved RAG chunks
    static_findings:   list[dict]     # all Slither findings
    audit_history:     list[dict]     # prior on-chain records

    # Model metadata
    ml_label:          str            # raw ML label
    ml_windows_used:   int
    ml_truncated:      bool
    num_nodes:         int
    num_edges:         int
    ml_probabilities:  dict[str, float]  # full 10-class vector always included

    # Output
    recommendation:    str            # LLM narrative or rule-based fallback
    narrative:         str | None     # full Markdown narrative
    error:             str | None     # any non-fatal error during the run
```

### Report JSON Example (abbreviated)

```json
{
  "contract_address": "0x1234...abcd",
  "timestamp": "2026-05-27T15:42:11Z",
  "overall_verdict": "CONFIRMED",
  "overall_severity": "HIGH",
  "risk_score": 0.872,
  "vulnerability_findings": [
    {
      "vulnerability_class": "Reentrancy",
      "ml_probability": 0.872,
      "ml_tier": "CONFIRMED",
      "verdict": "CONFIRMED",
      "severity": "HIGH",
      "evidence_sources": ["ml:0.872", "ml_tier:CONFIRMED", "slither:reentrancy-eth", "rag:0.81"],
      "location": "function withdraw() lines 45–72",
      "explanation": "External call made before state update — classic reentrancy pattern matching TheDAO exploit."
    },
    {
      "vulnerability_class": "DenialOfService",
      "ml_probability": 0.312,
      "ml_tier": "SUSPICIOUS",
      "verdict": "WATCH",
      "severity": "INFO",
      "evidence_sources": ["ml:0.312", "ml_tier:SUSPICIOUS"],
      "explanation": "Non-trivial DoS signal detected; insufficient corroboration for confirmed finding. Monitor in next audit."
    }
  ],
  "tools_activated": ["static_analysis", "rag_research", "audit_check"],
  "path_taken": "deep",
  "ml_probabilities": {
    "Reentrancy": 0.872, "CallToUnknown": 0.641, "DenialOfService": 0.312, "..."
  },
  "narrative": "## Severity\nHIGH\n\n## Vulnerability Summary\n..."
}
```

---

## 13. MCP Server Architecture

### Server Inventory

| Server | Port | Status | Wraps | Notes |
|--------|------|--------|-------|-------|
| `sentinel-inference` | :8010 | ✅ Implemented | ML predictor `/predict`, `/batch_predict` | Passes three-tier schema through; mock mode for development |
| `sentinel-rag` | :8011 | ✅ Implemented | FAISS+BM25+reranker `/search` | Hybrid retrieval; BM25 for keyword, FAISS for semantic |
| `sentinel-audit` | :8012 | ✅ Implemented | AuditRegistry on-chain reader `/get_audit_history` | Reads from deployed contract |
| `graph-inspector` | :8013 | 🔲 Phase 1 | GNN attention + CodeBERT gradients `/hotspots`, `/subgraph`, `/features` | Wraps `ml/src/inference/predictor.py` with attention hooks |
| `econ-sim` | :8015 | 🔲 Phase 3 | Pattern-matching economic scenarios for TOD/DoS | Canned scenarios, not full simulation |

### Transport Design

All servers use MCP over SSE (HTTP persistent event stream):
- `/sse` — SSE connection endpoint (one per client session)
- `/messages/` — JSON-RPC POST endpoint (tool invocations)
- `/health` — liveness probe for monitoring

The agents process opens short-lived SSE connections per tool call (one TCP handshake ≈ 1ms
on localhost). Connection pooling to be considered in Phase 2 if RTT measurements show
this is a bottleneck.

### inference_server Mock Mode

`MODULE1_MOCK=true` in `agents/.env` activates the mock path, which returns a realistic
three-tier schema without calling the actual ML model. The mock schema exactly mirrors the
live predictor output — swapping mock → real requires zero code changes in the agents.

---

## 14. Escalation Tiers

The key design constraint: **cost proportional to risk**. LLM calls are expensive; Slither
is fast; RAG adds meaningful latency. The escalation tiers enforce this constraint explicitly.

| Tier | Condition | Nodes activated | LLM calls | Expected latency |
|------|-----------|-----------------|-----------|-----------------|
| **Tier 0 — Fast pass** | All probs < DEEP_THRESHOLDS | ml_assessment → evidence_router → synthesizer | 0 | < 5s |
| **Tier 1 — Static only** | SUSPICIOUS classes above threshold for static-only classes (GasException, UnusedReturn, MishandledException) | + static_analysis → audit_check | 0 | 5–15s |
| **Tier 2 — Standard deep** | Any CONFIRMED class, or SUSPICIOUS class above threshold for rag+static classes | + rag_research (parallel) | 0 | 15–30s |
| **Tier 3 — LLM adjudication** | Any DISPUTED verdict after Tier 2 | + cross_validator LLM call | 1 | 30–50s |
| **Tier 4 — Full narrative** | Any CONFIRMED or LIKELY verdict | + synthesizer LLM call | 1 (additive) | +10–20s |

**Maximum LLM calls per audit: 2** (Tier 3 adjudication + Tier 4 narrative, if both triggered).  
**Minimum LLM calls: 0** (Tier 0, majority of contracts in a large scan).

---

## 15. Special Case: ExternalBug

### Why ExternalBug Is Different

ExternalBug (oracle manipulation, malicious callback exploitation) requires reasoning about
what an external contract *does*, not what the audited contract *calls*. The GNN only sees
the audited contract's graph.

The graph extractor (`_select_contract()`) correctly filters out interface definitions
(`IPriceOracle`, `IVault`, `ICallback`) — these are not the contract being audited. But
this means the GNN has no visibility into the trust relationship between the contract
and its external dependencies.

Additionally, contracts that call external functions via typed interfaces show
`call_target_typed=1.00` in the node features. The GNN interprets typed calls as *safer*
than raw `.call()` (which is correct for normal usage), but this is the opposite signal
needed for oracle manipulation detection.

**Conclusion:** ExternalBug cannot be reliably detected at ML layer without inter-contract
analysis. The agent layer is the right place.

### Agent-Layer Solution

**Step 1 — Slither external call enumeration**

```python
def _extract_external_call_summary(sl_obj, flagged_classes: set) -> list[dict] | None:
    if "ExternalBug" not in flagged_classes:
        return None

    calls = []
    for contract in sl_obj.contracts:
        if contract.is_from_dependency() or contract.is_interface:
            continue
        for fn in contract.functions:
            for called_contract, called_fn in fn.high_level_calls:
                calls.append({
                    "caller_function": fn.name,
                    "called_contract": called_contract.name,
                    "called_function": called_fn.name if called_fn else "unknown",
                    "is_interface":    called_contract.is_interface,
                    "is_in_value_path": any(
                        v.name in ("value", "amount", "price", "balance", "rate")
                        for v in fn.local_variables
                    ),
                })
    return calls or None
```

This returns structured data: which functions call which external contracts, whether
those contracts are interface definitions, and whether the call path involves value-sensitive
variables.

**Step 2 — Targeted RAG query**

When ExternalBug is flagged, `rag_research` constructs a query that names the called interfaces:

```
"oracle manipulation ExternalBug: {contract_name} calls {called_interfaces} 
 in value-sensitive context: {value_variable_names}"
```

This retrieves exploit writeups specifically about oracle manipulation and callback attacks,
not generic "external call" patterns.

**Step 3 — LLM context enrichment**

The synthesizer includes the external call summary in the LLM prompt when ExternalBug is
SUSPICIOUS or higher:

```
External calls detected:
  - borrow() → IPriceOracle.getPrice() [interface, value-sensitive: True]
  - borrow() → IVault.deposit() [interface, value-sensitive: True]

Question: Could an attacker manipulate IPriceOracle.getPrice() to affect the outcome
of borrow()? Does the contract validate the returned price?
```

This gives the LLM the right context to reason about oracle manipulation without requiring
inter-contract graph analysis.

---

## 16. Implementation Plan

### Phase 0 — Foundation (COMPLETE)

All Phase 0 items are implemented and tested (46/46 tests pass):
- Per-class routing with DEEP_THRESHOLDS and ROUTING_RULES
- Scoped Slither detectors via CLASS_TO_DETECTORS
- SqliteSaver checkpointing
- `routing_decisions` append-reducer in state
- `verdicts`, `confirmations`, `contradictions` fields in AuditState
- Rule-based verdict computation in synthesizer
- LLM narrative generation with 45s timeout fallback
- Full test suite covering routing logic, verdict model, graph compilation

---

### Phase A — Three-Tier ML Output (Immediate — no retraining)

**Estimated effort:** 3–4 hours  
**Unblocked:** yes

| Task | File | Change |
|------|------|--------|
| A1 | `ml/src/inference/predictor.py` | Add `tier_confirmed_threshold`, `tier_suspicious_threshold` params to `__init__`. Update `_format_result()` to emit `probabilities`, `confirmed`, `suspicious`, `label`, `tier_thresholds`. |
| A2 | `ml/api/` (when built) | Update response schema docs |
| A3 | `agents/src/orchestration/routing.py` | `compute_active_tools()` and `build_routing_decisions()` read from `probabilities` dict. Add tier annotation to log strings. Add `WATCH` to verdict logic. |
| A4 | `agents/src/orchestration/nodes.py` | `static_analysis`: read flagged classes from `probabilities`. `rag_research`: build per-class queries from `confirmed`+top `suspicious`. `synthesizer`: include WATCH verdicts for SUSPICIOUS classes. `ml_assessment`: update log format. |
| A5 | `agents/src/orchestration/state.py` | Update `ml_result` field comment to document three-tier schema. |
| A6 | `agents/src/mcp/servers/inference_server.py` | Update `_mock_prediction()` to return three-tier schema. Update schema comment in `_call_inference_api()`. |
| A7 | `agents/tests/` | Update `_ml()` helper in `test_routing_phase0.py`. Add tests: three-tier routing, WATCH verdict, SUSPICIOUS-above-threshold routing. |

**Verification:** Run `Predictor.predict_source()` on `ml/scripts/test_contracts/01_reentrancy_classic.sol`.
Confirm `probabilities` has 10 entries, `confirmed` has Reentrancy≈0.62, `suspicious` has DoS≈0.31.

---

### Phase B — ExternalBug Agent Enhancement

**Estimated effort:** 2–3 hours  
**Unblocked:** yes (Slither is already in-process in static_analysis node)

| Task | File | Change |
|------|------|--------|
| B1 | `agents/src/orchestration/nodes.py` | Add `_extract_external_call_summary()`. Call it in `static_analysis` when ExternalBug is flagged. Attach result to static_findings or state directly. |
| B2 | `agents/src/orchestration/nodes.py` | Update `rag_research` query construction for ExternalBug: include interface names and value-sensitive flag from external_call_summary if available. |
| B3 | `agents/src/orchestration/nodes.py` | Update `synthesizer` LLM prompt to include `external_call_summary` block when ExternalBug is SUSPICIOUS or higher. |
| B4 | `agents/tests/` | Add test: static_analysis returns external_call_summary for a contract with typed external calls. |

---

### Phase 1 — Evidence Richness (Graph Inspector)

**Estimated effort:** 1–2 days  
**Unblocked:** Run 4 checkpoint exists (`ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt`)

| Task | File | Change |
|------|------|--------|
| P1-1 | `ml/src/inference/predictor.py` | Add attention hook registration during forward pass. Add `predict_with_hotspots()` method that returns both prediction and per-class attention scores. |
| P1-2 | NEW: `ml/api/routers/hotspots.py` | Add `/hotspots` POST endpoint wrapping `predict_with_hotspots()`. Returns `[{class, fn_name, lines, node_ids, score}]`. |
| P1-3 | NEW: `agents/src/mcp/servers/graph_inspector_server.py` | MCP server at :8013. Tools: `get_hotspots(contract_code, classes)`, `get_subgraph(node_ids)`, `get_features(node_ids)`. Calls ML API `/hotspots` endpoint. |
| P1-4 | `agents/src/orchestration/nodes.py` | Add `graph_explain` node. Calls `graph_inspector_server` with top CONFIRMED+SUSPICIOUS classes. Stores in `ml_hotspots` and `graph_explanations`. Falls back gracefully if unavailable. |
| P1-5 | `agents/src/orchestration/graph.py` | Wire `graph_explain` node between `ml_assessment` and `evidence_router`. |
| P1-6 | `agents/src/orchestration/nodes.py` | Update `rag_research` query construction to use `ml_hotspots` function names and feature descriptions when available. |
| P1-7 | `agents/src/orchestration/nodes.py` | Update `synthesizer` LLM prompt to include graph hotspot context when available. |

---

### Phase 2 — Cross-Validator and Routing Intelligence

**Estimated effort:** 1 day  
**Requires:** Phase A complete

| Task | File | Change |
|------|------|--------|
| P2-1 | `agents/src/orchestration/nodes.py` | Implement `cross_validator` node with rule-based verdict for CONFIRMED/LIKELY/WATCH/SAFE and LLM adjudication for DISPUTED only. |
| P2-2 | `agents/src/orchestration/graph.py` | Wire `cross_validator` between `audit_check` and `synthesizer`. |
| P2-3 | `agents/src/orchestration/nodes.py` | Update `synthesizer` to use `cross_validator` verdicts when available (fallback to rule-based when Phase 2 not yet wired). |
| P2-4 | `agents/src/orchestration/graph.py` | Upgrade routing to LangGraph `Send` API for per-class fan-out (allows class-specific RAG queries to run in parallel). |
| P2-5 | `agents/tests/` | Add cross_validator tests: CONFIRMED with slither match, DISPUTED with no match, WATCH for SUSPICIOUS. |

---

### Phase 3 — Advanced Tools

**Estimated effort:** 2–3 days  
**Requires:** Phase 2 complete

| Task | Description |
|------|-------------|
| P3-1 | `econ_sim_server :8015` — canned scenario engine for TOD/DoS (pattern matching, not execution). TOD: "can sequenced tx pair change conditional outcome?" DoS: "is there a loop over unbounded array with external calls?" |
| P3-2 | Wire `econ_assessment` node for TOD and DenialOfService classes. |
| P3-3 | Mythril integration (scoped to hotspot functions only, subprocess-isolated, 60s hard timeout). Only for contracts with CONFIRMED Reentrancy or IntegerUO verdict. |

---

### Phase 4 — Feedback Loop Closure

**Estimated effort:** 1 day  
**Requires:** On-chain deployment

| Task | Description |
|------|-------------|
| P4-1 | Model drift corrections: when on-chain record has human-confirmed FP → write to `ml/data/feedback/corrections.csv` for active-learning next run. |
| P4-2 | Class-aware RAG ingestion: GitHub exploit fetcher maps each new writeup to vulnerability class. Per-class RAG queries improve over time. |
| P4-3 | Disagreement logging: log every case where ML flags a class but Slither finds nothing (and vice versa). Builds `ml/data/feedback/disagreements.csv`. |

---

## 17. Success Metrics

The agents system is measurable against a ground-truth test set.

### Primary Metrics

| Metric | Target | Measurement method |
|--------|--------|--------------------|
| **CONFIRMED precision** | > 0.85 | Fraction of CONFIRMED verdicts that are true positives on known-vulnerable contracts |
| **SAFE recall** | > 0.95 | Fraction of known-clean contracts that receive SAFE verdict (false negative rate) |
| **WATCH coverage** | > 0.70 | Fraction of true positives that appear as at least WATCH (not missed entirely) |
| **DISPUTED rate** | < 0.20 | Fraction of flagged contracts where all evidence disagrees |
| **Audit latency (Tier 1)** | < 30s | Static-only path, no LLM, end-to-end wall clock |
| **Audit latency (Tier 2)** | < 50s | With Slither + RAG, no LLM |
| **Audit latency (Tier 4)** | < 75s | Full path with LLM narrative |
| **Fast-path rate** | > 0.50 | Fraction of contracts taking Tier 0/1 path (cost control at scale) |

### Secondary Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Evidence breadth** | ≥ 2 sources for CONFIRMED | CONFIRMED verdicts with only `ml:` in evidence_sources are suspect |
| **ExternalBug recall** | > 0.60 (vs 0.00 without agent enhancement) | Measured with contracts known to have oracle manipulation vulnerability |
| **LLM call rate** | < 0.30 of deep-path audits trigger Tier 3 adjudication | High rate indicates ML over-flagging or Slither under-detecting |
| **Report completeness** | `routing_trace` populated for every contract | Verifies auditability requirement |

---

## 18. Scope Boundaries

### What This Module Does NOT Build

**Separate agent processes (Orchestrator/Triage/Analysis pattern):**
LangGraph node boundaries provide the same cognitive separation as separate agent processes
without inter-process message passing overhead. The cross_validator node achieves the
"debate pattern" in one structured call at lower cost.

**Full Mythril symbolic execution on whole contracts:**
Too slow (minutes per contract) and prone to OOM crashes. Mythril is in scope for Phase 3
scoped to hotspot functions only, subprocess-isolated with a 60-second hard timeout.

**Foundry fuzz tests in the agent loop:**
Foundry fuzzing takes seconds to minutes per property. Not compatible with real-time
audit latency targets. Belongs in an offline analysis pipeline, not the agent loop.

**Hardcoded vulnerability triggers ("if any external call exists → flag ExternalBug"):**
60–70% of production DeFi contracts have external calls. Manual rules at this level produce
unacceptable false positive rates. ExternalBug handling uses Slither structured data + LLM
reasoning, not keyword matching.

**Per-class tuning of ML tier thresholds inside routing.py:**
If ML tier thresholds need per-class adjustment, that belongs in the predictor config file.
The agent's DEEP_THRESHOLDS and the ML's tier_thresholds are separate parameters for
separate purposes — conflating them would create a maintenance nightmare.

**Cross-contract dependency graph (Phase 3+):**
Full cross-contract analysis (tracking CALL relationships across a batch of contracts) is
in scope for Phase 3 after the core pipeline is stable. It requires topological batch
ordering and significantly more Slither API surface than the current implementation.

**On-chain write from the agent module:**
AuditRegistry writes are handled by the ZK proof pipeline (Module 2) and the API layer
(Module 6). The agent module only reads from the registry. Writing requires the ZK proof
to be available — that dependency is outside this module's scope.
