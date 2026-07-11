# Orchestration — ML Integration in Agents

> **Scope:** the "ML module" inside `~/projects/sentinel/agents/` — the code
> that consumes ML signals from Module 1 (`~/projects/sentinel/ml/`) and uses
> them to drive the audit graph. This is a **cross-module reference**: the
> agent layer is the consumer, Module 1's FastAPI is the producer.
>
> **Source-of-truth is the code** in `src/orchestration/`, `src/mcp/servers/`,
> and `ml/api.py`. Verified: 2026-06-23.

---

## 1. Position in the System — Producer vs Consumer

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Module 1 (ml/)                                                         │
   │  FastAPI inference server (ml/api.py)                                   │
   │  ─────────────────────────────────                                       │
   │  /predict    contract_code → per-class probabilities                    │
   │  /hotspots   contract_code → function-level GNN attention scores        │
   │  /health     liveness probe                                              │
   │                                                                         │
   │  Run 12 checkpoint (active): F1=0.7004 tuned, F1=0.8743 honest OOD     │
   │  Schema: v9 (NODE_FEATURE_DIM=12, NUM_NODE_TYPES=14, NUM_EDGE_TYPES=12)│
   │  10 vuln classes (see § 3)                                              │
   │                                                                         │
   │  THIS IS THE PRODUCER. ml/api.py holds the model in memory.             │
   └────────────────────────────────────┬────────────────────────────────────┘
                                        │ HTTP (httpx)
                                        │ POST /predict, POST /hotspots
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │  Agents ML module (this dir)                                            │
   │  ─────────────────────────────                                          │
   │  Integrates ML signals into the LangGraph audit pipeline.               │
   │  TWO integration points:                                                │
   │    1. via MCP :8010 (sentinel-inference) — for graph nodes              │
   │    2. direct HTTP /hotspots — for graph_inspector_server :8013          │
   │                                                                         │
   │  Agents is the CONSUMER. It does not hold the model.                   │
   └─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Two Integration Paths

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Path A — via MCP (default for the audit graph)                             │
  │                                                                             │
  │  ② ml_assessment node (orchestration/nodes/ml_assessment.py)               │
  │       │                                                                     │
  │       │  async with sse_client("http://localhost:8010/sse")                 │
  │       │      await session.call_tool("predict", {"contract_code": ...})     │
  │       │                                                                     │
  │       ▼                                                                     │
  │  sentinel-inference MCP server (mcp/servers/inference_server.py) :8010      │
  │       │                                                                     │
  │       │  list_tools() → "predict" (inputSchema: contract_code required)     │
  │       │  call_tool("predict") → _call_inference_api()                       │
  │       │                                                                     │
  │       │  shared httpx.AsyncClient (lifespan-managed, A-20)                  │
  │       │  POST {MODULE1_INFERENCE_URL}/predict                              │
  │       │  body: {"source_code": contract_code}                              │
  │       │                                                                     │
  │       ▼                                                                     │
  │  Module 1 ml/api.py — POST /predict handler                                 │
  │       │                                                                     │
  │       │  Returns: label, probabilities, confirmed, suspicious,              │
  │       │            vulnerabilities, tier_thresholds, thresholds,             │
  │       │            truncated, windows_used, num_nodes, num_edges            │
  │       │                                                                     │
  │       ▼                                                                     │
  │  _call_inference_api returns dict → wraps in TextContent                    │
  │       │                                                                     │
  │       ▼                                                                     │
  │  ml_assessment parses → state["ml_result"] (three-tier schema)             │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Path B — direct HTTP (graph_inspector_server, for GNN attention)            │
  │                                                                             │
  │  ⑦ graph_explain node (orchestration/nodes/graph_explain.py)               │
  │       │                                                                     │
  │       │  async with sse_client("http://localhost:8013/sse")                 │
  │       │      await session.call_tool("get_graph_hotspots",                 │
  │       │                               {"contract_code": ...,               │
  │       │                                "flagged_classes": [...]})           │
  │       │                                                                     │
  │       ▼                                                                     │
  │  sentinel-graph-inspector MCP server :8013                                   │
  │       │                                                                     │
  │       │  list_tools() → "get_graph_hotspots"                                │
  │       │  call_tool → _analyze_hotspots_gnn()  [preferred]                   │
  │       │                                                                     │
  │       │  fresh httpx.AsyncClient (per-call; no pooling here)                │
  │       │  POST {SENTINEL_ML_API_URL}/hotspots                                │
  │       │  body: {"source_code": contract_code}                               │
  │       │                                                                     │
  │       ▼                                                                     │
  │  Module 1 ml/api.py — POST /hotspots handler                                │
  │       │                                                                     │
  │       │  Returns: {hotspots: [{fn_name, lines, node_id,                    │
  │       │                       node_type, score}],                          │
  │       │            hotspot_stats: {num_nodes, total_function_nodes},       │
  │       │            label, confirmed, suspicious}                            │
  │       │                                                                     │
  │       ▼                                                                     │
  │  _analyze_hotspots_gnn() transforms to graph_inspector format              │
  │       │                                                                     │
  │       ▼                                                                     │
  │  ml_hotspots → state["ml_hotspots"] + state["graph_explanations"]           │
  └─────────────────────────────────────────────────────────────────────────────┘
```

### Why Two Paths?

| Path | When | Latency | Used by |
|------|------|---------|---------|
| **A — via MCP** | the audit graph needs the ML prediction | one SSE handshake + one HTTP call (~20-50ms TCP + 5-15s inference) | `ml_assessment` (every audit) |
| **B — direct HTTP** | graph needs function-level GNN attention, not the prediction | one HTTP call (no MCP wrapper) | `graph_explain` (deep path only) |

Path B bypasses the MCP server because `graph_inspector_server` IS the MCP server — it calls Module 1 itself on behalf of the node. This is the "MCP server as thin client of Module 1" pattern.

---

## 3. The 10 Vulnerability Classes (v9 Schema, Module 1 Output)

```
  Module 1's ml/api.py emits a per-class probability vector. The agents
  module knows all 10 classes by name and uses them everywhere:
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Class                          DEEP_THR  ROUTING_RULES                │
  │  ───────────────────────────    ────────  ─────────────────────────    │
  │  Reentrancy                          0.35  static_analysis, rag_research
  │  IntegerUO                           0.35  static_analysis, rag_research
  │  GasException                        0.40  static_analysis            │
  │  Timestamp                           0.35  static_analysis, rag_research
  │  TransactionOrderDependence          0.35  static_analysis, rag_research
  │  ExternalBug                         0.40  static_analysis, rag_research
  │  CallToUnknown                       0.40  static_analysis, rag_research
  │  MishandledException                 0.40  static_analysis            │
  │  UnusedReturn                        0.45  static_analysis            │
  │  DenialOfService                     0.30  static_analysis, rag_research
  └────────────────────────────────────────────────────────────────────────┘
   Source: src/orchestration/routing.py:23-54  (DEEP_THRESHOLDS, ROUTING_RULES)
```

### Why DoS is 0.30 and UnusedReturn is 0.45
- **DoS** is rare in training data → use a lower threshold to investigate borderline cases
- **UnusedReturn** is noisy → use a higher threshold to avoid false-positive fan-out

### Why thresholds are LOWER than the inference threshold (0.50)
The model already decides "vulnerable vs safe" at 0.50. Routing thresholds are
deliberately lower (0.30-0.45) so that borderline ML predictions still get
investigated by Slither + RAG + LLM. We want second opinions on edge cases,
not blind trust in either direction.

---

## 4. The ml_result Schema (Three-Tier, 2026-05-27)

`ml_result` is what `ml_assessment` stores in `AuditState`. Defined in `state.py:55-74`.

```
  ml_result : dict
  ├── label             : "safe" | "suspicious" | "confirmed_vulnerable"
  │                       (derived from confirmed/suspicious/empty)
  │
  ├── probabilities     : dict[str, float]      ←  full 10-class vector (ALWAYS present)
  │     {                                                       }
  │     "Reentrancy":                 0.87,                      ← used by routing.py:_iter_class_probs
  │     "IntegerUO":                  0.12,                      ← used by consensus_engine (A.6)
  │     "GasException":               0.29,                      ← used by explainer (A.8)
  │     "Timestamp":                  0.08,                      ← used by cross_validator debate prompt
  │     "TransactionOrderDependence":0.05,
  │     "ExternalBug":                0.04,
  │     "CallToUnknown":              0.02,
  │     "MishandledException":        0.11,
  │     "UnusedReturn":               0.18,
  │     "DenialOfService":            0.03,
  │     }
  │
  ├── confirmed         : list[dict]   ←  classes with prob ≥ 0.55
  │     [{"vulnerability_class": "Reentrancy",
  │       "probability": 0.87, "tier": "CONFIRMED"}, ...]
  │
  ├── suspicious        : list[dict]   ←  classes with 0.25 ≤ prob < 0.55
  │     [{"vulnerability_class": "GasException",
  │       "probability": 0.29, "tier": "SUSPICIOUS"}, ...]
  │
  ├── vulnerabilities   : list[dict]   ←  LEGACY alias for confirmed (backward compat)
  │                                       no `tier` field
  │
  ├── tier_thresholds   : dict         ←  decision boundaries
  │     {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10}
  │
  ├── thresholds        : list[float]  ←  per-class tuned decision thresholds
  │                                       (10 entries, one per class)
  │
  ├── truncated         : bool         ←  True if source > 512 CodeBERT tokens
  │                                       (tail not analysed)
  │
  ├── windows_used      : int          ←  how many token windows were scored (>1 for long)
  │
  ├── num_nodes         : int          ←  AST node count (from graph_extractor)
  │
  └── num_edges         : int          ←  AST edge count
```

### Downstream consumers of ml_result

```
  ml_assessment  ──► state["ml_result"]
                          │
                          ├──► evidence_router  (per-class routing decisions)
                          │      src/orchestration/routing.py:build_routing_decisions()
                          │      uses probabilities (all 10 classes)
                          │
                          ├──► consensus_engine  (A.6 — weighted vote)
                          │      src/orchestration/consensus.py:consensus_vote()
                          │      uses confirmed/suspicious + per-class reliability
                          │
                          ├──► cross_validator  (A.4 — debate prompt)
                          │      src/orchestration/nodes/cross_validator.py
                          │      uses confirmed + top probabilities
                          │
                          ├──► explainer  (A.8 — LIME attribution)
                          │      src/orchestration/attribution.py
                          │      uses per-class probabilities
                          │
                          ├──► synthesizer  (top_vulnerability, risk_probability)
                          │      src/orchestration/nodes/synthesizer.py
                          │      uses max(probabilities) for top vuln
                          │
                          └──► reflection  (A.3 — failure-mode detection)
                                 src/orchestration/nodes/reflection.py
                                 flags known failure modes (truncated, ExternalBug FP)
```

---

## 5. Routing — From ML Signal to Deep/Fast Path

`routing.py` (263 lines) is the single source of truth for ML-signal-driven routing.

```
                  ┌────────────────────────────────────────────────────┐
                  │  ml_result (10-class probability vector)           │
                  └─────────────────────┬──────────────────────────────┘
                                        │
                                        ▼
              ┌─────────────────────────────────────────────────────┐
              │  _iter_class_probs(ml_result)                      │
              │  prefers "probabilities" dict; falls back to       │
              │  "vulnerabilities" list (legacy schema)            │
              └─────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────────────────┐
              │  build_routing_decisions(ml_result)                │
              │  → list[str] of human-readable decisions           │
              │  e.g. "Reentrancy prob=0.872 >= 0.35 → static_..  │
              │  +rag_research"                                    │
              │  e.g. "GasException prob=0.290 < 0.40 → skip"     │
              │  → state.routing_decisions (append-reducer)        │
              └─────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────────────────┐
              │  compute_active_tools(ml_result)                   │
              │  returns sorted list of tool node names to run     │
              │  empty list → fast path                            │
              │  non-empty   → fan-out to those tools              │
              └─────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────────────────┐
              │  evidence_router merges ML + quick_screen signals  │
              │  ─ Two-signal fast-path gate ─                    │
              │  • ML all classes < DEEP_THRESHOLDS, AND           │
              │  • quick_screen zero High/Critical hits            │
              │  Either fails → escalate to deep path              │
              └─────────────────────┬───────────────────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
        FAST PATH                                DEEP PATH
        (synthesizer directly)                    (fan-out then synthesize)
```

### CLASS_TO_DETECTORS — The bridge to Slither

```python
# src/orchestration/routing.py:62-112
CLASS_TO_DETECTORS = {
    "Reentrancy":          ["reentrancy-eth", "reentrancy-no-eth",
                            "reentrancy-events-and-order", "reentrancy-benign"],
    "IntegerUO":           ["integer-overflow", "toctou", "unchecked-lowlevel"],
    "ExternalBug":         ["arbitrary-send-eth", "low-level-calls",
                            "unchecked-send", "controlled-delegatecall"],
    # ... 7 more classes
}
DETECTOR_TO_CLASSES = {det: classes for det, classes in inverted(CLASS_TO_DETECTORS)}
```

This inverted map is used by `static_analysis` to scope Slither to only the relevant detectors (3-8x faster than running all 90+) and by `synthesizer` to match Slither findings back to ML-flagged classes for verdict computation.

---

## 6. Verdict Computation — Two Algorithms

### Algorithm 1: Rule-based (`compute_verdict` in `routing.py:187`)

```
  Inputs: cls, prob, static_findings, rag_results, path_taken
  Output: (verdict, evidence_sources)

  if path_taken == "fast":           return ("LIKELY",     [f"ml:{prob:.3f}"])
  if prob >= 0.50 and slither_match: return ("CONFIRMED",  [ml, slither])
  if prob >= 0.50 and rag_score≥0.80: return("CONFIRMED",  [ml, rag])
  if prob >= 0.50 and rag_score≥0.50: return("LIKELY",     [ml, rag])
  if prob >= 0.50:                   return ("DISPUTED",   [ml])
  return ("SAFE", [])  # shouldn't reach here if routing is correct
```

### Algorithm 2: LLM-adjudicated (A.4 — `cross_validator` debate)

```
  Inputs: ml_result, static_findings, rag_results, contract source
  Output: verdicts, confirmations, contradictions, debate_transcript

  DEBATE_MODE=on (default) → 3 sequential LLM calls, ONE outer timeout:
    ┌─────────────────────────────────────────────────────────┐
    │  ① Prosecutor   reads source + evidence                 │
    │       │  argues VULNERABLE                              │
    │       ▼                                                │
    │  ② Defender      given prosecutor's case                │
    │       │  argues FALSE POSITIVE (ML over-prediction       │
    │       │  is explicitly named in prompt per Ali directive)│
    │       ▼                                                │
    │  ③ Judge         given both sides + source              │
    │       │  renders {class: verdict} JSON                  │
    │       ▼                                                │
    │  debate_transcript: {prosecutor, defender, judge}       │
    └─────────────────────────────────────────────────────────┘
    DEBATE_TIMEOUT_S=240 (entire chain, not per-call)

  DEBATE_MODE=off → single classification call
                   CROSS_VALIDATOR_TIMEOUT_S=90

  AGENTS_DISABLE_LLM=1 → silent fallback to Algorithm 1 (rule-based)
```

### Algorithm 3: Consensus vote (A.6/A.7 — `consensus_engine`)

```
  Per-class weighted vote over {ML, Slither, Aderyn}.
  ML is DELIBERATELY discounted (Ali directive 2026-06-21).

  ML_WEIGHT_SCALE = 0.5  (env-tunable)
  ACCURACY_WEIGHTS = {class: {ml, slither, aderyn}}  (per-class reliability, in consensus.py)

  consensus_vote(prob, slither, aderyn, cls):
      score = (ACCURACY_WEIGHTS[cls]["ml"]    × ML_WEIGHT_SCALE × prob)
            + (ACCURACY_WEIGHTS[cls]["slither"] × slither_hit)
            + (ACCURACY_WEIGHTS[cls]["aderyn"]  × aderyn_hit)
      if score ≥ CONFIRMED_THR AND (slither or aderyn):   return CONFIRMED
      elif score ≥ LIKELY_THR:                              return LIKELY
      elif ml_only:                                         return DISPUTED
      else:                                                 return SAFE

  Key invariant: ML alone (slither=False, aderyn=False) NEVER returns CONFIRMED.
  This is enforced by the `AND (slither or aderyn)` clause.
```

---

## 7. Why ML is a HINT, Not Ground Truth (Ali Directive 2026-06-21)

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Run 12's ML model is not reliable enough to be trusted alone.      │
  │  Known issues (per MEMORY.md):                                      │
  │    • ExternalBug over-prediction (s_Form001 p=0.96 false positive)  │
  │    • Reentrancy / Timestamp are mostly TP                          │
  │    • Class-definition mismatches vs Slither in some cases          │
  │                                                                     │
  │  Mitigation layers (defence in depth):                              │
  │    1. ML_WEIGHT_SCALE = 0.5        (consensus can't CONFIRM w/o    │
  │                                    corroboration)                   │
  │    2. Two-signal fast-path gate    (ML OR Slither must flag)       │
  │    3. Per-class DEEP_THRESHOLDS    (deliberately LOWER than 0.50  │
  │                                    to catch borderline cases)       │
  │    4. Prosecutor/Defender/Judge    (LLM debate reads source,        │
  │                                    weighs ML as one signal)         │
  │    5. cross_validator prompt       (explicitly tells LLM that       │
  (nodes/cross_validator.py)      ML is over-prediction-prone)  │
  │    6. reflection (A.3)             (flags known ML failure modes)  │
  │    7. explainer (A.8)               (LIME attribution so consumer   │
  │                                    can see ml_pct vs others)        │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Module 1 Producer (Cross-Module Reference)

The agents module does NOT hold the model. Module 1 does. To understand the producer, look at `~/projects/sentinel/ml/`:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Module 1 (~/projects/sentinel/ml/)                                    │
  │  ──────────────────────────────────────                                │
  │  Entry point:     ml/api.py              FastAPI app                   │
  │  Model loading:   ml/src/models/         GCB-P1 (GNN+CodeBERT)        │
  │  Active ckpt:     ml/checkpoints/        Run 12 FINAL.pt (~280 MB)    │
  │  Schema:          v9 (NODE_FEATURE_DIM=12, NUM_NODE_TYPES=14,          │
  │                       NUM_EDGE_TYPES=12, NUM_CLASSES=10)               │
  │                                                                        │
  │  Endpoints consumed by agents:                                         │
  │    POST /predict      body: {source_code}                              │
  │                        → ml_result (3-tier schema, see § 4)            │
  │    POST /hotspots     body: {source_code}                              │
  │                        → function-level GNN embedding-norm scores      │
  │    GET  /health       → {status, version, checkpoint}                  │
  │                                                                        │
  │  Calibration:    ml/calibration/temperatures_run12.json                │
  │  Drift monitor:  mlops_config.json + drift_baseline_run12.json         │
  │  Docker:         ml/deploy/Dockerfile.inference                        │
  │                                                                        │
  │  Default URL:    http://localhost:8001 (env: MODULE1_INFERENCE_URL)    │
  │  M6+ (Docker):   http://ml-server:8001 (compose service name)         │
  └────────────────────────────────────────────────────────────────────────┘
```

### Why the two URLs?

| Env var | Default | Used by |
|---------|---------|---------|
| `MODULE1_INFERENCE_URL` | `http://localhost:8001` | `inference_server.py` (MCP :8010) |
| `SENTINEL_ML_API_URL` | `http://localhost:8000` | `graph_inspector_server.py` (MCP :8013) |

These are **two different defaults pointing to the same service** in local dev (both 8001/8000 are Module 1; the discrepancy is legacy from earlier versions of the API). In Docker Compose both should resolve to the same `ml-server` hostname.

---

## 9. Mock Mode (Development Without Module 1)

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  MODULE1_MOCK=true                                                   │
  │  (agents/.env or environment)                                        │
  └──────────────────────────────┬───────────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  inference_server._MOCK_MODE = True                                 │
  │  → _call_inference_api() returns _mock_prediction() INSTEAD of HTTP │
  │                                                                      │
  │  Mock returns realistic 3-tier predictions:                         │
  │    label:           "safe" | "suspicious" | "confirmed_vulnerable"  │
  │    probabilities:   full 10-class vector (deterministic from code)  │
  │    confirmed:       list of classes with prob ≥ 0.55                │
  │    suspicious:      list of classes with 0.25 ≤ prob < 0.55         │
  │    tier_thresholds: {confirmed: 0.55, suspicious: 0.25, ...}        │
  │                                                                      │
  │  Heuristic:                                                           │
  │    if "call.value" or "transfer(" in code:                           │
  │        Reentrancy=0.72, IntegerUO=0.54, Timestamp=0.31              │
  │    else:                                                              │
  │        Reentrancy=0.08, IntegerUO=0.12, ...                          │
  │                                                                      │
  │  Schema exactly matches Module 1's real output → swap mock→real      │
  │  requires ZERO changes to tool handlers or graph nodes.              │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 10. GNN Hotspots — The `/hotspots` Endpoint (graph_inspector_server)

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  ⑥ graph_explain (deep path) calls get_graph_hotspots(                   │
  │       contract_code, flagged_classes                                     │
  │  )                                                                       │
  │       │                                                                   │
  │       ▼                                                                   │
  │  _analyze_hotspots_gnn(contract_code, flagged_classes)                   │
  │       │                                                                   │
  │       │  POST {SENTINEL_ML_API_URL}/hotspots                              │
  │       │       body: {source_code: contract_code}                          │
  │       │                                                                   │
  │       ├─── 200 OK ──► transform to hotspot list                           │
  │       │                                                                   │
  │       ├─── non-200 ──► log warning, return None                           │
  │       │                                                                   │
  │       └─── connection error ──► log warning, return None                 │
  │                                                                           │
  │  If None returned → FALLBACK to _analyze_hotspots_slither()              │
  │                                                                           │
  │  FALLBACK CHAIN:                                                          │
  │   1. ML API /hotspots   (real GNN, embedding L2 norm per function)       │
  │   2. Slither analysis   (Phase 1 logic — structural proxy scoring)        │
  │   3. Mock data          (if Slither unavailable, MOCK_MODE=true)          │
  │                                                                           │
  │  Result includes analysis_mode field so caller knows which path was used: │
  │     "gnn_attention"  /  "slither"  /  "mock"                              │
  └──────────────────────────────────────────────────────────────────────────┘

  Why L2 norm?
  ─────────────
  The /hotspots endpoint returns per-function GNN embedding norms.
  Higher L2 norm = the GNN concentrated more structural message-passing
  signal on that node = the model "looked at" that function more.
  This is REAL model attention, not a Slither-based proxy.
```

---

## 11. Fallback Chain Visualized

```
  graph_explain (deep path)
       │
       ▼
  get_graph_hotspots MCP tool call (:8013)
       │
       ▼
  _analyze_hotspots_gnn  ───────────┐
       │                            │ 200 OK + valid data
       │                            ▼
       │                     returns hotspots (real GNN)
       │                     analysis_mode="gnn_attention"
       │
       ├─► non-200 / connection error
       │   │
       │   ▼
       │   _analyze_hotspots_slither  (Slither structural scoring)
       │       │
       │       ├─► Slither installed
       │       │       → returns hotspots (proxy)
       │       │         analysis_mode="slither"
       │       │
       │       └─► Slither unavailable OR MOCK_MODE
       │               → returns mock data
       │                 analysis_mode="mock"
       │
       └─► (no GraphInspector call → return empty)


  IMPORTANT: graph_inspector_server.py does NOT call _mock_prediction()
  — that is inference_server.py's fallback. The graph_inspector
  fallback is its own Slither logic + its own mock data structure.
```

---

## 12. Shared HTTP Client (inference_server.py)

```
  ┌────────────────────────────────────────────────────────────────────┐
  │  Sentinel-inference MCP server :8010                              │
  │  ──────────────────────────────────                               │
  │  One httpx.AsyncClient per server lifetime (NOT per call).        │
  │  Managed via Starlette lifespan context.                          │
  │                                                                    │
  │  Why: new client per call = new TCP+TLS handshake (~20-50ms).     │
  │       Shared client reuses connection.                            │
  │                                                                    │
  │  ┌──────────────┐    lifespan     ┌──────────────────────┐        │
  │  │ Starlette    │ ──────────────► │  _on_startup         │        │
  │  │ app          │                 │  _http_client =      │        │
  │  │              │                 │    httpx.AsyncClient │        │
  │  │              │                 │    (timeout=30s)     │        │
  │  │              │ ◄────────────── │  _on_shutdown        │        │
  │  │              │    lifespan     │  await _http_client  │        │
  │  │              │     exit        │        .aclose()     │        │
  │  └──────────────┘                 └──────────────────────┘        │
  │                                                                    │
  │  All predict/batch_predict tool calls share this client.          │
  │  (graph_inspector_server.py is a different file with a            │
  │   different pattern: fresh client per call.)                      │
  └────────────────────────────────────────────────────────────────────┘
```

---

## 13. Configuration — All env vars that affect ML integration

```
  ┌─── Module 1 URLs ────────────────────┬────────────────────────────────┐
  │ MODULE1_INFERENCE_URL                │ http://localhost:8001          │
  │ SENTINEL_ML_API_URL                  │ http://localhost:8000  (used   │
  │                                      │   by graph_inspector_server)   │
  └──────────────────────────────────────┴────────────────────────────────┘
  ┌─── MCP server ports ────────────────┬────────────────────────────────┐
  │ MCP_INFERENCE_URL                    │ http://localhost:8010/sse      │
  │ MCP_GRAPH_INSPECTOR_URL              │ http://localhost:8013/sse      │
  │ MCP_INFERENCE_PORT                   │ 8010                           │
  │ MCP_GRAPH_INSPECTOR_PORT             │ 8013                           │
  └──────────────────────────────────────┴────────────────────────────────┘
  ┌─── Timeouts ─────────────────────────┬────────────────────────────────┐
  │ MODULE1_TIMEOUT                      │ 30.0   (HTTP /predict)         │
  │ GRAPH_INSPECTOR_HOTSPOTS_TIMEOUT     │ 60     (HTTP /hotspots)        │
  └──────────────────────────────────────┴────────────────────────────────┘
  ┌─── Mock mode ────────────────────────┬────────────────────────────────┐
  │ MODULE1_MOCK                         │ false  (true = use _mock_      │
  │                                      │         prediction, no HTTP)   │
  │ GRAPH_INSPECTOR_MOCK                 │ false  (true = use Slither     │
  │                                      │         structural scoring or  │
  │                                      │         mock data only)        │
  └──────────────────────────────────────┴────────────────────────────────┘
  ┌─── ML discounting (consensus) ───────┬────────────────────────────────┐
  │ ML_WEIGHT_SCALE                      │ 0.5    (env-tunable; ML alone  │
  │                                      │         → never CONFIRMED)     │
  │ ACCURACY_WEIGHTS                     │ per-class, in consensus.py     │
  └──────────────────────────────────────┴────────────────────────────────┘
  ┌─── LLM enable ───────────────────────┬────────────────────────────────┐
  │ AGENTS_DISABLE_LLM                   │ 0  (1 = rule-based fallback,   │
  │                                      │   skips cross_validator debate,│
  │                                      │   synthesizer narrative,       │
  │                                      │   reflection)                  │
  │ DEBATE_MODE                          │ on  (off = single-pass)        │
  │ CROSS_VALIDATOR_LLM_MODEL            │ fast                           │
  │ CROSS_VALIDATOR_TIMEOUT_S            │ 90                             │
  │ DEBATE_TIMEOUT_S                     │ 240                            │
  │ SYNTHESIZER_TIMEOUT_S                │ 120                            │
  └──────────────────────────────────────┴────────────────────────────────┘
```

---

## 14. File Map — Every File That Participates in ML Integration

```
  agents/
  │
  ├── src/orchestration/                  ← THE ML MODULE
  │   ├── state.py            AuditState — ml_result, ml_hotspots,
  │   │                        graph_explanations, confidence_by_class,
  │   │                        consensus_verdict, metric_attribution
  │   │                        (line 55-74: ml_result three-tier schema)
  │   │
  │   ├── routing.py          Per-class thresholds + tool routing
  │   │                        DEEP_THRESHOLDS, ROUTING_RULES,
  │   │                        CLASS_TO_DETECTORS, DETECTOR_TO_CLASSES
  │   │                        compute_active_tools, compute_verdict,
  │   │                        prob_to_severity, compute_overall_verdict
  │   │                        (line 23-34: DEEP_THRESHOLDS)
  │   │
  │   ├── nodes/              14 async node implementations (P2 split from nodes.py)
  │   │   ├── ml_assessment.py       calls MCP :8010 /predict; sets ml_result + model_hash
  │   │   ├── quick_screen.py        Tier 0: Slither + Aderyn High/Critical scan
  │   │   ├── evidence_router.py     logs routing_decisions (no LLM, no contract_code)
  │   │   ├── rag_research.py        calls MCP :8011 search; uses ml_result
  │   │   ├── static_analysis.py     Slither + Aderyn scoped to ML-flagged classes
  │   │   ├── graph_explain.py       calls MCP :8013 get_graph_hotspots
  │   │   ├── formal_verification.py Halmos symbolic execution (P8a)
  │   │   ├── audit_check.py         calls MCP :8012 get_audit_history
  │   │   ├── consensus_engine.py    A.6/A.7 weighted vote; ML discounted
  │   │   ├── cross_validator.py     A.4 P/D/J debate; P4 injection guard
  │   │   ├── synthesizer.py         fuse() verdicts; P4 injection guard; final_report
  │   │   ├── reflection.py          A.3 self-critique
  │   │   ├── explainer.py           A.8 LIME attribution
  │   │   ├── visualizer.py          A.9 interactive HTML hotspot report
  │   │   └── _helpers.py            _call_mcp_tool, _llm_enabled, AderynRunError
  │   ├── verdict/                   verdict production package (P2)
  │   │   ├── fuse.py                sole verdict producer → verdict_provable + verdict_full
  │   │   ├── evidence.py            Evidence dataclass + constructors
  │   │   ├── reliability.py         L3→L1 fallback reliability weights
  │   │   ├── emit.py                emit_evidence(), emit_halmos_evidence()
  │   │   └── verdict.py             verdict constants
  │   │
  │   ├── consensus.py        A.6 — consensus_vote (weighted ML/Slither/Aderyn)
  │   ├── confidence.py       A.7 — track_confidence (Bayesian staged)
  │   ├── attribution.py      A.8 — attribute_verdict (LIME-style {ml_pct, slither_pct, rag_pct})
  │   ├── graph.py            build_graph(), audit_graph (lazy, PEP 562)
  │   └── visualizer.py       A.9 — generate_hotspot_html (interprets ml_hotspots)
  │
  ├── src/mcp/servers/                     ← MCP WRAPPERS OF MODULE 1
  │   ├── inference_server.py       :8010  predict, batch_predict
  │   │                                  _call_inference_api() → HTTP /predict
  │   │                                  _mock_prediction() (when MODULE1_MOCK=true)
  │   │                                  shared httpx.AsyncClient (lifespan)
  │   │                                  (line 174-237: HTTP bridge to Module 1)
  │   │                                  (line 240-313: _mock_prediction — full 3-tier)
  │   │
  │   └── graph_inspector_server.py :8013  get_graph_hotspots
  │                                      _analyze_hotspots_gnn() → HTTP /hotspots
  │                                      _analyze_hotspots_slither() → fallback
  │                                      Fallback chain: ML API → Slither → mock
  │                                      (line 11-32: PHASE 2 transition note)
  │                                      (line 153-232: _analyze_hotspots_gnn)
  │                                      (line 253-...: _analyze_hotspots_slither)
  │
  ├── scripts/
  │   └── smoke_inference_mcp.py       SSE smoke test for :8010
  │
  └── tests/
      ├── test_inference_server.py     Unit tests for :8010
      ├── test_graph_routing.py        Includes TestMlAssessmentNode
      └── test_smoke_e2e.py            End-to-end with fixtures
```

---

## 15. Failure Modes — What Happens When ML Integration Breaks

```
  ┌─────────────────────────────────────┬──────────────────────────────────────┐
  │ Failure                            │ Behaviour                             │
  ├─────────────────────────────────────┼──────────────────────────────────────┤
  │ Module 1 unreachable (HTTP error)  │ _call_inference_api() falls back to   │
  │                                     │ _mock_prediction() (if MODULE1_MOCK=  │
  │                                     │ true). graph continues with mock     │
  │                                     │ data.                                │
  │                                     │                                      │
  │ Module 1 timeout (>30s)            │ _call_inference_api() falls back to  │
  │                                     │ _mock_prediction().                  │
  │                                     │                                      │
  │ Module 1 returns HTTP 4xx/5xx      │ _call_inference_api() RAISES (not    │
  │                                     │ swallowed). ml_assessment captures   │
  │                                     │ as state["error"], sets              │
  │                                     │ state["ml_result"] = {}.             │
  │                                     │ graph still continues to             │
  │                                     │ synthesizer (rule-based).            │
  │                                     │                                      │
  │ MCP :8010 server down              │ ml_assessment gets connection error, │
  │                                     │ captures as state["error"],          │
  │                                     │ state["ml_result"] = {}.             │
  │                                     │ evidence_router sees empty           │
  │                                     │ ml_result → fast path with NO ML.    │
  │                                     │                                      │
  │ ML API /hotspots unreachable       │ _analyze_hotspots_gnn() returns      │
  │                                     │ None → fallback to                  │
  │                                     │ _analyze_hotspots_slither() (still   │
  │                                     │ gets per-function signals).         │
  │                                     │ analysis_mode="slither" in result.   │
  │                                     │                                      │
  │ ml_result empty/malformed          │ _iter_class_probs() yields nothing.  │
  │                                     │ build_routing_decisions() emits      │
  │                                     │ "all classes below threshold → fast │
  │                                     │ path". quick_screen signal is the    │
  │                                     │ sole driver.                         │
  │                                     │                                      │
  │ LLM unavailable (AGENTS_DISABLE_   │ cross_validator, synthesizer         │
  │  LLM=1)                             │ narrative, reflection all use        │
  │                                     │ rule-based fallbacks. Graph still    │
  │                                     │ completes.                           │
  │                                     │                                      │
  │ Contract > 512 CodeBERT tokens     │ ml_result.truncated = True.          │
  │                                     │ reflection (A.3) flags "truncated   │
  │                                     │ contract" as a known failure mode.   │
  │                                     │ risk score is conservative.          │
  │                                     │                                      │
  │ ExternalBug high-confidence        │ Known ML over-prediction (s_Form001  │
  │                                     │ false positive, p=0.96 on a 26-line │
  │                                     │ KV store). Defender prompt in debate │
  │                                     │ explicitly names this. reflection    │
  │                                     │ flags it. consensus_vote() can't     │
  │                                     │ CONFIRM without Slither/Aderyn hit.  │
  └─────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 16. Quick Reference — Source Code Locations

| Concept | File:Line |
|---------|-----------|
| `ml_assessment` node | `agents/src/orchestration/nodes/ml_assessment.py` |
| `ml_result` schema (3-tier) | `agents/src/orchestration/state.py:55-74` |
| `DEEP_THRESHOLDS` | `agents/src/orchestration/routing.py:23-34` |
| `ROUTING_RULES` | `agents/src/orchestration/routing.py:43-54` |
| `CLASS_TO_DETECTORS` | `agents/src/orchestration/routing.py:62-112` |
| `DETECTOR_TO_CLASSES` (inverted) | `agents/src/orchestration/routing.py:116-119` |
| `compute_active_tools()` | `agents/src/orchestration/routing.py:141-156` |
| `build_routing_decisions()` | `agents/src/orchestration/routing.py:159-184` |
| `compute_verdict()` (rule-based) | `agents/src/orchestration/routing.py:187-245` |
| `prob_to_severity()` | `agents/src/orchestration/routing.py:248-253` |
| `compute_overall_verdict()` | `agents/src/orchestration/routing.py:259-263` |
| Two-signal fast-path gate | `agents/src/orchestration/graph.py` + `orchestration/README.md` |
| `rag_research` node | `agents/src/orchestration/nodes/rag_research.py` |
| `static_analysis` node | `agents/src/orchestration/nodes/static_analysis.py` |
| `graph_explain` node | `agents/src/orchestration/nodes/graph_explain.py` |
| `cross_validator` node | `agents/src/orchestration/nodes/cross_validator.py` |
| Ali directive (ML = HINT) | `agents/src/orchestration/nodes/cross_validator.py` |
| Cross_validator prompt (names ML over-prediction) | `agents/src/orchestration/nodes/cross_validator.py` |
| `synthesizer` node | `agents/src/orchestration/nodes/synthesizer.py` |
| `reflection` node | `agents/src/orchestration/nodes/reflection.py` |
| `explainer` node (LIME attribution) | `agents/src/orchestration/nodes/explainer.py` |
| `fuse()` (P2 sole verdict producer) | `agents/src/orchestration/verdict/fuse.py` |
| `consensus_vote()` (A.6) | `agents/src/orchestration/consensus.py` |
| `track_confidence()` (A.7) | `agents/src/orchestration/confidence.py` |
| `attribute_verdict()` (A.8) | `agents/src/orchestration/attribution.py` |
| Inference MCP HTTP bridge | `agents/src/mcp/servers/inference_server.py:174-237` |
| `_mock_prediction()` (3-tier) | `agents/src/mcp/servers/inference_server.py:240-313` |
| Shared httpx client (lifespan) | `agents/src/mcp/servers/inference_server.py:147-167` |
| Hotspots GNN call | `agents/src/mcp/servers/graph_inspector_server.py:153-232` |
| Hotspots Slither fallback | `agents/src/mcp/servers/graph_inspector_server.py:253-...` |
| Module 1 producer | `~/projects/sentinel/ml/api.py` |
| Run 12 checkpoint | `~/projects/sentinel/ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt` |
| Calibration (3-tier thresholds) | `~/projects/sentinel/ml/calibration/temperatures_run12.json` |

---

## See Also

- `~/projects/sentinel/agents/DIAGRAM.md` — top-level module diagram
- `~/projects/sentinel/agents/src/orchestration/README.md` — written companion (text-heavy)
- `~/projects/sentinel/agents/AGENTS_STATE_AND_REDESIGN_2026-06-14.md` — pre-Phase A state doc
- `docs/plan/agents/2026-06-17-extended-capability/` — Phase A/B/C/D plans (A.6/A.7/A.4/A.3/A.8/A.9)
- `docs/proposal/agent_proposal/AGENTS_EXTENDED_CAPABILITY_FINAL_PROPOSAL.md`
- `~/projects/sentinel/ml/` — Module 1 producer (cross-module reference)
