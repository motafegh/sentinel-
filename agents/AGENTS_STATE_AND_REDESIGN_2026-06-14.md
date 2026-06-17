# Agents Module — State Audit & Redesign Plan (2026-06-14)

> **Purpose:** Working reference for agents module alignment with new ml/data_module setup.
> Written from source-code read of `agents/src/` and `docs/proposal/SENTINEL_AGENTS_V3.md`.
> Do not re-read source files when starting this work — use this document.
>
> **Context:** Run 12 ep46, f1_tuned=0.6941. Run 13 will drop GasException (NUM_CLASSES=10→9),
> change class distributions significantly. Agents were Phase 1 complete as of 2026-05-30.

---

## 1. Does Agents Have Duplicate Submodules with data_module?

**Short answer: No — not true code duplication. But there IS a naming confusion.**

### What agents/src/ingestion/ does
Maintains the **RAG knowledge base** for the LLM reasoning layer:
- Fetches DeFiHackLabs exploit reports (Markdown README files) from a local git clone
- Chunks + embeds them into a FAISS + BM25 hybrid index
- Incrementally updates the index on a Dagster/cron schedule
- Polls AuditRegistry on Sepolia for confirmed audit findings → feeds back into RAG

### What data_module/ does
Builds the **training dataset** for the ML model:
- Ingests Solidity `.sol` files from DIVE, SolidiFI, SmartBugs, BCCC
- Runs Slither/graph_extractor on each to produce graph + token representations
- Labels, deduplicates, splits, and exports as PyG-compatible shards

**These are completely different data types, different goals, different downstream consumers.** The `Deduplicator` in agents uses SHA256 of document content; data_module uses graph content hashes. Both are called "deduplicator" but share no code and are conceptually distinct.

### The naming confusion
`agents/src/ingestion/` is an unfortunate name because `data_module/sentinel_data/ingestion/` also exists. A future reader will wonder if they overlap. Recommendation: rename agents' ingestion to `agents/src/rag_pipeline/` to make the purpose clear. This is low-priority housekeeping.

---

## 2. Current Implementation Status

### Phase 0 + Phase 1 — COMPLETE (as of 2026-05-30, 219 tests passing)

#### Graph topology (implemented, all nodes working)
```
START → ml_assessment → quick_screen → evidence_router
                        (Tier 0: always)       │
                                 ┌─────────────┴──────────────────┐
                                 │ deep path                       │ fast path
                    ┌────────────┼───────────────┐                 │
               rag_research  static_analysis  graph_explain        │
                    └────────────┼───────────────┘                 │
                            audit_check                            │
                                 │                                 │
                          cross_validator                          │
                                 │                                 │
                            synthesizer ←────────────────────────--┘ → END
```

#### MCP servers (4, all implemented)

| Server | Port | What it does | Status |
|---|---|---|---|
| `inference_server.py` | 8010 | Proxies `/predict` and `/hotspots` to ml inference API at `:8001` | ✅ Done |
| `rag_server.py` | 8011 | FAISS+BM25 hybrid search over DeFiHackLabs corpus | ✅ Done |
| `audit_server.py` | 8012 | On-chain AuditRegistry lookup (Sepolia) | ✅ Done |
| `graph_inspector_server.py` | 8013 | Real GNN embedding-norm hotspots via `/hotspots` endpoint | ✅ Done (Phase 1 A2) |

#### Key components

| File | Lines (approx) | Status |
|---|---|---|
| `orchestration/nodes.py` | ~500 | ✅ All nodes implemented |
| `orchestration/routing.py` | ~200 | ✅ DEEP_THRESHOLDS, CLASS_TO_DETECTORS, verdict logic |
| `orchestration/state.py` | ~120 | ✅ Full AuditState TypedDict |
| `orchestration/graph.py` | ~100 | ✅ LangGraph topology |
| `rag/retriever.py` | ~200 | ✅ Hybrid FAISS+BM25 |
| `rag/embedder.py` | ~100 | ✅ Nomic embed v1.5 |
| `rag/chunker.py` | ~100 | ✅ Sliding window chunker |
| `ingestion/pipeline.py` | 313 | ✅ Incremental FAISS+BM25 pipeline |
| `ingestion/feedback_loop.py` | 470 | ✅ AuditRegistry polling + RAG ingestion |
| `ingestion/scheduler_dagster.py` | ~80 | ✅ Daily 02:00 UTC schedule |
| `llm/client.py` | ~100 | ✅ LM Studio client (local LLM, Windows host port 1234) |

### NOT YET BUILT

| Item | From which plan | Priority |
|---|---|---|
| A6: HIGH_VALUE_RAG_CLASSES routing distinction | SENTINEL_AGENTS_V3.md | Low (Phase 2 will rebuild routing anyway) |
| Phase 2: Investigator loop (multi-hop LLM reasoning) | V3 §4 | Post-Run-13 |
| Phase 3: `econ_sim` node (price manipulation cost estimator, port 8015) | V3 §5 | Far future |
| Phase 3: Mythril scoped to hotspot functions | V3 §3 | Far future |
| Phase 4: Echidna / Halmos integration | V3 §6 | Far future |
| `batch_predict` multi-contract API | V3 §1.2 | Low |
| LLM disagreement feedback loop logging | V3 §1.2 | Post-Run-13 |

---

## 3. Staleness Issues — What Needs Updating for Run 13

### 3.1 GasException — DEFERRED (still live in Run 12, remove after Run 13 lands)

`agents/src/orchestration/routing.py` hardcodes GasException in:
- `DEEP_THRESHOLDS["GasException"] = 0.40`
- `ROUTING_RULES["GasException"] = ["static_analysis"]`
- `CLASS_TO_DETECTORS["GasException"] = ["costly-loop", "calls-loop", "incorrect-exp"]`

**Status:** NOT FIXED YET — Left alone (2026-06-17) because:
- The model STILL outputs GasException (10 classes, including GasException)
- Run 13's plan to drop GasException (NUM_CLASSES=9) has NOT shipped yet
- Removing it now would break routing for a class the model still predicts

**Fix timing:** Only remove these 3 entries from `routing.py` **after** Run 13 training completes and the model is deployed with NUM_CLASSES=9. Then update `DETECTOR_TO_CLASSES` (auto-built at import time from `CLASS_TO_DETECTORS` — will be correct after deletion).

### 3.2 Class name mismatch: "TOD" vs "TransactionOrderDependence" — ✅ FIXED (2026-06-17)

**Status:** CLOSED — Fixed in commit 8c50fb8d7 (2026-06-17 02:45 UTC).

What the bug was: `routing.py` used `"TOD"` as the class key. The model outputs `"TransactionOrderDependence"`. This meant routing never triggered for ToD — probabilities from the model came in as `"TransactionOrderDependence"` but were looked up in `DEEP_THRESHOLDS` as `"TOD"`, causing a silent miss (empty tool list returned, contract never enters deep analysis).

Why tests didn't catch it: All test fixtures and the mock predictor independently invented `"TOD"` too — so they were internally consistent with the buggy code, but never with the real ML API.

**Fix applied (renamed "TOD" → "TransactionOrderDependence" everywhere):**
- `agents/src/orchestration/routing.py` (3 places: DEEP_THRESHOLDS, ROUTING_RULES, CLASS_TO_DETECTORS)
- `agents/src/mcp/servers/graph_inspector_server.py` (2 places: _CLASS_STRUCTURAL_SIGNALS, _DETECTOR_CLASS_MAP)
- `agents/src/mcp/servers/inference_server.py` mock predictor (2 places)
- Test fixtures + docs (4 places across test_routing_phase0.py, test_smoke_e2e.py, README.md)

All 53+ tests pass post-fix. TransactionOrderDependence contracts now correctly route to static_analysis + rag_research.

### 3.3 DEEP_THRESHOLDS values are not calibrated to Run 12

The current thresholds (0.30–0.45) were set before any honest training data. Run 12 at ep40 shows real per-class AUC-PR:
- ExternalBug: 0.921, Reentrancy: 0.840, Timestamp: 0.928, IntegerUO: 0.778
- DoS: 0.334 (low — keep threshold at 0.30), ToD: 0.464 (medium)
- CallToUnknown: 0.981 (high — could raise threshold to 0.50)

Thresholds should be re-tuned based on Run 12/13 per-class F1 and AUC-PR. This is not blocking but will reduce false positives in deep-path routing.

### 3.4 inference_server.py calls MODULE1 at :8001 — wrong port

```python
# inference_server.py
_MODULE1_URL: str = os.getenv("MODULE1_INFERENCE_URL", "http://localhost:8001")
```

The ml inference server (`api.py`) actually runs on port 8001 by default (when started with `uvicorn ml.src.inference.api:app --port 8001`). This is consistent. But:
- No startup script ensures ml inference server is running before agents start
- No health check from inference_server to MODULE1 at startup
- `_MOCK_MODE` env var exists for development — keep but document

### 3.5 audit_server.py hardcodes AuditRegistry address

```python
# Somewhere in audit_server.py
AUDIT_REGISTRY = os.getenv("AUDIT_REGISTRY", "0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf")
```

This address is the old Sepolia deployment. After contracts are redeployed (to support multi-class scores), this address must change. It's env-overridable but the default will be wrong.

### 3.6 RAG knowledge base built from unverified exploit data

The FAISS index was built from DeFiHackLabs README files (markdown). DeFiHackLabs is high-quality but covers historical DeFi exploits — not always the same vulnerability taxonomy as SENTINEL's 9 classes. The RAG evidence quality could be improved by:
1. Adding Web3Bugs dataset (already planned in data-source-addition-plan-2026-06-13.md)
2. Adding verified contract examples from data_module v3 export (contracts where we have high-confidence labels) as RAG documents

This is enhancement, not a bug — the current RAG still works.

### 3.7 feedback_loop.py score threshold is single-score based

```python
# feedback_loop.py
if score >= 5734:   # field element for 0.70 human-readable (5734 = 0.70 * 8192)
```

This checks a single field element against a threshold. After contracts are redesigned for multi-class output, the on-chain score format changes — this threshold and decoding logic will need to be updated.

Also: `feedback_loop.py` reads `data/reports/{contract_address}.json` (written by synthesizer) to recover `vulnerability_class`. This bridge must stay intact when contracts change their event schema.

---

## 4. Data Flow — How Agents Currently Connects to ml Module

```
Caller (contract_code + contract_address)
    │
    ▼
ml_assessment node
    → calls inference_server MCP (port 8010)
    → inference_server calls ml FastAPI (port 8001) POST /predict
    → ml/src/inference/api.py
    → ml/src/inference/predictor.py  (loads Run 12 checkpoint)
    → ml/src/inference/preprocess.py (calls graph_extractor.py + tokenizer)
    → Returns: { label, probabilities[9/10 classes], confirmed, suspicious, thresholds, ... }
    │
    ▼
quick_screen node (always runs)
    → runs Slither + Aderyn in-process on contract_code via temp file
    → Returns: { slither: [...], aderyn: [...] }
    │
    ▼
evidence_router node
    → routing.py: compute_active_tools(ml_result)
    → Two-signal gate: fast path requires ML clean AND quick_screen clean
    → Fan-out to: rag_research, static_analysis, graph_explain (deep path)
    │
    ▼
[deep path nodes run in parallel]
    rag_research   → rag_server MCP (port 8011) → FAISS+BM25 over DeFiHackLabs
    static_analysis → Slither scoped to flagged classes (CLASS_TO_DETECTORS)
    graph_explain  → graph_inspector_server MCP (port 8013) → /hotspots
    audit_check    → audit_server MCP (port 8012) → AuditRegistry Sepolia
    │
    ▼
cross_validator → LLM (LM Studio port 1234) adjudicates per-class verdicts
    │
    ▼
synthesizer → writes final_report JSON
             → writes data/reports/{contract_address}.json (BRIDGE for feedback_loop)
```

---

## 5. Changes Needed Before Agents Can Support Run 13

**Order matters:**

1. ✅ **Fix class name "TOD" → "TransactionOrderDependence"** in `routing.py` (10 min, bug fix) — **DONE 2026-06-17, commit 8c50fb8d7**
2. **Remove GasException from routing.py** (10 min, after Run 13 trains) — DEFERRED until Run 13 lands
3. **Update AUDIT_REGISTRY address** after contract redeployment (env var — 5 min)
4. **Re-tune DEEP_THRESHOLDS** using Run 12/13 per-class AUC-PR values (30 min)
5. **Rebuild RAG index** once Web3Bugs data is ingested (1 hr — runs pipeline.py)
6. **Update feedback_loop.py score threshold** after contracts redesign (30 min)
7. **Rename `agents/src/ingestion/` → `agents/src/rag_pipeline/`** (housekeeping, 1 hr)

---

## 6. Tests

```
agents/tests/
  test_audit_server.py         ← on-chain AuditRegistry lookup
  test_chunker.py              ← RAG chunk splitting
  test_deduplicator.py         ← document dedup
  test_github_fetcher.py       ← DeFiHackLabs fetcher
  test_graph_routing.py        ← LangGraph routing
  test_inference_server.py     ← MCP inference server (mock mode)
  test_retriever_filters.py    ← FAISS+BM25 retrieval
  test_routing_phase0.py       ← DEEP_THRESHOLDS + compute_active_tools
  test_smoke_e2e.py            ← end-to-end smoke (Phase 1 A5, 7 tests)
```

219+ tests passing as of 2026-05-30. After TOD → TransactionOrderDependence fix (2026-06-17, commit 8c50fb8d7), all 53+ tests in test_routing_phase0.py + test_smoke_e2e.py verified PASS. After Run 13 lands and GasException is removed, re-run full `poetry run pytest agents/tests/` to verify again.

---

## 7. Key Design Decisions Still Open

### Decision A: What does agents receive when contracts go multi-class?
After `AuditRegistry` is redesigned for multi-class scores, `audit_check` node's response format changes. The `AuditState.audit_history` field currently expects `AuditResult` structs with a single `scoreFieldElement`. This needs to map to multi-class scores once contracts are updated.

### Decision B: RAG quality improvement
Should the RAG corpus include verified contract examples from data_module v3 (high-confidence labels)? This would let the RAG retrieve similar contracts from the training corpus. Implementation: add a `SentinelCorpusFetcher` to `agents/src/rag/fetchers/` that reads from the v3 export's labels.parquet.

### Decision C: LLM for synthesis
Currently uses LM Studio (local LLM, Windows host at port 1234). For production, this should be replaced with a proper LLM API call (Claude API, etc.). The `llm/client.py` should be environment-configurable.

---

## 8. File Locations

| What | Path |
|---|---|
| Graph topology | `agents/src/orchestration/graph.py` |
| Node functions | `agents/src/orchestration/nodes.py` |
| Routing + thresholds | `agents/src/orchestration/routing.py` ← **TOD fix ✅ done (2026-06-17); GasException removal pending Run 13** |
| State TypedDict | `agents/src/orchestration/state.py` |
| MCP servers | `agents/src/mcp/servers/` |
| RAG retriever | `agents/src/rag/retriever.py` |
| RAG ingestion pipeline | `agents/src/ingestion/pipeline.py` |
| Feedback loop | `agents/src/ingestion/feedback_loop.py` |
| LLM client | `agents/src/llm/client.py` |
| DeFiHackLabs corpus | `agents/data/defihacklabs/` |
| FAISS+BM25 index | `agents/data/index/` |
| Audit reports (synthesizer output) | `agents/data/reports/` |
| Proposal (V3, current) | `docs/proposal/SENTINEL_AGENTS_V3.md` |
