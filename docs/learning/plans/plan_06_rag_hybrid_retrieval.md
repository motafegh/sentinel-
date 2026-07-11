# Plan: Doc 06 — RAG: Hybrid FAISS+BM25 Retrieval with Reciprocal Rank Fusion

**Spec:** `docs/learning/LEARNING_DOCS_SPEC.md`
**Target:** `docs/learning/06_rag_hybrid_retrieval.md`
**Session:** 3 of 5
**Prerequisite docs:** Doc 01 (Pipeline), Doc 02 (Evidence/Fuse), Doc 05 (MCP)

---

## Recall from previous docs

**From Doc 01 (Pipeline):** You learned that `rag_research` is one of the deep-path nodes that runs in parallel with `static_analysis`, `graph_explain`, and `formal_verification`. It feeds `rag_results` into state, which the synthesizer includes in the final report.

**From Doc 02 (Evidence/Fuse):** You learned that RAG emits `Evidence(kind=SEMANTIC, deterministic=True)` via `emit_rag_evidence()`. The evidence has a `source="rag"` and uses `similarity` as strength. RAG evidence goes into both `verdict_provable` and `verdict_full` (it's deterministic — the embedding model is deterministic per input).

**From Doc 05 (MCP):** You learned that the RAG server runs on port 8011 as an MCP server over SSE. `HybridRetriever` is lazily loaded in `_on_startup()`, not at import time. The `_call_mcp_tool()` helper is used to call the RAG server's `search` tool.

**Connection to this doc:** This doc explains what happens inside the RAG server — how hybrid retrieval works (FAISS + BM25 + RRF), how queries are constructed, and the P7 zero-match diagnosis that fixed 26.5% of reports getting empty RAG results.

**Key concepts carried forward:** `rag_research` node, `rag_results` in state, `Evidence(source="rag")`, MCP server on port 8011, `_call_mcp_tool()`.

---

## Step 1: Read source files

- [ ] `agents/src/rag/retriever.py` — `HybridRetriever` class, FAISS + BM25, Reciprocal Rank Fusion, `search()` method
- [ ] `agents/src/rag/embedder.py` — embedding model (nomic-embed-text-v1.5 via LM Studio), `embed()` method
- [ ] `agents/src/rag/chunker.py` — chunking strategy (split by headers/sections, overlap)
- [ ] `agents/src/rag/build_index/__init__.py` — package re-exports
- [ ] `agents/src/rag/build_index/_orchestrator.py` — index builder orchestrator
- [ ] `agents/src/rag/build_index/_pipeline.py` — pipeline stages (fetch → chunk → embed → index)
- [ ] `agents/src/rag/build_index/_io.py` — I/O helpers (save/load FAISS index, chunks)
- [ ] `agents/src/rag/build_index/_metadata.py` — metadata extraction
- [ ] `agents/src/rag/build_index/_paths.py` — path constants
- [ ] `agents/src/rag/fetchers/base_fetcher.py` — `BaseFetcher` interface
- [ ] `agents/src/orchestration/nodes/rag_research.py` — query construction (P7 fix), `_VULN_CLASS_TO_RAG_KEYWORDS` mapping, fallback query, `SENTINEL_DETERMINISTIC` skip
- [ ] `agents/src/mcp/servers/rag_server.py` — RAG MCP server (port 8011), `search` tool handler, lazy loading

## Step 2: Read scratch files

- [ ] `~/.claude/scratch/p2_5_p3_quarantine_plan_20260625.md` — corpus quality findings (22 broken contracts quarantined, 61 remain)

## Step 3: Read data

- [ ] `agents/data/index/index_metadata.json` — corpus stats, chunk count, embedding model, dimensions
- [ ] Run diagnostic:
  ```bash
  cd agents && source .venv/bin/activate
  python3 -c "
  import pickle
  from collections import Counter
  chunks = pickle.load(open('data/index/chunks.pkl', 'rb'))
  print(f'Total chunks: {len(chunks)}')
  vt = Counter(ch.metadata.get('vuln_type', 'NONE') for ch in chunks)
  for k, v in vt.most_common(5): print(f'  {k}: {v}')
  "
  ```

## Step 4: Read tests

- [ ] `agents/tests/test_rag_query.py` — 9 tests (skip when no classes, skip when unknown, reentrancy keywords, IntegerUO keywords, no code in query, fallback query, external bug call summary, all classes mapped, mappings descriptive)

## Step 5: Write sections

- [ ] **TL;DR:** Hybrid retrieval (FAISS semantic + BM25 keyword), Reciprocal Rank Fusion (RRF), nomic-embed-text-v1.5, 752 chunks from DeFiHackLabs, P7 zero-match fix (skip unknown, map class names, remove code, fallback query)
- [ ] **The Problem:** Need to retrieve relevant exploit post-mortems for a contract being analyzed. Pure semantic (FAISS) misses exact CVE numbers/addresses. Pure keyword (BM25) misses semantic similarity. Need both
- [ ] **How We Arrived at This Design:** invariant (RAG adds evidence if relevant, not if irrelevant) → constraint (embedding model is text, not code) → simplest retrieval (hybrid FAISS+BM25 with RRF) → stress-test (zero-match diagnosis) → measure (26.5% zero-match → fixed)
- [ ] **The Solution:** Hybrid retrieval diagram:
  ```
  Query → [FAISS (semantic)] → ranked list A
         → [BM25 (keyword)]  → ranked list B
         → RRF fusion: score(chunk) = Σ 1/(rank + 60)
         → top-k chunks
  ```
  Embedding pipeline: query → nomic-embed-text → 768-dim vector → FAISS L2 search. Query construction flow (P7 fix). Corpus structure (752 chunks, DeFiHackLabs, 97.9% unlabeled)
- [ ] **Key Code:**
  - `HybridRetriever` class (retriever.py) — `search(query, k)` method, FAISS index, BM25 index, RRF fusion
  - RRF formula: `score(chunk) = Σ 1/(rank_i + 60)` — combines rankings without score calibration
  - `_VULN_CLASS_TO_RAG_KEYWORDS` (rag_research.py) — 10 ML class names → RAG-friendly keywords (e.g., "IntegerUO" → "integer overflow underflow arithmetic")
  - `sanitize_for_prompt` NOT used here — RAG query is not an LLM prompt, it's an embedding query
  - Fallback query (rag_research.py) — if first query returns 0, try `f"{topic} vulnerability exploit"`
- [ ] **Design Decision:** FAISS+BM25 vs dense-only vs sparse-only vs learned fusion (tradeoff table: semantic coverage, keyword coverage, training data needed, complexity)
- [ ] **Technology Choice:** nomic-embed-text-v1.5 (5-question framework: category, alternatives, why this model, when code-specific embedder is better, migration trigger)
- [ ] **Anti-Patterns:**
  - ❌ Dense-only retrieval — "semantic is enough." Breaks: misses exact CVE numbers, contract addresses, function names that BM25 catches. Right: hybrid
  - ❌ Solidity code in the query — "more context = better retrieval." Breaks: text embedder (nomic-embed-text) can't handle code syntax, produces garbage embeddings. Right: use keywords only, not raw code
- [ ] **Mistakes & Fixes:**
  - 26.5% of reports had 0 RAG results (22/83 reports). All had `topic="unknown"` — ML returned no flagged classes, so the query was literally `"smart contract unknown vulnerability..."`. Fix: skip RAG when no classes flagged (nothing to search for)
  - Solidity code in query (`mapping(address => uint) balances`) confused the text embedder. The embedding of code-like text is meaningless. Fix: remove code from query, use mapped keywords instead
  - Class name mismatch: ML uses "IntegerUO" but corpus uses "integer overflow". No mapping existed. Fix: `_VULN_CLASS_TO_RAG_KEYWORDS` dict with 10 entries
  - 97.9% of chunks have `vuln_type="other"` — almost no labels in the corpus. This is a data quality problem, not a retrieval problem. Future: better labeling pipeline
- [ ] **What Would Break Without This:** Remove RAG → no exploit precedent in report, no `Evidence(source="rag")`. Remove hybrid → miss keyword matches (CVE numbers, addresses). Remove RRF → need score calibration between FAISS and BM25 (hard). Remove keyword mapping → back to zero-match on class names
- [ ] **At Scale:** 752 chunks (current, rebuild ~2 min) / 7,520 (~20 min) / 75,200 (~hours, need incremental indexing) / 752,000 (need sharded FAISS)
- [ ] **Try It Yourself:**
  ```
  cd agents && source .venv/bin/activate
  python3 -c "
  from src.rag.retriever import HybridRetriever
  r = HybridRetriever()
  results = r.search('reentrancy exploit attack pattern', k=3)
  for res in results: print(res.get('id', '?'), res.get('score', '?'))
  "
  pytest tests/test_rag_query.py -v
  ```
- [ ] **Limitations:** 97.9% unlabeled chunks (data quality), no code-specific embedder (text embedder only), no incremental indexing (full rebuild to update), no query expansion, BM25 not tuned (default params), corpus is English-only
- [ ] **Transferable Patterns:** (1) Hybrid retrieval — semantic + keyword, neither alone is enough (2) Reciprocal Rank Fusion — combine rankings without score calibration (3) Query engineering — know your embedding model's domain (text ≠ code). Each with interview story + when wrong.

## Step 6: Verify

- [ ] Open `retriever.py` and verify RRF formula uses `1/(rank + 60)`
- [ ] Open `rag_research.py` and verify `_VULN_CLASS_TO_RAG_KEYWORDS` has 10 entries
- [ ] Verify fallback query exists (if first returns 0, try simpler query)
- [ ] Confirm chunk count from `index_metadata.json` or the diagnostic script
- [ ] Confirm test count: 9 tests in `test_rag_query.py`
