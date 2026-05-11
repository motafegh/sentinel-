══════════════════════════════════════════════════════════════════
SENTINEL — FINAL COMPREHENSIVE IMPROVEMENT LIST
Generated: 2026-04-16 | Full source-code review + BCCC dataset audit
══════════════════════════════════════════════════════════════════

This document supersedes "SENTINEL — IMPROVEMENT ACTION LIST.md".
It incorporates all original 24 items plus 15 new issues discovered
during the deep source-code review session on 2026-04-15/16.

Tiers:
  TIER A  — No retrain. Code fix or config only. Apply any time.
  TIER B  — Requires retraining. Data pipeline or model change.
  TIER C  — Architecture change. Significant design work.
  TIER D  — Research / stretch. Experimental, uncertain payoff.

Severity tags (for Tier A items):
  [CRITICAL]  Silent breakage in production — fix before next deploy
  [HIGH]      Wrong behaviour, security risk, or data loss possible
  [MEDIUM]    Correctness or reliability issue under edge conditions
  [LOW]       Code quality, observability, missing tests

═══════════════════════════════════════════════════════════════════
TIER A — APPLY WITHOUT RETRAIN
═══════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────
CRITICAL FIXES (fix before any further deployment)
──────────────────────────────────────────────────────────────────

A-11  [CRITICAL]  run_proof.py — checkpoint format incompatibility
      File:   zkml/src/ezkl/run_proof.py

      Problem:
        run_proof.py loads the production checkpoint with
        weights_only=True, then immediately calls load_state_dict()
        on the raw loaded object.
        New-format checkpoints (all checkpoints from April 2026 onward)
        are a dict with keys: "model", "optimizer", "epoch",
        "best_f1", "config".
        run_proof.py has no code to detect or handle this format.
        Result: load_state_dict() receives a dict instead of an
        OrderedDict and raises a RuntimeError with a confusing
        message about unexpected keys "model", "optimizer", etc.
        This is a SILENT breakage — the proof pipeline compiles fine,
        only fails at runtime when a new checkpoint is used.

      Fix:
        ckpt = torch.load(ckpt_path, weights_only=True)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt   # legacy format
        model.load_state_dict(state_dict)

      Risk:   none — backward-compatible

---

A-12  [CRITICAL]  setup_circuit.py — assert statements stripped by -O
      File:   zkml/src/ezkl/setup_circuit.py

      Problem:
        Three assertions at lines 119, 152, 207 guard critical EZKL
        operations (calibrate_settings, compile_circuit, gen_pk/vk).
        When Python is run with the -O flag (optimise mode, common in
        Docker and production deployments), all assert statements are
        stripped from the bytecode.
        A failed EZKL step passes silently → a corrupt circuit is
        built on top of it → the final proof is invalid with no error.

      Fix:
        Replace all assert statements with explicit checks:
          res = await ezkl.calibrate_settings(...)
          if not res:
              raise RuntimeError("EZKL calibrate_settings failed")
        Do the same for compile_circuit and gen_pk/gen_vk calls.

      Risk:   none

---

A-13  [CRITICAL]  inference_server.py — mock response has extra "mock" key
      File:   agents/src/mcp/servers/inference_server.py

      Problem:
        _mock_prediction() constructs a response dict and adds
        "mock": True as a key.
        The real api.py returns a PredictResponse Pydantic model
        which has no "mock" field.
        Downstream code that processes the response using the
        PredictResponse schema will either ignore the key (if lenient)
        or raise a validation error (if strict).
        The mock path also hardcodes confidence=0.85 without
        matching the threshold/label/num_nodes/num_edges fields
        that the real response returns.

      Fix:
        _mock_prediction() should return a dict that exactly mirrors
        PredictResponse fields:
          {"label": "vulnerable", "confidence": 0.85,
           "threshold": 0.50, "truncated": False,
           "num_nodes": 12, "num_edges": 18}
        Remove "mock": True key entirely.
        If the mock flag is needed for testing, pass it as a
        separate out-of-band value (header, log line, etc.).

      Risk:   none

──────────────────────────────────────────────────────────────────
HIGH FIXES
──────────────────────────────────────────────────────────────────

A-14  [HIGH]  embedder.py — embed_query() has no retry logic
      File:   agents/src/rag/embedder.py

      Problem:
        embed_chunks() has exponential backoff retry (3 attempts,
        2^n seconds between retries) for API failures.
        embed_query() has no retry — one transient LM Studio timeout
        causes the entire RAG search to fail.
        Every search call goes through embed_query().
        embed_chunks() is called once during indexing.
        The function that is called most often has the weakest error
        handling.

      Fix:
        Apply the same exponential backoff decorator/pattern from
        embed_chunks() to embed_query().
        Same max_retries=3, same backoff logic.

      Risk:   none

---

A-15  [HIGH]  rag_server.py — relative import breaks outside agents context
      File:   agents/src/mcp/servers/rag_server.py

      Problem:
        from src.rag.retriever import HybridRetriever
        This is a relative import that assumes the working directory is
        agents/. Running the server from any other directory (tests,
        Docker, the project root) fails with ModuleNotFoundError.
        inference_server.py uses the same pattern and has the same bug.

      Fix:
        Use absolute imports with sys.path insertion:
          import sys, pathlib
          sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
          from agents.src.rag.retriever import HybridRetriever
        Or configure agents/ as a proper package with __init__.py and
        install it via pyproject.toml.

      Risk:   low — test with existing agents/tests/ after change

---

A-16  [HIGH]  pipeline.py — FileLock not released on exception
      File:   agents/src/ingestion/pipeline.py

      Problem:
        _write_index() acquires a FileLock to prevent concurrent
        writes to the FAISS index on disk.
        If an exception occurs during the write (disk full, permission
        error, corrupted data), the lock is not released because there
        is no try/finally block.
        Subsequent ingestion runs fail immediately with a LockTimeout
        and the index becomes permanently stale until the lock file
        is manually deleted.

      Fix:
        with FileLock(lock_path, timeout=30):
            ... all write operations ...
        The context manager already handles release on exception.
        Confirm the existing code uses "with" — if it uses
        lock.acquire() / lock.release() manually, switch to "with".

      Risk:   none

──────────────────────────────────────────────────────────────────
MEDIUM FIXES
──────────────────────────────────────────────────────────────────

A-17  [MEDIUM]  predictor.py — threshold not bounds-validated
      File:   ml/src/inference/predictor.py

      Problem:
        Predictor.__init__ accepts threshold from the checkpoint
        config dict or as a constructor argument.
        No validation that 0.0 < threshold < 1.0.
        A misconfigured checkpoint with threshold=0 would classify
        every contract as "vulnerable" silently.
        threshold=1.0 would classify everything as "safe".

      Fix:
        In __init__, after reading threshold:
          if not 0.0 < self.threshold < 1.0:
              raise ValueError(
                  f"threshold must be in (0, 1), got {self.threshold}"
              )

      Risk:   none

---

A-18  [MEDIUM]  api.py — missing specific error handling for OOM and missing files
      File:   ml/src/inference/api.py

      Problem:
        /predict wraps the entire predict_source() call in a bare
        except Exception → HTTP 500.
        Two important cases are not handled specifically:
        1. torch.cuda.OutOfMemoryError — OOM on GPU for very large
           contracts. Should return HTTP 507 (Insufficient Storage)
           or HTTP 413 (Request Entity Too Large) with a clear message.
        2. FileNotFoundError — if CodeBERT weights or graph data are
           missing. Returns a confusing "internal server error" instead
           of a clear startup misconfiguration message.

      Fix:
        except torch.cuda.OutOfMemoryError:
            raise HTTPException(413, "Contract too large for GPU memory")
        except FileNotFoundError as e:
            raise HTTPException(503, f"Model resource missing: {e}")
        except Exception as e:
            raise HTTPException(500, str(e))

      Risk:   none

---

A-19  [MEDIUM]  api.py — no guard for unloaded predictor
      File:   ml/src/inference/api.py

      Problem:
        /predict accesses request.app.state.predictor directly.
        If startup() failed (model load error, bad checkpoint) or
        was never called (test environment), state.predictor is None
        or does not exist, causing an AttributeError that leaks a
        Python traceback to the caller.

      Fix:
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(503, "Model not loaded — server is starting up")

      Risk:   none

---

A-20  [MEDIUM]  inference_server.py — new httpx client per request
      File:   agents/src/mcp/servers/inference_server.py

      Problem:
        _call_inference_api() creates a new httpx.AsyncClient() on
        every call. This means a new TCP connection is opened for every
        prediction request — no connection reuse or pooling.
        Under any moderate load (>2 concurrent requests) this adds
        ~20-50ms per request in TCP handshake overhead and can exhaust
        file descriptors.

      Fix:
        Create a single shared httpx.AsyncClient at server startup
        and reuse it. Use lifespan context manager or module-level
        client with proper cleanup on shutdown.
          client: httpx.AsyncClient | None = None
          async def startup():
              global client
              client = httpx.AsyncClient(timeout=90.0)

      Risk:   low — needs shutdown cleanup to avoid ResourceWarning

---

A-21  [MEDIUM]  deduplicator.py — timestamps recorded but never used; stale-doc risk
      File:   agents/src/ingestion/deduplicator.py

      Problem:
        The deduplicator records a first_seen timestamp for each
        document hash but never checks it. There is no TTL or expiry
        logic. A document that was updated upstream (new version of a
        DeFiHackLabs report, updated contract source) will be
        permanently skipped once its hash is recorded — even if the
        content changed on the next ingestion run.
        This creates a silent "stale forever" problem for any document
        that gets updated but keeps the same URL or filename.

      Fix (two-part):
        Part 1 — TTL expiry (prevents stale docs):
          Add max_age_days: int = 30 to __init__.
          In is_duplicate(): if now - first_seen > max_age_days: return False
          This forces re-ingestion of any document older than 30 days.
        Part 2 — Content hash check (detects updated docs):
          Store content_hash alongside URL hash.
          On re-ingestion, if content_hash differs: return False (ingest again).

      Risk:   low — increases re-indexing slightly for long-running deployments

---

A-22  [MEDIUM]  retriever.py — untyped return and unenforced list assumption
      File:   agents/src/rag/retriever.py

      Problem:
        get_index_info() has no return type annotation. The return
        value structure is implicit — callers must inspect the method
        body to know what keys exist.
        The chunks property is documented as returning a list but has
        no runtime check. If the underlying store returns a generator
        or set, downstream .index() calls fail with AttributeError.

      Fix:
        1. Add return type: def get_index_info(self) -> dict[str, Any]
           and define a TypedDict or dataclass for the return shape.
        2. In the chunks property getter:
             result = self._store.get_all()
             if not isinstance(result, list):
                 result = list(result)
             return result

      Risk:   none

---

A-23  [MEDIUM]  pipeline.py — skipped documents not logged at point of skip
      File:   agents/src/ingestion/pipeline.py

      Problem:
        When a document is skipped by the deduplicator, the skip is
        recorded in a counter but not logged immediately at the point
        of skipping. The summary is only logged at the end of the
        entire pipeline run.
        If the pipeline crashes mid-run, the reason for skipping
        specific documents is lost — making it hard to debug why
        important new docs were silently dropped.

      Fix:
        At the point of skip:
          logger.debug("skip | hash={} url={}", doc_hash, doc.url)
        Keep the existing end-of-run summary counter.

      Risk:   none

---

A-24  [MEDIUM]  sentinel_model.py — forward() type hint too broad
      File:   ml/src/models/sentinel_model.py

      Problem:
        forward(self, graphs: object, ...) uses object as the type
        for the graph batch argument.
        The actual expected type is torch_geometric.data.Batch.
        IDE autocomplete and static analysers (mypy, pyright) cannot
        check that the correct type is passed, hiding type errors until
        runtime.

      Fix:
        from torch_geometric.data import Batch
        def forward(self, graphs: Batch, ...) -> torch.Tensor:

      Risk:   none — purely declarative change

──────────────────────────────────────────────────────────────────
LOW / OBSERVABILITY FIXES (original list, still valid)
──────────────────────────────────────────────────────────────────

A-01  [LOW]  predictor.py — defensive model.eval() in _score()
      File:   ml/src/inference/predictor.py
      Change: add self.model.eval() at top of _score() before torch.no_grad()
      Why:    external code calling model.train() makes Dropout non-deterministic.
      Risk:   none

---

A-02  [LOW]  inference_server.py — stale docstring in _call_inference_api
      File:   agents/src/mcp/servers/inference_server.py
      Change: docstring says "risk_score" + "vulnerabilities list" — update to
              label, confidence, threshold, truncated, num_nodes, num_edges
      Risk:   none

---

A-03  [LOW]  inference_server.py — stale batch comment
      File:   agents/src/mcp/servers/inference_server.py
      Change: comment says "schema validation is advisory in mcp 1.x" — WRONG
              for 1.27.0 (schema IS enforced). Remove or correct.
      Risk:   none

---

A-04  [LOW]  inference_server.py — handle_sse return type hint
      File:   agents/src/mcp/servers/inference_server.py
      Change: declared -> None but returns Response() → fix to -> Response
      Risk:   none

---

A-05  [LOW]  predictor.py — add parameter_summary() at startup
      File:   ml/src/inference/predictor.py
      Change: call self.model.parameter_summary() after load_state_dict()
              to confirm trainable/frozen param counts on every startup.
      Risk:   none

---

A-06  [MEDIUM — see also A-18/A-19]  api.py — add inference timeout
      File:   ml/src/inference/api.py
      Change: wrap predict_source() in asyncio.wait_for(timeout=INFERENCE_TIMEOUT)
              use asyncio.to_thread() to avoid blocking the event loop
              return HTTP 504 on timeout
      Why:    Slither can hang indefinitely on pathological Solidity
      Risk:   low — adds new exception path

---

A-07  [LOW]  preprocess.py — add CI smoke test
      File:   ml/tests/test_inference_smoke.py (new)
      Change: test that process_source() on a minimal contract returns
              graph.x.shape[1] == 8 and tokens["input_ids"].shape == (1, 512)
              catches 17-dim vs 8-dim mismatch at CI time
      Risk:   none

---

A-08  [LOW]  rag_server.py — add full query DEBUG log
      File:   agents/src/mcp/servers/rag_server.py
      Change: logger.debug("search | full_query='{}'", query) before
              the existing logger.info with query[:60] truncation
      Risk:   none

---

A-09  [LOW]  agents/.env — add MODULE1_TIMEOUT=120
      File:   agents/.env
      Change: add MODULE1_TIMEOUT=120 (current default 30s times out on cold load)
      Risk:   none

---

A-10  [LOW]  rag_server.py — write unit tests
      File:   agents/tests/test_rag_server.py (new)
      Change: tests for list_tools schema, _handle_search with mock retriever,
              k cap enforcement, filters passthrough, unknown tool name,
              retriever error returns structured TextContent not exception
      Risk:   requires mocking HybridRetriever at module level

═══════════════════════════════════════════════════════════════════
TIER B — REQUIRES RETRAIN
═══════════════════════════════════════════════════════════════════

B-01  HIGHEST PRIORITY: Multi-label classification using BCCC folder structure
      Files:  ml/src/models/sentinel_model.py
              ml/src/training/ (trainer.py, dataset loader)
              ml/src/inference/predictor.py
              ml/src/inference/api.py
              agents/src/mcp/servers/inference_server.py

      ── STEP 0 STATUS: RESOLVED ──────────────────────────────────
      The BCCC structure audit was completed on 2026-04-15.
      Result B confirmed: the dataset IS genuinely multi-label.

      BCCC-SCsVul-2024/SourceCodes/ contains 12 folders:
        CallToUnknown, DenialOfService, ExternalBug, GasException,
        IntegerUO, MishandledException, NonVulnerable, Reentrancy,
        Timestamp, TransactionOrderDependence, UnusedReturn, WeakAccessMod

      Contract distribution:
        68,433 unique contract hashes total
        40,267 (58.8%) appear in exactly one folder
        28,166 (41.2%) appear in 2–9 folders simultaneously
        766 contradictory (appear in NonVulnerable AND a vuln folder)

      The current binary classification discards this multi-label
      information. The preprocessing script bakes only binary_label
      into graph.y via a "first-folder-wins" deduplication.

      Current contract_labels_correct.csv has 44,442 rows — all with
      class_count=1 because the multi-label structure was collapsed
      during data extraction.
      ─────────────────────────────────────────────────────────────

      Model change:
        classifier = nn.Linear(64, 12)   # raw logits, no Sigmoid
        At inference: torch.sigmoid(logits) → [B, 12] probabilities
        Loss: BCEWithLogitsLoss (numerically more stable than BCE+Sigmoid)

      API response change:
        PredictResponse gains:
          vulnerabilities: list[{class: str, probability: float}]
          label: str  → most likely class or "safe" if all < threshold
        Remove: single confidence field (replaced by per-class probs)

      Why this is the highest priority retrain item:
        The information already exists in BCCC. Binary classification
        throws it away. Multi-label output is what audit reports need —
        not just "vulnerable" but "reentrancy: 0.81, access_control: 0.34."

      Effort: ~1 week (data pipeline + training + API + MCP update)

---

B-02  Remove Sigmoid from model, use BCEWithLogitsLoss in training
      Files:  ml/src/models/sentinel_model.py
              ml/src/training/trainer.py

      Change:
        sentinel_model.py: classifier = nn.Linear(64, 1) (raw logits, no Sigmoid)
        trainer.py: use F.binary_cross_entropy_with_logits(logits, labels)
        predictor._score(): apply torch.sigmoid(scores) after forward()

      Why: BCEWithLogitsLoss is numerically more stable (log-sum-exp trick
           prevents gradient explosion on extreme predictions).
           Architecture unchanged — checkpoint format compatible.
           HOWEVER: new checkpoint is incompatible with old predictor code
           (sigmoid applied twice). Coordinate update of predictor + trainer.

      Effort: 1 day + full retraining run
      Note: combine with B-01 in the same retraining run

---

B-03  Sliding window tokenisation for contracts > 512 tokens
      Files:  ml/src/inference/preprocess.py
              ml/src/training/ (must retrain on same windowing strategy)

      Change in preprocess.py _tokenize():
        if true_token_count <= 512: current path (unchanged)
        else: sliding window with stride=256
          windows = chunk tokens into overlapping 512-token segments
          run CodeBERT on each window → collect CLS vectors
          max-pool across windows element-wise → [1, 768]

      Why: real DeFi contracts routinely exceed 512 tokens. Functions
           defined after token 512 are invisible to the transformer path.
           Reentrancy in a 2000-token contract is currently undetectable
           if the vulnerable function is in the second half.

      Important: must retrain — model trained on truncated inputs will
                 not generalise to windowed inputs at inference.

      Effort: ~3 days + full retraining run

---

B-04  Edge features — include edge type in graph (CALLS/READS/WRITES/etc)
      Files:  ml/src/inference/preprocess.py
              ml/src/data/graphs/ast_extractor_v4_production.py
              ml/src/models/gnn_encoder.py

      Change:
        _EDGE_TYPES dict is already defined in preprocess.py — but unused.
        Build edge_attr tensor [E, 5] — one-hot over edge types.
        GATConv(edge_dim=5) enables edge feature attention.

      Why: CALLS vs WRITES vs READS edges carry different security info.
           A function WRITING a state variable after an external CALL
           is the reentrancy pattern. The model currently infers this
           indirectly — edge types make it explicit.

      Effort: ~2 days + full retraining run (rebuild all 68K .pt graphs)

---

B-05  Global max pool experiment
      Files:  ml/src/models/gnn_encoder.py

      Change: swap global_mean_pool → global_max_pool
              or concat both → [B,128] (update FusionLayer gnn_dim=128)

      Why: max pool preserves single extreme activation — better for
           vulnerability detection where one vulnerable function dominates.
           Mean pool dilutes it by the number of safe functions.

      Effort: 1 day + full retraining run
      Note: combine with B-01/B-02 in the same retraining run

---

B-06  Richer node features — extend 8-dim to 11-dim
      Files:  ml/src/inference/preprocess.py
              ml/src/data/graphs/ast_extractor_v4_production.py
              ml/src/models/gnn_encoder.py

      Three new dimensions:
        [8]  has_external_call  — binary: calls msg.sender.call or address.call
        [9]  uses_tx_origin     — binary: uses tx.origin (auth bypass risk)
        [10] modifier_count     — int: number of modifiers protecting function

      Why: has_external_call is the most direct reentrancy pre-condition.
           Currently the model infers it from complexity + reentrant flags.
           Explicit signal reduces the learning burden.

      Must rebuild all 68K training graphs + retrain from scratch.
      Effort: ~3 days + full retraining run

═══════════════════════════════════════════════════════════════════
TIER C — ARCHITECTURE CHANGES
═══════════════════════════════════════════════════════════════════

C-01  Multi-contract analysis — analyse all user-defined contracts in one file
      Files:  ml/src/inference/preprocess.py
              ml/src/data/graphs/ast_extractor_v4_production.py

      Problem (confirmed in source):
        _extract_graph() does: contract = contracts[0]
        Only the FIRST contract in the file is analysed.
        A contract that inherits from a user-defined base contract
        (e.g. withdraw() in Vault calls _withdraw() in VaultBase)
        is analysed incomplete. The vulnerability in VaultBase is ignored.

      Change:
        user_contracts = [c for c in sl.contracts if not c.is_from_dependency()]
        Iterate ALL user_contracts for node insertion.
        Prefix node keys with contract name to avoid collisions.
        Add cross-contract INHERITS edges between user contracts.

      No infinite loop risk: Slither resolves full inheritance tree before
      our code runs. We only traverse non-dependency contracts — finite set.

      Requires: rebuild all training graphs + retrain
      Effort: ~2 days implementation + full retraining run

---

C-02  Batch inference endpoint in api.py
      Files:  ml/src/inference/api.py
              ml/src/inference/predictor.py
              agents/src/mcp/servers/inference_server.py

      Change:
        predictor.predict_batch(source_codes: list[str]) → list[dict]
        api.py: POST /batch_predict, max 20 contracts per call
        inference_server.py: send one HTTP call instead of sequential loop

      Why: 10 contracts in one forward pass ≈ 12s vs 80s sequential (6x).
           GPU parallelism — batched tensor ops use full GPU utilisation.

      No retrain needed. Effort: ~2 days.

---

C-03  Rate limiting and request queuing for api.py
      Files:  ml/src/inference/api.py

      Change:
        Add slowapi or fastapi-limiter for rate limiting.
        Add asyncio.Queue with max depth — reject with 503 when full.
        Add /queue_depth health metric.

      Why: without limits, concurrent large contracts cause OOM.
           Required for M6 public API. Acceptable to defer until then.

      Effort: ~1 day

---

C-04  LangGraph state checkpointing for MCP session resilience
      Files:  agents/src/orchestration/ (M5 — not built yet)

      Change: when M5 LangGraph is built, checkpoint graph state to Redis
              after each node execution.
              On MCP server restart, agent resumes from last checkpoint.

      Why: inference_server.py restart drops all SSE sessions.
           A 5-agent audit failing at step 4 restarts from step 1 today.

      Effort: ~1 day (implement during M5 build)

═══════════════════════════════════════════════════════════════════
TIER D — RESEARCH / STRETCH
═══════════════════════════════════════════════════════════════════

D-01  LoRA fine-tuning of CodeBERT
      Files:  ml/src/models/transformer_encoder.py
      Change: apply LoRA adapters to query/value matrices (r=8, ~500K params)
      Why:    improves semantic understanding of rare vulnerability classes
      When:   dataset grows to 200K+ contracts OR rare class F1 < 0.60
      Effort: ~3 days + training run with forgetting monitoring

---

D-02  GMU fusion instead of concat+MLP
      Files:  ml/src/models/fusion_layer.py
      Change: gate = sigmoid(W · concat[gnn, transformer]) → [B, 64]
              output = gate ⊙ proj(gnn) + (1-gate) ⊙ proj(transformer)
      Why:    dynamic per-input modality weighting per contract class
      When:   after B-01 retrain, compare F1 on val set
      Effort: 1 day + retrain

---

D-03  GraphSAGE for protocol-level analysis
      Files:  ml/src/models/gnn_encoder.py
      Change: replace GATConv with SAGEConv
      Why:    if SENTINEL expands to protocol-level graphs (entire DeFi
              ecosystems) — GAT O(N²) attention scales poorly;
              GraphSAGE sampling handles larger graphs
      When:   only if analysing full protocol graphs, not individual contracts
      Effort: 1 day + retrain

---

D-04  Cross-attention fusion (GNN queries Transformer token sequence)
      Files:  ml/src/models/transformer_encoder.py
              ml/src/models/fusion_layer.py
      Change: TransformerEncoder returns full last_hidden_state [B, 512, 768]
              FusionLayer uses nn.MultiheadAttention(query=gnn, key/value=tokens)
      Why:    GNN structure directly attends to relevant code tokens —
              more expressive than concatenation
      When:   only after B-01/B-02/B-03 done and F1 plateau reached
      Effort: ~3 days + retrain with architecture search

═══════════════════════════════════════════════════════════════════
RECOMMENDED EXECUTION ORDER
═══════════════════════════════════════════════════════════════════

PHASE 1 — IMMEDIATE (before any further deployment):
  A-11  run_proof.py checkpoint compat         [CRITICAL — fix now]
  A-12  setup_circuit.py assert → explicit     [CRITICAL — fix now]
  A-13  mock prediction schema mismatch        [CRITICAL — fix now]

PHASE 2 — THIS WEEK (no retrain, all code fixes):
  A-14  embed_query retry logic               [HIGH]
  A-15  rag_server absolute imports           [HIGH]
  A-16  pipeline FileLock try/finally         [HIGH]
  A-17  predictor threshold bounds check      [MEDIUM]
  A-18  api.py OOM + FileNotFoundError        [MEDIUM]
  A-19  api.py unloaded predictor guard       [MEDIUM]
  A-06  api.py inference timeout              [MEDIUM]
  A-20  httpx connection pooling              [MEDIUM]
  A-21  deduplicator TTL expiry               [MEDIUM]
  A-01  predictor defensive eval()            [LOW]
  A-02  inference_server docstring            [LOW]
  A-03  inference_server batch comment        [LOW]
  A-04  handle_sse return type               [LOW]
  A-05  parameter_summary at startup          [LOW]
  A-07  CI smoke test                         [LOW]
  A-08  rag_server full query debug log       [LOW]
  A-09  agents/.env MODULE1_TIMEOUT           [LOW]
  A-10  test_rag_server unit tests            [LOW]
  A-22  retriever type annotations            [MEDIUM]
  A-23  pipeline skip logging                 [MEDIUM]
  A-24  sentinel_model forward type hint      [MEDIUM]

PHASE 3 — AFTER M6 END-TO-END INTEGRATION:
  ONE COMBINED RETRAINING RUN covering:
    B-01  multi-label classification (primary goal)
    B-02  BCEWithLogitsLoss stability
    B-05  max pool experiment
    C-01  multi-contract graph analysis
  All four require rebuilding training data + retraining anyway.
  Doing them together saves 3 separate training runs (~10 days vs ~30).

PHASE 4 — AFTER FIRST RETRAIN, EVALUATE AND ADD:
  B-03  sliding window (if long-contract F1 is low)
  B-04  edge features (if reentrancy F1 needs improvement)
  B-06  richer node features (if overall F1 plateaus)
  C-02  batch inference (if throughput is a bottleneck)
  C-03  rate limiting (before M6 public API launch)
  C-04  LangGraph checkpointing (implement during M5)

PHASE 5 — RESEARCH (only if time and clear F1 gain needed):
  D-01, D-02, D-03, D-04

═══════════════════════════════════════════════════════════════════
QUICK REFERENCE — ITEM COUNTS
═══════════════════════════════════════════════════════════════════

  Tier A:  24 items  (3 critical, 3 high, 9 medium, 9 low)
  Tier B:   6 items  (all require retraining)
  Tier C:   4 items  (architecture changes)
  Tier D:   4 items  (research)
  ─────────────────────────────────────────────────────
  Total:   38 items

  Status:
    A-11, A-12, A-13  — CRITICAL, fix before next deployment
    B-01 Step 0       — RESOLVED (multi-label confirmed, use BCEWithLogitsLoss + Linear(64,12))
    All remaining     — open

═══════════════════════════════════════════════════════════════════
QUICK REFERENCE — HIGHEST IMPACT vs LOWEST EFFORT
═══════════════════════════════════════════════════════════════════

  Fix now, 15 min each:
    A-11  run_proof.py checkpoint compat — prevents silent ZKML breakage
    A-12  setup_circuit.py assert fix    — prevents silent proof corruption
    A-01  defensive eval() in _score()   — one line

  Fix this week, 1-2 hours each:
    A-06  inference timeout              — prevents server hangs
    A-07  CI smoke test                  — catches 17-dim mismatch at CI
    A-14  embed_query retry              — one-liner copy from embed_chunks
    A-19  api.py unloaded predictor guard — 3 lines

  High model quality impact:
    B-01  multi-label (biggest model improvement — 41% of data is multi-label)
    C-01  multi-contract (biggest coverage improvement — fixes contracts[0] bug)
    B-03  sliding window (real DeFi contracts are >512 tokens)

  High interview/portfolio differentiation:
    B-01  specific vulnerability class probabilities in audit report
    B-03  sliding window for production contracts
    C-01  cross-contract inheritance analysis

══════════════════════════════════════════════════════════════════
