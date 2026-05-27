══════════════════════════════════════════════════════
SENTINEL — IMPROVEMENT ACTION LIST
Generated: 2026-04-15 | Post deep-review session
══════════════════════════════════════════════════════

Tiers:
  TIER A — No retrain needed. Code fix or config only. Apply any time.
  TIER B — Requires retrain. Data pipeline + model change + full retraining run.
  TIER C — Architecture change. Significant design work before implementation.
  TIER D — Research / stretch. Experimental, uncertain payoff.

Priority within each tier ordered by impact vs effort.

══════════════════════════════════════════════════════
TIER A — APPLY WITHOUT RETRAIN
══════════════════════════════════════════════════════

A-01  predictor.py — defensive model.eval() in _score()
      File:   ml/src/inference/predictor.py
      Change: add self.model.eval() at top of _score() before torch.no_grad()
      Why:    if any external code calls model.train() between __init__ and
              _score(), Dropout re-enables and predictions become non-deterministic.
              One Python method call per inference — microseconds.
      Risk:   none

A-02  inference_server.py — fix stale docstring in _call_inference_api
      File:   agents/src/mcp/servers/inference_server.py
      Change: docstring still says "risk_score" and "vulnerabilities list" —
              actual response is label, confidence, threshold, truncated,
              num_nodes, num_edges
      Risk:   none

A-03  inference_server.py — fix stale batch comment
      File:   agents/src/mcp/servers/inference_server.py
      Change: comment in _handle_batch_predict says "schema validation is
              advisory in mcp 1.x" — WRONG for 1.27.0, schema IS enforced
      Risk:   none

A-04  inference_server.py — fix handle_sse return type hint
      File:   agents/src/mcp/servers/inference_server.py
      Change: handle_sse declared -> None but returns Response()
              fix type hint to -> Response
      Risk:   none

A-05  sentinel_model.py — add parameter_summary() call at startup
      File:   ml/src/inference/predictor.py
      Change: call self.model.parameter_summary() after load_state_dict()
              in Predictor.__init__ — logs trainable/frozen split on every
              startup to confirm checkpoint loaded correctly
      Risk:   none

A-06  api.py — add inference timeout
      File:   ml/src/inference/api.py
      Change: wrap predictor.predict_source() in asyncio.wait_for() with
              configurable timeout (default 60s, env: INFERENCE_TIMEOUT)
              return HTTP 504 on timeout instead of hanging forever
              use asyncio.to_thread() to avoid blocking event loop
      Why:    Slither + solc can hang on pathological Solidity — circular
              imports, deeply nested inheritance, large generated code
      Risk:   low — adds a new exception path, must test timeout behaviour

A-07  preprocess.py — add CI smoke test
      File:   ml/tests/test_inference_smoke.py  (new file)
      Change: single test that runs ContractPreprocessor.process_source()
              on a known minimal contract and asserts:
                graph.x.shape[1] == 8
                tokens["input_ids"].shape == (1, 512)
              catches graph_builder.py accidental use immediately
      Why:    the 17-dim vs 8-dim mismatch produces a runtime crash with no
              clear error message — this test catches it at CI time
      Risk:   none

A-08  rag_server.py — add full query DEBUG log alongside truncated INFO log
      File:   agents/src/mcp/servers/rag_server.py
      Change: add logger.debug("search | full_query='{}'", query) before
              the existing logger.info with query[:60] truncation
      Why:    debugging wrong RAG results requires the full query
      Risk:   none

A-09  agents/.env — add MODULE1_TIMEOUT=120
      File:   agents/.env
      Change: add MODULE1_TIMEOUT=120 for Docker Compose / cold-start safety
              current 30s default times out on CPU inference or cold model load
      Risk:   none

A-10  test_rag_server.py — write unit tests for rag_server.py
      File:   agents/tests/test_rag_server.py  (new file)
      Change: tests for: list_tools schema, _handle_search mock retriever,
              k cap enforcement, filters passthrough, dataclasses.asdict()
              serialisation, unknown tool name, retriever error returns
              structured TextContent not exception
      Why:    rag_server.py currently has zero unit tests
      Risk:   requires mocking HybridRetriever at module level — slightly
              complex setup but standard unittest.mock pattern

══════════════════════════════════════════════════════
TIER B — REQUIRES RETRAIN
══════════════════════════════════════════════════════

B-01  HIGHEST PRIORITY: Multi-label classification using BCCC folder structure
      Files:  ml/src/models/sentinel_model.py
              ml/src/training/ (trainer.py, dataset loader)
              ml/src/inference/predictor.py
              ml/src/inference/api.py
              agents/src/mcp/servers/inference_server.py

      Step 0 — FIRST: audit BCCC folder structure
        Run:  find ~/projects/sentinel/BCCC-SCsVul-2024 -name "*.sol" | \
                sed 's|.*/BCCC-SCsVul-2024/||' | cut -d'/' -f1 | sort | uniq -c
        Check: does any contract filename appear in multiple vulnerability folders?
        Result A (single-label): each contract in exactly one folder
          → multi-class softmax, Linear(64→12), CrossEntropyLoss
          → label = folder name → integer index
        Result B (multi-label): contracts appear in multiple folders
          → multi-label sigmoid, Linear(64→12), BCEWithLogitsLoss
          → label = binary vector [0,1,0,1,...] per class

      Model change:
        classifier = nn.Sequential(
            nn.Linear(64, 12),     # 12 vulnerability classes (no "safe" class)
            # no Sigmoid here — use BCEWithLogitsLoss for stability
        )
        At inference: torch.sigmoid(logits) → [B, 12] probabilities

      API response change:
        PredictResponse gains:
          vulnerabilities: list[{class: str, probability: float}]
          label: str  → most likely class (argmax) or "safe" if all < threshold
        Remove: single confidence field (replaced by per-class probabilities)

      inference_server.py change:
        _mock_prediction() updated to return per-class structure
        smoke test updated

      Why this is TIER B priority 1:
        The information already exists in BCCC. Binary classification throws
        it away. Multi-label output is what audit reports need — not just
        "vulnerable" but "reentrancy at 0.81, access_control at 0.34."
        This is also a significant interview differentiator.

      Effort: ~1 week (data pipeline + training + API + MCP update)

---

B-02  Remove Sigmoid from model, use BCEWithLogitsLoss in training
      Files:  ml/src/models/sentinel_model.py
              ml/src/training/trainer.py

      Change:
        sentinel_model.py: remove nn.Sigmoid() from classifier
          classifier = nn.Linear(64, 1)  # raw logits
        trainer.py: use F.binary_cross_entropy_with_logits(logits, labels)
        predictor._score(): apply torch.sigmoid(scores) manually after forward

      Why: BCEWithLogitsLoss is numerically more stable than BCELoss on
           sigmoid outputs. Prevents nan loss on extreme predictions.
           No architectural change — same model, same checkpoint format
           (incompatible with existing checkpoint — retrain required)

      Effort: 1 day + full retraining run
      Note: combine with B-01 — do both in the same retraining run

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

      Why: real DeFi contracts routinely exceed 512 tokens. Reentrancy
           in a 2000-token contract currently means the model sees only
           the first 512 tokens. Functions defined after token 512 are
           invisible to the transformer path.

      Important: must retrain with sliding window — model trained on
                 truncated inputs, if you switch to windowed at inference
                 only, the distribution shifts and scores become unreliable.

      API change: agent/inference_server timeout must increase (W × 8s per contract)
      Effort: ~3 days + full retraining run

---

B-04  Edge features — include CALLS/READS/WRITES/EMITS/INHERITS type in graph
      Files:  ml/src/inference/preprocess.py
              ml/src/data/graphs/ast_extractor_v4_production.py
              ml/src/models/gnn_encoder.py

      Change in preprocess.py _extract_graph():
        build edge_attr tensor [E, 5] — one-hot over edge types
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

      Change in gnn_encoder.py:
        GATConv(edge_dim=5)  — enables edge feature attention

      Why: CALLS vs WRITES vs READS edges carry different security information.
           A function WRITING a state variable after an external CALL is the
           reentrancy pattern. Currently the model infers this indirectly
           from node features — edge types make it explicit.
           _EDGE_TYPES dict already defined in preprocess.py — just unused.

      Effort: ~2 days + full retraining run (rebuild all 68K .pt graph files)

---

B-05  Global max pool experiment
      Files:  ml/src/models/gnn_encoder.py

      Change: swap global_mean_pool → global_max_pool
              or concat both: torch.cat([mean_pool, max_pool], dim=1) → [B,128]
              if concat: update FusionLayer gnn_dim=128

      Why: max pool preserves single-node extreme activations — more
           appropriate for vulnerability detection where one function
           dominates. Mean pool dilutes a single vulnerable function
           by the number of safe functions in the contract.

      Effort: 1 day + full retraining run
      Note: combine with B-01/B-02 — same retraining run

---

B-06  Richer node features — extend 8-dim to 11-dim
      Files:  ml/src/inference/preprocess.py
              ml/src/data/graphs/ast_extractor_v4_production.py
              ml/src/models/gnn_encoder.py

      Three new dimensions:
        [8]  has_external_call  — binary: calls msg.sender.call or address.call
        [9]  uses_tx_origin     — binary: uses tx.origin (auth bypass risk)
        [10] modifier_count     — int: number of modifiers protecting function

      Why: has_external_call is the most direct reentrancy pre-condition signal.
           Currently the model must infer it from complexity + reentrant flags.
           Making it explicit reduces the learning burden.

      Change in gnn_encoder.py: GATConv(in_channels=11)
      Must rebuild all 68K training graphs + retrain from scratch
      Effort: ~3 days + full retraining run

══════════════════════════════════════════════════════
TIER C — ARCHITECTURE CHANGES
══════════════════════════════════════════════════════

C-01  Multi-contract analysis — analyse all user-defined contracts in one file
      Files:  ml/src/inference/preprocess.py
              ml/src/data/graphs/ast_extractor_v4_production.py

      Change in _extract_graph():
        user_contracts = [c for c in sl.contracts if not c.is_from_dependency()]
        iterate ALL user_contracts for node insertion (not just [0])
        prefix node keys with contract name to avoid collisions:
          key = f"{contract.name}.{func.canonical_name}"
        add cross-contract INHERITS edges between user contracts only:
          for parent in contract.inheritance:
              if not parent.is_from_dependency():
                  _add_edge(contract.name, parent.name)

      Why: contracts that inherit from user-defined base contracts are currently
           analysed incomplete. withdraw() in Vault calls _withdraw() in VaultBase —
           VaultBase is ignored today. The vulnerability lives in VaultBase.

      No infinite loop risk: Slither resolves full inheritance tree before
      our code runs. is_from_dependency() is Slither's pre-computed flag.
      We only add edges between non-dependency contracts — finite set.

      Requires: rebuild all training graphs + retrain
      Effort: ~2 days implementation + full retraining run

---

C-02  Batch inference endpoint in api.py
      Files:  ml/src/inference/api.py
              ml/src/inference/predictor.py
              agents/src/mcp/servers/inference_server.py

      Change:
        predictor.py: add predict_batch(source_codes: list[str]) → list[dict]
          process all contracts → batch graph + batch tokens → single forward pass
        api.py: add POST /batch_predict endpoint
          PredictBatchRequest: contracts: list[str] (max 20)
          PredictBatchResponse: results: list[PredictResponse]
        inference_server.py: _handle_batch_predict sends one HTTP call
          instead of sequential loop

      Why: 10 contracts in one forward pass ≈ 12s instead of 80s (6x speedup)
           GPU parallelism — batched tensor operations use full GPU utilisation
           Current sequential loop serialises 10 × 8s = 80s for 10 contracts

      No retrain needed for the model itself.
      Effort: ~2 days

---

C-03  Rate limiting and request queuing for api.py
      Files:  ml/src/inference/api.py
              ml/src/inference/ (new: queue.py)

      Change:
        Add slowapi or fastapi-limiter for rate limiting
        Add asyncio.Queue with max depth — reject with 503 when full
        Add /queue_depth health metric

      Why: 10 concurrent requests at 8s each = 80s queue depth, OOM risk.
           Without limits, malicious or buggy callers can saturate the GPU.
           Acceptable without for M4 (internal service). Required for M6 public API.

      Effort: ~1 day

---

C-04  LangGraph state checkpointing for MCP session resilience
      Files:  agents/src/orchestration/ (M5 — not built yet)

      Change: when M5 LangGraph is built, checkpoint graph state to Redis
              after each node execution. On MCP server restart, agent
              reconnects and resumes from last checkpoint rather than
              restarting the audit from scratch.

      Why: inference_server.py restart (deploy, crash) drops all SSE sessions.
           Without checkpointing, a 5-agent audit that fails at step 4
           must restart from step 1.

      Effort: ~1 day (implement during M5 build)

══════════════════════════════════════════════════════
TIER D — RESEARCH / STRETCH
══════════════════════════════════════════════════════

D-01  LoRA fine-tuning of CodeBERT
      Files:  ml/src/models/transformer_encoder.py

      Change: add peft dependency, apply LoRA adapters to query/value matrices
              (r=8, ~500K trainable params added to frozen CodeBERT)
      Why:    improves semantic understanding of rare vulnerability classes
              (flash loan, oracle manipulation) that benefit most from
              task-specific fine-tuning
      When:   dataset grows to 200K+ contracts OR rare class F1 < 0.60
      Effort: ~3 days + training run with careful forgetting monitoring

---

D-02  GMU fusion instead of concat+MLP
      Files:  ml/src/models/fusion_layer.py

      Change: replace MLP with explicit gate mechanism
              gate = sigmoid(W · concat[gnn, transformer]) → [B, 64]
              output = gate ⊙ proj(gnn) + (1-gate) ⊙ proj(transformer)
      Why:    dynamic per-input modality weighting — learns when to trust
              structure vs semantics per contract class
      When:   experiment after B-01 retrain, compare F1 on val set
      Effort: 1 day + retrain

---

D-03  GraphSAGE for protocol-level analysis
      Files:  ml/src/models/gnn_encoder.py

      Change: replace GATConv with SAGEConv
      Why:    if SENTINEL expands to protocol-level graphs (entire DeFi
              ecosystems, 100s of interacting contracts) — GAT O(N²)
              attention becomes a bottleneck, GraphSAGE sampling scales better
      When:   only if analysing full protocol graphs, not individual contracts
      Effort: 1 day + retrain

---

D-04  Cross-attention fusion (GNN queries Transformer token sequence)
      Files:  ml/src/models/transformer_encoder.py
              ml/src/models/fusion_layer.py
              ml/src/models/sentinel_model.py

      Change: TransformerEncoder returns full last_hidden_state [B, 512, 768]
              FusionLayer uses nn.MultiheadAttention(query=gnn, key/value=tokens)
      Why:    GNN structure directly attends to relevant code tokens — more
              expressive than concatenation
      When:   only after B-01/B-02/B-03 done and F1 plateau reached
      Effort: ~3 days + retrain with architecture search

══════════════════════════════════════════════════════
RECOMMENDED EXECUTION ORDER
══════════════════════════════════════════════════════

  NOW (this week, before M5):
    A-01 through A-10  — all code fixes, no risk, no retrain

  AFTER M6 integration complete (system end-to-end):
    B-01 first  — audit BCCC structure, implement multi-label
    B-02        — combine with B-01 in same retraining run
    B-05        — combine with B-01/B-02 in same retraining run
    C-01        — multi-contract analysis (combine with above retrain)

  ONE COMBINED RETRAINING RUN covers B-01 + B-02 + B-05 + C-01:
    - multi-label output (B-01)
    - BCEWithLogitsLoss stability (B-02)
    - max pool experiment (B-05)
    - multi-contract graphs (C-01)
    All require rebuilding training data and retraining anyway.
    Do them together, not sequentially.

  AFTER FIRST RETRAIN, evaluate results:
    B-03  — sliding window (if long-contract F1 is low)
    B-04  — edge features (if reentrancy F1 needs improvement)
    B-06  — richer node features (if overall F1 plateau reached)
    C-02  — batch inference (if throughput is a bottleneck)

  RESEARCH (only if time and clear F1 gains needed):
    D-01, D-02, D-03, D-04

══════════════════════════════════════════════════════
QUICK REFERENCE — IMPACT vs EFFORT
══════════════════════════════════════════════════════

  Highest impact, lowest effort:
    A-01  defensive eval() in _score()
    A-06  inference timeout — prevents server hangs
    A-07  CI smoke test — catches 17-dim vs 8-dim immediately
    A-10  test_rag_server.py — fills the zero-test gap

  Highest impact, medium effort:
    B-01  multi-label classification — biggest model improvement
    C-01  multi-contract analysis — biggest coverage improvement
    C-02  batch inference — biggest throughput improvement

  High value, interview-differentiating:
    B-01  multi-label (specific class probabilities in audit report)
    B-03  sliding window (handles real production contracts)
    C-01  multi-contract (handles real production Solidity)

══════════════════════════════════════════════════════