```
══════════════════════════════════════════════════════════════════
SENTINEL — MASTER PLAN
Generated: 2026-04-15 | Complete reference across all decisions
══════════════════════════════════════════════════════════════════

This document is the single source of truth for what to build next,
in what order, and why. Three tracks run in parallel:

  TRACK 1: Immediate code fixes (no retrain, no architecture change)
  TRACK 2: Agent completion (M4.3 → M5 → M6)
  TRACK 3: ML upgrade (multi-label retrain + downstream updates)

Tracks 1 and 2 proceed now. Track 3 starts after M6 is complete.

══════════════════════════════════════════════════════════════════
CURRENT SYSTEM STATE (2026-04-15)
══════════════════════════════════════════════════════════════════

  LIVE AND WORKING:
    Module 1   ml/src/inference/   FastAPI port 8001, GPU inference confirmed
    Module 2   zkml/               Deployed Sepolia, ZKMLVerifier + AuditRegistry
    Module 3   mlops/              MLflow + DVC + Dagster (basic setup)
    M4.1       inference_server.py port 8010, mock=False, real inference wired
    M4.2       rag_server.py       port 8011, 1339 chunks, smoke-tested

  NEXT TO BUILD:
    M4.3 → M5 → M6

  PENDING RETRAIN:
    Binary → Multi-label upgrade (Track 3)

  MODEL:
    Architecture:   GNN (3-layer GAT) + CodeBERT (frozen) + FusionLayer + head
    Checkpoint:     run-alpha-tune_best.pt  477MB  RTX 3070
    Output:         binary score [0,1] — "vulnerable" or "safe"
    Threshold:      0.50 (tuned on val set, F1-macro criterion)
    Known limit:    binary collapses 11 vulnerability classes into one score
                    41.2% of BCCC contracts are genuinely multi-label

══════════════════════════════════════════════════════════════════
TRACK 1 — IMMEDIATE CODE FIXES
No retrain. No architecture change. Apply any time.
══════════════════════════════════════════════════════════════════

A-01  predictor.py — defensive model.eval() in _score()
      File:   ml/src/inference/predictor.py
      Fix:    add self.model.eval() at start of _score() before torch.no_grad()
      Why:    if anything calls model.train() between __init__ and _score(),
              Dropout re-enables and predictions become non-deterministic.
              One method call — microseconds. Eliminates an entire bug class.

A-02  preprocess.py — remove stale ASTExtractor import
      File:   ml/src/inference/preprocess.py  line 77
      Fix:    delete: from ml.src.data.graphs.ast_extractor import ASTExtractor
      Why:    ASTExtractor is imported but never called — confirmed by reading
              the full file. It creates a false dependency on a module that
              produces 17-dim features (incompatible with the trained model).
              The comment on line 75-76 saying "used by process()" is WRONG.

A-03  inference_server.py — fix stale docstring in _call_inference_api
      File:   agents/src/mcp/servers/inference_server.py
      Fix:    docstring says "risk_score" and "vulnerabilities list" —
              actual api.py response is: label, confidence, threshold,
              truncated, num_nodes, num_edges
      Why:    misleading documentation causes bugs when next developer
              tries to parse the response based on the docstring

A-04  inference_server.py — fix stale batch comment
      File:   agents/src/mcp/servers/inference_server.py
      Fix:    comment in _handle_batch_predict says "schema validation is
              advisory in mcp 1.x" — WRONG for mcp 1.27.0
              schema IS enforced at protocol level
      Why:    incorrect comment will mislead during M5 agent development

A-05  inference_server.py — fix handle_sse return type hint
      File:   agents/src/mcp/servers/inference_server.py
      Fix:    handle_sse declared -> None but returns Response()
              change to -> Response
      Why:    type correctness, no runtime impact but confusing

A-06  sentinel_model.py — call parameter_summary() at startup
      File:   ml/src/inference/predictor.py
      Fix:    add self.model.parameter_summary() after load_state_dict()
              in Predictor.__init__
      Why:    logs trainable/frozen split on every startup — confirms
              checkpoint loaded correctly, useful during debugging

A-07  api.py — add inference timeout
      File:   ml/src/inference/api.py
      Fix:    wrap predictor.predict_source() in asyncio.wait_for() with
              configurable timeout:
                result = await asyncio.wait_for(
                    asyncio.to_thread(predictor.predict_source, body.source_code),
                    timeout=float(os.getenv("INFERENCE_TIMEOUT", "60.0"))
                )
              return HTTP 504 on asyncio.TimeoutError
      Why:    Slither + solc can hang on pathological Solidity. Without this
              the endpoint hangs forever, connections pile up, server OOMs.
              asyncio.to_thread() prevents blocking the event loop.

A-08  ml/tests — add inference smoke test
      File:   ml/tests/test_inference_smoke.py  (new)
      Fix:    single test: ContractPreprocessor.process_source() on minimal
              Solidity → assert graph.x.shape[1] == 8
                       → assert tokens["input_ids"].shape == (1, 512)
      Why:    the 17-dim vs 8-dim mismatch (graph_builder.py accidentally
              used) crashes at runtime deep in the model with no clear error.
              This test catches it at CI time.

A-09  rag_server.py — add full query DEBUG log
      File:   agents/src/mcp/servers/rag_server.py
      Fix:    add logger.debug("search | full_query='{}'", query) before
              the existing INFO log with query[:60] truncation
      Why:    debugging wrong RAG results requires the full query string

A-10  agents/.env — add MODULE1_TIMEOUT=120
      File:   agents/.env
      Fix:    add MODULE1_TIMEOUT=120
      Why:    current 30s default times out on CPU inference or cold
              model load. 120s is safe for production Docker Compose use.

A-11  test_rag_server.py — write missing unit tests
      File:   agents/tests/test_rag_server.py  (new)
      Fix:    tests for: list_tools schema validation, _handle_search
              with mocked HybridRetriever, k cap enforcement, filters
              passthrough, dataclasses.asdict() serialisation, unknown
              tool name returns TextContent not exception, retriever
              RuntimeError returns structured error TextContent
      Why:    rag_server.py currently has zero unit tests — coverage gap

══════════════════════════════════════════════════════════════════
TRACK 2 — AGENT COMPLETION
M4.3 → M5 → M6
══════════════════════════════════════════════════════════════════

────────────────────────────────────────────────
M4.3 — sentinel-audit MCP server
────────────────────────────────────────────────

  File:     agents/src/mcp/servers/audit_server.py
  Port:     8012
  Pattern:  identical SSE wiring to inference_server.py and rag_server.py

  PREREQS (check before starting):
    grep SEPOLIA agents/.env           → SEPOLIA_RPC_URL must be set
    find zkml/ -name "AuditRegistry*"  → need ABI JSON location

  TOOLS — read-only first, write tool after RPC confirmed working:

    get_latest_audit(contract_address: str)
      → calls AuditRegistry.getLatestAudit(contractAddress) via web3
      → returns: {score, proofHash, timestamp, agent, verified}

    get_audit_history(contract_address: str, limit: int = 10)
      → calls AuditRegistry.getAuditHistory(contractAddress)
      → returns: list of AuditResult structs

    submit_audit(contract_address, score, zk_proof, public_signals)  [LATER]
      → requires valid ZK proof — implement after Phase 3 of Track 3
      → calls AuditRegistry.submitAudit() — requires MIN_STAKE (1000 SENTINEL)

  CONTRACT:  0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf (Sepolia proxy)
  WEB3:      web3 ^7.15.0 already in agents/pyproject.toml
  PORT MAP:  8012 (avoids 8001 Module1, 8010 inference MCP, 8011 rag MCP)

  ENV VARS TO ADD to agents/.env:
    MCP_AUDIT_PORT=8012
    SEPOLIA_RPC_URL=<rpc_url>
    AUDIT_REGISTRY_ADDRESS=0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf

  DELIVERABLES:
    audit_server.py
    scripts/smoke_audit_mcp.py
    tests/test_audit_server.py

────────────────────────────────────────────────
M5 — LangGraph orchestration
────────────────────────────────────────────────

  Files:    agents/src/orchestration/graph.py
            agents/src/orchestration/state.py
            agents/src/orchestration/nodes.py

  CONNECTS ALL 3 MCP SERVERS:
    MultiServerMCPClient config:
      "sentinel-inference": {"url": "http://localhost:8010/sse"}
      "sentinel-rag":       {"url": "http://localhost:8011/sse"}
      "sentinel-audit":     {"url": "http://localhost:8012/sse"}

  STATE SCHEMA (TypedDict):
    contract_code:    str
    contract_address: str
    ml_result:        dict | None    ← from inference MCP predict tool
    rag_results:      list | None    ← from rag MCP search tool
    audit_history:    list | None    ← from audit MCP get_audit_history tool
    static_findings:  dict | None    ← from Slither direct call
    final_report:     dict | None    ← synthesizer output
    error:            str | None

  GRAPH NODES:
    ml_assessment    → calls predict tool
    rag_research     → calls search tool (query built from ml_result)
    audit_check      → calls get_audit_history tool
    static_analysis  → calls Slither directly (no MCP — local tool)
    synthesizer      → assembles final report from all node outputs

  CONDITIONAL ROUTING:
    ml_assessment → check confidence
      confidence > 0.70 → rag_research → static_analysis → synthesizer
      confidence ≤ 0.70 → synthesizer  (fast path — low risk contract)

  STATE CHECKPOINTING (LangGraph MemorySaver or Redis):
    checkpoint after each node — if MCP server restarts mid-audit,
    agent reconnects and resumes from last checkpoint, not from start

  DELIVERABLES:
    agents/src/orchestration/graph.py
    agents/src/orchestration/state.py
    agents/src/orchestration/nodes.py
    agents/tests/test_graph_routing.py
    scripts/smoke_langgraph.py

────────────────────────────────────────────────
M6 — Five agents + full integration
────────────────────────────────────────────────

  FILES:
    agents/src/agents/static_analyzer.py  → Slither + Mythril
    agents/src/agents/ml_intelligence.py  → calls inference MCP
    agents/src/agents/rag_researcher.py   → calls rag MCP
    agents/src/agents/code_logic.py       → AST analysis + access control
    agents/src/agents/synthesizer.py      → structured AuditReport

  PYDANTIC AUDIT REPORT SCHEMA:
    class AuditReport(BaseModel):
        contract_address: str
        overall_label:    str           "vulnerable" | "safe"
        confidence:       float
        vulnerabilities:  list[VulnFinding]
        rag_evidence:     list[ExploitReference]
        static_findings:  list[StaticFinding]
        on_chain_history: list[AuditRecord]
        recommendation:   str
        timestamp:        datetime
        model_version:    str

  MODULE 1 TIMEOUT UPDATE:
    When wired into Docker Compose, set MODULE1_TIMEOUT=120
    in docker-compose.yml environment section

  MODULE 1 BATCH ENDPOINT (C-02 from improvement list):
    Add POST /batch_predict to api.py at M6 time
    Predictor.predict_batch() for 6x throughput on multi-contract protocols

  END-TO-END TEST:
    Run full audit on a real deployed vulnerable contract from DeFiHackLabs
    Verify: MCP tools called → report generated → proof verified on Sepolia

══════════════════════════════════════════════════════════════════
TRACK 3 — MULTI-LABEL ML UPGRADE
Start AFTER M6 is complete and end-to-end system is verified.
Full retrain required. ZKML circuit must be rebuilt after retrain.
══════════════════════════════════════════════════════════════════

BACKGROUND — why this matters:
  Current model: binary output (0/1) — throws away 11-class label structure
  BCCC dataset:  41.2% of contracts genuinely appear in multiple vuln folders
  What we gain:  "Reentrancy: 0.81, IntegerUO: 0.23" instead of "vulnerable: 0.75"
  What stays:    all 68,555 graph .pt files untouched — only labels change

THE TWO HASH SYSTEMS (never mix them):
  BCCC SHA256:   hash of file content   → BCCC filename e.g. 48e59d16.sol
  Internal MD5:  hash of file path      → .pt filename e.g. a1b2c3d4.pt
  Bridge:        graph.contract_path inside .pt → extract SHA256 from basename
                 → look up SHA256 in BCCC folder scan → get vuln set
                 → write row: MD5_stem, [11-dim multi-hot]

OUTPUT VECTOR — 11 classes (alphabetical, NonVulnerable excluded):
  Index  Class
    0    CallToUnknown
    1    DenialOfService
    2    ExternalBug
    3    GasException
    4    IntegerUO
    5    MishandledException
    6    Reentrancy
    7    Timestamp
    8    TransactionOrderDependence
    9    UnusedReturn
    10   WeakAccessMod

"Safe" = all 11 probabilities below threshold. No "safe" output node needed.

────────────────────────────────────────────────
PHASE 0 — Build multilabel_index.csv
────────────────────────────────────────────────

  New file:  ml/scripts/build_multilabel_index.py
  Output:    ml/data/processed/multilabel_index.csv
  Columns:   md5_stem, CallToUnknown, DenialOfService, ExternalBug,
             GasException, IntegerUO, MishandledException, Reentrancy,
             Timestamp, TransactionOrderDependence, UnusedReturn, WeakAccessMod
  Rows:      68,555  (one per .pt file)

  SHORTCUT — use existing CSV first:
    contract_labels_correct.csv already has file_hash (SHA256) + Class01-12 columns
    Use it for the 44,442 contracts it covers — saves rescanning 111K files
    For remaining ~24K .pt files not in CSV: scan BCCC folders as fallback
    For .pt files whose SHA256 is not found anywhere: binary fallback (graph.y)

  ALGORITHM:
    Step 1:  Read contract_labels_correct.csv
             Build: sha256 → [11-dim multi-hot] from Class01..Class11 columns
             (skip Class12:NonVulnerable — not an output class)

    Step 2:  Scan BCCC-SCsVul-2024/SourceCodes/ 11 vuln folders
             Augment sha256 map with any hashes not in CSV
             (handles the ~24K contracts dropped from CSV due to Slither failures
             that were still successfully processed as .pt files somehow)

    Step 3:  Handle 766 contradictory labels (in vuln folder AND NonVulnerable)
             Resolution: keep all vulnerability labels, ignore NonVulnerable tag
             These are almost certainly mislabels in BCCC

    Step 4:  For each .pt file in ml/data/graphs/:
             md5_stem = Path(pt_file).stem
             Load graph — read graph.contract_path
             sha256 = Path(graph.contract_path).stem  ← BCCC SHA256 filename
             Look up sha256 in map from Steps 1+2
             If found:    row = (md5_stem, multi-hot from map)
             If not found: row = (md5_stem, binary fallback from graph.y)
                           log WARNING — these are non-BCCC contracts

    Step 5:  Write multilabel_index.csv
             Print class distribution summary (pos count per class)
             → needed for pos_weight computation in trainer

  VERIFICATION:
    Row count == 68,555
    ~41% of rows have sum(cols) > 1  (multi-label contracts)
    Zero duplicate md5_stem values

────────────────────────────────────────────────
PHASE 1 — Update DualPathDataset
────────────────────────────────────────────────

  File:  ml/src/datasets/dual_path_dataset.py

  Changes:
    __init__: add label_csv: Path | None = None parameter
              if label_csv provided: load as {md5_stem: tensor[11, float32]}
              if not provided: fall back to graph.y binary (backward compat)

    __getitem__: when multilabel mode:
                 label = label_map[md5_stem]  → tensor [11] float32
                 NOT graph.y
                 NOT squeezed to scalar

    dual_path_collate_fn:
                 labels stack to [B, 11] float — no squeeze(1)

  VERIFICATION:
    Unit test: dataset[0] in multilabel mode → label.shape == (11,)
               label.dtype == torch.float32
    Unit test: dataset[0] in binary mode → unchanged behaviour

────────────────────────────────────────────────
PHASE 2 — Update SentinelModel
────────────────────────────────────────────────

  File:  ml/src/models/sentinel_model.py

  Changes:
    __init__: add num_classes: int = 1 parameter
              store as self.num_classes

    classifier:
      OLD: nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
      NEW: nn.Linear(fusion_output_dim, num_classes)
           NO Sigmoid — raw logits → BCEWithLogitsLoss handles sigmoid internally

    forward():
      OLD: scores = self.classifier(fused).squeeze(1)   → [B]
      NEW: scores = self.classifier(fused)               → [B, num_classes]
           if self.num_classes == 1: scores = scores.squeeze(1)  → [B] compat

  Backward compat: num_classes=1 restores binary behaviour exactly
                   existing binary checkpoint still loadable for comparison

  VERIFICATION:
    SentinelModel(num_classes=11).forward() → output shape [B, 11]
    SentinelModel(num_classes=1).forward()  → output shape [B]  (unchanged)

────────────────────────────────────────────────
PHASE 3 — Update trainer.py
────────────────────────────────────────────────

  File:  ml/src/training/trainer.py

  Changes:

    TrainConfig additions:
      num_classes: int = 11
      label_csv: str = "ml/data/processed/multilabel_index.csv"
      REMOVE: focal_gamma, focal_alpha  (FocalLoss no longer used)

    pos_weight computation (run once at trainer startup):
      Load all training labels from multilabel_index.csv
      For each class c:
        pos_count[c] = sum(label[:,c] == 1) across training set
        neg_count[c] = total_train - pos_count[c]
        pos_weight[c] = neg_count[c] / pos_count[c]
      pos_weight = torch.tensor([...], device=device)  # [11]

    Loss function:
      OLD: FocalLoss(gamma=2.0) or BCELoss
      NEW: nn.BCEWithLogitsLoss(pos_weight=pos_weight)
           Why: log-sum-exp trick prevents nan on extreme logits
                pos_weight handles per-class imbalance precisely

    train_one_epoch():
      labels are [B, 11] float32 — pass directly to BCEWithLogitsLoss
      no .float() cast or .view(-1) needed

    evaluate():
      probs = torch.sigmoid(logits)          → [B, 11]
      preds = (probs >= threshold).long()    → [B, 11]  binarized

      Metrics to compute and log:
        val_f1_macro    = f1_score(y_true, preds, average='macro')
        val_f1_micro    = f1_score(y_true, preds, average='micro')
        val_hamming     = hamming_loss(y_true, preds)
        val_f1_{class}  = f1_score per class for all 11 (log to MLflow)

      Primary checkpoint metric: val_f1_macro (consistent with binary training)

    Checkpoint save format — add num_classes to config dict:
      {"model": state_dict, "optimizer": ..., "epoch": ...,
       "best_f1": ..., "config": {..., "num_classes": 11}}

  MLflow experiment: sentinel-multilabel
  (separate from sentinel-training — clean comparison with binary run)

────────────────────────────────────────────────
PHASE 4 — Update predictor.py
────────────────────────────────────────────────

  File:  ml/src/inference/predictor.py

  Changes:

    CLASS_NAMES (module-level constant — same order as training vector):
      CLASS_NAMES = [
          "CallToUnknown", "DenialOfService", "ExternalBug",
          "GasException", "IntegerUO", "MishandledException",
          "Reentrancy", "Timestamp", "TransactionOrderDependence",
          "UnusedReturn", "WeakAccessMod",
      ]

    __init__:
      Load num_classes from checkpoint config dict (default 11)
      Instantiate SentinelModel(num_classes=num_classes)

    _score():
      scores = self.model(batch, input_ids, attention_mask)  → [1, 11] logits
      probs  = torch.sigmoid(scores).squeeze(0)              → [11] float
      probs_list = probs.cpu().tolist()

      vulnerabilities = sorted(
          [{"class": CLASS_NAMES[i], "probability": round(p, 4)}
           for i, p in enumerate(probs_list) if p >= self.threshold],
          key=lambda x: x["probability"],
          reverse=True,
      )

      return {
          "label":           "vulnerable" if vulnerabilities else "safe",
          "vulnerabilities": vulnerabilities,   # [] if all below threshold
          "threshold":       self.threshold,
          "truncated":       tokens["truncated"],
          "num_nodes":       int(graph.num_nodes),
          "num_edges":       int(graph.num_edges),
      }

────────────────────────────────────────────────
PHASE 5 — Update api.py
────────────────────────────────────────────────

  File:  ml/src/inference/api.py

  New Pydantic model:
    class VulnerabilityResult(BaseModel):
        vulnerability_class: str
        probability: float = Field(..., ge=0.0, le=1.0)

  Updated PredictResponse:
    class PredictResponse(BaseModel):
        label:           str                        "safe" or "vulnerable"
        vulnerabilities: list[VulnerabilityResult]  sorted desc by probability
        threshold:       float
        truncated:       bool
        num_nodes:       int
        num_edges:       int
    REMOVE: confidence: float  (replaced by per-class probabilities)

  /predict endpoint: no logic change — response_model handles new shape

────────────────────────────────────────────────
PHASE 6 — Update inference_server.py
────────────────────────────────────────────────

  File:  agents/src/mcp/servers/inference_server.py

  _mock_prediction():
    Return new multi-label schema:
      {
        "label":           "vulnerable" | "safe",
        "vulnerabilities": [{"vulnerability_class": "Reentrancy",
                              "probability": 0.72}],
        "threshold":       0.50,
        "truncated":       False,
        "num_nodes":       42,
        "num_edges":       58,
      }
    REMOVE: "mock": True  (fixes A-03 simultaneously)

  _call_inference_api() docstring: update to reflect new schema
  predict tool description: update to mention per-class vulnerability probs

────────────────────────────────────────────────
PHASE 7 — Update tests and smoke scripts
────────────────────────────────────────────────

  ml/tests/test_api.py:
    - assert "vulnerabilities" in response, is a list
    - each item has "vulnerability_class" (str) and "probability" (float [0,1])
    - assert "confidence" NOT in response
    - safe contract → label=="safe" AND vulnerabilities==[]
    - vulnerable contract → label=="vulnerable" AND vulnerabilities non-empty

  agents/tests/test_inference_server.py:
    - update _mock_prediction test: assert new schema shape
    - assert no "mock" key in mock output
    - update _handle_predict test: validate new response structure

  agents/scripts/smoke_inference_mcp.py:
    - remove assert "confidence" in parsed
    - add assert "vulnerabilities" in parsed
    - add assert isinstance(parsed["vulnerabilities"], list)
    - remove assert parsed.get("mock") is not True
      (mock key no longer exists)

────────────────────────────────────────────────
PHASE 8 — Retrain
────────────────────────────────────────────────

  Command (from project root):
    cd ~/projects/sentinel
    poetry run python ml/scripts/train.py

  TrainConfig:
    num_classes = 11
    label_csv   = "ml/data/processed/multilabel_index.csv"
    lr          = 3e-4
    epochs      = 40
    batch_size  = 32
    loss        = BCEWithLogitsLoss with pos_weight (computed from training labels)

  What stays the same — DO NOT CHANGE:
    GNNEncoder architecture (3-layer GAT, in_channels=8, heads=8)
    TransformerEncoder (frozen CodeBERT CLS token)
    FusionLayer (832 → 256 → 64 MLP)
    Graph .pt files (68,555 files untouched)
    Token .pt files (untouched)
    Split index files (untouched — position indices still valid)

  Expected training metrics:
    Baseline target:  val_f1_macro > 0.50
    Aim:              val_f1_macro > 0.65
    Note:             macro-F1 weights all 11 classes equally
                      micro-F1 will be higher (dominated by IntegerUO)
                      watch per-class F1 — WeakAccessMod and Timestamp
                      will likely be the hardest classes

  MLflow experiment:  sentinel-multilabel (not sentinel-training)

────────────────────────────────────────────────
PHASE 9 — Per-class threshold tuning
────────────────────────────────────────────────

  File:  ml/scripts/tune_threshold.py  (update existing)

  Sweep thresholds per class on val set
  Report: per-class F1 at each threshold + macro-F1
  Pick: threshold per class that maximises that class's F1
        OR single shared threshold that maximises macro-F1
        (single threshold simpler, per-class more accurate — try both)

  Update predictor.py to support per-class thresholds if needed:
    self.thresholds: list[float]  (one per class)

────────────────────────────────────────────────
PHASE 10 — Rebuild ZKML circuit
────────────────────────────────────────────────

  WHY: EZKL circuit was compiled from the binary (1-output) model.
       Multi-label model has 11 outputs — completely different circuit.
       Old proving_key.pk, verification_key.vk, ZKMLVerifier.sol all invalid.

  STEPS after retrain checkpoint is verified:
    1. Rebuild proxy model (knowledge distillation) for 11 outputs
       proxy: Linear(64→32) → ReLU → Linear(32→11) — 11 output nodes
       train proxy to agree with full model ≥95% per class

    2. Re-export proxy to ONNX (opset 11, same as before)

    3. Re-run EZKL pipeline:
       gen_settings → calibrate → compile_circuit →
       setup (new pk + vk) → prove → verify

    4. Re-generate ZKMLVerifier.sol from new verification key

    5. Redeploy ZKMLVerifier to Sepolia (new contract address)

    6. Update AuditRegistry to reference new ZKMLVerifier address

    7. Update agents/.env: ZKML_VERIFIER_ADDRESS=<new address>

  KEEP old circuit files in zkml/ until new circuit is verified on Sepolia
  KEEP old ZKMLVerifier deployed — don't invalidate existing audit records

══════════════════════════════════════════════════════════════════
ADDITIONAL ML IMPROVEMENTS — AFTER FIRST RETRAIN
These are separate from multi-label. Decide based on val results.
══════════════════════════════════════════════════════════════════

B-03  Sliding window tokenisation for contracts > 512 tokens
      Impact:  high — real DeFi contracts often exceed 512 tokens
               functions defined after token 512 are currently invisible
      What:    tokenize in overlapping 512-token windows (stride 256)
               run CodeBERT on each window → max-pool CLS vectors
      Cost:    W × inference time per long contract (W = num windows)
               requires retraining on same windowing strategy
               MODULE1_TIMEOUT must increase to 120+ seconds
      When:    if long-contract F1 is below target after multi-label retrain

B-04  Edge features — CALLS/READS/WRITES/EMITS/INHERITS type in graph
      Impact:  medium-high — edge type carries security signal
               WRITES after external CALL = reentrancy pattern, direct signal
      What:    build edge_attr [E, 5] one-hot tensor in preprocess.py
               _EDGE_TYPES dict already defined — currently unused
               GATConv(edge_dim=5) in gnn_encoder.py
      Cost:    rebuild all 68,555 graph .pt files + retrain
      When:    if reentrancy per-class F1 needs improvement

B-05  Global max pool experiment
      Impact:  medium — max pool better for single-vulnerable-function detection
      What:    swap global_mean_pool → global_max_pool in gnn_encoder.py
               or concat both: [mean; max] → [B, 128], update FusionLayer
      Cost:    retrain (combine with B-03 or B-04 run)
      When:    if mean pool is diluting single-function vulnerability signals

B-06  Richer node features — extend 8-dim to 11-dim
      Impact:  medium — has_external_call is the most direct reentrancy signal
      What:    add 3 dims: has_external_call, uses_tx_origin, modifier_count
               update GNNEncoder(in_channels=11)
               rebuild all training graphs
      Cost:    rebuild 68K graphs + retrain (most expensive B-tier)
      When:    only after other B-tier improvements are exhausted

C-01  Multi-contract analysis
      Impact:  high — contracts inheriting from user-defined bases currently
               miss the parent's vulnerable functions entirely
      What:    iterate all non-dependency contracts in _extract_graph()
               prefix node keys: "ContractName.function_canonical_name"
               add cross-contract INHERITS edges (user contracts only)
               Slither's is_from_dependency() prevents infinite loops
      Cost:    rebuild all training graphs + retrain
      When:    combine with B-04 (both require graph rebuild)

D-01  LoRA fine-tuning of CodeBERT
      Impact:  high for rare classes — semantic fine-tuning improves
               understanding of flash loan and oracle manipulation patterns
      What:    peft library, LoRA adapters on query/value matrices (r=8)
               ~500K trainable params added alongside frozen CodeBERT
      When:    if rare class F1 < 0.60 after multi-label retrain
               requires dataset 200K+ or targeted augmentation

══════════════════════════════════════════════════════════════════
OVERALL EXECUTION ORDER
══════════════════════════════════════════════════════════════════

  WEEK 1 (now):
    Track 1: A-01 through A-11 — all immediate fixes
    Track 2: M4.3 sentinel-audit MCP server

  WEEK 2:
    Track 2: M5 LangGraph orchestration + state machine

  WEEK 3-4:
    Track 2: M6 five agents + Pydantic AuditReport + end-to-end test

  AFTER M6 VERIFIED END-TO-END:
    Track 3 Phase 0: build_multilabel_index.py → multilabel_index.csv
    Track 3 Phase 1: DualPathDataset multi-label mode
    Track 3 Phase 2: SentinelModel num_classes parameter
    Track 3 Phase 3: trainer BCEWithLogitsLoss + pos_weight
    Track 3 Phase 4: predictor per-class output
    Track 3 Phase 5: api.py new PredictResponse
    Track 3 Phase 6: inference_server.py mock update
    Track 3 Phase 7: all tests + smoke scripts
    Track 3 Phase 8: retrain (40 epochs, ~days)
    Track 3 Phase 9: per-class threshold tuning
    Track 3 Phase 10: ZKML circuit rebuild + Sepolia redeploy

  AFTER FIRST RETRAIN — evaluate val metrics, then pick from:
    B-03 sliding window      (if long-contract accuracy is low)
    B-04 edge features       (if reentrancy F1 is low)
    B-05 max pool            (if single-function vulns are missed)
    C-01 multi-contract      (if inherited-base vulns are missed)
    Combine into ONE retrain run — never retrain for one change at a time

══════════════════════════════════════════════════════════════════
WHAT NEVER CHANGES WITHOUT FULL RETRAIN
══════════════════════════════════════════════════════════════════

  These are locked to the current checkpoint:
    GNNEncoder in_channels = 8      (feature dimension)
    GNNEncoder heads = 8            (GAT attention heads)
    TransformerEncoder TOKENIZER    "microsoft/codebert-base"
    TransformerEncoder MAX_LENGTH   512
    FusionLayer dimensions          gnn_dim=64, transformer_dim=768
    Node feature order              [type_id, vis, pure, view, payable,
                                     reentrant, complexity, loc]
    Node insertion order in graph   CONTRACT → STATE_VARs → FUNCTIONs
                                    → MODIFIERs → EVENTs

  If any of these change: all 68,555 graph .pt files must be rebuilt
  and the model retrained from scratch. The existing checkpoint becomes invalid.

══════════════════════════════════════════════════════════════════
KNOWN ISSUES TO NOT FORGET
══════════════════════════════════════════════════════════════════

  STALE:
    preprocess.py imports ASTExtractor but never uses it  (A-02)
    inference_server.py docstring says risk_score/vulnerabilities (A-03)
    inference_server.py batch comment wrong about mcp schema (A-04)

  PARKED (carried from previous handovers):
    v4.0 git tag
    Module 2 zkml milestone docs
    M3.6 Dockerfile
    GitHub Actions push (ingest.yml written, not pushed)
    CI_MODE embedder — CPU fallback for GitHub Actions
    Index rebuild with chunk_size=1536 (current index uses 512)
    DVC versioning of RAG index
    TRANSFORMERS_OFFLINE=1 + HF_TOKEN → .env
    SWC Registry + Rekt.news + Immunefi fetchers
    test_rag_server.py unit tests (A-11)
    solc-select confirmed on 0.8.20

══════════════════════════════════════════════════════════════════
PORT MAP — COMPLETE REFERENCE
══════════════════════════════════════════════════════════════════

  1234   LM Studio (Windows host, WSL2 gateway)
  3000   Dagster UI
  8000   Module 6 API gateway (M6 — not built)
  8001   Module 1 ML inference FastAPI (ACTIVE)
  8010   sentinel-inference MCP server (ACTIVE, mock=False)
  8011   sentinel-rag MCP server (ACTIVE)
  8012   sentinel-audit MCP server (M4.3 — not built)
  11434  Ollama (parked)

══════════════════════════════════════════════════════════════════
```