══════════════════════════════════════════════════════════════════
SENTINEL — FINAL ACTION PLAN
2026-04-16 | Incorporates master plan + source review + BCCC CSV audit
══════════════════════════════════════════════════════════════════

Four tracks:
  TRACK 0  Backup + file structure cleanup    do FIRST, takes ~1 hour
  TRACK 1  Immediate code fixes               no retrain, this week
  TRACK 2  Agent completion                   M4.3 → M5 → M6
  TRACK 3  Multi-label ML upgrade             after M6, full retrain

══════════════════════════════════════════════════════════════════
CURRENT SYSTEM STATE
══════════════════════════════════════════════════════════════════

  LIVE:
    Module 1    ml/src/inference/   FastAPI port 8001, GPU confirmed
    Module 2    zkml/               Sepolia: ZKMLVerifier + AuditRegistry
    Module 3    mlops/              MLflow + DVC + Dagster basic
    M4.1        inference_server.py port 8010, mock=False
    M4.2        rag_server.py       port 8011, 1339 chunks

  NOT YET BUILT:
    M4.3  audit_server.py
    M5    LangGraph orchestration
    M6    Five agents + API gateway

  MODEL (binary, production):
    Checkpoint:   run-alpha-tune_best.pt  477 MB  RTX 3070
    Output:       binary score [0,1] → "vulnerable" | "safe"
    Threshold:    0.50  (tuned on val, F1-macro criterion)
    Known limit:  41.2% of BCCC contracts are genuinely multi-label

══════════════════════════════════════════════════════════════════
KEY DATASET FINDING (informs Track 3)
══════════════════════════════════════════════════════════════════

  BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv  (111,897 rows, 290+ columns)

  This is the AUTHORITATIVE multi-label source.

  Structure:
    Col 1:    row index
    Col 2:    SHA256 hash  ← maps to .sol filenames in SourceCodes/
    Cols 3-N: 280+ numerical features (LOC, AST, opcodes, bytecode, ABI)
    Last 12:  Class01:ExternalBug through Class12:NonVulnerable  (binary 0/1)

  The 12 Class columns allow multi-label: a row can have multiple 1s.
  111,897 rows ≈ 111,177 total source files across 12 folders.
  This means ~one row per folder-file-occurrence, NOT per unique SHA256.
  → Same SHA256 in 3 folders = 3 rows, each with a different Class=1.
  → To get multi-hot per SHA256: GROUP BY SHA256, then OR the Class columns.

  Why this matters for Track 3 Phase 0:
    Instead of scanning 12 directories manually, read the CSV,
    group by SHA256, OR the Class01-Class11 columns → multi-hot [11].
    More reliable (authoritative BCCC labels) and covers all 111k contracts.

  The existing ml/src/data/bccc_dataset.py loads this CSV —
  check it before writing build_multilabel_index.py.

══════════════════════════════════════════════════════════════════
TRACK 0 — BACKUP + FILE STRUCTURE CLEANUP
Do this BEFORE touching any code.
══════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────
STEP 0-A: Create git backup tag
────────────────────────────────────────────────────────────────

  From project root:
    git add -A
    git commit -m "chore: pre-cleanup snapshot — binary baseline complete"
    git tag v1.0-binary-baseline
    git log --oneline -5   # confirm tag exists

  This preserves the full working binary system before ANY changes.
  The tag can be checked out at any point: git checkout v1.0-binary-baseline

────────────────────────────────────────────────────────────────
STEP 0-B: File structure audit — what is active vs legacy
────────────────────────────────────────────────────────────────

  ACTIVE (keep in place, no rename):
    ml/src/models/        sentinel_model.py, gnn_encoder.py,
                          transformer_encoder.py, fusion_layer.py
    ml/src/datasets/      dual_path_dataset.py
    ml/src/training/      trainer.py, focalloss.py
    ml/src/inference/     predictor.py, preprocess.py, api.py
    ml/scripts/           train.py, tune_threshold.py,
                          run_overnight_experiments.py,
                          create_splits.py, create_label_index.py
    ml/tests/             conftest.py, test_api.py
    ml/checkpoints/       run-alpha-tune_best.pt  (PRODUCTION — never delete)
                          sentinel_best.pt         (baseline reference)
                          run-more-epochs_best.pt  (experimental)
    ml/data/graphs/       68,556 .pt graph files
    ml/data/tokens/       68,570 .pt token files
    ml/data/splits/       train/val/test_indices.npy
    ml/data/processed/    contract_labels_correct.csv, label_index.csv
    agents/src/           all files (all active, Apr 11-15)
    agents/tests/         all files
    agents/scripts/       smoke_inference_mcp.py, smoke_rag_mcp.py, test_k_cap.py
    zkml/src/             all files (ezkl + distillation)
    zkml/ezkl/            proof artifacts (proof.json etc)
    contracts/            entire Foundry project
    CLAUDE.md, pyproject.toml, poetry.lock
    check_verify.sh, submit_audit.sh

  LEGACY — MOVE to ml/_archive/ and agents/_archive/ (git mv):
    ml/src/data/graphs/ast_extractor.py       (OLD 17-dim extractor, superseded)
    ml/src/data/graphs/graph_builder.py       (superseded by preprocess.py)
    ml/src/data/solidifi_dataset.py           (SolidiFI not in current pipeline)
    ml/src/data/validate_solidifi.py          (SolidiFI validation)
    ml/src/data/validate_dataset.py           (initial setup, done)
    ml/src/data/bccc_dataset.py               (KEEP if used, archive if not)
    ml/src/tools/slither_wrapper.py           (62K — superseded by preprocess.py)
    ml/src/tools/slither_wrapper_turbo.py     (unclear usage — archive)
    ml/src/validation/models_v2.py            (ad-hoc validation, done)
    ml/src/validation/statistical_validation.py
    ml/src/validation/test_full_dataset_final.py
    ml/src/validation/test_models.py, test_real_data.py, models.py
    ml/scripts/test_sentinel_model.py         (move to ml/tests/)
    ml/scripts/test_gnn_encoder.py            (move to ml/tests/ or archive)
    ml/scripts/test_fusion_layer.py           (move to ml/tests/ or archive)
    ml/scripts/test_dataloader.py             (move to ml/tests/ or archive)
    ml/scripts/test_dataset.py                (move to ml/tests/ or archive)
    ml/scripts/comprehensive_data_validation.py   (initial setup, done)
    ml/scripts/fix_labels_from_csv.py         (one-time utility, done)
    ml/scripts/analyze_token_stats.py         (analysis done)
    analysis/validation_report.json           (static artifact)
    run_full_dataset_overnight.py             (root-level, legacy)

  EXTRACTION SCRIPTS — RENAME and MOVE to ml/data_extraction/:
    ml/scripts/ast_extractor_v4_production.py → ml/data_extraction/ast_extractor.py
    ml/scripts/tokenizer_v1_production.py     → ml/data_extraction/tokenizer.py
    (These are data-generation tools, not model scripts. The "v4" and "production"
     suffixes cause confusion — the clean names make it obvious they're in data_extraction/)

  LARGE CACHE FILES in ml/data/processed/ — MOVE to ml/data/processed/_cache/:
    bccc_full_dataset_results.json     (375 MB — BCCC feature extraction)
    bccc_v0419_only.json
    bccc_v0424_only.json
    bccc_v0425_only.json
    bccc_v0426_only.json
    contracts_ml_ready_clean.parquet   (33 MB)
    contracts_ml_ready_csv.parquet     (92 MB)
    contracts_metadata.parquet         (43 MB)
    test_enriched.json, test_fix_100.json
    bccc_full_dataset_results_OLD.json.bak
    (These are pre-computed artefacts from earlier experiments.
     Moving to _cache/ makes it clear they are not active pipeline inputs.)

  DOCS — ARCHIVE legacy (move to docs/_archive/):
    "Complete Architecture.md"         (232 KB — superseded by modular docs)
    handover-2026-04-06.md             (superseded by current session docs)
    ML_FILE_INVENTORY.md               (outdated — will regenerate)
    PROJECT_TREE.md                    (will regenerate after restructure)
    QUICKSTART.md                      (restricted)
    ROADMAP.md                         (restricted)

────────────────────────────────────────────────────────────────
STEP 0-C: Proposed clean directory structure
────────────────────────────────────────────────────────────────

  sentinel/
  ├── CLAUDE.md
  ├── pyproject.toml / poetry.lock
  ├── check_verify.sh / submit_audit.sh
  │
  ├── ml/
  │   ├── src/
  │   │   ├── models/           GNNEncoder, TransformerEncoder, FusionLayer, SentinelModel
  │   │   ├── datasets/         DualPathDataset, dual_path_collate_fn
  │   │   ├── training/         trainer.py, focalloss.py
  │   │   └── inference/        predictor.py, preprocess.py, api.py
  │   ├── scripts/              train.py, tune_threshold.py,
  │   │                         run_overnight_experiments.py,
  │   │                         create_splits.py, create_label_index.py,
  │   │                         build_multilabel_index.py  (NEW Track 3)
  │   ├── data_extraction/      ast_extractor.py, tokenizer.py
  │   │                         (was: ast_extractor_v4_production.py etc.)
  │   ├── tests/                conftest.py, test_api.py,
  │   │                         test_inference_smoke.py  (NEW Track 1)
  │   ├── data/
  │   │   ├── graphs/           68,556 × .pt graph files
  │   │   ├── tokens/           68,570 × .pt token files
  │   │   ├── splits/           train/val/test_indices.npy
  │   │   └── processed/
  │   │       ├── contract_labels_correct.csv
  │   │       ├── label_index.csv
  │   │       ├── multilabel_index.csv  (NEW Track 3)
  │   │       └── _cache/       large pre-computed artifacts
  │   ├── checkpoints/          sentinel_best.pt, run-alpha-tune_best.pt,
  │   │                         run-more-epochs_best.pt
  │   ├── logs/
  │   └── _archive/             ml/src/data/, tools/, validation/,
  │                              scripts/test_*.py, legacy scripts
  │
  ├── agents/
  │   ├── src/
  │   │   ├── mcp/servers/      inference_server.py, rag_server.py,
  │   │   │                     audit_server.py  (NEW M4.3)
  │   │   ├── orchestration/    graph.py, state.py, nodes.py  (NEW M5)
  │   │   ├── agents/           five agents  (NEW M6)
  │   │   ├── rag/              retriever.py, embedder.py, etc.
  │   │   ├── ingestion/        pipeline.py, scheduler_cron.py, etc.
  │   │   └── llm/              client.py
  │   ├── scripts/              smoke scripts
  │   ├── tests/
  │   └── data/
  │
  ├── zkml/
  │   ├── src/
  │   │   ├── ezkl/             run_proof.py, setup_circuit.py, extract_calldata.py
  │   │   └── distillation/     proxy_model.py, train_proxy.py, export_onnx.py, etc.
  │   └── ezkl/                 proof artifacts, keys, settings
  │
  ├── contracts/                Foundry project (complete)
  │
  └── docs/
      ├── SENTINEL-architecture.md
      ├── SENTINEL-modules.md
      ├── SENTINEL_ACTION_PLAN_FINAL.md   ← this file
      ├── SENTINEL_FINAL_IMPROVEMENT_LIST.md
      ├── ML_ARCHITECTURE.md
      ├── ML_TRAINING.md
      ├── ML_DATASET_PIPELINE.md
      ├── ML_INFERENCE.md
      ├── ML_SCRIPTS.md
      ├── ZKML_PIPELINE.md
      ├── CONTRACTS.md
      ├── PROJECT_DOCUMENTATION.md
      ├── ENCODING_REFERENCE.md
      └── _archive/             legacy docs

────────────────────────────────────────────────────────────────
STEP 0-D: Execute the cleanup
────────────────────────────────────────────────────────────────

  Order matters — remove dead import BEFORE archiving old extractor:

  1. Fix A-02 first (remove ASTExtractor dead import from preprocess.py)
     → then ml/src/data/graphs/ast_extractor.py can safely be archived

  2. Run imports test to confirm nothing breaks:
       poetry run python -c "from ml.src.inference.preprocess import ContractPreprocessor"
     VERIFY: no ImportError

  3. Create archive directories:
       mkdir -p ml/_archive/src_data ml/_archive/src_tools ml/_archive/src_validation
       mkdir -p ml/_archive/scripts ml/data_extraction
       mkdir -p ml/data/processed/_cache
       mkdir -p docs/_archive

  4. Use git mv for all moves (preserves history):
       git mv ml/src/data/graphs/ast_extractor.py ml/_archive/src_data/
       git mv ml/src/data/graphs/graph_builder.py ml/_archive/src_data/
       git mv ml/src/tools/ ml/_archive/src_tools/
       git mv ml/src/validation/ ml/_archive/src_validation/
       git mv ml/scripts/ast_extractor_v4_production.py ml/data_extraction/ast_extractor.py
       git mv ml/scripts/tokenizer_v1_production.py ml/data_extraction/tokenizer.py
       git mv ml/scripts/test_*.py ml/_archive/scripts/
       git mv ml/scripts/comprehensive_data_validation.py ml/_archive/scripts/
       git mv ml/scripts/fix_labels_from_csv.py ml/_archive/scripts/
       git mv ml/scripts/analyze_token_stats.py ml/_archive/scripts/
       git mv ml/data/processed/bccc_*.json ml/data/processed/_cache/
       git mv ml/data/processed/contracts_*.parquet ml/data/processed/_cache/
       git mv ml/data/processed/*.parquet ml/data/processed/_cache/
       git mv ml/data/processed/*.bak ml/data/processed/_cache/
       git mv "docs/Complete Architecture.md" docs/_archive/
       git mv docs/handover-2026-04-06.md docs/_archive/
       git mv docs/ML_FILE_INVENTORY.md docs/_archive/
       git mv docs/PROJECT_TREE.md docs/_archive/
       git mv run_full_dataset_overnight.py ml/_archive/scripts/

  5. Write a README.md inside ml/_archive/:
       # Archive
       Files moved here are no longer part of the active pipeline.
       They are preserved in git history under their original paths.
       See: git log --follow -- ml/_archive/<filename>

  6. VERIFY everything still imports cleanly:
       poetry run python -c "from ml.src.models.sentinel_model import SentinelModel; print('OK')"
       poetry run python -c "from ml.src.datasets.dual_path_dataset import DualPathDataset; print('OK')"
       poetry run python -c "from ml.src.inference.predictor import Predictor; print('OK')"
       poetry run python -c "from ml.src.inference.preprocess import ContractPreprocessor; print('OK')"
     All must print "OK" with no errors.

  7. Git commit the cleanup:
       git add -A
       git commit -m "chore: file structure cleanup — archive legacy, rename extraction scripts"
       git tag v1.0-clean-structure

────────────────────────────────────────────────────────────────
STEP 0-E: Tag milestones going forward
────────────────────────────────────────────────────────────────

  v1.0-binary-baseline      created in 0-A (before any changes)
  v1.0-clean-structure      created in 0-D (after cleanup)
  v1.1-code-fixes           create after Track 1 complete
  v1.2-m4.3-complete        create after M4.3 complete
  v1.3-m5-complete          create after M5 complete
  v2.0-m6-complete          create after M6 end-to-end verified
  v2.1-multilabel-data      create after Phase 0 multilabel_index.csv verified
  v3.0-multilabel-retrain   create after Track 3 retrain checkpoint verified

══════════════════════════════════════════════════════════════════
TRACK 1 — IMMEDIATE CODE FIXES
After Track 0. No retrain. No architecture change.
══════════════════════════════════════════════════════════════════

Format for each item:
  WHAT:   what to change
  WHY:    why it matters
  VERIFY: command to confirm it works

──────────────────────────────────────────────────────────────────
CRITICAL — fix before any further deployment
──────────────────────────────────────────────────────────────────

A-12  run_proof.py — checkpoint format incompatibility
      File: zkml/src/ezkl/run_proof.py  lines 135-146
      WHAT: current code:
              state_dict = torch.load(TEACHER_CHECKPOINT, weights_only=True)
              teacher.load_state_dict(state_dict)
            new checkpoints are dicts: {"model": state_dict, "optimizer": ...}
            load_state_dict() receives a dict → RuntimeError at runtime
            Fix:
              ckpt = torch.load(path, weights_only=False)
              state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
              model.load_state_dict(state_dict)
            Apply to BOTH teacher and proxy checkpoint loads.
      WHY:  silently breaks ZKML proof generation on first retrain
      VERIFY:
        poetry run python -c "
        import torch
        ckpt = torch.load('ml/checkpoints/run-alpha-tune_best.pt', weights_only=False)
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        print('format:', 'new dict' if isinstance(ckpt, dict) else 'legacy')
        print('keys:', list(state_dict.keys())[:3], '...')
        "
        Expected: prints "format: new dict" and first 3 weight keys

A-13  setup_circuit.py — assert statements stripped by -O
      File: zkml/src/ezkl/setup_circuit.py  lines 119, 152, 207
      WHAT: replace each assert with explicit check:
              assert res, "gen_settings failed..."
            →
              if not res:
                  raise RuntimeError("gen_settings failed — check ONNX opset 11")
            Same for lines 152 and 207.
      WHY:  python -O (common in Docker) strips all assert statements.
            A failed EZKL step passes silently → corrupt circuit built on top.
      VERIFY:
        python -O -c "
        import ast, pathlib
        src = pathlib.Path('zkml/src/ezkl/setup_circuit.py').read_text()
        assert 'assert res' not in src, 'assert still present'
        print('OK — no bare assert statements')
        "

──────────────────────────────────────────────────────────────────
HIGH
──────────────────────────────────────────────────────────────────

A-02  preprocess.py — remove dead ASTExtractor import
      File: ml/src/inference/preprocess.py  line 77
      WHAT: delete: from ml.src.data.graphs.ast_extractor import ASTExtractor
            also remove the comment on lines 75-76 claiming "used by process()"
      WHY:  ASTExtractor is NEVER instantiated anywhere in the file.
            It creates a false dependency on the old 17-dim extractor.
            After Track 0, ml/src/data/graphs/ast_extractor.py moves to _archive —
            this import would break unless removed first.
      VERIFY:
        poetry run python -c "from ml.src.inference.preprocess import ContractPreprocessor; print('OK')"

A-14  embedder.py — embed_query() has no retry
      File: agents/src/rag/embedder.py
      WHAT: embed_chunks() has exponential backoff (3 attempts, 2^n seconds).
            embed_query() has none — one LM Studio timeout kills every RAG search.
            Apply the same retry pattern to embed_query().
      WHY:  embed_query() is called on EVERY search; embed_chunks() only on ingest.
      VERIFY:
        python -c "
        import inspect
        from agents.src.rag.embedder import Embedder
        src = inspect.getsource(Embedder.embed_query)
        assert 'retry' in src.lower() or 'attempt' in src.lower(), 'no retry found'
        print('OK — retry logic present in embed_query')
        "

A-15  rag_server.py — relative import breaks outside agents context
      File: agents/src/mcp/servers/rag_server.py
      WHAT: change:
              from src.rag.retriever import HybridRetriever
            to:
              import sys, pathlib
              sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
              from agents.src.rag.retriever import HybridRetriever
      WHY:  fails with ModuleNotFoundError when started from project root or Docker
      VERIFY:
        cd ~/projects/sentinel
        poetry run python -c "import agents.src.mcp.servers.rag_server; print('OK')"

A-16  pipeline.py — FileLock not released on exception
      File: agents/src/ingestion/pipeline.py
      WHAT: ensure _write_index() uses "with FileLock(...)" context manager,
            not manual acquire/release calls that skip release on exception
      WHY:  if write fails (disk full etc), lock is never released.
            All subsequent ingestion runs fail with LockTimeout until manual fix.
      VERIFY:
        python -c "
        import inspect
        from agents.src.ingestion.pipeline import IngestionPipeline
        src = inspect.getsource(IngestionPipeline._write_index)
        assert 'with FileLock' in src or 'with filelock' in src.lower(), 'not using context manager'
        print('OK — FileLock used as context manager')
        "

──────────────────────────────────────────────────────────────────
STANDARD — same week
──────────────────────────────────────────────────────────────────

A-01  predictor.py — defensive model.eval() in _score()
      File: ml/src/inference/predictor.py
      WHAT: add self.model.eval() at start of _score() before torch.no_grad()
      VERIFY:
        poetry run python ml/tests/test_api.py::test_predict_consistent_on_same_input
        (should already pass — confirms eval mode enforced)

A-03  inference_server.py — stale docstring in _call_inference_api
      File: agents/src/mcp/servers/inference_server.py
      WHAT: docstring says "risk_score" + "vulnerabilities list"
            correct to: label, confidence, threshold, truncated, num_nodes, num_edges
      VERIFY: read — confirm docstring matches PredictResponse fields in api.py

A-04  inference_server.py — stale batch comment
      File: agents/src/mcp/servers/inference_server.py
      WHAT: comment "schema validation is advisory in mcp 1.x" is wrong for 1.27.0
      VERIFY: remove or correct

A-05  inference_server.py — handle_sse return type hint
      File: agents/src/mcp/servers/inference_server.py
      WHAT: declared -> None but returns Response() → fix to -> Response

A-06  predictor.py — call parameter_summary() at startup
      File: ml/src/inference/predictor.py
      WHAT: add self.model.parameter_summary() after load_state_dict() in __init__
      VERIFY:
        poetry run python -c "
        from ml.src.inference.predictor import Predictor
        p = Predictor('ml/checkpoints/run-alpha-tune_best.pt')
        " 2>&1 | grep -i "trainable\|frozen"
        Expected: prints trainable: 239,041 / frozen: 124,645,632

A-07  api.py — add inference timeout + thread safety
      File: ml/src/inference/api.py
      WHAT:
        result = await asyncio.wait_for(
            asyncio.to_thread(predictor.predict_source, body.source_code),
            timeout=float(os.getenv("INFERENCE_TIMEOUT", "60.0")),
        )
        except asyncio.TimeoutError → HTTP 504
      WHY: Slither hangs on pathological Solidity without this
      VERIFY:
        # Start api server, send minimal contract, confirm 200 response
        poetry run uvicorn ml.src.inference.api:app --port 8001 &
        curl -s -X POST http://localhost:8001/predict \
          -H "Content-Type: application/json" \
          -d '{"source_code": "pragma solidity ^0.8.0; contract T { }"}' | python -m json.tool
        Expected: JSON with label, confidence, threshold fields

A-08  ml/tests/ — add inference smoke test
      File: ml/tests/test_inference_smoke.py  (new)
      WHAT: single test — process_source() on minimal Solidity:
              assert graph.x.shape[1] == 8
              assert tokens["input_ids"].shape == (1, 512)
      WHY:  catches 17-dim vs 8-dim mismatch from old ASTExtractor at CI time
      VERIFY:
        poetry run pytest ml/tests/test_inference_smoke.py -v
        Expected: 1 passed

A-09  rag_server.py — full query DEBUG log
      File: agents/src/mcp/servers/rag_server.py
      WHAT: logger.debug("search | full_query='{}'", query) before INFO log

A-10  agents/.env — add MODULE1_TIMEOUT=120
      File: agents/.env
      WHAT: add MODULE1_TIMEOUT=120

A-11  agents/tests/ — write unit tests for rag_server
      File: agents/tests/test_rag_server.py  (new)
      WHAT: list_tools schema, _handle_search with mock retriever,
            k cap, filters passthrough, unknown tool → TextContent,
            retriever error → structured TextContent not exception
      VERIFY:
        cd ~/projects/sentinel
        poetry run pytest agents/tests/test_rag_server.py -v
        Expected: all tests pass

────────────────────────────────────────────────────────────────
After all Track 1 items complete:
  git add -A
  git commit -m "fix: all Track 1 code fixes — critical, high, standard"
  git tag v1.1-code-fixes
────────────────────────────────────────────────────────────────

══════════════════════════════════════════════════════════════════
TRACK 2 — AGENT COMPLETION
M4.3 → M5 → M6
══════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────
M4.3 — sentinel-audit MCP server
────────────────────────────────────────────────────────────────

  File:     agents/src/mcp/servers/audit_server.py
  Port:     8012
  Pattern:  same SSE wiring as inference_server.py and rag_server.py

  PREREQS — verify before starting:
    grep SEPOLIA_RPC_URL agents/.env   → must be set
    find zkml/ -name "AuditRegistry*"  → confirm ABI JSON location
    cast call 0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf \
      "getLatestAudit(address)" 0x0000000000000000000000000000000000000001 \
      --rpc-url $SEPOLIA_RPC_URL        → confirm RPC works

  TOOLS — read-only first, write tool after RPC confirmed:

    get_latest_audit(contract_address: str)
      → AuditRegistry.getLatestAudit(contractAddress)
      → returns: {score, proofHash, timestamp, agent, verified}

    get_audit_history(contract_address: str, limit: int = 10)
      → AuditRegistry.getAuditHistory(contractAddress)
      → returns: list of AuditResult structs

    submit_audit(contract_address, score, zk_proof, public_signals)  [AFTER TRACK 3]
      → implement only after Track 3 ZKML rebuild is complete
      → requires valid ZK proof + MIN_STAKE (1000 SENTINEL)

  CONTRACT:   0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf (Sepolia proxy)
  WEB3:       web3 ^7.15.0 already in agents/pyproject.toml

  ENV VARS to add to agents/.env:
    MCP_AUDIT_PORT=8012
    SEPOLIA_RPC_URL=<your rpc url>
    AUDIT_REGISTRY_ADDRESS=0x14E5eFb6DE4cBb74896B45b4853fd14901E4CfAf

  DELIVERABLES:
    agents/src/mcp/servers/audit_server.py
    agents/scripts/smoke_audit_mcp.py
    agents/tests/test_audit_server.py

  VERIFY:
    # Start server
    cd ~/projects/sentinel && poetry run python -m agents.src.mcp.servers.audit_server &
    # Run smoke test
    poetry run python agents/scripts/smoke_audit_mcp.py
    # Run unit tests
    poetry run pytest agents/tests/test_audit_server.py -v
    Expected: all pass, server responds on port 8012

  git tag v1.2-m4.3-complete

────────────────────────────────────────────────────────────────
M5 — LangGraph orchestration
────────────────────────────────────────────────────────────────

  Files:
    agents/src/orchestration/graph.py
    agents/src/orchestration/state.py
    agents/src/orchestration/nodes.py

  MCP CLIENT CONFIG:
    "sentinel-inference": {"url": "http://localhost:8010/sse"}
    "sentinel-rag":       {"url": "http://localhost:8011/sse"}
    "sentinel-audit":     {"url": "http://localhost:8012/sse"}

  STATE SCHEMA (TypedDict):
    contract_code:    str
    contract_address: str
    ml_result:        dict | None
    rag_results:      list | None
    audit_history:    list | None
    static_findings:  dict | None
    final_report:     dict | None
    error:            str | None

  GRAPH NODES:
    ml_assessment   → predict tool
    rag_research    → search tool (query from ml_result)
    audit_check     → get_audit_history tool
    static_analysis → Slither direct (no MCP)
    synthesizer     → assembles final report

  CONDITIONAL ROUTING:
    ml_assessment → route based on score:
      BINARY PHASE:  if confidence > 0.70 → rag + static → synthesizer
                     if confidence ≤ 0.70 → synthesizer (fast path)

    NOTE — routing after Track 3 multi-label upgrade:
      Replace "confidence > 0.70" check with a helper function:
        def _is_high_risk(ml_result: dict) -> bool:
            # Binary phase: uses "confidence" field
            # Multi-label phase: uses max(v["probability"] for v in vulnerabilities)
            ...
      Design this as a helper NOW so Track 3 is a one-line swap.

  CHECKPOINTING:
    LangGraph MemorySaver or Redis — checkpoint after each node
    MCP restart resumes from last checkpoint

  DELIVERABLES:
    agents/src/orchestration/graph.py
    agents/src/orchestration/state.py
    agents/src/orchestration/nodes.py
    agents/tests/test_graph_routing.py
    agents/scripts/smoke_langgraph.py

  VERIFY:
    poetry run python agents/scripts/smoke_langgraph.py \
      --contract contracts/test/fixtures/vulnerable_reentrancy.sol
    Expected: full audit report JSON printed, all 4-5 nodes executed
    Check MLflow or logs to confirm all nodes ran

  git tag v1.3-m5-complete

────────────────────────────────────────────────────────────────
M6 — Five agents + full integration
────────────────────────────────────────────────────────────────

  AGENT FILES:
    agents/src/agents/static_analyzer.py  Slither + Mythril
    agents/src/agents/ml_intelligence.py  inference MCP
    agents/src/agents/rag_researcher.py   rag MCP
    agents/src/agents/code_logic.py       AST + access control
    agents/src/agents/synthesizer.py      AuditReport

  PYDANTIC AUDIT REPORT (binary phase):
    class AuditReport(BaseModel):
        contract_address: str
        overall_label:    str           "vulnerable" | "safe"
        confidence:       float         raw binary score [0,1]
        vulnerabilities:  list[VulnFinding]
        rag_evidence:     list[ExploitReference]
        static_findings:  list[StaticFinding]
        on_chain_history: list[AuditRecord]
        recommendation:   str
        timestamp:        datetime
        model_version:    str

    NOTE — after Track 3: confidence → max(vuln probabilities) across 11 classes

  BATCH ENDPOINT (C-02):
    Add POST /batch_predict to api.py
    Predictor.predict_batch() → single forward pass for up to 20 contracts

  END-TO-END VERIFY:
    Select a known vulnerable DeFi contract from DeFiHackLabs corpus
    # 1. Start all three MCP servers (ports 8010, 8011, 8012)
    # 2. Run full audit via LangGraph
    poetry run python agents/scripts/smoke_full_audit.py \
      --address <known_vulnerable_contract_on_sepolia>
    # 3. Verify audit report JSON has all required fields
    # 4. Check Sepolia: AuditRegistry.getLatestAudit(address) returns the proof
    Expected: complete AuditReport, proof hash on Sepolia, all MCP calls logged

  git tag v2.0-m6-complete

══════════════════════════════════════════════════════════════════
TRACK 3 — MULTI-LABEL ML UPGRADE
Start AFTER v2.0-m6-complete tag. Full retrain required.
══════════════════════════════════════════════════════════════════

Background:
  Current model: binary output — collapses 11 vulnerability class structure
  BCCC dataset:  41.2% of contracts appear in multiple vulnerability folders
  Goal:          "Reentrancy: 0.81, IntegerUO: 0.23" instead of "vulnerable: 0.75"
  Unchanged:     all 68,555 graph/token .pt files — labels loaded externally

THE TWO HASH SYSTEMS (never mix):
  BCCC SHA256:   hash of file contents → BCCC CSV col 2, BCCC .sol filename
  Internal MD5:  hash of file path     → .pt filename in ml/data/graphs/
  Bridge:        graph.contract_path inside .pt → Path(...).stem = SHA256

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

"Safe" = all 11 probabilities below threshold. No explicit "safe" output node.

────────────────────────────────────────────────────────────────
PHASE 0 — Build multilabel_index.csv
────────────────────────────────────────────────────────────────

  New file:  ml/scripts/build_multilabel_index.py
  Output:    ml/data/processed/multilabel_index.csv
  Columns:   md5_stem, CallToUnknown, DenialOfService, ExternalBug,
             GasException, IntegerUO, MishandledException, Reentrancy,
             Timestamp, TransactionOrderDependence, UnusedReturn, WeakAccessMod
  Rows:      68,555

  LABEL SOURCE — BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv:
    This is the authoritative multi-label ground truth.
    It has 111,897 rows: one per folder-file-occurrence (not per unique SHA256).
    Same SHA256 in 3 folders = 3 rows in CSV, each with a different Class=1.
    → Group by SHA256 and OR the Class01-Class11 columns → multi-hot [11].

  BEFORE writing the script — check bccc_dataset.py:
    ml/src/data/bccc_dataset.py may already load the BCCC CSV.
    Read it and reuse any existing CSV loading logic.
    Also check contracts_ml_ready_clean.parquet (33 MB) — it may already
    have pre-grouped multi-hot labels from an earlier preprocessing run.

  ALGORITHM:

    Step 1 — Load BCCC CSV and build sha256 → multi-hot [11] mapping:
      CLASS_NAMES = [
          "CallToUnknown", "DenialOfService", "ExternalBug", "GasException",
          "IntegerUO", "MishandledException", "Reentrancy", "Timestamp",
          "TransactionOrderDependence", "UnusedReturn", "WeakAccessMod",
      ]
      CLASS_COLS = [
          "Class08:CallToUnknown", "Class09:DenialOfService", "Class01:ExternalBug",
          "Class02:GasException",  "Class10:IntegerUO",       "Class03:MishandledException",
          "Class11:Reentrancy",    "Class04:Timestamp",       "Class05:TransactionOrderDependence",
          "Class06:UnusedReturn",  "Class07:WeakAccessMod",
      ]  # note: indices must match CLASS_NAMES order exactly

      bccc_df = pd.read_csv("BCCC-SCsVul-2024/BCCC-SCsVul-2024.csv")
      sha256_col = bccc_df.columns[1]   # second column = SHA256 hash

      # Group by SHA256, OR across rows for each class column
      multi_hot = (
          bccc_df.groupby(sha256_col)[CLASS_COLS]
          .max()                        # max(0,1) == OR for binary columns
          .reset_index()
          .rename(columns={sha256_col: "sha256"})
      )
      # Result: sha256 → multi-hot [11] (one row per unique SHA256, ~68k rows)

    Step 2 — Map each .pt file to its SHA256 and look up multi-hot:
      rows = []
      unknown_count = 0

      for pt_file in sorted(Path("ml/data/graphs").glob("*.pt")):
          md5_stem = pt_file.stem
          graph = torch.load(pt_file, weights_only=False)
          sha256 = Path(graph.contract_path).stem

          if sha256 in multi_hot_lookup:
              row = [md5_stem] + multi_hot_lookup[sha256]
          else:
              # SHA256 not in BCCC (very rare — non-BCCC source or processing artifact)
              # Use binary label as fallback: graph.y == 0 → all zeros (safe)
              # graph.y == 1 → unknown class, treat as all zeros and log WARNING
              logger.warning("sha256 not in BCCC | md5=%s sha256=%s y=%d",
                             md5_stem, sha256, graph.y.item())
              row = [md5_stem] + [0] * 11
              unknown_count += 1

          rows.append(row)

    Step 3 — Write CSV and print summary:
      df = pd.DataFrame(rows, columns=["md5_stem"] + CLASS_NAMES)
      df.to_csv("ml/data/processed/multilabel_index.csv", index=False)

      print(f"Total rows: {len(df)}")
      print(f"Multi-label rows (sum > 1): {(df[CLASS_NAMES].sum(axis=1) > 1).sum()}")
      print(f"Unknown (not in BCCC): {unknown_count}")
      print("\nPer-class positive count (use for pos_weight):")
      for c in CLASS_NAMES:
          print(f"  {c}: {df[c].sum()}")

  VERIFY:
    poetry run python ml/scripts/build_multilabel_index.py
    Expected output:
      Total rows: 68,555
      Multi-label rows: ~28,000+ (>40%)
      Per-class positives printed (save these numbers for trainer pos_weight)

  git tag v2.1-multilabel-data

────────────────────────────────────────────────────────────────
PHASE 1 — Update DualPathDataset
────────────────────────────────────────────────────────────────

  File: ml/src/datasets/dual_path_dataset.py

  WHAT:
    __init__: add label_csv: Path | None = None
              if label_csv: load as {md5_stem: tensor([11], float32)}
              if None: fall back to graph.y binary mode (backward compat)

    __getitem__: when multilabel mode:
                 label = label_map[md5_stem]  → tensor [11] float32
                 NOT graph.y, NOT squeezed to scalar

    dual_path_collate_fn:
                 stack [11] tensors → [B, 11] float — no squeeze(1)

  VERIFY:
    poetry run python -c "
    from pathlib import Path
    from ml.src.datasets.dual_path_dataset import DualPathDataset
    ds = DualPathDataset(
        graphs_dir=Path('ml/data/graphs'),
        tokens_dir=Path('ml/data/tokens'),
        indices_path=Path('ml/data/splits/val_indices.npy'),
        label_csv=Path('ml/data/processed/multilabel_index.csv'),
    )
    graph, tokens, label = ds[0]
    assert label.shape == (11,), f'expected (11,) got {label.shape}'
    assert label.dtype == __import__('torch').float32, f'expected float32 got {label.dtype}'
    print('OK — label shape:', label.shape, 'dtype:', label.dtype)
    "

────────────────────────────────────────────────────────────────
PHASE 2 — Update SentinelModel
────────────────────────────────────────────────────────────────

  File: ml/src/models/sentinel_model.py

  WHAT:
    __init__: add num_classes: int = 1
    classifier:
      OLD: nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
      NEW: nn.Linear(fusion_output_dim, num_classes)   # NO Sigmoid
    forward():
      OLD: scores = self.classifier(fused).squeeze(1)  → [B]
      NEW: scores = self.classifier(fused)              → [B, num_classes]
           if self.num_classes == 1: scores = scores.squeeze(1)  # compat

  VERIFY:
    poetry run python -c "
    import torch
    from ml.src.models.sentinel_model import SentinelModel
    # Multi-label mode
    m = SentinelModel(num_classes=11)
    out = m(
        __import__('torch_geometric.data', fromlist=['Batch']).Batch(),  # dummy
        torch.zeros(2, 512, dtype=torch.long),
        torch.zeros(2, 512, dtype=torch.long),
    )
    " 2>&1 || echo "(forward needs graph Batch — shape test via trainer instead)"

    # Simpler shape check:
    poetry run python -c "
    from ml.src.models.sentinel_model import SentinelModel
    m = SentinelModel(num_classes=11)
    print('classifier:', m.classifier)
    import torch
    dummy = torch.zeros(2, 64)
    out = m.classifier(dummy)
    assert out.shape == (2, 11), f'expected (2,11) got {out.shape}'
    print('OK — classifier output shape:', out.shape)
    "

────────────────────────────────────────────────────────────────
PHASE 3 — Update trainer.py
────────────────────────────────────────────────────────────────

  File: ml/src/training/trainer.py

  WHAT:
    TrainConfig:
      ADD:    num_classes: int = 11
              label_csv: str = "ml/data/processed/multilabel_index.csv"
      REMOVE: focal_gamma, focal_alpha

    pos_weight (compute once at startup from TRAINING split labels only):
      labels = load all training labels from multilabel_index.csv
      For each class c in range(11):
        pos = labels[:, c].sum()
        neg = len(labels) - pos
        pos_weight[c] = neg / pos
      pos_weight_tensor = torch.tensor([...], device=device)   # [11]
      Print pos_weight values → log to MLflow as config

    Loss: nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    train_one_epoch():
      labels: [B, 11] float32 — pass directly, no .float() or .view(-1)

    evaluate():
      probs = torch.sigmoid(logits)              → [B, 11]
      preds = (probs >= threshold).long()        → [B, 11]
      Metrics to log:
        val_f1_macro    = f1_score(y_true, preds, average='macro')  ← PRIMARY
        val_f1_micro    = f1_score(y_true, preds, average='micro')
        val_hamming     = hamming_loss(y_true, preds)
        val_f1_{class}  for all 11 classes (separate MLflow log per class)
      Primary checkpoint metric: val_f1_macro

    Checkpoint:
      {"model": state_dict, "optimizer": ..., "epoch": ...,
       "best_f1": ..., "config": {..., "num_classes": 11, "class_names": CLASS_NAMES}}

    MLflow experiment: "sentinel-multilabel"

  VERIFY (dry run — 1 batch, no GPU required):
    poetry run python -c "
    from ml.src.training.trainer import TrainConfig, train
    cfg = TrainConfig(
        num_classes=11,
        label_csv='ml/data/processed/multilabel_index.csv',
        epochs=1,
        max_batches_per_epoch=2,   # add this param or mock
    )
    print('TrainConfig OK:', cfg.num_classes, 'classes')
    "

────────────────────────────────────────────────────────────────
PHASE 4 — Update predictor.py
────────────────────────────────────────────────────────────────

  File: ml/src/inference/predictor.py

  MODULE-LEVEL CONSTANT (must match training vector order exactly):
    CLASS_NAMES = [
        "CallToUnknown", "DenialOfService", "ExternalBug",
        "GasException", "IntegerUO", "MishandledException",
        "Reentrancy", "Timestamp", "TransactionOrderDependence",
        "UnusedReturn", "WeakAccessMod",
    ]

  __init__:
    Load num_classes from ckpt["config"]["num_classes"] (default 11)
    Instantiate SentinelModel(num_classes=num_classes)

  _score():
    scores = self.model(batch, input_ids, attention_mask)   → [1, 11] logits
    probs  = torch.sigmoid(scores).squeeze(0)               → [11] float
    vulnerabilities = sorted(
        [{"class": CLASS_NAMES[i], "probability": round(p, 4)}
         for i, p in enumerate(probs.cpu().tolist())
         if p >= self.threshold],
        key=lambda x: x["probability"],
        reverse=True,
    )
    return {
        "label":           "vulnerable" if vulnerabilities else "safe",
        "vulnerabilities": vulnerabilities,
        "threshold":       self.threshold,
        "truncated":       tokens["truncated"],
        "num_nodes":       int(graph.num_nodes),
        "num_edges":       int(graph.num_edges),
    }

  VERIFY (requires retrained checkpoint — run after Phase 8):
    poetry run python -c "
    from ml.src.inference.predictor import Predictor
    p = Predictor('ml/checkpoints/multilabel_best.pt')
    result = p.predict_source('pragma solidity ^0.8.0; contract T { function withdraw() external { msg.sender.call{value: address(this).balance}(\"\"); } }')
    assert 'vulnerabilities' in result
    assert isinstance(result['vulnerabilities'], list)
    print('OK — result:', result['label'], result['vulnerabilities'][:2])
    "

────────────────────────────────────────────────────────────────
PHASE 5 — Update api.py
────────────────────────────────────────────────────────────────

  File: ml/src/inference/api.py

  New model:
    class VulnerabilityResult(BaseModel):
        vulnerability_class: str
        probability: float = Field(..., ge=0.0, le=1.0)

  Updated PredictResponse:
    class PredictResponse(BaseModel):
        label:           str                          "safe" or "vulnerable"
        vulnerabilities: list[VulnerabilityResult]    sorted desc by probability
        threshold:       float
        truncated:       bool
        num_nodes:       int
        num_edges:       int
    REMOVE: confidence: float

  /predict: no logic change

  VERIFY:
    poetry run pytest ml/tests/test_api.py -v
    Expected: all tests pass with new schema

────────────────────────────────────────────────────────────────
PHASE 6 — Update inference_server.py
────────────────────────────────────────────────────────────────

  File: agents/src/mcp/servers/inference_server.py

  _mock_prediction(): return multi-label schema:
    {
      "label":           "vulnerable" | "safe",
      "vulnerabilities": [{"vulnerability_class": "Reentrancy",
                            "probability": 0.72}],  # or [] for safe
      "threshold":       0.50,
      "truncated":       False,
      "num_nodes":       42,
      "num_edges":       58,
    }
    REMOVE: "mock": True key

  VERIFY:
    poetry run pytest agents/tests/test_inference_server.py -v
    Expected: all tests pass

────────────────────────────────────────────────────────────────
PHASE 7 — Update all tests and smoke scripts
────────────────────────────────────────────────────────────────

  ml/tests/test_api.py:
    assert "vulnerabilities" in response (list)
    each item: "vulnerability_class" (str) + "probability" (float [0,1])
    assert "confidence" NOT in response
    safe contract → label=="safe" AND vulnerabilities==[]
    vuln contract → label=="vulnerable" AND vulnerabilities non-empty

  agents/tests/test_inference_server.py:
    update _mock_prediction test: new schema, assert no "mock" key
    update _handle_predict test: validate new response structure

  agents/scripts/smoke_inference_mcp.py:
    replace assert "confidence" with assert "vulnerabilities"

  VERIFY:
    poetry run pytest ml/tests/ agents/tests/ -v
    Expected: all tests pass

────────────────────────────────────────────────────────────────
PHASE 8 — Retrain
────────────────────────────────────────────────────────────────

  Command:
    cd ~/projects/sentinel
    nohup poetry run python ml/scripts/train.py > ml/logs/multilabel_train.log 2>&1 &

  TrainConfig:
    num_classes = 11
    label_csv   = "ml/data/processed/multilabel_index.csv"
    lr          = 3e-4
    epochs      = 40
    batch_size  = 32

  DO NOT CHANGE:
    GNNEncoder       (3-layer GAT, in_channels=8, heads=8)
    TransformerEncoder (frozen CodeBERT, max_length=512)
    FusionLayer      (832→256→64 MLP)
    Graph .pt files  (68,555 files untouched)
    Token .pt files  (untouched)
    Split indices    (position arrays unchanged)

  Monitor:
    tail -50 ml/logs/multilabel_train.log
    poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db  → http://localhost:5000

  Expected metrics:
    Baseline target:  val_f1_macro > 0.50
    Aim:              val_f1_macro > 0.65
    Note:             WeakAccessMod and Timestamp will likely be the hardest classes
                      micro-F1 will be higher (dominated by IntegerUO)

  VERIFY (after training completes):
    # Check best checkpoint exists
    ls -lh ml/checkpoints/multilabel_best.pt
    # Check MLflow for val_f1_macro curve
    poetry run mlflow ui  → experiment "sentinel-multilabel"
    # Quick inference test (Phase 4 VERIFY block above)

  git tag v3.0-multilabel-retrain

────────────────────────────────────────────────────────────────
PHASE 9 — Per-class threshold tuning
────────────────────────────────────────────────────────────────

  File: ml/scripts/tune_threshold.py  (update existing)

  Sweep thresholds on val set.
  Try both single shared threshold and per-class thresholds.
  Report: per-class F1 at each threshold + macro-F1.
  Pick: whichever maximises macro-F1.

  VERIFY:
    poetry run python ml/scripts/tune_threshold.py \
      --checkpoint ml/checkpoints/multilabel_best.pt \
      --label-csv ml/data/processed/multilabel_index.csv
    Expected: table of thresholds × F1 per class, plus recommended threshold(s)

────────────────────────────────────────────────────────────────
PHASE 10 — Rebuild ZKML circuit
────────────────────────────────────────────────────────────────

  WHY: existing EZKL circuit was compiled from 1-output binary model.
       Multi-label model has 11 outputs — completely different circuit.
       Old proving_key.pk, verification_key.vk, ZKMLVerifier.sol: all invalid.

  PREREQS: A-12 and A-13 already fixed in Track 1

  STEPS:
    1. Rebuild proxy model for 11 outputs:
         proxy: Linear(64→32) → ReLU → Linear(32→11)
         train to agree with full model ≥95% per class
         File: zkml/src/distillation/proxy_model.py (update num_classes)

    2. Re-export proxy to ONNX (opset 11, same pipeline):
         poetry run python zkml/src/distillation/export_onnx.py

    3. Re-run EZKL pipeline with fixed code (no asserts, dict checkpoint compat):
         poetry run python zkml/src/ezkl/setup_circuit.py
         poetry run python zkml/src/ezkl/run_proof.py

    4. Re-generate ZKMLVerifier.sol from new verification key

    5. Redeploy ZKMLVerifier to Sepolia (new address)

    6. Update AuditRegistry to reference new verifier address

    7. Update agents/.env:
         ZKML_VERIFIER_ADDRESS=<new_address>

  Keep old ZKMLVerifier deployed — do not invalidate existing audit records.

  VERIFY:
    bash check_verify.sh   # existing script — update addresses first
    Expected: "proof verified successfully" on Sepolia

══════════════════════════════════════════════════════════════════
POST-RETRAIN IMPROVEMENTS (pick based on val results)
══════════════════════════════════════════════════════════════════

Combine chosen improvements into ONE retraining run.
Never run separate retrains for one change at a time.

B-03  Sliding window for contracts > 512 tokens
      If: long-contract class F1 is below target
      What: overlapping 512-token windows, stride 256, max-pool CLS vectors
      Cost: rebuild inference path + retrain

B-04  Edge type features in graph
      If: Reentrancy class F1 needs improvement
      What: edge_attr [E, 5] one-hot — _EDGE_TYPES dict already in preprocess.py
            GATConv(edge_dim=5) in gnn_encoder.py
      Cost: rebuild all 68K .pt graphs + retrain

B-05  Global max pool
      If: single-function vulnerability signals are being diluted
      What: swap global_mean_pool → global_max_pool
            or concat [mean; max] → [B, 128], update FusionLayer
      Cost: retrain (no graph rebuild needed)

B-06  Richer node features 8-dim → 11-dim
      If: overall F1 plateau reached
      What: add has_external_call, uses_tx_origin, modifier_count
            GATConv(in_channels=11)
      Cost: rebuild all 68K .pt graphs + retrain

C-01  Multi-contract analysis
      If: inherited-base vulnerability misses are significant
      What: iterate all non-dependency contracts in _extract_graph()
            prefix node keys with contract name, add cross-contract INHERITS edges
      Cost: rebuild all 68K .pt graphs + retrain

D-01  LoRA fine-tuning of CodeBERT
      If: rare class F1 < 0.60 and data > 200K contracts
      What: peft library, LoRA r=8 on query/value (~500K trainable params)

══════════════════════════════════════════════════════════════════
OVERALL EXECUTION ORDER WITH TAGS
══════════════════════════════════════════════════════════════════

  TODAY:
    Track 0-A: git tag v1.0-binary-baseline
    Track 1-A12, A13: CRITICAL fixes
    Track 1-A02, A14, A15, A16: HIGH fixes
    Track 0-B,C,D: file structure cleanup
    git tag v1.0-clean-structure

  THIS WEEK:
    Track 1 A-01, A-03 through A-11: remaining fixes
    git tag v1.1-code-fixes

  WEEK 1-2:
    M4.3 audit MCP server
    git tag v1.2-m4.3-complete

  WEEK 2-3:
    M5 LangGraph orchestration
    git tag v1.3-m5-complete

  WEEK 3-4:
    M6 five agents + AuditReport + end-to-end test
    git tag v2.0-m6-complete

  AFTER M6 VERIFIED:
    Track 3 Phase 0: build multilabel_index.csv
    git tag v2.1-multilabel-data
    Track 3 Phases 1-7: code changes
    Track 3 Phase 8: retrain (~2-3 days compute)
    git tag v3.0-multilabel-retrain
    Track 3 Phase 9: threshold tuning
    Track 3 Phase 10: ZKML rebuild + Sepolia redeploy

  AFTER RETRAIN:
    Evaluate per-class F1 → pick from B/C improvements
    ONE combined retrain for all chosen improvements

══════════════════════════════════════════════════════════════════
WHAT NEVER CHANGES WITHOUT FULL RETRAIN
══════════════════════════════════════════════════════════════════

  GNNEncoder    in_channels=8, heads=8, 3-layer GAT
  TransformerEncoder  tokenizer="microsoft/codebert-base", max_length=512
  FusionLayer   gnn_dim=64, transformer_dim=768, concat=832
  Node features [type_id, vis, pure, view, payable, reentrant, complexity, loc]
  Node order    CONTRACT → STATE_VARs → FUNCTIONs → MODIFIERs → EVENTs

  Change any of these → rebuild all 68,555 .pt graph files + retrain.

══════════════════════════════════════════════════════════════════
PORT MAP
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
PARKED ITEMS — carry forward
══════════════════════════════════════════════════════════════════

  v4.0 git tag
  Module 2 zkml milestone docs
  M3.6 Dockerfile
  GitHub Actions push (ingest.yml written, not pushed)
  CI_MODE embedder — CPU fallback for GitHub Actions
  Index rebuild with chunk_size=1536 (current: 512)
  DVC versioning of RAG index
  TRANSFORMERS_OFFLINE=1 + HF_TOKEN → .env
  SWC Registry + Rekt.news + Immunefi fetchers
  solc-select confirmed on 0.8.20

══════════════════════════════════════════════════════════════════
