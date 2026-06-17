# ML Core — Visual Diagrams

Interactive Mermaid diagrams for `ml/README.md`. Rendered natively on GitHub.
For tensor-shape step-by-step flows, see the ASCII diagrams in `ml/README.md`.

**Schema version:** v9 · **Model version:** v8.1 (four-eye) · **Data module:** `sentinel-data` v0.1.0

---

## Module Dependency

```mermaid
flowchart LR
    subgraph DM["data_module/ — sentinel-data v0.1.0"]
        direction TB
        CLI["cli.py\n9-stage CLI entry point"]
        ING["ingestion/\n5 connector types\nSHA-256 manifests"]
        PRE["preprocessing/\nflatten · compile · dedup · normalize · segment"]
        REP["representation/\ngraph_schema.py (CANONICAL v9)\ngraph_extractor.py (thin adapter)\ntokenizer.py (thin adapter)\norchestrator.py · cache_manager.py"]
        LBL["labeling/\ncrosswalks · merger · gate"]
        VER["verification/\nsemantic_checker · tool_validator · gate"]
        SPL["splitting/\n4 strategies · dedup_enforcer\nNonVulnerable 3:1 cap"]
        REG["registry/\nSQLite catalog · YAML mirror · lineage"]
        ANA["analysis/\n5 read-only exploratory tools"]
        EXP["export/\nchunker · 4 writers\nformat_schema v1.yaml"]
        CLI --> ING --> PRE --> REP --> LBL --> VER --> SPL --> REG --> ANA --> EXP
    end

    subgraph ML["ml/ — sentinel-ml"]
        direction TB
        MCLI["ml/scripts/train.py"]
        MDS["SentinelDataset\nStage 7B v2 export artifacts"]
        MMDL["SentinelModel v8.1\nFour-eye classifier"]
        MINF["Inference API\nFastAPI port 8001"]
        MCLI --> MDS --> MMDL --> MINF
    end

    REP -. "schema (canonical) → ml re-imports\nextractor + tokenizer → data_module re-imports" .-> ML
    EXP -->|"sharded export\n(graphs · tokens · labels.parquet · metadata.parquet)"| MDS

    style DM fill:#eef2ff,stroke:#818cf8,stroke-width:2px
    style ML fill:#fff7ed,stroke:#fb923c,stroke-width:2px
```

---

## Data Module Pipeline (9 Stages)

```mermaid
flowchart TD
    subgraph S1["Stage 1a — Ingest"]
        CONN["5 connector types\ngit · huggingface · zenodo · etherscan · manual"]
        MAN["SHA-256 manifests\nper-file integrity"]
        CONN --> MAN
    end

    subgraph S1B["Stage 1b — Preprocess"]
        FLAT["flatten.py\nsolc --flatten"]
        COMP["compiler.py\n2-pass: exact pragma → nearest"]
        DEDUP["deduplicator.py\nSHA-256 → address → AST near-dup (0.85)"]
        NORM["normalizer.py\nstrip comments · SPDX · whitespace"]
        SEG["segmenter.py\nversion bucket + has_unchecked_block"]
        FLAT --> COMP --> DEDUP --> NORM --> SEG
    end

    subgraph S2["Stage 2 — Represent"]
        ORCH["orchestrator.py\nv2 manifest-driven"]
        GS["graph_schema.py\nv9 · NODE_FEATURE_DIM=12\n14 node types · 12 edge types\n12 features · 10 classes"]
        GE["graph_extractor.py\nextract_contract_graph()"]
        TK["windowed_tokenizer.py\n[4, 512] stride=256"]
        CACHE["cache_manager.py\ncontent-addressed\n(schema + extractor version)"]
        VER["versioner.py\nprevents v8/v9 silent mix"]
        ORCH --> GS --> GE
        ORCH --> TK
        GE & TK --> CACHE --> VER
    end

    subgraph S3["Stage 3 — Label"]
        XW["crosswalks/\nper-source class maps"]
        MERGE["merger.py\nmulti-source · tier precedence\n99% co-occurrence flagging"]
        GATE1["gate.py\nmin-viable-corpus\ntotal ≥ 4000\nper-class ≥ 300/100"]
        XW --> MERGE --> GATE1
    end

    subgraph S4["Stage 4 — Verify"]
        CA["class_auditor.py\n10×10 co-occurrence matrix"]
        SC["semantic_checker.py\n7 extractable · 3 NOT_EXTRACTABLE"]
        TV["tool_validator.py\nSlither per-class agreement"]
        FP["fp_estimator.py\nstratified-by-(source,tier)"]
        NC["negative_checker.py\nNonVulnerable contamination"]
        VG["gate.py\nVERIFIED · PROVISIONAL\nBEST-EFFORT · FAIL"]
        CA & SC & TV & FP & NC --> VG
    end

    subgraph S5["Stage 5 — Split"]
        SPL["4 strategies\nrandom · stratified · project · temporal"]
        DE["dedup_enforcer.py\nreassigns straddling dedup groups"]
        NVC["nonvulnerable_cap.py\n3:1 positive ratio cap"]
        LA["leakage_auditor.py\nJaccard similarity 0.50 threshold"]
        SPL --> DE --> NVC --> LA
    end

    subgraph S6["Stage 6 — Register + Analyze"]
        CAT["catalog.py\nSQLite + YAML mirror\n4 base + 2 system tables"]
        LIN["lineage_tracker.py\nDAG of transformations"]
        FD["feature_dist.py\ncomplexity_proxy_risk"]
        CO["cooccurrence.py\noverlap_detector.py\ndrift_monitor.py"]
        CAT --> LIN
        FD --> CO
    end

    subgraph S7["Stage 7 — Export"]
        CHUNK["chunker.py\nExportManifest\nartifact_hash LAST"]
        LW["label_writer.py\nlabels.parquet"]
        MW["metadata_writer.py\nmetadata.parquet"]
        GW["graph_writer.py\ngraphs/graphs-{shard}.pt"]
        TW["token_writer.py\ntokens/tokens-{shard}.pt"]
        FS["format_schema/v1.yaml\n494-line formal spec"]
        CHUNK --> LW & MW & GW & TW
        FS -. "defines" .-> CHUNK
    end

    S1 --> S1B --> S2 --> S3 --> S4 --> S5 --> S6 --> S7

    style S1 fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style S1B fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style S2 fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
    style S3 fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    style S4 fill:#fce7f3,stroke:#f472b6,stroke-width:2px
    style S5 fill:#d1fae5,stroke:#34d399,stroke-width:2px
    style S6 fill:#ede9fe,stroke:#a78bfa,stroke-width:2px
    style S7 fill:#fff7ed,stroke:#fb923c,stroke-width:2px
```

---

## System Lifecycle

```mermaid
flowchart TD
    RAW["17+ curated sources\nSolidiFI · DIVE · SmartBugs Curated\nWeb3Bugs · DISL · BCCC (deferred)"]

    subgraph SHARED["Shared Preprocessing Layer"]
        GS["graph_schema.py (v9)\nNODE_TYPES(14) · EDGE_TYPES(12) · FEATURE_NAMES(12)\nFEATURE_SCHEMA_VERSION=v9 · NODE_FEATURE_DIM=12\nThin re-export shim → sentinel_data.representation.graph_schema"]
        GE["graph_extractor.py\nextract_contract_graph()\nThin re-export shim → sentinel_data.representation.graph_extractor"]
        GS --> GE
    end

    subgraph DM["Data Module — sentinel-data v0.1.0"]
        direction TB
        ING["Stage 1a: ingest\n5 connector types · SHA-256 manifests"]
        PRE["Stage 1b: preprocess\nflatten · compile · dedup@0.85 · normalize · segment"]
        REP["Stage 2: represent\nv9 graph .pt + windowed tokens"]
        LBL["Stage 3: label\ncrosswalk YAMLs · tier precedence · gate"]
        VER["Stage 4: verify\nsemantic · tool · FP · co-occurrence · gate"]
        SPL["Stage 5: split\n4 strategies · dedup_enforcer · 3:1 cap"]
        REG["Stage 6: register\nSQLite catalog · lineage"]
        ANA["Stage 7a: analyze\nfeature_dist · cooccurrence · drift"]
        EXP["Stage 7b: export\nchunker · 4 writers · format_schema v1.yaml"]
        ING --> PRE --> REP --> LBL --> VER --> SPL --> REG --> ANA --> EXP
    end

    subgraph OFFLINE["Offline Data Preparation"]
        EXPORT[("Export Artifacts\nsentinel-v2-*/\ngraphs/*.pt · tokens/*.pt\nlabels.parquet · metadata.parquet\nmanifest.json + artifact hash")]
        SPLITS[("Splits\ntrain · val · test\nby contract_id from parquet")]
        EXP --> SPLITS
    end

    subgraph TRAIN["Training — train.py + MLflow"]
        DS["SentinelDataset\nStage 7B · v2 export artifacts\n5-tuple: (graph, tokens, y, contract_id, tier)\nLRU shard cache (4 shards)\n3 integrity gates"]
        SM["SentinelModel v8.1\nFour-eye: GNN · Transformer · Fused · CFG\n8-layer GNN (2+3+3 phases)\n~2.5M GNN + ~590K LoRA params"]
        LOSS["AsymmetricLoss\n(γ⁻=2.0, γ⁺=1.0, clip=0.01)\n+ auxiliary loss λ=0.3\n+ JK entropy reg λ=0.005\n+ DoS loss λ=0.5"]
        CKPT[("checkpoints/\nbest.pt + _thresholds.json\nGCB-P1-Run12-v3dospatched\n-20260613_FINAL.pt")]
        DS --> SM --> LOSS --> CKPT
    end

    TT["tune_threshold.py\n0.05–0.95 grid per class"]
    CT["calibrate_temperature.py\nper-class temperature scaling"]
    OE["interpretability/\n29 experiment scripts"]
    MR["promote_model.py\nMLflow Staging → Production"]

    subgraph INFER["Inference API — src/inference/ — port 8001"]
        IC["InferenceCache\ncontent-addressed · TTL 24 h"]
        CP["ContractPreprocessor\npreprocess.py\nsliding window max 4 windows"]
        FW["SentinelModel.forward()\n4 eyes → classifier → logits"]
        DD["DriftDetector\nKS test every 50 requests"]
        RESP["POST /predict\nthree-tier: CONFIRMED ≥ 0.55\nSUSPICIOUS ≥ 0.25\nNOTEWORTHY < 0.25"]
        HS["POST /hotspots\nGNN attention hotspots\n+ ML prediction"]
        IC -->|"cache hit"| FW
        IC -->|"cache miss"| CP --> FW --> RESP
        FW --> DD
        FW --> HS
    end

    RAW --> ING
    EXP -->|"sharded export"| DS
    CKPT --> TT -->|"writes _thresholds.json"| CKPT
    CKPT --> CT -->|"writes temperatures.json"| CKPT
    CKPT --> OE
    CKPT --> MR
    CKPT --> IC & FW

    style SHARED fill:#eef2ff,stroke:#818cf8,stroke-width:2px
    style DM fill:#eef2ff,stroke:#818cf8,stroke-width:2px
    style OFFLINE fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
    style TRAIN  fill:#fff7ed,stroke:#fb923c,stroke-width:2px
    style INFER  fill:#fdf2f8,stroke:#e879f9,stroke-width:2px
```

---

## Shared Preprocessing Layer

```mermaid
flowchart TD
    subgraph SCHEMA["graph_schema.py — single source of truth (v9)"]
        direction LR
        NT["NODE_TYPES (14)\nSTATE_VAR=0 · FUNCTION=1 · MODIFIER=2\nEVENT=3 · FALLBACK=4 · RECEIVE=5\nCONSTRUCTOR=6 · CONTRACT=7\nCFG_NODE_CALL=8 · CFG_NODE_WRITE=9\nCFG_NODE_READ=10 · CFG_NODE_CHECK=11\nCFG_NODE_OTHER=12 · CFG_NODE_ARITH=13"]
        ET["EDGE_TYPES (12)\nCALLS=0 · READS=1 · WRITES=2 · EMITS=3\nINHERITS=4 · CONTAINS=5 · CONTROL_FLOW=6\nREVERSE_CONTAINS=7 (runtime only)\nCALL_ENTRY=8 · RETURN_TO=9\nDEF_USE=10 · EXTERNAL_CALL=11"]
        FN["FEATURE_NAMES (12)\ntype_id · visibility · uses_block_globals\nview · payable · complexity · loc\nreturn_ignored · call_target_typed\nhas_loop · external_call_count\nin_unchecked_block"]
        CONST["NODE_FEATURE_DIM = 12\nNUM_NODE_TYPES = 14\nNUM_EDGE_TYPES = 12\nNUM_CLASSES = 10\nFEATURE_SCHEMA_VERSION = v9\n_MAX_TYPE_ID = 13.0"]
    end

    subgraph SHIM["ml/src/preprocessing/ — thin re-export shims"]
        GS_SHIM["graph_schema.py\nre-exports from\nsentinel_data.representation.graph_schema"]
        GE_SHIM["graph_extractor.py\nre-exports from\nsentinel_data.representation.graph_extractor"]
    end

    subgraph CANONICAL["data_module/sentinel_data/representation/ — canonical source"]
        GS_CAN["graph_schema.py (251 lines)\nSlither version assertion ≥0.9.3\nNODE_TYPES · EDGE_TYPES · FEATURE_NAMES\nCLASS_NAMES · NodeType IntEnum\ninvariant assertions at import time"]
        GE_CAN["graph_extractor.py\nthin adapter → ml.src.preprocessing.graph_extractor"]
        TK_CAN["tokenizer.py\nthin adapter → ml.src.data_extraction.windowed_tokenizer"]
    end

    subgraph EXTRACTOR["graph_extractor.py — canonical implementation"]
        CFG["GraphExtractionConfig\nmulti_contract_policy (most_derived)\ninclude_edge_attr · solc_binary · allow_paths"]
        EXC["Exception hierarchy\nSolcCompilationError → HTTP 400\nSlitherParseError    → HTTP 500\nEmptyGraphError      → HTTP 400"]
        FN2["extract_contract_graph(sol_path, config)\nreturns Data: x[N,12] · edge_index[2,E] · edge_attr[E]\nICFG-Lite edges (CALL_ENTRY, RETURN_TO)\nDEF_USE data-flow edges\nEXTERNAL_CALL edges (v9)\nNever returns None — always raises on failure"]
    end

    ORCH["orchestrator.py (data_module Stage 2)\nv2 manifest-driven batch extraction"]
    PP["preprocess.py — ContractPreprocessor\nOnline inference · one contract per request"]

    CANONICAL -. "re-exported by" .-> SHIM
    GS_CAN -->|"imports"| GE_CAN
    GS_SHIM -->|"used by"| ORCH
    GS_SHIM -->|"used by"| PP

    ORCH --> G[("data_module/data/representations/\n.pt files (v9)")]
    PP --> API["Inference API\nPOST /predict"]

    style SCHEMA fill:#eef2ff,stroke:#818cf8,stroke-width:2px
    style SHIM fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    style CANONICAL fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
    style EXTRACTOR fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
```

---

## Model Architecture (v8.1 Four-Eye)

```mermaid
flowchart LR
    SRC(["Solidity\nSource"])

    subgraph GNN["GNN Path — gnn_encoder.py (v8.1)"]
        direction TB
        GE2["graph_extractor.py\nextract_contract_graph()"]
        NF["x  [N, 12]\nnode features (v9)"]
        TE["type_embedding\nEmbedding(14, 16)\n→ [N, 16]"]
        CONCAT["concat\n[N, 28]"]
        EA["edge_emb  Embedding(12,64)\nedge_attr[E] → [E, 64]"]
        SKIP["IMP-G2: input_proj\nLinear(28, 256) skip"]
        P1["Phase 1 (Layers 1+2)\nEdges 0-5 · heads=8\nStructural aggregation"]
        P2["Phase 2 (Layers 3-5)\nEdges 6,8,9,10,11 · heads=4\nCF + ICFG-Lite + DEF_USE + EXTERNAL_CALL"]
        P3["Phase 3 (Layers 6-8)\nREVERSE_CONTAINS up (layers 6+7)\nCONTAINS down (layer 8, IMP-G3)\nheads=1"]
        JK["JK Attention\nLearned aggregation\nover 3 phase outputs"]
        GE2 --> NF & EA
        NF --> CONCAT
        TE --> CONCAT
        CONCAT --> SKIP
        SKIP --> P1
        EA --> P1
        P1 --> P2 --> P3 --> JK
    end

    subgraph TFM["Transformer Path — transformer_encoder.py"]
        direction TB
        TOK["GraphCodeBERT tokenizer\ninput_ids [B,512] or [B,W,512]\nattn_mask [B,512] or [B,W,512]"]
        CB["microsoft/graphcodebert-base\n125M params  frozen\nFlash Attention 2"]
        LOR["LoRA  r=16  α=32\n~590K trainable params\nQ+V projections · all 12 layers"]
        TE2["last_hidden_state\n[B, 512, 768] or [B, W*L, 768]\nall token positions"]
        WP["WindowAttentionPooler\nLearned attention over\nW window-CLS tokens → [B, 768]"]
        TOK --> CB
        LOR -. "adapts" .-> CB
        CB --> TE2 --> WP
    end

    subgraph FUS["CrossAttentionFusion — fusion_layer.py"]
        direction TB
        NP["node_proj  Linear(256→256)\n[N,256] → [N,256]"]
        TP["token_proj  Linear(768→256)\n[B,512,768] → [B,512,256]"]
        TN["token_norm  LayerNorm(768)\nBUG-C2 fix"]
        DB["_scatter_to_dense\n[B, max_nodes, 256] + node_real_mask"]
        N2T["Node→Token MHA\nQ=nodes  K=V=tokens\nmask PAD token positions\n→ enriched_nodes [B,n,256]"]
        T2N["Token→Node MHA\nQ=tokens  K=V=nodes\nmask padded node positions\n→ enriched_tokens [B,512,256]"]
        POL["Masked mean pool\npooled_nodes  [B,256]\npooled_tokens  [B,256]"]
        FOUT["concat → [B,512]\nLinear → ReLU → Dropout\n[B, 128]"]
        NP --> DB --> N2T & T2N
        TP --> TN --> N2T & T2N
        N2T & T2N --> POL --> FOUT
    end

    subgraph EYES["Four-Eye Classifier — sentinel_model.py"]
        direction TB
        GNN_POOL["GNN Eye\nFUNCTION+MODIFIER+FALLBACK+RECEIVE+CONSTRUCTOR\nmax+mean → [B,512] → Linear → [B,128]"]
        TF_POOL["Transformer Eye\nWindow-pooled CLS → [B,768]\nLinear(768,128) → [B,128]"]
        FUSED_EYE["Fused Eye\nCrossAttentionFusion → [B,128]"]
        CFG_EYE["CFG Eye (IMP-R7-2)\nPhase2 pool over CFG nodes [8-13]\nmax+mean → [B,512] → Linear → [B,128]"]
        CONCAT2["cat([gnn_eye, tf_eye, fused_eye, cfg_eye])\n[B, 512]"]
        CLS["Classifier\nLinear(512,256) → ReLU → Dropout\nLinear(256,10) → raw logits [B,10]"]
        AUX["Auxiliary Heads (training only)\nLinear(128,10) per eye + aux_phase2\nλ=0.3 main, λ=0.20 phase2"]
        GNN_POOL & TF_POOL & FUSED_EYE & CFG_EYE --> CONCAT2 --> CLS
        GNN_POOL & TF_POOL & FUSED_EYE & CFG_EYE --> AUX
    end

    PRED["Predictor._score\nsigmoid → probs [B,10]\nper-class thresholds\n→ vulnerabilities list\nlabel: vulnerable / safe\nThree-tier: CONFIRMED ≥ 0.55\nSUSPICIOUS ≥ 0.25\nNOTEWORTHY < 0.25"]

    SRC --> GNN & TFM
    JK -->|"node_embs [N,256]\nbatch [N]"| FUS
    WP -->|"[B, 768]"| FUS
    FOUT --> FUSED_EYE
    JK --> GNN_POOL
    JK -->|"raw Phase2\nCFG nodes only"| CFG_EYE
    WP --> TF_POOL
    CLS --> PRED

    style GNN fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style TFM fill:#fce7f3,stroke:#f472b6,stroke-width:2px
    style FUS fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    style EYES fill:#d1fae5,stroke:#34d399,stroke-width:2px
```

---

## GNN Three-Phase Architecture

```mermaid
flowchart LR
    IN["Input\nx[N,28] · edge_index[2,E] · edge_attr[E]\n(NODE_FEATURE_DIM=12 + type_emb=16)"]

    subgraph P1["Phase 1 — Structural (Layers 1+2)"]
        direction TB
        L1["Layer 1 (conv1)\nEdges 0-5 · heads=8\nadd_self_loops=True\nIMP-G2 skip from input_proj"]
        L2["Layer 2 (conv2)\nEdges 0-5 · heads=8\nadd_self_loops=True"]
        LN1["LayerNorm"]
        L1 --> L2 --> LN1
    end

    subgraph P2["Phase 2 — CFG + ICFG (Layers 3+4+5)"]
        direction TB
        L3["Layer 3 (conv3)\nCONTROL_FLOW(6) only\nheads=4 · no self-loops"]
        L3B["Layer 4 (conv3b)\nCALL_ENTRY(8) + RETURN_TO(9)\nheads=4 · no self-loops"]
        L3C["Layer 5 (conv3c)\nCF(6)+CE(8)+RT(9)+DU(10)+EC(11) joint\nheads=4 · integration layer"]
        LN2["LayerNorm"]
        L3 --> L3B --> L3C --> LN2
    end

    subgraph P3["Phase 3 — Bidirectional CONTAINS (Layers 6+7+8)"]
        direction TB
        L4["Layer 6 (conv4)\nREVERSE_CONTAINS up\nCFG→FUNCTION\nheads=1"]
        L4B["Layer 7 (conv4b)\nREVERSE_CONTAINS up\nsecond hop\nheads=1"]
        L4C["Layer 8 (conv4c)\nCONTAINS down (IMP-G3)\nFUNCTION→CFG\nheads=1"]
        LN3["LayerNorm"]
        L4 --> L4B --> L4C --> LN3
    end

    JK["JK Attention\n Learned weights\n over 3 phases"]
    OUT["node_embs [N,256]"]

    IN --> P1 --> P2 --> P3 --> JK --> OUT

    style P1 fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style P2 fill:#fce7f3,stroke:#f472b6,stroke-width:2px
    style P3 fill:#d1fae5,stroke:#34d399,stroke-width:2px
```

---

## SentinelDataset (Stage 7B)

```mermaid
flowchart TD
    subgraph INIT["SentinelDataset.__init__()"]
        direction TB
        EXP["SentinelDatasetExport\nmanifest.json loading"]
        G1["Gate 1 — format schema\nmust be v1"]
        G2["Gate 2 — graph schema\nmust match FEATURE_SCHEMA_VERSION (v9)"]
        G3["Gate 3 — artifact hash\nSHA-256 integrity check"]
        LBL["labels.parquet\n{contract_id: (y_tensor[10], tier)}"]
        IDS["get_split_contract_ids(split)\nfilter to contracts in shard_index"]
        EXP --> G1 --> G2 --> G3 --> LBL --> IDS
    end

    subgraph ITEM["__getitem__(idx)"]
        direction TB
        CID["contract_id = _contract_ids[idx]"]
        ENTRY["shard_index[contract_id]\n→ (shard, pos_in_shard)"]
        GLOAD["Load graph shard\ngraphs/graphs-{shard:05d}.pt\nLRU-cached (4 shards)\ngraph.get_example(pos)"]
        TLOAD["Load token shard\ntokens/tokens-{shard:05d}.pt\nLRU-cached (4 shards)\ntoken_shard[pos] → [4, 512]"]
        AMASK["attention_mask\n(input_ids != pad_token_id).long()"]
        Y["y = _label_lookup[contract_id]\nfloat32 [10]\nconfidence_tier"]
        CID --> ENTRY --> GLOAD & TLOAD
        TLOAD --> AMASK
        GLOAD & AMASK & Y --> OUT["return (graph, tokens, y, contract_id, confidence_tier)\n5-tuple"]
    end

    subgraph COLL["sentinel_collate_fn(batch)"]
        direction TB
        BG["Batch.from_data_list(graphs)\nPyG batched graph"]
        BT["torch.stack input_ids [B,4,512]\ntorch.stack attention_mask [B,4,512]"]
        BY["multi-label: [B,10] float32"]
        BC["contract_ids: list[str]"]
        BTIER["confidence_tiers: list[str | None]"]
        BG & BT & BY & BC & BTIER --> RET["return (Batch, token_dict, y, contract_ids, confidence_tiers)"]
    end

    INIT --> ITEM --> COLL

    style INIT fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style ITEM fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
    style COLL fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
```

---

## Training Flow

```mermaid
flowchart TD
    subgraph SETUP["Setup"]
        CFG["TrainConfig\nbatch=8 · accum=8 → effective=64\nASL loss · BF16 · submodule compile\nearly_stop_patience=30\npos_weight_cap=10.0"]
        DS["SentinelDataset\nStage 7B v2 export artifacts\nWeightedRandomSampler\n(timestamp-size: 4× Timestamp+ >150 nodes)"]
        MDL["SentinelModel v8.1\nfour_eye_v8 · 8-layer GNN\nGNN prefix K=48 (warmup 15 ep)"]
        CFG --> DS --> MDL
    end

    subgraph LOOP["Training Loop (per epoch)"]
        direction TB
        FW["forward()\nGNN encoder → 4 eyes → classifier → logits\n+ GNN prefix injection (warmup epochs 0-14)"]
        AUX["Auxiliary loss\n0.3 × (aux_gnn + aux_tf + aux_fused)\n+ 0.20 × aux_phase2"]
        MAIN["Main loss\nASL(logits, labels)\n+ dos_loss_weight × DoS gradient"]
        JK_REG["JK entropy regularization\nλ=0.005 · pushes toward log(3)"]
        TOT["total_loss = main + aux + jk_reg"]
        BACKWARD["backward() · grad_clip=1.0\nBF16 autocast"]
        EVAL["eval_metrics()\nper-class F1 · AUROC · ECE\nthreshold tuning"]
        SAVE["checkpoint()\nbest.pt + _thresholds.json"]
        LOG["StructuredLogger\nstep_metrics.jsonl\nepoch_summary.jsonl (37 fields)\nalerts.jsonl (KILL/WARN_SKIP/WARN)"]
        FW --> AUX --> TOT
        MAIN --> TOT
        JK_REG --> TOT
        TOT --> BACKWARD --> EVAL --> SAVE
        EVAL --> LOG
    end

    subgraph CAL["Calibration"]
        TC["tune_threshold.py\n0.05–0.95 grid per class"]
        CT["calibrate_temperature.py\nper-class temperature scaling\nminimise BCE NLL on val set"]
        PM["promote_model.py\nMLflow Staging → Production"]
        TC --> CT --> PM
    end

    LOOP -->|"best model"| CAL

    style SETUP fill:#eef2ff,stroke:#818cf8,stroke-width:2px
    style LOOP fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
    style CAL fill:#fff7ed,stroke:#fb923c,stroke-width:2px
```

---

## Inference Pipeline

```mermaid
flowchart TD
    REQ["POST /predict\ncontract_code: str"]

    subgraph PRE["Preprocessing"]
        TMP["Write to temp .sol file"]
        GE["extract_contract_graph()\nv9 graph .pt\nNODE_FEATURE_DIM=12"]
        TK["retokenize_windowed()\n[4, 512] windows · stride=256\nmax_windows=4"]
        TMP --> GE --> TK
    end

    subgraph MODEL["SentinelModel.forward()"]
        GNN["GNN encoder\n8-layer three-phase GAT\n→ node_embs [N,256]"]
        TFM["Transformer encoder\nGraphCodeBERT + LoRA\n→ token_embs [B,512,768]"]
        PREFIX["GNN prefix injection\nK=48 nodes → [B,48,768]\nsuppressed if _current_epoch < warmup"]
        FUS["CrossAttentionFusion\n→ fused_eye [B,128]"]
        EYES["Four-Eye pooling\nGNN eye + TF eye + Fused eye + CFG eye\n→ [B,512]"]
        CLS["Classifier\nLinear(512,256) → Linear(256,10)\n→ logits [B,10]"]
        GNN --> FUS
        TFM --> FUS
        PREFIX -->|"prepended to inputs_embeds"| TFM
        FUS --> EYES
        GNN --> EYES
        TFM --> EYES
        EYES --> CLS
    end

    SIG["sigmoid → probs [B,10]"]
    THR["per-class thresholds\n0.25-0.55 range\n(loaded from _thresholds.json)"]
    TIER["Three-tier classification\nCONFIRMED ≥ 0.55\nSUSPICIOUS ≥ 0.25\nNOTEWORTHY < 0.25"]
    RESP["POST /predict response\nlabel · confirmed · suspicious\nprobabilities · thresholds\ntruncated · windows_used\nnum_nodes · num_edges"]

    REQ --> PRE --> MODEL --> SIG --> THR --> TIER --> RESP

    style PRE fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style MODEL fill:#fce7f3,stroke:#f472b6,stroke-width:2px
```

---

## Export Format (v1)

```mermaid
flowchart LR
    subgraph INPUT["Input Artifacts"]
        GRAPHS[("data_module/data/representations/\n.pt files (v9 schema)")]
        TOKENS[("data_module/data/representations/\n.tokens.pt files")]
        LABELS[("data_module/data/labels/\nmulti-label CSVs")]
        META[("data_module/data/preprocessed/\nmeta.json sidecars")]
    end

    subgraph EXPORT["chunk_export() — Stage 7"]
        CHUNK["chunker.py\nExportManifest\nartifact_hash written LAST"]
        GW["graph_writer.py\ngraphs/graphs-{shard:05d}.pt\nPyG Batch · shard_size=5000"]
        TW["token_writer.py\ntokens/tokens-{shard:05d}.pt\ntorch.Tensor [N,4,512]"]
        LW["label_writer.py\nlabels.parquet\n10 class cols (int8) + contract_id + source + split + tier"]
        MW["metadata_writer.py\nmetadata.parquet\nsolc_version · version_bucket · loc · n_functions\nnode_count · edge_count · ..."]
        CHUNK --> GW & TW & LW & MW
    end

    subgraph OUTPUT["Export Artifact"]
        MF["manifest.json\nschema_version · graph_schema_version\nshard_count · total_contracts · artifact_hash\ntrain/val/test contract lists"]
        SHARDS[("graphs/*.pt + tokens/*.pt\nlabels.parquet + metadata.parquet")]
        HASH["artifact_hash\nSHA-256 of all shard files\n+ manifest fields (excluding hash itself)"]
        MF --> SHARDS
        HASH -. "verified by" .-> MF
    end

    INPUT --> EXPORT --> OUTPUT

    subgraph CONSUMER["ml/ Consumer"]
        SDS["SentinelDataset\n3 integrity gates\nformat schema v1\ngraph schema v9\nartifact hash"]
        SHARDS --> SDS
    end

    style INPUT fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style EXPORT fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
    style OUTPUT fill:#fff7ed,stroke:#fb923c,stroke-width:2px
    style CONSUMER fill:#fdf2f8,stroke:#e879f9,stroke-width:2px
```

---

## Constants Reference

| Constant | Value | Source |
|---|---|---|
| `FEATURE_SCHEMA_VERSION` | `"v9"` | `sentinel_data/representation/graph_schema.py:77` |
| `NODE_FEATURE_DIM` | `12` | `sentinel_data/representation/graph_schema.py:83` |
| `NUM_NODE_TYPES` | `14` | `sentinel_data/representation/graph_schema.py:84` |
| `NUM_EDGE_TYPES` | `12` | `sentinel_data/representation/graph_schema.py:85` |
| `NUM_CLASSES` | `10` | `sentinel_data/representation/graph_schema.py:210` |
| `_MAX_TYPE_ID` | `13.0` | `sentinel_data/representation/graph_schema.py:216` |
| Model version | `v8.1` | `ml/src/training/trainer.py:121` |
| Architecture tag | `"four_eye_v8"` | `ml/src/training/trainer.py:119` |
| GNN hidden_dim | `256` | `ml/src/training/trainer.py:204` |
| GNN layers | `8` (2+3+3) | `ml/src/training/trainer.py:205` |
| LoRA rank | `16` | `ml/src/training/trainer.py:218` |
| LoRA alpha | `32` | `ml/src/training/trainer.py:219` |
| Transformer backbone | `microsoft/graphcodebert-base` | `ml/src/models/transformer_encoder.py:136` |
| Fusion output | `128` | `ml/src/training/trainer.py:197` |
| Max windows | `4` | `ml/src/inference/predictor.py:84` |
| Window size | `512` | `ml/src/data_extraction/windowed_tokenizer.py:41` |
| Stride | `256` | `ml/src/data_extraction/windowed_tokenizer.py:42` |
| Effective batch | `64` (8×8 accum) | `ml/src/training/trainer.py:229,270` |
| Loss | `ASL (γ⁻=2.0, γ⁺=1.0, clip=0.01)` | `ml/src/training/trainer.py:294-300` |
| API version | `3.0.0` | `ml/src/inference/api.py:126` |
| Export format schema | `v1` | `data_module/sentinel_data/export/format_schema/v1.yaml` |
| Data module version | `0.1.0` | `data_module/sentinel_data/__init__.py` |
| Pipeline stages | `9` (ingest→export) | `data_module/sentinel_data/cli.py:71-81` |
| Enabled sources | `5` (SolidiFI, DIVE, SmartBugs, Web3Bugs, DISL) | `data_module/config.yaml` |
