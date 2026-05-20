# M1 — ML Core: Visual Diagrams

Interactive Mermaid diagrams for `ml/README.md`. Rendered natively on GitHub.
For tensor-shape step-by-step flows, see the ASCII diagrams in `ml/README.md`.

---

## System Lifecycle

```mermaid
flowchart TD
    RAW["BCCC-SCsVul-2024\nraw .sol files + labels"]

    subgraph SHARED["Shared Preprocessing Layer"]
        GS["graph_schema.py\nNODE_TYPES(13) · EDGE_TYPES(11) · FEATURE_NAMES(11)\nFEATURE_SCHEMA_VERSION=v8 · NODE_FEATURE_DIM=11"]
        GE["graph_extractor.py\nextract_contract_graph()"]
        GS --> GE
    end

    subgraph OFFLINE["① Offline Data Preparation"]
        AE["ast_extractor.py\n11 workers · solc version-pinned"]
        TK["tokenizer.py\nCodeBERT tokenizer"]
        BM["build_multilabel_index.py"]
        CS["create_splits.py · stratified"]
        GRAPHS[("graphs/\n~41K .pt (v8)\nx[N,11] · edge_index[2,E] · edge_attr[E]")]
        TOKENS[("tokens/\n~41K .pt\ninput_ids · attn_mask")]
        CSV[("multilabel_index.csv\n68,523 rows × 10 classes")]
        SPLITS[("splits/\ntrain 47,966 · val 10,278\ntest 10,279")]
        AE --> GRAPHS
        TK --> TOKENS
        BM --> CSV --> CS --> SPLITS
    end

    subgraph TRAIN["② Training — scripts/train.py + MLflow"]
        DS["DualPathDataset\nbinary: graph.y\nmulti-label: multilabel_index.csv\nRAM cache support"]
        SM["SentinelModel v7\nThree-eye: GNN · Transformer · Fused\n7-layer GNN (2+3+2 phases)"]
        LOSS["BCEWithLogitsLoss\nor FocalLoss"]
        CKPT[("checkpoints/\nbest.pt + _thresholds.json")]
        DS --> SM --> LOSS --> CKPT
    end

    TT["tune_threshold.py\n0.05–0.95 grid per class"]
    OE["run_overnight_experiments.py\n4 hyperparameter configs\n--start-from N"]
    MR["promote_model.py\nMLflow Staging → Production"]

    subgraph INFER["③ Inference API — src/inference/ — port 8001"]
        IC["InferenceCache\ncontent-addressed · TTL 24 h"]
        CP["ContractPreprocessor\npreprocess.py"]
        FW["SentinelModel.forward()\nsliding window if > 512 tokens"]
        DD["DriftDetector\nKS test every 50 requests"]
        RESP["POST /predict\nlabel · vulnerabilities · probabilities"]
        IC -->|"cache hit"| FW
        IC -->|"cache miss"| CP --> FW --> RESP
        FW --> DD
    end

    RAW --> AE & TK & BM
    AE -->|"calls extract_contract_graph()"| GE
    CP -->|"calls extract_contract_graph()"| GE
    CSV -->|"label_csv arg"| DS
    GRAPHS & TOKENS & SPLITS --> DS
    CKPT --> TT -->|"writes _thresholds.json"| CKPT
    CKPT --> OE
    CKPT --> MR
    CKPT --> IC & FW

    style SHARED fill:#eef2ff,stroke:#818cf8,stroke-width:2px
    style OFFLINE fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
    style TRAIN  fill:#fff7ed,stroke:#fb923c,stroke-width:2px
    style INFER  fill:#fdf2f8,stroke:#e879f9,stroke-width:2px
```

---

## Shared Preprocessing Layer

```mermaid
flowchart TD
    subgraph SCHEMA["graph_schema.py — single source of truth (v8)"]
        direction LR
        NT["NODE_TYPES (13)\nSTATE_VAR=0 · FUNCTION=1 · MODIFIER=2\nEVENT=3 · FALLBACK=4 · RECEIVE=5\nCONSTRUCTOR=6 · CONTRACT=7\nCFG_NODE_CALL=8 · CFG_NODE_WRITE=9\nCFG_NODE_READ=10 · CFG_NODE_CHECK=11\nCFG_NODE_OTHER=12"]
        ET["EDGE_TYPES (11)\nCALLS=0 · READS=1 · WRITES=2 · EMITS=3\nINHERITS=4 · CONTAINS=5 · CONTROL_FLOW=6\nREVERSE_CONTAINS=7 (runtime)\nCALL_ENTRY=8 · RETURN_TO=9\nDEF_USE=10"]
        CONST["NODE_FEATURE_DIM = 11\nNUM_EDGE_TYPES = 11\nFEATURE_SCHEMA_VERSION = v8"]
    end

    subgraph EXTRACTOR["graph_extractor.py — canonical implementation"]
        CFG["GraphExtractionConfig\nmulti_contract_policy (most_derived)\ninclude_edge_attr · solc_binary · allow_paths"]
        EXC["Exception hierarchy\nSolcCompilationError → HTTP 400\nSlitherParseError    → HTTP 500\nEmptyGraphError      → HTTP 400"]
        FN["extract_contract_graph(sol_path, config)\nreturns Data: x[N,11] · edge_index[2,E] · edge_attr[E]\nICFG-Lite edges (CALL_ENTRY, RETURN_TO)\nDEF_USE data-flow edges\nNever returns None — always raises on failure"]
    end

    AE["ast_extractor.py\nOffline batch · 11 parallel workers\nattaches: contract_path · y=0"]
    PP["preprocess.py — ContractPreprocessor\nOnline inference · one contract per request"]

    SCHEMA -->|"imports"| EXTRACTOR
    EXTRACTOR -->|"called by"| AE
    EXTRACTOR -->|"called by"| PP

    AE --> G[("ml/data/graphs/\n~41K .pt files (v8)")]
    PP --> API["Inference API\nPOST /predict"]

    style SCHEMA fill:#eef2ff,stroke:#818cf8,stroke-width:2px
    style EXTRACTOR fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
```

---

## Model Architecture (v7 Three-Eye)

```mermaid
flowchart LR
    SRC(["Solidity\nSource"])

    subgraph GNN["GNN Path — gnn_encoder.py (v8)"]
        direction TB
        GE2["graph_extractor.py\nextract_contract_graph()"]
        NF["x  [N, 11]\nnode features (v8)"]
        EA["edge_emb  Embedding(11,64)\nedge_attr[E] → [E, 64]"]
        P1["Phase 1 (Layers 1+2)\nEdges 0-5 · heads=8\nStructural aggregation"]
        P2["Phase 2 (Layers 3-5)\nEdges 6,8,9,10 · heads=1\nCFG + ICFG-Lite + DEF_USE"]
        P3["Phase 3 (Layers 6-7)\nEdge 7 (REVERSE_CONTAINS)\nheads=1 · Reverse aggregation"]
        JK["JK Attention\nLearned aggregation\nover 3 phase outputs"]
        GE2 --> NF & EA
        NF & EA --> P1 --> P2 --> P3 --> JK
    end

    subgraph TFM["Transformer Path — transformer_encoder.py"]
        direction TB
        TOK["CodeBERT tokenizer\ninput_ids [B,512] or [B,W,512]\nattn_mask [B,512] or [B,W,512]"]
        CB["CodeBERT  microsoft/codebert-base\n124M params  frozen\nFlash Attention 2"]
        LOR["LoRA  r=16  α=32\n~590K trainable params\nQ+V projections · all 12 layers"]
        TE["last_hidden_state\n[B, 512, 768] or [B, W*L, 768]\nall token positions"]
        WP["WindowAttentionPooler\nLearned attention over\nW window-CLS tokens → [B, 768]"]
        TOK --> CB
        LOR -. "adapts" .-> CB
        CB --> TE --> WP
    end

    subgraph FUS["CrossAttentionFusion — fusion_layer.py"]
        direction TB
        NP["node_proj  Linear(256→256)\n[N,256] → [N,256]"]
        TP["token_proj  Linear(768→256)\n[B,512,768] → [B,512,256]"]
        DB["_scatter_to_dense\n[B, max_n, 256] + node_real_mask"]
        N2T["Node→Token MHA\nQ=nodes  K=V=tokens\nmask PAD token positions\n→ enriched_nodes [B,n,256]"]
        T2N["Token→Node MHA\nQ=tokens  K=V=nodes\nmask padded node positions\n→ enriched_tokens [B,512,256]"]
        POL["Masked mean pool\npooled_nodes  [B,256]\npooled_tokens  [B,256]"]
        FOUT["concat → [B,512]\nLinear → ReLU → Dropout\n[B, 128]"]
        NP --> DB --> N2T & T2N
        TP --> N2T & T2N
        N2T & T2N --> POL --> FOUT
    end

    subgraph EYES["Three-Eye Classifier — sentinel_model.py"]
        direction TB
        GNN_POOL["GNN Eye\nFunction-level pool\nmax+mean → [B,256]\nLinear(256,128) → [B,128]"]
        TF_POOL["Transformer Eye\nWindow-pooled CLS → [B,768]\nLinear(768,128) → [B,128]"]
        FUSED_EYE["Fused Eye\nCrossAttention output → [B,128]"]
        CONCAT["cat([gnn_eye, tf_eye, fused_eye])\n[B, 384]"]
        CLS["Classifier\nLinear(384,192) → ReLU → Dropout\nLinear(192,10) → raw logits [B,10]"]
        AUX["Auxiliary Heads (training only)\nLinear(128,10) per eye\nλ=0.3 loss weighting"]
        GNN_POOL & TF_POOL & FUSED_EYE --> CONCAT --> CLS
        GNN_POOL & TF_POOL & FUSED_EYE --> AUX
    end

    PRED["Predictor._score\nsigmoid → probs [B,10]\nper-class thresholds\n→ vulnerabilities list\nlabel: vulnerable / safe"]

    SRC --> GNN & TFM
    JK -->|"node_embs [N,256]\nbatch [N]"| FUS
    WP -->|"[B, 768]"| FUS
    FOUT --> FUSED_EYE
    JK --> GNN_POOL
    WP --> TF_POOL
    CLS --> PRED

    style GNN fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style TFM fill:#fce7f3,stroke:#f472b6,stroke-width:2px
    style FUS fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    style EYES fill:#d1fae5,stroke:#34d399,stroke-width:2px
```

---

## DualPathDataset Loading Flow

```mermaid
flowchart TD
    subgraph INIT["DualPathDataset.__init__()"]
        direction TB
        DIR["graphs_dir + tokens_dir"]
        DISC["Discover .pt files\ngraph_hashes ∩ token_hashes\n= paired_hashes (sorted)"]
        IDX["Apply split indices\n(train/val/test .npy)\nif indices= provided"]
        CSV2["label_csv= (optional)\nbuild _label_map\nmd5_stem → float32[10]\nMulti-label mode"]
        CACHE["cache_path= (optional)\nload pickle → cached_data dict\nschema version validation\nrandom 10-hash integrity check"]
        VAL["validate=True\nload sample[0] eagerly\ncatch format issues at startup"]
        DIR --> DISC --> IDX
        CSV2 --> IDX
        CACHE --> VAL
        IDX --> VAL
    end

    subgraph ITEM["__getitem__(idx)"]
        direction TB
        H["hash_id = paired_hashes[idx]"]
        CCHECK{"cached_data\navailable?"}
        DISK["torch.load graph .pt\ntorch.load tokens .pt\n(weights_only=True)\nsafe globals registered"]
        MEM["read from\ncached_data dict"]
        ESHAPE["edge_attr shape guard\nsqueeze(-1) if ndim > 1\nnormalises old [E,1] → [E]"]
        TSHAPE["Validate token shapes\n[512] or [W, 512] accepted"]
        LBLCHECK{"label_csv\nprovided?"}
        MLBL["label = _label_map[hash_id]\nfloat32 [10]"]
        BLBL["label = graph.y\nlong [1]"]
        OUT["return (graph, tokens, label)"]
        H --> CCHECK
        CCHECK -->|yes| MEM
        CCHECK -->|no| DISK
        MEM & DISK --> ESHAPE --> TSHAPE --> LBLCHECK
        LBLCHECK -->|multi-label| MLBL
        LBLCHECK -->|binary| BLBL
        MLBL & BLBL --> OUT
    end

    subgraph COLL["dual_path_collate_fn(batch)"]
        direction TB
        BG["Batch.from_data_list(graphs)\nexclude metadata keys\nPyG batched graph"]
        BT["torch.stack input_ids\n[B,512] or [B,W,512]\ntorch.stack attention_mask"]
        BL["multi-label: [B,10] float32\nbinary:     [B]    long (squeeze)"]
        BG & BT & BL --> RET["return (Batch, token_dict, labels)"]
    end

    INIT --> ITEM --> COLL

    style INIT fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style ITEM fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
    style COLL fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
```
