# M1 — ML Core: Visual Diagrams

Interactive Mermaid diagrams for `ml/README.md`. Rendered natively on GitHub.
For tensor-shape step-by-step flows, see the ASCII diagrams in `ml/README.md`.

---

## System Lifecycle

```mermaid
flowchart TD
    RAW["BCCC-SCsVul-2024\nraw .sol files + labels"]

    subgraph SHARED["Shared Preprocessing Layer"]
        GS["graph_schema.py\nNODE_TYPES · EDGE_TYPES · FEATURE_NAMES\nFEATURE_SCHEMA_VERSION · NODE_FEATURE_DIM=8"]
        GE["graph_extractor.py\nextract_contract_graph()"]
        GS --> GE
    end

    subgraph OFFLINE["① Offline Data Preparation"]
        AE["ast_extractor.py\n11 workers · solc version-pinned"]
        TK["tokenizer.py\nCodeBERT tokenizer"]
        BM["build_multilabel_index.py"]
        CS["create_splits.py · stratified"]
        GRAPHS[("graphs/\n~68K .pt\nx · edge_index · edge_attr")]
        TOKENS[("tokens/\n~68K .pt\ninput_ids · attn_mask")]
        CSV[("multilabel_index.csv\n68,523 rows × 10 classes")]
        SPLITS[("splits/\ntrain 47,966 · val 10,278\ntest 10,279")]
        AE --> GRAPHS
        TK --> TOKENS
        BM --> CSV --> CS --> SPLITS
    end

    subgraph TRAIN["② Training — scripts/train.py + MLflow"]
        DS["DualPathDataset\nbinary: graph.y\nmulti-label: multilabel_index.csv"]
        SM["SentinelModel\nGNN · LoRA · CrossAttention"]
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
    subgraph SCHEMA["graph_schema.py — single source of truth"]
        direction LR
        NT["NODE_TYPES\nCONTRACT=7 · STATE_VAR=0\nFUNCTION=1 · MODIFIER=2\nEVENT=3 · FALLBACK=4\nRECEIVE=5 · CONSTRUCTOR=6"]
        ET["EDGE_TYPES\nCALLS=0 · READS=1\nWRITES=2 · EMITS=3\nINHERITS=4"]
        CONST["NODE_FEATURE_DIM = 8\nNUM_EDGE_TYPES = 5\nFEATURE_SCHEMA_VERSION = v1"]
    end

    subgraph EXTRACTOR["graph_extractor.py — canonical implementation"]
        CFG["GraphExtractionConfig\nmulti_contract_policy · include_edge_attr\nsolc_binary · allow_paths"]
        EXC["Exception hierarchy\nSolcCompilationError → HTTP 400\nSlitherParseError    → HTTP 500\nEmptyGraphError      → HTTP 400"]
        FN["extract_contract_graph(sol_path, config)\nreturns Data: x[N,8] · edge_index[2,E] · edge_attr[E]\nNever returns None — always raises on failure"]
    end

    AE["ast_extractor.py\nOffline batch · 11 parallel workers\nattaches: contract_path · y=0"]
    PP["preprocess.py — ContractPreprocessor\nOnline inference · one contract per request"]

    SCHEMA -->|"imports"| EXTRACTOR
    EXTRACTOR -->|"called by"| AE
    EXTRACTOR -->|"called by"| PP

    AE --> G[("ml/data/graphs/\n~68K .pt files")]
    PP --> API["Inference API\nPOST /predict"]

    style SCHEMA fill:#eef2ff,stroke:#818cf8,stroke-width:2px
    style EXTRACTOR fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
```

---

## Model Architecture

```mermaid
flowchart LR
    SRC(["Solidity\nSource"])

    subgraph GNN["GNN Path — gnn_encoder.py"]
        direction TB
        GE2["graph_extractor.py\nextract_contract_graph()"]
        NF["x  [N, 8]\nnode features"]
        EA["edge_emb  Embedding(5,16)\nedge_attr[E] → [E, 16]"]
        C1["GATConv conv1\nin=8 → heads=8 → [N,64]\nReLU + Dropout"]
        C2["GATConv conv2\nin=64 → heads=8 → [N,64]\nReLU + Dropout"]
        C3["GATConv conv3\nin=64 → heads=1 → [N,64]"]
        GE2 --> NF & EA
        NF & EA --> C1 --> C2 --> C3
    end

    subgraph TFM["Transformer Path — transformer_encoder.py"]
        direction TB
        TOK["CodeBERT tokenizer\ninput_ids [B,512]\nattn_mask [B,512]"]
        CB["CodeBERT  microsoft/codebert-base\n124M params  frozen"]
        LOR["LoRA  r=8  α=16\n~295K trainable params\nQ+V projections · all 12 layers"]
        TE["last_hidden_state\n[B, 512, 768]\nall token positions"]
        TOK --> CB
        LOR -. "adapts" .-> CB
        CB --> TE
    end

    subgraph FUS["CrossAttentionFusion — fusion_layer.py"]
        direction TB
        NP["node_proj  Linear(64→256)\n[N,64] → [N,256]"]
        TP["token_proj  Linear(768→256)\n[B,512,768] → [B,512,256]"]
        DB["to_dense_batch\n[B, max_n, 256] + node_real_mask"]
        N2T["Node→Token MHA\nQ=nodes  K=V=tokens\nmask PAD token positions\n→ enriched_nodes [B,n,256]"]
        T2N["Token→Node MHA\nQ=tokens  K=V=nodes\nmask padded node positions\n→ enriched_tokens [B,512,256]"]
        POL["Masked mean pool\npooled_nodes  [B,256]\npooled_tokens [B,256]"]
        FOUT["concat → [B,512]\nLinear → ReLU → Dropout\n[B, 128]  ← LOCKED"]
        NP --> DB --> N2T & T2N
        TP --> N2T & T2N
        N2T & T2N --> POL --> FOUT
    end

    CLS["Classifier\nLinear(128, 10)\nraw logits [B, 10]\nno Sigmoid inside model"]
    PRED["Predictor._score\nsigmoid → probs [B,10]\nper-class thresholds\n→ vulnerabilities list\nlabel: vulnerable / safe"]

    SRC --> GNN & TFM
    C3 -->|"node_embs [N,64]\nbatch [N]"| FUS
    TE -->|"[B, 512, 768]"| FUS
    FOUT --> CLS --> PRED

    style GNN fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style TFM fill:#fce7f3,stroke:#f472b6,stroke-width:2px
    style FUS fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
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
        CSV2["label_csv= (optional)\nbuild _label_map\nmd5_stem → float32[10]"]
        CACHE["cache_path= (optional)\nload pickle → cached_data dict\nintegrity check on load"]
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
        DISK["torch.load graph .pt\ntorch.load tokens .pt\n(weights_only=True)"]
        MEM["read from\ncached_data dict"]
        ESHAPE["edge_attr shape guard\nsqueeze(-1) if ndim > 1\nnormalises old [E,1] → [E]"]
        LBLCHECK{"label_csv\nprovided?"}
        MLBL["label = _label_map[hash_id]\nfloat32 [10]"]
        BLBL["label = graph.y\nlong [1]"]
        OUT["return (graph, tokens, label)"]
        H --> CCHECK
        CCHECK -->|yes| MEM
        CCHECK -->|no| DISK
        MEM & DISK --> ESHAPE --> LBLCHECK
        LBLCHECK -->|multi-label| MLBL
        LBLCHECK -->|binary| BLBL
        MLBL & BLBL --> OUT
    end

    subgraph COLL["dual_path_collate_fn(batch)"]
        direction TB
        BG["Batch.from_data_list(graphs)\nPyG batched graph"]
        BT["torch.stack input_ids     [B,512]\ntorch.stack attention_mask [B,512]"]
        BL["multi-label: [B,10] float32\nbinary:     [B]    long (squeeze)"]
        BG & BT & BL --> RET["return (Batch, token_dict, labels)"]
    end

    INIT --> ITEM --> COLL

    style INIT fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style ITEM fill:#f0fdf4,stroke:#4ade80,stroke-width:2px
    style COLL fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
```
