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

    subgraph OFFLINE["Offline Data Preparation"]
        AE["reextract_graphs.py\nsolc version-pinned"]
        TK["retokenize_windowed.py\nCodeBERT tokenizer"]
        BM["build_multilabel_index.py"]
        CS["create_splits.py · stratified"]
        CC["create_cache.py"]
        GRAPHS[("graphs/\n~41K .pt (v8)\nx[N,11] · edge_index[2,E] · edge_attr[E]")]
        TOKENS[("tokens_windowed/\n~41K .pt\n[4, 512] windows, stride=256")]
        CSV[("multilabel_index.csv\n68,523 rows × 10 classes")]
        SPLITS[("splits/v10_deduped/\ntrain · val · test .npy")]
        CACHE[("cached_dataset_v10.pkl\npaired (graph, tokens)")]
        AE --> GRAPHS
        TK --> TOKENS
        BM --> CSV --> CS --> SPLITS
        GRAPHS & TOKENS & SPLITS --> CC --> CACHE
    end

    subgraph TRAIN["Training — train.py + MLflow"]
        DS["DualPathDataset\nmulti-label from CSV\nRAM cache support"]
        SM["SentinelModel v8.1\nFour-eye: GNN · Transformer · Fused · CFG\n8-layer GNN (2+3+3 phases)"]
        LOSS["AsymmetricLoss\n(γ⁻=2.0, γ⁺=1.0, clip=0.01)"]
        CKPT[("checkpoints/\nbest.pt + _thresholds.json")]
        DS --> SM --> LOSS --> CKPT
    end

    TT["tune_threshold.py\n0.05–0.95 grid per class"]
    OE["interpretability/\n21 experiment scripts"]
    MR["promote_model.py\nMLflow Staging → Production"]

    subgraph INFER["Inference API — src/inference/ — port 8001"]
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
    CACHE --> DS
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
        ET["EDGE_TYPES (11)\nCALLS=0 · READS=1 · WRITES=2 · EMITS=3\nINHERITS=4 · CONTAINS=5 · CONTROL_FLOW=6\nREVERSE_CONTAINS=7 (runtime only)\nCALL_ENTRY=8 · RETURN_TO=9\nDEF_USE=10"]
        CONST["NODE_FEATURE_DIM = 11\nNUM_EDGE_TYPES = 11\nFEATURE_SCHEMA_VERSION = v8"]
    end

    subgraph EXTRACTOR["graph_extractor.py — canonical implementation"]
        CFG["GraphExtractionConfig\nmulti_contract_policy (most_derived)\ninclude_edge_attr · solc_binary · allow_paths"]
        EXC["Exception hierarchy\nSolcCompilationError → HTTP 400\nSlitherParseError    → HTTP 500\nEmptyGraphError      → HTTP 400"]
        FN["extract_contract_graph(sol_path, config)\nreturns Data: x[N,11] · edge_index[2,E] · edge_attr[E]\nICFG-Lite edges (CALL_ENTRY, RETURN_TO)\nDEF_USE data-flow edges\nNever returns None — always raises on failure"]
    end

    AE["reextract_graphs.py\nOffline batch extraction"]
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

## Model Architecture (v8.1 Four-Eye)

```mermaid
flowchart LR
    SRC(["Solidity\nSource"])

    subgraph GNN["GNN Path — gnn_encoder.py (v8.1)"]
        direction TB
        GE2["graph_extractor.py\nextract_contract_graph()"]
        NF["x  [N, 11]\nnode features (v8)"]
        TE["type_embedding\nEmbedding(13, 16)\n→ [N, 16]"]
        CONCAT["concat\n[N, 27]"]
        EA["edge_emb  Embedding(11,64)\nedge_attr[E] → [E, 64]"]
        SKIP["IMP-G2: input_proj\nLinear(27, 256) skip"]
        P1["Phase 1 (Layers 1+2)\nEdges 0-5 · heads=8\nStructural aggregation"]
        P2["Phase 2 (Layers 3-5)\nEdges 6,8,9,10 · heads=4\nCF + ICFG-Lite + DEF_USE"]
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
        TOK["CodeBERT tokenizer\ninput_ids [B,512] or [B,W,512]\nattn_mask [B,512] or [B,W,512]"]
        CB["CodeBERT  microsoft/codebert-base\n124M params  frozen\nFlash Attention 2"]
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
        DB["_scatter_to_dense\n[B, 1024, 256] + node_real_mask"]
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
        CFG_EYE["CFG Eye (IMP-R7-2)\nPhase2 pool over CFG nodes [8-12]\nmax+mean → [B,512] → Linear → [B,128]"]
        CONCAT["cat([gnn_eye, tf_eye, fused_eye, cfg_eye])\n[B, 512]"]
        CLS["Classifier\nLinear(512,256) → ReLU → Dropout\nLinear(256,10) → raw logits [B,10]"]
        AUX["Auxiliary Heads (training only)\nLinear(128,10) per eye + aux_phase2\nλ=0.3 main, λ=0.20 phase2"]
        GNN_POOL & TF_POOL & FUSED_EYE & CFG_EYE --> CONCAT --> CLS
        GNN_POOL & TF_POOL & FUSED_EYE & CFG_EYE --> AUX
    end

    PRED["Predictor._score\nsigmoid → probs [B,10]\nper-class thresholds\n→ vulnerabilities list\nlabel: vulnerable / safe"]

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
    IN["Input\nx[N,27] · edge_index[2,E] · edge_attr[E]"]

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
        L3C["Layer 5 (conv3c)\nCF(6)+CE(8)+RT(9)+DU(10) joint\nheads=4 · integration layer"]
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

---

## Training Flow

```mermaid
flowchart TD
    subgraph SETUP["Setup"]
        CFG["TrainConfig\n8 batch · 8 accumulation = 64 effective\nASL loss · BF16 · submodule compile"]
        DS["DualPathDataset\ncached_dataset_v10.pkl\nWeightedRandomSampler (3× any-vuln)"]
        MDL["SentinelModel v8.1\nfour_eye_v8 · 8-layer GNN\nGNN prefix K=48 (warmup 15 ep)"]
        CFG --> DS --> MDL
    end

    subgraph LOOP["Training Loop (per epoch)"]
        direction TB
        FW["forward()\nGNN encoder → 4 eyes → classifier → logits"]
        AUX["Auxiliary loss\n0.3 × (aux_gnn + aux_tf + aux_fused)\n+ 0.20 × aux_phase2"]
        MAIN["Main loss\nASL(logits, labels)\n+ dos_loss_weight × DoS gradient"]
        JK_REG["JK entropy regularization\nλ=0.005 · pushes toward log(3)"]
        TOT["total_loss = main + aux + jk_reg"]
        BACKWARD["backward() · grad_clip=1.0\nBF16 autocast"]
        EVAL["eval_metrics()\nper-class F1 · AUROC · ECE\nthreshold tuning"]
        SAVE["checkpoint()\nbest.pt + _thresholds.json"]
        FW --> AUX --> TOT
        MAIN --> TOT
        JK_REG --> TOT
        TOT --> BACKWARD --> EVAL --> SAVE
    end

    subgraph CAL["Calibration"]
        TC["tune_threshold.py\n0.05–0.95 grid per class"]
        CT["calibrate_temperature.py\npost-training temperature scaling"]
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
        GE["extract_contract_graph()\nv8 graph .pt"]
        TK["retokenize_windowed()\n[4, 512] windows"]
        TMP --> GE --> TK
    end

    subgraph MODEL["SentinelModel.forward()"]
        GNN["GNN encoder\n8-layer three-phase GAT\n→ node_embs [N,256]"]
        TFM["Transformer encoder\nCodeBERT + LoRA\n→ token_embs [B,512,768]"]
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
    THR["per-class thresholds\n0.25-0.55 range"]
    TIER["Three-tier classification\nCONFIRMED ≥ 0.55\nSUSPICIOUS ≥ 0.25\nNOTEWORTHY < 0.25"]
    RESP["POST /predict response\nlabel · confirmed · suspicious\nprobabilities · thresholds"]

    REQ --> PRE --> MODEL --> SIG --> THR --> TIER --> RESP

    style PRE fill:#e0f2fe,stroke:#38bdf8,stroke-width:2px
    style MODEL fill:#fce7f3,stroke:#f472b6,stroke-width:2px
```
