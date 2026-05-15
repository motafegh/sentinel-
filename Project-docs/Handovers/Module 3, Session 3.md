.
# SENTINEL — MASTER HANDOVER DOCUMENT
**Last Updated:** February 20, 2026 — Module 3, Session 3
**Covers:** Session 2 + Session 3 (merged)
**Author:** Ali Motafegh
**Next Session:** FusionLayer verification → SentinelModel → End-to-end test → M3.2 DONE

════════════════════════════════════════════════════

## ACTIVE MILESTONE
  Name:    Core Model — Full Dual-Path Pipeline
  Number:  M3.2
  Status:  75% complete
  Goal:    Build full dual-path model (GNN + Transformer + Fusion + Head)
           that takes a contract graph + tokens → vulnerability score [B]

════════════════════════════════════════════════════

## MODULE STATUS

| Module   | Status         | Notes                                                 |
| -------- | -------------- | ----------------------------------------------------- |
| Module 1 | ✅ Complete     | Slither analysis, 101,897 contracts                   |
| Module 2 | ✅ Complete     | 68,555 paired graph/token files                       |
| Module 3 | 🔄 In Progress | Session 3 complete — GNN + Transformer + Fusion built |

════════════════════════════════════════════════════

## CUMULATIVE WORK — WHAT IS BUILT

### ✅ Token Stats Diagnostic
File: ml/scripts/analyze_token_stats.py
Result:
  - Total scanned:   68,568 token files
  - Truncated:       66,201 (96.5%) — contracts hit 512-token limit
  - Not truncated:   2,367  (3.5%)  — median 320 tokens
  - Decision: keep truncation (expected, not a bug); head+tail optimization parked

### ✅ DataLoader + Custom Collate
File: ml/src/datasets/dual_path_dataset.py (modified)
Changes:
  - Added: `from torch_geometric.data import Data, Batch`
  - Added: `dual_path_collate_fn()` OUTSIDE the class at bottom
  - Fixed:  labels use `.squeeze(1)` not `.squeeze()`
Verified shapes (batch_size=4):
  - graphs.x:        [N, 8]
  - graphs.batch:    [N]
  - input_ids:       [4, 512]
  - attention_mask:  [4, 512]
  - labels:          [^4]

### ✅ GNNEncoder — DONE + VERIFIED
File: ml/src/models/gnn_encoder.py (NEW)
File: ml/src/models/__init__.py   (NEW, empty)
Architecture:
  GATConv(8→8,   heads=8, concat=True)   → [N, 64] + ReLU + Dropout(0.2)
  GATConv(64→8,  heads=8, concat=True)   → [N, 64] + ReLU + Dropout(0.2)
  GATConv(64→64, heads=1, concat=False)  → [N, 64]
  global_mean_pool(x, batch)             → [B, 64]
Verified:
  - Output [32, 64] on real DataLoader batch ✅
  - Labels distribution: 18 vulnerable / 14 safe (real class imbalance confirmed) ✅

### ✅ TransformerEncoder — DONE + VERIFIED
File: ml/src/models/transformer_encoder.py (NEW)
Architecture:
  AutoModel.from_pretrained("microsoft/codebert-base")
  All 199 params frozen (requires_grad = False)
  torch.no_grad() in forward()
  forward() → outputs.last_hidden_state[:, 0, :] → [B, 768]
Verified:
  - Frozen params: 199
  - Trainable params: 0
  - Output shape [4, 768] ✅

### FusionLayer — WRITTEN
File: ml/src/models/fusion_layer.py (NEW)
Architecture:
  torch.cat([gnn_out, transformer_out], dim=1)  → [B, 832]
  Linear(832→256) + ReLU + Dropout(0.3)
  Linear(256→64)  + ReLU
  Output: [B, 64]
Status: Code written.

### ❌ SentinelModel — NOT BUILT YET
File: ml/src/models/sentinel_model.py
Flow: graph + tokens
  → GNNEncoder        → [B, 64]
  → TransformerEncoder → [B, 768]
  → FusionLayer        → [B, 64]
  → Linear(64→1) + Sigmoid → [B] score

### ❌ End-to-End Test — NOT BUILT YET
File: ml/scripts/test_sentinel_model.py

════════════════════════════════════════════════════

## ALL FILES CHANGED ACROSS BOTH SESSIONS

### New Files
```

sentinel/
├── ml/
│   ├── src/
│   │   ├── datasets/
│   │   │   └── dual_path_dataset.py      ← MODIFIED (collate_fn + squeeze fix)
│   │   └── models/
│   │       ├── __init__.py               ← NEW (empty package marker)
│   │       ├── gnn_encoder.py            ← NEW ✅ VERIFIED
│   │       ├── transformer_encoder.py    ← NEW ✅ VERIFIED
│   │       └── fusion_layer.py           ← NEW ⚠️  NOT VERIFIED
│   └── scripts/
│       ├── analyze_token_stats.py        ← NEW (token diagnostic)
│       ├── test_dataloader.py            ← NEW (collate_fn shape check)
│       └── test_gnn_encoder.py           ← NEW ✅ (pipeline verification)

```

════════════════════════════════════════════════════

## KEY DECISIONS (ALL SESSIONS)

1. **Truncation: keep as-is, truncate from end**
   Reason: 96.5% truncation is expected baseline behavior.
   Head+tail optimization deferred until after baseline F1.

2. **Labels use .squeeze(1) not .squeeze()**
   Reason: .squeeze() with no arg breaks on batch_size=1 (removes all size-1 dims).
   .squeeze(1) only removes dim 1 — safe at all batch sizes.

3. **GNN: 3×GAT layers, global mean pool → [B, 64]**
   Reason: 3 hops covers most vulnerability patterns; mean pool avoids
   requiring prior knowledge of which node is dangerous.

4. **GAT: heads=8 + concat=True for layers 1–2, heads=1 + concat=False for layer 3**
   Reason: Multiple heads learn different relationship patterns in parallel;
   final layer collapses to a single clean embedding for the fusion layer.

5. **CodeBERT frozen (requires_grad=False + torch.no_grad() in forward)**
   Reason: 68K contracts too small to fine-tune 125M params without overfitting.
   LoRA fine-tuning parked until after baseline F1 is established.

6. **AutoModel (base), not AutoModelForSequenceClassification**
   Reason: We build our own fusion + classification head; ForSequenceClassification
   adds a head we would discard anyway.

7. **FusionLayer: concat + MLP, not GMU**
   Reason: GMU is the stretch goal from 01_MODULE_ML_CORE.md;
   simple concat+MLP is the MVP — establish baseline F1 first.

8. **FusionLayer output: [B, 64]**
   Reason: Matches GNN output dim; keeps classification head simple (Linear 64→1);
   consistent dim throughout the model tail.

════════════════════════════════════════════════════

## CONCEPTS TAUGHT (SESSIONS 2 + 3)

- GATConv: multi-head graph attention, how heads + concat control output dims
- global_mean_pool: why mean pooling over nodes (not single-node) for graph classification
- DataLoader collate_fn: why PyG graphs need a custom batching function
- Pretrained models: what they are and why we freeze them
- requires_grad=False vs torch.no_grad(): param-level vs operation-level gradient control
- [CLS] token: BERT's built-in sequence summary, always at position 0
- last_hidden_state[:, 0, :]: how to extract CLS from BERT output → [B, 768]
- Concatenation fusion: torch.cat([B,64],[B,768], dim=1) → [B,832], why dim=1 not dim=0
- MLP compression: why we compress 832→256→64 before classification head

════════════════════════════════════════════════════

## OPEN ISSUES — MUST ADDRESS BEFORE TRAINING

| Priority  | Issue                                                         | Action                                            |
| :-------- | :------------------------------------------------------------ | :------------------------------------------------ |
| 🟡 MEDIUM | Label verification: 24,113 folder-based labels (not from CSV) | Spot-check ~50 samples before full training run   |
| 🟢 LOW    | DVC setup: graphs/tokens/splits not versioned                 | `dvc init` + `dvc add` — do at end of any session |
| 🟢 LOW    | MLflow setup: needed before training experiments              | Set up in M3.3                                    |

════════════════════════════════════════════════════

## PARKED TOPICS

- Head+tail truncation for CodeBERT (first 256 + last 254 tokens) → after baseline F1
- LoRA fine-tuning for CodeBERT → after baseline F1
- GMU (Gated Multimodal Unit) replacing FusionLayer → stretch goal
- DVC versioning of graphs/tokens/splits → end of any session
- Optuna hyperparameter search → after baseline F1 below target

════════════════════════════════════════════════════

## DATA ASSETS

| Asset | Path | Format | Count |
| :-- | :-- | :-- | :-- |
| Graphs | ml/data/graphs/ | PyG .pt | 68,555 |
| Tokens | ml/data/tokens/ | dict .pt | 68,568 |
| Label Index | ml/data/processed/label_index.csv | CSV | 68,555 |
| Train Indices | ml/data/splits/train_indices.npy | numpy | 47,988 |
| Val Indices | ml/data/splits/val_indices.npy | numpy | 10,283 |
| Test Indices | ml/data/splits/test_indices.npy | numpy | 10,284 |

Label distribution: 35.7% safe / 64.3% vulnerable

════════════════════════════════════════════════════


## ARCHITECTURE REFERENCE — FULL DATA FLOW

```
DISK              DATASET           DATALOADER       MODEL
────              ───────           ──────────       ─────
68,555 graph.pt → DualPathDataset → DataLoader  →   GNNEncoder          ✅ VERIFIED
68,555 token.pt   lazy loading      batch_size=32    3×GAT + mean pool
                  indices filter    collate_fn        → [B, 64]
                  hash pairing           ↓
                                    batched_graphs    TransformerEncoder  ✅ VERIFIED
                                    batched_tokens →  CodeBERT frozen
                                    batched_labels    CLS token
                                                       → [B, 768]
                                                      ↓
                                                      FusionLayer         ⚠️ NOT VERIFIED
                                                      concat → [B, 832]
                                                      MLP → [B, 64]
                                                      ↓
                                                      SentinelModel       ❌ NOT BUILT
                                                      Classification Head
                                                      Linear(64→1) + Sigmoid
                                                      → [B] score

Model target: F1-macro ≥ 85%
Loss:         Focal Loss (γ=2, α=0.25)
Optimizer:    AdamW, lr=1e-4
```

════════════════════════════════════════════════════

*End of Master Handover — Covers Module 3 Sessions 2 + 3*
*M3.2 Status: 75% — next session closes it out completed 

