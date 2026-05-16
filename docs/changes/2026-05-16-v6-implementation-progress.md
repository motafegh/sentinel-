# SENTINEL v6 — Implementation Progress Record
**Date:** 2026-05-16 (updated across two sessions)
**Plan reference:** `docs/changes/2026-05-16-v6-complete-plan.md`

This document tracks what has actually been implemented vs what the plan prescribes,
notes deviations and their rationale, and lists exact next commands to run.

---

## Status Summary

| Phase | Description | Code Status | Data Status |
|---|---|---|---|
| 0 — Feature schema v4 | All 4 graph feature bugs fixed | **COMMITTED** (commits `bef1f2a`, `310e738`) | **IN PROGRESS** — re-extraction running (PID 187896, ~23% @ 22:37) |
| 1 — Windowed tokenization | Model + dataset + tokenizer script | **COMMITTED** (commit `b38c9da`) | **NOT YET RUN** — retokenize pending |
| 2 — GNN arch (256 dim, 6 layers) | hidden_dim, depth, edge_emb, classifier, WindowAttentionPooler | **COMMITTED** (commit `2bb0e16`) | — |
| 3 — Training config (ASL, 100ep) | AsymmetricLoss, epochs=100, patience=30, LoRA LR 0.3× | **COMMITTED** (commit `64dfc5a`) | — |
| 4 — Data augmentation (DoS, Timestamp) | Clean single-label contracts | **NOT STARTED** | — |
| 5 — Re-extraction + cache rebuild | Run scripts after phase 0 + 1 code | **IN PROGRESS** | — |
| 6 — v6.0 training | Launch train.py with all fixes | **NOT STARTED** | — |

---

## Phase 0 — Feature Schema v4: COMPLETE

All four feature bugs fixed and committed. The extractor now produces v4 graphs.

### What changed in `ml/src/preprocessing/graph_extractor.py`

| Bug | What was wrong | Fix |
|---|---|---|
| `return_ignored` always 0 | `op.lvalue is None` is never True — Slither always creates TupleVariable | Check if `id(lval)` appears in any subsequent `op.read` across all IR ops |
| `Transfer`/`Send` invisible in ext_calls | Only `HighLevelCall`/`LowLevelCall` were counted | Added `Transfer, Send` to `_compute_external_call_count()` |
| `Transfer`/`Send` typed as CFG_READ | Priority check only looked for HL/LL calls | Added `Transfer, Send` to `_cfg_node_type()` priority-1 check |
| `block.timestamp` invisible | `SolidityVariableComposed` not in `state_variables_read` | New `_compute_uses_block_globals()` scans raw IR ops for `SolidityVariableComposed` |
| `loc` scale dominance | Raw loc=133+ vs binary features [0,1] — 2538× scale | `log1p(loc) / log1p(1000)`, range now [0, 1] |
| `pure` feature low-value | `pure=1` only for non-vulnerable functions — near-zero gradient | Replaced with `uses_block_globals` at feat[2] |

### What changed in `ml/src/preprocessing/graph_schema.py`

- `FEATURE_SCHEMA_VERSION` bumped from `"v3"` to `"v4"`
- `FEATURE_NAMES[2]` updated: `"pure"` → `"uses_block_globals"`
- Schema history and feature layout docs updated

### v4 feature vector (12 dimensions)

```
[0]  type_id / 12.0          — node type (0=CONTRACT … 12=CFG_NODE_RETURN)
[1]  visibility               — 0/1 (public=1)
[2]  uses_block_globals  ★   — 1.0 if func reads block.timestamp/number/etc. (was pure)
[3]  view                     — 0/1
[4]  payable                  — 0/1
[5]  complexity               — cyclomatic complexity, normalized
[6]  loc                 ★   — log1p(lines) / log1p(1000), range [0,1] (was raw count)
[7]  return_ignored      ★   — 1.0 if call return value not used in any subsequent op (was always 0)
[8]  call_target_typed        — 0/1 (typed interface vs dynamic call)
[9]  in_unchecked             — 0/1 (inside unchecked block)
[10] has_loop                 — 0/1
[11] external_call_count ★   — log1p(count)/log1p(20), now includes Transfer/Send (was 0 for ETH loops)
```
★ = changed from v3

---

## Phase 1 — Windowed Tokenization: CODE DONE, NOT COMMITTED

### Files modified (not yet committed)

- `ml/src/models/transformer_encoder.py` — `forward()` handles `[B, L]` and `[B, W, L]`
- `ml/src/models/sentinel_model.py` — `forward()` builds `flat_mask = [B, W*L]` for fusion
- `ml/src/datasets/dual_path_dataset.py` — shape validation accepts `[max_windows, 512]`
- `ml/scripts/train.py` — `--cache-path` flag added

### New files (not yet committed)

- `ml/scripts/retokenize_windowed.py` — windowed tokenizer producing `[max_windows, 512]` output

### How windowed mode works

```
Contract source code
        ↓
HuggingFace tokenizer (return_overflowing_tokens=True, stride=256)
        ↓
W raw windows, W ∈ [1, very large]
        ↓
_select_windows(): if W > max_windows, linspace sub-sample to cover start/middle/end
        ↓
Pad to exactly max_windows with zero-windows (attention_mask=0 on padding windows)
        ↓
Save: input_ids [max_windows, 512], attention_mask [max_windows, 512]

In model forward():
    TransformerEncoder: [B, max_windows, 512] → reshape [B*max_windows, 512] → CodeBERT
                        → [B*max_windows, 512, 768] → reshape [B, max_windows*512, 768]
    SentinelModel:      flat_mask = attention_mask.view(B, max_windows*512) for CrossAttentionFusion
                        transformer_eye = token_embs[:, 0, :] — CLS of window 0 (contract header)
```

### Why outputs `[max_windows, 512]` not `[W, 512]` (bug fix applied)

Variable W per contract would crash `torch.stack()` in the DataLoader collate function.
All contracts must have the same tensor shape for batching. The fix: always pad to `max_windows`
with zero-attention-mask windows. CrossAttentionFusion's `key_padding_mask` (from `attention_mask==0`)
correctly masks out padding windows so they contribute zero to cross-attention.

### Deviations from plan

| Plan says | What we implemented | Rationale |
|---|---|---|
| TransformerEncoder returns `[B, 768]` after WindowAttentionPooler | Returns `[B, W*L, 768]` — all tokens | More fine-grained: CrossAttentionFusion can attend to specific tokens in any window |
| `max_windows=8` | `max_windows=4` | 8GB VRAM limit: 4×512×B=8 = 16,384 positions; 8× would push VRAM limits |
| WindowAttentionPooler for transformer eye | Transformer eye uses CLS of window 0 only | Window 0 CLS covers pragma + contract opening; full WindowAttentionPooler is Phase 2 work |
| Token files overwrite `ml/data/tokens/` | Written to `ml/data/tokens_windowed/` | Keeps old single-window tokens available as fallback; cleaner separation |

### Still missing from Phase 1 (deferred to Phase 2 code)

- **WindowAttentionPooler for transformer eye**: The plan described pooling W window-CLS embeddings
  via learned attention. Currently transformer eye = CLS of window 0 only. Contracts where the
  relevant vulnerability appears in window 2+ will have weakened transformer eye signal. This
  should be added as part of Phase 2 architecture work.

---

## Phase 2 — GNN Architecture: COMMITTED (2bb0e16)

### Changes

| File | Change |
|---|---|
| `gnn_encoder.py` | hidden_dim 128→256, edge_emb_dim 32→64, num_layers 4→6 defaults; conv3b (2nd CF hop), conv4b (2nd RC hop); `_head_dim` 16→32 |
| `sentinel_model.py` | gnn_hidden_dim/num_layers/edge_emb defaults updated; classifier `Linear(384,10)` → `Sequential[384→192→ReLU→Dropout→10]`; added `self.window_pooler` |
| `transformer_encoder.py` | `WindowAttentionPooler` class added |
| `trainer.py` | MODEL_VERSION "v5.2"→"v6.0"; gnn_hidden_dim/gnn_layers/gnn_edge_emb_dim defaults; `gnn_layers > 6` warning (was `> 4`) |
| `train.py` | `--gnn-hidden-dim` 128→256, `--gnn-layers` 4→6, `--gnn-edge-emb-dim` 32→64 |

### WindowAttentionPooler (deviation from Phase 1)
Phase 1 left transformer eye as `token_embs[:, 0, :]` (CLS of window 0 only). Phase 2 implemented the WindowAttentionPooler:
- Extracts CLS from each window (position 0, 512, 1024, 1536 for max_windows=4)
- Single linear layer `[768→1]` produces per-window attention score
- Softmax over W windows → weighted sum → [B, 768]
- Single-window fallback: returns CLS directly (no allocation overhead)

### Why conv3b / conv4b (second hops)
v5.x only had 1 CF hop (conv3): CALL node → TMP variable node was the longest CFG path reachable.
With conv3b (2nd CF hop): CALL node → TMP → WRITE node, which is the CEI pattern (Check-Effect-Interaction).
conv4b (2nd RC hop): after two reverse-CONTAINS steps, the contract-level node carries phase-3 signal.

---

## Phase 3 — Training Config: COMMITTED (64dfc5a)

### Changes

| Item | Before | After |
|---|---|---|
| `losses.py` | did not exist | `AsymmetricLoss(gamma_neg, gamma_pos, clip)` — AMP-safe |
| `trainer.py` loss branches | bce / focal | bce / focal / **asl** |
| `trainer.py` epochs default | 60 | 100 |
| `trainer.py` patience default | 10 | 30 |
| `trainer.py` lora_lr_multiplier | 0.5 | 0.3 |
| `train.py` --loss-fn choices | bce, focal | bce, focal, **asl** |
| `train.py` --asl-* flags | absent | `--asl-gamma-neg`, `--asl-gamma-pos`, `--asl-clip` |

### ASL rationale
BCE treats all 440K (sample, class) cells equally. 85%+ are negative. ASL with gamma_neg=4
gives easy negatives (p≈0 but y=0) near-zero gradient weight, freeing capacity for the 15%
positive cells — especially the severely data-starved DoS class (377 train samples out of 31,092).

---

## Other Fixes Applied This Session

### `ml/scripts/reextract_graphs.py`
- Docstring `--check-edge-types 7` corrected to `--check-edge-types 8` (NUM_EDGE_TYPES=8, not 7)

### `ml/scripts/validate_graph_dataset.py`
- Usage example updated: `--check-edge-types 8` (was 7), `--check-dim 12` (correct)
- Comment updated: "use 8 for v5/v6 schema — REVERSE_CONTAINS=7 added"

---

## Exact Next Commands (Sequential Order)

Steps 1–7 complete. Steps 2–3 (re-extraction) currently running.

| Step | Status |
|---|---|
| 1. Commit Phase 1 (b38c9da) | ✅ DONE |
| 2. Re-extract graphs (PID 187896) | ⏳ RUNNING — ~23% @ 22:37 (est. ~45 min remaining) |
| 3. Validate graphs | ⏳ pending re-extraction |
| 4. Re-tokenize windowed | ⏳ pending validation |
| 5. Rebuild cache | ⏳ pending retokenization |
| 6. Commit Phase 2 (2bb0e16) | ✅ DONE |
| 7. Commit Phase 3 (64dfc5a) | ✅ DONE |
| 8. Launch v6 training | ⏳ pending cache |

---

### Step 3: Validate re-extracted graphs (run after PID 187896 exits)
```bash
source ml/.venv/bin/activate
PYTHONPATH=. python ml/scripts/validate_graph_dataset.py \
    --check-dim 12 \
    --check-edge-types 8 \
    --check-contains-edges \
    --check-control-flow
```
Gate: 0 validation errors, ghost graphs ≤ 100 (0.2%).

### Step 4: Re-tokenize with windowed tokenizer (~1–2 hours)
```bash
PYTHONPATH=. python ml/scripts/retokenize_windowed.py \
    --input ml/data/processed/multilabel_index_deduped.csv \
    --output ml/data/tokens_windowed \
    --max-windows 4 \
    --workers 11
```
Expected: 44,420 `.pt` files in `ml/data/tokens_windowed/`,
each with shape `input_ids=[4, 512]`, `attention_mask=[4, 512]`.

### Step 5: Rebuild cache for windowed tokens
```bash
PYTHONPATH=. python ml/scripts/create_cache.py \
    --graphs-dir ml/data/graphs \
    --tokens-dir ml/data/tokens_windowed \
    --label-csv ml/data/processed/multilabel_index_deduped.csv \
    --output ml/data/cached_dataset_windowed.pkl \
    --workers 8
```

### Step 8: Launch v6 training
All defaults are now baked in — these flags just override for clarity.
```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v6.0-20260517 \
    --experiment-name sentinel-v6 \
    --tokens-dir ml/data/tokens_windowed \
    --cache-path ml/data/cached_dataset_windowed.pkl \
    --loss-fn asl \
    --gradient-accumulation-steps 8 \
    --label-smoothing 0.05
```
All other v6 values (gnn_hidden_dim=256, gnn_layers=6, epochs=100, patience=30,
lora_lr_multiplier=0.3, asl_gamma_neg=4.0, asl_gamma_pos=1.0, asl_clip=0.05)
are now the TrainConfig defaults.

---

## Not-in-Plan Items Found

| Item | Where | Action |
|---|---|---|
| `--check-edge-types 7` stale (should be 8) | `reextract_graphs.py`, `validate_graph_dataset.py` | Fixed in this session |
| `--cache-path` CLI flag missing from `train.py` | `train.py` | Added in this session |
| Batch collation crash on variable W | `retokenize_windowed.py` | Fixed: always pad to `max_windows` |
| Windowed tokenizer writes to `tokens_windowed/` not `tokens/` | Design decision | Documented above |
| `FEATURE_SCHEMA_VERSION="v3"` in MEMORY.md | Memory file | Updated below |
| v5.3 "RUNNING" in MEMORY.md | Memory file | Updated below (KILLED epoch 47) |

---

## v6 Targets vs v5.2 Baselines

| Class | v5.2 Tuned F1 | v6.0 Target | Primary fix driving improvement |
|---|---|---|---|
| IntegerUO | 0.732 | ≥ 0.75 | Windowed tokens + hidden_dim=256 |
| GasException | 0.407 | ≥ 0.45 | 6-layer GNN + CF signal depth |
| Reentrancy | 0.322 | ≥ 0.40 | ASL + augmentation + 2nd CF layer |
| MishandledException | 0.342 | ≥ 0.50 | return_ignored fix (was always 0) |
| UnusedReturn | 0.238 | ≥ 0.45 | return_ignored fix (was always 0) |
| Timestamp | 0.174 | ≥ 0.30 | uses_block_globals + windowed tokens |
| DenialOfService | 0.329 | ≥ 0.35 | Transfer/Send fix + augmentation |
| CallToUnknown | 0.284 | ≥ 0.35 | Windowed tokenization |
| TOD | 0.283 | ≥ 0.30 | uses_block_globals + windowed |
| ExternalBug | 0.262 | ≥ 0.30 | Deeper GNN + cleaner data |
| **Macro avg** | **0.3422** | **≥ 0.43** | All fixes combined |

**Behavioral gates (primary pass/fail):**
- Detection rate ≥ 80% (v5.2: 36%)
- Safe specificity ≥ 80% (v5.2: 33%)
