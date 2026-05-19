# v8 Graph Extension — ICFG-Lite + DEF_USE + Full Re-Extraction

**Date:** 2026-05-19  
**Commits:** `47896a7` → `3a7b387` (8 commits)  
**Status:** Phase 1 + Phase 2 complete; v8 cache ready for training

---

## Summary

v7 training finished at best F1=0.2651 (epoch 23, +54.4% vs v6). JK weight analysis showed
Phase 2 weight decaying 0.35→0.18 over training, confirming the structural ceiling: the model
learned to distrust intra-function CFG signal because no cross-function edges existed. This
session implements the full v8 graph extension to break that ceiling.

---

## v7 Training Final Results (OPS-2 closed)

- **Best F1-macro: 0.2651** at epoch 23 (2026-05-19 05:49)
- **Per-class best:** IntegerUO=0.583 · GasException=0.301 · MishandledException=0.276 · TOD=0.228 · Timestamp=0.227 · DoS=0.019 (gradient zeroed)
- **Killed:** epoch 34, patience 10/30
- **Plateau pattern:** val F1 flat epochs 24–33 (0.2432–0.2618), step loss still declining (~0.14) — structural ceiling confirmed
- **JK weights at plateau (ep 33):** Phase1=0.050 · Phase2=0.182 · Phase3=0.768
- **Checkpoint:** `ml/checkpoints/v7.0_best.pt` (epoch 23, 258 MB)
- **Root cause of ceiling:** No cross-function CFG edges → model cannot trace reentrancy patterns spanning callee functions

---

## OPS-1 — Flash Attention

- flash-attn 2.8.3 installed via pre-built wheel (no compilation)
- CodeBERT (RoBERTa architecture) does NOT support `attn_implementation="flash_attention_2"` — only decoder models do
- `TransformerEncoder` correctly falls back to SDPA; this is the performance ceiling for RoBERTa
- flash-attn available for any future backbone swap to a decoder model

---

## Phase 1 — Extractor Refactor

### PLAN-1A — Validated `node.internal_calls` at CFG-node level

Confirmed Slither's `Node.internal_calls` returns callees at the specific CFG node (not whole
function). The IR-level fallback (`InternalCall` ops) is not needed.

### PLAN-1C — Accumulated CFG node maps across function iterations

`graph_extractor.py` extraction loop now accumulates three dicts before calling the new ICFG/DEF_USE helpers:
- `_func_entry_map: dict[str, int]` — canonical_name → entrypoint node index
- `_func_terminal_map: dict[str, list[int]]` — canonical_name → terminal node indices
- `_func_cfg_maps: dict[str, dict[Node, int]]` — canonical_name → local CFG node map

### PLAN-1D — `_add_icfg_edges()` — CALL_ENTRY(8) + RETURN_TO(9)

New helper emits cross-function CFG edges:
- **CALL_ENTRY(8):** calling CFG node → callee ENTRYPOINT node. Enables signal to flow from call site into callee body.
- **RETURN_TO(9):** callee terminal nodes → call-site successor nodes. Enables post-call state to flow back to caller.

Semantics validated on synthetic contract: 2 CALL_ENTRY + 1 RETURN_TO for a function with one internal call (second call has no successor → no RETURN_TO). Correct.

### PLAN-1E — `_add_def_use_edges()` — DEF_USE(10)

New helper emits intra-function data-flow edges:
- DEF: any IR op where `lvalue` is `LocalVariable` (not `TemporaryVariable` — intra-node SSA; not `StateVariable` — covered by READS/WRITES)
- USE: any IR op reading a variable name that appears in DEF set
- Deduplicates `(def_node, use_node)` pairs per function
- Keys by `lval.name` (consistent with BUG-M1 fix)

Validated: `newBal = balance + amount` (DEF at node 1) → `require(newBal > balance)` (USE at node 2) + `balance = newBal` (USE at node 3) → 2 correct DEF_USE edges emitted.

### PLAN-1F — Schema v8 (`graph_schema.py`)

```python
FEATURE_SCHEMA_VERSION = "v8"   # was "v7"
NUM_EDGE_TYPES         = 11     # was 8
EDGE_TYPES["CALL_ENTRY"] = 8    # NEW
EDGE_TYPES["RETURN_TO"]  = 9    # NEW
EDGE_TYPES["DEF_USE"]    = 10   # NEW
```

Import-time assertion `assert len(EDGE_TYPES) == NUM_EDGE_TYPES` catches any future mismatch.

### PLAN-1G — GNN encoder Phase 2 mask (`gnn_encoder.py`)

- `Embedding(8, 64)` → `Embedding(11, 64)` to cover all 11 edge types
- `cfg_mask` extended: `CONTROL_FLOW(6) | CALL_ENTRY(8) | RETURN_TO(9) | DEF_USE(10)`
- `add_self_loops=False` maintained for Phase 2 (directional signal preserved)

### PLAN-1B — 2,000-contract structural comparison gate

Script: `ml/scripts/validate_v8_extraction.py`

All 5 criteria passed:

| Criterion | Result |
|-----------|--------|
| Structural parity (legacy edges 0–6 bit-identical) | 1999/2000 PASS (1 Slither non-determinism, re-runs clean) |
| P99 edges per graph < 5,000 | P99=1,786 ✓ |
| Max edges per graph < 10,000 | max=3,707 ✓ |
| New types fire (non-zero counts) | CALL_ENTRY=12,630 · RETURN_TO=11,311 · DEF_USE=55,680 ✓ |
| DataLoader batch_size=8 | 8 graphs / 722 nodes batched cleanly ✓ |

---

## Phase 2 — Full Re-Extraction

### PLAN-2A — Archive v7 graphs
41,577 v7 graphs copied to `ml/data/archive/graphs_v7/`. Originals kept until overwritten.

### PLAN-2B — v8 re-extraction (`reextract_graphs.py`)
- **Command:** `python ml/scripts/reextract_graphs.py --multilabel-csv multilabel_index_cleaned.csv --workers 10`
- **Result:** 41,576 ok · 73 ghost (0.2%) · 2,875 skip (solc fail) · 0 fail
- **Time:** 29 minutes at 25.3 c/s (10 workers)
- **Note:** checkpoint from prior v7 run deleted first (prevented --resume from skipping all)

### PLAN-2C/D — Validation
- Node feature dim=11 on all sampled graphs ✓
- `edge_attr.max()=10` across all sampled graphs ✓
- New types 8, 9, 10 all present ✓

### PLAN-2E — Label cleaner on v8 graphs
`python ml/scripts/label_cleaner.py` (reads `multilabel_index_deduped.csv`, writes `multilabel_index_cleaned.csv`):
- **3,665 labels removed** (vs 17,722 in prior run — smaller delta because starting from already-cleaned CSV)
- Breakdown: UnusedReturn −1,665 · MishandledException −632 · Reentrancy −562 · Timestamp −423 · CallToUnknown −383

### PLAN-2F — Inject augmented DoS contracts
- 104 augmented graphs present (DoS pairs + augmented contracts)
- 6 augmented contracts fail: nested `interface` syntax inside contract body (invalid Solidity) — acceptable loss, 54/60 DoS pairs extract cleanly
- No new CSV rows injected (already present from prior run)

### PLAN-2G — Cache rebuild
- **Output:** `ml/data/cached_dataset_v8.pkl` (2.2 GB, 41,576 pairs, schema v8 embedded)
- **Skipped:** 2,948 stems (token present, graph missing — same contracts that fail compilation)

### PLAN-2H — Splits
Existing splits remain valid — same CSV order, same MD5 pairing:
- train=29,103 · val=6,236 · test=6,237 · total=41,576

### PLAN-2I — v8 Dataset Statistics
```
Graphs: 41,576 pairs
Nodes:  mean=125  P50=89   P99=623   max=1,735
Edges:  mean=248  P50=145  P99=1,801 max=6,516

Edge type distribution:
  CALLS(0)         :    437,968
  READS(1)         :    641,801
  WRITES(2)        :    678,879
  EMITS(3)         :         12  (rare)
  INHERITS(4)      :    105,010
  CONTAINS(5)      :  3,672,916
  CONTROL_FLOW(6)  :  3,140,025
  CALL_ENTRY(8)    :    257,829  ← NEW
  RETURN_TO(9)     :    232,814  ← NEW
  DEF_USE(10)      :  1,159,688  ← NEW
  REVERSE_CONTAINS : runtime-only (added by dataset, not stored)
```

---

## Phase 3 Preparation

### PLAN-3G — Fix stale `--run-name` default (`train.py:68`)
`"multilabel-v5-fresh"` → `"sentinel-v8"`

### Cache default updated (`train.py:113`)
`cached_dataset_deduped.pkl` → `cached_dataset_v8.pkl`

### Phase 2 ablation param (`gnn_encoder.py`, `sentinel_model.py`, `trainer.py`, `train.py`)
New `phase2_edge_types` / `gnn_phase2_edge_types` / `--phase2-edge-types` parameter:
- `None` (default) = all v8 types: CF(6) + CALL_ENTRY(8) + RETURN_TO(9) + DEF_USE(10)
- `[6, 8, 9]` = ICFG-only (PLAN-3A ablation)
- `[6, 10]` = DFG-only (PLAN-3B ablation)

Enables the three planned ablation runs without any code changes.

---

## Next Steps (Phase 3)

| Run | Edge types | Command flag | Purpose |
|-----|-----------|--------------|---------|
| v8-A | CF + CALL_ENTRY + RETURN_TO | `--phase2-edge-types 6 8 9 --run-name v8.0-A-20260519` | Isolate ICFG contribution |
| v8-B | CF + DEF_USE | `--phase2-edge-types 6 10 --run-name v8.0-B-20260519` | Isolate DFG contribution |
| v8-AB | CF + CALL_ENTRY + RETURN_TO + DEF_USE | *(default, no flag)* `--run-name v8.0-AB-20260519` | Joint effect |

Training command (same hyperparameters as v7):
```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. \
  python ml/scripts/train.py \
  --run-name v8.0-AB-20260519 \
  --epochs 100 \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --use-compile \
  2>&1 | tee ml/logs/v8.0-AB-20260519.log
```
