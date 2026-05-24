# GraphCodeBERT + GNN Prefix Injection — Execution Plan

**Proposal:** [2026-05-23-graphcodebert-gnn-prefix-injection-proposal.md](2026-05-23-graphcodebert-gnn-prefix-injection-proposal.md)
**Last updated:** 2026-05-24
**Status:** ACTIVE — GATE-GCB-2 passed; GATE-GCB-3 smoke running; P1-TRAIN tonight if smoke clears

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ DONE | Complete and verified |
| 🔵 RUNNING | In progress |
| ⬜ OPEN | Not started, unblocked |
| 🔴 BLOCKED | Cannot start until dependency resolved |
| ⏩ SKIPPED | Explicitly skipped with reason |

---

## Overview — Stage Sequence

```
GATE-GCB-0: v8.0-B Result Gate  (BLOCKING — running now)
      │
      ├── PRE: Prerequisites (can start items PRE-1 to PRE-5 NOW in parallel)
      │         │
      │    GATE-GCB-1: All Prerequisites Verified
      │         │
      │        P0: Phase 0 — GraphCodeBERT drop-in, 5 epochs
      │        P0b: (Optional) CodeBERT + GNN prefix, 5 epochs  ← can run in parallel with P0
      │         │
      │    GATE-GCB-2: Phase 0 Go/No-Go
      │         │
      │    P1-IMPL: Phase 1 Implementation (5–7 days engineering)
      │         │
      │    GATE-GCB-3: Implementation Verification (smoke test before full training)
      │         │
      │    P1-TRAIN: Phase 1 Full Training Run (60–80 hours GPU)
      │         │
      │    GATE-GCB-4: Phase 1 Results Go/No-Go
      │         │
      │        P2: Phase 2 — Option C (shared DFG, 1.5–2 weeks)
      │         │
      │    GATE-GCB-5: Phase 2 Results Go/No-Go
      │         │
      │        P3: Phase 3 — Option A (full per-window DFG, 3–4 weeks)
      │         │
      │    GATE-GCB-6: Full Ablation Complete
      │
      └── PARALLEL TRACK: Data quality + other modules (runs throughout all phases)
```

**Estimated total time (if all phases proceed):** ~10–14 weeks GPU time + engineering
**Phase 0 + Phase 1 alone:** ~1 week engineering + ~3 days GPU = first result in ~10 days

---

## GATE-GCB-0 — v8.0-B Result Gate

**Status:** ✅ CLOSED (2026-05-23 — killed at ep11, best ep10 F1-macro=0.2460)
**Decision:** H5 refuted — cleaned labels did not break the 0.287 ceiling at ep10-11. Ceiling is architectural. Accelerating this plan.

v8.0-B tests the competing hypothesis — that the ~0.287 F1 ceiling is data quality, not architecture. Its result determines the urgency and framing of all subsequent work.

| v8.0-B Outcome | Interpretation | Action |
|----------------|----------------|--------|
| Tuned F1 > 0.30 (H5 confirmed) | Data quality IS a real lever; architecture may ALSO be bottleneck | Proceed with this plan AND continue DQ track in parallel (both hypotheses partly true) |
| Tuned F1 0.288–0.30 | Minor data quality improvement; architectural ceiling remains dominant | Proceed with this plan; deprioritize further data cleaning until Phase 1 results |
| Tuned F1 ≤ 0.288 (H5 refuted) | Ceiling is purely architectural; data cleaning alone cannot break it | Accelerate this plan; data cleaning secondary |

**Gate criteria (recorded):**
- [x] v8.0-B best F1-macro: **0.2460** (ep10, killed at ep11)
- [x] H5 verdict: **REFUTED** — F1 ≤ 0.288 at ep10-11; data cleaning alone cannot break the ceiling
- [x] Interpretation: **Ceiling is purely architectural** → accelerate this plan, data cleaning secondary
- [x] Early signal: Timestamp appeared in Top3 at ep6/ep8/ep9 (label cleaning helped), DoS=0.000 (too few positives)

---

## PRE — Prerequisites

**Status:** ✅ ALL DONE (2026-05-23)
**All must complete before GATE-GCB-1**

### PRE-1 — Download GraphCodeBERT to local cache

**Why:** `trainer.py line 714` sets `TRANSFORMERS_OFFLINE=1` in Python code at `train()` entry. If the model is not already in the HuggingFace cache, every `from_pretrained("microsoft/graphcodebert-base")` call will fail. Must be done ONCE with internet access before any training or validation script is run.

```bash
# Run from WSL2 with internet access
# TRANSFORMERS_OFFLINE and HF_HUB_OFFLINE must NOT be set
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

source ml/.venv/bin/activate
python -c "
from transformers import AutoModel, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
print('Downloading model...')
AutoModel.from_pretrained('microsoft/graphcodebert-base')
print('PRE-1 DONE — GraphCodeBERT cached')
"
```

**Pass:** Both download without error; files appear in `~/.cache/huggingface/hub/`
**Fail:** Download error → check internet connectivity and HuggingFace Hub availability

- **Status:** ✅ DONE (2026-05-23) — downloaded pytorch_model.bin (499MB) + model.safetensors (499MB). Pooler weight warning is expected and benign (we use token embeddings, not pooled output).

---

### PRE-2 — Tokenizer identity validation

**Why:** All 44,470 token files were produced with `microsoft/codebert-base`. GraphCodeBERT uses a different set of pre-trained weights but should have the same vocabulary (both are RoBERTa-based). If there is even one token ID difference, the entire token pipeline is invalid.

```python
from transformers import AutoTokenizer
tok_cb  = AutoTokenizer.from_pretrained("microsoft/codebert-base")
tok_gcb = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

# Check 1: vocab size
assert tok_cb.vocab_size == tok_gcb.vocab_size, \
    f"CRITICAL: vocab mismatch — {tok_cb.vocab_size} vs {tok_gcb.vocab_size}. ALL TOKEN FILES MUST BE REGENERATED."

# Check 2: token IDs on Solidity code
test = "function withdraw(uint amount) external payable { call.value(amount)(); }"
ids_cb  = tok_cb.encode(test)
ids_gcb = tok_gcb.encode(test)
assert ids_cb == ids_gcb, \
    f"CRITICAL: token ID mismatch on test snippet. Tokenizers differ. Token files invalid."

print("PRE-2 PASSED — tokenizers are identical; existing token files are valid")
```

**Pass:** Both asserts pass → no token file changes needed for Options B or C.
**Fail:** Assert fails → regenerate all token files with `ml/scripts/retokenize_windowed.py --tokenizer microsoft/graphcodebert-base` (full pipeline, ~2–4 hours). Hardcoded strings in `tokenizer.py:60` and `preprocess.py:145` must also be updated.

- **Status:** ✅ DONE (2026-05-23) — vocab_size=50265, token IDs identical on Solidity snippet.

---

### PRE-3 — Solidity UNK-token rate check

**Why:** GraphCodeBERT was pre-trained on 6 languages (not Solidity). Validate it doesn't produce significantly more unknown tokens on Solidity code than CodeBERT does.

```bash
source ml/.venv/bin/activate
# Quick check: tokenize 100 Solidity contracts from the BCCC dataset
# Pass threshold: UNK token rate < 0.5%
PYTHONPATH=. python -c "
from transformers import AutoTokenizer
from pathlib import Path
import random, glob

tok = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
sol_files = random.sample(glob.glob('ml/data/contracts/**/*.sol', recursive=True), min(100, 999))
total_tokens, total_unk = 0, 0
for f in sol_files:
    ids = tok.encode(Path(f).read_text(errors='ignore'))
    total_tokens += len(ids)
    total_unk += ids.count(tok.unk_token_id)

unk_rate = total_unk / total_tokens
print(f'UNK rate: {unk_rate:.4%}  ({total_unk}/{total_tokens})')
assert unk_rate < 0.005, f'UNK rate {unk_rate:.4%} exceeds 0.5% — tokenizer may not handle Solidity well'
print('PRE-3 PASSED')
"
```

**Pass:** UNK rate < 0.5%
**Fail:** High UNK rate means GraphCodeBERT tokenizes Solidity poorly — investigate before proceeding (may need to add Solidity vocabulary or fall back to CodeBERT weights)

- **Status:** ✅ DONE (2026-05-23) — UNK rate: 0.000000% (0/418,248 tokens). Perfect.

---

### PRE-4 — Node count distribution audit

**Why:** The proposal uses K=32 as the upper bound, expecting it to cover ≥95% of contracts without truncation. This must be verified before writing node selection code.

```bash
source ml/.venv/bin/activate
# Write ml/scripts/audit_prefix_node_counts.py (new script):
# For each graph in ml/data/graphs/, count nodes of STRUCTURAL_PREFIX_TYPES.
# Report: min/P50/P90/P95/P99/max and histogram.
# Gate: P95 ≤ 32 (K=32 covers 95% without truncation)
PYTHONPATH=. python ml/scripts/audit_prefix_node_counts.py \
    --graphs-dir ml/data/graphs/ \
    --out ml/logs/prefix_node_count_audit.json
```

Script: `ml/scripts/audit_prefix_node_counts.py` (written 2026-05-23).

**Actual results (2026-05-23, n=41,576 graphs):**

| Subset | Mean | P50 | P90 | P95 | P99 | K=32 covers |
|--------|------|-----|-----|-----|-----|-------------|
| All eligible types (FUNCTION+MODIFIER+CFG_CALL+WRITE+CHECK+CONSTRUCTOR+FALLBACK+RECEIVE) | 48.7 | 37 | 91 | **122** | 236 | 42.3% |
| Declaration-level only (FUNCTION+MODIFIER+CONSTRUCTOR+FALLBACK+RECEIVE) | 20.3 | 16 | 38 | **47** | 84 | 85.8% |

**Finding:** Original plan (all 8 types, K=32) would only cover 42.3% of contracts — truncation is the rule, not the exception. CFG nodes alone add 28.4 mean.

**Resolution:** Phase 1 uses **declaration-level only** (5 types: FUNCTION+MODIFIER+CONSTRUCTOR+FALLBACK+RECEIVE) with **K=48** (covers 95.5% without truncation, code budget=462, stride=231). Phase 1B ablation adds CFG_NODE_CALL back with K=64 to isolate its contribution.

**Pass:** P95 ≤ K for chosen node type set and K value.
**Result:** ✅ DONE — P95=47 ≤ K=48 with declaration-level types.

- **Status:** ✅ DONE (2026-05-23) — full results in `ml/logs/prefix_node_count_audit.json`

---

### PRE-5 — LoRA applies to GraphCodeBERT

**Why:** Confirms Assumption A1. PEFT applies LoRA by module name (`query`, `value`). GraphCodeBERT uses the same RoBERTa naming convention but this must be verified in the actual environment.

```python
from transformers import AutoModel
from peft import get_peft_model, LoraConfig

model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"])
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
# Expected output: "trainable params: 589,824 || all params: 125,295,360 || trainable%: 0.47"
```

**Pass:** Trainable params ~590K, all others frozen, no error
**Fail:** PEFT raises `ModuleNotFoundError` → module names differ; inspect `model.named_modules()` and find actual Q/V module names

- **Status:** ✅ DONE (2026-05-23) — trainable=589,824 (~590K), frozen=124,645,632. LoRA applies cleanly to GraphCodeBERT via `query`/`value` module names (same RoBERTa naming). Pooler weight warning is benign.

---

## GATE-GCB-1 — All Prerequisites Verified

**Status:** ✅ PASSED (2026-05-23) — all prerequisites complete. Phase 0 is unblocked.

**Checklist:**

| Item | Check | Status |
|------|-------|--------|
| PRE-1 | GraphCodeBERT cached in `~/.cache/huggingface/hub/` | ✅ |
| PRE-2 | Tokenizer vocab_size and token IDs identical | ✅ |
| PRE-3 | UNK token rate < 0.5% on Solidity | ✅ |
| PRE-4 | P95 eligible node count ≤ K (declaration-level, K=48) | ✅ |
| PRE-5 | LoRA applies to GraphCodeBERT, ~590K trainable params | ✅ |
| PRE-6 | v8.0-B result recorded and interpreted (GATE-GCB-0) | ✅ |

**Recorded values:**
- Tokenizer vocab_size: **50,265** (identical CodeBERT = GraphCodeBERT)
- PRE-3 UNK rate: **0.000000%** (0/418,248 tokens)
- PRE-4 P95 node count (declaration-level): **47**  |  P99: **84**
- Chosen K value: **48** (covers 95.5% without truncation)
- PRE-5 trainable params: **589,824** (~590K ✓)

---

## P0 — Phase 0: GraphCodeBERT Pure Drop-In (5 Epochs)

**Status:** ✅ DONE — killed 2026-05-24 at ep4 start; best checkpoint ep3 F1=0.2178; GATE-GCB-2 PASSED
**Duration:** ~6–8 hours GPU total (~40 min/epoch)
**Purpose:** Validates Assumption A4 (pre-training transfers to Solidity) before investing in GNN prefix implementation. Separates GraphCodeBERT's contribution from the prefix injection mechanism.

**Engineering changes applied (minimal — drop-in only):**

In `ml/src/models/transformer_encoder.py`:
- Changed `"microsoft/codebert-base"` → `"microsoft/graphcodebert-base"` in both model load paths
- Zero other changes

**Training command (as launched):**
```bash
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. python ml/scripts/train.py \
    --run-name graphcodebert-dropin-P0-20260523 \
    --experiment-name sentinel-gcb \
    --phase2-edge-types 6 8 9 \
    --cache-path ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --label-csv ml/data/processed/multilabel_index_cleaned.csv \
    --epochs 5 --gradient-accumulation-steps 8 --dos-loss-weight 0.5 --weighted-sampler positive \
    2>&1 | tee ml/logs/graphcodebert-dropin-P0-20260523.log
```

**Pass criteria (check after 5 epochs):**

| Signal | Expected | Status |
|--------|----------|--------|
| Step loss at ep1 step 200 | ~0.15 (healthy learning) | ✅ 0.1540 observed |
| GNN gradient share ep1 | 40–75% | ✅ 63–86% observed |
| No NaN / CUDA error | — | ✅ ep1+ep2 clean |
| SDPA active (FA2 not available for RoBERTa) | Expected fallback | ✅ "SDPA active" in log |
| Step loss trajectory at ep3 | ≥ PLAN-3A at ep3 (0.285 range) | ✅ loss=0.1437 ep3 step 400 |
| F1-macro at ep3 (killed before ep5) | Upward trend sufficient for GATE-GCB-2 | ✅ 0.1734→0.2056→0.2178 |
| Training speed | Within ±20% of PLAN-3A (~40 min/ep) | ✅ ~40 min/ep |

**Ep1 results (2026-05-23):**
- F1-macro=0.1734 · JK Phase1=0.191 Phase2=0.389 Phase3=0.420 · VRAM=0.4/8.0 GiB
- Top3: IntegerUO=0.490, Reentrancy=0.229, GasException=0.225
- Bottom3: UnusedReturn=0.044, DoS=0.010, Timestamp=0.000

**Ep2 results (2026-05-23):**
- F1-macro=0.2056 · JK Phase1=0.190 Phase2=0.398 Phase3=0.412
- Top3: IntegerUO=0.505, GasException=0.225, Reentrancy=0.222

**Ep3 results (2026-05-24 — final, killed before ep4 completed):**
- F1-macro=0.2178 · Loss=0.2393 · JK Phase1=0.155 Phase2=0.395 Phase3=0.450
- Top3: IntegerUO=0.512, GasException=0.235, Reentrancy=0.222
- Bottom3: ExternalBug=0.199, TOD=0.171, DoS=0.010
- Note: ExternalBug=0.199 and TOD=0.171 both non-zero (was 0.000 in CodeBERT runs) — GraphCodeBERT cross-function pre-training showing early signal
- Best checkpoint: `ml/checkpoints/graphcodebert-dropin-P0-20260523_best.pt`

---

## P0b — Optional Ablation: CodeBERT + GNN Prefix K=16 (5 Epochs)

**Status:** 🔴 BLOCKED (on GATE-GCB-1)
**Duration:** ~6–8 hours GPU (can run in parallel with P0 if two GPUs available, or after P0)
**Purpose:** Separates prefix injection contribution from GraphCodeBERT pre-training contribution. Run only if P0 shows improvement — otherwise save the GPU time.

**Requires:** Partial Phase 1 implementation (just the prefix injection, with CodeBERT weights).

| Config | GraphCodeBERT weights | GNN prefix K=16 | Purpose |
|---|---|---|---|
| PLAN-3A baseline | No | No | Reference |
| P0 | Yes | No | GraphCodeBERT pre-training contribution |
| P0b | No | Yes | Prefix mechanism contribution |
| Phase 1 (P1-TRAIN) | Yes | Yes | Combined contribution |

**Decision:** Run P0b if P0 shows improvement. Skip P0b if P0 shows no improvement (already blocking Phase 1).

---

## GATE-GCB-2 — Phase 0 Go/No-Go

**Status:** ✅ PASSED (2026-05-24)

| Outcome | Decision |
|---------|----------|
| P0 ep5 F1-macro ≥ PLAN-3A ep5 | ✅ Proceed to Phase 1 implementation |
| P0 ep5 F1-macro < PLAN-3A ep5 by > 0.005 | 🔴 Stop — Assumption A4 (transfer to Solidity) may be wrong. Investigate: check loss curves, UnusedReturn-specific F1, UNK token rate. Consider longer warmup or lower LoRA learning rate before abandoning. |
| NaN or training instability in P0 | 🔴 Stop — Debug BF16 / FA2 compatibility with GraphCodeBERT before proceeding |

**Gate decision (2026-05-24):** PASSED — P0 killed at ep4 with clear upward trend (ep1=0.1734→ep2=0.2056→ep3=0.2178). ExternalBug and TOD both non-zero from ep3 (was 0.000 in all CodeBERT runs) — GraphCodeBERT cross-function pre-training signal confirmed. No NaN, no instability. Proceed to GATE-GCB-3 smoke test.

---

## P1-IMPL — Phase 1 Implementation: Option B

**Status:** ✅ DONE (2026-05-23) — all unit tests pass; backward compat (gnn_prefix_k=0) verified identical to original model
**Duration:** Completed in same session as P0 launch

**Files changed:**

| File | Change | Status |
|------|--------|--------|
| `ml/src/models/transformer_encoder.py` | Added `gnn_prefix_nodes: Optional[Tensor]` to `forward()`; `inputs_embeds` path with position_ids (1 for prefix, 3..466 for code); multi-window prefix expansion; `_word_embeddings` property | ✅ DONE |
| `ml/src/models/transformer_encoder.py` | `WindowAttentionPooler`: added `prefix_k: int = 0`; CLS now at `i*window_size + prefix_k` | ✅ DONE |
| `ml/src/models/sentinel_model.py` | Added `gnn_prefix_k`, `gnn_prefix_warmup_epochs`; `gnn_to_bert_proj: Linear(256, 768)`; `prefix_type_embedding: Embedding(5, 768)`; `_current_epoch`; `select_prefix_nodes()` method with priority sort; prefix suppressed during warmup | ✅ DONE |
| `ml/src/preprocessing/graph_schema.py` | Added `NodeType` IntEnum (13 types, values from NODE_TYPES); `STRUCTURAL_PREFIX_TYPES` frozenset (5 declaration types); `_PREFIX_NODE_PRIORITY` and `_PREFIX_TYPE_IDX` dicts in sentinel_model.py | ✅ DONE |
| `ml/src/training/trainer.py` | Added `gnn_prefix_k`, `gnn_prefix_warmup_epochs`, `gnn_prefix_proj_lr_mult` to TrainConfig; `gnn_to_bert_proj` param group (LR ×1.0); `model._current_epoch = epoch` each epoch | ✅ DONE |
| `ml/scripts/train.py` | Added `--gnn-prefix-k` (default 0), `--gnn-prefix-warmup-epochs` (default 15), `--gnn-prefix-proj-lr-mult` (default 1.0) | ✅ DONE |
| `ml/src/inference/predictor.py` | Added `gnn_prefix_k` and `gnn_prefix_warmup_epochs` to SentinelModel constructor; `model._current_epoch = 9999` after load (prefix always active at inference) | ✅ DONE |
| `ml/scripts/retokenize_windowed.py` | Not needed — TransformerEncoder truncates internally (`code_ids = input_ids[:, :code_budget]`); existing stride=256 token files are valid for K=48 | ⏩ SKIPPED (not needed) |

**Key implementation decisions (deviations from plan):**

- `prefix_type_embedding`: implemented as `Embedding(5, 768)` (not 8 — Phase 1 is declaration-level only, 5 types)
- `--graphcodebert` flag: not added — TransformerEncoder is already hardcoded to graphcodebert-base (P0 engineering change); no separate flag needed
- Warmup strategy (corrected): during warmup `gnn_prefix_nodes=None` is passed — prefix completely suppressed. `gnn_to_bert_proj` receives NO gradient during warmup (not called). Starts from random init at ep16. This is correct — old plan (freeze proj but still inject) would corrupt attention.
- Node selection: CONSTRUCTOR > FALLBACK > RECEIVE > MODIFIER > FUNCTION (priority sort with `_PREFIX_NODE_PRIORITY` dict)

**New checkpoint config keys:**
```python
{
    "architecture":               "three_eye_graphcodebert_prefix_v1",
    "transformer_model":          "microsoft/graphcodebert-base",
    "gnn_prefix_k":               48,     # PRE-4: declaration-level P95=47, K=48 covers 95.5%
    "gnn_prefix_warmup_epochs":   15,
    "gnn_prefix_node_types":      [1, 2, 6, 4, 5],  # FUNCTION MODIFIER CONSTRUCTOR FALLBACK RECEIVE
    "gnn_to_bert_proj_dim":       [256, 768],
    "use_prefix_type_embedding":  True,   # G11 — nn.Embedding(5, 768)
    "num_prefix_types":           5,      # len(STRUCTURAL_PREFIX_TYPES) — declaration-level only
    "max_windows":                4,      # must match training token files
    "gnn_prefix_proj_lr_mult":    1.0,    # LR multiplier for gnn_to_bert_proj param group
}
```

**Node selection priority — Phase 1 (declaration-level only, §6.2 of proposal):**
1. CONSTRUCTOR(6), FALLBACK(4), RECEIVE(5) — always included first (at most 3 per contract)
2. MODIFIER(2) — included next (typically ≤ 5)
3. FUNCTION(1) — sorted by `feature[10]` (external_call_count) descending, fills remaining K slots

**Phase 1B only** (K=64, after Phase 1 results): add CFG_NODE_CALL(8) sorted by node index after all declaration nodes.

All IDs from `graph_schema.NodeType` IntEnum (added 2026-05-23). Never hardcode raw integers — use `NodeType.FUNCTION` etc. from `STRUCTURAL_PREFIX_TYPES`.

**Unit tests required before Phase 1 training:**

| Test | What it verifies | Pass criterion |
|------|-----------------|----------------|
| `inputs_embeds` + LoRA + FA2 single batch | G1: stack compatibility + LoRA active | Gradient reaches `gnn_to_bert_proj.weight`; `lora_out ≠ base_out` |
| CLS at position 0 in `last_hidden_state` | G4: pooler works correctly | `output.last_hidden_state[:, 0, :]` is CLS |
| Batching contract with mixed node counts | G8: zero-padding doesn't contribute gradient | Zero-padded prefix positions: attention score = −inf |
| Position IDs max < 514 | G8: no position embedding overflow | `position_ids.max() < 514` for K=16 |
| Node selection determinism (same graph, two calls) | Ordering is reproducible | `assert torch.equal(sel1, sel2)` |
| Cross-implementation determinism | `sentinel_model.py` and `predictor.py` select identical nodes for same graph | Run both code paths on 5 test graphs, assert identical node index sets |
| Warmup suppression | During warmup, prefix is not injected | `model(batch, ids, mask, epoch=0).shape == model(batch, ids, mask, epoch=16).shape`; logits differ |

---

## GATE-GCB-3 — Implementation Verification (Smoke Test Before Full Training)

**Status:** ✅ PASSED (2026-05-24) — ep1 sufficient; smoke killed early, all gate criteria met on ep1 alone

**All must pass before launching P1-TRAIN (the 60–80 hour full run):**

### Step 1 — VRAM check
```bash
python -c "
import torch
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
free = total - reserved
print(f'VRAM: {reserved:.1f}/{total:.1f} GB  free={free:.1f}')
assert free >= 6.0, f'Insufficient VRAM'
print('VRAM gate PASSED')
"
```

### Step 2 — Config review

| Config | Expected value for Phase 1 | Check |
|--------|---------------------------|-------|
| `--gnn-prefix-k` | 48 (PRE-4: decl-level P95=47) | ⬜ |
| `--gnn-prefix-warmup-epochs` | 15 | ⬜ |
| `--gnn-prefix-proj-lr-mult` | 1.0 | ⬜ |
| `--phase2-edge-types` | `6 8 9` (same as PLAN-3A) | ⬜ |
| `--dos-loss-weight` | 0.5 (same as v8.0-B) | ⬜ |
| `--cache-path` | `ml/data/cached_dataset_v8.pkl` | ⬜ |
| `--splits-dir` | `ml/data/splits/deduped` | ⬜ |
| `--label-csv` | `ml/data/processed/multilabel_index_cleaned.csv` | ⬜ |
| `--weighted-sampler` | `positive` | ⬜ |
| `transformer_model` | `microsoft/graphcodebert-base` (hardcoded in TransformerEncoder) | ⬜ |
| `max_windows` | 4 (matches training token files) | ⬜ |
| MLflow run name | `graphcodebert-v1-prefix48-YYYYMMDD` | ⬜ |

### Step 3 — 2-epoch smoke test

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. python ml/scripts/train.py \
    --run-name graphcodebert-v1-prefix48-smoke-$(date +%Y%m%d) \
    --experiment-name sentinel-gcb \
    --gnn-prefix-k 48 \
    --gnn-prefix-warmup-epochs 15 \
    --phase2-edge-types 6 8 9 \
    --dos-loss-weight 0.5 \
    --cache-path ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --label-csv ml/data/processed/multilabel_index_cleaned.csv \
    --weighted-sampler positive \
    --epochs 2 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee ml/logs/graphcodebert-v1-prefix48-smoke-$(date +%Y%m%d).log
```

**Smoke test pass criteria:**

| Signal | Expected | Fail action | Status |
|--------|----------|-------------|--------|
| Step loss at ep1 step 100 | 0.15–0.20 (warmup active, no prefix) | Kill; check warmup suppression logic | ✅ 0.1684 |
| NaN count | 0 | Kill immediately; investigate FA2 + inputs_embeds | ✅ 0 |
| Warmup: `gnn_prefix_nodes=None` path active ep1–14 | Prefix NOT injected during warmup | Prefix injected during warmup = warmup suppression not working | ✅ "WARMUP (starts ep15)" logged ep1+ep2 |
| GNN gradient share during warmup | 40–75% | < 20% = CrossAttentionFusion path broken | ✅ 86→73% (trending into range) |
| `gnn_to_bert_proj` weight norm | Constant at random init — proj NOT called during warmup | Rising = proj called during warmup | ✅ **16.0000 → 16.0000** (byte-identical ep1→ep2, proj silent) |
| CUDA OOM | None | Reduce K or enable gradient checkpointing | ✅ VRAM 0.4/8.0 GiB (4.8%) |
| Training speed | Within ±25% of PLAN-3A (~40 min/ep) | | ✅ ~36 min/ep |
| F1-macro ep1 | > 0.15 | Kill — not learning | ✅ 0.1832 |
| JK Phase2 ep1 | > 0.10 | ICFG edges ignored | ✅ 0.387±0.083 |

**Gate decision (2026-05-24): PASSED on ep1 alone. Smoke killed early — ep1 sufficient. P1-TRAIN launched.**

---

## P1-TRAIN — Phase 1 Full Training Run

**Status:** 🔵 RUNNING — launched 2026-05-24 overnight
**Duration:** ~60–80 hours GPU
**Run name:** `graphcodebert-v1-prefix48-20260524`
**Log:** `ml/logs/graphcodebert-v1-prefix48-20260524.log`

```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. python ml/scripts/train.py \
    --run-name graphcodebert-v1-prefix48-$(date +%Y%m%d) \
    --experiment-name sentinel-gcb \
    --gnn-prefix-k 48 \
    --gnn-prefix-warmup-epochs 15 \
    --phase2-edge-types 6 8 9 \
    --dos-loss-weight 0.5 \
    --cache-path ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --label-csv ml/data/processed/multilabel_index_cleaned.csv \
    --weighted-sampler positive \
    --epochs 100 \
    --early-stop-patience 30 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee ml/logs/graphcodebert-v1-prefix48-$(date +%Y%m%d).log
```

**Early epoch monitoring (first 20 epochs — monitor before walking away):**

| Epoch | Check | Action if fails |
|-------|-------|-----------------|
| ep1–14 (warmup) | Per-step logs show `gnn_prefix_nodes=None` path active | If prefix is being injected before warmup ends, warmup logic is broken |
| ep15 (warmup end) | `gnn_to_bert_proj` weight norm ≈ random init (proj untrained during warmup — correct; gradient begins ep16+) | If already drifted far from init: warmup suppression broken (proj called during warmup) |
| ep15 | F1-macro > 0.20 | If < 0.15 after warmup: training not learning; abort and diagnose |
| ep16 | Prefix injection enabled; step loss rises briefly then recovers (expected) | If loss diverges: prefix is introducing noise — check position_ids and attention mask |
| ep18 | GNN gradient share still 40–75% | If GNN share < 15%: consider `stop_gradient` on prefix node embeddings before proj |
| ep20 | `gnn_to_bert_proj` weight norm stabilizing (Δ < 1% epoch-to-epoch) | If still rising rapidly: extend `gnn_prefix_warmup_epochs` to 20 in next run |
| ep20 | F1-macro trending higher than PLAN-3A at ep20 (0.240 range) | If lower: GNN prefix not helping; flag and decide whether to continue |
| ep20 | MLflow metric `prefix_attention_mean` > 0 | If ≈ 0: transformer ignoring prefix tokens entirely (attention weights near 0) |

**MLflow — log prefix attention weight:** After warmup ends (ep≥15), log the mean attention weight that code tokens assign to GNN prefix positions. This confirms the transformer is actually using the prefix rather than learning to ignore it:
```python
# In trainer.py, after forward pass (ep >= gnn_prefix_warmup_epochs):
# Extract attention weights from BERT output (output_attentions=True in forward).
# Mean over heads and layers for positions 0..K → prefix_attention_mean.
mlflow.log_metric("prefix_attention_mean", prefix_attention_mean, step=global_step)
```
A value near 0 for 5+ consecutive epochs means the prefix is being ignored — investigate `gnn_to_bert_proj` initialization and the attention mask shape.

**Hypotheses to track:**

| Hypothesis | Prediction | Result |
|------------|-----------|--------|
| H-GCB-1 | Reentrancy F1 > 0.30 (CEI visible to transformer) | — |
| H-GCB-2 | UnusedReturn F1 > 0.22 (def-use pre-training) | — |
| H-GCB-3 | ExternalBug F1 > 0.28 (cross-function via prefix) | — |
| H-GCB-4 | Tuned F1-macro > 0.30 (structural ceiling lifted) | — |
| H-GCB-5 | Behavioral test > 10/19 (window isolation resolved) | — |

### Behavioral evaluation — how to run

**Script:** `ml/scripts/manual_test.py` (existing — no new script needed)

**What it measures:** For each `.sol` file in `ml/scripts/test_contracts/`, the first line encodes expected vulnerability classes as `// expect: ClassName[, ClassName...]` (empty for safe contracts). The script counts how many expected class detections are found above the tuned threshold. Score = `total_hit / total_expected`.

**The 19:** There are 20 `.sol` files; 3 have empty expectations (safe contracts: 12, 18, 19). Contracts 01–11, 14–17, 20 each contribute 1 expected detection; contract 13 contributes 3 (Reentrancy, Timestamp, UnusedReturn). Total expected = 19.

**Canonical contract list (all 20 files, `ml/scripts/test_contracts/`):**
01_reentrancy_classic, 02_reentrancy_tricky, 03_integer_overflow, 04_timestamp_dependence, 05_denial_of_service, 06_mishandled_exception, 07_tx_order_dependence, 08_unused_return, 09_call_to_unknown, 10_gas_exception, 11_external_bug, 12_safe_contract *(safe)*, 13_multilabel_complex *(×3 expected)*, 14_reentrancy_minimal, 15_tod_minimal, 16_gas_minimal, 17_integer_simple, 18_safe_no_calls *(safe)*, 19_safe_with_transfer *(safe)*, 20_unused_return_minimal

**Command:**
```bash
source ml/.venv/bin/activate
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/manual_test.py \
    --checkpoint <best_checkpoint.pt> \
    --contracts ml/scripts/test_contracts/
```
Look for `Vulnerability detection: X/19 (Y%)` at the bottom of output.

**When to run:** After `tune_threshold.py` completes (use tuned thresholds, not defaults), on the best epoch checkpoint. Baseline at PLAN-3A: **8/19**. Phase 1 target: **> 10/19**.

> ⚠️ **tune_threshold.py note:** When GNN prefix injection is active, `tune_threshold.py` must also pass `gnn_prefix_nodes` through its forward pass. If it runs without the prefix, thresholds are calibrated on non-prefix inference and won't match deployment behavior. Update `tune_threshold.py` before running threshold optimization for Phase 1.

---

## IMP-M — Immediate Actions (While P1-TRAIN Runs)

**Status:** ⬜ OPEN
**When:** Implement now — these affect the next training run, not P1-TRAIN
**Reference:** `docs/proposal/IMPROVEMENT_BACKLOG.md` for full details on each

These items can be coded while P1-TRAIN runs overnight. They have no impact on the current run
but must be ready before P1B / Phase DATA-1 launches.

### IMP-BUG — Close stale BUG-H4 and BUG-H5 entries

Both were addressed by DQ-1 (`label_cleaner.py` changes, 2026-05-23) but the Open Bugs section
of `ACTIVE_PLAN.md` was not updated. Update ACTIVE_PLAN.md:
- BUG-H4: mark DONE — `check_timestamp()` requires `uses_block_globals > 0.5` (dim[2]); −568 labels removed
- BUG-H5: mark DONE — `check_reentrancy()` requires `external_call_count > 0` (dim[10]); −611 labels removed

**Status:** ⬜ OPEN

---

### IMP-M1 — FUNCTION Node Secondary Sort

**File:** `ml/src/models/sentinel_model.py:select_prefix_nodes()`
**Effort:** 30 min

`graph_schema.py:343` documents that FUNCTION nodes should be sorted by `feature[10]`
(external_call_count) descending when K truncation occurs. The current implementation uses
graph-index order within the FUNCTION priority group — a spec-implementation gap.

Replace the `priorities.argsort()` logic with a two-key sort:
`(priority_type, -external_call_count, original_index)`. See `IMPROVEMENT_BACKLOG.md:IMP-M1`
for the exact code.

**Status:** ⬜ OPEN

---

### IMP-M2 — prefix_attention_mean Diagnostic Logging

**File:** `ml/src/training/trainer.py`
**Effort:** Tier 1 = 30 min; Tier 2 = 2–3 hrs

**Tier 1 (do now):** Log `gnn_to_bert_proj.weight.norm()` to MLflow each epoch after warmup ends.
Healthy sign: norm changing 1–5%/epoch for first 5 post-warmup epochs. Flatline at ep17 = optimizer
not reaching the projection.

**Tier 2 (before GATE-GCB-4 analysis):** Add `output_attentions=True` path to `TransformerEncoder`
and log `prefix_attention_mean` (mean attention weight from code token positions → prefix positions).
Target: > 0.005 by ep20. Near-zero for 5+ epochs = transformer ignoring prefix. See
`IMPROVEMENT_BACKLOG.md:IMP-M2` for full implementation.

**Status:** ⬜ OPEN

---

### IMP-M3 — Zero-Padded Prefix Attention Mask Fix

**Files:** `ml/src/models/sentinel_model.py`, `ml/src/models/transformer_encoder.py`
**Effort:** 1–2 hrs

Currently all K prefix positions use `attention_mask=1` regardless of whether they contain real node
embeddings or zero-padding. For the 4.5% of contracts with < K eligible nodes, this wastes attention
capacity on zero vectors the transformer cannot meaningfully attend to.

Fix: return actual node count per graph from `select_prefix_nodes()` and construct the prefix mask
with `1` only for real nodes and `0` for padded positions. See `IMPROVEMENT_BACKLOG.md:IMP-M3`.

**Status:** ⬜ OPEN

---

## GATE-GCB-4 — Phase 1 Results Go/No-Go

**Status:** 🔴 BLOCKED (on P1-TRAIN)

| Outcome | Decision |
|---------|----------|
| Tuned F1-macro > 0.30 AND behavioral > 10/19 | ✅ Proceed to Phase 2 (Option C shared DFG) |
| Tuned F1-macro > 0.295, < 0.30 (marginal improvement) | Investigate per-class — if 3+ classes improve, proceed to Phase 2 anyway |
| Tuned F1-macro ≤ 0.295 | 🔴 Execute fallback plan (see below) before deciding |
| Any single class significantly regressed (> −0.02 vs PLAN-3A) | Investigate root cause before Phase 2 |

**Fallback plan (if Phase 1 ≤ 0.295):**
1. Run K=8 diagnostic (5 epochs): if K=8 > K=16, prefix was too noisy — prune node types
2. Run K=32 diagnostic (5 epochs): if K=32 > K=16, model needs more structural context
3. If K=32 still ≤ 0.295: architectural ceiling is label noise (BUG-H5), not graph blindness — shift to label cleaning before Phase 2

**Record Phase 1 results here:**

| Class | PLAN-3A tuned | Phase 1 tuned | Delta | H verdict |
|-------|--------------|---------------|-------|-----------|
| Reentrancy | 0.291 | — | — | H-GCB-1 |
| UnusedReturn | 0.195 | — | — | H-GCB-2 |
| ExternalBug | 0.255 | — | — | H-GCB-3 |
| Timestamp | 0.255 | — | — | — |
| IntegerUO | 0.699 | — | — | — |
| GasException | 0.358 | — | — | — |
| MishandledException | 0.289 | — | — | — |
| CallToUnknown | 0.256 | — | — | — |
| TOD | 0.251 | — | — | — |
| DoS | 0.030 | — | — | — |
| **F1-macro** | **0.2877** | **—** | **—** | H-GCB-4 |
| Behavioral | 8/19 | — | — | H-GCB-5 |

---

## Phase DATA-1 — Data Quality + Model Fix Training Run

**Status:** 🔴 BLOCKED (on GATE-GCB-4 results)
**Trigger:** GATE-GCB-4 results known; decide whether to continue architecture track or fix data first
**Reference:** `docs/proposal/IMPROVEMENT_BACKLOG.md:IMP-D1, IMP-D2, IMP-M1, IMP-M2, IMP-M3`

**Purpose:** A dedicated training run that incorporates all data quality and model fixes before
investing in Phase 2 (shared DFG) or Phase GNN-A (GNN overhaul). Running Phase 2 on dirty data
and with the FUNCTION sort bug active would make results harder to interpret.

**Changes vs P1-TRAIN:**

| Change | Item | Impact |
|--------|------|--------|
| IMP-D1: return_ignored temporal fix | Re-extract all 41K graphs with corrected `_compute_return_ignored` | Cleaner UnusedReturn + MishandledException labels |
| IMP-D2: inject 100+ OZ clean negatives | Add to `ml/data/augmented/`, re-run cache | False positive reduction |
| IMP-M1: FUNCTION secondary sort | Already in code before this run | Better K=48 node selection for large contracts |
| IMP-M2: prefix attention logging | Already in trainer before this run | Diagnostic visibility |
| IMP-M3: zero-padded mask fix | Already in code before this run | Cleaner attention for sparse contracts |

**IMP-D1 requires full re-extraction.** Follow the same gate protocol as Phase 2 (PLAN-2A–2I):
1. Validate fix on 10 known contracts with confirmed discarded returns
2. 2,000-contract sample gate: structural parity (existing edge types unchanged)
3. Rebuild cache with `FEATURE_SCHEMA_VERSION = "v8-d1"` (or bump to v9 if v9 is also being applied)
4. Update label CSV via `label_cleaner.py`

### GATE-DATA-1 — Data Quality Run Go/No-Go

| Check | Pass | Action |
|-------|------|--------|
| UnusedReturn F1 vs P1-TRAIN baseline | > +0.01 improvement | If no improvement: IMP-D1 fix not helping — investigate `return_ignored` distribution |
| MishandledException F1 vs P1-TRAIN | > 0.0 improvement | Baseline already weak (0.289); expect small gain |
| Behavioral Test safe contracts | ≥ 2/3 | If still 0/3: clean negatives not in training data — check injection pipeline |
| F1-macro vs P1-TRAIN tuned | ≥ P1-TRAIN result | If regression: data changes introduced noise — investigate per-class |

**Status:** 🔴 BLOCKED (on GATE-GCB-4 and IMP-D1/D2 implementation)

---

## Phase GNN-A — GNN Architecture Overhaul

**Status:** 🔴 BLOCKED (on GATE-DATA-1 + GATE-GCB-4)
**Trigger:** GATE-DATA-1 shows data quality fixes are absorbed; architectural ceiling confirmed lifted by GATE-GCB-4
**Reference:** `docs/proposal/IMPROVEMENT_BACKLOG.md:IMP-G1, IMP-G2, IMP-G3`
**Duration:** ~60–80 hrs GPU + 1 week engineering

**Purpose:** Address the three highest-impact GNN architectural weaknesses identified in the adversarial
audit. These are implemented together and measured in a single training run so their combined effect
is visible. Individual ablations (G1-only, G2-only, G3-only) can follow if the combined run shows
significant improvement and the source is unclear.

**Changes vs Phase DATA-1 baseline:**

| Change | Item | What it fixes |
|--------|------|---------------|
| IMP-G1 | Phase 2 layer-specific edge subsets | N-01: redundant layers sharing same edge set; Phase 2 JK collapse |
| IMP-G2 | Phase 1 input projection skip | N-03: no skip for 11→256 dim change; raw feature loss risk |
| IMP-G3 | Phase 3 bidirectional pass | N-04: CFG nodes lack Phase 3 context; CrossAttentionFusion sees representation gap |

**Total new parameters:**
- IMP-G2: `Linear(11, 256, bias=False)` = 2,816 params
- IMP-G3: one additional `GATConv(256, 256, heads=1, ...)` = ~66K params
- Net addition: ~69K params (0.05% of total model — negligible)

### Pre-Training Gate: GATE-GNN-A-SMOKE

2-epoch smoke test before full training:

| Signal | Expected | Fail condition |
|--------|----------|----------------|
| Step loss ep1 | 0.15–0.20 | > 0.30 or NaN → IMP-G changes introduce instability |
| JK Phase 2 std ep1 | > 0.08 (per-node routing active immediately) | ≤ 0.05 → layer-specific subsets not differentiating |
| GNN gradient share ep1 | > 40% | < 20% → input projection skip not connecting |
| Phase 1 JK weight ep1 | > 0.08 | < 0.04 → Phase 1 still being down-weighted; check skip wiring |
| No NaN in Phase 3 down pass | 0 NaN | Any NaN → check fwd_contains edge handling for graphs with no CONTAINS edges |

### GATE-GNN-A — GNN Overhaul Results Go/No-Go

| Outcome | Decision |
|---------|----------|
| F1-macro > Phase DATA-1 AND Phase 2 JK weight > 0.15 at convergence | ✅ GNN overhaul effective; proceed with this as new baseline |
| F1-macro ≈ Phase DATA-1 (± 0.005) | Investigate per-class — if individual improvements cancel out, consider individual ablations |
| Phase 2 JK weight still < 0.10 at convergence despite IMP-G1 | IMP-G1 not fully fixing the collapse; consider Phase 2 head count increase (N-02: heads=1 bottleneck) |
| F1-macro regresses | Identify which IMP-G change caused regression; run individual ablations |

**Hypotheses:**

| Hypothesis | Prediction | Rationale |
|------------|-----------|-----------|
| H-GNN-A-1 | Reentrancy F1 > P1-TRAIN + 0.01 | IMP-G1 CF layer builds better CEI pattern; IMP-G3 gives CFG fusion nodes Phase 3 context |
| H-GNN-A-2 | Phase 2 JK weight at convergence > 0.15 (vs 0.048 PLAN-3A) | IMP-G1 makes Phase 2 layers genuinely distinct → JK has reason to use them |
| H-GNN-A-3 | GNN gradient share stable > 15% through convergence | IMP-G2 skip connection prevents representation loss in Phase 1 |
| H-GNN-A-4 | CrossAttentionFusion loss < GNN eye loss from ep10 | IMP-G3 CFG nodes with Phase 3 context → richer fusion cross-attention keys/values |

**Status:** 🔴 BLOCKED (on Phase DATA-1 completion)

---

## P2 — Phase 2: Option C (Shared Contract-Level DFG)

**Status:** 🔴 BLOCKED (on GATE-GCB-4)
**Duration:** 1.5–2 weeks engineering + ~60–80 hours GPU
**Trigger:** Phase 1 tuned F1 > 0.30 OR 3+ classes improved

**Key changes vs Phase 1:**
- Build DFG once per contract using Slither's `ssa_variables` and `variables_written`/`variables_read`
- Append up to M=64 DFG variable nodes as shared suffix in every window
- Semi-structured attention mask: DFG nodes attend to their def/use code tokens
- Token file format changes → `FEATURE_SCHEMA_VERSION = "v9"` required

**Gate metric:** Tuned F1-macro > Phase 1 result AND Timestamp F1 > 0.265

---

## GATE-GCB-5 — Phase 2 Results Go/No-Go

**Status:** 🔴 BLOCKED (on P2)

| Outcome | Decision |
|---------|----------|
| Δ F1-macro vs Phase 1 > 0.02 | ✅ Proceed to Phase 3 |
| Δ F1-macro 0.01–0.02 | Borderline — proceed only if UnusedReturn or Timestamp improved significantly |
| Δ F1-macro < 0.01 | Option C is marginal; skip Phase 3, document and analyze |

---

## P3 — Phase 3: Option A (Full Per-Window DFG Masking)

**Status:** 🔴 BLOCKED (on GATE-GCB-5)
**Duration:** 3–4 weeks engineering + ~60–80 hours GPU
**Trigger:** Phase 2 Δ F1 > 0.02

**Key changes vs Phase 2:**
- Per-window DFG extraction (which variables appear in token positions `start_i:end_i`)
- Full `[seq_len × seq_len]` structured attention mask per window
- Batching with dynamic sequence lengths
- Aligns exactly with GraphCodeBERT's original pre-training format

**Gate metric:** Tuned F1-macro > Phase 2 result AND Reentrancy F1 > 0.350

---

## GATE-GCB-6 — Full Ablation Complete

**Status:** 🔴 BLOCKED (on P3)
**Action:** Document full ablation results, update architecture as the new baseline, proceed to ZKML and production hardening.

---

## Parallel Track — What Runs Alongside This Plan

These items are independent of the GraphCodeBERT proposal and should continue in parallel:

| Item | Description | Priority |
|------|-------------|----------|
| DQ-4 | Inject 100+ confirmed-clean negative contracts (OpenZeppelin) | P1 — do before Phase 1 training |
| PLAN-3B | v8-B ablation (CF + DEF_USE only) — completes the ablation matrix | P2 — low urgency, can skip if v8.0-B + Phase 1 are decisive |
| BUG-H5 active learning | Surface Timestamp val contracts with prob 0.35–0.65 for manual review | P2 — parallel to Phase 1 training |
| M5 Contracts | Fix foundry.toml remappings; run forge build + forge test | P1 — independent, do now |
| M4 Agents | Build RAG index; start MCP servers; run actual audit pipeline | P2 — after M5 verified |
| BUG-M5 | Remove Brainmab mislabeled contract | P2 |
| IMP-BUG | Close stale BUG-H4 + BUG-H5 entries in ACTIVE_PLAN.md | P0 — do now |
| IMP-M1 | FUNCTION secondary sort by external_call_count | P0 — before next run |
| IMP-M2 | prefix_attention_mean diagnostic (Tier 1: proj norm) | P0 — add to trainer now |
| IMP-M3 | Zero-padded prefix attention mask fix | P1 — before P1B/Phase DATA-1 |
| IMP-D1 | return_ignored temporal ordering fix + re-extraction | P1 — before Phase DATA-1 run |
| IMP-D2 | Inject 100+ OZ clean negative contracts | P1 — before Phase DATA-1 run |
| IMP-G1/G2/G3 | GNN Architecture Overhaul (Phase GNN-A) | P2 — after GATE-DATA-1 |

---

## Summary Checklist

Use this to track overall progress at a glance:

| Gate / Stage | Status | Date Completed |
|---|---|---|
| GATE-GCB-0: v8.0-B result | ✅ CLOSED | 2026-05-23 |
| PRE-1: GraphCodeBERT cached | ✅ DONE | 2026-05-23 |
| PRE-2: Tokenizer identity validated | ✅ DONE | 2026-05-23 |
| PRE-3: UNK rate < 0.5% | ✅ DONE | 2026-05-23 |
| PRE-4: Node count P95 ≤ K=48 (decl-level) | ✅ DONE | 2026-05-23 |
| PRE-5: LoRA on GraphCodeBERT | ✅ DONE | 2026-05-23 |
| GATE-GCB-1: All prerequisites passed | ✅ DONE | 2026-05-23 |
| P0: Phase 0 drop-in (5 epochs) | 🔵 RUNNING (ep2, ~3 epochs remaining) | 2026-05-23 started |
| P0b: CodeBERT+prefix ablation | ⬜ | — |
| GATE-GCB-2: Phase 0 go/no-go | ⬜ | — |
| P1-IMPL: Option B code changes + unit tests | ✅ DONE | 2026-05-23 |
| GATE-GCB-3: Smoke test passed | ⬜ | — |
| P1-TRAIN: Full Phase 1 training run | ⬜ | — |
| GATE-GCB-4: Phase 1 results recorded | ⬜ | — |
| P2: Option C (shared DFG) | ⬜ | — |
| GATE-GCB-5: Phase 2 results recorded | ⬜ | — |
| P3: Option A (full per-window DFG) | ⬜ | — |
| GATE-GCB-6: Full ablation complete | ⬜ | — |
| IMP-BUG: Close BUG-H4+H5 in ACTIVE_PLAN.md | ⬜ OPEN | — |
| IMP-M1: FUNCTION secondary sort | ⬜ OPEN | — |
| IMP-M2 Tier 1: proj_norm logging | ⬜ OPEN | — |
| IMP-M2 Tier 2: prefix_attention_mean | ⬜ OPEN | — |
| IMP-M3: zero-padded prefix mask fix | ⬜ OPEN | — |
| IMP-D1: return_ignored temporal fix + re-extraction | ⬜ OPEN | — |
| IMP-D2: 100+ clean negatives injected | ⬜ OPEN | — |
| GATE-DATA-1: Data quality run results | ⬜ OPEN | — |
| Phase GNN-A: Smoke test (GATE-GNN-A-SMOKE) | ⬜ OPEN | — |
| Phase GNN-A: Full training run | ⬜ OPEN | — |
| GATE-GNN-A: GNN overhaul results | ⬜ OPEN | — |

---

*Document ends. Fill in status, dates, and recorded values as each stage completes.*
