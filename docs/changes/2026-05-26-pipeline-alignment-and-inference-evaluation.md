# 2026-05-26 — Pipeline Alignment, Inference Evaluation, Three-Tier Design

## Summary

Three-session arc completing all training→inference pipeline alignment work for Run 4, verifying
it end-to-end with the real checkpoint, evaluating the model on 20 test contracts, and designing
the three-tier suspicion output format for the agent module.

---

## 1. compare_pipelines.py — Full Pipeline Audit

A new audit document (`sentinel_pipeline_comparison_audit.md`) was authored and then fully
validated against source code. Every claimed finding was verified by reading the actual files.
Several claims in the audit were wrong; the source code was ground truth.

### Confirmed Gaps Fixed in Source Code

| ID | File | Gap | Fix |
|----|------|-----|-----|
| Window advance | `preprocess.py` | Online tokenizer advanced by `stride=256` → overlap 254 tokens, not 256; mismatched HF offline which advances `(L-2)-stride = 254` | Changed `start += stride` → `start += _CONTENT_CAP - stride` |
| WindowAttentionPooler path | `predictor.py` | Old inference sent `[1,512]` → single-window fallback in pooler; training used `[B,4,512]` → learned multi-window attention. Completely different code path. | Rewrote `_score_windowed()` to stack all windows into `[1,4,512]` and call model once — same path as training |
| predict() truncation | `predictor.py` | `predict(sol_path)` read source → tokenized to 512 tokens max → long contracts silently truncated | `predict()` now delegates to `predict_source()` → always windowed |
| `_warmup()` prefix bypass | `predictor.py` | Warmup used 2-node (STATE_VAR-only) graph → `select_prefix_nodes()` found 0 eligible nodes → prefix injection never exercised at startup | Rewrote `_warmup()`: 3-node graph (CONTRACT+FUNCTION+STATE_VAR) with CALLS+CONTAINS edges; `[1,4,512]` token format |
| Solc binary detection | `preprocess.py` | Online path used bare `GraphExtractionConfig()` → no solc binary → `[Errno 2] No such file or directory: 'solc'` | Added `_make_extraction_config(source)` that detects pragma version and resolves venv binary in `.solc-select/artifacts/` |
| G21 double-multiply bug | `compare_pipelines.py` | `raw_type_ids = (type_ids_off * _MAX_TYPE_ID).round().long()` applied 12× scale to already-integer type IDs → all CFG nodes appeared orphaned | Changed to `raw_type_ids = type_ids_off` |
| G21 scope | `compare_pipelines.py` | Check verified all non-CONTRACT nodes had CONTAINS parents; FUNCTION/STATE_VAR never have CONTAINS parents by design | Scoped to CFG_NODE_* only |
| v7/v8 comments | `graph_schema.py` | Four stale "v7" comments in a v8 file | Fixed to "v8" |

### compare_pipelines.py Check Updates

| Check | Before | After | Reason |
|-------|--------|-------|--------|
| O2 | DIFF | PASS | Inference now uses batched `[1,4,512]` — aligned with training |
| O3 | WARN (truncation risk) | PASS | predict() delegates to windowed path |
| O5 | WARN (2-node warmup) | PASS | warmup now uses FUNCTION node + `[1,4,512]` |
| G21 | FAIL (all contracts) | PASS | Fixed double-multiply type_id bug |
| M1/M2 | FAIL (mixed dtype CPU) | PASS | Added `.float()` cast; checkpoint now actually loaded |
| M5 | FAIL (mixed dtype) | PASS | Same `.float()` fix |

### Tokenizer Identity Verification

Verified programmatically: `microsoft/graphcodebert-base` and `microsoft/codebert-base` share
identical BPE vocabulary (50,265 tokens, cls=0, sep=2). Every token ID is identical for any
Solidity source input. Training (`retokenize_windowed.py` uses codebert-base), offline dataset,
and online inference all produce the same token IDs. M3 (logit agreement) PASS confirms this
end-to-end.

Comment added to `preprocess.py` `TOKENIZER_NAME` constant documenting why codebert-base is used
even though the model backbone is graphcodebert-base.

### Final Results

```
compare_pipelines.py --checkpoint ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt

PASS : 1062
FAIL :    0  ← must be zero for safe deployment
WARN :   23  ← all design-level (G13 edge type diversity in small test suite,
               S5/S6/S6b architectural blind spots — none are code gaps)

M1 ✓ PASS  model forward (offline graph) — Run 4 checkpoint loaded
M2 ✓ PASS  model forward (online graph)
M3 ✓ PASS  logit agreement offline==online (max_diff < 1e-5)
M4 ✓ PASS  prefix node selection identical
M5 ✓ PASS  online batched forward [1,4,512] (training-aligned)
M6 ✓ PASS  GNN entropy scalar validity
```

---

## 2. Run 4 — Complete

**Killed:** 2026-05-26 20:12 at epoch 44. F1 locked 0.31–0.34 for 12 epochs — capacity ceiling.
**Best checkpoint:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` — **epoch 32, F1=0.3362**

This is the highest F1 ever achieved in SENTINEL training (+0.0485 over previous best PLAN-3A
tuned 0.2877; +0.0734 over Run 1).

**Full analysis:** `docs/ml/gcb-p1-run4-final-analysis.md`

---

## 3. Test Contract Evaluation — 20 Contracts

Ran `Predictor.predict_source()` on all 20 test contracts in `ml/scripts/test_contracts/`.

**Aggregate results (at tuned thresholds):**

```
Exact-match: 1/20 (04_timestamp_dependence.sol only)
Label F1:    0.304   Precision: 0.259   Recall: 0.368
TP=7  FP=20  FN=12  TN=161
```

**Per-class summary:**

| Class | TP | FP | FN | Status |
|-------|----|----|-----|--------|
| Timestamp | 2 | 0 | 0 | ✓ Clean |
| Reentrancy | 3 | 4 | 1 | Partial (co-fires with CallToUnknown) |
| CallToUnknown | 1 | 5 | 0 | High FP rate |
| IntegerUO | 1 | 3 | 1 | Partial |
| DenialOfService | 0 | 0 | 1 | Missed |
| ExternalBug | 0 | 1 | 1 | Missed (structural signal absent) |
| GasException | 0 | 4 | 2 | Missed (FP on arithmetic patterns) |
| MishandledException | 0 | 0 | 1 | Missed |
| TransactionOrderDependence | 0 | 3 | 2 | Missed |
| UnusedReturn | 0 | 0 | 3 | Missed (semantic gap) |

**Full raw probability table (all 10 classes, threshold=0):**

```
Contract                            Exp                       CU   DoS    EB   Gas   Int   MHE    Re    TS   TOD    UR
01_reentrancy_classic.sol           Reentrancy             0.64✗ 0.31  0.26  0.30  0.31  0.29  0.62★ 0.20  0.28  0.25
02_reentrancy_tricky.sol            Reentrancy             0.84✗ 0.27  0.20  0.19  0.20  0.21  0.83★ 0.14  0.20  0.20
03_integer_overflow.sol             IntegerUO              0.26  0.31  0.27  0.41✗ 0.67★ 0.27  0.29  0.20  0.38✗ 0.28
04_timestamp_dependence.sol         Timestamp              0.24  0.31  0.33  0.32  0.40  0.27  0.33  0.47★ 0.26  0.28
05_denial_of_service.sol            DenialOfService        0.28  0.36· 0.27  0.28  0.24  0.26  0.31  0.21  0.25  0.27
06_mishandled_exception.sol         MishandledException    0.58✗ 0.36  0.23  0.29  0.27  0.38· 0.60✗ 0.21  0.26  0.27
07_tx_order_dependence.sol          TransactionOD          0.28  0.33  0.27  0.40  0.49  0.27  0.30  0.22  0.35· 0.27
08_unused_return.sol                UnusedReturn           0.70✗ 0.31  0.32  0.30  0.37  0.30  0.68✗ 0.20  0.28  0.30·
09_call_to_unknown.sol              CallToUnknown          0.76★ 0.31  0.30  0.26  0.38  0.28  0.78✗ 0.20  0.25  0.31
10_gas_exception.sol                GasException           0.34  0.34  0.33  0.36· 0.37  0.32  0.37  0.23  0.31  0.28
11_external_bug.sol                 ExternalBug            0.32  0.33  0.30· 0.34  0.39  0.30  0.32  0.23  0.29  0.29
12_safe_contract.sol                SAFE                   0.39  0.30  0.30  0.36  0.64✗ 0.34  0.35  0.17  0.32  0.24
13_multilabel_complex.sol           Re,TS,UR               0.30  0.35  0.34  0.40✗ 0.43  0.34  0.40★ 0.51★ 0.32  0.32·
14_reentrancy_minimal.sol           Reentrancy             0.33  0.33  0.23  0.23  0.23  0.23  0.34· 0.17  0.23  0.23
15_tod_minimal.sol                  TransactionOD          0.32  0.33  0.31  0.31  0.31  0.31  0.36  0.23  0.29· 0.28
16_gas_minimal.sol                  GasException           0.31  0.41  0.33  0.38· 0.32  0.33  0.38  0.23  0.32  0.28
17_integer_simple.sol               IntegerUO              0.27  0.32  0.27  0.33  0.41· 0.28  0.29  0.21  0.32  0.28
18_safe_no_calls.sol                SAFE                   0.24  0.31  0.23  0.42✗ 0.68✗ 0.24  0.26  0.19  0.39✗ 0.24
19_safe_with_transfer.sol           SAFE                   0.52✗ 0.30  0.32  0.34  0.46  0.34  0.50✗ 0.20  0.31  0.26
20_unused_return_minimal.sol        UnusedReturn           0.37  0.32  0.36✗ 0.45✗ 0.54✗ 0.38  0.37  0.24  0.37✗ 0.34·

★=TP  ✗=FP  ·=FN (missed)  blank=TN
```

**Key observations from raw probabilities:**

1. **Most missed classes have signal just below threshold** — DoS=0.36 (thr=0.45), Gas=0.36 (thr=0.40),
   TOD=0.35 (thr=0.35), UnusedReturn=0.32–0.34 (thr=0.35). The model does have signal; the hard
   threshold suppresses it.

2. **Reentrancy + CallToUnknown co-fire on all external-call patterns** — contracts 01, 02, 06, 08,
   09, 19 all trigger both. Both classes share the same structural signal (a `.call()` with no
   re-entrancy guard). They are not well-separated in the learned representation space.

3. **Minimal contracts (14–17) cluster at 0.22–0.38 across all classes with no dominant signal.**
   Training corpus skews toward larger multi-function contracts. The model lacks structural features
   to distinguish a 10-line minimal contract — everything scores just below threshold.

4. **Safe contracts with arithmetic fire IntegerUO/GasException** — contracts 12 and 18 have
   arithmetic/loop patterns that the GNN correctly identifies as potentially dangerous but cannot
   distinguish from safe usage.

5. **ExternalBug is structurally invisible** — detecting it requires reasoning about what an external
   contract *might* do, which requires inter-contract analysis. The GNN only sees the current
   contract's graph.

---

## 4. Design Finding: Three-Tier Suspicion Output

Detailed discussion and proposal in `docs/proposal/2026-05-27-three-tier-inference-output.md`.

**Core finding:** The current binary threshold design (above = flagged, below = silent) is wrong
for a security oracle. The model's actual output is a continuous probability vector. Setting a hard
threshold converts "I'm somewhat confident this is suspicious" into "SAFE" — which is misleading
and costly.

**Key insight from raw prob table:** With a `SUSPICIOUS` tier at prob ≥ 0.25, all missed classes
except ExternalBug would be caught:
- DoS: 0.36 → SUSPICIOUS ✓
- GasException: 0.36 → SUSPICIOUS ✓
- TOD: 0.35 → SUSPICIOUS ✓
- UnusedReturn: 0.32–0.34 → SUSPICIOUS ✓
- Reentrancy minimal: 0.34 → SUSPICIOUS ✓

**Design resolution:** Model outputs full 10-class probability vector + three-tier classification.
Agent module makes final verdict using RAG + static analysis + LLM reasoning. Threshold is an
operational parameter, not a model parameter.

---

## Files Changed This Session

| File | Change |
|------|--------|
| `ml/src/inference/preprocess.py` | Window advance fix; solc detection; tokenizer name comment |
| `ml/src/inference/predictor.py` | _warmup() 3-node graph; _score_windowed() batched; predict() delegates; FUNCTION node |
| `ml/src/preprocessing/graph_schema.py` | v7→v8 comment fixes |
| `ml/scripts/compare_pipelines.py` | O3/O5 PASS; G21 double-multiply fix; M1/M2 checkpoint load + .float(); header comments |

---

## Commits

- `a6d4323` — fix(trainer): lower JK STD collapse threshold 0.05→0.015
- `de4fe10` — fix(trainer): fix prefix_attention_mean diagnostic unpack error
- `4970177` — fix issues (solc binary detection, window advance, warmup, _score_windowed)
