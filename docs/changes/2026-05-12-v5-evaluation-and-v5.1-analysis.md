# v5 Phase 6 Evaluation — Results and v5.1 Root-Cause Analysis

**Date:** 2026-05-12
**Model:** `ml/checkpoints/v5-full-60ep_best.pt` (epoch 43, raw F1-macro 0.5736)
**Thresholds:** `ml/checkpoints/v5-full-60ep_best_thresholds.json`
**Author:** motafegh

---

## 1. Training Run Summary (Phase 5C)

The full 60-epoch run was launched 2026-05-11 resuming from the epoch-10 check-run
weights (F1=0.3856). The run was stopped at epoch 44 by the user after observing a
first patience tick (patience=1/10, epoch 44 F1=0.5731 slightly below epoch 43 best).

| Epoch | Loss | F1-macro | Notes |
|-------|------|----------|-------|
| 1–3 | 1.00→0.88 | 0.19–0.22 | Warmup; GNN dominant |
| 4–12 | 0.87→0.78 | 0.22→0.45 | LR peak; TF dominant |
| 13–22 | 0.77→0.70 | 0.45→0.53 | LR descent; Fused rising |
| 23–35 | 0.69→0.60 | 0.53→0.57 | Fused takes lead at ep23 |
| 36–44 | 0.60→0.52 | 0.57→0.57 | Fused dominant (52%); GNN at 6.7% |

Best checkpoint: epoch 43, raw F1-macro **0.5736**.

**Eye gradient share at epoch 44 (B2900):**
- GNN eye: ~6.7%
- TF eye: ~40%
- Fused eye: ~52%

---

## 2. Threshold Tuning (Phase 6a)

Run command:
```bash
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/tune_threshold.py \
  --checkpoint ml/checkpoints/v5-full-60ep_best.pt --batch-size 32 --num-workers 2
```

### Per-class tuned results

| Class | Threshold | Tuned F1 | v4 floor | vs floor |
|-------|-----------|----------|----------|---------|
| CallToUnknown | 0.40 | 0.4659 | 0.397 | +0.069 ✅ |
| DenialOfService | 0.90 | 0.4490 | 0.384 | +0.065 ✅ (gate ❌) |
| ExternalBug | 0.50 | 0.5657 | 0.434 | +0.132 ✅ |
| GasException | 0.45 | 0.6029 | 0.507 | +0.096 ✅ |
| IntegerUO | 0.45 | 0.8412 | 0.776 | +0.065 ✅ |
| MishandledException | 0.40 | 0.5588 | 0.459 | +0.100 ✅ |
| Reentrancy | 0.50 | 0.6163 | 0.519 | +0.097 ✅ |
| Timestamp | 0.70 | 0.5533 | 0.478 | +0.075 ✅ |
| TOD | 0.50 | 0.5703 | 0.472 | +0.098 ✅ |
| UnusedReturn | 0.50 | 0.6046 | 0.495 | +0.110 ✅ |

**Tuned F1-macro: 0.5828** — gate requires >0.58 ✅ CLEARED

**DoS analysis:** 137 val positives (1.3% prevalence). Even at threshold=0.90 (most
selective), best F1 is 0.449. This is a data-ceiling issue, not a tuning issue.
DoS gate (>0.55) FAILED — data augmentation required.

### Gate summary

| Gate | Required | Result |
|------|----------|--------|
| F1-macro tuned | > 0.58 | 0.5828 ✅ |
| All classes ≥ v4 floor | 10/10 | 10/10 ✅ |
| DoS tuned F1 | > 0.55 | 0.449 ❌ |
| Behavioral detection | > 70% | see §3 |
| Behavioral safe specificity | > 66% | see §3 |

---

## 3. Behavioral Test Suite (Phase 6b)

Run command:
```bash
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/manual_test.py \
  --checkpoint ml/checkpoints/v5-full-60ep_best.pt \
  --contracts ml/scripts/test_contracts/
```

### Per-contract results

| # | Contract | Expected | Detected | Result |
|---|----------|----------|----------|--------|
| 01 | reentrancy_classic | Reentrancy | CallToUnknown, **Reentrancy** | ✅ (FP: CTU) |
| 02 | reentrancy_tricky | Reentrancy | CallToUnknown, **Reentrancy** | ✅ (FP: CTU) |
| 03 | integer_overflow | IntegerUO | CallToUnknown, GasException, Reentrancy | ❌ |
| 04 | timestamp_dependence | Timestamp | IntegerUO | ❌ |
| 05 | denial_of_service | DenialOfService | (none) | ❌ |
| 06 | mishandled_exception | MishandledException | CallToUnknown, IntegerUO, Reentrancy | ❌ |
| 07 | tx_order_dependence | TOD | IntegerUO | ❌ |
| 08 | unused_return | UnusedReturn | CallToUnknown, ExternalBug | ❌ |
| 09 | call_to_unknown | CallToUnknown | ExternalBug, Reentrancy, UnusedReturn | ❌ |
| 10 | gas_exception | GasException | IntegerUO | ❌ |
| 11 | external_bug | ExternalBug | (none) — GHOST GRAPH | ❌ |
| 12 | safe_contract | (safe) | CallToUnknown, IntegerUO, Reentrancy | ❌ FP |
| 13 | multilabel_complex | Reentrancy, Timestamp, UnusedReturn | CallToUnknown, GasException, Reentrancy | ~ (1/3) |
| 14 | reentrancy_minimal | Reentrancy | CallToUnknown, IntegerUO | ❌ |
| 15 | tod_minimal | TOD | ExternalBug, IntegerUO, MishandledException | ❌ |
| 16 | gas_minimal | GasException | (none) | ❌ |
| 17 | integer_simple | IntegerUO | (none) | ❌ |
| 18 | safe_no_calls | (safe) | ExternalBug, IntegerUO | ❌ FP |
| 19 | safe_with_transfer | (safe) | CallToUnknown, IntegerUO, Reentrancy | ❌ FP |
| 20 | unused_return_minimal | UnusedReturn | (none) — GHOST GRAPH | ❌ |

**Detection rate: 3/19 (15%)** — unchanged from v4 failure baseline
**Safe-contract specificity: 0/3 (0%)** — worse than v4 (was 33%)
**Mandatory Contract B (CEI safe): FAILED** — Reentrancy=0.851, CallToUnknown=0.901

### Ghost graphs confirmed

Contracts 11, 13, and 20 produced 2-node, 0-edge ghost graphs due to the interface
selection bug in `_select_contract()` (see §4.1). They cannot be scored meaningfully.

### v5 vs v4 behavioral comparison

| Metric | v4 | v5 |
|--------|----|----|
| Detection rate | 15% (3/19) | 15% (3/19) |
| Safe specificity | 33% (1/3) | 0% (0/3) |
| Mandatory A/B | A✅ B❌ | A✅ B❌ |

Despite validation F1 improving from 0.5422 → 0.5828 (+3.9pp), behavioral performance
is identical to v4. The CFG architecture did not solve the core ordering-blindness
failure.

---

## 4. Root Cause Analysis

### 4.1 Interface selection bug in `_select_contract()` (confirmed)

**File:** `ml/src/preprocessing/graph_extractor.py`, line ~594

When a `.sol` file defines interfaces before the main contract (common pattern for
protocol contracts: `interface IOracle { ... } contract LendingPool { ... }`),
`candidates[0]` returns the interface. Interfaces have zero-node functions.
Result: CONTRACT node + FUNCTION shell = 2 nodes, 0 edges. Ghost graph.

**Confirmed:** `11_external_bug.sol` defines `IPriceOracle` (interface) first.
`_select_contract()` returns `IPriceOracle` with `is_interface=True`.

**Scale:** ~10% of 3,000 sampled training graphs have ≤3 nodes (**≈6,852 ghost
graphs** in the 68K training set). Ghost graphs contribute near-zero gradient and
poison the GNN's ability to learn from these contracts.

**Fix (one line):** Prefer non-interface, non-abstract contracts before falling back.
See proposal `docs/proposals/2026-05-12-v5.1-analysis-and-plan.md` §2.1.

### 4.2 CFG_RETURN node floods GNN eye pooling (root cause of GNN collapse)

**File:** `ml/src/models/sentinel_model.py` — GNN eye pooling

In a 3,000-graph sample, for graphs with CFG edges:
- CFG_RETURN share of CFG node mass: **mean=77%, median=93%**
- Graphs where ALL CFG nodes are RETURN: 29.6%
- Graphs with NO CFG_CALL node: 55.8%

The GNN eye uses `global_max_pool + global_mean_pool` over ALL nodes. With 77% of
CFG node mass being uninformative CFG_RETURN bookkeeping nodes, the pooled vector
is dominated by noise. The meaningful nodes — CFG_CALL (9%), CFG_WRITE (7%),
CFG_COND (5%) — contribute only 21% of the node mass.

**This explains the GNN eye gradient collapse to 6.7% by epoch 44.** Even with
`aux_loss_weight=0.4`, the GNN eye signal would remain diluted at the pooling
level unless the pooling strategy changes.

The fix is to pool at the FUNCTION level (not all-node level). Phase 3
(reverse-CONTAINS) already aggregates CFG signals up into FUNCTION nodes. Pooling
only over FUNCTION/FALLBACK/MODIFIER nodes removes the return-node flood.

### 4.3 Safe contract count is not the problem (assumption corrected)

Training has **17,116 safe contracts (35.7%)** — not "< 200" as initially suspected.
The model has seen ample safe examples. The behavioral failure on safe contracts is
not data scarcity — it is shortcut learning. The model learned "has `call{value}` +
balance mapping = Reentrancy" from the SolidiFI distribution. This shortcut fires on
both the vulnerable and CEI-safe contracts because both contain `call{value}` and a
balance mapping. Only the execution order differs.

### 4.4 Training data distribution

| Class | Train count | Train % |
|-------|-------------|---------|
| IntegerUO | 25,042 | 52.2% |
| Reentrancy | 11,633 | 24.3% |
| GasException | 12,156 | 25.3% |
| MishandledException | 10,599 | 22.1% |
| UnusedReturn | 7,872 | 16.4% |
| TOD | 8,199 | 17.1% |
| ExternalBug | 7,711 | 16.1% |
| CallToUnknown | 5,587 | 11.6% |
| Timestamp | 5,101 | 10.6% |
| **DenialOfService** | **695** | **1.4%** |
| Safe (all zeros) | 17,116 | 35.7% |

IntegerUO (52%) dominates. The model's broad IntegerUO firing (threshold 0.45
triggered on 8 of 20 behavioral test contracts) directly traces to this imbalance.

### 4.5 26.9% zero-edge training graphs

807 of 3,000 sampled training graphs have no edges at all (zero-edge). These include
the ~10% ghost graphs (interface selection bug) plus legitimate single-function or
purely declarative contracts. Zero-edge graphs contribute minimal gradient through
the GNN eye.

---

## 5. Decision

v5.0 behavioral gates are NOT cleared:
- Detection: 15% < 70% required
- Safe specificity: 0% < 66% required
- DoS tuned: 0.449 < 0.55 required
- Mandatory B test: FAILED

v5.0 validation gates ARE cleared:
- Tuned F1-macro: 0.5828 > 0.58 ✅
- All 10 classes above v4 floor ✅

**Decision:** Do NOT promote v5.0 checkpoint. Proceed to v5.1 rebuild addressing the
confirmed root causes. v4 remains production fallback.

See [docs/proposals/2026-05-12-v5.1-analysis-and-plan.md](../proposals/2026-05-12-v5.1-analysis-and-plan.md)
for the full v5.1 plan.

---

## 6. Files Changed / Produced

| File | Change |
|------|--------|
| `ml/checkpoints/v5-full-60ep_best.pt` | Final v5 checkpoint (epoch 43, raw F1=0.5736) |
| `ml/checkpoints/v5-full-60ep_best_thresholds.json` | Per-class thresholds (10 classes) |
| `ml/checkpoints/v5-full-60ep_best.state.json` | Training state (epoch 44, patience 1/10) |
| `ml/logs/train_v5_full60ep.log` | Full 44-epoch training log |
