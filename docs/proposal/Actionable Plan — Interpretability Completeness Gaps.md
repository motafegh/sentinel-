## Actionable Plan — Interpretability Completeness Gaps
**Date:** 2026-05-31 - 10PM 
Ordered by impact on downstream conclusions. Each item references the exact section in `docs/proposal/INTERPRETABILITY_AUDIT_AND_COMPLETENESS.md`.

---

### P1 — BUG-2 · EXP-S3 "Dead Feature" Retraction
**Ref:** COMPLETENESS § Part 1 — BUG-2 (lines 83–115)  
**Impact:** The reported finding "dim 7 is a dead feature" is in the master report and experiment index. It is **factually wrong** and will mislead any reader.

**Actions:**
1. Fix [exp_s3_feature_distribution.py](ml/scripts/interpretability/exp_s3_feature_distribution.py):
   - Remove the `mean_call_depth_norm` field (dim 7 over CFG nodes)
   - Add a note in the output explaining: `return_ignored` (dim 7) is intentionally 0.0 on CFG nodes — it is a function-level feature set only on FUNCTION nodes in `graph_extractor.py`
   - Optionally replace with `mean_return_ignored_function` computed over FUNCTION nodes only to get the real distribution
2. Re-run and save output
3. Update [docs/interpretability/exp_s3_feature_distribution.md](docs/interpretability/exp_s3_feature_distribution.md) — retract the "dead feature" finding, replace with correct explanation

---

### P2 — INCOMPLETE-2 · EXP-E1 Add DEF_USE to Phase 2 BFS
**Ref:** COMPLETENESS § Part 2 — INCOMPLETE-2 (lines 191–214)  
**Impact:** The 37.7% CEI reachability finding (Analysis 1) is still valid. But Phase 2 reachability for IntegerUO and UnusedReturn is **underestimated** because DEF_USE (type 10) is missing from the BFS edge set. One line fix.

**Actions:**
1. Fix [exp_e1_receptive_field.py](ml/scripts/interpretability/exp_e1_receptive_field.py) line ~115:
   ```python
   PHASE2_EDGE_TYPES = {
       EDGE_TYPES["CONTROL_FLOW"],   # 6
       EDGE_TYPES["CALL_ENTRY"],     # 8
       EDGE_TYPES["RETURN_TO"],      # 9
       EDGE_TYPES["DEF_USE"],        # 10  ← ADD THIS
   }
   ```
2. Re-run Analysis 1 (CEI reachability) with the corrected edge set and report the updated percentage
3. Update [docs/interpretability/exp_e1_receptive_field.md](docs/interpretability/exp_e1_receptive_field.md) with the corrected reachability number and a note that the prior run excluded DEF_USE

---

### P3 — INCOMPLETE-5 · EXP-L5 Fix Pooling to Match Actual Model
**Ref:** COMPLETENESS § Part 2 — INCOMPLETE-5 (lines 259–289)  
**Impact:** Phase 2 probing AUROC values (e.g. Reentrancy 0.618) are measured on mean-only `[256]` vectors. The real model uses max+mean concatenated `[512]`. Phase 2 signal is **understated** by an unknown margin, which affects conclusions about whether Phase 2 contains linearly-decodable vulnerability information.

**Actions:**
1. Fix [exp_l5_probing_classifiers.py](ml/scripts/interpretability/exp_l5_probing_classifiers.py) lines ~194–201:
   ```python
   # Replace:
   pooled = phase_emb[func_mask].mean(0)   # [256]
   # With:
   func_embs = phase_emb[func_mask]        # [K, 256]
   pooled = torch.cat([
       func_embs.max(0).values,            # [256] max
       func_embs.mean(0),                  # [256] mean
   ], dim=0)                               # [512]
   ```
   Also update the probing classifier input size from 256 → 512
2. Re-run and compare AUROC numbers to prior run
3. Update [docs/interpretability/exp_l5_probing_classifiers.md](docs/interpretability/exp_l5_probing_classifiers.md) with corrected AUROCs and a note on what changed

---

### P4 — INCOMPLETE-3 · EXP-E4 Extend Direction Test Beyond CONTROL_FLOW
**Ref:** COMPLETENESS § Part 2 — INCOMPLETE-3 (lines 219–234)  
**Impact:** The "direction adds 0% discriminative power" finding is stated globally but only tested for CONTROL_FLOW. DEF_USE direction (def→use ordering for IntegerUO), CALL_ENTRY vs RETURN_TO asymmetry (cross-function direction for Reentrancy) were never tested. A wrong conclusion here could cause Run 5 architecture decisions to overlook directional signals.

**Actions:**
1. Extend [exp_e4_direction_sensitivity.py](ml/scripts/interpretability/exp_e4_direction_sensitivity.py) to also test bidirectionality for:
   - DEF_USE (type 10) — relevant to IntegerUO, UnusedReturn
   - CALL_ENTRY (type 8) — relevant to Reentrancy, ExternalBug
   - RETURN_TO (type 9) — relevant to UnusedReturn
   Each test: add reverse edges for that type → recompute WL collision rate → compare to directed baseline
2. Re-run, report per-edge-type direction sensitivity
3. Update [docs/interpretability/exp_e4_direction_sensitivity.md](docs/interpretability/exp_e4_direction_sensitivity.md) — scope the existing "no direction effect" claim to CF only; add new findings for DEF_USE/CALL_ENTRY/RETURN_TO

---

### P5 — INCOMPLETE-8 · EXP-L9 Fix Rollout Pass Criterion
**Ref:** COMPLETENESS § Part 2 — INCOMPLETE-8 (lines 340–355)  
**Impact:** The current criterion (≥2 CALL/WRITE nodes in top-10) is satisfied by **both** vulnerable and safe contracts because any contract with external calls has CALL/WRITE nodes. The experiment produces no discriminative signal — it always passes regardless of whether the model correctly attends to the CEI pattern.

**Actions:**
1. Rewrite the pass criterion in [exp_l9_attention_rollout.py](ml/scripts/interpretability/exp_l9_attention_rollout.py): instead of absolute count, compare the **relative attribution rank** of the specific CALL→WRITE CEI-pattern nodes in the vulnerable contract vs. the same node types in the safe contract. Pass = vulnerable contract's CALL/WRITE nodes rank higher (on average) than safe contract's CALL/WRITE nodes
2. Re-run on the same test contracts (already have outputs from today)
3. Update [docs/interpretability/exp_l9_attention_rollout.md](docs/interpretability/exp_l9_attention_rollout.md)

---

### P6 — Update All Stale Reports in docs/interpretability/
**Ref:** COMPLETENESS § Part 6 — "Findings that need re-evaluation" (lines 549–556)  
**Impact:** All 24 files in `docs/interpretability/` predate the COMPLETENESS doc by ~7 hours. They contain findings the COMPLETENESS doc has now partially invalidated, and are missing the Phase B results entirely.

**Actions — in order:**
1. **Master report** ([INTERPRETABILITY_MASTER_REPORT.md](docs/interpretability/INTERPRETABILITY_MASTER_REPORT.md)):
   - Retract BUG-2 "dead feature" claim from S3 section
   - Update L4 finding: "fn_call_count ranks 2nd" → "complexity (CFG block count) ranks 2nd"
   - Update E1 finding after DEF_USE fix (P2 above) runs
   - Scope E4 direction finding to CF only
   - Add Phase B findings: B1 (Phase 2 gradient starvation confirmed), B2 (main head miscalibrated, eyes OK), B3 (JK weight std per class), B4 (UnusedReturn saliency)
   - Add L6 result: 1/4 pass — model blind to CEI/IntegerUO/Timestamp structure
   - Update trust table in Part 6 based on COMPLETENESS doc conclusions

2. **Individual reports** (update only those with corrected findings):
   - [exp_s3_feature_distribution.md](docs/interpretability/exp_s3_feature_distribution.md) — retract dead feature
   - [exp_l4_gradient_saliency.md](docs/interpretability/exp_l4_gradient_saliency.md) — fix feature names
   - [exp_a2_cfg_inheritance.md](docs/interpretability/exp_a2_cfg_inheritance.md) — update with correct results
   - [exp_e1_receptive_field.md](docs/interpretability/exp_e1_receptive_field.md) — update after P2 re-run
   - [exp_l3_attention_visualization.md](docs/interpretability/exp_l3_attention_visualization.md) — add conv3b results
   - [exp_l5_probing_classifiers.md](docs/interpretability/exp_l5_probing_classifiers.md) — update after P3
   - [exp_l6_counterfactual.md](docs/interpretability/exp_l6_counterfactual.md) — add today's results (1/4 pass)
   - [exp_e4_direction_sensitivity.md](docs/interpretability/exp_e4_direction_sensitivity.md) — update after P4
   - [exp_l9_attention_rollout.md](docs/interpretability/exp_l9_attention_rollout.md) — update after P5
   - Add new reports for B1, B2, B3, B4 (no docs exist yet for these)

3. **EXPERIMENT_INDEX.md** — add B1–B4 rows, update status/trust for fixed experiments

---

### Deferred — Not Blocking (post-Run 5)
**Ref:** COMPLETENESS § Part 3 — MISSING-3 and MISSING-4 (lines 397–427)

| Item | Ref | Why deferred |
|------|-----|-------------|
| DEF_USE chain length distribution | MISSING-3 / N3 | Graph stats query, no model needed; informative but doesn't change Run 5 config |
| STATE_VAR multi-function sharing for TOD | MISSING-4 / N4 | Validates TOD encodability; answer doesn't change E1/E2 which are already committed |
| L10 full training ablation | COMPLETENESS § L10 note | Needs `--ablate-edge-type` in train.py + weeks of training; genuine post-Run-5 work |

---

**Suggested execution order:** P1 → P2 → P3 → P4 → P5 → P6 (docs last, once all re-runs are done so reports reflect final numbers).