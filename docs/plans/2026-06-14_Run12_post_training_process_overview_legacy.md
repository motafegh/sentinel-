# Post-Training Process — Overview (2026-06-14)

> **Purpose:** The end-to-end post-training workflow. What happens AFTER Run 12 finishes. Maps to specific docs for each step.
>
> **When:** Run 12 completes (early stop or ep100). Estimated: within 24-48 hours from now.
>
> **Scope:** Run 12 final validation → 4 pre-Run-13 fixes → v4 export → Run 13 launch → monitoring → production promotion (if Run 13 is the new SOTA).
>
> **See also:** [run_12_to_13_handoff_2026-06-14.md](run_12_to_13_handoff_2026-06-14.md) (the 6-step workflow) and [run13_plan_2026-06-14.md](run13_plan_2026-06-14.md) (the 4 fixes details).

---

## 1. The Complete Post-Training Workflow

```
Run 12 completes
    │
    ├─→ [Day 0] Run 12 final validation (2-3 hours)
    │     - Reproducibility (L.1)
    │     - Performance analysis
    │     - OOD/contamination check (A.1)
    │     - Run 12 final report
    │     - Save artifacts (checkpoints, logs)
    │     - Promote to Staging (F.2)
    │
    ├─→ [Day 1] Apply 4 pre-Run-13 fixes (1-2 days)
    │     - Fix 1: Drop GasException → NUM_CLASSES=9 (30 min)
    │     - Fix 2: Extend L4 to drop `loc` (15 min)
    │     - Fix 3: Strip Solidifi `bug_*` prefix (1-2 hours)
    │     - Fix 4: Inject 658 BCCC ME contracts (1-2 days)
    │
    ├─→ [Day 3-4] Build v4 export (1-2 days)
    │     - Ingest BCCC ME (525 compilable)
    │     - Preprocess + represent + label + split + export
    │     - Validate 6 GREEN + 1 AMBER gates
    │     - Update docs (v4-readiness, architecture, CHANGELOG)
    │
    ├─→ [Day 5] Launch Run 13 (1 day)
    │     - Same config as Run 12
    │     - Cron monitor (same as Run 12)
    │     - 4 hypotheses to test
    │
    ├─→ [Day 5-7] Run 13 monitoring (continuous, ~40 hours)
    │     - Watch ep10/20/30/40/50 f1_tuned
    │     - Compare to Run 12 trajectory
    │     - Detect early regression
    │
    └─→ [Day 7-8] Run 13 final validation + Production promotion
          - Same as Run 12 (steps above)
          - Compare Run 13 vs Run 12: which is better?
          - Promote Run 13 best to Production (or Run 14 if not)
```

---

## 2. Existing Post-Training Process Specs (in `ml/testing_specs/`)

| Spec | What | When |
|---|---|---|
| `00_rules.md` | Master rules (gate assertions, completion attestation) | Always |
| `A_benchmark_runs.md` | Contamination check + external benchmark (SmartBugs, SolidiFI) | Before any benchmark claim |
| `B_contract_deep_dive.md` | Manual contract-level inspection | When class-level signal unclear |
| `C_diagnostic_checks.md` | C.2 behaviour checks (smoke inference) | Before promotion |
| `D_smoke_preflight.md` | Pre-launch smoke test | Before any new run |
| `E_preprocessing_consistency.md` | Verify cache validity (v10 `--relabel-timestamp` trap) | Before new run |
| `F_new_run_checklist.md` | F.1 pre-launch, F.2 promotion, F.3 post-run | **Every run lifecycle** |
| `G_ablation_protocol.md` | Ablation study (per-component impact) | When comparing configs |
| `H_issue_triage.md` | Bug/finding triage workflow | When issues found |
| `I_regression_guard.md` | Promotion gates (behaviour checks) | **Before promotion** |
| `J_schema_migration.md` | Schema version bump process | When FEATURE_SCHEMA_VERSION changes |
| `K_inference_api.md` | API response validation | After checkpoint change |
| `L_release_readiness.md` | L.1 reproducibility, L.2 release gate, L.3 handoff | **Before reporting findings** |

**The 12-letter alphabet is the EXISTING post-training process.** This overview is the wrapper.

---

## 3. What's NEW (added this session, 2026-06-14)

These are the session-specific docs that wrap the alphabet:

| Doc | Purpose |
|---|---|
| `run_12_to_13_handoff_2026-06-14.md` | The 6-step workflow (this overview's child) |
| `run13_plan_2026-06-14.md` | The 4 fixes details (Fix 1-4) |
| `project_bccc_2tool_audit_2026-06-14.md` | Source of 658 ME contracts (in MEMORY) |
| `project_feature_leakage_audit_2026-06-14.md` | Why dropping GasException + extending L4 (in MEMORY) |
| `check_run12_status.sh` + README | Run 12 cron monitor (in ml/scripts/) |
| `/tmp/bccc_me_*.json` | 658 ME contract extraction list (operational) |

---

## 4. Decision Flowchart: "What to do with a completed run"

```
                    Run N completes
                          │
                          ▼
            ┌─────────────────────────────┐
            │ Is it better than the best? │
            │  (per L.1 + I_regression_guard)│
            └─────────┬───────────────────┘
                      │
         ┌────────────┼────────────┐
         │ YES                       │ NO
         ▼                          ▼
   Promote to Staging       Save checkpoint, log
         │                  as "failed experiment"
         ▼
   Test on benchmark
   (per A_benchmark_runs)
         │
   ┌─────┴──────┐
   │            │
   ▼            ▼
 Stays       Promote to
 Staging     Production
              (per I.3)
```

---

## 5. The 4 "would be helpful to create" items I added this turn

These are the things that DIDN'T exist before and were created based on user feedback "any other things that would be helpful to be created and added to current post training process":

### 5.1 — `run_12_to_13_handoff_2026-06-14.md` (NEW)

The end-to-end 6-step workflow. Why needed: the existing testing_specs cover individual procedures (F, A, L, etc.) but no single doc says "what's the order of operations". This handoff is the master sequence.

### 5.2 — `run13_plan_2026-06-14.md` (NEW)

The 4 fixes with file-by-file changes. Why needed: each fix is small (15 min - 2 days) but they have dependencies (Fix 1-3 must complete before Fix 4 can start). Need a single doc showing the dependency graph + scripts to create.

### 5.3 — Updated `docs/CHANGELOG.md`

Added 5 new sections (Stage 7, Run 10-12, Run 12 validation, Run 13 plan, post-Phase-5 decisions). Why needed: was last updated 2026-06-08, missing 6 days of work.

### 5.4 — This overview (`post_training_process_2026-06-14.md`, NEW)

The wrapper that ties everything together. Why needed: 12 testing_specs + 2 new plans + CHANGELOG + scripts = a lot to navigate. This overview is the single entry point.

---

## 6. What's STILL missing (would be helpful to add)

These are things that would be helpful but I haven't created yet (would need more work or decisions from Ali):

### 6.1 — `data_module/temp/live_plans/inference_hardening_2026-06-XX.md` (TODO)

**What it would cover:** After Run 12 best is identified, harden the inference path:
- Verify the checkpoint loads in `ml/src/inference/predictor.py`
- Test the FastAPI server (per K_inference_api.md K.1-K.5)
- Run a smoke test with 10 known contracts (5 safe, 5 vulnerable)
- Verify the response schema matches expectations
- Test edge cases: empty input, malformed Solidity, very large contracts

**Why not done yet:** Run 12 is still training. Will be created as part of Step 2.4 of the handoff plan.

### 6.2 — `data_module/temp/live_plans/autoML_baseline_2026-06-XX.md` (TODO)

**What it would cover:** The deferred Phase 4 Stage 6 work — run AutoML (XGBoost, LightGBM, CatBoost, etc.) on the v3 data as a baseline. If AutoML > SENTINEL, the model is over-engineered.

**Why not done yet:** Requires verified labels (we have them now) but the Phase 5 work was for the v1.3 dataset, not v3. Need to re-run on v3. Parked.

### 6.3 — `data_module/temp/live_plans/seam_swap_completion_2026-06-13.md` (already exists, needs decision)

**Current state:** 3 open questions for Ali (Q1, Q2, Q3). All deferred to post-Run-12. **Once Ali answers, can be executed in 2-3 hours.**

### 6.4 — `data_module/temp/live_plans/run14_cgt_ingestion_2026-06-XX.md` (TODO)

**What it would cover:** If Run 13 < 0.72, plan the CGT ingestion (2-3 days, +3,103 contracts). Already partially covered in `data-source-addition-plan-2026-06-13.md` (now updated with BCCC finding).

### 6.5 — `data_module/temp/live_plans/runN_solidifi_promotion_2026-06-XX.md` (TODO)

**What it would cover:** The final promotion decision: when to declare a run as the "production" model. Includes statistical significance testing (is f1_tuned=0.6941 actually better than Run 11's 0.3384?). Maybe a bootstrap CI.

**Why not done yet:** Need to know Run 12's final state first. Will be created as part of Step 2.4 (Run 12 final report).

---

## 7. Summary: Process Coverage

| Activity | Existing? | Where? |
|---|---|---|
| New run validation (pre-launch) | ✅ Yes | `F_new_run_checklist.md` F.1 + `D_smoke_preflight.md` |
| New run validation (post-run) | ✅ Yes | `F_new_run_checklist.md` F.3 + `L_release_readiness.md` |
| Performance analysis | ✅ Yes | `A_benchmark_runs.md` A.1-A.2 (contamination + benchmark) |
| Promotion to Staging | ✅ Yes | `I_regression_guard.md` I.2 + `C_diagnostic_checks.md` C.2 |
| Promotion to Production | ✅ Yes | `I_regression_guard.md` I.3 + `L_release_readiness.md` L.4 |
| Inference validation | ✅ Yes | `K_inference_api.md` K.1-K.5 |
| Session handoff | ✅ Yes | `L_release_readiness.md` L.5 |
| Bug triage | ✅ Yes | `H_issue_triage.md` |
| **End-to-end workflow** | **🆕 Added** | **`post_training_process_2026-06-14.md` (this file)** + handoff + Run 13 plan |
| **Run-specific plan** | **🆕 Added** | **`run_12_to_13_handoff_2026-06-14.md`** + **`run13_plan_2026-06-14.md`** |

**Coverage is comprehensive.** The 12 testing_specs cover the general case; the 2 new plans cover the specific Run 12 → Run 13 case.

---

## 8. Next actions (what Ali should do now)

1. **Wait for Run 12 to finish** (~24-48 hours). Cron will notify.
2. **After Run 12 finishes**: Review the Run 12 final report (will be created automatically per handoff Step 2.4).
3. **Decide on Run 13 priority**: is the 1-2 week Run 13 prep worth doing now, or focus on seam swap first, or something else?
4. **Decide on Q1-Q3 from seam_swap_completion_2026-06-13.md**: when to do the seam swap completion.

For Claude: continue waiting for cron notifications + prepare to execute handoff Step 2 when Run 12 finishes.
