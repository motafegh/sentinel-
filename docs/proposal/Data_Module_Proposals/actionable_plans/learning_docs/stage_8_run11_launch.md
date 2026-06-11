# Stage 8 — Run 11 Launch

**Date:** 2026-08-05
**Status:** NOT STARTED. This is the final stage.
**Reading time:** 15-20 minutes.
**Goal:** After this doc, you can answer all 6 items in `LEARNING_CHECKLIST.md` §"Stage 8" from memory.

---

## 1️⃣ The Problem

### What Stage 8 has to deliver

Stage 8 is the **end** of the v2 build, not a new development stage. Its purpose is to launch Run 11 ("v2-baseline") on the v2 corpus, confirm it starts cleanly, and document the first-epoch results.

Run 11 is the first training run on the verified, multi-source, deduplicated v2 export. The first epoch's metrics are the baseline against which all future v2 runs are compared.

### Why this is a separate stage

The launch is a separate stage (not a task in Stage 7) because there is a clear handoff: Stage 7 delivers "v2 is ready"; Stage 8 starts "v2 is being trained." The 12-condition pre-launch checklist is the gate between the two.

---

## 2️⃣ The Solution

### The 12-condition pre-launch checklist (N-11)

| # | Condition | Why |
|---|---|---|
| 1 | Run 9 best checkpoint archived | Prevents accidental overwrite |
| 2 | v2 catalog entry reachable | Proves the export works |
| 3 | 7 v2-readiness gates GREEN | Stage 7's exit criteria |
| 4 | Predictor tier fix active | Prevents hidden threshold bug |
| 5 | EMITS edge fixed | Prevents under-representation |
| 6 | Launch command programmatically extracted | Prevents hand-typed typos (F21 lesson) |
| 7 | `--gnn-prefix-warmup-epochs=5` | Run 8/9 value, NOT 15 (reverts a working change) |
| 8 | MLflow backend `sqlite:///mlruns.db` | `file:///` is corrupt (experiments 1,2,3) |
| 9 | Timestamped `--run-name` | Prevents Run 9 silent-overwrite incident |
| 10 | `--early-stop-patience 30` + `--threshold-tune-interval 10` | Matches Run 7/8/9 |
| 11 | `--jk-entropy-reg-lambda 0.005` | Run 7/8/9 value (Run 9 typo was 0.0075) |
| 12 | Watcher copied from `run8_watcher.sh` with F1>0.1 floor | Proven infrastructure + safety net |

### Timestamped `--run-name` (D-8.4)

The MLflow run name is `v2-baseline-$(date +%Y%m%d-%H%M%S)`. The timestamp prevents the Run 9 incident where the same `--run-name` was used twice and the new ep1 overwrote the ep14 best.pt.

### The watcher (D-8.5)

A copy of `ml/scripts/run8_watcher.sh` with two changes:
1. F1>0.1 floor — filters out F1=0.0 saves (the Run 9 incident path)
2. Per-class collapse detection — any class going from > 0.1 to 0.0 in one epoch emits WARNING

### Per-class P/R/F1 reported separately (D-8.3)

The Run 8 audit found "DoS predicts positive for 76.8% of all test rows" — that was a precision problem, not F1. Reporting precision/recall separately surfaces this. The first-epoch report includes per-class precision, recall, AND F1.

### Run 11 uses v1 architecture (D-8.1)

Per the deferred-schema decision, Run 11 uses the v1 model architecture with the v9 schema. No schema changes in v2 (those are v2.1). This lets us attribute Run 11's results to the **data** change, not the model change.

---

## 3️⃣ The Broader Context

### What Stage 8 enables

- Run 11 trains on the v2 corpus → first-epoch val F1 is the baseline
- Future v2 runs compare against this baseline
- The v2 module's promise is validated if Run 11 ep1 > Run 9 ep1

### What breaks if Stage 8 is wrong

- Non-timestamped run name → Run 9 silent-overwrite repeats
- Wrong MLflow backend → corrupt experiment data
- Missing watcher F1 floor → degenerate F1=0.0 runs saved as "best"
- Wrong `gnn_prefix_warmup_epochs` → reverts a working change (15 vs 5)
- Missing per-class P/R → precision problems hidden by F1-macro

---

## 4️⃣ Verification — Stage 8 exit criteria

| # | Check | Status |
|---|---|---|
| 1 | Run 9 finished + checkpoint archived | ⏳ |
| 2 | `--dataset-version` flag works | ⏳ |
| 3 | 12-condition checklist all verified | ⏳ |
| 4 | Run 11 starts cleanly | ⏳ |
| 5 | First-epoch val F1 logged | ⏳ |
| 6 | Per-class P/R/F1 reported | ⏳ |
| 7 | MLflow registered with timestamped name + sqlite | ⏳ |
| 8 | Watcher running with F1>0.1 floor | ⏳ |
| 9 | ep10 threshold tune verified active | ⏳ |

---

## 5️⃣ The "got it" checklist

1. **Why timestamped `--run-name`?** Run 9's silent-overwrite incident: same name used twice, new ep1 overwrote ep14 best.pt. Timestamp makes every run unique.

2. **What's the watcher F1>0.1 floor?** Filters out F1=0.0 saves. The Run 9 incident path was "best_f1 reset to 0.0" → watcher emits "new best" → old best overwritten.

3. **Why `gnn_prefix_warmup_epochs=5` not 15?** Run 8/9 reduced from 15 to 5 and the change was active. Using 15 reverts a working change (F16, 8-P1).

4. **Why sqlite MLflow backend?** `file:///` is corrupt (experiments 1,2,3). `sqlite:///mlruns.db` is the proven backend.

5. **Why per-class P/R/F1 separately?** Run 8's "DoS predicts positive for 76.8%" was a precision problem hidden by F1-macro. Separate P/R surfaces this.

6. **What's the expected first-epoch F1?** >= 0.20 (matching Run 7's ep1), >= 0.2395 (matching Run 9's ep1). The v2 module's promise is validated if Run 11 ep1 > Run 9 ep1.

If you can answer all 6, Stage 8 is mastered and the v2 build is complete.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 8"
- **09_stage_8_run11_launch.md** — the design + intent document
- **Sentinel_v2_Data_Module_Integration_Proposal.md** §5 (Aug 5 entry), §9 (v2-readiness)
- **project_run8_audit_findings.md** (in MEMORY) — the watcher infrastructure
- **project_run9_resume.md** (in MEMORY) — the silent-overwrite incident

When you're ready, say **"Stage 8 is mastered — the v2 build is complete."**
