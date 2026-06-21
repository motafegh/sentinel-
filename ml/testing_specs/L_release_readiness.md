# L — Release Readiness

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.
>
> **Last revised: 2026-06-17** (overhaul per `2026-06-17_testing_suite_overhaul_plan.md`).
> Added L.4.1 (auto-reproducibility) and L.5.1 (auto-floating-findings detection).
> Behavioral probes (C.2.4) and label quality (F.1.0) are now required for promotion.

---

## When This File Applies

- Before declaring any run result as a reportable finding
- Before promoting a checkpoint to Production (use after `I_regression_guard.md`)
- Before comparing two runs and claiming one is definitively better
- At session close — confirm all findings are written, all context is persisted
- After a long investigation session where multiple findings accumulated

This file covers three concerns in one:
- **L.1–L.3:** Reproducibility — is the result trustworthy?
- **L.4:** Release gate — is the model ready for Production?
- **L.5:** Session handoff — is the session context persisted?

Always load alongside: `I_regression_guard.md` (promotion gates) and
`F_new_run_checklist.md` F.3 (post-run documentation requirements).

**End-to-end workflow**: For the full post-training → promotion workflow
(reproducibility check, performance analysis, OOD/contamination check,
final report, save artifacts, MLflow promotion, monitoring handover),
see `data_module/temp/live_plans/post_training_process_2026-06-14.md`.
This file (L) is the per-gate detail; that doc is the runner.

---

## L.1 — Reproducibility Pre-Conditions

"Run N is better than Run M" is only a defensible claim if both runs are
reproducible. These checks must be done before making any such claim.

### L.1.1 — RNG Seed

Read the checkpoint `.pt` file metadata and the `epoch_summary.jsonl` header
to confirm the seed used for the run. Do not assume the seed from memory.

- Confirm `seed` is recorded in the checkpoint `config` dict
- Confirm it matches the seed in `MEMORY.md` Training History for that run
- If seeds differ between checkpoint and MEMORY.md, the Training History entry
  is the suspect — correct it before making any reproducibility claims

### L.1.2 — Tokenizer Determinism

CodeBERT tokenization is deterministic when run offline. Confirm:

```bash
echo $TRANSFORMERS_OFFLINE
```

Expected: `1`. If unset, online tokenizer calls may produce non-deterministic
behaviour across sessions. Set before any training or evaluation:

```bash
export TRANSFORMERS_OFFLINE=1
```

### L.1.3 — Data Export Hash Verification

DVC was retired for the data pipeline (Stage 7B seam swap, 2026-06-12). The
v3 export is the source of truth. Verify the export's artifact hash matches
the run's recorded `export_artifact_hash` in `epoch_summary.jsonl`:

```bash
# Active v3 export: data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/
# Verify its manifest:
python -c "import json; print(json.load(open('data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/MANIFEST.json'))['artifact_hash'])"
# Compare to the value recorded in the run's epoch_summary.jsonl
jq '.[0].export_artifact_hash' ml/logs/<run_name>/epoch_summary.jsonl
```

Both values must match. A mismatch means the data was re-exported between
the run start and the current check — do not claim reproducibility without
re-running on the original data version.

For split files:
```bash
sha256sum data_module/data/splits/v3/{train,val,test}.jsonl
```
Compare against the SHA recorded in the run's pre-flight log.

### L.1.4 — Dependency Lockfile

Confirm `poetry.lock` has not changed between the runs being compared:

```bash
git diff HEAD~1 poetry.lock --stat
```

Any diff in `poetry.lock` means a dependency version changed, which may affect
results. If the lockfile changed, the comparison is not apple-to-apple —
document the dependency delta before drawing conclusions.

---

## L.2 — Two-Source Metric Confirmation

A metric is not confirmed until it has been read from two independent sources
and they agree. This is Rule 2 Layer 2 from `00_rules.md`, applied to metrics.

### L.2.1 — F1 Two-Source Check

For any `val_f1_macro` claimed as the basis for a promotion or comparison:

**Source 1:** `epoch_summary.jsonl` — `f1_macro_tuned` field of the best epoch row:
```bash
jq 'select(.f1_macro_tuned == ([inputs | .f1_macro_tuned] | max))' \
    ml/logs/epoch_summary.jsonl 2>/dev/null | tail -1
```
(Adjust path to the correct run's log file.)

**Source 2:** MLflow model registry — read from the registered run via:
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
versions = client.get_latest_versions("sentinel-vulnerability-detector", stages=["Staging"])
run = client.get_run(versions[0].run_id)
print(run.data.metrics["val_f1_macro"])
```

Both sources must agree within tolerance. Read the run config or `MEMORY.md`
for the declared tolerance — do not hardcode a tolerance value here. If the
two sources disagree by more than tolerance, that disagreement is itself a
finding and must be documented before the value is used in any decision.

### L.2.2 — Per-Class F1 Source

Per-class F1 values should be read from the benchmark run output
(`benchmark_run9_*.py` or equivalent), not from the training log.
Training log per-class F1 is computed on the validation split; benchmark
F1 is computed on the held-out test benchmark. These are different numbers.
Do not conflate them in any written finding or comparison.

---

## L.3 — When Results Are Non-Reproducible

If a re-run produces a different F1 than the original:

1. Check L.1.1–1.4 first — confirm seed, tokenizer mode, DVC status, lockfile
2. If all four match and results still differ, the delta is genuine variance —
   document the delta (original value, re-run value, difference)
3. Investigate seed sensitivity: if the delta is > 0.01 F1, the result is
   sensitive to initialisation. Document as a finding, do not average away
4. Do not promote a checkpoint whose F1 cannot be reproduced within tolerance
   on a second run with the same configuration

---

## L.4 — Production Release Gate

This is the final gate before a Production promotion. It aggregates all
previous procedures. Each item must have a written attestation from the
referenced procedure — do not re-run; confirm the attestation exists and is
dated after the current checkpoint was saved.

| Gate | Procedure | Required attestation |
|---|---|---|
| Data pipeline verified | `B_data_pipeline.md` B.3 | Written to run doc |
| Contamination check passed | `A_benchmark_runs.md` A.1 | Written to run doc |
| Smoke suite passed | `D_smoke_preflight.md` D.1 | Written to run doc |
| VRAM gate passed | `D_smoke_preflight.md` D.2 | Written to run doc |
| Calibration files present | `I_regression_guard.md` I.3.2 | `<stem>_thresholds.json` + `<stem>_temperatures.json` exist |
| **Behavioural probes pass** | **`C_diagnostic_checks.md` C.2.4** | **`<stem>_behavioral_probes.json` `all_passed=true`** |
| **Label quality OK** | **`F_new_run_checklist.md` F.1.0** | **`<stem>_label_quality.json` no FAILs** |
| Drift baseline is `source=warmup` | `I_regression_guard.md` I.3.5 | Confirmed from baseline JSON |
| F1 dry-run gate passed | `I_regression_guard.md` I.3.6 | Dry-run output printed/logged |
| Two-source F1 confirmed | L.2.1 above | Both sources read and agree |
| Reproducibility checks done | L.1.1–1.4 above | Seed, tokenizer, DVC, lockfile |
| **L.4.1 Auto-reproducibility** | **NEW (2026-06-17)** | **Auto-run hash compare; see below** |
| No open KILL-level issues | `H_issue_triage.md` H.3 | No unresolved KILL issues in `ISSUES.md` |
| API validation passed | `K_inference_api.md` K.3–K.5 | `/health` verified, round-trip tests passed |
| `MEMORY.md` Training History updated | `F_new_run_checklist.md` F.3 | Entry added for this run |

If any gate is missing its attestation: do not promote. Run the referenced
procedure and produce the attestation first.

### L.4.1 — Auto-Reproducibility Check (NEW 2026-06-17)

**Why this exists:** The previous L.4 was a checklist that required humans
to confirm reproducibility by re-running. Humans forget, skip steps, or
declare "looks reproducible" without verification. L.4.1 makes this
automated.

**What it does:** Before any Production promotion, automatically:
1. Re-run the same evaluation suite on the same checkpoint
2. Compare outputs to the recorded evaluation results
3. Hash the model + tokenizer state
4. Confirm seed, lockfile, tokenizer mode haven't changed

**Run the check:**

```bash
# Compare current checkpoint to recorded results
python ml/scripts/auto_reproducibility_check.py \
    --checkpoint ml/checkpoints/Run12_FINAL.pt \
    --reference ml/checkpoints/Run12_reference_eval.json \
    --output ml/checkpoints/Run12_reproducibility.json
```

**Pass criteria:**
- Hash of model state matches reference (within tolerance for non-deterministic ops)
- Re-run F1 within ±0.005 of reference
- Seed, lockfile, tokenizer mode all match

**Why this matters:** A model that produces different F1 on re-run is
not safe to promote. Either there's a non-determinism source (e.g., a
random seed not set), or the model is unstable. Either way, block promotion.

---

## L.5 — Session Handoff

This section is the enforcement mechanism for Rule 3 (no floating findings)
from `00_rules.md`. Complete it before ending any session where more than
one meaningful finding, decision, or change was made.

### L.5.1 — Auto-Detection of Floating Findings (NEW 2026-06-17)

**Why this exists:** Rule 3 (no floating findings) was enforced by humans
remembering to write findings. L.5.1 makes it automated.

**What it does:** Before any session close, the L.5 script (`ml/scripts/
session_close.py`) automatically:
1. Scans the conversation log for finding patterns (e.g., "BUG-", "found",
   "discovered", "issue", "FP", "FN", "broken", "wrong")
2. For each candidate, checks if it has a corresponding entry in
   `ISSUES.md` or relevant audit doc
3. For any finding WITHOUT a corresponding written entry, raises a
   `FloatingFindingError` and refuses to close the session
4. Writes a `## Open Questions` block to MEMORY.md with the unresolved items

**This is the AI-assistance layer.** Humans don't have to remember to write
findings — the script catches what they forgot.

**Run the check:**

```bash
# At session close
python ml/scripts/session_close.py --session-log <path>
```

### L.5.2 — Mandatory Writes Before Session Close

In order, before closing the session:

1. **`MEMORY.md` Current State** — update to reflect actual current state:
   - Current best checkpoint and its F1
   - Current Production model (if changed)
   - Any ongoing investigation or open blocker
   - Schema version (if changed)

2. **`MEMORY.md` Training History** — if a run was completed:
   - Add a row with run name, epoch, val F1 (tuned), key config changes, outcome
   - Read the existing table format before adding — do not invent new columns

3. **Open bugs/findings** — for each finding discovered during this session:
   - Write to `ISSUES.md` (or relevant audit doc) per `H_issue_triage.md` H.5
   - Assign a `BUG-<ID>` or `FIND-<ID>` identifier
   - Do not leave findings only in the conversation

4. **Decisions as ADR entries** — for any architectural or procedure decision
   made during the session: confirm it is recorded in `docs/ml/adr/` or in the
   relevant spec file via a proper update (not just in `MEMORY.md`)

5. **Open questions** — any unresolved question that affected work during the
   session: append to an `## Open Questions` section in the most relevant doc

### L.5.3 — Verification Before Close

Answer each of these before ending the session:

- Is there any finding that exists only in this conversation? → Write it now
- Is there any plan or decision that exists only in this conversation? → Write it now
- Could a fresh Claude session (with no memory of this conversation) resume work
  correctly using only the written docs? → If no, what is missing?

### L.5.4 — Handoff Summary

Append to `MEMORY.md` before closing:

```
## Session Handoff — <ISO date>
Completed: [what was done this session]
Open: [what was not finished]
Next session must read: [specific files, by name]
Blockers: [anything preventing progress, or "none"]
```

Keep each field to one or two lines. The purpose is to give the next session
a 30-second orientation, not a transcript.

---

## L.6 — Completion Attestation

After completing a release readiness check or session handoff, append to the
relevant doc:

```
## Procedure Attestation — L_release_readiness — <ISO date>
Scope: Release gate / Session handoff / Both
Steps completed:
  L.1.1 RNG seed confirmed:              YES/NO (seed: <value>)
  L.1.2 TRANSFORMERS_OFFLINE=1:         YES/NO
  L.1.3 DVC status clean:               YES/NO
  L.1.4 poetry.lock unchanged:          YES/NO
  L.2.1 F1 two-source confirmed:
    epoch_summary value:                 <value>
    MLflow value:                        <value>
    Agreement:                           YES/NO (delta: <value>)
  L.2.2 per-class F1 source correct:    YES/NO/N/A
  L.4 release gate:
    All attestations present:            YES/NO
    Any gate missing:                    [gate name] / none
  L.5.1 mandatory writes done:          YES/NO
  L.5.2 floating findings cleared:      YES/NO
  L.5.3 handoff summary written:        YES/NO
Steps skipped:     [any skipped + explicit reason]
Promotion outcome: APPROVED / BLOCKED (reason) / N/A
Written to:        [path of this attestation]
```
