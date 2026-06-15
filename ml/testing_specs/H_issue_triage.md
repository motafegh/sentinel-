# H — Issue Triage

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.
>
> **Last revised: 2026-06-14** (post-Run-12 launch). **No routing-table changes** — the 11 alert codes in §H.2 are still valid and the source-of-truth is unchanged. Observed in Run 12: 18 alerts fired by ep44, all `WARN` tier `[9.3.6b]` (AUC-PR<0.1, minority classes) and `[9.3.6c]` (F1-AUC divergence). Both are expected for minority classes and do not require abort (per §H.4 decision tree: class death + F1 still improving = MONITOR).

---

## When This File Applies

- An alert fires in `alerts.jsonl` during training
- A guardrail counter trips (all-zeros collapse, class death, GNN collapse)
- A training run ends earlier than expected or produces suspicious metrics
- Deciding whether a signal requires immediate abort vs. continued monitoring
- Before escalating an observed symptom to a new `BUG-` entry

Always load alongside: `C_diagnostic_checks.md` (log reading) and
`G_ablation_protocol.md` G.2.4 (monitoring thresholds during a run).

---

## H.1 — Alert Tier Definitions

Read `ml/src/training/training_logger.py` docstring before acting on any alert.
Three tiers are defined there:

| Tier | Constant | Behaviour | Required response |
|---|---|---|---|
| `KILL` | `training_logger.KILL` | Raises `TrainingAbortError` immediately | **Stop. Do not save checkpoint.** Restart from last clean checkpoint. |
| `WARN_SKIP` | `training_logger.WARN_SKIP` | Logs alert, returns `skip=True` to caller | Batch skipped automatically; count skips; if skip rate > 0.5% investigate |
| `WARN` | `training_logger.WARN` | Logs alert, training continues | Monitor; triage using H.2; escalate if sustained |

**KILL checkpoints are corrupted.** The `TrainingAbortError` message says
"do NOT save checkpoint" explicitly. Never resume from a checkpoint saved
after a KILL event — Adam `exp_avg`/`exp_avg_sq` will be NaN, producing
irreproducible behaviour on the next run.

---

## H.2 — Alert-to-Action Routing Table

Locate the `[spec_code]` in the alert message, then follow the action. All
`[N.N.N]` codes are sourced from `training_logger.py` Spec references.

| Alert code | Source check | Immediate action | Escalate if |
|---|---|---|---|
| `[9.1.1]` KILL: NaN/Inf loss | `check_loss()` | Abort. Restart from last clean checkpoint. Check LR, BF16 overflow, data quality | Always file a finding |
| `[9.1.2]` KILL: NaN/Inf param | `check_parameters()` | Abort. Do not save. Identify the param name in the alert — it is logged. | Always file a finding |
| `[9.1.3]` KILL: Adam NaN state | `check_adam_state()` | Abort. Optimizer state permanently corrupted. Fresh AdamW required. | Always file a finding |
| `[9.2.1]` WARN_SKIP: all-zero labels | `check_batch()` | Counted automatically. If persistent across many epochs, check weighted sampler config and label CSV. | > 5% of batches skipped in a single epoch |
| `[9.2.2]` WARN_SKIP: NaN in graphs.x | `check_batch()` | Check graph re-extraction (E.6). May indicate stale `.pt` files. | Any single occurrence |
| `[1.3]` WARN: feature dim mismatch | `check_inputs()` | Read `graph_schema.py` — `NODE_FEATURE_DIM` must match. Likely schema migration needed (J). | Always |
| `[9.3.1]` WARN: VRAM > 7500 MB | `check_vram()` | Note headroom. If VRAM > 7900 MB, OOM is imminent. Reduce `batch_size` or `fusion_max_nodes`. | VRAM > 7900 MB |
| `[9.3.2]` WARN: grad norm spike | `check_grad_norm()` | Check `grad_clip` (default 1.0). Single spike is noise; sustained (3+ consecutive) = LR too high. | 3+ consecutive spikes |
| `[9.3.3]` WARN: aux_phase2 weight near zero | `check_aux_head()` | Phase 2 auxiliary head may be disconnected. Check `aux_phase2_loss_weight` ≠ 0.0 and `"phase2"` key in `aux` dict. | Weight < 1e-6 for 5+ epochs |
| `[9.3.4]` WARN: JK entropy < 0.5 | `check_jk_entropy()` | JK collapsed to single phase. Check `jk_entropy_reg_lambda` (default 0.005; 0.01 forced uniform in Run 3). | Entropy < 0.5 for 5+ epochs |
| `[9.3.6b]` WARN: AUC-PR < 0.1 | `compute_auc_metrics()` | Near-random probability signal for that class. Check positive count in split; consider `dos_loss_weight` or sampler. | Any minority class |
| `[9.3.6c]` WARN: F1-AUC divergence | `check_f1_auc_divergence()` | F1 improving but AUC-ROC degrading = model fitting threshold artefact. Check `eval_threshold` and label smoothing. | Sustained 3+ epochs |
| `[9.3.6d]` WARN: Brier > 0.4 | `compute_brier()` | Severe miscalibration. After run: run `calibrate_temperature.py` before reporting probabilities. | Any class |
| `[2.7]` WARN: loss spike > 5× | `check_loss()` | Single spike is acceptable. 3+ spikes in one epoch = instability. Reduce effective LR or check data quality. | 3+ spikes in one epoch |

---

## H.3 — Trainer Guardrail Counters

These are separate from the structured logger — they live inside the
`train()` epoch loop in `trainer.py` and log via `loguru`. Read the
`trainer.py` guardrails block (search `BUG-M10`) before interpreting them.

### H.3.1 — All-Zeros Collapse

- **Trigger:** `val_metrics["hamming"] > 0.85` for 3+ consecutive epochs
- **Meaning:** Model predicts all-zeros on everything
- **Log level:** `logger.critical()`
- **Investigation order:**
  1. Check `asl_gamma_neg` — 4.0 caused all-zeros collapse (BUG-C4; reduce to 2.0 default)
  2. Check weighted sampler mode (`use_weighted_sampler` config field)
  3. Check that `dos_loss_weight > 0.0` (0.0 = zero DoS gradient)
  4. Check `asl_clip` — 0.05 caused oscillation near the zero boundary (BUG-M2)

### H.3.2 — Class Death

- **Trigger:** `f1_{ClassName} == 0.0` for 5+ consecutive epochs
- **Meaning:** Model has stopped predicting that class entirely
- **Log level:** `logger.warning()`
- **Investigation order:**
  1. Check positive count for the dead class in the training split
     (`B.5` label quality checks)
  2. Check `pos_weight_min_samples` — if the class has < 3000 positives,
     its `pos_weight` should be > 1.0; if loss_fn="asl", pos_weight is not applied
  3. Check `class_label_smoothing` for that class — if set too high (> 0.3),
     positive signal is diluted to near-noise
  4. Check `eval_threshold=0.35` — if probabilities cluster at 0.30–0.34,
     the class appears dead at training-time eval but may recover with threshold tuning

### H.3.3 — GNN Collapse

- **Trigger:** `gnn_grad_share < 0.10` for 5+ consecutive epochs
- **Meaning:** GNN is contributing < 10% of gradient; model is relying entirely
  on CodeBERT/LoRA path
- **Log level:** `logger.critical()`
- **Investigation order:**
  1. Check `gnn_lr_multiplier` — default 2.5; if reduced, GNN share drops
  2. Check `fusion_lr_multiplier` — if > 0.5, CrossAttentionFusion dominates
     (observed: 4–5× GNN gradient norm at full fusion LR, RC1 fix)
  3. Check `gnn_use_jk` — JK disabled reduces GNN representational capacity
  4. Check LayerNorm in GNN — if gradients are vanishing through LN, the GNN
     receives near-zero signal regardless of LR

---

## H.4 — When to Abort vs. Monitor

Use this decision tree before stopping a run:

```
Alert fires
  ├─ KILL tier? ─────────────────────── ABORT immediately. Never save checkpoint.
  └─ WARN or WARN_SKIP
        ├─ Single occurrence, first time? ─── Log to findings doc. Continue.
        └─ Sustained (3–5+ consecutive epochs)?
              ├─ GNN collapse (H.3.3)? ────── Abort and adjust LR multipliers.
              ├─ All-zeros collapse (H.3.1)? ─ Abort. Fix loss/sampler config.
              ├─ VRAM > 7900 MB? ────────── Abort. Reduce batch size.
              ├─ Loss spikes > 3/epoch? ───── Abort. Reduce LR or grad_clip.
              └─ Class death (H.3.2) or JK
                  entropy < 0.5 sustained? ── Document. Allow run to complete
                                              unless Hamming also collapses.
```

**Minimum epoch before aborting for underperformance:** 30 epochs.
The `eval_threshold=0.35` was introduced specifically because stopping at
ep 30 while loss was still declining (observed in v5.x) was a false early
stop. Do not abort a run for low F1 before ep 30.

---

## H.5 — Filing a New Finding

When an alert or guardrail reveals a previously undocumented failure mode:

1. Assign a `BUG-<ID>` (next sequential after the current highest in the
   audit doc; do not reuse IDs)
2. Record: symptom, trigger condition, epoch first observed, config values
   at the time, and which alert code (if any) fired
3. Record the resolution or the investigation path if unresolved
4. Write to the audit doc before the session closes (Rule 3 — no floating findings)

Do not create a `BUG-` entry for:
- Single-occurrence WARN_SKIP on a large run (expected noise)
- JK entropy < 0.5 for 1–2 epochs early in training (initialisation transient)
- Loss spike on epoch 1 (common during aux warmup epochs 1–8)

---

## H.6 — Log File Locations

All three structured log files are written to `ml/logs/<run_name>/`
(controlled by `config.log_dir`, defaulting to that path). Read the
`StructuredLogger.__init__()` docstring in `training_logger.py` before
querying them — field names in `epoch_summary.jsonl` follow the 37-field
spec schema (Spec §8) and must not be assumed from memory.

| File | Content | Key fields |
|---|---|---|
| `step_metrics.jsonl` | Per-optimizer-step data | `step`, `epoch`, `loss`, `grad_norm_total`, `gnn_share`, `lr`, `vram_mb` |
| `epoch_summary.jsonl` | One JSON line per epoch | `epoch`, `train_loss`, `per_class_f1`, `auc_roc_per_label`, `brier_score_per_label`, `ece`, `jk_weight_entropy`, `loss_spike_count` |
| `alerts.jsonl` | KILL / WARN_SKIP / WARN events | `timestamp`, `level`, `message`, plus alert-specific fields |

---

## H.7 — Completion Attestation

After triaging an alert or guardrail event, append to the relevant run doc:

```
## Procedure Attestation — H_issue_triage — <ISO date>
Run: <run_name>
Alert / symptom: <alert code or guardrail name>
First observed: epoch <N> / step <N>
Alert tier: KILL / WARN_SKIP / WARN
Action taken: ABORT / MONITOR / SKIP
Investigation steps followed:
  H.2 routing table consulted:    YES/NO
  H.3 guardrail section read:     YES/NO / N/A
  H.4 abort decision applied:     YES/NO
Finding created: BUG-<ID> / NO
Steps skipped:    [any skipped + explicit reason]
Resolved:         YES / NO / ONGOING
Written to:       [path of this attestation]
```
