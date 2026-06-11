# G — Ablation Protocol

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.

---

## When This File Applies

- Running a controlled ablation (single flag change vs baseline)
- Comparing two run configs to diagnose a regression
- Before publishing per-class F1 numbers in any doc or issue
- When asked "does feature X help?" about any `TrainConfig` flag

Always load alongside: `F_new_run_checklist.md` (F.1 baseline gate) and
`D_smoke_suite.md` (D.1 smoke before committing to full ablation).

---

## G.1 — Ablation Flags in `TrainConfig`

These are the `TrainConfig` fields explicitly designed for ablation.
Each entry lists the default, the ablated value to test, and the expected
signal direction if the feature is working. All values sourced from
`ml/src/training/trainer.py`.

### G.1.1 — Architecture Flags

| Flag | Default | Ablated (off) | Expected direction |
|---|---|---|---|
| `gnn_use_jk` | `True` | `False` | Macro-F1 drops; CEI classes (Reentrancy, ExternalBug, TOD) most affected |
| `gnn_jk_mode` | `'attention'` | `'cat'` or `'max'` | Attention mode allows per-node routing; weaker modes collapse to global weight |
| `use_edge_attr` | `True` | `False` | Phase-specific edge masking lost; CFG/ICFG distinction disabled |
| `gnn_edge_emb_dim` | `64` | `0` (requires `use_edge_attr=False`) | Companion to `use_edge_attr` |
| `drop_complexity_feature` | `False` | `True` | Reduces complexity-proxy dominance; complexity held 34–36% gradient share (L4 experiment) |
| `appnp_alpha` | `0.0` (disabled) | `0.2` | Phase 1 teleport prevents CEI signal decay; Run 8 recommendation |
| `gnn_prefix_k` | `0` (disabled) | `48` | GNN prefix tokens injected into BERT; monitor `prefix_attention_mean` |
| `gnn_phase2_edge_types` | `None` (all types) | `[specific IDs]` | Phase 2 ablation: restricts CFG-mask edge types in Phase 2 |

### G.1.2 — Loss Function Flags

| Flag | Default | Ablated value | Note |
|---|---|---|---|
| `loss_fn` | `'asl'` | `'bce'` or `'focal'` | ASL is default since v6; BCE was used before; focal needs `focal_gamma` / `focal_alpha` |
| `asl_gamma_neg` | `2.0` | `0.0` or `4.0` | 4.0 caused all-zeros collapse (BUG-C4); reduce to test sensitivity |
| `asl_clip` | `0.01` | `0.05` | 0.05 caused oscillation at p≈0.03–0.06 (BUG-M2) |
| `dos_loss_weight` | `0.5` | `0.0` or `1.0` | 0.0 = zero DoS gradient; 1.0 = full gradient |
| `aux_loss_weight` | `0.3` | `0.0` | Disables all three pathway auxiliary heads |
| `aux_phase2_loss_weight` | `0.20` | `0.0` | Disables Phase 2 CEI-weighted auxiliary loss; 0.0 = no Phase 2 supervision |
| `jk_entropy_reg_lambda` | `0.005` | `0.0` or `0.01` | 0.01 forced uniform 33/33/33 JK weighting in Run 3 — do not exceed |

### G.1.3 — Per-Group LR Multipliers

These are tuned per-group LR multipliers in `TrainConfig`. Changing any one
requires a full run (not smoke) to see the gradient share effect:

| Group | Multiplier field | Default | Rationale |
|---|---|---|---|
| GNN (`model.gnn.*`) | `gnn_lr_multiplier` | `2.5` | GNN collapsed to ~10% gradient share by ep 8 in v5.1-fix28 |
| LoRA adapter | `lora_lr_multiplier` | `0.3` | Prevents CodeBERT catastrophic forgetting (was 0.5 in v5) |
| Fusion + classifier + aux heads | `fusion_lr_multiplier` | `0.5` | CrossAttentionFusion (821K params) produced 4–5× GNN gradient norm at full LR (RC1) |
| GNN prefix proj | `gnn_prefix_proj_lr_mult` | `5.0` | Cold-start for projection; raised from 1.0 (NH-5) |
| Other params | — | `1.0` (base) | Everything not in the above groups |

**Weight decay for LoRA:** `weight_decay=0.0` (set in the LoRA param group
only; rest of the model uses `config.weight_decay=1e-2`). Standard PEFT
practice — L2 decay competes directly with the LoRA adaptation signal.

### G.1.4 — Label Smoothing Flags

Per-class label smoothing (`class_label_smoothing`) calibrated to estimated
noise rates per class. Default values from `TrainConfig`:

```python
{
    "CallToUnknown":               0.10,
    "DenialOfService":             0.18,
    "ExternalBug":                 0.10,
    "GasException":                0.12,
    "IntegerUO":                   0.08,
    "MishandledException":         0.12,
    "Reentrancy":                  0.14,   # confirmed 14% noise (no external calls)
    "Timestamp":                   0.05,   # structural check exists; lower noise
    "TransactionOrderDependence":  0.10,
    "UnusedReturn":                0.10,
}
```

Ablation: set all values to `0.0` (or set to uniform `0.05` to compare
against per-class calibration). The uniform `label_smoothing` float field
is superseded by `class_label_smoothing`; do not set both.

---

## G.2 — Controlled Ablation Procedure

One ablation = one flag change vs a locked baseline. Never change two flags
simultaneously — the interaction effect cannot be attributed.

### G.2.1 — Before Starting

1. Verify the baseline checkpoint exists and has a recorded val F1-macro
   (from `F_new_run_checklist.md` F.1 or `ml/logs/<run_name>/`)
2. Confirm the splits used by the baseline (`splits_dir` in baseline config)
   — the ablation **must use the same splits**
3. Confirm `random_seed=42` (or document if baseline used a different seed)
4. Run smoke (`D.1`) on the ablated config before committing to a full run

### G.2.2 — Run Naming Convention

Ablation runs must be named to encode the change:

```
<base_run_name>-abl-<flag>-<value>
```

Examples:
- `v8-run1-abl-jk-off` (gnn_use_jk=False)
- `v8-run1-abl-appnp-0.2` (appnp_alpha=0.2)
- `v8-run1-abl-gamma_neg-0.0` (asl_gamma_neg=0.0)

This ensures MLflow runs are uniquely named and the checkpoint path encodes
what was changed.

### G.2.3 — Launching an Ablation Run

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/train.py \
    --run-name v8-run1-abl-jk-off \
    --experiment-name sentinel-multilabel \
    --splits-dir ml/data/splits/v10_deduped \
    --epochs 100 \
    --gradient-accumulation-steps 8 \
    # add the single ablated flag here, e.g.:
    # --no-gnn-use-jk
```

If the ablated flag does not have a CLI argument, edit `TrainConfig` directly
and document the change in the run's MLflow description before launching.

### G.2.4 — Monitoring During the Run

Watch these metrics in MLflow (or `ml/logs/<run_name>/epoch_summary.jsonl`)
to detect an ablation-induced failure before the run finishes:

- `gnn_grad_share` per epoch: should stay > 10%; below for 5 epochs = GNN collapse
- `jk_phase{1,2,3}_weight` (when JK is enabled): all should be > 5% after ep 3
- `ph2_ph1_grad_ratio`: near 0 = Phase 2 not receiving gradient (CEI patterns not learning)
- `val_f1_macro` vs baseline: if more than 0.05 lower by ep 30, the ablation is harmful
- `nan_batch_count`: any NaN/Inf at > 0.5% of batches = abort (A38 Gate)

### G.2.5 — Comparing Results

When both runs are complete, compare these metrics in this order:

1. **Val macro-F1** (primary): use `f1_macro_tuned` (per-class threshold sweep)
   rather than fixed-threshold `f1_macro` — fixed threshold hides minority-class
   improvement (eval_threshold=0.35 is training-time only; tuned is the fairer metric)
2. **Per-class F1** for CEI classes (Reentrancy=6, ExternalBug=2, TOD=8) — most
   sensitive to architecture changes
3. **Per-class F1** for minority classes (DoS=1, Timestamp=7, UnusedReturn=9)
4. **GNN grad share** mean over last 10 epochs — below 10% = structural GNN
   not contributing
5. **Training wall time** — use only to break ties, not as a primary criterion

**Minimum epoch requirement:** do not compare runs before epoch 30. The
`eval_threshold=0.35` patience setting was introduced specifically to prevent
early stopping while minority classes are still learning (observed: stopped
at ep 30, loss=0.8855, still improving — v5.x run).

---

## G.3 — Known Ablation Results (Do Not Re-Test Without New Justification)

These experiments have been run and their conclusions are locked. Re-running
wastes GPU time unless a code change invalidates the result.

| Flag | Ablated value | Result | Run / Evidence |
|---|---|---|---|
| `gnn_use_jk` | `False` | Phase-level signal loss; JK enabled in all v5.2+ runs | Phase 1-A1 2026-05-14 |
| `asl_gamma_neg` | `4.0` | All-zeros collapse (BUG-C4; 60% zero-label rows drove model to predict nothing) | BUG-C4 fix 2026-05-xx |
| `asl_clip` | `0.05` | Oscillation at p≈0.03–0.06 (BUG-M2) | BUG-M2 fix |
| `gnn_lr_multiplier` | `1.0` (no boost) | GNN share collapsed to ~10% by ep 8 | v5.1-fix28 observation |
| `fusion_lr_multiplier` | `1.0` (full LR) | Fusion 4–5× GNN gradient norm; CodeBERT Reentrancy bias overwhelmed GNN | RC1 fix 2026-05-16 |
| `pos_weight_min_samples` | `0` (no cap) | Reentrancy 2.82× amplification caused behavioral collapse v5.2 | BUG-H3 fix |
| `jk_entropy_reg_lambda` | `0.01` | Forced uniform 33/33/33 JK weights in Run 3 — no per-node routing | C-3 comment in `trainer.py` |
| `drop_complexity_feature` | `False` | Complexity held 34–36% gradient share (L4 experiment); ablation pending Run 8 | Config comment |
| `appnp_alpha` | `0.2` | Pending — scheduled for Run 8 | `appnp_alpha` default=0.0 comment |

---

## G.4 — Per-Class Threshold Tuning

After each full run, `tune_threshold.py` (called inside `evaluate()` via
`tune_thresholds=True`) sweeps 19 candidate thresholds per class over [0.1, 0.9]
and picks the per-class threshold that maximises each class's F1 on val.

- Tuned thresholds are saved in the checkpoint under `tuned_thresholds`
  (list of 10 floats, one per class, in CLASS_NAMES index order)
- On resume (Fix #35), cached thresholds are restored from the checkpoint
  so the sweep is not repeated until `threshold_tune_interval=10` epochs pass
- When comparing two ablation runs, always compare `f1_macro_tuned` —
  fixed `f1_macro` at `eval_threshold=0.35` is for training stability only
- The inference threshold (`config.threshold=0.5`) is separate from the
  tuned thresholds; inference uses the per-class tuned values, not 0.5

---

## G.5 — Completion Attestation

After completing this section, append to the ablation run doc:

```
## Procedure Attestation — G_ablation_protocol — <ISO date>
Run: <run_name>
Ablated flag: <field_name> = <value>  (baseline was <baseline_value>)
Baseline run: <run_name>
Same splits:   YES/NO  (if NO — results are not comparable)
Same seed:     YES/NO
Smoke passed:  YES/NO
Steps completed:
  G.2.1 baseline verified:          PASS/FAIL
  G.2.2 run name follows convention: YES/NO
  G.2.3 run launched:               YES/NO
  G.2.4 monitoring checks:
    gnn_grad_share > 10%:            PASS/FAIL
    nan_batch_count < 0.5%:          PASS/FAIL
    ph2_ph1_grad_ratio not near 0:   PASS/FAIL
  G.2.5 comparison (ep >= 30):
    val_f1_macro_tuned (ablated):    N
    val_f1_macro_tuned (baseline):   N
    delta:                           ±N
    per-class CEI delta:             [Reentrancy±N, ExternalBug±N, TOD±N]
    per-class minority delta:        [DoS±N, Timestamp±N, UnusedReturn±N]
    conclusion:                      HELPFUL / HARMFUL / NEUTRAL
Steps skipped:     [any skipped + explicit reason]
New findings:      [link to audit doc entry, or "none"]
Written to:        [path of this attestation]
```
