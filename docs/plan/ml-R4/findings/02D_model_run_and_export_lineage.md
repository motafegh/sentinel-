# 02D — Model Run and Export Lineage

- **Run ID:** R4-P1-LINEAGE-20260716
- **Phase:** 1
- **Date:** 2026-07-16
- **Status:** COMPLETE

## Confirmed Lineage

### Checkpoint: GCB-P1-Run12-v3dospatched-20260613_FINAL.pt

| Property | Value | Source |
|---|---|---|
| Export trained on | `sentinel-v3-smartbugs-2026-06-13` | Launch log line 4 |
| Export hash verified | YES (artifact hash check during dataset init) | Launch log line 8 |
| Train contracts | **18,027** | Launch log line 14 |
| Val contracts | **1,831** | Launch log line 25 |
| Test contracts | **0 (not loaded)** | Test set not used during training |
| Total contracts loaded | 19,858 | 18,027 + 1,831 |
| Total export size | 22,493 (manifest) | 2,635 not loaded (test set) |
| Architecture | four_eye_v8 | Launch log line 34 |
| Loss | AsymmetricLoss(gamma_neg=2.0, gamma_pos=1.0, clip=0.01) | Launch log line 36 |
| DataLoader | workers=4, pin_memory=True, AMP=True, TF32=True, grad_accum=8, effective_batch=64 | Launch log line 27 |
| Optimizer | AdamW with param-group LR multipliers | Launch log line 38 |
| Scheduler | OneCycleLR | trainer.py |
| Epochs trained | 51 (patience_counter=0 at save) | state.json |
| Best F1 | 0.6800766276074683 | state.json |
| Experiment name | sentinel-v12 | Launch log line 48 |
| MLflow tracking | sqlite:///mlruns.db, experiment created at launch | Launch log line 48 |
| Torch compile | Active on GNN/fusion/classifier/aux submodules | Launch log line 39 |
| LoRA | r=16 alpha=32 on query/value modules | Launch log line 32 |

### Split discrepancy: Run12 training vs current v3 split

| Metric | Run12 (from log) | Current Phase 0 v3 split | Delta |
|---|---|---|---|
| Train | 18,027 | 18,596 | -569 |
| Val | 1,831 | 1,983 | -152 |
| Test | 0 (not loaded) | 1,914 | -1,914 |
| Train+Val | **19,858** | **22,493** | **-2,635** |

The current v3 split (train=18,596 + val=1,983 + test=1,914 = 22,493) has **2,635 MORE contracts** than the Run12 training population (19,858). Possible explanations:

1. **Export regenerated after Run12** — the export `sentinel-v3-smartbugs-2026-06-13` may have been updated after Run12 training completed (repackaged shards, added sources)
2. **Split version mismatch** — Run12 may have used a v3-split variant (e.g., v3a, v3b) that excluded some contracts
3. **DoS patch effect** — 2,655 DoS labels were zeroed in a post-Run12 data audit patch. If contracts with ONLY DoS as their positive label were removed from training, this would explain the count difference (2,655 zeroed vs 2,635 fewer contracts)

**Additional finding:** The DoS/Reentrancy co-occurrence patch (2,655 DoS labels zeroed) was applied in the same time frame (2026-06-13). Run12 launched at 2026-06-13 23:31. If the DoS patch was applied BEFORE the export was regenerated but AFTER Run12 started, the export at Phase 0 time would include the DoS-patched version. This needs exact timestamp comparison.

### Checkpoint file chain

```
_best.pt (epoch 51, F1=0.68008)
  → byte-identical _FINAL.pt (same bytes, different name)
    → referenced in mlops_config.json as active checkpoint
      → DVC-tracked: remote=localbackup→/mnt/d/sentinel-dvc-remote
        → DVC md5: f1a04c12bda6ac4ebb0ba03b8b0b0cbc
```

### Threshold file chain

```
GCB-P1-Run12-v3dospatched-20260613_FINAL_thresholds.json
  → per-class F1-tuned thresholds at epoch 51 (or epoch 50 if tuned on validation set before final save)
    → referenced in mlops_config.json as active thresholds
      → DVC-tracked with checkpoint
```

### Calibration status

```
ml/calibration/temperatures_run12.json  (exists, hash recorded)
  → NOT loaded by inference API
    → mlops_config.json has no calibration_ref key
      → Explicit absence confirmed
```

## Indirect / Hypothesized Lineage

The following are naming hints, NOT confirmed evidence:

| Hint | Source | Status |
|---|---|---|
| "v3dospatched" in checkpoint name | Checkpoint filename | HINT — suggests v3 split with DoS label patch |
| 2,655 DoS labels zeroed | Data audit finding | CORROBORATES — DoS patch existed at Run12 time |
| 19,858 contracts vs 22,493 | Launch log vs Phase 0 split | DISCREPANCY — likely DoS patch + test set exclusion |

## MLflow Status

- Tracking URI: `sqlite:///mlruns.db`
- Experiment: `sentinel-v12` (created at Run12 launch, NOT the same as `sentinel-retrain-v2` in mlops_config.json)
- The experiment name in mlops_config.json is `sentinel-retrain-v2` — this may reference a different experiment
- **No runs queried from MLflow DB** (requires mlflow Python package or direct sqlite query)

## Unresolved Lineage Items

1. **Exact split version used by Run12** — not confirmed whether it was v3, v3a, v3-dospatched, or an ad-hoc subset. The 2,635-contract gap needs reconciliation.
2. **Test set not loaded** — Run12 did not evaluate on the test set during training. Final eval metrics may be from a separate evaluation script.
3. **Experiment name mismatch** — mlops_config.json says `experiment: sentinel-retrain-v2` but launch log shows MLflow experiment `sentinel-v12` was created. These may be different experiments.
4. **Threshold tuning epoch** — thresholds.json says F1-tuned, but whether tuning used val or test set is unclear.
5. **OOD evaluation** — `evaluate_run12_on_v0.py` may exist but was not read in this recovery.

## Recommended Phase 2 Actions

- Query MLflow `mlruns.db` for Run12 params to confirm split version, data config hash, and export artifact hash
- Compare export manifest hashes between Run12 training time and current Phase 0 to detect regeneration
- If export was regenerated after Run12, Phase 0's protected artifacts reference the POST-Run12 version, not the Run12 training version
