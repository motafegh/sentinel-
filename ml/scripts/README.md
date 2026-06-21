# ml/scripts — Executable Scripts

Entry-point scripts for training, evaluation, auditing, promotion, monitoring, and
diagnostics. Each script has a single responsibility (per `ml/CLAUDE.md` convention).

**Run from repo root** with the ML venv active:
```bash
source ml/.venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_cache
PYTHONPATH=. python ml/scripts/<script>.py [args]
```

---

## Training

### `train.py`

Training entry point. Parses 50+ CLI args into `TrainConfig`, calls `train()` from
`ml/src/training/trainer.py`.

**Produces:** `ml/checkpoints/<run_name>_best.pt`, `ml/logs/<run_name>/` (3-stream JSONL)

**Key args:**
- `--run-name` — must include date (e.g. `v10-20260603`)
- `--epochs` (default 100), `--batch-size` (default 8), `--gradient-accumulation-steps` (default 8)
- `--loss-fn` — `asl` (default), `focal`, `bce`
- `--resume` — resume from checkpoint
- `--no-jk` — disable JK attention (ablation)
- `--gnn-prefix-k` — GNN prefix injection (0=disabled, 48 for Phase 1)
- `--smoke-subsample-fraction` — fraction of data for smoke runs

**Gates checked at startup:** GNN layer count (warns if != 8 for three-phase arch)

---

## Post-Training Pipeline

The standard post-training workflow is:

```
train.py → tune_threshold.py → calibrate_temperature.py → promote_model.py
```

### `tune_threshold.py`

Per-class decision-threshold optimiser. Sweeps 19 thresholds per class over [0.05, 0.95]
in 0.05 steps, picks the threshold that maximises per-class F1.

**Reads:** checkpoint `.pt`, v2 export dir (`data_module/data/exports/sentinel-v2-baseline-2026-06-12/`)
**Produces:** `<checkpoint_stem>_thresholds.json` next to the checkpoint

**Architecture-aware:** loads model config from checkpoint, handles edge_embedding resize
when checkpoint predates current schema, strips `_orig_mod.` prefix from torch.compile.

### `calibrate_temperature.py`

Post-training per-class temperature scaling. Fits one scalar temperature T_c per class
(10 classes) by minimising BCE NLL on the validation set via L-BFGS.

**Reads:** checkpoint `.pt`, legacy cache (`cached_dataset_v9.pkl`)
**Produces:** `ml/calibration/temperatures_<run>.json`, `_stats.json`, `_ece_comparison.png`

**Dependencies:** Uses shared utilities from `ml/scripts/interpretability/utils.py` (`load_model`,
`load_val_split`, `collect_predictions`, `add_common_args`).

### `promote_model.py`

Promotes a checkpoint to the MLflow Model Registry. Logs the checkpoint as an MLflow
artifact, registers it, and transitions it to the requested stage.

**Enforced gates (all stages):**
1. `<stem>_behavioral_probes.json` must exist and `all_passed=true`
2. `ml/checkpoints/v3_label_quality.json` must exist with no FAILs

**Additional Production gates:**
3. `val_f1_macro` must exceed current Production model's F1
4. `--require-baseline` path must exist and have `source='warmup'`

**Produces:** MLflow run with checkpoint + thresholds logged as artifacts

---

## MLOps Operations

### `set_active_checkpoint.py`

Atomic update of `ml/mlops_config.json` to point at a new checkpoint. Also auto-detects
companion `_thresholds.json` next to the checkpoint. Uses `.tmp` + rename for crash safety.

```bash
python ml/scripts/set_active_checkpoint.py GCB-P1-Run12-v3dospatched-20260613_FINAL
```

### `build_warmup_baseline.py`

Generates synthetic warmup JSONL for drift baseline building. Until real production
traffic is available, this produces realistic synthetic data (distributions derived from
v3 training export + SmartBugs Wild eval stats).

**Produces:** `<output>.jsonl` — records with `num_nodes`, `num_edges`, `confirmed_count`, `suspicious_count`

**Next step after running:** `compute_drift_baseline.py --source warmup`

### `compute_drift_baseline.py`

Builds `drift_baseline.json` for T2-B KS drift detection. Two modes:
- `--source warmup` (recommended): reads JSONL from API warmup buffer
- `--source training` (NOT recommended — causes false alerts on modern contracts)

**Requires:** minimum 30 warmup records for reliable baseline.

---

## Data Integrity

### `check_contamination.py`

Four-tier contamination audit checking whether SmartBugs Curated contracts leaked
into the BCCC training corpus:

1. **Tier 1** — Exact content hash (SHA256 of raw bytes)
2. **Tier 2** — Normalised content hash (strip comments, collapse whitespace)
3. **Tier 3** — Token Jaccard similarity (≥0.75 = near-duplicate)
4. **Tier 4** — Structural graph fingerprint (num_nodes, num_edges, function names)

```bash
PYTHONPATH=. python -m ml.scripts.check_contamination --jaccard-threshold 0.75
```

### `dedup_multilabel_index.py`

Content-deduplicates `multilabel_index.csv`. BCCC stores the same `.sol` in multiple
category directories; path-based MD5 creates separate rows. This script:
1. Hashes content (not path) of every `.sol`
2. Groups rows sharing a content hash
3. Merges labels with OR
4. Rebuilds stratified 70/15/15 splits

Optional `--relabel-timestamp`: source-verified Timestamp label relabeling using both
graph features (`feat[2]` = uses_block_globals) and source patterns (block.timestamp, etc.)

---

## Validation Gates

### `compile_smoke_test.py`

Gate 3.1: torch.compile smoke test. Three checks:
1. Compile succeeds on all targeted submodules
2. Compiled forward pass numerically close to eager (max abs diff < 0.05)
3. 2-epoch training stability with variable batch shapes

```bash
TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/compile_smoke_test.py
```

### `vram_gate_test.py`

Gate 5.3: VRAM worst-case test for `fusion_max_nodes=2048`. Runs a realistic
training step (forward + backward + optimizer.step + AMP scaler) on synthetic
graph data at max node count.

Thresholds: PASS < 7500 MB, WARN 7500-8000 MB, FAIL > 8000 MB.

### `auto_reproducibility_check.py`

L.4.1: Confirms model state, git commit, and lockfile match a reference before
promotion. Blocks promotion if hashes don't match.

### `session_close.py`

L.5.1: Auto-detects floating findings at session close. Scans recent files and
session logs for finding patterns (FP, FN, bug discovered, etc.), checks if each
has a corresponding `ml/audit_docs/ISSUES.md` entry. Refuses to close if findings
are unwritten.

---

## Benchmarks

### `benchmark_run9_smartbugs.py`

Evaluates Run 9 on SmartBugs Curated (143 contracts, 6 mappable categories + 4 FP
probe categories). Reports per-class P/R/F1 under both tier thresholds (0.55/0.25)
and tuned per-class thresholds.

### `benchmark_run9_solidifi.py`

Evaluates Run 9 on SolidiFI (350 contracts, 7 categories, 341/350 clean of BCCC
contamination). Includes near-dup flagging for 9 Unchecked-Send contracts.

### `diag_per_eye_solidifi.py`

Per-eye diagnostic on SolidiFI. Runs forward pass with `return_aux=True` to capture
predictions from all four eyes (GNN, Transformer, Fused, CFG/Phase2) plus the combined
output. Produces top-K rank analysis, per-eye average probability tables, and notable
contract breakdowns.

---

## Cron / Monitoring

### `check_run12_status.sh`

Cron-based training monitor (every 5 min). Reads epoch_summary.jsonl, detects:
- Process death → critical notification
- New epoch complete → log + notification
- New best F1 → highlighted notification
- NaN/Inf in metrics → critical notification
- New alert in alerts.jsonl → notification
- Stalled training (30 min no step_metrics update) → warning

Uses state file `ml/logs/.run12_check_state` to track last-known best F1 + epoch.
Notifications: tries `notify-send` → PowerShell toast → silent log.

```cron
*/5 * * * * /home/motafeq/projects/sentinel/ml/scripts/check_run12_status.sh
```

### `push_log_snapshot.sh`

Cron-based JSONL log snapshot (every 30 min during training). Copies the three
JSONL streams from the active run to `ml/training_snapshots/<run_name>/`, commits,
and pushes to origin/main.

```cron
*/30 * * * * /home/motafeq/projects/sentinel/ml/scripts/push_log_snapshot.sh
```

---

## Subdirectories

### `smoke/` — Infrastructure Smoke Tests

Master runner + 8 per-fix smoke tests for validating infrastructure changes.

| Script | Tests |
|--------|-------|
| `run_all.py` | Master runner — runs smoke_fix{1..8} in phase-aware order |
| `_common.py` | Shared utilities (paths, pass_/fail_, schema checks, graph loading) |
| `smoke_fix1.py` – `smoke_fix8.py` | Individual fix validation tests |
| `test_drift_detector_a1.py` | Drift detector unit test |
| `test_phase_a_final.py` | Phase A final validation |
| `test_run12_loads_a5.py` | Run 12 checkpoint loading |
| `verify_c21_path.py` | C.2.1 inference path verification |
| `env_check.py` | Environment variable checks |
| `move_checkpoints_to_archive.py` | Archive old checkpoints |

**Usage:** `python ml/scripts/smoke/run_all.py` (all), `--phase 1` (one phase), `--fix 2` (one fix)

### `eval/` — Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_run12_on_v0.py` | Run 12 on v0.1 OOD benchmark (66 contracts, 6 classes) |
| `round_trip_v3.py` | Round-trip test on v3 data |
| `smartbugs_wild_full_eval.py` | Full 47K SmartBugs Wild evaluation |
| `smartbugs_wild_speed_test.py` | Speed benchmark on SmartBugs Wild subset |
| `famous_contracts_test.py` | Test on well-known DeFi contracts |
| `verify_wild_predictions.py` | Verify wild evaluation predictions |

### `audit/` — Contamination & Data Quality

| Script | Purpose |
|--------|---------|
| `check_contamination_v3.py` | v3 training data contamination check |
| `check_contamination_wild.py` | SmartBugs Wild contamination check |
| `analyze_wild_ood.py` | OOD subset analysis of wild evaluation |
| `calibrate_temperature_v3.py` | Temperature calibration for v3 |
| `bccc/` | BCCC-specific audit scripts (aderyn_retry, deep_dive_v2, etc.) |

### `interpretability/` — Model Behavior Analysis

34 experiment scripts organized by series:

- **A-series** (audit): pooling audit, CFG inheritance, JK entropy, aux contribution
- **B-series** (behavior): gradient norm, per-eye ECE, JK weight distribution, saliency
- **E-series** (edge): receptive field, WL distinguishability, message propagation, direction sensitivity
- **L-series** (learning): JK weight analysis, edge ablation, attention visualization, gradient saliency,
  probing classifiers, counterfactual contracts, calibration-size analysis, permutation importance,
  attention rollout, training ablation
- **S-series** (structural): structural trace, edge enrichment, feature distribution, ICFG path audit
- **val scripts**: validate findings from A/E experiments

**Shared:** `utils.py` (model loading, prediction collection, common args), `test_contracts/`

### `util/` — Shell Utilities

| Script | Purpose |
|--------|---------|
| `launch_eval.sh` | Launch evaluation jobs |
| `watch_smartbugs_eval.sh` | Monitor SmartBugs evaluation progress |

### `_legacy_data_pipeline/` — Deprecated

Old data pipeline scripts (archive_v8_data, build_multilabel_index, create_cache,
create_splits, reextract_graphs, retokenize_windowed, validate_graph_dataset). Kept
for reference; replaced by `data_module/` pipeline.

### `archive/` — Historical Scripts

~50 archived scripts from earlier development phases (audit tasks, label cleaning,
augmentation, validation sections, etc.). Not used in current workflow.

---

## Script Dependency Map

```
train.py
  → ml/src/training/trainer.py (TrainConfig, train())

tune_threshold.py
  → ml/src/datasets/ (SentinelDataset, sentinel_collate_fn)
  → ml/src/inference/predictor.py (_ARCH_TO_FUSION_DIM)
  → ml/src/models/sentinel_model.py (SentinelModel)

calibrate_temperature.py
  → ml/scripts/interpretability/utils.py (load_model, load_val_split, collect_predictions)

promote_model.py
  → mlflow (tracking, model registry)
  → checks: behavioral_probes.json, label_quality.json, drift_baseline.json

check_contamination.py
  → ml/src/inference/preprocess.py (ContractPreprocessor for Tier 4)
  → ml/src/utils/hash_utils.py (get_contract_hash)

dedup_multilabel_index.py
  → ml/src/utils/hash_utils.py (get_contract_hash)

set_active_checkpoint.py
  → ml/mlops_config.json (read/write)
```
