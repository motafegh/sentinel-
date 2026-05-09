# SENTINEL Autoresearch Harness

Automated hyperparameter search for the v4 retrain sprint.

Goal: find a knob combination where `auto_experiment.py --regime confirm`
produces a **tuned F1-macro > 0.5069** with no per-class floor breach —
beating the v3 baseline on the fixed `val_indices.npy` held-out split.

---

## How it works

```
auto_experiment.py --regime smoke   →  1 epoch · 10% train subsample · ~3–5 min
auto_experiment.py --regime confirm →  5 epochs · full train · ~30–60 min
```

The agent (Claude or a human) iterates: smoke → if F1 looks promising →
confirm → keep branch or discard. The skill file `program.md` specifies
the exact search space, floor constraints, and stop conditions.

---

## Directory layout

```
ml/autoresearch/
├── README.md           This file
├── program.md          Skill file — the agent reads this before acting
└── runs/               Per-run log files (created at runtime, untracked)
    └── auto-<tag>-<idx>.log

ml/autoresearch/results.tsv   Ledger of all runs (append-only, untracked)
```

---

## Starting a session

```bash
# 1. Ensure you're on a clean base branch
git checkout main
git pull

# 2. Verify the hash guard passes
poetry run python ml/scripts/compute_locked_hashes.py --check
# → OK: all 5 entries match.

# 3. Verify trainer imports
poetry run python -c "from ml.src.training.trainer import TrainConfig, train; print(TrainConfig())"

# 4. Create a session branch
git checkout -b autoresearch/2026-05-<date>

# 5. Make the runs log directory
mkdir -p ml/autoresearch/runs

# 6. Point your agent at program.md and start
#    (or run the loop manually — see program.md §5)
```

---

## Running a single smoke experiment (manual)

```bash
poetry run python ml/scripts/auto_experiment.py \
    --regime smoke \
    --run-name auto-v4-001 \
    --experiment-name sentinel-retrain-v4 \
    --loss-fn focal --gamma 2.0 --alpha 0.25 \
    --lora-r 8 --lora-alpha 16 \
    --batch-size 16 --lr 3e-4 \
    > ml/autoresearch/runs/auto-v4-001.log 2>&1

grep '^SENTINEL_SCORE=' ml/autoresearch/runs/auto-v4-001.log
```

## Running a confirm experiment

```bash
poetry run python ml/scripts/auto_experiment.py \
    --regime confirm --max-epochs 5 \
    --run-name auto-v4-001-confirm \
    --experiment-name sentinel-retrain-v4 \
    --loss-fn focal --gamma 2.0 --alpha 0.25 \
    --lora-r 8 --lora-alpha 16 \
    --batch-size 16 --lr 3e-4 \
    > ml/autoresearch/runs/auto-v4-001-confirm.log 2>&1

grep '^SENTINEL_SCORE=' ml/autoresearch/runs/auto-v4-001-confirm.log
```

## Validate a winner manually

```bash
# Re-run tuning on an existing checkpoint without retraining
poetry run python ml/scripts/tune_threshold.py \
    --checkpoint ml/checkpoints/<winner>_best.pt

# Then check the macro F1 against 0.5069
cat ml/checkpoints/<winner>_best_thresholds.json | python -c \
    "import json,sys; d=json.load(sys.stdin); print('Tuned F1-macro:', d['overall_metrics']['f1_macro'])"
```

---

## Stopping a session

`Ctrl-C` stops the current run. The most recently committed branch is the
current candidate. To resume:

```bash
# See what candidates exist
git branch | grep autoresearch/

# Resume from a specific candidate checkpoint
poetry run python ml/scripts/auto_experiment.py \
    --regime confirm \
    --run-name auto-resume \
    --resume-from ml/checkpoints/<candidate>_best.pt \
    ...same knobs...
```

---

## Reviewing results

```bash
# Tabular view of all runs
column -t -s $'\t' ml/autoresearch/results.tsv

# MLflow UI (compare all sentinel-retrain-v4 runs)
mlflow ui --port 5000
```

---

## Promoting a winner

```bash
poetry run python ml/scripts/promote_model.py \
    --checkpoint ml/checkpoints/<winner>_best.pt \
    --stage Staging \
    --val-f1-macro <confirmed_f1> \
    --note "v4 winner: <knob summary>"
```

Then update `docs/STATUS.md` "Active Checkpoint" and create a dated changelog
under `docs/changes/`.

---

## VRAM budget (RTX 3070 8 GB)

| Setting | Max safe |
|---------|---------|
| batch_size | 16 (8 is safer with lora_r=16) |
| lora_r | 16 (32 → OOM) |
| batch_size=32 | forbidden |
| use_amp=False | forbidden (must be True on 8 GB) |

`PEAK_VRAM_MB` in each log shows actual peak. Values > 7800 MB are risky.
OOM exits with code 2 — reduce batch or lora_r before retrying.

---

## Known issues

1. **Smoke F1 ≠ confirm F1**: smoke uses a 10% subsample and 1 epoch.
   The 0.42 promote threshold is a heuristic — calibrate after the first
   5 confirmed runs if it feels too conservative or too loose.

2. **DoS class (DenialOfService)**: 137 training samples, lowest support.
   `--weighted-sampler DoS-only` upsamples it 39×. Focal loss also helps.
   The floor is 0.35 F1 — achievable but tight without remediation.

3. **Drift between smoke and confirm**: a smoke winner with F1=0.44 may
   confirm at 0.50 or 0.49 — the variance is real. Confirm before promoting.

4. **WSL2 thermal throttle**: long confirm runs on AC power are stable.
   Battery power throttles the GPU. Prefer plugged-in overnight sessions.
