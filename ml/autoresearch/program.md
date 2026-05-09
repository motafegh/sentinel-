# SENTINEL Autoresearch Skill — v4 Sprint

You are an autoresearch agent driving the SENTINEL v4 hyperparameter search.
Your job is to find a set of knobs for which `auto_experiment.py --regime confirm`
produces a tuned F1-macro **> 0.5069** on `ml/data/splits/val_indices.npy`
with **no per-class F1 dropping more than 0.05 below the v3 floor**.

Read this file completely before acting. Do not skip any section.

---

## 1. Goal

Beat the v3 baseline on the held-out validation split:

```
gate:  tuned val F1-macro > 0.5069
floor: no class drops > 0.05 F1 from v3 tuned values (see §6)
split: ml/data/splits/val_indices.npy  — permanently locked; never regenerate
```

This is a **conservative hyperparameter search** over a fixed architecture.
The Architecture Playground (post-v4) is the place for structural changes.

---

## 2. Allowed file edits (v4 sprint)

You MAY edit:
- `ml/src/training/trainer.py` — pure Python additions/changes to training logic
- CLI args to `ml/scripts/auto_experiment.py`

You MUST treat as READ-ONLY (do not edit, do not `git rm`):
- `ml/src/preprocessing/` — graph schema (locked)
- `ml/src/models/` — architecture (locked)
- `ml/data/splits/` — eval split (permanently locked)
- `ml/data/graphs/`, `ml/data/tokens/` — data freeze for v4
- `zkml/`, `contracts/` — downstream, not your concern
- `ml/scripts/auto_experiment.py` — the harness itself (edit trainer.py instead)

The locked-file hash guard inside `auto_experiment.py` will **exit 1** if any
of the architecture/schema files differ from the committed hashes, before
training starts. A hash mismatch means you edited something forbidden.

---

## 3. Forbidden changes

- Modifying `fusion_output_dim`, `NUM_CLASSES`, `GNNEncoder.in_channels`,
  `ARCHITECTURE`, or `edge_attr` shape — these are v4-sprint locks.
- `poetry add` or any new dependency.
- Regenerating splits (`create_splits.py`) or graph files (`ast_extractor.py`).
- Using an MLflow experiment name other than `sentinel-retrain-v4`.
- Editing `val_indices.npy` or anything in `ml/data/splits/`.

---

## 4. Search space (knobs you may vary)

All runs fine-tune from the v3 checkpoint (`ml/checkpoints/multilabel-v3-fresh-60ep_best.pt`)
with model-only resume (fresh optimizer). This means 5 confirm epochs start from F1≈0.5069
territory, making the gate achievable. Do NOT pass `--base-checkpoint ""` during v4 sprint.

```
--loss-fn       ∈ {bce, focal}        (focal is EXPERIMENTAL — also test bce+pos_weight)
--gamma         ∈ [1.0, 3.0]          (only relevant with loss_fn=focal)
--alpha         ∈ [0.20, 0.40]        (only relevant with loss_fn=focal)
--lora-r        ∈ {8, 16}             (lora_r=32 → OOM on 8 GB; forbidden)
--lora-alpha    ∈ {16, 32}
--batch-size    ∈ {8, 16}             (batch_size=32 → OOM with LoRA r≥8; forbidden)
--lr            ∈ [1e-4, 5e-4]
--weighted-sampler ∈ {none, DoS-only, all-rare}
```

Edits to `trainer.py` (if any) must stay in pure Python and must not import
new packages. Examples of legal trainer.py edits:
- Changing the OneCycleLR schedule shape
- Adding a DoS class cosine annealing variant
- Adjusting gradient clip value

---

## 5. The loop

```
REPEAT:
  1. Create a branch:
       git checkout -b autoresearch/<tag>-<idx>

  2. (Optionally) edit ml/src/training/trainer.py.
     Pure Python only. No new imports. No architecture changes.

  3. Run smoke:
       poetry run python ml/scripts/auto_experiment.py \
           --regime smoke \
           --run-name auto-<tag>-<idx> \
           --experiment-name sentinel-retrain-v4 \
           <knob args> \
           > ml/autoresearch/runs/auto-<tag>-<idx>.log 2>&1

  4. Parse score:
       grep '^SENTINEL_SCORE=' ml/autoresearch/runs/auto-<tag>-<idx>.log

  5. Append to results ledger (ml/autoresearch/results.tsv):
       <timestamp>  <branch>  smoke  <knobs>  <smoke_f1>  -  <peak_vram_mb>  <notes>

  6. Decide:
       if smoke_f1 > 0.42:             # promote threshold
           run confirm (--regime confirm, same knobs, same branch)
           parse confirm score
           update results.tsv (confirm_f1 column)
           if confirm_f1 > 0.5069 AND no floor breach:
               KEEP branch as candidate — do NOT delete
               set current_best = confirm_f1
           else:
               git reset --hard <base>; git branch -D autoresearch/<tag>-<idx>
       else:
           git reset --hard <base>; git branch -D autoresearch/<tag>-<idx>

UNTIL stop condition (see §7).
```

---

## 6. Per-class floor values (v3 tuned — must not drop below)

| Class | v3 F1 | Floor (v4 minimum) |
|-------|-------|---------------------|
| CallToUnknown | 0.394 | 0.344 |
| DenialOfService | 0.400 | 0.350 |
| ExternalBug | 0.435 | 0.385 |
| GasException | 0.550 | 0.500 |
| IntegerUO | 0.821 | 0.771 |
| MishandledException | 0.492 | 0.442 |
| Reentrancy | 0.536 | 0.486 |
| Timestamp | 0.479 | 0.429 |
| TransactionOrderDependence | 0.477 | 0.427 |
| UnusedReturn | 0.486 | 0.436 |

`auto_experiment.py` logs floor breaches to MLflow and prints them to stderr.
Check the log: a macro-F1 win with a floor breach is **NOT** a valid winner.

---

## 7. Stop conditions

Stop when any of these is reached:

- **12 confirmed runs total** — compute budget exhausted.
- **Three consecutive confirms below 0.5069** — signal exhausted; change strategy.
- **Human interrupt** (Ctrl-C) — the most recent committed branch is the candidate.

When stopping, report:
1. The branch name of the best candidate (if any).
2. The `results.tsv` contents.
3. Whether the gate was met.

---

## 8. OOM handling

Exit code 2 = OOM or runtime failure. Do not retry the same knobs.
Instead:
- Reduce `--batch-size` (try 8 if currently 16).
- Reduce `--lora-r` (try 8 if currently 16).
- If OOM persists with batch=8 lora_r=8: flag to operator; stop.

`PEAK_VRAM_MB` in the log shows how close you ran to the 8192 MB limit.
Values above 7800 MB are risky — reduce a knob before the next run.

---

## 9. Results ledger format (ml/autoresearch/results.tsv)

Tab-separated. Append one row per run. Never edit past rows.

```
timestamp	branch	regime	loss_fn	gamma	alpha	lora_r	lora_alpha	batch	lr	sampler	smoke_f1	confirm_f1	peak_vram_mb	notes
2026-05-09T...	autoresearch/v4-001	smoke	focal	2.0	0.25	8	16	16	3e-4	none	0.4129	-	3421	baseline reproduction
```

---

## 10. Phase B (post-v4 — do not enter until v4 sprint closes)

After a v4 winner is promoted to MLflow Staging, the Architecture Playground
opens. Use `program-playground.md` (not this file) for Phase B runs.
Phase B uses `--mode playground` which relaxes all locks except `val_indices.npy`.
