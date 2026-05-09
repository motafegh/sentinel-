# 2026-05-09 — Handover to Local Laptop Session

You are continuing SENTINEL development from a fresh local session on
your **Lenovo RTX 3070 8 GB laptop** (Windows / WSL2 + VS Code), where
the data is pulled, dependencies are installed, and the GPU is
available. This document is the entry point for that session.

---

## 1. Where work continues

Active feature branch (this PR adds the round-1 work):

```
claude/m1-step1-locked-hashes
```

It contains:

- `ml/scripts/compute_locked_hashes.py` — script that pins the v4-sprint
  locked-file hashes (used by round-2's autoresearch harness).
- `ml/locked_files.sha256` — partial sidecar (4 source-file hashes
  computed from a clean cloud checkout). It is **incomplete** until you
  re-run the script on the laptop to add `ml/data/splits/val_indices.npy`.

The branch is a draft PR. Do not merge it yet — it needs the laptop
step in §3 first.

> Note: the autoresearch plan §7 originally suggested
> `ml/data/splits/locked_files.sha256` as the sidecar path, but that
> directory is gitignored (DVC-managed). The file lives at
> `ml/locked_files.sha256` instead. The script default and all docs in
> this PR already reflect that.

---

## 2. The plans you should reference (all on `main`)

These were produced and merged in PRs #34 and #35. Read them in this
order at the start of your local session:

| Order | File | Why |
|---|---|---|
| 1 | `docs/changes/2026-05-09-module-plans-overview.md` | Entry point + execution order |
| 2 | `docs/changes/2026-05-09-project-state-synthesis.md` | Cross-module briefing — current state, system design, data flow, predictions |
| 3 | `docs/changes/2026-05-09-M1-ml-deep-dive.md` | Full M1 picture — process, design, decisions, flow, files, assumptions, criteria, **and the Phase B Architecture Playground in §12** |
| 4 | `docs/changes/2026-05-09-M1-ml-plan.md` | M1 plan proper — v4 sprint (§2), autoresearch hook (§3), open audits (§4), Architecture Playground (§6), acceptance (§7) |
| 5 | `docs/changes/2026-05-09-M1-autoresearch-integration-plan.md` | Autoresearch harness contract — `SENTINEL_SCORE=` stdout protocol (§5.1), `program.md` skill spec (§5.2), v4-sprint freeze guard (§7), Phase B extension (§14) |

All other modules (`M2-…`, `M3-…`, etc.) are documented in their own
plan files dated 2026-05-09 — read them when you reach those modules.
M1 is the current work.

---

## 3. First commands on the laptop (round-1 finishing step)

You need to:

1. Pull the latest `main` and check out the round-1 branch.
2. Make sure `ml/data/splits/val_indices.npy` is present (`dvc pull` if
   needed).
3. Re-run `compute_locked_hashes.py --write`. This will produce a
   complete 5-entry sidecar (the 4 source-file hashes already in the
   sidecar will match the bytes in main, so only `val_indices.npy` is
   added).
4. Run `--check` to verify, then commit the updated sidecar to the
   round-1 branch and merge the PR.

```bash
# in your repo working tree
git fetch origin
git checkout claude/m1-step1-locked-hashes
git pull --rebase origin main           # bring the branch up to date

# ensure data is present
dvc pull                                # if you don't already have ml/data/splits/

# regenerate the sidecar with val_indices.npy included
poetry run python ml/scripts/compute_locked_hashes.py --write

# expected output: "Wrote 5 entries to ml/locked_files.sha256"
poetry run python ml/scripts/compute_locked_hashes.py --check
# expected output: "OK: all 5 entries match."

# also run the dataset validator (M1 deep-dive §1 step 1)
poetry run python ml/scripts/validate_graph_dataset.py
# expected output: 68,523 / 68,523 PASS, exit 0

git add ml/locked_files.sha256
git commit -m "data: add val_indices.npy hash on laptop"
git push
# then merge the PR (or ask Claude in the next session to)
```

If `--write` shows that any of the 4 source-file hashes *changed*, that
means main has moved since the cloud commit — that's fine; the new
hashes are what you want. Re-commit, re-push.

---

## 4. Round 2 — what comes next (start a new Claude session for this)

Round 2 is the autoresearch harness itself. Reference doc:
`docs/changes/2026-05-09-M1-autoresearch-integration-plan.md`. The
deliverables, in dependency order:

1. **`ml/scripts/auto_experiment.py`** (new file) — single-run wrapper.
   See plan §5.1 for the exact CLI contract:
   - emits `SENTINEL_SCORE=<float>`, `PEAK_VRAM_MB=<int>`, `REGIME=<...>`
     as the last stdout lines
   - exit codes 0/1/2/3 per plan §5.1
   - calls `compute_locked_hashes.py --check` programmatically as part
     of pre-flight; aborts on mismatch
   - implements `--regime smoke` (10 % subsample, 1 epoch) and
     `--regime confirm` (full data, 5 epochs)
   - imports `train()` from `ml.src.training.trainer` and
     `tune_threshold` programmatically (don't shell out)
   - **build it with arbitrary keyword forwarding into `TrainConfig`
     from day one** (autoresearch plan §14, §3.3 of M1 plan) so Phase B's
     `--model-file` flag is one CLI addition later, not a refactor

2. **`ml/autoresearch/program.md`** (new file) — skill file. See
   autoresearch plan §5.2 for required sections (search space,
   forbidden edits in v4-sprint mode, loop, stop conditions). The
   v4-sprint locks come from §3 of the same plan.

3. **`ml/autoresearch/README.md`** (new file) — operator doc. See
   autoresearch plan §5.5.

4. **Sanity reproduction** (M1 deep-dive §1 step 3) — once the harness
   exists, run `--regime confirm` with v3 hyperparameters and verify
   tuned F1 ≈ 0.5069 ± 0.01.

Suggested round-2 PR scope: **just the harness skeleton + program.md +
README.md**, with the smoke/confirm regimes wired but no agent session
yet. Sanity reproduction lives in round 3.

When you start the next Claude session, paste this:

> Continue SENTINEL M1 work. Read
> `docs/changes/2026-05-09-handover-to-laptop.md` and then proceed to
> round 2 (autoresearch harness — `ml/scripts/auto_experiment.py`,
> `ml/autoresearch/program.md`, `ml/autoresearch/README.md`) per
> `docs/changes/2026-05-09-M1-autoresearch-integration-plan.md` §5.

---

## 5. Round 3+ — the rest of M1

| Round | Goal | Source-of-truth doc |
|---|---|---|
| 3 | Sanity reproduction: `--regime confirm` with v3 knobs reproduces 0.5069 ± 0.01 | M1 deep-dive §1 step 3 |
| 4 | First overnight v4 search session | M1 deep-dive §1 step 4, autoresearch plan §6 example |
| 5 | Promote winner via `promote_model.py --stage Staging` | M1 deep-dive §1 step 5 |
| 6 | Open-audit hardening (#9, #10, #12, #14, #15, #16, #17, #18) | M1 plan §4 |
| 7 (after v4) | Phase B Architecture Playground opens | M1 deep-dive §12, M1 plan §6 |

After M1 closes, the cross-module order from the project-state-synthesis
takes over: M3 → M5 → M2 → M4 → M6.

---

## 6. Environment notes (laptop-side)

These were verified from the cloud session and should already be true
on your laptop, but flag anything that doesn't match:

- `python --version` → 3.12 (project standard)
- `poetry --version` → installed
- `dvc --version` → installed; remote configured
- `ml/data/graphs/` populated (~68,523 .pt files)
- `ml/data/splits/val_indices.npy` present after `dvc pull`
- `ml/checkpoints/multilabel-v3-fresh-60ep_best.pt` present
- `ml/checkpoints/multilabel-v3-fresh-60ep_best_thresholds.json` present
- `TRANSFORMERS_OFFLINE=1` set in your shell rc (per
  `SENTINEL-CONSTRAINTS.md`)
- GPU free: `nvidia-smi` shows the 3070 with no python process holding
  VRAM
- MLflow tracking URI reachable (defaults to `sqlite:///mlruns.db` —
  fine)

---

## 7. Things deliberately *not* started here

These were **not touched** in the cloud session and are explicitly
deferred to the appropriate later round:

- The autoresearch harness itself (`auto_experiment.py`) — round 2
- Any actual training runs — round 3+
- M2 / M3 / M5 / M6 module work — after M1
- Phase B Architecture Playground — after v4 closes
- Any modification of `ml/src/models/` or `ml/src/preprocessing/` —
  these are v4-sprint-locked and the hash sidecar enforces it
- `ml/data/splits/val_indices.sha256` (the autoresearch plan mentioned
  this as a separate file; it has been folded into the unified
  `ml/locked_files.sha256` sidecar instead, since `val_indices.npy` is
  already in that file's set)

---

## 8. Quick sanity check before you start round 2

On the laptop, after finishing §3:

```bash
# Hash guard works
poetry run python ml/scripts/compute_locked_hashes.py --check
# → OK: all 5 entries match.

# Existing dataset validator works
poetry run python ml/scripts/validate_graph_dataset.py
# → exit 0

# Existing trainer imports cleanly (no torch surprises)
poetry run python -c "from ml.src.training.trainer import TrainConfig, train; print(TrainConfig())"
# → prints a TrainConfig dataclass

# v3 checkpoint loads
poetry run python -c "
from ml.src.inference.predictor import Predictor
p = Predictor('ml/checkpoints/multilabel-v3-fresh-60ep_best.pt')
print('thresholds:', p.thresholds.tolist())
"
# → prints 10 floats
```

If all four pass, you're cleared to start round 2.
