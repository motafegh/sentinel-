# 2026-05-09 — M1 · Autoresearch Integration Plan (RTX 3070 8 GB Laptop)

Companion to: `docs/changes/2026-05-09-M1-ml-plan.md` §3 ("Plan B —
Autoresearch Harness").
Spec refs: `docs/Project-Spec/SENTINEL-M1-ML.md`,
`docs/Project-Spec/SENTINEL-EVAL-BACKLOG.md`,
`docs/Project-Spec/SENTINEL-CONSTRAINTS.md`.
Status ref: `docs/STATUS.md` (Open Half-Open Loops → "Autoresearch").

Upstreams reviewed:
- karpathy/autoresearch (master) — original H100 / Linux design
- jsegov/autoresearch-win-rtx (master) — Windows desktop RTX fork
- miolini/autoresearch-macos / trevin-creator/autoresearch-mlx / andyluo7/autoresearch — out of scope (different platforms)

Hardware target: **Lenovo laptop, RTX 3070 mobile, 8 GB VRAM**, Windows.

---

## 1. Reality Check Before Anything Else

Two hard mismatches mean we **cannot use either upstream as-is**.

### 1.1 Hardware

`jsegov/autoresearch-win-rtx` README states explicitly:

| Constraint | Source | Affects us? |
|---|---|---|
| Tested on RTX 3080 10 GB | README "Tested" section | Yes — only 10 GB desktop class |
| Ampere requires **≥10 GB VRAM** | "tiered VRAM floors by architecture" | **Yes — RTX 3070 = 8 GB → below floor** |
| **Desktop-only**, "no laptop GPUs" | Known Limitations | **Yes — laptop disqualified** |
| RTX 2060 6 GB explicitly out of matrix | README | Confirms 8 GB is a hard line |

Conclusion: a laptop RTX 3070 is **out of the supported matrix** in two
independent ways (laptop + 8 GB on Ampere). The fork's training script
(a small GPT trained on TinyStories) might still run with reduced batch
size, but the contract is "5-minute meaningful val_bpb" — broken at this
VRAM tier.

karpathy upstream targets H100, also unsuitable.

### 1.2 Domain / Metric

autoresearch's built-in success criterion is `val_bpb` (validation
**bits per byte**) — a next-token prediction metric for autoregressive
LMs. SENTINEL's success criterion (per
`SENTINEL-EVAL-BACKLOG.md` §"Active Baseline") is:

```
tuned F1-macro on val_indices.npy, no per-class drop > 0.05 from v3 floors
```

These are not interchangeable. `train.py` in autoresearch is also a
small GPT trainer (~200 LOC). SENTINEL's `ml/src/training/trainer.py`
(~700 LOC) wires CodeBERT + LoRA + GAT + CrossAttention. We can't drop
our trainer into the autoresearch repo and have anything sensible
happen.

### 1.3 Conclusion: port the **pattern**, not the code

We adopt:

- The **agent-loop pattern**: branch → modify → train → grep score → keep/reset
- The `program.md` skill-file format
- The fixed time-budget discipline
- `results.tsv` ledger
- The "agent edits exactly one file" principle (limits blast radius)

We do **not** adopt:

- Either repo's `train.py` (we already have a better one)
- The `val_bpb` metric (not relevant to SENTINEL)
- The 5-minute wall-clock budget verbatim (see §4 for what we use)
- `uv` (we use Poetry — already established in project)
- The fork's hardware support matrix (it locks us out)

This integration lives entirely inside the `ml/` package; nothing is
forked or vendored from either upstream repo.

---

## 2. Source Survey — What's Already in `ml/`

(Verified 2026-05-09.)

```
ml/scripts/run_overnight_experiments.py    Sequential MLflow sweep (4 hand-coded
                                            TrainConfig variants). This is the
                                            closest existing analogue to
                                            autoresearch — but it's static.

ml/scripts/train.py                         Thin CLI over TrainConfig + train().
                                            Resume modes already wired.

ml/src/training/trainer.py                  TrainConfig dataclass exposes the
                                            knobs that an autoresearch agent
                                            could legally vary. See §6.

ml/scripts/tune_threshold.py                Per-class threshold tuning that
                                            produces the *tuned* F1-macro that
                                            is the actual SENTINEL_SCORE.

ml/checkpoints/                             multilabel-v3-fresh-60ep_best.pt
                                            (gate: 0.5069 tuned F1-macro)

ml/data/splits/val_indices.npy              FIXED split — never regenerate;
                                            authoritative gate evaluator.

ml/scripts/auto_experiment.py               DOES NOT EXIST yet (this plan creates).

ml/autoresearch/program.md                  DOES NOT EXIST yet (this plan creates).
```

Knobs already exposed by `TrainConfig` (verified in `trainer.py` lines
250–303 and CLI plumbing in `train.py`):

| Knob | Default | Notes |
|---|---|---|
| `epochs` | 40 | int |
| `batch_size` | 32 | int (8 GB VRAM cap likely 16) |
| `lr` | 3e-4 | float |
| `loss_fn` | `"bce"` | `"bce" \| "focal"` (`_VALID_LOSS_FNS`) |
| `focal_gamma` | 2.0 | float |
| `focal_alpha` | 0.25 | float |
| `lora_r` | 8 | int |
| `lora_alpha` | 16 | int |
| `num_workers` | 2 | int |
| `early_stop_patience` | int | already settable via CLI (Fix #20) |
| `grad_clip` / `warmup_pct` / `use_amp` / `log_interval` | various | also CLI-exposed (Fix #20) |

Knobs **not** exposed (require source change before agent can vary):

- weighted sampler for DenialOfService (per M1 plan §2.4)
- LoRA target_modules list (currently auto from arch field)
- gnn_hidden_dim / gnn_heads (architecture freeze applies — see §7)

---

## 3. Critical Architectural Constraints The Agent Must Not Touch

From `SENTINEL-INDEX.md` §"Critical Cross-Cutting Rules" — **any agent
that changes these silently breaks downstream M2/M5**:

```
fusion_output_dim = 128              ← LOCKED
GNNEncoder in_channels = 8           ← LOCKED (tied to 68,523 graphs)
NUM_CLASSES = 10                     ← LOCKED (append-only)
edge_attr shape = [E] 1-D            ← LOCKED
ARCHITECTURE = "cross_attention_lora" ← LOCKED for v4 to be comparable to v3
```

Plus:

- `val_indices.npy` is the gate — agent must never regenerate splits
- Active checkpoint name format is `multilabel-v<N>-<runname>_best.pt`
- MLflow experiment for v4 is `sentinel-retrain-v4` (do not pollute v3's)

The skill file (`program.md` in §5.2) **enumerates these and forbids
their modification explicitly**.

---

## 4. Time Budget For 8 GB Ampere Laptop — The "5-Minute" Question

autoresearch's 5-minute budget assumes a tiny GPT on H100/3080. On
RTX 3070 8 GB laptop running our architecture, even validation alone
over `val_indices.npy` (10,278 graphs, batch 16) takes a few minutes.
A meaningful F1 number requires at least one epoch of training on a
representative subset.

### 4.1 Two budget regimes

**Regime A — Smoke (~3–5 min/run, fast iteration, weak signal):**
- 1 epoch of training on a stratified 10 % subsample of train (≈4,800)
- Full val pass over `val_indices.npy`
- `tune_threshold.py` with `--max-batches 100`
- Agent metric: tuned F1-macro on the **subsampled trainer's** val
  pass — useful for *direction-finding*, not gate-passing

**Regime B — Confirmation (~30–60 min/run, used to confirm a smoke win):**
- N=5 epochs on full train
- Full val + full `tune_threshold.py`
- Agent metric: tuned F1-macro on full val — the only number
  comparable to the v3 0.5069 gate

The skill file routes runs through Regime A by default; the agent must
explicitly request a Regime B confirmation when smoke F1 exceeds a
threshold (suggest: smoke F1 > 0.42, calibrated empirically vs v3 trace).

### 4.2 Why two regimes (and not just "5 minutes")

A single fixed budget either:
- (a) runs full epochs and we get ~10 runs per night (acceptable signal,
  bad search efficiency), or
- (b) runs subsampled and we get 100+ runs but every winner is suspect
  on the real val split.

Two-tier amortises this — search fast on subsamples, confirm only the
winners on full data. Standard ML autotuning practice.

### 4.3 Concrete VRAM budget for RTX 3070 8 GB

Verified empirically during v3 (`docs/changes/2026-05-04-resume-batch-size-fix.md`):

| Knob | Safe range on 8 GB | Notes |
|---|---|---|
| batch_size | 8–16 | 32 OOMs on 8 GB with current arch + LoRA r=16 |
| lora_r | 8–16 | r=32 borderline; agent must test before committing |
| max_seq_len (CodeBERT) | 512 | windowed inference handles longer; do not raise |
| AMP | required | `use_amp=True` is non-negotiable on 8 GB |

The skill file lists these as "you may search within these ranges; values
outside cause OOM and an automatic discard."

---

## 5. File-Level Plan

### 5.1 `ml/scripts/auto_experiment.py` — NEW

Single-run wrapper that:

1. Parses `--gamma --alpha --lora-r --lora-alpha --batch-size --lr
   --loss-fn --weighted-sampler --regime [smoke|confirm] --run-name
   --experiment-name --max-epochs`
2. Pre-flight checks (must pass or exit non-zero before training):
   - `validate_graph_dataset.py` returns 0
   - `val_indices.npy` mtime matches v3-locked hash (recorded in
     `ml/data/splits/val_indices.sha256` — a new sidecar this plan adds)
   - GPU has ≥ 7 GB free
   - MLflow tracking URI reachable
3. Constructs `TrainConfig`, calls `train()` from `trainer.py`
4. Loads best checkpoint, calls `tune_threshold.py` programmatically
   (or imports its function path) to get tuned F1-macro
5. Logs all knobs and the final number to MLflow under the requested
   experiment name
6. Emits the autoresearch-style results to stdout. **Last line of
   stdout must be**:

```
SENTINEL_SCORE=<tuned_f1_macro_float>
PEAK_VRAM_MB=<int>
REGIME=<smoke|confirm>
```

(Multiple lines, but `SENTINEL_SCORE=` is parsed in isolation by
the agent — same pattern as autoresearch's `^val_bpb:` grep.)

7. Exit codes:
   - `0` — clean run, score emitted
   - `1` — pre-flight failed (no run started)
   - `2` — OOM or runtime failure during training
   - `3` — train succeeded but tune_threshold failed (rare; treat as 0
     with `SENTINEL_SCORE=0.0` to discard)

### 5.2 `ml/autoresearch/program.md` — NEW

Skill file for the agent. Mirrors karpathy's `program.md` structure but
encodes SENTINEL constraints. Required sections:

```
## 1. Goal
   Maximise tuned val F1-macro on ml/data/splits/val_indices.npy.
   Gate: > 0.5069 (v3 baseline).

## 2. Allowed file edits
   ONLY ml/src/training/trainer.py
   AND ml/scripts/auto_experiment.py CLI args.
   READ-ONLY: everything else, especially:
     - ml/src/preprocessing/  (graph schema is locked)
     - ml/src/models/         (architecture is locked: ADR-007, fusion=128)
     - ml/data/splits/        (val_indices.npy must not change)
     - ml/data/graphs/, ml/data/tokens/   (data freeze)
     - zkml/, contracts/      (downstream — not your concern)

## 3. Forbidden changes
   - Anything that touches fusion_output_dim, NUM_CLASSES,
     GNNEncoder.in_channels, ARCHITECTURE, edge_attr shape
   - Adding dependencies (no `poetry add`)
   - Regenerating splits or graphs
   - Touching MLflow experiment naming convention

## 4. Search space (knobs you may vary)
   focal_gamma     ∈ [1.0, 3.0]
   focal_alpha     ∈ [0.20, 0.40]
   lora_r          ∈ {8, 16}     (32 forbidden on 8 GB)
   lora_alpha      ∈ {16, 32}
   batch_size      ∈ {8, 16}     (32 forbidden on 8 GB)
   lr              ∈ [1e-4, 5e-4]
   loss_fn         ∈ {"bce", "focal"}
   weighted_sampler∈ {None, "DoS-only", "all-rare"}

## 5. Loop
   REPEAT:
     1. git checkout -b autoresearch/<tag>-<idx>
     2. (optional) edit trainer.py — pure Python, do not import new packages
     3. uv-equivalent: poetry run python ml/scripts/auto_experiment.py \
            --regime smoke <knob args> --run-name auto-<tag>-<idx>
     4. grep "^SENTINEL_SCORE=" run.log → score
     5. log to ml/autoresearch/results.tsv (untracked)
     6. if score > current_best:
            re-run with --regime confirm
            if confirmed > 0.5069: keep branch, update best_pointer
        else:
            git reset --hard previous; delete branch

## 6. Per-class floor enforcement
   After --regime confirm, the script also writes per-class F1 to MLflow
   under tags. Agent must check that no class drops > 0.05 F1 vs v3 floor
   before declaring a winner. Floors are listed in
   docs/Project-Spec/SENTINEL-EVAL-BACKLOG.md.

## 7. Stop conditions
   - 12 confirmed runs total (compute budget)
   - OR three consecutive confirms below 0.5069 (signal exhausted)
   - OR human interrupt
```

### 5.3 `ml/autoresearch/results.tsv` — RUNTIME

Columns (tab-separated, untracked, append-only):

```
timestamp	branch	regime	gamma	alpha	lora_r	lora_alpha	batch	lr	loss_fn	sampler	smoke_f1	confirm_f1	peak_vram_mb	notes
```

### 5.4 `ml/data/splits/val_indices.sha256` — NEW (sidecar)

Pin the val split hash so the agent can detect accidental regeneration.
Compute once now, commit, never modify:

```bash
sha256sum ml/data/splits/val_indices.npy > ml/data/splits/val_indices.sha256
```

`auto_experiment.py` step 2 verifies this before training.

### 5.5 `ml/autoresearch/README.md` — NEW

Short operator doc:

- How to start a session: `git checkout -b autoresearch/<tag>` then point
  Claude (or any agent) at `ml/autoresearch/program.md`
- How to stop: Ctrl-C; the most recent committed branch is the latest
  candidate
- How to validate a winner manually: `poetry run python
  ml/scripts/auto_experiment.py --regime confirm --resume <ckpt>`
- Known issues (8 GB VRAM, smoke vs confirm tradeoff)

---

## 6. What An Autoresearch Run Looks Like (concrete example)

```bash
# Operator opens a new session
git checkout claude/review-project-status-LFRYv
git checkout -b autoresearch/2026-05-09

# Hand the agent: ml/autoresearch/program.md + this plan
# Agent does:
git checkout -b autoresearch/2026-05-09-001
# (no trainer.py edit — first run is a baseline reproduction)
poetry run python ml/scripts/auto_experiment.py \
    --regime smoke \
    --run-name auto-2026-05-09-001 \
    --experiment-name sentinel-retrain-v4 \
    --gamma 2.0 --alpha 0.25 --lora-r 8 --lora-alpha 16 \
    --batch-size 16 --lr 3e-4 --loss-fn focal \
    > run.log 2>&1
grep '^SENTINEL_SCORE=' run.log
# → SENTINEL_SCORE=0.4129          (smoke estimate)
# → not better than current_best (none yet); ledger row written

git checkout -b autoresearch/2026-05-09-002
# Agent edits trainer.py — adds weighted sampler for DoS class
poetry run python ml/scripts/auto_experiment.py \
    --regime smoke ... --weighted-sampler DoS-only > run.log 2>&1
# → SENTINEL_SCORE=0.4318
# Better than 001; promote to confirm:
poetry run python ml/scripts/auto_experiment.py \
    --regime confirm ... --weighted-sampler DoS-only > run.confirm.log 2>&1
# → SENTINEL_SCORE=0.5142
# Above 0.5069 gate AND no per-class floor breach → keep branch.
```

---

## 7. Architecture-Freeze Guard (defence-in-depth)

Even with `program.md` listing forbidden edits, the agent might attempt
them. Add a runtime guard inside `auto_experiment.py` pre-flight:

```python
# Hash-pin the locked files; refuse to run if any changed
LOCKED = {
    "ml/src/models/sentinel_model.py": "<sha256>",
    "ml/src/models/gnn_encoder.py":    "<sha256>",
    "ml/src/preprocessing/graph_schema.py": "<sha256>",
    "ml/src/preprocessing/graph_extractor.py": "<sha256>",
    "ml/data/splits/val_indices.npy": "<sha256>",
}
```

Hashes computed once at plan-acceptance time. If a run starts and any
hash mismatches, exit `1` with a clear message. The agent's loop will
then reset and try a different edit.

This is not a security control (the agent could disable it) — it's a
guardrail against accidental drift.

---

## 8. What This Replaces / Supersedes

- `ml/scripts/run_overnight_experiments.py` becomes legacy. Its 4
  hand-coded experiments can stay as a sanity-check launcher, but new
  search work goes through `auto_experiment.py`.
- The "Plan B — Autoresearch Harness" subsection of
  `2026-05-09-M1-ml-plan.md` is now elaborated here. The pointer in
  that file should be updated to reference this document (see §11).

---

## 9. Phased Build

**Phase 1 — bootstrap (no agent yet):**
- Implement `auto_experiment.py` skeleton with both regimes
- Implement pre-flight checks + locked-files hash guard
- Manually run smoke + confirm on a known-good knob set; verify score
  matches v3 within ±0.01

**Phase 2 — skill file:**
- Write `ml/autoresearch/program.md` per §5.2
- Write `ml/autoresearch/README.md`
- Add `ml/data/splits/val_indices.sha256`

**Phase 3 — first agent session (operator-supervised):**
- Run a 4-hour session manually walking the agent through the loop
- Capture failures (OOM, missing flags, bad parses) and fix
  `auto_experiment.py`

**Phase 4 — overnight session:**
- Run unattended 8–12h
- Morning: review `results.tsv`, MLflow runs, pick winner

**Phase 5 — promote winner:**
- `tune_threshold.py` final pass
- `promote_model.py --stage Staging --val-f1-macro <new>`
- New dated changelog: `docs/changes/2026-05-1<x>-v4-training-complete.md`
- `docs/STATUS.md` "Active Checkpoint" updated

Each phase is an independent commit; do not bundle.

---

## 10. Risks and Decision Points

| Risk | Mitigation |
|---|---|
| Smoke regime under-predicts confirm F1 | Calibrate threshold (smoke > 0.42 → confirm) on first 5 confirmed runs; adjust |
| Agent edits a locked file anyway | Locked-files hash guard in `auto_experiment.py` rejects |
| 8 GB OOM under unusual knob combo | Pre-flight VRAM probe + per-knob bounds in `program.md` |
| MLflow run pollution | Dedicated experiment `sentinel-retrain-v4`; agent forbidden to use any other name |
| Agent loops forever wasting compute | Stop conditions in `program.md` §7 (12 confirms or 3 below-gate streak) |
| Wall-clock variance on a laptop (thermal throttle, background apps) | Record `peak_vram_mb` AND wall-clock in `results.tsv`; flag outliers |
| Per-class floor breach masked by macro-F1 win | `auto_experiment.py` writes per-class F1 to MLflow; agent loop must check before promoting |

**Decision points the operator owns (not the agent):**

- What `<tag>` to use for branches
- Whether to retire and restart after a failed confirm streak
- Whether to graduate a smoke-best to confirm (overrideable)
- Whether to merge the winning autoresearch branch back to the working
  feature branch

---

## 11. Cross-Doc Updates Required When This Lands

- `docs/changes/2026-05-09-M1-ml-plan.md` §3 — replace the placeholder
  with: "See `2026-05-09-M1-autoresearch-integration-plan.md` for the
  detailed design."
- `docs/STATUS.md` "Open Half-Open Loops" — update the Autoresearch row
  to reference this plan.
- `docs/changes/INDEX.md` — already updated by this commit.
- `docs/Project-Spec/SENTINEL-M1-ML.md` — append "Autoresearch" subsection
  pointing here. Spec rules apply unchanged.

---

## 12. Acceptance Criteria

- [ ] `ml/scripts/auto_experiment.py` exists, both regimes implemented,
      exit codes per §5.1
- [ ] `ml/autoresearch/program.md` exists, encodes constraints per §5.2
- [ ] `ml/autoresearch/README.md` documents the operator workflow
- [ ] `ml/data/splits/val_indices.sha256` committed
- [ ] Locked-files hash guard in `auto_experiment.py` operational
- [ ] Smoke regime completes a single run in < 8 minutes on RTX 3070 8 GB
      laptop
- [ ] Confirm regime reproduces v3 tuned F1-macro within ±0.01 on the
      v3 hyperparameters (sanity check)
- [ ] At least one autoresearch session produces a checkpoint with
      tuned F1-macro > 0.5069 and no per-class floor breach
- [ ] Operator can stop any session with Ctrl-C and resume manually with
      `--resume`
- [ ] No locked file (architecture / split / data) changed during the
      session — verified by hash guard

---

## 13. Out of Scope (explicitly)

- Forking either upstream autoresearch repo
- Running autoresearch's `train.py` (TinyStories GPT) — irrelevant to SENTINEL
- Multi-GPU search — single 3070 only
- Distributed search across machines
- AMD / MacOS / MLX variants of autoresearch
- Any LLM-driven *architectural* search (ARCHITECTURE is frozen for v4)
- Search over data preprocessing — graph and token data are frozen
